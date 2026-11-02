import Foundation
import CoreML
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Metal
import CryptoKit
import Darwin

// This executable compiles the iOS Engine/Models/Tensor/Image/Audio sources
// unchanged. Only argument handling, artifact writing and reporting live here.
private struct Arguments {
    let models: URL, portrait: URL, audio: URL, output: URL
    let chunks: Int
    let compute: String
    var units: MLComputeUnits { compute == "cpu" ? .cpuOnly : compute == "all" ? .all : .cpuAndGPU }
    static func parse() throws -> Arguments {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let help = "Usage: run_mac_smoke.sh --models BUNDLE --portrait IMAGE --audio AUDIO --output DIRECTORY [--chunks 1] [--compute cpu-gpu|all|cpu]"
        if arguments.contains("--help") { print(help); exit(0) }
        let supported = Set(["--models", "--portrait", "--audio", "--output", "--chunks", "--compute"])
        var values: [String: String] = [:]
        var index = 0
        while index < arguments.count {
            let key = arguments[index]
            guard supported.contains(key), index + 1 < arguments.count, values[key] == nil else {
                throw SoulXError.invalidManifest(help)
            }
            values[key] = arguments[index + 1]
            index += 2
        }
        guard let models = values["--models"], let portrait = values["--portrait"],
              let audio = values["--audio"], let output = values["--output"],
              let chunks = Int(values["--chunks"] ?? "1"), chunks > 0,
              ["cpu-gpu", "all", "cpu"].contains(values["--compute"] ?? "cpu-gpu") else {
            throw SoulXError.invalidManifest(help)
        }
        return Arguments(models: URL(fileURLWithPath: models), portrait: URL(fileURLWithPath: portrait),
                         audio: URL(fileURLWithPath: audio), output: URL(fileURLWithPath: output),
                         chunks: chunks, compute: values["--compute"] ?? "cpu-gpu")
    }
}

private func sha256(_ url: URL) throws -> String {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    var hash = SHA256()
    while let data = try handle.read(upToCount: 8 << 20), !data.isEmpty { hash.update(data: data) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}

private final class Recorder: @unchecked Sendable {
    private let lock = NSLock()
    private let directory: URL
    private let started = Date.timeIntervalSinceReferenceDate
    private var stages: [[String: Any]] = []
    private var windows: [[String: Any]] = []
    private var report: [String: Any]

    init(_ args: Arguments) throws {
        directory = args.output
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let gpu = MTLCreateSystemDefaultDevice()
        report = [
            "schema_version": 1, "date_utc": ISO8601DateFormatter().string(from: Date()), "status": "running",
            "hardware": ["gpu_model": gpu?.name ?? "unverified; Metal device unavailable",
                         "evidence": "MTLCreateSystemDefaultDevice and ProcessInfo APIs on this Mac",
                         "physical_system_memory_bytes": ProcessInfo.processInfo.physicalMemory,
                         "physical_vram": gpu?.hasUnifiedMemory == true ? "Unified system memory; no dedicated VRAM" : "Unverified",
                         "visible_vram": "Metal recommendedMaxWorkingSetSize is a budget, not physical VRAM",
                         "metal_recommended_working_set_bytes": gpu?.recommendedMaxWorkingSetSize ?? 0,
                         "driver_runtime": "Core ML / Metal supplied by \(ProcessInfo.processInfo.operatingSystemVersionString)",
                         "co_resident_load": "Unknown; conversion and device tests may be co-resident",
                         "thermal_state_at_start": ProcessInfo.processInfo.thermalState.rawValue],
            "execution_environment": "Native macOS executable; shared iOS inference runtime sources",
            "execution_classification": args.compute == "cpu" ? "Requested CPU-only Core ML inference on Mac" : "Requested Mac Core ML inference; actual CPU/GPU/ANE placement unverified",
            "scope": "Mac functional smoke and Mac timings only. No iPhone FPS, ANE placement, CUDA parity, lip-sync or perceptual quality claim.",
            "requested_compute_units": args.compute, "requested_chunks": args.chunks,
            "workload": "Original full PRO architecture, two denoising steps per 33-frame window; 28 visible frames; first-use model loading included",
            "models_directory": args.models.path, "portrait": args.portrait.path, "audio": args.audio.path,
            "model_manifest_sha256": try sha256(args.models.appendingPathComponent("manifest.json")),
            "portrait_sha256": try sha256(args.portrait), "audio_sha256": try sha256(args.audio),
            "swift_compiler": ProcessInfo.processInfo.environment["SOULX_SWIFT_COMPILER"] ?? "unverified",
            "git_revision": ProcessInfo.processInfo.environment["SOULX_GIT_REVISION"] ?? "unverified",
            "inference_started": false, "inference_completed": false
        ]
        if let root = ProcessInfo.processInfo.environment["SOULX_RUNTIME_ROOT"] {
            var hashes: [String: String] = [:]
            for file in ["SoulXEngine.swift", "SoulXModels.swift", "SoulXTensor.swift", "SoulXImage.swift", "SoulXAudio.swift"] {
                hashes[file] = try sha256(URL(fileURLWithPath: root).appendingPathComponent(file))
            }
            report["runtime_source_sha256"] = hashes
        }
        persist()
    }

    func set(_ key: String, _ value: Any) {
        lock.lock(); defer { lock.unlock() }
        report[key] = value; persist()
    }

    func stage(_ text: String) {
        lock.lock(); defer { lock.unlock() }
        stages.append(["elapsed_seconds": Date.timeIntervalSinceReferenceDate - started, "message": text])
        report["stages"] = stages
        report["last_stage"] = text
        persist()
        print("SOULX_MAC \(text)")
        fflush(stdout)
    }

    func chunk(_ chunk: SoulXChunk) throws {
        lock.lock(); defer { lock.unlock() }
        guard let first = chunk.frames.first, let last = chunk.frames.last else {
            throw SoulXError.invalidTensor("Generated window has no frames")
        }
        let index = windows.count
        try Self.writePNG(first, to: directory.appendingPathComponent(String(format: "window_%02d_first.png", index)))
        try Self.writePNG(last, to: directory.appendingPathComponent(String(format: "window_%02d_last.png", index)))
        if index == 0 { try Self.writePNG(first, to: directory.appendingPathComponent("first.png")) }
        try Self.writePNG(last, to: directory.appendingPathComponent("last.png"))
        windows.append(["index": index, "start_frame": chunk.startFrame, "frames": chunk.frames.count,
                        "inference_seconds": chunk.inferenceSeconds, "mac_generation_fps": chunk.generationFPS,
                        "playback_fps": chunk.fps, "audio_samples": chunk.samples.count])
        report["windows"] = windows
        report["generated_frames"] = windows.reduce(0) { $0 + ($1["frames"] as? Int ?? 0) }
        persist()
    }

    private static func writePNG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw SoulXError.invalidImage
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw SoulXError.invalidImage }
    }

    private func persist() {
        do {
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: directory.appendingPathComponent("timings.json"), options: .atomic)
        } catch { print("SOULX_MAC report_write_error=\(error.localizedDescription)") }
    }
}

@main private enum MacSmoke {
    static func main() async {
        var recorder: Recorder?
        let started = Date.timeIntervalSinceReferenceDate
        do {
            let args = try Arguments.parse()
            let log = try Recorder(args); recorder = log
            let manifest = try SoulXManifest.load(from: args.models)
            log.set("profile", ["name": manifest.profile, "height": manifest.height, "width": manifest.width,
                                "sampling_steps": manifest.samplingSteps, "block_count": manifest.blockCount,
                                "decoder_mode": manifest.decoderMode])
            guard let source = CGImageSourceCreateWithURL(args.portrait as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw SoulXError.invalidImage }
            let engine = SoulXEngine(modelsDirectory: args.models, computeUnits: args.units)
            log.set("inference_started", true)
            let initialization = Date.timeIntervalSinceReferenceDate
            try await engine.initialize(image: image, onProgress: { log.stage($0) })
            log.set("initialization_seconds", Date.timeIntervalSinceReferenceDate - initialization)
            try await engine.generate(audioURL: args.audio, maximumChunks: args.chunks,
                                      onProgress: { log.stage($0) }, onChunk: { try log.chunk($0) })
            await engine.reset()
            log.set("total_seconds", Date.timeIntervalSinceReferenceDate - started)
            log.set("inference_completed", true)
            log.set("status", "passed")
            log.stage("Mac smoke completed")
        } catch {
            recorder?.set("total_seconds", Date.timeIntervalSinceReferenceDate - started)
            recorder?.set("error", error.localizedDescription)
            recorder?.set("status", "failed")
            recorder?.stage("Mac smoke failed")
            print("SOULX_MAC_ERROR \(error.localizedDescription)")
            exit(1)
        }
    }
}
