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
    let cache: URL?
    let saveFrames: Bool
    let timings: Bool
    let audioCPU: Bool
    let prefetchAudio: Bool
    let prefetchBlocks: Bool
    let keepResident: Bool
    let chunks: Int
    let compute: String
    var units: MLComputeUnits { compute == "cpu" ? .cpuOnly : compute == "all" ? .all : .cpuAndGPU }
    static func parse() throws -> Arguments {
        let arguments = Array(CommandLine.arguments.dropFirst())
        let help = "Usage: run_mac_smoke.sh --models BUNDLE --portrait IMAGE --audio AUDIO --output DIRECTORY [--reference-cache METADATA] [--save-frames] [--model-timings] [--audio-cpu] [--prefetch-audio] [--prefetch-blocks] [--keep-models-resident] [--parallel-frames] [--gpu-low-precision] [--fast-prediction] [--dit-compute cpu|cpu-gpu|cpu-ane|all] [--chunks 1] [--compute cpu-gpu|all|cpu]"
        if arguments.contains("--help") { print(help); exit(0) }
        let supported = Set(["--reference-cache", "--models", "--portrait", "--audio", "--output", "--chunks", "--compute", "--dit-compute"])
        var values: [String: String] = [:]
        var flags = Set<String>()
        var index = 0
        while index < arguments.count {
            let key = arguments[index]
            if ["--save-frames", "--model-timings", "--audio-cpu", "--prefetch-audio", "--prefetch-blocks", "--keep-models-resident",
                "--gpu-low-precision", "--fast-prediction", "--parallel-frames"].contains(key) {
                guard flags.insert(key).inserted else { throw SoulXError.invalidManifest(help) }
                index += 1
                continue
            }
            guard supported.contains(key), index + 1 < arguments.count, values[key] == nil else {
                throw SoulXError.invalidManifest(help)
            }
            values[key] = arguments[index + 1]
            index += 2
        }
        guard let models = values["--models"], let portrait = values["--portrait"],
              let audio = values["--audio"], let output = values["--output"],
              let chunks = Int(values["--chunks"] ?? "1"), chunks > 0,
              ["cpu-gpu", "all", "cpu"].contains(values["--compute"] ?? "cpu-gpu"),
              ["cpu", "cpu-gpu", "cpu-ane", "all"].contains(values["--dit-compute"] ?? "all") else {
            throw SoulXError.invalidManifest(help)
        }
        #if SOULX_BENCH_BASELINE
        guard flags.subtracting(["--save-frames"]).isEmpty, values["--dit-compute"] == nil else {
            throw SoulXError.invalidManifest("The historical benchmark uses sequential inference with the CPU speech encoder.")
        }
        #endif
        return Arguments(models: URL(fileURLWithPath: models), portrait: URL(fileURLWithPath: portrait),
                         audio: URL(fileURLWithPath: audio), output: URL(fileURLWithPath: output),
                         cache: values["--reference-cache"].map { URL(fileURLWithPath: $0) },
                         saveFrames: flags.contains("--save-frames"), timings: flags.contains("--model-timings"),
                         audioCPU: flags.contains("--audio-cpu"),
                         prefetchAudio: flags.contains("--prefetch-audio"), prefetchBlocks: flags.contains("--prefetch-blocks"),
                         keepResident: flags.contains("--keep-models-resident"), chunks: chunks, compute: values["--compute"] ?? "cpu-gpu")
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
    private let saveFrames: Bool
    private let started = Date.timeIntervalSinceReferenceDate
    private var stages: [[String: Any]] = []
    private var windows: [[String: Any]] = []
    private var report: [String: Any]

    init(_ args: Arguments) throws {
        directory = args.output
        saveFrames = args.saveFrames
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
            "audio_lookahead": args.prefetchAudio, "partition_preload": args.prefetchBlocks,
            "models_resident": args.keepResident,
            "parallel_frames": CommandLine.arguments.contains("--parallel-frames"),
            "gpu_low_precision_accumulation": CommandLine.arguments.contains("--gpu-low-precision"),
            "fast_prediction_specialization": CommandLine.arguments.contains("--fast-prediction"),
            "dit_compute_units": CommandLine.arguments.firstIndex(of: "--dit-compute").map { CommandLine.arguments[$0 + 1] } ?? "inherit",
            "audio_compute": args.audioCPU ? "cpu" : "inherit",
            "model_timings": args.timings,
            "reference_initialization": args.cache == nil ? "on-device encoder" : "verified portrait cache",
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
            for file in ["SoulXEngine.swift", "SoulXModels.swift", "SoulXTensor.swift", "SoulXImage.swift", "SoulXAudio.swift", "SoulXReferenceCache.swift", "SoulXAudioLookahead.swift"] {
                let url = URL(fileURLWithPath: root).appendingPathComponent(file)
                if FileManager.default.fileExists(atPath: url.path) { hashes[file] = try sha256(url) }
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
        var pixels = SHA256()
        for (offset, frame) in chunk.frames.enumerated() {
            guard let bytes = frame.dataProvider?.data else { throw SoulXError.invalidImage }
            pixels.update(data: bytes as Data)
            if saveFrames {
                try Self.writePNG(frame, to: directory.appendingPathComponent(String(format: "frame-%06d.png", chunk.startFrame + offset)))
            }
        }
        let pixelHash = pixels.finalize().map { String(format: "%02x", $0) }.joined()
        var window: [String: Any] = ["rgba_sha256": pixelHash, "index": index, "start_frame": chunk.startFrame, "frames": chunk.frames.count,
                        "inference_seconds": chunk.inferenceSeconds, "mac_generation_fps": chunk.generationFPS,
                        "playback_fps": chunk.fps, "audio_samples": chunk.samples.count]
        #if !SOULX_BENCH_BASELINE
        window["audio_encoder"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(chunk.audioEncoderStatus))
        if let profile = chunk.windowProfile {
            window["model_timings"] = try JSONSerialization.jsonObject(with: JSONEncoder().encode(profile))
        }
        #endif
        windows.append(window)
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
            #if SOULX_BENCH_BASELINE
            let engine = SoulXEngine(modelsDirectory: args.models, computeUnits: args.units)
            log.set("audio_compute", "cpu")
            log.set("benchmark_reference", "Historical runtime with CPU Wav2Vec encoder and original conditioning/DiT compute selection")
            #else
            let engine = SoulXEngine(modelsDirectory: args.models, computeUnits: args.units,
                                    audioComputeUnits: args.audioCPU ? .cpuOnly : nil,
                                    collectModelTimings: args.timings, prefetchAudio: args.prefetchAudio, prefetchBlocks: args.prefetchBlocks)
            #endif
            log.set("inference_started", true)
            let initialization = Date.timeIntervalSinceReferenceDate
            try await engine.initialize(image: image, referenceCacheURL: args.cache, sourceImageURL: args.portrait, onProgress: { log.stage($0) })
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
