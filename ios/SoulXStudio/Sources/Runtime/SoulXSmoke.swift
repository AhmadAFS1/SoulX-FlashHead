import Foundation
import CoreML
import CoreGraphics
import ImageIO
import Metal
import UniformTypeIdentifiers
import Darwin

/// Launch with --smoke for an actual, finite one-window inference run. A report
/// is persisted before expensive work and after every stage so a device memory
/// termination leaves the last completed stage visible to the operator.
public enum SoulXSmoke {
    private static let latch = SoulXSmokeLatch()
    @discardableResult
    public static func runIfRequested() async -> Bool {
        let arguments = ProcessInfo.processInfo.arguments
        guard arguments.contains("--smoke") else { return false }
        guard latch.begin() else { return true }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let resultDirectory = documents.appendingPathComponent("SmokeResult", isDirectory: true)
        let units: MLComputeUnits
        let computeLabel: String
        if arguments.contains("--cpu-only") { units = .cpuOnly; computeLabel = "cpuOnly" }
        else if arguments.contains("--cpu-gpu") { units = .cpuAndGPU; computeLabel = "cpuAndGPU" }
        else if arguments.contains("--cpu-ane") { units = .cpuAndNeuralEngine; computeLabel = "cpuAndNeuralEngine" }
        else { units = .all; computeLabel = "all" }
        let chunkArgument = arguments.firstIndex(of: "--smoke-chunks").flatMap { index in
            index + 1 < arguments.count ? Int(arguments[index + 1]) : nil
        }
        let maximumChunks = min(10, max(1, chunkArgument ?? 1))
        let saveFrames = arguments.contains("--smoke-save-frames")
        let recorder = SoulXSmokeRecorder(directory: resultDirectory, computeLabel: computeLabel)
        let started = Date.timeIntervalSinceReferenceDate
        do {
            try FileManager.default.createDirectory(at: resultDirectory, withIntermediateDirectories: true)
            recorder.set("requested_windows", maximumChunks)
            recorder.set("save_all_frames", saveFrames)
            recorder.set("workload", "Requested PRO, up to \(maximumChunks) generated windows; first-use model loading may be included")
            recorder.stage("Starting real on-device inference")
            let models = documents.appendingPathComponent("Models/current", isDirectory: true)
            let status = SoulXEngine.modelStatus(at: models)
            guard status.isReady else { throw SoulXError.missingModels(status.missingModels) }
            let manifest = try SoulXManifest.load(from: models)
            recorder.set("profile", ["name": manifest.profile, "height": manifest.height, "width": manifest.width,
                                     "decoder_mode": manifest.decoderMode, "sampling_steps": manifest.samplingSteps,
                                     "block_count": manifest.blockCount])
            let imageURL = documents.appendingPathComponent("Smoke/portrait.png")
            guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw SoulXError.invalidImage }
            let engine = SoulXEngine(modelsDirectory: models, computeUnits: units)
            let initialization = Date.timeIntervalSinceReferenceDate
            recorder.set("inference_started", true)
            let cacheCandidate = documents.appendingPathComponent("Smoke/reference-cache/metadata.json")
            let cacheURL = FileManager.default.fileExists(atPath: cacheCandidate.path) ? cacheCandidate : nil
            recorder.set("reference_initialization", cacheURL == nil ? "Original Wan encoder on device" : "Verified precomputed original Wan portrait features; no generated video cached")
            try await engine.initialize(image: image, referenceCacheURL: cacheURL, sourceImageURL: imageURL, onProgress: { recorder.stage($0) })
            recorder.set("initialization_seconds", Date.timeIntervalSinceReferenceDate - initialization)
            try await engine.generate(audioURL: documents.appendingPathComponent("Smoke/audio.wav"), maximumChunks: maximumChunks,
                                      onProgress: { recorder.stage($0) }, onChunk: { chunk in
                guard let first = chunk.frames.first, let last = chunk.frames.last else {
                    throw SoulXError.invalidTensor("Smoke inference returned no frames")
                }
                if saveFrames {
                    let framesDirectory = resultDirectory.appendingPathComponent("Frames", isDirectory: true)
                    try FileManager.default.createDirectory(at: framesDirectory, withIntermediateDirectories: true)
                    for (offset, frame) in chunk.frames.enumerated() {
                        let filename = String(format: "frame-%06d.png", chunk.startFrame + offset)
                        try writePNG(frame, to: framesDirectory.appendingPathComponent(filename))
                    }
                }
                if chunk.startFrame == 0 {
                    try writePNG(first, to: resultDirectory.appendingPathComponent("first.png"))
                }
                try writePNG(first, to: resultDirectory.appendingPathComponent("window-\(chunk.startFrame)-first.png"))
                try writePNG(last, to: resultDirectory.appendingPathComponent("last.png"))
                let metrics: [String: Any] = ["frames": chunk.frames.count, "inference_seconds": chunk.inferenceSeconds,
                                        "generation_fps": chunk.generationFPS, "playback_fps": chunk.fps,
                                        "audio_samples": chunk.samples.count, "start_frame": chunk.startFrame]
                recorder.set("window", metrics)
                recorder.set("window_\(chunk.startFrame)", metrics)
            })
            recorder.set("total_seconds", Date.timeIntervalSinceReferenceDate - started)
            recorder.set("inference_completed", true)
            recorder.set("status", "passed")
            recorder.stage("Real inference windows completed")
            await engine.reset()
        } catch {
            recorder.set("total_seconds", Date.timeIntervalSinceReferenceDate - started)
            recorder.set("error", String(describing: error))
            recorder.set("localized_error", error.localizedDescription)
            recorder.set("status", "failed")
            recorder.stage("Smoke inference failed")
        }
        return true
    }

    private static func writePNG(_ image: CGImage, to url: URL) throws {
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw SoulXError.invalidImage
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw SoulXError.invalidImage }
    }
}

private final class SoulXSmokeLatch: @unchecked Sendable {
    private let lock = NSLock()
    private var started = false
    func begin() -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !started else { return false }
        started = true
        return true
    }
}

private final class SoulXSmokeRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private let reportURL: URL
    private let started = Date.timeIntervalSinceReferenceDate
    private var stages: [[String: Any]] = []
    private var report: [String: Any]

    init(directory: URL, computeLabel: String) {
        reportURL = directory.appendingPathComponent("report.json")
        var machine = utsname()
        uname(&machine)
        let machineCapacity = MemoryLayout.size(ofValue: machine.machine)
        let identifier = withUnsafePointer(to: &machine.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: machineCapacity) { String(cString: $0) }
        }
        let gpu = MTLCreateSystemDefaultDevice()?.name ?? "unverified; Metal device unavailable"
        #if targetEnvironment(simulator)
        let executionEnvironment = "iOS Simulator"
        #else
        let executionEnvironment = "Physical Apple device"
        #endif
        report = [
            "schema_version": 1,
            "status": "running",
            "execution_environment": executionEnvironment,
            "inference_started": false,
            "inference_completed": false,
            "date_utc": ISO8601DateFormatter().string(from: Date()),
            "hardware": [
                "machine": identifier,
                "gpu_model": gpu,
                "gpu_evidence": "MTLCreateSystemDefaultDevice()?.name; machine from uname",
                "physical_system_memory_bytes": ProcessInfo.processInfo.physicalMemory,
                "physical_vram": "Unified system memory; no dedicated VRAM",
                "visible_vram": "Not exposed by iOS; process memory limit unknown",
                "driver_runtime": "Core ML / Metal supplied by \(ProcessInfo.processInfo.operatingSystemVersionString)",
                "co_resident_load": "Unknown; no other-app process inspection",
                "thermal_state_at_start": ProcessInfo.processInfo.thermalState.rawValue
            ],
            "requested_compute_units": computeLabel,
            "execution_classification": computeLabel == "cpuOnly" ? "Requested CPU-only Core ML inference; check inference_started/completed" : "Requested Core ML inference on \(executionEnvironment); actual CPU/GPU/ANE placement unverified; check inference_started/completed",
            "workload": "Requested PRO, one generated window, cold first-use compilation/loading may be included",
            "parity": "Visual and numerical CUDA parity not established by this smoke test"
        ]
    }

    func set(_ key: String, _ value: Any) {
        lock.lock(); defer { lock.unlock() }
        report[key] = value
        persist()
    }

    func stage(_ message: String) {
        lock.lock(); defer { lock.unlock() }
        let elapsed = Date.timeIntervalSinceReferenceDate - started
        stages.append(["elapsed_seconds": elapsed, "message": message])
        report["stages"] = stages
        report["last_stage"] = message
        print("SOULX_SMOKE \(message) elapsed_seconds=\(elapsed)")
        persist()
    }

    private func persist() {
        do {
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: reportURL, options: .atomic)
        } catch { print("SOULX_SMOKE report_write_error=\(error.localizedDescription)") }
    }
}
