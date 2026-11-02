import Foundation
import CoreGraphics
import CoreML

public struct SoulXChunk: @unchecked Sendable {
    public let frames: [CGImage]
    public let samples: [Float]
    public let startFrame: Int
    public let inferenceSeconds: Double
    public let fps: Double
    public var generationFPS: Double { Double(frames.count) / max(inferenceSeconds, 0.000001) }
}

/// On-device PRO inference. Model calls and mutable temporal state have a single
/// owner. The awaited chunk callback provides explicit backpressure: the engine
/// never creates an unbounded sequence of full-resolution video buffers.
public actor SoulXEngine {
    public let modelsDirectory: URL
    private let computeUnits: MLComputeUnits
    private var store: SoulXModelStore?
    private var reference: MLMultiArray?
    private var referenceColors: SoulXColorStatistics?
    private var motion: [Float] = []
    private var motionTimes = 1
    private var noise = SoulXNoise(seed: 42)
    private var audioHistory = [Float](repeating: 0, count: 128000)
    private var outputFrame = 0
    private var generationID = UUID()
    private var generating = false

    public init(modelsDirectory: URL, computeUnits: MLComputeUnits = .all) {
        self.modelsDirectory = modelsDirectory
        self.computeUnits = computeUnits
    }

    public nonisolated static func modelStatus(at directory: URL) -> SoulXModelStatus {
        do {
            let manifest = try SoulXManifest.load(from: directory)
            let missing = manifest.modelNames.filter { SoulXModelStore.sourceURL($0, directory: directory) == nil }
            return SoulXModelStatus(missingModels: missing, isReady: missing.isEmpty,
                                    profileDescription: "PRO · \(manifest.width) × \(manifest.height) · 2 steps")
        } catch {
            return SoulXModelStatus(missingModels: ["manifest.json (\(error.localizedDescription))"],
                                    isReady: false, profileDescription: "Core ML models required")
        }
    }

    public func reset() {
        generationID = UUID()
        reference = nil
        referenceColors = nil
        motion = []
        audioHistory = [Float](repeating: 0, count: 128000)
        outputFrame = 0
        store?.unloadAll()
        store = nil
    }

    public func initialize(image: CGImage, referenceCacheURL: URL? = nil, sourceImageURL: URL? = nil, seed: UInt64 = 42,
                           onProgress: @Sendable (String) -> Void = { _ in }) async throws {
        guard !generating else { throw SoulXError.invalidTensor("Finish or cancel the current stream before changing portraits") }
        reset()
        onProgress("Checking model files")
        let loaded = try SoulXModelStore(directory: modelsDirectory, computeUnits: computeUnits)
        let configuration = loaded.manifest
        onProgress("Preparing portrait")
        let (input, colors) = try SoulXImage.prepare(image, height: configuration.height, width: configuration.width)
        try Task.checkCancellation()
        let started = Date.timeIntervalSinceReferenceDate
        let latent: MLMultiArray
        if let referenceCacheURL {
            onProgress("Verifying bundled portrait features")
            guard let sourceImageURL else { throw SoulXError.invalidReferenceCache("The original portrait file URL is required to verify cached features") }
            latent = try SoulXReferenceCache.load(metadataURL: referenceCacheURL, manifest: configuration, sourceImageURL: sourceImageURL)
        } else {
            onProgress("Encoding portrait on device")
            latent = try loaded.predict("reference", inputs: ["image": input], output: "reference")
        }
        try Task.checkCancellation()
        _ = try SoulXTensor.checked(latent, shape: configuration.latentShape, name: "reference")
        let values = SoulXTensor.values(latent)
        guard values.allSatisfy(\.isFinite) else { throw SoulXError.invalidTensor("Non-finite reference latent") }
        let plane = configuration.latentHeight * configuration.latentWidth
        var prefix = [Float](repeating: 0, count: 16 * plane)
        for c in 0..<16 { prefix.replaceSubrange((c * plane)..<((c + 1) * plane), with: values[(c * 9 * plane)..<(c * 9 * plane + plane)]) }
        // The original reference encoder is only needed during initialization.
        loaded.unload("reference")
        store = loaded
        reference = latent
        referenceColors = colors
        motion = prefix
        motionTimes = 1
        noise = SoulXNoise(seed: seed)
        print("SOULX_EVENT reference_ready source=\(referenceCacheURL == nil ? "device_encoder" : "verified_original_wan_cache") seconds=\(Date.timeIntervalSinceReferenceDate - started) width=\(configuration.width) height=\(configuration.height)")
        onProgress("Portrait ready")
    }

    public func generate(audioURL: URL,
                         maximumChunks: Int? = nil,
                         onProgress: @Sendable (String) -> Void = { _ in },
                         onChunk: @Sendable (SoulXChunk) async throws -> Void) async throws {
        guard !generating else { throw SoulXError.invalidTensor("A generation is already running") }
        guard let store, let reference, let colors = referenceColors else { throw SoulXError.notInitialized }
        if let maximumChunks, maximumChunks < 1 { throw SoulXError.invalidTensor("maximumChunks must be positive") }
        generating = true
        var completed = false
        defer {
            // An interrupted window may have advanced audio without committing
            // motion. Require fresh initialization before another generation.
            if !completed { reset() }
            generating = false
        }
        let token = generationID
        let configuration = store.manifest
        onProgress("Reading audio")
        let audio = try SoulXAudio.readMono16k(audioURL)
        let totalFrames = Int(ceil(Double(audio.count) * Double(configuration.fps) / Double(configuration.sampleRate)))
        let samplesPerFrame = configuration.sampleRate / configuration.fps
        let samplesPerChunk = configuration.usefulFrames * samplesPerFrame
        var emitted = 0
        var emittedChunks = 0
        while emitted < totalFrames {
            try check(token)
            let started = Date.timeIntervalSinceReferenceDate
            let firstSample = emitted * samplesPerFrame
            var chunkAudio = [Float](repeating: 0, count: samplesPerChunk)
            let available = min(samplesPerChunk, audio.count - firstSample)
            if available > 0 { chunkAudio.replaceSubrange(0..<available, with: audio[firstSample..<(firstSample + available)]) }
            audioHistory.removeFirst(samplesPerChunk)
            audioHistory.append(contentsOf: chunkAudio)
            onProgress("Encoding audio · window \(emitted / configuration.usefulFrames + 1)")
            // Normalization is inside the exported audio model.
            let waveform = try SoulXTensor.make([1, 128000], values: audioHistory, type: .float32)
            let features = try store.predict("audio", inputs: ["waveform": waveform], output: "features")
            let context = try store.predict("conditioning", inputs: ["audio": SoulXAudio.windows(features)], output: "context")
            store.unload("audio")
            store.unload("conditioning")
            try check(token)
            let plane = configuration.latentHeight * configuration.latentWidth
            var latentValues = noise.normals(count: 16 * 9 * plane)
            for step in 0..<2 {
                try check(token)
                restoreMotion(into: &latentValues, plane: plane)
                let latent = try SoulXTensor.make(configuration.latentShape, values: latentValues)
                let timestep = try SoulXTensor.make([1], values: [configuration.samplingSteps[step]], type: .float32)
                var hidden = try store.predict("patch", inputs: ["latent": latent, "reference": reference], output: "hidden")
                for index in 0..<configuration.blockCount {
                    try check(token)
                    let name = String(format: "blocks_%02d", index)
                    onProgress("Generating · step \(step + 1)/2 · part \(index + 1)/\(configuration.blockCount)")
                    let stage = Date.timeIntervalSinceReferenceDate
                    hidden = try store.predict(name, inputs: ["hidden": hidden, "context": context, "timestep": timestep], output: "hidden_out")
                    print("SOULX_EVENT partition name=\(name) step=\(step) seconds=\(Date.timeIntervalSinceReferenceDate - stage)")
                }
                let prediction = try store.predict("head", inputs: ["hidden": hidden, "timestep": timestep], output: "flow")
                _ = try SoulXTensor.checked(prediction, shape: configuration.latentShape, name: "flow")
                let flow = SoulXTensor.values(prediction)
                let time = configuration.samplingSteps[step] / 1000
                let next = configuration.samplingSteps[step + 1] / 1000
                for i in latentValues.indices {
                    let initial = latentValues[i] - time * flow[i]
                    // Preserve a draw at the final zero-noise step so repeatable
                    // Swift sessions consume the same stream across windows.
                    latentValues[i] = (1 - next) * initial + next * noise.normal()
                }
                guard latentValues.allSatisfy(\.isFinite) else { throw SoulXError.invalidTensor("Denoising produced non-finite values") }
            }
            store.unloadLastBlock()
            restoreMotion(into: &latentValues, plane: plane)
            try check(token)
            onProgress("Decoding generated video")
            let denoised = try SoulXTensor.make(configuration.latentShape, values: latentValues)
            let decodeInput = try SoulXTensor.temporalFirst(denoised, height: configuration.latentHeight, width: configuration.latentWidth)
            let decoded = try store.predict("decoder", inputs: ["latent": decodeInput], output: "frames")
            let visibleCount = min(configuration.usefulFrames, totalFrames - emitted)
            var images: [CGImage] = []
            images.reserveCapacity(visibleCount)
            var feedback = [Float]()
            feedback.reserveCapacity(5 * 3 * configuration.height * configuration.width)
            // History positions 0...4 are never displayed or fed back. Decode
            // still computes them because window-mode causal context needs them.
            for frame in configuration.historyFrames..<configuration.framesPerWindow {
                try check(token)
                let rgb = try SoulXImage.correctedFrame(decoded, index: frame, height: configuration.height, width: configuration.width,
                                                        reference: colors, strength: configuration.colorCorrectionStrength)
                if frame >= configuration.framesPerWindow - configuration.historyFrames { feedback.append(contentsOf: rgb) }
                if images.count < visibleCount { images.append(try SoulXImage.cgImage(rgb, height: configuration.height, width: configuration.width)) }
            }
            store.unload("decoder")
            onProgress("Updating motion history")
            let feedbackInput = try SoulXTensor.make([1, 5, 3, configuration.height, configuration.width], values: feedback)
            let encoded = try store.predict("encoder", inputs: ["frames": feedbackInput], output: "motion")
            _ = try SoulXTensor.checked(encoded, shape: [1, 16, 2, configuration.latentHeight, configuration.latentWidth], name: "motion")
            let nextMotion = SoulXTensor.values(encoded)
            store.unload("encoder")
            guard nextMotion.allSatisfy(\.isFinite) else { throw SoulXError.invalidTensor("Motion encoder produced non-finite values") }
            try check(token)
            motion = nextMotion
            motionTimes = 2
            let duration = Date.timeIntervalSinceReferenceDate - started
            let chunk = SoulXChunk(frames: images, samples: Array(chunkAudio.prefix(visibleCount * samplesPerFrame)),
                                   startFrame: outputFrame, inferenceSeconds: duration, fps: Double(configuration.fps))
            print("SOULX_EVENT window_ready frames=\(visibleCount) seconds=\(duration) generated_fps=\(chunk.generationFPS)")
            outputFrame += visibleCount
            emitted += visibleCount
            emittedChunks += 1
            onProgress("Streaming generated frames")
            try await onChunk(chunk)
            if let maximumChunks, emittedChunks >= maximumChunks { break }
        }
        try check(token)
        completed = true
        onProgress("Generation complete")
    }

    private func check(_ token: UUID) throws {
        try Task.checkCancellation()
        guard generationID == token else { throw CancellationError() }
    }

    private func restoreMotion(into latent: inout [Float], plane: Int) {
        for c in 0..<16 { for t in 0..<motionTimes {
            let source = (c * motionTimes + t) * plane, destination = (c * 9 + t) * plane
            latent.replaceSubrange(destination..<(destination + plane), with: motion[source..<(source + plane)])
        } }
    }
}
