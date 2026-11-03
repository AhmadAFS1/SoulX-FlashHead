import Foundation
import CoreGraphics
import CoreML

public struct SoulXChunk: @unchecked Sendable {
    public let frames: [CGImage]
    public let samples: [Float]
    public let startFrame: Int
    public let inferenceSeconds: Double
    public let fps: Double
    public let windowProfile: SoulXWindowProfile?
    public let audioEncoderStatus: SoulXAudioEncoderStatus
    public var generationFPS: Double { Double(frames.count) / max(inferenceSeconds, 0.000001) }
}

/// On-device PRO inference. Model calls and mutable temporal state have a single
/// owner. The awaited chunk callback provides explicit backpressure: the engine
/// never creates an unbounded sequence of full-resolution video buffers.
public actor SoulXEngine {
    public let modelsDirectory: URL
    private let computeUnits: MLComputeUnits
    private let audioComputeUnits: MLComputeUnits?
    private let prefetchAudio: Bool
    private let prefetchBlocks: Bool
    /// Keep Core ML models loaded across passes and windows instead of
    /// unloading each one after its prediction. Opt-in: it trades resident
    /// memory (about 1.4 GB of PRO320 partitions) for fewer loads and warm-up.
    private let keepModelsResident: Bool
    private let collectModelTimings: Bool
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

    public init(modelsDirectory: URL, computeUnits: MLComputeUnits = .all,
                audioComputeUnits: MLComputeUnits? = nil,
                collectModelTimings: Bool = false, prefetchAudio: Bool = false, prefetchBlocks: Bool = false,
                keepModelsResident: Bool = false) {
        self.modelsDirectory = modelsDirectory
        self.computeUnits = computeUnits
        self.audioComputeUnits = audioComputeUnits
        self.prefetchAudio = prefetchAudio
        self.prefetchBlocks = prefetchBlocks || ProcessInfo.processInfo.arguments.contains("--prefetch-blocks")
        self.keepModelsResident = keepModelsResident || ProcessInfo.processInfo.arguments.contains("--keep-models-resident")
        self.collectModelTimings = collectModelTimings || ProcessInfo.processInfo.arguments.contains("--model-timings")
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
        let loaded = try SoulXModelStore(directory: modelsDirectory, computeUnits: computeUnits,
                                         audioComputeUnits: audioComputeUnits,
                                         collectTimings: collectModelTimings)
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
        let supportsOverlap = configuration.height == 320 && configuration.width == 176
        let worker = prefetchAudio && supportsOverlap ? try SoulXAudioLookahead(modelsDirectory: modelsDirectory,
                                                             computeUnits: computeUnits, audioComputeUnits: audioComputeUnits,
                                                             collectModelTimings: collectModelTimings) : nil
        try await withThrowingTaskGroup(of: SoulXPrefetchedAudio.self) { group in
            var pendingAudioFrame: Int?
            var completedAudio: SoulXPrefetchedAudio?
            var prefetchAllowed = worker != nil
            var blockPreloadAllowed = prefetchBlocks && supportsOverlap
            var emitted = 0
            var emittedChunks = 0
            while emitted < totalFrames {
                try check(token)
                store.beginWindow()
                let started = Date.timeIntervalSinceReferenceDate
                let firstSample = emitted * samplesPerFrame
                let chunkAudio = Self.audioSlice(audio, start: firstSample, length: samplesPerChunk)
                Self.append(chunkAudio, to: &audioHistory)
                onProgress("Encoding audio · window \(emitted / configuration.usefulFrames + 1)")
                let context: MLMultiArray
                if let prefetchedFrame = pendingAudioFrame {
                    guard prefetchedFrame == emitted else { throw SoulXError.invalidTensor("Audio lookahead frame mismatch") }
                    let waitStart = ProcessInfo.processInfo.systemUptime
                    let outcome: Result<SoulXPrefetchedAudio, Error>
                    if let completedAudio {
                        outcome = .success(completedAudio)
                    } else if let next = await group.nextResult() {
                        outcome = next
                    } else {
                        throw SoulXError.invalidTensor("Missing prefetched audio context")
                    }
                    let waited = ProcessInfo.processInfo.systemUptime - waitStart
                    try check(token)
                    switch outcome {
                    case .success(let result):
                        context = try result.makeContext()
                        store.adoptPrefetchedAudioStatus(result.audioEncoderStatus)
                        store.recordPrefetchedAudio(models: result.modelTimings, packing: result.packingSeconds,
                                                    worker: result.preparationSeconds, wait: waited)
                    case .failure(let error):
                        prefetchAllowed = false
                        store.recordAudioPrefetchFallback(error, waited: waited)
                        context = try serialAudioContext(store: store)
                    }
                    pendingAudioFrame = nil
                    completedAudio = nil
                } else {
                    context = try serialAudioContext(store: store)
                }
                try check(token)
                let visibleCount = min(configuration.usefulFrames, totalFrames - emitted)
                let nextFrame = emitted + visibleCount
                let hasNextChunk = nextFrame < totalFrames && emittedChunks + 1 < (maximumChunks ?? Int.max)
                if prefetchAllowed, let worker, hasNextChunk {
                    if store.canOverlapAudio(starting: true) {
                        var history = audioHistory
                        Self.append(Self.audioSlice(audio, start: nextFrame * samplesPerFrame, length: samplesPerChunk), to: &history)
                        let snapshot = history
                        let fallback = store.audioEncoderStatus.fallbackActive
                        guard group.addTaskUnlessCancelled(priority: .utility, operation: {
                            try await worker.context(for: snapshot, inheritCPUFallback: fallback)
                        }) else { throw CancellationError() }
                        pendingAudioFrame = nextFrame
                        store.nextAudioPrefetch("scheduled", active: true)
                    } else {
                        store.nextAudioPrefetch("serial: insufficient headroom or unsupported profile", active: false)
                    }
                }
                let plane = configuration.latentHeight * configuration.latentWidth
                var latentValues = noise.normals(count: 16 * 9 * plane)
                for step in 0..<2 {
                    try check(token)
                    restoreMotion(into: &latentValues, plane: plane)
                    let latent = try SoulXTensor.make(configuration.latentShape, values: latentValues)
                    let timestep = try SoulXTensor.make([1], values: [configuration.samplingSteps[step]], type: .float32)
                    let patch = try store.predict("patch", inputs: ["latent": latent, "reference": reference], output: "hidden")
                    let pass = try await runBlocks(patch, context: context, timestep: timestep, step: step, store: store,
                                                   token: token, preload: blockPreloadAllowed, onProgress: onProgress) {
                        if pendingAudioFrame != nil, completedAudio == nil, !store.canOverlapAudio(starting: false) {
                            group.cancelAll()
                            if case .success(let result) = await group.nextResult() {
                                completedAudio = result
                            } else {
                                pendingAudioFrame = nil
                            }
                            prefetchAllowed = false
                            store.nextAudioPrefetch("drained: low headroom", active: false)
                        }
                    }
                    blockPreloadAllowed = pass.preload
                    let prediction = try store.predict("head", inputs: ["hidden": pass.hidden, "timestep": timestep], output: "flow")
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
                restoreMotion(into: &latentValues, plane: plane)
                try check(token)
                onProgress("Decoding generated video")
                let denoised = try SoulXTensor.make(configuration.latentShape, values: latentValues)
                let decodeInput = try SoulXTensor.temporalFirst(denoised, height: configuration.latentHeight, width: configuration.latentWidth)
                let decoded = try store.predict("decoder", inputs: ["latent": decodeInput], output: "frames")
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
                if !keepModelsResident { store.unload("decoder") }
                onProgress("Updating motion history")
                let feedbackInput = try SoulXTensor.make([1, 5, 3, configuration.height, configuration.width], values: feedback)
                let encoded = try store.predict("encoder", inputs: ["frames": feedbackInput], output: "motion")
                _ = try SoulXTensor.checked(encoded, shape: [1, 16, 2, configuration.latentHeight, configuration.latentWidth], name: "motion")
                let nextMotion = SoulXTensor.values(encoded)
                if !keepModelsResident { store.unload("encoder") }
                guard nextMotion.allSatisfy(\.isFinite) else { throw SoulXError.invalidTensor("Motion encoder produced non-finite values") }
                try check(token)
                motion = nextMotion
                motionTimes = 2
                let duration = Date.timeIntervalSinceReferenceDate - started
                let chunk = SoulXChunk(frames: images, samples: Array(chunkAudio.prefix(visibleCount * samplesPerFrame)),
                                       startFrame: outputFrame, inferenceSeconds: duration, fps: Double(configuration.fps),
                                       windowProfile: store.windowProfile, audioEncoderStatus: store.audioEncoderStatus)
                print("SOULX_EVENT window_ready frames=\(visibleCount) seconds=\(duration) generated_fps=\(chunk.generationFPS)")
                outputFrame += visibleCount
                emitted += visibleCount
                emittedChunks += 1
                onProgress("Streaming generated frames")
                try await onChunk(chunk)
                if let maximumChunks, emittedChunks >= maximumChunks { break }
            }
        }
        try check(token)
        completed = true
        onProgress("Generation complete")
    }

    /// A failed speculative load disables further speculation for this stream.
    /// The normal loader still runs when the partition itself is needed.
    private func runBlocks(_ patch: MLMultiArray, context: MLMultiArray, timestep: MLMultiArray, step: Int,
                           store: SoulXModelStore, token: UUID, preload: Bool,
                           onProgress: @Sendable (String) -> Void, beforeBlock: () async throws -> Void)
        async throws -> (hidden: MLMultiArray, preload: Bool) {
        var hidden = patch
        let blockCount = store.manifest.blockCount
        func predict(index: Int, name: String, started: Double) throws {
            try check(token)
            onProgress("Generating · step \(step + 1)/2 · part \(index + 1)/\(blockCount)")
            hidden = try store.predict(name, inputs: ["hidden": hidden, "context": context, "timestep": timestep], output: "hidden_out")
            if !keepModelsResident { store.unload(name) }
            print("SOULX_EVENT partition name=\(name) step=\(step) seconds=\(ProcessInfo.processInfo.systemUptime - started)")
        }

        guard preload else {
            for index in 0..<blockCount {
                try await beforeBlock()
                try predict(index: index, name: String(format: "blocks_%02d", index), started: ProcessInfo.processInfo.systemUptime)
            }
            return (hidden, false)
        }

        var allowed = true
        try await withThrowingTaskGroup(of: SoulXLoadedBlock.self) { blocks in
            var pendingIndex: Int?
            for index in 0..<blockCount {
                try await beforeBlock()
                try check(token)
                let name = String(format: "blocks_%02d", index)
                let started = ProcessInfo.processInfo.systemUptime
                if let pending = pendingIndex {
                    guard pending == index else { throw SoulXError.invalidTensor("Partition lookahead index mismatch") }
                    guard let next = await blocks.nextResult() else { throw SoulXError.invalidTensor("Missing partition lookahead") }
                    try check(token)
                    switch next {
                    case .success(let block):
                        try store.installPrefetchedBlock(block, expected: name, waitSeconds: ProcessInfo.processInfo.systemUptime - started)
                    case .failure(let error):
                        allowed = false
                        store.recordBlockPrefetchFallback(error)
                    }
                    pendingIndex = nil
                }
                let nextIndex = index + 1
                if allowed, nextIndex < blockCount {
                    try store.prepareCurrentBlock(name)
                    let nextName = String(format: "blocks_%02d", nextIndex)
                    if let request = store.blockLoadRequest(nextName) {
                        guard blocks.addTaskUnlessCancelled(priority: .userInitiated, operation: {
                            try request.load()
                        }) else { throw CancellationError() }
                        pendingIndex = nextIndex
                    }
                }
                try predict(index: index, name: name, started: started)
            }
        }
        return (hidden, allowed)
    }

    private static func audioSlice(_ audio: [Float], start: Int, length: Int) -> [Float] {
        var result = [Float](repeating: 0, count: length)
        let available = min(length, audio.count - start)
        if available > 0 { result.replaceSubrange(0..<available, with: audio[start..<(start + available)]) }
        return result
    }

    private static func append(_ samples: [Float], to history: inout [Float]) {
        history.removeFirst(samples.count)
        history.append(contentsOf: samples)
    }

    private func serialAudioContext(store: SoulXModelStore) throws -> MLMultiArray {
        let waveform = try SoulXTensor.make([1, 128000], values: audioHistory, type: .float32)
        let packed = try store.packedAudio(waveform: waveform)
        let context = try store.predict("conditioning", inputs: ["audio": packed], output: "context")
        if !keepModelsResident { store.unload("conditioning") }
        return context
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
