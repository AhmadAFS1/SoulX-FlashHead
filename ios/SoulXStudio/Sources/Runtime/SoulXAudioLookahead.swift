import CoreML
import Foundation

/// A completed worker result contains only value types. The inference actor
/// creates its own MLMultiArray when it consumes this exact FP16 context.
struct SoulXPrefetchedAudio: Sendable {
    fileprivate let values: [Float]
    let preparationSeconds: Double
    let modelTimings: [String: SoulXModelTiming]
    let packingSeconds: Double
    let audioEncoderStatus: SoulXAudioEncoderStatus

    func makeContext() throws -> MLMultiArray {
        try SoulXTensor.make([1, 9, 32, 1536], values: values, type: .float16)
    }
}

/// Separates audio preparation from the actor executing the current DiT
/// window. The caller supplies the same 128,000 samples as serial generation.
actor SoulXAudioLookahead {
    private let store: SoulXModelStore
    init(modelsDirectory: URL, computeUnits: MLComputeUnits, audioComputeUnits: MLComputeUnits? = nil,
         collectModelTimings: Bool = false) throws {
        self.store = try SoulXModelStore(directory: modelsDirectory, computeUnits: computeUnits,
                                         audioComputeUnits: audioComputeUnits, collectTimings: collectModelTimings)
    }

    func context(for history: [Float], inheritCPUFallback: Bool = false) throws -> SoulXPrefetchedAudio {
        try Task.checkCancellation()
        guard history.count == 128_000 else { throw SoulXError.invalidTensor("Audio prefetch requires eight seconds of 16 kHz samples") }
        let started = ProcessInfo.processInfo.systemUptime
        store.beginWindow()
        defer { store.unloadAll() }

        if inheritCPUFallback { store.inheritAudioCPUFallback() }
        let packed: MLMultiArray = try autoreleasepool {
            let waveform = try SoulXTensor.make([1, 128_000], values: history, type: .float32)
            return try store.packedAudio(waveform: waveform)
        }

        try Task.checkCancellation()
        let context = try store.predict("conditioning", inputs: ["audio": packed], output: "context")
        _ = try SoulXTensor.checked(context, shape: [1, 9, 32, 1536], name: "conditioning.context")
        guard context.dataType == .float16 else { throw SoulXError.invalidTensor("Audio prefetch expects an FP16 PRO context") }
        let values = SoulXTensor.values(context)
        guard values.allSatisfy(\.isFinite) else { throw SoulXError.invalidTensor("Non-finite prefetched audio context") }
        try Task.checkCancellation()
        let profile = store.windowProfile
        return SoulXPrefetchedAudio(values: values,
                                   preparationSeconds: ProcessInfo.processInfo.systemUptime - started,
                                   modelTimings: profile?.models ?? [:], packingSeconds: profile?.audioPackingSeconds ?? 0,
                                   audioEncoderStatus: store.audioEncoderStatus)
    }
}
