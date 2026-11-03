import Foundation
import CoreML
import Darwin
#if os(iOS)
import os
#endif

public enum SoulXError: LocalizedError {
    case invalidManifest(String)
    case missingModels([String])
    case invalidTensor(String)
    case invalidImage
    case invalidAudio(String)
    case notInitialized
    case modelOutput(String)
    case invalidReferenceCache(String)

    public var errorDescription: String? {
        switch self {
        case .invalidManifest(let reason): return "The model manifest is invalid: \(reason)"
        case .missingModels(let names): return "Import the Core ML model folder first. Missing: \(names.joined(separator: ", "))."
        case .invalidTensor(let reason): return "Model tensor mismatch: \(reason)"
        case .invalidImage: return "The selected image could not be decoded."
        case .invalidAudio(let reason): return "The audio could not be read: \(reason)"
        case .notInitialized: return "Initialize the selected portrait before generating."
        case .modelOutput(let name): return "The Core ML model did not produce \(name)."
        case .invalidReferenceCache(let reason): return "The avatar reference cache is invalid: \(reason)"
        }
    }
}

public struct SoulXManifest: Decodable, Sendable {
    public var schemaVersion: Int
    public var profile: String
    public var height: Int
    public var width: Int
    public var blockCount: Int
    public var framesPerWindow: Int
    public var historyFrames: Int
    public var fps: Int
    public var sampleRate: Int
    public var samplingSteps: [Float]
    public var colorCorrectionStrength: Float
    public var decoderMode: String

    enum CodingKeys: String, CodingKey {
        case schemaVersion = "schema_version", profile, height, width
        case blockCount = "block_count", framesPerWindow = "frames_per_window"
        case historyFrames = "history_frames", fps, sampleRate = "sample_rate"
        case samplingSteps = "sampling_steps", colorCorrectionStrength = "color_correction_strength"
        case decoderMode = "decoder_mode"
    }

    public static func load(from directory: URL) throws -> SoulXManifest {
        let value = try JSONDecoder().decode(Self.self, from: Data(contentsOf: directory.appendingPathComponent("manifest.json")))
        guard value.schemaVersion == 1, value.width >= 16, value.height >= 16,
              value.width % 16 == 0, value.height % 16 == 0,
              (1...30).contains(value.blockCount), value.framesPerWindow == 33,
              value.historyFrames == 5, value.fps == 25, value.sampleRate == 16000,
              value.samplingSteps == [1000, 625, 0], value.decoderMode == "window",
              (0...1).contains(value.colorCorrectionStrength) else {
            throw SoulXError.invalidManifest("This runtime requires PRO, 33/5 frames, 25 FPS, 16-kHz audio, [1000,625,0] sampling and window decode.")
        }
        return value
    }

    public var modelNames: [String] {
        ["reference", "audio", "patch", "conditioning"] + (0..<blockCount).map { String(format: "blocks_%02d", $0) } + ["head", "decoder", "encoder"]
    }
    public var latentHeight: Int { height / 8 }
    public var latentWidth: Int { width / 8 }
    public var latentShape: [Int] { [1, 16, 9, latentHeight, latentWidth] }
    public var usefulFrames: Int { framesPerWindow - historyFrames }
}

public struct SoulXModelStatus: Sendable {
    public let missingModels: [String]
    public let isReady: Bool
    public let profileDescription: String
}

public struct SoulXModelTiming: Encodable, Sendable {
    public fileprivate(set) var requestedComputeUnits = ""
    public fileprivate(set) var calls = 0
    public fileprivate(set) var loads = 0
    public fileprivate(set) var loadSeconds = 0.0
    public fileprivate(set) var inputPreparationSeconds = 0.0
    public fileprivate(set) var predictionSeconds = 0.0
    public fileprivate(set) var prefetchedLoads = 0
    public fileprivate(set) var waitForPrefetchSeconds = 0.0
}

/// Effective compute-unit requests describe Core ML policy, not hardware
/// placement within a mixed policy.
public struct SoulXAudioEncoderStatus: Encodable, Sendable {
    public let requestedComputeUnits: String
    public let effectiveComputeUnits: String
    public let fallbackActive: Bool
    public let retriedThisWindow: Bool
}

/// Optional per-window diagnostics. Memory is sampled at model boundaries.
public struct SoulXWindowProfile: Encodable, Sendable {
    public fileprivate(set) var models = [String: SoulXModelTiming]()
    public fileprivate(set) var audioPackingSeconds = 0.0
    public fileprivate(set) var audioPreparation = "serial"
    public fileprivate(set) var audioEncoderStatus: SoulXAudioEncoderStatus?
    public fileprivate(set) var workerAudioSeconds = 0.0
    public fileprivate(set) var waitForAudioSeconds = 0.0
    public fileprivate(set) var nextAudioPrefetch = "off"
    public fileprivate(set) var audioPrefetchFallback: String?
    public fileprivate(set) var largestPartitionWeightBytes: Int?
    public fileprivate(set) var blockPrefetchFallback: String?
    public fileprivate(set) var peakSampledFootprintBytes: UInt64?
    public fileprivate(set) var minimumSampledHeadroomBytes: Int?
}

/// The loader transfers the new model to one consumer. Only the inference
/// actor subsequently owns the handle, mutates the store or calls prediction.
struct SoulXLoadedBlock: @unchecked Sendable {
    let name: String
    let model: MLModel
    let loadSeconds: Double
}

/// Opt-in Core ML execution options for speed experiments, read from the
/// process arguments so the app, the in-app smoke mode and the Mac runner
/// share them. Defaults leave every model exactly as before.
///   --gpu-low-precision   allowLowPrecisionAccumulationOnGPU (approximate)
///   --fast-prediction     specializationStrategy = .fastPrediction
///   --dit-compute UNITS   compute units for patch/blocks/head only:
///                         cpu | cpu-gpu | cpu-ane | all
enum SoulXExecutionOptions {
    static let arguments = ProcessInfo.processInfo.arguments
    static let lowPrecisionGPU = arguments.contains("--gpu-low-precision")
    static let fastPrediction = arguments.contains("--fast-prediction")
    static let ditComputeUnits: MLComputeUnits? = {
        guard let index = arguments.firstIndex(of: "--dit-compute"), index + 1 < arguments.count else { return nil }
        switch arguments[index + 1] {
        case "cpu": return .cpuOnly
        case "cpu-gpu": return .cpuAndGPU
        case "cpu-ane": return .cpuAndNeuralEngine
        case "all": return .all
        default: return nil
        }
    }()

    static func isTransformer(_ name: String) -> Bool {
        name.hasPrefix("blocks_") || name == "patch" || name == "head"
    }

    static func apply(to configuration: MLModelConfiguration) {
        if lowPrecisionGPU { configuration.allowLowPrecisionAccumulationOnGPU = true }
        if fastPrediction {
            if #available(macOS 14.4, iOS 17.4, *) {
                configuration.optimizationHints.specializationStrategy = .fastPrediction
            }
        }
    }
}

struct SoulXBlockLoadRequest: Sendable {
    let name: String
    let url: URL
    let computeUnits: MLComputeUnits

    func load() throws -> SoulXLoadedBlock {
        try Task.checkCancellation()
        return try autoreleasepool {
            let configuration = MLModelConfiguration()
            configuration.computeUnits = computeUnits
            SoulXExecutionOptions.apply(to: configuration)
            let start = ProcessInfo.processInfo.systemUptime
            let model = try MLModel(contentsOf: url, configuration: configuration)
            try Task.checkCancellation()
            return SoulXLoadedBlock(name: name, model: model, loadSeconds: ProcessInfo.processInfo.systemUptime - start)
        }
    }
}

/// Owns loaded Core ML models on a single inference actor.
final class SoulXModelStore {
    let directory: URL
    let manifest: SoulXManifest
    let computeUnits: MLComputeUnits
    private let audioComputeUnits: MLComputeUnits?
    private var models: [String: MLModel] = [:]
    private let largestPartitionBytes: Int?
    private let collectTimings: Bool
    private var fallbackActive = false
    private var retryingAudioOnCPU = false
    private var retriedAudioThisWindow = false
    private var audioOverlapsDiT = false
    private(set) var windowProfile: SoulXWindowProfile?

    init(directory: URL, computeUnits: MLComputeUnits, audioComputeUnits: MLComputeUnits? = nil,
         collectTimings: Bool = false) throws {
        self.directory = directory
        self.computeUnits = computeUnits
        self.audioComputeUnits = audioComputeUnits
        self.manifest = try SoulXManifest.load(from: directory)
        let count = manifest.blockCount
        let partitionSizes = (0..<count).compactMap {
            Self.partitionSize(String(format: "blocks_%02d", $0), directory: directory)
        }
        self.largestPartitionBytes = partitionSizes.count == count ? partitionSizes.max() : nil
        self.collectTimings = collectTimings
        let missing = manifest.modelNames.filter { Self.sourceURL($0, directory: directory) == nil }
        guard missing.isEmpty else { throw SoulXError.missingModels(missing) }
    }

    static func sourceURL(_ name: String, directory: URL) -> URL? {
        for ext in ["mlmodelc", "mlpackage", "mlmodel"] {
            let file = directory.appendingPathComponent(name).appendingPathExtension(ext)
            if FileManager.default.fileExists(atPath: file.path) { return file }
        }
        return nil
    }

    private static func partitionSize(_ name: String, directory: URL) -> Int? {
        guard let source = sourceURL(name, directory: directory) else { return nil }
        let path = source.pathExtension == "mlmodelc" ? "weights/weight.bin" : "Data/com.apple.CoreML/weights/weight.bin"
        let attributes = try? FileManager.default.attributesOfItem(atPath: source.appendingPathComponent(path).path)
        guard let size = (attributes?[.size] as? NSNumber)?.intValue, size > 0 else { return nil }
        return size
    }

    func unload(_ name: String) { models.removeValue(forKey: name) }
    func beginWindow() {
        retriedAudioThisWindow = false
        audioOverlapsDiT = false
        windowProfile = collectTimings ? SoulXWindowProfile(largestPartitionWeightBytes: largestPartitionBytes) : nil
        windowProfile?.audioEncoderStatus = audioEncoderStatus
    }

    func recordAudioPacking(_ seconds: Double) {
        windowProfile?.audioPackingSeconds += seconds
    }

    var audioEncoderStatus: SoulXAudioEncoderStatus {
        SoulXAudioEncoderStatus(requestedComputeUnits: Self.label(requestedAudioUnits),
                                effectiveComputeUnits: Self.label(effectiveAudioUnits),
                                fallbackActive: fallbackActive, retriedThisWindow: retriedAudioThisWindow)
    }

    func inheritAudioCPUFallback() {
        if !fallbackActive { unload("audio") }
        fallbackActive = true
    }

    func adoptPrefetchedAudioStatus(_ status: SoulXAudioEncoderStatus) {
        if status.fallbackActive { inheritAudioCPUFallback() }
        retriedAudioThisWindow = status.retriedThisWindow
        windowProfile?.audioEncoderStatus = audioEncoderStatus
    }

    /// The projection can return finite context from invalid Wav2Vec features.
    /// Validate at the encoder boundary, then retry only that specific failure.
    func packedAudio(waveform: MLMultiArray) throws -> MLMultiArray {
        defer { windowProfile?.audioEncoderStatus = audioEncoderStatus }
        func encodeAndValidate() throws -> MLMultiArray {
            try autoreleasepool {
                defer { unload("audio") }
                let features = try predict("audio", inputs: ["waveform": waveform], output: "features")
                try Task.checkCancellation()
                let start = collectTimings ? ProcessInfo.processInfo.systemUptime : 0
                defer {
                    if collectTimings { recordAudioPacking(ProcessInfo.processInfo.systemUptime - start) }
                }
                return try SoulXAudio.windows(features)
            }
        }

        do {
            return try encodeAndValidate()
        } catch SoulXAudioFeatureError.nonFiniteFeatures where effectiveAudioUnits != .cpuOnly {
            try Task.checkCancellation()
            retriedAudioThisWindow = true
            retryingAudioOnCPU = true
            defer { retryingAudioOnCPU = false }
            print("SOULX_EVENT audio_encoder_retry reason=nonfinite_features requested=\(Self.label(requestedAudioUnits)) retry=cpuOnly")
            // Failed CPU validation and unrelated model errors propagate.
            let packed = try encodeAndValidate()
            fallbackActive = true
            print("SOULX_EVENT audio_encoder_fallback reason=nonfinite_features requested=\(Self.label(requestedAudioUnits)) effective=cpuOnly")
            return packed
        }
    }

    func recordPrefetchedAudio(models: [String: SoulXModelTiming], packing: Double, worker: Double, wait: Double) {
        windowProfile?.audioPreparation = "lookahead"
        windowProfile?.audioPackingSeconds = packing
        windowProfile?.workerAudioSeconds = worker
        windowProfile?.waitForAudioSeconds = wait
        for (name, timing) in models { windowProfile?.models[name] = timing }
    }

    func nextAudioPrefetch(_ status: String, active: Bool) {
        audioOverlapsDiT = active
        windowProfile?.nextAudioPrefetch = status
    }

    func recordAudioPrefetchFallback(_ error: Error, waited: Double) {
        windowProfile?.audioPreparation = "serial fallback"
        windowProfile?.audioPrefetchFallback = error.localizedDescription
        windowProfile?.waitForAudioSeconds = waited
    }

    func canOverlapAudio(starting: Bool) -> Bool {
        guard manifest.height == 320, manifest.width == 176 else { return false }
        #if os(iOS)
        guard let bytes = largestPartitionBytes, bytes <= 300 * 1_024 * 1_024 else { return false }
        let minimumHeadroom = (starting ? 2_048 : 1_536) * 1_024 * 1_024 + extraPartitionReserve(bytes)
        let maximumFootprint = UInt64(starting ? 1_280 : 2_048) * 1_024 * 1_024
        guard os_proc_available_memory() >= minimumHeadroom,
              let footprint = Self.footprint() else { return false }
        return footprint < maximumFootprint
        #else
        return true
        #endif
    }

    /// Load the current partition before scheduling its successor so the
    /// device headroom check includes the current model's live weights.
    func prepareCurrentBlock(_ name: String) throws {
        guard models[name] == nil else { return }
        try Task.checkCancellation()
        let start = collectTimings ? ProcessInfo.processInfo.systemUptime : 0
        try autoreleasepool { _ = try model(name) }
        if windowProfile != nil {
            var timing = windowProfile?.models[name] ?? SoulXModelTiming()
            timing.loads += 1
            timing.loadSeconds += ProcessInfo.processInfo.systemUptime - start
            windowProfile?.models[name] = timing
            sampleMemory()
        }
    }

    func blockLoadRequest(_ name: String) -> SoulXBlockLoadRequest? {
        guard models[name] == nil, enoughMemoryForPreload() else { return nil }
        let url = directory.appendingPathComponent(name).appendingPathExtension("mlmodelc")
        // Speculation is limited to complete compiled PRO ML Program bundles.
        // Source packages follow normal compilation on the inference actor.
        let manager = FileManager.default
        #if os(iOS)
        // Avoid overlapping audio with two double-size six-layer partitions.
        if audioOverlapsDiT, (largestPartitionBytes ?? Int.max) > 180 * 1_024 * 1_024 { return nil }
        #endif
        for path in ["coremldata.bin", "model.mil", "weights/weight.bin"] {
            let attrs = try? manager.attributesOfItem(atPath: url.appendingPathComponent(path).path)
            guard ((attrs?[.size] as? NSNumber)?.intValue ?? 0) > 0 else { return nil }
        }
        return SoulXBlockLoadRequest(name: name, url: url, computeUnits: units(for: name))
    }

    func installPrefetchedBlock(_ block: SoulXLoadedBlock, expected: String, waitSeconds: Double) throws {
        guard block.name == expected && models[expected] == nil else {
            throw SoulXError.invalidTensor("Unexpected prefetched partition")
        }
        models[expected] = block.model
        if windowProfile != nil {
            var timing = windowProfile?.models[expected] ?? SoulXModelTiming()
            timing.loads += 1
            timing.prefetchedLoads += 1
            timing.loadSeconds += block.loadSeconds
            timing.waitForPrefetchSeconds += waitSeconds
            windowProfile?.models[expected] = timing
            sampleMemory()
        }
    }

    func recordBlockPrefetchFallback(_ error: Error) {
        windowProfile?.blockPrefetchFallback = error.localizedDescription
    }

    private func extraPartitionReserve(_ weightBytes: Int) -> Int {
        // INT8 storage may expand to wider on-device weights during a load.
        2 * max(0, weightBytes - 140 * 1_024 * 1_024)
    }

    private func enoughMemoryForPreload() -> Bool {
        #if os(iOS)
        // On the A14 a separate reference run disappeared at a last observed
        // 3.2 GB footprint. Leave room for a whole additional partition's
        // loading/prediction scratch and enforce a lower process footprint.
        guard let bytes = largestPartitionBytes, bytes <= 300 * 1_024 * 1_024 else { return false }
        let minimumHeadroom = (audioOverlapsDiT ? 2_304 : 1_536) * 1_024 * 1_024 + extraPartitionReserve(bytes)
        guard os_proc_available_memory() >= minimumHeadroom,
              let footprint = Self.footprint() else { return false }
        return footprint < 2 * 1_024 * 1_024 * 1_024
        #else
        return true
        #endif
    }

    func unloadAll() {
        models.removeAll()
        windowProfile = nil
    }

    private static func footprint() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : nil
    }

    private func sampleMemory() {
        guard windowProfile != nil else { return }
        if let footprint = Self.footprint() {
            let old = windowProfile?.peakSampledFootprintBytes ?? 0
            windowProfile?.peakSampledFootprintBytes = max(old, footprint)
        }
        #if os(iOS)
        let headroom = os_proc_available_memory()
        let old = windowProfile?.minimumSampledHeadroomBytes ?? Int.max
        windowProfile?.minimumSampledHeadroomBytes = min(old, headroom)
        #endif
    }

    private var requestedAudioUnits: MLComputeUnits { audioComputeUnits ?? computeUnits }
    private var effectiveAudioUnits: MLComputeUnits {
        (fallbackActive || retryingAudioOnCPU) ? .cpuOnly : requestedAudioUnits
    }

    private func units(for name: String) -> MLComputeUnits {
        if name == "audio" { return effectiveAudioUnits }
        if SoulXExecutionOptions.isTransformer(name), let override = SoulXExecutionOptions.ditComputeUnits { return override }
        return computeUnits
    }

    private static func label(_ units: MLComputeUnits) -> String {
        switch units {
        case .cpuOnly: return "cpuOnly"
        case .cpuAndGPU: return "cpuAndGPU"
        case .cpuAndNeuralEngine: return "cpuAndNeuralEngine"
        case .all: return "all"
        @unknown default: return "unknown"
        }
    }

    private func model(_ name: String) throws -> MLModel {
        if let cached = models[name] { return cached }
        guard let source = Self.sourceURL(name, directory: directory) else { throw SoulXError.missingModels([name]) }
        let url: URL
        if source.pathExtension == "mlmodelc" {
            url = source
        } else {
            // Keep generated binaries separate from portable source packages.
            // Folder imports replace the whole model directory and this cache.
            let attributes = try FileManager.default.attributesOfItem(atPath: source.path)
            let stamp = (attributes[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0
            let root = directory.appendingPathComponent(".compiled", isDirectory: true)
            try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
            let destination = root.appendingPathComponent("\(name)-\(Int(stamp))").appendingPathExtension("mlmodelc")
            if !FileManager.default.fileExists(atPath: destination.path) {
                let temporary = try MLModel.compileModel(at: source)
                try FileManager.default.moveItem(at: temporary, to: destination)
            }
            url = destination
        }
        let configuration = MLModelConfiguration()
        configuration.computeUnits = units(for: name)
        SoulXExecutionOptions.apply(to: configuration)
        let loaded = try MLModel(contentsOf: url, configuration: configuration)
        models[name] = loaded
        return loaded
    }

    func predict(_ name: String, inputs: [String: MLMultiArray], output: String) throws -> MLMultiArray {
        // Bound transient Objective-C/Core ML objects to a single prediction;
        // a whole two-step window can load twenty large block partitions.
        return try autoreleasepool {
            try Task.checkCancellation()
            let wasLoaded = models[name] != nil
            let start = collectTimings ? ProcessInfo.processInfo.systemUptime : 0
            let model = try model(name)
            let prepared = collectTimings ? ProcessInfo.processInfo.systemUptime : 0
            if !wasLoaded { sampleMemory() }
            try Task.checkCancellation()
            var values: [String: MLFeatureValue] = [:]
            for (key, tensor) in inputs {
                guard let constraint = model.modelDescription.inputDescriptionsByName[key]?.multiArrayConstraint else {
                    throw SoulXError.invalidTensor("\(name) has no input named \(key)")
                }
                let actual = tensor.shape.map(\.intValue)
                let expected = constraint.shape.map(\.intValue)
                guard actual == expected else { throw SoulXError.invalidTensor("\(name).\(key): \(actual), expected \(expected)") }
                values[key] = MLFeatureValue(multiArray: try SoulXTensor.cast(tensor, to: constraint.dataType))
            }
            let provider = try MLDictionaryFeatureProvider(dictionary: values)
            let predictStart = collectTimings ? ProcessInfo.processInfo.systemUptime : 0
            let result = try model.prediction(from: provider)
            if windowProfile != nil {
                var timing = windowProfile?.models[name] ?? SoulXModelTiming()
                timing.requestedComputeUnits = Self.label(units(for: name))
                timing.calls += 1
                if !wasLoaded {
                    timing.loads += 1
                    timing.loadSeconds += prepared - start
                }
                timing.inputPreparationSeconds += predictStart - prepared
                timing.predictionSeconds += ProcessInfo.processInfo.systemUptime - predictStart
                windowProfile?.models[name] = timing
                sampleMemory()
            }
            guard let array = result.featureValue(for: output)?.multiArrayValue else { throw SoulXError.modelOutput("\(name).\(output)") }
            return array
        }
    }
}
