import Foundation
import CoreML

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

/// Actor-confined loader. Only one DiT partition is retained at a time; this
/// bounds resident weights on A14. Compilation is persisted separately from
/// the portable .mlpackages and subsequent predictions reuse those binaries.
final class SoulXModelStore {
    let directory: URL
    let manifest: SoulXManifest
    let computeUnits: MLComputeUnits
    private var models: [String: MLModel] = [:]
    private var blockName: String?

    init(directory: URL, computeUnits: MLComputeUnits) throws {
        self.directory = directory
        self.computeUnits = computeUnits
        self.manifest = try SoulXManifest.load(from: directory)
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

    func unload(_ name: String) { models.removeValue(forKey: name) }
    func unloadLastBlock() {
        if let blockName { unload(blockName) }
        blockName = nil
    }
    func unloadAll() { models.removeAll(); blockName = nil }

    private func model(_ name: String) throws -> MLModel {
        if let cached = models[name] { return cached }
        if name.hasPrefix("blocks_"), let old = blockName, old != name { unload(old) }
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
        configuration.computeUnits = computeUnits
        let loaded = try MLModel(contentsOf: url, configuration: configuration)
        models[name] = loaded
        if name.hasPrefix("blocks_") { blockName = name }
        return loaded
    }

    func predict(_ name: String, inputs: [String: MLMultiArray], output: String) throws -> MLMultiArray {
        // Bound transient Objective-C/Core ML objects to a single prediction;
        // a whole two-step window can load twenty large block partitions.
        return try autoreleasepool {
            try Task.checkCancellation()
            let model = try model(name)
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
            let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: values))
            guard let array = result.featureValue(for: output)?.multiArrayValue else { throw SoulXError.modelOutput("\(name).\(output)") }
            return array
        }
    }
}
