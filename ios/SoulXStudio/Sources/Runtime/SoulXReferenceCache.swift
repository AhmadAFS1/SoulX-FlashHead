import Foundation
import CoreML
import CryptoKit

/// Original Wan conditioning features for a fixed portrait. This format never
/// contains generated video, audio features, motion history, or DiT results.
enum SoulXReferenceCache {
    static let wanCheckpointSHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
    static let tensorFilename = "reference.f32le"

    struct Metadata: Codable {
        let schemaVersion: Int
        let profile: String
        let height: Int
        let width: Int
        let shape: [Int]
        let dtype: String
        let tensorFile: String
        let tensorSHA256: String
        let preparedImageSHA256: String
        let sourceImageSHA256: String
        let sourceWanSHA256: String
        let sourceModelSHA256: String
        let createdUTC: String
        let provenance: [String: String]

        enum CodingKeys: String, CodingKey {
            case schemaVersion = "schema_version", profile, height, width, shape, dtype
            case tensorFile = "tensor_file", tensorSHA256 = "tensor_sha256"
            case preparedImageSHA256 = "prepared_image_sha256", sourceWanSHA256 = "source_wan_sha256"
            case sourceImageSHA256 = "source_image_sha256"
            case sourceModelSHA256 = "source_model_sha256", createdUTC = "created_utc", provenance
        }
    }

    static func sha256(_ data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }

    static func data(_ values: [Float]) -> Data {
        let words = values.map { $0.bitPattern.littleEndian }
        return words.withUnsafeBytes { Data($0) }
    }

    static func load(metadataURL: URL, manifest: SoulXManifest, sourceImageURL: URL) throws -> MLMultiArray {
        let attributes = try FileManager.default.attributesOfItem(atPath: metadataURL.path)
        guard ((attributes[.size] as? NSNumber)?.intValue ?? Int.max) < 131072 else {
            throw SoulXError.invalidReferenceCache("Metadata is too large")
        }
        let metadata = try JSONDecoder().decode(Metadata.self, from: Data(contentsOf: metadataURL))
        guard metadata.schemaVersion == 1, metadata.dtype == "float32-le", metadata.tensorFile == tensorFilename else {
            throw SoulXError.invalidReferenceCache("Unsupported schema, tensor name, or encoding")
        }
        guard metadata.height == manifest.height, metadata.width == manifest.width,
              metadata.shape == manifest.latentShape, metadata.profile == manifest.profile else {
            throw SoulXError.invalidReferenceCache("Cache shape/profile does not match the installed model profile")
        }
        guard metadata.sourceWanSHA256 == wanCheckpointSHA256 else {
            throw SoulXError.invalidReferenceCache("Cache was not produced with the supported original Wan checkpoint")
        }
        // Validate the exact bundled source file. Core Graphics resizing can
        // vary slightly across macOS/iOS; the prepared tensor hash is retained
        // as export provenance rather than used as a cross-platform gate.
        guard sha256(try Data(contentsOf: sourceImageURL)) == metadata.sourceImageSHA256 else {
            throw SoulXError.invalidReferenceCache("Cache belongs to a different portrait file")
        }
        let tensorURL = metadataURL.deletingLastPathComponent().appendingPathComponent(tensorFilename)
        let count = manifest.latentShape.reduce(1, *)
        let tensorAttributes = try FileManager.default.attributesOfItem(atPath: tensorURL.path)
        guard (tensorAttributes[.size] as? NSNumber)?.intValue == count * 4 else {
            throw SoulXError.invalidReferenceCache("Tensor byte count does not match its shape")
        }
        let bytes = try Data(contentsOf: tensorURL)
        guard sha256(bytes) == metadata.tensorSHA256 else {
            throw SoulXError.invalidReferenceCache("Tensor SHA-256 does not match; replace the cache")
        }
        let values = bytes.withUnsafeBytes { buffer in
            (0..<count).map { Float(bitPattern: UInt32(littleEndian: buffer.loadUnaligned(fromByteOffset: $0 * 4, as: UInt32.self))) }
        }
        guard values.allSatisfy(\.isFinite) else { throw SoulXError.invalidReferenceCache("Tensor contains non-finite features") }
        return try SoulXTensor.make(manifest.latentShape, values: values, type: .float32)
    }

    static func write(latent: MLMultiArray, preparedImage: MLMultiArray, manifest: SoulXManifest,
                      sourceImageData: Data, sourceWanSHA256: String, sourceModelSHA256: String,
                      provenance: [String: String], directory: URL) throws {
        guard sourceWanSHA256 == wanCheckpointSHA256 else { throw SoulXError.invalidReferenceCache("Unexpected source checkpoint SHA-256") }
        _ = try SoulXTensor.checked(latent, shape: manifest.latentShape, name: "cached reference")
        _ = try SoulXTensor.checked(preparedImage, shape: [1, 3, 1, manifest.height, manifest.width], name: "cache portrait input")
        let values = SoulXTensor.values(latent)
        guard values.allSatisfy(\.isFinite) else { throw SoulXError.invalidReferenceCache("Cannot save non-finite features") }
        let bytes = data(values)
        let metadata = Metadata(schemaVersion: 1, profile: manifest.profile, height: manifest.height, width: manifest.width,
                                shape: manifest.latentShape, dtype: "float32-le", tensorFile: tensorFilename,
                                tensorSHA256: sha256(bytes), preparedImageSHA256: sha256(data(SoulXTensor.values(preparedImage))),
                                sourceImageSHA256: sha256(sourceImageData),
                                sourceWanSHA256: sourceWanSHA256, sourceModelSHA256: sourceModelSHA256,
                                createdUTC: ISO8601DateFormatter().string(from: Date()), provenance: provenance)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try bytes.write(to: directory.appendingPathComponent(tensorFilename), options: .atomic)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(metadata).write(to: directory.appendingPathComponent("metadata.json"), options: .atomic)
    }
}
