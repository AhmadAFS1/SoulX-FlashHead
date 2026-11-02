import Foundation
import CoreML
import CoreGraphics
import AVFoundation

// CPU helper correctness checks. This does not execute a Core ML model, GPU,
// Neural Engine, or iPhone workload. run.sh prints the host hardware evidence.

func check(_ condition: @autoclosure () -> Bool, _ message: String) throws {
    if !condition() { throw NSError(domain: "SoulXValidation", code: 1, userInfo: [NSLocalizedDescriptionKey: message]) }
}

let raw = UnsafeMutablePointer<Float>.allocate(capacity: 6)
for (i, v) in [Float(1), 2, 99, 3, 4, 99].enumerated() { raw[i] = v }
let strided = try MLMultiArray(dataPointer: raw, shape: [2, 2], dataType: .float32, strides: [3, 1], deallocator: { $0.assumingMemoryBound(to: Float.self).deallocate() })
try check(SoulXTensor.values(strided) == [1, 2, 3, 4], "noncontiguous tensor read")
SoulXTensor.write([5, 6, 7, 8], into: strided)
try check(raw[0] == 5 && raw[1] == 6 && raw[3] == 7 && raw[4] == 8 && raw[2] == 99, "strided write")
let half = try SoulXTensor.cast(strided, to: .float16)
try check(SoulXTensor.values(half) == [5, 6, 7, 8], "float16 conversion")
let source = (0..<(16 * 9 * 4)).map(Float.init)
let latent = try SoulXTensor.make([1,16,9,2,2], values: source)
let transposed = try SoulXTensor.temporalFirst(latent, height: 2, width: 2)
let output = SoulXTensor.values(transposed)
for t in 0..<9 { for c in 0..<16 { for p in 0..<4 {
    try check(output[(t * 16 + c) * 4 + p] == source[(c * 9 + t) * 4 + p], "NCTHW -> NTCHW")
} } }
print("PASS strided tensor I/O, FP16 conversion, temporal transpose")

let features = try SoulXTensor.make([1,200,12,768])
for frame in 0..<200 { for p in 0..<(12 * 768) { SoulXTensor.write(Float(frame), into: features, at: frame * 12 * 768 + p) } }
let windows = try SoulXAudio.windows(features)
for frame in 0..<33 { for slot in 0..<5 {
    let expected = Float(min(199, max(0, 167 + frame + slot - 2)))
    let offset = (frame * 5 + slot) * 12 * 768
    try check(SoulXTensor.read(windows, at: offset) == expected && SoulXTensor.read(windows, at: offset + 9215) == expected, "stream audio alignment")
} }
print("PASS all 165 audio-window center/slot endpoints including final clamp")

let colors: [SIMD3<Float>] = [SIMD3(0,0,0), SIMD3(1,1,1), SIMD3(1,0,0), SIMD3(0,1,0), SIMD3(0,0,1), SIMD3(0.2,0.5,0.8)]
var maxError: Float = 0
for color in colors {
    let reconstructed = SoulXImage.rgb(SoulXImage.lab(color))
    maxError = max(maxError, abs(reconstructed.x-color.x), abs(reconstructed.y-color.y), abs(reconstructed.z-color.z))
}
try check(maxError < 0.0001, "Lab roundtrip \(maxError)")
let planar: [Float] = [1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
let image = try SoulXImage.cgImage(planar, height: 2, width: 2)
let (prepared, _) = try SoulXImage.prepare(image, height: 2, width: 2)
let normalized = SoulXTensor.values(prepared)
for i in planar.indices { try check(abs(normalized[i] - (planar[i] * 2 - 1)) < 0.001, "Image row/orientation or channel order mismatch \(i)") }
print("PASS Lab roundtrip maxerror=\(maxError), RGB image preparation orientation/channel order")

let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 48000, channels: 2, interleaved: false)!
let wavURL = FileManager.default.temporaryDirectory.appendingPathComponent("soulx-audio-\(UUID().uuidString).wav")
defer { try? FileManager.default.removeItem(at: wavURL) }
do {
    var settings = format.settings
    settings.removeValue(forKey: AVLinearPCMIsNonInterleaved)
    let file = try AVAudioFile(forWriting: wavURL, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4800)!
    buffer.frameLength = 4800
    for i in 0..<4800 {
        buffer.floatChannelData![0][i] = sin(Float(i) * 2 * .pi * 440 / 48000) * 0.25
        buffer.floatChannelData![1][i] = buffer.floatChannelData![0][i]
    }
    try file.write(from: buffer)
}
let pcm = try SoulXAudio.readMono16k(wavURL)
try check(abs(pcm.count - 1600) <= 1 && pcm.allSatisfy(\.isFinite), "48k stereo ->16k mono sample count \(pcm.count)")
try check(pcm.map { abs($0) }.max()! > 0.20, "audio amplitude")
print("PASS actual AVAudioConverter 48k stereo WAV ->16k mono samples=\(pcm.count)")

var a = SoulXNoise(seed: 42), b = SoulXNoise(seed: 42)
let aa = a.normals(count: 100000), bb = b.normals(count: 100000)
try check(aa == bb, "noise reproducibility")
let mean = aa.reduce(0, +) / Float(aa.count)
let variance = aa.reduce(Float(0)) { $0 + ($1 - mean) * ($1 - mean) } / Float(aa.count)
try check(abs(mean) < 0.02 && abs(variance - 1) < 0.03, "normal distribution")
print("PASS repeatable finite Gaussian stream mean=\(mean) variance=\(variance)")

let cacheDirectory = FileManager.default.temporaryDirectory.appendingPathComponent("soulx-cache-test-\(UUID().uuidString)")
defer { try? FileManager.default.removeItem(at: cacheDirectory) }
try FileManager.default.createDirectory(at: cacheDirectory, withIntermediateDirectories: true)
let cacheImageURL = cacheDirectory.appendingPathComponent("portrait-fixture")
let cacheImageBytes = Data("Raw source-file identity fixture; no model inference".utf8)
try cacheImageBytes.write(to: cacheImageURL)
let cacheManifest = SoulXManifest(schemaVersion: 1, profile: "test", height: 16, width: 16, blockCount: 1,
                                 framesPerWindow: 33, historyFrames: 5, fps: 25, sampleRate: 16000,
                                 samplingSteps: [1000,625,0], colorCorrectionStrength: 1, decoderMode: "window")
let cacheLatent = try SoulXTensor.make(cacheManifest.latentShape, values: (0..<576).map { Float($0) / 100 }, type: .float32)
let cacheImage = try SoulXTensor.make([1,3,1,16,16], type: .float32)
try SoulXReferenceCache.write(latent: cacheLatent, preparedImage: cacheImage, manifest: cacheManifest,
                             sourceImageData: cacheImageBytes, sourceWanSHA256: SoulXReferenceCache.wanCheckpointSHA256,
                             sourceModelSHA256: String(repeating: "0", count: 64), provenance: ["test": "Synthetic format fixture; no inference"], directory: cacheDirectory)
let cacheMetadataURL = cacheDirectory.appendingPathComponent("metadata.json")
let cacheTensorURL = cacheDirectory.appendingPathComponent(SoulXReferenceCache.tensorFilename)
let originalCacheMetadata = try Data(contentsOf: cacheMetadataURL)
let originalCacheTensor = try Data(contentsOf: cacheTensorURL)
func loadCache() throws -> MLMultiArray {
    try SoulXReferenceCache.load(metadataURL: cacheMetadataURL, manifest: cacheManifest, sourceImageURL: cacheImageURL)
}
let cachedRoundtrip = try loadCache()
try check(SoulXTensor.values(cachedRoundtrip) == SoulXTensor.values(cacheLatent), "Reference cache exact float32 little-endian roundtrip")
func rejectCache(_ label: String, _ body: () throws -> Void) throws {
    do { try body() }
    catch SoulXError.invalidReferenceCache { return }
    throw NSError(domain: "SoulXValidation", code: 2, userInfo: [NSLocalizedDescriptionKey: "Cache accepted \(label)"])
}
var metadataObject = try JSONSerialization.jsonObject(with: originalCacheMetadata) as! [String: Any]
metadataObject["shape"] = [1,16,8,2,2]
try JSONSerialization.data(withJSONObject: metadataObject).write(to: cacheMetadataURL)
try rejectCache("wrong shape") { _ = try loadCache() }
try originalCacheMetadata.write(to: cacheMetadataURL)
var corruptCacheTensor = originalCacheTensor
corruptCacheTensor[0] ^= 1
try corruptCacheTensor.write(to: cacheTensorURL)
try rejectCache("wrong tensor hash") { _ = try loadCache() }
try originalCacheTensor.write(to: cacheTensorURL)
try Data("different portrait".utf8).write(to: cacheImageURL)
try rejectCache("wrong portrait") { _ = try loadCache() }
try cacheImageBytes.write(to: cacheImageURL)
metadataObject = try JSONSerialization.jsonObject(with: originalCacheMetadata) as! [String: Any]
metadataObject["source_wan_sha256"] = String(repeating: "0", count: 64)
try JSONSerialization.data(withJSONObject: metadataObject).write(to: cacheMetadataURL)
try rejectCache("wrong source checkpoint") { _ = try loadCache() }
try originalCacheMetadata.write(to: cacheMetadataURL)
var nonfiniteBytes = originalCacheTensor
nonfiniteBytes.withUnsafeMutableBytes { $0.storeBytes(of: Float.infinity.bitPattern.littleEndian, as: UInt32.self) }
try nonfiniteBytes.write(to: cacheTensorURL)
metadataObject = try JSONSerialization.jsonObject(with: originalCacheMetadata) as! [String: Any]
metadataObject["tensor_sha256"] = SoulXReferenceCache.sha256(nonfiniteBytes)
try JSONSerialization.data(withJSONObject: metadataObject).write(to: cacheMetadataURL)
try rejectCache("non-finite tensor with valid checksum") { _ = try loadCache() }
print("PASS reference cache exact roundtrip and rejection of bad shape/hash/portrait/checkpoint/non-finite values")
print("VALIDATION_CLASS=Mac_CPU_helpers_only; no model inference and no iPhone throughput measurement")
