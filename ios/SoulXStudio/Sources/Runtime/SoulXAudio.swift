import Foundation
import AVFoundation
import CoreML

enum SoulXAudio {
    static func readMono16k(_ url: URL) throws -> [Float] {
        let accessed = url.startAccessingSecurityScopedResource()
        defer { if accessed { url.stopAccessingSecurityScopedResource() } }
        let file = try AVAudioFile(forReading: url)
        guard file.length > 0, file.processingFormat.sampleRate > 0,
              let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: file.processingFormat, to: format),
              let input = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 4096),
              let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 4096) else {
            throw SoulXError.invalidAudio("Unsupported or empty audio format")
        }
        var samples: [Float] = []
        var reachedEnd = false
        var readError: Error?
        while true {
            try Task.checkCancellation()
            output.frameLength = 0
            var conversionError: NSError?
            let status = converter.convert(to: output, error: &conversionError) { requested, inputStatus in
                if reachedEnd { inputStatus.pointee = .endOfStream; return nil }
                do {
                    let remaining = file.length - file.framePosition
                    if remaining <= 0 { reachedEnd = true; inputStatus.pointee = .endOfStream; return nil }
                    let count = min(requested, input.frameCapacity, AVAudioFrameCount(min(remaining, Int64(UInt32.max))))
                    try file.read(into: input, frameCount: count)
                    if input.frameLength == 0 { reachedEnd = true; inputStatus.pointee = .endOfStream; return nil }
                    inputStatus.pointee = .haveData
                    return input
                } catch {
                    readError = error; reachedEnd = true; inputStatus.pointee = .endOfStream; return nil
                }
            }
            if let readError { throw readError }
            if let conversionError { throw conversionError }
            if let pointer = output.floatChannelData?[0], output.frameLength > 0 {
                samples.append(contentsOf: UnsafeBufferPointer(start: pointer, count: Int(output.frameLength)))
            }
            if status == .endOfStream { break }
            if status == .error { throw SoulXError.invalidAudio("Audio sample-rate conversion failed") }
            if status == .inputRanDry && reachedEnd && output.frameLength == 0 { break }
        }
        guard !samples.isEmpty, samples.allSatisfy(\.isFinite) else { throw SoulXError.invalidAudio("Empty or non-finite PCM samples") }
        return samples
    }

    /// Exact generate_video.py stream alignment: keep eight seconds including
    /// the new 28-frame slice, take centers 167...199, clamp +/-2 in 0...199.
    static func windows(_ features: MLMultiArray) throws -> MLMultiArray {
        _ = try SoulXTensor.checked(features, shape: [1, 200, 12, 768], name: "audio.features")
        let values = SoulXTensor.values(features)
        let stride = 12 * 768
        var packed = [Float](repeating: 0, count: 33 * 5 * stride)
        for frame in 0..<33 { for slot in 0..<5 {
            let center = min(199, max(0, 167 + frame + slot - 2))
            let destination = (frame * 5 + slot) * stride
            packed.replaceSubrange(destination..<(destination + stride), with: values[(center * stride)..<((center + 1) * stride)])
        } }
        return try SoulXTensor.make([1, 33, 5, 12, 768], values: packed)
    }
}
