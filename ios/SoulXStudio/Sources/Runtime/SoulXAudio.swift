import Foundation
import AVFoundation
import CoreML

enum SoulXAudioFeatureError: LocalizedError {
    case nonFiniteFeatures

    var errorDescription: String? { "The audio encoder produced non-finite conditioning features." }
}

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
        let layers = 12, channels = 768
        let source = try SoulXTensor.cast(features, to: .float16)
        let output = try MLMultiArray(shape: [1, 33, 5, 12, 768], dataType: .float16)

        // Production features already contain Float16. Copy each selected time
        // slice once per slot without materializing the other 165 time slices
        // or converting the entire eight-second history through Float.
        try source.withUnsafeBytes { bytes in
            let input = bytes.bindMemory(to: Float16.self)
            let strides = source.strides.map(\.intValue)
            // A backend can emit NaNs which the next layer maps to a finite,
            // identical context for every speech input. Check each used slice
            // before projection so the caller can retry the encoder on CPU.
            for center in 165..<200 {
                let start = center * strides[1]
                if strides[2] == channels && strides[3] == 1 {
                    guard input[start..<(start + layers * channels)].allSatisfy(\.isFinite) else {
                        throw SoulXAudioFeatureError.nonFiniteFeatures
                    }
                } else {
                    for layer in 0..<layers { for channel in 0..<channels {
                        guard input[start + layer * strides[2] + channel * strides[3]].isFinite else {
                            throw SoulXAudioFeatureError.nonFiniteFeatures
                        }
                    } }
                }
            }
            output.withUnsafeMutableBytes { outputBytes, outputStrides in
                let target = outputBytes.bindMemory(to: Float16.self)
                let contiguousSlice = strides[2] == channels && strides[3] == 1 &&
                    outputStrides[3] == channels && outputStrides[4] == 1
                for frame in 0..<33 {
                    for slot in 0..<5 {
                        let center = min(199, 167 + frame + slot - 2)
                        let readOffset = center * strides[1]
                        let writeOffset = frame * outputStrides[1] + slot * outputStrides[2]
                        if contiguousSlice {
                            memcpy(target.baseAddress! + writeOffset, input.baseAddress! + readOffset,
                                   layers * channels * MemoryLayout<Float16>.stride)
                        } else {
                            for layer in 0..<layers {
                                let from = readOffset + layer * strides[2]
                                let to = writeOffset + layer * outputStrides[3]
                                for channel in 0..<channels {
                                    target[to + channel * outputStrides[4]] = input[from + channel * strides[3]]
                                }
                            }
                        }
                    }
                }
            }
        }
        return output
    }
}
