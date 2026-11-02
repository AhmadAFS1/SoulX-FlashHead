import Foundation
import CoreML

enum SoulXTensor {
    static func make(_ shape: [Int], values: [Float]? = nil, type: MLMultiArrayDataType = .float16) throws -> MLMultiArray {
        let array = try MLMultiArray(shape: shape.map(NSNumber.init), dataType: type)
        if let values {
            guard values.count == array.count else { throw SoulXError.invalidTensor("Expected \(array.count) values, received \(values.count)") }
            write(values, into: array)
        } else {
            memset(array.dataPointer, 0, array.count * bytes(type))
        }
        return array
    }

    static func bytes(_ type: MLMultiArrayDataType) -> Int {
        switch type { case .float16: return 2; case .float32, .int32: return 4; default: return 8 }
    }

    private static func isContiguous(shape: [Int], strides: [Int]) -> Bool {
        var stride = 1
        for i in shape.indices.reversed() {
            if shape[i] > 1 && strides[i] != stride { return false }
            stride *= shape[i]
        }
        return true
    }

    private static func offset(_ logical: Int, shape: [Int], strides: [Int]) -> Int {
        var remainder = logical, result = 0
        for i in shape.indices.reversed() {
            result += (remainder % shape[i]) * strides[i]
            remainder /= shape[i]
        }
        return result
    }

    static func read(_ array: MLMultiArray, at offset: Int) -> Float {
        switch array.dataType {
        case .float16: return Float(array.dataPointer.assumingMemoryBound(to: Float16.self)[offset])
        case .float32: return array.dataPointer.assumingMemoryBound(to: Float.self)[offset]
        case .double: return Float(array.dataPointer.assumingMemoryBound(to: Double.self)[offset])
        default: return Float(array.dataPointer.assumingMemoryBound(to: Int32.self)[offset])
        }
    }

    static func write(_ value: Float, into array: MLMultiArray, at offset: Int) {
        switch array.dataType {
        case .float16: array.dataPointer.assumingMemoryBound(to: Float16.self)[offset] = Float16(value)
        case .float32: array.dataPointer.assumingMemoryBound(to: Float.self)[offset] = value
        case .double: array.dataPointer.assumingMemoryBound(to: Double.self)[offset] = Double(value)
        default: array.dataPointer.assumingMemoryBound(to: Int32.self)[offset] = Int32(value)
        }
    }

    static func values(_ array: MLMultiArray) -> [Float] {
        let shape = array.shape.map(\.intValue)
        let count = array.count
        return array.withUnsafeBytes { bytes in
            let strides = array.strides.map(\.intValue)
            let packed = isContiguous(shape: shape, strides: strides)
            func unpack<T>(_ type: T.Type, _ convert: (T) -> Float) -> [Float] {
                let pointer = bytes.baseAddress!.assumingMemoryBound(to: type)
                if packed { return UnsafeBufferPointer(start: pointer, count: count).map(convert) }
                return (0..<count).map { convert(pointer[offset($0, shape: shape, strides: strides)]) }
            }
            switch array.dataType {
            case .float16: return unpack(Float16.self, Float.init)
            case .float32: return unpack(Float.self) { $0 }
            case .double: return unpack(Double.self, Float.init)
            default: return unpack(Int32.self, Float.init)
            }
        }
    }

    static func write(_ values: [Float], into array: MLMultiArray) {
        let shape = array.shape.map(\.intValue)
        array.withUnsafeMutableBytes { bytes, strides in
            let packed = isContiguous(shape: shape, strides: strides)
            func pack<T>(_ type: T.Type, _ convert: (Float) -> T) {
                let pointer = bytes.baseAddress!.assumingMemoryBound(to: type)
                if packed {
                    for i in values.indices { pointer[i] = convert(values[i]) }
                } else {
                    for i in values.indices {
                        pointer[offset(i, shape: shape, strides: strides)] = convert(values[i])
                    }
                }
            }
            switch array.dataType {
            case .float16: pack(Float16.self, Float16.init)
            case .float32: pack(Float.self) { $0 }
            case .double: pack(Double.self, Double.init)
            default: pack(Int32.self, Int32.init)
            }
        }
    }

    static func cast(_ array: MLMultiArray, to type: MLMultiArrayDataType) throws -> MLMultiArray {
        if array.dataType == type { return array }
        return try make(array.shape.map(\.intValue), values: values(array), type: type)
    }

    static func checked(_ array: MLMultiArray, shape: [Int], name: String) throws -> MLMultiArray {
        guard array.shape.map(\.intValue) == shape else { throw SoulXError.invalidTensor("\(name): \(array.shape), expected \(shape)") }
        return array
    }

    /// NCTHW latent to NTCHW tiny-decoder input; preserves normalized latent values.
    static func temporalFirst(_ latent: MLMultiArray, height: Int, width: Int) throws -> MLMultiArray {
        let output = try make([1, 9, 16, height, width])
        let source = values(latent)
        let plane = height * width
        let pointer = output.dataPointer.assumingMemoryBound(to: Float16.self)
        for t in 0..<9 { for c in 0..<16 { for p in 0..<plane {
            pointer[(t * 16 + c) * plane + p] = Float16(source[(c * 9 + t) * plane + p])
        } } }
        return output
    }
}

/// Portable deterministic noise. A seed establishes repeatability within this
/// runtime, not bit-for-bit equivalence to CUDA's generator. Golden parity tests
/// must pass the same saved noise tensors to both implementations.
struct SoulXNoise {
    var state: UInt64
    private var spare: Float?
    init(seed: UInt64) { state = seed }
    mutating func uniform() -> Double {
        state &+= 0x9e3779b97f4a7c15
        var z = state
        z = (z ^ (z >> 30)) &* 0xbf58476d1ce4e5b9
        z = (z ^ (z >> 27)) &* 0x94d049bb133111eb
        z ^= z >> 31
        return (Double(z >> 11) + 0.5) / 9007199254740992.0
    }
    mutating func normal() -> Float {
        if let value = spare { spare = nil; return value }
        let radius = sqrt(-2 * log(uniform())), angle = 2 * Double.pi * uniform()
        spare = Float(radius * sin(angle))
        return Float(radius * cos(angle))
    }
    mutating func normals(count: Int) -> [Float] { (0..<count).map { _ in normal() } }
}
