import Foundation
import CoreGraphics
import CoreML

struct SoulXColorStatistics: Sendable {
    let mean: SIMD3<Float>
    let std: SIMD3<Float>
}

enum SoulXImage {
    /// Aspect-fill center crop, corresponding to resize_and_centercrop. Native
    /// interpolation may differ at edges from Pillow and is part of parity QA.
    static func prepare(_ image: CGImage, height: Int, width: Int) throws -> (MLMultiArray, SoulXColorStatistics) {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        let success = rgba.withUnsafeMutableBytes { bytes -> Bool in
            guard let context = CGContext(data: bytes.baseAddress, width: width, height: height,
                                          bitsPerComponent: 8, bytesPerRow: width * 4,
                                          space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                          bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue) else { return false }
            let scale = max(CGFloat(width) / CGFloat(image.width), CGFloat(height) / CGFloat(image.height))
            let w = CGFloat(image.width) * scale, h = CGFloat(image.height) * scale
            context.interpolationQuality = .high
            context.draw(image, in: CGRect(x: (CGFloat(width) - w) / 2, y: (CGFloat(height) - h) / 2, width: w, height: h))
            return true
        }
        guard success else { throw SoulXError.invalidImage }
        let pixels = width * height
        var rgb = [Float](repeating: 0, count: pixels * 3)
        var normalized = rgb
        for p in 0..<pixels { for c in 0..<3 {
            let value = Float(rgba[p * 4 + c]) / 255
            rgb[c * pixels + p] = value
            normalized[c * pixels + p] = value * 2 - 1
        } }
        return (try SoulXTensor.make([1, 3, 1, height, width], values: normalized, type: .float32), statistics(rgb, pixels: pixels))
    }

    static func lab(_ rgb: SIMD3<Float>) -> SIMD3<Float> {
        func linear(_ v: Float) -> Float { v > 0.04045 ? powf((v + 0.055) / 1.055, 2.4) : v / 12.92 }
        let r = linear(rgb.x), g = linear(rgb.y), b = linear(rgb.z)
        let xyz = SIMD3<Float>((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047,
                              0.2126729 * r + 0.7151522 * g + 0.0721750 * b,
                              (0.0193339 * r + 0.1191920 * g + 0.9503041 * b) / 1.08883)
        func f(_ x: Float) -> Float { let v = min(1, max(1e-8, x)); return v > 0.008856 ? powf(v, 1 / 3) : (903.3 * v + 16) / 116 }
        let x = f(xyz.x), y = f(xyz.y), z = f(xyz.z)
        return SIMD3(116 * y - 16, 500 * (x - y), 200 * (y - z))
    }

    static func rgb(_ lab: SIMD3<Float>) -> SIMD3<Float> {
        let fy = (lab.x + 16) / 116, fx = lab.y / 500 + fy, fz = fy - lab.z / 200
        func inverse(_ f: Float) -> Float { f * f * f > 0.008856 ? f * f * f : (116 * f - 16) / 903.3 }
        let x = inverse(fx) * 0.95047
        let y = lab.x > 903.3 * 0.008856 ? fy * fy * fy : lab.x / 903.3
        let z = inverse(fz) * 1.08883
        func gamma(_ v: Float) -> Float { min(1, max(0, v > 0.0031308 ? 1.055 * powf(v, 1 / 2.4) - 0.055 : 12.92 * v)) }
        return SIMD3(gamma(3.2404542 * x - 1.5371385 * y - 0.4985314 * z),
                     gamma(-0.9692660 * x + 1.8760108 * y + 0.0415560 * z),
                     gamma(0.0556434 * x - 0.2040259 * y + 1.0572252 * z))
    }

    static func statistics(_ planar: [Float], pixels: Int) -> SoulXColorStatistics {
        var sum = SIMD3<Double>(repeating: 0), squared = sum
        for p in 0..<pixels {
            let v = lab(SIMD3(planar[p], planar[pixels + p], planar[2 * pixels + p]))
            let d = SIMD3<Double>(Double(v.x), Double(v.y), Double(v.z))
            sum += d; squared += d * d
        }
        let mean = sum / Double(pixels), variance = squared / Double(pixels) - mean * mean
        return SoulXColorStatistics(mean: SIMD3(Float(mean.x), Float(mean.y), Float(mean.z)),
                                    std: SIMD3(Float(sqrt(max(0, variance.x))), Float(sqrt(max(0, variance.y))), Float(sqrt(max(0, variance.z)))))
    }

    static func correctedFrame(_ frames: MLMultiArray, index: Int, height: Int, width: Int,
                               reference: SoulXColorStatistics, strength: Float) throws -> [Float] {
        let plane = height * width
        guard frames.shape.map(\.intValue) == [1, 33, 3, height, width] else { throw SoulXError.invalidTensor("decoder frames must be NTCHW [1,33,3,H,W]") }
        let strides = frames.strides.map(\.intValue)
        var result = [Float](repeating: 0, count: plane * 3)
        for c in 0..<3 { for y in 0..<height { for x in 0..<width {
            let v = SoulXTensor.read(frames, at: index * strides[1] + c * strides[2] + y * strides[3] + x * strides[4])
            guard v.isFinite else { throw SoulXError.invalidTensor("Non-finite pixel in decoder output") }
            result[c * plane + y * width + x] = min(1, max(0, v))
        } } }
        if strength <= 0 { return result }
        let source = statistics(result, pixels: plane)
        let scale = SIMD3(reference.std.x / (source.std.x < 1e-8 ? 1 : source.std.x),
                          reference.std.y / (source.std.y < 1e-8 ? 1 : source.std.y),
                          reference.std.z / (source.std.z < 1e-8 ? 1 : source.std.z))
        for p in 0..<plane {
            let original = SIMD3(result[p], result[plane + p], result[2 * plane + p])
            let transferred = rgb((lab(original) - source.mean) * scale + reference.mean)
            let blended = original * (1 - strength) + transferred * strength
            result[p] = blended.x; result[plane + p] = blended.y; result[2 * plane + p] = blended.z
        }
        return result
    }

    static func cgImage(_ planar: [Float], height: Int, width: Int) throws -> CGImage {
        let plane = height * width
        var rgba = [UInt8](repeating: 255, count: plane * 4)
        for p in 0..<plane { for c in 0..<3 {
            rgba[p * 4 + c] = UInt8(min(255, max(0, planar[c * plane + p] * 255)))
        } }
        let data = Data(rgba)
        guard let provider = CGDataProvider(data: data as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue | CGBitmapInfo.byteOrder32Big.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent) else {
            throw SoulXError.invalidImage
        }
        return image
    }
}
