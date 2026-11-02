import CoreGraphics
import Foundation
import ImageIO
import UIKit

enum StudioAvatar: String, CaseIterable, Identifiable {
    case indian
    case girl

    var id: String { rawValue }
    // User-confirmed labels for the provenance-recorded repository portraits.
    var title: String { self == .indian ? "Indian" : "Asian" }
    var portraitURL: URL? {
        Bundle.main.url(forResource: "portrait", withExtension: "png", subdirectory: "Avatars/\(id)")
    }
    var referenceCacheURL: URL? {
        Bundle.main.url(forResource: "metadata", withExtension: "json", subdirectory: "Avatars/\(id)/reference-cache")
    }
    func image() throws -> CGImage {
        guard let url = portraitURL,
              let source = CGImageSourceCreateWithURL(url as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw NSError(domain: "SoulXStudio", code: 3, userInfo: [NSLocalizedDescriptionKey: "The bundled \(title) portrait is unavailable."])
        }
        return image
    }
}
