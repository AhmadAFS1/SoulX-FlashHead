import Foundation
import CoreGraphics
import CoreML
import CryptoKit
import ImageIO
import Metal

func failure(_ message: String) -> NSError {
    NSError(domain: "SoulXReferenceExporter", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
}

func sha256File(_ url: URL) throws -> String {
    let file = try FileHandle(forReadingFrom: url)
    defer { try? file.close() }
    var hash = SHA256()
    while let chunk = try file.read(upToCount: 1 << 20), !chunk.isEmpty { hash.update(data: chunk) }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}

func sha256Tree(_ root: URL) throws -> String {
    var directory: ObjCBool = false
    guard FileManager.default.fileExists(atPath: root.path, isDirectory: &directory) else { throw failure("Missing model: \(root.path)") }
    if !directory.boolValue { return try sha256File(root) }
    guard let enumerator = FileManager.default.enumerator(at: root, includingPropertiesForKeys: [.isRegularFileKey]) else { throw failure("Cannot enumerate model") }
    var paths: [URL] = []
    for case let path as URL in enumerator {
        if try path.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true { paths.append(path) }
    }
    var hash = SHA256()
    for path in paths.sorted(by: { $0.path < $1.path }) {
        hash.update(data: Data(path.path.dropFirst(root.path.count + 1).utf8))
        let file = try FileHandle(forReadingFrom: path)
        while let chunk = try file.read(upToCount: 1 << 20), !chunk.isEmpty { hash.update(data: chunk) }
        try file.close()
    }
    return hash.finalize().map { String(format: "%02x", $0) }.joined()
}

do {
    var options: [String: String] = [:]
    let arguments = Array(CommandLine.arguments.dropFirst())
    guard arguments.count % 2 == 0 else { throw failure("Arguments must be --name value pairs") }
    for index in stride(from: 0, to: arguments.count, by: 2) { options[arguments[index]] = arguments[index + 1] }
    func required(_ key: String) throws -> URL {
        guard let value = options[key] else { throw failure("Required: --models DIR --image FILE --output DIR --checkpoint Wan2.1_VAE.pth [--compute cpuOnly|all]") }
        return URL(fileURLWithPath: value).standardizedFileURL
    }
    let models = try required("--models"), imageURL = try required("--image")
    let output = try required("--output"), checkpoint = try required("--checkpoint")
    let compute = options["--compute"] ?? "cpuOnly"
    guard ["cpuOnly", "all"].contains(compute) else { throw failure("--compute must be cpuOnly or all") }
    let manifest = try SoulXManifest.load(from: models)
    let checkpointHash = try sha256File(checkpoint)
    guard checkpointHash == SoulXReferenceCache.wanCheckpointSHA256 else { throw failure("Original Wan checkpoint hash does not match supported weights") }
    guard let source = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
          let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw SoulXError.invalidImage }
    // Matches the app's exact CGImage decoder and SoulXImage.prepare path.
    // Callers must supply the bundled PNG itself, without a second renderer.
    let (input, _) = try SoulXImage.prepare(image, height: manifest.height, width: manifest.width)
    guard let modelURL = SoulXModelStore.sourceURL("reference", directory: models) else { throw failure("Missing reference model") }
    let configuration = MLModelConfiguration()
    configuration.computeUnits = compute == "cpuOnly" ? .cpuOnly : .all
    let compiled = modelURL.pathExtension == "mlmodelc" ? modelURL : try MLModel.compileModel(at: modelURL)
    defer { if compiled != modelURL { try? FileManager.default.removeItem(at: compiled) } }
    let model = try MLModel(contentsOf: compiled, configuration: configuration)
    let start = ProcessInfo.processInfo.systemUptime
    let result = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: input)]))
    guard let latent = result.featureValue(for: "reference")?.multiArrayValue else { throw SoulXError.modelOutput("reference") }
    let inferenceSeconds = ProcessInfo.processInfo.systemUptime - start
    let provenance = [
        "execution": compute == "cpuOnly" ? "Mac CPU-only Core ML prediction; no GPU/ANE inference" : "Mac Core ML .all; actual CPU/GPU/ANE placement unverified",
        "requested_compute_units": compute,
        "gpu_model_present": MTLCreateSystemDefaultDevice()?.name ?? "unverified",
        "physical_system_memory_bytes": String(ProcessInfo.processInfo.physicalMemory),
        "physical_vram": "Apple unified memory; no dedicated VRAM",
        "visible_vram": "Not measured; CPU-only default",
        "hardware_evidence": "MTLDevice.name and ProcessInfo.physicalMemory",
        "driver_runtime": "Core ML supplied by \(ProcessInfo.processInfo.operatingSystemVersionString)",
        "co_resident_load": "Unknown; correctness export, not a throughput benchmark",
        "workload": "One original Wan portrait encoding, \(manifest.height)x\(manifest.width), 33 repeated still frames ->9 latent frames",
        "inference_seconds": String(inferenceSeconds),
        "source_image_file_sha256": try sha256File(imageURL),
        "source_model": modelURL.lastPathComponent,
        "content": "Reference conditioning only. No generated video or DiT output."
    ]
    try SoulXReferenceCache.write(latent: latent, preparedImage: input, manifest: manifest,
                                 sourceImageData: Data(contentsOf: imageURL), sourceWanSHA256: checkpointHash, sourceModelSHA256: sha256Tree(modelURL),
                                 provenance: provenance, directory: output)
    let reloaded = try SoulXReferenceCache.load(metadataURL: output.appendingPathComponent("metadata.json"), manifest: manifest, sourceImageURL: imageURL)
    guard SoulXTensor.values(reloaded) == SoulXTensor.values(latent) else { throw failure("Cache roundtrip differs from original reference output") }
    print("Verified original-Wan reference cache: \(output.path); shape=\(manifest.latentShape); inference_seconds=\(inferenceSeconds)")
} catch {
    FileHandle.standardError.write(Data("Reference cache export failed: \(error.localizedDescription)\n".utf8))
    exit(1)
}
