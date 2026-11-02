import AVFoundation
import CoreML
import SwiftUI
import UIKit

@MainActor
final class StudioModel: ObservableObject {
    @Published var portrait: UIImage?
    @Published var selectedAvatar: StudioAvatar?
    @Published var output: UIImage?
    @Published var audioName: String?
    @Published var message = "Choose a portrait and install the models to begin."
    @Published var error: String?
    @Published var isBusy = false {
        didSet { UIApplication.shared.isIdleTimerDisabled = isBusy }
    }
    @Published var isInitialized = false
    @Published var isGenerating = false
    @Published var isInstalling = false
    @Published var isBuffering = false
    @Published var modelReady = false
    @Published var missingModels: [String] = []
    @Published var profileDescription = "Core ML models required"
    @Published var generatedFrames = 0
    @Published var generationFPS = 0.0
    @Published var audioDuration = 0.0
    @Published var computeMode = ComputeMode.automatic
    @Published var overlapInference = UserDefaults.standard.object(forKey: "SoulXOverlapInference") as? Bool ?? true {
        didSet { UserDefaults.standard.set(overlapInference, forKey: "SoulXOverlapInference") }
    }
    @Published var memoryMB = 0
    @Published var thermal = "Nominal"
    private var audioURL: URL?
    private var engine: SoulXEngine?
    private var work: Task<Void, Never>?
    private var inferenceTotal = 0.0
    private let player = ChunkPlayer()

    enum ComputeMode: String, CaseIterable, Identifiable {
        case automatic = "Automatic"
        case neuralEngine = "Neural Engine + CPU"
        case gpu = "GPU + CPU"
        var id: String { rawValue }
        var units: MLComputeUnits {
            switch self { case .automatic: .all; case .neuralEngine: .cpuAndNeuralEngine; case .gpu: .cpuAndGPU }
        }
    }

    static var documents: URL { FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0] }
    var modelsDirectory: URL { Self.documents.appendingPathComponent("Models/current", isDirectory: true) }
    var canInitialize: Bool { modelReady && portrait != nil && !isBusy }
    var canGenerate: Bool { isInitialized && audioURL != nil && !isBusy }
    var generationProgress: Double { audioDuration > 0 ? min(1, Double(generatedFrames) / 25 / audioDuration) : 0 }

    init() {
        player.onFrame = { [weak self] frame in self?.output = UIImage(cgImage: frame) }
        player.onBuffering = { [weak self] buffering in self?.isBuffering = buffering && self?.isGenerating == true }
        player.onFailure = { [weak self] error in self?.work?.cancel(); self?.fail(error) }
        refreshModels()
        selectAvatar(.indian)
    }

    func refreshModels() {
        let status = SoulXEngine.modelStatus(at: modelsDirectory)
        modelReady = status.isReady
        missingModels = status.missingModels
        profileDescription = status.profileDescription
        sampleResources()
    }

    func selectAvatar(_ avatar: StudioAvatar) {
        guard !isBusy else { return }
        guard selectedAvatar != avatar else { return }
        let source: CGImage
        do { source = try avatar.image() }
        catch { fail(error); return }
        // Preserve the exact decoded source pixels used by reference-cache export.
        portrait = UIImage(cgImage: source)
        selectedAvatar = avatar
        output = nil
        generatedFrames = 0
        generationFPS = 0
        inferenceTotal = 0
        isInitialized = false
        engine = nil
        message = modelReady ? "\(avatar.title) selected. Initialize to prepare its reference." : "\(avatar.title) selected. Install the Core ML models next."
    }

    func importAudio(_ url: URL) {
        guard !isBusy else { return }
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let directory = Self.documents.appendingPathComponent("Audio", isDirectory: true)
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            let destination = directory.appendingPathComponent(UUID().uuidString + "." + url.pathExtension)
            try FileManager.default.copyItem(at: url, to: destination)
            audioURL = destination
            audioName = url.lastPathComponent
            audioDuration = 0
            Task {
                if let duration = try? await AVURLAsset(url: destination).load(.duration), audioURL == destination {
                    let seconds = CMTimeGetSeconds(duration)
                    if seconds.isFinite && seconds > 0 { audioDuration = seconds }
                }
            }
            message = isInitialized ? "Ready to generate your talking portrait." : "Audio selected. Initialize the portrait when ready."
        } catch { fail(error) }
    }

    func initializePortrait() {
        guard canInitialize, let image = portrait?.cgImage else { return }
        guard let avatar = selectedAvatar, let cacheURL = avatar.referenceCacheURL, let sourceURL = avatar.portraitURL else {
            error = "This app build is missing the selected avatar’s reference features. Install a build with a reference cache matching your model profile."
            return
        }
        isBusy = true
        isInitialized = false
        error = nil
        output = nil
        generatedFrames = 0
        generationFPS = 0
        inferenceTotal = 0
        let localEngine = SoulXEngine(modelsDirectory: modelsDirectory, computeUnits: computeMode.units,
                                     prefetchAudio: overlapInference, prefetchBlocks: overlapInference)
        engine = localEngine
        work = Task {
            do {
                try await localEngine.initialize(image: image, referenceCacheURL: cacheURL,
                                                 sourceImageURL: sourceURL, seed: 42) { [weak self] status in
                    Task { @MainActor in self?.message = status; self?.sampleResources() }
                }
                try Task.checkCancellation()
                isInitialized = true
                message = "Portrait initialized. Select audio and start generation."
            } catch is CancellationError { message = "Initialization stopped." }
            catch { fail(error) }
            isBusy = false
            work = nil
            sampleResources()
        }
    }

    func start() {
        guard canGenerate, let engine, let audioURL else { return }
        error = nil
        isBusy = true
        isGenerating = true
        isBuffering = true
        generatedFrames = 0
        generationFPS = 0
        inferenceTotal = 0
        player.stop()
        work = Task {
            do {
                try await engine.generate(audioURL: audioURL, onProgress: { [weak self] status in
                    Task { @MainActor in self?.message = status; self?.sampleResources() }
                }, onChunk: { [weak self] chunk in
                    guard let self else { throw CancellationError() }
                    try await self.receive(chunk)
                })
                message = "Finishing playback…"
                try await player.finish()
                try Task.checkCancellation()
                message = "Generation complete. Initialize again to start a fresh session."
            } catch is CancellationError {
                if error == nil { message = "Generation stopped. Initialize to start a new session." }
            }
            catch { fail(error) }
            player.stop()
            isBusy = false
            isGenerating = false
            isBuffering = false
            isInitialized = false
            work = nil
            sampleResources()
        }
    }

    private func receive(_ chunk: SoulXChunk) async throws {
        try Task.checkCancellation()
        generatedFrames += chunk.frames.count
        inferenceTotal += chunk.inferenceSeconds
        generationFPS = inferenceTotal > 0 ? Double(generatedFrames) / inferenceTotal : 0
        sampleResources()
        try await player.enqueue(chunk)
    }

    func stop() {
        work?.cancel()
        player.stop()
        isInitialized = false
        message = "Stopping after the current model prediction…"
    }

    func changeExecutionSettings() {
        guard !isBusy else { return }
        isInitialized = false
        engine = nil
        message = "Initialize the portrait again to apply your execution settings."
    }

    func importModels(_ source: URL) {
        guard !isBusy else { return }
        isBusy = true
        isInstalling = true
        message = "Installing model files…"
        let destination = modelsDirectory
        work = Task {
            do {
                try await Task.detached(priority: .userInitiated) {
                    let scoped = source.startAccessingSecurityScopedResource()
                    defer { if scoped { source.stopAccessingSecurityScopedResource() } }
                    let fm = FileManager.default
                    let parent = destination.deletingLastPathComponent()
                    try fm.createDirectory(at: parent, withIntermediateDirectories: true)
                    let staging = parent.appendingPathComponent("import-" + UUID().uuidString)
                    defer { try? fm.removeItem(at: staging) }
                    try fm.copyItem(at: source, to: staging)
                    let status = SoulXEngine.modelStatus(at: staging)
                    guard status.isReady else {
                        throw NSError(domain: "SoulXStudio", code: 1, userInfo: [NSLocalizedDescriptionKey: "The selected folder is incomplete. Missing: " + status.missingModels.joined(separator: ", ")])
                    }
                    if fm.fileExists(atPath: destination.path) {
                        _ = try fm.replaceItemAt(destination, withItemAt: staging)
                    } else { try fm.moveItem(at: staging, to: destination) }
                }.value
                engine = nil
                isInitialized = false
                output = nil
                generatedFrames = 0
                generationFPS = 0
                inferenceTotal = 0
                refreshModels()
                message = "Models installed. Choose a portrait and initialize."
            } catch { fail(error) }
            isBusy = false
            isInstalling = false
            work = nil
        }
    }

    func fail(_ failure: Error) { error = failure.localizedDescription; message = "Unable to continue. See the error for details." }

    func sampleResources() {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "Nominal"
        case .fair: thermal = "Warm"
        case .serious: thermal = "Hot"
        case .critical: thermal = "Critical"
        @unknown default: thermal = "Unknown"
        }
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        if result == KERN_SUCCESS { memoryMB = Int(info.phys_footprint / 1_048_576) }
    }
}
