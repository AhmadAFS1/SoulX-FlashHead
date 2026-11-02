import SwiftUI
import UniformTypeIdentifiers

@MainActor
struct StudioView: View {
    @ObservedObject var studio: StudioModel
    @Environment(\.scenePhase) private var scenePhase
    @State private var showAudioImporter = false
    @State private var showSettings = false

    var body: some View {
        NavigationStack {
            content
            .background(Color(uiColor: .systemBackground))
            .navigationTitle("SoulX Studio")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button { showSettings = true } label: {
                        Image(systemName: "slider.horizontal.3")
                            .frame(minWidth: 44, minHeight: 44)
                    }
                    .accessibilityLabel("Models and settings")
                }
            }
            .sheet(isPresented: $showSettings) { ModelSettingsView(studio: studio) }
            .fileImporter(isPresented: $showAudioImporter, allowedContentTypes: [.audio]) { result in
                do { studio.importAudio(try result.get()) }
                catch { studio.fail(error) }
            }
            .alert("SoulX Studio", isPresented: Binding(get: { studio.error != nil }, set: { if !$0 { studio.error = nil } })) {
                Button("OK", role: .cancel) { studio.error = nil }
            } message: { Text(studio.error ?? "") }
            .onChange(of: scenePhase) { _, phase in
                if phase == .active { studio.refreshModels() }
                if phase == .background && studio.isBusy && !studio.isInstalling { studio.stop() }
            }
        }
    }

    private var content: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                portraitStage
                inputs
                controls
                if studio.generatedFrames > 0 { measurements }
                else if studio.isBusy {
                    Text("\(studio.memoryMB) MB memory · \(studio.thermal)")
                        .font(.caption).foregroundStyle(.secondary).monospacedDigit()
                }
                Text(studio.message)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal, 20)
            .padding(.bottom, 24)
        }
    }

    private var portraitStage: some View {
        ZStack {
            Color(uiColor: .secondarySystemBackground)
            if let image = studio.output ?? studio.portrait {
                Image(uiImage: image)
                    .resizable()
                    .scaledToFit()
                    .accessibilityLabel(studio.output == nil ? "Selected reference portrait" : "Generated SoulX video frame")
            } else {
                VStack(spacing: 14) {
                    Image(systemName: "person.crop.rectangle")
                        .font(.largeTitle)
                        .foregroundStyle(.secondary)
                    Text("Choose an avatar")
                        .font(.headline)
                    Text("Select a bundled portrait below\nto prepare your talking avatar.")
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .padding(24)
            }
            VStack {
                Spacer()
                HStack {
                    Label(stageStatus, systemImage: stageSymbol)
                        .font(.caption.weight(.semibold))
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        .background(.regularMaterial, in: Capsule())
                    Spacer()
                }
                .padding(14)
            }
        }
        .frame(height: 350)
        .frame(maxWidth: .infinity)
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
    }

    private var stageStatus: String {
        if studio.isGenerating { return studio.isBuffering ? "Generating next frames" : "Playing generated video" }
        if studio.output != nil { return "Generated output" }
        if studio.isInitialized { return "Portrait ready" }
        return studio.portrait == nil ? "Local generation" : "Reference image"
    }

    private var stageSymbol: String {
        if studio.isGenerating { return studio.isBuffering ? "hourglass" : "play.fill" }
        return studio.isInitialized ? "checkmark.circle.fill" : "iphone"
    }

    private var inputs: some View {
        VStack(spacing: 16) {
            HStack(spacing: 12) {
                ForEach(StudioAvatar.allCases) { avatar in
                    Button { studio.selectAvatar(avatar) } label: {
                        AvatarCard(avatar: avatar, isSelected: studio.selectedAvatar == avatar)
                    }
                    .accessibilityLabel(avatar.title)
                    .accessibilityAddTraits(studio.selectedAvatar == avatar ? .isSelected : [])
                }
            }
            .disabled(studio.isBusy)
            Button { showAudioImporter = true } label: {
                StudioInputRow(title: studio.audioName == nil ? "Choose audio" : "Change audio", subtitle: studio.audioName ?? "WAV, M4A or another audio file", symbol: "waveform")
            }
            .disabled(studio.isBusy)
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder private var controls: some View {
        if studio.isInstalling {
            HStack(spacing: 14) {
                ProgressView()
                Text("Installing Core ML models…").font(.subheadline.weight(.medium))
            }
            .frame(minHeight: 52)
        } else if !studio.modelReady {
            Button { showSettings = true } label: {
                Label("Install Core ML models", systemImage: "square.and.arrow.down")
                    .foregroundStyle(.black)
                    .frame(maxWidth: .infinity, minHeight: 36)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
        } else if studio.isBusy {
            HStack(spacing: 14) {
                ProgressView()
                Text(studio.isGenerating ? "Generating on this iPhone" : "Preparing portrait…")
                    .font(.subheadline.weight(.medium))
                Spacer(minLength: 4)
                Button("Stop", role: .destructive) { studio.stop() }
                    .buttonStyle(.bordered)
            }
            .frame(minHeight: 52)
        } else if studio.isInitialized {
            Button { studio.start() } label: {
                Label("Generate & play", systemImage: "play.fill")
                    .foregroundStyle(.black)
                    .frame(maxWidth: .infinity, minHeight: 36)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!studio.canGenerate)
        } else {
            Button { studio.initializePortrait() } label: {
                Label("Initialize portrait", systemImage: "sparkles")
                    .foregroundStyle(.black)
                    .frame(maxWidth: .infinity, minHeight: 36)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.large)
            .disabled(!studio.canInitialize)
        }
    }

    private var measurements: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Generation").font(.headline)
            HStack(alignment: .firstTextBaseline) {
                Text(studio.generationFPS, format: .number.precision(.fractionLength(1)))
                    .font(.title2.weight(.semibold)).monospacedDigit()
                Text("frames / second").font(.subheadline).foregroundStyle(.secondary)
                Spacer()
                Text("\(studio.generatedFrames) frames").font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
            Text("\(studio.memoryMB) MB memory · \(studio.thermal)")
                .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            if studio.audioDuration > 0 {
                ProgressView(value: studio.generationProgress)
                    .accessibilityLabel("Audio generated")
                Text("\(min(Double(studio.generatedFrames) / 25, studio.audioDuration), format: .number.precision(.fractionLength(1))) of \(studio.audioDuration, format: .number.precision(.fractionLength(1))) seconds generated")
                    .font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
        }
    }
}

private struct AvatarCard: View {
    let avatar: StudioAvatar
    let isSelected: Bool
    @State private var thumbnail: UIImage?

    var body: some View {
        HStack(spacing: 10) {
            Group {
                if let thumbnail {
                    Image(uiImage: thumbnail).resizable().scaledToFill()
                } else {
                    Image(systemName: "person.crop.rectangle").font(.title2).foregroundStyle(.secondary)
                }
            }
            .frame(width: 48, height: 70)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 6) {
                Text(avatar.title).font(.subheadline.weight(.semibold))
                Image(systemName: isSelected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(isSelected ? Color.orange : Color.secondary)
            }
            Spacer(minLength: 0)
        }
        .padding(10)
        .frame(maxWidth: .infinity, minHeight: 90, alignment: .leading)
        .background(Color(uiColor: .secondarySystemBackground), in: RoundedRectangle(cornerRadius: 12))
        .overlay {
            RoundedRectangle(cornerRadius: 12).strokeBorder(isSelected ? Color.orange : Color.clear, lineWidth: 2)
        }
        .contentShape(RoundedRectangle(cornerRadius: 12))
        .accessibilityElement(children: .ignore)
        .task(id: avatar.id) {
            if let image = try? avatar.image() { thumbnail = UIImage(cgImage: image) }
        }
    }
}

private struct StudioInputRow: View {
    let title: String
    let subtitle: String
    let symbol: String

    var body: some View {
        HStack(spacing: 14) {
            Image(systemName: symbol)
                .font(.title3)
                .foregroundStyle(.orange)
                .frame(width: 26)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).font(.body.weight(.medium))
                Text(subtitle).font(.caption).foregroundStyle(.secondary).lineLimit(2)
            }
            Spacer(minLength: 8)
            Image(systemName: "chevron.right").font(.caption.weight(.semibold)).foregroundStyle(.tertiary)
        }
        .frame(minHeight: 52)
        .contentShape(Rectangle())
    }
}
