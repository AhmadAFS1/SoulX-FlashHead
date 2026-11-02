import SwiftUI
import UniformTypeIdentifiers

struct ModelSettingsView: View {
    @ObservedObject var studio: StudioModel
    @Environment(\.dismiss) private var dismiss
    @State private var showModelImporter = false

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    Label(studio.modelReady ? "Models installed" : "Models needed", systemImage: studio.modelReady ? "checkmark.circle.fill" : "square.and.arrow.down")
                        .foregroundStyle(studio.modelReady ? Color.green : Color.primary)
                    Text(studio.profileDescription).font(.subheadline).foregroundStyle(.secondary)
                    Button("Import model folder") { showModelImporter = true }
                        .disabled(studio.isBusy)
                    Button("Check installed models") { studio.refreshModels() }
                        .disabled(studio.isBusy)
                } header: { Text("Core ML models") } footer: {
                    Text("Choose an exported profile folder from Files. Importing replaces the active profile. From your Mac, copy the profile’s contents into this app’s Documents/Models/current folder. The dimensions above come from its manifest.")
                }
                if !studio.missingModels.isEmpty {
                    Section("Missing files") {
                        ForEach(studio.missingModels, id: \.self) { name in
                            Text(name).font(.caption).textSelection(.enabled)
                        }
                    }
                }
                Section {
                    Picker("Compute", selection: $studio.computeMode) {
                        ForEach(StudioModel.ComputeMode.allCases) { mode in Text(mode.rawValue).tag(mode) }
                    }
                    .disabled(studio.isBusy)
                    .onChange(of: studio.computeMode) { _, _ in studio.changeComputeMode() }
                } header: { Text("Device execution") } footer: {
                    Text("Automatic lets Core ML choose the available processors. Selecting Neural Engine does not guarantee every operation runs there. Changing this setting requires portrait initialization again.")
                }
                Section("About this build") {
                    LabeledContent("Inference", value: "On device")
                    LabeledContent("Generator", value: "SoulX FlashHead PRO")
                    LabeledContent("Output cadence", value: "25 fps")
                    Text("Generation speed is measured separately. Playback buffers when the model cannot keep up with the audio.")
                        .font(.footnote).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Models & settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar { ToolbarItem(placement: .confirmationAction) { Button("Done") { dismiss() } } }
            .fileImporter(isPresented: $showModelImporter, allowedContentTypes: [.folder]) { result in
                do { studio.importModels(try result.get()); dismiss() }
                catch { studio.fail(error) }
            }
        }
    }
}
