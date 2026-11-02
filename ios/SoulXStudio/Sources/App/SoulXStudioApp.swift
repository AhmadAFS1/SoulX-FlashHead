import SwiftUI

@main
struct SoulXStudioApp: App {
    @StateObject private var studio = StudioModel()
    @State private var runningValidation = false
    var body: some Scene {
        WindowGroup {
            StudioView(studio: studio)
                .tint(.orange)
                .preferredColorScheme(.dark)
                .disabled(runningValidation)
                .task {
                    let arguments = ProcessInfo.processInfo.arguments
                    let isProbe = arguments.contains("--probe")
                    guard isProbe || arguments.contains("--smoke") else { return }
                    runningValidation = true
                    UIApplication.shared.isIdleTimerDisabled = true
                    let resultFolder = isProbe ? "ProbeResult" : "SmokeResult"
                    studio.message = "Running local model validation. Results are saved in Documents/\(resultFolder)."
                    if isProbe { await SoulXComponentProbe.runIfRequested() }
                    else { await SoulXSmoke.runIfRequested() }
                    studio.message = "Model validation finished. Open Documents/\(resultFolder)/report.json for the result."
                    runningValidation = false
                    UIApplication.shared.isIdleTimerDisabled = false
                }
        }
    }
}
