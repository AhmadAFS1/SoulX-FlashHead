import CoreGraphics
import CoreML
import Darwin
import Foundation
import ImageIO
import Metal

/// A real compiled-model probe that does not require the rest of the PRO bundle.
/// Reports are persisted before model loading and prediction, including when a
/// device memory termination prevents the operation from returning.
enum SoulXComponentProbe {
    private static let latch = ComponentProbeLatch()

    @discardableResult
    static func runIfRequested() async -> Bool {
        let arguments = ProcessInfo.processInfo.arguments
        guard let argumentIndex = arguments.firstIndex(of: "--probe") else { return false }
        guard latch.begin() else { return true }
        let component = argumentIndex + 1 < arguments.count ? arguments[argumentIndex + 1] : ""
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let directory = documents.appendingPathComponent("ProbeResult", isDirectory: true)
        let units: MLComputeUnits
        let computeLabel: String
        if arguments.contains("--cpu-only") { units = .cpuOnly; computeLabel = "cpuOnly" }
        else if arguments.contains("--cpu-gpu") { units = .cpuAndGPU; computeLabel = "cpuAndGPU" }
        else if arguments.contains("--cpu-ane") { units = .cpuAndNeuralEngine; computeLabel = "cpuAndNeuralEngine" }
        else { units = .all; computeLabel = "all" }
        let recorder = ComponentProbeRecorder(directory: directory, component: component, computeLabel: computeLabel)
        let started = Date.timeIntervalSinceReferenceDate
        defer { recorder.stopSampling() }
        do {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try recorder.stage("Checking component inputs")
            recorder.startSampling()
            guard component == "reference" else { throw probeError("Supported invocation: --probe reference") }
            let modelURL = documents.appendingPathComponent("Probe/reference.mlmodelc", isDirectory: true)
            guard FileManager.default.fileExists(atPath: modelURL.path) else {
                throw probeError("Missing Documents/Probe/reference.mlmodelc")
            }
            let imageURL = documents.appendingPathComponent("Smoke/portrait.png")
            guard let imageSource = CGImageSourceCreateWithURL(imageURL as CFURL, nil),
                  let image = CGImageSourceCreateImageAtIndex(imageSource, 0, nil) else {
                throw probeError("Could not decode Documents/Smoke/portrait.png")
            }
            try recorder.update(["source_image": ["width": image.width, "height": image.height],
                                 "model_path": "Documents/Probe/reference.mlmodelc"])
            try recorder.stage("Preparing real portrait input")
            let prepareStart = Date.timeIntervalSinceReferenceDate
            let (prepared, _) = try SoulXImage.prepare(image, height: 576, width: 320)
            try recorder.update(["preparation_seconds": Date.timeIntervalSinceReferenceDate - prepareStart])
            try Task.checkCancellation()
            let configuration = MLModelConfiguration()
            configuration.computeUnits = units
            try recorder.stage("Loading compiled reference model")
            let loadStart = Date.timeIntervalSinceReferenceDate
            let model = try MLModel(contentsOf: modelURL, configuration: configuration)
            try recorder.update(["load_seconds": Date.timeIntervalSinceReferenceDate - loadStart,
                                 "model_loaded": true])
            try recorder.stage("Compiled reference model loaded")
            guard let constraint = model.modelDescription.inputDescriptionsByName["image"]?.multiArrayConstraint else {
                throw probeError("Reference model must have a multi-array input named image")
            }
            let expectedInput = [1, 3, 1, 576, 320]
            guard constraint.shape.map(\.intValue) == expectedInput else {
                throw probeError("Reference input shape \(constraint.shape) does not match \(expectedInput)")
            }
            let input = try SoulXTensor.cast(prepared, to: constraint.dataType)
            try recorder.update(["input": ["shape": input.shape.map(\.intValue), "data_type": String(describing: input.dataType)]])
            let provider = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: input)])
            try Task.checkCancellation()
            try recorder.update(["inference_started": true])
            try recorder.stage("Predicting reference from real portrait")
            let predictionStart = Date.timeIntervalSinceReferenceDate
            let prediction = try await model.prediction(from: provider)
            let predictionSeconds = Date.timeIntervalSinceReferenceDate - predictionStart
            try recorder.update(["prediction_seconds": predictionSeconds, "inference_completed": true])
            try recorder.stage("Prediction returned; validating output")
            guard let reference = prediction.featureValue(for: "reference")?.multiArrayValue else {
                throw probeError("Reference model did not return a multi-array named reference")
            }
            let shape = reference.shape.map(\.intValue)
            let expected = [1, 16, 9, 72, 40]
            try recorder.update(["output_shape": shape, "output_data_type": String(describing: reference.dataType)])
            guard shape == expected else { throw probeError("Reference output shape \(shape) does not match \(expected)") }
            var minimum = Float.infinity, maximum = -Float.infinity
            var total = 0.0, nonfinite = 0
            for value in SoulXTensor.values(reference) {
                if value.isFinite {
                    minimum = min(minimum, value)
                    maximum = max(maximum, value)
                    total += Double(value)
                } else { nonfinite += 1 }
            }
            try recorder.update(["output_element_count": reference.count, "output_nonfinite_count": nonfinite])
            guard nonfinite == 0 else { throw probeError("Reference output contains \(nonfinite) non-finite elements") }
            try recorder.update(["output_statistics": ["min": Double(minimum), "max": Double(maximum), "mean": total / Double(reference.count)],
                                 "status": "passed", "total_seconds": Date.timeIntervalSinceReferenceDate - started])
            try recorder.stage("Reference prediction passed finite and shape checks")
        } catch {
            try? recorder.update(["status": "failed", "error": String(describing: error),
                                  "localized_error": error.localizedDescription,
                                  "total_seconds": Date.timeIntervalSinceReferenceDate - started])
            try? recorder.stage("Component probe failed")
            print("SOULX_PROBE failed: \(error)")
        }
        return true
    }

    private static func probeError(_ message: String) -> NSError {
        NSError(domain: "SoulXComponentProbe", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
}

private final class ComponentProbeLatch: @unchecked Sendable {
    private let lock = NSLock()
    private var started = false
    func begin() -> Bool {
        lock.lock(); defer { lock.unlock() }
        guard !started else { return false }
        started = true
        return true
    }
}

private final class ComponentProbeRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private let reportURL: URL
    private let started = Date.timeIntervalSinceReferenceDate
    private var report: [String: Any]
    private var stages: [[String: Any]] = []
    private var largestFootprint: UInt64 = 0
    private var sampler: DispatchSourceTimer?

    init(directory: URL, component: String, computeLabel: String) {
        reportURL = directory.appendingPathComponent("report.json")
        var machine = utsname()
        uname(&machine)
        let capacity = MemoryLayout.size(ofValue: machine.machine)
        let identifier = withUnsafePointer(to: &machine.machine) {
            $0.withMemoryRebound(to: CChar.self, capacity: capacity) { String(cString: $0) }
        }
        #if targetEnvironment(simulator)
        let environment = "iOS Simulator"
        #else
        let environment = "Physical Apple device"
        #endif
        report = [
            "schema_version": 1,
            "status": "running",
            "date_utc": ISO8601DateFormatter().string(from: Date()),
            "component": component,
            "execution_environment": environment,
            "execution_classification": computeLabel == "cpuOnly" ? "Requested CPU-only Core ML component prediction; inspect inference_started/completed" : "Requested Core ML component prediction on \(environment); actual CPU/GPU/ANE placement unverified; inspect inference_started/completed",
            "requested_compute_units": computeLabel,
            "model_loaded": false,
            "inference_started": false,
            "inference_completed": false,
            "workload": "Requested PRO reference encoder, one real portrait, H576 W320; component-only validation",
            "parity": "Finite and shape checks do not establish CUDA parity, generated video quality, or end-to-end FPS",
            "hardware": [
                "machine": identifier,
                "gpu_model": MTLCreateSystemDefaultDevice()?.name ?? "unverified; Metal device unavailable",
                "gpu_evidence": "MTLCreateSystemDefaultDevice()?.name; machine from uname",
                "physical_system_memory_bytes": ProcessInfo.processInfo.physicalMemory,
                "physical_vram": "Unified system memory; no dedicated VRAM",
                "visible_vram": "Not exposed by iOS; process memory limit unknown",
                "driver_runtime": "Core ML / Metal supplied by \(ProcessInfo.processInfo.operatingSystemVersionString)",
                "co_resident_load": "Unknown; no other-app process inspection"
            ]
        ]
    }

    func startSampling() {
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue.global(qos: .utility))
        timer.schedule(deadline: .now() + 1, repeating: 1)
        timer.setEventHandler { [weak self] in try? self?.update([:]) }
        sampler = timer
        timer.resume()
    }

    func stopSampling() { sampler?.cancel(); sampler = nil }

    func update(_ values: [String: Any]) throws {
        lock.lock(); defer { lock.unlock() }
        report.merge(values) { _, incoming in incoming }
        try persist()
    }

    func stage(_ message: String) throws {
        lock.lock(); defer { lock.unlock() }
        let elapsed = Date.timeIntervalSinceReferenceDate - started
        stages.append(["elapsed_seconds": elapsed, "message": message,
                       "process_footprint_bytes": Self.footprint() ?? 0,
                       "thermal_state": ProcessInfo.processInfo.thermalState.rawValue])
        report["last_stage"] = message
        report["stages"] = stages
        try persist()
        print("SOULX_PROBE \(message) elapsed_seconds=\(elapsed)")
    }

    private func persist() throws {
        if let footprint = Self.footprint() {
            largestFootprint = max(largestFootprint, footprint)
            report["process_footprint_bytes"] = footprint
            report["peak_observed_footprint_bytes"] = largestFootprint
            report["memory_evidence"] = "task_info TASK_VM_INFO phys_footprint, sampled at stages and every second; not guaranteed instantaneous peak"
        }
        report["thermal_state"] = ProcessInfo.processInfo.thermalState.rawValue
        report["last_saved_utc"] = ISO8601DateFormatter().string(from: Date())
        let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: reportURL, options: .atomic)
    }

    private static func footprint() -> UInt64? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : nil
    }
}
