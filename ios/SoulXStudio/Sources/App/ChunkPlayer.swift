import AVFoundation
import CoreGraphics
import Foundation

/// Bounded local playback. The audio clock owns frame pacing; empty queues produce buffering.
@MainActor
final class ChunkPlayer {
    private var queued: [SoulXChunk] = []
    private var playback: Task<Void, Never>?
    private let audioEngine = AVAudioEngine()
    private let audioNode = AVAudioPlayerNode()
    private var configured = false
    private var generation: UInt64 = 0
    var onFrame: ((CGImage) -> Void)?
    var onBuffering: ((Bool) -> Void)?
    var onFailure: ((Error) -> Void)?
    var playedFrames = 0

    func enqueue(_ chunk: SoulXChunk) async throws {
        let epoch = generation
        while queued.count >= 2 {
            try Task.checkCancellation()
            guard epoch == generation else { throw CancellationError() }
            try await Task.sleep(for: .milliseconds(25))
        }
        guard epoch == generation else { throw CancellationError() }
        queued.append(chunk)
        if playback == nil {
            playback = Task { [weak self] in await self?.drain(epoch: epoch) }
        }
    }

    func finish() async throws {
        while playback != nil || !queued.isEmpty {
            try Task.checkCancellation()
            try await Task.sleep(for: .milliseconds(30))
        }
    }

    func stop() {
        generation &+= 1
        playback?.cancel()
        playback = nil
        queued.removeAll()
        audioNode.stop()
        audioEngine.stop()
        playedFrames = 0
    }

    private func configure() throws {
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setActive(true)
        if !configured {
            let format = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1)!
            audioEngine.attach(audioNode)
            audioEngine.connect(audioNode, to: audioEngine.mainMixerNode, format: format)
            configured = true
        }
        if !audioEngine.isRunning { try audioEngine.start() }
    }

    private func schedule(_ buffer: AVAudioPCMBuffer) {
        // Scheduling must return before play(); awaiting the completion overload
        // here would wait for audio that has not started.
        audioNode.scheduleBuffer(buffer, completionHandler: nil)
    }

    private func drain(epoch: UInt64) async {
        defer { if generation == epoch { playback = nil; onBuffering?(true) } }
        do {
            try configure()
            while !queued.isEmpty && generation == epoch {
                try Task.checkCancellation()
                let chunk = queued.removeFirst()
                guard !chunk.frames.isEmpty else { continue }
                onBuffering?(false)
                let fps = chunk.fps
                let duration = Double(chunk.frames.count) / fps
                let frameCount = max(1, Int((duration * 16_000).rounded()))
                let format = AVAudioFormat(standardFormatWithSampleRate: 16_000, channels: 1)!
                guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frameCount)),
                      let samples = buffer.floatChannelData?[0] else { throw CocoaError(.fileReadCorruptFile) }
                buffer.frameLength = AVAudioFrameCount(frameCount)
                for i in 0..<frameCount { samples[i] = i < chunk.samples.count ? chunk.samples[i] : 0 }
                // Stop between complete chunks to reset player-time origin. No artificial time elapses during buffering.
                audioNode.stop()
                schedule(buffer)
                audioNode.play()
                var displayed = -1
                let fallbackStart = ContinuousClock.now
                while generation == epoch {
                    try Task.checkCancellation()
                    guard audioEngine.isRunning && audioNode.isPlaying else {
                        throw NSError(domain: "SoulXStudio", code: 2, userInfo: [NSLocalizedDescriptionKey: "Audio playback was interrupted. Initialize the portrait to start a fresh session."])
                    }
                    let elapsed: Double
                    if let nodeTime = audioNode.lastRenderTime, let playerTime = audioNode.playerTime(forNodeTime: nodeTime) {
                        elapsed = Double(max(0, playerTime.sampleTime)) / playerTime.sampleRate
                    } else {
                        let delta = fallbackStart.duration(to: .now)
                        elapsed = Double(delta.components.seconds) + Double(delta.components.attoseconds) / 1e18
                    }
                    if elapsed >= duration { break }
                    let index = min(chunk.frames.count - 1, max(0, Int(elapsed * fps)))
                    if index != displayed {
                        displayed = index
                        onFrame?(chunk.frames[index])
                        playedFrames += 1
                    }
                    try await Task.sleep(for: .milliseconds(8))
                }
            }
            if generation == epoch { audioNode.stop() }
        } catch is CancellationError {
            if generation == epoch { audioNode.stop() }
        } catch {
            if generation == epoch {
                audioNode.stop()
                queued.removeAll()
                onFailure?(error)
            }
        }
    }
}
