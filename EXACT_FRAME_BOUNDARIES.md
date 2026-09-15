# Persistent-call exact frame boundaries

> **GPU provenance — local runs:** NVIDIA GeForce RTX 4070, 12 GB (12,282 MiB visible). This applies to the local SoulX/MuseTalk inference and receiver tests described here; CPU-only checks do not establish GPU performance. Historical or upstream results on other GPUs retain their separate attribution. See [GPU run provenance](docs/research/GPU_RUN_PROVENANCE.md) for dates, evidence, and attribution limits.

Implemented September 9, 2026 in `soulx_rtc/calls.py` and
`soulx_rtc/boundaries.py`. Applies to persistent `/calls` and its WebRTC wall,
not the separate finite `/sessions` API. Existing running processes must restart
to load these changes.

## Contract and flow

1. The approved idle video's first frame is decoded and resized before creating
   the call. Its immutable RGB copy is the canonical completion target. For
   image-only calls, the prepared avatar image is the target.
2. Idle continues while generation prepares. On consumption of the first
   generated speech frame, capture the latest frame returned by the video track.
   Do not use the older model-conditioning snapshot as the display anchor.
3. Replace the first four speech display frames with a cosine fade. Weights are
   0, 0.25, 0.75, 1. The first is exactly the captured RGB; the fourth is exactly
   that slot's generated RGB. Intermediate targets follow generated motion.
   This replaces frames, rather than inserting frames or shifting speech audio.
4. A generation underrun holds the last displayed RGB and does not advance the
   fade. Existing audio/video horizon coupling still bounds audio advancement.
5. Once generated video and useful audio are drained, wait until the final audio
   packet's full 20-ms sender slot has elapsed. Hold the final speech image until
   then. This is sender scheduling, not knowledge of the receiver's jitter buffer.
6. Return over four video frames from the final displayed speech RGB to the
   fixed predecoded canonical RGB. First and last endpoints are exact. No idle
   decoder frames are consumed during this fade. At 25 FPS, endpoints are
   120 ms apart and occupy four 40-ms slots.
7. Queue admission for this peer waits for the canonical endpoint to be emitted.
   Other peers can still render. Reset the source idle index to zero; its decoder
   seeks back on the next idle read. The canonical first frame is deliberately
   repeated once when source playback resumes. No stale prior idle position is
   shown. New queued speech captures whichever frame was most recently sent.

One PeerConnection, stable tracks, and monotonic RTP clocks are retained. Neural
synthetic idle chunks are not treated as separate speech turns. Interrupt/close
cancel pending fades and increment the epoch; an old fade cannot resume after
interruption. Interrupt remains an immediate cancellation, not a guaranteed
graceful return to canonical idle.

## Telemetry and scope of exactness

`GET /calls/{id}` includes `returning_idle`, `boundary_contract`, and the latest
128 `boundary_events`. Each frame event records direction, turn ID, epoch,
index, frame count, 90-kHz video PTS and alpha. Endpoint events include expected
and actual SHA-256 plus an exact RGB equality check. Cancellation is explicit.
Telemetry hashes only endpoints and does not retain full-resolution images.

Exact means **pre-encode sender RGB**. H.264 is lossy; browser display, network
loss and receiver playout cannot be certified by sender hashes. A short fade can
still show ghosting when poses differ significantly. This is a compositor rule,
not a diffusion-model first/last-frame constraint or proof of invisible joins.
The decoder still reads idle on a CPU thread; seek cost may create a late slot.

## MuseTalk comparison

The local MuseTalk reference stages a decoded completion idle first frame in
`scripts/webrtc_tracks.py` (`stage_completion_idle_video`), atomically switches
idle on completion, and waits for final audio packet duration. Its scheduler
also blends source/pose transitions. SoulX now implements the analogous stable
transport, predecoded idle target, sender-frame capture and completion gating.
Unlike MuseTalk's interior cosine weights, these boundary weights explicitly
include zero and one to satisfy exact endpoint equality. This does not establish
equal lip-sync quality, GPU throughput, browser behavior or ten-session capacity.

## Reproducible verification

Run from the repository:

```sh
.venv/bin/python -m pytest tests/test_boundaries.py tests/test_calls.py tests/test_call_recording.py tests/test_wall.py tests/test_codec.py tests/test_rtc.py tests/test_rtc_loopback.py -q --junitxml=benchmarks/boundary-continuity-tests.xml
```

Coverage includes exact endpoint values, epoch cancellation, starvation without
fade advancement, final audio packet timing, native 480x832 MuseTalk idle rewind,
and real local H.264/Opus PeerConnections with both encoder configurations.
The WebRTC cases use a synthetic moving video and the installed MuseTalk
`idle_active_listening.mp4` avatar video, resized to 64x64 to isolate transport.
Two queued speech turns must each emit all four entry and all four exit frames,
in order, before the next turn's entry. Receiver audio/video timestamps must be
strictly increasing, both RTP streams must report zero packet loss, and the call
must survive interruption with one negotiation and clean up successfully.

Speech is supplied by a deterministic fake inference engine in these transport
tests. They do not benchmark the GPU model or demonstrate generated face quality.
Tests using the deployment's MuseTalk fixture explicitly skip if it is absent.
JUnit output records the actual pass/skip/failure counts for each run.

The final regression run passed **42 tests**, with no skips and four existing
Triton deprecation warnings, in 25.69 seconds. A recorder regression additionally
checks that the MP4 has both video and audio streams; a negative test corrupts
endpoint telemetry and verifies that automated validation rejects it.

## Native GPU recording validation

Final evidence: [`boundary-native-gpu-final.json`](benchmarks/boundary-native-gpu-final.json)
and [recorded footage](benchmarks/boundary-native-gpu-final-peer0.mp4).
The server was restarted with the final implementation and remains listening
on its existing authenticated port 1111.

The final run completed two turns in 17.274 seconds of benchmark call time.
All four entry/exit fades passed the seven aggregate telemetry checks, both
receiver RTP streams had zero packet loss and strictly increasing timestamps,
and one negotiation served both turns. The recording contains decoded 480x832
H.264 video and AAC audio. Cleanup passed with zero calls, sessions and GPU states.
Received wire FPS was 23.843; arrival gaps were 52.66 ms at p95 and 103.73 ms
maximum. There were **90 underrun slots**. Wire FPS includes source idle and
held frames; it is not neural generation throughput. This validates boundary
semantics, not consistently real-time generation, imperceptible transitions,
browser/TURN behavior or ten-session capacity. Contact-sheet inspection of the
preceding valid recording confirmed portrait avatar content and mouth/expression
changes; it is not a frame-by-frame perceptual or audio-sync certification.

The actual SoulX server was tested with the same MuseTalk idle avatar at native
480x832, 25-FPS pacing, four diffusion steps, eager optimized real RoPE,
staged memory, and veryfast H.264. Input was the first second of the existing
MuseTalk `kokoro_duration_matrix/multi_sentence.wav` fixture, played in two
consecutive turns over one local PeerConnection. Source hashes and exact
configuration are stored in each benchmark JSON.

The first run (`benchmarks/boundary-native-gpu.json`) completed both turns and
passed all entry/exit endpoints and transport checks, but had 80 underrun slots.
Its old recorder produced audio only; do not use that MP4 as video evidence.

The second run (`benchmarks/boundary-native-gpu-recorded.json`) produced a verified
480x832 H.264 + AAC MP4, passed the four fades, and had 66 underrun slots and
23.00 received wire FPS. It failed the immediate cleanup check: the HTTP call
had disappeared before its asynchronously closing GPU state was released.
A subsequent health check confirmed the state did release. This observation
led to a fix: concurrent close paths now await one shielded cleanup task, and
the call remains registered until release completes. The deterministic delayed
release regression verifies both callers wait. These failed aspects remain in
the evidence rather than being silently overwritten.

The corrected persistent recorder preconfigures the portrait encoder geometry
before either audio or video is muxed, and recording exceptions enter benchmark
errors instead of remaining hidden in a detached recorder task.

To validate a completed benchmark's boundary telemetry:

```sh
.venv/bin/python -m soulx_rtc.verify_boundaries benchmarks/boundary-native-gpu-final.json
```

This verifies every completed turn's entry and exit sequence and exact endpoint
hashes, strictly increasing transition PTS, a shared canonical idle target, no
pending outro, and the benchmark's transport result. Cleanup and underrun metrics
must also be inspected separately. Long tests exceeding the bounded telemetry
window require incremental collection; missing events fail verification.
