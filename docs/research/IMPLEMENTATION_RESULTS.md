# Native portrait generation and persistent calls: implementation evidence

Implementation started 2026-09-06 from `b3c47de` and continued on September 7. Experiments below ran on the local RTX 4070 with 12,282 MiB visible. An unrelated OmniVoice process initially retained approximately 6,150 MiB, rising to 6,510 MiB during later portrait TensorRT tests; it was not stopped. This is a constrained, co-resident deployment test, not an isolated whole-GPU capacity claim.

## Decision

The repository now implements native rectangular generation, cached conditioning, optional exact-shape TensorRT partitions, and persistent H264/Opus call/turn/interrupt APIs. It produces footage at the exact MuseTalk full-video dimensions and at true 9:16.

**Do not migrate production from MuseTalk on this evidence.** The requested 25% whole-model speedup and ten simultaneous real-time speakers have not been demonstrated. Transport continuity is implemented; invisible generated first/last-frame joins and perceptual lip-sync remain separate, unpassed gates.

The delivery and publication checklist is in [implementation status](IMPLEMENTATION_STATUS.md). Completed tests and later unvalidated changes are distinguished below.

## What changed

- `flash_head_model.py`: audio projection and cross-attention K/V once per chunk; profile timestep/RoPE constants; optional FP32 real rotary implementation. Self-attention is still recomputed for every changing denoising latent.
- `utils.py`: cached reference Lab statistics. Compact/reference/staged modes tile only frame-independent color correction, not the temporally coupled VAE.
- `engine.py`: native width/height, independent RNG/motion state, rolling audio history, append/recondition, detailed CUDA phases, bounded reference cache and explicit memory modes.
- `trt_backend.py`, export tools: exact-shape BF16 FFN and LTX decoder engines with checksum/checkpoint/runtime/device/binding validation, explicit workspace ownership, fresh-process reload and no silent fallback.
- `calls.py`, server/worker/browser: stable peer/tracks, idempotent audio turns, interruption epochs, last-sent-frame reconditioning, bounded queues, active-call admission and lead-based microbatch scheduling. GPU failure marks all affected calls unhealthy.
- Call-specific warmup now covers source-idle/interruption reconditioning before readiness. A separate video clock prevents old idle frames from creating a catch-up backlog. Missed transport slots are disclosed, never counted as generation.
- Optional pinned `veryfast` H264 configuration addresses CPU encoding separately from neural throughput. The upstream encoder remains available for attribution/rollback.

The finite `/sessions` API remains available. See [the current call contract and commands](../../CONTINUOUS_WEBRTC.md) and [TensorRT reproduction](../../TENSORRT_EXPERIMENTS.md).

## Assets and resolution

All new avatar tests derive from the first frame of MuseTalk's certified `idle_active_listening.mp4`, in `assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/`. The original source is **480×832 at 24 FPS**, not exact 9:16.

- Source file SHA-256: `099877cef231ce12dede03843c558d10c2fa1e9c4e054c83be595e81a00f6ae4`.
- Canonical decoded RGB SHA-256: `47b05c6bdd63466e13381dc6cf21545e827bea0bc668c5798cbf7c69f7076b33`.
- Existing English audio: `short.wav` and `multi_sentence.wav` from MuseTalk's `generated/webrtc_quality/2026-08-08/kokoro_duration_matrix/`. Input hashes and useful sample counts are recorded in each benchmark.

480×832 uses native 15×26 spatial latents. True 9:16 at 576×1024 uses 18×32. Neither is a square neural output stretched into a portrait container. These are four-step, 25-output-FPS experiments. They reuse the avatar identity anchor, **not the original source video's motion**: SoulX generates the whole frame.

For a cheaper exact-9:16 delivery option, a separate preview centre-crops the native 480×832 TensorRT output to **468×832**, removing six pixels from each side without resizing. It reuses the same neural frames and is explicitly a post-generation crop, not another native model profile or an end-to-end cropped-WebRTC benchmark. A FaceTime client can similarly use a 9:16 `object-fit: cover` viewport while receiving the original 480×832 stream. This changes framing, not generation throughput; it must not be called native 576×1024 performance.

## Unpaced useful generation FPS

Warmup, model preparation and MP4 encoding are outside these timed intervals. Useful frames exclude nine-frame overlap, unrequested tail padding, held images and source idle. The MP4 playback rate is 25 FPS even when generation takes longer than real time.

| Native profile | Baseline median | Candidate median | Repeats / interpretation |
| --- | ---: | ---: | --- |
| 512×512, one 10-second job | 41.54 FPS | 43.11 FPS | Five alternating baseline/FP32-rotary repeats; earlier conditioning implementation |
| 480×832, one 10-second job | 26.07 FPS | 27.51 FPS | Five alternating matched repeats, reference offload and expandable allocator; approximately 5.5% gain |
| 480×832, full staged offload | 12.21 FPS | 12.21 FPS | Three repeats; CPU/GPU weight transfers erase the cache gain |
| 480×832, ten independent 10-second jobs | Not measured in this sweep | 27.60 FPS | Five warm repeats; median 90.58 seconds for 2,500 useful frames |
| 576×1024, true 9:16 | Not measured | 10.88 FPS | Three four-step staged-offload repeats; 250 frames take median 22.97 seconds |
| 512×512, 30 TRT FFNs + TRT VAE | Not a matched A/B | 44.26 FPS | Three fresh-process full-video runs; reference offload; no 25% claim |
| 480×832, 30 TRT FFNs + transient-workspace TRT VAE | Not a matched A/B | 28.49 FPS | Three full-video repeats beside the later 6,510-MiB workload |
| 480×832, combined TRT, ten independent 10-second jobs | Not a matched A/B | 28.57 FPS | Five repeats; median 87.51 seconds for 2,500 useful frames |

Evidence: [square A/B](../../benchmarks/implementation/square-ab.json), [matched portrait A/B](../../benchmarks/implementation/portrait-allocator-ab.json), [staged portrait](../../benchmarks/implementation/portrait-staged-ab.json), [true 9:16](../../benchmarks/implementation/portrait-9x16-staged.json), [combined TensorRT](../../benchmarks/implementation/square-trt-combined-reference.json).

The [ten-job portrait sweep](../../benchmarks/implementation/portrait-ten-jobs.json) ranged from 27.57 to 27.61 aggregate FPS, with p95 completion wall 90.67 seconds. Shared batch-one scheduling gave first chunks to peers approximately 0.83–8.25 seconds into the unpaced run. This is aggregate throughput, not 27.60 FPS for each of ten users.

The [combined TRT ten-job sweep](../../benchmarks/implementation/portrait-trt-ten-jobs.json) ranged from 28.48 to 28.69 aggregate FPS, median 28.57, with p95 completion wall 87.74 seconds. This is approximately 3.5% above the earlier optimized-PyTorch ten-job median, but these are separate sequential sweeps with changed co-resident memory, not a matched alternating TRT A/B. Neither proves the requested 25% gain.

The matched portrait baseline took about 9.59 seconds for 250 frames; the candidate took about 9.09 seconds. Raw recurrent output is not pixel-identical: mean absolute RGB difference was approximately 0.808/255, with an isolated maximum of 191. The CPU cached-math tests preserve the original arithmetic exactly where expected, but changed GPU fusion/precision can feed differences into later chunks. Small whole-frame error is not proof of identical mouth detail or lip-sync.

The earlier local ten-job MuseTalk result was 43.76 aggregate FPS, rendering a 256×256 face crop into its full video. The earlier SoulX result was 41.05 at native 512 square. A new single-job 44.26 result is not a valid claim that SoulX now beats MuseTalk for ten jobs, or at the same output/task quality.

### Reproduce useful-FPS measurements

Run from the repo root after stopping this checkout's SoulX GPU service and verifying its worker has exited. Keep unrelated services untouched and use a fresh output filename. The following is the native portrait ten-job TRT workload:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/workspace/SoulX-FlashHead/.trt-experiment/site-packages \
.venv/bin/python -m soulx_rtc.experiment \
  --width 480 --height 832 --fps 25 --steps 4 --batch 1 \
  --sessions 10 --seconds 10 --repeats 5 --modes real \
  --memory-mode reference \
  --trt-ffn .trt-experiment/portrait-ffn-bf16 \
  --trt-vae .trt-experiment/portrait-vae-bf16/vae.engine \
  --image benchmarks/implementation/musetalk-idle-anchor.png \
  --audio /workspace/MuseTalk/generated/webrtc_quality/2026-08-08/kokoro_duration_matrix/multi_sentence.wav \
  --output benchmarks/implementation/portrait-trt-ten-jobs-rerun.json
```

For matched PyTorch A/B, remove both TRT flags and `PYTHONPATH`, use `--sessions 1 --modes baseline real --record`, and retain five repeats. For the native true-9:16 experiment, remove TRT flags, use `--width 576 --height 1024 --memory-mode staged --sessions 1 --repeats 3 --modes real --record`. Preparation/compile time is reported separately. These choices reproduce the workload/profile, not a guarantee of identical timing under changing host load.

## TensorRT findings and memory failures

All 30 BF16 FFN partitions were built from actual checkpoint weights, reloaded and checked on captured activations. The median isolated layer speedup was approximately 1.034×; some layers were slower. The BF16 LTX decoder measured 160.66 ms compiled PyTorch versus 136.65 ms TensorRT, approximately 1.176×. Its output was finite, with mean absolute decoder error 0.00157 and maximum 0.05664 against the BF16 reference.

These are **kernel** results, not additive whole-model gains. The combined full-video path measured 44.26 useful FPS, and stayed opt-in. BF16 engines exist locally under the ignored `.trt-experiment/` directory; weights, engines, captured tensors and environments are not committed. Optional FP16-with-FP32-normalization code is a further experiment, not a tested production profile.

Initial full-FFN and combined default-memory runs ran out of VRAM; reference offload and releasing unused allocator blocks made the combined square run succeed. Default-allocator portrait preparation and one mixed portrait A/B also failed under the co-resident workload. Failure evidence is preserved rather than replaced by successful reruns: [failure notes](../../benchmarks/implementation/failures.json), [FFN full-model failure](../../benchmarks/implementation/square-trt-ffn.json), [combined startup failure](../../benchmarks/implementation/square-trt-combined.json).

PyTorch peak allocated/reserved values exclude TensorRT-owned allocations. For example, the successful combined run's roughly 2,912 MiB allocated figure is **not its total process/device VRAM**. Do not compare that number directly to an all-PyTorch peak as a memory reduction claim.

The native portrait exports also completed: median isolated FFN speedup 1.154×, decoder 246.66→208.62 ms (1.182×), finite outputs. Their full-model persistent-workspace run then failed during motion re-encoding when the unrelated process occupied 6,510 MiB. The decoder's 664.22-MiB workspace is now allocated only for decoding, synchronized before release, and returned to Torch's pool before motion encoding. This made the [full native portrait run](../../benchmarks/implementation/portrait-trt-transient-workspace.json) complete at 28.49 useful FPS.

Post-run native TensorRT snapshots reported approximately 5,208 MiB of total process GPU memory versus approximately 2,877 MiB in Torch's allocated counter. Only about 153 MiB was free according to CUDA at that snapshot. These are not total-process **peak** samples and show very little operational headroom; further co-resident growth can still make the service fail. The [failed predecessor](../../benchmarks/implementation/portrait-trt-combined.json) is retained. Decoder workspace reuse changes memory lifetime, not model dimensions, four-step denoising or RNG semantics.

A later service restart briefly overlapped the previous idle SoulX worker and failed to load; no benchmark samples came from that attempt. Both were stopped before a clean restart. A per-checkout OS file lock now rejects duplicate SoulX GPU owners before model allocation, including standalone engine builds. It does not reserve memory against other checkouts or unrelated services, and OmniVoice was left running throughout.

## Actual receiver-side call tests

The first call exposed cold reconditioning compilation and an idle-clock backlog. Its 66.53-second recording is retained as diagnostic evidence, not the recommended demo. Call-specific warmup and an elapsed-time video clock were then implemented.

**Codec correction:** subsequent actual-encoder assertions found that this early ramp selected VP8: H264 preferences had been applied after `setRemoteDescription`. The order is now fixed for both APIs. The early rows below are valid delivery/overload evidence, but **not verified H264 results**. Later reruns must report the active `H264Encoder` or `FastH264Encoder`, not infer a transport codec from the recorded MP4. The separate historical source-bank relay set preferences before making its offer and is not affected by this answerer bug.

The warmed source-idle C1 recording completed two three-second speech turns plus an interruption through one negotiation, with continuous increasing audio/video timestamps and zero reported RTP packet loss. First generated media took approximately 2.95 and 3.36 seconds from turn acceptance. There were two accounted underrun slots. Recording adds a second decode/encode path and is not a pure throughput measurement.

The following overload ramp used one three-second turn per peer, source idle, batch one, four steps, native 480×832 and an intentionally permissive ten-active-call limit:

| Active peers | Completion wall | Total underrun slots | Transport checks |
| --- | ---: | ---: | --- |
| 2 | 12.19 s | 151 | Passed |
| 4 | 25.19 s | 1,142 | Passed |
| 6 | 36.17 s | 2,830 | Passed |
| 10 | 68.26 s | 10,074 | Passed |

“Transport passed” means media arrived, PTS increased, one negotiation remained, no reported packet loss occurred, and useful turns completed. It does **not** mean real-time smoothness, invisible seams or perceptual lip-sync. The ramp already failed the real-time capacity gate, so it must not be promoted as ten supported speakers. These short overload probes are not the longer production/mobile/network acceptance suite.

The source recorder uses aiortc's default MP4 encoder settings; its nominal codec rate can be 30 while received timestamps are variable. Use the actual receiver PTS/arrival arrays, not `r_frame_rate`, to assess delivery. Offline demonstration MP4s contain exactly 250 native frames and 10 seconds of audio at 25 FPS.

### Verified H264 reruns with native portrait TensorRT

These later tests used native 480×832, four steps, batch one, all 30 TRT FFNs, the transient-workspace TRT decoder and the opt-in `FastH264Encoder` (`libx264`, `veryfast`, two threads). The benchmark checks the actual encoder, RTP reception and cleanup. No matched upstream-H264 versus fast-H264 speedup is claimed.

| Test | Completion wall | Underrun slots | Interpretation |
| --- | ---: | ---: | --- |
| Source idle, one peer: two turns, mid-speech interrupt, recovery | 23.14 s | 1 | H264/Opus transport and cleanup passed |
| Source idle, two speakers, one 3-second turn each | 10.54 s | 134 | Transport passed; real-time smoothness failed |
| Source idle, four speakers | 20.35 s | 935 | Transport passed; real-time smoothness failed |
| Source idle, six speakers | 31.81 s | 2,622 | Transport passed; real-time smoothness failed |
| Source idle, ten speakers | 58.90 s | 9,153 | Transport passed; real-time smoothness failed |
| Source idle, ten connected peers, only one speaker, two turns | 18.87 s | 27 | Not ten active speakers; not a strict smoothness pass |
| Generated idle, one peer: two turns, mid-speech interrupt, recovery | 27.58 s | 0 | Short transport/playback-accounting pass; not perceptual certification |

Evidence: [C1](../../benchmarks/implementation/calls-h264-trt-c1.json), [C2](../../benchmarks/implementation/calls-h264-trt-c2.json), [C4](../../benchmarks/implementation/calls-h264-trt-c4.json), [C6](../../benchmarks/implementation/calls-h264-trt-c6.json), [C10](../../benchmarks/implementation/calls-h264-trt-c10.json), [mixed ten](../../benchmarks/implementation/calls-h264-trt-mixed10.json), [generated idle C1](../../benchmarks/implementation/calls-generated-idle-h264.json).

The source-idle C1 sender's first-media latency was approximately 1.91/1.71 seconds for the initial turns and 1.66 seconds for recovery. This measures server acceptance to sender playback, **not** acceptance to receiver decode/audible playback. The interruption was requested after 15 generated frames and 27,840 useful audio samples had been sent, not before speech began. Its receiver wire rate was 22.85 FPS with p95 inter-arrival gap 55.7 ms. Generated-idle C1 measured 25.00 wire FPS and p95 gap 47.4 ms. Both retained monotonic audio/video PTS, one negotiation and zero reported RTP packet loss. Wire rates include idle/held frames and are not additional neural throughput results.

### Sustained generated-idle validation

The [30-minute one-peer run](../../benchmarks/implementation/calls-generated-idle-soak30m.json) completed in 1,819.82 seconds, including final mid-speech interruption and recovery. It completed **209 useful speech turns**, with 210 submitted IDs including the interrupted turn. Actual H264, one negotiation, monotonic audio/video PTS, zero reported RTP packet loss and cleanup all passed. It received 45,493 video frames at 24.987 wire FPS, with p95 arrival gap 44.0 ms and maximum 191.3 ms.

This was **not a strict smoothness pass**: it counted 23 missed/underrun video slots and 10,535 held frames, approximately 23% of delivered frames. Many holds occur while waiting for turn preparation/first chunks, outside the video-underrun counter. Only 34,944 frames were newly generated, including 14,856 neural-idle frames. The latest 64 turn summaries had sender-first-media median 2.84 seconds and p95 2.93 seconds; these are not whole-run quantiles or receiver TTFF. Continuous RTP does not imply continuous natural motion or low conversational latency.

Sixty resource samples found server RSS 275.61–363.81 MiB, GPU-worker CPU RSS steady at 6,186.65 MiB, retained audio 458,240–642,560 bytes and sampled queued chunks 0–1. After cleanup, calls/sessions/GPU states all returned to zero and server RSS was 293.56 MiB. These finite sampled observations do not prove a universal memory bound or total GPU-memory peak. Other host work remained active.

All 31 timestamped receiver-image checksums and the compressed raw timestamp checksum verified. The [contact sheet](../../benchmarks/implementation/calls-generated-idle-soak30m-contact.png) shows a coherent avatar across the sampled interval without the previously rejected collapsed-face behavior. One image per minute cannot certify lip-sync or invisible transitions.

The soak ran the source hashes recorded in its `health_before`: it preceded the cross-peer idle-admission fix, explicit audio-hold accounting/20-ms scheduling allowance, and later startup/staged-memory changes. It is not a 30-minute validation of those later changes.

### Audio-boundary diagnostic and final regression work

A waveform check of the short generated-idle receiver recording found a **20-ms inserted gap** in its first `short.wav` turn despite zero counted video underruns. Half-second correlation windows shifted from approximately 3.0865 to 3.1065 seconds; the recovery turn retained a consistent offset around 23.6265 seconds. This illustrates why video counters alone cannot certify audio continuity. See [the reproducible diagnostic](../../benchmarks/implementation/audio-before-short.json) and `python -m soulx_rtc.check_recorded_audio --help`.

The sender now permits at most one 20-ms audio packet of scheduling lead after video begins, avoiding the strict-horizon callback race while still bounding advancement through a real stall. `audio_hold_samples` separately discloses withheld useful speech samples. CPU tests verify both the boundary allowance and subsequent stall accounting. Waveform correlation after lossy Opus/AAC is a diagnostic, not a perceptual lip-sync score or a receiver acknowledgement.

After the soak, the unrelated GPU process grew to approximately **7,456 MiB**. A clean native TRT restart failed while deserializing an FFN engine; the measured approximately 5,164-MiB steady SoulX runtime cannot coexist with that load. A staged PyTorch restart also failed during Wav2Vec preprocessing. Both failures produced zero benchmark samples and are retained: [TRT restart](../../benchmarks/implementation/portrait-trt-restart-headroom-failure.json), [staged restart](../../benchmarks/implementation/portrait-staged-restart-headroom-failure.json). OmniVoice was not stopped. The subsequent staged path moves the existing DiT reload after audio and keeps it on CPU after decode; startup TRT reference offloading also reduces deserialization peaks. These are separate memory-lifetime changes, not retroactive improvements to the earlier FPS figures.

Moving the DiT reload alone passed audio but still failed in denoising. The final staged path additionally offloads Wav2Vec weights outside audio preprocessing. It then started successfully at native 480×832, four steps, verified two-session/two-chunk batch-one GPU isolation with maximum pixel difference zero, and completed [the final C1 H264 regression](../../benchmarks/implementation/calls-final-staged-c1.json). Three turns, a mid-speech interruption and recovery completed in 46.48 seconds, with successful cleanup. It recorded 251 video underrun slots and **481,920 audio-hold samples (10.04 seconds)**. Its 24.62 wire FPS includes many held frames and must not be presented as real-time generation. Sender first-media latencies were 3.40–5.88 seconds across the tested turns. A sampled final chunk took about2.00 seconds for24 new frames; this is a chunk timing, not a repeated useful-FPS benchmark.

That final C1 path reported approximately3,855 MiB peak Torch allocation; this still is not a sampled total-device peak. The tested fallback fits at the cost of transfer overhead and plainly fails conversational smoothness. It validates the current protocol/audio-hold instrumentation under overload, **not** elimination of every waveform gap in the faster TRT mode. Revalidating the final scheduler/audio changes on the fast native TRT profile requires additional available GPU memory; the unrelated service was not stopped to obtain it.

The [final two-peer generated-idle handoff test](../../benchmarks/implementation/calls-final-staged-c2.json), with only one rendering slot, completed one three-second turn for each peer in21.77 seconds. Both actual H264 transports and cleanup passed, demonstrating that an idle peer no longer indefinitely monopolizes admission. It still had159 video underrun slots and is not a two-speaker real-time result.

### Automated and isolation coverage

The CPU suite passed **34 tests**, covering actual H264/Opus peer negotiation with both encoder modes, retry/cancel/cleanup, bounded generated silence, cross-peer idle-admission yielding, audio boundary/stall accounting, scheduler admission/private outputs, cached math, strict TRT bindings/workspace ownership, GPU-owner lock release and waveform-correspondence silence rejection. Four warnings originate in upstream Triton's deprecated autotuner arguments. Saved implementation MP4s were checked for FFmpeg decoding errors.

The native GPU service checked two independent sessions over two chunks in reversed batch-one execution order, with **maximum pixel difference zero**. Actual native batch-two GPU validation was not performed in this constrained configuration; CPU batch math and scheduler tests are not substitutes. Fixed TRT profiles remain batch one. Browser JavaScript syntax is checked, but real mobile browsers, TURN, loss/jitter injection and slow receivers are still untested.

## Footage

- [Exact MuseTalk size: 480×832, 10 seconds](../../benchmarks/implementation/portrait-allocator-ab-real.mp4) — offline generated output, 27.51 useful-FPS profile.
- [True 9:16: 576×1024, 10 seconds](../../benchmarks/implementation/portrait-9x16-staged-real.mp4) — offline generated output, not a real-time performance claim.
- [Exact 9:16 delivery crop: 468×832, 10 seconds](../../benchmarks/implementation/portrait-9x16-cropped.mp4) — centre-cropped from the tested 480×832 TensorRT output; no new neural generation or FPS claim.
- [Native portrait continuous generated-idle WebRTC recording](../../benchmarks/implementation/calls-generated-idle-h264-peer0.mp4) — actual H264 receiver media, two turns, mid-speech interruption and recovery.
- [Source-idle H264 WebRTC recording](../../benchmarks/implementation/calls-h264-trt-c1-peer0.mp4) — cheaper idle strategy, with separate source/generation transitions.
- [Native portrait TensorRT output](../../benchmarks/implementation/portrait-trt-transient-workspace-real.mp4) — offline full-model output, 28.49 useful-FPS profile.
- [Combined TensorRT square output](../../benchmarks/implementation/square-trt-combined-reference-real.mp4) — actual full-model output, not an isolated-layer visualization.

Decoded contact sheets show a coherent avatar without the earlier low-resolution/two-step facial collapse. This is limited visual inspection, not a blinded quality or lip-sync pass.

## Remaining product gates

The model still has no exact first/last-frame API. Generated-idle mode keeps one recurrent model state across silence and speech, but spends GPU during idle. Source-idle mode saves compute and reconditions from the last nine **sent**, not receiver-acknowledged, frames; it cannot guarantee an invisible switch to a separately moving source plate. Interruption does not claim RNG rewind or receiver-playout rollback.

Ten 25-FPS speakers require at least 250 useful FPS plus headroom. The measured profiles are nowhere near that. Keep MuseTalk production, use SoulX as an experimental whole-frame renderer, and require perceptual endpoint/lip-sync review, mobile/TURN/network testing, unified multi-tenant compute admission and a measured capacity margin before migration. A larger available GPU budget may change memory/offload behavior; it cannot be assumed to supply the missing throughput.
