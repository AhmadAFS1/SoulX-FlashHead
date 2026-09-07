# Implementation status: native portrait calls and performance

The [September 7 full-code audit](../architecture/README.md) and [next optimization plan](NEXT_OPTIMIZATION_PLAN.md) document remaining opportunities and newly identified source/measurement gaps. The [executed evidence validation](EVIDENCE_VALIDATION_2026-09-07.md) rechecks historical results without rerunning the GPU. Completion below refers to the prior engineering round, not every possible optimization or product gate.

Implemented and tested on September 6–7, 2026, from baseline `b3c47de`. The engineering experiment, sustained test and final fallback regressions are delivered. Product performance/quality targets are separate and must not be inferred from checked implementation tasks.

## Delivered engineering

- [x] Native 480×832 generation matching the certified MuseTalk full-video dimensions, plus native true 9:16 at 576×1024. Existing idle avatar and speech-fixture hashes are recorded.
- [x] Detailed CUDA phase profiling and useful-FPS benchmark harness with warmup separation, repeated A/B and ten-job sweeps, input/code provenance, raw trajectories and retained failures.
- [x] Within-chunk audio and cross-attention K/V caching, reference-color statistics, timestep/RoPE constants and opt-in FP32 real rotary math. Baseline remains available.
- [x] Separate BF16 TensorRT environment, all 30 exact-shape FFN partitions and deterministic VAE decoder, for square and portrait profiles. Runtime validates artifact/checkpoint/device/bindings and fails closed.
- [x] Low-headroom memory modes and transient decoder workspace; GPU-owner lock prevents duplicate loads from this checkout. Co-resident OmniVoice remains untouched.
- [x] Persistent peer/H264/Opus tracks, audio turns, idempotent retries, queued-turn bounds, cancellation epochs, last-sent-frame reconditioning, and browser controls.
- [x] Shared GPU owner, fair lead-based scheduling, compatible-call microbatch implementation, separate connected/rendering limits, release/health/error handling.
- [x] Source, hold and recurrent generated-idle modes. Generated silence has bounded render-ahead and audio storage; it does not claim exact endpoints or zero turn-start latency.
- [x] Corrected H264 negotiation order and verified the actual encoder. Optional version-pinned two-thread `veryfast` encoder; upstream encoder retained.
- [x] **34 automated tests passed**, plus JavaScript and shell syntax checks. GPU batch-one reordered two-session/two-chunk isolation produced maximum pixel difference zero in the measured TRT service.
- [x] Actual native H264 call recordings, mid-speech interruption/recovery, active-speaker C1/C2/C4/C6/C10 ramp and ten-connected/one-speaker probe.
- [x] 30-minute generated-idle resource, delivery, drift-snapshot and cleanup assessment: 209 completed turns, transport/cleanup passed, strict smoothness did not. The soak preceded the final scheduler/audio/memory follow-ups; see exact source hashes and qualifications in results.
- [x] Final native-size staged fallback starts beside the later7,456-MiB co-resident workload, passes GPU isolation, C1 H264 interruption/recovery/cleanup and two-peer idle-to-speech admission handoff. Substantial audio/video holds are measured; it is not a real-time pass.
- [x] Code, documentation, tests, footage and benchmark evidence are included in this implementation commit for `AhmadAFS1/SoulX-FlashHead`. Model weights, environments and machine-specific engines remain excluded.

## Measured outcomes

| Profile | Useful generation throughput | Meaning |
| --- | ---: | --- |
| Native 480×832 matched PyTorch A/B | 26.07 → 27.51 FPS | Approximately 5.5% gain, five alternating repeats |
| Native 480×832 TensorRT, ten 10-second jobs | 28.57 aggregate FPS | Median 87.51 seconds for 2,500 frames; five repeats |
| Native true 9:16, 576×1024 | 10.88 FPS | Four-step staged offload; 10 seconds of footage takes 22.97 seconds to generate |
| Native 512×512 TensorRT | 44.26 FPS | Single-job sweep, not matched ten-job superiority over MuseTalk |

The short generated-idle H264 run completed three turns plus an interruption in 27.58 seconds, with zero counted underruns and successful peer/GPU-state cleanup. The ten-active-speaker run had 9,153 underrun slots; even the ten-connected/one-speaker source-idle probe had 27. Neither is a strict smoothness pass. Delivered wire/idle/held FPS is never substituted for useful neural FPS.

## Plan disposition and unpassed gates

- P1 invariant conditioning/profiling: implemented; matched portrait gain measured. This is not pixel-identical recurrent GPU output or a perceptual lip-sync pass.
- P2 selective TensorRT: implemented and run end to end, with exact-shape builds, reloads and failures preserved. The requested **25% whole-model improvement was not achieved**. Motion encoding, custom attention and recurrence remain in PyTorch. Optional FP16 decoder code is untested; BF16 is the measured path.
- P3 serving/scheduler/media: implemented and live-tested locally, but native GPU batch-two, full burst/slow-receiver/network/mobile/TURN acceptance and unified multi-tenant compute admission remain untested or unimplemented.
- P4 quantization/model changes: not promoted or calibrated. INT8, smaller/fewer-step profiles, whole-block/attention export and a hybrid face compositor require separate numerical/perceptual validation. Earlier low-resolution/two-step output collapsed; those changes cannot count as same-quality wins.
- Exact first/last frames and invisible source/generated transitions: **not guaranteed by the model**. Interruption uses last sent frames, not receiver-acknowledged audio/video rollback.
- Ten simultaneous 25-FPS speakers require at least 250 useful FPS plus headroom: **failed on this configuration**. Generated-idle peers also consume rendering capacity.
- Production migration: **do not migrate from MuseTalk based on these results**. Keep SoulX as an experimental whole-frame renderer with conservative single-call admission.

At the end of the implementation tests, the local endpoint was `http://127.0.0.1:8765/`, using the slower fully staged PyTorch fallback. At that time, the faster measured native TensorRT profile did not fit beside the increased unrelated GPU workload. Its final scheduler/audio follow-ups need a new GPU run with sufficient headroom; no permission to stop OmniVoice was received and it was left untouched. This is a historical deployment/memory observation, not a current liveness or GPU-fit guarantee.

Read [the measured results and footage](IMPLEMENTATION_RESULTS.md), [complete test catalog](TEST_CATALOG.md), [persistent API/run instructions](../../CONTINUOUS_WEBRTC.md), and [TensorRT reproduction](../../TENSORRT_EXPERIMENTS.md). The original [optimization plan](OPTIMIZATION_PLAN.md) and [92-file MuseTalk audit](MUSETALK_DOC_AUDIT.md) remain historical references, not claims that every proposed product gate passed.
