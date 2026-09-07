# MuseTalk versus SoulX: what transfers, what still needs work

Audit: 2026-09-07. SoulX source `ac2c2bbb7308f87880c2ae4b01563d1e4d203c02`; MuseTalk source `e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e`.

**Recommendation: transfer MuseTalk's serving discipline and experiment methodology, not its tensor shapes or precision settings. Keep MuseTalk production until SoulX passes matched quality, native-resolution and concurrent-delivery gates.**

[Detailed SoulX architecture](../architecture/README.md) · [Next implementation plan](NEXT_OPTIMIZATION_PLAN.md) · [Validated evidence](EVIDENCE_VALIDATION_2026-09-07.md) · [Existing 92-document MuseTalk audit](MUSETALK_DOC_AUDIT.md)

## The models solve different-sized problems

| Dimension | This MuseTalk checkout | This SoulX Lite checkout |
| --- | --- | --- |
| Neural output | 256×256 face patch | Whole RGB frame, e.g. 480×832 |
| Final frame | Patch composited into a pre-existing source plate | Newly generated head/body/background appearance within the canvas |
| Audio model | Whisper hidden features aligned to face frames | Rolling eight-second Wav2Vec features with twelve layers |
| Main renderer | Conditioned 2-D UNet invocation at fixed timestep | Four passes through a 30-block spatiotemporal DiT |
| Latent | Face VAE latent around 4×32×32; prepared masked/full-face conditioning | 128×5×H/32×W/32, combined with reference |
| Temporal dependence | Frame work can be collected from prepared source cycles | Next 24-frame chunk depends on last-nine-frame re-encoding |
| Decoder | 2-D face VAE | 3-D LTX decoder, five latent times to 33 RGB frames |
| Pose control | Select a source pose/frame/transition and its matching prepared tensors | Portrait, audio, seed and recurrent motion; no semantic endpoint solver |
| Cheap idle | Existing source video | Source/hold available, but visual seam to generated motion remains |
| Neural idle | Not required to animate an existing motion plate | Full generation on silence if continued generated trajectory is wanted |

The MuseTalk source anchors are its [VAE wrapper](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/musetalk/models/vae.py), [UNet wrapper](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/musetalk/models/unet.py), and [audio processor](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/musetalk/utils/audio_processor.py). These links identify the local audited commit; access depends on repository visibility.

A 480×832 delivery frame does not imply that MuseTalk runs its neural model at 480×832. Conversely, evaluating SoulX only on its face pixels does not remove the cost of generating its full canvas.

## What the MuseTalk scheduler actually does

The inspected [GPU scheduler](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/scripts/hls_gpu_scheduler.py) owns one rendering thread and overlaps CPU preparation/composition with it.

Preparation obtains matching source frame, mask, crop coordinates, face latent and audio conditioning. Position/audio feature preparation is cached where invariant. CPU pinned staging buffers support transfer. Jobs are sliced into frame rows and collected into compatible batch buckets; padding is trimmed from actual results.

The scheduling policy explicitly favors startup slices, then warm work, with chunk-completion and fairness behavior. It can keep producing for a turn while previously produced content plays. Composition uses a CPU pool and sequence numbers so out-of-order CPU completion does not reorder delivered frames. Batch callbacks reduce Python handoff overhead.

The scheduler's `_memory_bucket` returns one for its semaphore accounting. That value is a lease unit, **not** the neural GPU batch size. Porting that constant as a VRAM formula would be incorrect.

In SoulX, the model batch unit is an entire temporal chunk per session. A MuseTalk batch of eight face frames and a SoulX batch of eight recurrent 33-frame windows have radically different memory and latency.

## Optimization transfer matrix

| MuseTalk technique / evidence | SoulX status | Additional work that is justified |
| --- | --- | --- |
| Shared weights and single GPU owner | Already implemented | Unify finite jobs, calls, generated idle and cold preparation under one admission budget |
| Startup-aware/fair scheduling, preparation overlap | Partial equivalent | Deadline-aware scheduling; separate generation completion from turn playout completion |
| Prepared immutable avatar/pose cache | Reference LRU and private motion state implemented | Active-template leases, cold-start queue priority and complete preparation provenance |
| Cached positional/audio features | Timestep/RoPE and within-chunk audio K/V implemented | Microbatch Wav2Vec only after profiling; shifted windows are not invariant |
| Exact-shape TRT UNet artifacts | FFN islands and decoder artifacts implemented | New B2 profiles or larger DiT partitions; cannot load MuseTalk weights/engines |
| Stagewise VAE and selective precision | BF16 whole decode wrapper implemented | Target LTX motion encoder and sensitive operations with new captures/calibration |
| Safe-five INT8 VAE policy | No direct stage equivalence | Independent per-layer sensitivity study; keep normalization/feedback numerically safe |
| GPU uint8 conversion | Already implemented | Measure transfer/IPC; don't count an existing optimization twice |
| Shrink blending to nonzero-mask ROI | No equivalent face compositor in SoulX | Skip discarded-overlap postprocessing if mathematically independent; hybrid compositor is a different product |
| Fixed-point blending | No active matching stage | Do not add integer arithmetic without a stage to accelerate |
| Batched callbacks / asynchronous composition | Chunk IPC already implemented; no source blend needed | Measure serialization and encoder backpressure before introducing shared-memory rings |
| Exact 8/16/32 frame buckets | Not transferable verbatim | Test B1/B2 chunk efficiency and deadlines first; all dependent TRT shapes must match |
| Persistent audio transport and audible-end gating | Implemented, with sender-horizon holds | Receiver-aware measurements, sample-accurate queued-turn continuation |
| Matched source pose/frame/latent ownership | Not equivalent | Treat source/generated transition quality as a separate model/asset problem |
| Cheap source idle | Implemented | Default candidate for many idle peers; verify every approved seam |
| NVENC/HLS packaging changes | Current RTC uses H264 RTP, not HLS | Measure encode CPU and transport overhead; use a version-safe optional hardware encoder only if justified |
| Lower crop size or temporal cadence | Earlier aggressive SoulX profile failed | Separate quality tier, never label lower quality a same-quality speedup |

## What previous MuseTalk measurements do and do not say

The refreshed [July optimization results](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/docs/webrtc_generation_optimization_results_2026-07-03.md) recorded approximately 5.89 ms for full float blending, 6.30 ms for full fixed-point blending, and 1.89 ms for fixed-point blending restricted to the active mask ROI. The important gain was avoiding inactive area, not a universal claim that integer math is faster. The reported maximum pixel difference was one.

The same historical document's exact-8 versus exact-16 GPU microbenchmarks concern a UNet stage. Hundreds of face-stage FPS on an RTX 3090 are not hundreds of full SoulX frames on this RTX 4070. Its live larger-batch intervals and dual-engine OOM observations also show why maximum isolated batch throughput is not the same as smooth per-peer delivery.

The [TRT artifact notes](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/docs/trt_artifacts/README.md) describe selective precision and exact artifact contracts. The [late-VAE-block plan](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/docs/next_bottleneck_vae_late_block_plan_2026-06-11.md) includes proposed, not all measured, improvements. Neither justifies copying an INT8 policy into a different 3-D recurrent VAE.

### Profile-selection nuance

The current [selection script](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/scripts/select_unet_trt_profile.py) defaults its CLI preference to `exact16` when available. The inspected [deployment launcher](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/scripts/vast_onstart.sh) instead supplies a default preference of `split8`. The later published bundle is batch-eight oriented. “MuseTalk always prefers batch eight” is therefore too broad; state which launcher, available artifacts and profile actually ran.

In [trt_runtime.py](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/scripts/trt_runtime.py), the split wrapper can process a larger divisible batch in exact-eight pieces. That is artifact reuse, not concurrent execution by itself. The stagewise VAE can ensure/build missing batch profiles on demand; production readiness should prewarm all permitted shapes to keep compilation off the media path.

## Audio and continuity lessons

MuseTalk's [audio timeline helper](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/scripts/webrtc_audio_timeline.py) detects sustained activity and optionally trims only leading/trailing silence with padding, then uses the same normalized media for inference and playout. This avoids a mismatch between an audio container's silent tail and audible speech.

Its [persistent tracks](https://github.com/AhmadAFS1/MuseTalk/blob/e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e/scripts/webrtc_tracks.py) keep silence RTP alive, arm prepared speech, publish audio media-start progression and mark completion after the final packet occupies its transport tick. These are useful sender-clock principles; even they are not direct proof of remote acoustic playout.

SoulX should retain one continuous 16 kHz conditioning timeline and one 48 kHz transport timeline with explicit turn boundaries. Optional edge-silence normalization must apply to both, preserve intended dramatic pauses, and never silently trim internal silence. Whole-chunk model alignment must not be mistaken for a requirement to insert silence between every queued TTS segment.

MuseTalk's source poses have matching frame/latent/mask ownership and explicit transition assets. SoulX's use of a still from the same source bank is asset reuse, not reproduction of that source motion. Replaying the six original MP4s in one peer is a control experiment, not evidence of generated endpoint control.

## Fair performance comparison

The historical local square ten-job comparison reported SoulX ~41.05 aggregate FPS and compiled MuseTalk ~43.76. The MuseTalk comparator generated face patches and resized its composed result to 512²; it did not establish a native 480×832 production TRT result.

The newer native SoulX ten-job TRT median is **28.57 useful FPS**. The newer square TRT single-job figure is **44.26 FPS**. They must not be combined into “both models do 43 FPS at native portrait resolution.”

A new head-to-head must use:

- The exact approved source-bank identity and the same speech PCM, with fixture hashes.
- The same final delivered dimensions, useful duration, codec and quality acceptance criteria.
- Both backend-only and complete receiver-side timing; native neural dimensions explicitly recorded.
- Same GPU, controlled/timestamped co-residency, warmup policy and repeated alternating runs.
- C1/C2/C4/C6/C10 active-speaker ramps, plus ten-connected duty-cycle scenarios.
- Separate useful speech, neural idle, source idle, held frames, audio holds, dropped frames and codec drain.
- Per-peer tail latency and quality, not just the sum of frames.

## Migration decision

There is still worthwhile engineering work: queued-turn lookahead, unified admission, better DiT execution boundaries, memory lifetime planning, motion-encoder acceleration and stronger evidence gates.

There is **no measured basis** for promising ten simultaneous 25-FPS SoulX speakers on this 12-GB 4070. That target needs at least 250 useful FPS before headroom, roughly 8.75 times the historical native result. A 25% gain would reach about 35.71 FPS, still below two simultaneous 25-FPS speakers before headroom.

Keep MuseTalk for the current source-controlled FaceTime product. Evaluate SoulX as a separately labeled whole-frame/generative-motion tier. Promote only after the [next plan's](NEXT_OPTIMIZATION_PLAN.md) quality and concurrent-delivery gates pass.
