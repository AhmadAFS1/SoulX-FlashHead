# Throughput, latency and TensorRT: a quality-gated plan

This is the **historical pre-implementation plan** written at `b3c47de`. Follow [implementation status](IMPLEMENTATION_STATUS.md), the [complete test catalog](TEST_CATALOG.md), [measured results](IMPLEMENTATION_RESULTS.md) and [the implemented call API](../../CONTINUOUS_WEBRTC.md) for what was actually built and tested. Proposed speedups below are not measured results unless explicitly labelled historical evidence.

Status: source-derived proposals, **not measured speedups**, unless explicitly labeled historical evidence. Scope is installed SoulX Lite on this RTX 4070, native 512² / four steps / 25 FPS. Keep the current working backend as the rollback path.

## Define the target correctly

Useful generation FPS = sum of newly generated, requested frames / shared unpaced completion wall time. Exclude overlap, final padding, held frames, repeated wire frames, startup compilation and codec drain. Report warmup separately. Live receive FPS, GPU-stage FPS, first-frame latency and per-peer stalls are distinct metrics.

Previous matched local ten-job results are **41.05 useful FPS SoulX versus 43.76 compiled MuseTalk**; neither is a ten-speaker real-time result. Those were single sweeps with different model tasks, not a statistically significant model ranking. MuseTalk's compatible local compiled baseline was used because the older saved TRT artifact did not deserialize here. The other-host 94–97 backend FPS result cannot be substituted for this 4070 baseline.

| Objective | Required useful FPS | Relative to 41.05 |
| --- | ---: | ---: |
| +25% throughput | 51.31 | 1.25× |
| Ten active speakers at 15 generated FPS | 150 | 3.65× |
| Ten active speakers at 20 generated FPS | 200 | 4.87× |
| Ten active speakers at native 25 FPS | 250 | 6.09× |

A 25% throughput gain reduces 60.90-second completion wall to **48.72 seconds**, a 20% time reduction. It would provide about two theoretical 25-FPS streams before headroom and jitter, not ten. Ten connected calls with mostly cheap idle playback is a different target; it requires a measured speaker-duty-cycle/burst admission policy.

## What MuseTalk actually achieved, and transferability

All source documents, including superseded experiments, are indexed in [the 92-file audit](MUSETALK_DOC_AUDIT.md). Paths in this table refer to that MuseTalk checkout.

| MuseTalk work / evidence | SoulX applicability | Decision |
| --- | --- | --- |
| Shared HLS/RTC GPU scheduler, overlapping preparation/composition | Share weights and batch compatible sessions, but chunk recurrence replaces independent frame rows | Already partly implemented; improve deadline fairness and prep priority |
| GPU-process isolation in this fork's earlier experiments | Separates media Python work from model dispatch | Keep; already present |
| Exact TRT batch artifacts, real-input capture, load validation and fail-closed backend identity | Same methodology, different shapes/weights/runtime | Strong transfer; build new artifacts, never reuse MuseTalk engines |
| Safe-five INT8 VAE + FP16 TRT UNet | Lite has a different 3-D LTX VAE, pixel norm, BF16 DiT and recurrent feedback | Transfer selective precision and validation strategy, not stage names/scales |
| GPU uint8 postprocess; May29 post slice ~6.9→0.9 ms, only 0–2% RTC FPS gain | Already GPU uint8 in SoulX | No double-counted new gain; inspect remaining synchronization/IPC |
| July3 mask-ROI shrink + fixed-point blend, 5.89→1.89 ms, max pixel difference 1 | SoulX has no face-patch compositing stage | Not directly applicable; only a future hybrid compositor would use it |
| July3 batch frame callback | SoulX already sends chunks across IPC/queues | Keep chunk handoff; consider measured shared-memory ring optimization |
| Static geometry/PE caches | Reference color statistics, timestep/RoPE tables and within-chunk audio conditioning are invariant | High-priority code-specific experiments below |
| Larger 8/16/24/28 buckets | SoulX batch means whole temporal chunks, not face frames; native B2 barely beat B1 | Do not copy sizes; test B1/B2 first, include memory and tails |
| Lower generation FPS with duplicated playback | Can reduce work per second but changes audio interpolation and trained temporal cadence | Separate quality experiment, not identical-quality acceleration |
| 192/224 MuseTalk ROI | Faster but blurred/smeared; rejected | Do not repeat resolution cuts as a quality-equivalent win |
| Avatar cache/S3, strict restore, warmup, sticky workers | Reference templates are smaller but same lifecycle principles apply | Port content hashes, active leases, bounded cache and readiness/drain policy |
| Persistent video + persistent audio RTP, audible-end gate, pose ownership | Model-independent serving requirements | Essential missing product work; see continuous-call design |
| NVENC/HLS process/manifest optimizations | HLS packaging does not exist in native RTP; encoder factory/version hooks differ | Verify active codec first; optimize only if measured bottleneck |

Important chronology/caveats:

- Monolithic MuseTalk TRT VAE produced fast but gray/collapsed faces; exact shapes alone did not solve all errors. Partitioning and preserving sensitive normalization were necessary. Export success is not correctness.
- May29 FP16 UNet stage improved roughly 35%, while full WebRTC aggregate improved only 7–10%. RTX6000Ada's corresponding UNet addition improved aggregate only 2–4%; strict smooth capacity stayed four at 20 FPS.
- July3 exact16 runtime failed in combinations that exhausted VRAM; July4 exact16-only later passed. July4's C6 average interval of 85 ms is still not strict six-by-20 FPS, regardless of the document's looser “support target” wording. July10/11 published a validated **batch8-only** immutable artifact; current `select_unet_trt_profile.py` prefers that unless exact16 is explicitly requested.
- Five-stage INT8 gave meaningful memory and throughput gains, but not a universal 25%. Some early two-stage tests improved VAE microbench while concurrent end-to-end generation got slower. Late-up-block INT8 improved isolated VAE 6.59% but did not move live capacity and remained unpromoted.
- More workers or VRAM did not reliably move strict capacity. Repeated frames and permissive HLS segment thresholds must not be presented as newly generated real-time frames.

## Establish the critical path before changing it

The prior native B1 chunk profile was about 15.1 ms audio, 306.2 ms DiT, 231.9 ms **combined VAE/color/re-encode/transfer**, total about 553 ms. Latest warm service observations were similar (~303 ms DiT, ~229 ms combined). The combined label is not a measurement of VAE decode alone.

First add an opt-in profiler in `Engine.generate()` with CUDA events and host timers around:

1. Audio window assembly, preprocessing, host→device transfer and Wav2Vec separately.
2. Audio projection, each DiT step, self/cross attention and MLP groups.
3. VAE decode, RGB↔Lab correction, last-nine VAE encode and posterior sampling separately.
4. Float→uint8/layout, device→host, IPC, enqueue, encode, network and receiver decode separately.

Synchronize only at controlled measurement boundaries; don't add per-layer global synchronizations to the production loop. Disable tensor capture during timing. Report wall and event time, compile events, effective batch occupancy and wasted final-chunk frames. Record GPU clocks/power/utilization and co-resident memory without logging credentials.

Using that coarse profile only as an Amdahl illustration, DiT is ~55% of the measured work. A 25% total throughput gain from DiT alone needs roughly **36% DiT time reduction**. Removing the entire ~3% audio slice would yield only ~3% overall. Re-profile after each win rather than adding percentage estimates together.

## Ranked code-specific experiments

### P1: Hoist invariant conditioning, preserve output semantics

**Within a chunk**, `WanModelAudioProject.forward()` recomputes `audio_proj` from unchanged context at every denoising step. Every block's cross-attention then recomputes `k(context)`, `norm_k` and `v(context)` although only the query changes.

Proposed interface: `prepare_chunk_conditioning(context)` produces projected audio and optional per-block K/V; a denoiser forward consumes them for all four steps. Cache **only inside that chunk**, separately per session/profile. Do not cache self-attention K/V: those depend on changing noisy video latents.

- Audio projection can run once instead of four times.
- Each cross-attention K/V projection can run once instead of four times.
- All-block BF16 K/V storage is approximately `30×2×5×32×1536×2 bytes = 28.125 MiB/session`, plus projected context/workspace. That is modest but not free on today's almost-full GPU.
- Keep the session RNG sequence, normalization dtype and operation ordering stable; cache keyed to actual context generation, not just avatar identity. Compare full sequential output, not merely one projection tensor.
- Export the denoiser core separately after hoisting: fewer repeated operators and a clearer TensorRT boundary. This may help PyTorch compilation even without TRT.

**Per profile/reference**, precompute the expanded RoPE grid and fixed timestep embeddings. RoPE currently builds complex/float64 views and grid multipliers in each self-attention call. Prototype real cosine/sine rotations with FP32 accumulation and BF16 return; numerical similarity is necessary but GPU throughput/long-run motion quality must be measured. Consider caching the reference half of the patch Conv3d only after profiling: its linearity permits decomposition, but extra launches/memory could erase the small saving.

`match_and_blend_colors_torch()` also recomputes the unchanged portrait's RGB→Lab conversion, mean and standard deviation every chunk. Cache those per-reference statistics and compile/fuse the source-color path. Do **not** silently disable correction: corrected frames feed the next motion encoder, so changing color changes future motion state as well as display pixels.

### P2: TensorRT BF16/FP16 partitions, not an all-or-nothing export

The installed SoulX environment is Torch 2.7.1+cu128 / Python 3.10, distinct from MuseTalk's Torch 2.5.1+cu121 / TRT10.3 family. A separate experiment environment in the Torch2.7/CUDA12.8 family is the starting point; exact patch-version dependencies still need resolver/import tests. [Torch-TensorRT 2.7 release compatibility](https://github.com/pytorch/TensorRT/releases/tag/v2.7.0) targets Torch2.7, CUDA12.8 and TRT10.9.

Start with one warmed B1 profile and fixed 512²/33-frame shapes. Make latent-prefix replacement and stochastic sampling explicit outside the exported denoiser. First prefix length one and recurrent length two must both be tested; do not pad history by inventing a second reference frame. [Torch-TensorRT's shape guide](https://docs.pytorch.org/TensorRT/v2.7.0/user_guide/dynamic_shapes.html) describes min/opt/max profiles; broad dynamic shapes are not required for this fixed workload.

Export surfaces / hazards:

| Surface | Main risk | First strategy |
| --- | --- | --- |
| RoPE / timestep math | Complex tensors and float64 operations; apparent export can retain unsupported ops | Precompute real constants, validate real FP32 rotation; keep outside TRT initially |
| FlashAttention/Sage custom calls | Custom kernel capture/converter coverage, version dependence | Compare supported SDPA path versus existing fast attention; leave efficient custom op in PyTorch if necessary |
| DiT MLP, projections, norms | BF16/FP16 accumulation and many partition boundaries | Test a complete block or MLP group; retain sensitive reductions in FP32 |
| LTX VAE decode | Conv3d, pixel norm, temporal padding/unpatchify; non-causal context | Exact B1 video-latent decoder partition, then smaller measured bottlenecks |
| Motion encoder | Distribution object and RNG semantics | Export tensor mean/log-variance computation; sample with existing private generator outside engine |
| Python orchestration | Mutation, branches, distributions and device checks | Keep scheduler/session/RNG management in Python; fixed tensor-only wrapper |

NVIDIA documents BF16/FP16/INT8 formats, but format availability does not guarantee converter coverage or fast kernels for this graph. [TensorRT data formats](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/data-format-desc.html). Inspect converter diagnostics and the resulting partition graph on the chosen version. Avoid dozens of tiny TRT islands that repeatedly cross back to PyTorch. A successful `torch.export` is **not** an ONNX/TRT engine, and neither implies a 25% win over the already compiled PyTorch baseline.

At the time this historical plan was written, no TensorRT/ONNX packages or engine had been installed/built. The later implementation performed the isolated CUDA TensorRT experiment; see [TensorRT experiments](../../TENSORRT_EXPERIMENTS.md), [test catalog](TEST_CATALOG.md), and the preserved OOM evidence. The plan's proposed P2 work is therefore historical context, not current status.

### P3: Scheduler and media latency without sacrificing useful throughput

- Prioritize sessions by remaining playable lead and deadline, with bounded starvation fairness. Keep first-chunk priority bounded so a stream of new callers cannot starve ongoing speech.
- Limit preparation in front of active generation; prewarm approved portrait templates. Use explicit active/rendering admission instead of interpreting ten connected peers as ten compute slots.
- Avoid waiting to fill B2 when B1 can satisfy a near deadline; measure actual occupancy. Native B2's prior 41.05 versus B1's 40.66 FPS is too small to assume a gain.
- Profile D2H/IPC before shared memory/pinned-buffer work. Reusable buffers need ownership until encoding and recurrence finish. Removing `cuda.synchronize()` alone cannot eliminate required transfer completion.
- Keep the existing H264 encoder baseline for attribution. Verify the actual active codec/context, CPU load, encode tails and session limits before testing NVENC; environment variables alone did not prove encoder activation in the older MuseTalk integration.
- Separate idle asset playback from active model compute. Streaming generated silence indefinitely consumes nearly the same whole-frame model budget as speech; cheap idle clips save compute but introduce a reconditioning/continuity problem described separately.
- Smaller first chunks might lower TTFF, but alter temporal shapes/history/attention and compile profiles. With nine overlap frames, a 17-frame window emits only eight useful frames; overhead becomes `17/8`, much worse than `33/24`. Treat as a separate quality/latency profile, not a simple queue knob.

### P4: Quantization / architectural changes only after the above

Selected DiT linear-layer INT8 or weight-only quantization may reduce memory or speed GEMMs; neither is guaranteed at B1. Use representative portrait/audio/silence/motion-prefix captures and calibrated per-layer sensitivity. Keep sensitive normalization, output head, early/late VAE stages and recurrence in higher precision initially. Do not transplant MuseTalk calibration or quantize the entire LTX VAE because its class has a similar name.

Other high-risk branches: two-step native512, 384², altered audio FPS, recurrent latent reuse, decoder overlap removal, feature caching across shifted Wav2Vec windows, distilled smaller models, or a face-only SoulX hybrid. Each changes model behavior or conditioning. The previous combined256²/two-step/15-FPS profile visibly collapsed after several seconds despite passing transport; it remains rejected. It does not isolate whether size, steps, FPS, or their combination caused failure.

## Reproducible acceptance protocol

1. Capture fixed seeds and real tensors for opening/recurrent/tail chunks, speech/silence/interruptions and multiple portraits. Pin weights, source hashes, code, dependencies, GPU and profile in artifact metadata.
2. Compare eager reference, current compiled baseline and candidate on the **same** frames/audio. Validate full trajectory: errors feed the next chunk. Require finite outputs, no batch-row contamination and no new recurrent drift. Measure mouth-region/detail and aligned face geometry, not whole-frame MAE alone.
3. Serialize/reload in a fresh process and re-run exact-shape tests with fallback disabled. Verify checksum and metadata; reject incompatible artifacts before serving. Keep old known-good engine/profile intact.
4. Run at least five warm repeats, alternating A/B order, of ten independent ten-second requests. Report median and p95 wall, useful FPS, TTFF, peak allocated/reserved/device VRAM, CPU RSS and queue/encode tails. Record video separately from timing runs.
5. Ramp actual WebRTC active speakers C1/C2/C4/C6/C10 for 30 seconds, then sustained multi-turn calls. Require complete content, bounded p95/p99 arrival gaps, unchanged lip sync, no repeat-frame inflation and healthy cleanup. Test burst arrivals and one deliberately slow receiver.
6. For the +25% goal, require at least 1.25× useful FPS against the current compiled same-quality baseline, without latency/memory/quality regressions. Do not promote on kernel FPS alone.
7. For ten native simultaneous speakers, require at least 250 useful FPS **plus operating headroom** and per-session real-time delivery. For ten mixed idle/active calls, publish the separately tested active-speaker limit and overload policy.

None of these future performance gates is represented as passed by this documentation change. The near-term engineering recommendation is P1 + profiling, then a focused P2 experiment; production migration remains gated on the whole call contract, not just a faster denoiser.
