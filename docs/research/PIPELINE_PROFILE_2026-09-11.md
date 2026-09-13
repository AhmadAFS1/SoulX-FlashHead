# FlashHead pipeline profiling and optimization experiments — 2026-09-11

September 13 follow-up: [implemented media/cache/IPC optimizations and recorded validation](MEDIA_OPTIMIZATION_2026-09-13.md). The historical measurements below are unchanged; the newer report supersedes the idle-decoding deployment status, not the neural-compute comparison.

Status: completed. Seventy-two measured GPU renders, four individual-avatar live
call tests, one ten-connected/one-speaker test, CPU decode controls, and the
63-pass regression suite are retained. The service is restored with profiling
enabled and the original rendering defaults. Equal-canvas superiority over
MuseTalk and ten-speaker smoothness were **not** achieved.

This report extends the [architecture guide](../architecture/README.md) and
[optimization plan](NEXT_OPTIMIZATION_PLAN.md). It audits the deployed inference
and media critical path, not training performance. Source is based on main HEAD
`3fcaa4219df25174b54f80ee9f5d21e0f70789ab` **plus the existing local changes**.
Those changes were preserved in an isolated worktree before new instrumentation.

Evidence directory: `/workspace/experiments/flashhead-pipeline-ARsFTh`.
Do not run historical service-control scripts or reuse old PIDs. The private
runtime snapshot in this directory contains environment secrets and must not be
committed, printed, or published.

## Measurement contract

- RTX 4070, 12,282 MiB; the unrelated OmniVoice process remained resident at
  5,312 MiB. FlashHead was gracefully paused after verifying no live sessions,
  and experiments took exclusive ownership of the FlashHead GPU lease.
- Four-step Lite, compiled DiT/VAE, cached conditioning, real RoPE, packed QKV,
  lean postprocessing, resident compact INT8 **storage with BF16 compute**.
  This is not integer matrix multiplication.
- Phone baseline: 320×576, 3,584 MiB Torch allocator cap, 25 playback FPS.
  320×576 is approximately portrait 9:16, not mathematically exact 9:16.
- Fixed real ten-second WAV, 16 kHz, seed 50, 250 useful generated frames.
  A complete clip warms each variant; three measured repetitions follow per
  avatar. Preparation, compiler warmup, quality comparison, and video encoding
  are outside the generation timer. All repeats are retained.
- Avatars: FlashHead girl; MuseTalk man; MuseTalk yongen; extracted MuseTalk
  production-idle anchor. Source paths and input hashes are in the JSON.
  The portrait tests use still anchors, not motion-driving input videos.
- GPU phase measurements are CUDA-event **intervals**, including possible GPU
  starvation between launches, not an assertion that every millisecond is a
  running kernel. Host intervals measure enqueue/CPU time and must not be added
  to GPU intervals. One warmed operator trace separates kernel work further.
- Variants run in blocks, not randomized alternating trials. Small percentage
  differences are descriptive measurements, not statistically established gains.
  Raw-pixel MAE/PSNR detect numerical changes, not lip-sync or perceptual quality.
- Finite clips skip the final feedback encode only because they are explicitly
  terminal. Persistent calls must retain recurrent state and whole-chunk tail
  padding. Offline useful FPS is not WebRTC useful FPS or ten-speaker capacity.

## Whole serving path and measured boundaries

| Stage | Implementation and work | Timing/limitation |
| --- | --- | --- |
| Avatar selection | `avatars.py` validates approved opaque IDs; `calls.py` decodes a reference and canonical idle anchor | Client call-creation wall time; reference cache hit and host preparation time |
| Reference preparation | `Engine.prepare`: content/profile-keyed LRU of up to eight immutable templates; repeated 33-frame reference VAE encode, reference Lab statistics | Cold preparation versus synchronized warm-cache wall time in offline runner |
| Speech request | Wall requests Kokoro, waits for the complete WAV, then uploads; server validates/decodes audio | `upload_read`, `upload_decode`, `transport_resample` time the ingress work; turn/admission clock starts afterward, not at LLM/TTS request start |
| Admission | Bounded turn queue; one active neural call in the selected service profile | `stage_ms.admission_wait`; queued wait is not inference cost |
| Recondition | Encode the last nine sender-displayed RGB frames to restart from actual displayed state | `stage_ms.recondition_rpc`, including worker queue/IPC; preserves endpoint contract |
| Append | Append 16-kHz audio to a private rolling state; construct 48-kHz transport audio separately | `stage_ms.append_rpc` |
| Audio | Eight-second waveform normalization, host→device copy, FP32 Wav2Vec forward, hidden-layer stack and frame-context indexing | `audio_normalize_h2d`, `wav2vec`, `audio_stack`, final `audio` indexing interval |
| Conditioning | Audio projection and per-layer cross-attention K/V once per chunk; cached rotary/timestep constants; private noise/reference tensors | `conditioning_noise_reference` |
| Denoising | Thirty transformer blocks × four passes; global full-frame self-attention, audio cross-attention and MLP | `dit_step_0` through `dit_step_3`; largest measured region |
| Decode | LTX VAE decodes five latent positions into 33 RGB frames; noncausal temporal decoder | `decode_0`; nine overlap frames cannot simply be omitted from decoder input |
| Color | Drop unused overlap before per-frame Lab correction; four-frame tiles bound activations | `color_0`; corrected final nine frames feed subsequent motion state |
| Feedback | Causal VAE encoder on the final nine corrected frames, posterior sampling with session RNG | `motion_encode_0`; skipped only for terminal finite output |
| Pixel transfer | Scale/clamp to uint8 on GPU, contiguous THWC, copy only useful frames to CPU | `uint8_0`, `transfer_0`; no float32 or discarded overlap transfer |
| Worker return | Spawned single-owner worker returns RGB through process IPC to scheduler | `rpc_wall_ms`; `rpc_overhead_ms` includes queueing, serialization and parent scheduling, not pure memcpy |
| Playout | Bounded ready chunks; monotonic persistent RTP tracks; starvation holds video and gates audio | Held/underrun/missed-slot counters; first-ready and first-media latency |
| Boundary blending | Exact sender-displayed entry anchor, cosine entry blend, canonical predecoded idle target, final blend back | Existing boundary events and automated verifier; equality is pre-encode RGB, not post-H264 receiver pixels |
| Idle | Per-peer source-video decoder or still image; no neural idle work for the source policy | Idle decode mean/max/count: wall time around the thread call, including executor queue/scheduling, not exclusive codec CPU time |
| Transport encode | aiortc H264 veryfast, two codec threads, packetization and normal rate control | Bounded 128-call encoder wall-time window; actual receiver frame timing separately |

The existing `audio_ms` aggregate also includes conditioning setup. Do not label
that whole aggregate “Wav2Vec inference.” Per-state audio labels get distinct
suffixes when profiling a batch, avoiding overwritten timing keys.

## Correctness constraints that limit easy caching

1. **Audio hidden states are not a causal KV cache.** Shifting the normalized
   eight-second window changes normalization, interpolation and noncausal
   attention context. Reusing the old hidden states changes conditioning.
2. **Motion encoding is recurrent work, not avatar preparation.** The next chunk
   depends on the final nine corrected frames. Caching it across turns/people or
   skipping it in a persistent call breaks state continuity.
3. **Decoder overlap is context.** Its temporal convolutions are noncausal.
   Lean mode safely removes overlap before per-frame color work, not before
   decoding. Spatial/temporal tiling requires derived halos and seam tests.
4. **A compiler can change the recurrent video.** Fusing BF16 color arithmetic
   can remove intermediate rounding; a small color difference becomes different
   motion latents and then different future geometry. A faster function is not
   automatically a numerically equivalent whole-video optimization.
5. **INT8 storage is a memory optimization.** Each block linear dequantizes a
   weight matrix before BF16 `F.linear`. True low-precision GEMM is a separate
   kernel and calibration project. Re-expanding every weight permanently would
   give back the memory needed by the decoder.
6. **Ten connections are not ten active generators.** Source idle is cheap, but
   ten speakers at 25 FPS demand 250 useful generated FPS before headroom. A
   single-stream 54 FPS result is only about two streams' theoretical demand;
   it does not qualify even two simultaneous real calls without scheduler tests.

## MuseTalk comparison boundaries

MuseTalk's best prior valid result on this machine is compiled PyTorch batch 8,
44.69 FPS, with a 256×256 synthesized face composited into a 512×512 canvas.
FlashHead generates the whole image, including pose/background. Equal canvas
size is an application-output comparison, not equal neural work or quality.

The older TensorRT MuseTalk artifact failed to load in the previous isolated
experiment; there is no valid local TensorRT FPS comparison. Historical results
on other GPUs are not substituted. Reference evidence:
`/workspace/experiments/flashhead-optimization/musetalk-compiled-b8-4g.json`
and the handoff in that directory. MuseTalk was not rerun in this experiment.

Higher throughput also does not mean lower first-frame latency: prior MuseTalk
first composed-frame medians were about 87–190 ms depending on batch; FlashHead
produces a 24-frame chunk before returning it. RTC first speaking media includes
reconditioning, queueing, transfer and the A/V start gate on top of model time.

## Results, candidate decisions and next priorities

### Four-avatar compiler matrix

`profile-320.json`: completed, 48 measured ten-second renders, plus four full
warmup clips. Median useful FPS across three repetitions per cell:

| Avatar | Existing baseline | Compiled color | Compiled audio | Both compiled |
| --- | ---: | ---: | ---: | ---: |
| Girl | 54.06 | 55.50 | 54.65 | 56.26 |
| Man | 53.91 | 55.77 | 54.64 | 56.55 |
| Yongen | 54.05 | 55.85 | 54.88 | 56.47 |
| Production idle anchor | 54.21 | 54.58 | 54.63 | 56.50 |
| Pooled median, 12 renders | 54.06 | 55.71 | 54.65 | 56.47 |

Both options: +4.47% pooled median throughput, not +25%. Baseline measured range
53.14–54.48 FPS; combined 55.93–56.77 FPS. Avatar content has little influence
on this fixed-shape neural workload. Baseline median first-chunk latency by
avatar is 424–432 ms; combined is 404–409 ms. This excludes preparation and RTC.

The first uncached girl template took 5,195 ms including shape/compiler setup;
subsequent newly encountered avatars took 147–194 ms. Warm template lookup and
state creation had a 2.06 ms synchronized median. Prewarming approved avatars
can remove a real cold-call penalty, but will not accelerate ongoing chunks.

Color-only raw RGB error against the corresponding baseline first repeat:
MAE 1.79–2.68/255, PSNR 32.64–36.09 dB. Audio-only: MAE 0.92–1.13,
PSNR 40.32–43.78 dB. Combined: MAE 1.70–2.78, PSNR 32.29–37.31 dB.
These are differences, not proof of worse/better perceptual quality. Sparse
girl contact sheets showed plausible frames but are not a lip-sync review.
The full paired MP4s are retained for all four avatars. These options remain
experimental and default off; they are not promoted based on FPS alone.

### Where the milliseconds go

Mean across 108 baseline steady, full, nonterminal chunks (first and last chunks
excluded from each of twelve runs). Total phase interval 422.03 ms; mean wall
422.21 ms. These are per **24 new frames**, not per single frame.

| Region | ms/chunk | Share of phase interval |
| --- | ---: | ---: |
| Audio normalize/H2D + Wav2Vec + stack/index | 13.88 | 3.3% |
| Conditioning/noise/reference | 5.90 | 1.4% |
| Four DiT passes | 232.84 | 55.2% |
| VAE decode | 116.34 | 27.6% |
| Color correction | 14.08 | 3.3% |
| Motion feedback encode | 33.96 | 8.0% |
| Uint8 layout + CPU transfer | 4.82 | 1.1% |
| Remaining bookkeeping phases | 0.21 | <0.1% |

Wav2Vec itself: 12.28 ms. Each DiT pass: 57.78–59.50 ms. Merely removing all
audio and color work would yield only about a 7.1% whole-chunk speedup and would
obviously change the model. A +25% throughput gain requires removing 20% of
total time, around 84 ms/chunk. That requires material DiT/VAE improvements,
not only Python/HTTP cleanup. Four-pass DiT and VAE decode together account for
82.8% of this path.

The 250-frame job still runs eleven full neural chunks; the last output chunk
contains only ten useful frames. Therefore 24/steady-chunk-time and measured
250/job-time are intentionally different rates.

### Operator-level evidence

One warmed nonterminal chunk, `kernels-320.operators.txt` and its gzipped Chrome
trace, confirms the active attention backend is FlashAttention. Approximate
attributed GPU costs: `aten::mm` 148.52 ms plus `aten::addmm` 33.36 ms (45.4%
combined); cuDNN convolution 141.21 ms (35.3%); FlashAttention 18.75 ms (4.7%).
Do not add framework operator rows to their corresponding kernel rows: the
profiler table contains both views of the same work. Its total self CUDA time
is 400.55 ms; profiler instrumentation makes it a different measurement from
the untraced repeated phase averages above.

The two largest fused INT8 conversion/multiply kernels consume 8.69 and 8.53 ms,
with additional smaller conversion kernels. cuDNN NCHW→NHWC and reverse layout
conversions consume 10.57 and 5.94 ms. These are concrete optimization targets,
but eliminating them alone cannot save the required 84 ms/chunk. Pageable
device→host memcpy itself is 1.98 ms in this trace; a large CPU `copy_` interval
must not be misread as 183 ms of physical PCIe copying—it includes waiting.

### Convolution and precision controls

`kernels-320.json` retains nine completed renders before a failed layout trial:

| Candidate | Median FPS | Pixel control | Decision |
| --- | ---: | --- | --- |
| Independent baseline | 54.20 | Repeats 1 and 2 exactly match repeat 0 | Stable deterministic control in this run |
| Compiled color with `emulate_precision_casts=True` | 55.43 | MAE 1.46, PSNR 40.29 dB, maximum 109 | Rounding emulation reduces drift but does not restore equality; not default |
| cuDNN autotuning, limit 10 | 53.27 | Exact RGB match | No speed win in this profile; not enabled |
| Change compiled VAE weights to channels-last | Failed | Compiled stride assertion during warmup | No measured FPS; changing frozen weight layout after compile is invalid here |

The planned graph variant in that failed script did not execute; graph testing
uses a separate subsequent process. A clean compilation/layout trial is distinct
from the failed in-place mutation and must be reported separately.

### Prioritized next implementation work

| Priority | Concrete change | Why / acceptance gate |
| --- | --- | --- |
| 1 | Tune large MLP GEMMs or implement a calibrated weight-only fused-dequant GEMM, starting with measured 1536↔8960 shapes | Matrix multiply is ~45% of GPU work; current INT8 buffers still expand into BF16 weights. Preserve session RNG, compare ten-second and long recurrent videos, measure startup and peak allocation, and retain the current kernel fallback. |
| 2 | Make VAE tensor layout consistent before compilation; tune exact 33-frame decode and 9-frame encode convolution shapes under a bounded workspace | Convolutions are ~35%; layout conversions add measurable cost. Validate intermediate strides and compile variants, then posterior parameters, RGB and seams. Do not mutate captured/compiled weight storage in place. |
| 3 | Benchmark larger TensorRT islands or a motion-encoder posterior-parameter export | Previous isolated FFN/decoder improvements do not imply whole-model gains. Keep the tested fast attention backend, control partition transfers/workspace, and leave posterior sampling in the private RNG path. Motion encode alone is only ~8%. |
| 4 | Deadline-aware B1/B2 batching and distinct generation/playout ownership | Needed to turn raw throughput into useful concurrent service. First use measured chunk durations in a CPU simulation; then verify private-state equivalence, admission, interruption and actual per-peer deadlines. Current B1 results cannot qualify B2 or ten speakers. |
| 5 | Prewarm approved avatar templates and keep call preparation out of a speech deadline | Newly encountered references cost ~150–194 ms after compile; a cached reference costs ~2 ms. Capacity/memory-bounded startup preparation improves cold calls, not steady neural FPS. |
| 6 | Use measured RPC/codec overhead to justify bounded shared-memory RGB buffers or transfer overlap | Never infer the whole RPC residual is a memcpy. Add explicit slot ownership, epoch invalidation and slow-receiver tests before replacing safe copies. Keep H264 packetization/bitrate behavior. |
| 7 | Stream TTS/audio admission and bounded next-turn render-ahead | Reduces perceived speech-onset latency; does not make the DiT faster. Requires sample-accurate turn ownership and preserving exact entry/exit anchors and barge-in rollback. |

Replacing FlashAttention is lower priority than GEMM/VAE work: even eliminating
all its measured GPU time yields only about 5% improvement on this profile.
Removing HTTP authentication, increasing the connected-peer limit, or displaying
idle frames at 25 FPS does not raise useful generated throughput.

Ten 25-FPS active speakers require at least 250 FPS, ~4.6× this phone baseline;
300 FPS with 20% extra capacity requires ~5.5×. That is a different-scale target
from a 25% optimization. No migration recommendation should rely on ten idle
connections or on reducing resolution/steps without a quality gate. The previous
low-resolution/two-step visual failure is a reason not to present fewer steps
as a same-quality performance win.

### Same-canvas square check

`profile-512.json`: completed, nine measured renders after separate warmups.
Four steps, 512×512, 4,096 MiB allocator cap, same girl portrait/audio/seed:

| Variant | Median useful FPS | Repeats | Maximum measured Torch allocation |
| --- | ---: | --- | ---: |
| Existing baseline | 42.60 | 42.62, 42.04, 42.60 | 3,596 MiB |
| Compiled color + audio | 43.72 | 43.84, 43.66, 43.72 | 3,596 MiB |
| DiT CUDA graph, eager color/audio | 42.53 | 42.53, 42.25, 42.54 | 3,605 MiB |
| Prior compiled MuseTalk batch 8 reference | 44.69 | Five historical repeats | Different neural work/runtime; see comparison contract |

The combined candidate is +2.63% over FlashHead's baseline, still 2.16% below
the previous MuseTalk median. It also differs from baseline RGB (MAE 3.20,
PSNR 30.17 dB). Graph output exactly matches baseline but supplies no speed win.
This experiment does **not** establish FlashHead superiority at equal canvas
size, much less against a functioning tuned MuseTalk TensorRT deployment.

### A higher-impact MuseTalk-style architecture option

MuseTalk's decisive advantage is not just TensorRT: it synthesizes only the
face crop and reuses the full-frame body/background, masks and pose assets.
An experimental hybrid could generate a tightly framed SoulX head/face patch
and compose it onto the existing idle/full-body plate. This would reduce neural
tokens and decoded pixels while retaining a full-resolution delivery canvas.

That is **not implemented or benchmarked here**, and is not equivalent to
downscaling the final video. It needs stable face tracking, compatible head pose,
identity/lighting matching, masks, jaw/neck blending, and per-avatar calibration.
SoulX generates head motion, so naive pasting can introduce double edges and
neck seams. It also gives up unconstrained full-frame/head-motion generation.
Prototype it as a separate renderer/quality tier with synchronized real footage
and compare lip-sync/identity and boundaries before considering migration.
This architecture change is a more plausible route to a large throughput gap
than expecting another attention flag to remove most of the remaining cost.

### Clean layout retry

`layout-320.json` completed six measured renders. Invalidating Dynamo's compiled
assumptions before changing VAE weight strides fixed the earlier stride error.
No cuDNN autotuning was enabled in this retry. Baseline median 54.21 FPS;
channels-last median 55.33 FPS (+2.07%), with three individual values
55.33 / 54.93 / 55.51. It required 47.99 seconds of new warmup/compilation.
Raw RGB differs (MAE 1.10/255, PSNR 41.08 dB); the baseline's other two repeats
were exact matches. This is a valid small experimental gain, not an unchanged
output claim. It was not combined with other options or promoted to the service.

### Implemented instrumentation and test coverage

The main checkout now contains CUDA audio substage and host-interval timing,
per-turn ingress/admission/reconditioning/append/first-ready timing, worker RPC
wall/residual timing, bounded H264 encoder timing, idle decode timing/counts,
and approved-avatar selection in the persistent-peer benchmark. Detailed GPU
timing is opt-in via `--profile`; status/health JSON exposes the measurements.
`--compile-color` and `--compile-audio` are new default-off experiment switches.
No frame/audio ownership, boundary-blend math, pose behavior, authentication
policy, resolution, precision-storage policy or admission limit was relaxed.

`full-tests-final.log`: **63 passed, one skipped, 32.89 seconds** with CUDA hidden
from the test process. The skipped test is the explicitly opt-in Chromium wall
smoke test. This suite includes real local H264/Opus peers with deterministic CPU
fixtures, endpoint verification, interruption/retry/cleanup, avatar catalog,
recording and bounded-encoder-stat checks; it is not a GPU quality certification.
Actual GPU/WebRTC calls are a separate test below. Existing user changes were
preserved; only the new scoped patch was applied from the isolated worktree.

### Reproduction and evidence

Drivers are retained in `benchmarks/pipeline_20260911/`. Use fresh output paths;
the drivers reject an existing result file. Offline GPU tests require a planned
maintenance window, sufficient co-resident headroom, and no other FlashHead
GPU owner. Never stop unrelated services or reuse a historical PID.

Run from the repository with its existing venv and model links. Common offline
environment: `PYTHONPATH=.`,
`TORCHINDUCTOR_CACHE_DIR=/workspace/SoulX-FlashHead/.torchinductor`,
`TORCHINDUCTOR_COMPILE_THREADS=4`,
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Commands below omit the common environment assignments and use placeholder
`FRESH` output paths that must be replaced with a new experiment directory:

```bash
.venv/bin/python benchmarks/pipeline_20260911/profile_avatars.py --output FRESH/profile-320.json
.venv/bin/python benchmarks/pipeline_20260911/kernel_candidates.py --output FRESH/kernels-320.json
.venv/bin/python benchmarks/pipeline_20260911/profile_avatars.py --width 512 --height 512 --budget 4096 --variants baseline both graph --avatars girl --output FRESH/profile-512.json
.venv/bin/python benchmarks/pipeline_20260911/kernel_candidates.py --variants baseline channels_last --reset-for-layout --output FRESH/layout-320.json
.venv/bin/python benchmarks/pipeline_20260911/run_live_calls.py --output-dir FRESH
env CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/pytest -q tests
```

The default kernel sweep deliberately retains the original failed in-place
layout trial; use the separate reset-for-layout command for the successful
version. Earlier offline JSON files retain their own three core-source hashes;
`environment-final-source.json` inventories the final installed Python source
and package versions. Instrumentation-only refinements occurred between runs;
do not claim that every evidence file used byte-identical instrumentation.
The initial `telemetry-only.patch` is an installation aid, not a complete final
diff (ingress timing was added afterward). Original dirty-tree changes are not
part of a new optimization claim.

All raw JSON, logs, paired MP4s and operator traces remain under the evidence
directory above. In particular, `profile-320-baseline-{girl,man,yongen,idle}.mp4`
and `profile-320-both-{girl,man,yongen,idle}.mp4` are complete ten-second clips.
No model weights, secrets, environments or machine-specific engine binaries
are copied into the documentation/driver directory.

## Actual GPU/WebRTC results

`call-{girl,man,yongen,default}.json` and matching receiver MP4s: each individual
avatar completed two five-second speech turns on one persistent H264/Opus peer.
The default source-video avatar additionally completed interruption/recovery.
Every run passed transport, boundary verification and peer/GPU-state cleanup.
All had zero **reported generation underruns**, but this is not a blanket claim
of zero pacing/audio imperfections:

| Avatar | Receiver wire FPS | Normal-turn first speaking media | Initial call creation | Missed video slots |
| --- | ---: | --- | ---: | ---: |
| Girl | 25.01 | 0.619–0.709 s | 3,790 ms | 0 |
| Man | 25.01 | 0.628–0.637 s | 416 ms | 0 |
| Yongen | 25.01 | 0.616–0.671 s | 295 ms | 0 |
| Production idle video | 24.84 | 0.714–0.726 s | 183 ms | 2 |

The first girl call includes cold parent-process/image utilities and reference
preparation; this HTTP timer does not isolate their individual contributions.
It must not be reported as 3.79 seconds of VAE inference. The already-warmed
default anchor has a different cache history. Yongen counted 960 audio-hold
samples (20 ms); the default run, including interruption/recovery, counted
4,800 (100 ms). Still-image avatars count their normal idle holds in
`held_frames`; those 64 held frames are not 64 inference stalls. Default's
interrupted and recovery turns started media after 0.852 and 0.671 seconds.

The normal source-video turns spent 70.5–84.2 ms in reconditioning RPC. The
single-peer rolling H264 encoder mean was 6.47–7.15 ms/frame. Per-request PCM
resampling was approximately 20 ms in the single-peer fixtures; it still runs
on the event-loop thread, so offloading it is a concrete burst-latency task.
These tests use uploaded fixture WAVs, not Kokoro synthesis or an external
Safari/phone/network path.

### Ten connected peers: a media-scaling failure, not a pass

`call-ten-connected.json`: ten independent peers, one speaker, two five-second
turns; the four avatar IDs were cycled across peers. All transport and cleanup
checks passed; the speaking peer's exact sender-endpoint verification passed.
However, **six underrun slots** were recorded and strict smoothness failed.

- Speaking girl: 24.69 wire FPS, 0.815–0.826 s first media, six missed slots.
- Other still-image peers: about 24.49–24.62 wire FPS, eight or nine missed slots.
- Two production-video idle peers: **11.87–11.90 wire FPS**, 212 and 214 missed
  slots. Their mean awaited idle decode times were **57.68 and 58.50 ms/frame**,
  exceeding the 40 ms 25-FPS frame budget before encoding.
- Encoder means rose to 11.04–12.93 ms/frame across peers.
- Mean generation RPC residual rose from **34–44 ms/chunk** in single-peer
  tests to **173 ms/chunk**. Mean engine wall rose only to 454 ms, while full
  RPC wall reached 627 ms. This residual includes queueing, serialization and
  event-loop scheduling: it is not a measured 173 ms physical copy.

The RPC aggregates use each report's final ten chunk samples, not a complete
call trace; first-media and admission values come from each turn's own record.
Each chunk is 12.66 MiB of RGB at 320×576, so worker serialization is worth
isolating. The local benchmark also decodes all peers and records peer zero on
this same host; it includes client CPU load and is not a server-only capacity
measurement. It nevertheless demonstrates that the tested wall configuration
is not smooth. No ten-active-speaker test or claim is made.

### Isolating the idle cost

`idle-cpu.json` / `profile_idle_decode.py`: the existing source is H264,
480×832, 24 FPS, 241 frames. In a separate CPU-only sequential diagnostic,
default and 1/2/4-thread decoders all took roughly 15.5–15.7 ms per output-frame
request after warmup. Changing decoder thread count alone did not help.

A 55-frame split (after five warmup frames) measured 14.31 ms decoding,
1.82 ms reformat-to-RGB, and 1.51 ms with a persistent reformatter. All sixty
compared RGB frames matched exactly. That small reformatter saving does not
explain or fix the 58 ms awaited live cost. Additional live thread/executor and
CPU contention matters. The container exposes 64 CPUs but has a cgroup-v1 CPU
quota of 15.36 CPU equivalents; the host's thread count is not an unlimited
CPU budget. These are diagnostic controls, not a successful ten-peer rerun.

## Revised implementation order for this product

For **neural FPS**, prioritize large GEMM/quantized-kernel and VAE work from the
table above. For the **WebRTC wall**, the fresh live evidence changes the order:

1. Add a bounded shared **CPU idle-frame cache** for approved assets/profiles,
   predecode off the event loop, and keep independent per-peer playback indices.
   One 241-frame 320×576 RGB clip costs about 127 MiB; share immutable frames
   across peers instead of decoding that same asset per peer. Define LRU/byte
   limits, source-content/profile invalidation, exact canonical endpoints,
   wraparound, read-only storage and fallback before implementation. Avoid
   cloning a whole RGB cache for each caller. This targets media deadlines,
   not neural FPS, and has **not** been implemented in this task.
2. Separate idle/media executor capacity from other CPU work and move ingress
   resampling off the event loop. Measure queue wait separately from actual
   decoding/encoding, then rerun the mixed-video ten-connected fixture.
3. Isolate worker serialization and introduce a bounded shared-memory RGB ring
   only with explicit slot ownership/cancellation/slow-reader tests. Preserve
   the process boundary that protects media callbacks from the model's GIL work.
4. Only after these gates pass, evaluate two **active** speakers with measured
   deadlines and B1/B2 batching. Do not raise the production active-call limit
   from one simply because ten transports connect.

The useful outcome is a measured optimization map, small opt-in compiler gains,
and newly exposed media bottlenecks—not a claim that all optimizations are done
or that migration is justified. Keep MuseTalk available; qualify a SoulX quality
and concurrency tier before replacing it.

## Restored service

The existing service was restored from its freshly captured launch, with only
`--profile` added. Health is ready at port 1111, 320×576/four steps/25 FPS,
compact resident INT8-storage, 3,584 MiB allocator cap, one active call, ten
connected-call limit, source idle, H264 veryfast, anonymous access. Both new
compiler switches remain off. All benchmark calls and GPU states were released.
The unrelated OmniVoice service remained running. `health-final.json` records
the final check; use live health/process discovery rather than any recorded PID.
