# Iteration log: toward 40 FPS @ 576×320 or 56 FPS @ 448×256

**Goal (user, 2026-09-21):** ≥40 FPS at 576×320, or ≥56 FPS at 448×256, with acceptable video
quality, via quantization and pipeline optimization — not resolution alone. Document every change
and reference the docs as iterations land.

**Audience:** the engineer continuing this. Living document; newest iteration at the bottom.

## Hardware and provenance

RTX 4070 SUPER, SM89 (Ada), 12,282 MiB visible (physical class **unverified**), driver 595.84,
Torch 2.7.1+cu128, CUDA 12.8, TensorRT 10.9.0.34, SageAttention 2.2.0 SM89. Co-resident:
one idle `ltx23-soulx-transfer` process (234 MiB); OmniVoice TTS stopped. All numbers are
**fresh local GPU inference** on a quiet GPU unless marked `[E]` (estimate) or `[diag]`
(instrumented, not a throughput claim). Workload: 250 frames, seed 50, `indian150-a`, 2 steps.

## Starting point

| build | FPS | total s | dit | vae_decode | motion_enc | doc |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 576×320 full stack | 18.90 | 13.23 | 3.578 | 8.134 | 1.117 | `PRO_PHASE123_RESULTS_2026-09-21.md` |
| 448×256 + output reuse | 32.13 | 7.78 | 2.201 | 4.595 | 0.695 | `PRO_448x256_30FPS_2026-09-21.md` |

## The constraint that decides where effort goes

| target | budget | decoder alone today | verdict |
| --- | ---: | ---: | --- |
| 40 FPS @ 576×320 | 6.25 s | **8.13 s** | decoder exceeds the whole budget by 30% |
| 56 FPS @ 448×256 | 4.46 s | **4.60 s** | decoder exceeds the whole budget by 3% |

The decoder is FP16 TensorRT with convs measured compute-bound, INT8 at parity (Ada shares the
tensor-core tier), tactic forcing measured 0–7 ms — i.e. **at its floor for the work it does**
(`PRO_THROUGHPUT_PLAN_2026-09-21.md` §3, `PRO_PHASE123_RESULTS_2026-09-21.md`). So the targets
require the decoder to do **less work**, or **more than one stream** on the idle SM capacity
(convs run at 30–46% of peak). No remaining precision tier exists on SM89.

## Plan

1. Profile 448×256 — where did the time go after the resolution change? (engines may not be at
   floor at the new, smaller tile shapes)
2. Recalibrate the static INT8 activation scale at 448×256 (calibrated at 6,480 tokens; now 4,032)
3. Map the resolution curve with one intermediate point (512×288) so the trade is a curve, not two dots
4. **Concurrency with shared weights**: two streams in one process sharing DiT, VAE and the TRT
   arena — the structural lever for aggregate throughput, and the user's original goal
5. Decoder work reduction — whatever the profile says is not at floor

## Iteration log

### Iteration 1 — profile the 448×256 build `[diag]`

Artifact: `benchmarks/pro_30fps_20260921/latency-448/latency-summary.json` (`--latency-detail
stages`, 21 engine calls/window, output reuse on). Instrumented run; proportions are the result.

| component | 576×320 ms/win | 448×256 ms/win | ratio |
| --- | ---: | ---: | ---: |
| decoder `trt.execute` | 617.1 (42.2%) | **379.8 (44.7%)** | 0.615 |
| DiT forward | 393.6 (26.9%) | 242.4 (28.6%) | 0.616 |
| decoder non-engine ops | 246.3 (16.8%) | 115.8 (13.6%) | 0.470 |
| motion VAE encode | 156.9 (10.7%) | 77.1 (9.1%) | 0.491 |
| casts in+out | 14.7 | 9.2 | 0.63 |
| **window** | **1,463.9** | **848.9** | **0.580** |

Per-engine share is unchanged: tail `12-13-14` 54%, mid `8-9-10` 34%, low `4-5-6` 12%.

**Reading:** every GPU-bound component scaled with the 0.622 area factor. The engines did not
find better tactics at the smaller tiles; they are at floor at both shapes. 56 FPS = 496
ms/window needs −353 ms (−42%) from a window whose engine compute alone is 380 ms.
**Single-stream 56 @ 448×256 is closed by the same arithmetic as 40 @ 576×320.** The only
structural lever left is running more than one stream against the idle SM capacity.

**Finding that reshapes the plan:** `soulx_rtc/engine.py` already implements shared-weight
multi-session generation for LITE — docstring "Session state is small; model weights are
shared", `generate(states)` over a list of `GenerationState`, DiT microbatched across
sessions, decode kept sequential. It hardcodes `"lite"`. The pipeline itself keys reference
latents per `person_name` in dicts. A PRO two-stream harness is a port, not an invention.

### Iteration 2 — recalibrate the static INT8 activation scale at 448×256

Artifact: `benchmarks/pro_30fps_20260921/calib-448/results.json` → `int8-amax-448.json`
(`--calibrate-int8`, diagnostic run, 60 sites, 4,032 tokens).

The scales shipped at 576×320 were calibrated at 6,480 tokens
(`PRO_PHASE123_RESULTS_2026-09-21.md` §2.1) and were deliberately **not** applied to the
448×256 arm in `PRO_448x256_30FPS_2026-09-21.md` until re-measured. Result:

| site | amax @ 576×320 (6,480 tok) | amax @ 448×256 (4,032 tok) |
| --- | ---: | ---: |
| `ffn.2` min over 30 blocks | 7.41 | 7.14 |
| `ffn.2` max over 30 blocks | 30.88 | 30.85 |

**The per-site activation range is resolution-invariant** — the max is a per-row property of
the block, not of the token count. The 576×320 scales would have been safe; the 448×256 set is
used anyway for matched provenance. Expected gain `[E]`: the 576×320 measurement was −30.3
ms/window on a 1,463.9 ms window; scaled by the 0.62 area factor that is ≈ −19 ms on an
849 ms window, ≈ +2.2%.

**Measured `[M]`** (`benchmarks/pro_30fps_20260921/lowres-scaled/results.json`, RTX 4070 SUPER,
fresh GPU inference, 250 frames, seed 50, `indian150-a`, output-buffer reuse on in both arms):

| arm | FPS | DiT s | motion enc s | VAE decode s | VRAM MiB (nvidia-smi peak) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lowres-fast` — dynamic ffn.2 activation scale | 32.13 | 2.201 | 0.695 | 4.595 | 8,807 |
| `lowres-scaled` — static ffn.2 scale from `int8-amax-448.json` | **32.98** | **2.023** | 0.694 | 4.583 | 8,769 |

DiT −178 ms over 9 windows = −19.8 ms/window, i.e. the estimate landed within 1 ms. FPS
+2.6% (32.13 → 32.98). This is now the 448×256 reference build.

**Quality gate.** Two reviews, because the two say different things:

| pairing | opening corr | oral edge ratio | mouth-centre dist | what it measures |
| --- | ---: | ---: | ---: | --- |
| vs 576×320 control `p1-ctl-raw` (`review-lowres-scaled/`, resolution drift waived) | 0.918 | 1.121 | 63.2 px | independent trajectories at two resolutions — read distributionally; the previous 448 build scored 0.939 / 1.070 / 61.4 against the same control |
| vs same-resolution dynamic-scale build `lowres-full` (`review-lowres-scaled-vs-full/`) | **0.968** | **1.051** | **2.8 px** | aligned trajectories (same seed, same latent shape): the isolated effect of the static scale |

The same-resolution pairing is the one that isolates the change: lip motion tracks the
dynamic-scale build (0.968, 2.8 px centre distance) and the static scale adds ≈5% oral edge
energy — between the 1.026 accepted at 576×320 and the 1.149 that got the all-FFN variant
rejected. Colour drift over 8 windows (`colour_drift.py`, control = `lowres-full`): slope
R +0.018 / G +0.007 / B +0.000 per window, max 0.181/255 (control 0.284/255). Visual check of
`review-lowres-scaled-vs-full/mouth-comparison.mp4`: accepted.

### Iteration 3 — GPU headroom probe (decides whether concurrency can pay)

Artifacts: `benchmarks/pro_30fps_20260921/probe-contended/results.json`; matmul rates in the
session scratchpad. Method: a saturating FP16 4096² matmul loop ran alongside the 448×256
pipeline; each was measured solo and concurrent.

| load | solo | concurrent | kept |
| --- | ---: | ---: | ---: |
| matmul (it/s) | 486.2 | 365.4 | 75% |
| pipeline (FPS) | 32.13 | 15.40 | 48% |
| **sum of fractions** | | | **1.23** |

A sum of 1.0 would mean the GPU was already saturated and concurrency could only time-slice;
1.23 means ~23% of a second workload runs *for free* in the gaps of the first. The probe
load is greedier than a second pipeline stream would be, so a two-stream aggregate is
plausibly 1.2–1.4× single-stream — **a real lever, and the only structural one left, but not
a doubling.** Against 56 FPS @ 448×256 that is 38–45 FPS aggregate, so concurrency plus the
remaining single-stream squeezes is the path, and the target should be read as aggregate.

Enabler landed in the same iteration: `install_overlap_skip` is now **session-keyed**
(`set_overlap_skip_session`), fixing a latent bug — its persisted decoder cache was per-VAE,
so any shared-VAE multi-person use (which `engine.py` already does for LITE via
`copy.copy(self.pipeline)`) would have handed session B session A's tail.


### Iteration 4 — two concurrent sessions on one GPU (the aggregate lever)

Code: `benchmarks/pro_quantization_v2_20260918/sessions.py` (new, in `_sources()`),
`run.py --sessions N --session-seed-stride K`, `soulx_rtc/pro_vae_stage_backend.py`
(engine output pools keyed per caller stream), `soulx_rtc/pro_attention_backends.py`
(stream probe + fence), `flash_head/src/pipeline/flash_head_pipeline.py` (eight
`torch.cuda.synchronize()` timing fences in `generate()` gated on `verbose_timing`; they
only ever bracketed `time.time()` prints, and the harness times with CUDA events).

**Design.** One pipeline object, one set of weights and TensorRT engines; per session a
CUDA stream, a noise generator, the motion-feedback latents, the overlap-skip decoder
cache (`set_overlap_skip_session`), a rolling audio history and pinned host buffers. The
host issues windows round-robin with at most one window per session in flight; the D2H
copy is `non_blocking` into pinned memory behind an event instead of `.cpu()`. Sharing one
TensorRT execution context across streams is safe only because the stage backend already
runs every engine call on one engine stream fenced to the caller stream.

**Three bugs found on the way, each of which single-stream measurement could never show:**

1. *Cross-stream tensor race in my own code.* The first version cloned the reference latent
   on the default stream and read it on the session stream. `torch.cuda.Stream()` streams
   are non-blocking with respect to the default stream, so the DiT was seeded with whatever
   the block held. Window 0 came out non-finite. Fix: produce per-session tensors on the
   session stream after a `wait_stream` on the producer.
2. **SageAttention SM89 ignores the current CUDA stream.** After fix 1 the run still
   produced non-finite pixels, and the device-side check proved they were real. A probe
   (`kernel_respects_current_stream`: build q/k/v at the end of a ~100 ms dependent matmul
   chain on a fresh stream, call the kernel there, compare with a synchronised reference)
   showed `sageattn_qk_int8_pv_fp8_cuda` **and** `sageattn_qk_int8_pv_fp16_cuda` differ
   from their reference on 3/3 trials, while `torch._int_mm`, `torch._scaled_mm` and
   `flash_attn_func` match bit-for-bit. The source confirms it: every launch in
   `csrc/qattn/sm89_*.cu` and `csrc/fused/fused.cu` of the 2.2.0 build at
   `/workspace/experiments/pro30-deps/SageAttention-sm89` is `<<<grid, block, smem>>>` with
   no stream argument, i.e. the legacy default stream. Every single-stream number in this
   programme was unaffected (everything else was on the default stream too). Two fixes:
   * *landed now:* the backend probes the kernel once at install time (outside any compiled
     region) and, if it ignores the stream, fences every call through the default stream
     (`default.wait_stream(current)` → kernel → `current.wait_stream(default)` +
     `record_stream`). The probe result is in the run manifest as `stream_safe`.
     `SOULX_SAGE_FORCE_FENCE=1` forces the fence for A/B measurement.
   * *building:* `/workspace/experiments/pro30-deps/SageAttention-sm89-stream`, the same
     source with every launch given `at::cuda::getCurrentCUDAStream()`, installed to
     `sage-sm89-stream` (the original install is untouched for reproducibility).
   The first probe version overflowed fp16 in its matmul chain and turned into a NaN test;
   the shipped probe layer-normalises each step. Lesson recorded because it briefly made the
   fence look broken.
3. *VRAM.* With a decode stream per session the run OOM'd in warmup (11.2 GiB in use,
   9.56 GiB allocated): the caching allocator keeps free blocks per stream, so two decode
   streams meant two sets of decoder activations, two engine output pools, and — because
   every stage call `record_stream`s its buffers on the engine stream — two windows of
   queued engine buffers pinned until the GPU reached them. Fix: **all decodes on one
   shared decode stream, one decode in flight**; the host issues a decode only after the
   previous one (either session) has completed, and while it waits the GPU runs the other
   session's DiT. Decodes were serialised on the engine stream anyway, so this costs no
   overlap that mattered: DiT(B) ‖ decode(A), and A's colour/encode/copy tail ‖ decode(B).

**Measured `[M]`** — RTX 4070 SUPER (12,282 MiB visible), fresh GPU inference, 250 frames per
session, seed 50, `indian150-a`, 448×256, recalibrated static ffn.2 scale, output-buffer
reuse on unless stated. `useful_fps` in multi-session rows is the AGGREGATE (sessions × frames
/ wall); "window p50" is the median wall time of one session's window from issue to
completion. VRAM is the nvidia-smi peak during generation.

| run | sessions | SageAttention | engine pools | aggregate FPS | per session | VRAM MiB | window p50 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| `lowres-scaled` (reference) | 1 | 2.2.0 as shipped | shared | **32.98** | 32.98 | 8,769 | 0.825 s |
| `single-stream-r01` | 1 | stream-patched | shared | 32.38 | 32.38 | 8,835 | 0.836 s |
| `sched1-r01` — scheduler path, 1 session, own streams | 1 | stream-patched | per session | 32.85 | 32.85 | 10,085 | 0.823 s |
| `sched1-default-r01` — scheduler path, default stream | 1 | stream-patched | per session | 33.10 | 33.10 | 8,919 | 0.823 s |
| `conc2-r01` | 2 | as shipped + fence | shared (aliasing bug) | 33.40 | 16.70 | 11,871 | 1.637 s |
| `conc2-r02` | 2 | stream-patched | shared (aliasing bug) | **33.43** | 16.71 | 11,471 | 1.630 s |
| `conc2-r03` | 2 | stream-patched | per session | OOM in warmup | | | |
| `conc2-r04` | 2 | stream-patched | off (fresh outputs) | 30.24 | 15.12 | 8,593 | 1.805 s |

Single-stream runs scatter 32.4–33.1 (±1%, consistent with the 0.86% sd established earlier),
so the patched SageAttention build, the scheduler path and the pinned asynchronous delivery
are all throughput-neutral. **Two concurrent sessions: +1.3% aggregate (33.43 vs 32.98).**
Each session's window takes 1.63 s instead of 0.83 s — the two streams time-slice the GPU
almost exactly, they do not fill each other's gaps.

**Fourth bug, found by the isolation check, not by timing.** With the shared engine pools
the two sessions were identical in window 0 and diverged from window 1 (mean |Δ| 0.5 →
3.7/255). The overlap-skip shim persists the decoder's causal cache across windows, and
under `reuse_outputs` those cache tensors ARE the ping-pong pool buffers; session B's seven
calls per engine rewrite both slots that session A's next window reads as `cache_in`. Fix:
pools keyed by (caller stream, session) (`set_output_pool_key`). That fix needs a second set
of pools (+~0.6 GB) and OOMs at 448×256 (`conc2-r03`); with reuse off (`conc2-r04`) **the two
sessions are bit-identical for all 250 frames** (`session_outputs[1].max_abs_diff_vs_session0
= 0`), at the known −8% cost of allocator churn. Also confirmed on the way: the stock
single-stream path is bit-reproducible across processes (`s30-base-raw` = `s30-reuse` at
576×320; `lowres-scaled` = `single-stream-r01` at 448×256, across two SageAttention builds),
and micro-tests show bf16 GEMM, `_int_mm`, `_scaled_mm`, SageAttention and cuDNN conv3d are
bitwise insensitive to buffer alignment and to the stream they run on.

**Why there is nothing to gain — the traces say the GPU is already busy.** Re-reading the
iteration-1 profiler traces (`dit-trace-r01`, `t3-trace-r01`, 576×320) for GPU idle time
rather than kernel totals: the GPU is busy 95% of the traced span. The idle is two things,
both host-side: (a) **~55 ms per window of `cudaFree` storms** — 46 frees of ~2.2 ms each,
i.e. the caching allocator releasing its cached blocks (each `cudaFree` is a device sync)
before a large decode allocation, the "reserved pool oscillation" that output-buffer reuse
was built for; at 448×256 with reuse on, `torch_reserved_mib` is **flat at 8,060 MiB** for
the whole run, so this gap is already gone; (b) ~25 ms per window of blocking device-to-host
copy (`cudaMemcpyAsync` of the fp32 window) plus ~15 ms of host work between windows, i.e.
~3–4% at 448×256, which the scheduler's pinned asynchronous delivery removes — and that is
the +1% that the scheduler rows show. Everything else is kernels back to back. The
iteration-3 headroom probe (sum of fractions 1.23) measured how well a *dense FP16 GEMM*
co-schedules with the pipeline, not how a second copy of the pipeline does; two copies
contend for the same tensor cores and memory system and get time-sliced.

**Conclusion for the goal.** Concurrency is not a lever on this card for this pipeline:
aggregate FPS ≈ single-stream FPS. The two-session harness stays as a validated tool
(isolation check, stream-safety probe, per-session pools) but the target must be met by
removing GPU work from the single stream. The measured single-stream budget at 448×256 is
849 ms/window: engines 380 (FP16 floor, iteration 1), DiT 243, decoder non-engine ops 116,
motion encode 77, audio 8, colour 5, host ~20. Reaching 56 FPS (496 ms) needs −353 ms;
reaching 40 FPS at 576×320 needs a 625 ms window against a decoder that alone takes 505 ms
at 448×256 scale. Neither is closable by the remaining non-engine levers (their entire sum is
~220 ms at 448×256), so the next iterations attack them anyway for what they are worth and
the target is reported against the arithmetic, not around it.

Videos (labelled, standing preference): `benchmarks/pro_30fps_20260921/visual/conc2-r04-threeway.mp4`
(single stream 448×256 32.98 FPS | concurrent session A | concurrent session B, 30.24 FPS
aggregate) and `conc2-r04-mouth-threeway.mp4` (mouth crop). Sessions A and B are bit-identical
by construction of that run, which is the point of the recording.

**Numerics of the scheduler path — resolved, and it is not a scheduler bug.** The
scheduler path is deterministic and stream-independent (`sched1-r01` = `sched1-default-r01`
= `conc2-r04` session 0, all sha `8ea13f58…`) but differs from the classic loop from window
0 (mean |Δ| 2.9/255, 69% of pixels in frame 0, max 70). Per-stage tensor dumps
(`SOULX_DUMP_DIR`, `dump-classic-r01` vs `dump-sched-r01`) put the first difference inside
the very first DiT forward of the warmup: **identical `x`, `timestep`, `context`, `y` in,
different output out** (max 0.58, mean 0.036 on a bf16 tensor). Kernel numerics are
exonerated by the alignment/stream micro-tests, so the compiled artifact itself differs.
Confirmed by `classic-freshcache-r01`: the **classic** path run with an empty
`TORCHINDUCTOR_CACHE_DIR` reproduces window 0 of `lowres-scaled` exactly and then differs by
2.0–2.7/255 in windows 1–2. Inductor autotunes its pointwise/reduction kernels among several
block configurations at first run and caches the winner on disk; a different reduction split
is a different accumulation order, the 2-step distilled denoiser amplifies that to visible
pixel differences, and every earlier "bit-reproducible across processes" observation was
cache reuse. Consequence for measurement: sha equality across runs is only meaningful within
one compile-cache family; quality gates stay the arbiter. Consequence for the scheduler:
none — its outputs are a second, equally valid family.

### Iteration 5 — inside the TensorRT engines: a quarter of the "floor" was layout copies

Iteration 1 called the stage engines a floor because their *convolution* FLOP rate did not
move with resolution. Re-reading the iteration-1 profiler trace by kernel family instead of
by stage showed that the engine time is not all convolution: `permutationKernelPL`
(cuTENSOR layout permutes), `cuInt8::nhwcTonchw`, `cuSliceLayer::naiveSlice` and
`cuReduceLayer` kernels inside the engines summed to ~26% of `trt.execute` at 576×320.
The TensorRT per-layer profiler (`IProfiler`) on the 448×256 engines, steady-state
signatures, 7 calls per engine per window:

| stage engine (448×256) | ms/call | conv | reformat | pointwise | reduce | slice |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `upsamples[12-14]` (96 ch, 4 frames 448×256) | 28.7 | 59% | **24%** | 7% | 6% | 4% |
| `upsamples[8-10]` (192 ch, 224×128) | 20.5 | 73% | 15% | 5% | 4% | 3% |
| `upsamples[4-6]` (192→384 ch, 112×64) | 6.2 | 85% | 9% | 2% | 4% | — |
| **per window (×7 ×3)** | **388** | 260 (67%) | **73 (19%)** | 23 | 20 | 12 |

The engine inspector explains it: TensorRT runs the convolutions in a channel-vectorized
layout (`1:8`, i.e. DHWC8) and the RMS norm (`ReduceL2` over C) plus the pointwise chain
in the linear layout, so every convolution gets a "Reformatting CopyNode" on its input
and its output, the causal-cache `Slice`/`Concat` get more, and even the fp16→fp16
identity `Cast` left by the bf16→fp16 export conversion became a Reformat layer. 17–30
Reformat layers per engine around 6–7 convolutions.

**Experiments (tail stage, steady signature, real calibration sample, shipped engine as the
reference — which is bit-reproducible run to run):**

| variant | reformat layers | ms/call | max Δ vs shipped (y, ref max 76.6) | verdict |
| --- | ---: | ---: | ---: | --- |
| shipped graph rebuilt | 29 | 29.1 | — | control |
| DHWC8 network I/O | 36 | 29.6 | — | worse: TRT still runs the reduce linear and converts x back |
| DHWC8 I/O + `DIRECT_IO` | build fails | | | "no conformant implementation" |
| graph `clean` (identity casts, Expand, +0, dynamic Pad folded into Conv pads) | 29 | 27.3 | **0.000 (bit-identical)** | exact, removes 6 no-op layers |
| `clean` + **norm as 1×1×1 conv** of (x/64)² | **20** | **26.1** | 0.125 (0.16%) | Reduce layers gone; kept |
| + DHWC8 for cache tensors only | 26 | 28.4 | 0.125 | worse |
| **FP8 Q/DQ convolutions** (opset 19, e4m3) | 0 | 29.1 | 1.57 (2%) | TRT 10.9 on this GPU has no FP8 Conv3d kernel: the convs become a generic "correlation" layer, same speed, worse accuracy. Dead, consistent with the earlier per-conv finding. |

The rewrite is `soulx_rtc/pro_stage_onnx.py` (`clean` exact; `norm_conv` numerically
equivalent: the 1/64 pre-scale keeps the fp16 sum of squares in range and is a power of
two), applied by `decoder_stages.py build --graph-rewrite norm-conv` and recorded per
engine in the plan. Two bugs on the way worth one line each: protobuf repeated-field
wrappers have unstable `id()`s (bookkeeping by node name), and `del graph.node[:]`
empties nodes still referenced elsewhere (copy first). The 9 engines rebuilt in 57 s
(`stage-fp16-448x256-normconv-r01`); against the bf16 PyTorch stage on the calibration
samples their max error is unchanged from the shipped build (e.g. tail y 0.500 → 0.500,
`8-9-10` 0.375 → 0.367).

**Measured end to end `[M]`** (`normconv-r01`, policy `lowres_normconv.json` → plan
`stage-fp16-448x256-normconv-r01`; RTX 4070 SUPER, 250 frames, seed 50, reuse on):

| | FPS | VAE decode s / 9 windows | nvidia-smi peak MiB | torch peak reserved MiB | engine arena MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lowres-scaled` (shipped engines) | 32.98 | 4.583 | 8,769 | 8,060 | 883 |
| `normconv-r01` | **33.15** | **4.532** | **8,253** | **7,538** | **301** |

**+0.5% FPS, −522 MiB VRAM.** The speed gain is far below what the per-layer profile
promised, and an interleaved A/B of the engine files on the real calibration samples
(no profiler, CUDA events, 40 reps × 3 rounds) says why: tail −0.20 ms/call (−0.7%),
mid −0.44 (−2.1%), low −0.15 (−2.2%), **−5.5 ms per window** in total. Two lessons:
(1) the per-layer profiler numbers and the probe's wall times both carry ±5% build-to-build
tactic variance (the *same* graph rebuilt ranged 26.2–29.2 ms/call), so any single build
comparison under ~5% is noise unless the engine files themselves are A/B'd; (2) the
pointwise (`kgen`) time grew by exactly what the Reduce and part of the reformat lost —
the work moved, it did not vanish. What did vanish is the Reduce layers' scratch memory:
the shared engine arena dropped from 883 to 301 MiB, which is the VRAM win. Quality is
neutral (`review-normconv/`: opening correlation 0.968, oral edge ratio 1.049, mouth-centre
distance 1.6 px against `lowres-scaled`; colour drift max 0.216/255 vs 0.181; window 0
differs by 0.116/255 mean, later windows by the usual compile-family 3–4/255).
Videos: `review-normconv/comparison.mp4`, `review-normconv/mouth-comparison.mp4`.
Kept as the 448×256 build (VRAM), not counted as a throughput step.

What is left inside the engines after the rewrite (tail: 4.5 ms reformat + 1.2 ms slice
of 26.1 in the profiler's accounting, ~2 ms in wall terms): six "cache_in copy" nodes (the concat of the 2-frame cache with the 4 new
frames), six full-tensor reformats feeding the cache-out `Slice` (TRT reformats the
4-frame tensor before slicing 2 frames of it), the network input and output. Those are the
next candidates, but they need TensorRT to slice before it reformats or a different cache
contract, so they are parked behind the end-to-end measurement of this step.

## 2026-09-22 — goal restated: 30 FPS on the 576×320 build

The user's standing goal is now **at least 30 FPS on the high-resolution (576×320) build**, with
quality risk explicitly accepted as long as every change is validated on video before it is
kept. Reference for every 576×320 measurement in this section: `benchmarks/pro_30fps_20260921/s30-reuse`
(19.62 FPS, `raw.npy` saved; RTX 4070 SUPER, 12,282 MiB visible, seed 50, `indian150-a`, 250 frames).

**Budget.** 30 FPS = 933 ms per 28-frame window. The reference window is 1,427 ms:

| component (576×320, per window) | ms | share |
| --- | ---: | ---: |
| VAE decode — three FP16 TensorRT stage engines | ≈ 630 | 44% |
| VAE decode — non-engine ops, boundary casts | ≈ 224 | 16% |
| DiT (2 steps, 30 blocks) | 394 | 28% |
| motion encode (5 frames) | 124 | 9% |
| audio + host + delivery | ≈ 55 | 4% |

−494 ms is needed. Every kernel-level lever is exhausted (iterations 1–5), so this section tries
**structural** ones, each behind a flag with a one-line rollback, each gated by `review.py` against
the reference plus colour drift plus a labelled video:

1. drop decoder residual blocks (the tail engine `upsamples[12-14]` alone is ~340 ms; every block
   is a `y = x + f(x)` residual, so bypassing one is well-defined and the stage engines can be
   rebuilt for any prefix of a group from the existing calibration samples);
2. drop DiT blocks (whole run, or only at the refinement step);
3. latent feedback: reuse the DiT's trailing latents as the next window's motion history instead
   of decoding, colour-correcting and re-encoding five frames (−124 ms if the model tolerates it);
4. CUDA graphs for the launch-bound low-resolution decoder segments.

Implementation, adversarial review and the two analyses ran as a multi-agent workflow
(`pro-30fps-576-levers`); the GPU experiments below are serial and single-tenant.

### Iteration 6 — structural levers built and reviewed (workflow `pro-30fps-576-levers`)

Ten agents: three implementers on disjoint files, an adversarial reviewer per implementation,
a fix pass where the reviewer found something, and two analyses. Everything is behind a flag
that defaults off; flag absent is the byte-identical shipped path.

**Decoder residual-block skipping** (`soulx_rtc/pro_decoder_ops.py`,
`decoder_stages.py`, `pro_vae_stage_backend.py`). `install_decoder_block_skip(vae, indices)`
replaces `upsamples[i]` by `SkippedResidualBlock`, an identity that advances the causal-cache
cursor by the two slots the block's two `CausalConv3d` would have consumed and never writes
them, so every downstream slot — including the head's — stays aligned under the overlap-skip
shim and `torch.compile`. Only same-width `ResidualBlock`s may be skipped (`upsamples[4]`
changes 192→384 and is refused). The bypassed block stays in the module tree as a child
because `clear_cache` sizes `_feat_map` by counting `CausalConv3d` modules after the install
— a bare identity would shrink the map and the head would `IndexError`; the install runs a
4×4 self-check that replays the cursor walk. Engines: `decoder_stages.py build --use-blocks
"4,5,6" "8,9,10" "12"` builds a **prefix** of each calibration group from the existing
samples (a prefix of k blocks needs `(x, *caches[:2k])`, in block order), records
`skipped_blocks`, and `install_stage_plan` refuses a plan whose `skipped_blocks` differ from
what the VAE actually has skipped. Non-prefix patterns need `capture --skip-blocks` first.
Reviewer found a blocker (scales looked up by the prefix key after rebinding → `KeyError`
on every prefix build), fixed; plus stale-calibration bookkeeping for groups downstream of a
skipped block (accepted for fp16, refused for INT8).

**DiT block skipping** (`soulx_rtc/pro_dit_ops.py`). Drop mode re-lists `model.blocks`
after quantization/static scales/attention backend (all keyed by original names) and
before prepared conditioning + compile (which iterate survivors). Step-aware mode wraps the
listed blocks to return the residual stream unchanged when `CURRENT_STEP[0]` equals the
chosen step — one Dynamo guard, two specialisations. Reviewer: drop mode would silently
corrupt an INT8 calibration run (amax collection keyed by block names) → now refused;
`--dit-skip-step` beyond the schedule → detectable via the manifest. `suggest_skip_sets()`
gives the ladder (evenly spaced middle blocks; the blocks before the last; step-1-only
variants; a step-0 control), every set keeping blocks 0 and 29.

**Latent feedback** (`flash_head_pipeline.py`). `pipeline.latent_feedback = "last2"`
replaces the 124 ms/window re-encode with the DiT's trailing two latents. Known mismatch,
documented in the code: the causal VAE's slot 0 is a single-frame keyframe latent, the DiT's
trailing latents are both 4-frame latents, and the colour-correction feedback loop
disappears. `"last2-fix0"` re-encodes only the single frame for slot 0 (~30 ms instead of
124; the 1×1×1 `conv1` has no temporal mixing so slot 0 equals the shipped one up to compile
numerics). Reviewer: sound.

**Analysis: CUDA graphs — dead, with the traces to prove it.** Inside decode the host runs
43–151 ms *ahead* of the GPU and the non-engine segments are 98.5–99.4% GPU-busy (576×320:
the per-frame head segment spans 15.83 ms with 15.72 ms of kernels). The ~1,500 small
launches per window are GPU time, not launch overhead. Inductor cudagraph trees also break on
the overlap-skip shim (a persisted output fed back as input: "accessing tensor output of
CUDAGraphs that has been overwritten", reproduced). Ceiling ≈ 1 ms/window. Not built. The
by-product is a non-engine breakdown per 576×320 frame: head segment 15.8 ms (12 low-res
convs at ~25% of bf16 peak with 36 NCHW↔NHWC transposes), `Resample[7]` 8.1 ms
(sub-pixel-foldable like `[11]`, ≈ −15 ms/window), `Resample[11]` 6.9 ms, and an apparent
graph break inside `decoder.head` leaving SiLU and copies eager.

**Analysis: missed levers.** Ranked, with code pointers: (C1) merge `Resample[7]`,
`Resample[11]` and the head into the `[8-10]`/`[12-14]` engines (≈ −50±25 ms; parked
earlier on VRAM, re-opened by the norm-conv arena shrink); (C2) INT8 FFN GEMM with a fused
dequant→GELU→requant epilogue (≈ −55±30 ms; the 232 MB INT32 `_int_mm` output is written
and read back today); (C3) `--lean-delivery --force-scheduler` (zero code, ≈ −25–35 ms of
window-boundary host serialisation); (C5) INT8 W8A8 on the self-attention projections; (C6)
2:4 sparsity probe; (C7) a TAEHV-class drop-in decoder if block dropping fails the video
gate. Nothing exact reaches 933 ms on its own.

**Ladder** (serial, single-tenant GPU, reference `s30-reuse` 19.62 FPS; every arm gated by
`review.py` + colour drift + labelled video): control on the norm-conv 576 plan; decoder
skips A `[12]` (skip 13,14), D `[8]+[12]` (skip 9,10,13,14), B `[12,13]`, C `[8,9]+[12]`;
DiT mid6, tail6, mid9, mid6-step1; latent feedback last2; lean delivery through the
scheduler. Results follow.

**Ladder, first pass (2026-09-22 10:55–11:02).** Engine builds: norm-conv control plan and
plans A (`[12]`) and C (`[8,9]+[12]`) built (arena 484 MiB for all three — the norm-conv
rewrite alone takes the 576×320 arena from 1,488 to 484 MiB); plans B and D failed with a
TensorRT `OutOfMemory` during tactic timing — a build process peaks at ~9.3 GB (calibration
samples, eager stage comparison at full resolution, 2 GB builder workspace) and a second GPU
tenant appeared at the same moment. Control run on the norm-conv plan: **19.71 FPS**
(reference 19.62; DiT 3.555 s, encode 1.110, decode 7.610 over 9 windows; torch reserved
10,420 MiB vs 11,160). Every following arm then failed in two seconds with "Another SoulX GPU
owner from this checkout is running": the `.gpu-owner.lock` flock had been taken at 11:01:22 —
the instant the control run released it — by another agent's job on this machine
(`experiments/ltx23_musetalk_lip_override_20260922/scripts/run_ltx.py`, launched from the
VS Code Codex extension, which starts a ComfyUI server with `--reserve-vram 3` and holds the
lease for up to an hour per phase). Not mine to stop. The ladder was rewritten to wait for a
free lease **and** a quiet GPU (`wait_lease.sh`: `flock -n` probe, <1.5 GB used, <10% util)
before every build and run, to retry once on a lease race, and to rebuild plans B and D
first. Measurements below carry the resident-process record from each run's
`environment_after` as evidence of a clean GPU.

**Ladder, measured (2026-09-22 11:07–11:19, lease-aware, GPU clean per `environment_after`).**
Control on the norm-conv plan: 19.71 FPS. Reference for quality: `s30-reuse` (19.62 FPS).
Gate columns: `review.py` opening correlation (lip motion vs reference; accepted builds score
0.94–0.97), oral edge ratio (mouth edge energy vs reference; accepted builds 0.95–1.07; the
rejected 1-step arm scored 1.32), colour-drift max over 9 windows (reference itself 1.48/255).

| arm | FPS | Δ ms/window | corr | edge ratio | drift max | verdict |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| decoder skip 14 (`[12,13]`) | 21.48 | −117 | 0.962 | **0.32** | 1.28 | lip-sync intact, mouth detail lost |
| decoder skip 13,14 (`[12]`) | 23.39 | −220 | 0.967 | **0.25** | 0.40 | same, worse |
| decoder skip 10,13,14 | 25.43 | −322 | 0.920 | **0.22** | 0.24 | same, worse |
| decoder skip 9,10,13,14 | **27.38** | −398 | 0.902 | **0.20** | 0.29 | same, worst |
| DiT drop 6 middle blocks | 20.50 | −80 | **0.58** | 0.67 | 0.67 | lip-sync broken |
| DiT drop 6 late blocks | 20.92 | −78 | **0.44** | 0.83 | 0.45 | broken |
| DiT drop 9 middle blocks | 21.54 | −119 | — | — | 0.75 | review found too few mouths; broken |
| DiT 6 middle blocks, step 1 only | 20.28 | −40 | 0.89 | **1.27** | 1.54 | over-sharpened, like the rejected 1-step |
| latent feedback `last2` | 21.57 | −124 | 0.90 | **2.65** | **2.56** | artefacts + colour drift |
| latent feedback `last2-fix0` | 20.67 | −90 | 0.941 | 0.93 | **2.77** (slope R +0.34/window) | motion fine, colour drifts without the pixel loop |
| `--lean-delivery --force-scheduler` | 19.51 | +14 | — | — | — | slower at 576×320; dropped |

Decoder-only arms share the DiT with the control, so window 0 (before the motion feedback
diverges the trajectories) isolates the decoder's own loss: see the fidelity table that
follows. Reading: **every structural lever buys its predicted time and fails the quality gate
as-is.** DiT block dropping is dead (the 2-step distilled sampler has no slack; lip sync
collapses). Latent feedback needs the colour-correction loop, which the pixel re-encode
provides and the latent path cannot; `last2-fix0` fixes the keyframe semantics but not the
drift. Decoder block skipping is the one lever whose failure mode is *soft*, not *wrong*:
lip motion tracks the reference (0.90–0.97), colour is stable, and what is lost is the
high-frequency detail the removed full-resolution blocks produce. That is exactly what a
decoder-only fine-tune (distillation from the full decoder on the same latents) can put back,
and it is the next step: fine-tune the surviving blocks of the pruned decoder to reproduce the
full decoder's frames, then recalibrate and rebuild its engines from the fine-tuned weights.
Videos for the user's own judgement: `benchmarks/pro_30fps_20260922/visual/skip-tail12-r01-*`,
`skip-mid8-tail12-r01-*`, `latfb-fix0-r01-*` (reference on the left/top).

**Exact win landed in parallel (lever C2):** `soulx_rtc/pro_int8_gemm.py` — a Triton INT8
GEMM for `ffn.0` whose epilogue dequantises, adds bias, applies the tanh-GELU and requantises
to int8 with `ffn.2`'s static scale, so the 232 MB INT32 intermediate is never written.
Bit-identical to the shipped compiled path (0 differing elements in 12 cases, `torch.equal`
end to end), 0.98 ms vs 1.55 ms for GEMM + requant at (6480,1536,8960), −29 ms/window at
576×320 measured at the FFN level. Wired as `--fused-int8-ffn`; end-to-end measurement below.

**Fused INT8 FFN, end to end (`fused-r01`):** DiT CUDA-event time 3.555 → 3.308 s over 9
windows (−27 ms/window, as predicted) but **19.66 FPS vs 19.71** — the window wall did not
move. Wall − stage spans (GPU idle) rose from 37 to 65 ms/window: at 576×320 the DiT phase
is host-issue-bound, so a pure GPU-time saving turns into idle and only a lever that also
removes host work (dropping blocks did both) shows up in FPS. The kernel is kept (exact, and
it will count once the host side is graph-captured: a DiT CUDA graph is now the obvious
follow-up, with ≈ 40–65 ms/window of measured GPU idle to recover — the cudagraph analysis
above ruled it out for the *decoder*, where the host runs far ahead, not for the DiT).

### Iteration 7 — distilling the pruned decoder

Data: `SOULX_DUMP_DIR` runs of the shipped 576×320 decoder **without** overlap-skip on all
seven fixtures (seed 50) plus `indian150-a` at seed 51 → 88 windows of (9 latents → 33
frames), the frames from the FP16 stage-engine decoder, i.e. the accepted teacher
(`benchmarks/pro_30fps_20260922/distill-data/`; the run filled the 204 GB shared disk to 100%
once — 35 `raw.npy` files of rejected arms were deleted). Script:
`benchmarks/pro_30fps_20260922/distill_decoder.py`. Student = stock VAE with
`upsamples[9,10,13,14]` bypassed; trainable = `upsamples[7]` (Resample feeding the pruned mid
group), `[8]`, `[11]` (Resample feeding the tail), `[12]`, `head` — 4.21 M parameters in fp32
under bf16 autocast, everything upstream frozen; forward mirrors `WanVAE_.decode` with
per-latent-frame truncated backprop through the shared causal cache; loss = mouth-weighted
(×3 on the review's mouth crop) L1 + L1 on spatial gradients; AdamW 5e-5 → 5e-6 cosine, 3
epochs over 77 training windows, `tts-conversational` (11 windows) held out.

Hold-out before training (pruned, untrained, vs teacher): PSNR 28.55 dB, mouth PSNR 28.77,
sharpness ratio 0.790, mouth sharpness ratio 0.785. Results follow.

**Lever C1 landed in code (merged span engines), measurement pending.** `soulx_rtc/pro_vae_stage_backend.py`
gained `DecoderSpanStage`: a TensorRT stage may now cover a contiguous slice of
`decoder.upsamples` that includes `Resample` layers and, as the last token of the last group,
the decoder `head` (`--groups "4,5,6" "7,8,9,10" "11,12,13,14,head"`). Cache slots are read from
the module classes (ResidualBlock 2, `upsample3d` Resample 1 for its `time_conv`, `upsample2d`
0, head 1); the Resample's first-call `"Rep"` sentinel is materialised as a zero 2-frame cache,
which is bit-identical to the stock path (CPU and CUDA, 3 frames, every cache slot compared);
`decoder.head` becomes a pass-through so the engine's `y` is the decoder output;
`install_stage_plan` validates the recorded span module classes against the live decoder and
refuses a mismatch (e.g. a plan captured with the sub-pixel fold installed on a stock decoder).
`Upsample(nearest-exact)` has no ONNX symbolic in torch 2.7, so the span exports
`Resize(nearest, floor)`, which equals nearest-exact for integer scale 2 (asserted at
construction, checked with the pure-numpy ONNX reference). The whole thing is behind the plan
format: an old plan installs exactly as before. Constraint for the harness: a span plan must be
run **without** `--subpixel-resample --channels-last-head` (those modules now live inside the
engines). Expected −50±25 ms/window at 576×320 from the eager Resample/head segments;
arena estimate 620–800 MiB (stock Resample form). Also from this agent: `--vae-weights` on
`capture`/`build` (the plan records the weights' sha256 and `install_stage_plan(vae, path,
vae_weights=…)` refuses a mismatch), which the harness now threads through.

**Distillation result (`distill/vae_skip9-10-13-14_ft.{pth,json}`, 3 epochs × 77 windows,
~3 s per window on the RTX 4070 SUPER, 12 min wall):**

| hold-out `tts-conversational`, 11 windows vs the full FP16 decoder | PSNR | mouth PSNR | sharpness ratio | mouth sharpness ratio |
| --- | ---: | ---: | ---: | ---: |
| pruned, untrained | 28.55 dB | 28.77 | 0.790 | 0.785 |
| after epoch 1 | 45.42 | 41.20 | 0.901 | 0.903 |
| after epoch 2 | 46.64 | 42.13 | 0.915 | 0.914 |
| **after epoch 3** | **47.18** | **42.61** | **0.924** | **0.924** |

Reading: 47 dB is an RMS error of ~1.1/255 against the teacher — the four removed blocks'
function is almost entirely absorbed by 4.2 M parameters in the surviving tail. The residual
8% sharpness gap is what the harness gate will judge. Next: `capture --skip-blocks 9 10 13 14
--vae-weights …` → `build --graph-rewrite norm-conv --vae-weights …` → harness with
`--vae-weights … --skip-decoder-blocks 9 10 13 14` → `review.py` + colour drift + video.

**Fine-tuned pruned decoder in the real pipeline (`ft-mid8-tail12-r01`; engines rebuilt
from the fine-tuned weights with `capture --skip-blocks 9 10 13 14 --vae-weights …` →
`build --graph-rewrite norm-conv --vae-weights …`, harness `--vae-weights … --skip-decoder-blocks
9 10 13 14`):** **27.78 FPS** (VAE decode 7.61 → 3.88 s over 9 windows; VRAM 8,825 MiB vs
10,895 for the control). Gate vs the 19.62 FPS reference: opening correlation **0.954**,
mouth-centre distance 3.1 px, colour drift max 1.37/255 (reference itself 1.48), oral edge
ratio **0.638** (untrained pruned: 0.198; accepted builds 0.95–1.07). Window-0 fidelity vs the
control (decoder-only): PSNR 33.2 → **43.9 dB**, mouth PSNR 32.6 → 39.6, mouth sharpness 0.74
→ 0.92. Verdict: lip sync and colour are those of the reference; the mouth is still softer
than the full decoder (edge energy at 64%). Kept as the working build; a second, longer,
edge-weighted fine-tune (6 epochs from these weights, gradient-loss weight 3, mouth weight 4)
is running to close the remaining gap. Video: `visual/ft-mid8-tail12-r01-*-vs-reference.mp4`.

**DiT CUDA graph, measured (`graph-r01`, `graph-fused-r01`, full decoder):**

| arm | FPS | window p50 | DiT s | VAE decode s | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| control (norm-conv plan) | 19.71 | 1.378 s | 3.555 | 7.610 | 10,895 |
| `--fused-int8-ffn` | 19.66 | 1.374 | 3.308 | 7.637 | 10,897 |
| `--dit-cuda-graph` | **21.13** | 1.289 | 3.381 | 7.015 | 11,485 |
| `--dit-cuda-graph --fused-int8-ffn` | **21.22** | 1.274 | 3.073 | 7.013 | 11,303 |

−89 ms/window from the graph alone (+7.2%), more than the 37–65 ms of DiT-phase idle: the
*decode* event span also fell by 66 ms/window, i.e. the host was pacing the decode too, and
with the DiT issued in one launch it runs far enough ahead to feed the decoder back to back.
The fused FFN now shows up (+0.5% on top). Both are exact (bitwise-identical FFN; the graph
replays the same kernels). Cost: ~600 MiB for the graph pools at 576×320.

**Race incident, recorded because it cost a run:** two of my own queued chains woke on the
same free-lease poll; the loser hit "Another SoulX GPU owner" and its whole chain skipped
forward. Chains are now strictly sequential in one queue.

**Fine-tune v2** (6 more epochs from v1, gradient-loss weight 3, mouth weight 4, lr 3e-5 →
3e-6; `distill/vae_skip9-10-13-14_ft2.{pth,json}`): hold-out PSNR 47.18 → **48.68 dB**, mouth
42.61 → **44.12**, sharpness 0.924 → **0.946**. In the pipeline (`ft-mid8-tail12-v2-r01`):
27.75 FPS, correlation **0.958**, oral edge ratio **0.796** (v1 0.638), mouth 3.4 px, drift
max 1.22/255.

**Combined candidate (`combo-v2-r01`)** = fine-tune v2 pruned decoder + `--dit-cuda-graph` +
`--fused-int8-ffn`: **28.69 FPS**, window p50 **937 ms** (target 933), DiT 3.085 s, decode
3.911 s, motion encode 1.049 s over 9 windows, VRAM 9,221 MiB. Gate: correlation **0.958**,
oral edge ratio **0.838**, mouth-centre distance **2.6 px**, colour drift max 1.25/255
(reference 1.48). The graph is worth −31 ms/window here (vs −89 on the full decoder: the
pruned decode is no longer host-paced). Video: `visual/combo-v2-r01-*-vs-reference.mp4`.
Remaining to 30 FPS: 4 ms/window; the merged Resample engines (`"7,8"` and `"11,12"` spans
around the skipped blocks) are next in the queue.

**Merged span engines, measured.** Full decoder, stock weights, groups `"4,5,6" "7,8,9,10"
"11,12,13,14,head"` (`spans-r01`, arena 675 MiB): **21.16 FPS** vs 19.71 control, decode 7.61 →
6.95 s over 9 windows (−73 ms/window) — inside the analysis estimate of −50±25. On the pruned
decoder the head cannot join the tail span (the group must end at index 14 and may not contain
skipped blocks), so the final chain used `"4,5,6" "7,8" "11,12"` with `--skip-blocks 9 10 13 14`
(arena 551 MiB): −21 ms/window.

### Final candidate so far — `final-v2-r01`: 29.63 FPS at 576×320

Fine-tune v2 pruned decoder (blocks 9, 10, 13, 14 bypassed; `upsamples[7,8,11,12]` + head
distilled), span engines `"4,5,6" "7,8" "11,12"` rebuilt from the fine-tuned weights with the
norm-conv rewrite, DiT CUDA graph, fused INT8 FFN; everything else as the 19.62 FPS reference.

| | reference | final-v2-r01 |
| --- | ---: | ---: |
| useful FPS, 250 frames | 19.62 | **29.63** |
| window p50 | 1,427 ms | **916 ms** (steady state 30.57 FPS; window 0 is 994 ms) |
| DiT / decode / encode s over 9 windows | 3.54 / 7.68 / 1.12 | 3.08 / 3.93 / 1.05 |
| nvidia-smi peak | 11,160 MiB | **9,039 MiB** |
| opening correlation | — | **0.956** |
| oral edge ratio | — | **0.818** |
| mouth-centre distance | — | 3.0 px |
| colour drift max (ref 1.48/255) | — | 1.25/255 |

Video: `benchmarks/pro_30fps_20260922/visual/final-v2-r01-vs-reference.mp4` and
`final-v2-r01-mouth-vs-reference.mp4`. The 250-frame average sits 0.4 FPS under the target
because window 0 pays the overlap-skip's cold full decode; steady state is over 30. Remaining
exact levers in flight: `--lean-delivery` (queued), the head inside the tail span despite the
skipped blocks (agent), a TensorRT engine for the motion encoder (117 ms/window, agent).

## Superseded result — 30.04 FPS (`final-v2-lean-r01`, 2026-09-22 19:11); see the final result below

**Hardware:** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM (physical class
unverified in this run), driver 595.84, Torch 2.7.1+cu128, TensorRT 10.9.0.34, SageAttention
2.2.0 stream-patched build (`sage-sm89-stream`). **Fresh local GPU inference**, seed 50,
`indian150-a`, 250 frames, 2 sampling steps, single tenant on the GPU for the whole run
(`environment_after.resident_gpu_processes` = this process only).

| | reference `s30-reuse` | `final-v2-r01` | `final-v2-lean-r01` | `final-v2-r02` (repeat) |
| --- | ---: | ---: | ---: | ---: |
| useful FPS (250 frames, includes window 0) | 19.62 | 29.63 | **30.04** | 29.68 |
| window p50 | 1,427 ms | 916 | **916** | 915 |
| DiT / decode / encode over 9 windows | 3.54 / 7.68 / 1.12 s | 3.08 / 3.93 / 1.05 | 3.08 / 3.92 / 1.05 | 3.08 / 3.92 / 1.05 |
| nvidia-smi peak / torch reserved | 11,160 / 11,160 MiB | 9,039 / 8,550 | **9,039 / 8,550** | 9,039 / 8,550 |
| raw frames | — | sha A | **identical to sha A** (bitwise) | — |

Quality gate against the reference (`review.py`, colour drift over 9 windows):

| | accepted-build range | `final-v2-lean-r01` |
| --- | --- | ---: |
| opening correlation (lip motion) | 0.94–0.97 | **0.957** |
| mouth-centre distance | ≤ 3.4 px | 2.9 px |
| oral edge ratio (mouth edge energy) | 0.95–1.07 | **0.818** |
| colour drift max / slope R | ref 1.48 / +0.106 per window | 1.15 / +0.085 |

Reading: lip sync, timing and colour are those of the reference; the mouth carries ~18% less
edge energy than the full decoder (the fine-tune recovered it from 20% to 82% of the
reference; a third fine-tune or skipping one block fewer would trade FPS back for it). Judge on
the labelled videos, which are frame-identical to the lean run:
`benchmarks/pro_30fps_20260922/visual/final-v2-r01-vs-reference.mp4` (full frame, reference
left) and `final-v2-r01-mouth-vs-reference.mp4` (mouth crop, reference top).

**What the build is (every item behind a flag; omit it to roll back):**

1. Decoder residual blocks `upsamples[9,10,13,14]` bypassed (`--skip-decoder-blocks 9 10 13 14`,
   `SkippedResidualBlock` keeps the causal-cache cursor aligned) — the only structural lever whose
   failure mode was softness rather than wrong motion.
2. The surviving `upsamples[7,8,11,12]` and the head **distilled** from the full FP16 decoder on
   88 windows of the pipeline's own latents (`distill_decoder.py`, two rounds: 3 epochs, then 6
   edge-weighted epochs; hold-out PSNR 28.6 → 48.7 dB, sharpness 0.79 → 0.95) →
   `distill/vae_skip9-10-13-14_ft2.pth` (`--vae-weights`, loaded before any decoder rewrite; the
   stage plan records the weights' sha256 and refuses a mismatch).
3. TensorRT FP16 span engines `"4,5,6" "7,8" "11,12"` built from those weights with the
   norm-conv ONNX rewrite (`decoder_stages.py capture --skip-blocks … --vae-weights …` →
   `build --graph-rewrite norm-conv --vae-weights …`; policy `policies/final_v2.json`); the span
   stages absorb `Resample[7]` and `Resample[11]`, so the harness runs **without**
   `--subpixel-resample --channels-last-head`.
4. `--dit-cuda-graph`: the two-step DiT forward captured once per step into CUDA graphs
   (possible only because SageAttention now launches on the current stream).
5. `--fused-int8-ffn`: Triton INT8 GEMM with fused dequant → GELU → requant epilogue for
   `ffn.0` (bit-identical to the compiled path).
6. `--lean-delivery`: device-side uint8 trim, one quarter of the bytes over the bus (bit-identical
   frames).
7. Unchanged from the reference: 2 distilled steps, INT8 W8A8 FFN with static `ffn.2` scales,
   FP8 attention projections, SageAttention INT8-QK/FP8-PV, prepared cross-attention
   conditioning, overlap-skip decoder cache, FP16 cache passthrough, output-buffer reuse.

Exact command (from `hires_span.sh`):
```
POLICY=benchmarks/pro_30fps_20260920/policies/final_v2.json  # decoder.plan -> stage-fp16-576x320-final-v2-r01
.venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py --policy $POLICY \
  --fixtures benchmarks/pro_30fps_20260920/tts-fixtures/fixtures.json --fixture-id indian150-a \
  --seed 50 --frames 250 --repeats 1 --sampling-steps 2 --timestep-variant distilled_aligned \
  --static-int8-scales benchmarks/pro_30fps_20260921/int8-amax.json --overlap-skip \
  --compile-vae-encode --vae-weights benchmarks/pro_30fps_20260922/distill/vae_skip9-10-13-14_ft2.pth \
  --skip-decoder-blocks 9 10 13 14 --dit-cuda-graph --fused-int8-ffn --lean-delivery --save-raw \
  --output benchmarks/pro_30fps_20260922/final-v2-lean-r01
```
(with `PYTHONPATH=/workspace/experiments/pro30-deps/sage-sm89-stream:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.`,
`SOULX_STAGE_REUSE_OUTPUTS=1`, `PYTORCH_CUDA_ALLOC_CONF` unset.)

Still in flight for headroom (to buy quality back at ≥ 30 FPS): the decoder head inside the tail
span despite the skipped blocks, and a TensorRT engine for the motion encoder (117 ms/window).

### Iteration 8 — the head inside the tail engine: faster, softer

`soulx_rtc/pro_vae_stage_backend.py` now accepts `"11,12,head"` as a group when every
`upsamples` index after the group is a `SkippedResidualBlock`. The skipped identities stay in
the module list and keep advancing the cursor; the span maps its own three caches onto slots
25, 26 and 31 through a new `slot_offsets` list, so slots 27–30 stay `None` and the total walk
still ends at 32 — bit-identical to the pruned eager decoder on CPU and on the GPU (3 frames,
all 33 slot states, plus the real `WanVAE.decode` entry point). `install_stage_plan` refuses a
plan whose recorded span says `after-skipped:13,14` when the live decoder has those blocks, and
vice versa.

Measured (`final-headspan-r01`, same weights and flags as the final candidate, no lean
delivery): **30.39 FPS**, window p50 **893 ms**, decode 3.93 → 3.72 s over 9 windows
(−23 ms/window), arena unchanged at 551 MiB, VRAM 8,975 MiB. Gate: opening correlation 0.950,
mouth-centre distance 4.0 px, colour drift max **0.884**/255 (the best of any arm), but oral
edge ratio **0.736** against 0.818 with the head eager.

**Conclusion: not shipped.** The head is precision-sensitive — moving its RMS-norm + SiLU +
3-channel convolution from eager bf16 into the FP16 engine costs ~0.08 of oral edge ratio for
23 ms/window. Since the goal is 30 FPS *with acceptable quality* and the head-eager build
already clears 30, the 0.35 FPS is not worth the mouth detail. Kept as a documented
alternative (`policies/final_headspan.json`, plan `stage-fp16-576x320-headspan-r01`) and as the
mechanism to use if a future build needs the time back. A distillation round that trains the
head *through* the FP16 engine would likely recover it; not attempted.

With lean delivery (`final-headspan-lean-r01`): **30.87 FPS**, window 892 ms, VRAM 8,939 MiB,
correlation 0.951, edge ratio 0.759, drift 0.884/255. So the full trade at 576×320 is
**+0.83 FPS for −0.06 oral edge ratio**; the head-eager build ships.

**Blocked:** the TensorRT motion-encoder engine (lever C4, 117 ms/window) was being built by an
agent that stopped on an account spend limit for its model. The encoder remains eager/compiled.

**Regression check.** `pytest tests/` on this working tree: 294 passed, 16 failed, 1 skipped.
The same 16 fail with every change of this programme stashed (verified by `git stash` and a
re-run), so they are pre-existing in this checkout — missing gitignored fixtures
(`examples/girl.png`), an RTC server binding, and policy/preflight fixtures. No test regressed.

### Iteration 9 — the sharpness gap was capacity, not optimisation

Rounds 1–3 of the distillation trained only `upsamples[7,8,11,12]` + head — 4.21 M parameters
— and plateaued: round 2 ended at hold-out sharpness 0.946, and round 3 (8 epochs, gradient
weight 6) moved its first epoch by **+0.002**, so it was stopped. Round 4 added the
384-channel group `upsamples[4,5,6]` that feeds the pruned levels, **26.19 M** trainable
parameters, 8 epochs from the round-2 weights, gradient weight 4, mouth weight 4, lr 3e-5 →
3e-6 (`distill/vae_skip9-10-13-14_ft4.{pth,json}`, 46 min):

| hold-out `tts-conversational` vs the full FP16 decoder | PSNR | mouth PSNR | sharpness | mouth sharpness |
| --- | ---: | ---: | ---: | ---: |
| pruned, untrained | 28.55 dB | 28.77 | 0.790 | 0.785 |
| round 1 (3 ep, 4.2 M) | 47.18 | 42.61 | 0.924 | 0.924 |
| round 2 (6 ep, 4.2 M) | 48.68 | 44.12 | 0.945 | 0.946 |
| round 3 (4.2 M, edge-weighted) | *stopped, +0.002 in epoch 1* | | | |
| **round 4 (8 ep, 26.2 M)** | **49.66** | **45.04** | **0.956** | **0.955** |

Both terms improved together once the capacity was there: the first epoch traded fidelity for
edges (46.53 dB / 0.950), and by epoch 8 fidelity had passed every earlier round as well.

## Final result — 30.22 FPS at 576×320 (`final-v4-lean-r01`, 2026-09-22 23:03)

**Hardware:** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM (physical class
unverified in this run), driver 595.84, Torch 2.7.1+cu128, TensorRT 10.9.0.34, SageAttention
2.2.0 stream-patched build. **Fresh local GPU inference**, seed 50, `indian150-a`, 250 frames,
2 sampling steps, single tenant on the GPU.

| | reference `s30-reuse` | **`final-v4-lean-r01`** |
| --- | ---: | ---: |
| useful FPS, 250 frames (includes the cold window 0) | 19.62 | **30.22** |
| steady-state FPS (windows 1–8) | — | 30.74 |
| window p50 | 1,427 ms | **912 ms** |
| DiT / decode / motion encode over 9 windows | 3.54 / 7.68 / 1.12 s | 3.09 / 3.91 / 1.05 s |
| nvidia-smi peak VRAM | 11,160 MiB | **9,039 MiB** |

Quality gate against the reference (`review.py`; accepted builds of this programme scored
0.94–0.97 correlation and 0.95–1.07 edge ratio):

| | reference band | `final-v4-lean-r01` | previous best `final-v2-lean-r01` |
| --- | --- | ---: | ---: |
| opening correlation (lip motion) | 0.94–0.97 | **0.970** | 0.957 |
| oral edge ratio (mouth detail) | 0.95–1.07 | **0.907** | 0.818 |
| mouth-centre distance | ≤ 3.4 px | **2.2 px** | 2.9 px |
| colour drift max / slope R (ref 1.48 / +0.106) | — | 1.32 / +0.117 | 1.15 / +0.085 |
| decoder-only window 0: mouth PSNR / mouth sharpness vs control | — | **30.5 dB / 0.954** | 28.7 / 0.953 |

**+54% FPS, −19% VRAM, and lip sync better than any earlier accepted arm.** The mouth still
carries ~9% less edge energy than the full decoder (was 80% less before any distillation);
everything else is at or better than reference. Videos:
`benchmarks/pro_30fps_20260922/visual/final-v4-lean-r01-vs-reference.mp4` (full frame,
reference left) and `final-v4-lean-r01-mouth-vs-reference.mp4` (mouth crop, reference top).

**The shipping build** (each item behind a flag; omit it to roll back):
`policies/final_v4.json` (span engines `"4,5,6" "7,8" "11,12"` built from the round-4 weights
with the norm-conv rewrite) plus
`--vae-weights benchmarks/pro_30fps_20260922/distill/vae_skip9-10-13-14_ft4.pth
--skip-decoder-blocks 9 10 13 14 --dit-cuda-graph --fused-int8-ffn --lean-delivery`, and
**not** `--subpixel-resample` / `--channels-last-head`. Everything else as the reference
(2 distilled steps, INT8 W8A8 FFN with static `ffn.2` scales, FP8 attention projections,
SageAttention INT8-QK/FP8-PV, prepared cross-attention conditioning, overlap-skip decoder
cache, FP16 cache passthrough, output-buffer reuse).

**Artifacts kept** (the shared 204 GB disk ran to 100% twice during this work, so regenerable
things were deleted): the shipping weights `distill/vae_skip9-10-13-14_ft4.pth` and their
round-2 initialiser `_ft2.pth` with both JSON logs; the plan
`stage-fp16-576x320-final-v4-r01` and its calibration; `final-v4-lean-r01` /
`final-v2-lean-r01` / `ctl-r01` with `raw.npy`; every arm's `results.json`, `video.mp4` and
`review-*/`; the labelled comparison videos. **Regenerable, deleted:** the 88-window
distillation dataset (3.2 GB — `hires_dump.sh <fixture> <seed>` over the seven TTS fixtures
plus `indian150-a` seed 51 rebuilds it in ~6 minutes), the `raw.npy` of gated non-shipping
arms, superseded calibrations and engines, round 1 and the abandoned round 3 checkpoints.

Remaining headroom if more speed is ever needed: the head inside the tail engine
(`policies/final_headspan.json`, +0.65 FPS for −0.15 edge ratio on the v2 weights — worth
re-testing on v4 since its head is better trained) and the TensorRT motion encoder
(117 ms/window, not built).

## 2026-09-26 — tiny VAE (TAEHV `taew2_1`) in place of the Wan VAE: 2x, not the same quality

**Question:** can TAESD replace the VAE, as it did for MuseTalk (about 5x)? The full write-up
is [`TINY_VAE_FEASIBILITY_2026-09-26.md`](TINY_VAE_FEASIBILITY_2026-09-26.md).

**Hardware:** fresh local GPU inference on NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible
(physical class unverified in this run), driver 595.84, Torch 2.7.1+cu128, CUDA 12.8.
- Wan span engines: TensorRT 10.9.0.34, runtime-only restore in `.restored-deps-20260926`.
- Tiny-VAE engines: TensorRT 10.3.0, borrowed.
- Attention: **FlashAttention-2 in every arm, the control included.** The SageAttention-2 SM89
  build directory was deleted, and its rebuild was blocked by the permission system. The DiT
  therefore takes 456.7 ms per window instead of 343.3, and only same-session ratios are
  measured.

**Answer:**
- **TAESD itself cannot be used.** It works on SD's 4-channel 2D latents; SoulX decodes Wan 2.1
  latents (16 channels, 4x causal temporal, 9 latents per 33 frames).
- **The Wan 2.1 counterpart by the same author, `taew2_1`, fits.** It is integrated behind
  `--tiny-vae-decoder`, `--tiny-vae-encoder`, `--tiny-vae-backend` and
  `--tiny-vae-decode-mode` in `run.py` (`soulx_rtc/pro_tiny_vae.py`). The default path is
  byte-identical (raw sha `e93fb5ab…` on both the edited and the HEAD `run.py`).
- **Conventions:** taew2_1 decodes the DiT-normalised latent as is and drops the first 3 of 36
  raw frames. Its encoder end-pads with copies of the last frame. The wrong latent convention
  costs about 20 dB, which is the MuseTalk lesson again.

**Measured** (`indian150-a`, seed 50, 250 frames, same session):

| arm | useful FPS | decode / encode ms per window | vs reference: corr / edge / mouth px | vs same-session control: corr / edge / mouth px | drift /255 |
| --- | ---: | --- | --- | --- | ---: |
| control, shipping VAE (flash2) | 26.81 | 432 / 116 | 0.939 / 0.907 / 2.6 | floor 0.984 / 0.857 / 1.3 | 1.26 |
| T3: taew2_1 decoder + encoder, TensorRT | **52.90** | **37 / 6.4** | 0.920 / 1.192 / 3.1 | 0.923 / 1.408 / 4.2 | 1.59 |
| T5: shipping decoder + taew2_1 encoder | 30.02 | 432 / 6.5 | 0.937 / 0.864 / 3.2 | 0.959 / 0.935 / 2.6 | 1.61 |
| T4: lighttaew2_1 decoder + encoder | 51.57 | 47 / 6.4 | 0.952 / 1.012 / 4.4 | 0.943 / 1.132 / 5.2 | 1.51 |

**Verdicts:**
- **T3 is not shippable as is.** The tiny decoder adds teeth speckle, a lower-teeth fleck,
  grainier stubble and about 16% more skin shimmer, and its edge ratio is 1.4-1.8x the paired
  control on 3 fixtures. Its lip-sync correlation is below the control floor, and fewer
  closed-mouth frames on plosives is plausible but unproven.
- **Why not 5x:** the DiT is 83-87% of the window after the swap. The projection with sage2
  (arithmetic, not measured) is about 67 FPS for T3 and about 34 FPS for T5. A zero-cost VAE
  would be about 2.5x.
- **Closed:** lightvaew2_1 (blurry, 175 ms compiled), window-mode tiny decode (no drift
  benefit, −1 FPS), and Wan-style front padding for the tiny encoder (lip lag).

**Next:**
1. Restore sage2.
2. Gate T5 on at least 3 fixtures × 2 seeds, plus a run of 60 s or more.
3. Distil taew2_1 on SoulX latents with the Wan decoder as teacher on the fly. Storing latents
   only (about 0.8 MB per window) fits the disk. This is the route to about 2x at the same
   quality.

**Videos:** `benchmarks/pro_30fps_20260922/visual/tae-*.mp4`, including
`tae-t5-4way-mouth-grid.mp4` (reference | control | T5 | T3).
