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
