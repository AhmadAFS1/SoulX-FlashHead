# PRO stacked optimizations: INT8 FFN, cross-attention FP8, encoder compile, decoder tail engines

**Audience:** the next engineer continuing the PRO throughput work.
**Supersedes** the projections in `RTX4070_RUN_RESULTS_2026-09-20.md` §4 with measurements.
**Superseded in part (2026-09-21):** the "Where this leaves the 20 FPS goal" lever table and
the tail-engine VRAM figure have been corrected in place after adversarial review of all four
components. The definitive forward plan is **`PRO_THROUGHPUT_PLAN_2026-09-21.md`**.

## Hardware and provenance

| Field | Value | Evidence |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER (SM89, Ada) | `nvidia-smi --query-gpu=name` |
| Visible VRAM | 12,282 MiB | `nvidia-smi --query-gpu=memory.total` |
| Physical VRAM class | **unverified in this run** | not exposed by any source artifact |
| Driver / Torch / CUDA / TRT | 595.84 / 2.7.1+cu128 / 12.8 / 10.9.0.34 | runtime query |
| Commit | `51225f5` + the changes listed below | `git rev-parse` |

**Execution class: fresh local GPU inference.** 250 frames, seed 50, 2 sampling steps
(`--timestep-variant distilled_aligned`), `--repeats 1`. Every FPS number is measured.
Estimates are marked `[E]`. No historical results are reused.

## Headline

All four changes land. Measured under **identical conditions** (OmniVoice TTS stopped,
no `expandable_segments`, fixture `indian150-a`):

| Arm | FPS | vs D2 | dit | vae_decode | motion_encode | peak VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| D2 baseline | 14.8809 | — | 4.158 | 10.615 | 1.377 | 10,679 MiB |
| **stackedA** (INT8 FFN + cross q/o + encoder compile) | 15.6278 | **+5.02%** | 3.845 | 10.630 | 1.087 | 10,711 MiB |
| **stackedB** (stackedA + decoder tail engines) | **16.0275** | **+7.71%** | 3.831 | 10.241 | 1.111 | **11,849 MiB** |

Against the original 4-step baseline (11.7423 FPS four-seed mean), stackedB is **+36.5%** `[E]`.

## What was implemented

### 1. INT8 linear weight-layout fix — `soulx_rtc/pro_quantization.py`

`Int8ComputeLinear` stored `quantized.t().contiguous()`, materialising a row-major `[K, N]`
buffer. That is N-contiguous from the GEMM's point of view, so cuBLAS silently fell back to
the non-tensor-core `ampere_igemm_int8_*_nn` kernel. Storing `[N, K]` and transposing at call
time gives the K-contiguous "TN" layout that selects the tensor-core tactic.

Measured at the real FFN shapes (M=6480), **GEMM only**:

| shape | old layout | new layout | speedup |
| --- | --- | --- | --- |
| 1536 -> 8960 | 2.932 ms | 0.982 ms | **2.99x** |
| 8960 -> 1536 | 2.770 ms | 0.826 ms | **3.35x** |

**Numerics are unchanged** — verified `torch.equal` on the GEMM output for both shapes.
`torch._int_mm` is exact integer arithmetic; only the memory layout moved.

**Do not size the end-to-end win from eager microbenchmarks.** Eager and compiled invert:

| per block (ffn.0 + ffn.2) | BF16 | FP8 | INT8 |
| --- | --- | --- | --- |
| eager | 5.267 ms | 7.382 ms | 13.677 ms |
| **compiled** | 5.093 ms | 3.194 ms | **2.952 ms** |

The shipped DiT runs `compile.dit = true`, so the compiled row is the relevant one:
INT8 wins by 0.241 ms/block, or **−0.26 s end-to-end at 4 steps** `[E]`. The prior plan's
−0.85 s estimate came from idle-GPU block timing and is roughly 3x optimistic.

### 2. Cross-attention q/o FP8 — policy only

`combined_fp16_sage_fp8_crossqo.json` already existed. Folded into the stacked policies.
60 modules convert (30 blocks x q,o); **no k/v converted**, the safety invariant holds.

### 3. Motion-feedback VAE encoder compile — `soulx_rtc/pro_quantization.py`, `run.py`

`motion_encode` was 1.377 s of BF16 3D convs running **uncompiled**:
`flash_head_pipeline` sets `COMPILE_VAE = True` as a module default, the benchmark harness
disables it at `run.py:406-408`, and `optimize_wan_vae(mode="compiled")` only ever restored
`vae.decode`. Added mode `compiled_all` plus a `--compile-vae-encode` flag.

Kept as a **distinct mode** rather than folded into `"compiled"` so `decoder_trial.py`'s
compiled-BF16 *control* keeps its exact historical meaning.

The compile is applied **before** `_instrument` wraps `vae.encode`; instrumenting first would
hand `torch.compile` the CUDA-event wrapper, breaking both the timing and the graph.

Measured: **1.377 -> 1.087 s, −0.290 s**, reproducible to ±0.002 s across all seven fixtures.

### 4. Decoder tail TensorRT FP16 stage engines — `upsamples[12,13,14]`

The shipped plan covered `upsamples[4,5,6]` and `[8,9,10]`. `Decoder3d` builds four levels of
three `ResidualBlock`s plus a `Resample`, so `upsamples` is indices 0-14 and the **full
resolution level 3 (`[12,13,14]`, 96 channels at 576x320) had no engine at all**. That is
also why the historical INT8 work found nothing: it quantized the 1/4 and 1/2 resolution
groups, not the hot one.

Built via `decoder_stages.py capture --groups 4,5,6 8,9,10 12,13,14` then `build --precision fp16`.
Measured: **vae_decode 10.630 -> 10.241 s, −0.389 s, +2.56% FPS.**

**Cost: +1,138 MiB peak VRAM** (10,711 -> 11,849 of 12,282). The new shared arena is
1,418.91 MiB against the shipped 709.10 MiB. **That arena cannot be tuned down** — rebuilding
with `--workspace-mib 384` instead of 2048 produced a byte-identical 1,418.91 MiB, because the
arena is set by the engines' I/O bindings (96x4x576x320 fp16 ~= 142 MB each, seven cache
tensors), not by builder scratch.

## The methodological trap: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

**This flag costs 0.725 s on the decoder.** It was introduced to make the tail engines fit,
and it nearly produced two wrong conclusions.

Same policy, same seed, same everything, allocator flag as the only variable:

| | dit | vae_decode | motion_encode | FPS |
| --- | --- | --- | --- | --- |
| D2 baseline | 4.162 | 10.964 | 1.376 | 14.8063 |
| + `expandable_segments` only | 4.165 | **11.689** | 1.377 | 14.1667 |
| delta | +0.003 | **+0.725** | +0.001 | **−4.3%** |

It touches nothing but the decoder, and it is large enough to swamp every change in this
document. Two conclusions were briefly drawn from contaminated arms and then withdrawn:
that the tail engines caused a decoder regression (they do not — they help by 0.389 s), and
that the stacked arm was net-neutral (it is +5.02%).

**Never set this flag in a throughput arm.** `run_matrix.sh` now unsets it explicitly.

## Second measurement trap: cold compile cache

The first run of any session is slow and is **not** a valid baseline. Reproduced three times:

- `rebaseline-r01` (first run, 20:43) produced a different `raw_rgb_sha256` from every later
  run of identical configuration, and was the minimum of its five-run null set.
- `baseline__indian150-a` (first run of the sweep) came in at 14.4547 while the other six
  baseline fixtures clustered at 14.54-14.79.

The cache is `.torchinductor` (334 MB). **Discard the first run, or warm the cache first.**

A consequence worth recording: **`raw_rgb_sha256` is not a usable acceptance gate.** Two runs
of byte-identical configuration produce different hashes across a cold/warm boundary. The
Step-3 bit-exactness criterion in `RTX4070_RUNBOOK_2026-09-20.md` cannot be met by any arm,
and the earlier rejection of `--lean-delivery` on those grounds was wrong — `lean-r01` and
`quality-steps4-seed50` are in fact byte-identical to each other.

## Kokoro TTS fixtures: testing beyond one utterance

All prior PRO work used a single 10 s utterance. Six phonetically-targeted fixtures were
synthesised with **Kokoro-82M** (`kokoro` 0.9.4, repo `hexgrad/Kokoro-82M`, voice `am_michael`,
speed 1.0, 24 kHz -> 16 kHz `resample_poly`, peak-normalised 0.95, truncated to exactly 10.0 s
= 250 frames at 25 fps). **Voice is held constant so speech content is the only variable.**

Registered with SHA256 in `benchmarks/pro_30fps_20260920/tts-fixtures/fixtures.json`
alongside `indian150-a` as the historical control.

Acoustic separation is real, and in the intended directions:

| fixture | centroid Hz | ZCR | voiced % | probes |
| --- | --- | --- | --- | --- |
| indian150-a (control) | 1776.9 | 0.149 | 52.9 | — |
| plosives | 1724.9 | **0.122** | 55.0 | bilabial closure/release |
| sibilants | **2033.9** | **0.175** | 56.3 | fricative energy, teeth exposure |
| open-vowels | 1859.4 | 0.144 | **65.1** | wide jaw excursion |
| rounded | **1719.2** | 0.127 | 59.1 | lip protrusion |
| numbers-rapid | 1754.6 | 0.134 | 62.2 | dense consonant transitions |
| conversational | 1753.0 | 0.124 | 58.7 | natural prosody, questions |

Install note: `kokoro` and `misaki` were installed with `pip install --no-deps` precisely so
pip could not touch the pinned `torch 2.7.1+cu128` / SageAttention / TensorRT stack.
Verified after install: torch 2.7.1+cu128, numpy 2.2.6, `torch.cuda.is_available()` True.

## Seven-fixture sweep

baseline vs stackedA, seed 50, 2-step, all seven fixtures (OmniVoice still resident here,
so the absolute FPS is lower than the headline table; the deltas are what matter):

| fixture | base FPS | stackedA FPS | gain |
| --- | --- | --- | --- |
| indian150-a* | 14.4547 | 15.3594 | +6.26% |
| tts-plosives | 14.5379 | 15.4022 | +5.95% |
| tts-sibilants | 14.7697 | 15.3169 | +3.71% |
| tts-open-vowels | 14.7913 | 15.3401 | +3.71% |
| tts-rounded | 14.7707 | 15.4122 | +4.34% |
| tts-numbers-rapid | 14.7796 | 15.0052 | +1.53% |
| tts-conversational | 14.7859 | 15.2983 | +3.47% |

\* cold-cache baseline; that pair overstates.

**The stage deltas are the result, not the FPS spread.** They are near-identical across every
speech type, which is the finding: these optimizations are **speech-content-independent**.

| stage | mean delta | min | max |
| --- | --- | --- | --- |
| dit | **−0.331 s** | −0.346 | −0.309 |
| motion_encode | **−0.295 s** | −0.296 | −0.294 |
| vae_decode | −0.013 s | −0.100 | +0.066 |

FPS gain mean +4.14%, sd 1.60 — the sd is the known FPS noise floor (2 sd ~= 1.7%), not
fixture sensitivity.

## Quality

`review.py` on baseline vs stackedA, same step count so no profile waiver needed:

| fixture | opening correlation | median mouth centre distance | median oral edge ratio |
| --- | --- | --- | --- |
| tts-sibilants | 0.9598 | 2.34 px | **1.0030** |
| tts-plosives | 0.9514 | 4.36 px | 1.0802 |

**Edge ratio ~= 1.00 on the teeth-heavy sibilants fixture** — essentially no texture change,
against 1.235 for the D2 step reduction. 250/250 frames detected on both arms. Visual check of
`reviews/stack-tts-sibilants/mouth-comparison.png` shows near-identical teeth and lip shapes.

Pixels do differ (79-83% of them, mean |delta| 3.3-5.3/255) with per-channel means within 0.2
— the same autoregressive divergence documented previously, not degradation.

## Per-component latency breakdown (2026-09-21)

**This is the first latency summary ever retained in this repo.** Artifact:
`benchmarks/pro_30fps_20260920/latency-stackedB/latency-summary.json`
(`--latency-detail stages`, stackedB policy, seed 50, 2 steps, `complete_coverage: true`,
`dropped_events: 0`, 9 windows, 27 TensorRT engine calls per window).

The instrumented run measured 15.8737 FPS and the harness stamps it
`latency_diagnostic.performance_claim: false` — **instrumentation costs about 1%, so do not
quote 15.8737 as a throughput number.** The uninstrumented stackedB figure is 16.0275 FPS.
The *proportions* below are the result; the absolute total is diagnostic.

### Per window (`pipeline.window` = 1,728.5 ms), CUDA time

| Component | ms/window | % of pipeline |
| --- | ---: | ---: |
| **decoder `trt.execute`** (engine compute) | **698.4** | **40.4%** |
| **DiT forward** | **426.6** | **24.7%** |
| **decoder non-engine ops** (conv1, middle, upsamples[0,1,2], Resample[3,7,11], head) | **271.0** | **15.7%** |
| motion VAE encode | 123.0 | 7.1% |
| decoder `trt.output_cache_cast_bf16` | 117.6 | 6.8% |
| everything else | 43.0 | 2.5% |
| decoder `trt.input_cast_contiguous` | 41.5 | 2.4% |
| audio encoder | 7.3 | 0.4% |
| **total** | **1,728.5** | **100%** |

### `vae.decode` internals (1,128.6 ms/window)

| Part | ms/window | % of decode |
| --- | ---: | ---: |
| `trt.execute` | 698.4 | 61.9% |
| non-engine decoder ops | 271.0 | 24.0% |
| output cast to BF16 | 117.6 | 10.4% |
| input cast / contiguous | 41.5 | 3.7% |
| **boundary casts combined** | **159.1** | **14.1% of decode, 9.2% of wall** |

### Two findings that change the plan

**1. The engines are doing real work — the cast tax is not the story.** An earlier hypothesis in
this cycle held that the TensorRT boundary casts dominated the decoder, and that this explained
both the INT8 null result and the tail engines' modest gain. **That hypothesis is refuted by
measurement.** Casts are 159.1 ms/window — 14.1% of decode, 9.2% of wall. `trt.execute` is 61.9%
of decode. The surviving explanation for the INT8 result is the hardware one: on Ada, INT8 and
FP16-with-FP16-accumulate share the same tensor-core tier and the shipped FP16 engines already
run an f16-accumulate tactic, so there was never arithmetic headroom to pay the Q/DQ reformat tax.

**2. The serialized enqueue lock is a non-issue.** `trt.bind_enqueue_fence` is the *outer* scope
wrapping `trt.execute`: 698.5 vs 698.4 ms. The lock, fence and binding cost **0.08 ms/window**.
Any plan premised on contention there is chasing nothing.

### Amdahl against the measured budget

The decoder (engines + non-engine ops + casts) is **1,128.6 ms of 1,728.5 ms = 65.3%**.
Because D2 and the stacked DiT work shrank everything around it, the decoder's share has *risen*
from 52.0% at the original 4-step baseline to 65.3% today.

- With DiT, motion encode and audio all at **zero**: 250 / 10.241 = **23.6 FPS ceiling** `[E]`
- To reach 20 FPS (1,389 ms/window) from 1,733: cut the decoder **1.43x**, or cut
  everything-else **2.37x**. The decoder route is the only tractable one.

## Where this leaves the 20 FPS goal

16.0275 FPS is **80.1%** of 20 FPS. Closing the remaining 24.8% `[E]`:

**SUPERSEDED 2026-09-21** by `PRO_THROUGHPUT_PLAN_2026-09-21.md`, which is the definitive
plan. The table below is kept only so the corrections are visible; **do not plan off it.**

| lever | est. delta (as written) | **corrected 2026-09-21** |
| --- | --- | --- |
| `chunk_frames` 28 -> 36 | −0.85 s | **Not a DiT lever.** 11 latent frames x 720 = 7,920 tokens; `7 x (334.0 x 7920/6480 + 92.6 x (62/51)^2) = ~3,823 ms` per 250 frames against today's `9 x 426.6 = 3,839` — a **16 ms wash** on the DiT. The −0.85 s is decoder/encoder amortisation and belongs to the decoder components. **Do not double-count it.** Unflagged product cost: buffered audio per window 1.32 s -> **1.64 s**. |
| 1 sampling step | −2.08 s | **−1.92 s.** The −2.08 was computed off the D2 baseline DiT (4.1623/2); against stackedB's faster DiT (3.8307 s) it is −1.92 s = **−212.8 ms/window**. MEASURED. Still the single biggest lever, still ungated on quality — **no 1-step arm exists anywhere in `benchmarks/`.** |
| decoder cache persistence | −1.7 to −1.9 s | **~−150 ms/window (~−1.35 s).** The MEASURED core is **98.87 ms/window** (the sig0+sig1 engine calls), +10.15 boundary casts, +~41 non-engine. **The blocker named here was only half the story:** `cached_decode` omits the two `clear_cache()` calls in `decode`, but `WanVAE_.encode` calls `clear_cache()` at `vae.py:771` **and** `:798`, and `clear_cache` resets the **decoder** `_feat_map` at `:898`. The pipeline runs `encode` between decodes, so the encoder wipes the decoder cache every window regardless of entry point. The fix is a `_feat_map` save/restore around `flash_head_pipeline.py:404`, **not** a `decode` -> `cached_decode` swap. New quality exposure: `cond_frame` is taken **after** `match_and_blend_colors_torch`, so persisting the cache removes colour correction from the cross-window feedback loop. |
| distilled decoder tail | −2.5 to −3.5 s | **−124 to −280 ms/window (−1.1 to −2.5 s).** FLOP scaling is the right model (conv-only arithmetic intensity 864 FLOP/B against a 563 machine balance — the convolutions are **compute-bound**, contrary to the whole-engine figure). Still weeks, still trips both the weights and source gates. |

At 15.598 s (the stackedB total), reaching 20 FPS needs 12.5 s — another **3.1 s**. The
2026-09-21 plan closes ~126 ms/window with no quality exposure and no rebuild (17.3 FPS), ~276
with the overlap-skip (19.1 FPS), and **clears 20 FPS only by spending a sampling step or the
motion-history window.**

**VRAM is now the more urgent constraint.** stackedB peaks at 11,849 of 12,282 MiB — 433 MiB
of headroom. A second PRO instance does not fit, so aggregate FPS across concurrent streams is
still bounded by one instance per GPU. If concurrency is the goal, the tail engines' VRAM cost
is arguably worse than the +2.56% they buy, and stackedA is the better shipping target at
+5.02%.

**Correction 2026-09-21 — quote the tail engines' VRAM cost as two numbers, not one.** The
"+1,138 MiB" conflated allocated and reserved. From `runs[0]` of `tailtest-stackedA-control` vs
`tailtest-stackedB-tail`: `peak_allocated_mib` 4,904.84 -> 6,431.49 (**+1,526.6 MiB**) while
`peak_reserved_mib` 10,014 -> 11,162 (**+1,148 MiB**). **Reserved is the number that decides
whether a second instance fits.** Reverting the tail group is a policy-file edit only (drop
`[12,13,14]` from `policies/stacked_b_full.json`), costs a MEASURED **43.2 ms/window** (0.4 FPS),
and returns **−1,148 MiB reserved**. Note also that "`peak_reserved` 11,164 minus
`peak_allocated` 6,430 = 4,734 MiB of slack" is an artifact — in `resources-0.json` the two are
**anti-correlated** and never peak together; instantaneous slack is 1.4 GiB at the allocated
peak and 6.2 GiB at the reserved peak. **Even with every VRAM lever in the 2026-09-21 plan the
process lands near 10.2 GiB: a second instance does not fit on a 12 GB card at any point in
this plan.**

## Files changed

| File | Change |
| --- | --- |
| `soulx_rtc/pro_quantization.py` | INT8 weight layout fix; new `compiled_all` mode |
| `benchmarks/pro_quantization_v2_20260918/run.py` | `--compile-vae-encode`, applied pre-instrumentation |
| `benchmarks/pro_quantization_v2_20260918/decoder_trial.py` | qualification gate now names the failing condition with magnitudes (strictness unchanged) |
| `benchmarks/pro_quantization_v2_20260918/review.py` | `--allow-steps-drift` (earlier cycle) |

New artifacts under `benchmarks/pro_30fps_20260920/`: `policies/`, `tts-fixtures/`,
`stage-calibration-r02/`, `stage-fp16-tail-build-r01/`, `matrix/`, `reviews/`, `tailtest-*`.

**Service note:** the OmniVoice TTS server on :8002 (PID 157695) was stopped to free 2,876 MiB
for the tail-engine test and **has not been restarted**. Its command line is preserved at
`scratchpad/omnivoice_cmdline.txt`; `cwd` was `/workspace/omnivoice-triton`.
