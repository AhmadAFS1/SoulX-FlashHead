# PRO throughput: phases 1-3 executed — 16.03 -> 18.60 FPS, VRAM -2.8 GiB

**Audience:** the engineer continuing this work.
**Executes** `PRO_THROUGHPUT_PLAN_2026-09-21.md`. Where the plan and the measurement
disagree, the measurement is recorded here and the plan's figure is called out.

## Hardware and provenance

RTX 4070 SUPER, SM89 (Ada), 12,282 MiB visible (physical class **unverified**), driver
595.84, Torch 2.7.1+cu128, CUDA 12.8, TensorRT 10.9.0.34, SageAttention 2.2.0 SM89.
All arms: **fresh local GPU inference**, 250 frames, seed 50, fixture `indian150-a`,
2 sampling steps (`distilled_aligned`) unless stated, OmniVoice stopped, no
`expandable_segments`, compile cache warmed and the first run of the session discarded.

## Headline

| Arm | FPS | dit | vae_decode | motion_enc | peak VRAM | reserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stackedB (start) | 16.0275 | 3.831 | 10.241 | 1.111 | 11,849 | ~10,430 |
| + 6.2 one-pass input prep | 16.1545 | 3.828 | 10.115 | 1.112 | 11,871 | 10,398 |
| + 6.1 FP16 cache passthrough | 16.9052 | 3.845 | 9.437 | 1.112 | 10,419 | 9,710 |
| + 3.1 sub-pixel + 3.2 channels-last head | 17.0930 | 3.830 | 9.274 | 1.118 | 9,941 | 9,232 |
| + 2.1 static INT8 activation scale | 17.3542 | **3.557** | 9.311 | 1.120 | 9,717 | 9,008 |
| **+ 1.1 overlap-skip (SHIPPING)** | **18.5950** | 3.569 | **8.052** | 1.419 | **9,017** | **8,308** |
| + cudaMallocAsync (optional) | 19.2442 | 3.571 | 7.573 | 1.423 | — | 11,168 |
| + 1 sampling step | 21.6311 | 1.768 | 7.989 | 1.412 | — | 8,330 | **REJECTED** |

**Shipping result: 16.0275 -> 18.5950 FPS (+16.0%), peak VRAM 11,849 -> 9,017 MiB
(-2,832 MiB).** Both quality gates pass. Against the original 4-step baseline
(11.7423 FPS) this is **+58.4%** `[E]`.

## Quality gates (both pass)

Articulation, `review.py`, shipping arm vs the same stack without 2.1/1.1:

| metric | value | reference |
| --- | ---: | --- |
| `opening_correlation` | **0.9708** | 0.95-0.96 across prior accepted arms |
| `median_edge_ratio_on_both_open` | **1.0492** | 1.0030 quantization stack, 1.235 for 4->2 steps |
| `median_mouth_center_distance_px` | 3.07 | — |
| `detected_pairs` | 250 / 250 | — |

Colour drift (the overlap-skip kill check — per-window mean RGB vs window 0, 8 windows):

| | R slope/window | max drift |
| --- | ---: | ---: |
| control | +0.1023 | 1.245/255 |
| overlap-skip | +0.1301 | 1.314/255 |

The plan's central worry — that persisting the decoder cache removes colour correction
from the cross-window feedback loop and lets colour drift accumulate — **did not
materialise**. Drift is statistically indistinguishable and under 0.6% of range.

## What landed, and what the measurements corrected

### 6.1 FP16 causal-cache passthrough — the biggest single win

`soulx_rtc/pro_vae_stage_backend.py`. The engines' `cache_in`/`cache_out` bindings are
already `float16`; the caller-owned cache was BF16, so all 27 engine calls per window
cast in and out. Now only `x` and `y` cross the boundary.

- **Plan said** -53.4 ms/window with -450 to -600 MiB.
- **Measured** -0.678 s / 250 frames = **-75.3 ms/window**, and **-1,452 MiB**.
  The plan's "upside 134 ms if it also removes the allocator purge -- do not plan on
  that" partially materialised.
- Quality: strictly *better* than before. This deletes a bf16 rounding step (11 -> 8
  mantissa bits) rather than adding one.
- Escape hatch: `SOULX_STAGE_CACHE_PASSTHROUGH=0`.

### 6.2 One-pass input preparation

`value.to(dtype).contiguous()` -> `value.to(dtype, memory_format=torch.contiguous_format)`.
Verified on this build: `.to(dtype)` **preserves** non-contiguous strides, so the old
form made a second full pass; and on already-contiguous same-dtype input the new form
returns the *same tensor object*. Measured **-0.126 s** (-14.0 ms/window). Bit-identical.

### 3.1 Sub-pixel `Resample[11]` and 3.2 channels-last head

`soulx_rtc/pro_decoder_ops.py` — runtime module swaps, deliberately **not** `vae.py`
edits, so the nine TensorRT engines stay valid. Together **-0.163 s** (-18.1 ms/window)
and **-478 MiB**. The plan projected -26.9 ms; the delivered figure is ~67% of that.

The sub-pixel fold is gated by an assertion that runs on **every install**, split in two
because the halves fail for unrelated reasons:

1. **Algebra**, in float64 end to end: measured **1.87e-14**. A wrong tap mapping or
   PixelShuffle channel order shows up only here.
2. **Representation**, at the live dtype: **0.66 ulp**. The folded coefficients are sums
   of two bf16 weights, which bf16 cannot hold exactly — ~1 ulp is expected *by
   construction*, not evidence of a wrong fold.

A first version conflated the two and rejected a correct implementation at
max|err|=1.03e-02. A negative control (scrambled PixelShuffle order) is rejected at 115%
of range, so the gate still discriminates.

### 2.1 Static INT8 activation scale — **the only numerical change that shipped**

`soulx_rtc/pro_quantization.py`. Skipping the per-row amax collapses the FFN's double
read of the 232 MB INT32 `_int_mm` output to one pass. Measured **dit 3.830 -> 3.557 s**
(-30.3 ms/window) against the plan's -33.5.

Two-run protocol: `--calibrate-int8` records per-site amax, `--static-int8-scales`
installs `amax * margin / 127`. Applied only to the 30 `ffn.2` sites (8960-wide input);
the 1536-wide sites fit in L2 and their second read is already free. Observed amax
spans 7.4-30.9 across blocks, so per-site scales matter. Default margin **1.25** —
activations above it saturate, and the calibration fixture is one utterance.

### 1.1 Overlap-skip — largest decoder win, and where the real bug was

`vae_decode` **9.311 -> 8.052 s (-139.9 ms/window)**, matching the plan's ~140 ms.

**Two defects were found and fixed during execution. Both were invisible to speed
measurement and to the colour gate.**

**(a) Mutual exclusion with the compiled encoder.** The obvious implementation wraps
`vae.encode` to save/restore `_feat_map`. `encode` is `torch.compile`d and guards on
that structure, so every window missed its guard:

| form | motion_encode | FPS |
| --- | ---: | ---: |
| wrap encode, compiled | 12.454 s | 9.8361 |
| restore inside decode, compiled | 2.611 s | 16.9637 |
| restore inside decode, encode uncompiled | 1.415 s | **18.4298** |

The fix restores the cache at the top of `decode` instead, and the runner now **drops
the encode compile when `--overlap-skip` is set** and records why. The compile is worth
0.295 s alone but costs 1.49 s here, against 1.17 s saved on decode.

**(b) A 4-frame lip-sync shift — caught only by the articulation gate.** The skip count
was derived from `pipeline.latent_motion_frames.shape[1]`, assumed to be stably 2. It is
not. Instrumented: `frames_per_window = [33, 33, 37, 33, 33]` — one window emitted 32
fresh frames instead of 28, injecting 4 extra frames and shifting every later frame.

This passed every other check. Speed looked excellent, the colour gate passed, raw pixel
difference was an unremarkable 6.86/255 with matching per-channel means. It surfaced as
`opening_correlation` **0.0439** at lag 0 — against **0.9584 at lag -4**, i.e. correct
content, wrong alignment. **The harness only validates the total frame count, so a
per-window imbalance is undetectable downstream by construction.**

Fixed by deriving the keep-count from the first window's actual output
(`(frames - motion_frames_px) // 4`), plus an assertion that fails loudly if any window
deviates. After the fix: uniform `[33]*5`, `opening_correlation` **0.9708**, and FPS
*improved* to 18.5950.

### 2.4 — **not implementable as specified**

The plan states `int8_w8a8` "is already in `SUPPORTED_LINEAR_SCHEMES` for both families".
It is not: `validate_policy` (`pro_quantization_v2.py:218`) restricts cross-attention
projections to `('bf16', 'fp8_e4m3_w8a8')`, and the schema carries **one scheme per
family**, so self-attn `o` cannot be INT8 while `q`/`k`/`v` stay FP8. Worth 2.5 ms/window
(under the ~3.3 ms single-run resolution) on the audio-conditioning path. **Dropped** —
widening a validator for an unmeasurable change on the lip-sync path is the wrong trade.

## Phase 3

### cudaMallocAsync — works, but wrong direction for concurrency

**19.2442 FPS (+3.5%)**, `vae_decode` 7.573. But `reserved` rises **8,308 -> 11,168 MiB**.
It hands back 2.9 GiB — the entire VRAM win — for 3.5% throughput. Since a second PRO
instance not fitting is the binding constraint on aggregate FPS, **not default**. Enable
with `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` if single-stream latency is all
that matters.

`max_split_size_mb:512` is a **no-op**: 18.4158 vs 18.4260, reserved identical at 9,396.

### 1 sampling step — REJECTED on quality

**21.6311 FPS (+16.3%)** and it is not usable. `median_edge_ratio_on_both_open` **1.3224**
— worse than the 4->2 reduction's 1.235, which already spent this budget once.
`opening_correlation` 0.861.

Visual inspection is unambiguous: speckled grey-green teeth, magenta oversaturation,
destroyed mouth texture. Artifact:
`benchmarks/pro_30fps_20260921/review-1step/mouth-comparison.png`. **This is not a
marginal call.** The plan rated it "the largest risk on the board" and that is correct.

### Not attempted

§1.2 merged engine and §1.3 tactic forcing both need an engine rebuild cycle; §6.4 arena
shrink was already shown inert (`stage-fp16-tail-build-r02` at `workspace_mib=384`
produced a byte-identical 1,418.9 MiB arena).

## Updated per-component grid (shipping config)

Artifact: `benchmarks/pro_30fps_20260921/latency-p2fixed/latency-summary.json`
(`--latency-detail stages`, shipping config, 250 frames, seed 50, 2 steps,
`complete_coverage: true`). The instrumented run reads 18.755 FPS and is stamped
`performance_claim: false`; the uninstrumented throughput number is 18.5950 FPS.
Compared against `benchmarks/pro_30fps_20260920/latency-stackedB/latency-summary.json`.

**1,728.5 -> 1,463.9 ms/window. TensorRT engine calls 27 -> 21.**

| Component | BEFORE | AFTER | delta ms |
| --- | ---: | ---: | ---: |
| decoder `trt.execute` | 698.4 (40.4%) | **617.1 (42.2%)** | −81.3 |
| DiT forward | 426.6 (24.7%) | **393.6 (26.9%)** | −33.0 |
| decoder non-engine ops | 271.1 (15.7%) | **246.3 (16.8%)** | −24.8 |
| motion VAE encode | 123.0 (7.1%) | 156.9 (10.7%) | **+33.9** |
| decoder output cast -> bf16 | 117.6 (6.8%) | **7.1 (0.5%)** | **−110.5** |
| decoder input cast/contiguous | 41.5 (2.4%) | **7.6 (0.5%)** | −33.9 |
| everything else | 43.0 (2.5%) | 28.1 (1.9%) | −14.9 |
| audio encoder | 7.3 (0.4%) | 7.4 (0.5%) | +0.1 |
| **total** | **1,728.5** | **1,463.9** | **−264.6** |

Three readings:

1. **The boundary-cast tax is effectively gone: 159.1 -> 14.7 ms/window, −91%.** The
   largest single line change in the grid, and far past the 53.4 ms the plan projected.
   Casts fall from the #4 component to 1.0% combined. The plan's stated hard floor of
   ~17 ms/window for this component is now the operating point.
2. **`trt.execute` fell 81.3 ms by doing less work, not faster work** — 21 calls per
   window instead of 27, because overlap-skip removed 6 redundant engine invocations.
   No kernel improved.
3. **Motion encode is the one regression (+33.9 ms) and is the price of overlap-skip**,
   which is mutually exclusive with the compiled encoder. Net trade is strongly positive
   (−250 ms on decode against +34), but reclaiming it is the obvious next target if the
   two can be made to coexist.

**The ranking is unchanged and the decoder still dominates**: 878.1 ms, **60.0%** of the
window (down from 65.3%). `trt.execute` alone is 42.2% — a *larger* share than before,
because everything around it shrank faster. Further real gains still have to come from
decoder engine compute, and SM89 has no precision tier below FP16, which points at the
engine-rebuild levers (§1.2 merged engines, §1.3 tactic forcing) rather than more
quantization.

## Where this leaves the goal

**20 FPS is not met.** 18.5950 FPS is 13.4445 s / 250 frames = **1,493.8 ms/window**
against the 1,388.9 ms the target needs — still **104.9 ms/window short**, i.e. 93.0% of
the way there. With cudaMallocAsync, 19.2442 FPS = 1,443.4 ms/window, **54.5 ms short**.
Neither clears it.

Closing that last 55-105 ms means the levers this round did not attempt: §1.2 merged
engine and §1.3 tactic forcing, both of which need an engine rebuild cycle. The one lever
large enough on its own — 1 sampling step — is rejected on quality above.

For concurrent streams the more useful number is VRAM: peak is now **9,017 MiB**, down
2,832 MiB. That is still one PRO instance per 12 GiB card (a second needs ~9 GiB more),
so aggregate FPS remains one-instance-bound. The decoder is still **59.9%** of the window
and remains the binding constraint.

## Files changed

| File | Change |
| --- | --- |
| `soulx_rtc/pro_vae_stage_backend.py` | FP16 cache passthrough; one-pass input prep; adapter dtype allow-list |
| `soulx_rtc/pro_quantization.py` | static INT8 activation scale + calibration helpers |
| `soulx_rtc/pro_decoder_ops.py` | **new** — sub-pixel Resample, channels-last head, overlap-skip |
| `benchmarks/pro_quantization_v2_20260918/run.py` | 6 flags; encode-compile/overlap-skip exclusion; calibration wiring |
| `benchmarks/pro_30fps_20260920/policies/phase1.json` | **new** — shipping policy |

Test suite unchanged throughout at **17 failed / 306 passed** (the same pre-existing
environmental failures documented in `PRO_STACKED_OPTIMIZATIONS_2026-09-20.md`).

## Reproducing the shipping arm

```bash
cd /workspace/SoulX-FlashHead
export PYTHONPATH=/workspace/experiments/pro30-deps/sage-sm89:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.
unset PYTORCH_CUDA_ALLOC_CONF          # costs +0.725 s on the decoder

.venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  --policy benchmarks/pro_30fps_20260920/policies/phase1.json \
  --fixtures benchmarks/pro_30fps_20260920/tts-fixtures/fixtures.json \
  --fixture-id indian150-a --seed 50 --frames 250 --repeats 1 \
  --sampling-steps 2 --timestep-variant distilled_aligned \
  --subpixel-resample --channels-last-head --overlap-skip \
  --static-int8-scales benchmarks/pro_30fps_20260921/int8-amax.json \
  --output benchmarks/pro_30fps_20260921/<arm>
```

Discard the first run of any session. `--compile-vae-encode` is accepted but
deliberately ignored alongside `--overlap-skip`.
