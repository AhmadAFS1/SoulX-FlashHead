# PRO throughput plan: 16.03 -> 20 FPS, after adversarial review

**Date:** 2026-09-21. **Audience:** the engineer who will execute this immediately.
**Supersedes** the lever table in `PRO_STACKED_OPTIMIZATIONS_2026-09-20.md` "Where this leaves
the 20 FPS goal" and closes the `[GAP]` items in `WAVE0_WAVE1_LANDED_2026-09-20.md`.

Four components were analysed independently and then adversarially refuted. Where the two
conflict, the refuter's number is the one carried here. Every number below is tagged
**MEASURED** (artifact named), **ESTIMATED** (arithmetic shown) or **UNKNOWN**.

Hardware: RTX 4070 SUPER, SM89 (Ada), 12,282 MiB visible, driver 595.84, Torch 2.7.1+cu128,
CUDA 12.8, TensorRT 10.9.0.34, SageAttention 2.2.0 SM89. Shipping config = **stackedB**
(`benchmarks/pro_30fps_20260920/policies/stacked_b_full.json`), 16.0275 FPS, 1,733 ms/window.

---

## 1. Where the time actually goes

### Per-window budget (MEASURED)

Artifact: `benchmarks/pro_30fps_20260920/latency-stackedB/latency-summary.json`
(`--latency-detail stages`, seed 50, 250 frames, 9 windows, 27 TRT engine calls/window,
`complete_coverage: true`, `dropped_events: 0`). The instrumented run measured 15.8737 FPS;
instrumentation costs ~1% and the harness stamps `performance_claim: false`. Proportions are
the result; the uninstrumented throughput number is 16.0275 FPS / 1,733 ms/window.

| Component | ms/window | % | status |
| --- | ---: | ---: | --- |
| **#1 decoder `trt.execute`** | **698.4** | **40.4%** | MEASURED |
| **#2 DiT forward** | **426.6** | **24.7%** | MEASURED |
| **#3 decoder non-engine ops** | **271.0** | **15.7%** | MEASURED (residual of `vae.decode` minus instrumented children; may contain GPU idle) |
| motion VAE encode | 123.0 | 7.1% | MEASURED |
| **#4a** `trt.output_cache_cast_bf16` | 117.6 | 6.8% | MEASURED |
| everything else | 43.0 | 2.5% | MEASURED |
| **#4b** `trt.input_cast_contiguous` | 41.5 | 2.4% | MEASURED |
| audio encoder | 7.3 | 0.4% | MEASURED |
| **`pipeline.window`** | **1,728.5** | **100%** | MEASURED |

`vae.decode` = 1,128.6 ms/window = **65.3% of the pipeline**. #4a+#4b = 159.1 ms = 14.1% of
decode, 9.2% of wall.

### #1 decoder `trt.execute` — 698.4 ms internal split

Per-engine, aggregated from `latency-events.jsonl` `metadata.engine` (MEASURED, reproduced
independently by the refuter to the digit; sums to 698.389):

| Engine | ms/window | calls/window | ms/call |
| --- | ---: | ---: | ---: |
| `stage-12-13-14` (96ch, 576x320) | 373.59 | 1+1+7 | 45.93 steady |
| `stage-8-9-10` (192ch, 288x160) | 241.34 | 1+1+7 | 29.68 steady |
| `stage-4-5-6` (192ch, 144x80) | 83.46 | 1+1+7 | 10.04 steady |

Conv-vs-overhead split (ESTIMATED: charge every non-conv inspector layer's traffic at the
measured 430 GB/s, attribute the residual to `CaskConvolution`):

| Engine | conv ms/call | achieved TFLOPS | % of ~284 peak | tactic |
| --- | ---: | ---: | ---: | --- |
| `stage-12-13-14-2` | ~25.45 of 45.93 | 86.5 | 30.5% | `sm75_...tilesize128x128x32_stage1` |
| `stage-8-9-10-2` | ~19.45 of 29.68 | 113.2 | 39.8% | `sm75_...tilesize256x64x32_stage1` |
| `stage-4-5-6-2` | ~7.79 of 10.04 | 129.9 | 45.8% | `sm80_...tilesize256x128x32_stage3` |

**Key correction from the review:** the whole-engine arithmetic intensities (194 / 388 / 752
FLOP/B vs a 563 FLOP/B machine balance) include the 41-49% Reformat traffic. **Conv-only**
intensities are **864 / 1,728 / 2,661 FLOP/B** — every convolution in every stage engine is
**compute-bound**, not bandwidth-bound. Reformat is 41.3% of `stage-12-13-14-2` traffic
(4.686 GB of 149.85 GB/window total engine traffic). MEASURED from the retained
`.layers.json` inspector output.

### #2 DiT forward — 426.6 ms internal split

Top split (ESTIMATED, calibrated against `compiled-profile-r01` and rescaled; predicted
1.320 vs measured 1.351 ms/block):

| Sub-line | ms/window | status |
| --- | ---: | --- |
| token-linear work (scales ~linearly with 6,480 tokens) | 334.0 | ESTIMATED |
| self-attention core (SageAttention2 INT8 QK / FP8 PV) | 92.6 | ESTIMATED — **currently unverifiable on this box**, see hazards |

Within the token-linear share, 60 block-forwards/window (30 blocks x 2 steps):

| Sub-line | ms/window | status |
| --- | ---: | --- |
| FFN pair (2 INT8 GEMMs + fused quantize/GELU) | ~183.7 | MEASURED standalone (3.0609 ms/block-fwd x 60) |
| &nbsp;&nbsp;of which INT8 GEMMs (`cutlass_80_tensorop_i16832gemm_s8`) | ~98.6 | MEASURED (1.644 x 60) |
| &nbsp;&nbsp;of which `triton_red_fused__to_copy_abs_amax_clamp_clamp_min_div_round_1` | **73.1** | MEASURED (1.2178 x 60) |
| &nbsp;&nbsp;of which other pointwise | ~11.0 | MEASURED (0.1838 x 60) |
| self-attn projections (120 FP8 linears) | 64.2 | ESTIMATED |
| cross-attn q/o projections | 35.8 | ESTIMATED |
| norm / modulation / residual pointwise | ~50 | ESTIMATED by subtraction |

**Key correction:** the dominant DiT intermediate is the **INT32 output of `torch._int_mm` at
[6480, 8960] = 232.2 MB**, not the 116.1 MB BF16 activation. The fused reduction reads it
**twice** (row amax, then quantize) and writes 58.1 MB: 522.5 MB at 430 GB/s = 1.215 ms
against 1.2178 MEASURED — a 0.2% match. That single kernel is 17% of the whole DiT.

### #3 decoder non-engine ops — 271.0 ms internal split

MEASURED in a compiled harness at the real shapes with real weights, cross-validated against
in-situ chunk timings (isolated chunk1 49.98 vs in-situ 49.6 ms/window, 0.8% agreement):

| Sub-line | ms/window | status |
| --- | ---: | --- |
| `Resample[11]` (upsample2d 192ch, 288x160 -> 576x320) | ~55.7 | MEASURED (6.96 ms/call x 8) |
| `head` (RMS_norm + SiLU + CausalConv3d Cout=3 @ 576x320), in situ | ~48.8 | MEASURED (6.10 ms/call x 8; isolated compiled is 4.496) |
| `conv1 + middle + upsamples[0,1,2] + Resample[3]` ("chunk1") | ~49.6 | MEASURED |
| `Resample[7]` (upsample3d 384->192, 144x80 -> 288x160) | ~42.3 | MEASURED (5.290 ms/call x 8) |
| `torch.cat` output accumulation (quadratic: ~160 MiB written, not 36.5) | ~0.7 | MEASURED |
| **unattributed residual** (27 graph-break re-entries, per-call Python validation, allocator host stalls, GPU idle) | **~74** | ESTIMATED by subtraction |

`AttentionBlock` inside `middle` runs **9** times/window (not 8) = 5.0 ms. Do not touch it.

### #4 boundary casts — 159.1 ms internal split

MEASURED byte totals reproduce exactly from the plan binding shapes: x_in 1,827.0, c_in
5,507.5, y_out 1,902.2, c_out 6,241.8 MB/window; cache share 76.6% of input bytes, 60.1% of
output bytes.

| Sub-line | ms/window | status |
| --- | ---: | --- |
| bf16->fp16 conversion of the 6 cache inputs | 24.94 | MEASURED-derived |
| mandatory strided->linear pass over `x` | 8.3 | MEASURED-derived (TRT bindings must be linear) |
| **redundant second pass** (`.to()` then `.contiguous()`) | 8.3 | MEASURED-derived |
| fp16->bf16 of the 6 cache outputs | 28.42 | MEASURED-derived |
| fp16->bf16 of `y` (engine output) | 8.66 | MEASURED-derived |
| **GPU idle inside the tail output-cast spans** (allocator purge) | **~80.6** | MEASURED (excess over the 432-468 GB/s the same code achieves on the other two engines) |

The bandwidth tell: `stage-12-13-14-2`'s output cast moves 1,080 MiB in 14.094 ms CUDA =
**80 GB/s**, while `stage-8-9-10-2` hits 413 GB/s and `stage-4-5-6-2` 436 GB/s doing the
*identical* dtype conversion. Host time on the same scope is 118.268 ms/call on the tail vs
0.119 and 0.078 ms on the other two. That is an allocator slow path, not arithmetic.

**`trt.bind_enqueue_fence` costs 0.08 ms/window over `trt.execute`.** The serialized enqueue
lock is not a bottleneck. MEASURED (698.47 vs 698.39).

---

## 2. The 344 ms/window gap

20 FPS = 12.5 s / 250 frames = **1,389 ms/window**. From 1,733 that is **-344 ms/window
(-19.9%)**.

**No single lever closes it.** The complete list of levers that individually exceed 100 ms:

| Lever | ms/window | confidence | blocker |
| --- | ---: | --- | --- |
| 1 sampling step | -212.8 MEASURED | high (speed) | quality, ungated, zero retained evidence at that operating point |
| overlap-skip (persist decoder causal cache) | ~-150 ESTIMATED (98.87 MEASURED core) | medium | removes colour correction from the cross-window feedback path |
| distilled / restructured decoder tail | -124 to -280 ESTIMATED | medium | weeks; retrains the VAE; invalidates all 9 engines |
| Myelin fusion in the FP16 engines | **~0, probably -150 (a LOSS)** | low | refuted — see §9 |

Amdahl bounds worth stating up front:

- Component #3 **cannot reach 20 FPS even if it were free**: 271.0 -> 0 gives
  16.0275 x 1733/1462 = **19.0 FPS**. ESTIMATED.
- With DiT, motion encode and audio all at zero, the decoder alone caps at **23.6 FPS**.
- The decoder (engines + non-engine + casts) is 65.3% of the window. Any plan that does not
  attack the decoder is arithmetic that cannot work.

**Combinations that close the gap:**

- **A (no quality exposure, no rebuild): -126 ms -> 17.3 FPS.** Falls 218 ms short.
- **B (A + overlap-skip): -266 ms -> ~18.9 FPS.** Falls 78 ms short. Carries colour-drift risk.
- **C (B + `WINDOW_HISTORY_FRAMES` 5->1): -316 to -322 ms -> ~19.5-19.6 FPS.** Still short, and
  adds an inter-window pose-discontinuity risk that no existing metric detects.
- **D (A + 1 sampling step): -339 ms -> ~19.9 FPS.** Essentially on target.
- **E (B + 1 sampling step): -479 ms -> ~22.1 FPS.** Clears 20 with margin, and is the only
  combination with margin for the estimates to be wrong.

**The honest summary: 20 FPS is reachable, but only by spending either one sampling step or
the motion-history window. Everything that is free has already been found and totals ~126 ms.**

---

## 3. Plan for #1 — decoder `trt.execute` (698.4 ms, 40.4%)

Ranked by value-per-risk. Four levers survive; five are dead (§9).

### 1.1 Overlap-skip: persist the decoder causal cache across window boundaries

- **Change.** Route decode through `cached_decode` (`flash_head/wan/modules/vae.py:856-881`,
  byte-identical to `decode` minus the two `clear_cache()` calls) and feed only latent frames
  3-9, **plus** save/restore `WanVAE_._feat_map` around the `self.vae.encode(cond_frame)` call
  at `flash_head/src/pipeline/flash_head_pipeline.py:404`.
- **Mechanism.** Window stride is 28 useful frames = 7 latent frames, so window N+1's latent
  frames 1-2 are exactly window N's frames 8-9. The first two engine calls per group
  (signatures 0 and 1) re-decode the motion overlap and are pure waste. After persistence all
  calls become sig2, so **no new engine signatures and no arena change**.
- **The correction that makes this work.** `WanVAE_.encode` calls `self.clear_cache()` at
  entry (`vae.py:771`) and exit (`vae.py:798`), and `clear_cache` resets `_feat_map` — the
  *decoder* cache — at `vae.py:898`. **Verified by reading the source.** The pipeline runs
  `encode` once per window between consecutive decodes, so the encoder wipes the decoder cache
  on every boundary regardless of which decode entry point is used. Swapping to `cached_decode`
  alone **saves nothing**. The clean fix (separating the two cache lifetimes in `clear_cache`)
  is a `vae.py` edit and therefore forces a 9-engine rebuild; the save/restore shim does not.
- **ms/window saved.** **98.87 MEASURED** in this component (sig0+sig1 spans:
  (8.161+43.895)+(5.322+28.281)+(4.496+8.712), reproduced to three decimals from
  `latency-events.jsonl`), **+10.15 MEASURED** boundary casts, **+~41 ESTIMATED** non-engine
  decoder ops. Decoder-wide **~150 ms/window**.
- **Confidence.** Medium. The arithmetic is exact; the temporal-contiguity argument checks out;
  the quality consequence does not.
- **VRAM.** +793 MiB allocated held across the boundary (not 831 — that figure is decimal MB
  mislabelled MiB). Effect on `peak_allocated` UNKNOWN; effect on **`peak_reserved` is the real
  risk and is unpriced**, because the persisted caches can no longer be recycled into DiT-phase
  activations. Reserved headroom is 1,118 MiB.
- **Quality risk.** **Higher than it looks.** `run.py:472` sets `color_correction_strength=1.0`
  and `flash_head_pipeline.py:391-395` takes `cond_frame` *after* `match_and_blend_colors_torch`.
  Today's decoder temporal context is therefore the re-encode of **colour-corrected** pixels.
  Persisting the cache substitutes raw pre-correction decoder state, **removing colour
  correction from the cross-window feedback loop** — the exact mechanism that suppresses
  inter-window colour drift. A single-window relative-L2 check will not see it.
- **Effort.** Medium. A stateful shim spanning two call sites, not the advertised one-line swap.
- **Cheapest kill check.** Run 250 frames with the shim and compare **per-window mean RGB
  against window 0** across all 9 windows. If the drift slope is non-zero where the control's
  is flat, the lever is dead on quality. ~16 s of GPU, no rebuild.

### 1.2 Merge the three engine groups into one engine spanning `upsamples[4..14]`

- **Change.** Generalise `ResidualStage` (`soulx_rtc/pro_vae_stage_backend.py:25-33`, which
  currently raises `"Stage must contain only contiguous Wan ResidualBlocks"` — **verified**) to
  absorb the eager `Resample[7]` and `Resample[11]`, then re-capture calibration with
  `--groups 4,5,6,7,8,9,10,11,12,13,14`.
- **Mechanism.** Engine calls/window fall 27 -> 9. Two of every three boundary cast pairs stop
  happening, the per-engine input NoOp reformat (283.1 MB in `stage-12-13-14-2`) and the
  terminal output reformat are eliminated at two of three boundaries, and TensorRT can hold
  NDHWC across what are currently engine boundaries instead of round-tripping to linear for the
  eager `Resample`.
- **This lives in `soulx_rtc/` and `benchmarks/`, NOT in `vae.py`** — `install_stage_plan`
  validates only `flash_head/wan/modules/vae.py`'s sha256
  (`pro_vae_stage_backend.py:287-292`, **verified by reading**). The engine gate is untouched.
- **ms/window saved.** 90-120 ESTIMATED, of which ~106 is directly measured boundary-cast work
  (`input_cast_contiguous` + `output_cache_cast_bf16` = 159.13 ms/window; the tail alone is
  22.640 + 98.655 = 121.3 ms, MEASURED per-engine).
- **PRICE IT JOINTLY WITH §6.1, NOT ADDITIVELY.** If the FP16-cache change lands first, most of
  the cast saving is already banked and this lever's marginal value drops to roughly **30-50 ms**
  (the eliminated reformats and the removed graph breaks).
- **Confidence.** Medium on mechanism, low on the marginal number after §6.1.
- **VRAM.** **The blocker.** One engine spanning 144x80 through 576x320 sets an arena at least
  as large as today's 1,418.9 MiB and plausibly larger, against 433 MiB of headroom. Sequence
  after §6.4 (shrink the arena) or after 1.4 (revert the tail), or treat as mutually exclusive.
- **Quality risk.** Low — same arithmetic, same precision, new tactic selection (so re-run the
  determinism gate, which the INT8 stage attempt already failed once).
- **Effort.** Large.
- **Cheapest kill check.** Build the merged engine offline and read `workspace_bytes` out of
  `results.json`. If it exceeds ~1,850 MiB the lever does not fit on this card and nothing else
  matters. ~30 s of build time, no pipeline run.

### 1.3 Force the two dominant engines off their Turing tactics (IAlgorithmSelector probe)

- **Change.** Add a TensorRT `IAlgorithmSelector` in `_build_engine`
  (`benchmarks/pro_30fps_20260919/build_decoder_engine.py:249-268`) that rejects any tactic
  whose name matches `sm75_*` / `tensor16x8x8` for the `CaskConvolution` layers, rebuild, and
  compare inspector output and per-call timing.
- **Mechanism.** The only engine that picked an `sm80` `tensor16x8x16` kernel
  (`stage-4-5-6-2`) is also the only one achieving 45.8% of peak; the two on Turing
  `tensor16x8x8` MMA — **half-rate on Ada for the same instruction slot** — sit at 30.5% and
  39.8%. That correlation is visible in the retained `.layers.json` files.
- **ms/window saved.** 60-85 ESTIMATED. Arithmetic: bringing the two `sm75` engines to
  `stage-4-5-6-2`'s measured 45.8% of peak yields 8.5 ms/call x 7 on the tail and 2.55 ms/call
  x 7 on 8-9-10 = 59.5 + 17.9, discounted for the residual method's sensitivity to the assumed
  430 GB/s.
- **Confidence.** Low-medium. **This is a probe, not a prediction** — TensorRT's autotuner did
  time these and chose `sm75`. But the dimension was previously ruled out on an arithmetic
  intensity computed *including* the overhead the levers exist to delete, which is circular.
- **VRAM.** 0.
- **Quality risk.** None arithmetically; re-run the determinism gate.
- **Effort.** Small (one selector class, ~10 s build per engine). No `vae.py` edit.
- **Cheapest kill check.** Build `stage-12-13-14` with the selector and read the tactic names
  out of the new `.layers.json`. If TensorRT has no non-`sm75` candidate for that layer the
  build will either fail or fall back, and you know in 10 seconds.

### 1.4 NEGATIVE / VRAM-only: revert the tail group to the stackedA two-group decoder

- **Change.** Drop `[12,13,14]` from the groups list in
  `benchmarks/pro_30fps_20260920/policies/stacked_b_full.json`. Policy file only. No rebuild,
  no gate.
- **ms/window.** **-43.2 (a deliberate LOSS, ~0.4 FPS)**, MEASURED from
  `tailtest-stackedA-control` vs `tailtest-stackedB-tail` (`stage_seconds.vae_decode` 10.6301
  vs 10.2409 s over 9 windows; the 0.389 s delta is well above the ~0.030 s resolution).
- **VRAM.** **-1,148 MiB reserved / -1,526.6 MiB allocated, MEASURED.** Correct the -1,138
  figure in `PRO_STACKED_OPTIMIZATIONS_2026-09-20.md`: `peak_allocated_mib` moves 4,904.84 ->
  6,431.49 and `peak_reserved_mib` 10,014 -> 11,162. **Reserved is the number that decides
  whether a second instance fits.**
- **Confidence.** High on both axes. Caveat: the two tailtest runs are `count=1`.
- **When to take it.** If the goal weights VRAM over single-stream FPS, this is the best
  VRAM-per-ms on the board by a wide margin. See §8 for why it still does not buy a second
  instance on a 12 GB card.

### Component #1 floor

Conv-only work is **~300 ms/window** of genuinely compute-bound arithmetic at the tactic tier
Ada offers. Below that, only a smaller decoder (§9 "restructured tail") moves the number.

---

## 4. Plan for #2 — DiT forward (426.6 ms, 24.7%)

### 2.1 Eliminate the second read of the 232 MB INT32 `_int_mm` output

- **Change.** Replace the two-pass reduction in the FFN with either (a) a single-pass
  row-blocked Triton kernel (quality-neutral, bit-identical if reduction order is preserved) or
  (b) a calibrated **static** activation scale for the 8960-wide site (numerical change, no
  custom kernel).
- **Mechanism.** `Int8ComputeLinear.forward` (`soulx_rtc/pro_quantization.py:80`) returns INT32
  and dequantises in a consumer kernel. Inductor's
  `triton_red_fused__to_copy_abs_amax_clamp_clamp_min_div_round_1` reads the [6480, 8960] INT32
  buffer **twice** — once for the row amax, once for the quantize — and writes 58.1 MB. One row
  post-GELU is 8960 x 2 B = 17.9 KB, comfortably inside Ada's 99 KB/block opt-in shared memory,
  so one-row-per-CTA is feasible.
- **ms/window saved.** **-33.5 MEASURED** (3.0609 -> 2.5028 ms/block-forward via the
  static-scale proxy, x 60 block-forwards). **~10 sigma** on `stage_seconds.dit` (sd 3.33
  ms/window). End-to-end 15.598 -> 15.296 s = **16.0275 -> 16.344 FPS, +1.97%**, 9.7% of the gap.
- **Confidence.** High (the proxy measurement is direct).
- **VRAM.** 0.
- **Quality risk.** Zero for route (a). Route (b) is a real numerical change and needs the
  standard oral-edge-ratio gate.
- **Effort.** Medium (Triton) / small (static scale + calibration).
- **Cheapest kill check.** Route (b) first: hard-code a static scale at the `ffn.2` input site
  and time one block-forward. If it does not reproduce ~2.50 ms, stop.
- **Negative already burned — do not repeat.** Token-chunking the FFN so the INT32 tile fits in
  the 48 MiB L2 **regresses at every chunk count**: 3.0535 unchunked vs 3.1013 / 3.3906 /
  3.4964 / 3.8129 at 2 / 4 / 6 / 12 chunks, bit-identical output (`maxabs_diff = 0`). MEASURED.
- **Corollary measured:** applying the same treatment to the four 1536-wide quantize sites is
  worth **0.2 ms/window** (2.5028 vs 2.4993) — 19.9 MB fits in L2, so that second read is
  already free. Not worth doing.

### 2.2 Drop to 1 sampling step

- **Change.** `--sampling-steps 1 --timestep-variant distilled_aligned`. `run.py:751` already
  has `choices=(1,2,3,4)`; `schedules.py:50` carries an explicit `1: [1000]` row; 1000.0 is in
  `DISTILLED_LEVELS`. No code change.
- **Mechanism.** `dit.forward` is invoked once per step. The saving is exactly one forward.
- **ms/window saved.** **-212.8 MEASURED.** `dit.forward` = 18 calls / 213.303 ms mean /
  3,839.456 ms total over 9 windows in `latency-summary.json`; 3.8307 s in
  `tailtest-stackedB-tail` (the actual 16.0275 arm); 3.8307/2 = 1.915 s / 250 frames = 212.8
  ms/window. Linearity is measured (`steps2-r01` dit 4.1623 vs `rebaseline-r01` 8.3641 =
  2.0095x). 15.598 -> 13.683 s = **16.0275 -> 18.27 FPS, +14.0%**, 62% of the gap.
  **Note:** `PRO_STACKED_OPTIMIZATIONS_2026-09-20.md:274` prices this at -2.08 s off the D2
  baseline DiT. Against stackedB's faster DiT it is **-1.92 s**. The doc is stale.
- **Confidence.** High on speed. **The risk is entirely quality.**
- **VRAM.** 0.
- **Quality risk.** **The largest on the board.** At 1 step the loop at
  `flash_head_pipeline.py:302` runs once with `t_i = 1.0` and `x_0 = noise - flow_pred`, so the
  model predicts the entire flow field from pure noise with **no terminal correction**.
  `schedules.py:30-32` records that the 2-step row was *deliberately* re-based from [1000,500]
  to [1000,250] so it terminates on 625.0 — that re-basing exists precisely because the terminal
  correction matters. **No 1-step arm exists anywhere in `benchmarks/`.** The 4->2 reduction
  already spent this budget once (median oral edge ratio **1.235**, against ~1.0030 for the
  entire quantization stack). The brief states quality is accepted *at 2 steps*, not that there
  is headroom below. And `flash_head_pipeline.py:252` shows window 0 starts with only **one**
  motion latent, so the weakest anchor in the sequence is exactly where a 1-step error would
  seed compounding.
- **Effort.** Zero to run; days to qualify.
- **Cheapest kill check.** The existing review harness on `tts-sibilants` and `tts-plosives`:
  oral edge ratio, opening correlation, mouth-centre distance, `--allow-steps-drift`. Compare
  **distributions across seeds**, never paired seeds — `flash_head_pipeline.py:349` draws a
  fresh randn per step, so seeds are not comparable across step counts.

### 2.3 Reduce the motion-history window: `WINDOW_HISTORY_FRAMES` 5 -> 1

- **Change.** 9 -> 8 latent frames, 6,480 -> 5,760 tokens. Not a flag — a module constant used
  at `run.py:90, 109, 470, 601`; lines 470/601 change what "reference" and "delivered window"
  mean, so a re-derived reference is needed first.
- **ms/window saved.** **-50 to -56** on the DiT alone: 37.1 from the token-linear term
  (334.0 x 11.1%, MEASURED geometry) plus 13-19 from attention (ESTIMATED, and **currently
  unverifiable on this box** — see hazards). Attention does **not** scale as (8/9)^2 = 0.790:
  the sage kernel tiles queries in 128-row blocks, so 6,480 = 51 padded row-blocks and 5,760 =
  exactly 45, giving 45^2/51^2 = **0.779**. 16.0275 -> ~16.50-16.55 FPS, 15-16% of the gap.
- **Cross-component effects on #1 and #3 are real and larger**, but are not claimed here.
- **Confidence.** Medium.
- **VRAM.** **-39 MB** (~-37 MiB) on the DiT phase — the buffer that shrinks is the 232.2 MB
  INT32 `_int_mm` output, not the 116.1 MB BF16 activation. Peak process VRAM is set by the
  decode phase (torch_allocated 6,064-6,132 MiB there vs 3,793-3,840 in the DiT phase,
  `resources-0.json`), so peak is unchanged or falls.
- **Quality risk.** **A conditioning change, not a precision change** — halving the temporal
  anchor at every window seam is different in kind from everything else on this board.
  **`review.py`'s metrics are all per-frame and will UNDER-DETECT the actual failure mode**,
  which is inter-window pose discontinuity.
- **Effort.** Small code, medium qualification.
- **Cheapest kill check.** Build the instrument first: a **per-seam pose-delta metric** compared
  against the within-window frame-to-frame delta distribution. It does not exist today. Without
  it, a passing review run means nothing.

### 2.4 Free rider: INT8 for self-attn `o` and cross-attn `q`/`o`

- **Change.** Policy JSON only: `fp8_e4m3_w8a8` -> `int8_w8a8` for self-attn `o`, cross-attn
  `q`, cross-attn `o`. **NOT `q`/`k`/`v` of self-attention.**
- **Mechanism.** MEASURED compiled at the real shape (M=6480, K=N=1536), timing the projection
  together with its residual consumer so the INT32->BF16 epilogue gets its chance to fuse:
  bf16 0.4530, fp8 0.4073, **int8 0.3868 ms**.
- **ms/window saved.** **-3.7 for the safe 3** (0.0205 x 3 x 60) = 1.1 sigma — **UNMEASURABLE
  on a single run**. -7.4 for all six (2.2 sigma). The earlier -15.5 figure was 2x optimistic.
- **Confidence.** High on the measurement, and the sign is right.
- **VRAM.** +0.7 MiB persistent (120 x 1536 x 4 B of per-output-channel FP32 scales).
- **Quality risk.** Low. **Do not convert `q`/`k`:** SageAttention2 re-quantizes Q and K to INT8
  per-thread inside the kernel (`quant_query_per_thread_int8_kernel`,
  `quant_key_per_thread_int8_kernel`, both visible in the trace at n=240), so routing them
  through an INT8 GEMM first is double quantization of the same values.
- **Effort.** Zero code. `int8_w8a8` is already in `SUPPORTED_LINEAR_SCHEMES`
  (`pro_quantization_v2.py:27`) for both families and `_validate_target` only requires BF16
  source weights and in/out features divisible by 16, which 1536 passes.
- **Take it opportunistically alongside a larger change; never as its own arm.** It is the only
  remaining arithmetic tier in this component now that `fast_accum` is dead.

### Component #2 floor

After 2.1, the DiT is ~393 ms/window of which ~99 is INT8 GEMM at 187-232 TOPS and ~93 is a
SageAttention kernel already at the SM89 floor. **There is no further precision tier.** The
only levers below that are fewer steps (2.2) or fewer tokens (2.3).

---

## 5. Plan for #3 — decoder non-engine ops (271.0 ms, 15.7%)

**State this first: annihilating this component entirely (271.0 -> 0) gives 19.0 FPS.
It cannot reach the target by itself.** The tier-1 bundle below is 36.3 ms = 2.1% end-to-end.

### 3.1 Rewrite `Resample[11]` as an algebraically exact sub-pixel convolution

- **Change.** Replace the nearest-upsample + `Conv2d(192,192,3)` at 576x320 with
  `Conv2d(192, 384, 3, p=1)` at 288x160 + `PixelShuffle(2)`, weights folded in closed form at
  install time.
- **Mechanism.** MAC count is unchanged (122.3 GMAC either way). The win is deleting the 270 MiB
  nearest-upsampled intermediate and getting a better GEMM shape (M=384 at 288x160 vs M=96 at
  576x320).
- **ms/window saved.** **-17.4 MEASURED twice** (compiled, full module including both
  rearranges, real weights, real shape [1,192,4,288,160]: original 6.948/6.975 vs sub-pixel
  4.787/4.796 ms/call, delta 2.16-2.18 x 8), plus ~0.5 for the t=1 rep invocation = **~17.9**.
- **Confidence.** High.
- **VRAM.** **-135 MiB peak transient ALLOCATED** (472.5 -> 337.5 MiB held). Effect on
  **reserved** — the number that gates a second instance — is UNPROVEN and probably much smaller.
- **Quality risk.** **Algebraically exact.** fp64 identity max|err| = **1.634e-13** on a +/-35
  range; bf16 delta at the real shape is 0.0469 on a +/-8.56 range = one ulp. Boundary cases
  verified by hand at p=0 and p=2H-1.
- **Effort.** Small-medium.
- **Cheapest kill check.** The fp64 identity assert itself. **Gate the install with it
  permanently, not once** — implementing this as a runtime module swap deliberately routes
  around the source-hash gate at `pro_vae_stage_backend.py:286-291`, so a future mistake in the
  weight fold produces no build-time error and only a ~1-ulp bf16 drift into `stage-12-13-14`.
- **17.4 ms is 1.0% end-to-end, ~1.2 sigma on FPS. Argue it on `stage_seconds.vae_decode`
  (0.157 s over a 250-frame run), never on FPS.**

### 3.2 `channels_last_3d` on the head's `CausalConv3d` only

- **Change.** Convert the cache+activation tensor before `F.pad`, hold the weight in
  `channels_last_3d`. **Narrow. The head only.**
- **ms/window saved.** **-9.5 to -9.8 MEASURED** (compiled 4.496 -> 3.312 ms/call x 8, +~0.3
  for the rep).
- **THE WIN EXISTS ONLY INSIDE THE COMPILED REGION.** MEASURED: **eager** head NCDHW 9.032 vs
  channels_last 10.543 ms/call — a **17% REGRESSION**; conv alone 5.510 -> 7.062, 28% worse.
  **Compiled** 4.496 -> 3.312, a 26% win. The entire lever is inductor fusing the layout
  conversion into the norm/SiLU/cat epilogue.
- **Confidence.** High, **conditional**. `results.json` carries a `dynamo_recompile_limit` key,
  and 3.3 introduces an invocation-index branch into the same compiled region. **Anything that
  drops `chunk4` out of the compiled graph — a recompile-limit fallback, a graph break, an eager
  debug path — silently flips a 9.5 ms win into a ~9 ms loss.** Add an assertion that the head
  is compiled.
- **VRAM.** **-20 MiB MEASURED eager.** The -204 MiB figure is plausible for the compiled path
  (cuDNN NCDHW->NHWC workspace disappearing) but is UNVERIFIED, and allocated-side only.
- **Quality risk.** **Bit-identical**: max|err| = 0.000e+00 in both eager and compiled on a
  +/-1.375 range — cuDNN's NCDHW path already transforms to NHWC internally, so it is literally
  the same kernel.
- **Effort.** Small.
- **Adjacent idea already dead:** under channels_last, padding Cout 3 -> 4 / 8 / 16 gives
  3.345 / 3.363 / 3.432 ms/call, **all worse** than Cout=3 at 3.312. MEASURED. "Untileable
  Cout=3" is not the binding constraint; the pad materialisation and 27x input re-read are.

### 3.3 Batch the latent-resolution prefix across the 8 steady latent frames

- **Change.** One t=8 pass for `conv1 + middle + upsamples[0,1,2] + Resample[3]`, keeping the
  rep invocation separate.
- **ms/window saved.** **-5.6 MEASURED, BIT-EXACT** (compiled: 9 sequential 49.983 vs 1 rep +
  1 batched-8 44.342 ms/window; `max|err| = 0.0` on the full [1,192,16,144,80] output). The
  49.98 isolated figure agrees with the 49.6 in-situ chunk1 to within 0.8% — the best
  cross-validation of the whole reconstruction.
- **Confidence.** High on the number.
- **VRAM.** **+60 to +80 MiB** (the batched `Resample[3]` output alone is [1,192,16,144,80] =
  67.5 MiB against 16.9 today), not +15-20.
- **Quality risk.** None — causality verified by hand: `conv1` batched with the 1-frame cache
  from invocation 0 reproduces the sequential windows exactly, and `Resample[3]`'s Rep branch
  zero-pads 2 frames on the left identically.
- **Effort.** **Medium-large, and this is the correction.** It restructures `WanVAE_.decode`'s
  per-frame loop, which lives in `flash_head/wan/modules/vae.py` — **the file the engine gate
  hashes**. Editing it invalidates all three shipped FP16 engines. Avoiding that means
  monkeypatching `WanVAE_.decode` from `soulx_rtc` **and** re-wrapping `torch.compile(vae.decode)`
  around the patched method.
- **Also fold in:** preallocate the decode output and write each invocation's frames into a
  slice, instead of `out = torch.cat([out, out_], 2)` (`vae.py:817-831`). Worth ~0.7 ms —
  **UNMEASURABLE alone**, but it is the same loop.
- **Cheapest kill check.** The bit-exactness assert, which is already written.

### 3.4 Skip the head `CausalConv3d` for decoder invocations 0 and 1

- **ms/window saved.** **-5.6 MEASURED standalone** (t=4 invocation 4.167 ms + t=1 invocation
  1.446 ms). **After 3.2 it is ~3.8 — at or under the measurement floor. UNMEASURABLE as its
  own arm.** Ship it inside 3.2/3.3 or not at all.
- **Correctness.** Verified on three legs: the head's cache slot is taken from the conv's
  *input* (`vae.py:504-521`), so `RMS_norm+SiLU+clone` must still run but the conv affects no
  cache; `delivery.lean_delivery` is false in the shipping run; `run.py:601` trims dim 0 by 5
  and the `isfinite` check at `run.py:611` runs on the **already-trimmed** window, so a NaN
  probe in frames 0-4 will not produce a false failure; `cond_frame` takes the **trailing** 5
  frames; colour correction reduces over H,W **per frame**
  (`flash_head/utils/utils.py:163-166`) so there is no frame coupling.
- **Two corrections.** The loop index lives at `vae.py:817` — **the engine-gate file**, which
  the original analysis flagged for 3.1/3.2 and forgot here. And you cannot skip the
  allocation: `torch.cat` still needs a correctly-shaped placeholder.
- **VRAM.** 0.
- **Effort.** Small, but it collides with 3.2 (an invocation-index branch inside `chunk4`).

### 3.5 Investigate the ~74 ms unattributed residual before funding anything else here

Two candidate causes, both cheap to test, both larger than every lever above:

- **(a) The TensorRT output-buffer allocator slow path.** Preallocate the fp16 **output**
  buffers per (engine, signature) instead of `torch.empty`-ing ~540 MiB every call on
  `stage-12-13-14`, and drop the `record_stream` deferral on them. Evidence:
  `trt.output_allocation` (declared `gpu=False`, so its host time is **pure allocator time**)
  costs **1.587 ms/call** on 12-13-14 vs 0.036 ms on the other two; the output cast's effective
  bandwidth is 80 GB/s on the tail vs 413/436 GB/s elsewhere. **Only the fp16 buffers can be
  reused** — they are consumed immediately by the cast, under one lock and one stream
  (`pro_vae_stage_backend.py:355-359`). The bf16 destinations must stay fresh, because
  `StageAdapter` writes them straight into `feat_cache`. This breaches the module docstring's
  deliberate "neither persists them nor aliases runtime output buffers" invariant for the fp16
  half — **make that an explicit decision, not a quiet edit.** Worth ~11.5 ms on the cast itself
  (books to #4) plus an UNKNOWN share of the residual. Budget 15-40 ms.
  **Kill check:** one run logging `torch.cuda.memory_stats()['num_alloc_retries']` per window.
- **(b) The 27 graph breaks.** Replace `@torch.compiler.disable` on `StageAdapter.forward`
  (`pro_vae_stage_backend.py:81`) with a `torch.library.custom_op` + `register_fake` wrapper so
  `vae.decode` compiles as **one** inductor graph instead of 4 regions x 9 invocations. Sizing:
  in-situ `chunk4` is 6.10 ms/call against the isolated compiled head at 4.50 — a 1.6 ms/call
  re-entry gap, 13 ms/window on the head alone; the component's total in-situ-minus-isolated gap
  is ~78 ms/window. Worth 10-40 ms, **low confidence**. `custom_op` exists in torch 2.7.1 and
  the `StageAdapter`'s shapes are static and signature-keyed, so the fake impl is trivial.
  **Caveat: one graph across the whole decoder raises peak liveness — sequence it after (a),
  not before.**

### Component #3 floor

The head's bandwidth floor is **~3.8-4.3 ms/window** (0.47 ms/call x 8; the earlier "8.5
ms/window" was an internal inconsistency). After 3.2 the head costs ~26.5 ms/window — **roughly
6x its floor**, and that is the largest single inefficiency left in the component. A fused
Triton replacement for `RMS_norm + SiLU + pad + Cout=3 conv` is the only lever that addresses
it (expected value ~8 ms beyond 3.2, range 0-16, **low confidence**) and it must beat cuDNN's
already-NHWC implicit-GEMM path on tiling, not merely dodge a pathology. **Gate it hard: time a
bare memory-movement skeleton first.** Do not fund it before 3.5.

---

## 6. The cross-cutting lever — precision plumbing (159.1 ms, 9.2%)

### Did the FP16-cache hypothesis survive refutation?

**Partly. The mechanism survived; the number did not.** The claim was "keeping the causal cache
in FP16 end-to-end is worth ~159 ms/window." The correct figure is **53.4 ms/window**. The rest
of the 159 ms **does not disappear — it moves**:

| Where the other 105.7 ms goes | ms | status |
| --- | ---: | --- |
| Mandatory strided->linear pass over `x`. **TensorRT bindings handed to `set_tensor_address` must be LINEAR memory**, so the permuted view out of `Resample` must be copied regardless of binding dtype. Unavoidable. | 8.3 | hard floor |
| `y` output fp16->bf16. Removable **only** by running the whole decoder in FP16, which edits `vae.py` and invalidates all 9 engines for 0.50% of the window. Not worth it. | 8.66 | hard floor in practice |
| **GPU idle inside the tail output-cast spans.** Not cast work at all — an allocator purge. Becomes component #3.5(a)'s problem, not a precision problem. | ~80.6 | re-attributed |
| Redundant second contiguous pass — genuinely removable, but by §6.2, not by the dtype change. | 8.3 | moved to §6.2 |

**So: the component's hard floor is ~17 ms, and levers 6.1 + 6.2 together are bounded at
~142 ms however credit is assigned between them.**

### 6.1 Keep the causal cache in FP16 end-to-end

- **Change.** Relax the bf16 input gate at `pro_vae_stage_backend.py:241/247` for **cache**
  bindings, cast only `outputs[0]` at :271, plumb a `cache_dtype` through `install_stages` into
  the `StageAdapter` guard at :102, driven by a `decoder.caller_cache_dtype` policy key.
- **Mechanism.** The engines already run FP16 — `binding_dtype` is `float16` for every binding
  in `stage-fp16-tail-build-r01/results.json`. The caller-owned cache is BF16, so every one of
  27 engine calls/window casts in **and** out. In torch 2.7.1+cu128, `t.to(float16)` on an
  already-fp16 tensor returns `t` itself and `.contiguous()` on a contiguous tensor returns `t`
  itself — **verified** — so this is a true **kernel deletion**, not a cheaper kernel.
- **ms/window saved.** **-53.4** (24.94 input + 28.42 output), 3.09% end-to-end,
  **16.0275 -> 16.54 FPS**, 15.5% of the gap. Upside 134 ms if it also removes the allocator
  purge — **do not plan on that.**
- **Confidence.** High. Attacked five ways and held.
- **VRAM.** **-450 to -600 MiB** (a win). Today a tail call holds old bf16 cache (405 MiB) +
  prepared fp16 copies (405) + `x` prepared (141) + engine outputs (566) + new bf16 copies
  (566); afterwards it holds old fp16 cache (405) + `x` prepared (141) + engine outputs (566).
  Steady-state live cache is identical at 2 bytes/element. **Verify with `peak_reserved_mib`,
  not the estimate.**
- **Quality risk.** **None. Lower than today.** `cache_in` and `cache_out` bindings are
  *already* float16, so every cache value already round-trips through fp16 range in both
  directions; nine shipped windows prove no overflow. This change **strictly deletes a bf16
  rounding step** (11 -> 8 mantissa bits). No dynamic-range hedge is needed.
- **Effort.** Small+. **Two corrections to the original sketch:** `StageAdapter` holds only
  `stage` and an opaque `execute` callable (:76-79) and has no `binding_dtype`; and
  `install_stages` is **also the eager path** (`decoder_stages.py:138`,
  `tests/test_pro_30fps.py:104-106`) where `execute` is the eager `ResidualStage` returning
  bf16 caches. The guard must accept both dtypes.
- **No engine rebuild.** `install_stage_plan` validates only `vae.py`'s sha256
  (`pro_vae_stage_backend.py:287-292`, verified). `pro_vae_stage_backend.py`'s own sha is
  recorded in the plan but never checked.
- **Cheapest kill check.** Assert the six cache tensors arrive as `torch.float16` on call 2+ of
  window 0 and that `prepared[i] is value` for each. If the identity does not hold, the fast
  path is not firing and the lever saves nothing.

### 6.2 One-token fix to the input preparation

- **Change.** `pro_vae_stage_backend.py:247`, currently
  `prepared.append(value.to(self.binding_dtype).contiguous())` — **verified in source**.
  Replace with `value.to(self.binding_dtype, memory_format=torch.contiguous_format)`.
- **Mechanism.** einops' `(b t) c h w -> b c t h w` yields stride
  `(106168320, 184320, 17694720, 320, 1)` — non-contiguous — and `.to(float16)` **preserves**
  those strides, so `.contiguous()` is a **second full pass**. Independently confirmed by a
  bandwidth reconciliation: under a one-pass model the four large engines measure 310.2 / 313.3
  / 344.8 / 350.1 GB/s, impossible when the *output* casts on the identical tensors of the
  identical engines measure 432.8-468.3 GB/s; **adding exactly one extra pass over `x` gives
  431.0 / 434.3 / 437.7 / 438.6 GB/s.**
- **ms/window saved.** **-8.3.** 0.48% end-to-end — **inside the 0.86% FPS sd, so it can never
  be demonstrated on FPS.** Argue on `stage_seconds.decode` at ~2.5x its resolution.
- **DO NOT use the `torch.empty(...); buf.copy_(value)` form.** Applied unconditionally to all
  seven inputs it forces a real 5.5 GB/window copy of the six FP16 caches back into existence
  and **silently destroys 6.1's entire 24.94 ms input saving.** The two levers are only
  independent if this patch preserves the no-op fast path.
- **Confidence.** High. **VRAM.** 0. **Quality.** Bit-identical. **Effort.** One line.
  **Ship it in the same change as 6.1.**

### 6.3 Allocator: run the free probes before building anything

The ~80.6 ms GPU bubble is real, is present in the **uninstrumented** run
(`tailtest-stackedB-tail` `stage_seconds.vae_decode` 10.2409 s = 1,137.9 ms/window vs the
instrumented 1,128.6), and is on the critical path. `resources-0.json` shows `torch_reserved`
oscillating **7,252 <-> 10,506 MiB**, which requires `cudaFree` of cached segments.

**But the canonical fix has already been measured to regress this pipeline.**
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`benchmarks/pro_30fps_20260920/iso-expandable`)
**did** cut peak reserved 9,982 -> 8,296 MiB, and the decoder got **1.074 s slower** over 250
frames against the **warm same-policy control** (`tailtest-baseline-control` `vae_decode`
10.6146 -> 11.6892 s) = **+119 ms/window**. *(The +0.725 s figure circulating in the brief is
measured against `rebaseline-r01`, whose `stage_seconds.dit` is 8.364 s versus 4.16 s
everywhere else — i.e. the cold-`.torchinductor` first run of a session, an invalid control by
the repo's own hazard list.)*

So: **do not build ping-pong buffers yet.** Run these four, each ~16 s or less, in one sitting:

1. `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync` — the one PyTorch backend built on CUDA's
   stream-ordered pool, which reclaims **without device-synchronising**. Completely different
   mechanism from `expandable_segments` (segment mapping vs who owns reclamation). Set
   `cudaMemPoolAttrReleaseThreshold` so the pool does not simply hoard, and verify
   `record_stream` behaviour — `pro_vae_stage_backend.py:263` calls it on every binding.
2. `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb` and/or `roundup_power2_divisions`. This workload's
   large-pool block sizes are few and fixed — 566,231,040 B engine-output sets, 141,557,760 B
   `x` buffers, 70,778,880 B cache tensors — the textbook profile for `max_split_size_mb`.
   Reserved climbs 7.5 -> 10.5 GiB while allocated sits at 3.8 GiB: that **is** split
   fragmentation. **Every retained `results.json` records an empty env; neither knob has ever
   been tried.**
3. **Falsify the competing hypothesis.** `gc.disable()` + `gc.freeze()` around the generate
   loop. Process RSS is 3,252-3,313 MiB and the stalls are **two per window at fixed call
   positions** (tail output-cast calls 1 and 4 of 7, all 9 windows, host 378-436 ms). That
   determinism is **also** a gen-2 collection signature. The reserved oscillation proves the
   allocator purges; it does **not** prove the purge costs the 400 ms of host time.
4. Log `num_alloc_retries` and `segment.all.freed` deltas per window, and add a
   `torch.cuda.synchronize()` immediately before the tail output-cast scope to prove where the
   idle sits.

- **ms/window saved.** **0 to 87. Planning number 25-30.** **Confidence: low.**
- **VRAM.** 0 for the env-var routes. **+405 MiB for tail-only ping-pong, +700 for all three
  groups — against ~409 MiB of real driver-level headroom** (`gpu_vram_used` 11,873 of 12,282).
  A permanently pinned 405 MiB block in the large pool is a plausible way to make fragmentation
  **worse**. Given the goal is concurrency, that is the wrong direction.
- **Correction to a figure in circulation:** "`peak_reserved` 11,164 vs `peak_allocated` 6,430
  = 4,734 MiB of slack" subtracts two peaks that **never co-occur**. In `resources-0.json` the
  two are anti-correlated: allocated peaks at ~6.1 GiB while reserved is at its **minimum**
  ~7.5 GiB (slack 1.4 GiB), and reserved peaks at ~10.5 GiB while allocated is at its minimum
  ~3.8 GiB (slack 6.2 GiB). Nine of 27 samples (33%) are in the high-allocated state, matching
  the dit+motion_encode share of the window (550/1728 = 32%) — **so the 2.3 GiB working-set
  oscillation is the DiT/decoder alternation, not the decoder alone.** Halving the decoder's
  transient traffic may reduce but not remove the retry.

### 6.4 Shrink the TensorRT workspace arena (a VRAM lever, not a latency lever)

`install_stage_plan` (`pro_vae_stage_backend.py:300-303`) allocates
`torch.empty(max(workspace_bytes), uint8, device='cuda')` and holds it for the whole run. From
`stage-fp16-tail-build-r01/results.json` that maximum is **1,487,831,040 B = 1,418.9 MiB**,
driven by `stage-12-13-14`. **That is 12% of the card and the single largest block in the
allocator** — both a direct VRAM cost and a prime fragmentation driver.

- **Change.** Rebuild the tail engines with `set_memory_pool_limit(WORKSPACE, 256 MiB)`.
- **Returns.** **~1,100 MiB.** 0-30 ms of latency (UNKNOWN; TRT may pick a slower tactic under
  a tighter budget).
- **Cost.** 9 rebuilds plus the tactic-reroll risk that produced the determinism-gate failure at
  `stage-int8-build-r02`. **Gate on both `peak_reserved_mib` and `stage_seconds.decode`.**
- **Known counter-evidence to read first:** `stage-fp16-tail-build-r02` used `workspace_mib=384`
  against r01's 2048 — a **5.3x swing** — and produced an **identical** arena (1418.9 MiB, same
  `workspace_bytes` on every one of the 9 signatures), identical tactics, and a marginally
  **worse** `stage-4-5-6-2` (58 layers / 32 Reformats vs 56 / 29). **`engine.device_memory_size`
  is activation memory, not tunable tactic scratch** — so the 256 MiB probe may return nothing.
  It is still the only route to 1.1 GiB and it is one build.
- **Bundle with it:** drop `cache_out_0` from every engine's output bindings and produce it
  caller-side. `ResidualStage.forward` (`pro_vae_stage_backend.py:44-48`, verified) computes the
  first cache update as `x[:, :, -2:].clone()` where `x` is the **stage input** — literally a
  slice of the tensor the caller already materialised as a contiguous FP16 buffer at :247. Worth
  ~4.7 ms/window and one fewer 70.8 MB escaping allocation per call. **UNMEASURABLE alone —
  never its own rebuild.**

---

## 7. Execution order

### The dependency that governs everything

**`flash_head/wan/modules/vae.py` is the only file `install_stage_plan` hashes**
(`pro_vae_stage_backend.py:287-292`). Any edit to it invalidates all 9 engines and forces a
recapture + rebuild + requalification. **Batch every `vae.py` edit into one engine cycle. Never
interleave.** By contrast `soulx_rtc/pro_vae_stage_backend.py`, `benchmarks/`, the policy JSONs
and runtime module swaps are all outside the gate.

### Phase 0 — free diagnostics and free kills (hours, no builds)

| # | Action | Why first |
| --- | --- | --- |
| 0.1 | Warm-cache run, discarded | Establishes the session baseline; see protocol |
| 0.2 | The four allocator/GC probes in §6.3 | They can redirect or delete §3.5(a) and §6.3 entirely, and cost 4 runs |
| 0.3 | Rebuild `stage-12-13-14` and count Reformats in the new `.layers.json` | Settles the Myelin-fusion question offline for ~10 s of build time |
| 0.4 | `IAlgorithmSelector` build probe (§1.3) | Reads tactic names out of the inspector; no pipeline run |
| 0.5 | Merged-group build probe (§1.2): read `workspace_bytes` | If it exceeds ~1,850 MiB the lever is dead on VRAM before any code is written |
| 0.6 | Static-scale proxy for §2.1 route (b) | One block-forward timing confirms -33.5 ms |

### Phase 1 — zero-rebuild, zero-`vae.py`, zero quality exposure (~-126 ms -> ~17.3 FPS)

Land as **one** change set, measured on `stage_seconds`, in this order:

1. **§6.1 FP16 cache + §6.2 the one-token fix** — **-61.7 ms**, and 6.2 must ship with 6.1 or it
   will be written in the form that destroys it.
2. **§2.1 INT32 single-pass FFN reduction** — **-33.5 ms**.
3. **§3.1 sub-pixel `Resample[11]`** — **-17.4 ms** (runtime module swap, fp64 assert gated).
4. **§3.2 `channels_last_3d` head** — **-9.5 ms** (assert the head is compiled).
5. **§2.4 INT8 self-attn `o` + cross `q`/`o`** — **-3.7 ms**, free rider, policy JSON only.

**Cumulative -125.8 ms/window -> 1,607 ms -> 17.28 FPS.** ESTIMATED as the sum of six
independently MEASURED deltas; the interaction risk is that 6.1 and 3.1 both reduce decoder
transient allocations and may partly overlap with whatever §6.3 finds.

### Phase 2 — the engine cycle (batch every `vae.py` edit here)

Everything below invalidates engines or needs a rebuild. **Do it once.**

1. **§1.1 overlap-skip.** Prefer the save/restore shim at
   `flash_head_pipeline.py:404` (no `vae.py` edit). Only if that proves unworkable, take the
   clean `clear_cache` split and pay the rebuild here.
   **-~140 ms after Phase 1** (the 10.15 ms of boundary casts is mostly already banked by 6.1).
   **Gated on the colour-drift check.**
2. **§3.3 batched latent prefix** (`vae.py` loop or monkeypatch) — **-5.6 ms**, +60-80 MiB.
3. **§3.4 head-conv skip for invocations 0/1** — **-3.8 ms**, rides along, never its own arm.
4. **§6.4 arena shrink + `cache_out_0` removal** — the VRAM cycle. Re-run the determinism gate.

**Cumulative after Phase 2: ~1,457 ms -> ~19.1 FPS**, with the colour-drift risk live.

### Phase 3 — the residual, and then the hard choice

1. **§3.5(a)/(b)** — whichever of allocator or graph breaks Phase 0 implicated. 15-40 ms,
   UNKNOWN.
2. **§1.2 merged engine** — only if 0.5 said it fits, and only after §6.4 freed the arena.
   ~30-50 ms marginal after Phase 1.
3. **§1.3 tactic forcing** — 60-85 ms if Phase 0.4 found a candidate; 0 otherwise.
4. **Then the quality decision.** To clear 20 FPS you must spend either **§2.2 (1 step,
   -212.8 ms)** or **§2.3 (`WINDOW_HISTORY_FRAMES` 5->1, -50 to -56 ms + cross-component)**.
   §2.3 is smaller and cheaper to reverse; §2.2 is the only one that clears the target alone.
   **Build §2.3's per-seam pose-delta metric before running §2.3**, and run §2.2's
   distribution-across-seeds gate before believing any 1-step result.

### Measurement protocol (non-negotiable)

- **The first run of any session is invalid.** Cold 334 MB `.torchinductor` cache, ~2% slow, and
  it produces a different `raw_rgb_sha256`. **Warm the cache, discard run 1.**
- **`raw_rgb_sha256` is NOT an acceptance gate.** Byte-identical configs produce different hashes
  across a cold/warm boundary. Use it only for within-session op-order checks.
- **Never set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in a throughput arm.** MEASURED
  +1.074 s on the decoder against the warm control.
- **Argue on `stage_seconds`, not FPS.** FPS sd ~0.86%, full range ~2.22%. `stage_seconds`
  resolves ~10x smaller deltas (`stage_seconds.dit` sd 0.0300 s over 250 frames ~= 3.33
  ms/window). **Anything below ~3.3 ms/window is UNMEASURABLE on a single run** and must ship
  bundled.
- **Watch VRAM every run.** Log `peak_reserved_mib` **and** `peak_allocated_mib` **and**
  `nvidia-smi` peak. They are anti-correlated and only reserved decides whether a second process
  fits. stackedB is at 11,849 of 12,282 MiB.
- **Keep the OmniVoice TTS server on :8002 stopped** for every throughput arm (2,876 MiB), and
  state in the artifact that it was.

Exact command shape for every throughput arm:

```bash
cd /workspace/SoulX-FlashHead
export NEW_DEPS=/workspace/experiments/pro30-deps/sage-sm89:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.
export OUT=benchmarks/pro_30fps_20260921

PYTHONPATH="$NEW_DEPS" .venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  --policy benchmarks/pro_30fps_20260920/policies/stacked_b_full.json \
  --fixtures benchmarks/pro_quantization_v2_20260918/fixtures.json \
  --fixture-id indian150-a --seed 50 --frames 250 --repeats 1 \
  --compile-vae-encode --latency-detail off \
  --output "$OUT/<arm-name>"
```

Latency attribution (diagnostic only, costs ~1%, never a throughput claim):

```bash
PYTHONPATH="$NEW_DEPS" .venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  ... --latency-detail stages --output "$OUT/latency-<arm-name>"
```

Engine plan preflight before any run that touches engines (milliseconds, fails before the lease):

```bash
.venv/bin/python benchmarks/pro_quantization_v2_20260918/plan_preflight.py \
  benchmarks/pro_30fps_20260920/stage-fp16-tail-build-r01/results.json
```

### Environment note — the reported hazard was a FALSE ALARM (corrected 2026-09-21)

An earlier draft of this plan reported that the box had moved to Python 3.12.14 / torch 2.8.0 /
triton 3.4.0, that `import tensorrt` failed, that the cp310 SageAttention build no longer
imported, and that self-attention was therefore unmeasurable on this machine.

**None of that is true.** That finding came from running the bare system interpreter
(`/usr/bin/python3`, which is indeed 3.12.14) instead of the project venv. Verified directly:

```
$ PYTHONPATH=/workspace/experiments/pro30-deps/sage-sm89:.pro-quant-deps:\
/workspace/experiments/ojin-components-deps:. .venv/bin/python -c ...
python 3.10.12 | torch 2.7.1+cu128
tensorrt 10.9.0.34
sageattention imported OK
SM89 fp8 kernel symbol (sageattn_qk_int8_pv_fp8_cuda) resolves OK
```

The stack matches `latency-stackedB`'s recorded provenance exactly. **No interpreter restoration
is required and nothing in this plan is blocked.** `tensorrt` and `sageattention` are not
installed into the venv itself — they are supplied by the `NEW_DEPS` PYTHONPATH that every run in
this project already sets, so a bare `.venv/bin/python` without that PYTHONPATH will also fail to
import them. That is by design, not breakage.

**Always use `.venv/bin/python` with `PYTHONPATH="$NEW_DEPS"`.** Never `python3`.


## 8. Expected outcome and the honest ceiling

Cumulative, starting from 1,733 ms/window / 16.0275 FPS:

| Stage | delta ms | ms/window | FPS | basis |
| --- | ---: | ---: | ---: | --- |
| stackedB today | — | 1,733 | 16.03 | MEASURED |
| + Phase 1 (6 high-confidence items) | -126 | 1,607 | **17.28** | ESTIMATED sum of MEASURED deltas |
| + Phase 2 (overlap-skip + prefix + head skip) | -150 | 1,457 | **19.06** | 98.87 MEASURED core, rest ESTIMATED |
| + Phase 3 residual/engine work (mid-band) | -60 | 1,397 | **19.88** | ESTIMATED, wide band |
| + 1 sampling step | -213 | 1,184 | **23.45** | MEASURED |

**Confidence band.** Phase 1 is tight: every line is an independent measurement, so
**17.0-17.4 FPS**. Phase 2 hinges on one 98.87 ms MEASURED core plus ~41 ms of estimated
non-engine follow-on, so **18.6-19.3 FPS**. Phase 3 is **17.9-20.3 FPS** — the band is wide
because §1.3 is a probe (0-85), §3.5 is UNKNOWN (0-40) and §1.2's marginal value after Phase 1
is uncertain (0-50).

**Is 20 FPS reachable? Yes, but not without spending something.**

- **If only the high-confidence items land: 17.3 FPS.** That is the honest ceiling of
  "free" — no quality exposure, no engine rebuild, no architecture change. It is 86% of target.
- **Everything short of a quality decision: ~19.1 FPS**, and even that carries the colour-drift
  risk of the overlap-skip.
- **20 FPS requires a quality decision**: 1 sampling step (clears it with margin, ~23.5 FPS, and
  is the single biggest lever on the board) or `WINDOW_HISTORY_FRAMES` 5->1 plus everything in
  Phases 1-3 landing near the top of its band. **A third option exists and is honest to
  surface: resolution is a product decision that has not been taken** — it is not assumed
  anywhere in this plan. ROI/crop reduction is ruled out by the user and does not appear.
- **The research route** — distilling a shorter decoder tail (drop one of the three
  `upsamples[12,13,14]` ResidualBlocks, or move `Resample[11]`'s 2x upsample after them so they
  run at 288x160) — is worth **-124 to -280 ms** and is the **only lever that lowers the ~300 ms
  arithmetic floor**. FLOP scaling is the right model here (conv-only AI 864 FLOP/B against a
  563 balance, so the convs are genuinely compute-bound). It is also weeks of work with no
  training pipeline, no reference data and no quality bar in this repo, and it trips **both**
  gates at once (`weights_sha256` and `source_sha256`).

### What this means for concurrent streams

**Blunt answer: the VRAM levers in this plan will not fit a second PRO instance on a 12 GB card,
and the plan should not be sold as if they will.**

- Today: 11,849 MiB peak of 12,282. One instance needs ~11.1 GiB.
- Best case with every VRAM lever: §6.1 (-450 to -600 MiB) + §6.4 (-1,100 MiB) ->
  **~10.2 GiB**. Or §6.1 + §1.4 tail revert (-1,148 reserved, costs 43 ms) -> ~10.1 GiB.
  Two instances need ~20.3 GiB. **Not close.**
- Therefore **aggregate FPS on this card comes from single-instance throughput, not from a
  second process.** Every ms in Phases 1-3 is directly aggregate FPS.
- **The VRAM work is still worth doing**, for three reasons: it removes the fragmentation that
  is plausibly costing ~80 ms/window (§6.3), it is what makes a second instance feasible on a
  16 GB or 24 GB card, and it buys headroom for the +793 MiB the overlap-skip holds and the
  +60-80 MiB the batched prefix adds — **Phase 2 needs Phase 1's VRAM savings to fit.**
- **Sequencing consequence:** §6.1 is not only the best latency-per-risk item in the plan, it is
  the VRAM prerequisite for Phase 2. It goes first for two independent reasons.

---

## 9. Do not bother

Everything below is refuted with evidence. **Do not re-litigate.**

### Refuted this round

| Claim | Verdict | Evidence |
| --- | --- | --- |
| **Boundary casts dominate the decoder / explain the INT8 null** | **WRONG (earlier speculation, now dead)** | Casts are 159.1 ms = 14.1% of decode, 9.2% of wall. `trt.execute` is 61.9% of decode. The engines do genuine convolution work. |
| **FP16 cache is worth ~159 ms/window** | **53.4 ms** | 8.3 ms is a mandatory linear-binding pass, 8.66 ms is the `y` cast (needs a full rebuild), ~80 ms is allocator idle, not cast work. §6 |
| **`fast_accum: true`** | **DEAD, -0.47 ms/window** | MEASURED interleaved A/B, 5 reps x 200 iters: at (6480,1536,1536) — the only shape it reaches in stackedB — OFF 0.2284 ms / 133.9 TFLOP/s vs ON 0.2271 / 134.6. **Max FP8 throughput observed anywhere, fast_accum ON, = 141.8 TFLOP/s = exactly the ~142 FP32-accumulate cap. There is NO ~284 TFLOP/s FP8 tier on Ada consumer.** The earlier 28.4 ms midpoint was ~60x the truth. |
| **Hand-written Triton kernel to fuse FFN amax + quantize** | **DEAD as scoped — inductor already emits it** | The compiled INT8 FFN emits ONE kernel, `triton_red_fused__to_copy_abs_amax_clamp_clamp_min_div_round_1`, whose name enumerates abs, amax, clamp, div AND round. The two-kernel split observed earlier is an artifact of the **FP8 tensorwise** path, where the amax is global and genuinely must materialise first. **Re-scoped to the INT32 two-pass problem it is worth -33.5 ms — see §2.1.** |
| **Restore Myelin fusion in the FP16 engines** (LpNorm rewrite / opt level 3->5 / drop STRONGLY_TYPED) | **WEAK-to-DEAD; variant (c) predicted to LOSE ~150 ms** | Across every build in this repo the Myelin fused conv appears with **exactly one** tactic family, `sm86_xmma_fprop_implicit_gemm_bf16bf16_bf16f32_f32` — **f32 accumulate, zero exceptions**. Every f16-accumulate tactic belongs to `CaskConvolution`, the **unfused** path. The BF16 build fused fully (19 layers, 0 Reformats) under a **byte-identical** builder (sha `ceff9bd9…`), the same `STRONGLY_TYPED`, the same optimization level 3, the same 512 MiB workspace, on a topologically identical ONNX — **so the element type is the only discriminator, and dropping STRONGLY_TYPED just reproduces the measured BF16 config: 1.4660/1.4696 s vs FP16's 1.3188/1.3195 s, +11.2% slower.** The fully-fused BF16 set moves **31% LESS** DRAM traffic (39.60 vs 57.26 GB/window) and is still slower — direct evidence these engines are not bandwidth-bound. |
| **`builder_optimization_level` 3 -> 5** | **DEAD, 0 ms** | BF16 fused completely **at level 3**. Level is demonstrably not the gate. Separately, `stage-fp16-tail-build-r02` swung `workspace_mib` 2048 -> 384 (5.3x) and produced an **identical** arena, identical tactics, and a marginally worse `stage-4-5-6-2`. Two build-config knobs, no movement. |
| **Strip the no-op `cache_x.to(x.device)` Cast nodes from the ONNX** | **DEAD, 0 ms** | The six surviving Casts are layers **7-12** of `stage-12-13-14-2`, immediately after the six Constants and **before** any Reduce, kgen, Slice or Cask conv. TensorRT hoisted all six cache inputs to the top and converted them in one block — the signature of a **layout conversion staged ahead of the convolutions**. The ONNX carries 12 and TRT folded 6: the 6 that survived survived **because they do work**. Deleting the node cannot delete the Cask kernel's NDHWC requirement; TRT re-materialises the same bytes as a CopyNode. |
| **Decode 2 latent frames per engine call (T=8 tail slice)** | **DEAD** | Three kills: ~2,360 MiB projected against a 1,418.9 MiB arena and 433 MiB headroom; it edits `vae.py:816` (engine gate + collides with the overlap-skip); and the physics saturate at -8.1% because cache traffic is a minority of non-conv bytes while the convs are compute-bound and scale linearly with T. ~48 ms at best for +950 to +1,400 MiB. |
| **CUDA graphs (decoder engine sequence or the 30-block DiT)** | **DEAD** | `trt.bind_enqueue_fence` minus `trt.execute` = **0.08 ms/window**, ~40x below the measurement floor. DiT: `dit.forward` host_total 426.045 ms against cuda_elapsed_total 3,839.456 — the CPU is **11.1%** of the GPU, the launch queue cannot drain, and 0.28 ms mean per kernel is two orders of magnitude above per-launch cost. Shape changes between sig0/sig1/sig2 would defeat capture independently. +several hundred MiB for zero gain. |
| **Any further precision reduction inside the decoder engines** | **DEAD** | Right answer, and the **reason matters**: not "bandwidth-bound" (the convs are compute-bound at 864-2,661 FLOP/B) but the **hardware tier** — every FP16 tactic is already `f16f16_f16f16_f16`, INT8 shares that exact tier on Ada, FP4/FP6 do not exist on SM89, and Q/DQ reformats would be added to a graph already 41-49% reformat. MEASURED: `stage-int8-trial-r01` 0.985x / 1.008x, -35% to -43% end-to-end, plus a determinism-gate failure. |
| **Mixed-dtype engine I/O bindings (x bf16, caches fp16)** | **DEAD, provably 0 ms** | TensorRT bindings handed to `set_tensor_address` must be **linear** memory. The permuted view out of `Resample` cannot be bound at any dtype, so the strided->linear pass is mandatory and a linear pass that *also* converts moves exactly the same bytes. Upper bound on top of §6.2 is **exactly 0**, for 9 rebuilds and a tactic reroll. |
| **Run the whole decoder in FP16** | **DEAD, 8.66 ms (not 16.9)** | An FP16 decoder still hands the engine a permuted view needing one linear pass. Only the `y` cast disappears: 0.50% of the window, below the FPS noise floor, in exchange for invalidating all 9 engines. (FP16 and BF16 are both 2 bytes, so the 271 ms of non-engine ops get no faster either.) |
| **`torch.compile(mode='max-autotune-no-cudagraphs')` on `vae.decode`** | **DEAD, subsumed** | `"Not enough SMs to use max_autotune_gemm mode"` fires on every compile (56 SMs), so it reaches only the pointwise/reduction tuner. Its entire 9.3 ms came from `Resample[11]` (7.086 -> 5.921) — the module §3.1 deletes, and §3.1's 4.787 is already well below it. Funding both is double-counting, and it adds minutes to cold start. |
| **Extend TRT coverage to `upsamples[0,1,2]`** | **DEAD, -3 to +1 net** | The 1.69x TRT ratio was measured on a 506 GMAC/call group; extrapolating it down to a 69 GMAC group on a 1.1M-element tensor is the wrong direction — at that size a TRT enqueue is launch-bound. The measured floor for any engine in this repo is 4.496 ms/call. New casts (5.0 ms) and nine more graph breaks make it net negative. **Only flips positive if §6.1 lands first.** |
| **`channels_last_3d` across the decoder; replacing the Resample rearranges with `conv3d((1,3,3))`** | **DEAD, -27.8% MEASURED regression** | And now with the mechanism: channels_last pays **only** where inductor can fuse the conversion into an adjacent epilogue (head, compiled: 4.496 -> 3.312) and loses everywhere it must materialise a transform (head, eager: 9.032 -> 10.543). A whole-decoder conversion forces transforms at every TRT boundary — the engines bind NCDHW (`pro_vae_stage_backend.py:236-245`) — and strands `RMS_norm`'s channel-axis reduction on strided layouts. |
| **Fused 2-tap sub-pixel kernel for `Resample[11]` (beyond §3.1)** | **DEAD, 0-12 ms realistic** | MEASURED: sub-pixel conv alone 4.250 ms/call vs conv+pixel_shuffle 4.766 — the **entire** fusible epilogue is 0.516 ms/call = 4.1 ms/window. The exact 2-tap form (four k=2 convs from the sparse fold) costs **5.179 ms/call, slower** than the dense sub-pixel at 4.766. Absolute ceiling 21 ms; realistic 0-12 for multi-week CUTLASS/Triton work. |
| **Fuse `Resample[11]` + head into the `stage-12-13-14` engine** | **WEAK, ~14 ms marginal (not 23.5)** | The 1.40x ratio was measured on three dense 3x3x3 ResidualBlocks at ~64 TFLOP/s. **The head is 5.73 GMAC/call at 2.7 TFLOP/s — ~62 GB/s against a ~450 GB/s ceiling, bandwidth-bound by 20x.** TRT will land at 1.0-1.1x there, not 1.4x. Plus: the head is in `decoder.head`, not `decoder.upsamples`; `ResidualStage` rejects non-ResidualBlocks; `install_stages` only walks `upsamples`. New stage class, new ONNX export, new cache accounting, +0 to +560 MiB against 433 MiB headroom. |
| **Hand-fuse the DiT norm/modulation/residual pointwise passes** | **DEAD, ~7 ms realistic (2.2 sigma)** | Those kernels measure 0.072-0.097 ms on a 39.8 MB round trip = **at or above achievable HBM bandwidth**, so they are already partly L2-served (19.9 MB in a 48 MiB L2). Achievable on this part is **420-475 GB/s, not 506** (506 is 100.4% of the 504.2 GB/s spec — impossible for a pure HBM pass). Large effort, sub-floor return, and it permanently blocks inductor from fusing across. |
| **Skip compute for the 1,440 tokens (22.2%) whose DiT output is discarded** | **DEAD, -1.4 ms** | `flash_head_pipeline.py:307` overwrites `noise[:, :latent_motion_frames.shape[1]]` every step and line 359 again before decode, so those outputs cannot reach the frame — **but** `pro_attention_backends.py:204` passes `is_causal: False` unconditionally, so every history row is a key and value for every position at all 30 blocks. The rows must be propagated. The residue is the last block's FFN + o-proj + head = 1.4 ms, for a correctness-critical indexing change across a block boundary that **no FPS gate would catch if it silently broke**. Making the temporal axis causal would unlock it — that is a redistillation measured in weeks. |
| **Packed QKV in the DiT** | **DEAD, ~0.24%** | `_pro_target_paths` rejects `packed_qkv` deliberately. |
| **Rowwise FP8 scaling** | **DEAD** | MEASURED to cost time. |
| **`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** | **DEAD in any throughput arm** | **+1.074 s on the decoder** over 250 frames against the warm same-policy control (`tailtest-baseline-control` 10.6146 -> 11.6892 s) = +119 ms/window. It *did* cut peak reserved 9,982 -> 8,296 MiB, so it is a VRAM tool only, and an expensive one. |
| **Token-chunking the FFN to fit the INT32 tile in L2** | **DEAD, regression at every chunk count** | 3.0535 unchunked vs 3.1013 / 3.3906 / 3.4964 / 3.8129 at 2/4/6/12 chunks, `maxabs_diff = 0`. MEASURED. |
| **Static-scaling the four 1536-wide DiT quantize sites** | **DEAD, 0.2 ms** | 2.5028 -> 2.4993. 19.9 MB fits in the 48 MiB L2, so the second read there is already free. |
| **Padding the head's Cout 3 -> 4/8/16 under channels_last** | **DEAD, all worse** | 3.345 / 3.363 / 3.432 vs 3.312 at Cout=3. MEASURED. |
| **Sub-pixel rewrite of `Resample[7]`** | **DEAD, 0.8 ms** | Isolated it looks worth 5.9 ms/window (4.723 -> 3.991 ms/call); re-measured on the **full module in the shipping-shaped graph** it collapses to 0.8 (5.290 -> 5.190). A cautionary example of the isolated-microbenchmark trap. |
| **Self-attention core precision** | **DEAD** | Already INT8 Q/K + FP8 P/V via `sageattn_qk_int8_pv_fp8_cuda`. **No lower tier exists on SM89.** |
| **Attacking the serialized enqueue lock** | **DEAD, 0.08 ms/window** | `trt.bind_enqueue_fence` 698.47 vs `trt.execute` 698.39. |

### One more correction that prevents a double-count

**`chunk_frames` 28 -> 36 is NOT a DiT lever.** `PRO_STACKED_OPTIMIZATIONS_2026-09-20.md:274`
lists it at -0.85 s in a table aimed at the 20 FPS gap. Essentially **none** of that belongs to
the DiT: 11 latent frames x 720 = 7,920 tokens, and
`7 x (334.0 x 7920/6480 + 92.6 x (62/51)^2) = ~3,823 ms per 250 frames` against today's
`9 x 426.6 = 3,839 ms` — **a 16 ms wash**. The real -0.85 s is decoder and encoder amortisation
(paying the fixed 5-history-frame cost 7 times instead of 9) and belongs to components #1 and
#3. **Do not count it twice.** It also carries an unflagged product cost: buffered audio per
window goes from 33/25 = 1.32 s to 41/25 = **1.64 s**, a conversational-responsiveness
regression for a WebRTC talking head that the FPS number hides.

