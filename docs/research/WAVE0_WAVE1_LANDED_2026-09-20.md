# mcPRO series: Wave 0, Wave 1 and D2 — landed source changes and the 4070 runs they require

Landed **2026-09-20**. Commits are tagged `REVERTABLE from mcPRO [n/3]` with a
`Revert-Tag: mcPRO` trailer; git tags `mcpro-before` (pre-change tree) and
`mcpro-latest` bracket the series. Commands to run these changes are in the
[RTX 4070 SUPER runbook](RTX4070_RUNBOOK_2026-09-20.md).

**Evidence class: STATIC SOURCE ANALYSIS plus CPU-only tests. NO new GPU inference.**
Every change here was written and validated on an Apple Silicon Mac with **no CUDA,
no GPU, and no torch installed** (`import torch` → `ModuleNotFoundError`, verified this
session). Nothing in this document is a speed measurement. **132 CPU tests pass locally**
(pytest 9.1.1, CPython 3.13.14, macOS arm64); the torch-dependent guards are skipped here
and must be run where torch is installed.

**Hardware for every reused figure below: NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB
visible VRAM; physical class recorded as `unverified in this run` by the source
artifact.** Driver **595.84**, Torch **2.7.1+cu128**, CUDA runtime **12.8** (build
toolchain `nvcc 12.1`, `CUDA_VERSION=12.1.1`), TensorRT **10.9.0.34**, SageAttention
**2.2.0** built for SM89. Evidence: the `environment` block inside
`benchmarks/pro_30fps_20260919/extended-r01/recurrence-candidate/results.json`, which
also records `git.dirty = True` at commit `877a07f` — every retained `[M]` number comes
from a dirty working tree and is a correspondingly weaker evidence source.

Label key: `[M]` measured GPU inference in a retained artifact · `[A]` arithmetic on
measured values · `[CPU]` CPU-only test or static check · `[GAP]` not measured anywhere.

---

## 0. The finding that reframes this work

**At 4 denoising steps, 30 FPS is arithmetically unreachable no matter what happens to
the decoder.** `[A]` Measured DiT is **8.474913 s** per 250-frame equivalent; the 30-FPS
budget is **8.333333 s**. DiT alone is **101.70%** of the entire budget. With a free
decoder, free motion encoder, free audio and zero remainder the ceiling is **24.4118 FPS**.

`[A]` Every surviving execution optimisation, at full expected value, reaches **14.883 FPS**;
at 50% of expected, **13.127 FPS**. The target needs **2.5549×**. Execution work falls short
by **2.02×**.

The user's goal is **aggregate** FPS across concurrent streams. That does not change the
arithmetic — aggregate FPS is still generated FPS — and two retained facts bound it:

1. `[CPU]` **There is no PRO serving path.** `soulx_rtc/engine.py:99` hardcodes `"lite"`.
   The WebRTC/calls/worker stack is LITE-only, with PRO-incompatible constants:
   `chunk_frames = 24` (`engine.py:41`) vs PRO's 28; `motion_frames_num=9`
   (`engine.py:175`) vs PRO's 5; `recondition` requiring `(9,H,W,3)` (`engine.py:243`);
   the shared-memory slot at `(batch, 24, H, W, 3)` (`worker.py:126`); turn length
   quantised to 24 (`calls.py:520`).
2. `[M]` **Batching bought ~5%.** The Sep-16 concurrency result — 5 streams at 15 FPS,
   ~77.36 aggregate FPS — is **LITE**, and LITE went 73 FPS at batch 1 → 77 FPS at batch 5.
   `[M]` GPU utilisation during PRO runs is **99.07%** mean, so concurrent streams
   time-slice rather than multiply.

Full-frame PRO's execution ceiling is therefore **~14.9 FPS at 4 steps, ~18.9 FPS at
2 steps**. ROI/crop reduction was ruled out by the user (body and background animation
must be preserved).

---

## 1. What landed

### Wave 0 — measurement and landing infrastructure (0 FPS; unblocks everything)

| Item | Change | Guard |
| --- | --- | --- |
| **Schedule table** | `flash_head/src/pipeline/schedules.py` (new, torch-free) replaces the `np.linspace` fallback in `flash_head_pipeline.py`. | `tests/test_sampling_schedule.py`, 10 tests `[CPU]` |
| **Source-drift guard** | `tests/test_source_manifest_drift.py` recomputes sha256 of every `source_sha256` key in the candidate run. | 4 tests `[CPU]` |
| **Stage-plan preflight** | `benchmarks/pro_quantization_v2_20260918/plan_preflight.py` (new, stdlib-only CLI). | `tests/test_plan_preflight.py`, 12 tests `[CPU]` |
| **Policy-schema contract** | `tests/test_policy_schema_contract.py` + `tests/test_cross_attention_policy.py`. | 106 tests `[CPU]` |
| **Test scaffolding** | `tests/conftest.py` — repo-root `sys.path`, `heavy` marker, dependency-probing collection. | suite runs on a machine with no torch |

**Schedule defect fixed.** `[CPU]` The old fallback computed
`np.linspace(1000, 1, sampling_steps)`. At `sampling_steps=3` that is raw
`[1000, 500.5, 1]`, which the shift transform maps to **t = {1000.0, 833.6109, 4.9801}**.
The third forward pass sat at **0.5% noise**, outside the distilled level set the student
was trained on, costing a full pass for nothing. Verified by executing the transform.

Post-shift levels at `shift=5.0` (all verified `[CPU]`):

| Steps | Before | After |
| ---: | --- | --- |
| 4 | `{1000, 937.5, 833.3333, 625, 0}` | unchanged |
| 3 | `{1000, 833.6109, 4.9801, 0}` ← defect | `{1000, 833.3333, 625, 0}` |
| 2 | `{1000, 833.3333, 0}` | `{1000, 625, 0}` |

The 2-step change makes its terminal non-zero level **625.0**, the level the 4-step
distilled schedule actually ends on. Untabulated step counts now raise instead of
interpolating off the level set.

**Source drift, confirmed.** `[CPU]` Four files differ from the candidate run's manifest:

```
benchmarks/pro_quantization_v2_20260918/profile.py
benchmarks/pro_quantization_v2_20260918/run.py
flash_head/src/pipeline/flash_head_pipeline.py
soulx_rtc/pro_vae_stage_backend.py
```

`flash_head/wan/modules/vae.py` still matches, so the retained FP16 engines will install.
But **the 11.7424 FPS baseline was measured on different source than this tree**, most
likely the Sep-19 `latency_scope` instrumentation, whose `gpu_validation.status` is
`"not_run"`. A re-baseline is mandatory before any comparison.

**Workspace arithmetic independently reproduced** `[CPU]` recomputation of `[M]` builds:

| Build | Shared arena | Summed across signatures |
| --- | ---: | ---: |
| `stage-bf16-build-r02` | **236.602 MiB** | 713.870 MiB |
| `stage-fp16-build-r01` (selected) | **709.102 MiB** | 1,739.905 MiB |
| `stage-int8-build-r02` (rejected) | 748.125 MiB | 2,071.510 MiB |

`[A]` The selected FP16 engines cost **+472.5 MiB** of shared arena over BF16, against
roughly **426 MiB** of observed device headroom. For a concurrency goal this is the
single largest identified recoverable-memory item, and it is a consequence of FP16
tactic/reformat selection, not of the model.

### Wave 1 — rebuild-free execution changes (no `vae.py` edit, no engine rebuild)

| Item | Change | Default |
| --- | --- | --- |
| **W1.1 cross-attention FP8 q/o** | Lifted the blanket rejection in `pro_quantization_v2.py`; added `cross_attn.q`/`o` targets. **k/v refused with a specific error.** | off; new policy file |
| **W1.3 `use_fast_accum`** | Threaded from an **optional** top-level policy key to the `torch._scaled_mm` call. | `False` — unchanged |
| **W1.4 lean delivery** | `pipeline.lean_delivery`: trim history frames before colour correction, cast to uint8 on device, finite-check pre-cast. | `False` |
| **H4 timing prints** | 7 per-window stdout lines behind `pipeline.verbose_timing`. | `False` |

> **Correction (2026-09-20, same day):** as first committed in `f370189`, this row
> was WRONG. The code trimmed *after* colour correction, so the claimed
> colour-correction saving did not exist; only the D2H reduction did. Found by
> adversarial review and fixed in `86d577e`, which moved the trim ahead of
> `postprocess.color_correction`. The row is accurate as of that commit.


**Why k/v are refused.** `CrossAttention.prepare_kv` (`flash_head_model.py:270-273`)
returns bare BF16 tensors once per window at M=288. There is nowhere in that contract to
carry an activation scale, and the GEMMs are far too small to pay for one. The refusal is
enforced at policy validation, at plan construction, and by a test.

**Why the early trim is safe.** `[CPU]` Verified by reading
`match_and_blend_colors_torch`: `source_mean`/`source_std` reduce over `dim=[2,3]` of a
`(B, T, H, W, C)` tensor — H and W only, keeping B and T. Every frame is corrected
independently, so `f(x)[n:] == f(x[n:])` exactly. `cond_frame` takes the **trailing**
`motion_frames_num`, which trimming the leading frames does not move.

**Every retained policy still validates** `[CPU]`, and `fast_accum` was deliberately
added to `allowed` but **not** to `required`, so all 19 recorded policies keep their
`policy_sha256`. Two new policy files were added rather than editing the selected
candidate in place:

- `policies/combined_fp16_sage_fp8_crossqo.json` — differs in exactly `{name, cross_attention_projections}`
- `policies/combined_fp16_sage_fp8_fastaccum.json` — differs in exactly `{name, fast_accum}`

---

## 2. Expected value — and why it is below the promotion gate

`[A]` Wave 1, all items at expected value: DiT −64 ms/window, remainder −16.25 ms/window.

```
21.290295 s  →  20.5680 s  →  12.155 FPS   (+3.5%)
at 50% of expected:            11.945 FPS   (+1.7%)
```

**This is below the ≥5% full-pipeline promotion gate, and that is expected.** Wave 1 must
be promoted as a bundle or the gate applied to the bundle. The same items must be
*ablated separately for quality* and *promoted together for speed*; those two requirements
pull in opposite directions and the run schedule below reflects it.

~~`[GAP]` `use_fast_accum` is unquantifiable without measurement.~~ **CLOSED 2026-09-21 by
measurement.** The second reading was correct: `CUBLASLT_MATMUL_DESC_FAST_ACCUM` is a **no-op**
on the SM89 `m16n8k32` path whose accumulator is architecturally f32. Interleaved A/B, 5 reps x
200 iterations with CUDA events, `torch._scaled_mm` E4M3 TN, at (M=6480, K=1536, N=1536) — the
**only** shape the flag reaches in stackedB, because the INT8 FFN took the other two:

| shape | fast_accum OFF | fast_accum ON | delta |
| --- | --- | --- | --- |
| (6480, 1536, 1536) — the only shape it reaches | 0.2284 ms / 133.9 TFLOP/s | 0.2271 ms / 134.6 TFLOP/s | **+0.6%** |
| (6480, 1536, 8960) | — | — | +11.7%, but **INT8 in stackedB** |
| (6480, 8960, 1536) | — | — | +4.7%, but **INT8 in stackedB** |

Across 6 linears x 60 block-forwards the flag is worth **0.47 ms/window** — 0.14 sigma on
`stage_seconds.dit`, roughly 7x below the resolution floor. **The ~200 ms/window reading is
dead.** The earlier 28.4 ms midpoint carried into planning was ~60x the truth.

---

## 3. What the 4070 must run, in order

Run `plan_preflight.py` first — it is free and fails in milliseconds.

**#0 — Re-baseline (mandatory, before anything is compared).** Current tree,
`policies/combined_fp16_sage_fp8.json`, `--latency-detail off`, same fixture and seed as
`extended-r01/recurrence-candidate`. Without it, every delta below is measured against a
number this tree has not been shown to produce.

**#1 — G9 quality ablation (promotion blocker, ranks above every speed run).** The
candidate cannot be promoted until the sync regression is attributed. Two retained arms
**disagree in sign**: `extended-r01` (1,500 frames, 295 windows) shows the candidate
*better* on own-crop (0.7404 → 0.7495, **+0.0091**) while `new-speech-correction-r01`
(45 windows) shows it worse (0.7629 → 0.7450, −0.0179). The larger arm is absent from
every current plan document. Report per-window deltas at **both lag 0 and per-variant
best lag**; a `flash2` arm is needed as an attention-off control, since both existing
levels quantise Q/K to INT8.

**#2 — Roofline microbenchmark (decides W1.3 and closes G4).** Sweep `torch._scaled_mm`
FP8-E4M3 with `use_fast_accum` False/True plus BF16 `torch.mm` at M=6480,
(K,N) ∈ {(1536,1536), (1536,8960), (8960,1536)}. Report TFLOP/s and the dispatched kernel
name. ~~`[GAP]` No retained artifact measures this GPU's achievable dense FP8 peak.~~
**RUN 2026-09-21. The answer is ~94% of a hard cap.** Max FP8 throughput observed on this part
under **any** configuration, fast_accum ON, is **141.8 TFLOP/s** — exactly the ~142
FP32-accumulate cap. **There is no ~284 TFLOP/s FP8 tier on consumer Ada.** Measured tier table
at M=6480:

| path | TFLOP/s (or TOPS) |
| --- | --- |
| BF16 `torch.mm` | 68.8 - 69.8 |
| FP8 `_scaled_mm`, fast_accum OFF | 128 - 138 |
| FP8 `_scaled_mm`, fast_accum ON | 134 - **141.8** |
| INT8 `_int_mm` | **187 - 232 TOPS** |

So the measured **131.4 TFLOP/s effective is ~93% of roofline — the FP8 lane has no headroom.**
The only open arithmetic tier left in the DiT is INT8 for the 6 square projections, measured
end-to-end at **-7.4 ms/window** for all six (-3.7 for the three that are safe; `q`/`k` must
**not** be converted, because SageAttention2 re-quantizes them to INT8 per-thread inside the
kernel — routing them through an INT8 GEMM first is double quantization of the same values).

**#3 — Latency trace (closes G1/G3, zero new code).** `--latency-detail stages` on the
candidate policy. Every per-kernel number in every current plan document describes the
**V2 reference**, not the candidate, and the candidate changed exactly the two things that
trace measures most. Also run a paired `--latency-detail off` control to measure the
instrumentation's own overhead: `pro_vae_stage_backend.py` now emits 5 spans per stage
invocation × 18 invocations per decode = **90 spans per decode**.

**#4 — Wave 1 bundle.** `--lean-delivery` plus `combined_fp16_sage_fp8_crossqo.json`,
established paired protocol: three fresh processes per seed/role, seeds 50/51/0/1,
Indian-man 1.50×, 320×576, 4 steps, shift 5, 2 motion latents, strength 1.0. Compare
`raw_rgb_sha256` against the non-lean run — it should be **identical**, since the op order
was preserved deliberately.

**#5 — Step-count arm, only after #1 clears.** `sampling_steps=2` now terminates on 625.0.
`[A]` DiT time is exactly linear in steps, so 4→2 is a 2× DiT gain: **8.474913 → 4.237457 s**.
Note `flash_head_pipeline.py:304` draws a fresh full randn per step, so **seeds are not
comparable across step counts** — compare distributions, not paired seeds.

### Accept/revert criteria

The CPU suite gates **correctness only, never promotion**. Every item needs a named
accept/revert criterion before it is pushed:

| Item | Accept | Revert |
| --- | --- | --- |
| W1.4 lean delivery | `raw_rgb_sha256` identical; remainder drops | any sha mismatch |
| W1.1 cross q/o | Δ `stage_seconds.dit` ≤ −30 ms/window | DiT unchanged or quality arm fails |
| ~~W1.3 fast_accum~~ | **RETIRED 2026-09-21 — measured no-op, 0.47 ms/window. Do not run this arm.** | — |
| 2-step | quality arm passes at best lag | any tooth/articulation regression |

---

## 4. Corrections to the published record

1. **`profile-cost-summary.json` is not wrong.** Its seven categories sum to exactly
   `5,349,836.576 µs`, matching the 269 `DeviceType.CUDA` rows in `aggregated.csv`
   — independently reproduced `[CPU]` with the stdlib `csv` module. Conv is **33.12%
   exclusive**; it is **41.06% including the layout kernels it launches**, because
   `aten::cudnn_convolution` is a CPU roll-up row. Both are correct statements of
   different quantities.
2. **BF16 engine workspace was omitted** from the Sep-19 plan (§1 above): 236.602 MiB,
   **472.5 MiB below** the selected FP16 engines.
3. **TensorRT fusion per se buys nothing.** `[M]` TRT BF16 vs compiled BF16 is
   0.987–1.034× — a wash. TRT FP16 vs TRT BF16 at identical coverage is **1.112–1.114×**.
   The entire fused-stage win is the FP16 tactic path, not fusion.
4. **The compiled-BF16 control is unstable across processes** `[M]`: 1.44785 / 1.52010 /
   1.63847 s for the same policy and latent — a **7.9% spread**, wider than the 5%
   promotion gate. Fixed-latent screening needs interleaved, order-rotated repeats in one
   process before its numbers mean anything.
5. **CUDA graphs are ruled out** `[M]`: GPU utilisation is 99.07% mean and kernel time is
   98.9% of generation wall time. There are no enqueue gaps to recover, despite ~8,150
   launches per window. They may still matter for p95 jitter.
6. **Channels-last already measured 1.103× whole-decoder at half the memory** `[M]`
   (`vae-channels_last.json`: 1.65963 → 1.50317 s, peak 2,174.06 MiB vs compiled's
   4,334.04 MiB). `[GAP]` It has never been combined with compile or with the TRT stages.
7. **PRO is already the CFG-distilled student** `[CPU]` (`flash_head_pipeline.py:110`,
   `:283-295`). There is no 2× available from removing guidance.
8. **Physical VRAM is unverified.** The source artifact records
   `physical_vram_class: "unverified in this run"`; only `visible_vram_mib: 12282` is
   evidenced. Earlier documents asserting "12 GB physical" overstate what the artifact
   supports.

---

## 5. Still open

- `[GAP]` No kernel trace of the actual candidate exists (**G1**) — the largest hole.
- `[GAP]` No per-layer timing inside any TRT engine (**G2**); `*.layers.json` carry layer
  type, shape, dtype and tactic id but **no time**.
- `[GAP]` The INT8 cliff at window 3–4 is unexplained. `[M]` DiT is identical to 0.1%, but
  the **unchanged BF16 motion encoder slowed 65%** (0.1533 → 0.2538 s/window), pointing at
  a global allocator/stream effect rather than INT8 convolution arithmetic. Device peaks
  were *not* higher than the accepted FP16 candidate.
- `[GAP]` PRO's actual resolution-scaling exponent. The two retained LITE datapoints
  disagree in direction (one sublinear, one superlinear) and neither is PRO.
- **Unresolved product question:** the pipeline generates 25-FPS content
  (`infer_params.yaml` `tgt_fps: 25`; `run.py:211` hard-rejects non-25-FPS fixtures), so
  "30 FPS" is **1.2× real-time**. `summarize_acceptance.py:76` counts missed deadlines
  against `28/25 = 1.12 s` and `:132-134` gates on aggregate `useful_fps` — **neither
  tests the 30-FPS streaming gate of `28/30 = 0.9333 s`**. A candidate can pass both
  published gates while missing every streaming deadline.


---

## 6. D2 — denoising steps 4 to 2 (commit 2/3)

**The largest single FPS lever available once ROI/crop reduction is excluded**, and
the highest-risk item in the plan. Nothing here is measured.

`[A]` DiT time is exactly linear in step count: 30 blocks x 4 steps = 120 block
executions per window, halving to 60.

| Configuration | s / 250 fr | FPS |
| --- | ---: | ---: |
| 4 steps `[M]` baseline | 21.290295 | **11.742** |
| 2 steps `[A]` projection | 17.052839 | **14.66** |

That is **+24.85%**. It is **not sufficient**: at 2 steps, 30 FPS would still
require a **4.74x** decoder against a best plausible execution gain of **1.518x**.

### What landed

| Flag | Default | Effect |
| --- | --- | --- |
| `--sampling-steps {1,2,3,4}` | `4` | Replaces the hardcoded literal at the `prepare_params` call and in `_fixed_profile`. |
| `--timestep-variant {shipped,distilled_aligned}` | `shipped` | `shipped` reproduces the original table exactly; `distilled_aligned` re-bases 2-step to terminate at 625.0. |
| `--skip-zero-weighted-noise` | off | Skips the terminal `randn` multiplied by exactly zero (414,720 elements/window). |

### The two schedule variants, and why both exist

`shipped` is the default because existing configurations depend on the original
behaviour: `flash_head/inference.py` sets `sample_steps = 20` for the CFG teacher,
and `soulx_rtc/engine.py:53` accepts `steps in (2, 4)`. For an untabulated count
`shipped` returns `None` and the caller runs the original `np.linspace` verbatim.

`distilled_aligned` is the corrected table. Post-shift levels at `shift=5`:

| Steps | `shipped` | `distilled_aligned` |
| ---: | --- | --- |
| 4 | `{1000, 937.5, 833.3333, 625, 0}` | identical |
| 3 | `{1000, 833.6109, 4.9801, 0}` ← defect | `{1000, 833.3333, 625, 0}` |
| 2 | `{1000, 833.3333, 0}` | `{1000, 625, 0}` |
| 20 | `np.linspace` fallback (teacher) | raises |

Choosing `distilled_aligned` is QUALITY-AFFECTING and never the default.

---

## 7. Six defects found by adversarial review, all fixed

The first draft of D2 was reviewed by 38 agents across four lenses. One was
reported CONFIRMED; **four more were marked "refuted" by the verifier and were
real on hand re-check.** All six are fixed in `86d577e` and regression-guarded by
`tests/test_shipped_behaviour_preserved.py`.

| # | Defect | Why it mattered |
| ---: | --- | --- |
| 1 | Wave 1 switches were instance-only attributes | `generate()` reads them unconditionally; `tests/test_pipeline_latency.py` builds the pipeline with `__new__`, so it raised `AttributeError`. **Invisible locally** because that module imports torch and is skipped on the dev Mac. Now class-level defaults. |
| 2 | The table raised on `sampling_steps=20` | Broke the CFG teacher entirely (`inference.py:36`). `shipped` now returns `None` and falls back. |
| 3 | The 2-step row was silently re-based | Changed `soulx_rtc.Engine(steps=2)`, a supported RTC config. Now opt-in. |
| 4 | `fast_accum` had no type check | `bool("false")` is `True`; a JSON string would silently enable fast-accumulate FP8 numerics. |
| 5 | `schedules.py` absent from `_sources()` | The table defining a run's timesteps was not hashed into its provenance. |
| 6 | `sweep.py` could not forward the new flags | Made the arm unmeasurable through the paired promotion harness. |

Plus the W1.4 trim-order correction recorded in section 1 above.

### Known limitation, not a defect

`sweep.py` forwards `--sampling-steps` to **both** roles, so it runs a paired
comparison *at* a fixed step count and cannot A/B 4-step against 2-step. That is
deliberate — an arm differing on one side only is not a controlled comparison —
but the D2 promotion sweep must therefore be assembled from direct `run.py`
calls. See the runbook, Step 5.

---

## 8. Test counts, and what they do not mean

**173 CPU tests pass** on an Apple Silicon Mac with no CUDA, no GPU and no torch
(pytest 9.1.1, CPython 3.13.14). `tests/conftest.py` skips modules whose
third-party imports are unavailable, including transitively through first-party
modules, and reports what it skipped.

That number gates **correctness only, never promotion**. None of these tests
measures speed, and several torch-dependent guards — including the one that
caught defect 1 — have never executed against these changes. The first real run
of `tests/test_pipeline_latency.py` since the Wave 1 switches were added will
happen on the 4070, at Step 0 of the runbook.
