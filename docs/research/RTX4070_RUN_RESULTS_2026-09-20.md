# RTX 4070 SUPER run results: mcPRO Wave 0/1 + D2 measured

**Audience:** the next engineer continuing the PRO 30-FPS work.
**Supersedes the projections in** `RTX4070_RUNBOOK_2026-09-20.md` **and**
`WAVE0_WAVE1_LANDED_2026-09-20.md` **with measurements.**

## Hardware and provenance

| Field | Value | Evidence |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER | `nvidia-smi --query-gpu=name` |
| Visible VRAM | 12,282 MiB | `nvidia-smi --query-gpu=memory.total` |
| Physical VRAM class | **unverified in this run** | not exposed by any source artifact |
| Driver | 595.84 | `nvidia-smi` |
| Torch / CUDA runtime | 2.7.1+cu128 / 12.8 | `torch.__version__`, `torch.version.cuda` |
| Commit | `51225f5` (tag `mcpro-latest`) | `git rev-parse` |

**Execution class of every number below: fresh local GPU inference**, single
250-frame run per cell, fixture `indian150-a`, policy
`combined_fp16_sage_fp8`, unless explicitly marked `[A]` arithmetic.
No historical results are reused. No static analysis is presented as measurement.

## Executive summary

1. **D2 (4→2 denoising steps) is real: +24.77% across four seeds.** The
   linearity assumption holds exactly — measured DiT ratio **2.009**.
2. **30 FPS is unreachable, and the binding constraint is the decoder, not DiT.**
   `vae_decode` alone is **131.6%** of the entire 30-FPS budget.
3. **D2's only real quality issue is a ~1-frame temporal shift.** At best lag it
   is statistically indistinguishable from the baseline; at lag 0 it is
   significantly worse (p = 0.006) purely because its sync peak moves later.
   **Visual inspection found no degradation** — see "Visual inspection" below.
4. **`--lean-delivery` must not be promoted**: it fails bit-exactness and buys
   +0.02%.
5. **Runbook Step 5 was not executable as written** — `review.py` refuses
   cross-step pairing. Fixed behind an opt-in flag.

## Measured results

Re-baseline was mandatory because source had drifted. It moved almost nothing:
published 11.7424 → measured **11.8511** (+0.93%), so the drift was inert.

| Arm | FPS | vs base | DiT (s) | Verdict |
| --- | --- | --- | --- | --- |
| Re-baseline, 4 steps | 11.8511 | — | 8.364 | BASE |
| **D2, 2 steps** | **14.8063** | **+24.94%** | 4.162 | speed accepted, quality contested |
| `--lean-delivery` | 11.8533 | +0.02% | 8.459 | **reject** |
| cross-attn FP8 q/o | 11.9906 | +1.18% | 8.105 | marginal miss |

Four-seed confirmation of D2 (seeds 50, 51, 0, 1):

| | 4-step | 2-step |
| --- | --- | --- |
| mean FPS | 11.7423 | 14.6510 |
| spread | 1.8% | 1.6% |
| **gain** | | **+24.77%** |

Run-to-run spread is ~1.6%. **The cross-attention arm's +1.18% FPS gain sits
inside that noise band**; only its DiT stage time (8.364 → 8.105 s, CUDA-event
timed) is trustworthy evidence for it.

### Stage decomposition (4-step baseline, 250 frames)

| Stage | Seconds | Share | vs 30-FPS budget (8.3333 s) |
| --- | --- | --- | --- |
| `vae_decode` | 10.965 | 52.0% | **131.6%** |
| `dit` | 8.364 | 39.6% | 100.4% |
| `motion_encode` | 1.376 | 6.5% | 16.5% |
| `audio` | 0.072 | 0.3% | 0.9% |

**This is the headline correction to the prior plan.** Earlier documents framed
DiT as the obstacle ("101.70% of budget"). Measured, DiT is 100.4% of budget but
the decoder is **131.6%** — larger than the whole budget by itself. With DiT,
motion_encode and audio all at zero the ceiling is **22.80 FPS `[A]`**.

### What 30 FPS would now require `[A]`

With D2 landed, non-decoder work is 4.162 + 1.376 + 0.073 = 5.611 s, leaving a
decoder budget of 2.722 s against a measured 10.965 s — a required **4.03×**
decoder speedup, against a best plausible execution gain of **1.518×**. The
shortfall is ~2.7×. (The prior 4.74× figure was computed before D2; D2 improves
it but does not close it.)

## Quality gate (Step 5)

Eight runs: {4-step, 2-step} × seeds {50, 51, 0, 1}, `--save-raw`.
45 SyncNet windows per run, 180 per arm pooled.

### Lip-sync, SyncNet

| Crop mode | Lag | base | cand | delta | Mann-Whitney p |
| --- | --- | --- | --- | --- | --- |
| own | best | 0.7236 | 0.7319 | **+0.0083** | 0.487 (ns) |
| shared | best | 0.7236 | 0.7019 | **−0.0217** | 0.198 (ns) |
| shared | 0 | 0.7117 | 0.6750 | **−0.0367** | **0.006** |

Shuffled-control mean ≈ 0.17, so the model is discriminating.

**At best lag the 2-step arm is statistically indistinguishable from baseline in
both crop modes.** That satisfies the runbook's literal acceptance.

**At lag 0 it is significantly worse.** The two readings are reconciled by a
temporal shift: the candidate's sync peak moves later. Score-weighted centroid
lag shifts **+0.088 frames** (own crop), and the argmax moves 0 → 1 in three of
four seeds. At 25 fps that is roughly one frame / 40 ms.

**This reproduces, and explains, the contradiction flagged in the handoff.** The
retained arms disagreed in sign (`extended-r01` +0.0091 own-crop, better;
`new-speech-correction-r01` −0.0179, worse) because own-crop-at-best-lag grants
the candidate two degrees of freedom — its own face boxes *and* its own lag —
which mask a shift that a controlled comparison exposes.

### Articulation / texture

`median_edge_ratio_on_both_open` (candidate ÷ baseline oral edge energy):

| seed | 50 | 51 | 0 | 1 | mean |
| --- | --- | --- | --- | --- | --- |
| ratio | 1.2428 | 1.3562 | 1.2970 | 1.0439 | **1.235** |

**Elevated in all four seeds.** The initial reading of this was that fewer
denoising steps leave high-frequency residue, i.e. an artifact signature.
**Visual inspection refuted that** — see below. `review.py` states this metric is
a diagnostic, not a certification, and in this case the metric alone would have
produced the wrong verdict.

### Visual inspection

Stills were compared **pose-matched** (frames selected by nearest `opening_norm`,
not by frame index — the two arms follow different trajectories, so equal indices
are different poses; an index-matched comparison is misleading and was discarded).

Findings across seeds 0 and 50 at openings 0.08 / 0.14 / 0.20:

- **No consistent degradation.** Some matched poses favour each arm. On seed 50
  the 2-step teeth are *cleaner and better defined* at all three openings, while
  the 4-step shows dark speckling at open=0.08. On seed 0 the result is mixed.
- The elevated edge energy corresponds in several poses to **more tooth
  definition**, not to artifacts.
- Full-frame quality — identity, skin, hair, background, lighting — is
  indistinguishable between arms.

**Temporal stability** (mean |frame[t] − frame[t−1]|, lower = less flicker):

| seed | full-frame 4-step | full-frame 2-step | mouth 4-step | mouth 2-step |
| --- | --- | --- | --- | --- |
| 50 | 2.718 | **2.418** | 13.837 | **13.254** |
| 51 | 2.660 | **2.578** | **13.341** | 13.753 |
| 0 | 2.535 | **2.438** | **12.767** | 13.407 |
| 1 | 2.680 | **2.413** | 13.719 | **13.095** |

**2-step flickers less at full-frame level in all four seeds**; the mouth region
is a wash (two seeds each way). There is no temporal-coherence penalty.

Artifacts for review:
`benchmarks/pro_30fps_20260920/visual-comparison/side-by-side-seed{0,50}.mp4`
and `mouth-zoom-seed{0,50}.mp4`, plus per-seed
`reviews/d2-seed*/comparison.mp4` and `mouth-comparison.mp4`.

### Verdict on D2

**Promotable subject to the lag shift being handled.** The quality objection
reduces to one thing: the output sits ~1 frame (40 ms) later against the audio,
which costs nothing at best lag but is a significant regression at lag 0
(p = 0.006). If the serving path applies a fixed A/V offset, the +24.77% is
available with no visual cost. If it cannot, expect measurably looser lip-sync.

The oral edge-energy elevation, on its own, is **not** grounds to revert — the
visual evidence contradicts the artifact hypothesis it suggested.

## Step 3 — `--lean-delivery`: the original analysis here was WRONG

**Corrected 2026-09-20, later the same day.** This section originally reported that
lean delivery corrupted 83.95% of pixels via autoregressive amplification. That
measurement was real but it was measuring the **wrong thing**: it compared
`lean-r01` against `rebaseline-r01`, which was the first run of the session against
a cold 334 MB `.torchinductor` cache.

Compared against a **warm** run of identical configuration, lean delivery is
bit-identical: `lean-r01` and `quality-steps4-seed50` both carry
`raw_rgb_sha256 = a8a6a405aa4f5d159a4d5992478d0129cac464ebee68837e99b851c8399cf28b`.
It is `rebaseline-r01` (`3b006313…`) that is the outlier.

Two consequences, both larger than the lean-delivery question itself:

1. **The pipeline is not bit-reproducible across a cold/warm compile-cache boundary**,
   so `raw_rgb_sha256` **cannot be used as an acceptance gate at all**. The Step-3
   criterion in `RTX4070_RUNBOOK_2026-09-20.md` is unsound as written — no arm can
   satisfy it, including a no-op.
2. The in-code claim at `flash_head/src/pipeline/flash_head_pipeline.py` that the
   surviving frames are "bit-identical" is **not** refuted by this run. Colour
   correction is per-frame independent (statistics reduce over H and W only) to
   ~1e-7, and the uint8 quantisation absorbs that.

**The practical verdict is unchanged: do not ship it.** It buys **+0.02%**, which is
nothing. But it was rejected for the wrong reason, and the reasoning is what matters
for the next arm that gets gated this way.

## Step 4 — cross-attention FP8 q/o, marginal miss

60 modules converted (30 blocks × q,o), **no k/v converted** — the safety
invariant holds. DiT 8.364 → 8.105 s = **28.8 ms/window saved** against a 30
ms/window bar and a 36 ms/window projection: ~80% of projection, just under the
bar.

> **Runbook erratum:** the Step 4 snippet computes `(baseline − candidate)` but
> accepts `<= -30`. Taken literally that passes only if the candidate is
> *slower*. Read the saving as positive.

## Defects and blockers found

1. **7 dead guard tests.** Wave 1 made `_pro_target_paths` in
   `soulx_rtc/pro_quantization_v2.py` unconditionally require `cross_attn.q/.o`,
   but the stub in `tests/test_pro_quantization_v2.py` is a bare
   `nn.Linear(32, 32)`. 12 passed at `mcpro-before`, 7 fail at `51225f5`.
   Production `CrossAttention` does define `q/k/v/o`, so the GPU path is
   unaffected — but those seven tests now fail at fixture construction and
   validate nothing. **Not yet fixed.**
2. **Fixture assets were missing.** `.gitignore` strips `**/*.png` and `**/*.wav`
   from this github-source branch, so `indian150-a` could not resolve and every
   GPU step failed instantly. Restored from branch
   `preserve/detached-work-20260919-c39f2a5`, each hash-verified against
   `fixtures.json`. **Any fresh clone of this branch hits the same wall.**
3. **Runbook Step 5 was not executable.** `review.py` compares a `_profile_key`
   that includes `steps`, so it refuses 4-step vs 2-step — the exact comparison
   Step 5 prescribes. The runbook also contradicts itself, warning "seeds are NOT
   comparable across step counts; compare distributions" while prescribing a
   paired tool.
4. **Five pre-existing RTC/call test failures**, all `examples/girl.png`
   missing — same gitignore cause, identical at `mcpro-before`.
5. **Four environment-dependent test failures**: two assert
   `"torch" not in sys.modules` in-process (needs a subprocess); two assume a
   weightless dev machine and see `drift` instead of `unavailable-locally`.

## Changes made to the tree

Only one file modified: `benchmarks/pro_quantization_v2_20260918/review.py`
— added opt-in `--allow-steps-drift`. Default behaviour is unchanged and still
refuses cross-step pairing; every other profile field stays enforced; the waiver
and both step counts are recorded in `review.json` under `profile_drift`.

Untracked outputs: `benchmarks/pro_30fps_20260920/`.

## Recommended next actions

1. **Decide on the ~1-frame shift** — this is now the only thing gating D2. If
   the serving path can apply a fixed 40 ms A/V offset, +24.77% is available
   with no visual cost.
2. **Confirm the visual call in motion** by watching
   `visual-comparison/mouth-zoom-seed{0,50}.mp4`. The stills and flicker metrics
   both say quality is fine; a human should sign that off before promotion.
3. **Fix the 7 dead guard tests** (give the stub a `cross_attn` with `q/k/v/o`).
4. **Point all remaining effort at `vae_decode`.** It is 52% of runtime and
   131.6% of the 30-FPS budget by itself. No amount of DiT work reaches 30 FPS.
5. **For aggregate/concurrent FPS**: the GPU ran at 100% utilisation throughout.
   Concurrency cannot multiply throughput on a saturated device — this is the
   same wall LITE batching hit (+5%). Aggregate FPS ≈ generated FPS, so the
   decoder is the constraint for the concurrency goal too. Reaching many
   concurrent streams needs either the decoder solved or more devices.
