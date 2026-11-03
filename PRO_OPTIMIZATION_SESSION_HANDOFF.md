# SoulX FlashHead PRO transformer optimization: session handoff (cloud → MacBook)

**Hardware for every new measurement in this document: no GPU.** This session ran on an **Intel Xeon @ 2.80 GHz Linux cloud container** with 4 vCPU (AVX-512), 15.7 GiB RAM and no GPU (`nvidia-smi` absent, no Metal or Core ML runtime). It used PyTorch 2.7.0+cpu and coremltools 9.0, which can convert but cannot predict on Linux. Evidence: `/proc/cpuinfo`, `/proc/meminfo`, and the `environment` block in every JSON this work wrote. The dates were 2026-11-02 and 2026-11-03 (UTC). The co-resident load was unisolated: two renders, or a render plus the fit, often shared the 4 cores, so CPU timings are indicative only.

Numbers quoted from earlier work are **reused historical results**, and each is labelled with its hardware: Apple M5 Pro (24 GiB unified memory), iPhone 12 Pro (A14), or NVIDIA GeForce RTX 4070 SUPER. This session ran **no Apple GPU, Neural Engine or iPhone measurement**. Every Apple speed effect below is a static operation count or a hypothesis, not a measured FPS.

Branch: **`claude/soulxflashhead-pro-optimization-7cjfod`**. It is based on `codex/ios-coreml-inference` at `43cb763` and pushed to `origin` (`AhmadAFS1/SoulX-FlashHead`). Commits are listed in [§2](#2-branch-and-commits).

---

## 0. Status at handoff (2026-11-03 ~01:00 UTC)

| Item | State |
|---|---|
| CPU reproduction of the iOS PRO window loop, with Core ML INT8/FP16 emulation | **Done.** Validated against recorded Core ML errors ([§5.1](#51-the-cpu-emulation-matches-the-shipped-core-ml-error-structure)) |
| Exact cross-attention output fold: −3.0% transformer MACs, unchanged Core ML partition I/O | **Done.** Exact in FP32 and converts on Linux. Quality renders are in progress |
| Calibrated FFN hidden-unit reduction (OBS + least-squares refit + per-timestep bias) | **Done.** Mostly a negative result: the PRO FFNs are dense. The adaptive variant `ffnk2` gives −1.5% MACs ([§5.4](#54-ffn-reduction-the-pro-ffns-are-dense-negative-result-for-most-blocks)) |
| INT4 FFN weights | **Rejected** (about 100× the INT8 error) ([§5.6](#56-lower-bit-weights)) |
| FP16-everywhere (Neural Engine eligibility) | **Probed only.** The emulation cannot represent Core ML's FP16 kernels; this needs a Mac experiment ([§5.7](#57-fp16--neural-engine-probe-cpu-emulation-only)) |
| A/B videos | Baseline video delivered. The teacher-forced FP32 floor and the negative control are finished. **15 renders were still queued** when this was written ([§6](#6-what-was-still-running-and-how-to-finish-it)) |
| Device validation (M5 Pro / iPhone) | **Not started.** It needs your Mac ([§7](#7-what-to-do-on-the-macbook)) |

**Bottom line so far.** Training-free, quality-neutral transformer savings at pro320 are about **3–5%** of MACs: the exact fold (−3.0%) plus small FFN cuts in blocks 0–3 and 29 (−1.5%). The PRO FFNs in blocks 5–28 carry no redundant units. Removing even the least useful 6.25% adds 5–22× the INT8 error the shipped model already carries. Large wins now require device-level work (Neural Engine eligibility, kernels or layouts, measured on the Mac or iPhone) or training (a smaller or fewer-step student). The new CPU A/B harness is built to evaluate such candidates before they reach a device.

---

## 1. What was asked and what happened

1. **Request.** Continue the PRO quantized-model work from `codex/ios-coreml-inference` and its `PRO_QUANTIZED_MODEL_REPORT.md`. Add more PRO transformer optimizations for iOS. Then render videos so that we can both check that quality did not drop.
2. **Branch access.** At first the codex branch existed only on your Mac (`~/.codex/worktrees/ios-coreml/...`). Remote Control was suggested. Note that a folder inside a hidden directory such as `~/.codex` cannot be shared through Remote Control. You then pushed the branch, and this session continued from `43cb763`.
3. **Constraint.** The cloud container had no GPU and no Core ML runtime. So the work was designed around (a) a faithful CPU reproduction of the iOS runtime with a Core ML numerics emulation, (b) optimizations that keep the Swift-facing model interface unchanged, so they drop into the existing app, and (c) CPU A/B videos plus metrics, with device speed left to the Mac.
4. **Push problem.** For most of the session `git push` returned HTTP 403 ("Claude doesn't have GitHub access ... for your organization"). A git bundle was handed over as a workaround. Access later started working, and everything is now pushed.

---

## 2. Branch and commits

Get it on the Mac:

```sh
cd <your SoulXFlashHead clone>
git fetch origin claude/soulxflashhead-pro-optimization-7cjfod
git switch claude/soulxflashhead-pro-optimization-7cjfod   # or: git worktree add ../SoulXFlashHead-claude claude/soulxflashhead-pro-optimization-7cjfod
```

Commits on top of `43cb763`, oldest first:

| Commit | Summary |
|---|---|
| `c7b6886` | CPU A/B renderer; exact cross-attention fold; FFN reduction tooling; emulation fidelity report |
| `b6b3d98` | Teacher-forced motion and per-window latent metrics |
| `37956b8` | Review fixes: FFN ratio names, validator scores variants against the same variant, resume checks |
| `8c8b545` | **Refit fix**: damping regularises toward the original W2 instead of shrinking it; hardening |
| `5f78cb5` | Thread count passed through the render queue |
| `062469d` | Vectorised, bit-identical port of Core ML Tools' INT8 quantizer; faster fit defaults |
| `713823b` | FP32 OBS and refit linear algebra (the Gram matrix stays FP64) |
| `5bcadd8` | OBS downdate batch 256 (the update is memory-bound) |
| `3258deb` | Adaptive per-layer FFN reduction (`ffnkK`) |
| `6ad304e` | Stale-fit detection on export resume, in validation and in the queue |
| `dccccbc` | Renderer memory halved (meta-device load, FP16 weight storage); bit-identical to before |
| `d240791` | Ignore the git bundle hand-off folder |
| `8af50b0` | Per-block FFN reduction evidence: `apple/validation/cpu_ffn_reduction_fit_pro320.json` |
| `4464253` | A/B summary tool (metrics against floors, comparison videos) |
| `0993b1d` | Load lock and `malloc_trim` (out-of-memory fix); fixture builder; `ffnmap` variant |
| (this commit) | This handoff document; Makefile `VARIANT`/`setup-cpu-ab`; bundle manifest records the variant; renderer clones checkpoint tensors (9 → 3.4 GB per render) |

---

## 3. Mac setup for this branch

Use the existing Apple environment, plus the extra CPU A/B packages:

```sh
make -f apple/Makefile setup-cpu-ab        # setup + apple/requirements-cpu-ab.txt (mediapipe, scikit-image, ...)
make -f apple/Makefile fetch               # pinned PRO, Wav2Vec2, Wan VAE and taew2_1 (hash-checked)
.venv-coreml/bin/python apple/scripts/make_cpu_ab_fixtures.py   # rebuilds the A/B audio; prints "ok" per pinned hash
make -f apple/Makefile test                # 9 earlier tests + 16 new tests (Core ML prediction tests also run on macOS)
```

The fixtures are 16 kHz mono PCM16. The calibration clips are used only to fit the FFN reduction, and the evaluation clips are held out:

| File | Content | SHA-256 |
|---|---|---|
| `cal_sichuan_0_13s.wav` | Official example podcast, 0–13.44 s (calibration) | `ded4834f…` |
| `cal_english_utt4_13s.wav` | LibriSpeech dummy utterance 4, first 13.44 s (calibration) | `44f09703…` |
| `eval_english_plosive.wav` | Utterance 38 ("…BIG GONG AND POUNDED…") + 0.5 s silence + utterance 24, 9.48 s | `7294437d…` |
| `eval_sichuan_40s.wav` | Podcast 40.00–49.48 s (held out) | `d0dc5900…` |

---

## 4. What was built (all under `apple/`)

| File | Purpose |
|---|---|
| `flashhead_coreml/cpu_runtime.py` | **CPU reproduction of `SoulXEngine.swift`** for the fixed avatar caches. It reproduces the Swift SplitMix64/Box–Muller noise bit for bit (380,160 draws per window), the `[1000,625,0]` re-noising update, motion restore (1 slot in the first window, 2 afterwards), the window-mode `taew2_1` decode (33 frames, frames 5–32 shown), Lab colour transfer with the ICC-corrected portrait statistics, re-encoding of frames 28–32 into motion, the 8 s audio history with Swift's window centres, and the truncating uint8 conversion. It has two modes. `fp32` is the portable reference. `coreml-int8` emulates the shipped bundle: Core ML Tools' per-channel INT8 weights, FP16 tensors between ops, FP32 layer/RMS norms and SDPA, and FP16 I/O at the ten partition boundaries. Other features: `--force-motion` teacher forcing; `--variant`; loads serialised by a lock (OOM fix). Output is `frames.npy`, `latents.npy`, `video.mp4` (with audio) and `run.json` (with hardware labels). |
| `flashhead_coreml/compare_emulation.py` | Rebuilds `validate_dit`'s seeded per-partition cases and compares the emulation error with the Core ML CPU error recorded on the M5 Pro. |
| `flashhead_coreml/dit.py` (changed) | `CrossAttention.fold_output_projection()` and `Blocks.fold_cross_output()` implement the **exact fold**. `CrossAttentionWeights` is an FP32 island for the explicit probabilities. `DiTBlock.ffn_step_bias` is a per-timestep FFN bias selected like the time modulation. Default behaviour is unchanged (the upstream parity tests pass). |
| `flashhead_coreml/export_dit.py` (changed) | Adds `--variant`. The FP32 selector also covers `CrossAttentionWeights`. The folded `[heads,128,dim]` operand gets INT8 `per_block [0,0,1]`, which reproduces the original W_o row scales exactly. Resume re-exports a partition when its FFN selection or fit files change. The manifest records the variant transforms with fit hashes. |
| `flashhead_coreml/validate_dit.py` (changed) | Scores variant exports against the same FP32 variant, after verifying fit hashes and widths. |
| `flashhead_coreml/ffn_pruning.py` | `capture` saves FFN inputs (a token subsample) from FP32 renders. `fit` builds a centred two-pass Gram per timestep, runs batched OBS removal of whole hidden units, does a damped least-squares refit toward the original W2, adds per-timestep compensating biases, and reports held-out error next to the dense INT8 error. It also provides `reduce_partition` and the variants `ffnNN`, `ffnkK` and `ffnmapL=F,...`. |
| `flashhead_coreml/variants.py` | Variant names: `baseline`, `fold`, `ffn25`, `ffnk2`, `ffnmap0=0.5,1=0.125`, combinable with `+` (for example `ffnk2+fold`). |
| `flashhead_coreml/quality.py` | Frame PSNR/SSIM, mouth-ROI PSNR and temporal-delta error. MediaPipe FaceMesh lip opening correlation, mouth-centre distance and oral edge-energy ratio (the historical review definitions). Colour shift, lip-audio correlation, per-window latent NRMSE, and labelled side-by-side videos with a ×8 difference panel. |
| `flashhead_coreml/ab_summary.py` | Runs every arm against the INT8 baseline and the floors, and writes the comparison videos. |
| `flashhead_coreml/bench_cpu_blocks.py` | CPU-only per-layer timing, baseline versus variant, plus static MACs. |
| `scripts/run-cpu-ab.sh` | Resumable render queue. Options: `SEED`, `FORCE_FROM`, `TAG`, `THREADS`. It refuses stale arms. |
| `scripts/make_cpu_ab_fixtures.py`, `requirements-cpu-ab.txt` | Reproducible inputs and the extra packages. |
| `tests/test_cpu_runtime.py`, `tests/test_export_variants.py` | 16 tests, among them: the emulation equals `DiTBlock` in FP32; the fold is exact; the fast INT8 port is bit-identical to coremltools' quantizer and to a real `linear_quantize_weights` model; the noise matches a scalar Swift transcription; OBS/refit least-squares properties; a folded and reduced partition converts with FP32 softmax and exact INT8 operand scales. |

The Swift runtime is **unchanged**. Every variant keeps each partition's `hidden`/`context`/`timestep` → `hidden_out` contract, so a variant bundle drops into the existing app.

---

## 5. Results so far

All of these are **CPU-only PyTorch on the Intel Xeon container (no GPU)** unless a row says otherwise.

### 5.1 The CPU emulation matches the shipped Core ML error structure

`apple/validation/cpu_emulation_fidelity_pro320.json` compares emulation error with error recorded on the **Apple M5 Pro using Core ML CPU_ONLY** (reused, `apple/validation/pro320_dit.json`). Both use the same seeded inputs and are measured against FP32.

| Partition | Emulation NRMSE (t=1000 / 625) | Recorded Core ML CPU (t=1000 / 625) |
|---|---|---|
| blocks_00 | 0.44% / 0.56% | 0.75% / 1.59% |
| blocks_03 | 1.79% / 1.86% | 2.03% / 2.20% |
| blocks_05 | 2.57% / 3.10% | 2.94% / 3.60% |
| blocks_06 (worst) | **3.66%** / 1.70% | **4.68%** / 2.57% |
| blocks_09 | 1.22% / 1.37% | 1.52% / 1.69% |

Across all 20 cases the emulation reaches 0.35–0.88× of the recorded error (median 0.86×), with the same pattern across partitions. It is therefore a **slight lower bound**. The tiny VAE runs in FP32 with FP16 I/O, whereas the bundle computes it in FP16. Both arms of every A/B share that VAE, so comparisons between transformer variants are unaffected.

Bit-exactness checks:
- The fast INT8 port equals coremltools' quantizer, including all-zero channels.
- The lean, FP16-storage renderer reproduced an earlier baseline window bit for bit.

### 5.2 Static cost per pro320 window

These are static counts, not measured speed. The baseline is 5.768 TMAC per window, matching the report.

| Variant | Transformer MACs/window | Reduction | Partition interface |
|---|---:|---:|---|
| `baseline` | 5.768 T | — | — |
| `fold` (exact) | 5.592 T | **−3.04%** | unchanged |
| `ffnk2+fold` | ≈5.50 T | **−4.6%** | unchanged |
| `ffn12.5+fold` (negative control) | 5.183 T | −10.1% | unchanged, but quality fails ([§5.5](#55-teacher-forced-floors-and-the-negative-control)) |

How the fold works: each latent frame's 220 cross-attention queries see only 32 audio keys. So `o(concat_h P_h v_h) + b = Σ_h P_h (v_h W_o[:,h]ᵀ) + b` replaces a 1980×1536×1536 projection (4.67 GMAC) with 0.68 + 1.17 GMAC per block-step. The probabilities stay FP32. Under the mixed policy the two contractions run in FP16. That is exact in real arithmetic, but rounding order changes, so the fold is not bit-exact on device.

### 5.3 FFN-unit removal method

For each block:

1. Capture FFN inputs from 24 FP32 calibration windows (both avatars, seed 1234, half of the tokens; 42,768 fit tokens and 4,752 held-out tokens per timestep).
2. Compute `a = gelu(W1 f + b1)` and its Gram matrix, centred per timestep.
3. Remove whole hidden units greedily by OBS score, `‖W2[:,j]‖² / [H⁻¹]_jj`, with H = G + λI and λ = 1% of the mean diagonal, in batches of 256.
4. Refit the kept W2 columns by damped least squares toward W2.
5. Restore each timestep's mean with one bias per timestep.

Evaluation uses held-out error relative to the FFN-output variance. `int8_dense` is the error the **shipped** INT8 per-channel weights already add to the dense FFN.

### 5.4 FFN reduction: the PRO FFNs are dense (negative result for most blocks)

Each cell below is held-out error after removal **plus INT8**, as a multiple of that block's dense INT8 error. Full data: `apple/validation/cpu_ffn_reduction_fit_pro320.json`.

| Block | INT8 error | 6.25% | 12.5% | 25% | 50% |
|---:|---:|---:|---:|---:|---:|
| 0 | 7.0e-5 | 0.9× | 0.7× | 0.9× | 1.6× |
| 1 | 4.0e-5 | 1.0× | 1.7× | 4.5× | 18.5× |
| 2 | 7.0e-5 | 1.4× | 3.4× | 12.4× | 52.6× |
| 3 | 1.2e-4 | 1.8× | 3.8× | 11.8× | 46.1× |
| 4 | 1.1e-4 | 2.7× | 6.9× | 21.4× | 77.5× |
| 5–12 | 1.1–1.6e-4 | 5.6–15.4× | 15.5–38.9× | 45.8–105× | 156–328× |
| 13–18 | 1.2–1.8e-4 | 16.9–22.5× | 41.2–56.1× | 105–149× | 313–438× |
| 19–28 | 1.2–1.5e-4 | 7.2–14.5× | 17.2–37.1× | 44.8–99.9× | 138–304× |
| 29 | 7.0e-5 | 1.7× | 4.1× | 12.4× | 46.3× |

Adaptive selections, where each layer gets the largest removal within K× its INT8 error:

| Variant | Layers reduced | FFN units removed | Transformer MACs |
|---|---|---:|---:|
| `ffnk1.5` | 0: 37.5%, 1: 6.25%, 2: 6.25% | 1.7% | −0.9% |
| **`ffnk2`** | 0: 50%, 1: 12.5%, 2/3/29: 6.25% | 2.7% | **−1.5%** |
| `ffnk3` | + layer 4: 6.25% | 2.9% | −1.7% |
| `ffnk5` | 0: 50%, 1: 18.75%, 2/3/29: 12.5%, 4: 6.25% | 3.8% | −2.1% |

For a deterministic export that does not depend on a re-fit's borderline decisions, use the explicit map: `ffnmap0=0.5,1=0.125,2=0.0625,3=0.0625,29=0.0625+fold`.

Two method bugs were found by review and fixed before these numbers were produced:
- The original refit shrank kept weights toward zero.
- The batched quantizer was not yet bit-identical.

### 5.5 Teacher-forced floors and the negative control

Setup: the girl avatar on `eval_english_plosive`, seed 42, nine windows. Each window reuses the INT8 baseline's motion latents, so differences cannot compound across windows.

| Arm compared with the INT8 baseline | Latent NRMSE per window (%) | Mean | Median frame PSNR per window (dB) |
|---|---|---:|---|
| **FP32 reference** (quantization floor) | 2.53, 1.42, 2.88, 1.41, 1.45, 1.62, 1.08, 1.46, 2.46 | **1.81%** | 45.6, 47.1, 45.2, 47.8, 46.5, 45.8, 47.7, 43.6, 41.0 |
| Uniform FFN −12.5% + fold (**rejected**) | 12.53, 5.54, 7.57, 4.27, 4.95, 4.16, 1.89, 3.88, 7.59 | **5.82%** | 33.6, 37.0, 33.7, 38.8, 39.1, 37.9, 43.5, 36.2, 32.6 |

The shipped INT8 model already differs from FP32 by about 1.8%. A uniform 12.5% FFN cut drifts 3.2× further, which agrees with the per-block sensitivities above. Results for `fold` and `ffnk2+fold` were still rendering ([§6](#6-what-was-still-running-and-how-to-finish-it)). The acceptance rule: a variant should sit inside the FP32 floor, and its lip, teeth and colour metrics should stay within the seed and FP32 floors in the free-running videos.

### 5.6 Lower-bit weights

The table gives held-out FFN-output error relative to signal energy, for blocks 1, 6 and 20.

| Scheme | Block 1 | Block 6 | Block 20 |
|---|---:|---:|---:|
| INT8 per-channel (shipped) | 3.5e-5 | 1.08e-4 | 1.37e-4 |
| **INT8, one scale per 32 inputs** | 1.0e-5 | 3.3e-5 | 4.3e-5 |
| INT4, per 32 inputs | 3.3e-3 | 1.06e-2 | 1.39e-2 |
| INT4, per 16 inputs | 2.4e-3 | 8.3e-3 | 1.05e-2 |

- INT4 weight-only quantization costs about 100× the INT8 error. It is **rejected** without calibration-aware training such as GPTQ, followed by video checks.
- Per-block INT8 (scale per 32) cuts the shipped quantization error about 3× for about 6% more weight bytes. That is a quality lever, not a speed one. It could also create headroom for FFN removal. It is not yet implemented in the exporter.

### 5.7 FP16 / Neural Engine probe (CPU emulation only)

This ran on `blocks_06` at t=1000 with validate_dit's seeded inputs:
- Mixed policy (shipped): 3.66%.
- Every op in FP16: 440%. The only cause is **LayerNorm overflow**: residual |x| reaches about 1,100, so x² overflows FP16 (65,504).
- FP16 RMSNorm alone and FP16 SDPA alone each stay at about 3.66%.
- Pre-scaling the norm inputs by 1/64 gives 3.53%. This is exact because LayerNorm is scale-invariant up to eps.

However, Core ML's real full-FP16 conversion gave 10–12% on the Mac, not 440%. Apple's FP16 kernels therefore behave differently from this emulation, especially SDPA accumulation. The Neural Engine route must be tested on the Mac ([§7.4](#74-neural-engine-experiment-highest-potential-unverified)).

### 5.8 Videos delivered in this session

- `artifacts/cpu-validation/renders/coreml-int8-baseline/girl-eval_english_plosive/video.mp4`: shipped INT8 PRO, emulated on CPU, girl avatar, 9.48 s of held-out English, seed 42. It is stable across the 8 s history rollover. CPU-only Intel Xeon render; it took 727 s.
- Comparison videos (FP32 | INT8 baseline | fold | fold+FFN | ×8 difference, and the negative control) are produced by `ab_summary.py` once the queued renders finish.

---

## 6. What was still running, and how to finish it

When this was written, the cloud session was running two workers (`artifacts/cpu-validation/worker.sh`, job lists `jobs1.txt` and `jobs2.txt`, two threads each). Finished arms are `coreml-int8-baseline/girl-eval_english_plosive`, `fp32-baseline-forced/...` and `coreml-int8-ffn12.5_fold-forced/...`. Remaining:

```
INT8 baseline: indian/sichuan, indian/english, girl/sichuan (three re-renders after an out-of-memory kill)
teacher-forced: fold, ffnk2+fold                         (girl/english)
free-running:   fold (4 clips), ffnk2+fold (girl/english, indian/sichuan), ffn12.5+fold (girl/english),
                fp32 baseline (girl/english, indian/sichuan), INT8 baseline seed 43 (girl/english)
```

If the cloud session finishes, its results will be pushed as `apple/validation/cpu_ab_*.json` along with the videos. To reproduce or finish on the Mac (a CPU run; the M5 Pro CPU should be faster than this container):

```sh
I=artifacts/cpu-validation/inputs; R=artifacts/cpu-validation/renders
A=girl:$I/eval_english_plosive.wav; B=indian:$I/eval_sichuan_40s.wav
C=indian:$I/eval_english_plosive.wav; D=girl:$I/eval_sichuan_40s.wav
apple/scripts/run-cpu-ab.sh $R coreml-int8 baseline $A $B $C $D
TAG=forced FORCE_FROM=$R/coreml-int8-baseline apple/scripts/run-cpu-ab.sh $R fp32 baseline $A
TAG=forced FORCE_FROM=$R/coreml-int8-baseline apple/scripts/run-cpu-ab.sh $R coreml-int8 fold $A
apple/scripts/run-cpu-ab.sh $R coreml-int8 fold $A $B $C $D
apple/scripts/run-cpu-ab.sh $R fp32 baseline $A $B
TAG=seed43 SEED=43 apple/scripts/run-cpu-ab.sh $R coreml-int8 baseline $A
# FFN variants need the fit files first (see 7.3), then e.g.:
apple/scripts/run-cpu-ab.sh $R coreml-int8 'ffnmap0=0.5,1=0.125,2=0.0625,3=0.0625,29=0.0625+fold' $A $B
PYTHONPATH=apple .venv-coreml/bin/python -m flashhead_coreml.ab_summary --output artifacts/cpu-validation/ab.json --videos artifacts/cpu-validation/videos
```

Each 9-window clip takes about 7.5 min with 4 threads alone on this Xeon, or about 12 min with 2 threads beside another render. Run at most two renders at once with 16 GB of RAM. The load lock already prevents overlapping load peaks.

---

## 7. What to do on the MacBook

### 7.1 Export the exact-fold bundle and measure it against the current bundle

```sh
# Reuse the already-exported support models to skip the slow Wan reference export:
mkdir -p artifacts/coreml/pro320-fold
cp -R artifacts/coreml/pro320/{audio,reference,decoder,encoder}.mlpackage artifacts/coreml/pro320-fold/
PYTHONPATH=apple .venv-coreml/bin/python -m flashhead_coreml.export_dit --height 320 --width 176 \
  --output artifacts/coreml/pro320-fold --variant fold --resume
PYTHONPATH=apple .venv-coreml/bin/python -m flashhead_coreml.validate_dit --checkpoint models/SoulX-FlashHead-1_3B/Model_Pro \
  --models artifacts/coreml/pro320-fold --height 320 --width 176          # NRMSE gate vs FP32, per partition
make -f apple/Makefile bundle OUTPUT=artifacts/coreml/pro320-fold DEVICE_OUTPUT=artifacts/coreml/device/pro320-fold
```

Then run the report's matched comparison (§8 there) with `--models artifacts/coreml/device/pro320-fold` against `.../device/pro320`. Use the same avatar cache, speech, seed and `--compute all`, with `--prefetch-audio --prefetch-blocks --model-timings --save-frames`, then `compare_frames.py`. What to expect:
- Partition prediction time should drop by about 3% if the M5 Pro GPU tracks MACs.
- Frames will not be bit-identical, because FP16 contraction order changes. Their PSNR should be in the high-40s dB, like the FP32 floor in §5.5.
- If the Mac does not get faster, the fold is not worth shipping. Small matmuls may map poorly, and that is a legitimate outcome.

### 7.2 Optional: check the CPU emulation against the real Mac Core ML

Render one clip on the Mac with `cpu_runtime --mode coreml-int8` and the same clip with `run_mac_smoke.sh --compute cpu` (CPU_ONLY Core ML), same seed. The noise stream is bit-compatible with Swift. Remaining differences come from CoreGraphics versus Pillow portrait resampling (colour statistics only) and from Core ML kernel numerics. A small frame difference here measures the emulation's fidelity directly.

### 7.3 FFN variants need a calibration fit, run once and kept local

The fit files are about 7 GB and are not in git. On the Mac:

```sh
PYTHONPATH=apple .venv-coreml/bin/python -m flashhead_coreml.ffn_pruning capture \
  --clip girl:artifacts/cpu-validation/inputs/cal_sichuan_0_13s.wav \
  --clip indian:artifacts/cpu-validation/inputs/cal_english_utt4_13s.wav      # ~20 min on this Xeon (24 FP32 windows)
PYTHONPATH=apple .venv-coreml/bin/python -m flashhead_coreml.ffn_pruning fit \
  --blocks 0,1,2,3,29        # only the layers the explicit map uses; the full 30-block fit took ~34 min here
```

Then export with `--variant 'ffnmap0=0.5,1=0.125,2=0.0625,3=0.0625,29=0.0625+fold'`, or use `ffnk2+fold` after a full fit. Validate with `validate_dit`; it checks fit hashes. Given −1.5% MACs for some complexity, ship this only if the Mac measurement shows the gain is real.

### 7.4 Neural Engine experiment (highest potential, unverified)

The A14 Neural Engine (about 11 TOPS) is idle for the DiT, because the FP32 norm and SDPA islands force GPU or CPU placement. Suggested Mac experiment:
1. Export one partition with `--precision float16`, with and without LayerNorm/RMSNorm input pre-scaling (divide by 64; exact by scale invariance; add this as an exporter flag).
2. Run `validate_dit` with CPU_ONLY and with `.all`.
3. Inspect the compute plan to see whether the partition lands on the Neural Engine.

If a pre-scaled FP16 partition passes the 5% gate and places on the Neural Engine, time it on the iPhone. That is the most promising route to a several-fold A14 speed-up. This session has no evidence that it will work.

### 7.5 iPhone

Follow the report's §7 row 1: an updated A14 baseline, sequential versus overlapped execution, then a variant bundle if the Mac shows a gain.

---

## 8. Lessons and gotchas from this session

- **Closed-loop videos re-roll.** Trajectories diverge after small numeric changes, as the historical CUDA notes already showed. Use teacher-forced runs for per-window fidelity, and lip/teeth/colour metrics for free-running clips. Do not use raw PSNR after window 0.
- **Memory.** A naive full-model CPU load needed about 11 GB at peak. `safetensors` tensors share one memory map of the 6 GB checkpoint. Writing emulated INT8 values into them made those pages private, and any surviving small tensor (norm weights, biases) kept the whole map resident: about 9 GB per render, steady state. The renderer now clones loaded tensors, loads on the meta device, stores emulated weights as FP16, serialises loads with a lock and trims freed memory. The result is 3.4 GB steady state per render. The out-of-memory killer took six renders before this was fixed.
- **CPU oversubscription.** Six PyTorch threads on four cores made both jobs 2–3× slower. Keep the total thread count at or below the core count.
- **`pkill -f` with a pattern that appears in your own command line kills your own shell.** Use explicit PIDs.
- **Remote Control** cannot serve folders inside a hidden directory under home (for example `~/.codex/...`). Use a worktree elsewhere, such as `~/SoulXFlashHead-claude`.
- **GitHub push from cloud sessions** needs the Claude GitHub App on the repository, or a refreshed GitHub link. A `git bundle` is a quick workaround.
- **Portrait ICC.** `girl/portrait.png` embeds Display P3. CoreGraphics colour-matches it, so the CPU renderer converts it to sRGB before computing the colour statistics.

## 9. Ranked next steps

1. **Mac:** export `fold` and measure it against the current bundle (§7.1). Ship it if the M5 Pro or A14 partition time drops and the frames stay within the FP32 floor.
2. **Mac:** the Neural Engine eligibility experiment with pre-scaled norms in FP16 (§7.4). This is the largest potential A14 win.
3. **Quality:** finish the CPU A/B set (§6) and review the comparison videos together. Decide on `ffnk2`/`ffnmap` only after the Mac timing.
4. **If more speed is required:** distil a smaller or fewer-block student. The FFN density result shows that training-free pruning cannot provide it. Use this repository's CPU A/B harness plus the device runs as the quality gate.
5. Optional quality lever: per-block INT8 (scale per 32) weights, about 3× lower FFN quantization error.
