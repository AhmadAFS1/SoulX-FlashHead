# Tiny VAE for SoulX-FlashHead PRO: can TAESD replace the VAE? (2026-09-26)

**Question (user, 2026-09-26):** analyse ways to improve efficiency inside the model. On
MuseTalk, TAESD gave about 5x at the same quality. Can the SD VAE be replaced with a TAESD
decoder here, and is that possible at all for SoulX-FlashHead?

**Audience:** engineers on this project who know the pipeline, the harness and
`PRO_40FPS_ITERATION_LOG_2026-09-21.md`.

**Status:** feasibility answered, implementation landed behind flags, measured end to end,
**not shipped**. Nothing was committed.

## Answer

1. **TAESD itself cannot be used.** SoulX does not use the SD VAE. It decodes **Wan 2.1**
   latents: 16 channels, 8x spatial, 4x causal temporal, with 9 latents per 33-frame window.
   TAESD decodes SD's 4-channel, per-image 2D latents, so the channel count, the latent
   basis and the time axis are all wrong.
2. **The TAESD author's Wan 2.1 model, TAEHV `taew2_1`, does fit.** It is now wired into the
   real 576x320 harness for the per-window decode and the per-window motion re-encode. The
   reference-image encode stays on Wan.
3. **It roughly doubles throughput. It does not give 5x.** Same session, same attention
   backend: **26.81 → 52.90 useful FPS (1.97x)**. The VAE stages fall from 548 to 44 ms per
   window, and the nvidia-smi sampled peak falls from 9,019 to 6,105 MiB. The DiT is then
   83-87% of the window, and it caps the pipeline. Even a zero-cost VAE would give only about
   2.5x on the shipping stage totals.
4. **Quality is not the same, so the tiny decoder is not shippable as a drop-in.**
   - taew2_1 adds teeth speckle, a white fleck on the lower teeth, grainier stubble and about
     16% more skin shimmer.
   - Its oral edge ratio fails the band on the high side (1.18-1.26 against the reference,
     1.4-1.8x the paired control).
   - Its paired lip-sync correlation (0.92-0.95) is below the 0.984 control-vs-control floor.
5. **The encoder-only swap (T5) is the lead candidate, not a proven ship.** It keeps the
   shipping decoder and gives **+12% (26.81 → 30.02 FPS)**, and its texture looked
   indistinguishable from the control. But it was run on 1 fixture and 1 seed, not under
   SageAttention; its paired gates sit beyond the control floor; and it adds 0.35/255 of
   colour drift.
6. **The route to about 2x at the same quality** is distilling taew2_1 on SoulX latents,
   with the Wan decoder as the teacher. That is the same recipe that rescued the pruned
   shipping decoder. It has not been attempted.

## 0. Provenance and labels

- **Hardware:** fresh local GPU inference on NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB
  visible; physical class unverified in this run. Compute capability 8.9, GPU UUID
  `GPU-69894a01-4488-37ac-5fd8-878224044365`.
- **Software** (from the `environment` blocks of the run `results.json` files): driver 595.84,
  Torch 2.7.1+cu128, CUDA runtime 12.8, cuDNN 9.7.1.26, Triton 3.3.1, flash_attn 2.8.0.post2.
- **TensorRT:**
  - The Wan span engines in the control used TensorRT 10.9.0.34. That was a runtime-only
    restore into `.restored-deps-20260926/trt-10.9.0.34/`; see its `PROVENANCE.txt`.
  - The tiny-VAE engines were built with TensorRT 10.3.0, borrowed read-only from
    `/workspace/.venvs/musetalk_trt_stagewise` through
    `benchmarks/tiny_vae_20260926/decoder_bakeoff/_trt10_path`.
- **Date and load:** all runs 2026-09-26. The end-to-end arms ran 21:45-22:12 UTC in one
  session, each holding the SoulX lease. `environment_after` lists only the run's own process.
- **Workload:** 576x320, 250 frames, `indian150-a` (plus `tts-plosives` and
  `tts-open-vowels`), seed 50, 2 distilled steps, static INT8 `ffn.2` scales, DiT CUDA graph,
  fused INT8 FFN, lean delivery.
- **Attention (important):** every arm, including the control, used **FlashAttention-2**
  self-attention.
  - The shipping policy needs the SM89 stream-patched SageAttention-2 build
    (`/workspace/experiments/pro30-deps`). That directory was deleted, and a rebuild was
    denied by the permission system.
  - The control policy `benchmarks/tiny_vae_20260926/policies/ctl_final_v4_flash2.json`
    differs from `final_v4.json` only in `self_attention_kernel.backend` (sage2 → flash2).
  - Under flash2 the DiT takes 456.7 ms per window, against 343.3 ms under sage2. So the
    absolute FPS is below the shipping 30.22, and only same-session ratios are measured.
- **Number labels used in this report:**
  - "Measured": fresh local GPU inference on this card, 2026-09-26.
  - "Reused": the 2026-09-22 shipping run `final-v4-lean-r01`, not re-run.
  - "Projection": arithmetic on stage totals, not measured.
  - "Static": meta-device operation counts, not timings.
- **Verification:** two independent verifier passes checked these results.
  - A GPU and CPU skeptic re-ran the conventions, metrics and eager timings.
  - A CPU-only quality critic inspected stills and took measurements from the mp4s.
  - Their corrections are applied throughout; §6.4 lists the claims they refuted or narrowed.

## 1. Why TAESD cannot decode SoulX latents

| | SD VAE / TAESD (MuseTalk) | Wan 2.1 VAE (SoulX-FlashHead PRO) |
|---|---|---|
| Latent | 4 channels, 8x spatial, one 2D image per latent | 16 channels, 8x spatial, **4x causal temporal** |
| Frames per call | 1 latent gives 1 frame (batched) | 9 latents give 33 frames (latent 0 gives 1 frame, each later latent 4) |
| State | none | causal 3D-conv caches, carried across windows by overlap-skip |
| Normalisation | SD scaling factor | per-channel mean and 1/std; the DiT works in normalised space (`vae.py:791-796`, `808-813`) |

The pipeline calls the VAE at three points (`flash_head/src/pipeline/flash_head_pipeline.py`):

| Call | Where | Shape | Can a tiny model take it? |
|---|---|---|---|
| Reference encode, once per session | `:253` | (1,3,33,576,320) → (16,9,72,40), used as the DiT's `y` every step | **No, it stays on Wan.** It is one-off and uncounted in the stage times. |
| Decode, every window | `:456` | (16,9,72,40) bf16 normalised → (1,3,33,576,320) bf16 in [-1,1] | Yes |
| Motion re-encode, every window | `:504` | last 5 colour-corrected frames (1,3,5,576,320) → (16,2,72,40) | Yes |

**Wan 2.1 candidates** (all downloaded to `models/tiny_vae/`, provenance in `SOURCES.txt`):

| Candidate | What it is | Params | Licence |
|---|---|---:|---|
| `taehv/taew2_1.pth` | TAEHV by the TAESD author. Plain 2D convs, MemBlock temporal memory, no attention, no norms. | 11.3 M | MIT |
| `lightx2v/lighttaew2_1.safetensors` | the same architecture, retrained by LightX2V; takes un-normalised latents | 11.3 M | Apache-2.0 |
| `lightx2v/lightvaew2_1.safetensors` | the Wan 2.1 VAE with every width cut to a quarter; loads strictly into SoulX's own `WanVAE_(dim=24)` | 8.0 M | Apache-2.0 |
| stock Wan 2.1 VAE, for comparison | | 126.9 M | |

`taew2_2*` are for Wan 2.2's 48-channel, 16x latent space and do not apply. The Wan 2.1
upstream licence was not re-verified; this is not legal advice.

## 2. Why the MuseTalk 5x does not transfer

The MuseTalk result is `/workspace/experiments/musetalk_sdvae_vs_taesd_new_avatars_20260925/README.md`:
fresh local GPU inference on the same card, 2026-09-25, 44.4 → 215.6 and 44.1 → 219.6 FPS
(4.86x and 4.98x). Five things differ here.

1. **Amdahl.**
   - In MuseTalk's SD-VAE arm, the decoder took 142.25 ms of the 180.2 ms per 8-frame
     batch, about 79%. MuseTalk's UNet is a single-step TensorRT FP16 batch-8 model.
   - In SoulX, decode is 47% of generation and motion encode 13%; the rest is a 1.3 B
     parameter DiT at 2 steps (shipping, reused).
   - So removing the VAE entirely would give at most about 2.5x (§6.3). The tiny arm reached
     52.90 FPS, which is 93% of the 57.0 FPS that a zero-cost decode and encode would allow
     in the same flash2 session (projection: 9.3254 − 3.891 − 1.048 s).
2. **The baseline here was already optimised.**
   - MuseTalk replaced a stock FP16 SD-VAE.
   - The SoulX shipping decoder is already block-pruned, distilled and running on FP16
     TensorRT spans with overlap-skip. It runs at 434 ms per window, against 1,384 ms warm
     for the stock eager Wan decoder (measured in the bake-off).
   - The decoder-alone ratios are in fact similar in size. MuseTalk went from 17.8 to 0.86 ms
     per frame at 256x256 (20.7x, compiled). Here taew2_1 on TensorRT runs 11.6x faster than
     the shipping decode and about 38x faster than the stock eager Wan decode.
3. **Crop versus the whole frame.**
   - MuseTalk decodes a 256x256 lower-face crop and composites it into real video.
   - SoulX generates every pixel of every 576x320 frame. Tiny-decoder texture error is
     therefore everywhere: stubble, skin and teeth.
4. **2D per frame versus 3D causal with 4x temporal compression.**
   - TAESD decodes one image per latent.
   - taew2_1 has to produce 4 frames from each 16-channel latent and carry MemBlock state
     across windows.
   - Its teeth speckle changes position from frame to frame while the mouth moves, and it
     settles on static frames (critic, frames 158-161). This is consistent with detail being
     re-synthesised per frame from a shared latent, but that is an interpretation, not a
     measurement.
5. **Closed loop.**
   - SoulX re-encodes the decoded, colour-corrected last 5 frames of each window into the
     next window's DiT conditioning. Decoder and encoder error therefore re-enter the
     recurrence. MuseTalk has no such loop.
   - The loop is why colour drift, the lip-sync trajectory and the paired gates move even
     when only the encoder is swapped (T5).

**Quality evidence on the MuseTalk side.** The MuseTalk README states that its runs "do not
... approve TAESD's visual quality"; its acceptance evidence is the side-by-side videos plus
a reconstruction MAE of 0.015 with the right latent convention. The same care was needed
here, and the latent convention again decided everything (about 20 dB, §3).

## 3. Conventions, resolved empirically

They were resolved on the GPU over 36 fresh DiT latent windows and confirmed independently by
the skeptic using upstream `decode_video` and `encode_video`.

| Model | Decoder input | Output | Frames | Encoder |
|---|---|---|---|---|
| taew2_1 | the DiT-normalised latent **as is** (39.63 dB; un-normalised gives 19.04 dB) | NTCHW [0,1], mapped `x*2-1`, bf16 NCTHW | cold: 36 raw frames, drop the first 3 → 33. Warm (carried state): 7 latents → 28 frames plus the 5-frame tail. | `(x+1)/2`, **append** 3 copies of the last frame (8 frames → 2 latents); output used as is |
| lighttaew2_1 | un-normalised `z/inv_std + mean` (39.04 dB; normalised gives 20.25 dB) | same | same | same padding; output needs `(e-mean)*inv_std` |
| lightvaew2_1 | the Wan protocol, `decode(z, [mean, inv_std])` (40.30 dB) | same as Wan | same as Wan | the Wan protocol |

- **Wrong settings:**
  - Front-padding the encoder the Wan way (3 copies of frame 0) is wrong. It raises slot-1
    error from 0.089 to 0.29 nRMSE, and the lip state visibly lags.
  - A ±1 frame offset on decode costs 7-8 dB.
- **Stream decode:** with carried state it equals a full-sequence decode within 7.45e-7 in
  CPU fp32. On the GPU in fp16 the skeptic measured at most **1 uint8 level** of difference,
  so it is exact only up to fp16 rounding.

## 4. Decoder bake-off (single window, 36 fresh DiT latent windows, reference = stock Wan decoder)

- **Latents:** 4 fixtures (`indian150-a`, `tts-plosives`, `tts-open-vowels`, `tts-sibilants`)
  × windows 0-8, seed 50, dumped from the real pipeline with flash2 and the stock eager
  decoder in the loop.
- **Reference:** reproduced the in-loop decode checksum exactly (max abs 0.0).
- **Colour correction:** none applied.

**Speed.** Median over the full contract (bf16 normalised latents in, bf16 [-1,1] NCTHW out),
in ms:

| Decoder | cold, 9 latents → 33 frames | warm, 7 latents → 28 frames |
|---|---:|---:|
| taew2_1 fp16 eager | 90.03 | 70.62 |
| taew2_1 fp16 torch.compile | 58.48 | 45.29 |
| **taew2_1 fp16 TensorRT 10.3** | **46.29** | **36.11** |
| lightvaew2_1 bf16 torch.compile | 208.00 | 175.38 |
| shipping pruned decoder, bf16 eager | 819.64 | 693.27 |
| stock Wan 2.1, bf16 eager | 1,628.75 | 1,383.99 |
| shipping harness decode (TensorRT spans + overlap-skip; reused) | 433.9 per window, average | |

**Quality**, window mode, mean / worst window:

| Decoder | PSNR full | PSNR mouth | mouth gradient ratio* | flicker ratio | worst colour offset /255 |
|---|---|---|---:|---:|---:|
| shipping pruned + ft4 | 49.01 / 48.43 | 44.85 / 43.47 | 0.957 | 0.962 | 0.07 |
| taew2_1 fp16 | 39.63 / 38.95 | 35.38 / 33.99 | 0.979 | 1.000 | 0.43 |
| lighttaew2_1 | 39.04 / 37.86 | 34.46 / 32.59 | 0.943 | 0.826 | 0.86 |
| lightvaew2_1 | 40.30 / 38.76 | 34.93 / 32.91 | 0.835 (blurred) | 0.769 | 0.54 |

\* The gradient ratio is mean |dx|+|dy| of luma, and it **counts grain as sharpness**.
taew2_1's 0.979 does not mean it keeps mouth detail. The harness oral edge ratio and the
stills show the added grain (§6).

- **Stream mode** (the overlap-skip analogue), full-frame PSNR mean: stock Wan with carried
  cache 48.31, shipping 42.38, taew2_1 38.52, lighttaew2_1 37.87, lightvaew2_1 38.69 dB. The
  overlap-skip mechanism itself costs up to about 7 dB, even on the stock decoder.
- **Closed here:** lightvaew2_1 is blurry and 175 ms even compiled. Its 16x fewer MACs do
  not turn into speed because narrow 3D convs do not scale.

## 5. Encoder bake-off (per-window motion re-encode, 36 real cond_frame clips)

The clips were rebuilt exactly as the pipeline builds them, and validated against the in-loop
latents to relative L2 at most 0.0066.

| Encoder | median ms | nRMSE slot 0 / slot 1 | systematic offset, slot 1 (RMS) |
|---|---:|---|---:|
| stock Wan, torch.compile (as `--compile-vae-encode`) | 116.50 | — | — |
| **taew2_1 fp16 torch.compile / TensorRT** | **6.62 / 6.45** | **0.138 / 0.089** | 0.0135 |
| lightvaew2_1, compile | 18.96 | 0.155 / 0.096 | 0.018 |
| lighttaew2_1, eager | 10.66 | 0.191 / 0.114 | 0.020 |
| baseline the shipping build already carries: Wan encode of shipping-decoder frames | — | 0.134 / 0.093 | 0.020 |
| rejected latent feedback `last2-fix0` (drifted 2.77/255) | — | 0 / 0.428 | **0.143** |

- The compiled Wan encoder standalone (116.5 ms) matches the harness stage (116.4 ms per
  window), so these timings carry over.
- taew2_1's latent error is about the size of the perturbation the pruned shipping decoder
  already introduces.
- Its systematic offset is 10x smaller than the rejected latent-feedback arm's.
- It keeps encoding colour-corrected pixels, so the correction loop stays closed. That
  argument held over 10 s end to end (§6.2, drift plateaus), but it is untested beyond 10 s.

## 6. End-to-end harness arms (the real pipeline)

**Implementation:**
- `soulx_rtc/pro_tiny_vae.py` (new) holds `TinyDecoder` and `TinyEncoder`, with eager,
  torch.compile and TensorRT backends and stream or window decode mode.
- The TinyDecoder cold output is bit-identical to the bake-off reference. A per-window guard
  enforces 33 frames.
- `run.py` gains four flags, and the default path stays byte-identical:
  `--tiny-vae-decoder`, `--tiny-vae-encoder`, `--tiny-vae-backend {eager,compile,tensorrt}`
  and `--tiny-vae-decode-mode {stream,window}`.
  - Proof of the unchanged default: `tae-ctl-r02` (edited `run.py`) and `tae-ctl-r03-origrunpy`
    (HEAD `run.py`) have the same raw sha256, `e93fb5ab…`.
  - The diff is +68/−1 lines.
- **Refused combinations:**
  - With a tiny decoder: `--overlap-skip`, `--skip-decoder-blocks`, `--vae-weights`, any
    non-`pytorch/bf16` decoder policy, and `--capture-manifest`.
  - With a tiny encoder: `--compile-vae-encode` and `--latent-feedback`.
  - With either tiny flag: `--sessions>1` and `--force-scheduler`.
- `tests/test_tiny_vae_flags.py`: 19 CPU tests, re-run for this report, 19 passed. The
  current `run.py` sha256 (`e90fa558…`) and `pro_tiny_vae.py` sha256 (`544dca46…`) equal the
  hashes recorded by the arms.

### 6.1 Speed

Measured on `indian150-a`, seed 50, same session, flash2 in every arm. Stages are in ms per
window.

| Arm | Run | Useful FPS (x control) | Window 0 / p50 (s) | DiT / decode / encode | nvidia-smi peak MiB (sampled) |
|---|---|---:|---|---|---:|
| Control: shipping VAE (pruned + ft4 Wan decoder, FP16 TRT spans, overlap-skip, compiled Wan encoder) | `tae-ctl-r01` | 26.81 (1.00) | 1.10 / 1.028 | 456.7 / 432.3 / 116.4 | 9,019 |
| Control repeat (HEAD `run.py`) | `tae-ctl-r03-origrunpy` | 26.89 | 1.10 / 1.025 | 457.2 / 433.8 / 116.7 | 8,949 |
| T1: taew2_1 decoder (compile) + Wan encoder | `tae-dec-r01` | 42.80 (1.60) | 0.66 / 0.648 | 455.0 / 47.0 / 116.4 | 6,767 |
| T2: taew2_1 decoder + encoder (compile) | `tae-decenc-r01` | 51.54 (1.92) | 0.55 / 0.538 | 455.2 / 46.8 / 6.4 | 6,609 |
| **T3: taew2_1 decoder + encoder (TensorRT)** | `tae-decenc-trt-r01` | **52.90 (1.97)** | 0.53 / **0.524** | 455.9 / **37.1** / **6.4** | **6,105** |
| T4: lighttaew2_1 decoder + encoder (compile) | `tae-light-decenc-r01` | 51.57 (1.92) | 0.55 / 0.538 | 455.5 / 46.6 / 6.4 | 6,609 |
| **T5: shipping decoder + taew2_1 encoder (compile)** | `tae-enc-r01` | **30.02 (1.12)** | 0.99 / 0.918 | 455.7 / 432.0 / 6.5 | 8,501 |
| T6: T3 with window-mode decode | `tae-decenc-trt-window-r01` | 51.95 (1.94) | 0.53 / 0.534 | 456.4 / 46.2 / 6.4 | 6,043 |

- **Robustness:** T3 gave 52.87 FPS on `tts-plosives` and 52.81 on `tts-open-vowels`, against
  controls of 26.82 and 26.92.
- **Warmup:** T3 15.0 s, against the control's 19.5 s with a warm compile cache (80.3 s cold).
- **Torch peak allocated:** 7,157 MiB for the control, 4,859 MiB for T3.
- **Timing integrity** (skeptic):
  - generation_s equals the stage sum plus 17.6-22.9 ms per window in every arm.
  - The microbenchmarks predict the harness stages: 46.3 + 8×36.1 = 335 ms, against 334 ms.
- **1.97x is against a flash2 control** at 26.8 FPS, not against the 30.22 shipping build.

### 6.2 Quality

Two comparisons: review.py against the `s30-reuse` reference, and review.py paired against the
same-session control. The accepted band is correlation 0.94-0.97, edge ratio 0.95-1.07 and
mouth distance ≤ 3.4 px. Drift is the maximum per-window mean-RGB drift against the run's own
window 0, in /255.

| Arm | vs reference: corr / edge / mouth px | vs control: corr / edge / mouth px | drift |
|---|---|---|---:|
| Shipping sage2 build, reused 2026-09-22 | 0.970 / 0.907 / 2.2 | — | 1.32 |
| Control r01 (flash2) | 0.939 / 0.907 / 2.6 | — | 1.264 |
| Control r03 (noise floor) | 0.953 / 0.771 / 2.7 | **0.984 / 0.857 / 1.3** | 1.247 |
| T1 | 0.939 / 1.263 / 3.0 | 0.950 / 1.407 / 3.7 | 1.709 |
| T2 | 0.945 / 1.180 / 3.2 | 0.947 / 1.325 / 3.9 | 1.633 |
| T3 | **0.920** / 1.192 / 3.1 | 0.923 / 1.408 / 4.2 | 1.591 |
| T4 | 0.952 / **1.012** / **4.4** | 0.943 / 1.132 / 5.2 | 1.507 |
| T5 | 0.937 / 0.864 / 3.2 | 0.959 / 0.935 / 2.6 | 1.610 |
| T6 | 0.935 / 1.252 / 3.5 | 0.937 / 1.397 / 4.0 | 1.610 |

**Other fixtures** (paired against each fixture's own control):

| Fixture | Arm | corr / edge / mouth px | drift (control) |
|---|---|---|---|
| `tts-plosives` | T3 | 0.949 / 1.830 / 3.4 | 1.544 (1.209) |
| `tts-plosives` | T4 | 0.966 / 1.510 / 2.5 | 1.642 |
| `tts-open-vowels` | T3 | 0.942 / 1.750 / 3.7 | 1.090 (0.766) |
| `tts-open-vowels` | T4 | 0.900 / 1.284 / 3.7 | 0.905 |

**How to read these gates:**
- **Nothing passes all three bands, the flash2 control included.** Both controls miss the
  edge-ratio band, and r01 also misses the correlation band by 0.001. The band was set against
  sage2 builds.
- **The paired comparison is the fairer one.** Every tiny-decoder arm is below the 0.984
  control floor on correlation and above the 1.3 px floor on mouth distance.
- **Single-seed noise is large.** Control r01 and r03 differ by 0.14 in edge ratio. T2 and T3
  differ only in backend, yet score 0.945 and 0.920. These single-seed results cannot rank
  arms that are close.

**Seams, drift and mouth closure:**
- **No window seams:** the ratio of change at window boundaries to change inside windows is
  1.055 for T3 and 1.080 for the control.
- **Drift is a step, not a ramp.** Every arm shows an R-channel step at windows 1-2 (T3 +1.34
  at window 1, control +0.73), then a plateau with similar slopes (R +0.101 per window for
  the control, +0.113 for T3, +0.122 for T5). It is bounded over 10 s. Longer sessions are
  unmeasured, and the metric mixes content (mouth opening, pose) into colour.
- **Weaker mouth closure on plosives is plausible, not proven.** On `tts-plosives` the control
  has 130 frames with the mouth closed (opening under 4 px), T3 has 112 and T4 has 116; the
  median opening rises from 0.025 to 0.046 (T3). `tts-open-vowels` goes 73 → 66 (T3) and 64
  (T4). This could be weaker bilabial closure, but it is one seed with no per-fixture noise
  floor. It also inflates the 1.83 plosives edge ratio, which is computed only on frames where
  both mouths are open.

**What it looks like** (critic, CPU, stills from the mp4s):

| Arm | Visual | Mouth Laplacian variance | Forehead mean abs frame difference |
|---|---|---:|---:|
| Reference (full decoder) | — | 119 | 5.46 |
| Control | clean teeth | 100 | 5.54 |
| taew2_1 arms (T1, T2, T3, T6) | dark grey-green blotches across the teeth on smiles (frames 109, 162, 212); a white fleck on the lower teeth when wide open (frame 189, every taew2_1 arm); darker, crunchier stubble | 150-152 | 6.1-6.5 (+16%; +13% plosives, +21% open-vowels) |
| T4 (lighttaew2_1) | right amount of detail, reached by smoothing: softer jaw and stubble, pinker lips, grey streaks on the right-hand teeth | 118 | 5.30 |
| T5 (encoder only) | not distinguishable from the control in texture | 105 | 5.22 |

- **No gross artefacts:** no smearing, 8x8 grid or banding in any arm, and the static
  background is unchanged (mean abs frame difference 0.44-0.50 everywhere).
- **Report author's spot check** of the new four-way grid `tae-t5-4way-mouth-grid.mp4`
  (frames 109, 162 and 189):
  - It confirms the T3 teeth blotches (frames 109 and 162) and the lower-teeth fleck
    (frame 189).
  - T5 is clean at frames 109 and 189.
  - At the frame-162 smile, T5's right-hand teeth show darker gaps than the control's, much
    milder than T3's blotches. T5 runs the shipping decoder, so this comes from a different
    DiT trajectory, not from decoder texture. That is one more reason T5 needs multi-seed
    gating.
- **Bitrate:** the tiny-decoder arms' mp4s run 20-30% higher bitrate at the same encode
  settings.

### 6.3 Projection with the shipping attention (arithmetic, NOT measured; sage2 cannot run here now)

The starting point is the reused 2026-09-22 stage totals (DiT 3.09, decode 3.9055, motion
encode 1.0526 s over 9 windows; generation 8.2725 s). Each arm's measured VAE stages are
swapped in.

| Arm | Generation s | Projected useful FPS |
|---|---:|---:|
| Shipping (measured 2026-09-22) | 8.2725 | 30.22 |
| T5: taew2_1 encoder only | 7.278 | ≈34.3 |
| T1 with the TensorRT decoder: taew2_1 decoder only | 4.701 | ≈53.2 |
| T3: taew2_1 decoder + encoder, TensorRT | 3.706 | ≈67.5 (2.2x) |
| Ceiling: zero-cost decode | 4.367 | ≈57 |
| Ceiling: zero-cost decode + encode | 3.314 | ≈75 (2.5x) |

### 6.4 Claims refuted or narrowed by verification

| Claim | Verdict |
|---|---|
| "Quality is the same" for any decoder-swap arm | **Refuted.** Visible teeth and stubble artefacts, and more skin shimmer. |
| T3 lip sync and mouth distance "within control noise" | **Refuted.** 0.920 against a control range of 0.939-0.953, and 4.2 px against a 1.3 px paired floor; T3 also fails the 0.94 band. |
| T5 "gates the same as shipping; ship-safe now" | **Narrowed.** Visually the same, but paired correlation is 0.959 against the 0.984 floor and drift is +0.35/255, on 1 fixture and 1 seed. It is the lead candidate, not proven. |
| taew2_1 "keeps sharpness" (0.98x) | **Misleading.** The gradient metric counts grain. |
| Bake-off "flicker 1.00x" | **Narrowed.** The background is stable, but forehead and teeth change 9-21% more from frame to frame. |
| "Stream decode is exact" | **Narrowed.** Exact in CPU fp32; at most 1 uint8 level in GPU fp16. |
| "Drift +0.3-0.45/255 on every fixture" | **Narrowed.** True for the taew2_1 arms (+0.32 to +0.45). lighttaew2_1 on `tts-open-vowels` is +0.14. |
| "Saves 2.9 GB VRAM" | **Holds in direction and size.** The values are sampled nvidia-smi maxima (9 samples for T3), not true peaks. |

## 7. Recommendation

1. **Do not ship the tiny decoder (T1, T2, T3, T4, T6) under the current acceptance
   criteria.** The speed is real, but the mouth cost is visible and the paired lip-sync gates
   fail.
2. **Take the encoder-only arm (T5) through a proper gate before shipping it.** It is the
   lowest-risk real gain: it removes 110 ms per window, projects to about 34 FPS with sage2,
   and leaves the decoder untouched.
   - Flags on top of the shipping build: add
     `--tiny-vae-encoder taew2_1 --tiny-vae-backend compile` and **remove**
     `--compile-vae-encode`. run.py refuses the combination. `--force-encode-compile` is a
     dead flag and may stay.
   - `hires_span.sh` always adds `--compile-vae-encode`, so run the arm through
     `benchmarks/tiny_vae_20260926/e2e/run_arm.sh`, which does not.
   - Once sage2 is restored, use `final_v4.json`, not the flash2 control policy:
     ```
     POLICY=benchmarks/pro_30fps_20260920/policies/final_v4.json bash benchmarks/tiny_vae_20260926/e2e/run_arm.sh <run> \
       --vae-weights benchmarks/pro_30fps_20260922/distill/vae_skip9-10-13-14_ft4.pth \
       --skip-decoder-blocks 9 10 13 14 --overlap-skip \
       --tiny-vae-encoder taew2_1 --tiny-vae-backend compile
     ```
     run_arm.sh points PYTHONPATH at the restored TensorRT 10.9 runtime; the sage2 build
     path has to be added to it once rebuilt.
   - **Gate before shipping:**
     - at least 3 fixtures × 2 seeds, with a per-fixture control-vs-control floor;
     - a run of 60 s or more (at least 55 windows) with a regenerated reference `raw.npy`;
     - SyncNet or closure checks at bilabial phonemes;
     - one more avatar;
     - a labelled video.
   - **Fallback** if T5 fails: E1, the TensorRT Wan motion encoder (§8).
   - **Rollback:** drop the two `--tiny-vae-*` flags and restore `--compile-vae-encode`. The
     default path is byte-identical.
3. **For about 2x at the same quality, distil taew2_1** (the next structural experiment).
   - Teacher: the full Wan decoder, run on the fly.
   - Data: store only latents, about 0.8 MB per window. 88 windows is about 70 MB, which fits
     the disk; the old 3.2 GB frame dataset does not.
   - Target the teeth speckle and stubble grain, with an edge-matched loss.
   - Re-gate with the same scripts; the target is the T3 arm at T5-level quality.
4. **Restore SageAttention-2 SM89 first.** Without it the shipping build cannot be reproduced,
   and no arm's absolute FPS is meaningful. The rebuild needs the user's go-ahead, because the
   permission system blocked it.

## 8. Remaining efficiency levers, re-ranked with what this study measured

The earlier audit (CPU, 2026-09-26) ranked levers by expected FPS per unit of risk and effort.
The table below updates that ranking. `[E]` marks an estimate, `[M]` a measurement. FPS
figures assume the sage2 shipping stage totals.

| # | Lever | ms per window | Evidence | Risk | Next step |
|---|---|---|---|---|---|
| 0 | **Restore the SageAttention-2 SM89 build** (not new; an environment regression) | −113 vs today's flash2 DiT [M: 456.7 against 343.3] | shipping 30.22 cannot be reproduced without it | none (it is the shipped state) | rebuild from `.deps/SageAttention` (d1a57a5, stream-patched), with approval |
| 1 | **taew2_1 motion encoder (T5)** | −110 [M] | +12% same session; ≈34.3 FPS projected | numerics-structural; paired gates beyond the floor | multi-seed, multi-fixture, long-session gate with sage2 (§7.2) |
| 2 | **Distil taew2_1 decoder** on SoulX latents | −395 decode [M ceiling: T3 37 against 432] | T3 ≈67.5 FPS projected with the encoder | structural | §7.3 |
| 3 | V2: prune RB[12] (or RB[8]) + distil round 5, Wan decoder | −100 to −107 (−70) [E] | the recipe that shipped | structural; edge already 0.907 | only if #2 fails |
| 4 | V1: head inside the tail engine, v4 weights | −23 [M on v2] | costs 0.06-0.08 edge on v2 | numerics | only if #2 fails |
| 5 | E1: TensorRT Wan motion encoder (code complete, no plan built) | −25 to −50 [E] | | numerics | only if #1 fails |
| 6 | D1: INT8 attention projections with a fused epilogue | −15 to −30 [E] | the largest DiT lever left after a tiny VAE | numerics | kill check ≤ 0.19 ms at (6480, 1536, 1536) |
| 7 | H1: host and delivery fixes (deferred isfinite sync, pinned non-blocking D2H, audio ring buffer) | −5 to −15 [E] | exact; its share doubles once the VAE is small (17.6-22.9 ms per window measured remainder in the tiny arms) | exact | low effort |
| 8 | D3: INT4 W4A4 FFN with a low-rank branch | −40 to −50 [E] | | numerics-high | high effort |
| 9 | D4: block-sparse self-attention | −15 to −40 [E] | | numerics or structural | medium-high effort |
| 10 | V3 / V4 / V5 / V7 (Wan-decoder structure and precision) | see the audit | moot if #2 passes | | |

**After a tiny VAE the DiT is 83-87% of the window.** Every further large gain has to come
from the DiT (D1, D3, D4). D5 (motion history 5 → 1: −143 ms measured, gate failed at
correlation 0.42) and D7 (DiT block distillation, not feasible on this card) stay closed.

**Closed by this study:**
- TAESD (SD) and `taew2_2`: incompatible latent spaces.
- lightvaew2_1 as the decoder: blurry, 175 ms compiled; not worth an end-to-end arm.
- lighttaew2_1 as is (T4): mouth distance 4.4-5.2 px; it matches edge energy by smoothing.
- Window-mode tiny decode (T6): no drift benefit (1.61 against 1.59), −1 FPS.
- Wan-style front padding for the tiny encoder: the lip state lags.
- "5x from the VAE": the zero-cost ceiling is about 2.5x.

**Not tried** (expected small, since the tiny calls total 44 ms per window): CUDA graphs
around the tiny VAE calls, and fusing the tiny decode and encode into one engine.

## 9. Artifacts

**Code (new files, plus flag-gated edits to run.py):**
- `soulx_rtc/pro_tiny_vae.py`
- `benchmarks/pro_quantization_v2_20260918/run.py` (+68/−1 lines, flag-gated; the original
  is saved as `/tmp/claude-0/-workspace/5a04e616-f8a6-4bae-9052-d4d650c8f46c/scratchpad/run.py.orig-20260926`)
- `tests/test_tiny_vae_flags.py`

**Policies:**
- `benchmarks/tiny_vae_20260926/policies/ctl_final_v4_flash2.json`: `final_v4` with flash2.
- `benchmarks/tiny_vae_20260926/policies/tiny_v4_flash2.json`: decoder `{pytorch, bf16}`,
  flash2.
- A sage2 twin of `tiny_v4_flash2.json` does not exist yet. It would differ only in
  `self_attention_kernel.backend`.

**Weights:** `models/tiny_vae/{taehv/taew2_1.pth, lightx2v/lighttaew2_1.safetensors, lightx2v/lightvaew2_1.safetensors}`,
with `models/tiny_vae/SOURCES.txt` (taehv commit 011dfc2; HF lightx2v/Autoencoders 02cbfd1).
taew2_1 sha256 is `d26151e7…`.

**Study directory:** `benchmarks/tiny_vae_20260926/`
- Contract and CPU probes: `contract_flops_meta.{py,json}`, `load_smoke.{py,json,out}`,
  `convention_probe_cpu*.{py,json}`.
- Latents: `latents/` (36 windows, 32 MB, `index.json`).
- Decoder bake-off: `decoder_bakeoff/results.{md,json}`, `quality.json`, `speed.json`,
  scripts, and `visual/` (`grid_*.png`, `mouth3x_*.png`, `sbs_indian150-a_w4_33f_10fps.mp4`).
- Encoder bake-off: `encoder_bakeoff/results.{md,json}`, `quality.json`, `speed.json`, and
  `visual/motion_latents_*.png`.
- End to end: `e2e/results.json` (all rows), `e2e/window_seams.json`, `e2e/adapter_check.json`,
  and the scripts `run_arm.sh`, `gate_arm.sh`, `review_vs_ctl.sh`, `compare_video.sh`,
  `robustness.sh` and `collect.py`, plus `e2e/stills/*.png` and `e2e/logs/`.

**Runs:** `benchmarks/pro_30fps_20260922/tae-*` and `review-tae-*`. Only
`tae-decenc-trt-r01/raw.npy` (132 MB) was kept.

**Labelled comparison videos** in `benchmarks/pro_30fps_20260922/visual/`:
- `tae-arms-full-4up.mp4` and `tae-arms-mouth-grid.mp4`: the arms side by side.
- `tae-decenc-trt-r01-vs-reference.mp4` and `tae-decenc-trt-r01-mouth-vs-reference.mp4`: the
  best-speed arm.
- `tae-ctl-r01-*`, `tae-dec-r01-*`, `tae-decenc-r01-*`, `tae-light-decenc-r01-*` and
  `tae-decenc-trt-window-r01-*`, each in full-frame and mouth variants.
- `tae-tts-plosives-ctl-vs-tiny-mouth.mp4` and `tae-tts-open-vowels-ctl-vs-tiny-mouth.mp4`.
- **T5, added for this report** (CPU ffmpeg only, from the existing run mp4s):
  - `tae-t5-4way-mouth-grid.mp4` and `tae-t5-4way-full-4up.mp4`: reference | control | T5 | T3,
    each panel labelled with its FPS.
  - `tae-enc-r01-vs-reference.mp4` and `tae-enc-r01-mouth-vs-reference.mp4`.

**Critic stills** (session scratchpad, not in the repo):
`/tmp/claude-0/-workspace/5a04e616-f8a6-4bae-9052-d4d650c8f46c/scratchpad/tae_stills/`.

**Restored runtime:** `.restored-deps-20260926/trt-10.9.0.34/` (703 MB), runtime only.
Deleting it breaks the Wan span engines, and with them the control and shipping arms.

**Tiny TensorRT engines:** deleted to save disk. Their sha256 values are in each run's
`results.json` under `tiny_vae`; they rebuild on first use (about 45 s, as reported by the e2e
agent).

## 10. Open items

1. **Environment (blocking reproduction of the shipping build):**
   - The SageAttention-2 SM89 build is missing, and its rebuild needs approval.
   - There is no single TensorRT install: 10.9 is runtime-only and restored, while 10.3 is
     borrowed from another venv.
   - The tiny engines have no persistent cache, so cold start includes about 45 s of engine
     builds.
2. **Gate evidence:**
   - Only seed 50, and a noise floor only on `indian150-a`.
   - `tts-sibilants`, `tts-rounded`, `tts-numbers-rapid` and `tts-conversational` were never
     run end to end. T5 was run on `indian150-a` only.
   - Only one avatar was tested. It is bearded with teeth often visible, which is exactly
     where the taew2_1 artefacts sit.
   - No audio-to-lip metric (SyncNet); the plosive-closure signal needs one.
   - No session longer than 10 s.
   - The gates run on H.264 mp4s at about 500-650 kbps.
3. **Reference data:**
   - `s30-reuse/raw.npy` and `final-v4-lean-r01/raw.npy` are gone, so the colour-drift gate
     compares each run with its own window 0. Regenerate a reference raw (about 132 MB).
   - The real-inputs latents are gone; `latents/` replaces them.
4. **Deployment:**
   - `--sessions>1` is refused with any tiny flag, and the per-session tiny state is not
     implemented.
   - There is no automatic runtime fallback to Wan on a guard trip.
5. **Minor code gaps (not bugs):**
   - A bare-PATH `--tiny-vae-*` spec skips the validation-time TensorRT/LightVAE refusal; it
     still raises at install time.
   - A tiny decoder combined with `--latent-feedback` is not refused.
6. **GPU tenancy:** `comfy-h3` ComfyUI processes use the GPU outside the SoulX lease (6-9 GB
   seen). All measured runs waited for a quiet GPU, but one dump attempt hit an OOM.
7. **Disk:** 998 MB free at the time of writing, on a 100%-used 204 GB overlay. This study's
   largest items are `.restored-deps-20260926` (703 MB) and `tae-decenc-trt-r01/raw.npy`
   (132 MB).
