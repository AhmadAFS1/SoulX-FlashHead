# Motion re-encode bake-off: tiny encoders vs the Wan 2.1 encoder at 576x320 (2026-09-26)

**Hardware:** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM (nvidia-smi), driver 595.84, torch 2.7.1+cu128, CUDA runtime 12.8, cuDNN 9.7.1. TensorRT 10.3.0 was borrowed read-only from `/workspace/.venvs/musetalk_trt_stagewise` (the same symlinks the decoder bake-off used).

**Labels:**
- Every number measured here is fresh local GPU inference on the RTX 4070 SUPER, 2026-09-26 (quality 2026-09-26T21:19:58Z, speed 2026-09-26T21:15:34Z / 2026-09-26T21:16:39Z).
- Shipping-build numbers are reused from `final-v4-lean-r01` (2026-09-22), not re-run.
- The FPS projections are arithmetic on those reused totals. They are not measurements.

**Co-resident load:** none. Every nvidia-smi snapshot taken during the runs lists only this bake-off's own process, which held the SoulX lease.

**Workload:** the per-window motion re-encode, `flash_head_pipeline.py:502-504`. Input: the trailing 5 colour-corrected frames of a window, (1,3,5,576,320) bf16 in [-1,1]. Output: 2 DiT-normalised latents, (16,2,72,40) bf16. Slot 0 is frame 0 encoded alone; slot 1 covers frames 1-4.

## 1. Inputs, and how they were validated

- **Clips:** 36 cond_frame clips, one per window, from 4 fixtures (indian150-a, tts-plosives, tts-open-vowels, tts-sibilants; seed 50; windows 0-8). They were built from the saved DiT latents in `benchmarks/tiny_vae_20260926/latents/` exactly as the pipeline builds them:
  1. Stock Wan decode, bf16 eager (the decoder bake-off showed this reproduces the in-loop decode bit-exactly).
  2. Lean trim `[:, :, 5:]`.
  3. `match_and_blend_colors_torch` against `reference-150x.png` at strength 1.0, in bf16.
  4. Keep the trailing 5 frames.

- **Validation against the dump:** the motion latents that window w+1 injects in-loop (`latent[w+1][:, :2]`) are the torch.compiled Wan encode of window w's cond_frame. On the 32 window pairs where both exist, my eager Wan encode of the reconstructed clip matches them to rel L2 at most 0.0066 and max abs 0.035 (nRMSE 0.0074 / 0.0070, cosine 0.99998 / 0.99998). So the clips are the pipeline's clips, and this difference is the eager-vs-compiled noise floor.

- **Reference:** `WanVAE.encode` with stock weights, bf16 eager. The shipping ft4 weights have a bf16-identical encoder (checked on CPU by the earlier VAE-contract analysis, reused), so this is also the shipping encoder.

## 2. Conventions, resolved empirically

Error is nRMSE against the Wan encode, averaged over 36 clips and 16 channels, for slot 0 / slot 1.

| variant | slot 0 nRMSE | slot 1 nRMSE | cos slot 0 / 1 |
|---|---:|---:|---|
| taew2_1 fp16 front0 in01 asis | 0.138 | 0.089 | 0.9928 / 0.9972 |
| taew2_1 fp16 front0 in01 raw2norm | 0.966 | 0.848 | 0.6094 / 0.6026 |
| taew2_1 fp16 front1 in01 asis | 0.136 | 0.174 | 0.9931 / 0.9888 |
| taew2_1 fp16 front1 in01 raw2norm | 0.965 | 0.852 | 0.6107 / 0.6013 |
| taew2_1 fp16 front2 in01 asis | 0.136 | 0.246 | 0.9932 / 0.9787 |
| taew2_1 fp16 front2 in01 raw2norm | 0.965 | 0.866 | 0.6107 / 0.5934 |
| taew2_1 fp16 front3 in01 asis | 0.136 | 0.290 | 0.9932 / 0.9695 |
| taew2_1 fp16 front3 in01 raw2norm | 0.964 | 0.864 | 0.6109 / 0.5966 |
| taew2_1 fp16 front0 inpm1 asis | 4.845 | 5.787 | 0.3938 / 0.2541 |
| taew2_1 fp16 front0 inpm1 raw2norm | 3.105 | 3.520 | 0.3655 / 0.2083 |
| lighttaew2_1 fp16 front0 in01 asis | 1.876 | 1.806 | 0.7464 / 0.8744 |
| lighttaew2_1 fp16 front0 in01 raw2norm | 0.191 | 0.114 | 0.9873 / 0.9955 |
| lighttaew2_1 fp16 front1 in01 asis | 1.877 | 1.821 | 0.7450 / 0.8676 |
| lighttaew2_1 fp16 front1 in01 raw2norm | 0.189 | 0.183 | 0.9877 / 0.9880 |
| lighttaew2_1 fp16 front2 in01 asis | 1.877 | 1.851 | 0.7455 / 0.8575 |
| lighttaew2_1 fp16 front2 in01 raw2norm | 0.188 | 0.250 | 0.9880 / 0.9780 |
| lighttaew2_1 fp16 front3 in01 asis | 1.878 | 1.883 | 0.7454 / 0.8512 |
| lighttaew2_1 fp16 front3 in01 raw2norm | 0.188 | 0.290 | 0.9880 / 0.9701 |
| lighttaew2_1 fp16 front0 inpm1 asis | 12.146 | 13.441 | 0.3258 / 0.2751 |
| lighttaew2_1 fp16 front0 inpm1 raw2norm | 5.697 | 6.287 | 0.4012 / 0.2799 |
| taew2_1 fp32 front0 in01 asis | 0.138 | 0.089 | 0.9928 / 0.9972 |
| taew2_1 bf16 front0 in01 asis | 0.135 | 0.090 | 0.9931 / 0.9972 |
| lighttaew2_1 fp32 front0 in01 raw2norm | 0.191 | 0.114 | 0.9873 / 0.9955 |
| lightvaew2_1 bf16 inpm1 wan-scale | 0.155 | 0.096 | 0.9930 / 0.9968 |
| lightvaew2_1 bf16 inpm1 no-scale | 1.888 | 1.785 | 0.7437 / 0.8735 |
| lightvaew2_1 bf16 in01 wan-scale | 1.244 | 1.328 | 0.5280 / 0.4856 |

`frontK` means K copies of frame 0 are prepended and the rest of the padding to 8 frames is copies of the last frame. front0 is upstream `encode_video` (end padding); front3 is the Wan-style guess that frame 0 sits alone in slot 0. `in01` feeds (x+1)/2; `inpm1` feeds x as is. `asis` uses the output directly; `raw2norm` applies (e-mean)*inv_std.

**Resolved conventions**, which match the decoder bake-off's decoder conventions:
- **taew2_1:** input in [0,1]; end padding (front0); output used as is, because it is already DiT-normalised.
- **lighttaew2_1:** input in [0,1]; end padding; output is un-normalised and needs (e-mean)*inv_std.
- **lightvaew2_1:** the Wan protocol unchanged, i.e. input in [-1,1] and `encode(x, wan_scale)`.

**Wrong conventions fail badly:**
- A wrong output normalisation gives nRMSE 0.85-1.9.
- Input in [-1,1] gives 3-13.
- The Wan-style front pad leaves slot 0 unchanged but raises slot 1 from 0.089 to 0.29. That is a temporal lag of the lip state (see the visual).

**Precision and implementation checks** (all on taew2_1):
- fp16, fp32 and bf16 give the same error.
- My static encode core matches upstream `encode_video` exactly (max abs 0.0, fp32).
- TensorRT FP16 matches eager fp16 to a max abs of 0.016.

## 3. Latent error vs the Wan encoder, per slot (36 clips)

How to read the columns:
- **nRMSE** is RMSE divided by that channel's std of the Wan latent over all clips (slot 0 std 0.21-0.62, slot 1 std 0.35-0.92).
- **cos** is cosine similarity over the flattened slot.
- **offset** is the systematic per-channel mean offset, averaged over clips. It is given as the RMS over the 16 channels and the largest channel (channel number in brackets).

| arm | slot 0 nRMSE | slot 1 nRMSE | cos slot 0 / 1 (min) | slot 0 offset RMS / max | slot 1 offset RMS / max |
|---|---:|---:|---|---|---|
| taew2_1 encoder | 0.138 | 0.089 | 0.9928 / 0.9972 (0.9915 / 0.9966) | 0.0336 / 0.104 (ch 13) | 0.0135 / 0.032 (ch 5) |
| lightvaew2_1 encoder | 0.155 | 0.096 | 0.9930 / 0.9968 (0.9920 / 0.9957) | 0.0269 / 0.049 (ch 6) | 0.0178 / 0.043 (ch 5) |
| lighttaew2_1 encoder | 0.191 | 0.114 | 0.9873 / 0.9955 (0.9851 / 0.9948) | 0.0477 / 0.134 (ch 13) | 0.0203 / 0.051 (ch 5) |
| taew2_1, Wan-style front pad (wrong alignment) | 0.136 | 0.290 | 0.9932 / 0.9695 (0.9920 / 0.8932) | 0.0314 / 0.094 (ch 13) | 0.0524 / 0.151 (ch 4) |
| taew2_1 encoder on shipping-decoder frames (vs Wan encode of same clip) | 0.133 | 0.090 | 0.9931 / 0.9972 (0.9917 / 0.9965) | 0.0333 / 0.100 (ch 13) | 0.0129 / 0.032 (ch 5) |
| lightvaew2_1 encoder on shipping-decoder frames (vs Wan encode of same clip) | 0.145 | 0.097 | 0.9936 / 0.9968 (0.9922 / 0.9959) | 0.0238 / 0.047 (ch 7) | 0.0150 / 0.038 (ch 5) |
| taew2_1 encoder on taew2_1-decoder frames (vs Wan encode of same clip) | 0.108 | 0.087 | 0.9960 / 0.9973 (0.9955 / 0.9969) | 0.0167 / 0.047 (ch 13) | 0.0103 / 0.029 (ch 4) |
| Wan encoder on shipping-decoder frames (perturbation the shipping build already carries) | 0.134 | 0.093 | 0.9942 / 0.9971 (0.9894 / 0.9957) | 0.0249 / 0.062 (ch 13) | 0.0201 / 0.036 (ch 11) |
| Wan encoder on taew2_1-decoder frames (perturbation a tiny decoder brings) | 0.304 | 0.189 | 0.9660 / 0.9868 (0.9571 / 0.9838) | 0.0793 / 0.233 (ch 13) | 0.0436 / 0.110 (ch 4) |
| taew2_1 decoder + taew2_1 encoder (vs shipping-path Wan encode of stock frames) | 0.324 | 0.194 | 0.9603 / 0.9869 (0.9507 / 0.9832) | 0.0900 / 0.280 (ch 13) | 0.0393 / 0.080 (ch 4) |
| REJECTED latent feedback last2-fix0 | 0.000 | 0.428 | 1.0000 / 0.9550 (1.0000 / 0.9507) | 0.0000 / 0.000 (ch 0) | 0.1429 / 0.263 (ch 9) |
| REJECTED latent feedback last2 | 1.396 | 0.428 | 0.7588 / 0.9550 (0.7266 / 0.9507) | 0.3211 / 0.684 (ch 7) | 0.1429 / 0.263 (ch 9) |

The error is uniform across fixtures: taew2_1 scores 0.135-0.139 on slot 0 and 0.088-0.091 on slot 1 in every fixture. Windows 0 and 8 are no worse than the middle windows.

## 4. Pixel view of what the DiT is conditioned on

Each arm's 2 latents were decoded with the stock Wan decoder into 5 frames.
- **vs Wan round trip** compares them with the decode of the Wan encoder's latents. This is what the DiT sees differently.
- **round trip vs input** compares them with the clip that was encoded.
- For scale, the Wan encoder's own round trip scores 43.04 dB full frame / 40.33 dB mouth against the input.

| arm | vs Wan round trip: PSNR full / mouth (worst) | mean RGB offset /255, frame 0 | mean RGB offset /255, frames 1-4 | round trip vs input: PSNR full / mouth |
|---|---|---|---|---|
| taew2_1 encoder | 44.62 / 41.10 (43.50 / 37.90) | -0.08, -0.11, -0.13 | -0.13, -0.10, +0.00 | 41.79 / 38.53 |
| lightvaew2_1 encoder | 42.78 / 39.62 (41.43 / 35.68) | -0.09, -0.54, +0.06 | +0.23, -0.43, -0.13 | 40.48 / 37.65 |
| lighttaew2_1 encoder | 41.77 / 38.96 (40.89 / 35.32) | -0.43, -0.22, -0.50 | -0.64, -0.48, -0.65 | 40.31 / 37.33 |
| taew2_1, Wan-style front pad (wrong alignment) | 31.18 / 23.65 (24.65 / 19.09) | -0.13, -0.15, -0.16 | +0.02, -0.01, +0.06 | 30.98 / 23.56 |
| taew2_1 encoder on shipping-decoder frames (vs Wan encode of same clip) | 45.06 / 41.62 (43.76 / 38.40) | -0.14, -0.08, -0.10 | -0.12, -0.09, +0.03 | 42.24 / 39.03 |
| lightvaew2_1 encoder on shipping-decoder frames (vs Wan encode of same clip) | 43.10 / 39.96 (41.71 / 36.20) | -0.16, -0.54, +0.06 | +0.18, -0.40, -0.08 | 40.81 / 37.98 |
| taew2_1 encoder on taew2_1-decoder frames (vs Wan encode of same clip) | 45.09 / 41.58 (43.71 / 38.21) | -0.23, -0.07, -0.04 | -0.18, -0.10, +0.03 | 40.92 / 37.24 |
| Wan encoder on shipping-decoder frames (perturbation the shipping build already carries) | 48.19 / 44.97 (46.97 / 42.51) | +0.01, -0.04, -0.06 | +0.03, -0.02, -0.01 | 43.55 / 40.89 |
| Wan encoder on taew2_1-decoder frames (perturbation a tiny decoder brings) | 40.70 / 36.54 (39.54 / 33.99) | +0.01, -0.11, -0.14 | +0.03, -0.06, -0.05 | 41.83 / 38.31 |
| taew2_1 decoder + taew2_1 encoder (vs shipping-path Wan encode of stock frames) | 39.95 / 35.87 (39.06 / 33.57) | -0.22, -0.17, -0.18 | -0.15, -0.16, -0.02 | 40.92 / 37.24 |
| REJECTED latent feedback last2-fix0 | 28.95 / 28.47 (27.67 / 26.68) | +0.00, +0.00, +0.00 | -1.12, +1.58, -3.80 | 29.18 / 28.56 |
| REJECTED latent feedback last2 | 23.15 / 21.89 (22.13 / 19.39) | +3.39, +1.61, -8.33 | -0.13, -0.15, -7.33 | 23.27 / 21.97 |

## 5. What the DiT will see, and the drift question

The 2 motion latents replace `noise[:, :2]` at every denoising step of the next window. Under the shipping overlap-skip decoder they are never decoded, so only the DiT sees them.

**Why the latent-feedback arms drifted.** They fed the DiT's own output latents back into the next window. Its slot 1 differs from the Wan encode of the colour-corrected frames by:
- a systematic offset of RMS 0.143 over channels, up to 0.263 on channel 9, with channel 9 keeping the sign of its mean in 100% of clips (all channels: 99%);
- nRMSE 0.428.

In pixels that is -1.12, +1.58, -3.80 /255 (RGB). Nothing re-anchors those latents to colour-corrected pixels, so the offset compounds from window to window. That matches the measured drift of 2.56-2.77/255 (slope R +0.34/window).

**How the taew2_1 encoder compares on the same windows:**
- **Systematic offset:** RMS 0.0135 on slot 1, **10.6x smaller** than latent feedback; the largest channel is 0.032, 8.2x smaller.
- **Slot 0:** RMS 0.0336, its largest term. Channel 13 sits at +0.104 (0.27 of that channel's std) and has the same sign in 100% of clips. That is still 4.2x below latent feedback's slot 1.
- **In pixels:** at most 0.12/255 mean RGB offset, against 3.04/255 for last2-fix0.

**Why its offset should not compound.** The tiny encoder still encodes the colour-corrected pixels every window, so the pixel colour-correction loop stays closed. Its bias is re-applied to freshly corrected frames each window, which gives a bounded constant offset, not a random walk.

**Calibration against perturbations the DiT already copes with:**
- **The shipping build:** swapping the stock decoder for the shipping pruned decoder changes the Wan-encoded motion latents by nRMSE 0.134 / 0.093 (offset RMS 0.0249 / 0.0201). The shipping build already carries that perturbation, and its lip-sync and colour gates are in band (reused: correlation 0.970, mouth distance 2.2 px, colour drift 1.32/255; its edge ratio of 0.907 is below the band). The taew2_1 encoder's error is the same size: 0.138 / 0.089. On the shipping decoder's own frames, the input of an encoder-only arm, it is again the same (0.133 / 0.090).
- **A tiny decoder:** the taew2_1 decoder recommended by the decoder bake-off perturbs the Wan-encoded motion latents about twice as much, 0.304 / 0.189. Adding the taew2_1 encoder on top barely raises that (combo 0.324 / 0.194), and on taew2_1-decoded frames its own error is smaller still (0.108 / 0.087). In a full tiny-VAE arm the decoder, not the encoder, is the dominant risk.

**Visual check** (`visual/motion_latents_*.png`, inspected by eye):
- taew2_1, lightvaew2_1 and lighttaew2_1 round trips cannot be told apart from the Wan round trip at 1x or at the 2x mouth crop. Their x8 difference maps show only low-level texture noise.
- In the open-vowel clip the lips close between frame 0 and frame 4. The Wan-style front pad keeps the mouth open at frame 4, a lag that would hurt lip sync.
- last2-fix0 shows a visible colour and contrast shift: darker skin, and pinker, more saturated lips.

## 6. Speed of one re-encode call at 576x320

Method: the full contract from bf16 [-1,1] NCTHW in to (16,2,72,40) bf16 DiT-normalised out, including the tiny encoders' layout, range, padding and normalisation conversions. Input is a real clip (indian150-a window 4). Timing uses CUDA events, 3 warm-up calls and 20 timed calls, and reports the median.

| case | median ms (cuDNN benchmark off = harness default) | median ms (benchmark on) | peak over resident MiB | nRMSE vs Wan eager (this clip) |
|---|---:|---:|---:|---:|
| stock Wan 2.1 bf16 eager | 150.11 | 146.90 | 1372 | 0.0000 |
| stock Wan 2.1 bf16 torch.compile (harness --compile-vae-encode) | 116.50 | 113.25 | 1335 | 0.0074 |
| lightvaew2_1 bf16 eager | 23.88 | 23.34 | 345 | 0.1222 |
| lightvaew2_1 bf16 torch.compile | 18.96 | 18.28 | 336 | 0.1225 |
| taew2_1 fp16 eager nchw | 10.65 | 10.34 | 278 | 0.1127 |
| taew2_1 fp16 torch.compile nchw | 6.62 | 6.36 | 360 | 0.1127 |
| taew2_1 fp16 eager channels_last | 13.11 | 12.90 | 638 | 0.1127 |
| taew2_1 fp16 torch.compile channels_last | 6.59 | 6.35 | 360 | 0.1127 |
| taew2_1 bf16 eager nchw | 10.84 | 10.54 | 278 | 0.1117 |
| taew2_1 bf16 torch.compile nchw | 7.68 | 6.64 | 360 | 0.1120 |
| taew2_1 bf16 eager channels_last | 14.17 | 13.17 | 638 | 0.1117 |
| taew2_1 bf16 torch.compile channels_last | 7.65 | 6.61 | 360 | 0.1120 |
| lighttaew2_1 fp16 eager nchw | 10.66 | 10.32 | 278 | 0.1519 |
| lighttaew2_1 fp16 eager channels_last | 13.14 | 12.89 | 638 | 0.1519 |
| taew2_1 fp16 TensorRT (default stream) | 6.45 | 6.48 | 16 + 382 workspace | 0.1126 |
| taew2_1 fp16 TensorRT (side stream) | 6.45 | 6.47 | 16 + 382 workspace | 0.1126 |
| shipping harness motion_encode stage (reused, final-v4-lean-r01) | 117.0 ms/window | | | |

Notes on the speed table:
- **Calibration:** the stock Wan encoder, compiled the way `--compile-vae-encode` compiles it, measures 116.5 ms here, against the harness's 117.0 ms/window, so these standalone timings carry over.
- **taew2_1:** 6.45 ms as a TensorRT FP16 engine, 6.62 ms with torch.compile, 10.65 ms eager. That is **18x** faster than the shipping encoder and saves about 110 ms per window.
- **TensorRT build:** 7.6 s; the engine uses a 382 MiB workspace.
- **Other encoders:** lighttaew2_1 has the same architecture and speed. lightvaew2_1 needs 19 ms compiled.
- **Settings:** channels_last hurts eager and makes no difference compiled. The cuDNN benchmark setting changes the fp16 and Wan timings by 4% or less; bf16 taew2_1 compiled goes from 7.68 to 6.64 ms.

## 7. Projection (arithmetic on the reused 2026-09-22 stage totals, NOT measured)

Formula: `generation_s = 8.2725 - 1.05255 (motion_encode) + 9 * encode_ms [- 3.90554 + (46.29 + 8*36.11)/1000 for the taew2_1 TRT decoder]`.

| arm | projected generation s | projected useful FPS |
|---|---:|---:|
| shipping (reused, measured 2026-09-22) | 8.273 | 30.22 |
| encoder_only_taew2_1_trt | 7.278 | 34.3 |
| encoder_only_taew2_1_compile | 7.280 | 34.3 |
| encoder_only_taew2_1_eager | 7.316 | 34.2 |
| encoder_only_lightvaew2_1_compile | 7.391 | 33.8 |
| taew2_1_decoder_trt_only (decoder bake-off projection) | 4.702 | 53.2 |
| taew2_1_decoder_trt_plus_taew2_1_encoder_trt | 3.708 | 67.4 |

The encoder alone buys about +14% (30.2 to 34.3 FPS). With the taew2_1 decoder as well the projection is about 67 FPS (2.2x). That is still not 5x: the DiT, at about 343 ms per window, is untouched and becomes about 83% of the remaining time.

## 8. Recommendation

**The taew2_1 encoder is safe to test end to end, and it is the one to test.** Summary of the evidence:
- **Latent error:** equal to the perturbation the shipping build already carries, and smaller than what the tiny decoder introduces.
- **Offsets:** 4-10x smaller than the latent-feedback offsets that caused drift.
- **Colour loop:** it keeps the colour-correction loop closed.
- **Speed:** 18x faster than the shipping encoder.

**Contract** (fp16):
```
x = clip[0].transpose(0, 1).half()
x = (x + 1) / 2
x = cat([x, x[-1:].expand(3, ...)])   # end-pad to 8 frames
z = encoder(x)                        # (2, 16, 72, 40)
z = z.transpose(0, 1).bfloat16()      # (16, 2, 72, 40)
```
- The encoder is stateless per call, so it needs no reset. It can be a TensorRT FP16 engine, or torch.compile at +0.2 ms.
- Do NOT use the Wan-style front pad.

**Order of end-to-end arms** (each gated with review.py, colour drift over 9 windows against a regenerated control raw.npy, and a labelled video):
1. Encoder-only: the shipping build plus `--tiny-encoder taew2_1`. This isolates the encoder. Watch the colour-drift step at window 1, which is the first window conditioned on tiny-encoded latents, and the channel-13 slot-0 bias.
2. The taew2_1 stream decoder with the Wan encoder.
3. Both.

**The other candidates:**
- **lightvaew2_1:** second choice. Its latent error is similar (0.155 / 0.096), but it shows a -0.5/255 green offset and is 3x slower (19 ms).
- **lighttaew2_1:** worst on every metric (0.191 / 0.114, up to -0.65/255). Not recommended.

**Constraints for the integration:**
- The reference-image encode must stay on the Wan encoder.
- `--latent-feedback` must stay off.

## 9. Caveats

- **Single-step only.** Every measurement here is one window: the latents the next window would receive. The closed-loop effects over 9 windows (the DiT's response, lip sync, colour drift, mouth edge ratio) need the end-to-end harness arm and cannot be inferred from this bake-off.
- **Latent source.** The latents come from the decoder bake-off's dump, which used flash2 attention and the stock eager decoder in the loop, because the sage2 and TensorRT 10.9 dependency directories are missing. They are real SoulX DiT outputs but not bit-identical to the shipping sage2 build. The shipping build cannot be reproduced on this machine until those dependencies are restored.
- **Clip source.** The primary clips come from the stock decoder in window mode. The shipping build produces cond_frame through the pruned decoder in overlap-skip stream mode. The shipping-decoder arm here used window mode.
- **TensorRT.** TensorRT 10.3 was borrowed from another venv. Production needs a proper install. The engine and ONNX files were deleted after timing; their sha256 are in speed.json.
- **Colour-drift gate.** It still needs a regenerated control raw.npy, because every raw.npy on disk was deleted.

