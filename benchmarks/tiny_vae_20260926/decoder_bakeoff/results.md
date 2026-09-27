# Tiny-decoder bake-off at 576x320 on SoulX DiT latents (2026-09-26)

**Hardware / environment (every GPU number in this file).** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM
(nvidia-smi query recorded in quality.json / speed.json), driver 595.84, torch 2.7.1+cu128 (CUDA runtime 12.8, cuDNN 9.7.1),
TensorRT 10.3.0 python bindings (borrowed read-only from /workspace/.venvs/musetalk_trt_stagewise through the symlinks in
decoder_bakeoff/_trt10_path/). Runs on 2026-09-26: latent dumps 20:41-20:47 UTC, quality 20:47-20:51 UTC, speed 20:51-20:58 UTC.
Co-resident load: none during the quality and speed runs (nvidia-smi listed only this process at the start and after every speed
case); another project's ComfyUI/MuseTalk jobs used the GPU before and between runs, and one of them made the first dump attempt
fail with an out-of-memory error, so that attempt was re-run.
The 2026-09-22 shipping-build numbers are **reused** from benchmarks/pro_30fps_20260922/final-v4-lean-r01/results.json (same GPU model,
fresh local GPU inference at the time); they were not re-run.

All numbers measured here are **fresh local GPU inference on RTX 4070 SUPER**, unless marked as reused or projected.

## Answer and best candidate

* **The SD TAESD decoder itself cannot be used.** It decodes 4-channel single-image SD latents. SoulX uses Wan 2.1 latents
  (16 channels, 8x spatial, 4x causal temporal). The Wan 2.1 version from the same author, **taew2_1 (TAEHV)**, does fit: it takes the
  DiT's normalised latents directly, gives exactly 33 frames per 9-latent window with the upstream 3-frame trim, and can stream across
  windows with carried state (28 frames per 7 latents).
* **Best candidate: taew2_1, fp16, as a TensorRT FP16 engine** (torch.compile of the same core is 9-12 ms slower per call).
  * Speed: 36 ms for the steady-state 7-latent decode and 46 ms for the cold 9-latent decode. The shipping decoder takes 434 ms per
    window in the harness (reused figure), so this is about 12x faster.
  * Arithmetic projection, not measured: about 53 useful FPS instead of 30.2, because the DiT and the motion encode stay as they are.
    5x end to end is out of reach from the decoder alone.
* **The trade is fidelity.** Measured against the stock decoder:
  * PSNR: taew2_1 is 39.6 dB full frame and 35.4 dB on the mouth. The shipping decoder is 49.0 and 44.9 dB, so taew2_1 is about
    9.5 dB lower on both.
  * Sharpness: taew2_1 keeps mouth sharpness at 0.98x the reference; the shipping decoder is at 0.96x.
  * Temporal flicker: taew2_1 is 1.00x the reference.
  * Colour: the offset is at most 0.43/255 per channel in any window.
  * On the PNGs the losses are fine texture: stubble looks grainier and gaps between teeth are softer. Lip contours and teeth are
    preserved.
* **The other candidates are worse:**
  * **lighttaew2_1:** loses about 0.6-0.9 dB, is smoother (flicker 0.83x) and has a 0.7/255 colour shift.
  * **lightvaew2_1:** its PSNR looks similar, but only because it is visibly blurry (sharpness 0.82x, flicker 0.77x). It is also about 4-5x
    slower than taew2_1 (175 ms compiled against 36-45 ms).
* **Not measured yet:** the end-to-end effect. The Wan encoder re-encodes the tiny decoder's colour-corrected output as the next
  window's conditioning, so the lip-sync, edge-ratio and colour-drift gates and a labelled full-run video are still needed before
  shipping.

Inputs: 36 latent windows (16x9x72x40 bf16) from indian150-a, tts-open-vowels, tts-plosives, tts-sibilants (seed 50, measured windows 0-8). The 4 'real latent' windows listed in benchmarks/pro_30fps_20260919/real-inputs-r01/manifest.json no longer exist (its tensors/ directory is empty), so all windows here are fresh DiT latents from decoder_bakeoff/dump_latents.sh (fixtures indian150-a, tts-plosives, tts-open-vowels, tts-sibilants; seed 50; measured windows 0-8 of each run).

Reference: stock Wan 2.1 decoder, bf16 eager, stock weights; it reproduces the in-loop decode of the dump run bit-exactly (max abs diff on a strided checksum: {'indian150-a': 0.0, 'tts-open-vowels': 0.0, 'tts-plosives': 0.0, 'tts-sibilants': 0.0}).

## Conventions (resolved empirically, mean PSNR full frame over all windows)

| candidate | setting | PSNR full dB | PSNR mouth dB | frames |
|---|---|---:|---:|---:|
| taew2_1 latent=norm | temporal offset -3 | 27.44 | 20.51 | 33 |
| taew2_1 latent=norm | temporal offset -2 | 29.18 | 21.77 | 33 |
| taew2_1 latent=norm | temporal offset -1 | 32.36 | 24.80 | 33 |
| taew2_1 latent=norm | temporal offset 0 | 39.63 | 35.38 | 33 |
| taew2_1 latent=norm | temporal offset 1 | 32.18 | 24.59 | 32 |
| taew2_1 latent=norm | temporal offset 2 | 28.94 | 21.50 | 31 |
| taew2_1 latent=unnorm | temporal offset -3 | 18.54 | 15.72 | 33 |
| taew2_1 latent=unnorm | temporal offset -2 | 18.68 | 16.22 | 33 |
| taew2_1 latent=unnorm | temporal offset -1 | 18.89 | 17.06 | 33 |
| taew2_1 latent=unnorm | temporal offset 0 | 19.04 | 17.83 | 33 |
| taew2_1 latent=unnorm | temporal offset 1 | 18.87 | 17.05 | 32 |
| taew2_1 latent=unnorm | temporal offset 2 | 18.61 | 16.17 | 31 |
| taew2_1 output read as [-1,1] (wrong range) | offset 0 | 8.79 | n/a | 33 |
| lighttaew2_1 latent=norm | temporal offset -3 | 19.77 | 18.78 | 33 |
| lighttaew2_1 latent=norm | temporal offset -2 | 19.94 | 19.21 | 33 |
| lighttaew2_1 latent=norm | temporal offset -1 | 20.14 | 19.96 | 33 |
| lighttaew2_1 latent=norm | temporal offset 0 | 20.25 | 20.70 | 33 |
| lighttaew2_1 latent=norm | temporal offset 1 | 20.12 | 19.84 | 32 |
| lighttaew2_1 latent=norm | temporal offset 2 | 19.94 | 19.11 | 31 |
| lighttaew2_1 latent=unnorm | temporal offset -3 | 27.76 | 20.68 | 33 |
| lighttaew2_1 latent=unnorm | temporal offset -2 | 29.36 | 21.96 | 33 |
| lighttaew2_1 latent=unnorm | temporal offset -1 | 32.53 | 25.02 | 33 |
| lighttaew2_1 latent=unnorm | temporal offset 0 | 39.04 | 34.46 | 33 |
| lighttaew2_1 latent=unnorm | temporal offset 1 | 32.56 | 25.00 | 32 |
| lighttaew2_1 latent=unnorm | temporal offset 2 | 29.31 | 21.82 | 31 |
| lighttaew2_1 output read as [-1,1] (wrong range) | offset 0 | 8.75 | n/a | 33 |
| lightvaew2_1 decode(z, wan scale) | temporal offset -2 | 29.41 | 21.89 | 31 |
| lightvaew2_1 decode(z, wan scale) | temporal offset -1 | 32.84 | 25.15 | 32 |
| lightvaew2_1 decode(z, wan scale) | temporal offset 0 | 40.30 | 34.93 | 33 |
| lightvaew2_1 decode(z, wan scale) | temporal offset 1 | 33.14 | 25.41 | 32 |
| lightvaew2_1 decode(z, wan scale) | temporal offset 2 | 29.59 | 22.03 | 31 |
| lightvaew2_1 decode(z, no un-normalisation) | offset 0 | 20.34 | n/a | 33 |

NCTHW (Wan layout) into TAEHV: RuntimeError: Given groups=1, weight of size [256, 16, 3, 3], expected input[16, 9, 72, 40] to have 16 channels, but got 9 channels instead

Chosen: taew2_1: {'latent': 'norm', 'offset': 0, 'trim': 'drop first 3, last 0 of 36 raw frames', 'psnr_full_mean': 39.629373664558926, 'matches_assumed': True}; lighttaew2_1: {'latent': 'unnorm', 'offset': 0, 'trim': 'drop first 3, last 0 of 36 raw frames', 'psnr_full_mean': 39.04003140782282, 'matches_assumed': True}; lightvaew2_1: {'latent': 'normalised, decode(z, wan scale)', 'offset': 0, 'psnr_full_mean': 40.302935284034945}

## Quality, window mode (every window decoded from scratch, all 33 frames)

Ratios are candidate/reference (1.0 = same as the stock decoder); worst = lowest PSNR, ratio farthest from 1, largest |offset|.

| decoder | windows | PSNR full dB (mean / worst) | PSNR mouth dB (mean / worst) | sharpness full (mean / worst) | sharpness mouth (mean / worst) | flicker ratio (mean / worst) | colour offset R,G,B /255 (mean / worst abs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| shipping decoder (skip 9,10,13,14 + ft4), bf16 eager | 36 | 49.01 / 48.43 | 44.85 / 43.47 | 0.956 / 0.949 | 0.957 / 0.938 | 0.962 / 0.944 | 0.04, 0.00, -0.03 / 0.07, 0.02, 0.04 |
| taew2_1 (TAEHV), fp16 | 36 | 39.63 / 38.95 | 35.38 / 33.99 | 0.961 / 0.949 | 0.979 / 0.956 | 1.000 / 1.065 | -0.27, 0.24, 0.33 / 0.39, 0.30, 0.43 |
| lighttaew2_1 (LightTAE), fp16 | 36 | 39.04 / 37.86 | 34.46 / 32.59 | 0.940 / 0.927 | 0.943 / 0.924 | 0.826 / 0.779 | 0.68, 0.72, 0.35 / 0.86, 0.85, 0.50 |
| lightvaew2_1 (LightVAE, WanVAE_ dim=24), bf16 | 36 | 40.30 / 38.76 | 34.93 / 32.91 | 0.817 / 0.799 | 0.835 / 0.795 | 0.769 / 0.699 | 0.03, -0.15, -0.27 / 0.14, 0.26, 0.54 |
| taew2_1 (TAEHV), bf16 | 36 | 39.51 / 38.83 | 35.32 / 33.91 | 0.962 / 0.951 | 0.980 / 0.958 | 1.003 / 1.069 | -0.29, 0.30, 0.47 / 0.41, 0.36, 0.57 |
| lighttaew2_1 (LightTAE), bf16 | 36 | 39.06 / 37.88 | 34.46 / 32.59 | 0.942 / 0.929 | 0.945 / 0.926 | 0.829 / 0.783 | 0.66, 0.69, 0.14 / 0.85, 0.82, 0.29 |

## Quality, stream mode (overlap-skip analogue; 28 delivered frames per window)

| decoder | windows | PSNR full dB (mean / worst) | PSNR mouth dB (mean / worst) | sharpness full (mean / worst) | sharpness mouth (mean / worst) | flicker ratio (mean / worst) | colour offset R,G,B /255 (mean / worst abs) |
|---|---:|---:|---:|---:|---:|---:|---:|
| stock Wan decoder, stream (overlap-skip analogue) | 36 | 48.31 / 41.49 | 48.12 / 40.29 | 0.990 / 0.985 | 0.983 / 0.978 | 1.034 / 1.076 | -0.02, 0.03, 0.59 / 0.14, 0.13, 0.76 |
| shipping decoder (skip 9,10,13,14 + ft4), bf16 eager | 36 | 42.38 / 41.12 | 40.79 / 39.21 | 0.945 / 0.938 | 0.941 / 0.917 | 0.995 / 0.971 | 0.03, 0.02, 0.51 / 0.10, 0.10, 0.67 |
| lightvaew2_1 (LightVAE, WanVAE_ dim=24), bf16 | 36 | 38.69 / 37.34 | 34.23 / 32.29 | 0.804 / 0.790 | 0.822 / 0.781 | 0.800 / 0.750 | 0.34, 0.22, 0.21 / 0.55, 0.51, 0.52 |
| taew2_1 (TAEHV), fp16 | 36 | 38.52 / 37.84 | 34.76 / 33.34 | 0.952 / 0.943 | 0.961 / 0.936 | 1.017 / 1.059 | -0.38, 0.09, 0.59 / 0.56, 0.22, 0.79 |
| lighttaew2_1 (LightTAE), fp16 | 36 | 37.87 / 36.84 | 33.91 / 32.07 | 0.931 / 0.918 | 0.929 / 0.908 | 0.831 / 0.786 | 0.41, 0.55, 0.64 / 0.59, 0.72, 0.88 |

## Speed (median of timed calls, CUDA events; decode contract in/out included)

| case | median ms | min ms | max ms | peak alloc over resident MiB | frames | first call s |
|---|---:|---:|---:|---:|---:|---:|
| taehv fp16 eager nchw cold9 | 90.03 | 89.80 | 90.32 | 3790 | 33 | n/a |
| taehv fp16 eager nchw warm7 | 70.62 | 70.47 | 70.84 | 2924 | 28 | n/a |
| taehv fp16 torch.compile nchw cold9 | 58.48 | 58.18 | 58.93 | 1824 | 33 | 6.1 |
| taehv fp16 torch.compile nchw warm7 | 45.29 | 45.15 | 45.71 | 1398 | 28 | 5.2 |
| taehv fp16 eager channels_last cold9 | 85.30 | 85.03 | 85.66 | 2981 | 33 | n/a |
| taehv fp16 eager channels_last warm7 | 65.63 | 65.52 | 65.78 | 2294 | 28 | n/a |
| taehv fp16 torch.compile channels_last cold9 | 58.14 | 58.02 | 58.38 | 2069 | 33 | 3.1 |
| taehv fp16 torch.compile channels_last warm7 | 44.90 | 44.68 | 44.96 | 1585 | 28 | 3.1 |
| taehv bf16 eager nchw cold9 | 89.40 | 89.29 | 89.58 | 3790 | 33 | n/a |
| taehv bf16 eager nchw warm7 | 69.51 | 69.42 | 69.67 | 2924 | 28 | n/a |
| taehv bf16 torch.compile nchw cold9 | 57.73 | 57.60 | 58.08 | 1822 | 33 | 5.2 |
| taehv bf16 torch.compile nchw warm7 | 44.97 | 44.73 | 45.37 | 1397 | 28 | 5.1 |
| taehv bf16 eager channels_last cold9 | 84.88 | 84.54 | 85.01 | 2980 | 33 | n/a |
| taehv bf16 eager channels_last warm7 | 65.36 | 65.21 | 65.44 | 2294 | 28 | n/a |
| taehv bf16 torch.compile channels_last cold9 | 57.57 | 56.96 | 57.73 | 2069 | 33 | 3.0 |
| taehv bf16 torch.compile channels_last warm7 | 44.67 | 44.49 | 44.80 | 1586 | 28 | 2.9 |
| taehv fp16 TensorRT cold9 | 46.29 | 46.12 | 46.44 | 70 | 33 | n/a |
| taehv fp16 TensorRT warm7 | 36.11 | 36.01 | 36.20 | 59 | 28 | n/a |
| lightvaew2_1 bf16 eager cold9 | 263.57 | 263.30 | 263.97 | 291 | 33 | 0.5 |
| lightvaew2_1 bf16 eager warm7 | 225.57 | 224.96 | 225.94 | 285 | 28 | 0.3 |
| lightvaew2_1 bf16 torch.compile cold9 | 208.00 | 207.20 | 208.13 | 867 | 33 | 44.6 |
| lightvaew2_1 bf16 torch.compile warm7 | 175.38 | 175.11 | 175.66 | 1079 | 28 | 32.9 |
| shipping (pruned+ft4) bf16 eager cold9 | 819.64 | 818.82 | 821.55 | 1054 | 33 | 2.0 |
| shipping (pruned+ft4) bf16 eager warm7 | 693.27 | 692.08 | 694.64 | 1048 | 28 | 1.0 |
| shipping (pruned+ft4) bf16 torch.compile cold9 | 770.95 | 768.86 | 772.73 | 2303 | 33 | 39.2 |
| shipping (pruned+ft4) bf16 torch.compile warm7 | 645.40 | 643.70 | 646.66 | 2737 | 28 | 29.5 |
| stock Wan 2.1 bf16 eager cold9 | 1628.75 | 1627.15 | 1630.75 | 1054 | 33 | 1.6 |
| stock Wan 2.1 bf16 eager warm7 | 1383.99 | 1383.05 | 1385.06 | 1048 | 28 | 1.8 |
| taehv fp16 TensorRT cold9 (side stream) | 46.01 | 45.88 | 46.13 | 70 | 33 | n/a |
| taehv fp16 TensorRT warm7 (side stream) | 35.87 | 35.77 | 35.99 | 59 | 28 | n/a |

Shipping harness decode (reused, fresh local GPU inference, 2026-09-22): 433.9 ms/window average (FP16 TensorRT spans, overlap-skip; 1 cold + 8 warm windows), 30.22 useful FPS.

## End-to-end projection (arithmetic, NOT measured)

Shipping generation 8.272 s minus its decode total 3.906 s = 4.367 s, plus the candidate's cold9 + 8 x warm7 medians:

| decoder | projected generation s | projected useful FPS |
|---|---:|---:|
| shipping (measured 2026-09-22) | 8.272 | 30.22 |
| taew2_1 fp16 TensorRT | 4.702 | 53.17 |
| taew2_1 fp16 torch.compile | 4.788 | 52.22 |
| taew2_1 fp16 eager channels_last | 4.977 | 50.23 |
| lightvaew2_1 bf16 torch.compile | 5.978 | 41.82 |

## TensorRT build

```
{
 "T9": {
  "onnx": "benchmarks/tiny_vae_20260926/decoder_bakeoff/_trt/taehv_core_taew2_1_fp16_T9.onnx",
  "engine": "benchmarks/tiny_vae_20260926/decoder_bakeoff/_trt/taehv_core_taew2_1_fp16_T9.engine",
  "onnx_bytes": 19717477,
  "engine_bytes": 21264692,
  "export_s": 0.1857767105102539,
  "build_s": 19.33615517616272,
  "opset": 18,
  "tensorrt": "10.3.0",
  "maxabs_vs_eager_fp16": 0.0166015625,
  "psnr_u8_vs_eager_fp16": 59.77082118377764,
  "onnx_sha256": "ebcde8772440f745ac726f6935141deb733f12ec652ab029315aee49a8c46622",
  "engine_sha256": "e7288f81e157ee6f2577357ef09cd0b372f5b10887249b490e4acb4e6fd69bc5",
  "deleted_after_measurement": true
 },
 "T7": {
  "onnx": "benchmarks/tiny_vae_20260926/decoder_bakeoff/_trt/taehv_core_taew2_1_fp16_T7.onnx",
  "engine": "benchmarks/tiny_vae_20260926/decoder_bakeoff/_trt/taehv_core_taew2_1_fp16_T7.engine",
  "onnx_bytes": 19717477,
  "engine_bytes": 21239732,
  "export_s": 0.13965153694152832,
  "build_s": 15.110573530197144,
  "opset": 18,
  "tensorrt": "10.3.0",
  "maxabs_vs_eager_fp16": 0.01171875,
  "psnr_u8_vs_eager_fp16": 59.85623236642016,
  "onnx_sha256": "d2b3993a7229a498eb271be772ac988551ec0d02509a67766f82bc10e92a1dd6",
  "engine_sha256": "45ca9e3f72208c7220a6a6dc1cc94b4515510bebd28f51b18925e3c5a40d59e3",
  "deleted_after_measurement": true
 }
}
```

## Visuals

- `benchmarks/tiny_vae_20260926/decoder_bakeoff/visual/grid_indian150-a_w4_f24.png`
- `benchmarks/tiny_vae_20260926/decoder_bakeoff/visual/mouth3x_indian150-a_w4_f24.png`
- `benchmarks/tiny_vae_20260926/decoder_bakeoff/visual/sbs_indian150-a_w4_33f_10fps.mp4`
- `benchmarks/tiny_vae_20260926/decoder_bakeoff/visual/grid_tts-open-vowels_w6_f7.png`
- `benchmarks/tiny_vae_20260926/decoder_bakeoff/visual/mouth3x_tts-open-vowels_w6_f7.png`

## What the PNGs show (inspected by eye)

* **indian150-a w4 f24** (lips parted, a sliver of teeth):
  * **Shipping:** indistinguishable from the reference at 4x.
  * **taew2_1:** same lip outline, teeth sliver and mouth-corner shape. The moustache and stubble texture is slightly grainier,
    with faint blotchy or blocky micro-texture, and the skin is a touch lighter. There is no smearing.
  * **lighttaew2_1:** lips a little more saturated and pink, skin smoother, stubble detail reduced.
  * **lightvaew2_1:** clearly soft. Stubble is smeared, lip edges are blurred, and there is a slight purple cast in the mouth
    opening.
* **tts-open-vowels w6 f7** (open mouth, upper teeth visible):
  * **Reference and shipping:** crisp teeth with visible gaps between them.
  * **taew2_1:** the teeth row is there, but the individual gaps are less distinct. Lip edges stay sharp, and the stubble is
    grainier.
  * **lighttaew2_1:** teeth softer and skin smoother.
  * **lightvaew2_1:** softest of all. The teeth are still readable, but the lip contour is blurred.
* **Across both frames:** apart from taew2_1's faint blotchy micro-texture, no candidate shows a regular 8x8 latent-grid
  pattern, colour banding, or a visible global colour shift at 1x.
* **mp4 spot check:** I did not watch the mp4 as a video. I looked at frame 20 and at a strip of mouth crops over delivered frames
  12-19 extracted from it (indian150-a w4, smile with teeth).
  * **taew2_1:** keeps the structure of individual teeth, but with speckled dark gaps and grain, strongest on frame 12. The grain
    looked stable from frame to frame in that strip, with no obvious shimmer.
  * **lightvaew2_1:** merges the teeth into a soft white band.
  * **lighttaew2_1:** in between.
  * **Shipping:** matches the reference.


## Caveats and what was not measured

* **Missing inputs:** the 4 "real latent" windows (real-inputs-r01/tensors) no longer exist, so none were used.
* **The shipping build cannot currently be reproduced on this machine.**
  * `/workspace/experiments/pro30-deps` (the SM89 SageAttention-2 build) and `/workspace/experiments/ojin-components-deps`
    (TensorRT 10.9) are gone.
  * scratchpad/hires_dump.sh would therefore fail: its policy needs sage2 attention and the trt_stage_compile decoder.
  * The latents were dumped instead with decoder_bakeoff/dump_latents.sh and policies/dump_flash2_eagerdec.json. That policy keeps
    the same DiT quantisation but uses FlashAttention-2 and the stock bf16 eager decoder in the loop, with no overlap-skip.
  * The latents are genuine SoulX DiT outputs, but with flash2 instead of sage2 attention.
* **Shipping decoder numbers:** here it runs in eager bf16. The shipping build runs the same module layout as FP16 TensorRT
  spans, so its in-build fidelity can differ slightly from the rows above.
* **Metric definitions:**
  * Metrics are on uint8 frames from the pipeline's lean-delivery arithmetic, without colour correction. The pipeline's
    per-frame colour matching would largely remove the small global colour offsets.
  * "Sharpness" is the mean absolute luma gradient ratio. It is not review.py's oral edge ratio.
* **The motion re-encode stays on the Wan encoder in this study.** A tiny decoder changes the pixels that are re-encoded into the
  next window's DiT conditioning. That feedback effect needs an end-to-end harness run with the gates.
* **Stream mode:** the stream-mode rows emulate overlap-skip with each decoder's own carried state. The stock decoder itself scores
  48.3 dB mean and 41.5 dB worst in that mode, because the carried cache differs from the re-encoded history. The shipping decoder
  scores 42.4 dB. So part of every stream-mode gap is the overlap-skip mechanism, not the decoder.
* **TensorRT memory:** the peak-memory column for TensorRT excludes the engine's activation workspace (1,721 MiB for cold9 and
  1,339 MiB for warm7, allocated through torch at context creation).

