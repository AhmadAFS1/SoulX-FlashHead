# Ojin-disclosed components on the exact SoulX batch-five character

September 17, 2026. Postprocessing ran on **NVIDIA GeForce RTX 4070 SUPER**, physical
**12 GB class / 12,282 MiB visible** from `nvidia-smi`, driver **595.84**, PyTorch
**2.7.1+cu128 / CUDA 12.8**, TensorRT **10.16.1.11 cu13**, and MediaPipe **0.10.35**.
OmniVoice used 2,478 MiB and idle LTX used 234 MiB before the run; initial whole-device
use was 2,727 MiB. GPU super-resolution and CPU FaceLandmarker are distinguished below.

**Corrected test:** this run uses the exact SoulXFlashHead receiver recording and character
from `bf16-batch5-c5-recorded-peer0.mp4`. It does not use an LTX-generated talking clip as
the input. The earlier LTX test resulted from a misunderstanding and is supplementary only.

## Watch

[SoulX Lite: bicubic versus older SR versus FaceLandmarker + native-2x TensorRT](SoulX-LITE-default-closeup-character-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner.mp4)

[Speaking-mouth samples](SoulX-LITE-default-closeup-character-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner-mouths.png)
and [full-frame samples](SoulX-LITE-default-closeup-character-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner-frames.png)
come from identical decoded frame indices before output H.264 encoding.

## Exact SoulX provenance

The input is the existing [five-call SoulX receiver recording](../concurrency_15fps_20260916/bf16-batch5-c5-recorded-peer0.mp4):
H.264/AAC, 320x576, 15 FPS, 233 frames, and 15.533 seconds. Its retained run record identifies
avatar `default`, SoulX Lite/BF16, four denoising steps, seed 50 for peer 0, and the shared
ten-second `comparison-10s.wav` speech input. The avatar was backed by:

`/workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/idle_active_listening.mp4`

Its SHA-256 is `099877cef231ce12dede03843c558d10c2fa1e9c4e054c83be595e81a00f6ae4`,
matching the original batch-five evidence. This experiment postprocesses the actual received
SoulX video, including receiver H.264 compression; it does not rerun LTX inference.

## Result

FaceLandmarker found the mouth in **233/233 frames**. Mouth width averaged **57.6 source
pixels** (5th–95th percentile 49.7–64.6) on the 320-pixel-wide SoulX output.

| Treatment | Per-frame processing | Processing FPS | Wall including recording |
| --- | ---: | ---: | ---: |
| Bicubic 2x | 0.59 ms | 1,687.07 | 2.25 s |
| Earlier general 4x SR, resized to 2x | 58.38 ms | 17.13 | 15.91 s |
| FaceLandmarker + native-2x TensorRT | 14.96 ms | **66.85** | 9.42 s |

Processing FPS excludes input decode and output H.264 encoding, and includes synchronization,
transfers, and FaceLandmarker for the corrected path. The native-2x path exceeds the source's
15-FPS cadence. The older SR only narrowly exceeds it before encoding and leaves little room
for SoulX generation or concurrent sessions. TensorRT warned that the adapter uses the default
CUDA stream, so 66.85 FPS is the measured implementation rather than an optimized ceiling.

## Teeth finding

The six sampled speaking moments show the same core result more clearly on the requested SoulX
character:

- Bicubic preserves SoulX's soft, merged upper-tooth band.
- The public native-2x TensorRT substitute adds local contrast and cleaner face edges but leaves
  tooth boundaries close to the bicubic result. It is **not a visible teeth fix**.
- The older general SR produces conspicuous individual tooth lines, especially at 2.5 and 6.0
  seconds. Some lines look evenly repeated, outlined, or brace-like, indicating hallucinated
  texture rather than reliable recovery of dental anatomy.

Mean sampled mouth-region Laplacian variance was 11.8 for bicubic, 18.8 for the older SR,
and 24.5 for native-2x. Mean Sobel magnitude was 26.7, 27.5, and 26.8. These measure local
contrast and cannot establish correct teeth, lip-sync, or temporal stability. Exact values are
in [analysis.json](analysis.json).

The accessible Ojin components therefore remain useful infrastructure, but **FaceLandmarker plus
full-frame native-2x SR does not reproduce Ojin's reported Lite teeth result**. FaceLandmarker only
provides coordinates in this test. Ojin's unavailable learned mouth refiner, its routing code,
and exact SR checkpoint remain the leading missing components.

The [custom-refiner feasibility analysis](../mouth_refiner_feasibility_20260917/README.md)
details a possible independent training path. This experiment does not isolate the proprietary
refiner as the sole cause of Ojin's quality: his exact SR weights and orchestration are also
unknown. Generic upscaler failure establishes failure of these tested substitutes.

## Resource measurements

Whole-device VRAM includes the 2,727-MiB resident baseline. Both learned SR models were loaded and
warmed before all rows, so row-to-row VRAM differences do not measure model-loading requirements.
Torch counters exclude TensorRT-owned memory. CPU percentages above 100% span multiple logical CPUs.

| Metric | Bicubic | Earlier SR | FaceLandmarker + native-2x TRT |
| --- | ---: | ---: | ---: |
| Sampled whole-device VRAM peak | 3,142 MiB | 3,142 MiB | 3,154 MiB |
| Torch allocated peak | 2 MiB | 94 MiB | 21 MiB |
| Torch reserved peak | 118 MiB | 118 MiB | 118 MiB |
| Process RSS peak | 1,555 MiB | 1,649 MiB | 1,675 MiB |
| Host RAM used peak | 8,173 MiB | 8,271 MiB | 8,256 MiB |
| Sampled GPU use mean / peak | 0% / 0% | 25.1% / 34% | 10.7% / 13% |
| Process CPU mean / peak | 165.4% / 201.5% | 129.5% / 143.4% | 119.0% / 134.8% |
| Host CPU mean / peak | 9.6% / 11.6% | 7.5% / 11.9% | 6.8% / 11.2% |

The alternating 137/193-ms sampler can miss short peaks. The bicubic timed operation is CPU-only;
already-loaded models account for its device-memory baseline.

## Validation

- The shape-specific TensorRT engine is fixed at 320x576 input and 640x1152 output. Against the
  FP32 PyTorch model on three frames from this SoulX recording, it measured 73.8–74.4 dB PSNR,
  with maximum normalized error 0.00572. See [build.json](build.json).
- All three individual videos and the labeled comparison fully decode to 233 frames at 15 FPS
  with audio. See [media-validation.json](media-validation.json).
- [results.json](results.json) retains versions, hashes, all resource samples, exact output hashes,
  and per-frame mouth coordinates. [run.py](run.py), [package.py](package.py), and
  [analyze.py](analyze.py) retain the reproducible workflows.
