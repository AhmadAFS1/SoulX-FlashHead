# Ojin-disclosed components on the distant LTX Indian-man video

September 17, 2026. Existing LTX-generated video postprocessed on **NVIDIA
GeForce RTX 4070 SUPER**, physical **12 GB class / 12,282 MiB visible** from
`nvidia-smi`, driver **595.84**, PyTorch **2.7.1+cu128 / CUDA 12.8**, TensorRT
**10.16.1.11 cu13**, MediaPipe **0.10.35**. OmniVoice (2,478 MiB) and idle LTX
(234 MiB) remained resident; initial whole-device use was 2,727 MiB. Direct
snapshots, versions and samples are in [results.json](results.json). GPU SR and
CPU FaceLandmarker are distinguished below; comparison packaging and media
validation are CPU work.

**Result:** the accessible FaceLandmarker + native-2x TensorRT path processed
the more distant LTX clip at **39.61 FPS**, above its 24-FPS playback rate, and
detected the mouth in all **201/201 frames**. It provides a moderate clarity
increase, but does not reproduce the conspicuous individual-tooth enhancement
of the earlier general 4x model. The older model runs at only **7.15 FPS** and
sometimes makes the teeth unnaturally clean, segmented or outlined. Neither is
evidence of Ojin's unavailable mouth-refiner behavior.

## Watch the comparison

[Bicubic versus earlier SR versus FaceLandmarker + native-2x TensorRT](LTX23-distant-indian-man-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner.mp4)

[Pre-encode mouth samples](LTX23-distant-indian-man-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner-mouths.png)
and [full-frame samples](LTX23-distant-indian-man-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner-frames.png)
are taken from identical decoded source-frame indices. The three video columns
are all 960x1664, 24 FPS and preserve the source audio. Individual clips are in
descriptively named subdirectories.

## Source and test contract

The exact source is
`/workspace/LTX-2.3/lumatalk_completed_videos_20260701T025914Z/remake/generated_clips/11_indian_man_speaking.mp4`.
The associated LumaTalk [production notes](../../../LTX-2.3/lumatalk_completed_videos_20260701T025914Z/remake/README.md)
identify the speaking takes as LTX-generated with generated speech, and state
that the source clips remain untouched by the call compositor. The tested file
is H.264/AAC, 480x832, 24 FPS, 201 frames and 8.375 seconds. This experiment
decodes that already-compressed video; it does not rerun LTX or SoulX.

The FaceLandmarker mouth width averages **82.2 pixels** on the 480-pixel-wide
source, with 5th–95th percentiles 65.8–92.1 pixels. This records the tested
scale; it is not a dental-quality metric. The portrait shows the upper torso
and arm-length selfie framing, making it the requested more distant male case
relative to the previous square head/shoulder benchmark.

All learned treatments consume the same decoded RGB frames:

| Column | Processing |
| --- | --- |
| Bicubic 2x | CPU interpolation, 480x832→960x1664 |
| Earlier SR | Public general-x4v3 SRVGG, native 4x then area-resized to 2x; PyTorch FP16 GPU |
| Corrected accessible path | Google Tasks FaceLandmarker 0.10.35 plus public native-2x SRVGG TensorRT 10.16.1.11, full-frame 480x832→960x1664 |

FaceLandmarker supplies mouth coordinates, but no mouth pixels are altered from
those coordinates because Ojin's refiner and routing code are unavailable. The
native-2x model processes the full frame. Its public substitute checkpoint is
Philip Hofmann's `2xNomosUni_compact_multijpg_ldl`, not an Ojin checkpoint.
The exact accessibility boundaries and licenses are documented in the parent
[Ojin component audit](../ojin_components_20260917/README.md).

## Performance

All rows use the RTX 4070 SUPER profile above. `Processing FPS` times the
per-frame operation after input decode and before H.264 encoding. It includes
GPU synchronization and CPU/GPU transfers; the corrected path also includes
FaceLandmarker. The recording wall time includes synchronous software H.264
writing and input decode, so it is a packaging measurement rather than an
optimized streaming pipeline.

| Treatment | Per-frame processing | Processing FPS | Wall including recording |
| --- | ---: | ---: | ---: |
| Bicubic 2x | 1.38 ms | 722.16 | 5.02 s |
| Earlier general 4x SR → 2x | 139.81 ms | 7.15 | 32.97 s |
| FaceLandmarker + native-2x TensorRT | 25.24 ms | 39.61 | 16.83 s |

The corrected accessible postprocess itself fits within the source's 41.67-ms
24-FPS interval. The synchronous CRF-18 software encoder makes this diagnostic
recording slower than real time; no WebRTC/hardware-encoder integration was
tested. A real mouth refiner would add unknown cost. No concurrent-session,
end-to-end latency or production-capacity conclusion follows from this run.

## Resources

Whole-device VRAM includes the 2,727-MiB resident baseline. Both learned models
are loaded and warmed before all rows, so differences between rows do not give
model-loading requirements. Torch counters exclude TensorRT-owned allocations.
Process RSS is host RAM. CPU process values over 100% reflect more than one
logical CPU; host CPU is normalized across 32 logical CPUs.

| Metric | Bicubic | Earlier SR | FaceLandmarker + native-2x TRT |
| --- | ---: | ---: | ---: |
| Sampled whole-device VRAM peak | 3,308 MiB | 3,308 MiB | 3,320 MiB |
| Exact Torch allocated peak | 2 MiB | 200 MiB | 43 MiB |
| Torch reserved peak | 222 MiB | 222 MiB | 222 MiB |
| Process RSS peak | 1,648 MiB | 1,845 MiB | 1,837 MiB |
| Host RAM used peak | 8,271 MiB | 8,488 MiB | 8,417 MiB |
| Sampled GPU use mean / peak | 0% / 0% | 33.4% / 70% | 12.6% / 22% |
| Process CPU mean / peak | 143.6% / 193.4% | 115.4% / 136.6% | 114.5% / 134.2% |
| Host CPU mean / peak | 6.4% / 10.5% | 5.4% / 10.9% | 5.3% / 12.5% |

Samples use alternating 137/193-ms waits plus query time. They may miss short
peaks. The bicubic GPU figure is correctly zero because its timed operation is
CPU-only; the already loaded GPU models account for its device-memory baseline.

## Teeth observations

Seven native PNG moments at 0.5, 1.5, 2.5, 3.5, 4.5, 6.5 and 7.5 seconds were
reviewed at a fixed 224x124 output-pixel mouth window centered from the same
FaceLandmarker coordinates.

- The LTX source already contains separated upper teeth at 1.5 and 3.5 seconds.
  This clip therefore tests preservation/enhancement of existing structure,
  rather than recovery from an entirely featureless tooth band.
- The native-2x substitute adds mild edge definition while staying visually
  close to bicubic. It retains some blur/connected brightness and does not
  reveal a dramatic hidden dental structure.
- The earlier 4x model creates much stronger tooth divisions at 1.5 and 3.5
  seconds. Several divisions are unusually uniform/dark and the surrounding
  face is more processed, so this is likely partly learned invention rather
  than reliable recovery of true anatomy.
- Closed-mouth moments remain stable in the sampled frames. Full temporal
  flicker and phoneme accuracy were not ground-truth scored.

For descriptive contrast only, the sampled mouth-region Laplacian variance
averages 41.5 for bicubic, 101.5 for native-2x and 150.5 for the older SR. Mean
Sobel magnitude is 36.4, 37.3 and 39.3 respectively. These values confirm added
local high-frequency contrast; they do **not** certify correct teeth, temporal
consistency or lip-sync. Exact crop boxes and values are in [analysis.json](analysis.json).

The main conclusion is unchanged but better grounded at the requested distance:
**FaceLandmarker plus a fast native-2x TensorRT upscaler is operationally viable,
but that accessible stack alone is not Ojin's apparent dental solution.** The
missing learned mouth-refinement stage, its exact routing, and Ojin's exact SR
checkpoint remain material unknowns.

## Validation and reproduction

- The shape-matched TensorRT engine was built for exactly 480x832→960x1664 and
  checked against the FP32 PyTorch model on three decoded LTX frames: 70.3–71.8
  dB PSNR, maximum normalized error 0.0036. [Build record](build.json).
- All three individual clips and the labeled comparison fully decode to 201
  frames at 24 FPS with audio. [Media validation](media-validation.json).
- FaceLandmarker detected all 201 frames; per-frame coordinates are retained
  next to the corrected-path clip. Input, engine, task, output-buffer and video
  hashes are recorded. PNGs precede output H.264 encoding.
- [run.py](run.py), [package.py](package.py) and [analyze.py](analyze.py) retain
  the inference, packaging and limited sharpness-analysis workflows. No live
  service or default was changed.
