# Square-resolution teeth test: 512 and 1024 at 15 and 25 FPS

Source-detail follow-up: the [controlled 307/256/128/64-pixel experiment](../source_detail_20260917/README.md) fixes generation at 512×512/25 FPS and confirms that insufficient real pixels in the base face crop causes blur. The current evidence favors at least roughly 256 real face-crop pixels; enlarging the 307-pixel crop to a 1024 output did not create detail and caused a separate severe generation failure.

September 17, 2026: fresh GPU inference on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM**, driver **595.84**, Torch **2.7.1+cu128 / CUDA 12.8**. Direct `nvidia-smi` metadata is saved in each `results.json`; first pre-load device use was 2,487 MiB. OmniVoice remained resident. All four tests use an 8,704-MiB Torch allocator cap (not physical VRAM), **BF16, four denoising steps, seed 50, stock audio conditioning, one state, eager execution and staged weight offloading**. No INT8 or TensorRT was used.

## Results and videos

**512×512 is clearly better than 1024×1024 in this tested configuration.** Both 1024 runs develop severe whole-face blur over time, which also destroys dental detail. It is visible in native lossless RGB frames before video encoding. The 512 runs retain recognizable face and tooth-row detail throughout the sampled timeline, although teeth are still imperfect. The 25-FPS 512 clip has different articulation and sometimes more exposed teeth; this small sample does not establish universally sharper teeth at 25 FPS.

Watch the [labeled four-way comparison](soulx-LITE-output-resolution-512-vs-1024-and-15-vs-25fps.mp4), or inspect the [fixed-time contact sheet](frames-comparison.png). Individual outputs retain their requested native resolution and frame rate:

| Video | Output frames | Generation wall | Useful generated FPS | Peak Torch allocated / reserved | Visual observation |
| --- | ---: | ---: | ---: | ---: | --- |
| [512×512, 15 FPS](512-15/video.mp4) | 150 | 7.19 s | 20.87 | 3,836 / 4,188 MiB | Clearer face and tooth row; dental imperfections remain |
| [512×512, 25 FPS](512-25/video.mp4) | 250 | 11.22 s | 22.29 | 3,836 / 4,186 MiB | Clearer face and tooth row; often larger mouth opening |
| [1024×1024, 15 FPS](1024-15/video.mp4) | 150 | 20.67 s | 7.26 | 4,660 / 6,768 MiB | Severe blur develops across the face |
| [1024×1024, 25 FPS](1024-25/video.mp4) | 250 | 32.62 s | 7.67 | 4,659 / 6,856 MiB | Severe blur develops across the face |

All four clips are ten seconds long. Timing starts after loading, warmup, reference preparation and audio append; it includes generation of whole 24-frame chunks including discarded tail padding, while useful FPS counts only the final 150/250 frames. Encoding is outside the timing. These are single-run eager/offload timings, **not the maximum speed of a compiled resident service**, and cannot replace the earlier 320×576 concurrency figures. No concurrent-stream test was performed at these sizes.

## Controlled inputs and important differences from earlier videos

All four references come from the same source `/workspace/benchmarks/same-avatar/shared.png`, SHA-256 `cb7f390757c2d1a5e782a5fe0a0660d5d0fc848d54fc1f6eed72e4a3757feac9`. The original is 384×672. A fixed close square face crop `(38,64,345,371)` is resized to 512 or 1024 using Lanczos. Its 307-pixel width corresponds to approximately 1.25× horizontal magnification relative to the original width. The square composition differs from the previous 320×576 portrait crop; it deliberately keeps the close face without stretching it. Increasing canvas size does not create new real detail in this 307×307 source crop.

Audio is the same ten-second `benchmarks/comparison-10s.wav` fixture. All runs use seed 50, the same stock conditioning (no 0.5 residual hook), and direct PNG input without idle-video reconditioning. Each output receives the same single H264 CRF-18 encode with AAC audio. Seven native RGB PNGs per run at 0.5/1.5/2.5/3.5/4.5/6.5/8.5 seconds preserve pre-encode evidence. Frame selection rounds to the nearest frame; at 15 FPS that can differ from the requested time by 33 ms.

The runner temporarily allows exactly 1024×1024 inside its own process because the service currently limits total output pixels to 576×1024. The production geometry guard was not relaxed. Fitting the expanded shape and completing generation does not qualify its image quality; these 1024 results fail that check. Staged offloading was used consistently for the complete matrix to keep GPU memory bounded beside OmniVoice.

## Interpretation

This experiment rejects the proposed **1024×1024 as a teeth fix for this configuration**. The degradation appears across the face and increases along the sequence, suggesting a resolution-dependent generation/recurrence problem. The exact mechanism was not isolated. A low-detail source can limit achievable detail, but it alone does not explain why the generated face loses so much structure over time. The stock 1024 result should not be promoted based on nominal resolution.

The lossless frames rule out H264 as the sole cause of this 1024 failure. The 512 controls show that eager execution and offloading do not universally cause the same collapse. Further causal testing could inspect VAE round trips and per-chunk feedback at 1024 or compare full-precision/kernel variants; none of those hypotheses is established here.

At fixed 512 resolution, both generation frame rates retain substantially better detail than 1024. Changing FPS changes audio-frame alignment and recurrence, so output mouth poses can differ at the same utterance time. No dental ground truth or validated teeth score is available. Earlier portrait/WebRTC clips additionally differ in aspect ratio, source preparation, execution mode and codec path, so this matrix does not measure a clean percentage improvement over those clips.

## Evidence and reproduction

[Summary and decoded frame counts](summary.json), [packaging script](package.py), and [generation runner](../closer_capacity_20260916/run_square_matrix.py). Each run directory contains exact environment/profile metadata, reference, raw-frame hash, native PNGs and video. Loading/progress logs are retained alongside directories. All 800 video frames decoded, and the comparison video was decoded separately. The 30-FPS comparison layout repeats frames for presentation and downsizes 1024 panels to 512; inspect native videos/PNGs for original spatial detail. It does not imply 30-FPS generation.

Run each case in a fresh process with a new output directory:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/closer_capacity_20260916/run_square_matrix.py \
  --size 512 --fps 15 --output NEW_DIRECTORY
```

Repeat for `(512,25)`, `(1024,15)` and `(1024,25)` sequentially on this shared GPU. Packaging uses `.venv/bin/python benchmarks/square_teeth_20260917/package.py`.
