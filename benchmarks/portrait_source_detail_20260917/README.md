# Shoulder-visible 1.25× portrait: source-resolution isolation

September 17, 2026: fresh GPU inference on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM**, driver **595.84**, Torch **2.7.1+cu128 / CUDA 12.8**. This uses the exact earlier 320×576 [1.25× Indian-male reference](../distance_lipsync/evidence-indian-male-closer-20260916/reference-closer-125.png), which retains the shoulders and upper torso. Each run records a direct `nvidia-smi` snapshot. Co-resident load changed during collection and is detailed below; no unrelated process was stopped.

## Result

**The shoulder-visible composition confirms the source-detail effect.** Native 320×576 is the strongest overall reference. A 256×461 treatment remains visually close. A 128×230 treatment is usable but softer. At 64×115, the entire face and teeth become visibly blurred. Nominally enlarging the original to 640×1152 or 1280×2304 does not improve teeth because it adds no captured information; the extra Lanczos round trip slightly softens mouth detail in this run.

Watch the [six-way comparison video](soulx-LITE-portrait-source-width-320-256-128-64-640-1280-output320x576.mp4), the [lossless full-frame contact sheet](frames-comparison.png), or the [aligned mouth crops](mouth-comparison.png).

Focused review: [three-way 1280 / 256 / native 320 comparison](soulx-LITE-portrait-source-width-1280-vs-256-vs-native320.mp4). It is synchronized at 25 FPS with the shared audio and presents the requested order from left to right.

| Intermediate reference size | Video | Mouth-region edge energy | Opening trajectory vs native | Quality reading |
| --- | --- | ---: | ---: | --- |
| 1280×2304, upscaled | [video](int8-detail-1280/video.mp4) | 47.53 | 0.952 | No new detail; slightly softer than native |
| 640×1152, upscaled | [video](int8-detail-640/video.mp4) | 47.71 | 0.949 | No new detail; slightly softer than native |
| **320×576, native** | [video](int8-detail-320/video.mp4) | **58.69** | 1.000 | Best retained detail overall |
| 256×461 | [video](int8-detail-256/video.mp4) | 51.89 | 0.958 | Close to native; minor softening |
| 128×230 | [video](int8-detail-128/video.mp4) | 40.94 | 0.963 | Usable, visibly softer |
| 64×115 | [video](int8-detail-64/video.mp4) | 19.44 | 0.915 | Clear whole-face and dental degradation |

The edge metric is median Laplacian variance over a 96×52 mouth crop. It includes lips, beard, pose and H264 effects; it is supporting evidence rather than a validated teeth score. MediaPipe detected all 1,500 decoded frames. The native images at seven fixed utterance times also show the pattern before H264 encoding.

## Controlled settings

All six runs hold generation at 320×576 and 25 FPS for ten seconds, using four denoising steps, seed 50, the same audio, stock conditioning, one state, eager execution and the same recording path. They use the INT8-storage/BF16-compute compact profile and a 3,584-MiB Torch allocator cap. This matches the weight-storage mode used in the original distance experiments more closely than the later BF16 concurrency service. INT8 storage is not integer matrix multiplication.

For downsample treatments, the retained 320×576 PNG is reduced with Lanczos, then restored to 320×576 before model ingestion. Upscale treatments create 640/1280-wide PNGs with Lanczos; engine preprocessing returns them to the fixed 320×576 generation canvas. Upscaling changes samples through interpolation but cannot add real facial detail.

An initial BF16/staged native-control attempt failed during model loading because another local LTX service temporarily occupied about 5,180 MiB in addition to OmniVoice. Its zero-sample failure JSON/log are retained in `detail-320/`. The six successful INT8-storage runs form the matched quality matrix. The other process later reduced its allocation; it was never stopped.

## Throughput caveat

The native row started with 7,705 MiB whole-device use and measured 19.90 useful FPS. The 256 row started at 6,937 MiB and measured 30.96 FPS. The 128/64 rows started at 2,487 MiB, while 640/1280 started at 2,675 MiB; those four measured 31.07–31.12 FPS. This changing co-resident workload invalidates a cross-row throughput comparison. Source-resolution preprocessing itself happens outside the generation timing and does not change output tensor dimensions. These runs answer the quality question, not serving capacity.

## Combined conclusion

This rules out nominal file resolution as the main variable. The useful variable is **real facial detail after framing and preprocessing**:

- Showing shoulders is compatible with reasonable teeth when the native 320×576 reference is sharp enough.
- Moving closer helps because the face/mouth occupy more of the available real pixels and generated canvas.
- Upscaling an existing portrait to a larger file does not improve dental detail.
- Reducing the source far enough—especially to 64 pixels wide for this full portrait—causes obvious blur.
- The model still invents teeth, so native/high-detail input reduces one failure source but cannot guarantee anatomically clean teeth.

The practical choice among these treatments is the original native 320×576 file. The 256-wide reference is an acceptable lower-detail fallback. A truly higher-resolution original photo of the same composition would be needed to test whether additional captured detail beyond the current source improves output; these 640/1280 rows are interpolation controls only.

## Evidence and reproduction

[Run summary and media validation](summary.json), [CPU diagnostics](diagnostics.json), [runner](../closer_capacity_20260916/run_portrait_source_detail.py), [packager](package.py), and [analysis](analyze.py). Each successful directory contains its treatment reference, result metadata, raw-frame hash, seven lossless RGB frames, video and log.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/closer_capacity_20260916/run_portrait_source_detail.py \
  --detail-width 320 --output NEW_DIRECTORY
```

Repeat with 1280, 640, 256, 128 and 64 in fresh processes.
