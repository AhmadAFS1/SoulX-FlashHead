# Smiling versus neutral reference at 320 pixels

2026-09-17: fresh local GPU inference on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM (12 GB model)**, driver **595.84**, Torch **2.7.1+cu128 / CUDA 12.8**. Evidence: direct `nvidia-smi` snapshots in each run's `results.json`. The 3,584 MiB Torch allocator cap is not physical VRAM. Existing OmniVoice and LTX processes remained resident. Imagegen reference editing ran externally on unverified hardware; video packaging and validation ran on CPU.

## Watch

[Side-by-side comparison with audio](soulx-LITE-neutral-vs-smiling-reference-indian-man-1.25x-seed50.mp4): neutral on the left, smiling on the right. Each generated portrait is displayed at native 320×576 above a fixed 128×80 mouth region enlarged 2.5× with nearest-neighbor interpolation. Both columns show the same audio timestamp. No enhancement or sharpening was applied.

[Reference pair](references.png) · [Seven matched mouth samples](mouth-comparison.png) · [Full-frame samples](frames-comparison.png)

## Visual result

The smiling reference produces more exposed, brighter teeth in several sampled moments, especially 0.5 and 1.5 seconds. It also biases the face toward a wider smile. **It does not clearly solve tooth blur:** both runs still merge individual teeth into a soft bright strip in multiple samples, and the smiling run is not uniformly sharper. Around 2.5 seconds both correctly make a rounded open-mouth shape without visible teeth. At 4.5 seconds mouth opening differs noticeably between references despite identical audio.

This is a promising reference choice for appearance, but this single paired seed does not establish a reliable improvement in dental detail or lip-sync accuracy. These are visual observations, not a validated quantitative teeth score.

## Matched settings and limitations

Both runs were generated fresh: 10 seconds, 250 frames, 25 FPS, 320×576 source and output, seed 50, four denoising steps, stock conditioning, compact memory mode, eager execution, INT8 weight storage with BF16 compute. The audio and profile equality are checked by `package.py`; hashes are retained in run metadata. INT8 storage is not integer matrix multiplication.

The neutral input is the exact native shoulder-visible 1.25× Indian-man reference from the prior experiment. Imagegen edited it into a relaxed smile with visible upper teeth; its output was resized once with Lanczos to 320×576. The generated edit also changes texture and some facial details, so expression/teeth visibility is not perfectly isolated. See [generation provenance](reference-provenance.json). No historical video was reused.

| Reference | GPU | Initial device use | Generation time | Useful FPS |
| --- | --- | ---: | ---: | ---: |
| Neutral | RTX 4070 SUPER / 12,282 MiB | 2727 MiB | 8.01 s | 31.23 |
| Smiling | RTX 4070 SUPER / 12,282 MiB | 2727 MiB | 8.03 s | 31.15 |

Timing excludes model load, warmup, and encoding. These are quality comparisons under co-resident load, not deployment capacity measurements.

## Validation and reproduction

Both individual videos decode to exactly 250 frames at 320×576 and 25 FPS. The comparison decodes without errors, contains 250 frames at 640×856 and 25 FPS, and includes the shared audio. Seven lossless pre-encoding frames are retained per run. [Complete metadata](summary.json).

From the repository root, use fresh output directories:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/smile_reference_320_20260917/run.py --reference benchmarks/smile_reference_320_20260917/neutral-reference.png --output NEW_NEUTRAL_DIRECTORY
PYTHONPATH=. .venv/bin/python benchmarks/smile_reference_320_20260917/run.py --reference benchmarks/smile_reference_320_20260917/smiling-reference.png --output NEW_SMILING_DIRECTORY
```

The retained `package.py` assembles the `neutral/` and `smiling/` directories and validates media and matched settings.
