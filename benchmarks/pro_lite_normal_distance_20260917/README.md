# SoulX FlashHead PRO versus LITE at normal avatar distance

## Result

PRO again renders a substantially sharper face and mouth than LITE at the
original normal-distance framing. Its median whole-mouth edge energy is 184.84
versus 80.65 for LITE (2.29x), and its median central-oral edge energy on
open-mouth frames is 1100.38 versus 294.89 (3.73x).

Visual review agrees with the edge measurements, but the dental result is
mixed. PRO produces much brighter, harder tooth boundaries and removes the
soft gray blur seen in LITE. In several frames, especially around 3.5, 6.5 and
8.5 seconds, those teeth merge into a single bright strip with limited
individual-tooth structure. LITE is blurrier, yet sometimes suggests more
separate tooth divisions. PRO therefore fixes much of the softness bottleneck;
it does not fully fix dental anatomy or tooth segmentation.

The models also generate different mouth motion. PRO's median normalized
opening is 0.0859 versus 0.1041 for LITE, and p95 opening is 0.1867 versus
0.2792. Their opening trajectories correlate at 0.660. This means the videos
are a controlled released-model A/B, not a pixel-matched decoder-only ablation.

## Controlled setup

- Original normal-distance Indian-male reference: `reference-normal.png`, the
  retained 1.00x close reference from the September 16 distance experiment.
- Exact shared ten-second audio: `audio.wav`.
- 320x576, 25 FPS, 250 frames, four denoising steps, seed 50, BF16.
- Eager model and VAE execution for both variants; stock audio conditioning and
  color correction; no mouth-SR, sharpening, refinement, or post enhancement.
- Fresh local inference on NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible
  VRAM, driver 595.84, Torch 2.7.1+cu128, CUDA runtime 12.8. Co-resident GPU
  processes remained active and are recorded in each `results.json`.

| Variant | Generation time | Useful FPS | Peak Torch allocated | Peak Torch reserved |
| --- | ---: | ---: | ---: | ---: |
| LITE | 4.45 s | 56.18 | 4,784 MiB | 5,296 MiB |
| PRO | 35.09 s | 7.12 | 5,397 MiB | 6,620 MiB |

PRO fits the 12-GB card in eager mode, but this profile is about 7.9x slower
than LITE and is not real-time at 25 FPS. These are single-run measurements
under the recorded co-resident load, not deployment capacity claims.

## Interpretation

The normal-distance result matches the closer-reference experiment: the PRO
path's denser spatial latent representation and Wan2.1 VAE preserve much more
high-frequency information. The remaining connected-white-band artifact shows
that spatial detail retention and plausible tooth structure are two separate
problems. A practical SoulX teeth fix likely needs both a denser mouth
representation and a mouth-specific anatomical or identity constraint.

## Evidence

- [LITE video](lite/video.mp4) and [run record](lite/results.json)
- [PRO video](pro/video.mp4) and [run record](pro/results.json)
- [labeled comparison video](soulx-LITE-vs-PRO-indian-man-normal-1.00x-320x576-seed50.mp4)
- [native-pixel mouth crops](mouth-comparison.png)
- [full-frame contact sheet](frames-comparison.png)
- [per-frame diagnostics](diagnostics.json) and [media validation](media-validation.json)

The reusable runner now accepts `--reference` and `--framing`:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/pro_lite_teeth_20260917/run_variant.py \
  --variant lite \
  --reference benchmarks/distance_lipsync/evidence-indian-male-20260916/reference-close.png \
  --framing normal-1.00x \
  --output NEW_LITE_DIRECTORY
```
