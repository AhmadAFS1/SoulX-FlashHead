# Recreation of the 1.50x PRO-vs-LITE comparison video — 2026-09-21

The original `../soulx-LITE-vs-PRO-indian-man-1.50x-320x576-seed50.mp4` and its
`lite/video.mp4` / `pro/video.mp4` sources were missing from the working tree
(only the two `results.json` metric files from the original 2026-09-17 run
survived). This directory is a **fresh recreation**, not a byte-identical
replay: same reference image, same audio, same documented parameters, rerun
today. It exists so there is a real video to look at and commit; it is not
proof that it reproduces the original run's exact output bytes.

## What's unchanged from the original run

- Reference: `../reference-150x.png`, sha256
  `a0a2317609afe70ed6c3fa3c620654dc56b123981429b2316dc6f7cc8222d676` — verified
  identical to the value recorded in `../README.md`.
- Audio: `../audio.wav` (10.000 s, 16 kHz mono PCM16), byte-identical file
  reused from the original benchmark directory.
- Parameters: 320x576, 25 FPS, 250 frames, 4 sampling steps, seed 50, BF16,
  eager execution (`COMPILE_MODEL=False`, `COMPILE_VAE=False`), 1.50x framing,
  stock audio conditioning and color correction, no sharpening/mouth-SR/refinement.
- Checkpoints: same `models/SoulX-FlashHead-1_3B/Model_Lite` and `Model_Pro`
  weights on disk; hashes recorded fresh in each `results.json` this run.

## What's new / different

- Run date: 2026-09-21 (original was 2026-09-17).
- Hardware: **NVIDIA GeForce RTX 4070 SUPER**, physical 12 GB class,
  12,282 MiB visible (`nvidia-smi`), driver **595.84**, same GPU model as the
  original but a separate boot/session — co-resident processes and exact
  thermal/clock state were not reproduced.
- `run_variant.py` here is a copy of the harness with an added `sys.path`
  fix (the copy under `../pro_lite_teeth_20260917/run_variant.py` fails with
  `ModuleNotFoundError: No module named 'flash_head'` when run as a script
  from outside the repo root; this copy inserts the repo root on `sys.path`
  before importing). No inference logic was changed.
- Output layout: `lite/` and `pro/` here are new directories under
  `recreated_20260921/`, separate from the original (still-present)
  `../lite/results.json` and `../pro/results.json`, so the original metric
  files were not overwritten.

## Fresh measured results (this run)

| Variant | Model load | Generation time | Useful FPS |
| --- | ---: | ---: | ---: |
| LITE | 4.22 s | 4.55 s | 54.96 |
| PRO | 2.94 s | 35.02 s | 7.14 |

These closely track the original run's 4.45 s / 56.20 FPS (LITE) and
34.93 s / 7.16 FPS (PRO); the small deltas are consistent with ordinary
run-to-run variance (no compile cache, shared GPU, no fixed clock lock), not a
controlled re-benchmark. Full telemetry (resource sampling, checkpoint hashes,
GPU snapshots) is in each variant's `results.json`.

Video facts: both `lite/video.mp4` and `pro/video.mp4` are H.264/AAC, 320x576,
25 FPS, 250 frames, 10.000 s — matching the original file facts recorded in
`../media-validation.json`.

## Evidence

- [LITE video](lite/video.mp4) and [run record](lite/results.json)
- [PRO video](pro/video.mp4) and [run record](pro/results.json)
- [labeled comparison](soulx-LITE-vs-PRO-indian-man-1.50x-320x576-seed50-recreated.mp4)
- [full-frame contact sheet](frames-comparison.png)
- [media validation](media-validation.json)
- [GPU runner](run_variant.py), [packager](package.py)

## Qualitative check

A single extracted frame at 4.5 s shows the same direction as the original
report: LITE is softer around the mouth/eyes, PRO renders sharper facial and
dental edges. This is one frame, not a re-run of the original's Laplacian edge
energy analysis (`../analyze.py` was not rerun here).
