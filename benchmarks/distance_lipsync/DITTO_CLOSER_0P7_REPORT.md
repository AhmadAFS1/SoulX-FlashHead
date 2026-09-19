# Ditto: three framings at vad_alpha 0.7

GPU inference, September 16, 2026 UTC: **NVIDIA GeForce RTX 4070 SUPER**, 12 GB class / 12,282 MiB visible, driver 595.84; Torch 2.5.1+cu121, CUDA 12.1, TensorRT Ampere Plus backend. `nvidia-smi` reported 2,487 MiB already allocated before model load; workload identity was not verified. These are fresh offline renders, not a WebRTC or minimum-hardware test. [Direct run provenance](evidence-ditto-closer-0p7-20260916/results.json).

## Watch

- [New 0.7 three-framing video](evidence-ditto-closer-0p7-20260916/ditto-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-vad-alpha0p7.mp4).
- [0.5 versus 0.7 comparison](evidence-ditto-closer-0p7-20260916/ditto-indian-man-vad-alpha0.5-vs-0.7-three-framings.mp4): top row 0.5, bottom row 0.7; columns 1.00×, 1.25×, 1.50×; one shared audio track.
- Individual clips: [1.00×](evidence-ditto-closer-0p7-20260916/close-100-seed-50.mp4), [1.25×](evidence-ditto-closer-0p7-20260916/closer-125-seed-50.mp4), [1.50×](evidence-ditto-closer-0p7-20260916/closest-150-seed-50.mp4).

## Result

Increasing `vad_alpha` from 0.5 to 0.7 increased measured mouth opening at every framing. Zooming still made little difference to normalized mouth motion. This confirms a useful amplitude adjustment, **not** better phoneme accuracy or a demonstrated cure for stiffness. User review of the new videos is still needed for naturalness.

| Framing | Mean opening, 0.5 → 0.7 | Mean increase | p95 increase | 0.5/0.7 trajectory correlation | 0.7 correlation with 1.00× |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.00× | 0.04370 → 0.06097 | 39.5% | 26.5% | 0.971 | baseline |
| 1.25× | 0.04353 → 0.06009 | 38.0% | 28.1% | 0.970 | 0.995 |
| 1.50× | 0.04447 → 0.06110 | 37.4% | 28.2% | 0.968 | 0.993 |

Metric: MediaPipe inner-lip distance (13–14) divided by eye-corner distance (33–263). Larger displayed pixels alone do not increase this normalized measure. All correlations above are at zero frame lag; the framing lag scan also preferred zero. Removing the final 15 fade frames still gives mean increases of 39.1%, 37.6%, and 36.9%. Thus the measured increase is not solely a fade artifact. [Full per-frame measurements](evidence-ditto-closer-0p7-20260916/mouth-motion.json), [paired calculations and validation](evidence-ditto-closer-0p7-20260916/alpha-comparison.json).

## What changed and what did not

Only the requested expression blend changed from the retained 0.5 condition. Same Indian-male reference files, audio bytes, seed 50, model/configuration, 320×576 output, 25 FPS, 250 frames, and source-return fade over the final 15 frames. Input hashes and configuration identity match; the comparison script checks them. The runtime log confirms crop scale 2.3, vertical ratio −0.125, motion smoothing window 3, 50 sampling steps, and offline mode. See [source and settings audit](evidence-ditto-closer-0p7-20260916/source-audit.json).

In this checkout, `ctrl_vad` calculates `expression = generated * alpha + source * (1 - alpha)`. It constructs a lip mask but does not use that mask in the assignment: **the whole expression vector is blended**. Here 0.7 retains 70% of the generated expression, versus 50% previously. This is a fixed expression blend, not a changed speech-detection threshold, and not SoulX's audio cross-attention strength control. The nonlinear renderer means visible mouth opening need not scale exactly by 0.7/0.5.

Recorded generation/writing/muxing times were 7.677, 7.673, and 7.657 seconds respectively, excluding model loading and avatar setup. These single observations are not a throughput or latency ranking. No production settings or Ditto runtime source were changed; the reusable experiment runner still defaults to 0.5.

## Validation and limits

All 750 new frames had detected landmarks. All eight MP4s (three audio clips, three silent intermediates, two composites) fully decoded and contained 250 frames at 25 FPS; all five audio-bearing outputs contain a ten-second audio stream. CPU statistics/FFmpeg validation are distinct from GPU generation. The landmark model used its CPU delegate, with an NVIDIA OpenGL context initialized. Five sampled timestamps and the A/B frame at index 75 were visually inspected; no audio-listening review was performed. The 1.50× reference still crops the hair/headroom.

One avatar, utterance, seed, and render per condition; no repeated-render determinism test, blinded review, or phoneme scoring. Landmark detection does not guarantee precise lip localization, and trajectory agreement is not audiovisual correctness. The three conditions are **framings**, not three independently varied head orientations. No far-shot Ditto replication was run.

## Reproduce

From the SoulX checkout, with both model environments/checkpoints installed:

```bash
/workspace/.venvs/ditto/bin/python benchmarks/distance_lipsync/run_ditto_closer.py --vad-alpha 0.7 --output NEW_DIRECTORY
.venv/bin/python benchmarks/distance_lipsync/analyze_closer.py NEW_DIRECTORY --baseline NEW_DIRECTORY/close-100-seed-50.mp4
.venv/bin/python benchmarks/distance_lipsync/package_ditto_closer.py --directory NEW_DIRECTORY
.venv/bin/python benchmarks/distance_lipsync/compare_ditto_alpha.py benchmarks/distance_lipsync/evidence-ditto-closer-0p5-20260916 NEW_DIRECTORY
```

The runner refuses to overwrite an existing directory. See the [cross-model interpretation and user preference record](../../docs/research/MODEL_CHOICE_AND_FRAMING_2026-09-16.md) for why framing behaves differently.
