# Indian male: close, closer, and closest comparison

> **Strength-0.5 follow-up:** The same three framings under the experimental mouth-strength intervention are documented in [INDIAN_MALE_CLOSER_STRENGTH_0P5_REPORT.md](INDIAN_MALE_CLOSER_STRENGTH_0P5_REPORT.md).

> **GPU provenance:** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM; NVIDIA driver 595.84; run date 2026-09-16. Source: `nvidia-smi` captured in `evidence-indian-male-closer-20260916/results.json`. Generation used GPU inference; landmark analysis was CPU-side.

## Conclusion

Starting from the exact prior `close-seed-50.mp4`, two tighter variants were generated at 1.25× and 1.50× zoom. **Both closer framings preserve articulation; the 1.50× result has the strongest mouth-trajectory agreement with the original close clip and the clearest visible mouth detail.**

The practical tradeoff is composition: 1.50× sometimes crops the top of the hair/head during motion. The 1.25× version is the safer balance when headroom matters.

## Results

| Framing | Mean face width | Mean opening | Change vs close | p95 opening | Change vs close | Motion change vs close |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Close 1.00× | 133.3 px | 0.1054 | baseline | 0.2377 | baseline | baseline |
| Closer 1.25× | 167.6 px | 0.1009 | −4.2% | 0.2759 | +16.1% | +5.6% |
| Closest 1.50× | 199.6 px | 0.0928 | −12.0% | 0.2335 | −1.8% | −0.4% |

| Comparison with exact close baseline | Zero-lag trajectory correlation | Best-fit phase |
| --- | ---: | ---: |
| Closer 1.25× | 0.830 | +40 ms / one frame, correlation 0.855 |
| Closest 1.50× | 0.920 | 0 ms |

MediaPipe detected the face in every frame of all three 250-frame clips. The closest output therefore keeps stable measurable articulation while providing roughly 50% more face pixels than the baseline.

## Recommendation

- Use **1.25× / about 168 px face width** as the safer default framing for this 320-pixel-wide profile.
- Use **1.50× / about 200 px face width** when maximum mouth visibility matters and tighter hair/head cropping is acceptable.
- Avoid the previously tested far framing near 67 px face width, which collapsed mean mouth opening by 92.2%.

This is a controlled one-seed, one-utterance comparison. It is strong evidence for composition choice on this avatar, but not a phoneme-level lip-sync accuracy score or a universal threshold.

## Evidence

- `evidence-indian-male-closer-20260916/soulx-LITE-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-labeled.mp4`: labeled comparison, left to right.
- `evidence-indian-male-closer-20260916/contact-close-closer-closest.jpg`: matched-frame visual review.
- `evidence-indian-male-closer-20260916/results.json`: hashes, profile, GPU provenance, and timings.
- `evidence-indian-male-closer-20260916/mouth-motion.json`: per-frame landmark series and comparisons.
- `indian-male-closer-analysis.ipynb`: executed analysis and visualization.
