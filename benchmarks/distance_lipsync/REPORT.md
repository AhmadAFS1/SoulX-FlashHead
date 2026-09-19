# Character distance and SoulX FlashHead lip-sync output

> **Scope:** This report covers the initial female portrait. The stronger Indian male replication is documented in [INDIAN_MALE_REPORT.md](INDIAN_MALE_REPORT.md).

> **GPU provenance:** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM; NVIDIA driver 595.84; run date 2026-09-16. Source: `nvidia-smi` captured at experiment start in `evidence-20260916/results.json`. Generation was GPU inference; MediaPipe landmark analysis was CPU-side. The pre-load GPU allocation was 2,487 MiB and was not stopped.

## Conclusion

**Apparent character distance affected SoulX FlashHead's generated mouth motion in this controlled test.** Medium framing (72% of close scale) retained the broad close-up timing pattern but generated a larger mouth-opening trajectory. Far framing (50% scale) diverged more strongly and showed a consistent one-frame phase difference. This supports a framing/scale effect on the output, but does not prove worse perceived lip-sync or physical-distance causality.

## Controlled setup

- One identity derived from `examples/girl.png`.
- One 10-second utterance, 250 frames at 25 FPS.
- Close/medium/far subject scales: 1.00 / 0.72 / 0.50.
- Seeds 50 and 51 at every scale.
- Fixed SoulX FlashHead Lite profile: 320×576, four steps, compiled optimized path, real RoPE, INT8 weight storage, compact memory mode.
- Distance means apparent face scale in the reference image, not a measured camera-to-subject distance.

## Results

| Framing | Mean face width | Mean normalized opening | Change vs close | Mean p95 opening | Change vs close |
| --- | ---: | ---: | ---: | ---: | ---: |
| Close | 233.5 px | 0.0561 | baseline | 0.1282 | baseline |
| Medium | 169.2 px | 0.0710 | +26.7% | 0.1717 | +33.9% |
| Far | 118.5 px | 0.0617 | +10.0% | 0.1520 | +18.6% |

| Comparison with close, same seed | Zero-lag mouth-trajectory correlation | Best-fit phase shift | Correlation after shift |
| --- | ---: | ---: | ---: |
| Medium | 0.809 | 0 ms | 0.809 |
| Far | 0.602 | −40 ms (far leads close by one frame) | 0.705 |

MediaPipe detected the face in all 1,500 decoded frames. Cross-seed zero-lag trajectory correlation was 0.848 close, 0.852 medium, and 0.546 far, suggesting far framing also made the generated mouth motion less seed-stable.

## Interpretation

- Medium distance did not introduce a measurable phase shift, but it did change articulation magnitude.
- Far distance changed both trajectory shape and phase relative to the close reference.
- Throughput was effectively unchanged (~35.1 useful FPS), so the observed differences are behavioral rather than a speed tradeoff.
- The synthetic scale variants use a blurred background and feathered subject inset. That compositing is a confound; this experiment isolates a practical framing change, not physical distance alone.

The automated measure is inner-lip distance divided by eye-corner distance. It is useful for comparing generated motion but is not SyncNet, a phoneme-level score, or a human perception study. A production-quality conclusion requires natural close/medium/far source images, more identities and utterances, and a validated audiovisual metric or blinded review.

## Evidence

- `evidence-20260916/results.json`: configuration, hashes, GPU provenance, and run timings.
- `evidence-20260916/mouth-motion.json`: per-frame landmark series and correlations.
- `distance-lipsync-analysis.ipynb`: executed, reproducible analysis and chart.
- `evidence-20260916/soulx-LITE-female-framing-1.00x-vs-0.72x-vs-0.50x-seed50.mp4` and `evidence-20260916/soulx-LITE-female-framing-1.00x-vs-0.72x-vs-0.50x-seed51.mp4`: close / medium / far, left to right.
