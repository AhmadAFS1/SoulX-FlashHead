# Indian male: character distance and SoulX lip-sync output

> **Follow-up:** Two framings closer than the original close clip are compared in [INDIAN_MALE_CLOSER_REPORT.md](INDIAN_MALE_CLOSER_REPORT.md).

> **GPU provenance:** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM; NVIDIA driver 595.84; run date 2026-09-16. Source: `nvidia-smi` captured at experiment start in `evidence-indian-male-20260916/results.json`. Generation was GPU inference; MediaPipe landmark analysis was CPU-side. The pre-load GPU allocation was 2,487 MiB and was not stopped.

## Conclusion

**The Indian male result strongly confirms that closer framing produces better SoulX FlashHead lip-sync output.** At far scale the generated mouth is nearly static despite speech: mean normalized opening falls 92.2%, mouth-motion variability falls 74.6%, and trajectory agreement with the close output falls to 0.208. Visual review of both seeds confirms the numerical result.

Medium framing is usable but not equivalent to close. Its average mouth opening is only 9.8% lower, yet frame-to-frame motion is 26.3% higher, cross-seed stability is substantially worse, and some frames show exaggerated mouth/teeth shapes. Of the tested variants, close framing is clearly preferable.

## Controlled setup

- Indian male source: `/workspace/benchmarks/same-avatar/shared.png` (384×672); fitted close reference and generated scale variants are retained in the evidence directory.
- One 10-second utterance, 250 frames at 25 FPS.
- Close/medium/far subject scales: 1.00 / 0.72 / 0.50.
- Seeds 50 and 51 at every scale.
- Fixed SoulX FlashHead Lite profile: 320×576, four steps, compiled optimized path, real RoPE, INT8 weight storage, compact memory mode.
- Distance means apparent face scale in the reference image, not measured physical camera distance.

## Results

| Framing | Mean face width | Mean normalized opening | Change vs close | Mean p95 opening | Change vs close | Frame-change vs close |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Close | 133.4 px | 0.1092 | baseline | 0.2525 | baseline | baseline |
| Medium | 97.6 px | 0.0984 | −9.8% | 0.2304 | −8.7% | +26.3% |
| Far | 67.2 px | 0.0085 | −92.2% | 0.0413 | −83.7% | −74.6% |

| Comparison with close, same seed | Zero-lag mouth-trajectory correlation | Best-fit phase behavior |
| --- | ---: | --- |
| Medium | 0.690 | 0 ms in both seeds |
| Far | 0.208 | Opposite ±80 ms shifts; no stable direction and correlation remains weak |

MediaPipe detected the face in all 1,500 frames. Cross-seed mouth-trajectory correlation was 0.853 close, 0.563 medium, and 0.520 far. The far failure therefore is not missing tracking; it is a generated-motion difference.

## Practical implication

For this avatar and 320-pixel-wide output:

- Prefer close framing near the tested 133 px generated face width.
- Treat the tested far framing near 67 px as unacceptable for speech animation.
- Use medium framing cautiously and inspect teeth, mouth amplitude, and temporal stability.

The exact pixel thresholds are avatar/profile-specific. Natural close/medium/far source photographs, more voices, and a phoneme-level or human-rated audiovisual test are needed before turning them into a universal input validator.

## Evidence

- `evidence-indian-male-20260916/results.json`: configuration, hashes, GPU provenance, timings.
- `evidence-indian-male-20260916/mouth-motion.json`: per-frame landmark series and correlations.
- `indian-male-distance-analysis.ipynb`: executed reproducible analysis and chart.
- `evidence-indian-male-20260916/soulx-LITE-indian-man-framing-1.00x-vs-0.72x-vs-0.50x-seed50.mp4` and `evidence-indian-male-20260916/soulx-LITE-indian-man-framing-1.00x-vs-0.72x-vs-0.50x-seed51.mp4`: close / medium / far, left to right.
- `evidence-indian-male-20260916/contact-seed-50.jpg` and `contact-seed-51.jpg`: five matched time samples for visual review.
