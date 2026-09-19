# Indian male close framings at mouth-strength 0.5

> **GPU provenance:** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM; NVIDIA driver 595.84; Torch 2.7.1+cu128 / CUDA 12.8; run date 2026-09-16. The pre-load GPU allocation was 2,487 MiB and was not stopped. Source: `nvidia-smi` and run metadata captured in `evidence-indian-male-closer-strength-0p5-20260916/results.json`. Generation was GPU inference; MediaPipe analysis was CPU-side.

## Conclusion

All three close framings were regenerated with the experimental audio cross-attention strength fixed at 0.5. **The 1.25× framing is the best combined candidate in this test.** It preserves the unmodified pose's mouth trajectory most strongly (0.928 zero-lag correlation), reduces p95 opening by 24.6%, and retains more composition headroom than 1.50×.

The 1.50× framing still provides the largest visible mouth, but strength 0.5 attenuates it more heavily: mean opening falls 20.9% and p95 falls 32.1% versus its unmodified clip. It also retains the prior hair/head-cropping tradeoff. Thus “closer is better” holds directionally across the tested range for visibility, but the best combined setting is not simply the maximum zoom under every strength.

## Experimental control

- Three framings: 1.00×, 1.25×, 1.50×.
- One 10-second utterance, seed 50, 250 frames at 25 FPS.
- Fixed 320×576, four-step compiled SoulX FlashHead Lite profile.
- Thirty process-local forward hooks multiply every audio cross-attention output by 0.5.
- `torch._dynamo.reset()` was called before warmup/compilation.
- All three raw hashes differ from their matching unmodified clips, confirming that the intervention was captured.
- Production code, defaults, and checkpoint files were not changed.

## Results

| Framing | Face width | Mean opening at 0.5 | p95 opening at 0.5 | Mean change vs strength 1 | p95 change vs strength 1 | Trajectory correlation vs strength 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00× | 133.3 px | 0.0935 | 0.1779 | −11.3% | −25.2% | 0.874 |
| 1.25× | 167.0 px | 0.0915 | 0.2080 | −9.3% | −24.6% | **0.928** |
| 1.50× | 199.4 px | 0.0734 | 0.1585 | −20.9% | −32.1% | 0.869 |

Within the strength-0.5 set, trajectory correlation with the 1.00× pose is 0.783 for 1.25× and 0.762 for 1.50×. All 750 decoded frames have successful face landmarks. Generation throughput remains about 35.2–35.5 useful FPS; the intervention is a behavior control, not a speed optimization.

## Visual review

Five matched samples and the full labeled composite show:

- 1.00× remains coherent but has the least visible mouth detail.
- 1.25× keeps useful articulation while moderating the largest openings; it is the best balance.
- 1.50× is visibly clearest but quieter and more tightly cropped at strength 0.5.
- The intervention changes more than mouth amplitude; expression, gaze, and head motion can also differ.

This does not establish phoneme-level lip-sync accuracy, naturalness, or a production default. Strength 0.5 is an experimental cross-attention intervention and does not reconstruct smeared teeth detail.

## Evidence

- `evidence-indian-male-closer-strength-0p5-20260916/soulx-LITE-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-audio-strength0.5.mp4`: requested three-pose video.
- `evidence-indian-male-closer-strength-0p5-20260916/contact-strength-0p5.jpg`: matched-frame review.
- `evidence-indian-male-closer-strength-0p5-20260916/results.json`: hashes, hook count, GPU/profile metadata, and timings.
- `evidence-indian-male-closer-strength-0p5-20260916/mouth-motion.json`: landmark series and pose/strength comparisons.
- `indian-male-closer-strength-0p5-analysis.ipynb`: executed analysis and chart.
