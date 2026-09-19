# Ditto: Indian male at three framings and expression strength 0.5

GPU inference on 2026-09-16: NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM (12 GB class), driver 595.84. Torch 2.5.1+cu121 / CUDA 12.1, Ditto TensorRT Ampere Plus models. Pre-load device allocation was 2,487 MiB; its workload identity was not verified. Direct evidence: [results.json](evidence-ditto-closer-0p5-20260916/results.json). Landmark analysis and video decoding are separate from GPU generation.

## Result

Follow-up: [the matched 0.7 experiment](DITTO_CLOSER_0P7_REPORT.md) increases mouth-opening amplitude while preserving the same framing consistency. [Cross-model findings](../../docs/research/MODEL_CHOICE_AND_FRAMING_2026-09-16.md) record the user's model preferences and the source-based explanation.

Ditto preserved almost identical normalized mouth trajectories across all three tested framings. Closer framing increases mouth visibility, but these measurements do **not** demonstrate an improvement in audiovisual lip-sync accuracy. At 1.50×, the source crop removes some hair/headroom. The 1.25× framing remains a reasonable composition preference; neither preference constitutes an objective quality ranking.

| Framing | Mean face width, px | Mean normalized opening | p95 normalized opening | Correlation with 1.00× | Best trajectory lag |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.00× | 132.7 | 0.04370 | 0.08617 | baseline | baseline |
| 1.25× | 165.9 | 0.04353 | 0.08546 | 0.991 | 0 frames |
| 1.50× | 199.0 | 0.04447 | 0.08420 | 0.988 | 0 frames |

All 750 frames had detected facial landmarks. Five matched frames (25, 75, 125, 175, 225) were visually inspected; all individual and composite videos fully decoded. Video playback with listening was not performed. Correlation measures similarity to the baseline, not correctness against spoken phonemes. Face-detection success alone does not guarantee accurate lip landmarks.

## Control and reproducibility

The exact saved SoulX reference images at 1.00×, 1.25× and 1.50× were supplied to Ditto. Same 10-second `benchmarks/comparison-10s.wav`, seed 50, 25 FPS, 320×576 individual outputs. Each contains 250 frames. The composite has a separate 36-pixel label strip above each image, preserving the source headroom.

Ditto uses `overall_ctrl_info={'vad_alpha': 0.5}`. Inspection of `core/atomic_components/motion_stitch.py::ctrl_vad` shows that this blends the **entire expression vector** halfway toward the source expression. It is not the same operation as SoulX's 0.5 audio cross-attention multiplier, nor does it promise exactly half the visible mouth motion. Ditto also registers/crops the face before processing, which is a plausible explanation for its scale consistency; that mechanism was not isolated experimentally.

The established Ditto `fade_type='s'` and 15-frame return-to-source fade are fixed across all three runs. The final 0.6 seconds therefore include an intentional transition. This makes a direct numeric quality ranking against the previous SoulX clips inappropriate. No new strength-1 Ditto controls were rendered, so this test does not quantify attenuation relative to 1.0.

Reproduce from the SoulX repository:

```bash
/workspace/.venvs/ditto/bin/python benchmarks/distance_lipsync/run_ditto_closer.py --output NEW_DIRECTORY
.venv/bin/python benchmarks/distance_lipsync/analyze_closer.py NEW_DIRECTORY --baseline NEW_DIRECTORY/close-100-seed-50.mp4
.venv/bin/python benchmarks/distance_lipsync/package_ditto_closer.py --directory NEW_DIRECTORY
```

The packaging script defaults to the checked-in evidence directory; `--directory` selects the reproduced run. Run metadata records reference/audio hashes, Ditto revision, configuration hash and output hashes. Temporary silent MP4s are retained as intermediate evidence. Serving defaults were not changed.

## Playback

- [Labeled three-way comparison](evidence-ditto-closer-0p5-20260916/ditto-indian-man-framing-1.00x-vs-1.25x-vs-1.50x-vad-alpha0p5.mp4)
- [1.00×](evidence-ditto-closer-0p5-20260916/close-100-seed-50.mp4), [1.25×](evidence-ditto-closer-0p5-20260916/closer-125-seed-50.mp4), [1.50×](evidence-ditto-closer-0p5-20260916/closest-150-seed-50.mp4)
- [Contact sheet](evidence-ditto-closer-0p5-20260916/contact.jpg)
- [Per-frame metrics](evidence-ditto-closer-0p5-20260916/mouth-motion.json)

This is one avatar, one utterance and one seed. The evidence supports stability across these three close framings; it does not establish that closer is always better or cover Ditto at the earlier far-shot scales.
