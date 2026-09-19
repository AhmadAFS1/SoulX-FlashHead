# Closed-mouth SoulX feasibility check

September 17, 2026. Fresh GPU inference on **NVIDIA GeForce RTX 4070 SUPER**, 12 GB class / **12,282 MiB visible VRAM**, driver **595.84**, Torch **2.7.1+cu128**, CUDA **12.8**. Direct hardware and co-resident process snapshots are in `results.json`: 2,727 MiB device use before loading, including two Python processes reporting 2,478 and 234 MiB. A 6,144 MiB Torch allocator cap is not the GPU's physical capacity. Landmark analysis ran on CPU through MediaPipe, which also initialized an NVIDIA OpenGL context.

## Finding

The same original close-up MuseTalk portrait can drive SoulX Lite with silent waveform audio. Two new eight-second takes yielded closed-looking lips and very restrained motion. This supports an offline candidate workflow for subtle idle footage. It does not establish independent control of speaking-like head gestures while locking the mouth shut.

| Run, same RTX 4070 SUPER | Video | Detected frames | Maximum lip-gap / eye-span | Head roll p95–p5 |
| --- | --- | ---: | ---: | ---: |
| Seed 50 | [Watch](silence-seed50.mp4) | 192 / 192 | 0.001262 | 0.419° |
| Seed 51 | [Watch](silence-seed51.mp4) | 192 / 192 | 0.001475 | 0.281° |

All 384 generated RGB frames were measured. Visual inspection covered regularly sampled frames and the four largest measured lip gaps per take, in `silence-seed50-frames.jpg` and `silence-seed51-frames.jpg`. Those inspected frames show closed lips without visible teeth; seed 50 also shows a blink. Tiny nonzero landmark distances do not imply an actual open mouth. The measurements are proxies, not a certified all-frame visual or anatomical closure guarantee. The samples show little head movement and are not evidence of expressive silent speaking gestures.

## Method and code findings

- Source: the exact original `sample_ai_human_facetime_v1.png` in the MuseTalk close-up production bank, SHA-256 `0e893bd133c22de532ced133f91e136a2896663612c506b9801c64ddd3d1e919`.
- Native output: 480×832, 24 FPS, eight seconds per seed; Lite, four steps, shift 5, default history 2, eager optimized execution, INT8 weight storage with BF16 compute. No attenuation hooks or mouth repair.
- Input: 128,000 exact-zero samples at 16 kHz, processed normally by Wav2Vec. Silent waveform conditioning is distinct from arbitrarily zeroing learned embedding tensors.
- `soulx_rtc/engine.py:197` constructs the rolling audio window. `soulx_rtc/calls.py:579` already feeds silent audio for generated idle in the service.
- `flash_head/src/modules/flash_head_model.py:335` adds audio cross-attention to visual tokens across the frame. The current model/API has no independent jaw lock, closed-mouth prompt, or head-pose control.
- `flash_head/src/pipeline/flash_head_pipeline.py:150` accepts portrait and sampling settings; generation accepts audio embeddings, not a natural-language motion prompt. Reducing audio attention affects more than the mouth and is not a validated closure constraint.
- This experiment tests Lite only; it does not establish Pro's behavior with silent audio.

The generated MP4s are diagnostics, not replacements installed in MuseTalk. They have not been given the production bank's common first/last six-frame anchors, certified for looping, or prepared/tested through MuseTalk lip sync. They contain a silent AAC track from the existing recording helper; production base assets require no audio track.

For an offline base video, silence generation followed by candidate selection is feasible if this small amount of motion is sufficient. For expressive speaking motion with strictly closed lips, a separate motion/face-retargeting or mouth-repair stage would need implementation and validation; merely supplying speech then muting the MP4 leaves mouth articulation intact. A fresh SoulX sample is not guaranteed to reproduce an LTX clip's exact pose trajectory or boundary frame.

Reproduction (fresh output directory required):

```bash
PYTHONPATH=. .venv/bin/python benchmarks/closed_mouth_silence_20260917/validate.py
```

Per-frame measurements, runtime provenance, and source hash: [results.json](results.json). Generation script: [validate.py](validate.py).
