# Why the 1.25× SoulX framing moves more, and control options

September 16, 2026: fresh **CPU landmark analysis of existing videos**, not new SoulX generation. Those six source clips were generated on **NVIDIA GeForce RTX 4070 SUPER**, 12 GB class / 12,282 MiB visible, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8, compiled Lite, four steps, 320×576, 25 FPS, seed 50. Their run metadata records 2,487 MiB pre-load device use, identity unverified. Evidence: [original close run](../../benchmarks/distance_lipsync/evidence-indian-male-20260916/results.json), [tighter run](../../benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/results.json), [strength-0.5 run](../../benchmarks/distance_lipsync/evidence-indian-male-closer-strength-0p5-20260916/results.json).

## The observation is supported, beyond simple display magnification

Measured all 1,500 frames across the original and strength-0.5 close/closer/closest clips. Every frame had landmarks. Statistics below are **p95–p5 ranges**, not absolute peak-to-peak angles. Roll is the angle of the eye-corner line in the image; horizontal movement is eye-midpoint displacement divided by each clip's median eye span. These are 2D head-motion proxies, not calibrated 3D pose.

| Framing | Original roll range | Strength-0.5 roll range | Strength-0.5 horizontal range, eye-span units |
| --- | ---: | ---: | ---: |
| 1.00× close | 3.56° | 5.86° | 0.273 |
| 1.25× closer | 8.28° | 10.20° | 0.334 |
| 1.50× closest | 5.15° | 2.93° | 0.126 |

The 1.25× condition also has the largest full-clip yaw/pitch proxy ranges within each strength. The head movement is not merely more visible because the face is larger: normalized movement and rotation differ, and the still-larger 1.50× face moves less on these measures.

At strength 0.5, the 2.00–2.96-second window has a 15.72° robust roll range at 1.25×, versus 2.39° and 2.25° in the other framings. The one-second-window statistic can exceed the full-clip robust range because a brief movement occupies more of that shorter window. No transcript/word alignment was performed; this identifies a visible interval, not a specific word causing a gesture. [Samples at 1.48, 2.48, and 3.48 seconds](../../benchmarks/distance_lipsync/head-motion-samples-20260916.jpg) were visually inspected.

## Why this happens

The strongest implementation-backed explanation is **coupled generation conditioned on a different image**, not a special “1.25× movement” parameter.

1. SoulX synthesizes the whole video jointly. Audio cross-attention updates visual tokens throughout the portrait; there is no separate jaw rig and head-pose rig in this Lite serving path. See [`DiTAudioBlock.forward`](../../flash_head/src/modules/flash_head_model.py).
2. Cropping the reference changes its encoded composition, facial position/size, and visible head/shoulder context. Using the same random seed repeats the noise sequence, not the model's response to different conditioning. The same speech can therefore produce different head gestures as well as mouth movements.
3. The final nine generated frames condition the next 24 new frames. Once the trajectories differ, each clip carries its own motion history forward. The engine also samples the VAE posterior with the session RNG. See [`Engine.generate`](../../soulx_rtc/engine.py).
4. The experimental 0.5 hook multiplies all 30 audio cross-attention residual outputs. It does not isolate the mouth. In fact, roll range rises from 8.28° to 10.20° at 1.25× while falling from 5.15° to 2.93° at 1.50×. This is direct evidence against treating the existing scalar as a monotonic head-motion control. Cross-strength differences also include retracing/numerical effects because the retained strength-1 baseline is unhooked.

The exact reason this particular crop/seed makes this particular gesture stronger cannot be isolated from one utterance and seed. Learned composition/prosody associations are plausible, but not experimentally identified causes. More space around the head is not a proven explanation, and 1.25× is not a universal maximum-motion point.

## Can head movement be controlled separately from lip sync?

**Not with an existing validated native slider in this checkout.** Searches of the model, pipeline, engine, and server expose no independent head amplitude/yaw/pitch/roll control. `audio_guide_scale` is a pretrained-teacher branch, not a Lite motion dial. `sample_shift`, boundary fades, and changing WAV volume are not established independent controls either.

| Option | What it can do | Limitation/status |
| --- | --- | --- |
| Keep audio strength fixed; compare several seeds | Select a naturally calmer or more expressive trajectory | Offline selection, not a live amplitude guarantee; new words may behave differently |
| Reference framing or initial motion history | Bias composition and initial movement | Existing conditioning, not a monotonic head-motion control; effects can drift |
| Causal 2D head stabilization after generation | Attenuate translation/roll/scale without averaging away mouth articulation | Proposed, not implemented/tested; cannot undo true yaw/pitch; may shift background, need crop margin, or create blend artifacts |
| Spatially gate audio residuals outside the mouth | Attempt to separate articulation from broader motion | Research intervention, not validated; self-attention/temporal coupling and coarse latent resolution prevent a clean guarantee |
| Explicit pose-conditioned adapter/fine-tuning | Potential genuine head-pose/amplitude control alongside audio | Larger model/training project, not a hidden configuration option |

**September 17 update:** the user's review of the subsequent [38-setting sweep](../../benchmarks/movement_ranges_20260917/README.md) selects seeds **50, 51, 0, 1** with shift **5**, history **2**, and stock strength **1.0** as the [current movement protocol](MOVEMENT_SEED_PROTOCOL_2026-09-17.md). Seed choice gave the most useful head-movement variation; extreme shift/history settings caused unwanted distortion. This supersedes the strength-0.5 recommendation below. The new sweep's RTX 4070 SUPER hardware and runtime provenance are recorded in its report; this update adds no new inference.

Historical September 16 proposal: keep **1.25× and audio strength 0.5**, compare several seeds, and test mild **rigid-motion stabilization** on the chosen output. Estimate the transform from stable upper-face landmarks and apply one transform to each frame; do not smooth or blend the whole face over time, which can smear consonant closures. Use a causal transform filter for streaming; an offline centered smoother would add lookahead. Separately score mouth opening, phoneme closures, pose motion, image warping, added latency, and throughput before adopting it. Stabilizing pixels does not change the model's internal recurrent motion unless explicitly fed back, which is a different and riskier intervention.

No control was implemented or production default changed in this analysis. Increased head motion is not automatically a defect; it may be part of the naturalness the user prefers.

## Reproduction and limits

```bash
.venv/bin/python benchmarks/distance_lipsync/analyze_head_motion.py --output benchmarks/distance_lipsync/head-motion-20260916.json
```

[Script](../../benchmarks/distance_lipsync/analyze_head_motion.py), [per-frame data, windows, definitions, and hashes](../../benchmarks/distance_lipsync/head-motion-20260916.json). CPU MediaPipe inference initializes an NVIDIA OpenGL context but performs no new SoulX generation. Landmark noise, blink-related localization, perspective, and nonrigid facial deformation can affect the proxies. No 3D motion-capture ground truth, new seed sweep, or fresh lip-sync score was produced.
