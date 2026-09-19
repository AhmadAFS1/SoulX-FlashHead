# Strength, seed, shift and history experiments

2026-09-17: fresh GPU inference on **NVIDIA GeForce RTX 4070 SUPER, 12 GB class / 12,282 MiB visible VRAM**, driver **595.84**, Torch **2.7.1+cu128 / CUDA 12.8**. Evidence: direct per-run `nvidia-smi` snapshots in `results.json` and `other-parameters/results.json`. The 3,584 MiB Torch allocator cap does not change physical VRAM. Existing GPU processes were left running; their PIDs, names and allocations are recorded. Packaging and media validation are CPU work.

## Strength comparison recreated first

The September 13 four-level test remains documented in `docs/research/MOUTH_MOVEMENT_CONTROL_2026-09-13.md`, but its MP4 directory `/workspace/experiments/flashhead-mouth-3lwV6k/measured/outputs` is absent. The surviving September 16 strength-0.5 video compares three framings at a fixed strength; it is not a strength sweep.

These are fresh comparisons using the neutral, shoulder-visible 1.25× Indian-man reference at 320×576. They reproduce the four strength levels, not the exact historical avatar/history/compiled execution fixture.

- [Strength sweep, seed 50](soulx-LITE-audio-strength-1.0-0.9-0.75-0.5-indian-man-seed50.mp4)
- [Strength sweep, seed 51](soulx-LITE-audio-strength-1.0-0.9-0.75-0.5-indian-man-seed51.mp4)
- [Matched mouth samples, seed 50](mouths-seed50.png)
- [Matched mouth samples, seed 51](mouths-seed51.png)

Columns run left to right: **1.0, 0.9, 0.75, 0.5**. Each video includes the shared audio and 2.5× nearest-neighbor mouth crops below native-resolution portraits. The crops add no detail or sharpening. Lower strength visibly changes articulation and other expression details; it should not be interpreted as a linear percentage reduction in mouth opening or as a teeth reconstruction method.

All 30 audio cross-attention outputs receive the same scalar. An explicit unhooked seed-50 control is retained. The eager hook at strength 1.0 matched that control byte for byte, and all four strength settings produced distinct raw frame hashes for each seed.

## Other parameter sweeps

Each group changes one parameter while holding the neutral portrait, audio, output resolution and all other listed settings constant.

| Sweep | Values, left to right | Baseline settings |
| --- | --- | --- |
| [Seed](soulx-LITE-seed-50-vs-51-vs-52-vs-53-indian-man-1.25x.mp4) | 50, 51, 52, 53 | Strength 1, shift 5, history 2 |
| [Sampling shift](soulx-LITE-sampling-shift-1-vs-3-vs-5-vs-7-seed50-indian-man-1.25x.mp4) | 1, 3, 5, 7 | Strength 1, seed 50, history 2 |
| [Motion history](soulx-LITE-motion-history-1-vs-2-vs-3-seed50-indian-man-1.25x.mp4) | 1, 2, 3 latent frames | Strength 1, seed 50, shift 5 |

Seed 50 and 51 strength-1 clips are reused from this same fresh strength sweep. Other entries are freshly generated. The new-process baseline must match the original control hash before comparisons are assembled.

The shift sweep clears the engine's cached timestep embeddings after changing the schedule, so the requested shift actually reaches inference. Shift controls the denoising schedule; it is not a calibrated motion-amplitude control.

History is a more invasive, out-of-training-configuration experiment. The serving engine hardcodes nine overlapping frames. The offline runner adapts its generation method locally: **1/2/3 history latents correspond to 1/9/17 overlapping RGB frames and 32/24/16 newly emitted frames per 33-frame model window**. Audio-window advancement and output selection follow those chunk lengths. It retains enough decoded context for re-encoding. Every chunk checks the actual encoded history length. A two-latent identity control must match the standard engine exactly. The generated method and explicit replacements are retained under `other-parameters/`; production source files are not edited.

History changes also change chunk count, audio-window boundaries and the session's random-number consumption. This experiment measures the overall effect of that history configuration; it cannot isolate a pure history-strength effect. Seed and shift sweeps each use one utterance, and the shift/history sweeps use one seed. These videos support visual exploration, not universal quality or lip-sync claims.

## Initial visual review

Seven matched pre-encoding samples were inspected for each setting. Shift 1 produces conspicuous whole-face smearing in several samples, especially around 3.5 and 4.5 seconds. Shifts 3, 5 and 7 retain much clearer facial structure in this fixture. This makes shift 1 an unsuccessful quality setting here, rather than an attractive way to moderate movement.

Changing seeds visibly changes tilt, pose and mouth trajectories. Seed 52 has a pronounced early tilt at the 0.5-second sample. The history variants also change mouth shapes and head poses; all three decode, but these sparse samples do not establish which is most natural or best synchronized. There is no numerical motion, dental-quality or phoneme-accuracy score in this run.

Both identity checks passed exactly: the new-process baseline matches the strength-sweep unhooked control, and the history adapter at two latents matches that baseline. All five comparison videos passed full decode and frame/audio checks.

## Shared profile and validation

10-second shared waveform; 250 output frames; 25 FPS; 320×576 native output; four denoising steps; compact memory mode; INT8 weight storage with BF16 compute; eager execution. Stock color correction and audio preprocessing. The reference, audio and raw output hashes are retained. No checkpoints, server configuration or running services are changed.

`package.py` and `package_other.py` count decoded frames, check frame rates and audio presence, and fully decode comparison videos to detect errors. Source clips retain seven pre-encoding PNG samples. Exact results and statuses are in `results.json`, `validation.json`, `other-parameters/results.json` and `other-parameters/validation.json`.

Run scripts from the repository root with `PYTHONPATH=.` and `.venv/bin/python`. The inference scripts refuse to overwrite retained evidence. Use a fresh experiment directory for reruns.
