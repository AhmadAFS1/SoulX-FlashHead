# Working protocol: choose head movement by seed

Recorded September 17, 2026 after the user's visual review. This is a documentation decision, with no new inference or runtime changes. The supporting [38-setting experiment](../../benchmarks/movement_ranges_20260917/README.md) ran on **NVIDIA GeForce RTX 4070 SUPER, 12 GB class / 12,282 MiB visible VRAM**, driver **595.84**, Torch **2.7.1+cu128 / CUDA 12.8**. Its [run manifest](../../benchmarks/movement_ranges_20260917/results.json) records direct `nvidia-smi` evidence, per-run settings, and co-resident processes (2,478 MiB and 234 MiB at initialization). The tested profile was 320×576, 25 FPS, four steps, eager execution, compact memory, INT8 weight storage with BF16 compute.

## Selected protocol

**Use seeds `50`, `51`, `0`, and `1` as the preferred shortlist for head-movement variation.** These are selected alternatives, not a ranking or an ordered intensity scale. Use seed `50` as the existing comparison baseline, and compare `51`, `0`, and `1` when another movement trajectory is desired.

| Setting | Working value |
| --- | --- |
| `seed` / `base_seed` | Select from **50, 51, 0, 1** |
| `sample_shift` / `shift` | **5**, default |
| `motion_frames_latent_num` | **2**, default; nine overlapping RGB frames |
| Audio-conditioning `strength` / `gain` | **1.0**, stock conditioning; no attenuation hook needed |

Keep the chosen seed fixed within a generated take. For comparisons, hold the portrait, audio, resolution and other inference settings constant and record the seed used. Preserve the current 320×576, 25-FPS, four-step fixture when reproducing these reviewed clips.

## User's conclusion and rationale

The user found seed selection to have the most useful visible effect on head movement. The strength comparison did not show a meaningful enough difference for the user's purpose. Pushing the other parameters too far from their defaults produced unwanted distortion, so the preferred workflow is to vary the seed while leaving shift and history at their defaults.

The experiment supports treating extreme settings cautiously: shifts 0.5/1 produced severe smearing, history 4 degraded facial detail, and higher shifts changed head motion in different directions across seeds. This does not mean every non-default value fails or that distortion increases symmetrically on both sides of a default. The choice of seeds 50/51/0/1 is the user's visual preference for the reviewed portrait and audio, not a universal quality guarantee.

This protocol supersedes the earlier recommendation to use experimental strength 0.5 for the next head-motion comparison. Historical strength measurements remain valid as recorded; they do not define the current preferred settings. The seed shortlist is a documentation recommendation, not a newly implemented automatic seed-selection or rotation policy.

## Evidence

- [Eight-seed comparison](../../benchmarks/movement_ranges_20260917/soulx-LITE-seeds-0-1-7-42-50-51-99-1234-overview.mp4): first section includes 0/1/7/42; second includes 50/51/99/1234.
- [Wide shift and history tests, measurements and limitations](../../benchmarks/movement_ranges_20260917/README.md).
- [Recreated strength comparisons](../../benchmarks/strength_recreated_20260917/README.md).

No server settings, model weights, or serving-capacity claims change with this documentation update.
