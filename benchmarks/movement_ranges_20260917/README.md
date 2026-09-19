# Wide movement-parameter experiments

Fresh GPU inference, 2026-09-17, **NVIDIA GeForce RTX 4070 SUPER, 12 GB class / 12,282 MiB visible VRAM**, driver **595.84**, Torch **2.7.1+cu128 / CUDA 12.8**. Evidence: direct device/process snapshots in [results.json](results.json), including one device snapshot per render. Torch allocator caps are 3,584 MiB for histories 1–3 and 4,608 MiB for history 4; these are not physical VRAM. Existing services remain resident. Media assembly and MediaPipe/XNNPACK landmark analysis run on CPU; MediaPipe initializes an NVIDIA OpenGL context.

## User-selected working protocol

After reviewing these videos, the user selected **seeds 50, 51, 0, 1** for useful head-movement variation, keeping **shift 5**, **history 2**, and **stock strength 1.0**. Seed selection had the most useful visible impact for the user's purpose; pushing other settings too far caused unwanted distortion. This is the current working preference for the reviewed fixture, not an ordered seed-intensity scale. See the [durable movement protocol](../../docs/research/MOVEMENT_SEED_PROTOCOL_2026-09-17.md). This review decision adds no new GPU runs or runtime changes.

## Watch the comparisons

Every comparison shows native 320×576 portraits above fixed 2.5× nearest-neighbor mouth crops, with the same audio and matching timestamps. Crops add no detail; strong pose changes can move the mouth away from the fixed crop. No sharpening, interpolation or stabilization is applied.

| Video | Left-to-right columns | Sections |
| --- | --- | --- |
| [Low shifts](soulx-LITE-sampling-shift-0.5-1-2-5-seeds7-50-99-overview.mp4) | 0.5, 1, 2, **5 default** | Seeds 7 / 50 / 99, ten seconds each |
| [Middle shifts](soulx-LITE-sampling-shift-3-5-7-10-seeds7-50-99-overview.mp4) | 3, **5 default**, 7, 10 | Seeds 7 / 50 / 99, ten seconds each |
| [High shifts](soulx-LITE-sampling-shift-5-7-10-15-seeds7-50-99-overview.mp4) | **5 default**, 7, 10, 15 | Seeds 7 / 50 / 99, ten seconds each |
| [Motion history](soulx-LITE-motion-history-1-2-3-4-seeds7-50-99-overview.mp4) | 1, **2 default**, 3, 4 latents | Seeds 7 / 50 / 99, ten seconds each |
| [Eight seeds](soulx-LITE-seeds-0-1-7-42-50-51-99-1234-overview.mp4) | 0, 1, 7, 42; then 50, 51, 99, 1234 | Two ten-second groups |

All use the same neutral Indian-man portrait, ten-second audio, four denoising steps, 25 FPS, strength 1.0, eager execution, compact memory and INT8 weight storage with BF16 compute. Shift/history trials hold the other parameter at its default. These are 38 distinct settings plus three fresh stock-engine adapter controls and one resumed-process control; no historical videos are substituted. The repeated-seed baseline clips are shared across the comparison groups.

## Measurements

Mouth opening is inner-lip separation divided by eye span. Head roll is the image-plane angle of the eye-corner line; its range is p95 minus p5. Percent changes below are medians of three per-seed ratios against that seed's shift-5/history-2 baseline. Correlation compares mouth-opening trajectories at matching timestamps. It is not a phoneme-alignment or lip-sync score. Blur, face distortion, and landmark error can contaminate every metric; rendering collapse must not be mistaken for desirable movement reduction.

| Parameter | Value | Mean mouth opening change | Peak mouth opening change (p95) | Roll-range change | Seeds with less roll | Mouth-trajectory correlation | Face detections |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| shift | 0.5 | +10.3% | +4.2% | -47.9% | 3/3 | 0.819 | 750/750 |
| shift | 1 | +9.2% | +9.8% | -44.4% | 3/3 | 0.884 | 750/750 |
| shift | 2 | +6.0% | -0.7% | -19.2% | 3/3 | 0.911 | 750/750 |
| shift | 3 | +5.1% | -1.0% | -16.2% | 2/3 | 0.954 | 750/750 |
| shift | 5 | +0.0% | +0.0% | +0.0% | 0/3 | 1.000 | 750/750 |
| shift | 7 | +0.8% | +1.8% | -16.6% | 2/3 | 0.965 | 750/750 |
| shift | 10 | +0.4% | +0.1% | +30.2% | 1/3 | 0.933 | 750/750 |
| shift | 15 | -3.5% | -9.1% | +6.3% | 1/3 | 0.943 | 750/750 |
| history | 1 | +4.7% | -14.6% | +32.8% | 0/3 | 0.783 | 750/750 |
| history | 2 | +0.0% | +0.0% | +0.0% | 0/3 | 1.000 | 750/750 |
| history | 3 | -2.9% | -3.8% | -2.2% | 2/3 | 0.814 | 750/750 |
| history | 4 | -16.0% | -30.8% | +24.0% | 0/3 | 0.619 | 646/750 |

Full measurements: [CSV](metrics.csv), [summary JSON](summary.json), [per-frame landmarks](diagnostics.json). Missing detections preserve their original frame slots; differences do not bridge missing frames. Statistics use valid frames and expose detection counts. This is one portrait and one utterance; three repeated seeds do not establish population-wide behavior.

## What changed visibly

- **Shifts 0.5 and 1:** conspicuous face smearing in inspected frames. The lower roll measurements are not an acceptable quality improvement. Shift 2 also looks softer in the inspected seed-50 frame; its roll range is 16–21% lower across the three seeds, while mean mouth opening increases.
- **Shifts 7–15:** retain much more facial structure than the extreme low shifts, but head-motion changes depend on the seed. At shift 10, roll range changes by approximately +30%, −55% and +92% for seeds 7, 50 and 99. This is not a monotonic head-amplitude control.
- **History 1:** roll range rises by roughly 23–33% in all three seeds. It is a visible movement change, with a changed mouth trajectory as well; it has not been shown to improve synchronization.
- **History 3:** produces different gestures and articulation, with mixed changes relative to the default. It does not consistently lower both mouth opening and head roll in these runs.
- **History 4:** substantial smearing and loss of facial detail in inspected late frames. Face tracking fails on 104 of 750 frames across the three clips. Smaller measured mouth opening here is confounded by rendering degradation.
- **Seeds:** change pose and gesture trajectories at the default settings. Their numbers have no ordered meaning; a larger seed is not a stronger motion setting.

The videos are the primary evidence for appearance. No setting in this sweep establishes a clean, consistently weaker head-and-lip animation while preserving quality and proven synchronization. A small numerical difference should not be mistaken for a visibly useful adjustment.

## Parameter implementation and controls

Changing the seed changes the random motion trajectory; numerical seed order is not a strength ordering. Sampling shift transforms all four denoising timesteps. The runner clears cached timestep embeddings for every setting so the new schedule is used by both the model's conditioning and sampler.

Motion history needs a local experimental adaptation because the serving engine fixes its overlap at nine RGB frames. The model window stays at 33 frames. History 1/2/3/4 latents corresponds to **1/9/17/25 overlapping RGB frames**, leaving **32/24/16/8 new frames per chunk**. Five history latents would occupy the whole window and emit zero frames, so it is not a usable setting. The first chunk starts from the still reference; subsequent chunks use the requested generated history.

The adaptation changes output offset, history re-encoding length, chunk advancement and corresponding audio horizon together. It retains all decoded context before taking the history suffix, including when that suffix is longer than the newly generated portion. Actual returned chunk shapes and encoded history lengths are asserted on every chunk. The adapted default matches stock-engine raw outputs exactly for seeds 7, 50 and 99. Code hashes and a generated copy of the adapted method are retained. Production code and service settings are not modified.

Changing history also changes audio-window boundaries, chunk count and random-number consumption. These runs test the complete history configuration; they cannot isolate a pure history-strength effect. Non-default histories are experimental and differ from the standard model's inference setup. Timings exclude loading, warmup and encoding; co-resident load means these are not capacity claims.

The initial history-4/seed-7 attempt exceeded the 3,584 MiB allocator cap during VAE history encoding, despite free device memory. Its failure manifest and log are retained. Generation resumed without rerunning the 31 completed settings, and only history 4 received a 4,608 MiB cap. A fresh resumed-process seed-7 baseline was checked against the original raw hash. This changes memory allowance, not weights or numerical precision; the differing cap remains a throughput-comparison limitation.

## Validation and reproduction

All 38 individual clips contain 250 frames at 320×576 and 25 FPS, with audio. The four repeated-seed overviews contain 750 frames each; the seed overview contains 500 frames. All five overviews passed full ffmpeg decode. [Validation evidence](validation.json).

From the repository root, use `PYTHONPATH=.` and `.venv/bin/python` with `run.py`, then `analyze.py`, `package.py` and `report.py` in this directory. Generation refuses to overwrite retained results; explicit `--resume` continues a failed manifest while preserving successful clips. Create a fresh experiment directory for a new full rerun. `analyze.py --watch` can process finished clips while inference continues and reuses already-analyzed clips.
