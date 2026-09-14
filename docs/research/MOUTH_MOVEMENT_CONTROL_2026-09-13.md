# Lite mouth movement: analysis and attenuation experiment

## Findings from the deployed pipeline

Lite has no jaw-opening, viseme, or expression-strength parameter. Wav2Vec2
extracts features from normalized eight-second rolling audio; the audio adapter
projects them into normalized conditioning tokens. Thirty DiT blocks inject
audio through cross-attention while generating full portrait latents. The LTX
VAE decodes them into pixels. The final nine frames become motion history for
the next 24 useful frames. A neutral portrait does not constrain future mouth
opening, as the earlier still-reference experiment demonstrated.

Relevant code:

- `soulx_rtc/engine.py`: `_audio`, `generate`, `recondition`.
- `flash_head/src/modules/flash_head_model.py`: `DiTAudioBlock.forward`,
  `AudioProjModel.forward`, `prepare_conditioning`.
- `flash_head/src/pipeline/flash_head_pipeline.py`: `preprocess_audio`, `generate`.
- `flash_head/ltx_video/ltx_vae.py`: posterior sampling and latent decoding.

Lowering WAV gain is largely canceled by the configured feature normalization.
The `audio_guide_scale` path is teacher-only and does not control Lite.
`sample_shift` controls the denoising schedule, not articulation. Boundary
crossfades only affect transitions. Neither is an established mouth fix.

## Candidate controls

1. Multiply the audio cross-attention output by a strength scalar before adding
   it to visual features. Minimal extra computation; no new model weights.
   This may affect head motion and lip timing as well as mouth opening, and
   strength 0.75 is not a promise of 25% less opening.
2. Blend projected speech toward actual silence conditioning before building
   cached keys/values. This requires a validated silence reference. It remains
   unimplemented in this experiment.
3. Test posterior mean for motion history to isolate sampling variability.
   Separate random streams are needed to keep later denoising noise matched.
   Less randomness does not necessarily mean less mouth opening.

The first experiment uses process-local hooks on all 30 cross-attention outputs.
It exercises the compiled optimized path with its existing conditioning cache.
No production API/default or model checkpoint is changed. Levels: 1.0, 0.9,
0.75, 0.5. Same ten-second speech, seeds 50/51, two repeats per level, existing
male portrait with fixed nine idle frames and girl portrait with ordinary still
initialization. Comparisons are controlled within each avatar, not across their
different framing/history. An unhooked baseline is retained separately and its
pixel difference from hooked strength 1 is measured explicitly.

Entrypoint: `benchmarks/mouth_strength/run.py` with `--output` (new directory)
and `--male-input` (directory containing `anchor.png` and `motion.npy`). Run in
the repository's Python environment with the GPU available. It never stops or
starts services. Video encoding and reference preparation are outside timings;
reported useful FPS counts only the 250 speech frames, not padded/idle frames.

Prepare inputs with `benchmarks/mouth_strength/prepare_input.py --idle PATH
--output NEW_INPUT_DIR`. This selects the first decoded/cropped idle frame as
portrait and playback indices 31 through 39 at 25 FPS as motion history.
The input source used here is the existing MuseTalk certified
`sample_ai_human_facetime_closeup_production_v1/certified/idle_active_listening.mp4`.
Run scripts from the repository with `PYTHONPATH=.` and `.venv/bin/python`.
After generation, run `compare.py OUTPUT_DIR` and `measure.py OUTPUT_DIR` from
the same script directory (using their full paths). `measure.py` requires
MediaPipe's solutions API (tested environment: 0.10.9).

The initial attempt discovered that registering hooks after the unmodified
model has compiled may leave the existing graph active. Its 1.0 and 0.9 outputs
were identical, so that attempt was stopped and excluded. The corrected runner
calls `torch._dynamo.reset()` after registration and checks both baseline
deviation and nontrivial output changes from hooked strength 1 at reduced levels. It uses a device
scalar tensor so strength changes do not require a distinct graph per value.
The retraced graph did not satisfy a strict pixel-identity check at strength 1;
that attempt was also excluded. The final sweep preserves an unmodified male
seed-50 control clip and quantifies numerical deviation rather than claiming
this experimental hook is a bit-identical production replacement.

## Quality interpretation

### Confirmed practical result

The comparison review confirms that reducing the audio-conditioning strength
does reduce the visibly overexaggerated mouth movement in these subjects. The
effect is clearest at 0.5 and remains visible at 0.75, while 0.9 is a mild
adjustment. This is the intended behavior for a future user-facing control.
The control should remain separate from the teeth solution: the male clips still
show smeared or unstable teeth when the mouth is open, including at reduced
strength. Lowering movement can reduce how often the defect is exposed, but it
does not reconstruct dental detail.

These are experimental controls, not trained disentangled expression parameters.
Inspect opening, consonant closures, head motion, teeth consistency, and temporal
stability. A quieter-looking mouth alone is not a lip-sync quality pass. Existing
BF16/INT8 tests show that quantization alone does not explain smeared teeth.
Upstream issue 21 attributes Lite detail loss to LTX VAE compression and advises
Pro: https://github.com/Soul-AILab/SoulX-FlashHead/issues/21#issuecomment-4196878092.
Attenuation does not restore missing dental detail.

## Completed results

Thirty-two measured renders completed: two avatars × two seeds × four levels ×
two repeats. Each retained clip has 250 useful frames at 25 FPS (ten seconds);
each render computed 264 padded frames. All repeats matched exactly per
avatar/seed/strength. Four comparison composites fully decoded without errors,
each 1280×576 with 250 frames. These are offline generation recordings with the
original audio, not WebRTC receiver captures. No lip-sync score was computed.

MediaPipe detected a face in all 4,000 decoded frames from the 16 sweep clips.
Opening proxy: pixel distance between inner-lip landmarks 13/14, divided by
eye-corner distance 33/263. Each table row averages the two seeds' statistics;
changes are relative to the same avatar's hooked strength-1 baseline. The p95
column averages per-clip p95 values rather than pooling frames. Frame-to-frame
change is mean absolute change in normalized opening, not a validated jitter
or naturalness score. This is one utterance and two seeds, not a population study.

| Avatar | Strength | Mean opening change | p95 opening change | Frame-to-frame opening change | Median useful FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Male | 1.0 | baseline | baseline | baseline | 53.74 |
| Male | 0.9 | −0.40% | −1.78% | −4.69% | 53.75 |
| Male | 0.75 | −5.17% | −14.16% | −13.93% | 53.78 |
| Male | 0.5 | −17.72% | −35.11% | −28.77% | 53.43 |
| Girl | 1.0 | baseline | baseline | baseline | 53.28 |
| Girl | 0.9 | −4.92% | −7.89% | −5.64% | 53.08 |
| Girl | 0.75 | −9.87% | −27.00% | −17.11% | 53.29 |
| Girl | 0.5 | −18.58% | −37.00% | −35.42% | 52.56 |

Peak Torch allocation was 3,354.74 MiB in all measured conditions. The timings
include generation, audio features and output transfer, but exclude preparation,
warmup, encoding and subsequent landmark analysis. FPS differences are small;
this establishes no speed improvement. The unrelated GPU service remained running.

The unmodified versus retraced-hook strength-1 male/seed-50 control had RGB MAE
0.92065/255 and maximum absolute pixel error 99/255. Retracing/multiplication
can change numerical behavior that then propagates autoregressively. This is
not bit-identical, nor proven perceptually equivalent. The sweep compares all
levels under the same hooked graph; the separate unmodified movie is retained
for review. This control covers only that avatar/seed.

Sparse visual review examined frames 12, 37, 75, 125 and 175 for both seeds and
avatars. Strength 0.5 visibly reduces some large mouth openings and changes
expressions, gaze and head position. The lower opening measurements agree with
that observation. Male teeth smearing persists. A smaller mouth is not proof of
better phoneme accuracy. Strength 0.75 is a reasonable moderate candidate for
further lip-sync review; 0.5 is a stronger reduction with more visible changes
outside the mouth. Neither is promoted to a production default.

## Evidence and playback

Published numerical evidence and reproducible scripts are under
`benchmarks/mouth_strength/`. `results.json` contains individual measured runs,
input hashes and the control deviation; `summary.json` contains throughput and
repeatability; `mouth-metrics.json` includes all per-frame opening estimates.
`prepare_input.py` was executed and reproduced the existing portrait and motion
files byte for byte. All four scripts passed syntax checks; `git diff --check`
passed. No unrelated runtime regression suite was rerun for these offline tools.

Full local evidence directory:
`/workspace/experiments/flashhead-mouth-3lwV6k/measured/outputs`.
Earlier aborted attempts are retained in the parent and `retry` directories,
excluded from the reported matrix. Private runtime snapshots outside the repo
must never be published; lifecycle scripts contain historical process state.

Columns in each comparison: **1.0, 0.9, 0.75, 0.5**, left to right.

- [Male, seed 50](/workspace/experiments/flashhead-mouth-3lwV6k/measured/outputs/male-seed-50-comparison.mp4)
- [Male, seed 51](/workspace/experiments/flashhead-mouth-3lwV6k/measured/outputs/male-seed-51-comparison.mp4)
- [Girl, seed 50](/workspace/experiments/flashhead-mouth-3lwV6k/measured/outputs/girl-seed-50-comparison.mp4)
- [Girl, seed 51](/workspace/experiments/flashhead-mouth-3lwV6k/measured/outputs/girl-seed-51-comparison.mp4)
- [Unmodified male control](/workspace/experiments/flashhead-mouth-3lwV6k/measured/outputs/male-seed-50-unmodified.mp4)

Videos/contact sheets remain on the workspace; these absolute links require
workspace access and are not publicly hosted GitHub videos.

After testing, the original FlashHead service was restored and verified ready
on port 1111 with 320×576, four steps, 25 FPS, INT8 storage, shared-memory
transport and startup GC freezing unchanged. Calls, sessions and GPU states
were all zero. The experiment hooks do not exist in the serving process.
