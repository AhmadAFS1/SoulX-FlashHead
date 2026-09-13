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
different framing/history. An unhooked baseline must match strength 1 exactly.

Entrypoint: `benchmarks/mouth_strength/run.py` with `--output` (new directory)
and `--male-input` (directory containing `anchor.png` and `motion.npy`). Run in
the repository's Python environment with the GPU available. It never stops or
starts services. Video encoding and reference preparation are outside timings;
reported useful FPS counts only the 250 speech frames, not padded/idle frames.

## Quality interpretation

These are experimental controls, not trained disentangled expression parameters.
Inspect opening, consonant closures, head motion, teeth consistency, and temporal
stability. A quieter-looking mouth alone is not a lip-sync quality pass. Existing
BF16/INT8 tests show that quantization alone does not explain smeared teeth.
Upstream issue 21 attributes Lite detail loss to LTX VAE compression and advises
Pro: https://github.com/Soul-AILab/SoulX-FlashHead/issues/21#issuecomment-4196878092.
Attenuation does not restore missing dental detail.

Results and artifact links will be added after the controlled sweep completes.
