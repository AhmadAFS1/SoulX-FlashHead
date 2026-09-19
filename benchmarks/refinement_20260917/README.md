# SoulX clean-latent refinement experiment

2026-09-17, **fresh GPU inference on NVIDIA GeForce RTX 4070 SUPER**.
Physical VRAM: 12 GB class; visible VRAM: **12,282 MiB**, from `nvidia-smi`.
Driver 595.84, PyTorch 2.7.1+cu128, CUDA runtime 12.8. The driver-reported
CUDA 13.2 is not the PyTorch runtime. OmniVoice (2,478 MiB) and the idle LTX
research server (234 MiB) remained resident; initial/final device use was
2,727 MiB. The engine held the checkout's exclusive GPU lease during inference.
CPU-only unit tests, image packaging and media validation are labeled below.

## Result

**Do not enable this as a teeth fix.** Across this portrait, two seeds and four
restart strengths, sampled mouth crops show no consistent recovery of tooth
separation. The lower-noise passes often soften the mouth. Stronger restarts
change mouth opening/expression and sometimes tooth exposure, without reliably
resolving the blurry dental texture. Six continuous evaluations also change
the mouth but do not consistently solve the issue.

This is a negative result for the tested implementation/profile, not proof that
every possible refinement method fails. No LTX weights, negative-prompt guidance,
new dental conditioning or retraining were introduced. A second pass through
the same model does not necessarily supply the missing detail.

Visual review used seven fixed timestamps per clip at 0.5, 1.5, 2.5, 3.5, 4.5,
6.5 and 8.5 seconds, both full frames and fixed mouth crops. At seed 50, for
example, lower-noise refinement softens the 0.5/3.5/6.5-second mouth regions;
the strongest restart changes the 2.5-second opening. Seed 51 likewise retains
soft/simplified tooth rows. These are subjective sampled observations, not a
dental accuracy metric. Identical audio does not guarantee identical mouth
pose. Full synchronized videos are provided for motion review; frame-by-frame
flicker, identity and audio/video synchronization have not been quantitatively
certified. There is no evidence here to claim improved lip sync.

## Implementation

`soulx_rtc/engine.py` now accepts the optional `refinement_timesteps=()` keyword.
The empty default executes the original sampler. `soulx_rtc/refinement.py`
implements the extra stage immediately before decoding:

1. Finish the stock four-step sampler and retain its clean prediction.
2. At each refinement timestep, mix that prediction with fresh Gaussian noise:
   `x_t = (1-t)*x_clean + t*noise`.
3. Restore each session's existing motion prefix, run the same model with the
   same portrait/audio conditioning, and predict `x_clean = x_t - t*flow`.
4. Restore the motion prefix again before decoding and normal chunk continuation.

The refinement RNG is per-session, seeded with `(seed + 1000003) % (2**63-1)`.
It does not consume baseline diffusion/motion-encoding random draws. Refined
video still changes subsequent motion conditioning, as expected for streaming.
This is a whole-latent pass, not a mouth mask or pixel sharpening operation.

Values supplied to the option are raw pre-shift coordinates; the existing shift
of 5 is applied. The lowest-noise experimental values are outside the stock
four-step schedule, so no claim is made that the distilled checkpoint was
trained for them. The strongest setting repeats the stock final two values.

```python
# Experimental only; keep the other profile arguments unchanged.
engine = Engine(..., refinement_timesteps=(500, 250))
# Default: refinement_timesteps=()
```

Set the option before preparing sessions. The experiment changes it only between
completed calls. No live service configuration, model weights or defaults were
changed. The stock upstream FlashHeadPipeline.generate path is not modified;
the implementation is in this checkout's RTC Engine, used for these clips.

## Matched runs

All rows: **RTX 4070 SUPER**, 320×576, 25 FPS, 250 frames/10 seconds, seeds 50
and 51, INT8 weight storage with BF16 computation, compact memory mode,
3,584 MiB allocator cap (not physical VRAM), no compilation, native portrait,
same external audio, color correction 1.0. Optimized conditioning, real RoPE,
lean execution and fused QKV were enabled in every arm.

| Arm | Model calls/chunk | Raw schedule after normal four steps | Effective noise fractions | Seed 50 generation | Seed 51 generation |
| --- | ---: | --- | --- | ---: | ---: |
| Normal | 4 | None | None | 7.939 s | 8.043 s |
| Continuous six | 6 | Separate six-step schedule | See JSON | 9.455 s | 9.508 s |
| Refine 100/50 | 6 | 100, 50 | 0.3571, 0.2083 | 9.479 s | 9.528 s |
| Refine 50/25 | 6 | 50, 25 | 0.2083, 0.1136 | 9.479 s | 9.553 s |
| Refine 250/125 | 6 | 250, 125 | 0.6250, 0.4167 | 9.510 s | 9.541 s |
| Refine 500/250 | 6 | 500, 250 | 0.8333, 0.6250 | 9.523 s | 9.565 s |

Timings include the warmed engine's audio, generation, decode, color and motion
encoding loop; exclude model loading, warmup and MP4 encoding. These are two
individual timings, not a repeated statistical throughput benchmark. Added
latency is approximately 18–20%. Peak allocated memory was approximately
3,301–3,304 MiB; reserved 3,530–3,544 MiB. These allocator measurements exclude
co-resident processes. Exact per-run values are in [results.json](results.json).

Stock raw schedule: `[1000,750,500,250,0]`. The experiment-only six-step control
uses `linspace(1000,0,7)`, transformed by the same shift. It has equal model
evaluation count to the restart arms but changes time placement and consumes
more baseline RNG draws. It is not claimed to be an official trained six-step
schedule. Comparing the restart arms with baseline holds baseline RNG draws
fixed; comparing with continuous six does not.

## Validation and provenance

- **CPU-only:** 8 tests passed across `test_refinement.py` and
  `test_optimizations.py`; sampler tests cover a known flow oracle, distinct
  batch motion prefixes, deterministic refinement, input preservation,
  disabled behavior and invalid schedules. Static compilation and diff checks
  passed. Existing dependency deprecation warnings remain.
- **GPU:** disabling refinement after the first sweep reproduces all 250 raw
  baseline frames exactly (seed 50). The raw hash also matches the previous
  neutral-reference benchmark; the current baseline was generated afresh.
- **GPU:** all eight refined runs finish with exactly the same baseline RNG
  state as their matching normal run. Independent refinement RNG is therefore
  verified through real multi-chunk execution, not merely a CPU mock.
- **CPU-only media validation:** each clip has 250 video frames at 25 FPS and an
  audio stream, checked with ffprobe and full ffmpeg decoding. Four comparison
  videos contain native full frames plus enlarged fixed 128×80 mouth crops.
  Crop sheets use 2× nearest-neighbor enlargement, without sharpening.
- Thirteen inference clips: six conditions × two seeds plus a restored-default
  repeat. Both research processes exited and released their GPU leases after
  completion; unrelated resident applications were left intact.

Base checkout commit: `6d2ba83a85f58c8d8fbe7cfcc9f1dff432af8d66` plus the
documented local sampler change. Exact input hashes and runtime records are in
`results.json`; generation logs are `run.log` and `extra.log`.

Weight SHA-256:

- Model_Lite: `aaf1cde6e80ca23f740aae236c47954249f65b151db133cc0f77d3a138ccdf6e`
- VAE_LTX: `265ca87cb5dff5e37f924286e957324e282fe7710a952a7dafc0df43883e2010`

## Review and reproduction

- [All synchronized comparisons](review.html)
- [Lower-noise comparison, seed 50](soulx-LITE-4steps-vs-6steps-vs-low-noise-refinement-seed50.mp4)
- [Stronger comparison, seed 50](soulx-LITE-4steps-vs-6steps-vs-strong-refinement-seed50.mp4)
- [Lower-noise mouth crops, seed 50](mouths-seed50.png)
- [Stronger mouth crops, seed 51](mouths-seed51-strong.png)
- [Machine-readable results](results.json)
- [Media validation](media-validation.json)

From the repository root, use a fresh artifact directory/copy of the runner:
`PYTHONPATH=. .venv/bin/python benchmarks/refinement_20260917/run.py`, then
the same command with `--extra`. The runner refuses to overwrite clip
directories. Run `package.py` normally and with `--strong` for both sets of
review artifacts. Do not rerun into this preserved results directory.
