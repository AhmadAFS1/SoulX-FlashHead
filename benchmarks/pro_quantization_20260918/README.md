# PRO compute quantization: Indian-man 1.50× comparison

September 18, 2026. **Fresh local GPU inference on NVIDIA GeForce RTX 4070 SUPER, physical 12 GB class / 12,282 MiB visible VRAM**, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8. Direct `nvidia-smi` snapshots and per-run telemetry are retained. Two unrelated Python processes remained resident, reporting 2,478 and 234 MiB; initial device use was 2,727 MiB. CPU tests, aggregation, FFmpeg packaging, and landmark analysis are distinguished from generation below.

**Result:** the selected FP8 implementation generates about **9.4 useful FPS versus 7.1 FPS for stock PRO**, consistently **31–32% faster** across four seeds. Visible PRO tooth detail remains in the inspected crops, with seed-dependent articulation/pose differences. It does **not** reach 15 or 25 FPS on this GPU. Start with the [seed-50 video](comparison-seed50/comparison.mp4), [seed-51 video](comparison-seed51/comparison.mp4), and [raw teeth crops](comparison-seed50/mouth-comparison.png).
Accepted reference name: **[PRO FP8 v1](PRO_FP8/README.md)**. The source/configuration snapshot preserves this tested candidate. See the [further-quantization analysis](../../docs/research/PRO_FURTHER_QUANTIZATION_2026-09-18.md) for proposed next variants; no new inference accompanies that analysis.


## Scope and selected implementation

This implements and tests an experimental PRO optimization path. The live service and original model checkpoints are unchanged. The primary candidate retains FlashAttention2 and combines:

- **Real FP8 E4M3 weight-and-activation GEMMs** in all 60 PRO FFN linear layers. Weight scales are tensorwise; activation scales are recomputed from each actual input. Biases and external interfaces remain BF16. `torch._scaled_mm` dispatch was verified as SM89 E4M3 GPU kernels.
- Compiled transformer execution with the repository's existing prepared real-valued rotary implementation, cached per-profile timestep embeddings, and audio conditioning/cross-attention K/V prepared once per chunk.
- Compiled **BF16 Wan VAE decode**. The VAE weights, image/reference representation, audio pathways, output head, normalization, sampler, and four-step schedule are not quantized by the FFN policy.

The policy lives in [pro_quantization.py](../../soulx_rtc/pro_quantization.py); the explicit offline integration is [run.py](run.py). Lower precision does not isolate a teeth module: all FFNs affect the generated representation. Complete recurrent videos are the quality check.

The primary candidate was selected over a slightly faster SageAttention 1.0.6 combination because the additional gain was only approximately 0.5 FPS and its seed-50 mouth-opening trajectory differed more from stock. The faster trial remains inspectable in [its comparison](review-fp8-sage-optimized-seed50/comparison.mp4). Neither candidate establishes real-time 25-FPS service.

## Controlled inputs and timing

The exact reference and audio are reused from [the prior Indian-man 1.50× experiment](../pro_lite_150x_20260917/README.md). All runs use 320×576, native 25-FPS output, four denoising steps, shift 5, strength 1, two motion latent frames, five decoded history frames, and 28 new frames per PRO window. The ten-second comparisons contain exactly 250 frames. No mouth restoration, sharpening, endpoint replacement, or redubbing is applied. The fresh stock seed-50 raw RGB hash exactly matches the retained prior PRO 1.50× run, and reference/audio hashes also match: [reproduction check](stock-reproduction.json).

Model/config/reference/audio hashes, backend selection, policy coverage, actual GPU load, and source hashes are recorded in each `results.json`. Final validation runs also retain source snapshots. Each timing run follows two warmup chunks, resets reference history and the private RNG, and includes audio processing, denoising, VAE decode, color correction, motion feedback encode, and RGB transfer. MP4 encoding/review is outside generation timing. The first and final padded windows remain included in useful-FPS accounting. Compilation/warmup time is reported separately.

These are offline generation benchmarks with GPU stage events. They do not measure WebRTC delivery, interactive startup/audio buffering, interruptions, or concurrent-call capacity. Benchmark repeats are sequential, not a randomized interleaved A/B study.

## Initial attribution experiments

All rows below ran on **RTX 4070 SUPER / 12,282 MiB** with the same 1.50× seed-50 fixture and software above. See [summary.json](summary.json) and each run manifest for exact repetition counts and memory.

| Configuration | Useful FPS | Interpretation |
| --- | ---: | --- |
| Stock eager BF16 PRO | 7.16 median of 3 | Fresh reference |
| FP8 FFNs, compiled FFNs only | 7.70 | Quantization's initial full-pipeline gain |
| BF16 compiled FFNs + compiled Wan | 7.57 | Decoder/FFN compilation control |
| FP8 compiled FFNs + compiled Wan | 8.17 median of 3 | Combined first candidate |
| Compiled full DiT/conditioning + compiled Wan, BF16 | 8.63 | Matched advanced nonquantized implementation |
| Same advanced implementation, FP8 FFNs | 9.43 | Primary candidate before multi-seed validation |
| SageAttention 1 only | 7.47 | Independent quantized-attention trial |
| Advanced FP8 + SageAttention 1 | 9.93 | Faster alternative, more mouth-motion deviation |

The improvement from 7.16 to 9.43 FPS includes compilation/caching as well as quantization. Against the matched advanced BF16 implementation, the initial FP8 gain is approximately **9.2%**. It would be incorrect to attribute the entire approximately 32% stock-to-candidate gain to quantization alone.

FP8 reduces resident FFN storage by about 787.5 MiB before scale metadata. **The compiled candidate uses more peak memory than the stock eager pipeline**, because decoder/compiler buffers outweigh that saving. The initial advanced BF16 run reached 11,726 MiB sampled whole-device use; the FP8 counterpart reached 10,824 MiB, including co-residents. Whole-device figures differ from Torch allocator counters and exclude unsampled transients.

## Kernel and decoder evidence

[Kernel results](kernel-results.json) use actual first-step PRO activations from blocks 0, 14, and 29 at `[1,6480,1536]`, with their real checkpoint weights. Full FFN latency includes activation quantization and the two projections/GELU. Across these samples:

- BF16: approximately 5.57–5.60 ms per FFN.
- Compiled FP8: approximately 3.38–3.43 ms, about 1.6× faster.
- Eager FP8: approximately 7.95–7.97 ms, slower than BF16.
- Compiled INT8: approximately 6.60–6.86 ms, also slower than BF16. This implementation was not promoted to video testing.

FP8 relative output L2 errors on these three FFNs were approximately 1.2–4.8%; these are tensor diagnostics, not video quality guarantees. Tests included finite/zero inputs, changing inputs after compilation, state-dict save/reload, and profiler dispatch. Compiled outputs have small additional rounding differences. No kernel-level speedup is presented as a whole-pipeline speedup.

Captured baseline activation statistics cover all 30 FFN inputs over four steps in a first and recurrent window. The selected policy uses dynamic activation scaling, so this is diagnostic coverage rather than fitted static calibration. There is no claim of a completed layer-by-layer sensitivity sweep or validation on other identities.

Wan trials used exactly the same saved PRO latents and unchanged BF16 precision:

- [Channels-last decoder weights](vae-channels_last.json): approximately 1.66→1.51 s.
- [Full decoder compilation](vae-compiled.json): approximately 1.66→1.45 s; selected. Fixed-latent relative L2 change approximately 0.0035.
- [Isolated pointwise compilation](vae-pointwise.json): failed on a Torch data-dependent scalar tracing error; retained as a failed trial, not a working optimization.

SageAttention was installed only into `.pro-quant-deps`, using [the pinned optional requirement](requirements-attention.txt). The host CUDA compiler is 12.1, so this trial used the Triton-only 1.0.6 path rather than assuming the newer Ada FP8 extension could compile. It quantizes self-attention Q/K; audio cross-attention remains FlashAttention2. Its in-place key smoothing receives a clone to protect caller-owned inputs. The primary candidate does not require this package.

## Quality and final validation

Four final short-clip pairs completed on **RTX 4070 SUPER / 12,282 MiB**, at the exact shared profile above. Each input and comparison has 250 frames; all 1,000 frames per variant had detected faces. Seed 50 uses three timing repetitions per variant; the other seeds use one each.

| Seed | Stock useful FPS | Candidate useful FPS | Speedup | Stock versus candidate |
| --- | ---: | ---: | ---: | --- |
| 50 | 7.156 | 9.412 | 1.315× | [Video](comparison-seed50/comparison.mp4), [raw mouth crops](comparison-seed50/mouth-comparison.png) |
| 51 | 7.118 | 9.370 | 1.316× | [Video](comparison-seed51/comparison.mp4), [raw mouth crops](comparison-seed51/mouth-comparison.png) |
| 0 | 7.157 | 9.438 | 1.319× | [Video](comparison-seed0/comparison.mp4), [raw mouth crops](comparison-seed0/mouth-comparison.png) |
| 1 | 7.144 | 9.379 | 1.313× | [Video](comparison-seed1/comparison.mp4), [raw mouth crops](comparison-seed1/mouth-comparison.png) |

The selected implementation is consistently **about 31–32% faster than stock**, but still below either 15 or 25 useful FPS. The MP4s play at native 25 FPS; that playback rate is not generation throughput.

Visual inspection of seven raw mouth samples per seed shows preserved visible PRO tooth detail rather than a uniform loss of boundaries. It does not establish identical output: normalized mouth-opening trajectory correlations are 0.982, 0.910, 0.934, and 0.972 for seeds 50, 51, 0, and 1. Median mouth-center displacements are approximately 3.5, 6.5, 11.7, and 2.8 pixels, so head motion and articulation have changed in a seed-dependent way. Seed 51 also has fewer frames meeting the four-pixel mouth-opening threshold. These changes must remain part of the user's quality review.

On frames where both mouths are open, median candidate/stock oral edge-energy ratios range from 0.941 to 1.130. This supports the absence of a consistent edge-energy collapse, not proof of dental correctness. PRO's existing bright merged-tooth bands still occur; this experiment does not repair dental anatomy. No new identity/utterance generalization claim is made.

### Sixty-second recurrence check

The matched seed-50 long runs also completed on **RTX 4070 SUPER, physical 12 GB class / 12,282 MiB visible**, with the same runtime and co-residents stated above. Stock generated 1,500 frames in 209.91 seconds (**7.146 useful FPS**); the candidate took 159.25 seconds (**9.419 useful FPS**, 1.318×). Both had faces detected in all 1,500 frames. See the [60-second comparison](comparison-60s-seed50/comparison.mp4), [raw sample sheet](comparison-60s-seed50/mouth-comparison.png), and [decoded-video tooth crops across the minute](comparison-60s-seed50/teeth-through-60s.png).

The paired mouth-opening correlation was 0.911, median mouth-center displacement 5.6 pixels, and median oral edge-energy ratio on jointly open frames 1.145. Inspected samples retain visible tooth divisions through the minute; this is not an anatomical or flicker certification. The input repeats the same ten-second utterance six times. It does not test six independent utterances, interruptions, WebRTC, or a long live-call soak.

### Relative audio-sync diagnostic

A fresh GPU diagnostic used the existing MuseTalk/LatentSync SyncNet checkpoint, whose hash is recorded in [the results](sync-results-own-boxes.json), on the same **RTX 4070 SUPER / 12,282 MiB** and runtime. This evaluation did not generate video. It compared 16-frame lower-face windows against mel features across offsets of −5 through +5 frames. Higher cosine similarity is better within this diagnostic; these are **not official LSE-C/LSE-D scores**. Windows overlap and cover one identity/utterance.

Using each output's own face boxes:

| Seed | Stock best similarity | Candidate best similarity | Stock → candidate best lag | Stock → candidate zero-lag similarity |
| --- | ---: | ---: | --- | --- |
| 50 | 0.722 | 0.745 | +1 → +1 frame | 0.708 → 0.683 |
| 51 | 0.762 | 0.761 | +1 → +1 frame | 0.697 → 0.644 |
| 0 | 0.769 | 0.771 | +1 → +1 frame | 0.625 → 0.621 |
| 1 | 0.720 | 0.733 | 0 → +1 frame | 0.720 → 0.731 |

Positive lag pairs the video with later audio; one frame is 40 ms. Seed 1's candidate scores at zero and +1 frame are nearly tied (0.731 versus 0.733), so that change is weak evidence of an actual timing shift. Seed 51's zero-lag decrease remains a reason to review articulation even though its best-offset similarity is unchanged.

The initial [common-box diagnostic](sync-results.json) reused stock face boxes for both outputs and showed a seed-0 best-score decrease from 0.769 to 0.677. Re-cropping each output's own moving face removes that decrease (candidate 0.771). This demonstrates crop sensitivity; it does not prove lip-sync equivalence. Both analyses are retained rather than reporting only the more favorable one.

### Decision and remaining work

This is a working, tested offline FP8 prototype with a measured speed gain and inspectable PRO-versus-candidate footage. It is **not a qualified real-time replacement**. The main candidate preserves the chosen four-step, shift-5, history-2 profile and BF16 decoder, but quantization and compilation change the recurrent motion trajectory. No full layer-sensitivity sweep, static fitted calibration, INT4 video trial, other-identity evaluation, live integration, or concurrent-call qualification is claimed.

The next speed work should target Wan decode and the remaining transformer cost together. In the first final seed-50 candidate repeat, CUDA events account for 13.09 seconds of Wan decode and 11.76 seconds of DiT work within 26.56 seconds of generation. Decoder time alone already exceeds the ten-second output duration, so eliminating transformer cost would still not establish 25-FPS generation. Selective BF16 restoration remains available for quality sensitivity testing via the exclusion list.

[Final media checks](media-validation.json) verify panel dimensions, frame counts, 25-FPS playback, duration, audio presence, and complete decoding for all ten final source videos and five comparisons. These are CPU checks and make no generation-speed measurement.

All review measurements are relative diagnostics. Edge energy cannot distinguish correct tooth boundaries from contrast, ringing, or invented divisions. Landmark motion agreement does not prove phoneme accuracy. Raw pre-encode sample crops and same-settings encoded comparisons are both retained. [review.py](review.py) reports face detections, mouth opening/exposure, paired edge-energy diagnostics, and media metadata.

## Reproduction

Run from the repository root with its existing `.venv`. The GPU lease prevents a second SoulX owner from loading concurrently; other resident applications are left running. Use a new output directory for every run.

```bash
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_20260918/run.py \
  --output benchmarks/pro_quantization_20260918/NEW-stock-seed50 --seed 50

PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_20260918/run.py \
  --output benchmarks/pro_quantization_20260918/NEW-candidate-seed50 --seed 50 \
  --precision fp8 --optimized-dit --compile-dit --vae compiled --repeats 3

PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_20260918/review.py \
  --baseline benchmarks/pro_quantization_20260918/NEW-stock-seed50 \
  --candidate benchmarks/pro_quantization_20260918/NEW-candidate-seed50 \
  --output benchmarks/pro_quantization_20260918/NEW-comparison-seed50
```

Use `--frames 1500` for the 60-second recurrent test; its audio repeats the known ten-second utterance. This is a recurrence stress test, not six independent utterances. `--exclude blocks.0.ffn.0 ...` supports later selective precision restoration. The default policy is `none`; quantization is never silently enabled in the live engine.

The implementation uses private PyTorch `_scaled_mm` / `_int_mm` interfaces and is tested on the recorded Torch/CUDA/GPU combination. Other versions/hardware require the kernel checks again. Six new CPU policy checks plus six existing conditioning/rotary tests passed: [focused-tests.log](focused-tests.log). They are separate from the fresh GPU experiments.
