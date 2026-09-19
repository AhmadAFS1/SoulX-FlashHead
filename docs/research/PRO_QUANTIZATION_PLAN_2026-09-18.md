# PRO quantization plan: preserve teeth detail while improving generation speed

Implementation follow-up: [tested FP8 prototype, attribution experiments, four-seed comparisons and 60-second recurrence check](../../benchmarks/pro_quantization_20260918/README.md). Fresh September 18 inference on **NVIDIA GeForce RTX 4070 SUPER, physical 12 GB class / 12,282 MiB visible**, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8, with co-residents reporting 2,478 and 234 MiB. About 7.1→9.4 useful FPS; not real-time qualification or completion of every proposed research branch. The original planning snapshot below is retained.

September 18, 2026. **Planning and CPU source/log/checkpoint-header inspection only; no new GPU inference, quantization, training, or deployment.** Current hardware, directly queried with `nvidia-smi`: **NVIDIA GeForce RTX 4070 SUPER, physical 12 GB class / 12,282 MiB visible VRAM, SM 8.9**, driver 595.84, 2,727 MiB device memory in use at inspection. Project environment: Torch 2.7.1+cu128 / CUDA 12.8, FlashAttention 2.8.0.post2, Triton 3.3.1. Historical runs cited below record their own hardware and load.

## Recommendation and scope

Start from the released PRO checkpoint and Wan VAE. Apply post-training **mixed-precision quantization to selected transformer operations**, keeping the original spatial representation, four denoising steps, and initially the entire VAE at its reference precision. Evaluate teeth on generated videos, then restore higher precision wherever quantization visibly damages them.

There is no separately addressable teeth network in this implementation. Transformer weights act across the frame; quantizing only spatial mouth tokens is not a simple weight conversion. Keeping the VAE in BF16 also cannot protect against detail already lost in quantized transformer latents. Sensitivity experiments must identify which operations need protection.

The initial objective is one useful live stream at **320×576 / 25 FPS on the current GPU**. This is a provisional planning target, pending the user's speed/concurrency preference. Treat 15 FPS as a separately validated fallback profile. Concurrency is a later capacity experiment. Quantization is a promising experiment, not evidence that 25 FPS is achievable here.

The work should deliver the fastest measured configuration that retains the user's preferred PRO teeth appearance. Start with PTQ; training and mouth-specific model redesign are escalation paths, not prerequisites.

## Evidence that changes the plan

### PRO's advantage is architectural

The [matched normal-distance experiment](../../benchmarks/pro_lite_normal_distance_20260917/README.md) used 320×576, four steps, seed 50, BF16, and eager execution on **RTX 4070 SUPER / 12,282 MiB**, driver 595.84, Torch 2.7.1+cu128. Co-resident processes used 2,478 and 234 MiB. It measured **PRO 7.124 useful FPS** and LITE 56.18 useful FPS. These are historical single runs, not current capacity measurements.

The local [PRO config](../../models/SoulX-FlashHead-1_3B/Model_Pro/config.json) and [LITE config](../../models/SoulX-FlashHead-1_3B/Model_Lite/config.json) give:

| At 320×576, 33 decoded frames per window | PRO | LITE |
| --- | --- | --- |
| VAE | Wan2.1 | LTX |
| VAE temporal × spatial stride | 4×8×8 | 8×32×32 |
| Latent tensor, before batching | 16×9×72×40 | 128×5×18×10 |
| DiT patch size | 1×2×2 | 1×1×1 |
| Transformer tokens per window | 9×36×20 = **6,480** | 5×18×10 = **900** |
| Motion history at two latent frames | **5 decoded frames** | **9 decoded frames** |
| New frames per recurrent window | **28** | **24** |

PRO therefore processes 7.2× as many tokens in this profile. This does not imply exactly 7.2× runtime: attention, linear operations, and the different VAEs scale differently. Denser representation and decoder differences plausibly explain the detail advantage; existing comparisons do not isolate their separate contributions. The [authors' architecture description](https://arxiv.org/html/2602.07449v1) also distinguishes the VAE tradeoff.

The [square comparison](../../benchmarks/pro_lite_teeth_20260917/README.md) and normal-distance comparison show sharper PRO mouth detail, but also occasional merged white tooth bands. The goal is to preserve PRO's demonstrated advantage, not claim anatomically correct teeth in every frame.

### Transformer quantization alone cannot meet the provisional target

CPU reanalysis of the nine chunks in the historical [PRO log](../../benchmarks/pro_lite_normal_distance_20260917/pro.log) and [run manifest](../../benchmarks/pro_lite_normal_distance_20260917/pro/results.json), on the RTX 4070 SUPER configuration above:

| Stage | Mean time per chunk | Approximate share of generation |
| --- | ---: | ---: |
| Four denoising calls | 2.028 s | 52.0% |
| Wan VAE decode | 1.663 s | 42.6% |
| Motion-history VAE encode | 0.152 s | 3.9% |
| Color correction | 0.0097 s | 0.25% |
| Remaining work / timing boundaries | About 0.046 s | About 1.2% |

Total generation was 35.091 s for 250 useful frames; the final full chunk includes two discarded frames. These synchronized historical stage logs are a coarse breakdown, not a new kernel profile.

With all other historical work fixed, halving denoising time predicts only **9.63 useful FPS**. Setting denoising time to zero predicts **14.85 useful FPS**. Formula: `250 / (35.0913 - old_denoising_seconds + new_denoising_seconds)`. These are conditional arithmetic bounds, not measured improvements.

Reaching 25 FPS requires about **3.51×** the historical overall speed; 15 FPS requires about **2.11×**. Wan VAE acceleration must therefore be a first-class work item for the 25-FPS objective, initially without reducing its precision.

### Existing tooling needs PRO-specific adaptation

- [compact_weights.py](../../soulx_rtc/compact_weights.py) implements INT8 **storage**, then dequantizes weights for `F.linear`. It is not evidence of INT8 Tensor Core computation. The prior LITE INT8/BF16 capacity comparison also changed allocator caps; it does not isolate quantization's speed effect.
- [Engine](../../soulx_rtc/engine.py) currently loads LITE and embeds its temporal/latent assumptions. Replacing only the checkpoint name would be incorrect.
- [FFN export](../../soulx_rtc/trt_experiment.py) and [TRT installation](../../soulx_rtc/trt_backend.py) hardcode LITE weights/shapes. Capture currently retains only each FFN's first invocation. That is insufficient calibration coverage.
- Existing LTX VAE engines cannot decode PRO latents. Wan requires its own implementation and validation, including temporal caches.
- A historical **512×512 PRO** compile attempt failed during Wan decoder autotuning with CUDA OOM. It does not prove 320×576 compilation fails, but full-model compilation must not be assumed to fit.
- TorchAO, SageAttention, ONNX, and TensorRT were not installed in the inspected project `.venv`. Other repository experiments used separate environments. Pin a separate quantization environment and reproduce BF16 there before comparing precision changes.

## Stage 1 — Freeze fixtures and acceptance criteria

Use the [movement protocol](MOVEMENT_SEED_PROTOCOL_2026-09-17.md): seeds **50, 51, 0, 1**, shift **5**, two motion latent frames, and audio strength **1.0**. Keep four denoising steps and the original audio window, color correction, crop, and preprocessing. For PRO, two latent history frames mean **five decoded history frames**, not LITE's nine. Preserve first-window handling separately.

Primary fixtures:

1. Existing Indian-man normal reference and 1.25× closer reference, with `benchmarks/comparison-10s.wav`.
2. The exact portrait/audio from the PRO-to-MuseTalk experiment, evaluated on **raw PRO output**, before endpoint editing or redubbing.
3. Several additional portraits and held-out utterances covering visible teeth, closed lips, rapid speech, vowels, lip closures, silence, and speech/silence transitions. Include the deployed avatar identity as an explicit fixture. Select and hash these before tuning.

Begin with two primary fixtures and seed 50 for fast rejection. Expand survivors to all four seeds and held-out identities/audio. Use separate calibration, development, and final evaluation splits; seeds 50/51 may support development, with 0/1 and distinct audio/identities reserved for final evaluation. Do not repeatedly tune on the final split.

Proposed acceptance gates, to fix before comparing candidates:

- **Appearance:** no material increase over matched BF16 PRO in merged tooth bands, lost tooth boundaries, false teeth behind closed lips, shimmering, lip distortion, or identity drift. Review randomized A/B videos at natural size and synchronized enlarged mouth crops. Require the user's visual acceptance of the final speed/quality tradeoff.
- **Lip sync:** no systematic timing shift beyond one 25-FPS frame (40 ms), and no consistent regression beyond the evaluator's measured baseline variability. Report failures per fixture. The repository's relative SyncNet evaluator is a diagnostic, not an official dental or lip-sync certification.
- **Temporal behavior:** no increased boundary jumps or accumulating mouth/face errors across first, recurrent, and long sequences.
- **Speed:** at 25 FPS, a recurrent 28-frame chunk must complete within **1.12 s** to keep up. An illustrative 20% time reserve targets **≤0.896 s**, equivalent to 31.25 generated FPS. Validate p95 chunk latency and sustained throughput, not just the mean. First-frame latency includes audio buffering and startup and must be reported separately.
- **Resources:** fits the actual shared 12-GB host including compile/build/reload peaks and non-Torch allocations. Proposed operational reserve: at least 1 GiB device headroom under representative co-resident load, subject to measured workload variability.

A 15-FPS profile has different timing/conditioning implications and requires its own quality and streaming checks. Slowing playback, repeating frames, or discarding frames from an unchanged expensive 25-FPS generator is not evidence of a faster generator.

## Stage 2 — Reproduce and profile PRO

Extend the tested [PRO/LITE offline runner](../../benchmarks/pro_lite_teeth_20260917/run_variant.py) into a dedicated experiment harness before changing the live engine. Record checkpoint/config/input hashes, git revision and dirty diff, GPU/driver/runtime versions, active attention backend, memory budget, and co-resident load.

Establish two controls: original eager BF16 PRO for quality/reproduction, and the fastest quality-approved BF16 PRO implementation for fair quantization comparisons. If dependency upgrades are necessary, reproduce both controls in that environment. Keep an explicit `none` precision policy.

Use separate diagnostic and timing runs. Profile Wav2Vec, conditioning, each denoising step, FFNs, self-attention, cross-attention, Wan decode, feedback encode, color, copies, and later encoding/transport. CUDA events and profiler ranges should reveal actual kernels and copies; observers/tensor capture must be disabled for timing. Record cold start, warmup, and steady state separately. Alternate baseline/candidate order and collect at least three measured repetitions with median, dispersion, and latency tails.

At this stage derive an updated time budget. Keep the existing GPU lease mechanism; experiments must not silently evict resident services or treat a less-loaded run as a quantization win.

**Deliverable:** reproducible reference videos, mouth crops, workload manifest, and PRO stage/kernel profile.

## Stage 3 — Accelerate Wan while retaining its reference precision

This stage is essential to the 25-FPS target, based on the existing timing bound. Isolate Wan encode/decode from transformer tuning so quality effects remain attributable.

Try bounded `torch.compile` on decoder submodules or temporal processing units, with conservative autotuning/workspace settings. Compare exact-shape BF16 execution paths and measured layout/kernel choices. If needed, build a separate Wan BF16 TensorRT partition; inspect operator support and memory before attempting a full decoder export. Preserve temporal padding, cache reset/update, normalization, latent scaling, clipping, and feedback semantics.

Test with identical saved BF16 PRO latents to isolate decoder numerical changes, then with complete recurrent generation. Compilation/fusion can change rounding even when nominal precision is unchanged. Include fresh-process reload and peak-memory checks.

Do not remove overlap or decode only a mouth crop as a routine optimization: these change context and are separate experiments. If BF16 Wan acceleration is insufficient, selectively quantizing internal decoder convolutions is a later, higher-risk branch, keeping its input/output and sensitive stages in reference precision and validating on fixed latents first.

**Deliverable:** a measured decoder result and an updated realistic speed ceiling. Stop claiming a 25-FPS path if the remaining time budget is already exceeded.

## Stage 4 — Test actual low-precision transformer kernels

First kernel target: `blocks[i].ffn.0` and `blocks[i].ffn.2`, the two large FFN linear operations in each of 30 blocks. CPU inspection of the PRO safetensors header found **825,753,600 FFN weight parameters**, about **54.77% of the 1,507,694,912 stored model parameters**. These counts include checkpoint contents, not a claim that all stored tensors execute. BF16-to-eight-bit FFN storage could save about **787.5 MiB before scales/metadata**; memory savings are not a speed prediction.

At this profile test real inputs shaped `[1,6480,1536]`, the `1536→8960` projection, GELU, and `8960→1536` projection. Include quantization/scaling, layout conversion, bias, activation, launch and output-conversion costs. Synthetic square matrix benchmarks are insufficient.

| Candidate | Initial scope | Decision |
| --- | --- | --- |
| FP8 weights + FP8 activations | Selected FFN linears, BF16 interfaces | Preferred first real-compute PTQ experiment if the pinned SM89 kernels work and are faster |
| INT8 weights + INT8 activations | Same FFN scope | Alternative if FP8 is unsupported, slower, or visibly worse |
| Quantized attention through SageAttention | Self-attention first, existing FlashAttention baseline | Independent experiment, promoted by measured PRO quality and speed |
| INT4 weight-only | Selected FFNs only | Later memory/speed tradeoff after eight-bit tests; no automatic expectation of faster compute |

Use TorchAO as the first PyTorch integration candidate, subject to exact Torch/CUDA/SM89 compatibility. Current [TorchAO documentation](https://docs.pytorch.org/ao/stable/workflows/inference.html) distinguishes weight-only and activation-plus-weight recipes and documents their hardware constraints. Latest APIs must not be assumed compatible with the installed Torch 2.7.1 environment. Test supported scaling granularity on this GPU; keep a tensorwise FP8 option if the selected rowwise kernel is unavailable. INT8 per-token activation/per-output-channel weight scales are a candidate where supported. Outlier/clipping choices require sensitivity validation.

If PyTorch integration is unsuitable, extend the existing FFN TensorRT tooling with explicit scales and Q/DQ, PRO checkpoint provenance, and PRO shapes. [NVIDIA's explicit quantization guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-with-quantized-types.html) describes that route. Inspect the built engine/profiler to verify lower-precision GEMMs actually execute; export success or fake quantization is insufficient. FP8 export needs an appropriate supported opset/backend, not the existing BF16 opset-17 path unchanged.

SageAttention is a separate activation-quantized attention implementation with [official Ada examples](https://github.com/thu-ml/SageAttention/blob/main/bench/README.md). It is not an FP8 checkpoint conversion. The local attention function automatically prefers Sage when importable, so introduce explicit per-run/backend selection before installing it; otherwise even the BF16 control may silently change. Keep audio cross-attention on the reference backend initially.

Keep the Wan VAE, patch embedding, output head, audio encoder/projection, audio cross-attention projections, timestep/modulation pathways, normalization, residual arithmetic, and sampler at their validated precision. Preserve existing FP32 operations where used; do not globally cast everything to BF16. For a later Sage trial, use its validated internal arithmetic rather than ad hoc low-bit softmax casts.

**Kernel gate:** promote a backend only after correct finite outputs, actual lower-precision dispatch, and a useful full-operation latency gain beyond noise. A provisional ≥15% FFN gain is a reasonable engineering filter, not a product acceptance threshold.

## Stage 5 — Calibrate and identify sensitive layers

Collect representative PRO activations from every actual shifted denoising timestep, first/recurrent/tail windows, early and late speech, silence/transitions, multiple identities, and both framings. Include FFN input and post-GELU input to the second linear operation. Two motion-prefix lengths must be represented. Store streaming statistics and bounded samples instead of all full activation tensors.

Static schemes need calibrated scales; dynamic schemes still need this corpus for outlier, saturation, and sensitivity analysis. Start with simple supported scaling, then compare clipping or activation rescaling only where measurements justify it. If static ranges vary materially across timesteps, compare per-step scales/dispatch against their memory and runtime costs. Do not assume one first-chunk capture is representative.

Measure normalized tensor error and clipping/saturation per layer and timestep for debugging, then make precision decisions using complete videos:

1. Test six groups of five FFN blocks, one group quantized at a time; refine offending groups to individual blocks/projections.
2. Test the complete candidate FFN set, since individually safe perturbations can interact.
3. Restore sensitive operations to BF16 and remeasure both appearance and full-pipeline speed.
4. If still useful, expand to self-attention projection linears or the independently validated Sage path, one change at a time. Audio conditioning remains protected until evidence justifies touching it.

Fixed-latent/teacher-history probes help localize error but cannot establish streaming quality. Final comparisons must let each candidate feed back **its own** generated history. Initial/late blocks or denoising steps may be sensitive; determine this experimentally rather than asserting they exclusively generate teeth.

**Deliverable:** an explicit module precision manifest with backend, scale recipe, exclusions, calibration split, and sensitivity evidence.

## Stage 6 — Select by teeth quality and complete-pipeline performance

Use a staged matrix, not every combination at once:

| Run | Change relative to the matched control |
| --- | --- |
| B0 | Original eager BF16 PRO |
| B1 | Quality-approved BF16 implementation improvements |
| Q1 | B1 + FP8 FFNs |
| Q2 | B1 + INT8 FFNs, if needed |
| A1 | B1 + selected SageAttention path |
| M1 | Best measured combination with sensitive layers restored |
| M2 | Optional INT4 or decoder-quantization experiment, only if justified |

Save raw/pre-encode frames and same-settings encoded review clips. Use native-size views, mouth crops, full-frame contact sheets, and long videos. Score preselected teeth-visible moments plus systematic samples to avoid cherry-picking. Report landmark/tracker failures and mouth exposure separately; a candidate hiding its teeth must not score as a preservation success.

Use mouth-region perceptual differences, edge energy, lip landmarks, relative audio/video sync, and motion-compensated temporal residuals as supporting diagnostics. Pixel metrics need alignment and can penalize legitimate motion changes; BF16 is an appearance reference, not ground-truth anatomy. Increased edge energy may be noise, ringing, or invented divisions. Human review of tooth separation and temporal stability is the primary quality gate.

Advance finalists from ten-second clips to 60–120-second sequences and then a representative 10-minute streaming soak, including silence, repeated turns, interruptions, and resume. Report useful unique frames, trimmed/padded frames, repeats, tail latency, startup, memory growth, and drift. Maintain a speed/quality/memory table and choose a nondominated candidate; do not select the smallest weight file by default.

## Stage 7 — Integrate a winning PRO policy into serving

Only after offline evidence supports a candidate, generalize the live engine to explicit model profiles. Derive latent shapes, patch grids, history lengths, emitted-frame counts, audio alignment, warmup, timestep constants, cache keys, and TRT dispatch from the selected profile. Audit Lite-specific assumptions across engine, worker, scheduler, interruption/reconditioning, and media pacing. Existing `24`-frame and five-latent constants cannot silently survive a switch to PRO.

Add an explicit precision/backend configuration with validated manifests and a recoverable BF16 option. Verify checkpoint hashes, GPU/runtime compatibility, scales, actual quantized module coverage, and engine I/O precision on load. Quantization must be applied at a compatible point in the loader; subsequent `.to(dtype=...)`, projection fusion, compilation, or reload must not undo it. Keep packed weights resident and avoid retaining redundant full-precision GPU copies.

Meaningful implementation checks: PRO/LITE shape and chunk contracts, disabled-policy baseline parity, save/reload behavior, incompatible-artifact rejection, first/recurrent history, temporal cache reset, private RNG/session isolation, and receiver-side audio/video timing. If graph replay is used, verify new inputs change outputs correctly and dynamic quantization scales are refreshed. Keep decode caches and workspaces safe across sessions.

Run one-stream WebRTC first. Measure actual delivery with TTS/co-resident load. If concurrency is requested, validate every actual batch size, including partial batches, before a capacity ramp. Export/build/reload and rollback instructions belong with the final artifacts; no production switch is part of this planning task.

## Stop rules and follow-up choices

- If no quantized FFN backend wins full-operation latency, do not expand it just because its storage is smaller. Check kernel support, copies, and compile behavior before changing bit width.
- If decoder time alone consumes the target budget, transformer-only work cannot solve the target. Revisit Wan acceleration or the target hardware/profile using measured results.
- If eight-bit quality is poor, restore sensitive modules and retest. A small speed loss is acceptable if it preserves the reason for choosing PRO.
- If the best approved configuration remains below 25 FPS, report its measured FPS and quality honestly. Present a separately evaluated 15-FPS option or target-GPU test; do not label offline generation or repeated-frame playback real time.
- Consider quantization-aware fine-tuning only after PTQ shows a specific quality/speed gap worth solving. It needs separate data, compute, and validation planning; it is not the simplest first step. Two-step sampling, different resolution/history, new refiners, and distillation also remain separate hypotheses.

## Proposed implementation artifacts

Keep first implementation work under a new `benchmarks/pro_quantization_<run-date>/` directory: fixture manifest, profile/capture runner, bounded calibration summaries, kernel trials, layer-sensitivity table, per-run telemetry, raw-frame checks, comparison videos, and a final decision report. Reuse existing GPU lease, recording, provenance, and analysis utilities where their PRO assumptions are correct.

After a backend passes the kernel gate, add a small policy module such as `soulx_rtc/pro_quantization.py` for selection/exclusions and metadata, with a backend-specific adapter only where needed. Extend existing TRT tooling only if the TensorRT branch wins. Keep large weights/engines out of Git and retain reproducible build instructions.

The first implementation milestone is **a reproducible PRO profile, a BF16 Wan optimization trial, and one real FP8/INT8 FFN trial with matched teeth clips**. Those results determine whether broader quantization can satisfy the user's target before substantial serving integration work.
