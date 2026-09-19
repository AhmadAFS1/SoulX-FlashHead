# PRO quantization V2: independent implementation review

Follow-up: implementation repairs and fresh validation are documented in [the V2 repair report](../../benchmarks/pro_quantization_v2_20260918/README.md). The findings below describe the pre-repair implementation.

Review date: 2026-09-18. Verdict: **substantial partial implementation; not complete or qualified for replacement of the accepted PRO FP8 version.**

Hardware/evidence: this review used **CPU-only tests, static inspection, saved media, and historical GPU results**, with no new GPU inference. The GPU runs cited below recorded **NVIDIA GeForce RTX 4070 SUPER, 12 GB physical / 12,282 MiB visible VRAM**, driver 595.84, PyTorch 2.7.1+cu128, CUDA runtime 12.8 (nvcc 12.1). Their environment manifests identify the GPU and runtime. The reference seed-50 run recorded 2,727 MiB initially occupied; recorded co-resident allocations included 2,478 and 234 MiB. Timing results are not clean-device capacity measurements.

Scope: review against `PRO_QUANTIZATION_V2_IMPLEMENTATION_PLAN_2026-09-18.md`, actual source, tests and saved artifacts. Attached chat narratives and the multi-shape decoder proposal were treated as evidence, not instructions to execute. No runtime code or model weights were changed during this review.

## What has actually worked

The strict policy loader, explicit linear conversion plans, real FP8 self-attention projections, source/fixture manifests, capture infrastructure, decoder inventory, and comparison tooling are meaningful work. The causal-convolution adapter keeps preparation/cache handling in the original module and replaces its prepared convolution operation, a sound boundary to preserve.

Saved kernel results contain 96 self-projection cases covering blocks 0, 14 and 29, four projections, and eight captured invocations. All recorded eager/compiled outputs are finite. Dispatch traces include `aten::_scaled_mm` and SM89 FP8 GEMM kernel names: this is evidence of actual FP8 computation, not merely smaller weight storage. This does not establish behavior for every block or long video.

There are four completed 250-frame runs for the Indian-man 1.50× fixture, one per policy/seed pair. All numbers below are historical RTX 4070 SUPER runs dated 2026-09-18, using the runner's useful-generation FPS definition, not total cold-start application throughput.

| Seed | V2 PRO FP8 reference | Self-attention projections also FP8 | Relative FPS gain |
|---|---:|---:|---:|
| 50 | 9.4336 FPS | 9.8630 FPS | 4.55% |
| 51 | 9.3244 FPS | 9.7506 FPS | 4.57% |

Evidence: `benchmarks/pro_quantization_v2_20260918/runs/{pro-fp8-reference,self-all-fp8}-seed{50,51}-r01/results.json`.

These are useful pilot results, not final timing medians. Both gains are below the plan's provisional 5% combined-candidate threshold; no repeat distribution establishes reliability. No matched MuseTalk run exists. There is no evidence here that the candidate beats MuseTalk.

For seed 50, recorded decode time remains about 13.03 s, while DiT time falls from 11.72 s to 10.53 s. The decoder remains a major measured cost. Further transformer-only quantization cannot be assumed to solve the whole throughput problem.

## Blocking findings and required repairs

### 1. P1 — Decoder engines cannot cover the observed temporal shapes

Locations: `benchmarks/pro_quantization_v2_20260918/build_decoder_engine.py:88`, `:102`, and the build loop around `:274`; `soulx_rtc/pro_vae_quantization.py` engine-plan schema.

The builder loads the first matching captured prepared input and creates one fixed-shape engine per module. Runtime rejects any other shape. Every one of the four selected decoder targets has both `[1,192,3,290,162]` and `[1,192,6,290,162]` in the saved plan. An engine built from the first cannot handle the second.

Independent CPU probe reproduced rejection of the second shape before any CUDA execution. The saved plan also has an empty `engines` map; there is no completed quantized-decoder artifact to qualify.

Required repair: implement either verified dynamic-shape engines or a complete per-shape engine set. For the latter, version the plan schema, key artifacts by module plus exact prepared shape/dtype/layout, build every required shape, validate coverage before installation, and dispatch without changing causal state. Reject an unknown shape explicitly. Test both shapes, public decode/reset sequences, save/load, and end-to-end video. The attached multi-shape design describes future work; its proposed implementation is absent.

### 2. P1 — Engine export discards the fitted activation calibration

Location: `build_decoder_engine.py:129`.

`make_decoder_plan.py` fits and stores a scale from multiple captures, but `_onnx_model` recomputes the scale from the first input tensor. The fitted `target.calibration` is not supplied to export. Thus the plan does not describe the engine's actual activation quantization, and later inputs may clip differently from what the calibration intended.

Required repair: consume an explicit validated calibration record in export; serialize the exact scale/granularity and capture identity into engine metadata. If scales are per shape, fit and identify them per shape. Add a regression where a later calibration tensor has a larger maximum than the first and assert exported Q/DQ constants match the intended fitted scale.

Capture limitation also needs repair: current decoder captures are limited to two calls per module and all saved decoder inputs come from warmup window 0. Capture later recurrent windows and a separate utterance before treating this calibration as representative. Keep fitting and held-out evaluation distinct.

### 3. P1 — Fixed-latent decoder qualification rejects every quantized candidate

Location: `benchmarks/pro_quantization_v2_20260918/decoder_trial.py:117`.

`_configure_candidate` unconditionally raises when the decoder scheme is not BF16. Supplying valid engine artifacts cannot reach an installation path. This prevents the planned INT8 fixed-latent validation even after engines are built.

Independent CPU invocation reproduced this rejection.

Required repair: share a validated engine installer between `run.py` and `decoder_trial.py`. Keep artifact hashes, precision checks and shape coverage fail-closed. Exercise original BF16, BF16 adapter control and actual quantized engines on identical latent tensors, including first/recurrent slices, same-latent repeats, different-latent then original, cache reset and encode-after-decode. Fail the qualification on nonfinite values, shape/state errors, or violated declared tolerances.

### 4. P1 — Installing SageAttention can silently change cross-attention

Locations: `flash_head/src/modules/flash_head_model.py:43` and `:286`; v2 attention installation and runner initialization.

V2 explicitly selects the self-attention backend, but cross-attention still calls the global dispatcher. That dispatcher selects SageAttention whenever importable. Therefore an environment prepared for the proposed self-only Sage trial also changes cross-attention, including a purported FlashAttention-2 reference policy. This violates the intended isolation and can contaminate comparisons.

Independent CPU probe set availability true, replaced `sageattn` with a recorder, and invoked `CrossAttention`; the recorder was called. The saved pilot environment has SageAttention unavailable, so this is a future trial defect, not evidence that existing pilot outputs used Sage.

Required repair: explicitly pin cross-attention to the reference backend independently of import availability and self-attention policy. Test an environment where both backends are available; record actual self/cross dispatch, not just package availability.

### 5. P1 qualification gap — The new reference does not reproduce the accepted FP8 output

Evidence: preserved `benchmarks/pro_quantization_20260918/PRO_FP8/` and the V2 reference runs for seeds 50 and 51.

Raw RGB hashes differ for both seeds. Independently compared matching saved PNG timestamps also differ: mean absolute RGB pixel differences range about 1.20–8.41 levels for seed 50 and 2.04–9.93 for seed 51 on a 0–255 scale. This is not solely an encoded MP4 hash difference.

This does **not** by itself prove a quality regression or identify its cause. GPU/compiler changes can affect trajectories. However, the user-approved visual reference has not been shown equivalent to the refactored reference.

Required resolution: reproduce the original recipe with preserved inputs, seed, source/checkpoint identities, runtime and compile settings; run paired original/refactored controls; isolate differences in RNG setup, execution path and backend selection. Document unavoidable nondeterminism using repeat controls. Review mouth/teeth and synchronization if the trajectory changes. Preserve the original reference unchanged until equivalence or a newly accepted baseline is established.

### 6. P2 — TensorRT precision proof is insufficient

Location: `build_decoder_engine.py:288` and engine build configuration.

The boolean `quantization_dispatch_verified` comes from searching engine-inspector text for `int8`. A substring can refer to names or conversion layers and does not prove the target convolution executes INT8. The builder also does not request detailed profiling verbosity for inspection.

Required repair: retain detailed engine inspection and verify the intended convolution tactic/precision, with GPU profiler evidence where inspection is ambiguous. Distinguish quantize/dequantize operations from the convolution itself. Never approve the engine from its filename or one substring. Measure the full BF16-interface wrapper, including conversion and dispatch overhead.

### 7. P2 — The MuseTalk comparator does not yet enforce a fair warm timing contract

Location: `benchmarks/pro_quantization_v2_20260918/compare_musetalk.py:245`.

MuseTalk audio preparation is timed outside rendering and omitted from its application-delivery total, while PRO's generation includes audio processing. The first `generate_to_rgb` call is labeled warm rendering without an explicit discarded inference warmup. The comparator does not provide repeated paired timing or establish identical GPU conditions from both records.

Required repair: define and separately report cold setup, warmed generation, per-request audio processing and encoded delivery for both systems. Perform explicit warmup/reset, alternate repeated runs on the same GPU under recorded load, verify delivered frame count/resolution/audio duration, and report medians and ranges. Keep crop-versus-full-frame architectural differences visible. The current comparator has no completed saved run, so this defect has not produced a measured victory claim.

## Plan coverage

| Work item | Assessment |
|---|---|
| Preserve accepted PRO FP8 | Preservation artifacts exist; V2 output equivalence unresolved |
| Strict policies and conversion targeting | Implemented and unit-tested |
| Runner, manifests and failure records | Substantial implementation; not full qualification |
| Decoder cost inventory | Initial diagnostic completed |
| Activation capture | Implemented; decoder recurrence/calibration coverage incomplete |
| FP8 self projections | Actual kernel evidence and two short pilot seeds |
| BF16 versus FP8 kernel speed comparison | BF16 reference errors exist, but BF16 timing missing from kernel rows |
| SageAttention2 | Adapter exists; no completed GPU candidate; cross isolation needs repair |
| Causal decoder adapter | CPU mathematical checks exist; complete GPU/cache qualification missing |
| INT8 decoder | Plan/builder/runtime scaffolding; blocking defects and no built engine set |
| FP8 decoder | Export explicitly rejected; unsupported, not completed |
| INT4 | Readiness gate only; no implemented measured candidate |
| Four seeds × three timing repeats | Missing; only seeds 50/51, one run per policy |
| Teeth/mouth review | Seed-50 pilot artifact exists; insufficient for full acceptance |
| Lip-sync qualification | Tool exists; no completed result found |
| 60-second recurrence / other utterance | Missing |
| Five-minute serving / concurrency capacity | Missing; required only if serving/capacity claims are pursued |
| Matched MuseTalk benchmark | Missing; comparator needs repairs |
| Final implementation report and selected recipe | Not present before this review |

Optional Sage, FP8 decoder and INT4 branches need not all succeed. A measured rejection or well-supported hardware/software incompatibility can close an optional branch. A policy file or readiness check alone cannot count as a completed experiment. In particular, the INT4 gate checks for a benchmark callable but does not execute it before marking readiness complete; it is not INT4 performance evidence.

## Independent checks performed

From the repository root:

```bash
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_pro_quantization.py \
  tests/test_pro_quantization_v2.py \
  tests/test_pro_vae_quantization.py \
  tests/test_optimizations.py -q
```

Result: **32 passed, 4 warnings**, CPU-only. These tests do not exercise real TensorRT engines or full Wan decoder recurrence; passing them does not clear the findings above.

```bash
PYTHONPATH=. .venv/bin/python \
  benchmarks/pro_quantization_v2_20260918/verify_artifacts.py \
  --root benchmarks/pro_quantization_v2_20260918/runs \
  --output /tmp/terra-v2-audit-media.json
```

Result: **5 valid, 0 invalid, 4 excluded** under the verifier's checks. This confirms basic saved-media/hash checks for eligible runs, not teeth quality, audio synchronization, real-time performance or production readiness.

Additional CPU checks reproduced the three control-flow defects described above, checked saved kernel outputs for finiteness, and compared preserved/new reference frames. Historical GPU timings were read from result JSON, not regenerated.

## Recommended completion order

1. Preserve the accepted FP8 assets and current partial implementation; resolve the reference mismatch before fitting or ranking new variants.
2. Pin cross-attention dispatch and add a regression test with Sage available.
3. Complete decoder multi-shape coverage, calibration consumption and precision verification together. Expand recurrence calibration captures.
4. Enable real-engine fixed-latent trials and pass cache/reset/shape/numerical tests before producing candidate videos.
5. Benchmark BF16 and quantized kernels with identical inputs, then retain only candidates that improve the full pipeline. Record rejected optional branches honestly.
6. Run the intended Indian-man 1.50× comparisons for seeds 50, 51, 0 and 1, with three paired timing repeats, mouth/teeth review and lip-sync checks. Run the paired 60-second recurrence stress and a different utterance; add serving/capacity tests only for corresponding claims.
7. Repair and run the matched MuseTalk comparison. Publish the selected policy, all artifacts and limits. Do not label the work complete or faster than MuseTalk unless those acceptance gates pass.

The useful work should be retained. The next step is targeted completion and qualification, especially of the measured decoder bottleneck; the current evidence does not justify replacing PRO FP8 or claiming the requested performance goal has been achieved.
