# Further PRO quantization: analysis and implementation plan for 30 useful FPS

Current cross-version reference: [September 19 pipeline precision chart and next throughput targets](PRO_PIPELINE_PRECISION_AND_THROUGHPUT_2026-09-19.md), including LITE, the latest candidate's measured stage budget and priorities after the experiments below. The original planning tables here retain their September 18 scope; the consolidated chart also clarifies that the historical stock attention kernel was not recorded explicitly.

Execution tracking, 2026-09-19: [implementation and fresh GPU validation](../../benchmarks/pro_30fps_20260919/README.md). On the RTX 4070 SUPER described below, profiler/calibration repairs, explicit-cache fused TensorRT stages and native-SM89 FP8-PV attention have been implemented and exercised. The 24-process four-seed sweep measures **11.7830 versus 10.3978 useful FPS**, median paired gain **13.45%**. Real INT8 residual convolutions were verified but rejected for slower integrated performance; the surviving decoder uses explicitly labeled FP16 stages. Thirty FPS is not achieved. Offline recurrence/new-speech checks and 56 CPU tests are complete; relative sync-score declines keep the candidate experimental. The live server is restored. Executed and deferred branches are recorded in the linked report. The original plan below remains the acceptance contract, not a statement that every proposed branch was implemented or passed.

Date: **2026-09-18**. Original document status: **analysis and proposed work on September 18; subsequent September 19 execution is tracked above**. This follows the completed [V2 experiments](../../benchmarks/pro_quantization_v2_20260918/README.md), rather than treating the old implementation plan as unexecuted work.

**Hardware for reused PRO evidence: NVIDIA GeForce RTX 4070 SUPER, 12 GB physical / 12,282 MiB visible VRAM, compute capability 8.9.** Recorded environment: driver 595.84, PyTorch 2.7.1+cu128, CUDA runtime 12.8, host `nvcc` 12.1.105; approximately 2,727 MiB initial whole-device occupancy, with existing processes reporting 2,478 and 234 MiB. Decoder experiments used isolated TensorRT 10.9.0.34 and ONNX 1.17.0. These are retained run environments, not a new hardware measurement. Sources: [V2 report](../../benchmarks/pro_quantization_v2_20260918/README.md), [long-run results](../../benchmarks/pro_quantization_v2_20260918/recurrence/promoted-sage2-bf16-decoder-seed50-r01/candidate/results.json), [decoder build](../../benchmarks/pro_quantization_v2_20260918/runs/decoder-int8-twelve-target-build-r01).

Analysis: CPU source inspection, arithmetic on retained JSON, and primary upstream documentation. Inspected revision: `877a07fb980a3db118702ae0a6806008a651e91f`. Future measurements must record their own GPU UUID, physical/visible memory, versions, co-resident activity, timestamps and clocks/power/temperature when available. Current hardware must not be substituted for historical provenance.

## 1. Decision and scope

**Yes: more of PRO can potentially run in INT8, especially remaining BF16 Wan decoder convolutions. There is no evidence yet that blanket INT8 conversion can produce 30 FPS on this GPU.**

The accepted transformer already runs 60 FFN and 120 self-attention projection linears in FP8 W8A8. Self-attention already uses INT8 Q/K. FP8 and INT8 are both eight-bit formats: switching FP8 to INT8 does not halve storage again and may make this implementation slower. The useful next step is extending efficient low-precision computation to expensive remaining operations while removing integration overhead.

Recommended order:

1. Repair profiling and measure the exact accepted compiled policy; establish a fresh comparable MuseTalk baseline.
2. Prove a cache-correct fused decoder stage in BF16, then test calibrated INT8 in the same stage. Extend only stages that improve complete-decoder time and pass teeth checks.
3. Test native-SM89 SageAttention with FP8 P/V, then fuse repeated activation quantization/projections where profiling justifies it.
4. Test remaining cross-attention projections selectively. Consider actual four-bit transformer computation only after a compatible kernel beats current FP8.
5. Combine survivors, rerun four-seed Indian-man 1.50× comparisons, and test new utterances, recurrence and sustained streaming.
6. If measured budgets cannot reach 30 FPS, report the attainable result and separately scope architecture/distillation work. An incremental speedup is not a completed 30-FPS solution.

Keep original PRO geometry, four denoising steps and the selected movement protocol during this investigation. Lower resolution/fewer steps are separate ablations. Sharpening, redubbing, duplicated frames or interpolation do not count as quantization improvements.

## 2. Starting point and remaining precision opportunities

Reference for this cycle: [self_all_fp8_sage2_fp16.json](../../benchmarks/pro_quantization_v2_20260918/policies/self_all_fp8_sage2_fp16.json), approximately **10.3917 useful FPS** across twelve observations. Retain [PRO FP8 V1](../../benchmarks/pro_quantization_20260918/PRO_FP8/README.md), **9.4082 FPS** in the paired sweep, and stock PRO as attribution anchors.

These are load-time recipes derived from original PRO weights. Rebuild new selected buffers from original weights; never quantize an already rounded FP8 buffer again. Preserve both previous recipes and their evidence.

| Portion | Stock PRO | Accepted V2 | Next experiment / priority |
| --- | --- | --- | --- |
| 60 DiT FFN linears | BF16 | FP8 E4M3 W8A8; BF16 interface | Faster fused FP8; conditional actual W4A4/W4A8; existing INT8 loses |
| 120 self-attention Q/K/V/O linears | BF16 | FP8 E4M3 W8A8 | Shared input quantization / packed QKV with equivalent scaling |
| Self-attention core, 30 blocks | FlashAttention2, BF16 inputs | Sage2 INT8 Q/K, FP16 P/V, mixed FP16/FP32 accumulation | INT8 Q/K + FP8 P/V; exact kernel/build validation |
| Cross-attention Q/O projections | BF16 | BF16 | Selective FP8 or efficient INT8 after cost/sensitivity profiling |
| Cross-attention K/V projections | BF16 | BF16; prepared/cached within a generated window | Lower priority than repeatedly executed Q/O |
| Cross-attention core | FlashAttention2 | FlashAttention2 | Retain initially; no global dispatcher replacement |
| Wan decoder convolutions | BF16 | Compiled BF16 | Fused stages with actual INT8 convolution; largest remaining target |
| Decoder norms, residuals, resampling, head | Original mixed operations | Original operations under compilation | Fuse/layout-optimize; quantize only after sensitivity study |
| Motion-feedback VAE encoder | BF16 | BF16 | Compile/layout first, INT8 later if worthwhile; about 5.7% of wall time |
| Reference-image VAE encoder | BF16 | BF16 | Leave initially; primarily preparation cost |
| Wav2Vec audio encoder | FP32 in local default loading path | FP32 | Leave initially; audio stage about 0.3% of wall time |
| Audio projector, patch/time embedding, modulation, norms, residuals, DiT head | Original BF16/FP32 mix | Original mix | Retain initially; sensitive or small operations |
| Color correction, history selection, RGB transfer, encoding | Original behavior | Original behavior | Optimize measured copies/dispatch separately |

FP8 linears retain FP32 scales, BF16 biases/interfaces and the existing accumulation policy. Quantized attention still includes floating-point scaling/softmax/output. Neither policy is an entirely eight-bit pipeline. There is no isolated “teeth layer”: DiT latents, decoding and recurrent feedback all influence teeth. Initial late-layer protection is a testable quality precaution, not proof that other layers are teeth-insensitive.

Sources: [linear kernels](../../soulx_rtc/pro_quantization.py), [policy/conversion](../../soulx_rtc/pro_quantization_v2.py), [attention backends](../../soulx_rtc/pro_attention_backends.py), [pipeline](../../flash_head/src/pipeline/flash_head_pipeline.py), [Wan VAE](../../flash_head/wan/modules/vae.py).

## 3. Runtime budget and feasibility

Use one coherent run rather than summing unrelated stage medians. The [accepted 1,500-frame candidate](../../benchmarks/pro_quantization_v2_20260918/recurrence/promoted-sage2-bf16-decoder-seed50-r01/candidate/results.json) took **144.694432 s**, or **10.366674 FPS**. JSON SHA-256: `fcbd456d0f9516a7b6fd85c6af611f484d85b7e0b3728b1fca4260618a7f1a7c`.

Below, durations are divided by six: **250-frame-equivalent arithmetic, not a fresh short benchmark**. Stages use GPU events; wall minus their sum is an accounting residual, not an independently profiled CPU operation. Future overlapping execution requires critical-path wall measurement rather than adding overlapping events.

| Work | Seconds per 250-frame equivalent | Share of wall time |
| --- | ---: | ---: |
| DiT | 9.242595 | 38.326% |
| Wan decode | 13.099413 | 54.319% |
| Motion encode | 1.383762 | 5.738% |
| Audio | 0.072410 | 0.300% |
| Unattributed wall residual | 0.317559 | 1.317% |
| Total | **24.115739** | **100%** |

For 250 useful frames, 30 FPS permits **8.333333 s**, requiring **2.894× overall speedup**. Holding all work outside DiT/decode fixed:

`T250 = 9.242595 / DiT_speedup + 13.099413 / decoder_speedup + 1.773730`

| Hypothetical DiT speedup | Hypothetical decoder speedup | Useful FPS |
| ---: | ---: | ---: |
| 1× | 2× | 14.23 |
| 2× | 2× | 19.31 |
| 2× | 4× | 25.85 |
| 3× | 3× | 27.11 |
| 3× | 4× | **30.75** |
| 4× | 4× | 33.97 |
| DiT removed entirely | 1× | 16.81 |
| 1× | Decoder removed entirely | 22.69 |

These are idealized scenarios, **not forecasts**. A 3–4× improvement in already optimized stages is ambitious. Equal DiT/decoder speedups need about **3.406×** for 30 FPS and **4.991×** for 40 FPS with the remainder fixed. A 4× faster decoder still requires approximately **2.814×** faster DiT for 30 FPS.

Initial planning allocations per 250 frames: DiT ≤3.1 s, decoder ≤3.3 s, other ≤1.8 s, total ≤8.333 s. Component allocations can change; measured total is the gate.

Ordinary PRO windows emit 28 useful frames. For 30-FPS production throughput, the budget is **0.9333 s/window**; for existing 25-FPS playback it is **1.12 s/window**. Retained candidate chunk p95 is **2.6753 s**. Report throughput and latency separately. Keep native 25-FPS video timestamps: 30 useful FPS means generation headroom, not faster playback.

## 4. MuseTalk and LITE comparison contract

| Retained run | Hardware/software attribution | Speed | Scope |
| --- | --- | ---: | --- |
| Stock PRO, Indian-man 1.50× | RTX 4070 SUPER, 12 GB / 12,282 MiB; Torch 2.7.1+cu128; Sept 17–18 | About 7.1–7.2 FPS | Full-frame PRO |
| PRO FP8 V1, paired V2 sweep | Same retained Sept 18 GPU/environment above | 9.4082 FPS | Previous accepted recipe |
| Current PRO V2 | Same retained Sept 18 GPU/environment above | 10.3917 FPS | Current reference |
| LITE, Indian-man 1.50× | RTX 4070 SUPER, 12 GB / 12,282 MiB; Sept 17 | 56.199 FPS | Different model/latent/decoder; observed softer teeth |
| MuseTalk V1.5 FP16, batch 8 | RTX 4070 SUPER, 12 GB / 12,282 MiB; driver 595.84; Torch 2.5.1+cu121, runtime 12.1; Sept 17 | 30.87, 40.35, 39.92 FPS | Redub rendering + encoding; cached avatar preparation excluded |

Sources: [PRO/LITE](../../benchmarks/pro_lite_150x_20260917/README.md), [V1](../../benchmarks/pro_quantization_20260918/README.md), [V2 acceptance](../../benchmarks/pro_quantization_v2_20260918/sweeps/promoted-sage2-bf16-decoder-four-seed-r01/acceptance-summary.json), [MuseTalk JSON](../../benchmarks/pro_musetalk_redub_20260917/musetalk/results.json). MuseTalk initial device use was 3,156 MiB, including residents of 2,478, 234 and 424 MiB. Its three rows use different audio/base conditions, not identical repeated inputs. Older 37–44-FPS context includes a non-SUPER RTX 4070 and aggregate-concurrency results; do not substitute those here.

PRO generates the whole 320×576 frame; MuseTalk animates a 256×256 face region and composites onto prepared material. This is a workload difference, not a reason to ignore the requested speed target. LITE is not merely an INT8 PRO: latent/decoder design differs, so its speed does not predict quantized PRO speed.

Fresh comparison instructions:

1. Identify the exact MuseTalk backend used for the user's target. Existing `compare_musetalk.py` loads stock PyTorch V1.5; it does not benchmark the fastest installed TensorRT/compiled service. If that service is the target, add a separately named adapter.
2. Match GPU UUID, delivered dimensions, effective audio samples, useful frames and resident-load conditions. Run the models sequentially under the GPU lease with alternating order, not simultaneously.
3. Separate cold load/avatar preparation from warm generation. Include audio processing, model work, compositing/color correction and host RGB delivery for both warm rates. Report generation plus identical encoder/mux settings separately. Record native model/face resolutions too.
4. Use at least three fresh-process paired observations per condition with identical warmup/reset rules and all raw timings retained. Schedule extra repeats if variation prevents a decision.
5. Existing comparison code marks `comparison_qualified=false` because it uses retained PRO results. Add fresh alternating orchestration before changing that qualification.
6. **30 FPS** and **beating MuseTalk** are distinct gates. Initial victory threshold: repeatable ≥5% over the selected fresh MuseTalk profile with quality passing. Thirty FPS may still lose to a 40-FPS MuseTalk result.

## 5. Findings that change the implementation approach

### 5.1 Existing INT8 linears are slower than current FP8

[Real-activation FFN results](../../benchmarks/pro_quantization_20260918/README.md), at `[1,6480,1536]`, measured compiled FP8 around **3.38–3.43 ms**, BF16 **5.57–5.60 ms**, and existing compiled INT8 **6.60–6.86 ms** per full FFN. This is evidence about those implementations/shapes, not a universal INT8 limitation.

`Int8ComputeLinear` materializes INT32 GEMM output, converts to FP32, applies scales/bias and converts to BF16. Quantization/reduction/intermediate traffic are plausible costs to profile. A replacement needs a fused epilogue and must beat **compiled FP8 including all quantization/layout overhead**. Smaller stored weights do not establish a speed gain.

### 5.2 Decoder INT8 was real but integration was insufficient

Twelve targets were `model.decoder.upsamples.{4,5,6,8,9,10}.residual.{2,6}`, using 24 exact-shape engines. Inspection proves INT8 convolution computation. Four targets were about 5.1–5.3% slower than compiled BF16; six about 1.6% slower; one twelve-target video gained only about 0.9% over its self-projection-FP8 control. Combined Sage/INT8-decoder seed 1 had open-mouth edge ratio 0.837, versus 0.960 for Sage/BF16 decoder and 0.945 for INT8 decoder/Flash2.

[`TensorRTPreparedConv`](../../benchmarks/pro_quantization_v2_20260918/build_decoder_engine.py) converts BF16 input to contiguous FP32, allocates FP32 output, waits between caller/engine streams, executes one convolution and casts back to BF16. Adapters also create compilation boundaries. Launch, conversion, layout and cache traffic need attribution; earlier runs do not isolate each cause. Pose changes also affect edge ratios, so a ratio alone is not proof of intrinsic decoder blur.

Do not simply extend this adapter to every convolution. Replace the unit of integration with useful contiguous stages, compare floating-point/INT8 versions of the same stage, and measure the whole decoder.

### 5.3 Profiling has a concrete correctness defect

[`profile.py`](../../benchmarks/pro_quantization_v2_20260918/profile.py) exports legacy `self_cuda_time_total`/`cuda_time_total` and CUDA memory fields with missing values defaulted to zero. The retained smoke CSV has **267 rows and zero nonzero self-CUDA timings**. Installed Torch 2.7.1 aggregate events expose device-time fields; aliases on individual events do not guarantee aggregate aliases.

Read `self_device_time_total`, `device_time_total`, `self_device_memory_usage`, `device_memory_usage` with explicit legacy compatibility handling. Export device type and field provenance. Missing data must be unavailable, not an unexplained zero. Fail GPU timing qualification when CUDA activity exists but all aggregate device durations are zero; retain trace and failure reason. Verify actual CUDA events correlate with exported timings.

`run.py` also disables decoder compilation for decoder trace/capture to preserve module hooks. Keep this as **eager shape inventory** and add distinct **compiled performance profiling** without hooks that force graph breaks. Record compile/graph-break metadata. Do not sum nested inclusive times or treat inventory timing as exclusive compiled-kernel cost.

### 5.4 Decoder output calibration is undersampled

The old input scale pools multiple captures, but `build_decoder_engine.py` computes each output scale using **one selected captured input per shape**. Fit output distributions across all registered calibration samples. This is not a claim that input calibration was absent.

Fused-stage calibration also needs full stage inputs/cache state, outputs and internal Q/DQ boundaries. Existing prepared-convolution inputs do not supply that contract.

### 5.5 Sage toolchain requirements need precise attribution

Current SageAttention 2.2.0, pinned commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, uses compute-86/PTX and explicitly selects `sageattn_qk_int8_pv_fp16_cuda`. [Pinned upstream](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5#installation) requires CUDA ≥12.4 for Ada FP8 and ≥12.8 for SageAttention2++.

Use isolated CUDA 12.8 matching the Torch runtime where practical; preserve the working dependency directory. This is a Sage constraint, not a general inability of CUDA 12.1 to emit SM89: NVIDIA documents native Ada compilation starting with [CUDA 11.8](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html#building-applications-using-cuda-toolkit-11-8).

## 6. Phase A — reference preservation and valid measurement

**Owners:** existing V2 `profile.py`, `run.py`, `common.py`, `compare_musetalk.py`, `sweep.py` and focused tests. Proposed new artifact root: `benchmarks/pro_30fps_20260918/`; old artifacts remain immutable.

1. Snapshot stock/V1/V2 policies, weight/fixture hashes, source hashes, dependency versions and environment. Record git status and preserve unrelated changes.
2. Define useful FPS as requested frames divided by synchronized wall time from audio processing through RGB transfer. Include padded inference work, color correction and motion feedback. Exclude load/compile/warmup and encode/mux, reporting those separately.
3. Repair §5.3. Add CPU regression tests for field lookup/unavailable failure and a small CUDA-operation profiling smoke check. Profiler runs are diagnostics, not performance benchmarks.
4. Reproduce V2 on seeds 50/51, 250 frames; expand to all four standard seeds if different from retained evidence. Warm up initial/recurrent shapes, then reset RNG, reference state, decoder cache and prepared conditioning before timing. In-timing compilation invalidates warm-throughput results.
5. Profile the exact compiled V2. Split DiT into attention, projections, FFN, cross-attention, normalization/quantization where possible. Split decoder into convolution, norm/SiLU, resampling, residual/cache copies, format conversions and dispatch. Mark unresolved time.
6. Record kernel shapes/layouts/dtypes/counts, exclusive GPU time, inclusive stage wall time, Q/DQ/reformat costs, graph breaks and memory. Compute maximum whole-pipeline benefit from each target's share before implementing it.
7. Run the fresh MuseTalk contract in §4; reuse media assets but measure new timing.

**Exit:** baseline reproduces within measured variation, GPU trace fields are valid, and target ranking comes from the compiled policy. Resolve measurement/environment discrepancies before expanding precision coverage.

## 7. Phase B — fused decoder control followed by INT8

### B1. Implement a cache-correct floating-point stage

**Proposed, not yet existing:** `soulx_rtc/pro_vae_stage_backend.py` and `benchmarks/pro_30fps_20260918/{export_decoder_stage,build_decoder_stage,validate_decoder_stage}.py`. Reuse strict provenance/validation from `pro_vae_quantization.py`; use a new versioned stage schema, not a silent reinterpretation of per-convolution plans.

1. Select contiguous regions using Phase A: initially residual blocks around `upsamples.4–6` and `upsamples.8–10`, extending through resampling only when supported and beneficial. Compare one block, a resolution stage and possibly a whole decoder subcall. Record runtime-weighted coverage.
2. Implement tensor-only `stage(x, ordered_cache_inputs, phase) -> (y, ordered_cache_outputs)`. Specify binding order, shape, stride, dtype, cache ownership and index mapping. Export explicit tensors rather than mutable Python cache lists.
3. Match `WanVAE_.decode`: latent scale inversion, pre-decoder `conv2`, nine sequential latent slices for standard PRO, `_conv_idx` reset per slice, causal padding, cache update, concatenation and cache clearing at decode entry/exit. Within-decode caches differ from pipeline cross-window motion history. Do not carry decoder caches across calls where stock clears them.
4. Export initial/recurrent signatures separately when necessary. Prior prepared convolution inputs include temporal extents 3 and 6; enumerate actual stage boundaries instead of assuming the same shapes. Reject unknown signatures before mutating cache state.
5. Preserve temporal/spatial upsampling conventions, residual scaling, norm epsilon, attention and clamp behavior. Keep reductions in suitable floating point. Test all nine slices, repeated calls, reset, independent interleaved sessions and padded final windows. Nine latent slices cannot be treated as independent batches.
6. Build BF16 control first. If BF16 coverage is unavailable, label FP16 as a separate precision change and test overflow/quality. Original compiled BF16 remains the control. Do not label silent floating-point fallback as INT8.
7. Use persistent per-session contexts/workspaces and shape-specific buffers. Prefer caller-stream execution. If a separate stream is needed, enforce dependencies and allocator/tensor lifetimes, including `record_stream` where required. Do not reuse output storage while downstream consumers still need it; use owned outputs or safe slots.
8. Remove BF16→FP32→BF16 boundaries where a supported engine format allows it. Count remaining casts/layout changes. Measure bindings, transfers and complete stage wall cost, not just engine CUDA time.
9. CUDA graphs are a subsequent experiment requiring stable addresses, no replay allocations, correct changing-input behavior and cache updates. Raising a Dynamo recompile limit is not a solution to per-layer graph churn.

**Exit:** cache correctness and floating-point quality pass, and stage/full-decoder timing is known. If the floating-point control loses, identify unsupported operations or costly boundaries before adding INT8.

### B2. Calibrate the actual fused computation

1. Register calibration/development/acceptance splits before fitting. Seeds 50/51/0/1 and the existing two-turn-gap utterance have already been observed: retain them as regression fixtures, not fresh held-out evidence. Add new registered utterances/seeds for independent acceptance; changes after inspection start a new declared cycle.
2. Cover open/closed mouths, wide vowels, sibilants, silence transitions, early/late windows and both initial/recurrent signatures. Start with Indian-man 1.50×; add another existing consented benchmark portrait if claiming avatar generalization.
3. Capture stage inputs, caches, floating-point outputs and internal proposed Q/DQ tensors from exact V2. After changing upstream DiT precision, validate or recapture distributions before combining.
4. Stream full-tensor histograms/statistics plus a bounded representative reservoir. Record byte limits and dropped captures. Require complete signature coverage; a cap causing missing coverage is a failure, not successful calibration. Avoid loading a whole multi-minute activation set into VRAM.
5. Start with symmetric per-output-channel weights and per-tensor activations supported by selected TensorRT tactics. Compare max-abs with a small preregistered percentile/MSE clipping set on development data. Store scales, axis, clipping fraction and source hashes for every boundary.
6. Fit outputs across all calibration samples, not the first input of each shape. Calibrate actual internal fused-region boundaries. Handle residual branch scales correctly; floating-point residual addition is acceptable and must be timed.
7. Log saturation/reconstruction by tensor/window/mouth condition. Smoothing/rounding calibration, if needed, is a separate ablation. Keep cache storage BF16 initially; cache quantization needs its own recurrence experiment.

### B3. Build real INT8 stages and select precision coverage

1. Export explicit Q/DQ around selected activations/weights/outputs. Leave sensitive operations floating point. Keep INT8 decoder networks separate from the PyTorch FP8 transformer.
2. Pin TensorRT, ONNX, CUDA, builder flags, workspace ceiling, profiles, timing cache and UUID. Begin with retained 10.9 for attribution; an upgrade requires a separate control. A successful export/build does not establish fast INT8 Conv3d tactics for every shape.
3. Save detailed engine inspection: convolution input/output/weight types, tactic identifiers, reformat nodes. Confirm actual dispatch if inspection is insufficient. Report operator-count and time-weighted INT8 coverage. INT8 reformat/QDQ nodes alone do not prove quantized convolution.
4. Initially protect `model.decoder.conv1`, `model.decoder.head.2`, `model.decoder.upsamples.11.resample.1`, and `model.decoder.upsamples.{12,13,14}.residual.{2,6}` as previously. Preserve untested scaling/norm/head operations too. Protection is a starting hypothesis, not an assurance of tooth quality.
5. Calculate the protected-cost ceiling: if fraction `f` of decoder time is accelerated by `k`, ideal decoder gain is `1 / ((1-f) + f/k)` before overhead. Even infinitely fast selected kernels require at least 75% reducible time for a 4× decoder. Finite gains need greater coverage.
6. If protected layers dominate, extend expensive stages individually after fixed-latent sensitivity review. Test BF16 restoration by stage/channel group where supported. If late-stage quantization fails teeth quality, report the speed ceiling rather than silently relaxing quality.
7. Compare compiled BF16, fused floating-point control, fused INT8 and any required adapter-only control on identical latents, including caches/copies. Initial advancement threshold: repeatable ≥10% whole-decoder gain and quality pass. Recalculate the 30-FPS gap; a useful gain may still be insufficient.
8. Limit unproductive search to a documented small set of calibration/restoration alternatives. Save rejected engines/results. If speed is sufficient but quality is not, consider separately scoped quantization-aware training in Phase F.

## 8. Phase C — transformer acceleration

### C1. Native-SM89 SageAttention with FP8 P/V

1. Build pinned Sage2 in a new isolated dependency directory using CUDA 12.8, `TORCH_CUDA_ARCH_LIST=8.9`, bounded parallelism and the existing Torch ABI. Record compiler/module paths, version, commit/patches and binary hashes. Preserve the working FP16-PV installation.
2. Compare current compute-86/PTX FP16-PV, native-SM89 FP16-PV and native-SM89 FP8-PV to separate build and precision effects. Keep projections/decoder fixed.
3. Invoke exact `sageattn_qk_int8_pv_fp8_cuda` through `pro_attention_backends.py`; record accumulation mode, layout, scale, mask semantics and no-fallback execution. Preserve cross-attention's independent Flash2 dispatcher.
4. Extend capture to real post-normalization/rotary Q/K and real V at the attention call, covering all 30 blocks/four steps via bounded stratified sampling and early/late windows. Synthetic `[1,6480,12,128]` tensors are useful smoke tests, not adequate quality evidence.
5. Time quantization, layout conversions, kernel and output handling together against current Sage FP16-PV. Check finite outputs, relative error, changing inputs and attention scaling. Restore sensitive blocks via explicit per-block backend selection if necessary.
6. Verify whole-DiT and video benefit before combination with decoder INT8. Attention speedup applies only to measured attention time, not all DiT time.

### C2. Reduce FP8 overhead before replacing it with INT8

1. Profile `amax`, conversion, GEMM epilogue and intermediate writes. Benchmark complete FFNs and Q/K/V/O groups on real activations/shapes with changing inputs after compilation.
2. When Q/K/V use the exact same input, quantize once and reuse it. Preserve independent weight scales/biases. Packed QKV must preserve scaling, output split order, Q/K normalization and rotary behavior. Using one new weight scale changes the method and needs separate validation.
3. Test fused activation-quantization/epilogues, legal norm-quantize boundaries and GELU-to-quantize transitions. Approximate normalization/GELU only as separately labeled numerical changes.
4. An INT8 candidate must fuse accumulation/rescale/bias/output and avoid the current large INT32→FP32 intermediate. Compare total cost with compiled FP8. Stop if slower; report any storage-only benefit separately.
5. Add packing/shared-quantization policy fields only after implementation exists. Reject incompatible packed state, unknown targets/alignment, stale metadata and casts that recast quantized buffers before mutation.

### C3. Remaining cross-attention projections

1. Measure repeated Q/O and cached K/V in prepared generation. Start with expensive Q/O and an explicit small layer list from original weights. Keep core attention, normalization and audio conditioning unchanged.
2. V2 intentionally rejects non-BF16 cross-attention. Implement schema validation, target discovery, manifests, serialization and prepared-cache integration; a JSON edit alone is insufficient.
3. Compare BF16/FP8/efficient INT8 on real conditioned activations. Invalidate/rebuild caches on policy or audio changes. Check bilabials and silence/voice transitions for articulation/sync changes.
4. Expand only for repeatable DiT benefit. Restore BF16 rather than sacrificing lip closure/timing for a small gain.

### C4. Conditional actual four-bit computation

Four-bit weights/activations would reduce precision below current FP8. No faster compatible PRO four-bit implementation is established by this repo.

First prove an SM89 kernel on exact PRO matrix sizes with correct packing, activation conversion, scales, outlier handling and BF16 output. Include all costs and compare full FFN against current FP8. W4A16 can retain 16-bit arithmetic plus unpacking; it is not proof of four-bit compute or speed on the large 6,480-token workload. Do not assume Ada has Blackwell-native FP4 instructions.

Only then implement an explicit backend/serialized format, initially FFN-only, with FP8/BF16 restoration. Test both projections: GELU changes input distributions. Include any low-rank/outlier correction cost. Defer package selection until exact compatibility is verified; SVDQuant/Nunchaku is not an established drop-in SoulX PRO backend here.

## 9. Phase D — attribution and acceptance

| ID | Transformer | Decoder | Purpose |
| --- | --- | --- | --- |
| R | Accepted V2 | Compiled BF16 | Reference |
| D0 | R | Fused BF16, or separately labeled FP16 | Integration/compiler attribution |
| D1 | R | Fused mixed INT8 | Decoder quantization effect |
| A0 | Native-SM89 Sage FP16-PV | Reference | Build attribution |
| A1 | Native-SM89 Sage FP8-PV | Reference | Attention precision effect |
| P1 | Best FP8 fusion | Reference | Quantization/dispatch overhead |
| X1 | Selected cross projections | Reference | Remaining BF16 projection effect |
| W1 | Optional real four-bit FFNs | Reference | Further bit-width reduction |
| C | Passing combined transformer | Passing selected decoder | Combined speed/recurrence |

### Fixed-latent and numerical tests

- Decode identical saved baseline latents/cache inputs. Compare pre-encode floating-point frames and lossless RGB, whole-frame/mouth error, PSNR/SSIM and temporal differences. This isolates decoder precision from changed DiT trajectories.
- Record relative L2, max error, saturation, NaN/Inf and cache discrepancies. Initial decoded-tensor screening flag: relative L2 above 1%; it triggers investigation, not a dental-quality verdict. Set stage tolerances against floating-point controls before observing final acceptance outputs.
- Verify reset, changing-input behavior, all signatures, repeated calls, final padding and interleaved independent sessions. Shape/cache corruption is a hard failure.

### Visual, sync and recurrence tests

1. Exact hashed Indian-man 1.50× image/audio, seeds **50, 51, 0, 1**, shift **5**, motion history **2**, strength **1.0**, four steps, 320×576, native 25-FPS playback. Compare C versus R; retain stock/V1 anchors for the chosen candidate.
2. Generate labeled full-frame and synchronized mouth-crop videos plus unsharpened native lossless samples. Predetermine samples using reference openness and first/middle/last windows, not attractive candidate frames.
3. Review tooth separation, merged bright bands, flicker, gum/lip edges, mouth interior, complete closures, identity and pose. Watch transitions/recurrence boundaries. Record which artifacts were actually reviewed and by whom/what; static sheets cannot establish absence of flicker.
4. Track face coverage, mouth-opening trajectories and edge energy. Initial triage flags: jointly-open median edge ratio <0.90 or >1.15, opening correlation <0.85, or persistent tracking loss. Investigate/restore precision; ratios are pose-dependent, and high edge energy can be noise or false teeth. There is no automatic dental-quality pass.
5. Run relative SyncNet on own-output and shared-reference crops. Flag consistent lag worsening over one 25-FPS frame and inspect crop sensitivity. Do not call these official LSE-C/LSE-D without the standardized protocol.
6. Repeat the 1,500-frame known-audio stress test and 576-frame two-turn/silence regression, then at least two new registered utterances with acceptance seeds not used to fit scales. Repeated known audio tests recurrence, not linguistic diversity.
7. Revalidate the combined candidate; individually passing components do not establish combined quality, as previous decoder/Sage interaction demonstrates.

### Speed and memory gates

- At least three fresh-process paired observations per standard seed, alternating A/B and B/A under matched load/warmup/reset. Publish all runs/exclusions, paired ratios, medians/ranges and uncertainty. Do not imply broad statistical confidence from twelve observations alone.
- Incremental promotion: repeatable ≥5% full-pipeline gain over R and quality pass. The **30-FPS milestone** requires each standard-seed median ≥30 useful FPS and long-run ≥30 FPS, with chunk distribution reported. If only aggregate passes, state that weaker result.
- Sustained 30-FPS production headroom: aim for p95 ordinary 28-frame windows ≤0.9333 s. Also track missed 1.12-s native-playback deadlines and queue depth. Startup buffering cannot repair inadequate steady-state throughput.
- Report load/compile/warmup, first-output latency, host memory, TensorRT workspace, Torch allocated/reserved and whole-device memory. Retained long-run device peak was 10,608 MiB with co-residents; new engines must fit actual available memory.
- After offline acceptance, run five minutes through bounded output/encode/transport. Check queue growth, cache/workspace leaks, late latency drift and recurrent quality. Concurrency is separately qualified; single-stream 30 FPS is not multi-user capacity.

## 10. Phase E — implementation map and execution instructions

| File/work area | Required change | Completion evidence |
| --- | --- | --- |
| V2 `profile.py`, `run.py` | Device-field fix; separate compiled trace from inventory/capture | Nonzero CUDA smoke trace; graph/compile metadata |
| V2 capture + new stage calibration | Complete inputs/caches/outputs and bounded distributions | Hashed split/coverage/scale manifests |
| `pro_vae_quantization.py` + proposed stage backend | Versioned binding/cache contract and persistent buffers | Reset/signature/interleaving tests, full decode parity |
| Existing per-conv builder + new stage tools | Multi-sample output calibration; floating-point/QDQ controls | ONNX, engine/tactic inspection, stage/full-decoder timing |
| `pro_attention_backends.py` | Native-SM89 FP8-PV selection/provenance | Real-activation and video comparisons |
| Linear/policy modules | Cost-justified FP8 fusion and cross-attention targets | Full-operation gains; strict serialization/target checks |
| V2 trial/review/sweep/collect | Stage installation, paired accounting, quality artifacts | Fixed-latent and four-seed combined acceptance |
| MuseTalk comparator/orchestration | Fresh alternating runs and intended backend | Qualified generation/delivery tables |
| Focused `tests/` | Profiler regression, cache ownership, calibration coverage, no fallback, packing | CPU/GPU verification explicitly labeled |

### Commands using existing interfaces

Run from `/workspace/SoulX-FlashHead`. These are **future execution instructions, not commands run for this analysis**. Use new directories for every trial. The candidate policy must be implemented/validated first; proposed stage-export CLIs do not yet exist.

```bash
RUN_ROOT=benchmarks/pro_30fps_20260918
BASE_POLICY=benchmarks/pro_quantization_v2_20260918/policies/self_all_fp8_sage2_fp16.json
FIXTURES=benchmarks/pro_quantization_v2_20260918/fixtures.json

# Reproduce accepted V2 with isolated dependencies.
PYTHONPATH=.pro-sage2-deps:/workspace/experiments/ojin-components-deps:. \
  .venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  --policy "$BASE_POLICY" --fixtures "$FIXTURES" \
  --fixture-id indian150-a --seed 50 --frames 250 --repeats 1 \
  --output "$RUN_ROOT/reference-seed50-r01" --gpu-lock .gpu-owner.lock

# Repair aggregate fields first; verify compile metadata in this trace.
PYTHONPATH=.pro-sage2-deps:/workspace/experiments/ojin-components-deps:. \
  .venv/bin/python benchmarks/pro_quantization_v2_20260918/profile.py \
  --policy "$BASE_POLICY" --fixtures "$FIXTURES" \
  --fixture-id indian150-a --seed 50 --frames 56 \
  --component pipeline --mode trace \
  --output "$RUN_ROOT/profile-compiled-r01" --gpu-lock .gpu-owner.lock

# Run only after implementation and focused stage checks.
CANDIDATE_POLICY="$RUN_ROOT/policies/candidate.json"
PYTHONPATH=.pro-sage2-deps:/workspace/experiments/ojin-components-deps:. \
  .venv/bin/python benchmarks/pro_quantization_v2_20260918/sweep.py \
  --policy "$CANDIDATE_POLICY" --reference-policy "$BASE_POLICY" \
  --fixtures "$FIXTURES" --fixture-id indian150-a \
  --seeds 50 51 0 1 --frames 250 --repeats 3 \
  --output "$RUN_ROOT/acceptance-r01" --gpu-lock .gpu-owner.lock

# Add new focused regression files when implemented.
PYTHONPATH=.:/workspace/experiments/ojin-components-deps \
  .venv/bin/python -m pytest \
  tests/test_pro_quantization.py tests/test_pro_quantization_v2.py \
  tests/test_pro_vae_quantization.py tests/test_pro_v2_repairs.py \
  tests/test_optimizations.py -q
```

Native-SM89 candidate runs must select their new isolated dependency directory and record imported binaries. The current sweep shares one launch environment between roles; extend it for per-role environments before attributing an old/new build comparison, or the reference may silently use the new library. New acceptance fixtures need a new preregistered fixture manifest.

Existing `decoder_trial.py`: `--reference-policy`, `--candidate-policy`, `--captures`, `--output`, `--repeats`, `--max-latents`, `--gpu-lock`. Extend candidate installation for the stage schema first. Existing `review.py`: `--baseline`, `--candidate`, `--baseline-label`, `--candidate-label`, `--output`; use exact schedule-produced run directories. MuseTalk remains provisional until §4 is implemented.

Required future stage CLI contract: explicit checkpoint/source hashes, stage/cache schema, calibration/development manifest, precision, profiles, workspace limit, output and GPU lease. Hash any reused timing cache. Build/validation are separate commands. Failure saves status/reason/partial provenance, never a success manifest with missing engines.

### Required artifacts

```text
benchmarks/pro_30fps_20260918/
  README.md                  # Hardware, timing boundaries, decision, failures
  baseline-manifest.json      # Stock/V1/V2 hashes and environment
  fixtures.json               # Calibration/development/acceptance registration
  schedule.json               # Prewritten paired order and repeats
  profiles/                   # Compiled traces and separate eager inventory
  calibration/                # Input/cache/output distributions and coverage
  stages/                     # Schema, ONNX, engines, inspector, build logs
  kernels/                    # Real-activation full-operation timings
  policies/                   # Immutable tested configurations
  runs/                       # Results, source snapshots and telemetry
  reviews/                    # Full/mouth videos, native samples, review record
  recurrence/                 # Minute/five-minute bounded-output stress
  generalization/             # New utterances/seeds, separate from regressions
  musetalk/                   # Fresh matched generation/delivery comparison
  acceptance-summary.json     # All observations, exclusions and quality gates
```

Stream metrics/media for long tests and retain bounded native samples, not unbounded activations/full-video arrays. Any later runtime selection must point to an exact accepted manifest and retain rollback to V2/V1. This analysis does not change production selection.

## 11. Phase F — stop conditions and alternatives

Recalculate §3 after each meaningful stage result using actual gains including floating-point islands and boundaries. Stop variants that lose to full-operation controls or repeatedly fail quality after a small documented restoration sweep.

If mixed INT8 decoder or transformer gains fall short, quantization alone has not demonstrated the target. Separately scoped alternatives:

- **Decoder quantization-aware fine-tuning:** consider only when INT8 speed is sufficient but calibration/restoration cannot preserve quality. Use original PRO decoded outputs from identical saved latents as aligned teachers; split by utterance/seed/avatar; combine mouth/temporal and full-frame reconstruction objectives. Establish data/training budget first. Validate integrated recurrence, not just teacher-latent reconstruction. Outcome unmeasured.
- **Fewer steps/distillation:** three/two-step inference is a separately labeled schedule/quality ablation. A few-step model may not tolerate arbitrary step removal. Even ideal DiT halving plus 4× decoder only predicts 25.85 FPS with current other work, so fewer steps alone do not establish 30 FPS.
- **Smaller/distilled PRO-compatible decoder or latent redesign:** a training/model project with tooth-detail and feedback validation, not a precision flag or a guaranteed quality-preserving LITE substitution.
- **More capable hardware:** a separately measured hardware solution, not success on this same RTX 4070 SUPER.

Deliver the best measured quality-preserving candidate and its remaining gap even if 30 FPS is not achieved. Do not replace the objective with multi-GPU aggregate FPS, LITE output, lower resolution or faster playback timestamps.

## 12. Ordered completion checklist

- [ ] A: Preserve V2/V1; fix profiler; reproduce baseline; define fresh MuseTalk target.
- [ ] B1: Implement explicit stage/cache contract and floating-point control.
- [ ] B2: Register splits and calibrate all inputs/outputs/internal boundaries/signatures.
- [ ] B3: Prove real INT8 stage execution, whole-decoder gain and tooth preservation.
- [ ] C1: Compare old/native FP16-PV and native FP8-PV on real activations/videos.
- [ ] C2/C3: Implement cost-justified FP8 fusion and cross-attention changes.
- [ ] C4, conditional: Prove faster actual four-bit full FFNs before broad conversion.
- [ ] D: Combined four-seed, sync, recurrence and new-utterance acceptance.
- [ ] E: Publish matched generation/delivery, failures and sustained streaming result.
- [ ] F, if needed: State measured ceiling and separately scope training/architecture work.

## 13. Primary external references and limits

- [NVIDIA Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html): Ada FP8 and native SM89 compilation guidance; hardware support is not local performance evidence.
- [Ada compatibility guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html): general toolkit support differs from Sage's stricter requirement.
- [Pinned SageAttention](https://github.com/thu-ml/SageAttention/tree/d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5): exact APIs/build requirements; upstream examples are not SoulX benchmarks.
- [TensorRT 10.9 release notes](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/release-notes-10/10.9.0.html): version-specific convolution/FP8 limits. Do not apply newer runtime support to installed 10.9.
- [TensorRT 10.x quantized types](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html): explicit Q/DQ/scaling and weight-only versus activation quantization. The archive spans 10.x; validate actual APIs/tactics against the pinned runtime. Avoid assuming FP8/INT8 mixing inside one engine; separate PyTorch FP8 DiT and INT8 decoder engines are different networks.

The first implementation milestone is **measurement repair plus a fused floating-point decoder control**, followed by calibrated INT8 stages. This establishes whether broader INT8 helps this PRO model while retaining the teeth detail the user values.
