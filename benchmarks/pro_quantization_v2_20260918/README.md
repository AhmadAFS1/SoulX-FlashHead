# PRO quantization V2 results — 2026-09-18

## Decision

The promoted offline candidate is **`self-all-fp8-sage2-fp16`**. It starts from the accepted PRO FP8 recipe, further converts all 120 self-attention Q/K/V/O projections to FP8 E4M3 W8A8, and replaces only self-attention's FlashAttention2 kernel with SageAttention 2.2.0 using INT8 Q/K, FP16 P/V, and mixed FP16/FP32 accumulation. Cross-attention remains BF16 + FlashAttention2. The Wan decoder remains compiled BF16 to protect teeth detail.

This candidate passes the repository's 5% combined-speed gate, four-seed review, relative sync diagnostic, 60-second recurrence stress, and one different-utterance check. It is the best further-quantized PRO candidate from this cycle. It is **not faster than MuseTalk**: its measured single-stream rate is about 10.4 useful FPS, while the old 37–44 FPS MuseTalk records are historical context and were not reproduced as a fresh matched run in this cycle. Production/server selection is unchanged.

Hardware for every fresh GPU result below: **NVIDIA GeForce RTX 4070 SUPER, 12 GB physical / 12,282 MiB visible VRAM**, compute capability 8.9, driver 595.84, PyTorch 2.7.1+cu128, CUDA runtime 12.8, host `nvcc` 12.1.105. Initial whole-device occupancy was approximately 2,727 MiB from pre-existing resident processes. Individual `results.json` files retain GPU UUID, environment, process and whole-device memory, source hashes, package inventory, and timing boundaries.

## Promoted precision policy

| Component | Promoted precision/backend | Reason |
| --- | --- | --- |
| 60 transformer FFN linears | FP8 E4M3 weights + dynamic FP8 activations | Preserved accepted PRO FP8 base |
| 120 self-attention Q/K/V/O linears | FP8 E4M3 weights + dynamic FP8 activations | Further quantization that passed all four seeds |
| Self-attention core | SageAttention 2.2.0, INT8 Q/K + FP16 P/V, `fp16+fp32` accumulation | Exact public kernel is pinned; no fallback |
| Cross-attention projections/core | BF16 + FlashAttention2 | Explicitly protected and independently pinned |
| Wan decoder | Compiled BF16 | Teeth-sensitive path; INT8 variants did not justify their quality/runtime cost |
| Output/postprocessing | Original behavior | No sharpening, restoration, SR, redubbing, or endpoint replacement |

Policy: [`policies/self_all_fp8_sage2_fp16.json`](policies/self_all_fp8_sage2_fp16.json). The run manifest records `sageattn_qk_int8_pv_fp16_cuda`, package 2.2.0, `pv_accum_dtype=fp16+fp32`, and `fallback=false` after actual execution.

## Four-seed performance acceptance

The schedule was written before observation and ran each role in a fresh process with alternating A/B and B/A order. Each seed has three reference and three candidate observations at 250 useful frames, 320×576, 25 FPS playback, four denoising steps, shift 5, two motion latents, and strength 1.0.

| Seed | PRO FP8 median FPS | Candidate median FPS | Median gain |
| ---: | ---: | ---: | ---: |
| 50 | 9.4055 | 10.3941 | 10.51% |
| 51 | 9.4062 | 10.3951 | 10.51% |
| 0 | 9.4102 | 10.3897 | 10.41% |
| 1 | 9.4110 | 10.2922 | 9.36% |
| **All 12 runs per role** | **9.4082** | **10.3917** | **10.45%** |

All 24 media runs completed; artifact verification reports 24 valid, 0 invalid, 0 excluded. The complete values, ranges, stages, memory peaks, hashes, and run paths are in [`sweeps/promoted-sage2-bf16-decoder-four-seed-r01/acceptance-summary.json`](sweeps/promoted-sage2-bf16-decoder-four-seed-r01/acceptance-summary.json). The recorded schedule is in [`schedule.json`](sweeps/promoted-sage2-bf16-decoder-four-seed-r01/schedule.json).

The speedup comes from DiT attention. The promoted decoder is the same BF16 compiled path as the reference. This candidate is still below the 25 FPS real-time playback rate and far below the old MuseTalk context, so no real-time or multi-call capacity claim is made.

## Teeth, mouth motion, and sync evidence

Each exact accepted-sweep pair retains the full-frame comparison, synchronized mouth-crop video, lossless selected-frame sheet, selection rule, decoded media metadata, and per-frame face/mouth diagnostics.

| Seed | Faces detected | Opening correlation | Median mouth-center distance | Median open-mouth edge ratio | Review finding |
| ---: | ---: | ---: | ---: | ---: | --- |
| 50 | 250/250 | 0.967 | 3.23 px | 0.947 | Tooth divisions remain visible; no systematic banding or flicker in selected/full video |
| 51 | 250/250 | 0.872 | 5.52 px | 1.003 | Larger trajectory difference, but exposed teeth remain distinct |
| 0 | 250/250 | 0.932 | 7.87 px | 0.984 | Pose/trajectory differs; no systematic tooth blur category |
| 1 | 250/250 | 0.936 | 5.31 px | 0.948 | Slightly lower edge energy, visually retains tooth separation |

Review directories: [`seed 50`](sweeps/promoted-sage2-bf16-decoder-four-seed-r01/review-seed50), [`seed 51`](sweeps/promoted-sage2-bf16-decoder-four-seed-r01/review-seed51), [`seed 0`](sweeps/promoted-sage2-bf16-decoder-four-seed-r01/review-seed0), and [`seed 1`](sweeps/promoted-sage2-bf16-decoder-four-seed-r01/review-seed1). Edge energy is a diagnostic and is not treated as dental correctness by itself.

Relative SyncNet diagnostics completed with 100% face-box coverage. Across own-output crops, reference/candidate best lags differ by at most one 25 FPS frame; the same is true using shared reference crops. This passes the plan's block threshold, which triggers only for a consistent shift exceeding one frame. These are relative same-audio diagnostics, not official LSE-C/LSE-D. Evidence: [`sync-own.json`](sweeps/promoted-sage2-bf16-decoder-four-seed-r01/sync-own.json) and [`sync-shared.json`](sweeps/promoted-sage2-bf16-decoder-four-seed-r01/sync-shared.json).

## Recurrence and different-utterance checks

The paired 1,500-frame run repeats the known ten-second utterance six times. It is recurrence stress, not six independent samples.

| Variant | Useful FPS | Chunk p50 / p95 / max | First / middle / last chunk |
| --- | ---: | --- | --- |
| PRO FP8 | 9.3884 | 2.9465 / 2.9497 / 2.9498 s | 2.9319 / 2.9446 / 2.9459 s |
| Candidate | 10.3667 | 2.6674 / 2.6753 / 2.6805 s | 2.6751 / 2.6660 / 2.6663 s |

The candidate gained 10.42%, produced all 1,500 frames, and showed no late-window timing drift. The paired review detected all 1,500 faces; its 1,185 jointly open pairs have a 0.997 median edge ratio and 0.934 opening correlation. Early, middle, and late selected mouth samples retain teeth. Evidence and videos: [`recurrence/promoted-sage2-bf16-decoder-seed50-r01`](recurrence/promoted-sage2-bf16-decoder-seed50-r01).

A separately hashed, 23.05-second two-turn utterance with an internal silence gap was registered as `indian150-two-turns-gap` in [`fixtures.json`](fixtures.json). On 576 frames, the candidate reached 10.2457 FPS versus 9.1927 FPS (+11.46%). The review detected 576/576 faces, measured 0.972 opening correlation and a 1.061 open-mouth edge ratio, and showed correct sampled closures without persistent teeth during closure. Evidence and videos: [`generalization/two-turns-gap-seed50-r01`](generalization/two-turns-gap-seed50-r01).

## What happened to decoder INT8

The TensorRT export was repaired so each selected convolution has input, weight, and output Q/DQ. Detailed engine inspection now proves `CaskConvolution` receives and emits INT8, instead of mistaking surrounding reformat nodes for INT8 compute. Twelve non-protected decoder convolutions were calibrated from 96 prepared-input captures and built into 24 exact-shape engines.

The engines were faster in isolation, but Python/TensorRT dispatch and graph boundaries consumed the gain in the complete decoder:

- Four targets: about 5.1–5.3% slower than compiled BF16 in fixed-latent trials.
- Six targets: about 1.6% slower.
- Twelve targets: near break-even; one seed-50 video improved only about 0.9% over self-projection FP8.
- Combining the twelve-target decoder with SageAttention reduced seed-1 open-mouth edge ratio to 0.837. Separate isolation gave 0.960 for Sage + BF16 decoder and 0.945 for decoder INT8 + Flash2, showing an avoidable combined quality interaction.

For that reason decoder INT8 is retained as valid experimental evidence, but excluded from the promoted policy. The late/high-resolution and explicitly teeth-sensitive decoder layers stayed BF16 throughout. A future decoder attempt needs a coarser fused stage or full-decoder engine that removes per-convolution dispatch and graph breaks; simply quantizing more individual convolutions is not supported by these measurements.

Key evidence: [`runs/decoder-int8-twelve-target-build-r01`](runs/decoder-int8-twelve-target-build-r01), [`runs/decoder-int8-twelve-fixed-latents-r01`](runs/decoder-int8-twelve-fixed-latents-r01), and [`runs/review-self-fp8-decoder-int8-twelve-seed1-r02`](runs/review-self-fp8-decoder-int8-twelve-seed1-r02).

## SageAttention build and kernel evidence

The official SageAttention source is pinned at commit `d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5`, package 2.2.0. The host CUDA compiler is 12.1, so the package was built in an isolated target directory with compute-86 forward-compatible PTX for the Ada SM89 GPU. FP8-PV was not selected because the installed build toolchain cannot support that path; the promoted backend is explicitly INT8-QK/FP16-PV.

At the real self-attention shape `[1, 6480, 12, 128]` BF16, a fresh synthetic direct-kernel run measured 3.6545 ms median for FlashAttention2 and 2.4169 ms for SageAttention2, with output relative L2 0.00220 against FlashAttention2 on that synthetic input. This kernel-only result does not substitute for the end-to-end measurements above. Binary hashes, exact build command, constraints, and package provenance are in [`sage2-build-manifest.json`](sage2-build-manifest.json); direct timing is in [`sage2-kernel-benchmark.json`](sage2-kernel-benchmark.json).

## Validation and reproducibility

Focused test command:

```bash
PYTHONPATH=.:/workspace/experiments/ojin-components-deps .venv/bin/python -m pytest \
  tests/test_pro_quantization.py tests/test_pro_quantization_v2.py \
  tests/test_pro_vae_quantization.py tests/test_pro_v2_repairs.py \
  tests/test_optimizations.py -q
```

Result: **44 passed**, with four existing Triton deprecation warnings. Python bytecode compilation and `git diff --check` pass. Tests cover strict policy resolution, no partial conversion, self/cross-attention isolation, exact Sage2 FP16-PV kernel selection, capture scheduling, shape-dispatched decoder plans, ONNX Q/DQ scales, real precision-inspector requirements, cache-safe decoder installation, and fixed-latent contracts.

To reproduce one promoted run, keep the isolated dependencies out of the base environment and use a new output directory:

```bash
PYTHONPATH=.pro-sage2-deps:/workspace/experiments/ojin-components-deps:. \
  .venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  --policy benchmarks/pro_quantization_v2_20260918/policies/self_all_fp8_sage2_fp16.json \
  --fixtures benchmarks/pro_quantization_v2_20260918/fixtures.json \
  --fixture-id indian150-a --seed 50 --frames 250 --repeats 1 \
  --output NEW_UNIQUE_OUTPUT_DIRECTORY --gpu-lock .gpu-owner.lock
```

All final verification records are clean: 24/24 short sweep runs, 2/2 recurrence runs, and 2/2 held-out runs are valid. The accepted PRO FP8 snapshot and original weights remain unchanged. Rollback is selection of the preserved PRO FP8 recipe; the V2 policies are offline opt-in experiments and do not modify production server defaults.

## Remaining path to the user's throughput target

Further module-level quantization alone did not approach MuseTalk throughput. The measured decoder still consumes about 13 seconds per 250-frame clip and now dominates total time; even eliminating all remaining DiT time would not reach the historical MuseTalk range. The next technically justified work is a fused decoder-stage/full-decoder engine, a newer CUDA toolchain experiment for SageAttention's FP8-PV path, or a separate architectural experiment such as fewer denoising steps/distillation. Those change more than weight precision and require their own quality attribution. A fresh matched MuseTalk run, five-minute bounded-output serving test, and concurrency qualification remain open; no deployment or multi-user claim should be based on this offline result.
