# PRO quantization v2: detailed implementation and validation runbook

> **Execution update (September 18, 2026):** The runbook has now been implemented and exercised. See the [final V2 evidence report](../../benchmarks/pro_quantization_v2_20260918/README.md). The text below remains the pre-execution plan, so its proposed interfaces and unchecked checklist describe the original plan rather than current completion status.

**Status: implementation plan, not implemented results.** September 18, 2026. This document was prepared by CPU source inspection and review of existing experiment artifacts. No new GPU inference, dependency installation, or quantization experiment accompanied its creation.

**Reference hardware:** NVIDIA GeForce RTX 4070 SUPER, physical 12 GB class / **12,282 MiB visible VRAM**, verified in saved `nvidia-smi` snapshots. Reference driver 595.84; Torch 2.7.1+cu128; CUDA runtime 12.8; host CUDA compiler 12.1; SM89. The completed PRO FP8 experiments retained two unrelated GPU processes reporting 2,478 and 234 MiB. Recollect actual hardware/runtime/load for every future run; these figures are historical provenance, not a promise of the next environment.

## 1. Objective, deliverables, and scope

Produce a faster PRO variant while preserving the tooth detail the user accepted in **PRO FP8**. The ultimate performance objective is to exceed a freshly measured, fairly scoped MuseTalk baseline on the same GPU. The provisional planning target is **40–45 useful generated FPS**, not an established achievable outcome. Multiple simultaneous callers are a separate capacity requirement, not a consequence of reducing checkpoint size.

Deliver the following:

1. An intact, reproducible PRO FP8 reference.
2. An explicit precision policy covering every changed module and actual backend.
3. Operator profiles identifying remaining transformer and decoder costs.
4. Independently measured self-attention projection, attention-kernel, decoder, and optional four-bit FFN candidates.
5. A winning combined policy only if its measured speed and quality justify it.
6. Matched stock PRO / PRO FP8 / candidate videos using the Indian-man 1.50× fixture and seeds 50, 51, 0, 1.
7. A 60-second recurrent comparison and a longer stability test for any finalist intended for serving.
8. A same-GPU MuseTalk comparison with clear cold-start, warm-generation, and delivery boundaries.
9. A manifest, reproduction commands, numerical results, known failures, quality findings, and rollback instructions.

The first implementation cycle is an offline experiment. Keep 320×576 output, four denoising steps, shift 5, strength 1, and two motion latents fixed. Fewer steps, lower resolution, a different VAE, distillation, spatial sparsity, and mouth restoration are different experiments and must not be silently mixed into quantization attribution.

### 1.1 Read these before implementation

- [Accepted PRO FP8 snapshot](../../benchmarks/pro_quantization_20260918/PRO_FP8/README.md).
- [PRO FP8 preservation manifest](../../benchmarks/pro_quantization_20260918/PRO_FP8/manifest.json).
- [Completed v1 report](../../benchmarks/pro_quantization_20260918/README.md).
- [Further-quantization analysis](PRO_FURTHER_QUANTIZATION_2026-09-18.md).
- [Repository run-documentation requirements](../../AGENTS.md).

**Command convention:** sections marked **existing command** use interfaces already present. Sections marked **proposed interface** specify tools/flags that must be implemented before those commands can run. Paths listed as new files are proposed deliverables, not existing capabilities.

## 2. Fixed baseline contract and speed budget

### 2.1 Accepted PRO FP8 definition

| Property | Required value |
| --- | --- |
| Reference | `benchmarks/pro_lite_150x_20260917/reference-150x.png` |
| Audio | `benchmarks/pro_lite_150x_20260917/audio.wav` |
| Model | `models/SoulX-FlashHead-1_3B/Model_Pro` |
| Decoder | `models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth` |
| Audio encoder | `models/wav2vec2-base-960h` |
| Output | 320×576, 25 playback FPS |
| Denoising | Four steps, shift 5 |
| Recurrence | Two motion latent frames; five decoded history frames; 28 new output frames per window |
| Seeds | 50, 51, 0, 1 |
| Short duration | Exactly 250 useful frames / 10 seconds |
| FFNs | All 60 linear projections: FP8 E4M3 weights and dynamically scaled FP8 activations |
| Attention | BF16 projections and FlashAttention2 |
| Decoder | Compiled BF16 Wan decode |
| Other optimizations | Compiled DiT; prepared real rotary; timestep preparation; per-chunk audio/context K/V preparation |
| Quality transformations | No sharpening, SR, face substitution, endpoint replacement, or redubbing |

The saved recipe reconstructs FP8 buffers from original BF16 weights at load time. It is not an exported standalone FP8 checkpoint. Do not convert existing rounded FP8 buffers to INT4; always quantize the original weights for each experiment.

The PRO transformer configuration is 30 blocks, hidden dimension 1,536, FFN dimension 8,960, 12 heads of dimension 128. At this profile the visual sequence is 6,480 tokens. The latent window is `[16, 9, 72, 40]` before adding the VAE batch dimension. Validate these facts against the loaded checkpoint instead of relying only on hardcoded dimensions.

### 2.2 Why decoder work is mandatory

One saved PRO FP8 repeat generated 250 frames in **26.564 s**:

| Stage | Seconds | Approximate share |
| --- | ---: | ---: |
| DiT | 11.757 | 44% |
| Wan decode | 13.086 | 49% |
| Remaining work, including motion encode/audio/overhead | 1.722 | 6% |

Do not confuse parameter percentage with runtime percentage. FFN weights account for approximately 54.77% of the transformer checkpoint, but FFNs are only part of DiT runtime.

Use `FPS = 250 / (11.75657 / S_dit + 13.08618 / S_vae + 1.72159)` for initial scenario calculations only. Both major stages becoming 2× faster predicts 17.68 FPS; both 3× faster predicts 24.99 FPS; both 5× faster predicts 37.37 FPS. Eliminating all DiT cost with remaining work unchanged predicts 16.88 FPS. These are arithmetic scenarios, not expected kernel gains.

Recompute this budget after every surviving optimization. If measured decoder time already exceeds the target's full time budget, further transformer quantization cannot satisfy the target by itself.

## 3. Existing code and proposed file layout

### 3.1 Existing implementation points

| Existing file | Relevant behavior |
| --- | --- |
| `soulx_rtc/pro_quantization.py` | `Float8Linear`, `Int8ComputeLinear`, FFN-only replacement, BF16 decoder compilation |
| `flash_head/src/modules/flash_head_model.py` | Self/cross-attention projections, optional packed QKV, rotary, cached conditioning, transformer blocks |
| `flash_head/wan/modules/vae.py` | Causal Conv3d, decoder residual/resampling blocks, internal temporal caches, encode/decode wrappers |
| `flash_head/src/pipeline/flash_head_pipeline.py` | Denoising schedule, model calls, decoder, color correction, motion feedback encode |
| `benchmarks/pro_quantization_20260918/run.py` | Offline PRO runner and saved profile |
| `benchmarks/pro_quantization_20260918/kernels.py` | Real-input FFN microbenchmarks |
| `benchmarks/pro_quantization_20260918/vae_trial.py` | Isolated decoder timing/error on identical latents |
| `benchmarks/pro_quantization_20260918/review.py` | Paired media, face/mouth diagnostics, sample sheets |
| `benchmarks/pro_quantization_20260918/sync_check.py` | Relative same-audio SyncNet with optional own-face boxes |
| `soulx_rtc/gpu_lease.py` | Nonblocking checkout-level GPU ownership lock |
| `tests/test_pro_quantization.py` | CPU policy contracts |

### 3.2 New files to implement

Keep the accepted v1 sources and artifacts intact. Prefer new modules and a new experiment directory:

```text
soulx_rtc/
  pro_quantization_v2.py          # strict policy parsing, validation, staged replacement
  pro_attention_backends.py      # explicit per-module attention backend selection
  pro_vae_quantization.py        # cache-safe decoder adapters and engine contracts
benchmarks/pro_quantization_v2_20260918/
  README.md                      # living status, evidence, decisions
  policies/
    pro_fp8_reference.json
    self_qkv_fp8.json
    self_all_fp8.json
    attention_sage2.json
    decoder_int8_conservative.json
    decoder_fp8_conservative.json
    ffn_w4a8.json
  preflight.py
  run.py
  profile.py
  capture.py
  kernels.py
  decoder_trial.py
  build_decoder_engine.py
  sweep.py
  review.py
  sync_check.py
  compare_musetalk.py
  collect.py
  verify_artifacts.py
  fixtures.json
  calibration/                  # manifests, fitted scales, bounded captures
  runs/                         # unique, never overwritten
tests/
  test_pro_quantization_v2.py
  test_pro_vae_quantization.py
```

Not every optional backend requires its own new implementation if an existing tested adapter can be reused. Keep module responsibilities as above. Do not create unused serving abstractions or train a model during the first cycle.

## 4. Phase A — protect the reference and establish reproducibility

### A1. Inspect the working tree before editing

**Existing commands**, from `/workspace/SoulX-FlashHead`:

```bash
git status --short
git diff --stat
cat AGENTS.md
```

The working tree contains unrelated user work and untracked research files. Do not reset/clean it or assume a new git worktree contains the uncommitted PRO FP8 implementation. If using an isolated checkout, deliberately copy required source snapshots and reference weights/assets, preserving provenance. Avoid copying large checkpoints unnecessarily.

### A2. Verify the saved snapshot and source weights

Use this **existing-data verification command** before implementation. It reads files and hashes; it does not load models onto the GPU:

```bash
.venv/bin/python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path.cwd()
saved = root / 'benchmarks/pro_quantization_20260918/PRO_FP8'
m = json.loads((saved / 'manifest.json').read_text())
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
for original, expected in m['tested_source_sha256'].items():
    relative = Path(original).relative_to(root)
    assert digest(saved / 'sources' / relative) == expected, relative
for relative, expected in m['checkpoint_sha256'].items():
    assert digest(root / m['checkpoint_root'] / relative) == expected, relative
for field in ('reference', 'audio'):
    assert digest(Path(m[field]['path'])) == m[field]['sha256'], field
print('PRO FP8 snapshot, weights, and fixture hashes verified')
PY
```

This command assumes the recorded repository root. If the repository moved, remap the recorded root explicitly; never silently reinterpret arbitrary absolute paths.

Record current source hashes separately from saved hashes. A source change is not necessarily corruption, but it means the current runner is no longer automatically the saved baseline. Restore the saved source set only inside an isolated experiment checkout if necessary.

### A3. Implement `preflight.py`

Output a JSON record containing UTC time, hostname, git commit and dirty status, source hashes, GPU UUID/name/compute capability, visible VRAM, documented physical VRAM class, driver, Torch/CUDA/Triton/FlashAttention versions, CUDA compiler version, available disk/RAM, package inventory, environment overrides and resident GPU processes. Exclude credentials from environment capture.

Validate model geometry, fixture hashes, actual attention backend and required files. Do not label an SDPA fallback as FlashAttention2. Report CUDA runtime and CUDA compiler separately. Record whether compiler caches existed before the run.

Acquire one GPU lease for each inference/build/benchmark process. With multiple checkouts, use the same absolute lock file: `/workspace/SoulX-FlashHead/.gpu-owner.lock`. The existing default is checkout-local and is not sufficient to coordinate separate checkouts. Keep the lock handle alive through model destruction/process exit. Do not kill unrelated GPU processes to improve benchmark numbers.

### A4. Reproduce PRO FP8 once before adding precision changes

Create a unique run root; do not reuse this example if it already exists:

```bash
mkdir -p benchmarks/pro_quantization_v2_20260918/runs
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_20260918/run.py \
  --output benchmarks/pro_quantization_v2_20260918/runs/reference-pro-fp8-seed50-r01 \
  --seed 50 --precision fp8 --optimized-dit --compile-dit --vae compiled \
  --repeats 3
```

This is an **existing command**, valid only after checking that the current source set reproduces the saved reference. Validate 250 frames, matching profile, finite output, source/weight identity and approximate historical speed. Compare raw RGB hashes; exact equality is expected only under the same deterministic execution conditions. Investigate differences rather than silently tolerating them.

A speed drift above 5% from the historical range triggers a provenance/load/cache investigation, not automatic rejection. Preserve both historical and new timings. Do not normalize candidate measurements by an unexplained baseline drift.

**Phase A exit:** reference integrity check, current environment manifest, fresh reference video/timings, and an explicit explanation of any difference.

## 5. Phase B — implement strict policies and reliable experiment plumbing

### B1. Policy schema

Implement a versioned JSON policy. The following is a **proposed schema**, not a currently accepted runner input:

```json
{
  "schema_version": 1,
  "name": "self-all-fp8",
  "base": "pro-fp8-v1",
  "ffn": {"scheme": "fp8_e4m3_w8a8", "exclude": []},
  "self_attention_projections": {
    "scheme": "fp8_e4m3_w8a8",
    "include": ["blocks.*.self_attn.q", "blocks.*.self_attn.k", "blocks.*.self_attn.v", "blocks.*.self_attn.o"],
    "exclude": [],
    "packing": "separate"
  },
  "self_attention_kernel": {"backend": "flash2"},
  "cross_attention_projections": {"scheme": "bf16", "include": [], "exclude": []},
  "decoder": {"scheme": "bf16", "backend": "torch_compile", "plan": null},
  "compile": {"dit": true, "ffn_only": false},
  "prepared_conditioning": true
}
```

Supported schemes should be registered explicitly. Add INT4 or decoder backend values only when the associated implementation exists. Unknown keys, unsupported combinations, duplicate/conflicting rules, unmatched include patterns, and unknown exclusions must fail before mutation. An empty include list means no extra targets; it must not accidentally mean all modules.

Resolve patterns to sorted full module paths and save `resolved-policy.json`. Exclusions override includes, but every exclusion must refer to an otherwise valid target. Record exact source/target dtypes, scale granularity, quantized storage bytes, bias dtype, accumulation policy, backend version, group size, packed layout and fallback status for each module. Do not count a dequantized BF16 GEMM as FP8/INT4 compute.

### B2. Two-pass model conversion

The v1 replacement loop can modify earlier layers before encountering a bad later layer. V2 must validate the complete replacement set first:

1. Strict-load original checkpoint.
2. Put original modules on the intended device/dtype; freeze parameters and set evaluation mode.
3. Validate geometry, every target class/dtype/shape, backend support, exclusions, and packing state.
4. Build the complete conversion plan without changing the model.
5. Convert sequentially to avoid holding duplicate full models in VRAM.
6. If allocation/conversion fails, mark the run failed and discard the partial model. Retry from original weights in a new process; do not continue with mixed accidental state.
7. Install explicit attention/decoder adapters.
8. Prepare immutable conditioning/rotary structures.
9. Compile only after all replacements and layouts are final.
10. Never call a blanket `.to(dtype=...)` afterward: it could cast registered FP8 buffers back to BF16.

Keep FP8 scales FP32, BF16 external interfaces, and original bias handling. Either reject conversion twice or implement a separately tested reload path; do not silently double-quantize.

### B3. New runner CLI

Build the v2 runner from the existing offline runner. **Proposed interface**:

```text
run.py --policy FILE --fixture-id indian150-a --fixtures FILE
       --output NEW_DIRECTORY --seed N --frames N --repeats N
       [--save-raw] [--capture-manifest FILE] [--gpu-lock ABSOLUTE_PATH]
```

Required behavior:

- Preserve native profile and useful-frame accounting; do not drop expensive padding from elapsed time.
- Save requested and resolved policies, fixture identity, all relevant source hashes, checkpoint hashes and environment details.
- Snapshot sources with their repository-relative paths, not basenames that can collide.
- Save a starting manifest before model load and update it atomically through preflight/loading/warming/running/complete/failed states.
- Include complete error type/message/traceback, failing stage, GPU snapshot and partial timing records on failure.
- Create run directories with `exist_ok=False`; use new IDs for retries.
- Reset per-session RNG and reference/motion state between repeats, but not between recurrent chunks of one clip.
- Keep exactly the original audio history behavior; final slice padding stays in generation timing.
- Separate load, prepare, compile, warmup, generation, RGB transfer, encode/mux and evaluation times.
- Include RGB transfer in the primary generation metric, consistent with v1. Also report device-only stages; do not substitute them for the primary metric.
- Record first-window latency and the full chunk distribution, including p50/p95/p99/max and number of samples. Nine short-clip chunks are insufficient for a strong tail-latency claim.
- Derive audio repetition from actual input sample count, not a hardcoded 160,000-sample threshold when generalizing to other fixtures.

Preserve the original synchronization behavior for first attribution experiments. If removing per-step `synchronize()`/printing later, create a separate precision-preserving optimization row. Keep reliable event synchronization at measurement boundaries.

### B4. Backend selection and fallback handling

The existing Sage1 hook recognizes self-attention by sequence length. Replace this heuristic with an explicit backend attached only to `SelfAttention` instances in the experimental model. Do not globally change the shared attention function and accidentally affect cross-attention, another model, or another caller.

Validate head dimension 128, Q/K/V shapes/strides, scale convention, noncausal behavior, output dtype, mask expectations, and input mutation. Make backend choice explicit even when optional packages are installed. Fail unsupported paths by default. If a deliberate fallback is allowed for debugging, report it per invocation and disqualify that run from the intended backend's speed claim.

### B5. Focused CPU checks

Add tests for invalid policies causing no partial mutation; disabled policy preserving identity/state; exact target coverage; exclusion precedence; unknown/unmatched paths; geometry/dtype/alignment rejection; packed-QKV rejection or supported conversion; zero/near-zero positive finite scales; buffer/state-dict metadata; and original head/audio/encoder protections.

Do not pretend CPU tests validate CUDA kernel correctness or speed. Keep those in GPU experiment scripts. Run the existing six policy tests and existing conditioning/rotary tests after relevant changes, then only the new focused tests needed by this work.

**Phase B exit:** unchanged numerical behavior for the v2 reference policy, strict resolved manifest, reproducible runner and passing policy contracts.

## 6. Phase C — profile and capture representative inputs

### C1. Separate profiling from performance measurement

Implement `profile.py` with explicit `--component dit|decoder|pipeline` and `--mode trace|timing`. Do not report throughput from a run with tensor capture, per-layer `.item()`, CPU copies, verbose profiler instrumentation or serialization inside the timed loop.

For profiler runs, use CUDA/CPU activities, memory and shapes. Save Chrome trace plus an aggregated CSV. Report self CUDA time and inclusive time separately; never sum nested module totals. Correlate kernels back to source paths/shapes. Compare eager attribution with compiled execution carefully: compilation can fuse away Python module boundaries.

Collect at least initial and recurrent windows after compilation warmup. Profile one seed first; repeat only if another shape/path is used. In pure timing mode disable instrumentation except the minimum stage events and existing resource sampler.

### C2. Required cost breakdown

- FFN projections, GELU and activation scaling/casting.
- Self-attention Q/K/V/O GEMMs, Q/K normalization/rotary, attention matrix operations.
- Audio conditioning preparation and cross-attention Q/O versus cached K/V.
- Decoder Conv3d, Conv2d, normalization, upsampling, concatenation/padding, cache copies, layout conversions and clamp.
- Motion VAE encode, color correction, audio processing and RGB transfer.
- Python synchronization/launch overhead where observable.

Generate a decoder inventory with full module path, class, kernel/stride/padding/groups, input/output channels, observed spatial and temporal shapes, number of calls, cache index behavior and measured time. Resolve the final high-resolution stage from structure and actual shapes, not a guessed `upsamples` index.

### C3. Calibration and held-out split

Create `fixtures.json` with explicit paths/hashes, sample rate, duration, portrait framing, license/provenance where known, and split assignment. Use the known Indian fixture as mandatory acceptance evidence. Choose additional existing utterances with sibilants, vowels, closures and silence; keep their actual audio identities in the manifest.

Suggested fitting set: Indian 1.50× seed 50 on the known utterance, plus an additional utterance if available. Use seed 51 as development validation. Reserve seeds 0/1 and another utterance for final held-out calibration validation. These seeds have been viewed before, so do not describe them as a wholly unseen research benchmark; they are held out from fitting the new quantizer.

Capture all four denoising steps and both initial and recurrent windows. For long-sequence behavior include windows near the beginning, middle and end of a minute. Cover both closed and clearly exposed teeth; do not use only fixed timestamps that happen to show closed mouths.

For attention, capture post-normalization/rotary Q/K/V for kernel tests and pre-projection activations for projection tests. For FFNs, capture inputs to both projections, not only the first. For decoder tests, capture public VAE input latents and representative internal convolution inputs after causal padding/cache concatenation.

Maintain streaming statistics across all targeted layers: RMS, max absolute value, selected quantiles, channel/group outliers, clipping fraction and step/window index. Keep bounded raw tensors for representative modules; do not dump every activation of every frame to disk. Precompute capture storage estimates, record available disk, and stop with a manifest if the capture budget is exceeded.

**Phase C exit:** cost-ranked modules, supported shape inventory, disjoint fitting/evaluation manifests, and reusable tensors with hashes. The first decoder profile determines whether the proposed quantized subset can affect enough runtime to matter.

## 7. Phase D — FP8 self-attention projections

### D1. Implement the first extension without QKV fusion

Work in `pro_quantization_v2.py`, reusing the tested FP8 E4M3 conversion only where its contract fits. Target ordinary BF16 `nn.Linear` modules:

```text
blocks.0..29.self_attn.q
blocks.0..29.self_attn.k
blocks.0..29.self_attn.v
blocks.0..29.self_attn.o
```

These are 120 additional projections if all are selected. The existing 60 FP8 FFN projections remain unchanged. Start with `self_qkv_fp8` (90 additional projections), then add output projections in `self_all_fp8` (120). If error attribution requires it, test output-only separately.

Apply the two-pass validation from Phase B. Use a separate weight scale for each projection and dynamic activation scales. Preserve bias, output dtype, Q/K normalization, rotary and residual arithmetic. Keep FlashAttention2 for this experiment so projection and attention-math effects are separable.

The exact projection GEMM is typically `M=6480, K=1536, N=1536`; obtain actual observed dimensions for every path. The FP8 module must accept the actual rank/strides and reshape outputs correctly. Benchmark conversions, scales, bias and output layout as part of cost.

### D2. Real-input kernel and compiled correctness checks

For blocks 0, 14 and 29, test Q/K/V/O on captured data from all four steps in first and recurrent windows. Add any additional high-outlier block found by the capture analysis.

For each kernel:

1. Run original BF16, eager FP8 and compiled FP8 on the same input.
2. Compare shape, dtype, finite values, max/mean error and relative L2 with a nonzero denominator floor.
3. Run zero, small-magnitude, mixed-sign and representative outlier inputs.
4. Change the input after compile; ensure scales and outputs update and agree with the eager quantized implementation within measured backend rounding differences.
5. Confirm input tensors are not modified in place.
6. Save/reload quantized weights/scales/bias and verify reconstruction and output.
7. Warm each exact shape, then time at least five batches of repeated calls; include conversion overhead and synchronize CUDA events correctly.
8. Capture profiler kernel names confirming FP8 GEMMs actually execute. Inspect compiled dispatch as well as eager dispatch.
9. Measure full self-attention-module cost, not only isolated projection GEMMs.

Use the actual measured BF16/FP8 errors as diagnostics. Do not set a universal tensor-error threshold and call that a teeth guarantee. Reject nonfinite output, incorrect state reload, stale scales, shape corruption or an unintended fallback immediately.

### D3. Short videos and selective restoration

Run seed 50 and seed 51 with 250 frames. Compare to the fresh PRO FP8 reference before spending time on all seeds. If a candidate degrades articulation or teeth:

1. Compare QKV-only and output-only to identify the more sensitive family.
2. Restore groups of blocks in chunks of five, changing only one group per trial.
3. If a group clearly matters, subdivide it to individual blocks/projections.
4. Rank candidates by recovered visual quality and lost speed.
5. Save the resolved exclusion list; do not infer that first/last blocks are automatically the only sensitive ones.
6. Confirm the chosen exclusions on a development clip that was not used to choose the exact problematic frames.

Do not expand a sensitivity sweep when the projection kernels are already slower or the end-to-end gain is negligible. Preserve failed videos and their policies.

### D4. Optional packed-QKV optimization

Only after separate projection quantization works, consider packing. The model contains `packed_qkv`, `use_fused_qkv`, and a branch that accesses `.weight.chunk(...)`. Replacing the packed owner with a generic FP8 module without adapting this branch will fail.

Choose either a dedicated packed FP8 forward or keep three quantized projections. Preserve independent Q/K/V scales unless a measured shared-scale candidate passes quality. Compare the packed BF16 control too, to separate packing from quantization. Record packing layout in the serialized policy. Do not change packed state after compilation.

**Phase D exit:** a resolved projection policy with verified real FP8 dispatch, a useful whole-pipeline gain or a documented rejection, and initial paired mouth videos.

## 8. Phase E — explicit quantized attention backend

### E1. Dependency and toolchain isolation

The existing Sage1 package is installed under `.pro-quant-deps`; it is not the proposed Sage2 build. Do not upgrade the original `.venv` in place. Prepare a sibling experiment environment using the repository Python version and exact compatible package pins. Keep a lock/pip freeze, compiler version, build log, backend source revision and wheel hash.

The upstream documentation reviewed for this plan specifies CUDA compiler ≥12.4 for Ada FP8 and ≥12.8 for SageAttention2++. Torch's CUDA 12.8 runtime does not make the host 12.1 compiler sufficient. Before installation, check the selected pinned release against its own build requirements. Prefer a matching CUDA 12.8 toolchain for the experiment when feasible. Do not replace the system compiler or unrelated services' environment.

Verify Torch, Triton and FlashAttention import and run in the new environment before quantized attention. Rerun the PRO FP8 control in that same environment. A gain caused by a new runtime must not be attributed entirely to SageAttention.

### E2. Adapter contract

Implement the backend only for experimental `SelfAttention` objects. Start with the explicitly supported Ada INT8-QK/FP8-PV path. Record the exact API and accumulation option; the SageAttention2 paper's INT4 title does not establish that a selected public API uses INT4.

The adapter must:

- Accept the post-normalization/rotary BF16 Q/K/V layout used by SoulX.
- Pass the correct layout flag and noncausal attention setting.
- Preserve the intended attention scaling and BF16 output interface.
- Avoid a full FP32 attention-score materialization.
- Leave cross-attention on FlashAttention2.
- Clone only inputs the backend otherwise mutates, and include that copy cost in timings.
- Report unsupported shapes and fail explicitly instead of silently using another backend.
- Compile only in modes supported by the backend; independently test full-DiT compilation and graph breaks.

Test `[1,6480,12,128]` and the exact flattened equivalent used at the call site. Do not extrapolate upstream large-GPU speedups to this GPU or to the full video pipeline.

### E3. Experiment order

1. PRO FP8 reference with FlashAttention2 in the new environment.
2. Same policy with Sage2 only; original BF16 attention projections.
3. If both survive, combine Sage2 with the selected projection FP8 policy.
4. Repeat initial/recurrent input error checks, changed-input checks and two-seed video screening.
5. Add all four seeds only for a surviving candidate.

Keep the previous 9.93-FPS Sage1 trial as historical evidence. It does not replace a current same-environment control. If Sage2 is incompatible with the compiled graph or fails to improve full-pipeline time, preserve the failure and continue decoder work.

**Phase E exit:** pinned backend environment, dispatch evidence, attributable timing and explicit quality decision.

## 9. Phase F — selective Wan decoder quantization

This is the critical speed-feasibility work. Start its profiling/kernel feasibility early, before investing in an extensive INT4 transformer sweep.

### F1. Preserve the decoder's actual state model

Source behavior to retain:

- `CausalConv3d.forward(x, cache_x)` concatenates prior temporal features when needed, adjusts left temporal padding, calls `F.pad`, then performs convolution.
- `ResidualBlock.forward` and decoder traversal use `isinstance(layer, CausalConv3d)` to decide how to pass and advance caches.
- `count_conv3d` also uses that class to size cache lists.
- `Resample` has first-use `None` and `"Rep"` states as well as tensor caches.
- The nonstreaming model `decode(z, scale)` clears caches at entry, runs latent slices sequentially with caches shared within that call, and clears them at exit.
- At the pipeline level, generated RGB is color-corrected, the last motion frames are encoded, and those latents condition the next generated window. This is a separate recurrence from decoder-internal caches.

A generic `nn.Module` replacement can bypass class checks and change cache allocation even if its convolution is numerically correct. Do not replace `CausalConv3d` blindly or reset its state for every latent slice.

### F2. Preferred adapter design for the first prototype

Retain existing `CausalConv3d` instances and their cache traversal identity. Add an optional compute backend that receives the **already padded/concatenated input** at the point where the class would call ordinary convolution. Default `None` executes the exact existing path. Limit the change to a documented optional branch and keep it disabled outside v2.

The adapter's convolution must use the original stride/dilation/groups/kernel/bias and must not pad a second time. Cache creation, concatenation, index increments and first-use behavior stay in the original Python logic. An engine adapter must not register another nested `CausalConv3d`, which would inflate `count_conv3d` and shift cache ownership.

For Conv2d resample layers, preserve their original padding and channel layout explicitly. Do not carry over the already-padded Conv3d assumption to Conv2d. Keep nearest-exact upsampling and temporal reshaping untouched initially.

A state-explicit multi-layer partition can be evaluated later if single-layer dispatch overhead erases gains. Its contract must list every input/output cache tensor, initial/recurrent variant, shapes, dtype and update ordering. Prove its BF16 behavior first. Do not export mutable Python lists or `"Rep"` sentinels as opaque engine state.

Keep original BF16 weights during first correctness trials if needed for fallback/comparison, and count this memory honestly. Removing duplicate weights requires a separately verified serialization/reconstruction path. Do not describe an engine with retained BF16 copies as a memory-saving implementation.

### F3. Mechanical BF16 adapter control before quantization

Implement the backend adapter with ordinary BF16 computation first. On identical captured public VAE latents compare:

1. Original decoder.
2. Decoder with BF16 adapter and identical math.
3. Compiled reference decoder.
4. Engine/partition BF16 control if exporting.

Check output values, shape, dtype, cache count and reset behavior. For a same-operation PyTorch adapter, expect exact output equality under the same backend. For engine reordering, measure numerical differences and compare them with the existing compiled BF16 decoder's variation; never silently call them exact.

Exercise first latent slice, each recurrent slice, short cache lengths, temporal upsample transitions, complete nine-latent decode, repeated decode of the same latent, and decode of a different latent followed by the original again. Confirm no state leaks across public decode calls or separate sessions. Verify `encode` still works after `decode`, because `clear_cache()` initializes encoder state too.

### F4. Select the initial quantized subset by measured cost

Use the Phase C inventory to create a `decoder-plan.json` containing exact module paths and observed shapes. Conservative first policy:

- Candidate: internal residual Conv3d layers in middle/earlier upsample stages with material runtime cost.
- Candidate: spatial Conv2d resample operations if their real kernels are favorable.
- Initially protected: entry/latent scaling operations, normalization, residual additions, temporal cache interfaces, temporal-resample control logic, final high-resolution residual stage, final RGB convolution and clamp.
- Encoder remains BF16, including motion feedback encode.

Protecting late layers is a hypothesis about risk, not proof of teeth localization. Report what percentage of decoder time the selected subset actually covers. If it covers too little to matter, expand one cost-ranked stage at a time or stop the approach; do not promise full decoder speedups from a small protected subset.

### F5. Representative convolution backend feasibility

Build standalone candidates for the dominant captured shapes, including initial and recurrent cases. Evaluate INT8 and FP8 only where the pinned backend has supported kernels. For TensorRT:

1. Pin a version compatible with the experiment CUDA/Python stack; save the exact build/runtime versions.
2. Start with a minimal exact-shape convolution or small residual partition, not the entire Python decoder loop.
3. Build a BF16/FP16 control as supported and an explicit-Q/DQ candidate.
4. Preserve bias, padding, stride, groups and output dtype exactly.
5. Supply fitted scales through explicit quantization; do not rely on undocumented implicit precision selection.
6. Save ONNX/engine hashes, optimization profiles, workspace cap, tactic/build log, layer information and runtime bindings.
7. Inspect actual layer/kernel precision. Report high-precision fallback separately.
8. Measure device-resident execution including Q/DQ, layout transforms, padding if moved inside, binding overhead, and required synchronization/stream handoff.
9. Measure memory during build, warmup and execution separately, including non-Torch engine allocations.
10. Reject an unsupported or slower tactic. Do not rewrite all convolutions into materialized im2col plus GEMM without accounting for its substantial temporary memory/traffic.

PyTorch's `Float8Linear` cannot quantize a Conv3d just by changing its class or weight dtype. Backend support at the datatype level does not guarantee a fast tactic for every shape.

### F6. Fit quantization parameters

For INT8, start with symmetric per-output-channel weight scaling where supported, FP32 scales and calibrated activation ranges. For FP8, use the scale granularity actually supported by the selected convolution backend. Compare max-absolute versus percentile/MSE-based clipping on the fitting set; store the selected algorithm, percentiles, sample counts and clipping fractions.

Do not optimize clipping against the held-out comparison videos. Calibrate first-use and recurrent distributions; if one shared range fails, either use an explicitly supported state-specific policy with correct runtime dispatch or leave the layer higher precision. Never add hidden inference-time recalibration during timed runs.

Keep boundary tensors/cache state BF16 initially so the experiment isolates compute quantization. If cache quantization is explored later, it requires a separate policy and recurrence test because it changes accumulated temporal information.

### F7. Isolated decoder quality, then recurrent video quality

Stage 1 uses identical saved latent windows. Save raw BF16-control and candidate RGB before encoding. Measure whole-frame and face/mouth errors, inspect tooth boundaries, and compare frame-to-frame changes within the same motion trajectory. Include exposed-teeth frames from beginning, middle and end, not only periodic closed-mouth moments.

Stage 2 runs the full generator so decoder changes feed into motion encoding. Check whether small isolated errors accumulate into tooth flicker, mouth motion changes or face drift. A decoder that passes fixed-latent checks can still fail recurrence.

If quality fails, restore the highest-impact changed stage, then split the group until the speed/quality tradeoff is understood. Restore RGB/late-stage precision before proposing sharpening as compensation. Do not add postprocessing to make a quantization result appear acceptable.

### F8. Integrating engines with compilation

Keep engine calls outside unsupported TorchInductor tracing regions, or expose a well-defined custom operator with validated shape/dtype/stream semantics. Compare against the best compiled BF16 reference, not only an eager decoder. The cost of graph breaks, launches and copies must be included.

Do not compile the whole decoder blindly around mutable engine state. Verify execution on the intended CUDA stream, correct event dependencies, tensor lifetime until completion, no per-layer CPU round-trips, and safe destruction/reload. Disable graph capture unless the chosen engine/adapter supports the exact state and allocation pattern.

**Phase F exit:** a cache-correct adapter, an exact cost/coverage inventory, engine dispatch proof where applicable, fixed-latent and recurrent quality evidence, and a measured decoder/full-pipeline gain or an explicit no-go.

## 10. Phase G — conditional INT4 FFNs

Proceed only after decoder feasibility is understood and a candidate kernel supports the exact SM89 workload. Do not invest in a full INT4 video implementation based solely on smaller stored weights.

### G1. Backend feasibility before model conversion

Test both FFN shapes with captured inputs:

```text
First projection:  M=6480, K=1536, N=8960
Second projection: M=6480, K=8960, N=1536
```

Evaluate groupwise W4A8 if a compatible fast kernel exists; otherwise explicitly identify W4A16 as weight-only. Start with group size 128; try 64 only if actually supported. Use symmetric/asymmetric layouts according to the chosen kernel contract, not an assumed interchangeable format. Record padding, group axis, nibble ordering, zero points and scale dtype.

Run the same full-FFN measurement including GELU, activation conversion, unpacking/dequantization, both GEMMs and output conversion. Compare with **compiled PRO FP8 FFNs**, historically approximately 3.38–3.43 ms on representative blocks. Beating eager BF16 or reducing checkpoint size is insufficient.

Use a provisional kernel screening threshold of at least 10% faster median full-FFN time over five measurement batches. This is an engineering effort gate, not a universal hardware requirement. If improvement is smaller, consider it only when a separately measured memory/capacity benefit justifies the quality risk.

### G2. Quantization and reconstruction

Load original BF16 weights. Fit weight clipping/group scales on representative FFN activations. Start with the simplest supported groupwise quantizer. If poor accuracy but meaningful kernel gain justifies further work, use activation-aware reconstruction/AWQ/GPTQ-style fitting appropriate to these linear layers; do not assume an LLM library's model-level adapter supports SoulX.

Validate both projections independently and together. GELU output distributions differ from the first projection's input. Use a sensitivity-driven mix of INT4, FP8 and BF16; store exact exclusions. Norms, modulation, audio conditioning and output head remain protected.

Ensure serialization roundtrips packed data and scales exactly. Save the original checkpoint hash plus packing/backend version. Quantized checkpoints require a matching loader; never label a collection of packed buffers universally portable.

### G3. Stop conditions

Stop the INT4 branch if no compatible kernel beats FP8 on real shapes, errors persist after a small documented restoration sweep, memory benefits are insignificant relative to decoder/compile buffers, or the engine requires an incompatible runtime without a viable isolated control. Save the result as a failed or memory-only experiment, not a speed improvement.

Native NVFP4/MXFP4 compute recipes requiring Blackwell are outside the 4070 SUPER experiment. Four-bit storage on Ada is not the same hardware feature.

**Phase G exit:** measured full-FFN advantage and validated packed-state contracts before any claim of a faster INT4 video model.

## 11. Phase H — optional cross-attention extension and combination

Only test cross-attention quantization after self-attention and decoder results justify continued work. Query/output projections execute on visual features; K/V are already prepared once per chunk. Start with `blocks.*.cross_attn.q` and `.o`. Keep audio projection, conditioning K/V and normalization high precision initially.

Capture actual cross-attention batch/token shapes. Validate that cached conditioning tensors are reconstructed with the correct policy and cleared at clip/reset boundaries. Evaluate lip articulation carefully: reducing precision here directly changes the audio-to-image pathway.

Create combination policies only from independently surviving components. Use the same environment and compiler settings for controls and candidates. Minimum attribution rows:

| ID | FFN | Self projections | Self attention | Decoder | Purpose |
| --- | --- | --- | --- | --- | --- |
| R0 | BF16 | BF16 | Flash2 | Eager BF16 | Stock quality anchor |
| R1 | FP8 | BF16 | Flash2 | Compiled BF16 | Accepted PRO FP8 reference |
| P1 | FP8 | FP8 selected | Flash2 | Compiled BF16 | Projection attribution |
| A1 | FP8 | BF16 | Selected Sage2 | Compiled BF16 | Attention attribution |
| D1 | FP8 | BF16 | Flash2 | Selected eight-bit subset | Decoder attribution |
| W1 | Selected INT4/FP8 | BF16 | Flash2 | Compiled BF16 | Optional FFN attribution |
| C1 | FP8 | Selected | Selected | Selected | Combined surviving low-risk components |
| C2 | Selected INT4/FP8 | Selected | Selected | Selected | Optional stronger combination |

If a backend requires a new environment, add an R1 control in that environment before interpreting its row. Add one-at-a-time ablations of the final combination to quantify its components. Do not assume isolated gains multiply.

## 12. Phase I — quality and performance qualification

### I1. Screening and final measurement schedule

Use this fixed progression to avoid an unbounded experiment matrix:

1. CPU policy/state checks.
2. Exact-shape GPU kernel checks and dispatch verification.
3. One seed-50 short run for crash/artifact screening.
4. Seed-51 short run for a second trajectory.
5. Five interleaved timing observations for survivors, including fresh reference controls.
6. All four seeds, three timing repeats per seed for the final policy and its reference under the same environment.
7. A paired 60-second recurrence run, then a five-minute run if the candidate is intended for serving and still stable.
8. Same-GPU MuseTalk comparison.
9. Separate serving/concurrency qualification only after a meaningful single-stream result.

Use a saved schedule of reference/candidate runs, such as alternating A/B and B/A pairs. Record the schedule before observing timings. Do not run competing GPU benchmarks concurrently. Model loads can be sequential in separate processes; keep warm/cold-cache status visible and compare like with like.

Use medians and show every repeat/range. Report per-clip results rather than pooling thousands of dependent frames as independent samples. If confidence intervals are computed, resample independent runs or suitably grouped temporal segments and label small sample limits. Do not claim a tiny improvement is reliable when it lies inside observed run variation.

### I2. Performance gates

The following are proposed engineering gates, not pre-existing measured properties:

- A kernel candidate should generally improve the affected complete operator by at least 10% before a large implementation effort. Include all conversion/layout overhead.
- A combined video candidate should improve median useful FPS by at least 5% over fresh PRO FP8 controls, with no seed showing an unexplained >5% slowdown. Smaller gains may be retained only with a stated other benefit, such as verified capacity or memory improvement.
- Report cold load/compile, warm first-window latency and steady generation separately; no compile time may disappear from startup claims.
- For one 25-FPS stream, the recurrent 28-frame window deadline is 1.12 seconds. A provisional 20% time reserve means a p95 target ≤0.896 seconds; validate on enough long-run windows and report maximum misses too.
- Beating MuseTalk requires a fresh matched timing contract and a gain larger than observed measurement noise; use ≥5% as a provisional margin, not a mathematical proof from one run.
- Five 25-FPS callers require 125 aggregate useful FPS before reserve. Report actual completed unique frames per caller and deadlines, not batch-size times a single-stream number.
- No OOM, nonfinite output, hidden precision fallback, silent CPU offload, or unrecorded resolution/step change is acceptable.

Record Torch allocated/reserved peaks, whole-device sampled peaks, non-Torch engine/workspace allocations when available, and build/warmup memory. Quantized weights can coexist with large compiled buffers. Do not claim a smaller checkpoint automatically allows another concurrent stream.

### I3. Mouth/teeth quality rubric

For every comparison, retain stock PRO as an anchor and PRO FP8 as the user-accepted quality reference. Deliver full-frame video, a synchronized mouth crop video, and lossless sample sheets. Use original-size views as well as clearly labeled nearest-neighbor enlargements. Avoid enhancing only one panel.

Inspect these failure categories:

| Category | What to inspect |
| --- | --- |
| Tooth boundaries | Lost divisions, merged bright bands, erased upper/lower separation |
| Stability | Tooth divisions popping, crawling texture, brightness flicker at a stable mouth opening |
| Anatomy | New dark gaps, duplicated rows, implausible teeth appearing/disappearing |
| Mouth articulation | Timing of opening/closure, jaw range, sibilants and vowels |
| Face | Identity drift, lip shape, skin artifacts, head pose changes |
| Recurrence | Worsening artifacts across windows, reset boundaries, long-run drift |
| Silence | Persistent unintended mouth motion or new visible teeth during closure |

Select the same timestamp/frame indices across panels. Add stratified examples where the reference mouth is closed, partially open and clearly open, including chunk boundaries and late frames. Do not select only favorable candidate frames. Save the selection rule and indices before making the sheet.

Use metrics as diagnostics: face detection rate, mouth opening trajectory, mouth-center displacement, oral edge energy, fixed-latent pixel/perceptual error, and motion-compensated mouth temporal difference if alignment is reliable. Reject alignment failures instead of turning them into false flicker scores. Increased edge energy can indicate artifacts and is not an automatic quality pass.

For short controlled fixtures, expect all 250 faces to remain detected as in PRO FP8; investigate every new failure. Require no systematic new blur/flicker/anatomical failure category relative to PRO FP8 in the reviewed stratified examples and full videos. Record per-seed qualitative findings in a review table. Any acceptance of a visible tradeoff must be described explicitly rather than hidden in an aggregate average.

### I4. Relative audio-sync checks

The existing SyncNet utility is hardcoded to the ten-second fixture and overlapping evaluation windows. Generalize it before using other utterances or a long clip:

- Read the actual run's effective audio and frame rate from its manifest.
- Use each variant's own face boxes as the primary crop and the shared-reference boxes as a sensitivity diagnostic.
- Handle missing/invalid face boxes explicitly and report coverage; do not assume every frame has `face_box`.
- Derive valid audio/video windows from clip duration, the 16-frame visual window, 52 mel columns and offset range.
- Save evaluated offsets, window indices, per-window scores, best-offset and zero-offset means, and shuffled-audio control.
- Report the exact checkpoint/config hash and evaluator environment.
- Do not call these official LSE-C/LSE-D scores.

Compare score changes against reference-repeat/crop-method variability. A consistent new best-offset shift exceeding one 25-FPS frame (40 ms) triggers review and blocks an unqualified lip-sync-equivalence claim. An almost tied peak at neighboring offsets is not strong evidence of a real shift. A score drop that disappears with correct per-output crops must be documented as crop sensitivity, not silently deleted.

### I5. Long-run and reset checks

A 1,500-frame test repeats the known ten-second utterance six times. Label it as recurrence stress, not six independent speech samples. Also test a genuinely different utterance before claiming generalization. Sample exposed teeth throughout the minute.

For a serving finalist, run five minutes with bounded output buffering. The v1 runner accumulates all RGB frames in RAM; replace that long-run storage path with bounded transfer/recording queues or a separately timed disk sink. Report its cost consistently for both variants. Keep the timing definition fixed rather than excluding candidate-only output work.

Test sequential reset/reload with the same seed and with a new seed. Test two interleaved session states against independent references before any shared-model concurrency claim. Keep separate RNG, motion latents, audio history, person reference and decoder cache state. Interruption/reconditioning tests belong to serving integration and must not be inferred from a completed offline minute.

### I6. Media and artifact verification

`verify_artifacts.py` must check:

- Exactly 250/1,500/etc. useful frames as requested.
- Source panels 320×576 and the correct combined canvas dimensions.
- Native 25-FPS playback and expected duration.
- Audio present, correct effective audio identity, aligned start and matching duration.
- Full FFmpeg decode with error checking.
- PNG crops sourced from the documented raw frames; decoded-MP4 samples labeled as such.
- Correct panel labels and policy/run IDs.
- Source, weight, fixture, policy and engine hashes.
- Every report link resolves; every claimed result maps to a complete manifest.
- Failed/rejected runs remain marked and do not enter winner aggregation.

## 13. Phase J — fair MuseTalk target and capacity evaluation

### J1. Build the comparison adapter from existing local code

Reusable source exists at `benchmarks/pro_musetalk_redub_20260917/redub.py`, which uses native DWPose/S3FD/BiSeNet preparation, `load_all_model`, Whisper/AudioProcessor, cached latents/masks, `datagen`, UNet, VAE decode and blending. The installed MuseTalk repository also has `/workspace/MuseTalk/scripts/benchmark_pipeline.py` and `/workspace/MuseTalk/scripts/realtime_inference.py`.

Read and adapt these into the new `compare_musetalk.py`; do not execute the old redub script unchanged because it hardcodes other source videos/audio and output locations. Run MuseTalk in its compatible environment and PRO in its compatible environment, sequentially on the same physical GPU. Record their differences instead of forcing an unsupported common Torch version.

Define an adapter contract with `load`, `prepare_avatar`, `prepare_audio`, `reset_session`, `generate_to_rgb`, and `close`, plus separate timing for each. Preserve MuseTalk's normal cached-avatar advantage in its steady-state measurement. Count PRO portrait/reference preparation separately in the same way. If a moving source template is required for MuseTalk, document it and its preparation cost; do not include generating that template with PRO in MuseTalk's ordinary renderer timing or pretend the systems generate identical head motion.

### J2. Two timing contracts

Report both:

1. **Warm rendering:** prepared avatar, audio processing from a defined input boundary, all generated RGB and final composition/transfer; encoding excluded for both. Report audio-feature preparation separately if also reporting render-only throughput.
2. **Application delivery:** audio input through usable encoded frames, including audio conditioning, composition, transfers, encoding, queueing and delivery overhead in both systems.

Keep portrait identity, audio, requested output count, playback cadence, canvas and delivery encoding comparable. State that MuseTalk synthesizes a face crop whereas PRO generates the full frame. Equal delivery resolution is not equal neural work.

Historical 37–44-FPS MuseTalk artifacts and the 30.9–40.4-FPS redub calculations are context only. They have different hardware/runtime/timing profiles and cannot serve as a fresh victory threshold without a matched run.

### J3. Concurrency after single-stream qualification

Do not instantiate N copies and call that a batching optimization. Evaluate shared read-only model weights with isolated per-session state. The current prepared-conditioning closure in the offline runner contains a mutable `chunk` dictionary and is not a ready concurrent API; refactor it into session-owned arguments/state before interleaving callers.

Measure one, two, and then additional sessions only while memory and deadlines permit. Use each session's own audio/reference/RNG/motion history. Record per-session p95/max inter-frame delay, initial latency, generated unique frames, deadline misses, aggregate FPS and device memory. Compare the same concurrency levels with MuseTalk. If PRO remains below one-stream real-time throughput, report the offline gain and stop short of a multi-call serving claim.

**Phase J exit:** a clearly scoped speed comparison and, only if actually tested, a capacity result. An inability to beat MuseTalk is a valid finding and must remain visible.

## 14. Command cookbook

### 14.1 Existing commands available now

Run from `/workspace/SoulX-FlashHead`. These commands perform work only when intentionally executed in a later implementation phase.

```bash
# Existing CPU checks for the accepted implementation.
PYTHONPATH=. .venv/bin/python -m pytest \
  tests/test_pro_quantization.py tests/test_optimizations.py -q

# Existing stock PRO control; output must not already exist.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_20260918/run.py \
  --output benchmarks/pro_quantization_v2_20260918/runs/stock-seed50-r01 \
  --seed 50 --repeats 3

# Existing accepted PRO FP8 control.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_20260918/run.py \
  --output benchmarks/pro_quantization_v2_20260918/runs/pro-fp8-seed50-r01 \
  --seed 50 --precision fp8 --optimized-dit --compile-dit --vae compiled \
  --repeats 3 --save-raw

# Existing stock-versus-PRO-FP8 review; comparison directory must be new.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_20260918/review.py \
  --baseline benchmarks/pro_quantization_v2_20260918/runs/stock-seed50-r01 \
  --candidate benchmarks/pro_quantization_v2_20260918/runs/pro-fp8-seed50-r01 \
  --output benchmarks/pro_quantization_v2_20260918/runs/review-stock-pro-fp8-seed50-r01 \
  --label 'PRO FP8 reference'
```

The v1 review's baseline label is fixed to stock PRO. Do not use it unchanged to label PRO FP8 as stock when comparing against a v2 candidate. Implement configurable baseline/candidate labels in the v2 review.

### 14.2 Proposed commands — implement these interfaces first

The following commands are acceptance examples for the new scripts. They **do not work until the files/flags in this plan are implemented**. `preflight` creates a metadata file; run/capture/profile/review commands create new directories; collector/verifier read an existing tree. Give every retry a new run ID.

```bash
# After creating the new experiment directory and preflight tool.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/preflight.py \
  --output benchmarks/pro_quantization_v2_20260918/preflight.json

# Capture development activations separately from timing.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/capture.py \
  --policy benchmarks/pro_quantization_v2_20260918/policies/pro_fp8_reference.json \
  --fixtures benchmarks/pro_quantization_v2_20260918/fixtures.json \
  --split calibration --output benchmarks/pro_quantization_v2_20260918/calibration/capture-r01

# Profile decoder operator coverage.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/profile.py \
  --policy benchmarks/pro_quantization_v2_20260918/policies/pro_fp8_reference.json \
  --fixtures benchmarks/pro_quantization_v2_20260918/fixtures.json \
  --fixture-id indian150-a --seed 50 --component decoder --mode trace \
  --output benchmarks/pro_quantization_v2_20260918/runs/profile-decoder-r01

# Projection kernel gate.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/kernels.py \
  --component self-projections \
  --captures benchmarks/pro_quantization_v2_20260918/calibration/capture-r01/manifest.json \
  --policy benchmarks/pro_quantization_v2_20260918/policies/self_all_fp8.json \
  --output benchmarks/pro_quantization_v2_20260918/runs/kernels-self-fp8-r01

# Initial projection candidate.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  --policy benchmarks/pro_quantization_v2_20260918/policies/self_all_fp8.json \
  --fixtures benchmarks/pro_quantization_v2_20260918/fixtures.json \
  --fixture-id indian150-a --seed 50 --frames 250 --repeats 3 \
  --output benchmarks/pro_quantization_v2_20260918/runs/self-fp8-seed50-r01

# Exact-latent decoder trial after creating a resolved decoder plan/backend.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/decoder_trial.py \
  --reference-policy benchmarks/pro_quantization_v2_20260918/policies/pro_fp8_reference.json \
  --candidate-policy benchmarks/pro_quantization_v2_20260918/policies/decoder_int8_conservative.json \
  --captures benchmarks/pro_quantization_v2_20260918/calibration/capture-r01/manifest.json \
  --output benchmarks/pro_quantization_v2_20260918/runs/decoder-int8-fixed-latents-r01

# Correctly labeled accepted-reference versus candidate review.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/review.py \
  --baseline benchmarks/pro_quantization_v2_20260918/runs/pro-fp8-seed50-r01 \
  --candidate benchmarks/pro_quantization_v2_20260918/runs/self-fp8-seed50-r01 \
  --baseline-label 'PRO FP8 reference' --candidate-label 'PRO FP8 plus self projections' \
  --output benchmarks/pro_quantization_v2_20260918/runs/review-self-fp8-seed50-r01

# Generalized relative sync diagnostic; reads audio identity from run metadata.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/sync_check.py \
  --reviews benchmarks/pro_quantization_v2_20260918/runs/review-self-fp8-seed50-r01 \
  --crop-mode own \
  --output benchmarks/pro_quantization_v2_20260918/runs/sync-self-fp8-r01.json

# Winner evaluation only after combined-selected.json is created from surviving policies.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/sweep.py \
  --policy benchmarks/pro_quantization_v2_20260918/policies/combined-selected.json \
  --reference-policy benchmarks/pro_quantization_v2_20260918/policies/pro_fp8_reference.json \
  --fixtures benchmarks/pro_quantization_v2_20260918/fixtures.json \
  --fixture-id indian150-a --seeds 50 51 0 1 --frames 250 --repeats 3 \
  --output benchmarks/pro_quantization_v2_20260918/runs/final-four-seeds-r01

# Long comparison: run the reference separately with identical arguments/profile.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  --policy benchmarks/pro_quantization_v2_20260918/policies/combined-selected.json \
  --fixtures benchmarks/pro_quantization_v2_20260918/fixtures.json \
  --fixture-id indian150-a --seed 50 --frames 1500 --repeats 1 \
  --output benchmarks/pro_quantization_v2_20260918/runs/final-60s-seed50-r01

# Final read-only aggregation/verification.
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/collect.py \
  --root benchmarks/pro_quantization_v2_20260918/runs \
  --output benchmarks/pro_quantization_v2_20260918/summary.json
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_v2_20260918/verify_artifacts.py \
  --root benchmarks/pro_quantization_v2_20260918/runs
```

Implement `sweep.py` as a sequential process orchestrator using argument lists, not shell-concatenated user inputs. Generate/store the interleaved schedule before launch. Run references under the same environment as candidates. Stop or mark a cell failed on errors; never relabel an incomplete cell as complete.

The decoder engine builder must accept a resolved plan, calibration manifest, precision and unique output directory. Its exact invocation depends on the backend selected by feasibility profiling; save the fully resolved build command in the engine manifest. Do not invent a working TensorRT command before determining supported export/operator versions.

## 15. Required artifact schema and reporting

Every complete run directory should contain:

```text
results.json                 # status, environment, profile, stage/total times, memory
requested-policy.json
resolved-policy.json
fixtures-resolved.json       # original and effective repeated/padded audio identity
source-manifest.json
sources/                    # relevant code snapshot with relative paths
video.mp4
samples/                    # lossless selected frames; frame/time indices recorded
resources-*.json
stdout.log
stderr.log
```

Capture directories additionally need a manifest for tensor shape/dtype/module/step/window/seed/fixture/hash. Engine directories need calibration/ONNX/engine hashes, exact backend/toolchain, shapes, build flags, workspace cap, bindings and layer precision records. Review directories need comparison videos, crop indices/sheets, metrics, per-seed qualitative review and source run IDs.

A result row must identify whether it is fresh GPU inference, a GPU diagnostic without video generation, CPU analysis, hypothetical arithmetic, or reused historical evidence. Include exact GPU and VRAM near the top of reports, and per row if hardware differs. Do not aggregate different resolutions, runtimes, step counts or precision policies into one headline median.

The final report must include all materially relevant failures: unsupported kernels, slower INT4, decoder graph breaks, OOM, precision fallback, tooth degradation and recurrence problems. Explain why the selected policy is chosen, how much each component contributes, and what remains unresolved.

## 16. Failure handling and rollback

| Failure | Required response |
| --- | --- |
| Unknown/unmatched policy path | Fail before conversion; fix policy/schema and create a new run |
| Unexpected packed-QKV state | Reject or use the tested dedicated packed adapter; never access nonexistent FP8 `.weight` blindly |
| FP8 buffer accidentally recast | Discard model; reload original weights and reapply policy before compile |
| NaN/Inf or stale activation scale | Stop candidate; save failing shape/input metadata and fix numerical path |
| Engine unsupported shape/precision | Save build/layer diagnostics; try a documented supported partition or keep that layer BF16 |
| Decoder cache count/order changes | Reject adapter before video timing; restore causal-class/cache semantics |
| Engine faster alone, pipeline slower | Include dispatch/layout/graph-break cost; reject or improve partition granularity |
| Quality loss | Restore selected modules; rerun held-out checks; do not conceal with sharpening |
| New environment changes baseline | Add same-environment controls and separate runtime gains |
| OOM | Record build/runtime stage and peaks; reduce workspace/duplicate allocations or tested batch size; preserve primary profile and co-resident conditions |
| GPU lease unavailable | Exit cleanly or schedule later; do not start another model owner |
| No meaningful quantization path to target | Publish measured limit; retain PRO FP8 and describe architectural options separately |

Rollback is selection of the preserved PRO FP8 recipe in an isolated compatible environment with original checkpoint hashes. Never overwrite its snapshot, weights or comparison videos. Rebuild compiler caches for the correct policy/runtime instead of reusing incompatible compiled objects. Existing production integration remains outside these offline changes until a separately reviewed implementation actually supports it.

## 17. Implementation order and completion checklist

Use the following order. Decoder profiling/feasibility is deliberately early; it can invalidate the overall speed premise before an expensive transformer sweep.

- [ ] A: Verify preserved PRO FP8 files, sources, weights and fixtures.
- [ ] A: Record preflight and reproduce the accepted reference.
- [ ] B: Implement strict policy resolution and safe two-pass conversion.
- [ ] B: Create v2 runner, explicit attention routing, manifests and focused CPU checks.
- [ ] C: Profile DiT and Wan; capture bounded representative tensors and fitting/evaluation splits.
- [ ] F1–F5: Prove cache-safe BF16 decoder adapter and exact-shape quantized convolution feasibility.
- [ ] D: Implement separate FP8 self-attention projections and perform two-seed screening.
- [ ] F6–F8: Fit/select decoder quantization; pass fixed-latent and recurrent checks.
- [ ] E: Evaluate pinned Sage2 backend only with same-environment controls.
- [ ] G: Evaluate INT4 FFNs only after a kernel gate against current FP8.
- [ ] H: Test optional cross-attention and combine only surviving components.
- [ ] I: Complete four-seed timing/quality validation, 60-second recurrence and media checks.
- [ ] J: Measure MuseTalk with the same defined delivery/timing contract.
- [ ] J: Qualify shared-state isolation and concurrency only if the single-stream result warrants it.
- [ ] Reporting: Preserve a named winner with source/configuration/backend hashes and commands, or record a no-go with PRO FP8 retained as the best accepted version.

A complete implementation cycle need not execute every optional branch. It must explain each skipped/rejected branch using measured feasibility, preserve the accepted reference, and deliver an honest speed/quality decision. A completion claim for the user's ultimate objective requires actually exceeding the matched MuseTalk baseline while maintaining acceptable teeth and articulation; implementation effort alone is not completion of that objective.

## 18. External implementation references

These are backend documentation/research references, not performance measurements on this repository. Recheck the selected release's exact requirements when implementing:

- [SageAttention official repository](https://github.com/thu-ml/SageAttention): explicit APIs, compiler requirements, supported architectures and accumulation modes.
- [TorchAO quantized inference recipes](https://docs.pytorch.org/ao/stable/workflows/inference.html): weight-only versus weight-and-activation recipes and hardware constraints.
- [TensorRT convolution operator](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/Convolution.html): supported types and explicit quantization interface; actual tactic support still requires a local build.
- [TensorRT 10.x explicit quantization guidance](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html): archived, version-specific Q/DQ and scaling guidance relevant only if selecting a compatible 10.x backend.
- [TensorRT support matrix](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html): select the intended runtime release and hardware rather than assuming newest documentation matches the installed environment.
