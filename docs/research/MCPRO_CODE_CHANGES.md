# mcPRO series — code change reference

Every source change in commits `f370189`, `86d577e`, `7601cd1`, file by file.
Bracketed by git tags `mcpro-before` (pre-change tree) and `mcpro-latest`.

**No behaviour changes by default.** Every new capability is off unless an
explicit flag or policy key turns it on. A `run.py` invocation with no new
arguments executes the same path as before the series.

This is a code reference. Rationale and measurements live in
[the landed-changes doc](WAVE0_WAVE1_LANDED_2026-09-20.md); commands live in
[the 4070 runbook](RTX4070_RUNBOOK_2026-09-20.md).

```bash
git diff mcpro-before..HEAD            # everything
git log --oneline --grep=mcPRO         # the series
git revert 86d577e                     # drop just the step reduction
git reset --hard mcpro-before          # drop everything
```

---

## New files

| File | Lines | Purpose | Deps |
| --- | ---: | --- | --- |
| `flash_head/src/pipeline/schedules.py` | 108 | Denoising-step schedule tables, `shipped` and `distilled_aligned`. | stdlib only |
| `benchmarks/pro_quantization_v2_20260918/plan_preflight.py` | 173 | CPU pre-check of a TRT stage plan before the GPU lease. CLI + importable. | stdlib only |
| `benchmarks/pro_30fps_20260919/policies/combined_fp16_sage_fp8_crossqo.json` | 41 | Candidate policy: cross-attention q/o at FP8. | — |
| `benchmarks/pro_30fps_20260919/policies/combined_fp16_sage_fp8_fastaccum.json` | 39 | Candidate policy: `fast_accum: true`. | — |

Both new Python modules are importable without torch, deliberately, so they run
on a machine with no CUDA.

---

## Modified files

### `flash_head/src/pipeline/flash_head_pipeline.py` (+110)

| Change | Default | Notes |
| --- | --- | --- |
| Five **class-level** attributes on `FlashHeadPipeline` | all off | `lean_delivery`, `verbose_timing`, `skip_zero_weighted_noise`, `resolved_timesteps`, `timestep_variant`. Class-level, not `__init__`-only: `generate()` reads them unconditionally and callers legitimately build the object with `__new__`. |
| Timestep construction calls `raw_timestep_schedule(...)` | `variant="shipped"` | A `None` return means "untabulated under shipped", and the original `np.linspace` fallback runs verbatim. The CFG teacher at `sample_steps=20` depends on this. |
| `self.resolved_timesteps` recorded | — | Post-shift levels as Python floats, so the denoise loop can test the terminal step without a device comparison. |
| Terminal-`randn` skip in `denoise.update` | off | Fires only when `skip_zero_weighted_noise` is set **and** `resolved_timesteps[i+1] == 0.0`. |
| `postprocess.lean_trim` scope, **before** colour correction | off | Trims the leading `motion_frames_num` frames. |
| Device uint8 delivery in `postprocess.rgb_uint8_device` | off | Preserves the exact host op order so `raw_rgb_sha256` stays comparable; finite-check moved pre-cast. |
| 7 stdout timing lines gated on `verbose_timing` | silent | `torch.cuda.synchronize()` / `time.time()` scaffolding retained. |

**Return-type note:** with `lean_delivery` set, `generate()` returns device
`uint8` in `[T, H, W, C]` with history frames removed, instead of host-bound
`float32` in `[C, T, H, W]` with all 33 frames. Callers that already trim
themselves (`generate_video.py`, the gradio apps) must leave the flag off.

### `flash_head/inference.py` (+4)

`run_pipeline` returns early when `getattr(pipeline, "lean_delivery", False)`,
since `generate()` has already produced the final layout. Uses `getattr` with a
default so any externally constructed pipeline still works.

### `soulx_rtc/pro_quantization.py` (+17)

`Float8Linear.__init__` gains `fast_accum: bool = False`, stored on the instance
and passed to `torch._scaled_mm(..., use_fast_accum=self.fast_accum)`. Default
`False` is the setting every retained measurement used.

### `soulx_rtc/pro_quantization_v2.py` (+87)

| Change | Notes |
| --- | --- |
| `PolicyFloat8Linear.__init__` accepts `fast_accum` | Forwarded to `Float8Linear`. |
| `"fast_accum"` added to the policy `allowed` tuple, **not** to `required` | So all 19 retained policies keep validating and keep their recorded `policy_sha256`. |
| `fast_accum` type-validated as `bool` | Without it, JSON `"false"` reaches `bool(...)` and evaluates `True`. |
| Cross-attention scheme accepts `fp8_e4m3_w8a8` | Previously a blanket rejection. |
| Cross-attention `k`/`v`/`k_img`/`v_img` refused with a specific message | `prepare_kv` returns bare BF16 tensors with no scale carrier, at M=288 once per window. |
| Quantized cross-attention must name `include` rules | No implicit whole-family conversion. |
| `_pro_target_paths` returns a **3-tuple** | `(ffn_paths, attention_paths, cross_paths)`. **Signature change** — only in-repo caller is `build_conversion_plan`; archived snapshots under `benchmarks/**/sources/` keep their own copies. |
| `build_conversion_plan` plans cross-attention targets | Family `cross_attention_projection`. |
| `apply_conversion_plan` reads `fast_accum` from the resolved policy | Records `accumulation` as `fp32_scaled_mm_fast_accum` when set. |

### `benchmarks/pro_quantization_v2_20260918/run.py` (+83)

| Flag | Default | Effect |
| --- | --- | --- |
| `--sampling-steps {1,2,3,4}` | `4` | Replaces the hardcoded literal at the `prepare_params` call and in `_fixed_profile`. |
| `--timestep-variant {shipped,distilled_aligned}` | `shipped` | Selects the schedule table. |
| `--lean-delivery` | off | Rejected together with `--capture-manifest`, which expects float32 delivery. |
| `--skip-zero-weighted-noise` | off | Changes the generator stream. |

Also: `schedules.py` added to `_sources()` so the timestep table is hashed into
run provenance; a `delivery` block and a `sampling_schedule` block added to
`results.json`; `steps`, `skip_zero_weighted_noise` and `timestep_variant`
recorded in `profile` so `review.py` can see them; step count validated before
the GPU lease is acquired.

### `benchmarks/pro_quantization_v2_20260918/sweep.py` (+23)

Forwards `--sampling-steps`, `--timestep-variant`, `--lean-delivery` and
`--skip-zero-weighted-noise` to **both** roles, and records them in the sweep
manifest.

**Limitation:** forwarding to both roles means `sweep.py` runs a paired
comparison *at* a fixed step count and **cannot A/B 4-step against 2-step**. That
is deliberate — an arm differing on one side only is not controlled — but the D2
promotion sweep must be assembled from direct `run.py` calls.

---

## Tests added

| File | Tests | Needs torch |
| --- | ---: | --- |
| `tests/conftest.py` | — | Skips modules whose imports are unavailable, including transitively. |
| `tests/test_sampling_schedule.py` | 10 | no |
| `tests/test_source_manifest_drift.py` | 4 | no |
| `tests/test_policy_schema_contract.py` | 87 | no |
| `tests/test_cross_attention_policy.py` | 19 | no (stubs torch to run the real validator) |
| `tests/test_plan_preflight.py` | 12 | no |
| `tests/test_step_reduction.py` | 18 | no |
| `tests/test_shipped_behaviour_preserved.py` | 23 | no |
| `tests/test_wave1_execution.py` | 17 | **yes** — CPU torch, runs on the GPU box |

**173 pass** on a machine with no CUDA/GPU/torch. They gate **correctness only,
never promotion**: none measures speed.

---

## Behaviour explicitly preserved

Each of these is regression-guarded in `tests/test_shipped_behaviour_preserved.py`.

| Preserved | Guard |
| --- | --- |
| CFG teacher at `sample_steps=20` | `shipped` returns `None`, caller falls back to `np.linspace`. |
| `soulx_rtc.Engine(steps=2)` levels | `shipped` 2-step stays `[1000, 500]` → terminal 833.33. |
| All 19 retained policy `policy_sha256` values | `fast_accum` is optional, never required. |
| Retained `results.json` artifacts | New keys are additive; nothing is rewritten. |
| `raw_rgb_sha256` comparability | Delivery op order preserved exactly. |
| The selected candidate policy file | Untouched; new experiments are separate files. |

---

## Known gaps in this series

- **`tests/test_wave1_execution.py` has never executed.** It needs torch, absent
  on the dev machine. Its first run is Step 0 of the runbook.
- **No roofline microbenchmark (W0.3).** `combined_fp16_sage_fp8_fastaccum.json`
  will run, but a null result is uninterpretable without it.
- **`decoder_trial.py` fixed-latent control not fixed (W0.6).** Its cross-process
  spread is 7.9%, wider than the 5% promotion gate.
- **No TRT stage runtime contract tests (W0.8).** `TensorRTStage.__call__` and
  `install_stage_plan` remain untested.
- **W1.2 delayed FP8 activation scales not implemented.** Still
  `flat.abs().amax()` per call; `triton_red_fused_abs_amax_5` alone is
  27.545 ms/window at ~506 GB/s, i.e. 100% of DRAM bandwidth.
