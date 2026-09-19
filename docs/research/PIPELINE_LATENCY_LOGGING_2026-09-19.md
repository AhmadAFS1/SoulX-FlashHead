# Pipeline latency logging

Implemented **2026-09-19**. Hardware for the existing PRO study and current read-only snapshot: **NVIDIA GeForce RTX 4070 SUPER, 12 GB physical / 12,282 MiB visible**, driver **595.84**, Torch **2.7.1+cu128**, CUDA runtime **12.8**. Latest candidate engines use TensorRT **10.9.0.34**. [Validation and environment evidence](../../benchmarks/pipeline_latency_20260919/README.md).

**This change adds instrumentation and CPU validation; it does not add a new full-model GPU profile or a speed result.** The SoulX GPU lease was occupied during validation. Logging is off by default and is enabled explicitly for diagnostic runs. Existing policies, model weights and decoder engine files are preserved; the Wan VAE source used for engine hash validation is unchanged.

## What is recorded

| Component | Stage logging | Additional `modules` diagnostic |
| --- | --- | --- |
| Initialization | VAE loading, DiT weight load/placement, audio encoder loading, quantization and engine installation | No model-construction hooks |
| Reference preparation | Image read, resize/transfer, reference VAE encoding | Reference setup remains a total; detailed wrappers attach after policy preparation |
| Audio | History assembly, normalization, host-to-device transfer, Wav2Vec, hidden-state stacking, window indices/gather, conditioning transfer | Individual Wav2Vec modules |
| Prepared conditioning | Per-window audio projection/cached cross-attention K/V work | Individual projector and K/V modules |
| Denoising | Noise initialization, history injection, each model call with step number, latent update, final history injection | Each transformer block, Q/K/V/O, FFN linears/GELU, norms, explicit self/cross-attention backend calls |
| Decoder | Complete selected decoder call | Wan/LTX module calls: convolutions, norms, residual groups, resampling, head; PyTorch execution explicitly eager |
| Fused TensorRT stages | Input/cache cast and contiguity, output allocation, binding/enqueue/fencing, actual engine-stream interval, output/cache cast back to BF16 | Same engine boundaries; engine internals stay fused, not invented per-layer timings |
| Feedback/postprocessing | Color correction, motion frame selection, motion VAE encode, output FP32 conversion | Individual motion encoder modules |
| RGB delivery | RGB layout/clipping, history trim and device-to-host copy, finite check/uint8 conversion | Same |
| Video delivery | Encode/mux host wall duration | Same; external encoder internals are not attributed to CUDA events |

Functional operations inside compiled graphs, such as casts, concatenation and rotary arithmetic, are visible as operators/kernels in the accompanying Torch trace where the compiler exposes them. Fused kernels are not split into fictitious Python sub-operation timings. Use the policy-preserving trace for actual compiled cost; eager module timings help locate operations but do not predict compiled savings.

## Two diagnostic levels

- **`--latency-detail stages`** preserves the requested DiT/decoder compilation and TensorRT engines. Wrappers are outside compiled entrypoints; no per-layer hooks are inserted into those graphs. This is the first diagnostic to run on the latest candidate.
- **`--latency-detail modules`** explicitly disables PyTorch DiT/FFN/decoder compilation for that diagnostic process, while retaining its selected quantization and TensorRT engines. It logs individual module calls across transformer, audio encoder and VAE. The requested policy is saved unchanged; manifests record the eager override. This mode is not an equivalent throughput run.
- **`--latency-detail off`** is the default. New scopes do not create CUDA events or log files. Existing pipeline synchronization/print behavior is retained. No claim of zero Python overhead is made.

`profile.py --mode trace` supplies Chrome trace markers named `soulx/<component>` together with the existing operator/kernel CSV. Logging also works through `run.py` without a Torch trace. Activation capture, old decoder inventory and the new latency diagnostics cannot be combined in one run.

## Run the latest candidate

From `/workspace/SoulX-FlashHead`, with the **existing GPU lease available**, use a new output directory. These are reproduction commands, not GPU runs executed for this change. The native Sage dependency path is necessary to select the measured SM89 build.

```bash
PYTHONPATH=/workspace/experiments/pro30-deps/sage-sm89:.pro-quant-deps:/workspace/experiments/ojin-components-deps:. \
  .venv/bin/python benchmarks/pro_quantization_v2_20260918/profile.py \
  --policy benchmarks/pro_30fps_20260919/policies/combined_fp16_sage_fp8.json \
  --fixtures benchmarks/pro_quantization_v2_20260918/fixtures.json \
  --fixture-id indian150-a --seed 50 --frames 56 \
  --component pipeline --mode trace --latency-detail stages \
  --output benchmarks/pipeline_latency_20260919/latest-stages-r01
```

Run the same command with `--latency-detail modules --latency-max-events 100000` and a separate output such as `latest-modules-r01` for eager module attribution. The two levels must not be compared as quantization speed variants. Fifty-six useful frames cover two generated windows after two warmup windows; use a later, longer diagnostic to investigate sustained recurrence.

For a clean throughput measurement, use `run.py` with `--latency-detail off` and no profiling options, keeping the established repeated-process/seed protocol. Logging and profiling have overhead; no instrumented run qualifies a speed improvement on its own.

## Output and interpretation

| Artifact | Meaning |
| --- | --- |
| `latency-events.jsonl` | One span per line with ID/parent ID, component, setup/warmup/generation/delivery phase, repeat/window, step or engine metadata where relevant, input shapes/dtypes for wrappers, host start/duration, CUDA elapsed duration and stream, completion/error status |
| `latency-summary.json` | Per-phase/component counts, means, maxima and inclusive totals; environment/policy/source provenance, event count, dropped count, diagnostic execution mode |
| `trace.json`, `aggregated.csv` | Existing Torch profiler artifacts when `--mode trace` is used; named scopes help locate work and kernels |
| `results.json` | Requested policy, effective decoder/prepared-model execution, timing diagnostics, environment, source hashes and artifact links |

`host_ms` is inclusive CPU wall duration around the call. For asynchronous calls it mostly measures enqueue work; for a blocking transfer it includes waiting. `cuda_elapsed_ms` is an **inclusive stream interval**, including waits and gaps between launches, not the sum of kernel execution times. CPU-only scopes use `null` for unavailable CUDA durations rather than a misleading zero.

TensorRT engine events are recorded on its dedicated execution stream **after its input wait**; the enclosing caller-stream scope covers binding/enqueue and the return fence. Those two intervals overlap. Parent/child IDs preserve the call hierarchy; **do not sum nested spans, CPU time plus GPU time, or concurrent stream intervals**. Use exclusive CUDA kernel rows from the Torch trace to account for kernel work, and generation wall time for end-to-end latency.

Setup, warmup/first compilation, generation and delivery remain separate phases. Generation records include repeat/window identifiers; `denoise.model` carries its step number, and child module spans link to that parent. TensorRT metadata identifies the engine/signature file. No audio/image/tensor contents are logged.

The default cap is **50,000 pending spans per flush**, configurable with `--latency-max-events`. Excess spans are dropped with an explicit counter and `complete_coverage=false`; an incomplete log cannot substantiate a complete component cost table. Events are resolved and written at setup/warmup/repeat/completion boundaries, with one CUDA synchronization per nonempty GPU flush, not one per component. Long runs should use short bounded diagnostics or an appropriately sized cap. Logs are opt-in run artifacts, not an unbounded always-on production service logger.

To inspect the slowest **inclusive** generation components:

```bash
.venv/bin/python - <<'PY'
import json
from pathlib import Path
p = Path('benchmarks/pipeline_latency_20260919/latest-stages-r01/latency-summary.json')
data = json.loads(p.read_text())
print('Dropped spans:', data['dropped_events'])
rows = [r for r in data['aggregates'] if r['phase'] == 'generation']
for row in sorted(rows, key=lambda r: r['cuda_elapsed_total_ms'] or 0, reverse=True)[:20]:
    print(row['name'], 'calls=', row['calls'],
          'inclusive CUDA ms=', row['cuda_elapsed_total_ms'],
          'inclusive host ms=', row['host_total_ms'])
PY
```

Use this listing to navigate the trace, not as an additive pie chart. A large `decode.total` naturally contains smaller VAE/TRT spans.

## Implementation and validation

[Recorder and reversible wrappers](../../flash_head/utils/latency.py), [pipeline scopes](../../flash_head/src/pipeline/flash_head_pipeline.py), [audio/RGB boundaries](../../flash_head/inference.py), [TRT stream/cast scopes](../../soulx_rtc/pro_vae_stage_backend.py), [runner integration](../../benchmarks/pro_quantization_v2_20260918/run.py), [tests](../../tests/test_pipeline_latency.py).

The CPU suite tests bounded/error-safe logging, correct requested-stream event placement with mocks, wrapper restoration, unchanged outputs/RNG/motion history on the actual generation method with small CPU substitutes, and a compiled CPU graph that remains compiled under stage logging. Existing quantization/cache tests also run. [Recorded validation](../../benchmarks/pipeline_latency_20260919/README.md). CUDA events on the real model and TensorRT integration still require a GPU diagnostic; no test result here substitutes for that measurement.

This instrumentation supports the next experiments in the [current pipeline precision/throughput analysis](PRO_PIPELINE_PRECISION_AND_THROUGHPUT_2026-09-19.md). Live service request/queue/WebRTC timing outside this offline visual pipeline is not added by this change.
