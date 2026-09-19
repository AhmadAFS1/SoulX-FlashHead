# Further PRO quantization: analysis and priorities

Implementation handoff: [highly detailed v2 runbook](PRO_QUANTIZATION_V2_IMPLEMENTATION_PLAN_2026-09-18.md), with exact code targets, execution order, commands and acceptance criteria. This is a plan, not a new benchmark.

September 18, 2026. **CPU source/checkpoint-header inspection and arithmetic on saved measurements; no new GPU inference, quantization trial, or package installation.** The PRO measurements reused here were recorded on **NVIDIA GeForce RTX 4070 SUPER, physical 12 GB class / 12,282 MiB visible VRAM**, verified by saved `nvidia-smi`: driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8, with two co-residents reporting 2,478 and 234 MiB. The workload is the Indian-man 1.50× portrait, 320×576, four denoising steps, shift 5, two motion latents, native 25-FPS output.

The accepted version is preserved as **[PRO FP8 v1](../../benchmarks/pro_quantization_20260918/PRO_FP8/README.md)**. This retains the tested source/configuration recipe, source hashes, original checkpoint hashes and existing evidence. It reconstructs quantized buffers at load; it is not a separately exported quantized checkpoint. Future experimental variants must retain separate names and output directories.

## What is still available to quantize?

PRO FP8 quantizes only the 60 feed-forward linear layers. The decoder and attention paths still provide substantial scope. CPU inspection of the PRO safetensors header gives:

| Transformer parameter group | Parameters, including biases/norms | Present treatment | Next candidate |
| --- | ---: | --- | --- |
| Feed-forward | 826,068,480 | FP8 matrix weights/activations; BF16 biases/interfaces | Groupwise INT4 weights with an efficient activation-aware kernel |
| Self-attention | 283,392,000 | BF16 projections, FlashAttention2 | FP8 Q/K/V/output projections; separately quantized attention math |
| Cross-attention | 283,392,000 | BF16, conditioning K/V cached per chunk | FP8 query/output projections first, only after self-attention validation |
| Other transformer tensors | 114,842,432 | Higher precision | Protect initially: conditioning, modulation, normalization, output head |

Totals are transformer checkpoint counts, not the entire pipeline and not percentages of runtime. FFN matrix weights already quantized total 825,753,600 elements, approximately 54.77% of that checkpoint. Reducing these from eight to four bits would save only another **393.75 MiB before scales/packing**, not halve total pipeline VRAM. Existing compiled PRO FP8 uses more peak allocator memory than eager stock because of compilation buffers.

There is no isolated teeth module. Preserving selected decoder stages and conditioning pathways is a conservative hypothesis to test, not a guarantee that other layers cannot affect teeth.

## The speed constraint

The first final seed-50 PRO FP8 repeat generated 250 frames in 26.564 seconds. Saved CUDA stage events attribute 11.757 seconds to the transformer and 13.086 seconds to Wan decode. The remaining 1.722 seconds includes motion encode, audio and unassigned overhead. This is approximately **44% transformer, 49% decoder, 6% other**. See [the recorded result](../../benchmarks/pro_quantization_20260918/candidate-seed50/results.json).

Conditional arithmetic, holding remaining work fixed:

| Hypothetical improvement over current PRO FP8 | Predicted useful FPS |
| --- | ---: |
| Current measured repeat | 9.41 |
| Transformer 2× faster, decoder unchanged | 12.09 |
| Decoder 2× faster, transformer unchanged | 12.49 |
| Both 2× faster | 17.68 |
| Both 3× faster | 24.99 |
| Both 4× faster | 31.52 |
| Both 5× faster | 37.37 |

Formula: `250 / (11.75657 / transformer_speedup + 13.08618 / decoder_speedup + 1.72159)`. These are bounds/scenarios, not measured or promised quantization gains. Even removing all transformer time predicts only **16.88 FPS** with other work fixed. Stage partitions can change when execution changes.

### What does beating MuseTalk mean here?

The retained [compiled MuseTalk engine artifact](../../benchmarks/musetalk-engine-b8-compiled.json) records **RTX 4070**, not SUPER, Torch 2.5.1+cu121, 256×256 neural face generation on a 512×512 canvas: 37.04 aggregate FPS for one session and 43.76 aggregate FPS for ten sessions. Physical 12-GB-class hardware is documented in the historical provenance index; the artifact itself omits visible VRAM, driver, date and co-resident load. These are historical, differently scoped measurements, not a matched current-GPU comparison. The separately cited 44.69-FPS run's external JSON path is no longer available here.

A more recent [MuseTalk redubbing result](../../benchmarks/pro_musetalk_redub_20260917/musetalk/results.json) explicitly records September 17, **RTX 4070 SUPER / 12,282 MiB**, driver 595.84, Torch 2.5.1+cu121, and co-resident telemetry. Its 250-frame render-and-encode timings imply approximately 30.9–40.4 FPS, but preprocessing, avatar reuse, and output motion differ. This is also not a controlled comparison with PRO generation.

Use **40–45 useful aggregate FPS as a provisional engineering target**, then rerun MuseTalk and PRO under a defined same-GPU timing contract before claiming victory. Relative to 9.41 FPS, this demands about 4.25–4.78× overall acceleration. Precision reduction alone has not established such a path. At 25 FPS, even one live stream requires more throughput than current PRO FP8; multi-user capacity is not equivalent to fitting several model copies in memory. Measure shared-model batching, p95 latency, VRAM and delivered frames separately. Five active 25-FPS users require at least 125 useful FPS before headroom.

## Recommended experiments, in order of purpose

### 1. Broaden transformer FP8 coverage: the simplest next implementation

Target `blocks.*.self_attn.{q,k,v,o}` in [flash_head_model.py](../../flash_head/src/modules/flash_head_model.py). Start with separate projections to isolate errors, then evaluate packed QKV if worthwhile. The existing source supports a packed-QKV owner and alternate forward branches; replacement must respect which branch actually runs. Preserve Q/K normalization, rotary operations and BF16 interfaces. Do not quantize a packed weight with one scale without checking different Q/K/V distributions.

This quantizes previously untouched large GEMMs instead of replacing working FP8 FFNs with an uncertain lower-bit kernel. Benchmark real 6,480-token PRO shapes with scale/conversion overhead included. Cross-attention query/output GEMMs are a later extension: they touch the full visual sequence, but carry articulation risk. Cross-attention K/V are already cached once per chunk and are lower-priority speed targets.

### 2. Quantized self-attention math: a separate, complementary experiment

Our existing SageAttention 1 trial reached about **9.93 FPS**, versus about 9.4 for PRO FP8, but changed mouth motion more in the inspected seed. It is evidence of modest additional headroom, not evidence of a multi-fold pipeline improvement.

Evaluate a pinned SageAttention2/2++ Ada backend separately from projection quantization. Its documented CUDA API supports INT8 QK and FP8 PV; do not label every Sage2 installation INT4 because of the paper title. The current host compiler is CUDA 12.1 even though Torch carries CUDA 12.8. Upstream specifies at least CUDA 12.4 for Ada FP8 and 12.8 for 2++; a future trial needs an isolated compatible build environment. Keep audio cross-attention unchanged initially. [Official implementation and requirements](https://github.com/thu-ml/SageAttention).

### 3. Selective decoder quantization: the most important speed feasibility test

Wan decode is about half the current runtime. Inspect operator-level time first; existing totals do not establish which decoder blocks dominate. Candidate targets are the internal convolutional residual blocks in `vae.model.decoder.middle` and `vae.model.decoder.upsamples`, plus spatial upsample convolutions. Initially protect the final high-resolution residual stage, RGB head, normalization, residual accumulation and temporal cache interfaces. Protecting the final stage may also leave much of the cost untouched; measure coverage rather than assuming success.

Compare **INT8 weights/activations** and **FP8 weights/activations** only where the selected backend actually supplies fast kernels for the exact shapes. The Wan decoder uses causal Conv3d, Conv2d spatial resampling, SiLU, RMS normalization, and feature caches. It cannot use the current FP8 Linear wrapper unchanged. TensorRT explicit Q/DQ is one possible implementation route; current [convolution documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/_static/operators/Convolution.html) lists INT8/FP8, but does not establish efficient execution of every Wan shape in our pinned environment. First build and profile representative partitions, including conversions and cache transfer. A successful export or fake quantization is insufficient.

For calibration, capture initial and recurrent decoder states, closed/open mouths and multiple seeds. Use per-output-channel INT8 weight scales where supported; choose activation ranges from representative data and validate on held-out clips. Keep temporal state explicit and verify reset/cached behavior. First compare identical saved latents through each decoder, then full recurrent generation, because decoder errors feed back through motion encoding. If PTQ visibly erases tooth detail, restore sensitive blocks before considering training.

### 4. Four-bit FFNs: a measured kernel experiment, not the default bet

Test groupwise W4A8 (four-bit weights, eight-bit activations) or W4A16 (four-bit weights, BF16 activations), starting with group size 128 and optionally 64 if the kernel supports it. Use representative activation-aware reconstruction/clipping and restore sensitive projections to FP8/BF16. Start from original weights, not already rounded FP8 buffers, to avoid compounded quantization error.

The 6,480-token FFN workload differs substantially from small-batch autoregressive LLM decoding. Weight compression and unpacking costs can fail to improve this workload's compute time. Gate on full FFN latency versus the existing **3.38–3.43 ms compiled FP8** baseline, not versus stock BF16. The prior native INT8 FFN trial was slower than FP8, demonstrating that a precision label is not a speed result.

[TorchAO recipes](https://docs.pytorch.org/ao/stable/workflows/inference.html) distinguish INT4 weight-only and INT4-weight/FP8-activation configurations. Exact Torch 2.7.1/SM89 compatibility must be checked in isolation; latest APIs are not assumed compatible. Its native NVFP4/MXFP4 compute recipes require Blackwell-class hardware, so FP4 is not a straightforward native-compute upgrade on this 4070 SUPER. LLM kernel support also does not imply a ready SoulX implementation.

## Decision and comparison protocol

Retain PRO FP8 as the accepted quality reference. Name future candidates by their actual policy: broader FP8 projections, quantized attention, selective eight-bit decoder, or mixed INT4 FFNs. Do not call a partially INT4 model universally four-bit.

The most useful first work is **FP8 self-attention projections plus a decoder kernel/precision feasibility study**, each isolated before combining them. Run four-bit FFNs only if their actual kernels beat current FP8. Keep the original 320×576/four-step profile while measuring precision effects.

Every surviving candidate should be compared against both stock PRO and preserved PRO FP8 using the exact Indian-man 1.50× inputs, seeds 50/51/0/1, matched playback, raw mouth crops and a 60-second recurrence check. Keep calibration and held-out quality clips distinct. Evaluate tooth boundaries and stability, lip-sync with appropriate face crops, face/head motion, startup/p95 chunk time, throughput and memory. Then measure single- and multi-session throughput against a fresh MuseTalk control.

If decoder and transformer gains plateau below the target, fewer denoising steps, a distilled decoder, or changed spatial work are separate architectural/training experiments. They may be necessary to beat MuseTalk, but should not be advertised as stronger quantization or silently substituted into a precision comparison.
