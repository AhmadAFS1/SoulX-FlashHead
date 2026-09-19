# Quantization, TensorRT feasibility and closer-face follow-up

September 16, 2026. Current host: **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM**, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8. Direct `nvidia-smi` inspection before this follow-up showed 2,487 MiB device use; the co-resident OmniVoice process remains running. Quantization/export findings below are source inspection and retained benchmark analysis, not a new TensorRT engine build. Historical TensorRT inference was on **RTX 4070 (non-SUPER), 12,282 MiB**, driver 570.181 host lineage, CUDA 12.8, TensorRT 10.9, alongside OmniVoice with variable memory use.

## What the five-stream result establishes

The [capacity report](../../benchmarks/concurrency_15fps_20260916/README.md) records five active 15-FPS streams at 320×576/four steps with BF16 resident weights and GPU batch five. The 41-second/15-turn run had zero video repeats but three isolated 20-ms audio holds. Four is the better operating point; five is the highest tested video-cadence result. Six failed. This short test does not establish an absolute hardware limit across all possible optimizations.

The user reports damaged teeth in the original batch-five recording. Record that as a quality failure despite successful transport. The earlier [teeth investigation](MALE_TEETH_QUALITY_2026-09-13.md) found teeth artifacts with both BF16 and INT8. Neither BF16 nor batching has been isolated as the cause. A closer reference gives the mouth more generated pixels at fixed output dimensions, but it also changes conditioning and motion; closer is not guaranteed to eliminate dental artifacts.

## Can we quantize it?

Yes. [`compact_weights.py`](../../soulx_rtc/compact_weights.py) already implements per-output-channel symmetric INT8 storage for DiT block linear weights. Each forward converts the weight back to the input dtype and calls `F.linear`. It saves resident weight bytes while doing BF16 computation. Normalization, conditioning outside the blocks and output projection retain their original precision. It does not quantize the whole pipeline or provide integer matrix multiplication.

The [matched serving sweep](../../benchmarks/concurrency_15fps_20260916/README.md) observed approximately 38–40 aggregate generated FPS with INT8 storage and a 3,584-MiB allocator cap, versus 73–77 FPS with BF16 resident weights and a 7,168-MiB cap. These differ in both storage precision and memory budget: dequantization alone is **not** proven to explain the entire slowdown. VAE/color/feedback time also differed strongly, making allocator/workspace/kernel choices a plausible additional factor. An equal-cap alternating A/B test is needed to isolate the cause. On this machine's available memory, the tested INT8 configuration is not a throughput improvement.

For actual acceleration, the next candidate is selectively quantized **DiT feed-forward matrix multiplication** using INT8 weights/activations or FP8, while retaining the VAE, normalization, attention softmax and sensitive conditioning in their validated precision initially. FP8/INT8 export requires explicit quantization scales and compatible kernels. A cast or the existing storage wrapper does not supply this automatically. NVIDIA describes ONNX QuantizeLinear/DequantizeLinear (Q/DQ) export as the explicit quantization path; hardware/software support and actual fusion must be checked for the pinned build. [NVIDIA quantization documentation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/work-quantized-types.html), [support matrix](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/getting-started/support-matrix.html).

Calibrate using real activations from all four denoising steps, first and recurrent chunks, silence and voiced audio, multiple identities, and the closer references. The current exporter captures only the first invocation of each FFN, which is suitable for an export smoke test but insufficient representative quantization calibration. Keep held-out clips/seeds for teeth, lip closure, identity and recurrent drift checks. INT4 storage is a lower-priority experiment because visible errors and conversion overhead may outweigh memory savings here. No new quantized checkpoint is claimed by this analysis.

## Can we export to TensorRT?

**Selected components have already been exported, built, reloaded and run in this repository. A complete SoulX engine and a batch-five TensorRT FFN service have not.**

| Component | Existing evidence | Work required for current five-stream profile |
| --- | --- | --- |
| All 30 DiT feed-forward blocks | `trt_experiment.py` exports actual checkpoint weights through ONNX opset 17; BF16/FP16 engines and per-layer numerical/timing reports exist | Capture/build 320×576 profiles; support batch sizes 1–5 at dispatch and warmup; current CLI explicitly rejects FFN TRT when maximum batch is not one |
| LTX VAE decoder | `trt_vae_experiment.py` exports deterministic decode; historical native portrait kernel speedup ~1.182× | Rebuild exact current latent `[128,5,18,10]` and output `[1,3,33,576,320]` on this GPU/runtime, then benchmark complete calls; decode is still per session, so it is the simpler batch-five integration candidate |
| Full DiT | No full export/build evidence found | Handle attention/export operators, real-RoPE, audio conditioning, denoising-step inputs and variable session batches; test partition boundaries and numerical recurrence |
| Audio encoder / motion encoder | Retained PyTorch execution | Separate export feasibility tests; preserve audio window behavior and posterior/private RNG ownership |
| Entire serving pipeline | Scheduling, media, recurrence and RNG remain Python/PyTorch | Keep host orchestration; exporting deterministic tensor regions is the practical route |

See [existing TensorRT instructions](../../TENSORRT_EXPERIMENTS.md), [FFN builder](../../soulx_rtc/trt_experiment.py), [VAE builder](../../soulx_rtc/trt_vae_experiment.py), and [runtime validation](../../soulx_rtc/trt_backend.py). Existing native portrait FFNs expect `[1,1950,1536]`; the current 320×576 batch-five tensor is `[5,900,1536]`. Renaming old engine files or removing the CLI restriction is insufficient. `Engine` currently installs FFNs with the default batch-one shape, and live scheduling/warmup uses smaller batches as well as the maximum. Shape-specific dispatch or tested dynamic profiles are required.

Historical kernel improvements did not multiply into large whole-pipeline gains: the retained native TRT ten-job run was around 28.57 aggregate FPS on the older RTX 4070/profile. This cannot predict this SUPER's 320×576 result. Engine metadata checks GPU identity, runtime, shapes and checkpoint hashes; rebuild for the target. TRT workspace allocations also sit outside Torch allocator counters.

## Next implementation order

1. Establish the closer 1.25× face as the inspectable quality fixture, with the same 320×576/four-step BF16 profile. Review the new receiver recording before treating teeth as fixed.
2. Build a BF16 TensorRT VAE decoder for 320×576 first, keeping DiT in the working compiled BF16 path. This avoids the current batch-one FFN limitation and directly addresses substantial per-session decode work.
3. Prototype one selectively quantized FFN with representative calibration and compare full-shape latency and errors. Expand only if complete video quality survives.
4. Add FFN profile dispatch for all actual batch sizes before a combined C4/C5/C6 rerun. Retain zero-held-video, audio continuity, startup latency and total VRAM checks.

At ~77.36 generated FPS, six 15-FPS speakers require at least 90 FPS, about **16.3%** more aggregate throughput before serving overhead; an illustrative 20% spare-capacity budget would require 112.5 FPS (~45.4% more). Neither export success nor reduced model bytes proves those gains.
