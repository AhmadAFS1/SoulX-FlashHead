# TensorRT experiments: isolated, explicit and fail-closed

Use the [implementation status](docs/research/IMPLEMENTATION_STATUS.md) for measured end-to-end results. Kernel gains below are not claims of25% whole-model improvement. PyTorch remains the default backend.

## Environment and artifacts

```bash
.venv/bin/python -m pip install --target .trt-experiment/site-packages \
  -r requirements-trt-experiment.txt
```

The optional TensorRT10.9/ONNX environment is an isolated `--target` directory, not an upgrade to the working venv. The requirements pin CUDA runtime12.8.57 and protobuf3.20.3 to avoid substituting incompatible latest dependencies. Enable it only for an experiment command with `PYTHONPATH=/workspace/SoulX-FlashHead/.trt-experiment/site-packages`. Engine/ONNX/captured tensor files are ignored by Git; benchmark JSON and reproduction code are published.

Reference API documentation: [NVIDIA TensorRT Python API](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html). This code pins the10.9 API actually tested locally; do not assume latest11.x builder behavior is compatible.

## Real activation capture and FFN partition build

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m soulx_rtc.trt_experiment capture \
  --memory-mode staged --layers 0 1 2 --output .trt-experiment/new-captures

PYTHONPATH=/workspace/SoulX-FlashHead/.trt-experiment/site-packages \
  .venv/bin/python -m soulx_rtc.trt_experiment build \
  --layers 0 1 2 --precision bf16 --inputs .trt-experiment/new-captures \
  --output .trt-experiment/new-ffn-engines
```

Capture accepts native width/height and image/audio paths; defaults are the approved idle avatar's canonical image and existing MuseTalk English audio. It captures the actual first-chunk input to each selected feed-forward layer and the actual denoised latent before VAE decode. Specify all layer indices0–29 to build the complete FFN set. Each block is exported from its actual checkpoint weights; no MuseTalk engine/calibration is reused.

FFNs are token-independent, so CPU tracing uses one token with a dynamic sequence declaration, which is then specialized to the real captured sequence length before TensorRT parsing. Numerical checks and timing use the full actual captured tensor, not the one-token trace input. The deployed engine accepts one exact profile; wrong batch/resolution fails.

`--resume` skips layers already present in the completed report. Partial artifacts without a completed report are not silently considered validated. Use a new directory if an interrupted build left an incomplete engine.

The builder persists a timing cache across compatible partitions with device-mismatch checks enabled. This reduces repeated tactic-search time; it does not reuse weights or bypass per-layer numerical/reload checks. Captures require a fresh directory so tensors from different fixtures cannot silently mix.

## VAE decoder build

```bash
PYTHONPATH=/workspace/SoulX-FlashHead/.trt-experiment/site-packages \
  .venv/bin/python -m soulx_rtc.trt_vae_experiment \
  --input .trt-experiment/new-captures/vae-input.pt \
  --output .trt-experiment/new-vae-engine
```

Only the deterministic tensor decoder is exported. Motion encoding, posterior sampling/private RNG, denoising recurrence, color correction and scheduling remain in PyTorch. First/recurrent chunks still pass the same five-frame denoised latent shape to the decoder. Native33-frame decode and nine overlap frames are retained.

The optional `--precision fp16` variant preserves pixel-normalization reductions in explicit FP32 and compares against the original BF16 decoder outputs. It is a separate numerical/quality experiment, not an identical-output setting.

### Rebuild the measured 480×832 profile

Stop the SoulX service and verify its worker has exited first. These are sequential commands, not parallel jobs; the per-checkout GPU-owner lock enforces that. Use fresh directory names if the shown artifacts already exist. Keep unrelated services untouched.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  .venv/bin/python -m soulx_rtc.trt_experiment capture \
  --width 480 --height 832 --memory-mode staged --layers {0..29} \
  --output .trt-experiment/portrait-captures

PYTHONPATH=/workspace/SoulX-FlashHead/.trt-experiment/site-packages \
  .venv/bin/python -m soulx_rtc.trt_experiment build \
  --layers {0..29} --precision bf16 \
  --inputs .trt-experiment/portrait-captures \
  --output .trt-experiment/portrait-ffn-bf16

PYTHONPATH=/workspace/SoulX-FlashHead/.trt-experiment/site-packages \
  .venv/bin/python -m soulx_rtc.trt_vae_experiment \
  --input .trt-experiment/portrait-captures/vae-input.pt \
  --output .trt-experiment/portrait-vae-bf16
```

The brace expansion is Bash syntax. The native FFNs expect `[1,1950,1536]`; the decoder expects `[128,5,26,15]` and emits `[1,3,33,832,480]`. The caller discards the original nine overlap frames. These artifacts cannot be reused for 576×1024 or batch two.

## Runtime integration

Pass `--trt-ffn DIRECTORY` and/or `--trt-vae PATH/vae.engine` to `soulx_rtc.experiment` or `start_webrtc.sh`, with the isolated `PYTHONPATH` enabled. FFN artifacts are batch-one profiles. Each artifact verifies shape, checksum, checkpoint hash, TensorRT version and GPU name before use; deserialize/binding/execution errors fail instead of silently selecting another backend.

The30 FFN contexts share one explicitly allocated maximum-size workspace because execution is serialized by a single GPU owner. Original FFN GPU weights are removed before replacement-engine allocation. Inference uses a non-default PyTorch compute stream. Unused PyTorch allocations are released before VAE decode when FFN engines are enabled; this can affect throughput and is included in end-to-end timing. TensorRT allocations are outside PyTorch's normal allocator statistics: inspect total device usage too.

The integrated VAE uses a transient workspace: allocate/bind immediately before decode, wait for decode completion, and release the buffer before motion encoding. The native portrait workspace is approximately664 MiB, so retaining it across all stages caused an otherwise avoidable OOM in the tested co-resident configuration. The explicit synchronization and allocator overhead are included in full-model timing. Every subsequent execute binds a newly owned buffer; a dangling pointer is never used for inference.

`reference` memory mode can offload remaining PyTorch DiT weights during reference preparation, but cannot move TensorRT engine weights to CPU. `staged` offloading is rejected for FFN engines. Do not assume a model fitting in PyTorch also fits with TensorRT resident contexts and workspaces.

Reference-mode startup also offloads the remaining PyTorch DiT while deserializing replacement engines, then restores it. This reduces temporary startup overlap, not the steady TensorRT footprint. This late startup change was not GPU-revalidated for the full TRT set after the co-resident process grew to7,456 MiB; it cannot make the previously measured roughly5,164-MiB runtime fit alongside that load.

## Promotion gates

Successful export, engine reload and finite kernel outputs are necessary, not sufficient. Compare complete generated trajectories against the current compiled PyTorch baseline using identical avatar/audio/seed/resolution/steps. Include actual video inspection, lip-sync, recurrent drift, fresh-process reload, useful-FPS A/B repetitions, VRAM peaks and live peer stalls. A positive isolated FFN or decoder result must not be added arithmetically to claim an end-to-end improvement. Do not enable an engine by default until the combined gates pass.
