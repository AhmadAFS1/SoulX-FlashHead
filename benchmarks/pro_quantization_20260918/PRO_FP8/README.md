# PRO FP8 — preserved reference v1

September 18, 2026. Saved source/configuration snapshot of the already tested **NVIDIA GeForce RTX 4070 SUPER, physical 12 GB class / 12,282 MiB visible** result. Driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8; two co-residents reported 2,478 and 234 MiB during the runs. This preservation action is CPU file handling, not new GPU inference.

The name **PRO FP8** now refers to the selected 9.37–9.44 useful-FPS version: all 60 FFN linears use FP8 weights/activations; compiled DiT and BF16 Wan decode; FlashAttention2; four denoising steps. It includes the tested recipe, source snapshots, dependency inventory, original checkpoint hashes, and links to the existing four-seed/60-second footage. Original weights and comparison artifacts stay in their existing locations.

[Manifest](manifest.json) · [Saved reference result](reference-results.json) · [Quality/performance report](../README.md)

This is not a new standalone FP8 checkpoint: the tested implementation creates FP8 buffers when loading original BF16 weights. To reproduce in an isolated checkout, restore `sources/` to the corresponding repository-relative paths, use the recorded environment and original checkpoint hashes, then run:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/pro_quantization_20260918/run.py \
  --output benchmarks/pro_quantization_20260918/NEW-pro-fp8-seed50 --seed 50 \
  --precision fp8 --optimized-dit --compile-dit --vae compiled --repeats 3
```

Future experiments should use separate names and output directories. This snapshot retains the four source files verified against the saved inference hashes; additional VAE/helper sources and the dependency inventory were captured at preservation time. It is not a complete hermetic environment image.
