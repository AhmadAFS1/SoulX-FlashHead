# Experimental INT8 decoder rebuilds

The large `.pt` files under `benchmarks/pro_quantization_v2_20260918/calibration/`
are calibration captures. They are only needed when rebuilding the explicit
Q/DQ TensorRT decoder engines; runtime inference loads the generated `.engine`
files instead.

Use a new output directory for every build. The wrapper refuses to overwrite an
existing directory and delegates the actual build, calibration, validation, and
engine-plan generation to the repository's existing builder.

```bash
scripts/rebuild_int8_decoder_engines.sh \
  --variant twelve \
  --output benchmarks/pro_quantization_v2_20260918/runs/decoder-int8-twelve-rebuild-r01
```

Available variants are `qdq-output`, `six`, and `twelve`. To inspect the
resolved command without using a GPU:

```bash
scripts/rebuild_int8_decoder_engines.sh --variant twelve \
  --output /tmp/soulx-int8-rebuild --dry-run
```

The selected Python environment must provide CUDA, PyTorch, ONNX, and
TensorRT. Rebuilds are target-specific because the generated metadata records
the TensorRT version and GPU name.
