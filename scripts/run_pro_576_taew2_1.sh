#!/usr/bin/env bash
# Default SoulX-FlashHead PRO 576x320 build since 2026-09-27: the taew2_1 (TAEHV) tiny
# autoencoder replaces the Wan 2.1 VAE for the per-window decode and the motion re-encode.
# The reference-image encode stays on the Wan 2.1 encoder. Decision and evidence:
# docs/research/PRO_TAEW2_1_DEFAULT_VAE_2026-09-27.md.
#
# usage: scripts/run_pro_576_taew2_1.sh <output-dir> [extra run.py flags, e.g. --save-raw]
#
# Environment overrides:
#   FIXTURE  (indian150-a)   SEED (50)   FRAMES (250)
#   POLICY   (benchmarks/pro_30fps_20260920/policies/pro576_taew2_1_flash2.json)
#   BACKEND  (auto: tensorrt when a TensorRT builder is importable, else compile)
#   TRT_BUILDER_PATH (benchmarks/tiny_vae_20260926/decoder_bakeoff/_trt10_path)
#
# Do not add --overlap-skip, --compile-vae-encode, --skip-decoder-blocks or --vae-weights:
# they belong to the Wan decoder and run.py refuses them with a tiny decoder.
# Rollback to the Wan 2.1 build: docs/research/PRO_TAEW2_1_DEFAULT_VAE_2026-09-27.md, "Rollback".
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
  sed -n '2,17p' "$0"
  exit 2
fi
output="$1"
shift

policy="${POLICY:-benchmarks/pro_30fps_20260920/policies/pro576_taew2_1_flash2.json}"
builder="${TRT_BUILDER_PATH:-benchmarks/tiny_vae_20260926/decoder_bakeoff/_trt10_path}"
backend="${BACKEND:-auto}"

# The TensorRT backend builds its FP16 engines on first use (about 45 s) and caches them under
# benchmarks/tiny_vae_20260926/engines/, so it needs the TensorRT builder, not only the runtime.
if [[ "${backend}" == "auto" ]]; then
  if PYTHONPATH="${builder}" "${python_bin}" -c "import tensorrt; tensorrt.Builder" >/dev/null 2>&1; then
    backend=tensorrt
  else
    backend=compile
    echo "run_pro_576_taew2_1: no TensorRT builder under ${builder}; using --tiny-vae-backend compile" >&2
  fi
fi

unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH="${builder}:.pro-quant-deps:."
export SOULX_STAGE_REUSE_OUTPUTS="${SOULX_STAGE_REUSE_OUTPUTS:-1}"

exec "${python_bin}" benchmarks/pro_quantization_v2_20260918/run.py \
  --policy "${policy}" \
  --fixtures benchmarks/pro_30fps_20260920/tts-fixtures/fixtures.json \
  --fixture-id "${FIXTURE:-indian150-a}" --seed "${SEED:-50}" --frames "${FRAMES:-250}" --repeats 1 \
  --sampling-steps 2 --timestep-variant distilled_aligned --latency-detail off \
  --static-int8-scales benchmarks/pro_30fps_20260921/int8-amax.json \
  --dit-cuda-graph --fused-int8-ffn --lean-delivery \
  --tiny-vae-decoder taew2_1 --tiny-vae-encoder taew2_1 --tiny-vae-backend "${backend}" \
  "$@" \
  --output "${output}"
