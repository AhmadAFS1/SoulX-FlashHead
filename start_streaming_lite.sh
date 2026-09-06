#!/usr/bin/env bash
set -euo pipefail

SOULX_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SOULX_PORT="${SOULX_PORT:-7860}"

cd "$SOULX_DIR"

if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv. Install the SoulX-FlashHead environment first." >&2
  exit 1
fi

if [[ ! -f models/SoulX-FlashHead-1_3B/Model_Lite/diffusion_pytorch_model.safetensors ]]; then
  echo "Missing Model_Lite checkpoint." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="$SOULX_PORT"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SOULX_DIR/.torchinductor}"

exec .venv/bin/python gradio_app_streaming.py
