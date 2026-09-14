#!/usr/bin/env bash
# Run from the patched SoulX-FlashHead checkout. Additional CLI options follow.
set -euo pipefail
FLASHHEAD_ROOT="${FLASHHEAD_ROOT:-/workspace/SoulX-FlashHead}"
FLASHHEAD_PYTHON="${FLASHHEAD_PYTHON:-.venv/bin/python}"
cd -- "$FLASHHEAD_ROOT"
exec "$FLASHHEAD_PYTHON" -u -m soulx_rtc.server \
  --width 320 --height 576 --steps 4 --fps 25 \
  --optimized --real-rope --memory-mode compact \
  --lean --fused-qkv --int8-weights --cuda-memory-mib 3584 \
  --batch 1 --max-active-calls 1 --max-sessions 10 \
  --idle-policy source --h264-preset veryfast "$@"
