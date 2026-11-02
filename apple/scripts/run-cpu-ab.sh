#!/usr/bin/env bash
# Render CPU A/B clips sequentially (resumable). Usage, from the repo root:
#   apple/scripts/run-cpu-ab.sh OUTPUT_ROOT MODE VARIANT AVATAR:AUDIO [AVATAR:AUDIO ...]
# Each clip goes to OUTPUT_ROOT/<mode>-<variant>/<avatar>-<audio stem>/ and is
# skipped when its run.json already exists. CPU-only PyTorch; see cpu_runtime.py.
set -euo pipefail
root=$1 mode=$2 variant=$3
shift 3
python=${PYTHON:-.venv-coreml/bin/python}
for clip in "$@"; do
  avatar=${clip%%:*}
  audio=${clip#*:}
  stem=$(basename "${audio%.*}")
  out="$root/$mode-${variant//+/_}/$avatar-$stem"
  if [[ -f "$out/run.json" ]]; then
    echo "skip $out"
    continue
  fi
  mkdir -p "$out"
  echo "render $out"
  PYTHONPATH=apple "$python" -m flashhead_coreml.cpu_runtime --avatar "$avatar" --audio "$audio" \
    --mode "$mode" --variant "$variant" --output "$out" >"$out/render.log" 2>&1
  tail -1 "$out/render.log"
done
