#!/usr/bin/env bash
# Render CPU A/B clips sequentially (resumable). Usage, from the repo root:
#   apple/scripts/run-cpu-ab.sh OUTPUT_ROOT MODE VARIANT AVATAR:AUDIO [AVATAR:AUDIO ...]
# Each clip goes to OUTPUT_ROOT/<mode>-<variant>[-<tag>]/<avatar>-<audio stem>/ and is
# skipped when its run.json already exists. CPU-only PyTorch; see cpu_runtime.py.
# Optional environment: SEED (default 42); FORCE_FROM=<run root> teacher-forces
# motion latents from the same clip under that root; TAG names the arm;
# THREADS sets PyTorch threads (default 4).
set -euo pipefail
root=$1 mode=$2 variant=$3
shift 3
python=${PYTHON:-.venv-coreml/bin/python}
for clip in "$@"; do
  avatar=${clip%%:*}
  audio=${clip#*:}
  stem=$(basename "${audio%.*}")
  out="$root/$mode-${variant//+/_}${TAG:+-$TAG}/$avatar-$stem"
  extra=(--seed "${SEED:-42}" --threads "${THREADS:-4}")
  if [[ -n "${FORCE_FROM:-}" ]]; then extra+=(--force-motion "$FORCE_FROM/$avatar-$stem"); fi
  if [[ -f "$out/run.json" ]]; then
    "$python" - "$out/run.json" "${SEED:-42}" "${FORCE_FROM:+$FORCE_FROM/$avatar-$stem}" <<'PY'
import json, sys
from pathlib import Path
meta = json.loads(Path(sys.argv[1]).read_text())
forced = str(Path(sys.argv[3]).resolve()) if sys.argv[3] else None
recorded = str(Path(meta['forced_motion_from']).resolve()) if meta.get('forced_motion_from') else None
if meta['seed'] != int(sys.argv[2]) or recorded != forced:
    sys.exit(f"{sys.argv[1]} was rendered with seed={meta['seed']} forced={recorded}; use another TAG")
PY
    echo "skip $out"
    continue
  fi
  mkdir -p "$out"
  echo "render $out"
  PYTHONPATH=apple "$python" -m flashhead_coreml.cpu_runtime --avatar "$avatar" --audio "$audio" \
    --mode "$mode" --variant "$variant" --output "$out" "${extra[@]}" >"$out/render.log" 2>&1
  tail -1 "$out/render.log"
done
