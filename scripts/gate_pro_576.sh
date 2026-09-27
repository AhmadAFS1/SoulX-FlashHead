#!/usr/bin/env bash
# Quality gate for a 576x320 PRO run: review.py against a baseline run, plus the candidate's
# colour drift against its own window 0 (needs the run's raw.npy, i.e. run.py --save-raw).
#
# usage: scripts/gate_pro_576.sh <candidate-run-dir> [<label>]
#   BASELINE (benchmarks/pro_taew2_1_20260927/ship-r01: the adopted taew2_1 build) -- any run
#            dir with video.mp4 + results.json; benchmarks/pro_30fps_20260921/s30-reuse is the
#            historical Wan 2.1 reference (19.62 FPS).
# Writes <candidate-run-dir>/../review-<name>/ and prints correlation, edge ratio, mouth distance
# and the drift line. Bands and how to read them: docs/research/PRO_TAEW2_1_DEFAULT_VAE_2026-09-27.md.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"

if [[ $# -lt 1 ]]; then
  sed -n '2,11p' "$0"
  exit 2
fi
candidate="${1%/}"
name="$(basename "${candidate}")"
label="${2:-${name}}"
baseline="${BASELINE:-benchmarks/pro_taew2_1_20260927/ship-r01}"
review="$(dirname "${candidate}")/review-${name}"

rm -rf "${review}"
"${python_bin}" benchmarks/pro_quantization_v2_20260918/review.py \
  --baseline "${baseline}" --candidate "${candidate}" \
  --baseline-label "$(basename "${baseline}")" --candidate-label "${label}" \
  --output "${review}" > "${review}.log" 2>&1 || { tail -5 "${review}.log"; exit 1; }
"${python_bin}" - "${review}/review.json" "${name}" "$(basename "${baseline}")" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))["paired"]
print("%s vs %s: corr=%.3f edge_ratio=%.3f mouth_dist=%.1fpx open_pairs=%d" % (
    sys.argv[2], sys.argv[3], p["opening_correlation"], p["median_edge_ratio_on_both_open"],
    p["median_mouth_center_distance_px"], p["both_open_pairs"]))
PY
if [[ -f "${candidate}/raw.npy" ]]; then
  "${python_bin}" scripts/colour_drift.py "${candidate}/raw.npy" | grep "slope" | sed 's/^/  drift /'
else
  echo "  drift: skipped (no ${candidate}/raw.npy; run with --save-raw)"
fi
