#!/usr/bin/env bash
# macOS only. Export the candidate PRO320 bundles, gate them numerically, and
# benchmark every speed arm with the real Swift runtime and Core ML. Prints one
# summary table: warm FPS, speed-up, gates, pixel identity and lip/teeth
# metrics vs the baseline. Runs are Mac Core ML inference; per-operation
# CPU/GPU/Neural Engine placement is not verified by this script.
#
# Prerequisites (repository root):
#   make -f apple/Makefile setup-cpu-ab fetch
#   make -f apple/Makefile export bundle          # baseline: artifacts/coreml/{pro320,device/pro320}
#   .venv-coreml/bin/python apple/scripts/make_cpu_ab_fixtures.py
# Usage:
#   apple/scripts/mac-speed-sweep.sh [SPEECH.wav] [CHUNKS] [REPEATS]
# Environment:
#   AVATAR       girl | indian (default girl)
#   ARMS         space-separated subset of arm names (baseline always runs)
#   SKIP_EXPORT  1 = reuse already-exported bundles
#   SKIP_GATES   1 = skip the validate_dit gates (timings are then ungated)
set -euo pipefail
cd "$(cd "$(dirname "$0")/../.." && pwd)"
speech=${1:-artifacts/cpu-validation/inputs/eval_english_plosive.wav}
chunks=${2:-4}
repeats=${3:-2}
avatar_name=${AVATAR:-girl}
fit_dir=artifacts/cpu-validation/ffn-fit  # flashhead_coreml.ffn_pruning.DEFAULT_FIT
py=.venv-coreml/bin/python
base=artifacts/coreml/pro320
device=artifacts/coreml/device
sweep=artifacts/coreml/sweep/$(date -u +%Y%m%dT%H%M%SZ)
avatar=ios/SoulXStudio/Resources/Avatars/$avatar_name
[[ "$(uname)" == Darwin ]] || { echo "Run this on the Mac (Core ML prediction is macOS-only)."; exit 1; }
[[ -d $device/pro320 ]] || { echo "Missing $device/pro320: run 'make -f apple/Makefile export bundle' first."; exit 1; }
[[ -x $py ]] || { echo "Missing $py: run 'make -f apple/Makefile setup-cpu-ab' first."; exit 1; }
(( chunks >= 2 )) || { echo "CHUNKS must be at least 2 (warm FPS excludes the first window)."; exit 1; }
mkdir -p "$sweep"
exec > >(tee -a "$sweep/console.log") 2>&1
echo "Sweep $sweep  avatar=$avatar_name speech=$speech chunks=$chunks repeats=$repeats"
echo "Mac: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unverified); $(sw_vers -productVersion 2>/dev/null || echo unverified)"
echo "Git: $(git rev-parse --short HEAD 2>/dev/null || echo unverified)"

failed_exports=""
export_variant() {  # NAME then export_dit arguments; support models are refreshed from the baseline export
  local name=$1; shift
  local dir=artifacts/coreml/$name
  if [[ -n "${SKIP_EXPORT:-}" && -d $device/$name ]]; then return 0; fi
  mkdir -p "$dir"
  local model
  for model in audio reference decoder encoder; do
    [[ -d $base/$model.mlpackage ]] || { echo "Missing $base/$model.mlpackage"; return 1; }
    rsync -a --delete "$base/$model.mlpackage/" "$dir/$model.mlpackage/"
  done
  local log="$sweep/export-$name.log"
  echo "export $name: $*  (log $log)"
  if PYTHONPATH=apple "$py" -m flashhead_coreml.export_dit --height 320 --width 176 --output "$dir" --resume "$@" >"$log" 2>&1 &&
     PYTHONPATH=apple "$py" -m flashhead_coreml.bundle --input "$dir" --output "$device/$name" --compile >>"$log" 2>&1; then
    return 0
  fi
  echo "   export FAILED for $name; its arms are skipped (see $log)"
  rm -rf "${device:?}/$name"
  failed_exports="$failed_exports $name"
}

# Bundles. Each is exact in real arithmetic or within the FP16/INT8 contract;
# the gates below check every partition before any timing is trusted.
export_variant pro320-fold --variant fold || true
export_variant pro320-mixedgpu --precision mixed-gpu || true
export_variant pro320-b6 --blocks-per-chunk 6 || true
export_variant pro320-b10 --blocks-per-chunk 10 || true
export_variant pro320-ane --layout ane --precision float16 || true
if [[ -f $fit_dir/fit.json ]]; then
  export_variant pro320-ffnk2-fold --variant ffnk2+fold || true
else
  echo "No FFN fit at $fit_dir/fit.json: skipping pro320-ffnk2-fold"
fi

# Numerical gates: validate_dit NRMSE vs FP32 PyTorch per partition, random
# inputs, requested compute units. Results go to gates.jsonl for the summary.
# A cpu-ane gate checks numerics with the Neural Engine requested; it does not
# prove placement.
gate() {  # NAME COMPUTE [BLOCKS_PER_CHUNK]
  local name=$1 compute=$2 blocks=${3:-3}
  [[ -d artifacts/coreml/$name && -d $device/$name ]] || { echo "gate SKIP $name ($compute): not exported"; return 0; }
  local log="$sweep/validate-$name-$compute.log" status=0
  PYTHONPATH=apple "$py" -m flashhead_coreml.validate_dit --checkpoint models/SoulX-FlashHead-1_3B/Model_Pro \
    --models "artifacts/coreml/$name" --height 320 --width 176 --blocks-per-chunk "$blocks" --compute "$compute" \
    >"$log" 2>&1 || status=$?
  printf '{"bundle": "%s", "compute": "%s", "blocks_per_chunk": %s, "exit_status": %s, "log": "%s"}\n' \
    "$name" "$compute" "$blocks" "$status" "$log" >>"$sweep/gates.jsonl"
  if (( status == 0 )); then echo "gate PASS $name ($compute)"; else echo "gate FAIL $name ($compute): see $log"; fi
}
if [[ -z "${SKIP_GATES:-}" ]]; then
  gate pro320-fold cpu;        gate pro320-fold all
  gate pro320-mixedgpu cpu-gpu; gate pro320-mixedgpu all
  gate pro320-b6 cpu 6;        gate pro320-b6 all 6
  gate pro320-b10 cpu 10;      gate pro320-b10 all 10
  gate pro320-ane cpu-ane;     gate pro320-ane cpu-gpu;   gate pro320-ane all
  gate pro320-ffnk2-fold cpu;  gate pro320-ffnk2-fold all
else
  echo "SKIP_GATES set: timings below are not numerically gated"
fi

common=(--portrait "$avatar/portrait.png" --reference-cache "$avatar/reference-cache/metadata.json"
        --audio "$speech" --chunks "$chunks" --compute all --save-frames --model-timings)
# name|bundle|runtime flags   (macOS /bin/bash 3.2 compatible: no associative arrays)
fast="--prefetch-audio --prefetch-blocks"
specs=(
  "baseline|pro320|$fast"
  "resident|pro320|$fast --keep-models-resident"
  "resident_fastpred|pro320|$fast --keep-models-resident --fast-prediction"
  "resident_lowprec|pro320|$fast --keep-models-resident --gpu-low-precision"
  "fold_resident|pro320-fold|$fast --keep-models-resident"
  "mixedgpu_resident|pro320-mixedgpu|$fast --keep-models-resident"
  "b6_resident|pro320-b6|$fast --keep-models-resident"
  "b10_resident|pro320-b10|$fast --keep-models-resident"
  "ane_on_ane|pro320-ane|$fast --keep-models-resident --dit-compute cpu-ane"
  "ane_on_all|pro320-ane|$fast --keep-models-resident --dit-compute all"
  "resident_parallel|pro320|$fast --keep-models-resident --parallel-frames"
  "exact_stack|pro320-fold|$fast --keep-models-resident --parallel-frames"
  "exact_stack_fastpred|pro320-fold|$fast --keep-models-resident --parallel-frames --fast-prediction"
  "ane_stack|pro320-ane|$fast --keep-models-resident --parallel-frames --dit-compute cpu-ane"
  "ffnk2fold_resident|pro320-ffnk2-fold|$fast --keep-models-resident"
)

run_spec() {  # SPEC REPEAT
  local arm bundle flags
  IFS='|' read -r arm bundle flags <<<"$1"
  if [[ -n "${ARMS:-}" && " $ARMS " != *" $arm "* && $arm != baseline ]]; then return 0; fi
  [[ -d $device/$bundle ]] || { echo "== $arm: SKIP ($device/$bundle missing)"; return 0; }
  local out="$sweep/$arm-r$2"
  echo "== $arm (repeat $2): $bundle $flags"
  # shellcheck disable=SC2086
  apple/run_mac_smoke.sh --models "$device/$bundle" "${common[@]}" $flags --output "$out" \
    >"$out.log" 2>&1 || echo "   FAILED (see $out.log)"
}

# Alternate the arm order between repeats so thermal drift and cache warm-up
# do not always favour the same arms.
count=${#specs[@]}
for repeat in $(seq 1 "$repeats"); do
  if (( repeat % 2 == 1 )); then
    i=0; while (( i < count )); do run_spec "${specs[$i]}" "$repeat"; i=$((i + 1)); done
  else
    i=$((count - 1)); while (( i >= 0 )); do run_spec "${specs[$i]}" "$repeat"; i=$((i - 1)); done
  fi
done

[[ -z $failed_exports ]] || echo "Exports that failed:$failed_exports"
PYTHONPATH=apple "$py" -m flashhead_coreml.mac_sweep_summary --sweep "$sweep" --baseline baseline | tee "$sweep/summary.txt"
echo "Sweep folder: $sweep"
