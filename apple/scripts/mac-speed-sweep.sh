#!/usr/bin/env bash
# macOS only. Export the candidate PRO320 bundles and benchmark every speed
# arm with the real Swift runtime and Core ML, then print one summary table
# (warm FPS, speed-up, pixel identity and lip/teeth metrics vs the baseline).
#
# Prerequisites (repository root):
#   make -f apple/Makefile setup-cpu-ab fetch
#   make -f apple/Makefile export bundle          # baseline: artifacts/coreml/{pro320,device/pro320}
#   .venv-coreml/bin/python apple/scripts/make_cpu_ab_fixtures.py
# Usage:
#   apple/scripts/mac-speed-sweep.sh [SPEECH.wav] [CHUNKS] [REPEATS]
# Environment: AVATAR (girl|indian), ARMS (space-separated subset of arm names),
#              SKIP_EXPORT=1 to reuse already-exported bundles.
set -euo pipefail
speech=${1:-artifacts/cpu-validation/inputs/eval_english_plosive.wav}
chunks=${2:-4}
repeats=${3:-2}
avatar_name=${AVATAR:-girl}
py=.venv-coreml/bin/python
base=artifacts/coreml/pro320
device=artifacts/coreml/device
sweep=artifacts/coreml/sweep/$(date -u +%Y%m%dT%H%M%SZ)
avatar=ios/SoulXStudio/Resources/Avatars/$avatar_name
mkdir -p "$sweep"
[[ "$(uname)" == Darwin ]] || { echo "Run this on the Mac (Core ML prediction is macOS-only)."; exit 1; }
[[ -d $device/pro320 ]] || { echo "Missing $device/pro320: run 'make -f apple/Makefile export bundle' first."; exit 1; }

export_variant() {  # NAME then export_dit arguments; support models are copied from the baseline export
  local name=$1; shift
  local dir=artifacts/coreml/$name
  [[ -n "${SKIP_EXPORT:-}" && -d $device/$name ]] && return 0
  mkdir -p "$dir"
  for model in audio reference decoder encoder; do
    [[ -e $dir/$model.mlpackage ]] || cp -R "$base/$model.mlpackage" "$dir/"
  done
  PYTHONPATH=apple "$py" -m flashhead_coreml.export_dit --height 320 --width 176 --output "$dir" --resume "$@"
  PYTHONPATH=apple "$py" -m flashhead_coreml.bundle --input "$dir" --output "$device/$name" --compile
}

# Bundles. Each is exact in real arithmetic or within the FP16/INT8 contract;
# numerical gates are run below before any timing is trusted.
export_variant pro320-fold --variant fold
export_variant pro320-mixedgpu --precision mixed-gpu
export_variant pro320-b6 --blocks-per-chunk 6
export_variant pro320-b10 --blocks-per-chunk 10
export_variant pro320-ane --layout ane --precision float16
if [[ -f artifacts/cpu-validation/ffn-fit/fit.json ]]; then
  export_variant pro320-ffnk2-fold --variant ffnk2+fold
fi

# Numerical gates (NRMSE vs FP32 PyTorch per partition; requested compute units in the report).
validate() {  # NAME COMPUTE [BLOCKS]
  PYTHONPATH=apple "$py" -m flashhead_coreml.validate_dit --checkpoint models/SoulX-FlashHead-1_3B/Model_Pro \
    --models "artifacts/coreml/$1" --height 320 --width 176 --blocks-per-chunk "${3:-3}" --compute "$2" \
    >"$sweep/validate-$1-$2.log" 2>&1 && echo "gate PASS $1 ($2)" || echo "gate FAIL $1 ($2): see $sweep/validate-$1-$2.log"
}
validate pro320-fold cpu
validate pro320-mixedgpu cpu-gpu
validate pro320-ane cpu-ane
validate pro320-ane cpu-gpu

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
)
[[ -d $device/pro320-ffnk2-fold ]] && specs+=("ffnk2fold_resident|pro320-ffnk2-fold|$fast --keep-models-resident")

for repeat in $(seq 1 "$repeats"); do
  for spec in "${specs[@]}"; do
    IFS='|' read -r arm bundle flags <<<"$spec"
    if [[ -n "${ARMS:-}" && " $ARMS " != *" $arm "* && $arm != baseline ]]; then continue; fi
    out="$sweep/$arm-r$repeat"
    echo "== $arm (repeat $repeat): $bundle $flags"
    # shellcheck disable=SC2086
    apple/run_mac_smoke.sh --models "$device/$bundle" "${common[@]}" $flags --output "$out" \
      >"$out.log" 2>&1 || echo "   FAILED (see $out.log)"
  done
done

PYTHONPATH=apple "$py" -m flashhead_coreml.mac_sweep_summary --sweep "$sweep" --baseline baseline | tee "$sweep/summary.txt"
echo "Sweep folder: $sweep"
