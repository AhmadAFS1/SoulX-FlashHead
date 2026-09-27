#!/bin/bash
# usage: gate_arm.sh <run-name> [<label>]
# Same review as scratchpad gate576.sh (review.py against the 576x320 reference s30-reuse, 19.62 FPS),
# but the colour-drift step reads this study's same-session control raw.npy (tae-ctl-r01) as its
# "control" column, because s30-reuse/raw.npy (and every other raw.npy) was deleted from disk.
# colour_drift.py reports each file's per-window mean-RGB drift against ITS OWN window 0, so the
# candidate's drift figure does not depend on which control is passed.
cd /workspace/SoulX-FlashHead || exit 1
SP=/tmp/claude-0/-workspace/5a04e616-f8a6-4bae-9052-d4d650c8f46c/scratchpad
LOGDIR=benchmarks/tiny_vae_20260926/e2e/logs
NAME="$1"; LABEL="${2:-$1}"
REF=benchmarks/pro_30fps_20260921/s30-reuse
CTL=benchmarks/pro_30fps_20260922/${CTL_RUN:-tae-ctl-r01}
rm -rf "benchmarks/pro_30fps_20260922/review-$NAME"
.venv/bin/python benchmarks/pro_quantization_v2_20260918/review.py --baseline "$REF" --candidate "benchmarks/pro_30fps_20260922/$NAME" \
  --baseline-label reference-576x320-19.6fps --candidate-label "$LABEL" --output "benchmarks/pro_30fps_20260922/review-$NAME" > "$LOGDIR/review-$NAME.log" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "[review FAIL rc=$rc]"; tail -5 "$LOGDIR/review-$NAME.log"; exit $rc; }
.venv/bin/python -c "
import json; r=json.load(open('benchmarks/pro_30fps_20260922/review-$NAME/review.json')); p=r['paired']
print('$NAME review: corr=%.3f edge_ratio=%.3f mouth_dist=%.1fpx open_pairs=%d' % (p['opening_correlation'], p['median_edge_ratio_on_both_open'], p['median_mouth_center_distance_px'], p['both_open_pairs']))"
.venv/bin/python $SP/colour_drift.py "$CTL/raw.npy" "benchmarks/pro_30fps_20260922/$NAME/raw.npy" > "$LOGDIR/drift-$NAME.log" 2>&1
grep "slope" "$LOGDIR/drift-$NAME.log" | sed 's/^/  drift /'
