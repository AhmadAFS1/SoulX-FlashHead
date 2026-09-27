#!/bin/bash
# usage: review_vs_ctl.sh <run-name> [<label>]
# Paired review of an arm against the SAME-SESSION control tae-ctl-r01 (same DiT, attention, seed,
# fixture; only the VAE differs), isolating the tiny VAE's effect from the attention-backend change.
cd /workspace/SoulX-FlashHead || exit 1
LOGDIR=benchmarks/tiny_vae_20260926/e2e/logs
NAME="$1"; LABEL="${2:-$1}"
CTL=benchmarks/pro_30fps_20260922/${CTL_RUN:-tae-ctl-r01}
OUTDIR=benchmarks/pro_30fps_20260922/review-$NAME-vs-ctl
rm -rf "$OUTDIR"
.venv/bin/python benchmarks/pro_quantization_v2_20260918/review.py --baseline "$CTL" --candidate "benchmarks/pro_30fps_20260922/$NAME" \
  --baseline-label shipping-ctl-flash2 --candidate-label "$LABEL" --output "$OUTDIR" > "$LOGDIR/review-$NAME-vs-ctl.log" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "[review FAIL rc=$rc]"; tail -5 "$LOGDIR/review-$NAME-vs-ctl.log"; exit $rc; }
.venv/bin/python -c "
import json; r=json.load(open('$OUTDIR/review.json')); p=r['paired']
print('$NAME vs ctl: corr=%.3f edge_ratio=%.3f mouth_dist=%.1fpx open_pairs=%d' % (p['opening_correlation'], p['median_edge_ratio_on_both_open'], p['median_mouth_center_distance_px'], p['both_open_pairs']))"
