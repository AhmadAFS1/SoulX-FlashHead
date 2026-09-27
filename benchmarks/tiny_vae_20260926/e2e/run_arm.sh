#!/bin/bash
# usage: POLICY=<policy.json> [TRT_PATH=<dir>] [FIXTURE=<id>] [SEED=<n>] run_arm.sh <run-name> [extra run.py flags]
#
# 576x320 end-to-end arm for the tiny-VAE study (2026-09-26). Modelled on the scratchpad
# hires_span.sh, with two deliberate differences:
#  * it does NOT force --overlap-skip / --compile-vae-encode / --force-encode-compile: the
#    control passes them explicitly, tiny-decoder arms must not (run.py refuses them);
#  * PYTHONPATH: the SM89 SageAttention-2 build (/workspace/experiments/pro30-deps) and the
#    TensorRT 10.9 dir (/workspace/experiments/ojin-components-deps) no longer exist. TensorRT
#    10.9.0.34 was restored runtime-only into .restored-deps-20260926/trt-10.9.0.34 (see its
#    PROVENANCE.txt); SageAttention SM89 could not be rebuilt, so every arm uses the flash2
#    policies in benchmarks/tiny_vae_20260926/policies/ (FlashAttention-2 self-attention).
# Fixed shipping-DiT flags: 2 steps, distilled_aligned, static INT8 ffn.2 scales, DiT CUDA graph,
# fused INT8 FFN, lean delivery, --save-raw (raw.npy for the colour-drift gate).
# Output under benchmarks/pro_30fps_20260922/<run-name> so gate/video scripts find it.
cd /workspace/SoulX-FlashHead || exit 1
unset PYTORCH_CUDA_ALLOC_CONF
TRT=${TRT_PATH:-.restored-deps-20260926/trt-10.9.0.34}
export PYTHONPATH=$TRT:.pro-quant-deps:.
export SOULX_STAGE_REUSE_OUTPUTS=${SOULX_STAGE_REUSE_OUTPUTS:-1}
OUT=benchmarks/pro_30fps_20260922
LOGDIR=benchmarks/tiny_vae_20260926/e2e/logs
mkdir -p "$LOGDIR"
POLICY=${POLICY:?set POLICY}
NAME="$1"; shift
rm -rf "$OUT/$NAME"
.venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  --policy "$POLICY" \
  --fixtures benchmarks/pro_30fps_20260920/tts-fixtures/fixtures.json \
  --fixture-id "${FIXTURE:-indian150-a}" --seed "${SEED:-50}" --frames 250 --repeats 1 \
  --sampling-steps 2 --timestep-variant distilled_aligned --latency-detail off \
  --static-int8-scales benchmarks/pro_30fps_20260921/int8-amax.json \
  --save-raw --dit-cuda-graph --fused-int8-ffn --lean-delivery "$@" \
  --output "$OUT/$NAME" > "$LOGDIR/$NAME.log" 2>&1
rc=$?; [ $rc -ne 0 ] && { echo "[FAIL rc=$rc] $NAME"; grep -v "Warning\|warn" "$LOGDIR/$NAME.log" | tail -12; exit $rc; }
.venv/bin/python -c "
import json;d=json.load(open('$OUT/$NAME/results.json'));r=d['runs'][0]
rs=json.load(open('$OUT/$NAME/resources-0.json'))
print('$NAME','fps=%.4f'%r['useful_fps'],{k:round(v,3) for k,v in r['stage_seconds'].items()},
      'vram=%d'%max(x['gpu_vram_used_mib'] for x in rs),'reserved=%d'%max(x['torch_reserved_mib'] for x in rs),
      'peak_alloc=%d'%r['peak_allocated_mib'],'window_p50=%.3f'%r['chunk_distribution_s']['p50'],'raw_sha=%s'%d['raw_rgb_sha256'][:16])"
