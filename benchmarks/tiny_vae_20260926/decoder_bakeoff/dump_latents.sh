#!/bin/bash
# usage: dump_latents.sh <fixture-id> <seed>
# Variant of scratchpad/hires_dump.sh for the tiny-decoder bake-off (2026-09-26).
# Why a variant: the TensorRT 10.9 bindings and the SM89 SageAttention-2 build that
# hires_dump.sh's policy (hires_normconv.json: trt_stage_compile decoder + sage2) needs lived in
# /workspace/experiments/pro30-deps and /workspace/experiments/ojin-components-deps, which no
# longer exist on this machine. This run therefore uses policies/dump_flash2_eagerdec.json:
# identical DiT quantisation (fp8 projections, int8 FFN, static scales, compiled DiT), FlashAttention-2
# self-attention, and the STOCK Wan 2.1 decoder in eager bf16 inside the loop (no overlap-skip, no
# decoder rewrites), so every window is decode(9 latents) -> 33 frames -> colour-correct -> re-encode.
# Right after the run it keeps ONLY each measured decode call's latent (args[0], (16,9,72,40) bf16)
# plus a tiny strided checksum of the harness decode, and deletes the big dump.
cd /workspace/SoulX-FlashHead || exit 1
unset PYTORCH_CUDA_ALLOC_CONF
export PYTHONPATH=/workspace/experiments/pro30-deps/sage-sm89-stream:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.
FX="$1"; SEED="$2"; NAME="dump-$FX-s$SEED"
BK=benchmarks/tiny_vae_20260926/decoder_bakeoff
TMP=$BK/_dump_tmp/$NAME
RUN=$BK/_dump_tmp/$NAME-run
LOG=$BK/logs/$NAME.log
mkdir -p $BK/logs; rm -rf "$TMP" "$RUN"
SOULX_DUMP_DIR="$TMP" .venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  --policy $BK/policies/dump_flash2_eagerdec.json \
  --fixtures benchmarks/pro_30fps_20260920/tts-fixtures/fixtures.json \
  --fixture-id "$FX" --seed "$SEED" --frames 250 --repeats 1 \
  --sampling-steps 2 --timestep-variant distilled_aligned --latency-detail off \
  --static-int8-scales benchmarks/pro_30fps_20260921/int8-amax.json \
  --compile-vae-encode \
  --output "$RUN" > "$LOG" 2>&1
rc=$?
rm -f $TMP/forward-*.pt $TMP/encode-*.pt $TMP/audio-*.pt
.venv/bin/python $BK/strip_dump.py --dump "$TMP" --run "$RUN" --fixture "$FX" --seed "$SEED" --rc $rc \
  --out benchmarks/tiny_vae_20260926/latents
src=$?
# keep only the run's small records; drop the dump and the big outputs
mkdir -p $BK/dump_runs/$NAME
cp -f $RUN/results.json $BK/dump_runs/$NAME/ 2>/dev/null
rm -rf "$TMP" "$RUN"
echo "$NAME rc=$rc strip=$src latents: $(ls benchmarks/tiny_vae_20260926/latents/$FX-s$SEED-w*.pt 2>/dev/null | wc -l) free=$(df -h /workspace | tail -1 | awk '{print $4}')"
