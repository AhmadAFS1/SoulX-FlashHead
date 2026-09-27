#!/bin/bash
# Dump latents for several fixtures, each after a sustained-quiet GPU check, retrying runs that fail (e.g. another tenant OOM).
BK=/workspace/SoulX-FlashHead/benchmarks/tiny_vae_20260926/decoder_bakeoff
for FX in "$@"; do
  for attempt in 1 2 3 4; do
    bash $BK/wait_quiet.sh 3 || exit 1
    line=$(bash $BK/dump_latents.sh "$FX" 50 | tail -1); echo "[$(date +%H:%M:%S)] attempt $attempt: $line"
    n=$(ls /workspace/SoulX-FlashHead/benchmarks/tiny_vae_20260926/latents/$FX-s50-w*.pt 2>/dev/null | wc -l)
    [ "$n" -ge 9 ] && break
    grep -h "OutOfMemory\|Error" $BK/logs/dump-$FX-s50.log | tail -2
  done
done
