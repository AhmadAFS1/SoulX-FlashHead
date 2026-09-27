#!/bin/bash
# Block until the SoulX lease is free and the GPU has been quiet (<1500 MiB, <10% util) for N consecutive 10 s checks.
N=${1:-3}; LOCK=/workspace/SoulX-FlashHead/.gpu-owner.lock; ok=0
for i in $(seq 1 1080); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
  if flock -n "$LOCK" -c true 2>/dev/null && [ "$used" -lt 1500 ] && [ "$util" -lt 10 ]; then ok=$((ok+1)); else ok=0; fi
  [ $ok -ge $N ] && { echo "quiet for $N checks (used=${used}MiB util=${util}%) after $((i*10)) s"; exit 0; }
  sleep 10
done
echo TIMEOUT; exit 1
