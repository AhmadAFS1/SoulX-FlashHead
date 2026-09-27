#!/bin/bash
# Second/third fixture check (2026-09-26): for each fixture, the same-session control (shipping VAE,
# flash2) and the two tiny arms, then a paired review of each tiny arm against ITS fixture's control
# (no s30-reuse-style reference exists for these fixtures) and colour drift. raw.npy files are
# deleted as soon as drift is computed (disk).
cd /workspace/SoulX-FlashHead || exit 1
E=benchmarks/tiny_vae_20260926/e2e
SP=/tmp/claude-0/-workspace/5a04e616-f8a6-4bae-9052-d4d650c8f46c/scratchpad
OUT=benchmarks/pro_30fps_20260922
SHIP="--vae-weights benchmarks/pro_30fps_20260922/distill/vae_skip9-10-13-14_ft4.pth --skip-decoder-blocks 9 10 13 14 --overlap-skip --compile-vae-encode --force-encode-compile"
for FX in ${FIXTURES:-tts-plosives tts-open-vowels}; do
  C=tae-ctl-$FX-r01; T=tae-decenc-trt-$FX-r01; L=tae-light-decenc-$FX-r01
  bash $SP/wait_lease.sh >/dev/null && FIXTURE=$FX POLICY=benchmarks/tiny_vae_20260926/policies/ctl_final_v4_flash2.json bash $E/run_arm.sh $C $SHIP
  bash $SP/wait_lease.sh >/dev/null && FIXTURE=$FX TRT_PATH=benchmarks/tiny_vae_20260926/decoder_bakeoff/_trt10_path \
    POLICY=benchmarks/tiny_vae_20260926/policies/tiny_v4_flash2.json bash $E/run_arm.sh $T \
    --tiny-vae-decoder taew2_1 --tiny-vae-encoder taew2_1 --tiny-vae-backend tensorrt
  bash $SP/wait_lease.sh >/dev/null && FIXTURE=$FX POLICY=benchmarks/tiny_vae_20260926/policies/tiny_v4_flash2.json bash $E/run_arm.sh $L \
    --tiny-vae-decoder lighttaew2_1 --tiny-vae-encoder lighttaew2_1 --tiny-vae-backend compile
  for A in $T $L; do
    CTL_RUN=$C bash $E/review_vs_ctl.sh $A
    .venv/bin/python $SP/colour_drift.py "$OUT/$C/raw.npy" "$OUT/$A/raw.npy" > $E/logs/drift-$A.log 2>&1
    grep slope $E/logs/drift-$A.log | sed "s/^/  drift $A /"
    rm -f "$OUT/$A/raw.npy"
  done
  rm -f "$OUT/$C/raw.npy"
  df -h /workspace | tail -1
done
echo ROBUSTNESS_DONE
