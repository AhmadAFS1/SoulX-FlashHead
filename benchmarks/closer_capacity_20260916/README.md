# Closer Indian male under five-stream load

September 17 quality validation: [comparison against both earlier 25-FPS 1.25× clips](TEETH_VALIDATION_20260917.md) supports poorer dental definition in several sampled moments of this receiver recording. The 650-frame CPU analysis and aligned contact sheet do not isolate FPS as the cause; precision, conditioning, recurrence and codec paths also differ. Teeth quality remains unresolved despite successful video cadence.

September 16, 2026: fresh inference on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM**, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8. Direct pre-run `nvidia-smi` showed 2,487 MiB device use, with OmniVoice still resident. Same 7,168-MiB Torch allocator cap, 320×576, four steps, BF16 resident weights, compiled optimized real-RoPE, lean/fused-QKV, batch five, 15 FPS, shared-memory transport and H264 `veryfast` as the prior BF16 capacity test. Audio conditioning is the stock server path.

Watch [the new closer-face receiver video](closer125-c5-peer0.mp4). Five active peers each completed one ten-second turn in a 14.81-second benchmark wall interval, with **zero held/underrun video frames**. Receiver wire FPS ranged 15.004–15.008; maximum arrival gap was 80.74 ms. One peer recorded one 20-ms audio hold. All transport checks and cleanup passed. Full decode completed without errors; source-audio correlation is 0.9625. Detailed evidence: [run JSON](closer125-c5.json), compressed timestamps and [audio diagnostic](audio-check.json).

The reference is the exact preferred [Indian male 1.25× image](../distance_lipsync/evidence-indian-male-closer-20260916/reference-closer-125.png). `closer-125-idle.mp4` is a two-second still loop of that image, created using FFmpeg at 15 FPS, H264 CRF 10. Using it as the idle source also makes it the actual GPU reference and reconditioning anchor. This changes both identity/source asset and idle motion relative to the original capacity video, so it is a useful requested preview, not a controlled attribution of teeth improvement to zoom alone. Sparse inspection at six seconds shows visible teeth; it does not establish artifact-free dentition throughout the video.

## Reproduce

From the repository root, start the existing server with:

```bash
.venv/bin/python -m soulx_rtc.server --host 127.0.0.1 --port 8765 \
  --width 320 --height 576 --steps 4 --fps 15 --optimized --real-rope \
  --lean --fused-qkv --memory-mode compact --cuda-memory-mib 7168 \
  --idle-video benchmarks/closer_capacity_20260916/closer-125-idle.mp4 \
  --idle-policy source --idle-cache-mib 512 --chunk-transport shm \
  --freeze-startup-gc --max-active-calls 5 --batch 5 --max-sessions 5 \
  --h264-preset veryfast --profile
```

After health reports ready, use `python -m soulx_rtc.benchmark_calls` from the same venv with `--url ws://127.0.0.1:8765 --sessions 5 --speakers 5 --turns 1 --audio benchmarks/comparison-10s.wav --audio-seconds 10 --gap 0 --idle-seconds 1 --timeout 180 --record --record-peers 0 --snapshots-every 2 --compact-evidence --output NEW_PATH.json`.

The test server was stopped after collection. [Quantization and TensorRT feasibility](../../docs/research/QUANTIZATION_TENSORRT_2026-09-16.md) explains the next optimization candidates. Neither quantization nor a new TensorRT engine was enabled for this closer-face recording.
