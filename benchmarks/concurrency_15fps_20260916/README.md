# SoulX 15-FPS simultaneous-stream capacity — September 16, 2026

Quality follow-up: the user reported damaged teeth in this run's recorded avatar. Transport success does not certify teeth quality. A new [closer 1.25× Indian-male five-stream recording](../closer_capacity_20260916/README.md) is available, along with [quantization/TensorRT feasibility](../../docs/research/QUANTIZATION_TENSORRT_2026-09-16.md). It uses a different source and still idle anchor; zoom alone is not isolated.

This is fresh local GPU inference and real local WebRTC receiver evidence on an **NVIDIA GeForce RTX 4070 SUPER with 12,282 MiB visible VRAM**, driver **595.84**, Torch **2.7.1+cu128 / CUDA 12.8**. A co-resident OmniVoice process used about **2,478 MiB** before SoulX started and was left running. The selected SoulX profile uses a 7,168-MiB Torch allocator cap; that cap is not physical VRAM. Exact environment metadata is in [`environment.json`](environment.json), and 200-ms device samples are in [`gpu-samples.csv`](gpu-samples.csv).

## Result

**Five simultaneous active 15-FPS streams are the measured video-capacity ceiling for this machine and profile. Six fail.** This required adding an experimental GPU batch size of five. The five-stream one-turn run, 15-turn soak, and recorded validation all had zero video underrun/held frames, approximately 15.00 wire FPS per receiver, H264 verification, monotonic A/V PTS, zero packet loss, and cleanup success. Six active streams accumulated 132 repeated/underrun frames in one turn per peer.

This is a narrow laboratory ceiling, not a production admission target: batch-five useful generation averaged about **77.36 FPS** in retained recent-chunk telemetry against **75 FPS** continuous demand, only ~3.1% raw generation margin. The five-stream soak also recorded one 20-ms audio hold on three of five peers (2,880 samples total), while video stayed clean. Use **four** as the conservative operational limit until longer remote/TURN and mixed-turn testing passes; five is the demonstrated maximum for video cadence.

## Selected sweep

All rows use 320×576, four denoising steps, 15 FPS, source-video idle, shared-memory RGB transport, H264 `veryfast`, the same ten-second audio, and all connected peers speaking concurrently. They use the stock server audio conditioning; the earlier process-local mouth-strength-0.5 experiment is not installed in this serving path. That scalar preserves tensor shapes and is unlikely to alter the capacity boundary materially, but this run does not prove an exact strength-0.5 service result. Wire FPS includes held frames, so the underrun/held counters decide whether the neural renderer actually kept up.

| Profile | Active streams | Turns | Wall | Min–max wire FPS | Video underrun / held | Audio hold | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BF16, max batch 4 | 4 | 4 | 14.68 s | 15.000–15.001 | 0 / 0 | 0 | Pass |
| BF16, max batch 4 soak | 4 | 12 | 39.80 s | 15.001–15.001 | 0 / 0 | 0 | Pass |
| BF16, max batch 4 | 5 | 5 | 15.09 s | 14.996–15.001 | 21 / 21 | 65,280 samples | Fail |
| BF16, compiled audio/color, max batch 4 | 5 | 5 | 14.88 s | 15.003–15.003 | 23 / 23 | 72,000 samples | Fail |
| BF16, **max batch 5** | **5** | **5** | **14.81 s** | **15.003–15.006** | **0 / 0** | **0** | **Pass** |
| BF16, **max batch 5 soak** | **5** | **15** | **41.15 s** | **15.000–15.001** | **0 / 0** | **2,880 samples** | **Video pass; minor audio holds** |
| BF16, max batch 5, recorded | 5 | 5 | 14.68 s | 15.000–15.002 | 0 / 0 | 1,920 samples | Video/transport pass |
| BF16, max batch 5 | 6 | 6 | 16.72 s | 14.998–15.005 | 132 / 132 | 418,560 samples | **Fail** |

The BF16 batch-five chunks rendered 120 useful frames in about 1.55 seconds, or about 77.3 useful FPS. The run-wide GPU sampler observed 2,487–8,895 MiB total device memory, up to 100% GPU utilization, 183.2 W instantaneous power, and 66 °C. Those ranges cover the complete experiment and the co-resident process; they are not isolated per-run SoulX allocations. Retained Torch telemetry peaked at 6,148 MiB reserved in the batch-five profile.

## Memory-saving control

The INT8-storage/BF16-compute profile used a 3,584-MiB allocator cap. It passed one stream, delivered two without video underruns but with one 20-ms audio hold, and failed at three (54 repeated frames) and four (266). INT8 here compresses stored weights; it does not accelerate the matrix multiplications. This makes it a memory-availability profile, not the best throughput profile.

## Recorded output and integrity checks

[`bf16-batch5-c5-recorded-peer0.mp4`](bf16-batch5-c5-recorded-peer0.mp4) is the receiver recording from one peer while five spoke. `ffprobe` reports H264 320×576 at 15/1 FPS plus 48-kHz stereo AAC, duration 15.534 seconds. The reproducible waveform check found the ten-second source at 1.8065 seconds with correlation 0.9624; this verifies audio correspondence, not perceptual lip-sync. See [`bf16-batch5-c5-recorded-audio-check.json`](bf16-batch5-c5-recorded-audio-check.json) and the eight receiver snapshots.

The JSON artifacts retain server health/configuration, independent peer counters, WebRTC stats, transport checks, exact source hashes, recent GPU chunk telemetry, and cleanup results. Compressed timestamp companions contain full arrival/PTS arrays.

## Interpretation

Batch four cannot serve five synchronized speakers because the scheduler renders a four-state chunk and then a separate state, pushing useful generation below the 75-FPS demand. A single batch of five amortizes the four DiT passes across all five session states and reaches roughly 77 FPS. Six necessarily returns to a five-plus-one schedule and fails. Audio conditioning and VAE decode/feedback still scale per session, which is why batch five buys only enough headroom for the fifth stream rather than a larger jump.

The test keeps independent recurrence, audio and RNG state per session; it does not merge users into one identity. It also does not establish a long-duration production SLA, remote network behavior, perceptual quality for every peer, or capacity with generated neural idle. The accepted four-step 320×576 quality profile was retained; the earlier corrupted 256²/two-step shortcut was not used.
