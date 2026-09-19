# SoulX concurrency potential versus the MuseTalk evidence

September 16, 2026 update: fresh GPU inference and local WebRTC load testing established a five-stream 15-FPS video-cadence ceiling on **NVIDIA GeForce RTX 4070 SUPER, 12 GB class / 12,282 MiB visible**, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8. A co-resident OmniVoice process occupied about 2,478 MiB before SoulX and was left running. The selected profile peaked at 6,148 MiB Torch-reserved; the whole-device sampler peaked at 8,895 MiB including co-residency. Historical **RTX 4070 (non-SUPER)** and other-GPU results remain separate.

## Bottom line

SoulX now demonstrates **five simultaneous active 15-FPS video streams** at 320×576 and four denoising steps when all five states share one GPU batch. The one-turn run, 15-turn soak and recorded validation had zero video underrun/held frames; six streams failed with 132 repeats. The five-stream soak did record three isolated 20-ms audio holds, and measured batch-five generation was only about 77.36 FPS against 75 FPS demand. Therefore **five is the laboratory video ceiling and four is the conservative operational limit**, not a production five-user SLA. Many connected idle users still must not be confused with actively generated speakers.

The previous strength-0.5 offline clips measured 35.2–35.5 FPS under INT8 weight storage. Direct serving tests explain most of the discrepancy: the resident-BF16 profile reaches roughly 73 FPS at batch one and 77 FPS at batch five, while the memory-saving INT8-storage profile reaches only about 38–40 aggregate FPS. The WebRTC server used stock audio conditioning; the process-local 0.5 hook is not installed there. Its scalar does not change tensor shapes, but an exact strength-0.5 serving boundary remains unmeasured. Eight or ten continuously speaking avatars per GPU remain unestablished at accepted quality.

## Direct 15-FPS capacity result

Exact run report and artifacts: [September 16 capacity evidence](../../benchmarks/concurrency_15fps_20260916/README.md). This was real local GPU inference plus WebRTC sender/receiver work, not CPU arithmetic or an upstream claim.

| GPU/profile | Active streams | Evidence | Result |
| --- | ---: | --- | --- |
| RTX 4070 SUPER, 12,282 MiB; BF16 resident weights, batch 4 | 4 | 12-turn, 39.80-s soak | Zero video underruns and zero audio holds |
| Same GPU/profile, batch 4 | 5 | One turn per peer | 21 video underruns; fails |
| Same GPU/profile, compiled audio/color, batch 4 | 5 | One turn per peer | 23 video underruns; fails; numerical-output experiment not promoted |
| Same GPU/profile, **batch 5** | **5** | 15-turn, 41.15-s soak | Zero video underruns; 2,880 held audio samples total (three 20-ms events) |
| Same GPU/profile, batch 5 | 6 | One turn per peer | 132 video underruns and 418,560 held audio samples; fails |

All rows used 320×576, four steps, 15 FPS, optimized real-RoPE, lean/fused-QKV, compact resident memory, source-video idle, shared-memory chunk transport, H264 `veryfast`, and stock server audio conditioning. The 7,168-MiB allocator cap does not change the physical GPU. The service now accepts experimental `--batch 5`; its prior choices stopped at four. Five works because one 120-frame GPU render averages about 1.55 seconds (~77.3 FPS); a four-plus-one schedule misses the same 75-FPS aggregate deadline. Six again requires multiple scheduling waves and fails.

MuseTalk has broader multi-GPU testing and mature prepared-avatar batching. Its own evidence also rejects equating completion with smoothness: a result described as supporting six or eight calls can still run far below 20 FPS per caller. We should apply one explicit acceptance standard to both models.

## Capacity definitions and arithmetic

- **Connected calls:** signaling/media state exists; may show a cached idle or held frame.
- **Active speech renderers:** independent speech-conditioned frames must be generated on schedule.
- **Generated idle:** also consumes neural compute even without speech. Source-video replay does not consume equivalent model compute, but still costs media/encoding resources.
- **Wire FPS:** decoded/transmitted frames, potentially including held or idle frames. It is not necessarily useful generated FPS.
- **Engine FPS:** useful new frames divided by unpaced generation time, excluding loading/warmup as stated. It is not automatically a WebRTC/SLA capacity measurement.

For continuously active streams, demand is `N × target_fps`. At 25 FPS, 2/4/8/10 speakers need 50/100/200/250 **useful generated FPS** before overhead. For an illustrative 80% utilization budget, required capacity is `N × target_fps / 0.8`. The 20% reserve is a planning assumption, not a measured tail-latency guarantee.

The following retained arithmetic uses the earlier **INT8-storage offline** 1.25×/strength-0.5 result, **35.317 FPS**, at a different 25-FPS target. It is historical planning context, not the new BF16 15-FPS capacity result:

| Active speakers at 25 FPS | Demand, no reserve | Capacity with 20% reserve | Speedup over 35.317 FPS with reserve |
| --- | ---: | ---: | ---: |
| 1 | 25 | 31.25 | 0.88× |
| 2 | 50 | 62.5 | 1.77× |
| 3 | 75 | 93.75 | 2.65× |
| 4 | 100 | 125 | 3.54× |
| 8 | 200 | 250 | 7.08× |
| 10 | 250 | 312.5 | 8.85× |

These are **required improvements**, not predicted speedups or admissions to configure now. At 20 FPS the nominal frame demand is lower, but changing SoulX's `fps` changes its audio/frame indexing. Generating at 25 and merely dropping transmitted frames saves encoding/bandwidth, not model work. A genuine 20-FPS generation profile needs its own quality and sync tests.

For turn-taking, `connected_calls × speaking_fraction × target_fps` describes average demand only. Ten calls at a 30% speaking fraction average three active speakers, or 75 FPS at 25 FPS. That exceeds the historical INT8-storage result, and synchronized replies create worse bursts. Under an illustrative independent binomial model, the chance of two or more simultaneous speakers is about 85%; real calls need not be independent. Multiplying capacity by inverse duty cycle without queue/tail analysis is unsafe. Continuously generated listening animation removes much of the proposed idle saving.

## What SoulX has actually measured

| Evidence/profile | GPU and provenance | Measured result | What it establishes |
| --- | --- | --- | --- |
| September 16: direct 320×576, four-step, 15-FPS WebRTC ramp; BF16 resident weights, batch 5 | RTX 4070 SUPER, 12,282 MiB; direct `nvidia-smi` samples, driver 595.84, CUDA 12.8; OmniVoice about 2,478 MiB resident | C5: zero video underruns over 15 turns; C6: 132; batch-five recent chunks ~77.36 aggregate useful FPS | Five-stream local video ceiling with minor audio-hold caveat; four-stream conservative operating point |
| September 16: 320×576, four steps, compiled compact INT8 weight storage/BF16 compute, strength 0.5 | RTX 4070 SUPER, 12,282 MiB; direct per-run metadata, driver 595.84, CUDA 12.8; 2,487 MiB pre-load use | 35.2–35.5 useful FPS, three sequential ten-second clips | Current-profile engine baseline; no simultaneous-call test |
| September 11: 320×576, four steps, compact resident profile | RTX 4070, 12,282 MiB; report identifies driver 570.181 host lineage, CUDA 12.8, co-resident OmniVoice 5,312 MiB | 54.06 pooled baseline median; 56.47 with experimental audio/color compilation | Historical documentary evidence, not a matched SUPER comparison; 54.06/25 ≈ 2.16 stream-equivalents with little headroom for two |
| Earlier 512², four steps, ten engine jobs | RTX 4070, 12,282 MiB; historical report, driver 570.181, CUDA 12.8, OmniVoice left resident | Batch 1: 40.66 aggregate FPS; batch 2: 41.05 | Batching two sessions added only about 1%; ten jobs are not ten real-time users |
| Earlier 480×832, four steps, TRT FFNs + VAE, ten jobs | RTX 4070, 12,282 MiB; historical report attribution; tight shared-memory headroom, not an isolated GPU | 28.57 aggregate FPS median | About one speaker's demand; Torch allocator peaks omit TRT-owned memory |
| September 13: 320×576, ten connected / one active | RTX 4070, 12,282 MiB; retrospective host attribution, co-resident service about 6,030 MiB at inspection | About 250 aggregate **wire** FPS; 0 and 2 underruns in selected rounds, arrival-gap failures remain | Improved media delivery; not ten-speaker throughput or strict ten-peer smoothness |

Sources: [current run JSON](../../benchmarks/distance_lipsync/evidence-indian-male-closer-strength-0p5-20260916/results.json), [September 11 profile](PIPELINE_PROFILE_2026-09-11.md), [batch-1 engine JSON](../../benchmarks/engine-512-4-b1.json), [batch-2 JSON](../../benchmarks/engine-512-4-b2.json), [portrait TRT JSON](../../benchmarks/implementation/portrait-trt-ten-jobs.json), [media report](MEDIA_OPTIMIZATION_2026-09-13.md).

The 54-FPS historical report's external raw `profile-320.json` is absent in this workspace. Its summary is retained as **documentary**, not newly revalidated raw data. The discrepancy between 54 FPS on the prior 4070 and 35 FPS on the current SUPER is not a hardware ranking: software/profile, timing, power/clocks, host, and co-residency were not controlled across runs. It needs a matched rerun rather than a guessed explanation.

### Direct simultaneous-call failure evidence

The retained 480×832/four-step TRT WebRTC ramp on **RTX 4070, 12,282 MiB** passed transport/cleanup while accumulating the following underrun slots:

| Simultaneous calls | 1 | 2 | 4 | 6 | 10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Underrun slots | 1 | 134 | 935 | 2,622 | 9,153 |

These counts are workload-specific, not equal-duration rates; the one-call run also exercises more turn operations. Nevertheless, the multi-call rows directly fail smoothness. [Two-call raw evidence](../../benchmarks/implementation/calls-h264-trt-c2.json), [ten-call evidence](../../benchmarks/implementation/calls-h264-trt-c10.json). Earlier 256²/two-step/15-FPS trials delivered ten streams but produced severe facial corruption, so that shortcut is **quality-rejected**, not usable capacity. [Original report](../../benchmarks/REPORT.md).

## What the extensive MuseTalk tests say

Historical tests below target **20 FPS**, not SoulX's usual 25. MuseTalk synthesizes a 256×256 face region and composites it over prepared frames; SoulX synthesizes the whole output. Avatars, canvases, precision, dates, and hosts differ. This is a reconciliation of each deployment's evidence, **not a matched GPU/model leaderboard**.

| Tested GPU/profile | Retained results | Honest capacity interpretation |
| --- | --- | --- |
| Tesla V100-SXM2-32GB, 32,768 MB reported; May 22 PyTorch baseline, CUDA 12.1, driver 570.211.01 | C1 51-ms mean interval; C2 55 ms with 118-ms max; C3 91 ms; C6 only 3/6 completed | One near-target; two borderline, not a pristine hard-SLA pass. Larger later TRT profiles improved completion, not eight smooth streams |
| RTX 3090, 24 GB class; May 29 300-W host, mixed INT8 VAE + FP16 TRT UNet split8 | About 71–72 aggregate cadence-FPS; C4 56-ms mean / 433-ms max; C8 113-ms mean | Approximately three 20-FPS stream-equivalents is a planning inference; the quoted C4/C6/C8 table does not itself validate C3 |
| RTX 4090, 24,564 MiB; May 23 VAE TRT buckets 8/12, driver 565.77 | C4–C8 all complete; about 61–63 aggregate cadence-FPS; C8 130-ms mean / 1.468-s max | Eight completed, not eight smooth. Earlier 8/16 profile failed at higher load; not proof of GPU-wide limits |
| RTX 5000 Ada, about 32 GB reported; June 7 mixed INT8 VAE, PyTorch UNet | Report records three smooth 20-FPS streams; saturated plateau about 70–72 FPS | Documentary support for C3 on that profile; not the missing TRT-UNet profile |
| RTX 6000 Ada, 49,140 MiB; June 8 mixed INT8 VAE + FP16 TRT split8 | C4 51-ms mean / 72-ms max; C8 70-ms mean / 661-ms max; saturated 112–114 aggregate cadence-FPS | Report accepts C4 as near-target smooth; C8–C20 complete but fail 20-FPS smoothness |
| RTX 3090, 24 GB class; July 4 exact16-only UNet + mixed INT8 VAE | C4 61-ms mean; C6 85 ms / 1.398-s max; C8 116 ms / 2.333-s max | The document's “6×20fps practical target” means an operational target, **not demonstrated continuous 20 FPS**; C6 is about 11.8 FPS per stream by interval reciprocal |

Sources: [V100 raw summary](../../../MuseTalk/load_test_webrtc_v100_baseline_pytorch_20_20_batch4_ramp1_6.json), [V100 environment](../../../MuseTalk/docs/v100_webrtc_load_test_2026-05-22.md), [3090 May results](../../../MuseTalk/current_unet_trt_throughput_findings_2026-05-29.md), [4090 source/host context](../../../MuseTalk/current_cross_server_throughput_findings.md), [4090 raw summary](../../../MuseTalk/load_test_webrtc_4090_gpt_moving_avatar_20_20_4_5_6_8streams_8_12_libx264_20260523.json), [5000 Ada findings](../../../MuseTalk/docs/webrtc_load_test_findings_2026-06-07.md), [6000 Ada report](../../../MuseTalk/load_test_webrtc_rtx6000ada_int8_trt_unet_split8_20fps_20260608.md), [July 3090 results](../../../MuseTalk/docs/webrtc_generation_optimization_results_2026-07-03.md).

Driver, full runtime versions, and unrelated GPU load are unverified where these selected source reports do not record them. Reported MB/MiB labels are preserved, not silently treated as identical. Several June/July reports reference raw JSON under absent `tmp/` directories: their tables are documentary evidence. The retained V100/4090 summary JSON was inspected directly. `C / mean_interval` is an approximate cadence proxy used by those reports, not the exact sum of per-stream reciprocal means. The harness averages **completed sessions only**, so it must not be used to claim full-stage capacity for failed ramps. A scheduler `encode=0` in WebRTC means HLS encoding is bypassed, not that aiortc video encoding costs nothing.

## Why scaling differs, and what is already implemented

MuseTalk's [`_run_generation_batch`](../../../MuseTalk/scripts/hls_gpu_scheduler.py) packs audio features and precomputed avatar latents from multiple jobs, runs one UNet forward plus VAE decoding, and dispatches composition. Prepared base videos amortize most non-mouth appearance/motion work. Batch 16 means **frames**, not sixteen simultaneous users.

SoulX's [`Engine.generate`](../../soulx_rtc/engine.py) shares model weights while preserving each session's RNG, audio, reference, and motion history. It batches the four DiT denoising passes across sessions, but currently performs audio extraction and VAE decode/feedback per session. Batch 2 means **two independent 24-new-frame chunks**. Each decoded chunk contains 33 frames, of which nine are context. Its recurrent temporal work is materially different from packing independent MuseTalk face frames.

The [server](../../soulx_rtc/server.py) now exposes batches 1/2/4 plus experimental batch 5, a connected-session limit, and a separate active-call limit. Defaults remain ten connections but **one active call**; selecting batch five does not silently raise admission. The current TRT-FFN path explicitly requires batch one. [`calls.py`](../../soulx_rtc/calls.py) prioritizes speech and low buffered lead, bounds queues, and yields generated-idle work to speech. The worker serializes GPU execution; weights do not need to be duplicated for every call.

Remaining serving constraints: finite jobs and persistent speech lack one unified admission budget; reference preparation shares the GPU; active turns retain admission until playout completes; sequential per-session VAE work limits batching gains; CPU media/encoding and queue tails still matter. These are reasons to measure scheduling, not evidence that more asynchronous tasks or more CUDA streams create missing GPU throughput. Raising `max-sessions` or `max-active-calls` changes policy, not physical capacity.

## Realistic optimization potential

The historical 320×576 profile attributes about **55.2%** of chunk time to four DiT passes, **27.6%** to VAE decoding, and **8.0%** to motion-feedback encoding. Audio and color are each about 3.3%. Those are historical phase intervals, not newly measured kernel shares. Speed work must target DiT/VAE, not expect HTTP tuning alone to deliver 4–10×.

An Amdahl illustration: doubling only the 55.2% DiT region yields `1 / (0.448 + 0.552/2) ≈ 1.38×`, assuming every other cost stays fixed. Even making that region free yields only about 2.23×. This illustrates why a faster transformer kernel does not multiply whole-service capacity by its isolated speedup. Actual profiling must be repeated on the current SUPER profile before predicting gains.

Best next levers are matched batch-1/2/4 measurements; faster VAE/feedback and calibrated compute kernels; removal of avoidable per-session serialization; unified deadline-aware admission; bounded cached idle; and one independent worker per GPU with session affinity for scale-out. Current INT8 **weight storage** still computes in BF16—it is not proof of INT8 GEMM acceleration. Model-parallel multi-GPU latency improvements do not automatically beat independent replicas for total users.

The [upstream README](https://github.com/Soul-AILab/SoulX-FlashHead) advertises **96 FPS and three concurrent 25+ FPS streams on RTX 4090**. Treat this as an external claim, not a local benchmark. The [v1 paper](https://arxiv.org/html/2602.07449v1) names RTX 4090 in its abstract but says the tabulated FPS is measured on one H20 in its evaluation section; the timing/hardware attribution is not fully consistent. Neither source supplies our exact portrait, audio-strength hook, WebRTC tail metrics, or idle policy. Three active 25-FPS users is a sensible 4090 validation target; eight or ten is not supported by these claims.

## Remaining qualification work

1. Freeze the quality fixture using the September 17 [user-selected movement protocol](MOVEMENT_SEED_PROTOCOL_2026-09-17.md): Indian male, 1.25× reference, 320×576, four steps, 25 FPS; preferred seeds **50, 51, 0, 1**, stock strength **1.0**, shift **5**, history **2**. This replaces the earlier proposed strength-0.5 fixture; historical throughput measurements above retain their original settings. Framing changes alone preserve tensor dimensions, so they are not a meaningful throughput lever. Keep head-motion interventions separate until quality/latency-tested.
2. Repeat the batch-five boundary in an isolated maintenance window and for 30–60 minutes. The present direct test retained the real co-resident OmniVoice load and sampled clocks/power/memory, but does not provide a long-duration tail distribution.
3. Extend the completed B1/B2/B3/B4/B5 ramp with distinct avatars, staggered conversational turns, short-turn churn, and generated-idle cases. The current capacity run intentionally synchronized one shared audio fixture to stress active generation.
4. Use common receiver gates: useful generated FPS per speaker, declared arrival-gap/stall budget, audio holds, first-speaking-frame latency, lip-sync/identity quality, no increasing queues, per-session memory, interruption, and cleanup. Count held/source-idle frames separately. Include many short turns because chunk-tail padding and reconditioning overhead disproportionately affect conversation.
5. Repeat C4 and C5 remotely through TURN/browser playback and enforce an explicit audio-hold budget. Then qualify the exact profile on RTX 4090; do not extrapolate users from VRAM or FLOPS ratios. More GPUs can scale independent workers, but shared CPU/network/TTS bottlenecks must still be tested.

No GPU was rented and no unrelated process was stopped. The local service was restarted between explicitly recorded profiles, and batch five was added as an opt-in CLI choice. The current recommendation is **four admitted active 15-FPS speakers per tested worker**, with five exposed only as an experimental ceiling until longer remote and audio-continuity gates pass. This is not a blanket production-quality approval or a 25-FPS claim.

## Reproducible evidence audit

[Audit script](../../benchmarks/concurrency_20260916/audit.py) and [extracted evidence, file hashes, source availability, and sizing calculations](../../benchmarks/concurrency_20260916/evidence.json). Reproduce with `python3 benchmarks/concurrency_20260916/audit.py` from this checkout. Sources from the sibling MuseTalk repository remain linked and selected documentary excerpts are preserved in the evidence JSON. Model controls are analyzed separately in [the head-motion report](HEAD_MOTION_CONTROL_2026-09-16.md).
