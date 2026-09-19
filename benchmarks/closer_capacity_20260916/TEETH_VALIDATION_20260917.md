# Teeth comparison: retained 15-FPS and 25-FPS closer clips

September 17, 2026: CPU landmark/decoded-image analysis of existing videos, with MediaPipe XNNPACK CPU inference (its initialization also opens an NVIDIA EGL context). No new SoulX generation or throughput test. All three source videos were generated September 16 on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM**, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8. Earlier offline metadata recorded 2,487 MiB pre-load use with identity unverified; the later serving run identified resident OmniVoice. GPU provenance is retained in the source run JSON and linked reports.

## Finding

The user's observation is **partly validated by direct visual comparison**: several sampled moments in the earlier 25-FPS 1.25× videos show a more distinct upper tooth row, while the later 15-FPS receiver recording shows less distinct, irregular or band-like dental detail. This is not uniform across every frame. The earlier clips also contain blurred/fused teeth; neither is a clean dental reference.

See the [audio-aligned mouth contact sheet](teeth-comparison-20260917.png). Rows are earlier 25-FPS stock conditioning, earlier 25-FPS strength 0.5, and newer 15-FPS stock receiver. Columns sample audio times 0.5, 1.5, 2.5, 3.5, 4.5, 6.5 and 8.5 seconds, chosen uniformly across the utterance rather than selected for favorable appearances. Around 3.5 seconds the earlier tooth row is more distinct; around 1.5 seconds the newer teeth appear more irregular. At 2.5 and 8.5 seconds mouth opening differs substantially, which makes a direct sharpness comparison less meaningful.

The 15-FPS movie contains approximately 1.8265 seconds of leading idle, measured by the retained waveform correlation. That offset was removed before selecting frames. Native 96×52-pixel mouth crops are centered using lip landmarks and displayed at 2× nearest-neighbor scale; no sharpening is applied. Sampling uncertainty is approximately half a frame (20 ms at 25 FPS, 33 ms at 15 FPS), in addition to model audiovisual alignment differences.

## What the measurements do and do not support

All 650 speech frames had detected landmarks: 250 in each earlier clip and 150 in the receiver clip. Median mouth widths are 69.30, 67.44 and 66.02 pixels respectively. Thus even the same 1.25× source does not produce exactly the same output mouth scale/pose.

Whole-mouth Laplacian variance is 50.93, 55.99 and 63.51 respectively. The newer clip actually has **higher** broad mouth-region edge energy, despite worse-looking teeth in some samples. This metric includes lip outlines, mustache, compression and mouth opening; it does not measure individual tooth definition. We therefore reject using it to claim a quantitative percentage of teeth blur. No validated automated dental-quality score was computed.

## Why FPS alone is not established as the cause

| Setting | Earlier stock 25 FPS | Earlier strength-0.5 25 FPS | New 15 FPS |
| --- | --- | --- | --- |
| Dimensions / steps / recorded seed | 320×576 / 4 / 50 | Same | Same for recorded peer zero |
| Weight storage | INT8 with BF16 compute | Same | BF16 resident |
| Audio conditioning | Stock | 30 residual hooks at 0.5 | Stock |
| Reference and recurrent initialization | Direct PNG / prepare-call | Direct PNG / prepare-call | H264 still-loop frame, live idle reconditioning |
| Execution | Offline single state | Offline single state | Up to five states per GPU batch |
| Video path | One H264 file encode | One H264 file encode | Live H264 encoding, decoding and another H264 recording encode |
| Torch allocator cap | 3,584 MiB | 3,584 MiB | 7,168 MiB |

The seed is **not different** for the recorded peer: the benchmark submits seed `50 + peer_index`, and records peer zero. However, changed recurrence and numerical paths can change the seeded trajectory.

FPS is a plausible contributor because the engine passes it into audio preprocessing and maps frame indices to audio samples. A 24-new-frame chunk covers **1.60 seconds at 15 FPS versus 0.96 seconds at 25 FPS**. Lowering this setting changes conditioning and recurrent evolution; it is not simply displaying the identical generated frames more slowly. This explains why different mouth poses can occur at the same audio time, but does not prove dental degradation is inevitable at 15 FPS. The extra codec pass is another plausible source of lost fine detail; its contribution is not isolated here.

## Decisive follow-up

Generate matching 15- and 25-FPS BF16 clips from the exact same PNG, seed, stock conditioning, one-state execution and identical lossless capture. Compare at audio-aligned and similar mouth-open poses. Also create a 15-FPS downsample of the 25-FPS output: this separates the visual effect of lower display cadence from changing neural generation FPS. Then compare lossless output with the delivered WebRTC recording to measure codec losses. This controlled GPU experiment has not been performed in this validation.

For now, the earlier 25-FPS 1.25× clips remain the better observed dental-detail references in several moments. The five-stream result remains a throughput result; the teeth-quality concern is unresolved.

Reproduce CPU inspection from the repository root with `.venv/bin/python benchmarks/closer_capacity_20260916/validate_teeth.py`. [Per-frame diagnostics](teeth-diagnostics-20260917.json), [new receiver run](README.md), [earlier stock run](../distance_lipsync/INDIAN_MALE_CLOSER_REPORT.md), [earlier strength-0.5 run](../distance_lipsync/INDIAN_MALE_CLOSER_STRENGTH_0P5_REPORT.md).
