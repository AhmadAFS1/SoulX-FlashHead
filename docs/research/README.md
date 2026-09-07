# SoulX-FlashHead for continuous avatar calls

This is the pre-implementation research snapshot. The subsequent [implementation and measured results](IMPLEMENTATION_STATUS.md), [persistent call API](../../CONTINUOUS_WEBRTC.md), and [TensorRT experiments](../../TENSORRT_EXPERIMENTS.md) supersede statements below about missing runtime features. Historical measurements are retained unchanged.

Analysis and receiver-side testing: **2026-09-06**, RTX 4070 12 GB.

## Decision

**Keep MuseTalk as the production renderer; keep SoulX Lite as an experimental alternative.** Native SoulX works and can produce coherent 512×512 talking-head footage, but it has not demonstrated higher throughput than the stronger tested MuseTalk baseline, ten simultaneous real-time speakers, or the current MuseTalk pose/turn continuity contract. A wholesale migration is not justified by the evidence available today.

The most important architectural distinction: MuseTalk changes a **256×256 face crop inside an existing video**. SoulX generates the **whole frame from a portrait and audio**, including new head motion. Reusing the same portrait does not preserve the choreography or framing of the original 480×832 motion plates.

## Read / watch

- [Architecture and source walkthrough](ARCHITECTURE.md): tensors, temporal recurrence, audio, low-VRAM worker and missing APIs.
- [Optimization transfer and TensorRT plan](OPTIMIZATION_PLAN.md): what actually worked for MuseTalk, what transfers, priorities and release gates.
- [Continuous calls and first/last-frame control](CONTINUOUS_CALLS.md): exact boundaries, independent turns, idle switching and proposed persistent-call design.
- [Validation, footage and migration gates](VALIDATION.md): measured results, reproducible commands, limitations and acceptance matrix.
- [All 92 MuseTalk Markdown files reviewed](MUSETALK_DOC_AUDIT.md): full paths, SHA-256 hashes and per-file reading notes.
- [10-second native SoulX WebRTC recording](../../benchmarks/migration/soulx-closeup-native-10s-provenance.mp4).
- [23-second continuous SoulX WebRTC recording](../../benchmarks/migration/soulx-closeup-continuous-23s.mp4): two preassembled speech passages separated by two seconds of silence, one model state.
- [Original six-video bank through one persistent WebRTC session](../../benchmarks/migration/base-bank-webrtc.mp4): **source-video relay control, no SoulX inference**, native 480×832.

## What this change delivers

Source analysis, documentation, actual recordings, an exact-source asset audit/persistent-peer replay lab, a recording-boundary analyzer, and benchmark input provenance/audio-padding accounting. Twelve automated tests passed, including actual H264/Opus loopback media.

It does **not** install or claim a TensorRT engine, change model precision, deploy a new production multi-turn API, or migrate the MuseTalk runtime. Proposed optimizations below are explicitly unmeasured until implemented and A/B tested. Existing GPU services were left running; the unrelated co-resident service was not stopped to manufacture better numbers.

Prior measured performance and failed low-resolution footage remain in [the original report](../../benchmarks/REPORT.md). This research extends that report rather than replacing its evidence.
