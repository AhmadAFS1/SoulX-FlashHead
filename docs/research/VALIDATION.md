# Evidence, reproduction and migration decision

Historical pre-implementation evidence. Later native portrait, TensorRT and live-turn results are tracked in [implementation status](IMPLEMENTATION_STATUS.md). “Every MP4” and other scope statements below refer only to this historical migration-evidence directory, not the newer offline engine experiments.

Codec qualification discovered during implementation: the old native API answerer applied codec preferences too late, allowing VP8. Native API recordings without active-encoder evidence must not be called verified H264 transport. Both APIs now negotiate H264 before applying the offer and expose actual encoder identity. The standalone source-bank relay negotiated its H264 preference before the offer and is unaffected.

Date:2026-09-06. GPU:RTX4070,12282MiB visible, driver570.181. SoulX environment:Python3.10, Torch2.7.1+cu128, BF16, compiled DiT/VAE, aiortc1.14.0/PyAV16.1.0. Native service profile512²/four steps/25FPS/B1, localhost8765.

## Tests completed in this pass

| Test | Result | Interpretation |
| --- | --- | --- |
| All MuseTalk project Markdown |92 files read, hashes inventoried |Historical evidence reconciled with later code/results |
| Production six-video source audit |Six shared first/last six-frame handles;30/30 directed pairs exact |Original assets, not regenerated videos |
| Original-bank real WebRTC relay |1350/1350 frames,56.25s, continuous PTS, zero loss |Persistent local transport control; no AI inference |
| Native SoulX10s initial capture |250 frames; first video1.015s; reported server stall0 |Single active speaker, visual evidence |
| Native SoulX23.05s continuous input |577 frames; first video0.961s; reported server stall0 |One state across speech/silence/speech; not live append |
| Native10s provenance rerun |250 frames,480000 useful audio samples; first video0.995s; server stall0.0426s |Fixture hashes/seed recorded; small jitter disclosed |
| Recording decode/boundary analysis |Native10s/23s fully decoded; source relay FFmpeg decode clean |No exact generated endpoint matching demonstrated |
| Automated tests |12 passed; four upstream Triton deprecation warnings |CPU unit/media controls; no new TRT performance validation |

The provenance rerun had max receiver arrival gap0.259s, despite a much smaller server-accounted stall. The earlier10s and23s max gaps were0.200s and0.234s. Server stalls and receiver jitter measure different stages; neither should be replaced with “perfectly smooth.” Recordings add CPU decode/re-encode work and are not throughput benchmarks.

In the provenance rerun, first audio arrived at1.249s and first video at0.995s: audio's first decoded arrival was approximately253ms later. Arrival timing includes transport and decoder buffering; it is not a measured253ms content lip-sync offset, but it is a startup synchronization concern that needs a playout/PTS and perceptual check. Monotonic timestamps and complete frame counts alone do not validate lip-sync.

## Footage and raw evidence

- [Native10s, provenance rerun](../../benchmarks/migration/soulx-closeup-native-10s-provenance.mp4), [JSON](../../benchmarks/migration/soulx-closeup-native-10s-provenance.json).
- [Native23s speech/silence/speech](../../benchmarks/migration/soulx-closeup-continuous-23s.mp4), [JSON](../../benchmarks/migration/soulx-closeup-continuous-23s.json), [contact sheet](../../benchmarks/migration/continuous-contact-sheet.png).
- [Original six-video source bank in one peer](../../benchmarks/migration/base-bank-webrtc.mp4), [source hashes and transport JSON](../../benchmarks/migration/base-bank-webrtc.json).
- [Decoded native recording boundary measurements](../../benchmarks/migration/recording-boundaries.json).
- [Initial10s capture](../../benchmarks/migration/soulx-closeup-native-10s.mp4) and [JSON](../../benchmarks/migration/soulx-closeup-native-10s.json) retained; use the provenance rerun for reproducible fixture attribution.

Every MP4 is receiver-side media, not an offline fake labeled as a live result. The source-bank lab does transmit actual WebRTC, but deliberately has **no SoulX neural work**. The native footage uses the same approved avatar's canonical frame; SoulX has no base-video input that preserves all original motion.

### Exact input assets

Manifest: `/workspace/MuseTalk/configs/pose_test/sample_ai_human_ltx23_facetime_closeup_production_v1.json`.

Files under `assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/`:

| Physical file | Frames | Duration at24FPS |
| --- | ---: | ---: |
| idle_active_listening.mp4 |241 |10.041667s |
| speaking_direct_v14_subtle.mp4 |289 |12.041667s |
| speaking_direct_v15_reference_paced.mp4 |289 |12.041667s |
| nod_agree.mp4 |145 |6.041667s |
| empathetic_head_tilt.mp4 |241 |10.041667s |
| light_smile.mp4 |145 |6.041667s |

These are480×832/24FPS silent files. Neutral and active-listening logical IDs alias one physical clip; speaking has two physical variants. The shared decoded RGB frame SHA256 is `47b05c6bdd63466e13381dc6cf21545e827bea0bc668c5798cbf7c69f7076b33`. Full file hashes and each first/last handle hash are in the relay JSON. The extracted [anchor PNG](../../benchmarks/migration/closeup-certified-anchor.png) is the first decoded V14 frame.

English inputs are the existing `short.wav` and `multi_sentence.wav` in MuseTalk's `generated/webrtc_quality/2026-08-08/kokoro_duration_matrix/`. Native10s provenance rerun uses the first10s of `multi_sentence.wav`, seed50. The [23.05s combined waveform](../../benchmarks/migration/two-turns-with-gap.wav) uses short3.575s + silence2s + multi_sentence17.475s, seed51. No new paid video/TTS generation was requested.

### Audio accounting correction

The older continuous JSON counts1106880 **received packet samples**, which is23.06s: its last20ms packet includes10ms padding. Useful input is1106400 samples at48kHz/23.05s. The video is577 frames/23.08s. AAC recorder decoding can expose additional codec frame padding (1106944 samples); this is not evidence of extra speech or missing audio. The new benchmark separates useful samples from packet samples. Old evidence is retained unchanged, with its interpretation corrected here.

## Reproduce

Run from the SoulX repo with the existing service healthy. Do not start a second full model on this currently almost-full GPU.

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m soulx_rtc.replay_lab \
  --musetalk-root /workspace/MuseTalk \
  --manifest /workspace/MuseTalk/configs/pose_test/sample_ai_human_ltx23_facetime_closeup_production_v1.json \
  --output benchmarks/migration/base-bank-webrtc-rerun.json

.venv/bin/python -m soulx_rtc.benchmark_rtc \
  --sessions 1 --seconds 10 --record --save-frame --seed 50 \
  --images benchmarks/migration/closeup-certified-anchor.png \
  --audio /workspace/MuseTalk/generated/webrtc_quality/2026-08-08/kokoro_duration_matrix/multi_sentence.wav \
  --output benchmarks/migration/native10-rerun.json

.venv/bin/python -m soulx_rtc.benchmark_rtc \
  --sessions 1 --seconds 23.05 --record --seed 51 \
  --images benchmarks/migration/closeup-certified-anchor.png \
  --audio benchmarks/migration/two-turns-with-gap.wav \
  --output benchmarks/migration/continuous23-rerun.json

.venv/bin/python -m soulx_rtc.analyze_recordings \
  --anchor benchmarks/migration/closeup-certified-anchor.png \
  --videos benchmarks/migration/soulx-closeup-native-10s-provenance.mp4 benchmarks/migration/soulx-closeup-continuous-23s.mp4 \
  --output benchmarks/migration/boundaries-rerun.json
```

Use separate output names to preserve original evidence. `replay_lab` decodes only bounded frames and validates manifest path containment; it requires the original assets locally. Native service signaling is already available at `http://127.0.0.1:8765/`; these finite test sessions were closed after recording. No permanent open call or public mobile/TURN proof is implied by the saved MP4s.

The receiver benchmark uses direct local ICE and records peer0 only when requested. Its first-frame timer starts at the coordinated offer barrier, **after** upload/preparation. Its wall time includes negotiated playback and final observation/drain:11.957s for the10s provenance capture,25.029s for the23.05s capture. Do not divide useful frames by those walls and label the result raw model speed.

## Previously measured capacity — not rerun as a new result here

[Original report](../../benchmarks/REPORT.md) and existing JSON retain the same-4070 measurements:

| Workload | Result |
| --- | --- |
| Ten10s jobs, compiled MuseTalk B8 |2500 useful frames /57.12s =43.76FPS |
| Ten10s jobs, native SoulX B2 |2500 /60.90s =41.05FPS |
| Ten native SoulX WebRTC peers |2500 received /69.09s; ~51s stalls per peer |
| Ten low-cost256²/2-step/15FPS peers |Transport nearly real-time, but generated faces visibly collapse; rejected |

Those older GPU runs had different co-resident memory usage from this pass. Current device usage was~11876/12282MiB; SoulX worker5712MiB, unrelated service6150MiB. The working services were not killed, restarted or repurposed for this analysis. No new ten-peer native load sweep, isolated TensorRT build or matched production-bank MuseTalk GPU benchmark was run in this pass.

## Migration gates

| Requirement | Current status | Release requirement |
| --- | --- | --- |
| Native512 talking video |Demonstrated short runs |Representative long-form quality and lip-sync review |
| Exact original motion plates |Replay verified; SoulX only conditions on a still |Accept new motion or validate a separate hybrid renderer |
| First/last-frame control |No native exact endpoint API |Validated stateful/bridge policy; no visible live↔idle seam |
| Persistent interactive turns |Not in current SoulX clip API |Append/idempotency/barge-in, stable tracks, audible EOF and ownership tests |
| Throughput better than MuseTalk |Not demonstrated against strongest tested local baseline |Matched quality/inputs, repeated useful-FPS and tail-latency A/B |
|25% TensorRT improvement |Hypothesis only |Actual engine, reload/numerical/visual tests and ≥1.25× matched throughput |
|Ten simultaneous native speakers |Not supported by measured compute |≥250 usefulFPS with headroom and strict per-peer playback checks |
|Ten mixed idle/active calls |Plausible serving concept, untested |Publish measured active-speaker limit, burst admission and long-call memory bounds |
|Public mobile hosting |Local media only |HTTPS/auth/TURN, selected candidate proof and real device/network matrix |

Recommendation: retain MuseTalk production and its certified close-up bank. Prioritize SoulX invariant-conditioning/color caches and profiling, then a narrowly scoped BF16/FP16 TensorRT experiment and persistent-call implementation. Reconsider migration only when the combined quality, continuity, throughput and operational gates pass. Bigger full-frame resolution alone is not a migration criterion.
