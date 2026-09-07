# Persistent portrait WebRTC calls (experimental)

Internal contracts are explained in the [call/scheduler walkthrough](docs/architecture/05_WEBRTC_AND_SCHEDULING.md) and [VAE/continuity guide](docs/architecture/03_VAE_AND_CONTINUITY.md). Remaining queued-turn, admission and performance work is in the [next optimization plan](docs/research/NEXT_OPTIMIZATION_PLAN.md).

The implementation is separate from the original finite `/sessions` API. One call owns one peer, one H264 video sender and one Opus audio sender. New turns never replace tracks or restart RTP timestamps. Model weights remain shared in the single GPU process.

This is a local integration/test endpoint, **not a certification of invisible live/idle joins, mobile/TURN deployment, or ten active speakers**. The supplied idle video can be replayed without neural work; held/idle frames are counted separately from newly generated frames.

## Start with the existing MuseTalk idle avatar

```bash
./start_webrtc.sh --width 480 --height 832 --fps 25 --steps 4 \
  --optimized --real-rope --profile --memory-mode reference \
  --max-sessions 10 --max-active-calls 1 \
  --idle-video /workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/idle_active_listening.mp4
```

The startup script defaults to `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This setting materially affected whether the portrait profile fit beside the unrelated GPU workload. Do not run another full SoulX instance on this same almost-full device. `--memory-mode reference` offloads the DiT only for reference preparation/reconditioning; `staged` offloads it around every decode and has substantial PCIe overhead. `default` retains the square baseline behavior; `compact` only temporally tiles frame-independent color correction.

A nonblocking OS lock in `.gpu-owner.lock` permits one GPU owner per checkout. It is released when its process exits; a leftover empty file is not a stale held lock. This does not protect against GPU consumers in other repositories. Stop and verify the old worker has exited before starting another profile; do not stop unrelated services.

The later `staged` implementation also keeps the audio encoder on CPU outside audio preprocessing and delays the DiT reload until audio processing has finished. DiT remains on CPU after VAE decoding, until the next chunk needs denoising. These explicit transfers reduce stage overlap in VRAM but add audio-weight transfer cost. The earlier approximately12.2-FPS staged measurements predate this follow-up and are not timings of the final fallback.

`--idle-policy source` replays the original asset between turns and reconditions when speech starts. `hold` keeps the last sent image and continues the existing model history on the next turn, but looks frozen during silence. `generate` continues native model generation through silence and speech without reference resets; this spends GPU while idle and is not a ten-cheap-idle-call strategy. Generated-idle frames are separately counted and synthetic idle turns do not fill the user-turn idempotency history. The same configured source asset supplies the avatar's first frame in all three modes.

Generated silence renders one chunk ahead within the existing synthetic turn rather than re-arming playback every24 frames. Its zero audio is tracked by counters, not an ever-growing waveform. A queued speech turn stops further idle extension and waits for already-generated idle to drain; turn-boundary generation latency/holds can still occur. This is not a zero-latency endpoint guarantee.

Speech waiting on another peer also stops synthetic-idle extension so the idle call can release its rendering slot after draining. This avoids admission starvation at a one-call limit; it does not create enough compute for simultaneous speakers. Non-admitted generated-idle peers hold their last image.

480×832 is the exact certified source size, **not exact 9:16**. Native 576×1024 and 288×512 are valid 9:16 dimensions; they are different model profiles and require their own quality/memory/performance checks. Both native dimensions must be multiples of32. No square-output resize is used in the GPU engine's rectangular generation path.

Open `http://127.0.0.1:8765/`, select **Persistent FaceTime experiment**, then Start. The server-configured idle avatar is used; the finite-clip portrait upload does not select a different persistent-call identity. Choose a WAV file and use **Send audio turn**, then **Interrupt** or **Stop**. A failed send can be retried with the same turn ID. Finite-clip mode still uses `/sessions`.

## API

All non-root routes use the existing `SOULX_API_TOKEN` bearer middleware when configured. External binds still require a token. The client never receives arbitrary filesystem access.

| Method and route | Contract |
| --- | --- |
| `POST /calls` | JSON `{ "seed": 50 }`; returns prepared call ID. Source/hold idle does not generate neural frames; generate-idle does. |
| `POST /calls/{id}/offer` | One SDP offer; second offer returns409. ICE settings come from existing configuration. |
| `POST /calls/{id}/turns` | Raw supported audio body, header `X-Turn-ID`; ≤30s, up to four queued turns per call. Returns202. |
| Same turn ID + same bytes | Idempotent retry, returns existing turn state; no duplicate playback. |
| Same ID + different bytes |409 conflict. |
| `POST /calls/{id}/interrupt` | Increment cancellation epoch, discard old pending output, recondition from the last nine **sent** RGB frames. |
| `GET /calls/{id}` | Counters, active/queued state and latest64 turn summaries. |
| `DELETE /calls/{id}` | Idempotent closure, peer/decoder/queues/GPU state released. |

The per-call idempotency map retains up to2048 compact turn records and refuses more instead of forgetting IDs and accidentally replaying retries. Finished/interrupted waveform arrays are released. GPU audio state retains the rolling eight-second history plus bounded future input. Output is limited to two queued chunks plus the current chunk and nine sent history frames.

## Playback/state policy

Turns are generated on complete24-frame boundaries; their final padding is explicit and not extra speech. Audio/video start on a shared future media boundary. RTP timestamps remain continuous through idle, turns and interruptions. If generation falls behind, video holds and audio inserts silence rather than running speech ahead; the counters disclose underruns. This does **not** make overloaded calls perceptually real-time.

After video begins, audio may lead the sent-video horizon by at most one 20-ms Opus packet. This avoids an unnecessary silence packet when audio/video callbacks race on a frame boundary, while still bounding advancement during a real video stall. `audio_hold_samples` counts withheld useful speech samples separately from video underruns; divide by48,000 for seconds. It excludes intended idle and final padding. This is sender scheduling, not a receiver-playout acknowledgement.

The last speech image is held until all useful audio has been sent; only then can idle resume. Server send completion is not proof of remote audible completion. Receiver jitter-buffer/AV-sync checks remain mandatory before product promotion.

The video RTP clock is separate from the number of transmitted frames. When encoding misses a wall-clock slot, timestamps advance rather than sending a burst of old idle images. `missed_video_slots` records this explicitly; speech-time misses also contribute to underrun accounting. No generated content is skipped and no missed slot counts as a generated or sent frame. This prevents idle backlog from delaying a later turn, but does not remove real CPU/network overload.

Reconditioning on interruption uses the most recent nine sent images, not a future model state associated with discarded frames. It is explicitly **not** receiver-acknowledged rollback. Idle assets have their own head movement, so reconditioning happens before a newly admitted turn when source-idle mode is configured. Reference preparation and queued generation can still make the starting pose differ from the last displayed idle frame. The native model has no exact endpoint API; invisible transitions are an open visual-quality gate.

## Reproduce actual peer tests

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m soulx_rtc.benchmark_calls \
  --audio /workspace/MuseTalk/generated/webrtc_quality/2026-08-08/kokoro_duration_matrix/short.wav \
          /workspace/MuseTalk/generated/webrtc_quality/2026-08-08/kokoro_duration_matrix/multi_sentence.wav \
  --audio-seconds 3 --turns 2 --gap 2 --record --interrupt \
  --output benchmarks/implementation/calls-rerun.json
```

Use a fresh output filename. Recordings are actual receiver-side media. `wire_fps` includes idle/held images and is never called model throughput. The unpaced `soulx_rtc.experiment` runner is the separate throughput benchmark.

For sustained testing, add `--duration-seconds 1800` (overrides the turn count). Progress is printed every30 seconds; full turn audio is released on completion and only compact idempotency records remain. The recorder is optional and should be disabled during throughput/load measurements.

`--sessions 10 --speakers 1` tests ten connected peers with only one submitting speech. This is a mixed idle/active test, not a ten-speaker test. `--speakers 0 --duration-seconds 30` is an idle-only transport probe. Without `--speakers`, every peer submits turns.

For a long run, `--compact-evidence --snapshots-every 60` keeps full timestamp arrays in a hashed gzip companion and saves periodic native receiver images. Server/worker RSS, retained audio, queue depth and GPU-state counts are sampled; cleanup must return peer/state counts to the pre-test baseline. Snapshots are limited visual evidence, not a full perceptual motion/lip-sync assessment.

The persistent scheduler can microbatch up to `--batch` compatible calls, without waiting to fill a batch. Lead/last-scheduled ordering and the active-call limit are applied before admission. Each peer retains independent motion latents and RNG state. Larger batches need their own GPU-memory validation; the low-VRAM portrait service defaults to batch one. Fixed-shape TensorRT FFN engines currently require batch one.

`--max-sessions` bounds connected peers across both APIs; `--max-active-calls` bounds rendering calls. It does not constrain separately created finite `/sessions` workloads, which share the GPU scheduler. Do not expose both workload types to independent untrusted tenants without a unified active-compute admission policy. Current API authentication is a shared service token, not per-user ownership.

`--h264-preset veryfast` opts into a two-thread, zero-latency libx264 encoder with the profile's frame rate and a 90-kHz codec clock. It uses a process-local factory hook for the tested aiortc 1.14.0 and refuses unvalidated versions; no installed library files are edited. Call stats report the actual encoder class/codec. This changes compression/CPU cost, not model FPS, and still requires visual review. Omit it to retain the upstream encoder. Per-request access logs are off by default; `--access-log` enables them.

## Tested native TensorRT service

The local native artifacts were built separately from the square artifacts. After building them as described in [TensorRT experiments](TENSORRT_EXPERIMENTS.md), the tested single-call configuration is:

```bash
PYTHONPATH=/workspace/SoulX-FlashHead/.trt-experiment/site-packages \
./start_webrtc.sh --width 480 --height 832 --fps 25 --steps 4 \
  --optimized --real-rope --profile --memory-mode reference \
  --trt-ffn .trt-experiment/portrait-ffn-bf16 \
  --trt-vae .trt-experiment/portrait-vae-bf16/vae.engine \
  --h264-preset veryfast --idle-policy generate \
  --max-sessions 10 --max-active-calls 1 --validate-isolation \
  --idle-video /workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/idle_active_listening.mp4
```

This is experimental single-call admission, not a production ten-call capacity setting. Generated-idle peers also occupy rendering slots. The ten-job unpaced TRT sweep measured 28.57 aggregate useful FPS; the ten-active-peer live test failed smoothness. See [all results and limitations](docs/research/IMPLEMENTATION_RESULTS.md).

## Current constrained-host fallback

After the recorded TRT tests, the unrelated OmniVoice process grew to approximately7,456 MiB. The fast native TRT profile no longer fit alongside it. The final local service uses the command above **without `PYTHONPATH`, `--trt-ffn` or `--trt-vae`, and with `--memory-mode staged`**. It retains native480×832, four steps and the same avatar. Check `/health` for the actual backend rather than assuming the faster profile is active.

This fallback starts and passes GPU session-isolation/protocol checks, but does not sustain real-time generation. Its final C1 regression recorded substantial audio/video holds. It is left as an experimental runnable endpoint, not a smooth FaceTime deployment. Restoring the fast profile requires freeing enough GPU memory; do not stop another service without its owner's authorization.
