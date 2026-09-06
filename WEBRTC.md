# SoulX-FlashHead shared-model WebRTC experiment

This is a single-GPU experimental service, not a claim of full MuseTalk product
parity. It implements independent talking-head sessions, uploaded image/audio,
shared weights, fair microbatch scheduling, bounded output queues, H264/Opus
WebRTC, cancellation, API authentication, and ICE configuration. It does not
currently implement MuseTalk's pose protocol, TTS integration, persistent
multi-turn peers, S3 avatar storage, or worker control plane.

## Run

From this repository, using the separately installed SoulX virtual environment:

```bash
# Reference quality/rate. First startup compiles and warms kernels.
./start_webrtc.sh --size 512 --steps 4 --fps 25 --batch 1

# Stress-test only: this profile FAILED visual-quality validation (face collapse).
./start_webrtc.sh --size 256 --steps 2 --fps 15 --batch 2
```

Open http://127.0.0.1:8765. The server starts listening only after warmup. It
warms every microbatch size that can occur, including partial batches. Model
weights are loaded once, not once per connection. Do not run multiple server
workers on the same GPU.

The UI accepts an avatar and audio file, or uses the included examples. It
supports 1–30 second clips. Uploads are bounded to 20 MiB. Image inputs are
bounded to 16 megapixels. A maximum of ten admitted sessions is a resource
limit, **not** a guarantee of ten real-time speakers at every profile.

## Hosting

For a non-loopback bind, set `SOULX_API_TOKEN` through your secret manager or
shell environment, then add `--host 0.0.0.0`. Enter the token in the UI; API
clients send `Authorization: Bearer …`. No token is embedded in the HTML or
stored in browser local storage. TLS termination and network access controls
are your deployment's responsibility. Do not expose an unprotected GPU API.

Optional `SOULX_ICE_SERVERS` is JSON in browser/aiortc format:

```json
[{"urls":["turn:YOUR_TURN_HOST:3478"],"username":"YOUR_USER","credential":"YOUR_CREDENTIAL"}]
```

The default is host candidates only: appropriate for loopback or directly
routable networks. HTTP reverse proxying alone does not relay WebRTC media.
NAT/firewall deployments need working ICE candidates, open UDP paths or TURN.
Use HTTPS and short-lived TURN credentials for external deployment. Local
loopback testing does **not** establish public-network/TURN reliability.

## API

| Method/path | Purpose |
| --- | --- |
| `GET /health` | Profile, session count, recent GPU timings and memory |
| `GET /config` | ICE configuration for an authenticated client |
| `POST /sessions` | JSON `{seconds, seed}` for sample inputs, or multipart `image`, `audio`, `seconds`, `seed` |
| `POST /sessions/{id}/offer` | SDP `{type: "offer", sdp: "…"}` → SDP answer |
| `GET /sessions/{id}` | Generated/sent counts, first-chunk latency, playback stalls |
| `DELETE /sessions/{id}` | Cancel, close peer and release session state |

Generation begins when the peer connects. An unused, completed, or stalled
session is reclaimed after 60 seconds without activity. A full session queue
is skipped by the scheduler; it cannot block other clients. There is one
outstanding GPU chunk per session. The first chunk plus a 250-ms startup cushion
is the initial prebuffer; this cushion absorbs small scheduling bursts.

## Memory and correctness

- A dedicated GPU process isolates model dispatch from Python WebRTC/GIL load.
  CUDA tensors stay in that process; only bounded uint8 chunks cross IPC.
- Immutable shared model/reference tensors; per-session motion latents and RNG.
- Same-profile DiT microbatches; sequential VAE decode/encode bounds VRAM.
- Fixed batch bugs in RoPE and audio cross-attention; no first-session broadcast.
- Initial one-latent history and later two-latent history handled per row.
- VAE posterior sampling also uses each session's generator.
- Only newly emitted frames are copied to CPU, as uint8, after motion encoding.
- Two queued chunks per client plus one chunk being consumed. At 512² these
  are 18 MiB/chunk; at 256², 4.5 MiB/chunk. Whole videos are not retained by the server.
- A common playout clock gates audio behind available video. Underload produces
  real pauses; it is not hidden by counting repeated frames as model throughput.
- One terminal video frame and 100 ms of silence drain receiver jitter buffers.
  They are separately counted and excluded from useful generation/delivery FPS.

**Quality warning:** the tested 256²/two-step/15-fps profile developed severe
facial artifacts by 3–5 seconds. Receiving all frames at ten-way concurrency
does not make it a usable quality-equivalent talking-head workflow. Use the
512²/four-step/25-fps reference profile for normal experimentation. The failing
combined profile does not identify which individual parameter caused the damage.

The 15/20-fps modes resample the Wav2Vec temporal features to the requested
frame timeline and generate fewer frames for the same audio duration. They
do not simply change video timestamps or speed up the audio. These modes and
two-step/256-pixel generation are quality tradeoffs, not reference-quality equivalents.

## Reproduce tests

```bash
.venv/bin/python -m pytest -q tests

# Real inference, CPU-ready frames, independent seeds/audio histories.
.venv/bin/python -m soulx_rtc.benchmark_engine \
  --size 512 --steps 4 --batch 1 --sessions 1 10 \
  --output benchmarks/repeat-native.json

# Run against a warmed server. Actually receives and decodes SRTP media.
.venv/bin/python -m soulx_rtc.benchmark_rtc --sessions 10 \
  --output benchmarks/repeat-rtc.json

# Separate visual sample run; recording adds CPU work, so don't use it for capacity claims.
.venv/bin/python -m soulx_rtc.benchmark_rtc --sessions 1 --record --save-frame \
  --output benchmarks/sample-rtc.json
```

`--validate-isolation` on the server additionally generates two chunks for two
sessions in forward and reversed batch order and checks pixel differences.
This includes the recurrent motion state, not just a mock transport test.

Benchmarks exclude cold model/avatar setup and compile warmup, but include
audio encoding, all denoising/VAE work, overlap and tail padding, and conversion
to CPU-ready pixels. The WebRTC benchmark additionally measures actual delivery.
Only frames belonging to the requested duration count toward useful throughput.

For the MuseTalk comparison, use its existing interpreter and working directory;
do not install SoulX requirements into the MuseTalk environment. The comparison
helper prepares only the local `soulx_comparison_4070` avatar cache and does not
touch the saved TRT/server launch configuration. See benchmark JSON and the
experiment report for the exact backend and profile of each result.

```bash
cd /workspace/MuseTalk
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MUSETALK_VAE_DECODE_TIMING=0 \
TORCHINDUCTOR_CACHE_DIR=/workspace/SoulX-FlashHead/.torchinductor-musetalk \
TORCHINDUCTOR_COMPILE_THREADS=4 \
/workspace/.venvs/musetalk_trt_stagewise/bin/python \
  /workspace/SoulX-FlashHead/soulx_rtc/benchmark_musetalk.py \
  --batch 8 --compile --sessions 1 10 \
  --output /workspace/SoulX-FlashHead/benchmarks/musetalk-repeat.json
```

This standalone helper selects native models directly; it does not invoke
the TRT loader, API launch profiles, S3 storage or the worker control plane.
