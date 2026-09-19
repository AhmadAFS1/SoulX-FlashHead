# Ojin Avatar — local container examples

Run the Ojin avatar model on your own GPU and drive it from your own code. This repo contains
everything you need to go from the delivered image to a talking avatar video in one command.

The avatar service runs **entirely on your machine** — no internet access, no Ojin account, no
data leaves your network. It exposes a single WebSocket endpoint; you send speech audio and it
streams back lip-synced video frames at **1024×1024**.

## Requirements

- **NVIDIA Blackwell GPU** (RTX PRO 6000 / compute capability 12.0)
- **NVIDIA driver 580 or newer** — see below, this one bites
- **Docker** with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  (so containers can use the GPU)
- ~25 GB of disk for the image

Check both in one go:

```bash
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader
# expect: NVIDIA RTX PRO 6000 ..., 580.xx or newer, 12.0
```

> **Why 580 specifically.** The avatar is upscaled to 1024×1024 by a TensorRT engine, which needs a
> newer driver than the GPU itself does. On an older driver (e.g. 570) the container **refuses to
> start** and prints a diagnostic banner telling you exactly what's wrong — required vs detected
> driver version, the detected GPU and compute capability, and the escape hatch to run at native
> 512×512 without the upscaler. If startup stops with that banner, update the driver to **580+** and
> start it again. (It is a deliberate, fail-fast refusal — not a crash — even though it appears
> during startup.)

## 1. Load the image

We deliver the image as an encrypted archive, plus a download URL, a SHA-256 checksum, and a
passphrase (sent separately). Verify and load it in one step:

```bash
./scripts/load_image.sh '<download-url>' '<sha-256>'
```

That downloads, checks the SHA-256 **before** touching the file, decrypts, and loads it into
Docker as `oj-avatar-service:kit-clean`. If you already have the `.gpg` file locally, pass its
path instead of the URL — no network needed.

## 2. See it work (no accounts, one command)

```bash
docker compose --profile demo up
```

This starts the avatar service, waits until the model has finished loading, plays a bundled
6-second audio clip through it, and writes the result to **`out/avatar.mp4`** — open that file
and you'll see the avatar speaking the clip.

> First start takes **~2 minutes** while the model loads onto the GPU. Compose waits for the
> service to report ready, so the demo won't run early. Later starts are faster.

## 3. Talk to it with your microphone (optional)

```bash
cp .env.example .env          # add your speech-service keys
cd examples/live-mic
pip install -r requirements.txt
python bot.py
```

Speak, and the avatar answers in a window. The avatar itself is still local — only the
speech-to-text, language model, and text-to-speech are cloud services (Deepgram, Groq,
ElevenLabs — the stack we run in production). Each is an ordinary
[pipecat](https://github.com/pipecat-ai/pipecat) service and can be swapped for your own
provider, including self-hosted ones: see the `create_stt` / `create_llm` / `create_tts`
functions in `examples/live-mic/bot.py`.

## Using your own face

Drop any portrait image into `avatars/` and reference it by filename:

```bash
OJIN_CONFIG_ID=myface.png docker compose --profile demo up
```

`avatars/` is mounted read-only into the container; images never leave your machine.

## Writing your own client

`examples/headless/main.py` is a complete, ~150-line client — the same code as the
[Ojin SDK's speech-to-video example](https://github.com/ojinai/python-sdk/tree/main/examples/01-speech-to-video-mp4-generator),
pointed at your container instead of our hosted service. The essentials:

```python
from ojin.stv import OjinSTVClient, QueueOutput, STVVideoFrame

client = OjinSTVClient(
    api_key="local",                          # the local container accepts any value
    config_id="sample.png",                   # a filename in ./avatars
    ws_url="ws://localhost:8000/realtime",    # your container
    output=QueueOutput(max_video=10**9),
)
await client.connect_with_retry()
await client.start()
await client.say(pcm, sample_rate=16000, num_channels=1)   # mono 16-bit PCM

async for frame in client.output_stream():
    if isinstance(frame, STVVideoFrame) and frame.rgb is not None:
        ...  # frame.rgb is raw RGB24, frame.width x frame.height (1024x1024)
```

Install the SDK with `pip install "ojin-client[stv] @ git+https://github.com/ojinai/python-sdk.git@main"`.

## Layout

| Path | What it is |
|---|---|
| `docker-compose.yml` | Runs the delivered image with the right GPU/port/volume settings |
| `avatars/` | Face images, mounted read-only into the container |
| `examples/headless/` | Tier 1 — WAV in, MP4 out. No accounts, runs in Docker |
| `examples/live-mic/` | Tier 2 — microphone in, avatar in a window. Needs speech-service keys |
| `scripts/load_image.sh` | Verify + decrypt + load the delivered image |
| `.env.example` | Template for the live-mic keys and optional overrides |

## Notes & troubleshooting

- **Capacity.** The service handles **one session at a time** by default (one avatar per GPU at
  this quality). Additional clients queue and, if no slot frees within ~25 s, are rejected with
  a "try again later" close — they are never silently degraded. Raise it with
  `NUM_INFERENCE_SESSIONS` if you have headroom, but validate quality first.
- **Resolution.** Output is 1024×1024: the model renders 512×512 and a 2× upscaler (built for
  your GPU) runs on every frame. It is on by default.
- **"No capacity" right after startup.** The model is still loading. The examples retry
  automatically; `docker compose logs -f avatar` shows progress.
- **The container exits immediately.** Almost always the GPU isn't visible to Docker. Check
  `docker run --rm --gpus all nvidia/cuda:12.8.1-base-ubuntu22.04 nvidia-smi`.
- **Compose can't see the GPU.** Older Compose versions want `gpus: all` on the service instead
  of the `deploy.resources.reservations.devices` block used here.
