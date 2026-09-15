# Live SoulX FlashHead wall with Kokoro

> **GPU provenance — local runs:** NVIDIA GeForce RTX 4070, 12 GB (12,282 MiB visible). This applies to the local SoulX/MuseTalk inference and receiver tests described here; CPU-only checks do not establish GPU performance. Historical or upstream results on other GPUs retain their separate attribution. See [GPU run provenance](docs/research/GPU_RUN_PROVENANCE.md) for dates, evidence, and attribution limits.

Use the **Avatar** selector to choose an existing certified avatar before
creating the wall. See [existing avatar setup and cache limitations](EXISTING_AVATARS.md).

Open **http://127.0.0.1:8765/webrtc/wall** after starting the WebRTC service.
The single-session page at `/` also links to the wall.

## Setup and launch

Use the existing SoulX virtual environment and model setup from [WEBRTC.md](WEBRTC.md):

```bash
.venv/bin/python -m pip install -r requirements-tts.txt
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ./start_webrtc.sh \
  --size 512 --steps 4 --fps 25 --batch 1 \
  --max-sessions 10 --max-active-calls 2
```

Kokoro runs locally on CPU and loads on first speech. First use downloads the
Kokoro weights, selected voice, and English language resources. Later turns reuse
the model. No MuseTalk server or external TTS API is required. TTS is optional:
without its dependencies, the wall still connects calls and accepts audio files.

For the existing MuseTalk portrait avatar on this constrained host:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ./start_webrtc.sh \
  --width 480 --height 832 --steps 4 --fps 25 \
  --optimized --real-rope --memory-mode staged --h264-preset veryfast \
  --max-sessions 10 --max-active-calls 2 --idle-policy source \
  --idle-video /workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/idle_active_listening.mp4
```

Run one SoulX server at a time. `staged` reduces GPU memory use by transferring
weights between stages and can substantially slow generation. The wall displays
the actual geometry, frame rate, active-call limit, and avatar from `/health`.
Increasing `--max-active-calls` admits more concurrent speakers; it does not add
GPU throughput. Keep the four-step reference profile for visual assessment.
See [persistent-call profiles](CONTINUOUS_WEBRTC.md) for other memory/idle options.

## Browser workflow

1. Enter the API token if the service requires one; click **Refresh server**.
2. Choose the number of peers and first seed. Each tile gets its own call, peer
   connection, audio/video tracks, and seed (`first seed + tile index`). All calls
   use the server-configured avatar.
3. Click **Kokoro → Speak to all** to create, connect, synthesize, and broadcast in
   one action. Or **Create + connect wall** to inspect idle peers first.
4. Click **Listen** on one tile to hear it. Every tile starts muted for reliable
   autoplay; changing the audible tile mutes the previous one.
5. Edit the text and click **Speak** on one tile to test one speaker among idle
   peers, or broadcast another turn to all. Voice and speed apply to the next turn.
6. **Interrupt** clears speech while preserving the peer. **Stop + release all**
   closes the peers and releases their server state.

File upload supports the server decoder's WAV/FLAC/OGG formats, at most 30 seconds
and 20 MiB. Kokoro also refuses generated speech over 30 seconds; it never silently
truncates a sentence. Shorten text or increase speed if this limit is reached.
Broadcast synthesizes one WAV and uploads it independently to each live peer.
The calls remain connected between turns, so this exercises the same persistent
SoulX path as real multi-turn clients.

A failed speech request retains its original audio and turn ID. **Retry failed
sends** retries only those peers, using the API's idempotency contract. Successful
peers do not get a duplicate turn. Interrupt clears pending retries. Tile errors
identify capacity, token, connection, and queue failures. A failed deletion keeps
the tile and call ID available for another Stop attempt. Closing the page sends
best-effort authenticated deletion; the server also reaps disconnected calls.

## Reading the wall

- **Browser FPS, dropped frames, RTP packet loss, jitter, interval jitter-buffer
  delay, codec, and RTT** come from each actual `RTCPeerConnection.getStats()`.
  Browser FPS includes source-idle and held frames.
- **Neural frames, held frames, queued turns, and speech audio holds** come from
  the call API. Audio holds are withheld useful speech samples divided by 48,000.
  Neural frames include model tail padding and any generated idle.
- **First sender media** is the latest accepted turn's server latency. It does
  not measure browser-visible end-to-end or audible completion latency.
- **Kokoro synthesis / audio** shows synthesis time and resulting speech duration;
  RTF and cold/warm status appear beside the speech controls.

The wall does not restart connections during stats refresh. It exposes generation
starvation rather than treating repeated images as neural throughput. Public NAT
traversal and visual quality still need testing in the intended environment.

## Routes and hosting

| Route | Purpose |
| --- | --- |
| `GET /webrtc/wall` | Multi-peer browser wall |
| `GET /webrtc/wall.js` | Browser implementation |
| `GET /webrtc/tts/kokoro/status` | Availability, voices, limits, busy/load state |
| `POST /webrtc/tts/kokoro` | JSON `{text, voice, speed, language_code?}` → 24 kHz mono PCM WAV |

The two static wall assets are accessible like `/`; all APIs, including TTS and
ICE configuration, retain the existing bearer-token checks. Tokens stay in the
page's memory and are not written to browser storage or URLs. External binding
uses `SOULX_API_TOKEN` and `--host 0.0.0.0`; use the same HTTPS/ICE/TURN setup
described in [WEBRTC.md](WEBRTC.md). A web proxy alone does not carry WebRTC media.
For a remote machine you can forward TCP 8765 for the page, but media still needs
routable ICE candidates or TURN.

Kokoro accepts the curated US/UK voice list returned by its status route. One
synthesis runs at a time; overlapping requests return 429 with `Retry-After`.
Output includes the MuseTalk-compatible `X-Kokoro-Synthesis-Ms`,
`X-Kokoro-Audio-Seconds`, `X-Kokoro-Real-Time-Factor`, `X-Kokoro-Cold-Start`, and
`X-Kokoro-Device` headers. Inference runs outside the HTTP event loop.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_wall.py tests/test_calls.py tests/test_rtc.py tests/test_rtc_loopback.py

# Optional browser dependencies; not required to serve the wall.
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install chromium
SOULX_BROWSER_TEST=1 .venv/bin/python -m pytest -q tests/test_wall_browser.py
```

The Chromium regression uses deterministic CPU frames/speech, with real browser
ICE/DTLS/SRTP/H264/Opus. It verifies three peers, independent seeds, persistent
tracks across turns, audio selection, broadcast and per-tile speech, lost-response
retry, interruption, responsive layout, and cleanup during pending creation.
It is a UI/protocol check, not a benchmark of neural performance. Live GPU/Kokoro
validation should use this same page against the real service.
