# Live SoulX FlashHead wall with Kokoro

> **GPU provenance — local runs:** NVIDIA GeForce RTX 4070, 12 GB (12,282 MiB visible). This applies to the local SoulX/MuseTalk inference and receiver tests described here; CPU-only checks do not establish GPU performance. Historical or upstream results on other GPUs retain their separate attribution. See [GPU run provenance](docs/research/GPU_RUN_PROVENANCE.md) for dates, evidence, and attribution limits.

The wall is a port of MuseTalk's WebRTC latency wall
(`MuseTalk/templates/webrtc_wall.py` and `webrtc_player.py`) onto SoulX's
persistent calls. It has the same layout, the same group workflow and the same
per-tile player frames. The pose-protocol controls are replaced by SoulX's avatar,
seed and server-profile controls, plus interrupt and idempotent retry.

Open **http://127.0.0.1:8765/** (also served at `/webrtc/wall`) after starting the
WebRTC service. Use the **Avatar** selector to choose an existing certified avatar
before creating a group. See [existing avatar setup and cache limitations](EXISTING_AVATARS.md).
The older single-session finite-clip page moved from `/` to `/clip`.

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
the actual geometry, frame rate, batch, active-call limit, and avatar from `/health`.
Those are server launch options, shown read-only; restart the server to change them.
Increasing `--max-active-calls` admits more concurrent speakers; it does not add
GPU throughput. Keep the four-step reference profile for visual assessment.
See [persistent-call profiles](CONTINUOUS_WEBRTC.md) for other memory/idle options.

## Browser workflow

1. If the server requires a token, the **API token** field appears. Enter it; the
   wall refreshes when the field loses focus.
2. Choose the avatar, number of peers and first seed. Each tile gets its own call,
   peer connection, audio/video tracks, and seed (`first seed + tile index`).
3. Click **Create + connect group**, or go straight to **Kokoro → Start all**, which
   creates the group if needed, synthesizes once and sends the same audio to every
   tile. **Start all with file** does the same with an uploaded file.
4. Only one tile is audible, tile 1 at first. **Use audio** on another tile moves
   audio to it and mutes the rest, as in MuseTalk.
5. A tile's **Speak** sends the current text to that tile only. Its **Stop**, or
   **Interrupt all**, clears speech while the call stays connected.
6. **Stats** cycles every player frame's in-video debug panel: off, docked or overlay.
   The choice is remembered in this browser.
7. **Delete group** closes every call and releases its server state.

File upload supports the server decoder's WAV/FLAC/OGG formats, at most 30 seconds
and 20 MiB. Kokoro refuses generated speech over 30 seconds and never silently
truncates a sentence. Shorten the text or increase the speed if you hit this limit.

Each broadcast carries one turn ID. If a send fails, or its response is lost after
the server accepted it, **Retry failed peers** re-sends the same audio with the same
turn ID. Calls that already have that turn answer `duplicate`, so no tile hears it
twice. Interrupting clears pending retries for the interrupted tiles.

The endpoints box shows the group's shareable URL, `/webrtc/groups/{id}/wall`.
Opening it (or reloading) loads that group. Frames detached by a reload close their
peers, and the server releases those calls within about a second. The wall then
reports them as ended. **Reconnect peers** replaces every ended call with a fresh one
(same avatar and seed) and restarts playback on the others. Groups with no live
call are forgotten an hour after their last use. A call that never connects is
released after 60 seconds, as before.

## Reading the wall

The metric row summarizes the group:

| Metric | Source and meaning |
| --- | --- |
| Peers | Tiles in the group, and how many calls report a connected peer. |
| Speaking | Tiles whose current turn is preparing, armed, speaking or draining; queued turns across the group. |
| First media | Worst tile's latest turn, from server acceptance to the first sent speech frame. Server latency; not browser-visible end-to-end or audible latency. |
| Neural render | GPU frames per second over the last ten generated chunks (`/health.recent_chunks`), measured while rendering, against the per-speaker target. |
| Client buffer Δ | Worst tile's audio minus video jitter-buffer delay from `getStats()`, over the last two-second interval. |
| Underruns | Speech frames held or missed because generation fell behind, and the worst speech-audio hold in seconds (withheld samples ÷ 48,000). |
| GPU | `nvidia-smi` utilization and memory for the serving GPU (`/stats/gpu-live`, sampled at most once a second). |
| Kokoro | Last synthesis time, speech duration, RTF and cold/warm state. |

Tile stats combine the call API (turn state, queue, first media, audio hold, neural /
idle / held / underrun frames) with that tile's own `RTCPeerConnection.getStats()`
(browser FPS and dropped frames, jitter, jitter-buffer delta, negotiated codecs, RTT).
Browser FPS includes source-idle and held frames. Neural frames include model tail
padding and any generated idle. The docked/overlay debug panel inside each video
adds packet counts, loss, audio bitrate, ICE route and playback state.

The wall does not restart connections during its two-second refresh. Tiles are
reused, never re-created, because detaching a connected frame closes its peer and
ends the call. It exposes generation starvation rather than treating repeated
images as neural throughput. Public NAT traversal and visual quality still need
testing in the intended environment.

## Routes and hosting

| Route | Auth | Purpose |
| --- | --- | --- |
| `GET /`, `/webrtc/wall`, `/webrtc/lab`, `/webrtc/groups/{id}/wall` | page | The wall |
| `GET /webrtc/wall.js` | page | Wall implementation |
| `GET /webrtc/player/{call id}` | page | One tile's player; embeddable, also usable on its own |
| `GET /clip` | page | Legacy single-session finite-clip page |
| `POST /webrtc/groups/create?count=&seed=&avatar_id=` | token | Create N calls atomically (a JSON body is also accepted) |
| `GET` / `DELETE /webrtc/groups/{id}` | token | Group state with per-call stats / release every call |
| `POST /webrtc/groups/{id}/stream` | token | Multipart `audio_file`, optional `turn_id` and `session_ids` → one idempotent turn per call |
| `POST /webrtc/groups/{id}/interrupt` | token | Optional JSON `{session_ids}`; interrupt speech, keep peers |
| `POST /webrtc/groups/{id}/reconnect` | token | Replace ended calls with fresh ones (same seed) |
| `GET /stats/gpu-live` | token | `nvidia-smi` snapshot, MuseTalk field names |
| `GET /webrtc/tts/kokoro/status` | token | Availability, voices, limits, busy/load state |
| `POST /webrtc/tts/kokoro` | token | JSON `{text, voice, speed, language_code?}` → 24 kHz mono PCM WAV |

Group routes and response fields follow MuseTalk's `/webrtc/groups` API
(`group_id`, `sessions[].session_id`, `player_url`, `wall_url`,
`stream_all_url`; stream returns `started`, `failed` and `results`). Existing wall
scripts therefore port directly. SoulX adds `seed`, `stats` and `last_turn` per
session and the interrupt and reconnect routes. The pose protocol is not ported.
A group is only a list of call IDs. Media and GPU state stay in the individual
calls, so the per-call API in [CONTINUOUS_WEBRTC.md](CONTINUOUS_WEBRTC.md) is unchanged.

Pages are static files without secrets, public for `GET`/`HEAD` only. Every API
they call keeps the bearer-token check. The token stays in the wall page's memory.
Player frames request it from their same-origin parent with `postMessage`, so it is
never written to browser storage or URLs. A player opened on its own asks for the
token when you tap its overlay. External binding
uses `SOULX_API_TOKEN` and `--host 0.0.0.0`; use the same HTTPS/ICE/TURN setup
described in [WEBRTC.md](WEBRTC.md). A web proxy alone does not carry WebRTC media.
For a remote machine you can forward TCP 8765 for the page, but media still needs
routable ICE candidates or TURN. The player does not trickle ICE: aiortc answers
once, so each offer waits up to 20 seconds for candidate gathering.

Kokoro accepts the curated US/UK voice list returned by its status route. One
synthesis runs at a time; overlapping requests return 429 with `Retry-After`.
Output includes the MuseTalk-compatible `X-Kokoro-Synthesis-Ms`,
`X-Kokoro-Audio-Seconds`, `X-Kokoro-Real-Time-Factor`, `X-Kokoro-Cold-Start`, and
`X-Kokoro-Device` headers. Inference runs outside the HTTP event loop.

## Verification

```bash
.venv/bin/python -m pytest -q tests/test_groups.py tests/test_wall.py tests/test_calls.py tests/test_rtc.py tests/test_rtc_loopback.py

# Optional browser dependencies; not required to serve the wall.
.venv/bin/python -m pip install playwright
.venv/bin/python -m playwright install chromium
SOULX_BROWSER_TEST=1 .venv/bin/python -m pytest -q tests/test_wall_browser.py
```

`test_groups.py` is a **[CPU]** contract suite. It covers page/API authentication,
atomic creation and capacity, idempotent broadcast and retry, per-tile subsets,
interrupt, reconnect, deletion, the GPU-stats cache, and group speech reaching a
real aiortc peer. The Chromium regression is also **[CPU]**: deterministic frames
and speech, with real browser ICE/DTLS/SRTP/H264/Opus. It covers three peers with
independent seeds, token entry, one-click Kokoro broadcast, audio-leader switching,
lost-response retry without duplicates, per-tile speech, interruption, stats modes,
responsive layout, reload with reconnect, and cleanup. Neither suite measures
neural performance. Live GPU/Kokoro validation should use this same page against
the real service.
