# WebRTC wall rework (MuseTalk parity): validation, 27 September 2026

**GPU (live runs below):** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB physical (from
`nvidia-smi`), driver 595.84. The GPU had no other process during the live runs:
another session's MuseTalk probe had finished and the MuseTalk server was stopped.
Host RAM was tight (about 6 GB available). **Browser:** headless Chromium from Playwright on the same host,
host ICE candidates on loopback. This is not a public-network or TURN test.

## What changed

`/` and `/webrtc/wall` now serve a port of MuseTalk's WebRTC latency wall. Each
tile is an iframe running `/webrtc/player/{call id}`, driven by the new
MuseTalk-compatible `/webrtc/groups/*` API (`soulx_rtc/groups.py`). See
[WEBRTC_WALL.md](../../WEBRTC_WALL.md) for the workflow and routes.

- [musetalk_vs_flashhead_empty.png](musetalk_vs_flashhead_empty.png): MuseTalk's wall (target) next to the new FlashHead wall.
- [before_after_speaking_3peers.png](before_after_speaking_3peers.png): old SoulX wall (HEAD `cf6c308`) and the new wall, both with three peers speaking. **[CPU]** fake renderer, real idle video and Kokoro.
- [live_wall_session_2peers.mp4](live_wall_session_2peers.mp4): 28 s screen capture of the new wall against the live GPU server: create, Kokoro to all, stats docked/overlay, audio switch, mobile, delete. The headless capture has no audio track.
- `live_03-speaking.png`, `live_04-stats-docked.png`, `live_06-mobile.png`, [live_drive.log](live_drive.log): stills and log from that session.

## Live GPU session

Server: `serve-low-vram.sh --idle-video …closeup_production_v1/certified/idle_active_listening.mp4 --cuda-memory-mib 6144`
(lite model, 320×576, 4 steps, 25 fps, batch 1, INT8 weights, compact memory,
**one rendering call at a time**). The script's own 3,584 MiB Torch cap ran out of
CUDA memory in `engine.warmup` (VAE decode) before the server became ready, with
7.9 GB of the GPU free. The cap was raised for this run only; the script is unchanged.

| Measure (2 peers, one 6.83 s Kokoro turn to both) | Value |
| --- | --- |
| Group created → both tiles playing | 1.5 s |
| Tile 1 first media (server accept → first speech frame sent) | 0.55 s |
| Tile 2 first media | 8.7 s; it waits for tile 1 because only one call renders at a time |
| Neural render while rendering (`/health.recent_chunks`) | 68–69 FPS |
| Browser FPS / dropped frames / underruns | 25.0 / 0–3 / 0 |
| GPU memory, whole board | 4.3 GB |
| Kokoro (CPU, warm) | 2.06 s for 6.83 s of audio, RTF 0.30 |

68–69 FPS of render against a 25 FPS target suggests `--max-active-calls 2` could
serve two simultaneous speakers on this profile. That was **not tested** here.

## Old vs new player A/B (live GPU, one peer, same WAV)

A same-origin proxy served the HEAD wall at `/old/` and forwarded every API call to
the one live server. The two UIs alternated across four trials each. The values are
per-second medians of the inbound-RTP playout (jitter-buffer) delay during speech.
Raw data: [ab_jitter_1peer_run1.json](ab_jitter_1peer_run1.json), [ab_jitter_1peer_run2.json](ab_jitter_1peer_run2.json).

| UI | Audio buffer, ms | Video buffer, ms | Decoded FPS | Drops |
| --- | --- | --- | --- | --- |
| Old (HEAD) | 33.5, 29.1, 29.8, 34.6 | 8.8, 8.5, 8.1, 8.3 | 24.93 | 0 |
| New (iframe player) | 76.2, 29.9, 29.8, 29.7 | 9.5, 11.1, 12.1, 11.3 | 24.93 | 0 |

Audio matched in three of four trials. The first new-UI trial started higher, a
single outlier that did not recur. Video playout is about 3 ms higher with the iframe
player, consistently. That is under a tenth of a 40 ms frame. Delivered frame rate and
drops are identical. A much larger reading from an earlier CPU dev harness came from
that harness's own load (Inductor compile and in-process Kokoro), not from the player.

## CPU checks

- `tests/test_groups.py` (5 tests) and `tests/test_wall_browser.py` (Chromium, opt-in) are **[CPU]**, with deterministic frames.
- The full suite ran with CUDA hidden: 333 passed, 14 failed, 4 skipped. The 14 failures fail identically on the unmodified tree, and none imports a changed module.
- Four call/RTC tests need `examples/girl.png` and `examples/podcast_sichuan_16k.wav`. Those files are gitignored and absent on this host. With generated stand-ins, all 47 wall/call/RTC tests pass.
