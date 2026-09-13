# Bounded media optimizations and recorded WebRTC validation — 2026-09-13

This implements the media/cache/IPC track from the
[September 11 pipeline profile](PIPELINE_PROFILE_2026-09-11.md). It does not
change the neural model, precision, step count, color math or VAE/GEMM kernels.
Wire FPS and aggregate wire FPS below include idle frames; they are **not**
generated-model throughput or evidence of ten simultaneous speakers.

Evidence directory: `/workspace/experiments/flashhead-media-jRaXOE`.
Selected recordings and JSON are in `benchmarks/media_20260913/`.
Private launch snapshots remain outside git and contain environment secrets.
Never reuse their historical PIDs or publish their contents.

## Implemented changes

### Shared bounded CPU idle cache

`soulx_rtc/idle_cache.py`, integrated through `Service` and `IdleVideo`:

- One immutable RGB clip shared by all callers of the same content/profile.
  Keys hash the exact compressed bytes decoded, plus width/height/format.
- Native source FPS and independent per-peer indices preserve the existing
  24→25 FPS mapping, looping and rewind. The first cached frame is also the
  canonical anchor. No body/face pixels are modified by caching.
- A 512 MiB default payload budget, LRU eviction of unused clips, and pinned
  active leases. A serialized loader reserves bytes before allocating frames;
  oversized clips or a fully pinned budget fall back to the existing decoder.
- The budget covers retained/staged cached RGB, not all Python/codec/queued
  media overhead. Compressed input is separately bounded to 64 MiB, with small
  transient decode/resize allocations. The production clip is 241 frames and
  occupies 133,263,360 bytes (about 127.1 MiB), shared rather than per caller.
- Default idle is prewarmed before readiness. Cold loads run on a dedicated
  loader executor, not the event loop. Cancellation releases late acquisitions;
  closing a peer releases its pin exactly once.
- Rollback: `--idle-cache-mib 0` retains the streaming decoder.

### Bounded shared-memory RGB transport

`soulx_rtc/shared_chunks.py` and `worker.py` replace pickle transmission of whole
RGB chunks with one fixed B×24×H×W×3 shared slot. At batch one / 320×576 the slot
is 12.66 MiB. The parent serializes generate requests and copies returned rows
to independently owned arrays **before** releasing the slot. Queued/slow media
consumers never retain borrowed shared storage. This intentionally keeps a safe
parent copy rather than claiming a zero-copy encoder ring.

Descriptors and shapes are checked, only valid tail frames are copied, the
parent owns unlinking after worker shutdown, and initialization failure closes
the transport. Interruption discards stale epochs without giving another turn
the old shared buffer. Worker and parent copy times are exposed separately.
Rollback: `--chunk-transport pickle`.

### Media CPU work and process separation

Audio validation/decoding and 48-kHz PCM resampling run in a dedicated four-thread
media executor; resampling no longer blocks every peer on the RTP event loop.
Fallback idle decoding uses that executor, while full cache loads use their own
single loader. Cached playback is an in-memory lookup without an executor hop.

Portrait preparation now uses the exact PIL resize/center-crop branch directly,
avoiding an unnecessary Torch/torchvision import and tensor round trip in the
RTP process. Tests cover odd/even crop rounding against the model utility.
Torch remains in the GPU worker; requesting the existing in-process Kokoro TTS
can still load Torch in the parent and is a separate qualification case.

The benchmark now supports recording selected idle peers as well as a speaker,
sorts creation results by peer index, and serializes MP4 mux/encoding in one
dedicated executor per recording rather than blocking its receive event loop.
The recorded client still runs on the same host, so this is not a server-only
or remote-browser load test.

### Pause telemetry and optional startup GC policy

With `--profile`, health retains bounded event-loop lag and GC-duration samples.
A late-frame run measured a generation-2 GC pause of ~55 ms alongside a ~66 ms
event-loop lag. The optional `--freeze-startup-gc` first collects and then
freezes the long-lived startup heap before accepting callers. New call objects
retain normal cyclic GC; collection is not disabled. This is a process-wide,
standalone-server option, default off, and unfreezes on service cleanup.

## Correctness and regression checks

- Cached versus legacy production-video decoding: **303 frames**, including a
  complete loop and rewind, at **320×576**, maximum RGB error **zero**.
  `cache-pixel-validation.json` records the source/frame count and cache bytes.
- Unit tests cover immutable frames, independent peers, pinning, eviction,
  budget fallback, content invalidation, deduplicated concurrent loads, late
  cancellation cleanup, shared-buffer ownership/unlinking, descriptor errors,
  and new-cycle collection after startup freezing.
- Existing peer, boundary, recording, retry, interruption and cleanup tests
  remain in the suite. The optional Chromium wall test is not enabled by the
  ordinary CPU suite. Actual GPU/H264/Opus tests are reported separately.
- Final CPU regression suite: **72 passed, one skipped in 40.49 seconds**.
  The skipped test is the opt-in browser test, not a passing Safari test.

## Measurements and retained unsuccessful trials

Same machine, RTX 4070, 320×576, four steps, 25 FPS, compiled resident compact
INT8-storage profile, 3,584 MiB Torch cap, ten connected / one active limit.
The other GPU service remained running (about 6,030 MiB at initial inspection).
Two ten-second uploaded WAV turns per ten-peer test; one speaking girl, seven
other still-avatar peers and two video-idle peers. Both speaker and a video-idle
peer are recorded. No denoising/color/compiler quality shortcut is enabled.

| Run | Video-idle FPS | Reported underruns | Interpretation |
| --- | --- | ---: | --- |
| Fresh original service | 11.61 / 11.66 | 14 | Transport passes, smoothness fails |
| Cache + shared memory + media executor | 23.78 / 23.74 | 12 | Large delivery gain, still fails strict pacing |
| Above, async recording harness | Around 24.6 in the recorded idle peer | 13 | Recording no longer blocks the receiver loop; still has common stalls |
| Above, Torch-free portrait preparation | 24.82 / 24.89 | 3 | ~249 aggregate wire FPS; small late-frame count remains |
| Selected startup GC policy, round 1 | 24.96 / 25.03 | 0 | 250.07 aggregate wire FPS; arrival gaps still fail strict pacing |
| Selected startup GC policy, round 2 | 24.98 / 24.99 | 2 | 249.88 aggregate wire FPS; strict ten-peer smoothness still fails |

The first baseline used the earlier synchronous recorder. Later runs include
both server and recording-harness changes; do not interpret the comparison as
a tightly randomized isolated server-only A/B. Failed/intermediate artifacts
are retained, not overwritten by the final trial.

Mean final-ten-chunk RPC residual dropped from about **161 ms** in the fresh
baseline to about **21 ms** with shared memory. This residual includes queueing,
copying and parent scheduling, not pure physical memcpy. Cached idle lookup
averaged roughly **0.01 ms**, versus **57–59 ms** awaited legacy decoding under
the baseline wall load. Raw neural FPS is intentionally not rebranded as higher.

## Recording validation scope

The earlier single-call candidate recording contains **946 frames**, two full
ten-second turns, an interrupted turn and recovery. It measured **25.006 wire
FPS**, no missed video slots or reported underruns, strictly increasing PTS,
maximum presentation interval **40 ms**, maximum arrival gap **62.8 ms**, zero
whole-frame blackouts, and zero identical-frame runs during completed speech.
Exact sender boundary verification passed all nine checks; resource cleanup
passed. A sparse contact sheet covering the clip shows coherent identity and
framing without obvious blank/corrupt frames. Model mouth/teeth detail remains
soft in some sampled frames; this is not a claim of flawless generative quality.

`validate_recording.py` inspects all decoded frames for blackouts, large frame
changes and exact repeats during completed speech; presentation/arrival timing
comes from the real receiver. Thresholds are documented in the driver. These
checks plus visual frame review detect major corruption and pacing failures,
not every possible lip-sync, identity, texture or perceptual artifact. Boundary
hash equality is **sender RGB**, not post-H264 pixel equality. No Safari/mobile,
WAN/TURN, arbitrary-voice or long-duration production qualification is implied.

## Final validation / release

The selected service uses the unchanged compiled 320×576/four-step renderer,
`--idle-cache-mib 512 --chunk-transport shm --freeze-startup-gc --profile`.
Color/audio compilation remains off. Model storage is INT8, compute is BF16;
the Torch allocator cap is 3,584 MiB. Connected/active admission remains 10/1.

Fresh baseline aggregate wire FPS was **219.96**; the two selected ten-peer
runs measured **250.07** and **249.88**: approximately **13.6% higher** on average.
The video-idle peers improved from approximately **11.6 to 25 FPS** (about 2.15×).
Mean final-ten-chunk RPC residual fell from **160.80 ms** to **19.59 / 21.86 ms**
(about 87% lower). These are bundle measurements, not a causal attribution of
the entire gain to one change. Engine chunk wall times were 458.21 ms baseline
and 429.97 / 431.74 ms selected; no isolated neural-kernel speedup is claimed.

Both ten-peer runs passed transport and cleanup, with **zero / two underruns**.
They still had occasional **157–197 ms receiver arrival gaps** and 1–3 missed
video slots per peer. The round-one speaker and video-idle recordings passed
structural checks but **failed strict arrival pacing**. Newly recorded GC
pauses were below 6 ms in these runs; this does not explain every remaining gap.
Ten-connected smoothness is therefore **not fully qualified**.

The final selected-profile [single-call recording](../../benchmarks/media_20260913/single-release-peer0.mp4)
contains **945 frames**, two ten-second turns, an interruption and a completed
recovery turn. It measured **24.977 wire FPS**, **zero underruns**, one missed
video slot, a maximum presentation interval of **80 ms**, and a maximum arrival
gap of **100.49 ms** (p95 **43.93 ms**). It passed the driver's explicit pacing
thresholds (at least 24.5 FPS, zero underruns, presentation gap ≤80 ms and arrival
gap ≤150 ms). This is not a claim that every frame arrived exactly 40 ms apart.
There were zero black frames, zero identical-frame runs during completed speech,
monotonic PTS and no whole-frame cut above the validator's MAD threshold.
All **nine sender boundary checks** and resource cleanup passed. The
[contact sheet](../../benchmarks/media_20260913/single-release-contact.jpg)
shows coherent identity/framing without obvious corruption; sparse frame review
does not certify artifact-free motion or objective lip-sync quality.

### Retained release evidence

- [Summary and source hashes](../../benchmarks/media_20260913/summary.json).
- [Round one](../../benchmarks/media_20260913/candidate-gc-r1.json) and
  [round two](../../benchmarks/media_20260913/candidate-gc-r2.json), with MP4s and
  compressed receiver timestamps alongside them.
- [Single-call report](../../benchmarks/media_20260913/single-release.json) and
  [frame/pacing validation](../../benchmarks/media_20260913/single-release-video-check.json).
- [Ten-peer speaker check](../../benchmarks/media_20260913/candidate-gc-r1-video-check.json)
  and [video-idle check](../../benchmarks/media_20260913/candidate-gc-r1-idle-video-check.json).
- [Exact idle cache parity](../../benchmarks/media_20260913/cache-pixel-validation.json).
- `candidate-final.json` is the retained pre-GC intermediate, **not** the final
  selected deployment despite its historical filename. The original baseline
  and earlier failed trials remain in the experiment directory named above.

Reproduce recording validation without reloading models or restarting services:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/pipeline_20260911/validate_recording.py \
  benchmarks/media_20260913/single-release.json --output /tmp/fresh-video-check.json
PYTHONPATH=. .venv/bin/python -m soulx_rtc.verify_boundaries \
  benchmarks/media_20260913/single-release.json
CUDA_VISIBLE_DEVICES= PYTHONPATH=. .venv/bin/python -m pytest -q
```

The validator requires a fresh output path. New live runs use
`python -m soulx_rtc.benchmark_calls --help`; use a fresh output basename,
`--sessions 10 --speakers 1 --avatars portrait:girl portrait:man portrait:yongen default`
and `--record --record-peers 0 3` with the retained ten-second WAV fixture.

### Remaining optimization and migration gates

Larger GEMM/VAE kernel work and true quantized GEMM are **not implemented in this
round**. Two active speakers and ten active speakers remain unqualified. Next:
isolate remaining parent/receiver scheduling and encoder stalls with a remote
receiver; qualify Kokoro TTS under load and longer runs; then compare GEMM/VAE
candidates with identical neural dimensions, steps, assets and quality gates.
The media gain does **not** establish faster neural rendering than matched
MuseTalk, a 25% whole-model gain, or readiness for production migration.
