# RTX4070: SoulX-FlashHead versus MuseTalk

## Outcome

A functioning shared-model H264/Opus WebRTC service was implemented. **A
quality-comparable ten-speaker real-time replacement for MuseTalk has not been
demonstrated.** Native SoulX is compute-limited, not session-VRAM-limited. A
reduced profile delivered ten concurrent streams but failed visual quality.

The final reference service is running locally at **http://127.0.0.1:8765**.
The earlier Gradio process was stopped to avoid keeping a second model on GPU.
External/TURN hosting is configurable but has not been provisioned or tested.
For a practical starting point, use **one active 25-FPS reference-quality session**
on this GPU; queue additional jobs rather than promising ten simultaneous speakers.

The service defaults to 512², four denoising steps and 25 FPS. The 256²,
two-step, 15-FPS configuration is retained only as an explicitly rejected
quality experiment. This is not full MuseTalk product/API parity.

## Machine and method

- NVIDIA RTX4070, 12,282 MiB VRAM, driver 570.181, 200-W power limit.
- SoulX: PyTorch 2.7.1+cu128, BF16, FlashAttention 2, compiled DiT and LTX VAE.
- MuseTalk: existing Python environment, PyTorch 2.5.1+cu121, FP16. The saved
  TensorRT VAE engine is incompatible with this machine; it was not reused.
- OmniVoice remained resident (~2,194 MiB) and was not modified or stopped.
  GPU benchmarks ran sequentially, not with MuseTalk and SoulX generating together.
- SoulX upstream revision: `9bc03de06bb0de82cd6bc477804512ae06144bf2`, plus local changes.
- Same portrait and first ten seconds of the included 16-kHz audio for engine
  comparisons. Audio SHA256: `0bbc0e4d1f1e4ecaad1f425e311e8f3ced2d1011d65a30b46487389331b456b2`.
- Independent seeds and shifted audio per engine job; no generated-frame fanout.
  Ten-session RTC tests use the same portrait/audio and distinct generation seeds.
- Warm model/reference caches; cold loading and compilation excluded from
  throughput. Compilation/warmup can take several minutes and is completed
  before the service listens. Audio encoding, denoising, VAE work, overlap/tail
  padding and CPU-ready frame conversion are included in engine timings.
- MuseTalk additionally includes actual mask-based avatar composition at 512².
  Its neural face region is 256², so this is an application-throughput comparison,
  not equal neural work or equivalent visual quality.
- Torch memory numbers below are allocated/reserved within the model process,
  not total device use or a strict NVML per-process peak.

## Warm ten-second generation

All 25-FPS rows charge the complete work needed for 250 useful frames/session.
SoulX produces 24 new frames/chunk after discarding nine context frames; eleven
chunks are required. Extra tail/context work is charged but never counted as
additional useful frames.

| Engine/profile | Jobs | Useful frames | Wall seconds | Aggregate generated FPS |
| --- | ---: | ---: | ---: | ---: |
| MuseTalk eager FP16, batch 4 | 1 | 250 | 8.37 | 29.87 |
| MuseTalk eager FP16, batch 4 | 10 | 2,500 | 83.56 | 29.92 |
| MuseTalk compiled UNet + VAE decoder, batch 8 | 1 | 250 | 6.75 | 37.04 |
| MuseTalk compiled UNet + VAE decoder, batch 8 | 10 | 2,500 | 57.12 | 43.76 |
| SoulX 512² / 4 steps, batch 1 | 1 | 250 | 6.12 | 40.88 |
| SoulX 512² / 4 steps, batch 1 | 10 | 2,500 | 61.49 | 40.66 |
| SoulX 512² / 4 steps, batch 2 | 10 | 2,500 | 60.90 | 41.05 |
| SoulX 256² / 2 steps, batch 2, 25 FPS | 10 | 2,500 | 16.12 | 155.09 |

**Native SoulX did not beat the stronger tested MuseTalk baseline:** 41.05 versus
43.76 aggregate FPS for ten jobs. The eager-only comparison would have given a
misleadingly favorable conclusion. These are single measured sweeps, not a
statistically significant claim about a small architecture-level speed difference.
The compiled MuseTalk sweep warmed a full clip first; residual batch-shape
transition costs (including the final partial batch) remain charged to wall time.
The attempted compiled `decode` wrapper failed under the older diffusers/torch
stack; the comparison helper instead compiles the actual VAE decoder submodule.
No installed MuseTalk source or launcher was changed. Compiled MuseTalk's
ten-job run reserved 6,384 MiB, versus ~5,508 MiB for native batch-1 SoulX.

Native batch-1 SoulX reserved approximately **5,508 MiB**, with essentially no
growth between one and ten shared-model sessions. Its representative chunk cost
was ~15 ms audio, ~306 ms DiT, and ~232 ms VAE/color/transfer. Batching two sessions
barely improved useful throughput. Ten 25-FPS streams require **250 generated FPS**;
~41 FPS is about 1.6 such streams, not ten.

## Real WebRTC delivery

The first thread-only GPU implementation performed much worse online than in
the isolated engine benchmark: ten 15-FPS peers accumulated roughly 7–10 seconds
of stalls per ten-second clip. The dedicated spawned GPU process reduced
representative batch time from ~0.6 seconds to ~0.27 seconds. This supports
Python dispatch contention as a major cause; no kernel profiler is claimed.

The receiver test actually negotiated ICE/DTLS, received SRTP, and decoded H264
and Opus. It excluded terminal jitter-buffer-drain packets from productive counts.

| Isolated-process profile | Peers | Content/peer | Useful frames received | Total wall time | Playback stalls |
| --- | ---: | ---: | ---: | ---: | --- |
| 512² / 4 steps / 25 FPS | 10 | 10 s | 2,500 / 2,500 | 69.09 s | ~51.30–51.90 s per peer |
| 256² / 2 steps / 15 FPS | 10 | 10 s | 1,500 / 1,500 | 13.04 s | Eight zero; two ~0.189 s |
| 256² / 2 steps / 15 FPS | 10 | 30 s | 4,500 / 4,500 | 33.09 s | ~0.204–0.232 s per peer |

The 30-second run delivered approximately **14.88–15.00 useful FPS per peer after
startup**, not 25 FPS. Wall totals include connection/startup, differing first
frame times, and a one-second packet-drain observation period. They must not be
relabelled as unpaced model throughput. The ten-second run's first received
frames ranged from ~0.63 to 2.09 seconds. All requested audio samples arrived.

The tested low-cost process reserved **4,888 MiB**. A 250-ms startup cushion was
subsequently added to absorb small initial scheduling bursts; the rows above
are retained as measured and are not presented as tests of that later change.

The final reference-profile ten-peer test **did include** that startup cushion.
Every peer received all 250 useful video frames and all 480,000 audio samples,
with zero reported RTP packet loss and ten distinct first-frame hashes. First
received video ranged from 1.10 to 6.79 seconds. It took 69.09 seconds including
the final observation/drain interval. This directly demonstrates reliable local
delivery under overload, **not real-time operation**: each ten-second stream
accumulated over 51 seconds of playback stalls.

These are short loopback tests on this machine, not sustained Internet/TURN
capacity, a multi-hour reliability test, or a lip-sync quality score.

## Visual quality: the concurrency shortcut is rejected

The first received low-cost frame looked coherent, but later frames developed
severe facial smearing/noise. Inspect the [received ten-second video](rtc-256-2-15-sample.mp4),
[three-second frame](rtc-256-2-15-sample-3s.png), and [five-second frame](rtc-256-2-15-sample-5s.png).
The MP4 contains exactly 150 frames and ten seconds of audio/video; delivery
success did not imply acceptable generation quality.

The earlier upstream 512²/four-step reference remains visually coherent at the
same point: [reference frame](reference-512-4-3s.png). This is a qualitative
observation, not a blinded quality or SyncNet evaluation. Resolution, step count,
and frame timeline were changed together in the rejected profile; this experiment
does not establish which individual change caused the failure.

The final custom service was also recorded and inspected at reference quality:
[received 512²/25-FPS video](rtc-512-4-25-sample.mp4) and
[its five-second frame](rtc-512-4-25-sample-5s.png). The face remained coherent.
ffprobe confirms exactly 250 frames and ten seconds of video/audio. The single
recording received its first frame at ~1.00 seconds and reported ~55 ms of stall;
recording adds CPU overhead and was not used as the capacity benchmark.

## Implementation and validation

- Shared immutable model/reference tensors, per-session motion latents and RNG,
  including VAE posterior randomness; no concurrent mutation of one pipeline.
- Fixed batch-dimension bugs in upstream RoPE and audio cross-attention.
- GPU-only uint8 conversion after recurrent motion encoding; no overlap-frame transfer.
- Fair chunk scheduling and bounded queues; a full client cannot block others.
- Dedicated GPU process; no CUDA tensor sharing across processes.
- H264/Opus media, common clock, cancellation, admission limit, upload bounds,
  token authentication for non-loopback binds and configurable ICE servers.
- Nine automated CPU tests, including real local WebRTC media, exact final-frame
  and audio delivery, timestamp ordering, cancellation, queue isolation, API auth,
  admission, and batched versus separate attention/RoPE computations.
- Actual GPU two-session/two-chunk permutation test: **zero pixel difference**.
- Received MP4 validated with ffprobe: 256², 15 FPS, 150 frames, ten-second audio/video.
- Reference MP4 likewise validated: 512², 25 FPS, 250 frames, ten-second audio/video.
- Two different uploaded portraits also passed a three-second media smoke test:
  each received all 75 useful frames and 144,000 audio samples. This is not a
  ten-distinct-identity capacity test or a long-duration visual assessment.
- `pip check` retains the existing decord wheel-platform warning; the exercised
  inference and WebRTC paths do not use decord. All other checked dependencies resolve.

See [operation and reproduction instructions](../WEBRTC.md),
[architecture review](ARCHITECTURE_REVIEW.md), and the adjacent JSON files for raw evidence.

Final idle native-worker NVML footprint was about 5,712 MiB; Torch's final
reference-profile peak reserved memory was 5,514 MiB after the two-avatar test.
OmniVoice remained separately resident at 2,194 MiB. MuseTalk's tracked worktree
was unchanged; only the dedicated comparison avatar cache was added locally.

## What this does not establish

No claim is made of ten quality-equivalent 25-FPS sessions, a faster-than-TensorRT
MuseTalk result, production lip-sync quality, browser interoperability beyond
the provided client, external NAT/TURN reliability, persistent multi-turn peers,
MuseTalk pose/TTS/storage/control-plane parity, or speed on another GPU.

Native quality at ten-way concurrency would require substantially more model
throughput or a validated acceleration/quality technique. Merely sharing weights,
lowering VRAM, adding connections, or increasing prebuffer cannot supply that compute.
