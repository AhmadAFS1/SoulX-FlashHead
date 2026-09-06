# MuseTalk → SoulX-FlashHead architecture review

Reviewed on 2026-09-06 against the locally installed repositories. This review
is grounded in the local code, not historical performance reports from other GPUs.

## What carries over

MuseTalk's API (`/workspace/MuseTalk/api_server.py`) separates avatar preparation,
WebRTC peer creation/offer, complete-audio uploads, and media playout. Its
`scripts/hls_gpu_scheduler.py` serves both HLS and WebRTC: multiple jobs contribute
independent face-frame rows to one UNet/VAE batch. Composition runs in worker
threads and outputs are restored to frame order. `scripts/webrtc_tracks.py`
provides the shared playout gate, 90-kHz video timestamps and 48-kHz/20-ms audio.

The new SoulX service carries over shared-model ownership, per-session queues,
fair scheduling, independent media senders, H264/Opus, signaling, cancellation
and observable latency. It changes the scheduling unit to a 24-frame temporal
chunk, because each chunk must complete motion re-encoding before that session
can advance. It does not reuse the MuseTalk frame scheduler verbatim.

## Why the generation engines differ

| Area | MuseTalk local implementation | SoulX-FlashHead Lite |
| --- | --- | --- |
| Neural output | 256² face patch | Whole generated frame, normally 512² |
| Avatar cache | Background frames, masks, coordinates, face latents, blending geometry | Reference image/latents |
| Per-frame/chunk work | One UNet pass, SD-VAE decode, composite | Two/four DiT steps, LTX-VAE decode, color matching, motion re-encode |
| Temporal dependency | Face rows can be independently combined across jobs | Recurrent motion latents and RNG belong to each session |
| Audio | Whisper features prepared before a job | Rolling eight-second Wav2Vec context per chunk |
| Concurrency | Shared frame batches and prepared motion | Shared DiT chunk batches; sequential VAE for lower peak VRAM |

Equal 512² output containers do not mean equal neural work or equal visual
quality. Reduced SoulX resolution/steps/FPS are additional quality compromises.

## Local MuseTalk findings relevant to copying it

1. There were no prepared avatar materials under `results` at the start of this
   task. Source videos and older JSON reports are not caches. A fresh local
   `soulx_comparison_4070` cache was prepared from SoulX's example portrait.
2. The saved TensorRT VAE engine failed deserialization on this 4070 in the
   prior startup. The comparison therefore explicitly selects PyTorch; the
   normal TRT launcher and its saved settings were left unchanged.
3. `scripts/benchmark_pipeline.py` uses synthetic conditioning and separately
   sums stage timings. It excludes Whisper, real composition and WebRTC. The
   new comparison helper instead measures actual ten-second audio, UNet, VAE,
   CPU transfer and composition, with independent round-robin jobs.
4. MuseTalk's queue callback can synchronously wait up to 30 seconds for media
   delivery (`api_server.py`, callback near the WebRTC stream submission).
   A slow session can therefore hold the shared scheduler. SoulX's scheduler
   never waits for a full queue: it skips that session and serves another.
5. MuseTalk's default timestamp-locked selection can repeat the last image on
   underrun. Its output FPS can also exceed source generation FPS. Receiver
   frame counts alone are therefore not proof of productive model throughput.
6. The default strict queue permits 400 YUV420 frames (~150 MiB at 512² per
   session). SoulX uses two uint8-RGB chunks plus the chunk being consumed,
   rather than retaining all generated frames.
7. `scripts/avatar_cache.py` cleanup deletes fields on shared avatar objects
   without active-job leases. The new reference cache only drops its own
   reference; existing sessions continue owning their immutable tensors.
8. MuseTalk's `WEBRTC_H264_ENCODER` hook patches a factory that aiortc 1.14 no
   longer uses for H264 context creation. Setting it to NVENC is not evidence
   that NVENC is active. The new service explicitly uses aiortc's libx264 path.
9. MuseTalk supports much more product functionality: persistent turns, idle
   and live poses, pose plans, TTS hooks, group APIs, remote avatar persistence,
   and a worker control plane. Those are not claimed as implemented in this
   bounded-clip experimental SoulX service.

## SoulX implementation and correctness fixes

The upstream RoPE operation indexed `x[0]`, and the audio attention block used
`context.squeeze(0)` followed by a batch-flattened output. Both assumed one
session. They now preserve batch rows and the `(batch × frame)` mapping. Tests
compare batched versus separate attention computations. Multi-GPU USP batching
is deliberately rejected; this service owns one GPU.

Each session has a shallow pipeline view with its own generator and motion
history; loaded modules and reference tensors are shared. Initial history has
one latent frame, later history two. Prefix replacement is done separately for
each row rather than padding and overwriting both. VAE posterior sampling uses
the session generator too. All initialization and generation run on one GPU
executor in a dedicated spawned process, avoiding races on shared modules and
caches. The first thread-only WebRTC implementation suffered substantial
dispatch slowdown under ten peers despite good offline throughput. The process
boundary removes Python media callbacks from the GPU worker's interpreter/GIL;
only bounded CPU uint8 chunks pass through IPC. Initial thread-mode evidence is
retained under the `rtc-*-initial.json` filenames, not silently overwritten.

Color matching and recurrent motion encoding finish before conversion. Only
the newly emitted frames are converted to uint8 on GPU and transferred to host;
the nine overlap frames never enter the transport queue. Compilation and model
loading are measured separately from warm throughput.

## External claims versus local evidence

Upstream advertises Lite at 96 FPS / three 25+ FPS streams on an RTX4090—not ten
streams on this 4070. See the [official repository](https://github.com/Soul-AILab/SoulX-FlashHead).
Its optional SageAttention dependency is not installed in these measurements;
this build uses FlashAttention 2. No untested quantized workflow or another
model's FlashTalk benchmark is substituted for local FlashHead measurements.

aiortc 1.14 permits PyAV 15/16, so the service pins compatible aiortc 1.14.0 and
PyAV 16.1.0 in its own environment. See the [official changelog](https://aiortc.readthedocs.io/en/latest/changelog.html).

Public-network hosting remains separate from loopback media validation: ICE
reachability, selected candidate pair, TURN load, TLS, packet loss, reconnects,
and long-lived production operation need deployment-specific testing.
