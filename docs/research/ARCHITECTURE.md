# How SoulX-FlashHead Lite works in this fork

Historical source snapshot. Native rectangular generation, conditioning caches, optional TensorRT partitions and a persistent call API have since been implemented; see [implementation status](IMPLEMENTATION_STATUS.md) and [current call contract](../../CONTINUOUS_WEBRTC.md). The checkpoint's temporal/endpoint limitations remain.

Snapshot before this research change: SoulX fork `2f76e830e92f054e34480ed860f71792551b5e01`; upstream base `9bc03de06bb0de82cd6bc477804512ae06144bf2`. MuseTalk comparison checkout: `e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e`. Analysis concerns the installed **Lite** checkpoint, not Pro, teacher, another FlashTalk model, or upstream multi-GPU benchmarks.

## Model and data flow

```text
portrait ── resize / center crop ── reference VAE latents ───────────┐
audio ── rolling 8 s Wav2Vec2 window ── per-frame audio windows ─────┤
private motion prefix + private noise ── four DiT steps ── latents ─┘
                                                          │
                       LTX VAE decode: 33 whole RGB frames
                                                          │
                           per-frame Lab color correction
                                ├─ last 9 frames → VAE encode → next motion prefix
                                └─ drop first 9 → 24 new frames → GPU uint8
                                                                   │
                   bounded CPU IPC → WebRTC H264 video + Opus audio
```

The arrows into the denoiser represent conditioning; the portrait is not pasted unchanged into its output.

## Source map

| Source | Responsibility / important detail |
| --- | --- |
| [flash_head/inference.py](../../flash_head/inference.py) | Pipeline selection; original inference entrypoints. |
| [FlashHeadPipeline](../../flash_head/src/pipeline/flash_head_pipeline.py) | Model/VAE/audio loading, portrait preparation, timesteps, recurrent inference. |
| [WanModelAudioProject](../../flash_head/src/modules/flash_head_model.py) | Audio-conditioned DiT, 3-D RoPE, self-attention, audio cross-attention and MLPs. |
| [Wav2Vec2 wrapper](../../flash_head/audio_analysis/wav2vec2.py) | Resamples feature sequence to requested video cadence before transformer encoding. |
| [LtxVAE](../../flash_head/ltx_video/ltx_vae.py) | Per-channel latent normalization and video encode/decode wrapper. |
| [VAE implementation](../../flash_head/ltx_video/models/autoencoders/causal_video_autoencoder.py) | Conv3d residual blocks, spatial/temporal resampling, patchify/unpatchify. |
| [Checkpoint config mapping](../../flash_head/ltx_video/utils/diffusers_config_mapping.py) | Maps installed Diffusers-style config to local VAE; pixel norm and **non-causal decoder**. |
| [Color and crop utilities](../../flash_head/utils/utils.py) | RGB↔Lab conversion, reference statistics, per-frame correction, resize/center crop. |
| [Engine](../../soulx_rtc/engine.py) | Shared modules, private session views, chunk microbatches and useful-frame accounting. |
| [GPU worker](../../soulx_rtc/worker.py) | One spawned process owns CUDA and session state; only CPU arrays cross IPC. |
| [RTC service](../../soulx_rtc/server.py) | Admission, uploads, fair scheduling, bounded queues, clocks, tracks and cleanup. |
| [RTC benchmark](../../soulx_rtc/benchmark_rtc.py) | Actual receiver frames/timestamps/RTP stats and optional recording. |

## Exact installed tensor contract

The checkpoint under `models/SoulX-FlashHead-1_3B/Model_Lite/config.json` has width 1536, 30 blocks, 12 heads (128 dimensions/head), feed-forward width 8960, 128 output channels, 256 input channels and patch size `(1,1,1)`. VAE stride is `(8,32,32)` in time/height/width.

For square 512 input:

| Tensor | Shape / interpretation |
| --- | --- |
| Reference video | `[1,3,33,512,512]`, repeated conditioning portrait |
| Normalized reference | `[128,5,16,16]` |
| DiT noisy input `x` | `[B,128,5,16,16]` |
| Reference input `y` | `[B,128,5,16,16]`; concatenated by channel |
| Patchified sequence | `[B,1280,1536]`, since `5×16×16=1280` |
| Audio context | `[B,33,5,12,768]` before projection |
| Projected audio | `[B,5,32,1536]` |
| Decoded video per session | `[1,3,33,512,512]` |
| Emitted RGB chunk | `[24,512,512,3]` uint8, except final truncated chunk |

Some upstream inline shape comments describe other variants (e.g. temporal stride four). The installed config and executable rearrangements, not those stale comments, define Lite's stride eight.

`has_image_input=false` disables the separate CLIP-like image-attention branch; it does **not** make the model unconditional on images. The repeated reference latent is still concatenated with noise as `y`.

## Temporal inference and frame accounting

Four base timesteps `[1000,750,500,250,0]` are transformed by the shift-five schedule. For each step, each session's motion prefix replaces the corresponding initial latent positions; the DiT predicts flow; the code forms the next sample with the next timestep and newly generated private noise. This is not MuseTalk's one-pass UNet face update.

At initialization, history contains **one reference latent frame**. Later, the last nine color-corrected decoded frames are encoded and sampled into **two motion latents**. Those two latents condition the next 33-frame chunk. The first nine decoded frames are discarded from transport, leaving 24 newly emitted frames. This fork discards nine even on the first chunk; its first visible video frame is not a literal copy of the supplied portrait.

At 25 FPS, a useful chunk covers 0.96 seconds. A ten-second request needs 250 useful frames and `ceil(250/24)=11` chunks: 363 decoded frames, of which 99 are overlap and 14 are final unused output. Report **250** productive frames, not 363. Steady-state decode overhead is `33/24=1.375`; that ratio is not a guaranteed achievable speedup because temporal context is part of the learned computation.

Crucially, the installed LTX checkpoint has `decoder_causal=false`, `encoder_causal=true`; the local mapping uses `causal_decoder=False`. The class name `CausalVideoAutoencoder` does not prove its decoder is causal. Trimming latent time or caching decoder activations without a receptive-field analysis can alter frames that remain visible. Directly carrying denoiser latents also differs from encoding the **color-corrected decoded** nine-frame history used today.

## Audio context and latency

`Engine._audio()` constructs an eight-second, 16-kHz window ending at the next chunk's audio horizon. Initial history and audio beyond clip end are zero padded. Wav2Vec convolutional features are linearly interpolated to `8×fps`, then passed through its transformer; twelve hidden-state layers feed five-frame windows around each video frame. The current 33-frame conditioning slice contains nine history-frame contexts plus 24 future-to-the-cursor output contexts.

“Causal rolling” here means no samples after the current **chunk horizon** are read. It does not mean sample-by-sample causal Wav2Vec attention: tokens inside that eight-second window can attend within the window. Whole-file feature precomputation would change available context and interpolation. A transformer KV cache copied from an autoregressive language model is therefore not an equivalent replacement.

The HTTP service currently receives a complete audio upload. True incremental TTS must buffer enough audio for the selected chunk/lookahead policy, including turn boundaries. The measured ~one-second first-video time excludes upstream ASR/LLM/TTS and session upload/preparation; the benchmark starts its timer at the offer barrier. It is not speech-to-speech call latency.

## Low-VRAM serving already implemented

The fork has already fixed two upstream batch-one assumptions: RoPE preserves all batch rows, and audio cross-attention reshapes `(batch × temporal frame)` correctly. Shared weights do not imply shared motion history. Each session owns its pipeline view, `torch.Generator`, audio cursor and motion prefix; VAE posterior sampling uses that generator too. Reference templates use a content-hash/profile key and an eight-entry LRU. Eviction drops the cache reference, while active sessions retain their tensors.

The DiT runs a microbatch across independent sessions. VAE decode/color/re-encode runs serially by session to limit peak memory. All CUDA operations run in one spawned owner process; preparation and generation use that same serialized executor. CPU media callbacks cannot simultaneously mutate the model. This avoids the GIL interference observed in the earlier thread-only experiment, but a slow new-portrait preparation can still delay active generation.

`torch.compile` is already enabled for the DiT and VAE encode/decode. Only new frames are converted to uint8 on the GPU and copied to the CPU. Do not propose these existing features as new speedups.

At 512², one 24-frame RGB chunk is 18 MiB. Two queued chunks plus the consumer's current chunk are roughly **54 MiB per session**, before encoder/IPC copies and audio; ten peers can therefore consume hundreds of MiB of host memory even though GPU weights are shared. Queues are bounded and full sessions are skipped, not awaited by the shared scheduler. At 25 FPS, three chunks represent up to 2.88 seconds of generated video ahead of playout—important for interruption and state rollback.

The latest warm worker reserved about 5514 MiB in PyTorch and appeared as 5712 MiB in `nvidia-smi`. These are different accounting scopes. Total device usage was about 11876/12282 MiB because an unrelated service used 6150 MiB. No additional full-model process or TRT build was started on that nearly full card.

## Current API is a clip service, not the MuseTalk call protocol

Actual routes: `GET /`, `/health`, `/config`; `POST /sessions`; `GET/DELETE /sessions/{sid}`; `POST /sessions/{sid}/offer`. Uploads accept `image`, `audio`, `seconds`, `seed`. Limits include 30-second clips, bounded upload/image sizes and a reservation before asynchronous preparation. Non-loopback CLI binds require a bearer token.

One offer installs H264 and Opus tracks. A shared origin gates audio behind generated video. Starvation slows both media clocks rather than creating fake productive frames. At clip end, one video timestamp and five silent audio packets drain receiver buffers; tracks then end. Sessions expire after inactivity. There is **no** append-turn endpoint, persistent idle track, semantic pose plan, audible-EOF event contract, source-video conditioning input, or native `last_frame` parameter.

`max_sessions=10` limits admitted objects, not ten real-time generations. The application-level token is not per-user authorization. Public HTTPS/TURN, worker health after a dead subprocess, ownership-safe cancellation during preparation, idle lifetime, disconnect cleanup and long-call resource bounds need additional production hardening.

## Why copying MuseTalk verbatim will not work

MuseTalk's `scripts/hls_gpu_scheduler.py::_run_generation_batch` gathers independently prepared face latents from multiple jobs/poses into a frame batch, runs UNet/VAE, trims padding, then restores ordered composition. Each pose's source frame, face geometry, mask and conditioning must correspond. SoulX's scheduler unit is a recurrent 24-frame chunk, and the next chunk of a single session cannot be generated independently before its history exists.

MuseTalk's approved full canvas is 480×832 while the network face crop remains 256×256. SoulX's tested output is 512×512. A hypothetical native 480×832 profile has `15×26=390` spatial latent cells versus 256: 1.523× as many tokens and potentially ~2.32× dense attention score work. That is an architectural estimate, not measured latency; the current service only permits square sizes. Letterboxing a square render changes presentation, not model throughput or preserved original motion.
