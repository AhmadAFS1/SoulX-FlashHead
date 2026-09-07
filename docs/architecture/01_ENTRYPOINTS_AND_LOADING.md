# 1. Entrypoints, configuration and model loading

[Guide index](README.md) · [Next: audio and DiT](02_AUDIO_AND_DIT.md)

## There are three applications, not one serving implementation

| Entry | Purpose | State and output | Important boundary |
| --- | --- | --- | --- |
| [generate_video.py](../../generate_video.py) / inference shell scripts | Upstream offline inference | One pipeline; collects generated video and muxes audio with ffmpeg | Not the persistent-call scheduler |
| [gradio_app.py](../../gradio_app.py), [gradio_app_streaming.py](../../gradio_app_streaming.py) | Interactive demonstrations | Globally cached mutable pipeline; full video or successive MP4 segments | Not isolated per-call model state |
| [soulx_rtc/server.py](../../soulx_rtc/server.py), [start_webrtc.sh](../../start_webrtc.sh) | This fork's service | Shared child-process model, private states, bounded queues, persistent RTP tracks | The path relevant to concurrent FaceTime-style calls |

The streaming Gradio demo is **segmented file playback**, not WebRTC. Its producer generates chunks into a queue; the consumer groups three useful chunks (72 frames, 2.88 seconds at 25 FPS), writes segment media and keeps accumulated output for the final file. Its queue is not the service's bounded per-peer queue. A producer exception without a completion sentinel can leave its consumer waiting. Do not use its global mutable pipeline as a multi-tenant service.

The standalone streaming launcher defaults to a public bind and does not inherit the RTC token/lease controls. Exposing a demonstration UI requires an independent security decision.

## Upstream construction

[flash_head/inference.py](../../flash_head/inference.py) exposes the pipeline factory. Its sequence is:

1. Resolve the parallel device layout using [usp_device.py](../../flash_head/src/distributed/usp_device.py).
2. Select CUDA execution and a model variant under the weights root.
3. Construct [FlashHeadPipeline](../../flash_head/src/pipeline/flash_head_pipeline.py).
4. Load the variant's VAE and DiT configuration/checkpoint.
5. Move inference modules to their selected device/dtype, set evaluation mode and disable parameter gradients.
6. Load the audio feature extractor and the custom [Wav2Vec2Model](../../flash_head/audio_analysis/wav2vec2.py).
7. Enable the configured compilation wrappers for the model and VAE encode/decode paths.

The reference image is not the VAE's decoder input by itself. It is encoded into a conditioning latent, then combined with evolving noise in the DiT. Audio controls the transformer via projected cross-attention context.

### Variant boundaries

| Variant | VAE family | Typical upstream denoising configuration | RTC support in this fork |
| --- | --- | --- | --- |
| Lite | LTX causal video autoencoder, 128 latent channels, temporal/spatial strides 8/32 | Four steps | Implemented and measured |
| Pro | Wan VAE, 16 latent channels, strides 4/8 | Four steps | Not implemented by the Lite RTC tensor contract |
| Teacher path | Wan-family path; additional guidance logic | Twenty steps and audio CFG 3 in the upstream configuration | Not a tested serving mode |

Read [infer_params.yaml](../../flash_head/configs/infer_params.yaml) with the checkpoint configuration. A generic vendored class's default arguments are not proof of the installed model's settings. The RTC engine hardcodes Lite's five latent frames and 24-frame useful chunk; changing a model-name string cannot port the service to Pro.

The multi-GPU Pro launcher and distributed helpers are upstream single-generation parallelism. They are not a ten-session, independent-GPU routing layer for this service.

## What the installed Lite configuration selects

The inspected Lite DiT has width 1536, feed-forward width 8960, 30 blocks, 12 attention heads, head width 128, frequency embedding width 256, input channels 256, output channels 128, patch size 1×1×1 and VAE stride 8×32×32. Its configured text dimension is 4096, but text-conditioning modules are not used by the active audio-driven forward path.

The Lite autoencoder uses spatial patch size four, 128 latent channels, uniform posterior log variance, no quantization convolutions, and a noncausal decoder configuration. Its exact block sequence is described in [the VAE guide](03_VAE_AND_CONTINUITY.md). Library support for a feature does not imply the checkpoint selects it.

## Service startup and readiness

[start_webrtc.sh](../../start_webrtc.sh) enters the repository, uses the dedicated virtual environment, selects the GPU, sets a local Inductor cache and bounded compilation threading, and enables the expandable-segment allocator configuration. It delegates service flags to the module entrypoint; it does not install dependencies or build TensorRT engines on every call.

The parent [server](../../soulx_rtc/server.py) creates the [worker](../../soulx_rtc/worker.py). The worker owns all model tensors and states. Initialization acquires the checkout GPU lease, constructs the engine, installs requested TRT backends, and warms supported batch shapes.

Warmup covers reference preparation, first and recurrent motion-prefix lengths, and interruption/reconditioning/append paths. Optional session-isolation checks exercise the requested state workflow. Readiness should only follow successful initialization. Readiness is not a guarantee that a future co-resident allocation cannot cause OOM.

The current lease is an advisory file lock under this checkout. It prevents cooperating duplicate owners here; it is not a GPU-UUID-wide reservation. Another checkout, the upstream demos, or another application can still consume the device.

### Initial memory peak matters

Even staged operation begins through the normal pipeline loader. Moving modules away after construction can reduce steady-state memory without removing the initial overlap of allocations. Measure cold load, compile/warmup and steady-state peaks separately. Loading another copy just to inspect it can disrupt the running service; configuration/header inspection does not require doing that.

## Offline audio/windowing differences

The offline CLI offers rolling-stream and once-encoded audio paths. Rolling mode uses an eight-second context; once mode encodes the full available waveform and can therefore change feature context. Its first generated chunk and subsequent overlap handling are not interchangeable with the RTC engine's always-24-useful-frame output.

The CLI's loop bound uses `range((len(features) - 33) // 24)`. This can omit a terminal valid window and yields no iteration for exactly 33 feature frames. It is an offline path review finding, not evidence of the persistent service losing the same frames.

The CLI also uses `type=bool` for a face-crop argument, so a textual `False` is not a reliable false value in ordinary argparse semantics. It parses arguments twice. These cleanup items belong to offline correctness, below live-call scheduling in performance priority.

## Utilities and inactive library paths

- [facecrop.py](../../flash_head/utils/facecrop.py) and [cpu_face_handler.py](../../flash_head/utils/cpu_face_handler.py) implement optional face detection/alignment/cropping for upstream preparation. They are not a MuseTalk-style per-frame face compositor in RTC.
- [utils.py](../../flash_head/utils/utils.py) contains audio/video preparation, device helpers, normalization and color utilities. Color correction is active in generation.
- [audio_analysis/torch_utils.py](../../flash_head/audio_analysis/torch_utils.py) supplies the audio-side interpolation/helper operations.
- The LTX package includes an alternative factorized video autoencoder, generic transformer/attention modules, prompt enhancement and configuration-conversion utilities. Some helpers support the loaded causal VAE; the generic text/video transformer and prompt enhancer are not the Lite DiT.
- [wan/modules/vae.py](../../flash_head/wan/modules/vae.py) is the Pro-family VAE implementation, not the decoder used by the measured Lite RTC profile.
- Requirements files separate general model, WebRTC, and TensorRT experiment dependencies. A dependency being listed is not proof of the active attention kernel, codec, or engine version; runtime evidence must identify those.

See the [inventory](SOURCE_INVENTORY.md) for the complete file-level map.
