# SoulX-FlashHead system architecture and model weights

**September 17, 2026 — static source audit, CPU-only checkpoint metadata inspection and diagram rendering. No new GPU inference or training.** Machine snapshot: NVIDIA GeForce RTX 4070 SUPER, 12 GB class / 12,282 MiB visible, driver 595.84; Torch 2.7.1+cu128 / CUDA 12.8. OmniVoice and idle LTX were resident. Exact inventory and hardware evidence: [weight-inventory.json](weight-inventory.json). Source baseline `1349991d26189898f5ba4158f112b86afb49ba06` plus the inspected local changes.

## Architecture diagrams

Open the [interactive diagram viewer](index.html) to switch views, zoom and download each diagram. SVGs remain readable at arbitrary zoom; PDFs are standalone exports.

1. **Full repository and serving system:** [SVG](01-system.svg), [PNG](01-system.png), [PDF](01-system.pdf). Entrypoints, CPU service, GPU worker, media delivery, model loading and optional experimental paths.
2. **Inside LITE generation:** [SVG](02-neural-pipeline.svg), [PNG](02-neural-pipeline.png), [PDF](02-neural-pipeline.pdf). Speech conditioning, reference latents, transformer blocks, denoising and recurrent video generation.
3. **Refiner weights, training and inference:** [SVG](03-refiner-weights.svg), [PNG](03-refiner-weights.png), [PDF](03-refiner-weights.pdf). What a new checkpoint would contain and how a proposed mouth network could attach to the existing system.

Blue/green components are implemented; purple components are optional/experimental; orange components are proposed. The diagrams describe the inspected fork, including its local real-time service, rather than claiming these additions all exist in upstream SoulX.

## What “weights” means

Network **architecture** is executable structure: layers, connections, input channels and operations. **Weights** are the numeric tensors learned during training and loaded into that structure. A checkpoint is a file storing those tensors, sometimes alongside other state. Its extension does not tell us its quality or whether it is compatible with another network.

For example, code might define a convolution with 32 filters. Training learns the numbers in those filters; inference applies the saved filters to new images. Saving those numbers as `refiner_weights.pt` does not turn them into a complete standalone application. We also need matching model code, normalization, crop geometry and output interpretation.

SoulX already uses several learned networks. The main diffusion transformer generates changing video latents from audio, reference and motion conditions. A VAE encodes/decodes video between pixels and latents. Wav2Vec analyzes speech for conditioning; it does **not** synthesize the voice. Optional CPU Kokoro TTS or supplied audio provides the speech in this local service.

A custom mouth refiner would be another network with another checkpoint. Its intended task is much narrower: take an already generated mouth crop and predict a constrained visual correction. We could freeze all SoulX weights and train just this network. That is still one SoulX generator, but technically an additional learned model, not a weight-free improvement inside the existing generator.

We do not have Ojin's private checkpoint, matching architecture or training recipe. We therefore cannot state its size, reproduce it exactly, or establish that it alone explains his quality. Our network would be independently designed and trained.

## How the weights compare

These are **stored tensor-element counts**, including inactive modules and any stored buffers, obtained from installed checkpoint metadata. They are not active-parameter counts or measured runtime VRAM. Decimal GB/MB below show hypothetical two-byte storage for every element; loading precision, activations, caches and workspaces change actual memory use.

| Component | Stored elements | Hypothetical BF16 weights | Function |
| --- | ---: | ---: | --- |
| LITE diffusion checkpoint | 1,526,864,256 | 3.05 GB | Whole-frame latent generation, including audio adapter weights |
| PRO diffusion checkpoint | 1,507,694,912 | 3.02 GB | Alternative generator paired with a different VAE |
| LITE LTX VAE | 419,193,905 | 838 MB | Video compression and reconstruction |
| PRO Wan VAE | 126,892,531 | 254 MB | Different compression and reconstruction path |
| Wav2Vec speech encoder | 94,395,552 | 189 MB | Speech features; current code loads this encoder in FP32 |
| Public native-2× SRVGG substitute | 600,652 | 1.20 MB | Existing optional image-upscaling experiment |
| Proposed custom mouth refiner | **0.5–2 million design budget** | **1–4 MB estimate** | Not implemented, trained or benchmarked |
| Ojin's private mouth refiner | **Unknown** | **Unknown** | Cannot inspect unavailable checkpoint |

The nominal “1.3B” model name is not an exact count of everything serialized. Some stored modules are inactive in the current path. A much smaller mouth network is plausible because SoulX has already generated pose, motion and facial appearance; the new network only processes a small region. Small weight size does not guarantee low latency, correct teeth or adequate temporal consistency.

## Inside a typical LITE chunk

The reference image is repeated across 33 frames and VAE-encoded. At 320×576 output, a video latent has shape `128 channels × 5 times × 18 high × 10 wide`. Reference and noisy video are concatenated across channels. Patch size `1×1×1` gives 900 spatiotemporal tokens with transformer width 1536.

The transformer has 30 audio-conditioned blocks, 12 attention heads and feed-forward width 8960. Self-attention relates video tokens, audio cross-attention brings in projected speech features, and timestep conditioning controls denoising. Four usual sampling steps run this same network with the **same weights** at different noise levels. Increasing steps does not train more detail or create a new checkpoint.

The VAE decodes 33 RGB frames. The ordinary path color-corrects them, re-encodes the final nine frames for motion continuity, and discards nine overlapping frames to deliver 24 new frames. Those 24 frames span 0.96 seconds at 25 FPS or 1.6 seconds at 15 FPS; these are media durations, not measured computation latency. Terminal lean paths can skip unused feedback work.

LITE's VAE stride is `8×32×32`, with 128 latent channels. PRO uses Wan VAE stride `4×8×8`, 16 latent channels and transformer patches `1×2×2`. The smaller PRO VAE weight count therefore does not mean its latent computation is cheaper: its spatial grid is denser. The LTX VAE used by SoulX LITE is not the separate LTX 2.3 video generator.

## Service structure and proposed integration

The CPU parent handles HTTP, calls, audio preparation, bounded queues, scheduling and WebRTC. A spawned GPU worker owns shared model weights and private per-call RNG, audio, cursor and motion state. Compatible DiT requests can batch across calls; VAE work runs per state. Frames cross process boundaries through shared memory, with a pickle fallback, before encoding and delivery. Idle media and persistent clocks keep calls alive between turns.

The real-time engine currently constructs **LITE** and assumes its temporal shapes. Offline inference and Gradio support LITE or PRO; switching the real-time engine to PRO requires more than changing a model name. Gradio streaming produces video segments, separate from the WebRTC service.

The existing FaceLandmarker and SRVGG TensorRT utilities are opt-in experiments, not the default live service path. Landmark weights locate the mouth; SR weights upscale pixels. Neither is the proposed specialized tooth-restoration network. Compilation, TensorRT export and quantization change execution/storage; they do not teach missing dental anatomy.

A candidate refiner would track and align the mouth, use the current and past crops, predict a masked residual inside the oral region and composite it into the delivered frame. Training needs suitable sharp targets and temporal constraints; our existing character bank is not yet an adequate teeth dataset. No quality or latency result is implied by this diagram.

**Feedback must be handled twice:** ordinary `Engine.generate` retains native motion latents before returning RGB, but interruption/resumption uses `Engine.recondition` to encode recent **sent** frames. A safe integration must retain separate native frame history, matched to delivery timing, for that second path too. Otherwise fabricated teeth can become conditions for future generation. A display refiner also cannot by itself correct badly timed lip motion; preserving timing and lip boundaries is a constraint, not proof of improved lip sync.

## Source map and reproduction

| Layer | Main sources |
| --- | --- |
| CLI/demo and checkpoint assembly | `generate_video.py`, `gradio_app.py`, `gradio_app_streaming.py`, `flash_head/inference.py` |
| Latent generation and network internals | `flash_head/src/pipeline/flash_head_pipeline.py`, `flash_head/src/modules/flash_head_model.py`, `flash_head/ltx_video/ltx_vae.py`, `flash_head/wan/modules/vae.py` |
| GPU ownership, batching and recurrence | `soulx_rtc/worker.py`, `soulx_rtc/engine.py` |
| HTTP, calls, queues and media delivery | `soulx_rtc/server.py`, `soulx_rtc/calls.py`, `soulx_rtc/avatars.py`, `soulx_rtc/idle_cache.py`, `soulx_rtc/tts.py` |
| Existing experimental utilities | `soulx_rtc/face_landmarker.py`, `soulx_rtc/srvgg_trt.py`, `soulx_rtc/mouth_sr.py`, `soulx_rtc/refinement.py` |

`soulx_rtc/refinement.py` concerns extra latent passes using existing generator weights; it is not the proposed learned mouth network. For the detailed training feasibility and data audit, see [mouth-refiner analysis](../../../benchmarks/mouth_refiner_feasibility_20260917/README.md).

From the repository root, run `.venv/bin/python docs/architecture/system_design_20260917/build.py` with Graphviz `dot` installed to rebuild the checkpoint inventory and SVG/PNG/PDF exports. This reads checkpoint metadata on CPU and does not execute models. The HTML viewer has no external dependencies. The older [source walkthrough](../README.md) provides deeper subsystem documentation at its separately dated baseline.
