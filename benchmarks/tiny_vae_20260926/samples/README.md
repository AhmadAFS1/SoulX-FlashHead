# Sample recordings: Wan 2.1 VAE decoder vs taew2_1

**Hardware:** NVIDIA GeForce RTX 4070 SUPER (visible VRAM and driver from nvidia-smi: 595.84, 12282 MiB; physical
class unverified in this run), Torch 2.7.1+cu128. Date 2026-09-27. **Fresh local GPU inference.** The GPU
was idle (1 MiB used) and the SoulX lease was held for the decode; no co-resident GPU job was seen.

## Identical latents (only the decoder differs)
`decode_ab.py` decodes the DiT latents dumped from one real 576x320 run per fixture
(`../latents/`, 9 windows, seed 50, FlashAttention-2 DiT, no overlap-skip) twice: stock Wan 2.1
decoder (bf16, eager) and taew2_1 (fp16, eager). Frames 5..32 of each window are kept, as the
pipeline does, giving 250 frames at 25 fps with the fixture's audio.
The Wan side matches the pipeline's own decode output bit for bit (checked on strided samples of
windows 1 and 5), so the left panel is exactly what the model produced.

| fixture | PSNR taew2_1 vs Wan 2.1, full frame | mouth crop |
| --- | ---: | ---: |
| indian150-a | 39.31 dB | 34.47 dB |
| tts-plosives | 39.62 dB | 35.31 dB |

- `<fixture>-wan21-vs-taew21-full.mp4`: full frame, side by side.
- `<fixture>-wan21-vs-taew21-mouth.mp4`: mouth crop at 4x.
- `<fixture>-wan21-vs-taew21-stills.png`: mouth crops at 4 frames, pixel-exact 4x.

## Real pipeline runs (decoder and encoder swapped; trajectories diverge)
`pipeline-wan21-vs-taew21-{full,mouth}.mp4` place `../../pro_30fps_20260922/tae-ctl-r01` (shipping
Wan decoder, 26.81 FPS) and `tae-decenc-trt-r01` (taew2_1 decoder and encoder, TensorRT,
52.90 FPS) side by side. Both are fresh local GPU inference on the same card in one session,
2026-09-26, with FlashAttention-2 in both, because the SageAttention build was missing.
