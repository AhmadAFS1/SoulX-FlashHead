# LTX 2.3 transfer investigation — 2026-09-17

Follow-up focused on the user's actual root-component question:
[NAG, Prompt Relay patch ordering, and denoised-latent refinement controls](ROOT_COMPONENTS.md).
That follow-up supersedes any interpretation below that the historical Q3/Q4
comparison isolated quantization alone: the files also differ in model revision.

**Local hardware:** NVIDIA GeForce RTX 4070 SUPER, physical board class 12 GB;
12,282 MiB reported by `nvidia-smi`, driver 595.84. Python 3.10.12,
PyTorch 2.7.1+cu128, CUDA runtime 12.8. OmniVoice remained resident at
2,478 MiB. The isolated ComfyUI server was idle at about 182 MiB during the
SoulX tests. Exact snapshots are in each result JSON. Allocator limits are not
physical GPU capacity.

## Findings

September 17 supplementary postprocessing control: the existing more distant
`11_indian_man_speaking.mp4` LTX/LumaTalk take was run through actual Tasks
FaceLandmarker 0.10.35 and a public native-2x SRVGG TensorRT 10.16.1.11
substitute on **RTX 4070 SUPER, 12,282 MiB visible**, driver 595.84, Torch
2.7.1+cu128 / CUDA 12.8, with OmniVoice and idle LTX resident. The accessible
postprocess measured 39.61 FPS at 480x832→960x1664 and tracked 201/201 frames;
it gave modest clarity without a demonstrated dental fix. The [report and
recordings](../ojin_components_ltx_distant_20260917/README.md) distinguish this
fresh GPU result from the LTX-generation experiments below and from Ojin's
unavailable refiner/checkpoint/code.
This was produced after misunderstanding a request for the character in the SoulX
`bf16-batch5-c5-recorded-peer0.mp4`; the [corrected SoulX test](../ojin_components_soulx_distant_20260917/README.md)
is the primary answer to that request.

No teeth fix is established. The first transferred memory technique was tested
and rejected for service adoption. A VAE diagnostic suggests investigating
generated latents/conditioning before replacing the decoder.

The source fork is [AhmadAFS1/ltx-influencer-comfyui](https://github.com/AhmadAFS1/ltx-influencer-comfyui),
checked out at `172fb6f9b4c9ab4342e2d1f7f3968bf79e88e821` in
`/workspace/LTX-2.3`. This is a ComfyUI workflow/install repository, not a
SoulX-compatible model implementation.

### External historical evidence, not local benchmark results

The fork's `context.md` records an RTX 5060 Ti with 15.48 GiB visible VRAM.
Its preferred Q4 Prompt Relay run generated 249 frames at 480×832/24 FPS in
223.057 seconds, peaking at 14,046 MiB. This is approximately 21.5 seconds of
compute per second of output on that historical machine. Driver/runtime for
that specific run have not been independently verified here.

It jointly generates audio and video. The external-audio Lipdub workflow is
a different graph and uses an access-controlled IC-LoRA. The preferred Q4 graph
has no spatial latent upscaler. The matched Q3 comparison in the fork reports
only a modest Q4 advantage, with remaining dental artifacts. These facts do
not establish quantization, Prompt Relay, or 6+2 steps individually as a fix.
Also, 480×832 is 15:26, not exact 9:16 despite the historical sample label.

### Local GPU test: feed-forward chunking

`ffn_experiment.py` adapts the token-axis chunking idea in KJNodes'
`LTXVChunkFeedForward` to SoulX's stock Linear/GELU/Linear blocks. It applies a
temporary inference-only override, preserves the input, handles a partial
final chunk and multiple batch elements, and restores the original methods.
No serving code/defaults or weights were changed. CPU checks cover those
contracts and exception cleanup.

Local RTX 4070 SUPER results: 512×512, 25 FPS, four steps, BF16, staged memory,
uncompiled, seed 50, two seconds of the existing audio fixture. All runs use
the same saved square reference. OmniVoice and idle ComfyUI stayed resident.

| Variant | Generation seconds | Useful FPS | Peak allocated MiB | Peak reserved MiB |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 3.0045 | 16.642 | 3,835.91 | 4,186 |
| 256-token FFN chunks | 3.0945 | 16.158 | 3,825.66 | 4,162 |
| Original restored | 3.0378 | 16.459 | 3,835.91 | 4,186 |

Only 10.25 MiB of allocated-memory reduction, with about 3% longer generation
in this one short run. These are not steady-state service capacity numbers.
Chunking changes BF16 GEMM shapes: it is mathematically equivalent but not
pixel-identical. Across all 50 frames, mean absolute RGB difference is 0.889
on the 0–255 scale, maximum 73; restoring the original method reproduces the
baseline exactly. Both inspected 1.48-second frames still show smeared teeth.
Do not enable this as a dental fix or assume larger-profile memory gains.

Evidence: [results](ffn_256/results.json), [baseline clip](ffn_256/baseline.mp4),
[chunked clip](ffn_256/chunked.mp4), [restored clip](ffn_256/restored.mp4).

### Local GPU test: can SoulX's VAE retain clear teeth?

`vae_roundtrip.py` encodes nine repeated frames from the fork's committed Q4
sample at 5 seconds, then decodes using SoulX's unchanged VAE. This is a
480×832 BF16 VAE-only diagnostic, posterior seed 50, with no DiT, audio
conditioning, or color correction. The middle decoded frame retains visibly
separated upper and lower teeth. Full-frame RGB MAE is 1.802; this is a pixel
error measure, not an anatomical quality score. Peak allocated memory was
1,269.87 MiB and the synchronized encode/decode took 0.311 seconds.

[Source](q4-source-5s.png) · [SoulX VAE reconstruction](q4-through-soulx-vae.png)
· [run evidence](vae-roundtrip.json)

This demonstrates representational ability for one repeated-still example.
It does not rule out motion-dependent VAE degradation, recurrence, source
conditioning, or avatar-specific effects. It does weaken the claim that this
VAE necessarily smears all separated teeth.

### Compatibility audit

Both VAEs use 128 latent channels and 32× spatial / 8× temporal compression,
but their learned latent spaces and internal architectures are not established
as interchangeable. The SoulX decoder input convolution has shape
`[512,128,3,3,3]`; LTX 2.3 uses `[1024,128,3,3,3]`. Its block layout also differs.
Direct checkpoint replacement is not supported. Shapes/configuration were
read from local checkpoint headers, not inferred from their names; details
are in `/workspace/experiments/ltx23-soulx-transfer/vae-compatibility.json`.

Prompt Relay expects an LTX text-conditioned joint audio/video transformer.
SoulX's audio-conditioned DiT and distilled stochastic sampling path have
different contracts. Copying prompts, NAG, LTX weights, or its 6+2 sigma
schedule is not a valid drop-in adaptation. SoulX already supports efficient
attention and staged model offloading.

## Installed experiment environment

The isolated stack lives at `/workspace/experiments/ltx23-soulx-transfer`.
See its `README.md` for startup and local render validation. Its own venv and
loopback port 18189 leave the SoulX and OmniVoice environments untouched.
All five model files are verified against the fork's immutable size/hash lock.
This local adaptation uses PyTorch attention and tiled VAE decode; it does not
reproduce the historical SageAttention stack or claim the same output quality.

The local RTX 4070 SUPER smoke render completed successfully in **170.844 s**
with OmniVoice resident. Sampled whole-device VRAM peaked at **9,401 MiB**
(two-second sampling, not an allocator peak). The graph requested 49 frames;
the final playable H.264/AAC container contains 57 frames at 480×832/24 FPS,
2.375 seconds. That duration difference has not been isolated. Four inspected
frames show some soft/uneven teeth, so this is an installation/inference pass,
not a teeth-quality pass. The output is
`/workspace/experiments/ltx23-soulx-transfer/ComfyUI/output/research/q4_sdpa_tiled_2s_00001-audio.mp4`.
Research models were unloaded after the run; the idle server retained about
316 MiB while OmniVoice remained at 2,478 MiB. This initial smoke test did not
include a full-length run. Subsequent full-length controlled renders are
documented in [the component investigation](ROOT_COMPONENTS.md); no real-time
LTX capability is claimed.

## Next experiment justified by this evidence

Use a short **moving** sharp-teeth reference clip to measure SoulX VAE
reconstruction through time, then compare the same identity/audio against
SoulX-generated latents. If the reconstruction stays clear while generation
smears, prioritize the learned generation/conditioning path. LTX can supply
offline candidate training/reference clips, but speech alignment, identities,
temporal dental quality, and a SoulX-compatible training procedure need
validation before any fine-tuning claim. An online LTX second pass has no
established latency or audio-preservation suitability for SoulX calls.
