# 3. Video autoencoder, color correction and temporal continuity

[Guide index](README.md) · [Previous](02_AUDIO_AND_DIT.md) · [Next: engine and memory](04_ENGINE_AND_MEMORY.md)

Primary sources: [ltx_vae.py](../../flash_head/ltx_video/ltx_vae.py), [causal_video_autoencoder.py](../../flash_head/ltx_video/models/autoencoders/causal_video_autoencoder.py), [vae.py](../../flash_head/ltx_video/models/autoencoders/vae.py), [color utilities](../../flash_head/utils/utils.py), [engine](../../soulx_rtc/engine.py).

## Why the VAE is part of every recurrent chunk

The VAE has three jobs:

1. Encode the repeated still into reference conditioning during avatar preparation.
2. Decode the denoised five-position latent into 33 RGB frames.
3. Re-encode the last nine corrected RGB frames into the next chunk's two-position motion prefix.

Only the first job is naturally reusable across sessions with the same immutable reference. Decoder acceleration alone does not remove the third job. In the historical native TensorRT ten-job profile, decoding averaged about 213 ms/chunk and motion encoding about 73 ms/chunk.

The LTX wrapper applies the checkpoint's per-channel latent mean and standard deviation. Encoding normalizes sampled latent values; decoding reverses that normalization before calling the decoder. Exporting only the raw decoder while forgetting these transforms changes the model contract.

## Installed Lite encoder: concrete shape progression

The encoder first patchifies each 4×4 RGB spatial neighborhood into channels. It does not initially group time. Thus three RGB channels become 48 channels at H/4 × W/4. A causal Conv3d projects to the encoder width.

For a 33-frame input:

| Operation group | Time | Spatial size | Working channels |
| --- | ---: | --- | ---: |
| Spatial patchify + input convolution | 33 | H/4 × W/4 | 128 |
| Four residual blocks | 33 | H/4 × W/4 | 128 |
| First all-axis compression | 17 | H/8 × W/8 | 128 |
| Width-changing residual block, then three residual blocks | 17 | H/8 × W/8 | 256 |
| Second all-axis compression | 9 | H/16 × W/16 | 256 |
| Width-changing residual block, then three residual blocks | 9 | H/16 × W/16 | 512 |
| Third all-axis compression | 5 | H/32 × W/32 | 512 |
| Three residual blocks, then four residual blocks | 5 | H/32 × W/32 | 512 |
| PixelNorm, SiLU and posterior convolution | 5 | H/32 × W/32 | 129 before log-variance expansion |

The 129 posterior channels are 128 means and one uniform log-variance channel, expanded over the 128 latent channels. The diagonal Gaussian posterior samples `mean + std * noise`. This sampling is part of state evolution, not deterministic image compression.

Causal temporal convolutions pad the left side by replicating initial frames; downsampling preserves the initial temporal position, giving 33→17→9→5 and 9→5→3→2 for the motion path. [causal_conv3d.py](../../flash_head/ltx_video/models/autoencoders/causal_conv3d.py) and [conv_nd_factory.py](../../flash_head/ltx_video/models/autoencoders/conv_nd_factory.py) implement the convolution selection/padding mechanics.

PixelNorm normalizes channel RMS at each position. Its square operation follows the incoming tensor dtype in the current helper. BF16 and FP16 do not have the same exponent range: casually changing the entire VAE to FP16 risks overflow in sensitive normalization, even if convolutions themselves run.

## Decoder: five latent positions become 33 frames

The decoder starts by projecting the 128 latent channels to the high-width feature representation and reverses the residual/compression hierarchy. All-axis expansion uses convolution plus depth-to-space rearrangement; the temporal rule is 2T-1. Three expansions give 5→9→17→33, while spatial dimensions grow H/32→H/16→H/8→H/4.

Channel widths step down from 512 through 256 to 128. The output normalization/nonlinearity and convolution produce 48 channels, which unpatchify to RGB at H×W.

Although the class name includes “causal,” the **installed decoder is configured noncausal**. Its temporal convolutions replicate edges on both sides. The encoder's causal behavior therefore does not justify independently decoding each latent time position or reusing an autoregressive decoder cache.

Optional attention, timestep-conditioned and noise-injection blocks exist in the generic implementation, but the installed Lite block configuration does not enable them. Optimizing a dormant library branch will not improve this profile.

## Color correction and feedback

The generated RGB frames are corrected toward reference-image color statistics in Lab space. The optimized engine caches the reference statistics; memory-saving modes process a few frames at a time instead of materializing all 33 frames' color intermediates together.

The order is significant:

```text
decode 33
  -> correct each frame
  -> take last 9 corrected floating-point frames
  -> encode + sample posterior -> normalized motion latent
  -> remove first 9 overlap frames from visible output
  -> clamp / quantize / layout conversion -> CPU RGB
```

The recurrence uses corrected floating-point data, not the encoded video received by a client. Last-sent uint8 frames used for interruption recovery have already lost precision relative to this internal state.

A safe-looking optimization candidate is to remove the first nine frames **before per-frame color correction**, since they are neither emitted nor part of the last-nine feedback. This needs a parity test confirming that every correction statistic is per-frame and that output/feedback indexing stays identical. It does **not** permit dropping those frames before the noncausal VAE decode.

For an explicitly terminal finite job, next-chunk motion encoding may be unnecessary. It is not safe to skip it for a call that can append more speech or generated idle. Saving one ~73 ms terminal encode is a small per-job gain, not 25% of a ten-second workload.

## Geometry and useful-frame accounting

A ten-second request at 25 FPS contains 250 useful frames. The current finite engine needs eleven chunks, decoding 363 frames in total: 99 are overlap, 250 are requested output, and 14 are terminal excess. Persistent calls keep whole-chunk video alignment, so a turn can occupy 264 frame slots even if its useful speech is only ten seconds.

That overhead is not solved by relabeling 363 decoded frames as useful FPS. Nor can the overlap simply be deleted without changing the temporal model.

| Neural or delivery geometry | Interpretation |
| --- | --- |
| 512×512 | Square whole-frame generation |
| 480×832 | Native generation matching the MuseTalk source plates; ratio 15:26 |
| 576×1024 | Native true 9:16; substantially more tokens and decoder work |
| 468×832 | True 9:16 delivery crop of 480×832: six columns removed on each side |
| 288×512 | Validator-compatible native 9:16 candidate; not quality-certified |

Upscaling or cropping a completed frame is not evidence of generating natively at the new size. The service resizes approved reference frames to the requested geometry; changing aspect ratio can stretch the anchor unless preparation explicitly preserves aspect and crops/pads.

## First frame, last frame and turn seams

**Reference image:** establishes identity, appearance and conditioning. It does not force the first emitted frame to equal that image; the first nine decoded positions are withheld.

**Motion prefix:** constrains the next latent chunk using recent generated history. It promotes continuity but is not an exact RGB boundary equation.

**Last frame:** there is no API supplying a target final image or enforcing an exact endpoint. Seed control is repeatability input, not a requested pose or endpoint solver.

**Continuous speech:** maintaining one recurrent model state is the closest existing path to continuous motion. Separate independent jobs initialized from the same portrait reset history and need not join invisibly.

**Source idle:** plays the approved base video cheaply, but its head/body motion can diverge from the generated trajectory. Reconditioning from recent sent source frames makes a plausible handoff; it cannot promise that the generator's next image exactly matches the source choreography.

**Generated idle:** advances the same neural trajectory using silence. It costs approximately the same kind of chunk generation as speech and competes for GPU capacity.

**Interruption:** the service invalidates unsent work by epoch, waits for in-flight work and rebuilds motion from the last nine sent frames. Sent frames are not receiver-acknowledged frames; some old media may still be in network/jitter/decoder buffers. Exact client-visible rollback is not implemented.

Invisible joins require visual evaluation of actual turn/epoch/chunk boundaries with frame provenance. A low mean pixel difference or monotonic RTP timestamps alone is insufficient.

## Tiling is not ready to enable as-is

The generic [VAE base](../../flash_head/ltx_video/models/autoencoders/vae.py) contains spatial and temporal tiling helpers, but they assume a different architecture.

Spatial tiling derives latent tile size as `512 / 2**(len(encoder.down_blocks)-1)`. The installed causal encoder has ten block groups but only three all-axis spatial compressions after patchification. The formula yields a latent tile size of 1 instead of 16. A 25% overlap then produces `int(1 * 0.75) == 0`, invalid for the decoder's range step.

Temporal tiling references an encoder `patch_size_t` attribute not provided by this causal encoder and uses older split assumptions. “Enable tiling” is therefore not a verified low-VRAM workflow for this checkpoint.

A correct implementation needs explicit effective strides, causal/noncausal receptive-field halos, exact output indexing, overlap blending only where justified, posterior RNG handling, and spatial/temporal seam tests. Decode tiling may save activation memory while adding repeated halo computation; throughput must be measured, not assumed.

## Other VAE implementations in the repository

[video_autoencoder.py](../../flash_head/ltx_video/models/autoencoders/video_autoencoder.py) and [dual_conv3d.py](../../flash_head/ltx_video/models/autoencoders/dual_conv3d.py) contain an alternative factorized spatial/temporal convolution architecture. Its helper assumptions are not the installed Lite model's stride contract.

The [Wan VAE](../../flash_head/wan/modules/vae.py) uses 16 latent channels and cached causal features, with first-frame and subsequent temporal-group handling. Its distributed spatial variants exchange halo regions and gather outputs. These are Pro-family implementations, not a drop-in Lite decoder cache.

[vae_encode.py](../../flash_head/ltx_video/models/autoencoders/vae_encode.py) supplies generic encode/decode/scaling helpers. Audit the actually selected wrapper before reusing generic video-batching or tiling behavior.
