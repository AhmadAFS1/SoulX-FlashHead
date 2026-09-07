# 2. Audio conditioning and the diffusion transformer

[Guide index](README.md) · [Previous](01_ENTRYPOINTS_AND_LOADING.md) · [Next: VAE and continuity](03_VAE_AND_CONTINUITY.md)

Primary implementation: [flash_head_model.py](../../flash_head/src/modules/flash_head_model.py), [flash_head_pipeline.py](../../flash_head/src/pipeline/flash_head_pipeline.py), [wav2vec2.py](../../flash_head/audio_analysis/wav2vec2.py), and the optimized orchestration in [engine.py](../../soulx_rtc/engine.py).

## The useful unit of work

At the standard 25-FPS profile, a model chunk decodes 33 RGB frames, retains nine frames as overlap/history, and emits 24 new frames: **0.96 seconds of useful video**. Lite's causal encoder maps 33 frames to five latent time positions. Nine recurrent RGB frames map to two latent time positions.

Independent sessions can form a batch. Two successive chunks of the same session cannot be computed independently because the second needs the first chunk's reconstructed motion context.

## Tensor ledger

Here H and W are neural output height and width; B is the compatible session batch.

| Value | Shape / meaning |
| --- | --- |
| RGB reference clip | `[1, 3, 33, H, W]`, repeated input still, approximately [-1, 1] |
| Encoded reference per state | `[128, 5, H/32, W/32]`, normalized latent |
| Initial random latent per state | Same shape as reference |
| Model input after concatenation | `[B, 256, 5, H/32, W/32]` |
| DiT token sequence | `[B, N, 1536]`, N = 5 × H/32 × W/32 |
| Audio feature windows | `[B, 33, 5, 12, 768]` |
| Projected audio context | `[B, 5, 32, 1536]` |
| DiT prediction | `[B, 128, 5, H/32, W/32]` |
| Decoded RGB | `[B, 3, 33, H, W]` conceptually; RTC decodes rows serially |
| Delivered chunk | Up to 24 uint8 RGB frames, `[T, H, W, 3]` on CPU |

Token counts are 1280 at 512×512, 1950 at 480×832, and 2880 at 576×1024. The dimensions need not be square. Full self-attention is not restricted to a mouth ROI; changing canvas size changes both spatial decoding work and the transformer token problem.

## Rolling Wav2Vec preparation

The engine stores audio in a per-session NumPy buffer at 16 kHz. For a chunk at cursor c, its audio window ends near `round((c + 24) * 16000 / fps)`; up to eight seconds of preceding waveform are selected. Missing left context and the required right tail are zero-padded.

The CPU feature extractor normalizes the selected waveform. The custom Wav2Vec convolutional features are interpolated to the desired video-aligned sequence length **before** transformer processing; the transformer emits twelve hidden layers of width 768. For each of the 33 chunk-aligned video positions, five nearby feature positions (offsets -2 through +2, clipped at the available boundaries) are selected.

This is not an autoregressive audio transformer cache. Shifting an eight-second window changes normalization, transformer context and interpolation. Saving the prior window's hidden states and appending only new features is not guaranteed equivalent.

The current engine prepares audio one state at a time even when the DiT will run a batch. A same-length Wav2Vec microbatch is an experiment worth measuring, but the existing native TRT profile spends only about 14 ms per chunk in this stage. It is not the main remaining cost.

### Audio-to-latent-time projection

The first video position represents the first latent time position. Its 5 × 12 × 768 feature values are flattened (46,080 values) and projected to width 512.

The remaining 32 video positions are divided into four groups of eight. Within each group, the first position contributes three context slots, the six middle positions contribute one each, and the last contributes three: twelve slots. Flattening gives 12 × 12 × 768 = 110,592 values, also projected to 512.

The two paths join a shared projection stack: ReLU, a 512-wide projection, another ReLU, expansion to 32 × 1536 values, reshape and layer normalization. The result supplies 32 audio tokens for each of five latent times.

## Thirty repeated transformer blocks

A 1×1×1 Conv3d embeds the 256-channel concatenated noise/reference input, then flattens the spatiotemporal positions. Timestep sinusoidal features feed the time embedding and modulation projection.

Each of the 30 blocks performs:

1. **Adaptive self-attention normalization.** Non-affine layer normalization is modulated by timestep-dependent shift and scale. The six modulation groups provide the shift, scale and residual gates for attention and MLP branches.
2. **Self-attention.** Separate q, k and v linear projections produce twelve 128-wide heads. q/k RMS normalization reduces in FP32 and returns to the working dtype. Three-dimensional rotary position embeddings encode temporal and spatial positions. Attention spans the full N-token chunk, without a causal temporal mask.
3. **Attention residual.** The output projection is gated and added to the original sequence.
4. **Audio cross-attention.** The affine normalization path prepares queries. Tokens are reshaped by latent time to `[B*5, spatial_tokens, 1536]`; each time slice attends to its corresponding 32 audio tokens. q/k normalization and output projection follow the model's trained layout.
5. **Adaptive MLP branch.** Normalization/modulation, Linear(1536→8960), tanh-approximate GELU, Linear(8960→1536), then gated residual addition.

The output head applies normalization and two-way shift/scale modulation, projects tokens to 128 latent channels and unpatchifies. There is no face crop pasted into a separate source video.

### Attention dispatch is conditional

The model prefers an importable Sage implementation, then the supported FlashAttention implementations, then torch scaled-dot-product attention. Installation flags and dtype/layout determine what can actually run. Small unit tests deliberately exercise a compatible SDPA path; they do not certify the production fast-attention backend.

A full-block TensorRT conversion must compare against the **actual selected kernel**. Replacing a fast specialized attention path with a generic export can erase savings from fused linears.

## Four-step sampling, explicitly

For four steps, the base sequence is 1000, 750, 500, 250, 0. Applying shift 5 gives approximately:

```text
1000 -> 937.5 -> 833.3333 -> 625 -> 0
```

Before each model invocation, the leading noisy latent positions are replaced by the motion prefix. The first chunk starts with one latent position from the reference. Later chunks use two positions encoded from the last nine corrected RGB frames.

With a model prediction v at time t:

```text
x0    = x - (t / 1000) * v
xnext = (1 - tnext / 1000) * x0 + (tnext / 1000) * fresh_noise
```

The motion prefix is restored again before decoding. Noise is drawn initially and at each step, including the last step where its coefficient is zero. Removing that final draw may preserve the immediate arithmetic but change the session RNG stream consumed by later noise or posterior sampling. A refactor must define and test RNG semantics across multiple chunks, not just compare one output tensor.

## What has already been cached

| Cache | Valid reuse boundary | Must change when |
| --- | --- | --- |
| Reference latent and reference color statistics | Immutable image/profile template | Image bytes, geometry or relevant preparation/profile contract changes |
| Timestep modulation/constants | Same steps/profile/batch | Dtype, device, batch/profile or step configuration changes |
| Real-valued rotary tables | Same geometry/layout/profile | Token positions, shape, device or math mode changes |
| Projected audio tokens | Four denoising steps of one chunk | Audio window/chunk changes |
| Each block's audio K/V | Four denoising steps of one chunk | Projected context, layer weights, dtype or session changes |

All 30 blocks' cross-attention K/V for B=1 occupy about **28.125 MiB BF16**, plus approximately 0.469 MiB of projected audio context. Sharing these between unrelated audio turns would be a correctness bug. Sharing the immutable model weights is correct.

The optional real-RoPE path replaces complex arithmetic with FP32 real-valued rotary math before returning to the working dtype. Its native portrait A/B gain is measured as part of the recorded profile; it is not a universal claim that every individual cache or kernel contributes that same percentage.

## Unused weights are not an unused compute stage

The instantiated `text_embedding` and `audio_emb` parameters do not participate in the active Lite forward path. Checkpoint metadata gives 8,653,824 and 1,776,384 parameters respectively: about **19.894 MiB together at BF16**. Releasing them after strict checkpoint loading could recover small headroom. They are not a multi-gigabyte optimization and removing them does not eliminate active matrix multiplication.

For the proposed fusion/export/caching work and acceptance tests, see [the next plan](../research/NEXT_OPTIMIZATION_PLAN.md).
