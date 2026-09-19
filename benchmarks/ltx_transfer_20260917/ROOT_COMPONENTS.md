# What changes teeth in the LTX workflow?

2026-09-17. Local GPU experiments: **NVIDIA GeForce RTX 4070 SUPER**, physical
12 GB class, 12,282 MiB visible (`nvidia-smi`), driver 595.84, PyTorch
2.7.1+cu128 / CUDA 12.8. OmniVoice remains resident at approximately 2,478 MiB.
Host RAM limit is 31,156,338,688 bytes (`/sys/fs/cgroup/memory.max`), no swap.
The local attention/decode adaptation is SDPA plus tiled decoding, not the
historical SageAttention configuration. Historical preferred output was made
on an RTX 5060 Ti with 15.48 GiB visible VRAM per the fork, not this GPU.
Hardware/runtime claims below are separated from CPU source audits.

This investigation asks which **quality-changing mechanism** in the winning
workflow matters. Memory optimizations by themselves are not the question.

## Concrete source findings

### 1. NAG is active and changes the attention output

Node `7`, `LTX2_NAG`, has scale 15, alpha 0.4, tau 2.5, `inplace=false`.
Node `5710` is a first-nonempty switch and selects this model for both passes.
It is not a disconnected optional node.

The implementation computes attention against positive and negative text,
forms `15*A_positive - 14*A_negative`, limits its L1 norm relative to the
positive representation (tau 2.5), then blends 40% guided / 60% positive.
This makes the negative prompt affect a distilled model even with CFG=1.
The earlier saved IA2V/Lipdub graphs use CFG=1 without this NAG node.
The saved negative prompt contains `distorted face`, `bad lips`, and
`poor lip sync`, but no dedicated dental correction model.

Source: locked KJNodes `nodes/ltxv_nodes.py`, functions
`normalized_attention_guidance`, `ltxv_crossattn_forward_nag`, and `LTX2_NAG`.
The [NAG authors](https://github.com/ChenDarYen/Normalized-Attention-Guidance)
describe normalized attention-space guidance for few-step models. Their
method is not a claim that this particular teeth problem is solved.

### 2. Prompt Relay's temporal patch is overwritten in the saved chain

At pinned Prompt Relay commit `5cae3c0`, `_encode_relay` concatenates the global
and local text, encodes it, and registers cross-attention `forward` patches
for `attn2` and `audio_attn2`. The downstream NAG node registers patches under
those same keys. `ModelPatcher.add_object_patch` assigns by key, replacing
the earlier entry; it does not compose both functions.

NAG has a fallback for `transformer_options['promptrelay_mask_fn']`, but this
pinned Prompt Relay package never installs that callback. Therefore its text
still contributes, while its temporal routing is overwritten in this chain.
This is a code-path finding, not a visual ranking inferred from the node name.

CPU-only verification executed the exact pinned patch-registration methods on
a one-block fake model with video/audio attention. Both Relay methods were
replaced by `LTXVCrossAttentionPatch`; no Relay mask callback was installed.
It did not load diffusion weights or run GPU inference. Reproduction and JSON:
`/workspace/experiments/ltx23-soulx-transfer/ablation/audit_patch_order.py` and
`patch-order-audit.json`.

Consequently, simply setting NAG scale to zero would be a confounded ablation:
that would restore Prompt Relay routing at the same time. The no-NAG control
instead uses the pre-Relay model while keeping exactly the same concatenated
text conditioning.

### 3. The 6+2 arrangement performs a denoised-latent restart

First-pass sigmas: `1, .99375, .9875, .98125, .975, .909375, .725`, seed 189.
The next sampler consumes the first sampler's denoised output, uses **new noise
with seed 74**, and samples at `.725, .421875, 0`. This is not merely an eight-step
schedule split into two UI boxes. It restarts from a prediction of the clean
sample, injects noise, and changes the second pass's guide conditioning.

There is a less obvious detail: `LTX2AudioLatentNormalizingSampling` has all-one
normalization factors, so it does not attenuate audio amplitudes. Nevertheless
its wrapper replaces the callback and returns the last predicted clean latent
after each sampler invocation. It is therefore not a strict no-op at the end
of the nonzero-sigma first pass. The denoised restart is the relevant behavior,
not an audio multiplier magically sharpening teeth.

The uninterrupted-eight-step control tests the split/refinement package as a
whole, including the guide-conditioning transition and second noise seed.
It does not isolate each of those subcomponents independently.

### 4. Several older comparisons also changed the base models

| Ingredient | Earlier saved graphs | Winning saved graph |
| --- | --- | --- |
| Transformer | `distilled/...Q3_K_S.gguf` | `distilled-1.1/...Q4_K_M.gguf` |
| Text encoder | Gemma IQ4_XS | Gemma QAT UD-Q3_K_XL |
| Guidance | CFG=1; Lipdub LoRA on the Lipdub path | CFG=1 plus NAG |
| Conditioning | External audio / video guide on Lipdub path | Jointly generated speech, image guide, concatenated prompts |
| Sampling | IA2V ancestral CFG++ or Lipdub Euler paths | Euler, clean-latent 6+2 restart |
| Video VAE | SHA-256 `01ea62d0…5b8ee3b` | **Same locked file/hash** |

The Q3/Q4 comparison described as quantization-only in the fork and earlier
notes is not sufficient to establish a pure quantization effect: the locked
files come from distinct distilled-version directories. Lightricks identifies
v1.1 as a different aesthetic/audio revision in its
[model card](https://huggingface.co/Lightricks/LTX-2.3/blob/main/README.md).
All new local controls keep the same v1.1 Q4 and text encoder.

The unchanged VAE hash is direct evidence against a special replacement VAE
being the new component in this particular workflow comparison. The winning
graph also contains no face-restoration, dental LoRA, or spatial latent
upscaler node. Attention scales are all 1.0. SageAttention and chunking change
the computation/numerics and memory use; no isolated dental benefit has been
demonstrated for them.

## Local controlled renders

The requested full-length prompt, portrait, seeds 189/74, 480×832 resolution,
24 FPS, model weights and text conditioning are retained. Review compares the
six-step predicted-clean intermediate and final output, NAG on/off with
temporal Relay off in both arms, and split 6+2 versus uninterrupted eight
steps. This is one identity and one seed pair, not universal causal proof.
Joint speech generation means equal timestamps need not contain equal phonemes.

The first combined generation/decode attempt was killed with exit 137 during
video decoding; `memory.events` reported an OOM kill. No finished movie was
saved. The queued controls did not execute before that server exited.
The retry disables host pinning/asynchronous offload and separates saved
latents/audio from decoding to avoid holding the large transformer/text encoder
alongside the full decoded video. These operational changes apply to all new
controls; they are not teeth-quality changes.

Graphs, execution histories, device telemetry, saved latents, and review images
are retained in `/workspace/experiments/ltx23-soulx-transfer/ablation` and
`ComfyUI/output/ablation`.

### Completed v1.1 controls on RTX 4070 SUPER

| Phase | Execution time | Sampled whole-device peak MiB |
| --- | ---: | ---: |
| NAG 6+2 generation, saving final/intermediate latents and audio | 247.212 s | 8,057 |
| Decode those two clips | 54.651 s | 3,239 |
| Combined no-NAG 6+2 and NAG uninterrupted-eight generation | 340.797 s | 8,665 |
| Decode the three resulting final/intermediate clips | 81.670 s | 3,239 |
| v1.0 nominal-Q4 artifact, NAG 6+2 generation | 236.157 s | 8,057 |
| Decode the v1.0 final/intermediate clips | 54.364 s | 3,239 |
| v1.1 without NAG, uninterrupted-eight generation | 229.087 s | 7,863 |
| Decode the no-NAG uninterrupted-eight clip | 27.172 s | 3,239 |

All rows use the local GPU/runtime stated above. Peaks are sampled every two
seconds and include OmniVoice; they are not exact allocator peaks. The first
retry's text-encoding period briefly overlapped a separate SoulX source-detail
benchmark. Subsequent sampling was protected by the shared experiment lease.
These timings are operational records, not isolated throughput benchmarks.
The successful baseline generation plus two decodes totaled 301.863 seconds;
this excludes the failed attempt and administrative gaps/restarts.

The final 6+2 videos contain 249 H.264 frames, 480×832 at 24 FPS, 10.375 s;
the first-pass and uninterrupted-eight outputs contain 241 frames, 10.042 s.
The guide/frame handling differs at the tail; visual comparisons use early
timestamps through 8.5 seconds, not the final padded/guide interval.

The CPU-generated `ablation/mouth-review.png` uses landmark-centered **fixed
128×72 native-pixel crops**, enlarged 2× with nearest-neighbor interpolation.
No sharpening, adaptive bright-pixel tooth mask, or anatomical score is applied.

Visual observations in the seven sampled moments:

- Teeth detail is already present in the six-step predicted-clean intermediate.
  Therefore the final pass cannot be the sole origin of separated teeth.
- Removing NAG substantially changes mouth expression/exposure. The no-NAG
  sample has larger, very white, rectangular tooth rows in several moments,
  notably 4.5 and 6.5 s. Some edges are stronger than with NAG. This supports a
  role in regulating appearance, **not** the simplistic claim that NAG always
  sharpens teeth. Speech is jointly generated, so equal timestamps do not fix
  phonemes or mouth openness across this comparison.
- The 6+2 output has better tooth separation than the uninterrupted-eight
  output in some sampled moments (including parts of the lower row at 0.5 s),
  but the benefit is not uniform. The split/refinement package is a concrete
  candidate for further testing, not a demonstrated universal dental repair.
- Both NAG-on schedules still contain soft or simplified dental regions. A
  one-identity, one-seed-pair review cannot establish one exclusive root cause.

The additional v1.0 comparison changes only the transformer **artifact** to an
older nominal Q4_K_M file, retaining the current text encoder/graph. Header
inspection finds identical tensor names/shapes, but quantization allocation
also differs: the old file has 108 Q5_K tensors where the v1.1 file uses Q4_K.
It is therefore a checkpoint-artifact control, not a perfectly isolated
training-revision experiment. Its immutable revision/hash and metadata are
retained in `ablation/v10-control.lock` and `transformer-metadata.json`.

### Completed four-condition comparison and conclusion

All four v1.1 conditions completed: NAG on/off crossed with split 6+2 versus
continuous eight steps. A fifth condition substituted the v1.0 nominal-Q4
artifact into the NAG/6+2 graph. Five final videos and three predicted-clean
intermediate videos were inspected through fixed-time crop sheets; full video
files are retained for temporal review. This is sampled visual inspection,
not a frame-by-frame anatomical or lip-sync certification.

The last no-NAG/eight-step control also generates separated teeth; in the
sampled open-mouth moments they are large, bright and rectangular, much like
the no-NAG/6+2 result. Thus neither NAG nor the restart is necessary for *any*
tooth detail in this fixture. NAG produces the most conspicuous change in
appearance/articulation among these node ablations. The restart produces
smaller, sometimes favorable changes in tooth separation at the same total
denoising evaluation count. Some regions remain soft in every condition.

The v1.0 artifact also generates separated teeth under the new workflow, with
a softer lower row in some samples and exaggerated exposed teeth in others.
It does not reproduce an all-blurry failure merely by restoring the older
checkpoint. Version, quantization packing, audio realization, and the old
Lipdub-versus-joint-generation distinction prevent assigning the historical
gain to one isolated weight change.

**Result:** the actual quality-changing mechanisms identified are NAG and the
clean-latent refinement/restart. Temporal Prompt Relay routing is overwritten
in the locked graph; the VAE file is unchanged. The ablations do **not** prove
one exclusive root cause of the historical blurry-teeth failure. They do
identify concrete mechanisms to test in SoulX and eliminate the unsupported
claim that a special low-VRAM decoder or active temporal Prompt Relay alone
explains the improvement. SageAttention/fused-kernel effects, alternate text
encoders, prompt wording, reference preprocessing, and external-audio versus
joint-AV conditioning have not been independently ablated here.

Review artifacts:

- [Five playable comparisons](/workspace/experiments/ltx23-soulx-transfer/ablation/review.html)
- [Four-condition mouth comparison](/workspace/experiments/ltx23-soulx-transfer/ablation/factorial-mouth-review.png)
- [All eight final/intermediate crop rows](/workspace/experiments/ltx23-soulx-transfer/ablation/mouth-review.png)
- [Requested full-length NAG/6+2 clip](/workspace/experiments/ltx23-soulx-transfer/ComfyUI/output/ablation/full_nag_6plus2_10s_00003-audio.mp4)

The large research models were unloaded after testing and the shared GPU
experiment lease was released. SoulX production code/weights were not changed.

## Implication for SoulX

The transferable ideas under investigation are **normalized attention
guidance** and **an additional denoised-latent refinement pass**. SoulX already
re-noises within its distilled sampling loop, so the novelty would be a tested
extra refinement stage, not merely adding random noise. A NAG experiment needs
a meaningful contrasting condition; SoulX Lite's active cross-attention consumes
audio features, so using silence is not equivalent to an LTX negative text
prompt describing visual defects. The LTX ablations should determine which
mechanism warrants that implementation work before changing SoulX behavior.

The first detail-focused SoulX experiment should test a separate refinement
pass on its own predicted-clean latents, using SoulX's existing trained
timestep range and preserving its motion prefix. Compare stock output, extra
steps without a restart, and a clean-latent restart at the same total model
evaluation count. Keep seed streams separate so extra refinement noise does
not silently change the next chunk's baseline noise stream. Judge teeth in
motion, lip sync, identity and added latency—not edge energy alone. This is
an experimental target supported by the LTX comparison, not an implemented
or validated SoulX fix at the time of this LTX investigation.

Follow-up: the [SoulX refinement experiment](../refinement_20260917/README.md)
is now implemented and tested on the same NVIDIA GeForce RTX 4070 SUPER
(12,282 MiB visible; driver 595.84, CUDA runtime 12.8). Fresh 320×576 GPU runs
compare four restart strengths, normal four steps and six continuous steps
across two seeds. They show no consistent teeth improvement; the feature
remains opt-in and disabled by default. See that report for full runtime,
co-resident load, validation and limitations.
