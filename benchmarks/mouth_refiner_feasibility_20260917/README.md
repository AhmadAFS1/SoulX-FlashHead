# Can we train our own SoulX mouth refiner?

September 17, 2026. **Analysis and CPU media inspection only; no refiner training or new GPU inference.**
Machine snapshot: **NVIDIA GeForce RTX 4070 SUPER**, physical **12 GB class / 12,282 MiB visible**,
driver **595.84**; the existing SoulX environment uses PyTorch **2.7.1+cu128 / CUDA 12.8**.
`nvidia-smi` showed 2,727 MiB device use, with OmniVoice at 2,478 MiB and idle LTX at 234 MiB.
Historical timings below have their own linked run records. [Audit evidence](audit.json) records
the current device snapshot, source hashes, media inventory and public repository commit IDs.

## Decision

**Yes: we can implement and train an independent mouth-refinement model. Quality is contingent
on suitable training targets and cannot be inferred from writing or exporting a checkpoint.**
The best first experiment is a small, causal, identity-conditioned mouth restoration network
for the exact character in `bf16-batch5-c5-recorded-peer0.mp4`.

The main obstacle is **training supervision matching SoulX's errors while preserving speech
geometry**, rather than access to a particular file extension or to TensorRT. At about 58 source
pixels across the mouth, information is missing or ambiguous; a refiner must supply learned or
reference-guided structure. It cannot guarantee recovery of the subject's actual teeth from an
unresolved gray band or a closed-mouth portrait.

This would keep SoulX as the visual motion generator and add a small learned pixel-refinement
module. It is an additional network, but does not require another TTS system or a second
audio-to-video generator. Improving existing lip-sync is a separate problem; the initial objective
is to improve teeth **without worsening** SoulX's lip-sync.

## What is established about Ojin?

Ojin's [public notices](https://github.com/ojinai/kit-example/blob/main/THIRD-PARTY-NOTICES.md)
identify a proprietary mouth-refiner checkpoint called `refiner_weights.pt`, alongside
FaceLandmarker and an SRVGG upscaler. They do not disclose its architecture, input tensors,
training targets, losses, crop size, temporal mechanism or ordering relative to upscaling.
The `.pt` name does not tell us those things. We need our own architecture, weights and loader;
we cannot assume checkpoint compatibility.

**Correction to the previous causal claim:** the failed public-upscaler experiment does not
prove Ojin's refiner alone caused his reported improvement. The exact upscaler weights and
orchestration also differ. It demonstrates that our tested substitutes do not solve this
SoulX character's teeth. A dedicated learned refiner is a plausible next hypothesis.

## Evidence from this workspace

| Evidence | Consequence for a custom refiner |
| --- | --- |
| [Exact SoulX receiver experiment](../ojin_components_soulx_distant_20260917/README.md): 320x576 at 15 FPS; average mouth width 57.6 pixels | Target small, low-detail mouths, not only large aligned face crops. Keep this exact character in evaluation. |
| Native-2x SR retains the soft tooth band; older SR creates conspicuous, sometimes brace-like divisions | Generic sharpness enhancement is an inadequate training objective. Reusing its output as unquestioned ground truth can teach the same defects. |
| [PRO/LITE comparison](../pro_lite_normal_distance_20260917/README.md) changes both details and mouth motion; opening trajectories correlate at 0.660 | Same audio, seed and frame number do not make PRO and LITE pixel-aligned training pairs. PRO output also has tooth-band artifacts. |
| [Repeated-still Lite VAE round trip](../ltx_transfer_20260917/README.md) retains visible tooth divisions in one example | VAE reconstructions can provide aligned diagnostic pairs, but a VAE round trip alone does not reproduce every SoulX generation defect. Moving clips need separate assessment. |
| `Engine.generate` computes its next motion latents before returning RGB frames, but `Engine.recondition` re-encodes sent-frame history on interruption/resumption | A delivery-stage refiner must preserve separate native frame history for reconditioning as well as ordinary generation; otherwise enhanced teeth can feed back into SoulX. |

The prior 66.85-FPS result was **postprocessing of an existing compressed SoulX receiver video**,
not fresh SoulX generation plus refinement. New training-pair capture should use pre-encode SoulX
RGB; receiver recordings remain useful for testing actual delivered quality.

## Existing character data is not yet a teeth dataset

The exact bank is
`/workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1`.
Its idle-file hash matches the original five-call recording metadata.

The six certified clips total **56.25 seconds / 1,350 frames**, all silent H.264 at 480x832,
24 FPS. Only two are named as speaking clips, totaling **24.08 seconds / 578 frames**.
These counts include repeated boundary frames. The source portrait is 941x1672, but dimensions
alone do not establish visible dental detail.

Crucially, both nominal speaking clips have **closed lips in all twelve inspected frames**
(1, 3, 5, 7, 9, 11 seconds in each clip). See the [CPU-extracted sample sheet](existing-character-speaking-bank-samples.jpg).
This is sampled evidence, not a claim that every frame in all local assets has been inspected.
The bank's names and frame counts do not establish a supply of sharp exposed teeth or varied
speech mouth shapes. Its silent clips also cannot directly provide audio-aligned speech supervision.

Therefore, these assets are useful for character appearance and closed-mouth negatives, but
are not sufficient evidence of a ready-to-train dental target set. Existing blurry SoulX videos
are inputs/evaluation cases, not their own clean targets.

## Proposed first architecture — our design, not a claim about Ojin

Use a compact residual CNN/U-Net, initially targeting **0.5–2 million parameters** and a
**128x128 canonical mouth/context crop**, optionally producing a 256x256 patch for 2x delivery.
These are design budgets, not measured performance or a selected trained architecture.

1. Detect and stabilize the crop location, rotation and scale. Keep the current lip opening;
   do not average the mouth pixels or expression across frames.
2. Feed the current crop, two prior aligned input crops, an inner-mouth mask and optional
   cached features from approved same-character teeth references to the network. Past-only
   context avoids adding future-frame lookahead. Reference features guide appearance without
   pasting a fixed smile over every phoneme.
3. Predict a bounded RGB correction and a confidence/visibility mask. Restrict the dental
   correction to the oral interior, with minimal blending at its boundary. Preserve lips,
   tongue occlusions and closed mouths. Landmarks locate lips but do not segment individual
   teeth; the current tracker must expose inner-lip landmarks, and a learned visibility mask
   or annotated oral segmentation is additional work.
4. Composite the patch into the display frame. Start with a neutral bicubic full-frame 2x
   baseline to isolate the refiner's effect; assess learned full-frame SR separately later.
5. Keep temporal state per session, reset it on cuts/tracking loss/avatar changes, and handle
   idle/speech transitions explicitly. Do not feed refined display pixels back into SoulX's
   motion-latent encoder in the first experiment.

Existing [mouth tracking](../../soulx_rtc/face_landmarker.py) and
[crop/composition code](../../soulx_rtc/mouth_sr.py) provide scaffolding. The current feather mask
includes the outer lips and is deliberately padded; it needs a different mask for a conservative
teeth-only refiner. The existing general-SR model is not a trained dental predictor.

## Training strategy and the difficult pairing problem

**Stage 1: learn restoration on truly aligned sequences.** Start with clear talking-mouth video,
retain the clean target, and derive the degraded input from that same sequence. Vary native mouth
size around the actual 50–65-pixel range, blur, contrast loss, resampling and compression. Add
moving Lite-VAE reconstructions as one degradation family after measuring their errors. Use many
mouth shapes and closed/occluded-mouth negatives. Any synthetic teacher footage needs review for
consistent teeth and geometry; it is a proposed appearance target, not recovered anatomical truth.

**Stage 2: address the gap between these corruptions and real SoulX output.** Actual SoulX errors
include malformed or merged teeth, not just blur. Evaluate Stage 1 on raw SoulX sequences early.
For supervised adaptation, create or curate corrected versions of the *same SoulX sequence* that
preserve lip contours, head pose and timing. Offline restorers and manually corrected keyframes
can propose targets, but must be checked and propagated through time with occlusion handling.
Reject bad pairs. Do not directly train L1 loss between independently generated LTX/PRO/LITE
videos: alignment of lips does not guarantee aligned tooth visibility or shape.

**Stage 3: temporal training and identity specialization.** Train short sequences with causal
context. Start with masked reconstruction, perceptual/detail and lip-boundary preservation losses;
use motion-compensated, occlusion-masked temporal losses so real mouth motion is not penalized.
Add a small adversarial loss only if needed after geometry is stable. A sync-preservation loss
may use frozen audio/video features during training without adding another runtime voice model.
Avoid optimizing Sobel/Laplacian sharpness alone; it rewards the artifacts we already rejected.

A few approved sharp references can establish a consistent intended tooth appearance for this
character, but cannot alone supply all speech poses. A personalized pilot is a smaller task than
a general refiner for arbitrary avatars. A broader system needs diverse high-quality speakers,
mouth shapes and separate held-out identities. Data quantity is a tuning decision after the pilot,
not a guarantee that a chosen number of minutes or iterations will solve the problem.

Split whole source clips before producing frames/corruptions; never put adjacent frames from the
same take into both train and validation. Hold out reference images as well where testing reference
generalization. Evaluate on new speech, motion and seeds, including the actual normal-distance
SoulX character that the user selected.

## Relevant public research

[OrthoNet, IJCAI 2025](https://www.ijcai.org/proceedings/2025/0211.pdf) addresses talking-video
teeth artifacts using mouth crops, teeth-focused restoration and short/long temporal memory.
It describes 96x96 crops and training on HDTF plus additional footage. That supports the research
direction; it provides no local SoulX or RTX 4070 SUPER performance result. Its linked
[repository](https://github.com/TKing-Su/OrthoNet-io) currently exposes a website/demos, with no
model/training files or releases found in the audited tree. We can study the paper but cannot
currently install it as a verified ready-made teeth refiner.

[KEEP](https://github.com/jnjaby/KEEP) does provide video-face restoration weights and training
code. [CodeFormer](https://github.com/sczhou/CodeFormer) supplies face restoration and training
code with a fidelity/quality tradeoff. They are candidate offline baselines or target generators,
not established solutions to SoulX teeth or guaranteed real-time mouth modules. Their existing
checkpoints were not run in this analysis. Training on synthetic degraded/clean pairs also has
an established implementation example in [Real-ESRGAN's training guide](https://github.com/xinntao/Real-ESRGAN/blob/master/docs/Training.md).

## Hardware feasibility and real-time limits

Training a small cropped CNN on this **RTX 4070 SUPER / 12,282 MiB** is plausible with mixed
precision and modest sequence batches. Freeze SoulX and precompute its outputs so its weights
need not share training memory. Model activations, sequence length, losses and optimizer determine
the real memory requirement. No training duration, peak RAM/VRAM or speed is measured here.

For inference, use **2–5 ms per mouth** as an initial engineering target for the *new refiner*,
then measure tracking, crops, transfers, SR, compositing and encoding separately and together.
It is not a forecast. Profile the trained network before TensorRT export, then validate export
numerically and visually. A fast random-weight network would only prove a compute envelope.

Five concurrent calls are a much harder target. The historical
[batch-five result](../concurrency_15fps_20260916/README.md) was approximately **77.36 useful
generation FPS against 75 FPS demand**. Purely serial arithmetic allows only about **0.41 ms
extra per output frame** before consuming that margin. Consequently, even a 2–5-ms refiner cannot
be assumed to preserve five-call capacity; CPU/GPU overlap, batching and admission limits must
be measured. The 66.85-FPS postprocessing-only result is already below the five-call aggregate
75-FPS demand. Establish one-call quality first, then retest concurrency with the full pipeline.

## Smallest meaningful next experiment

1. Build a vetted paired mouth dataset with real exposed tooth detail and preserved speech
   geometry; use this exact character for a personalization/evaluation subset.
2. Train the small spatial restoration baseline, then add causal context and reference conditioning
   as separate ablations. Save our architecture/configuration alongside our checkpoint.
3. Compare against the unmodified SoulX frame, bicubic and current public SR on held-out clips.
   Review normal-speed playback and oral crops: visible tooth structure, stability, no brace-like
   lines, no teeth appearing behind closed lips, and no new lip-sync drift.
4. Measure incremental GPU/CPU time, VRAM/RAM, p95 processing latency and first-frame delay on
   the actual SoulX serving path. Approve more calls only after a new concurrent receiver test.

**Recommendation:** pursue a small personalized mouth refiner, with the training-target audit
as the first milestone. This analysis establishes a credible implementation path and identifies
the missing data; it does not establish a trained teeth fix or reproduce Ojin's checkpoint.
