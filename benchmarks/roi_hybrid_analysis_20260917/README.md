# SoulX-only mouth ROI feasibility and hybrid throughput

September 17, 2026. Code inspection plus diagnostic GPU inference on **NVIDIA
GeForce RTX 4070 SUPER**, physical 12 GB class / **12,282 MiB visible**, driver
595.84, PyTorch 2.7.1+cu128 / CUDA 12.8. The benchmark records direct GPU/process
snapshots in `eager-measurement/results.json`. OmniVoice (2,478 MiB) and idle LTX
(234 MiB) remained resident; initial whole-device use was 2,727 MiB. Tracking
and blending are CPU work. This adds an isolated benchmark, not a deployed
hybrid, altered production model, or validated teeth/lip-sync improvement.

## What "another model" means

MuseTalk synthesizes face pixels from existing audio. It does not synthesize a
voice. SoulX's Wav2Vec2 and MuseTalk's Whisper components extract different audio
features; the waveform and its voice remain the same. Their feature tensors are
not interchangeable without an adapter/training.

## Can the existing SoulX model do ROI work?

Yes, some forms are implementable with its existing weights. It is too strong
to call all SoulX-only ROI approaches impossible. The distinction is between
implementing a spatial restriction and obtaining a reliable quality gain.

| Approach | Existing weights usable? | Main constraint |
| --- | --- | --- |
| Generate a tightly framed face/head and composite it onto a plate | Yes, experimental | Gives the face more pixels, but generated pose/jaw/neck must match the plate; body/background motion is reused |
| Restrict an extra denoising pass to a mouth mask | Can be prototyped | Same compressed grid and untrained inpainting behavior; a mask does not skip full transformer computation |
| Full-frame Lite plus a second zoomed face pass using the same Lite weights | Yes, as two generation states sharing weights | Adds compute and independent head/mouth motion; no stock per-frame pose lock joins the two streams |
| Reliable audio-driven mouth inpainting inside SoulX | Needs new conditioning and task-specific training/fine-tuning | Must learn preservation, boundary context, lip synchronization and temporal/dental consistency |

The most plausible same-model, no-training experiment is **head/face-crop
generation plus compositing**, rather than a crop containing only the teeth.
The subsequent [single-Lite experiment](../single_lite_face_roi_20260917/README.md)
now executes it at normal distance with 256/384/512 crops and two seeds on the
same RTX 4070 SUPER profile. The aligned 384 and 512 branches measured roughly
39 and 28 FPS respectively, but neither establishes a teeth/lip-sync gain.
It trades freely moving body/background animation for a static plate; the
256 branch has severe visual failures. That follow-up is separate from the
1.50x hybrid measurements below.

## Code evidence

- [Pipeline conditioning](../../flash_head/src/pipeline/flash_head_pipeline.py):
  `prepare_params` accepts a portrait, target canvas, temporal length, schedule,
  seed and face-crop flag. It repeats the reference image and VAE-encodes it.
  `use_face_crop` crops the input portrait; it is not a tracked output mouth ROI.
- [Model input and attention](../../flash_head/src/modules/flash_head_model.py):
  `WanModelAudioProject.forward` concatenates noisy video `x` with reference `y`,
  patchifies every spatial/temporal location, then runs 30 transformer blocks.
  The installed Lite config has 256 input channels = 128 noisy + 128 reference.
  There is no explicit mouth-mask conditioning or trained local repair branch.
  Self-attention couples the sequence; audio cross-attention is not mouth-gated.
- [Lite config](../../models/SoulX-FlashHead-1_3B/Model_Lite/config.json):
  VAE stride is 8 temporal / 32 spatial, patch size 1x1x1. A 320x576 canvas gives
  a 10x18 spatial grid, or 900 tokens over five latent frames. Cropping a
  256x256 canvas would give 8x8/five frames = 320 tokens, but changes framing
  and model input distribution; token count is not a measured speedup.
- At the measured 1.50x framing, an approximately 85-pixel mouth spans about
  2.7 latent spacings. A mouth mask on that existing grid does not add spatial
  samples. Each latent also has 128 channels and a decoder receptive field, so
  this is not proof that the VAE cannot represent tooth boundaries.
- [Pipeline recurrence](../../flash_head/src/pipeline/flash_head_pipeline.py):
  the sampler preserves initial temporal motion latents, decodes the complete
  canvas, corrects color, and re-encodes the final nine Lite frames for the next
  chunk. Temporal-prefix preservation is different from spatial inpainting.
- [Existing refinement](../refinement_20260917/README.md): extra whole-latent
  evaluations with the same weights did not consistently restore tooth detail.
  That is evidence against "more of the same" as an assumed fix, not an executed
  test of masked or zoomed ROI refinement.

Masking the predicted updates would still execute the full model. Skipping
non-mouth tokens changes the attention context and positional behavior; it is
not an equivalent optimization. A proper masked sampler would also need to
preserve outside latents at the correct noise level, supply moving boundary
context, allow jaw motion, and account for decoder influence beyond the ROI.
Adding a mask channel cannot be done by simply loading the unchanged first
layer weights into a larger input layer.

## Diagnostic benchmark contract

[benchmark.py](benchmark.py) runs fresh SoulX Lite, then processes each emitted
chunk through a face stage before the next SoulX chunk. Both networks stay
resident on the same GPU. This measures sequential inference work rather than
adding advertised standalone FPS values.

- 1.50x Indian-male reference; same ten-second audio, seed 50, 320x576, four
  steps, 250 useful frames, 25-FPS target.
- SoulX BF16 eager official pipeline; MuseTalk 1.5 FP16 eager. No TensorRT,
  compilation, quantization, weight offload or overlap optimization.
- One warmup plus three measured repeats for Lite alone and hybrid face batches
  of eight and four. Final partial batches are executed, not padded/repeated.
- Every new generated frame is tracked with CPU FaceMesh, cropped/resized to
  256x256, encoded as both a masked and full reference face, passed through
  MuseTalk, decoded, copied to CPU, and blended into the lower face.
- A geometric feather mask and FaceMesh replace production DWPose/face parsing
  in this diagnostic adapter. No precomputed face boxes or face latents are
  used. It is not a qualification of the production compositor or teeth quality.
- Timing includes tracking, both VAE encodes, mouth network, VAE decode,
  transfers and blending. Model loading, reference preparation, video encoding,
  network delivery and audio acquisition are excluded. MuseTalk's full-utterance
  audio preprocessing is measured separately; audio is known ahead of time.
  This is favorable to throughput and does not establish live lookahead latency.
- Stage timings synchronize CUDA. Resource samples are approximately every
  half-second; short peaks can be missed. Exact Torch peaks are recorded too.
  System metrics include resident services. Batch-four reserved VRAM can retain
  batch-eight allocator caches; compare live allocation separately.

## Measured result: eager hybrid misses 25 FPS

### Recorded follow-up

The requested fresh recorded repeat is now available: [Lite versus Lite +
MuseTalk, 1.50x Indian-man reference](recorded-comparison/soulx-LITE-vs-SoulX-plus-MuseTalk-indian-man-1.50x.mp4),
[native mouth samples](recorded-comparison/soulx-LITE-vs-SoulX-plus-MuseTalk-indian-man-1.50x-mouths.png),
and [run record](recorded-comparison/results.json). This uses the same RTX 4070
SUPER, 12,282 MiB visible, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8 with the
same co-resident services. Encoding and PNG saves occur after timing. The
batch-eight recorded hybrid ran at **15.39 FPS**, using **11,352 MiB** in the
post-run whole-device snapshot. Its underlying SoulX RGB matched the fresh
baseline exactly.

In the seven inspected native samples, this experimental FaceMesh/geometric
mask adapter makes the lower face and mouth softer and changes articulation
substantially, including the rounded-mouth moment at 2.5 seconds. It does not
provide a teeth-quality improvement in these samples. This is evidence about
this adapter, not a controlled verdict on every MuseTalk crop/mask pipeline.
Both individual clips and the comparison fully decode to 250 video frames
at 25 FPS with audio; validation is retained beside the comparison.

All rows below ran on the RTX 4070 SUPER / 12,282 MiB described above. Values
are medians of three measured ten-second runs after warmup. The GPU remained
shared with the stated resident processes.

| Pipeline | Time for 250 frames | Useful FPS | FPS range | First completed 24-frame chunk |
| --- | ---: | ---: | ---: | ---: |
| Lite alone | 4.488 s | 55.71 | 55.70–55.85 | 0.503 s |
| Lite + MuseTalk, face batch 8 | 16.182 s | 15.45 | 15.41–15.51 | 1.585 s |
| Lite + MuseTalk, face batch 4 | 16.455 s | 15.19 | 15.18–15.23 | 1.605 s |

The first-chunk figure excludes model loading, reference preparation, incoming
audio accumulation and MuseTalk audio preparation. It is not browser first-frame
latency. A future implementation could release the first face batch before the
whole chunk is composited, but that would not eliminate the throughput deficit.

Batch eight's median chunk duration is 1.524 seconds for mostly 24-frame chunks;
24 frames occupy only 0.960 seconds at 25 FPS. The producer falls behind during
continued speech, even if a larger initial buffer hides the first underrun.

### Where the time goes

Batch-eight median stage time per useful output frame:

| Stage | Time |
| --- | ---: |
| SoulX generation, audio, decode and motion re-encode | 17.61 ms |
| Face tracking, cropping and upload | 4.09 ms |
| Encode masked and full new face into MuseTalk's VAE | 17.44 ms |
| MuseTalk audio conditioning and UNet | 5.05 ms |
| MuseTalk VAE decode and transfer | 17.92 ms |
| Lower-face blending | 2.58 ms |
| Total wall time | 64.73 ms |

Small differences between summed medians and median total come from aggregation
and loop overhead. MuseTalk audio preparation outside this loop measured 0.123 s
cold and 0.007 s warm for the known ten-second utterance. A live audio encoder
with finite lookahead requires a separate evaluation.

The main added expense is **face VAE encoding plus decoding**, about 35.36
ms/frame. The moving SoulX output requires new face encodes each frame. Existing
MuseTalk avatar loops can precompute those latents, so their rendering-only FPS
does not describe this hybrid.

### Resource use

Sampled values across the three measured runs, with exact Torch allocation
peaks. Whole-device measurements include the resident baseline; RSS means CPU
process RAM, not GPU memory. CPU process 100% means approximately one logical
core; system CPU is normalized over 32 logical CPUs.

| Metric | Lite alone | Hybrid batch 8 | Hybrid batch 4 |
| --- | ---: | ---: | ---: |
| Whole-device VRAM peak | 8,264 MiB | 11,348 MiB | 11,348 MiB |
| Exact Torch allocated peak | 4,787 MiB | 7,029 MiB | 6,665 MiB |
| Torch reserved peak | 5,302 MiB | 8,366 MiB | 8,366 MiB |
| GPU use mean / peak | 98.5% / 100% | 89.1% / 100% | 88.2% / 100% |
| Process RSS peak | 4,930 MiB | 6,172 MiB | 6,208 MiB |
| Host RAM used peak | 11,672 MiB | 13,255 MiB | 13,090 MiB |
| Process CPU mean / peak | 100.1% / 102.9% | 100.7% / 103.9% | 100.7% / 103.9% |
| Host CPU mean / peak | 6.3% / 9.8% | 6.6% / 11.5% | 6.6% / 13.2% |

Batch four reuses the allocator cache grown by batch eight; equal reserved and
device memory therefore do not establish equal minimum memory requirements.
The tested hybrid leaves 934 MiB of reported whole-device headroom. Compiled/TRT
workspace requirements could change that substantially. No services were
stopped and no model was moved off GPU between stages.

## Real-time implications and decision

At 25 FPS the total compute budget is 40 ms/frame. Lite alone costs about 17.95
ms/frame including loop overhead, leaving roughly **22 ms/frame** for the
complete additional stage. The measured hybrid adds roughly **46.8 ms/frame**.
It needs about **1.62x total throughput**, or approximately **2.1x acceleration
of the added work** while holding Lite cost fixed, just to reach 25 FPS without
encoding/network overhead or operating margin.

Making only the 5.05-ms UNet infinitely fast would still leave about 59.7
ms/frame, or 16.8 FPS. Eliminating the 17.44-ms face-encoding stage entirely
would still leave about 47.3 ms/frame, or 21.1 FPS. Neither is a sufficient fix
on its own. Realistic acceleration would need to address both VAE directions,
tracking/transfers/composition, and possibly SoulX too. These are arithmetic
bounds from the recorded profile, not executed optimizations.

This does not prove an optimized hybrid can never be real-time. It does show
that the actual eager implementation is not suitable for continuous 25-FPS
generation on this shared card. Cached-face benchmark FPS, TensorRT UNet-only
FPS, or adding two standalone FPS values would all overstate its capacity.
No compiled or TensorRT hybrid was measured; no claim is made about their
maximum achievable speed or quality.

For the user's preference to retain SoulX, the [completed single Lite
face/head-crop test](../single_lite_face_roi_20260917/README.md) demonstrates a
faster feasible route, but not a proven dental-detail fix. If unconstrained
whole-body/head motion and strictly mouth-only changes are both mandatory, a
trained mouth-inpainting or detail-refinement adaptation remains a longer-term
direction; it is not an existing inference switch. Neither tested path has a
demonstrated lip-sync or teeth-quality gain yet.

## Validation and reproduction

- Nine measured runs plus three warmups completed. All outputs contain 250
  useful frames; every hybrid run tracked all 250 faces and ran every frame
  through both face encodes, the UNet, decoding and blending.
- Raw SoulX RGB hashes match across **all nine measured runs**, confirming the
  mouth pass did not perturb its recurrent state or baseline random stream.
- The original timing output was calculated in memory and hashed. The recorded
  follow-up above adds footage and sampled visual inspection. No quality-scored footage,
  new WebRTC service, model fine-tuning or production change accompanies this
  performance analysis. This is not the earlier generic mouth-SR benchmark.
- Raw stage timings, chunk durations, hashes, environment and resource samples:
  [results.json](eager-measurement/results.json).

```bash
PYTHONPATH=. .venv/bin/python benchmarks/roi_hybrid_analysis_20260917/benchmark.py \
  --output benchmarks/roi_hybrid_analysis_20260917/NEW_RUN --repeats 3
```

Upstream architectural cross-checks: [MuseTalk's audio-driven face inpainting](https://github.com/TMElyralab/MuseTalk)
and [SoulX's portrait-video generation](https://github.com/Soul-AILab/SoulX-FlashHead).
The inspected local source and fresh measurements govern the findings here;
upstream performance on other GPUs is not substituted for this benchmark.
