# Single SoulX Lite face-ROI experiment

September 17, 2026. Fresh GPU inference on **NVIDIA GeForce RTX 4070 SUPER**,
physical 12 GB class / **12,282 MiB visible**, driver **595.84**, PyTorch
**2.7.1+cu128 / CUDA 12.8**. Direct `nvidia-smi` snapshots, runtime versions,
input hashes and per-run samples are in [results.json](results.json).
OmniVoice (2,478 MiB) and idle LTX (234 MiB) remained resident; initial
whole-device use was 2,727 MiB. Face tracking, alignment, blending, packaging
and media checks are CPU work. No live service was modified.

**Finding:** existing Lite weights can generate a face crop and return it to a
normal-distance portrait with enough measured throughput for a 25-FPS target.
This experiment does **not** establish a teeth or lip-sync improvement. The
256-pixel version visibly fails; 384 and 512 are more stable, but soft/connected
teeth remain. Alignment reduces position mismatch by suppressing head motion
against a static body/background. This is a useful feasibility result, not a
finished replacement for whole-frame SoulX.

## Watch the results

The requested hybrid rerun was completed and recorded **before** this test:
[SoulX Lite versus Lite + MuseTalk, 1.50x](../roi_hybrid_analysis_20260917/recorded-comparison/soulx-LITE-vs-SoulX-plus-MuseTalk-indian-man-1.50x.mp4).
That experimental adapter ran at 15.39 FPS and softened the sampled mouths;
[its report](../roi_hybrid_analysis_20260917/README.md) retains the separate
hybrid method, resource use and limitations. The single-model experiment below
uses the **normal 1.00x** portrait, so it is not a framing-matched quality A/B
against that hybrid.

Every single-model comparison has four labeled columns: **full-frame Lite,
256 ROI, 384 ROI, 512 ROI**. All are ten seconds, 25 FPS, with the same audio.

| Composition | Seed 50 | Seed 51 | What to look for |
| --- | --- | --- | --- |
| Align the face to the original portrait | [Watch](soulx-LITE-full-frame-vs-face-ROI256-384-512-aligned-normal-distance-seed50.mp4) | [Watch](soulx-LITE-full-frame-vs-face-ROI256-384-512-aligned-normal-distance-seed51.mp4) | More stable placement at 384/512; static body/hair; residual facial softness |
| Paste the generated square at a fixed location | [Watch](soulx-LITE-full-frame-vs-face-ROI256-384-512-fixed-normal-distance-seed50.mp4) | [Watch](soulx-LITE-full-frame-vs-face-ROI256-384-512-fixed-normal-distance-seed51.mp4) | Head/neck drift and boundary mismatch, especially 256 |

Inspect the pre-encode mouth samples for [seed 50](soulx-LITE-full-frame-vs-face-ROI256-384-512-aligned-normal-distance-seed50-mouths.png)
and [seed 51](soulx-LITE-full-frame-vs-face-ROI256-384-512-aligned-normal-distance-seed51-mouths.png).
These are equal-sized native 96x52-pixel regions enlarged 2x with nearest
neighbor, not a sharpened or learned reconstruction. Full-frame sheets are
saved beside each comparison. Individual native crops and composited clips
are retained under the corresponding `face-crop<size>-seed<seed>` directory.

## What was implemented

[run.py](run.py) uses **one SoulX Lite visual generator**, its existing VAE and
its existing Wav2Vec2 audio encoder. It loads no MuseTalk, Whisper or SR weights.
The original voice/audio is reused. MediaPipe FaceMesh is an additional small
CPU landmark model for geometry; “single model” here means one face/video
synthesis model, not literally one neural network anywhere in the pipeline.

1. Start with the original 320x576 normal-distance Indian-man portrait.
2. Extract the same 288x288 head/neck region at `(16,48)-(304,336)` and resize
   it to 256, 384 or 512 square. This reallocates generated pixels to the face;
   interpolation does not create new source detail.
3. Run the unchanged eager BF16 Lite pipeline at that square resolution:
   four steps, shift 5, color correction 1, 33-frame chunks with nine-frame
   overlap and 24 new frames, 25-FPS audio conditioning. Run seeds 50 and 51.
4. Compose each generated crop onto the original normal-distance portrait,
   using either a feathered square or a face aligned by eye/upper-nose landmarks.

The aligned branch estimates a per-frame 2D similarity transform from six
upper-face landmarks, excluding the moving mouth. It blends through a
feathered reference face-oval mask. Pixels where that mask is zero are asserted
to equal the original plate exactly **before video encoding**. It preserves
the static body/background and largely anchors the head. It does not generate
only mouth pixels, preserve unconstrained body/head animation, solve 3D pose,
or add trained mouth-mask conditioning to Lite.

One shape-specific warmup precedes the two measured seeds. Each seed resets
reference/motion state and random generators. Matching seed numbers across
different canvas shapes does not imply matching noise tensors or head motion.

## Throughput

All rows: the RTX 4070 SUPER / 12,282 MiB profile above. Ranges are the two
different seeds, **not** repeated-run confidence intervals. Useful frames = 250;
the final chunk's unused frames are discarded but their computation is timed.

| Generation canvas | Generation only | Generation + fixed paste | Generation + aligned face | Time for 250 aligned frames |
| --- | ---: | ---: | ---: | ---: |
| Full frame 320x576 | 55.79–55.82 FPS | — | — | 4.48 s (full frame) |
| Face 256x256 | 128.63–131.34 FPS | 103.92–105.25 FPS | 51.93–52.18 FPS | 4.79–4.81 s |
| Face 384x384 | 70.63–70.64 FPS | 60.84–61.54 FPS | 38.94–39.29 FPS | 6.36–6.42 s |
| Face 512x512 | 41.01–41.02 FPS | 37.27–37.42 FPS | 28.11–28.43 FPS | 8.79–8.89 s |

Generation includes audio features, denoising, VAE decode, color correction,
motion re-encode and transfer to CPU, with CUDA synchronization. Both alternative
CPU compositors execute in the diagnostic loop; each branch's FPS is calculated
from **measured generation time plus that compositor's measured time**. The
whole diagnostic wall time also includes the unused alternative and is retained.
Alignment/tracking/blending costs about 10.8–11.6 ms per retained frame.

Loading, warmup, source preparation, video encoding, audio acquisition and
network/browser delivery are excluded. This is measured stage throughput,
not an end-to-end live/WebRTC benchmark. At 512 the branch uses about 35.2–35.6
ms of a 40-ms frame budget, leaving little room for additional serial work or
jitter. The 384 branch has more margin at about 25.5–25.7 ms/frame. No compiled,
TensorRT, concurrent-avatar or optimized-overlap claim follows from these runs.

## Resources

All rows: RTX 4070 SUPER, driver/runtime and resident load as above. Peaks are
across the two seeds. Whole-device VRAM includes resident services. Process RSS
is host RAM, not VRAM. CPU process 100% is approximately one logical CPU; host
CPU is normalized across 32 logical CPUs.

| Canvas | Sampled device VRAM peak | Exact Torch allocated peak | Torch reserved peak | Process RSS peak | Host RAM used peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full frame | 8,294 MiB | 4,785 MiB | 5,332 MiB | 4,909 MiB | 11,844 MiB |
| 256 ROI | 7,864 MiB | 4,354 MiB | 4,890 MiB | 5,374 MiB | 12,262 MiB |
| 384 ROI | 8,094 MiB | 4,650 MiB | 5,120 MiB | 5,465 MiB | 12,358 MiB |
| 512 ROI | 8,670 MiB | 5,068 MiB | 5,696 MiB | 5,672 MiB | 12,551 MiB |

| Canvas | Sampled GPU use mean / peak | Process CPU mean / peak | Host CPU mean / peak |
| --- | --- | --- | --- |
| Full frame | 98% / 100% | 100.1–100.2% / 101.8% | 6.4–6.8% / 10.5% |
| 256 ROI | Sampling missed GPU bursts; see below | 101.4–101.7% / 104.3% | 5.9–6.4% / 10.0% |
| 384 ROI | 42.1–45.8% / 100% | 101.4% / 104.2% | 6.6–6.7% / 10.5% |
| 512 ROI | 58.0–63.4% / 100% | 101.0–101.2% / 102.9% | 6.0–6.5% / 10.8% |

Half-second `nvidia-smi` sampling returned zero utilization for **every 256 ROI
sample despite synchronized CUDA generation**. These samples cannot estimate
its GPU utilization; short bursts alternating with CPU composition can be
missed/aliased. All resource samples cover the loop executing **both** CPU
composition branches, so they do not describe an optimized aligned-only service.
RAM also includes retained video arrays for diagnostic recording and allocations
from preceding runs. These are observed experiment peaks, not minimum deployment
requirements. Raw samples are preserved without substituting inferred GPU use.

## Quality and geometry findings

The review used seven pre-encode frames per variant at approximately 0.5, 1.5,
2.5, 3.5, 4.5, 6.5 and 8.5 seconds, plus all-frame tracking diagnostics.

- **256 fails visually in both seeds.** Large head/identity changes, softness
  and mismatched face placement remain after alignment. Seed 50 loses tracking
  at frames 63 and 64 and returns the static plate for those two frames; that
  fallback is not acceptable continuous lip-sync behavior.
- **384 is a plausible geometry/throughput development baseline.** Head
  movement against the fixed plate still creates visible neck/jaw mismatches
  with a square paste. Face alignment reduces that displacement, but mouth
  detail remains soft and tooth divisions are not reliably improved.
- **512 is more stable geometrically**, and some samples look cleaner than
  384, but the extra work does not establish a consistent anatomical tooth gain
  over full-frame Lite. Bright connected tooth regions and soft boundaries
  remain in the inspected native and final samples.
- The rounded-mouth event around 2.5 seconds remains recognizable in 384/512,
  whereas mouth aperture differs elsewhere. That observation is **not** a
  lip-sync accuracy measurement. No SyncNet, annotated phoneme alignment or
  human perceptual evaluation was performed.

Mean upper-face landmark displacement relative to the static reference,
expressed in final 320x576 pixels (seed 50 / seed 51):

| Canvas | Before alignment | After alignment | Successfully tracked frames |
| --- | ---: | ---: | ---: |
| 256 | 41.42 / 19.66 px | 2.68 / 1.71 px | 248/250; 250/250 |
| 384 | 7.11 / 7.59 px | 1.01 / 0.88 px | 250/250 each |
| 512 | 3.54 / 3.57 px | 0.70 / 0.64 px | 250/250 each |

These landmarks are the anchors used by the transform. Smaller residuals show
that alignment did its geometric job; they are not independent evidence of
identity, dental accuracy or natural motion. A 2D transform cannot correct
out-of-plane pose or generated identity deformation.

## What the code investigation establishes

The stock [pipeline](../../flash_head/src/pipeline/flash_head_pipeline.py)
already takes a target canvas and VAE-encodes a repeated reference portrait.
The test works through those inputs; no new model weights or sampler changes
were required. The [model](../../flash_head/src/modules/flash_head_model.py)
concatenates noisy and reference latents and applies transformer blocks across
the complete crop. There is still no explicit mouth mask or per-frame body/pose
conditioning. Its preserved temporal prefix is motion history, not spatial
inpainting of a surrounding video.

The installed [Lite config](../../models/SoulX-FlashHead-1_3B/Model_Lite/config.json)
uses spatial VAE stride 32, temporal stride 8 and 1x1x1 patches. Five latent
frames represent each 33-frame chunk:

| Canvas | Spatial grid | Total tokens | Approximate reference mouth width / latent spacing |
| --- | --- | ---: | ---: |
| Full frame | 10x18 | 900 | 52 px / 1.62 |
| 256 crop | 8x8 | 320 | 46 px / 1.44 |
| 384 crop | 12x12 | 720 | 69 px / 2.17 |
| 512 crop | 16x16 | 1,280 | 92 px / 2.89 |

Mouth widths are source FaceMesh corner distance, geometrically scaled by the
crop resize, not measured generated dental resolution. The 256 path actually
allocates **fewer** pixels to this source mouth than the baseline. The 384/512
paths allocate more, but reuse the same source information, learned dental
prior and compressed representation; compositing back to 320x576 also resamples
the generated detail. A latent has 128 channels and a decoder receptive field,
so spacing is not a hard bound on representable tooth boundaries.

This narrows the practical bottleneck: **routing Lite to a face ROI and getting
sufficient compute throughput are feasible; reliable detail generation and
motion-preserving composition remain unresolved.** A different crop size alone
did not remove tooth softness. This test does not isolate training prior versus
VAE loss versus source limitation as a single proven cause.

For further single-generator work, 384 is a sensible compositor-development
starting point and 512 a quality reference. A real teeth/lip-sync fix needs an
independently demonstrated quality gain, likely from better conditioning or
targeted training, before service integration. Preserving freely moving bodies
while changing only the mouth additionally needs per-frame pose/boundary
conditioning; the static-plate experiment does not solve that requirement.

## Validation and reproduction

- Eight fresh generations completed: full frame plus three crop sizes, two
  seeds each. Twenty individual native/composited videos and four comparison
  videos fully decode to 250 frames at 25 FPS and contain audio.
- All 1,000 aligned 384/512 frames passed face detection; pixels outside the
  face mask were asserted unchanged in the pre-encode buffers. The two 256
  tracking failures are explicitly retained, not hidden or silently dropped.
- [Analysis and individual-media validation](analysis-and-media-validation.json)
  retains aggregates, failure indices, alignment residuals, source mouth scale
  and ffprobe records. Each comparison has its own `-validation.json`.
- Native PNG samples precede H.264 encoding, so the visible softness cannot be
  attributed solely to export compression. Visual findings are qualitative,
  with only one source identity, one utterance and two seeds.

To repeat into a new directory within this repository:

```bash
PYTHONPATH=. .venv/bin/python benchmarks/single_lite_face_roi_20260917/run.py \
  --output benchmarks/single_lite_face_roi_NEW_RUN
```

[The CPU review helper](../roi_hybrid_analysis_20260917/review.py) accepts repeated
`--video` and `--label` pairs plus `--prefix` to package comparisons. Video/PNG
encoding and media validation are outside the inference timing.
