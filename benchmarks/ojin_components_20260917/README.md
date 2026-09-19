# Ojin component audit and FaceLandmarker / native-2x TensorRT tests

September 17, 2026. Fresh local GPU inference: **NVIDIA GeForce RTX 4070 SUPER**,
physical **12 GB class / 12,282 MiB visible**, driver **595.84**, PyTorch
**2.7.1+cu128 / CUDA 12.8**, TensorRT **10.16.1.11, cu13 package**, MediaPipe
**0.10.35**. OmniVoice (2,478 MiB) and idle LTX (234 MiB) remained resident;
initial whole-device use was 2,727 MiB. Direct GPU/process snapshots and
versions are in [the main run record](matched-runtime/results.json).
Tracking is CPU/XNNPACK with an NVIDIA EGL context; SR and Lite use the GPU.
Packaging, aggregation and media validation are CPU work.

**Completed:** Google Tasks FaceLandmarker, Ojin's listed MediaPipe and TensorRT
versions, a locally built **native 2x SRVGG** engine, and full-frame **512→1024**
delivery have now been tested and recorded. **This is still not an exact Ojin
reproduction:** its upscaler checkpoint/build recipe and service/refinement
code are not in the public repository. A public, natively trained 2x checkpoint
is explicitly substituted. The proprietary mouth refiner is omitted.

The subsequent [custom mouth-refiner analysis](../mouth_refiner_feasibility_20260917/README.md)
assesses an independent architecture, training data, temporal stability and serving cost.
The failed substitute tests do not establish that Ojin's private refiner alone explains his
reported result; the other unpublished pipeline details remain uncontrolled.

The corrected requested follow-up uses [the exact SoulX batch-five character and
receiver recording](../ojin_components_soulx_distant_20260917/README.md). Rebuilt for
320x576→640x1152, the accessible path processed all 233 frames at **66.85 FPS**
with **233/233** FaceLandmarker detections. It sharpens local texture without visibly
repairing SoulX's soft tooth band. The older 4x SR runs at 17.13 FPS and creates
stronger but sometimes brace-like tooth divisions. The prior [LTX-video control](../ojin_components_ltx_distant_20260917/README.md)
resulted from a target misunderstanding; it remains supplementary rather than the
answer to the SoulX-character request.

The corrected path runs at **24.04–24.17 FPS** in this serial eager test, before
encoding/delivery and without mouth refinement. It does not demonstrate a teeth
fix. The public 2x model mostly preserves the source's softness; the earlier
4x model invents stronger, sometimes visibly unnatural tooth texture.

## Recordings

Three equal-resolution columns: **bicubic 2x baseline / earlier public 4x SR
resized to 2x / native 2x TensorRT with FaceLandmarker, no mouth refiner**.

- [Seed 50 comparison](matched-runtime/soulx-LITE-bicubic-vs-oldSR-vs-native2xTRT-FaceLandmarker-512to1024-seed50.mp4)
- [Seed 51 comparison](matched-runtime/soulx-LITE-bicubic-vs-oldSR-vs-native2xTRT-FaceLandmarker-512to1024-seed51.mp4)
- Pre-encode mouth samples: [seed 50](matched-runtime/soulx-LITE-bicubic-vs-oldSR-vs-native2xTRT-FaceLandmarker-512to1024-seed50-mouths.png), [seed 51](matched-runtime/soulx-LITE-bicubic-vs-oldSR-vs-native2xTRT-FaceLandmarker-512to1024-seed51-mouths.png).
- Tracker overlays: [seed 50](matched-runtime/facemesh-vs-FaceLandmarker-seed50.png), [seed 51](matched-runtime/facemesh-vs-FaceLandmarker-seed51.png); cyan = old FaceMesh, yellow = Tasks FaceLandmarker.

Individual native 512 and enhanced 1024 clips are retained in descriptively
named directories under `matched-runtime/`. All clips are ten seconds, 250
frames, 25 FPS, with the same speech audio. The video rate is playback cadence;
the measured production rate is listed separately below.

## What was actually public, and what matches

The public [Ojin repository](https://github.com/ojinai/kit-example/tree/819fe6e95613fdcc7bffd8676e8814282cd445f5)
contains client examples, deployment files and notices. Its README describes a
separately delivered encrypted container, requiring a download URL/checksum and
separately supplied passphrase. Its public tree has no inference server source,
SR checkpoint, SR engine or mouth-refiner checkpoint; no GitHub releases were
listed at audit time. The notices identify Ojin's own code and mouth-refiner
weights as proprietary. These are broader gaps than “only one inaccessible
weight file.” Pinned files and the recursive repository tree are in
[source-audit](source-audit/); downloads are hashed in [source-lock.json](source-lock.json).

| Component | Ojin disclosure | This test / remaining gap |
| --- | --- | --- |
| Generator and audio features | SoulX Lite, LTX VAE, Wav2Vec2 | Installed official Lite pipeline and weights; exact Ojin sampler/settings/weight hashes undisclosed |
| Mouth locator | Google `face_landmarker.task`, MediaPipe 0.10.35 | Google float16/version-1 task, Tasks VIDEO mode, CPU delegate, MediaPipe 0.10.35; Ojin's task hash and tracker options undisclosed |
| Upscaler | Delivery 512→1024, SRVGGNetCompact, TensorRT | Native 2x SRVGG, full-frame output, real TensorRT execution; exact Ojin checkpoint, network depth and precision/build settings undisclosed |
| TensorRT package | Manifest lists 10.16.1.11 / cu13 | Matched and rebuilt locally for this GPU |
| Mouth refiner | `refiner_weights.pt`, Ojin/Journee proprietary | Omitted; no synthetic stand-in claimed |
| Crop, refinement order, stabilization, blending | Service internals not published | Not reproduced; locator coordinates recorded, no invented claim about its missing refiner's I/O or routing |
| Hardware | README targets Blackwell RTX PRO 6000, compute capability 12.0 | Tested on Ada RTX 4070 SUPER; their exact VRAM and performance unverified |
| Transport | Public WebSocket client and container deployment | Inspected; local media recorded, no Ojin container or live service test |

The earlier test used the already available FaceMesh API and public general-x4v3
weights as a functional prototype. Those choices did not establish fidelity to
Ojin. FaceLandmarker and TensorRT were technically testable, and are now tested.

Ojin's 2x delivery description also does not expose the network's internal
learned scale or any hidden resizing. Our substitute is independently confirmed
to be native 2x; the public disclosure alone cannot prove those internals match.

The manifest's prose mentions CUDA 12.8, while its package appendix lists
TensorRT cu13. We matched the explicit TensorRT package versions; Torch remains
2.7.1+cu128. Other packages are not a byte-for-byte environment clone: local
NumPy remains 2.2.6, and ONNX 1.17.0 is a local export utility. No public Ojin
export recipe was available. The base serving venv was preserved; the new
packages live under `/workspace/experiments/ojin-matched-deps`.

## Implementation and checkpoint identity

- [FaceLandmarker adapter](../../soulx_rtc/face_landmarker.py): actual
  `mediapipe.tasks.vision.FaceLandmarker`, explicit CPU delegate, VIDEO mode,
  monotonically increasing 40-ms timestamps, one face, 20 outer-lip points.
  Default confidence thresholds are 0.5; these are local choices, not verified
  Ojin settings. One tracker is created per independent stream.
- [Native-2x TensorRT adapter](../../soulx_rtc/srvgg_trt.py): fixed 512-square
  RGB input, 1024-square output, no intermediate 4x image and no post-downscale.
  Engine metadata/hash, runtime/GPU and binding geometry are checked. There is
  no silent PyTorch fallback.
- The public substitute is Philip Hofmann's
  [2xNomosUni_compact_multijpg_ldl](https://github.com/Phhofm/models/releases/tag/2xNomosUni_compact_multijpg_ldl):
  **CC BY 4.0**, native 2x SRVGGNetCompact, 16 body convolutions, 64 features,
  PReLU. It is a photo-oriented, depth-of-field-preserving model trained with
  resize/JPEG degradations. This identifies why it was a plausible public
  architecture match; it is not evidence that Ojin uses it. Its license/training
  and parameter count are not attributed to Ojin.
- Weights SHA-256:
  `dec25af2672783ce2214c38d6a70caf8c3d8a640d6f4d87b543b264cafd74110`.
  SRVGG architecture retains the upstream Real-ESRGAN BSD-3-Clause notice.
- [Engine builder](build_engine.py) enables FP16 with FP32 I/O, a 512-MiB
  workspace limit and optimization level 3. Build time, engine/ONNX hashes,
  inspector output and numerical checks are retained in
  [build.json](matched-runtime/build.json) and [engine-layers.json](matched-runtime/engine-layers.json).

The executed path is **Lite generation → FaceLandmarker mouth coordinates →
full-frame native-2x SR**. No face warp or mouth blending occurs: the stage that
would use mouth coordinates to refine pixels is unavailable. Therefore changing
the tracker alone cannot change this test's image output. Its cost and tracking
behavior are still measured rather than silently skipped.

## Matched fresh generation results

All rows below: RTX 4070 SUPER / 12,282 MiB with the provenance above. Each
seed resets reference/motion state and random generators. Four steps, shift 5,
BF16 eager, 512 square, 33-frame chunks, nine-frame overlap, 24 new frames,
color correction 1, same ten-second audio. Loading, preparation, warmup,
encoding and network delivery are outside timing. CPU transfers, audio features,
motion re-encoding, tracking and SR are inside the measured loop.

| Path | Seed | Time for 250 useful frames | Useful FPS | First completed chunk |
| --- | ---: | ---: | ---: | ---: |
| Lite alone | 50 | 6.204 s | 40.30 | 0.697 s |
| Lite alone | 51 | 6.221 s | 40.19 | 0.706 s |
| Lite + FaceLandmarker + native2x TRT | 50 | 10.398 s | 24.04 | 1.054 s |
| Lite + FaceLandmarker + native2x TRT | 51 | 10.343 s | 24.17 | 1.045 s |

Per useful frame, the enhanced branch spends approximately **24.57–24.59 ms
generating Lite**, **6.95–7.07 ms tracking**, and **9.71–10.07 ms upscaling with
transfers**. Total is 41.37–41.59 ms/frame, above the 40-ms budget for 25 FPS.
This serial implementation has no demonstrated continuous 25-FPS headroom,
even before adding a real refiner, encoding or delivery. First-chunk times
exclude incoming audio accumulation and are not live first-frame latency.

Lite raw RGB hashes match exactly between baseline and enhanced branches for
each seed. The preceding control using MediaPipe 0.10.9 / TensorRT 10.9.0.34
also produced identical raw and enhanced RGB hashes to the matched-version run
for both seeds. That control ran at 24.27–24.56 FPS; it remains in the parent
[results.json](results.json), and is not presented as the matched environment.

### Isolating TensorRT's contribution

[Same-weights runtime test](matched-runtime/same-weights-runtime.json): the same
public native-2x model, same 25 sampled 512-square inputs, warmup, three repeats,
alternating backend order; standalone SR with transfers and quantization, no
Lite or tracker. Both backends remain resident. Same RTX 4070 SUPER profile.

| Backend | Median time/frame | Median FPS |
| --- | ---: | ---: |
| PyTorch FP16 | 16.72 ms | 59.82 |
| TensorRT FP16 enabled | 9.08 ms | 110.19 |

TensorRT is **1.84x faster** in this controlled SR test. Final 8-bit outputs
differ by at most one intensity level; mean absolute difference is 0.039/255.
Separate comparisons against the FP32 model on four portraits/frames yielded
71.2–77.1 dB PSNR. These establish engine fidelity, not teeth quality. The speed
change from our old 4x SR includes a different checkpoint/depth/scale as well as
TensorRT; it must not be attributed entirely to the runtime.

### Resources

Peaks across the two main seeds; whole-device figures include the 2,727-MiB
resident baseline. Lite and the warmed SR engine remain loaded in both arms.
Torch counters exclude TensorRT-owned allocations. CPU process 100% is about
one logical CPU; host CPU is normalized over 32 logical CPUs.

| Metric | Lite baseline arm | FaceLandmarker + native2x TRT arm |
| --- | ---: | ---: |
| Sampled whole-device VRAM peak | 8,784 MiB | 8,796 MiB |
| Torch allocated peak | 5,070 MiB | 5,070 MiB |
| Torch reserved peak | 5,724 MiB | 5,724 MiB |
| Process RSS peak | 5,738 MiB | 5,771 MiB |
| Host RAM used peak | 12,407 MiB | 12,438 MiB |
| GPU use, per-run mean / peak | 95.2–98.7% / 100% | 70.3–70.6% / 100% |
| Process CPU, per-run mean / peak | 99.9–100.4% / 105.8% | 101.4–101.5% / 113.7% |
| Host CPU, per-run mean / peak | 4.9–5.7% / 10.9% | 5.0–5.1% / 9.8% |

Resource sampling alternates 137/193-ms waits plus query time to avoid the
previous half-second sampling alignment. It can still miss short peaks. The
engine reports about 64 MiB of execution-context device memory. The 12-MiB
difference between arms is **not** the total SR requirement, since SR is already
loaded/warmed in the baseline arm. Host RAM includes retained frames for
recording; encoding itself is outside monitoring. Aggregates are in
[summary.json](matched-runtime/summary.json), with raw samples in the run record.

## Tracking and visual findings

FaceLandmarker tracked **500/500** frames. A separate FaceMesh control on the
same raw frames also tracked 500/500. Their 20 mouth landmarks differed by a
mean of **2.25–2.27 pixels** on the 512 canvas, with per-frame mean disagreement
p95 around 3.7 pixels. Neither is ground truth, so this is not an accuracy win.
FaceMesh's separate replay took 0.53–0.67 seconds per clip; its different process
and replay workload prevent treating that as a controlled isolated tracker
speed ratio against the integrated FaceLandmarker timing.

Visual review used seven native pre-encode samples per seed/treatment, shown
in the mouth and full-frame sheets. The public native-2x model retains much of
the original softness and connected tooth appearance. The older general-x4v3
model produces much stronger edges, but also obvious artificial comb-like
divisions/patches in some teeth and smoother skin. Stronger edges are not proof
of correct dental anatomy. Neither treatment establishes a consistent teeth
fix or better lip-sync. No ground-truth dental/phoneme evaluation or temporal
perceptual score was performed.

**Conclusion:** the previously untested accessible APIs/runtime/scale now have
real local evidence. Their inclusion alone did not solve the dental problem
with this public checkpoint. We still cannot assign Ojin's reported quality
to its mouth refiner alone: its exact SR weights, conditioning, blending and
processing code remain unknown too. Reproducing the actual implementation
requires access to its delivered container or those missing artifacts/settings.

## Validation and reproduction

- Ten main/control/comparison MP4s fully decoded: 250 frames each, 25 FPS,
  audio present. [Media validation](matched-runtime/media-validation.json).
- Actual TensorRT numerical validation, all-frame tracker results, pre-encode
  input/output hashes and warmup/memory provenance retained.
- CPU tracker checks cover real video timestamps, finite mouth coordinates,
  missing-face behavior and closed-tracker rejection. Existing mouth-SR CPU
  regression tests: **4 passed**. Python files compile.
- No production defaults or running service changed. Added opt-in adapters and
  isolated benchmark scripts. Mesa dispatcher packages `libgles2` and `libegl1`
  were installed for MediaPipe 0.10.35's shared library.

Install the pinned packages into an isolated directory; the existing SoulX
environment supplies Torch and common dependencies:

```bash
.venv/bin/python -m pip install --no-deps \
  --target /workspace/experiments/ojin-matched-deps \
  -r benchmarks/ojin_components_20260917/requirements.txt
```

Model URLs and checksums are in `source-lock.json`. To build/run fresh artifacts:

```bash
PYTHONPATH=/workspace/experiments/ojin-matched-deps:. .venv/bin/python \
  benchmarks/ojin_components_20260917/build_engine.py \
  --output benchmarks/ojin_components_NEW_RUN \
  --engine models/ojin-components/NEW_RUN.engine

PYTHONPATH=/workspace/experiments/ojin-matched-deps:. .venv/bin/python \
  benchmarks/ojin_components_20260917/run.py \
  --output benchmarks/ojin_components_NEW_RUN \
  --engine models/ojin-components/NEW_RUN.engine
```

[controls.py](controls.py), [compare_runtime.py](compare_runtime.py) and
[package.py](package.py) retain the matched replay, backend and media workflows.
They target this evidence directory. Performance scripts refuse to overwrite
prior measurement records; CPU packaging can be regenerated.
