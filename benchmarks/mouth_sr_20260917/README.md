# Mouth super-resolution: external lead, implementation and local results

2026-09-17, **NVIDIA GeForce RTX 4070 SUPER**, physical 12 GB class,
**12,282 MiB visible** (`nvidia-smi`), driver **595.84**, PyTorch
**2.7.1+cu128 / CUDA runtime 12.8**. Fresh SoulX capture and SR inference ran
locally on this GPU. OmniVoice (2,478 MiB) and idle LTX research server (234 MiB)
remained resident; initial device use was 2,727 MiB. Each GPU process held the
checkout's lease. Tracking uses MediaPipe 0.10.9 FaceMesh/XNNPACK on CPU, with
an NVIDIA EGL context initialized by MediaPipe. Packaging, validation and unit
tests are CPU work. External deployment hardware/performance is unverified.

## Finding and decision

Follow-up: [Ojin component audit and matched FaceLandmarker/TensorRT tests](../ojin_components_20260917/README.md)
now uses actual Tasks FaceLandmarker 0.10.35, TensorRT 10.16.1.11 and a public
native-2x SRVGG checkpoint at full-frame 512→1024 on the same RTX 4070 SUPER /
12,282 MiB, driver 595.84, Torch 2.7.1+cu128 / CUDA 12.8, with OmniVoice and LTX
resident. The integrated test measured 24.04–24.17 FPS and did not establish a
teeth fix. This audit also corrects the accessibility claim: Ojin's exact SR
checkpoint/engine recipe and service code were not public, in addition to its
proprietary refiner. The experiment below remains the earlier FaceMesh/general-x4v3
prototype and must not be described as “Ojin's implementation minus its refiner.”

The external lead is real and implementable in part. The new open-source
prototype runs **SoulX output → tracked mouth crop → learned SR → feathered
composition → delivery**, without feeding the enhanced frames into SoulX's
recurrent motion state. It is usable as an opt-in library and has a tested
chunk-by-chunk integration example. No running service or default was changed.

**Observed benefit is cosmetic sharpness, not a demonstrated anatomical teeth
fix.** In the inspected samples, learned full-frame SR makes contours and some
tooth boundaries more conspicuous than bicubic enlargement. It also smooths skin
and beard into a waxier appearance and can emphasize irregular fused/tooth-band
texture. The mouth-only 0.65 blend is subtler and leaves the rest of the portrait
alone. Some dental structure remains merged or synthetic. It does not justify
claiming recovered true teeth, eliminated flicker or improved phoneme accuracy.

The useful next architectural direction is a **separate learned output-detail
stage**, distinct from more steps through the same SoulX generator. A dedicated
temporally trained dental/mouth refiner is still missing. The downloaded SR
model is a generic image model, not that missing refiner.

## Verification of the supplied research

| Lead | Independently verified | What it does not establish |
| --- | --- | --- |
| [SoulX issue 18](https://github.com/Soul-AILab/SoulX-FlashHead/issues/18) | A Lite user reports poor tooth clarity without SR; their end-to-end 4090 example includes detection/crop/generation/SR/pasteback. | No named SR checkpoint, controlled teeth A/B, guaranteed fix or local performance result. |
| [Ojin third-party manifest](https://github.com/ojinai/kit-example/blob/819fe6e95613fdcc7bffd8676e8814282cd445f5/THIRD-PARTY-NOTICES.md) | The documented Lite-only package lists mouth-ROI landmarks, an SRVGG-based 2x upscaler and proprietary mouth-refiner weights. | A dependency manifest is not an efficacy benchmark and does not expose the refiner, exact routing, training or blending implementation. |
| [kegeai888 preset](https://github.com/kegeai888/SoulX-FlashHead/blob/bde99904fd77583edbdd6bffbcb59f24debfb069/webui.py) | `quality` selects eight steps and `stable`; balanced selects four and aggressive. | No teeth-specific controlled result or new checkpoint. Our previous six-step/refinement results do not constitute an eight-step test. |

These sources support trying enhancement, but the other chat's descriptions
“strongest fix” and “strongest evidence for fixing rubbery teeth” are too strong
for the published evidence alone. We have not claimed an exhaustive search for
every checkpoint, fork or derivative. No proprietary weights were accessed.

Source pins, saved relevant files and hashes are in `source-audit/`. The official
[Real-ESRGAN inference code](https://github.com/xinntao/Real-ESRGAN/blob/a4abfb2979a7bbff3f69f58f58ae324608821e27/inference_realesrgan.py)
specifies the open general-x4v3 architecture/weights used here.

## Implementation details

- Module: [soulx_rtc/mouth_sr.py](../../soulx_rtc/mouth_sr.py).
- Architecture: official SRVGGNetCompact, 64 features, 32 body convolutions,
  PReLU, **native 4x** general-x4v3 weights, denoise strength 1.0, CUDA FP16.
  The official architecture is vendored with BSD-3-Clause notice; only BasicSR
  registry import/decorator were removed. No Torch/CUDA environment upgrade or
  BasicSR dependency installation was needed.
- Delivery at 2x uses area resizing of the model's 4x output. Native output uses
  area resizing back to original dimensions. **This is not Ojin's exact 2x
  model**, nor an independently trained 2x network.
- One streaming FaceMesh tracker per session; one shared SR model can serve
  sequential sessions on the GPU owner thread. Twenty outer-lip landmarks locate
  the padded crop. Causal EMA stabilizes its center/extent; large jumps reset it.
- The current lip hull is padded and feathered, with mask weight 0.65 and zero
  weight at the crop boundary. Lip positions are not warped and past RGB frames
  are not averaged. The neural model can still change apparent lip/tooth edges;
  geometry preservation is not a lip-sync certification.
- Missing/invalid/clipped detections return unenhanced output (bicubic at 2x)
  and reset ROI history. No stale enhanced mouth is pasted into a new frame.
- Native mode preserves pixels outside its ROI exactly. At 2x, the surrounding
  frame uses the same bicubic interpolation as the control.
- Processing follows `Engine.generate`, after its motion-history encoding and
  CPU RGB transfer. Enhanced output never enters the generator's history here.
  It precedes delivery/video encoding. This version uses CPU tracking and
  composition, eager PyTorch SR, no TensorRT or overlap optimization.

The checkpoint is downloaded from the official release URL recorded in
[weights-lock.json](weights-lock.json), SHA-256
`8dc7edb9ac80ccdc30c3a5dca6616509367f05fbc184ad95b731f05bece96292`.
It is kept in the ignored `models/mouth-sr/` directory. Strict state loading
and `torch.load(weights_only=True)` are used.

## Controlled local experiment

Fresh ten-second raw RGB captures, one identity/audio, 25 FPS:

- **320×576, seeds 50 and 51:** exact neutral 1.25x shoulder-visible portrait,
  INT8 weight storage/BF16 compute, compact, 3,584 MiB Torch cap.
- **512×512, seed 50:** retained close square crop, BF16/staged, 8,704 MiB cap.
- All: four steps, shift 5, history 2, stock audio strength 1, eager execution,
  optimized conditioning, real RoPE, lean output, fused QKV, no latent refinement.

Each raw capture is processed into five arms: original, bicubic 2x, full-frame
SR 2x, mouth SR/native and mouth SR/2x. **Within each fixture all arms receive
identical pre-encoding pixels**. Source generation is not rerun per arm. The
square and portrait profiles have different framing/precision and cannot be
treated as a controlled resolution-only comparison.

The fresh 320 seed-50 source hash matches the earlier baseline exactly. Seven
lossless pre-encode frames per arm are saved. Mouth crop sheets align every
arm to landmarks detected on the original frame at the same timestamp.
Original/native crops are displayed with nearest-neighbor enlargement; the
bicubic control shows whether ordinary interpolation alone explains a gain.

### Postprocessor runtime on RTX 4070 SUPER

| Input profile | Bicubic 2x | Full-frame learned SR 2x | Mouth SR/native | Mouth SR/2x |
| --- | ---: | ---: | ---: | ---: |
| 320×576 seed 50 | 0.58 ms/frame | 59.56 ms/frame | 9.77 ms/frame | 9.27 ms/frame |
| 320×576 seed 51 | 0.56 ms/frame | 59.01 ms/frame | 9.49 ms/frame | 9.09 ms/frame |
| 512×512 seed 50 | 0.89 ms/frame | 90.24 ms/frame | 14.57 ms/frame | 15.84 ms/frame |

These include synchronized SR, transfers, tracking and composition, excluding
model loading, MP4 encoding and disk writes. They are individual offline runs
under the stated resident load, not repeated service-capacity tests. The models
are warmed, while first-frame landmark initialization contributes to the mean.
All mouth arms detected/applied on 250/250 frames. Standalone peak Torch
allocation was roughly 16–17 MiB for portrait mouth SR, 33 MiB for square mouth
SR, and 94/132 MiB for full-frame SR at portrait/square respectively. Exact
memory/timing records are in each `enhancement.json`; allocator counters exclude
CUDA/EGL context and other processes. Reserved memory retains preceding-arm
caches and is not an isolated minimum-footprint measurement.

### Combined chunk-by-chunk run on RTX 4070 SUPER

`stream.py` runs fresh 320×576/seed-50 SoulX inference and immediately applies
native mouth SR to each generated chunk before the next call. Both networks
remain loaded within the **3,584 MiB** Torch cap.

- 250 useful frames in **10.311 s = 24.246 FPS**.
- SoulX-only matched capture: **7.957 s** for the same useful frames.
- Peak allocated **3,306.01 MiB**, reserved **3,554 MiB**.
- Underlying SoulX raw hash is unchanged, verifying no feedback perturbation.
- Enhanced raw hash exactly matches the offline native-mouth arm, verifying
  tracking/composition continuity across 24-frame chunk boundaries.

This uncompiled combined profile is slightly below 25 useful FPS before encoding
or WebRTC. It is **not** a demonstrated real-time service or concurrency pass.
The timed loop includes whole generation chunks and trimmed final excess frames;
loading, warmup and recording are excluded. Compiled/TensorRT acceleration may
help but was not measured in this experiment.

## Validation and limits

- CPU tests: **12 passed**, covering ROI/background/input preservation,
  no-face/disabled fallback, border/invalid landmarks, moving-ROI state across
  chunk partitions, and existing refinement/optimization regression tests.
- Actual GPU strict checkpoint loading, three source captures, fifteen output
  arms and one integrated run completed. Full videos and sampled native PNGs
  are retained. The default engine itself was not changed by this experiment.
- ffprobe/frame counts and full ffmpeg decoding validate the media; this does
  not validate tooth anatomy, flicker, identity or phoneme synchronization.
- This is one avatar, two portrait seeds and one square seed, one utterance.
  No dental ground truth, blinded preference or trained temporal consistency
  assessment was available. Sharper edges can make incorrect geometry more
  visible. ROI smoothing only stabilizes the crop; it does not make an
  image-by-image SR network temporally aware.

## Use and reproduce

Existing environment requirements: Torch, NumPy, OpenCV and MediaPipe 0.10.9
(the solutions FaceMesh API). We reuse the already installed packages.

```python
from soulx_rtc.mouth_sr import SRVGGUpscaler, MouthEnhancer

# Construct the shared model once on the GPU owner thread.
sr = SRVGGUpscaler("models/mouth-sr/realesr-general-x4v3.pth")
mouth = MouthEnhancer(sr, scale=1, strength=0.65)  # one tracker per call
try:
    raw_chunk = engine.generate([state])[0]
    delivery_chunk = mouth.process_chunk(raw_chunk)
    # Send delivery_chunk to the encoder. Keep mouth alive across this call's chunks.
finally:
    mouth.close()  # close when the call ends, not between chunks
```

For the reproducible complete loop use [stream.py](stream.py). It is an opt-in
integration example, not a new live-service flag. Offline reproduction uses
`capture.py --size 320 --seed 50`, `enhance.py --capture 320-seed50`, then
`package.py --capture 320-seed50`, with `PYTHONPATH=.` and `.venv/bin/python`
from the repository root. Use a fresh artifact directory; retained captures and
enhancements refuse overwrite. Other fixtures use 320/51 and 512/50.

Review:

- [All comparisons](review.html)
- [320 seed 50 comparison](320-seed50/soulx-LITE-original-vs-bicubic-vs-full-frame-SR-vs-mouth-SR-320-seed50.mp4)
- [320 seed 51 comparison](320-seed51/soulx-LITE-original-vs-bicubic-vs-full-frame-SR-vs-mouth-SR-320-seed51.mp4)
- [512 square comparison](512-seed50/soulx-LITE-original-vs-bicubic-vs-full-frame-SR-vs-mouth-SR-512-seed50.mp4)
- [Portrait mouth crops](320-seed50/mouth-comparison.png)
- [Square mouth crops](512-seed50/mouth-comparison.png)
- [Integrated native-resolution result](stream-native/video.mp4)
- [Integrated timing/parity evidence](stream-native/results.json)

The earlier source-upscaling failures used interpolation **before generation**.
They do not test this learned **after-generation** enhancement. This finding
adds a possible mitigation to the [bottleneck assessment](../teeth_bottleneck_audit_20260917/README.md)
without proving a different originating cause or eliminating the need for a
true mouth-detail model.
