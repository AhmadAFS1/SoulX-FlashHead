# PRO320 generation performance

**Hardware for these November 2, 2026 runs was an Apple M5 Pro with a 20-core integrated GPU and 24 GiB of unified memory.** Metal, system_profiler and sysctl identified the hardware. macOS 27.0.1 build 26A434 supplied the Core ML and Metal runtimes. The native runner used Swift 6.4 and Xcode 27. The GPU shares system memory. A Metal working-set budget is recorded in the JSON as additional evidence. Core ML chose operation placement within each requested compute policy. GPU benchmark processes ran sequentially. Background system load was unisolated.

SoulX Studio now starts loading the next transformer partition while the current partition runs. It also prepares one subsequent audio window during denoising. The app enables both through **Models & settings → Overlap generation work**. Turning the switch off selects sequential execution. Both modes use the same PRO320 bundle, sampling schedule, portrait caches and seed.

## Matched results

Each trial rendered 112 frames of the same active speech at 176 × 320. Warm FPS is the sum of the 84 frames after the first window divided by their combined generation time. It includes speech processing, denoising, decoding, color correction and motion feedback. The table pools paired trials on the M5 Pro. The JSON records first-window times and total process times including initialization and PNG writes.

| Core ML compute request | Avatar | Paired trials | Reference warm FPS | Optimized warm FPS | FPS gain |
| --- | --- | ---: | ---: | ---: | ---: |
| Automatic | Indian | 2 | 9.07 | 15.24 | 68.1% |
| Automatic | Asian | 2 | 8.98 | 15.27 | 70.0% |
| CPU + GPU | Indian | 2 | 10.94 | 15.01 | 37.2% |
| CPU + GPU | Asian | 3 | 10.76 | 15.31 | 42.2% |

**Every displayed RGB pixel matched the reference** in all four-window trials. A separate 9.48-second Indian-avatar run crossed the eight-second audio-history boundary and ended with a partial 13-frame window. All 237 frames matched. The reference was commit `fec3bf7` with one placement correction described below. It used the same ten compiled partitions, all 30 PRO blocks, two denoising steps and speech input as the optimized run. These comparisons establish character, mouth and motion preservation on the tested clips. [Timings, run conditions and frame hashes](validation/pro320_runtime_optimization.json)

The earlier iPhone 12 Pro results belong to the previous runtime. iPhone FPS and memory behavior for this revision still require a physical-device run. [Earlier A14 validation](VALIDATION.md)

## Speech encoder correction

On this Mac the GPU-eligible Core ML speech encoder returned non-finite features for five distinct inputs, including silence and held-out speech. The projection converted them into the same finite context. Earlier end-to-end checks validated finite video frames and missed the invalid encoder output. CPU speech encoding matched the original FP32 PyTorch features within 0.022% normalized RMSE. The projection also passed its accuracy checks when supplied with valid features. [Backend measurements](validation/pro320_audio_backend_quality.json)

The runtime now validates the selected features before projection. On this specific numerical failure it retries the speech encoder once on CPU and remembers a successful fallback for that session. Conditioning, the PRO transformer and the video models retain the selected compute policy. The audio worker inherits the resolved encoder policy and reports it with the window. A failed CPU retry propagates to the caller.

For the reference timings we set just the historical speech encoder to CPU. This gives both sides of the comparison valid speech features. The optimized runs used automatic retry and include the first failed attempt in the first-window timing. The observed M5 backend failure has not been established on A14.

## Runtime behavior

The scheduler limits work in flight to one successor partition and one upcoming audio context. It predicts every partition in order. The audio worker receives a copy of the exact eight-second history, including final padding. Motion history and noise stay on the inference actor. The workers finish or cancel within the generation task.

On iOS the runtime checks live process headroom, current footprint, profile dimensions and partition size before starting concurrent work. Low headroom can disable overlap. Source packages use their normal compilation path. Partition preloading uses complete compiled models. Both bundled avatars support PRO320. The app can use the existing installed model files.

Audio packing copies the overlapping FP16 feature slices directly. On the M5 Pro CPU the median packing time for a synthetic full-size window fell from about 62.9 ms to 0.52 ms with validation enabled. Color correction reuses each Lab value and maps the decoder buffer once per frame. The CPU helper check for 28 frames fell from 195 ms to 76 ms, with the same float32 pixel hash. [Audio probe](validation/pro320_audio_optimization.json)

We also tested fused attention projections, larger partitions and convolution projections. Six-block partitions gained about 1.7% in an isolated Mac trial and doubled the largest partition payload. The normal app uses the existing three-block exports. [Transformer experiments](validation/dit_partition_trials_mac.md)

## Reproduce the comparison

From the repository root, point `speech` at a speech WAV. For the recorded four-window fixture we used the 4.48 seconds starting at 3.00 seconds in the local MuseTalk `data/audio/eng.wav`, resampled to mono 16 kHz. Use the same arguments and output directories dedicated to each run. The baseline switch builds the historical Swift files with the speech correction.

```bash
models=artifacts/coreml/device/pro320
avatar=ios/SoulXStudio/Resources/Avatars/girl
speech=artifacts/coreml/fps-work/speech.wav
common=(--models "$models" --portrait "$avatar/portrait.png"
        --reference-cache "$avatar/reference-cache/metadata.json"
        --audio "$speech" --chunks 4 --compute all --save-frames)

SOULX_BENCH_BASELINE=1 apple/run_mac_smoke.sh "${common[@]}" --output /tmp/soulx-reference
apple/run_mac_smoke.sh "${common[@]}" --prefetch-blocks --prefetch-audio \
  --model-timings --output /tmp/soulx-optimized

.venv-coreml/bin/python apple/flashhead_coreml/compare_frames.py \
  --baseline /tmp/soulx-reference --candidate /tmp/soulx-optimized
```

The smoke runner also accepts `--compute cpu-gpu` for the second policy in the table. The iOS `--smoke` diagnostic accepts `--prefetch-audio`, `--prefetch-blocks` and `--model-timings`. Its JSON includes the effective speech encoder policy, fallback status and per-stage timings.
