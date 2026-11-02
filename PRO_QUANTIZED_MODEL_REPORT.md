# SoulXFlashHead PRO quantization, Core ML implementation and performance report

**Engineering snapshot through `3cb2397`, November 2, 2026.** The Apple measurements cover the **Apple M5 Pro integrated GPU** and the **Apple A14 GPU in an iPhone 12 Pro**. The CUDA results are historical measurements on an **NVIDIA GeForce RTX 4070 SUPER**. This report reanalyzes saved runs. It adds no new inference measurements.

**The dominant recurring cost is executing the PRO denoiser twice for every 28 new frames.** A historically important exception was the Wan video decoder, which consumed about half the time in an earlier CUDA configuration. The tiny video autoencoder substantially reduced that cost. On Apple hardware, portrait initialization had a separate memory problem. Caching the two fixed portraits solved initialization on the phone. Overlapping model loads and speech preparation improved generation on the Mac. The remaining Mac transformer prediction calls alone average **1.48 seconds per warm window**.

| Measured result | Hardware and workload | Interpretation |
|---|---|---|
| **52.90 useful FPS** | Historical RTX 4070 SUPER, one PRO stream, 320 × 576, tiny streaming decoder and encoder | Highest measured complete PRO result found in the audited refs. The tiny decoder had visible quality tradeoffs. |
| **0.81–0.90 warm FPS** | Physical iPhone 12 Pro, Apple A14, 176 × 320, two fixed avatars | The first Apple implementation generated consecutive windows locally. The optimized revision still awaits a physical-phone speed measurement. |
| **15.24–15.27 warm FPS** | Apple M5 Pro, 176 × 320, optimized runtime, automatic Core ML selection | Up from **8.98–9.07 FPS** on the matched, speech-corrected reference. Displayed frames matched pixel for pixel on the tested clips. |

Resolution throughout is **width × height**. The profile names use height, so `pro320` produces 176 × 320 portrait video and `pro576` produces 320 × 576. Useful FPS counts newly generated frames. A 25 FPS video file specifies the playback clock. Batched aggregate throughput, cached-video replay and single-session generation have different denominators.

[Branch history](#2-repository-history-and-the-fast-pro-branch) · [Model and app](#3-what-was-ported) · [Conversion decisions](#4-conversion-failures-and-the-decisions-they-forced) · [Bottleneck breakdown](#5-where-the-time-goes) · [Optimization and quality results](#6-optimizations-implemented-tested-and-rejected) · [Next experiments](#7-plan-from-here-ranked-by-expected-value) · [Reproduction](#8-reproduce-and-inspect)

## 1. Evidence and execution environments

| Evidence | Hardware, memory and runtime | Scope |
|---|---|---|
| September 2026 CUDA manifests | **NVIDIA GeForce RTX 4070 SUPER**, 12,282 MiB visible in `nvidia-smi`. Earlier records identify the 12 GB physical class. The September 22/26 manifests mark physical class unverified. Driver 595.84, PyTorch 2.7.1+cu128, CUDA 12.8. The September 26 runs used FlashAttention 2.8.0.post2, TensorRT 10.3 for tiny VAE engines and 10.9.0.34 for Wan spans. | Reused GPU inference. The September 26 environment snapshot lists only the measured process. Some September 19 runs had co-residents using about 2,478 and 234 MiB. Read the individual manifest for its load. |
| November 1 local / November 2 UTC physical-device records | **Apple A14 GPU**, iPhone 12 Pro `iPhone13,3`, iOS 26.2 build 23C55. Metal and `uname` identify the device. ProcessInfo reports 5,994,840,064 bytes of shared system memory. Dedicated or available GPU VRAM and a fixed app memory limit are unreported. | Physical Core ML inference at `fec3bf7`. Other-app load is unknown. The compute-unit field records the requested policy. Actual per-operation CPU, GPU and Neural Engine placement remains unverified. |
| November 2 Mac runtime comparisons | **Apple M5 Pro**, 20-core integrated GPU, 24 GiB unified memory. Metal, `system_profiler` and `sysctl` supplied the evidence. The reported Metal working-set recommendation is 19,069,665,280 bytes, a budget within shared memory. macOS 27.0.1 build 26A434, Xcode 27.0, Swift 6.4. | Native Swift runtime, real Core ML weights and speech. GPU benchmark processes ran sequentially. Background system and CPU development load were unisolated. Automatic and CPU + GPU are requested policies. |
| Conversion, component tests and arithmetic | Same **M5 Pro / 24 GiB** host. Python 3.12.12, PyTorch 2.7.0, Core ML Tools 9.0 and NumPy 2.2.6. | CPU-only source/component comparisons explicitly request CPU execution. Operation counts below are static estimates. Simulator builds establish app compatibility. |

Primary evidence includes the [fast CUDA run](benchmarks/pro_30fps_20260922/tae-decenc-trt-r01/results.json), [physical A14 run](apple/validation/iphone12pro_asian_active_speech.json), [matched Mac run records](apple/validation/pro320_runtime_optimization.json) and [backend-specific speech checks](apple/validation/pro320_audio_backend_quality.json).

## 2. Repository history and the fast PRO branch

During the original audit, the remote refs and tags were fetched and `main` was fast-forwarded. The checkout of `main` at `3f95dae` contained the earlier Lite-oriented serving code. The preserved source was 17 commits ahead of it.

| Ref reviewed | Commit | Significance |
|---|---|---|
| `main` | `3f95dae` | Earlier serving implementation |
| `codex-ci-test-2026` | `de4c12c` | Older Lite/quality snapshot |
| `mcpro-before` / `mcpro-latest` | `e4ea816` / `51225f5` | Intermediate September 19/20 tags in the PRO history |
| `preserve/detached-work-20260919-c39f2a5-github-source` | `4c5c7aa` | Contains the later PRO quantization, pruned decoder, two-step sampling, tiny-VAE experiments and WebRTC work |
| `codex/ios-coreml-inference` | `fec3bf7`, then `3cb2397` | Separate worktree based on the preserved source. The first commit added the Apple implementation. The second added scheduling optimizations and speech validation. |

The original feasibility audit and static cost estimate are available in the neighboring main checkout. Its speed estimates preceded the actual Apple runs. The physical and Mac measurements in this report supersede those forecasts. [Initial feasibility audit](</Users/ahmadsmacair/Documents/ChatGPT/SoulXFlashHead/docs/research/COREML_PRO_IPHONE12_FEASIBILITY_2026-11-01.md>)

The 50+ FPS milestone is visible at `cf6c308`. The result files establish what it measured. The audit covered the advertised refs and retained benchmark files. Deleted or unpushed branches on another machine remain outside that scope. The approximately 67.5 FPS SageAttention-plus-tiny result was a projection assembled from different runs. The frequently cited 56.20 FPS stock result belongs to Lite. [Preserved source](https://github.com/AhmadAFS1/SoulX-FlashHead/tree/4c5c7aa72cb65045e9f26a3c9d04f08a819fb0a8) · [PRO/Lite comparison](benchmarks/pro_lite_150x_20260917/README.md)

### How the RTX 4070 SUPER reached 52.90 FPS

All rows are **historical RTX 4070 SUPER GPU inference** at 320 × 576. Earlier and later profiles differ in step count, attention implementation and decoder. Only the September 26 FlashAttention control and its tiny-VAE arms form the matched comparison below.

| Date and configuration | Useful FPS | What changed |
|---|---:|---|
| Sep 17 stock PRO | 7.16 | BF16 eager anchor |
| Sep 18 PRO FP8 V1 | 9.41 | Quantized feed-forward linears |
| Sep 19 PRO V2 / mixed FP8 + FP16 stages | 10.40 / 11.78 | Broader transformer quantization, specialized attention and fused decoder stages |
| Sep 22 earlier reference / final lean V4 | 19.62 / 30.22 | Two-step PRO, distilled block-pruned Wan decoder, CUDA graph, fused INT8 FFN and lean frame delivery. Final run used SageAttention2. |
| Sep 26 matched Wan control | 26.81 | FlashAttention2, pruned Wan decoder, Wan motion encoder |
| Sep 26 tiny motion encoder only | 30.02 | Retained Wan decoder |
| Sep 26 tiny decoder and encoder, reset each window | 51.95 | TensorRT TAEHV window mode, closest decoder semantics to the Apple port |
| Sep 26 tiny decoder and encoder, carried decoder state | **52.90** | TensorRT TAEHV streaming mode. Plosive and open-vowel fixtures repeated at 52.87 and 52.81 FPS. |

The peak run used the Indian 1.50× fixture, seed 50 and two denoising steps. It produced 250 useful frames in 4.726 seconds, including the final padded window. Its first window took 534 ms, and its median window took 524 ms. Model/avatar setup, warmup and video muxing were outside that FPS figure. The matched improvement **26.81 → 52.90** is 1.97×. Peak sampled device memory fell from 9,019 to 6,105 MiB. [Precision history](docs/research/PRO_PIPELINE_PRECISION_AND_THROUGHPUT_2026-09-19.md) · [30 FPS iteration log](docs/research/PRO_40FPS_ITERATION_LOG_2026-09-21.md) · [Matched control](benchmarks/pro_30fps_20260922/tae-ctl-r01/results.json) · [Tiny study](docs/research/TINY_VAE_FEASIBILITY_2026-09-26.md)

**Quality qualification matters here.** The tiny decoder produced teeth speckle, a lower-teeth fleck, grainier stubble and about 16% more skin shimmer in the retained review. Paired lip-motion correlation was 0.923 against a control-repeat floor of 0.984. That comparison measured motion relative to another render. Its coverage was short and seed-limited. The Apple port uses this experimental tiny-VAE family. Preserving the Apple reference pixels preserves its visual strengths and its existing artifacts. [Quality review and comparisons](docs/research/TINY_VAE_FEASIBILITY_2026-09-26.md)

### Why the MuseTalk result did not predict phone FPS

Your near-20-FPS reference is consistent with the retained MuseTalk iPhone 12 Pro study, which measured about 16.4–17.1 **shown** FPS in two playback runs. The 338.91 FPS RTX 4070 SUPER result was aggregate throughput at batch 16 over 180 seconds. MuseTalk runs a fixed-timestep UNet over a 256 × 256 face crop and composites it into source video. PRO generates full video frames, attends across space and time twice per window, and feeds generated motion into the next window. Those differences defeat a desktop-to-phone FPS ratio based on MuseTalk. These are historical observations from the neighboring MuseTalk checkout. [iPhone study](</Users/ahmadsmacair/code/musetalk-ios-bench/docs/ios_unet_bench_2026-10-30.md>) · [Batch-16 GPU result](</Users/ahmadsmacair/code/musetalk-ios-bench/docs/fps_comparisons/4070s_300fps_impl_20260928/combined/bench_gpu_path_stagewise16_taesdtrt_180s.json>)

## 3. What was ported

We reconstructed portable PyTorch modules from the original PRO weights and exported fixed-shape, iOS 18+ Core ML ML Programs. The Apple additions live in `apple/` and `ios/SoulXStudio/`. The branch isolates them from the CUDA serving path and dependencies.

The authoritative DiT checkpoint has 1,507,694,912 stored tensor elements and a roughly 6.03 GB FP32 file. The count includes stored buffers and inactive modules. Its SHA-256 is `e47e61b9023ea1aac60c0c0fff077289bc6cca443c71907e5a76a411480af250`. The fetcher pins the SoulX and Wav2Vec repositories and verifies the PRO, Wan and TAEHV checkpoints. [Weight inventory](docs/architecture/system_design_20260917/weight-inventory.json) · [Source pins and checksums](apple/flashhead_coreml/fetch_models.py)

The fast CUDA recipe covers 240 linear modules. It uses INT8 W8A8 for 60 FFN linears, FP8 W8A8 for 120 self-attention projections and FP8 for 60 audio cross-attention Q/O projections. The Apple exporter applies per-channel INT8 **weight compression** to learned matrices/convolutions. It preserves normalization, timestep and rotary constants separately. CUDA activation rounding, TensorRT engines and fused GPU kernels belong to the CUDA implementation. On Apple, compression describes storage. The runtime determines execution precision and processor placement. [CUDA policy](benchmarks/pro_30fps_20260920/policies/pro576_taew2_1_flash2.json) · [Apple exporter](apple/flashhead_coreml/export_dit.py)

### One session and one window

1. Choose the bundled **Indian** or **Asian** portrait. Indian uses the verified original shared portrait named in the SoulX benchmark. Asian uses the repository’s `examples/girl.png`. Raw image hashes tie each cache to the right avatar. [Asset provenance](ios/SoulXStudio/Resources/ASSET_SOURCES.md)
2. On Mac, the original Wan encoder prepares nine latent time positions from 33 copies of each still. The app ships those conditioning tensors. Initialization checks the image, tensor and checkpoint hashes, profile, shape and finite values, then computes portrait color statistics. The built-in caches contain portrait features. Generated frames still depend on the selected audio and seed. [Cache specification](apple/tools/reference_cache/README.md)
3. Decode local audio to 16 kHz mono. Maintain an eight-second history. Wav2Vec2 normalizes the waveform, resamples its convolutional features to 200 video positions before its transformer, and exposes all 12 hidden layers. The runtime selects 33 centers with five neighboring slots, then projects them into nine groups of 32 audio tokens.
4. Concatenate reference and noisy video latents, patchify, then execute **all 30 PRO blocks** for each of two denoising evaluations. The schedule is `[1000, 625, 0]`. Restore one motion latent at startup and two in subsequent windows. The Swift runtime owns stochastic updates and state reset. The exporter prepares real rotary/timestep constants and uses a temporally equivalent 2D patch projection.
5. Decode nine latents with Wan-compatible `taew2_1`. Window mode starts with zero decoder memory, expands to 36 frames and trims the first three. Withhold five history frames, color-correct the 28 useful frames in Lab space, and re-encode the last five corrected frames into two motion latents. The encoder pads the tail with three copies of the last frame. The CUDA streaming variant carries decoder state across windows.
6. Stream each finished window into a bounded, audio-clock-paced Swift player. Playback waits during generation underruns. Cancellation and source changes reset session state. The app exposes generation FPS separately from playback cadence. [Engine](ios/SoulXStudio/Sources/Runtime/SoulXEngine.swift) · [Playback](ios/SoulXStudio/Sources/App/ChunkPlayer.swift) · [App guide](ios/SoulXStudio/README.md)

| Contract | `pro320` | `pro576` |
|---|---|---|
| Output size | 176 × 320 | 320 × 576 |
| Reference / current latent, NCTHW | [1,16,9,40,22] | [1,16,9,72,40] |
| Tokens per denoising step | 1,980 | 6,480 |
| Transformer width / FFN width / heads | 1,536 / 8,960 / 12 | Same |
| Audio window / projected context | [1,33,5,12,768] / [1,9,32,1536] | Same |
| Useful frames per full window | 28 | 28 |

The default bundle contains **17 models**. These are the reference encoder, audio encoder, conditioning projection, patch projection, ten three-block DiT partitions, output head, tiny decoder and motion encoder. The local compiled `pro320` folder contains 2,111,605,735 file bytes, about 1.97 GiB. One DiT partition is about 134 MiB on disk. Audio is about 360 MiB and the reference encoder about 205 MiB. Resident memory also depends on activations, compiler workspaces and simultaneous model lifetimes. Model weights and generated media are regenerated locally. [Bundle verification](apple/flashhead_coreml/bundle.py)

The converters can resume a partially finished export and operate on individual components. The fixed timesteps avoid replicating a full time-embedding network in every partition. The app accepts compiled models or source packages, validates the profile and required files, and installs a replacement model directory transactionally. The phone build ID is `com.lingua.soulxstudio`. The SwiftUI host provides the two avatar cards, local audio selection, initialization, cancellation, buffering status and measured generation speed. The simulator was used for UI/build checks. Physical-device smoke mode saved actual generated frames and timing reports. The first large device transfer was interrupted, which led to a per-file deployer with resume and byte-length verification. [Deployment tool](apple/deploy_device.py) · [Smoke runner](ios/SoulXStudio/Sources/Runtime/SoulXSmoke.swift)

## 4. Conversion failures and the decisions they forced

**Numerical measurements in this section used the M5 Pro CPU**, original checkpoint weights and fixed production shapes. These are component comparisons against FP32 PyTorch. The GPU and ANE were unused in those checks.

| Experiment | Observed result | Implemented decision |
|---|---|---|
| Small synthetic DiT conversion | Passed, but the initial real three-block FP16 conversion reached about 10–12% normalized RMSE at the two timesteps | Require production weights and full token lengths before approving a precision policy. |
| Normalization alone in FP32 | Three-block error remained high. Full FP32 Core ML was close to PyTorch. | Retain both scaled dot-product attention and normalization in FP32. Use FP16 where validated, FP16 partition interfaces and INT8 compression of learned weights. All 13 DiT components and 24 test cases per profile passed. Worst component NRMSE was 4.68% at PRO320 and 4.05% at PRO576 under the 5% gate. |
| FP16 audio encoder | 3.17% NRMSE. Tested mixed configurations still gave about 2.29–3.26%. Tested FP32 computation with INT8 linear weights gave 16.87%. | Keep the production speech network in FP32. CPU source comparison gave about 0.021% NRMSE. |
| Lower-precision Wan reference encoder | 6.33% NRMSE on PRO576 and 5.47% on the mixed PRO320 attempt | Use FP32 computation. Both profiles reached approximately 0.021% NRMSE on CPU. |
| Tiny video encoder and decoder | PRO320 NRMSE was 0.397% for motion encoding and 0.330% for decoding | Retain FP16. The comparison is against the corresponding tiny model. Full Wan decoder quality is a separate criterion. |

The support gates required finite outputs, NRMSE ≤1%, cosine similarity ≥0.999 and maximum absolute error ≤0.05. We also checked audio window alignment, tail padding, rotary and temporal ordering, strided tensors, Lab conversion and recurrent feedback. Eleven Python tests passed, along with the native Swift helper checks and signed device/simulator builds. [PRO320 DiT](apple/validation/pro320_dit.json) · [PRO576 DiT](apple/validation/pro576_dit.json) · [PRO320 support](apple/validation/support_pro320.json) · [PRO576 support](apple/validation/support_coreml.json)

**Memory failure on the physical A14.** The 576-high reference-only run loaded the Core ML model. During prediction its last sampled process footprint was 3,209,285,632 bytes, and the process disappeared before returning a latent. The 320-high full-FP32 reference path also terminated with signal 9 during initialization. A crash log did not confirm a Jetsam cause. The test evidence supports avoiding this monolithic encoder on the A14. The cached-avatar path initialized in tens of milliseconds and completed generation. Arbitrary portrait encoding, the generic reference export and fixed-avatar generation are distinct capabilities. [576 interruption](apple/validation/iphone12pro_reference576_interrupted.json) · [320 interruption](apple/validation/iphone12pro_reference320_interrupted.json) · [Successful cached run](apple/validation/iphone12pro_pro320_cold.json)

## 5. Where the time goes

### RTX 4070 SUPER — the bottleneck shifted from Wan to the transformer

In a September 19 extended run on this GPU, the Wan decoder accounted for 51.9% of generation wall time and the DiT for 39.8%. At the later matched FlashAttention2 control, the DiT averaged 456.7 ms per window, Wan decoding 432.3 ms and motion encoding 116.4 ms. Replacing the two video stages cut their combined cost from approximately **549 to 44 ms per window**. [Extended-run analysis](docs/research/PRO_PIPELINE_PRECISION_AND_THROUGHPUT_2026-09-19.md) · [Matched tiny-VAE measurements](docs/research/TINY_VAE_FEASIBILITY_2026-09-26.md)

| Stage in the 52.90 FPS run, RTX 4070 SUPER | Total for nine windows | Share of 4.726 s generation |
|---|---:|---:|
| DiT | 4.103 s | 86.8% |
| Tiny decode | 0.334 s | 7.1% |
| Motion encode | 0.058 s | 1.2% |
| Audio | 0.073 s | 1.5% |
| Remaining wall time | 0.159 s | 3.4% |

The remainder is wall time minus the recorded stage totals. Treat it as unattributed overhead. With the rest of this run fixed, removing *all* tiny decode and motion-encode time would raise throughput only to about **57.7 FPS**. A large further gain must affect the transformer or the amount of work it performs. [Raw stage totals](benchmarks/pro_30fps_20260922/tae-decenc-trt-r01/results.json)

### iPhone 12 Pro — original runtime had two distinct latency problems

The following are **physical Apple A14** times at `fec3bf7`, automatic Core ML selection, 176 × 320. Each row is one complete 28-frame window. The columns are elapsed intervals between persisted progress markers. The front interval includes audio, projection and setup through the first partition marker. The denoising intervals include associated loading, patch/head work and dispatch. These traces do not isolate the duration of individual GPU kernels.

| Avatar / window | Front to first block | Denoise pass 1 | Denoise pass 2 | Decode + color / frame creation | Motion / delivery boundary | Window / generated FPS |
|---|---:|---:|---:|---:|---:|---:|
| Indian, first | 7.29 s | 61.04 s | 10.89 s | 1.62 s | 0.72 s | 81.57 s / 0.343 |
| Indian, next | 6.50 s | 11.86 s | 11.94 s | 0.72 s | 0.18 s | 31.20 s / 0.897 |
| Asian, first | 7.73 s | 63.50 s | 13.33 s | 1.75 s | 0.73 s | 87.04 s / 0.322 |
| Asian, next | 6.77 s | 13.58 s | 13.08 s | 0.79 s | 0.20 s | 34.42 s / 0.814 |

The Asian clip contained active speech. The Indian timing clip used a quiet opening. A separate initial Indian run took 85.28 s for its first window. First-use specialization/loading is consistent with the much slower first denoising pass, although the original instrumentation combines load and prediction. Warm denoising still accounted for **76–78%** of the window. Even eliminating the entire Asian front interval while holding everything else fixed would give only **1.01 FPS**. Future phone testing needs both first-frame latency and repeated warm windows. [Indian trace](apple/validation/iphone12pro_pro320_two_windows.json) · [Asian trace](apple/validation/iphone12pro_asian_active_speech.json) · [Device sample provenance](apple/validation/iphone12pro_asian_sample.json)

### M5 Pro — overlap hid work and exposed the remaining compute cost

Both sides below used the same full PRO bundle, seed 42, verified portrait cache and active local speech. For the historical reference we compiled the `fec3bf7` Swift runtime and selected CPU for **only the Wav2Vec encoder**, because of the speech defect in §6. Conditioning and DiT retained the requested policy. Warm FPS is pooled useful frames divided by pooled elapsed generation time after each trial’s first window.

| M5 Pro compute request / avatar | Paired trials | Reference FPS | Optimized FPS | Gain |
|---|---:|---:|---:|---:|
| Automatic / Indian | 2 | 9.07 | 15.24 | 68.1% |
| Automatic / Asian | 2 | 8.98 | 15.27 | 70.0% |
| CPU + GPU / Indian | 2 | 10.94 | 15.01 | 37.2% |
| CPU + GPU / Asian | 3 | 10.76 | 15.31 | 42.2% |

**New analysis of the saved Automatic traces.** The following values are mean **wall seconds per warm window** on the M5 Pro, pooled over 12 warm windows per variant across both avatars. Differences between adjacent progress markers partition the timeline into approximately additive intervals.

| Interval | Sequential reference | Optimized | Time removed |
|---|---:|---:|---:|
| Front through first denoising marker | 0.204 | 0.040 | 0.164 |
| Denoising pass 1 | 1.269 | 0.814 | 0.454 |
| Denoising pass 2 | 1.257 | 0.781 | 0.476 |
| Decode, Lab correction and frame creation | 0.294 | 0.169 | 0.125 |
| Motion update through emission marker | 0.079 | 0.031 | 0.048 |
| **Total** | **3.103** | **1.836** | **1.267** |

About **0.93 s of the 1.27 s saving** appeared in the two denoising intervals. Their optimized total was 1.595 s, or **86.9% of window wall time**. More detailed instrumentation further separates work that ran concurrently:

- The twenty transformer partition prediction calls averaged **1.478 s in total**, about 80.5% of the whole warm window. This is the strongest measured target for further improvement.
- Summed partition-load work averaged **0.782 s**, but 18 successor loads overlapped prediction. Explicit waits for the prefetched handles summed to about **0.09 ms**. The two first-partition loads and other setup still have costs.
- The audio worker averaged **0.135 s** of preparation, with about **0.008 ms** spent waiting for the completed context at consumption. The tiny decoder and motion encoder prediction calls averaged about **67 ms** and **17 ms** respectively.
- The mean of per-window peak *sampled* process footprints was about **1.39 GiB**. Samples were taken at model boundaries, so the number describes observed footprint rather than a guaranteed peak.

Load and worker durations overlap the wall intervals above. Adding them to 1.836 s would double-count work. These are Core ML API call timings. A model can specialize lazily inside prediction. No saved counter splits transformer time into measured FFN, attention and memory-bandwidth percentages.

The ten partitions account for roughly 2.6 GiB of compiled partition payload across twenty load requests per window. The same weights are visited on both passes. OS caches and memory mapping affect where those bytes come from. Timing model construction or observing high GPU utilization cannot establish a DRAM bandwidth limit. Operation traces and hardware counters are needed to decide whether a faster matrix kernel, a different layout or lower memory traffic would shorten the critical path.

The stage calculation used local `artifacts/coreml/fps-work/all-{ref,fast}-*/timings.json`, selecting windows 2–4. The boundaries were the audio start, first partition of each pass, decode start, motion update and streaming marker. Detailed prediction/load figures were summed across `blocks_00` through `blocks_09` before averaging per window. The committed record preserves run identities, hashes, timings and memory samples. [Matched run evidence](apple/validation/pro320_runtime_optimization.json) · [Benchmark method](apple/PERFORMANCE.md)

### Why the transformer is expensive — static work, independent of hardware

Each of two steps applies 30 blocks. For token count N, hidden width D=1,536 and feed-forward width F=8,960, the main multiply-accumulates per block per step are approximately:

| Operation | MACs |
|---|---:|
| Two FFN matrices | 2NDF |
| Self Q/K/V/O plus cross Q/O projections | 6ND² |
| Full self-attention score and value products | 2N²D |
| Audio K/V projections plus cross-attention products | 2×(9×32)D² + 2N×32D |

Counting the current Apple cross K/V projection each step, that is **5.77 trillion MACs per PRO320 window** or **0.412 trillion floating-point-equivalent operations per useful frame** at two operations per MAC. About **56.7%** of these MACs are FFN matrices, **29.2%** are self and cross Q/O-family projections as grouped above, **12.5%** are full self-attention products and **1.6%** are audio K/V and cross-attention products. At PRO576 the same approximation rises to **1.72 trillion operations per useful frame** and self-attention’s share rises to **32.2%**. This estimate excludes norms, activations, softmax, patch/head, audio, VAE, transfers and scheduling. Kernel efficiency determines the actual time shares. [Transformer source](apple/flashhead_coreml/dit.py)

Increasing the token count from 1,980 to 6,480 multiplies dense token work by about 3.27 and the self-attention products by about 10.71. If materialized, one all-head FP32 score tensor would occupy about **179 MiB at PRO320** or **1,922 MiB at PRO576**. A single FP16 FFN expansion is about 34 MiB or 111 MiB respectively. These are individual tensor sizes. Fusion, tiling, lifetimes and placement determine peak memory. Weight compression reduces weight storage while these activation shapes persist.

**Budget for a real-time product.** With 28 useful frames, 20 generated FPS allows **1.40 s/window**. Keeping up with 25 FPS audio/video playback allows **1.12 s/window**. At the measured M5 Pro Automatic mean of 1.836 s, that requires removing another 0.436 s or 0.716 s, respectively. Even if every other cost vanished and the measured block prediction time stayed fixed, 28 / 1.478 gives only **18.9 FPS**. At fixed non-block overhead, a 20 FPS target needs roughly a **30% cut in block prediction time**. These are conditional engineering budgets. The older A14 warm measurements imply a roughly **22–25×** throughput gap to 20 FPS. The Mac improvement supplies no measured multiplier for A14.

## 6. Optimizations implemented, tested and rejected

### Changes in the normal app

- **Bounded partition preloading.** While a partition predicts, a separate task can load its one successor. Prediction and mutation of temporal state remain ordered on the engine actor. Each pass begins with a synchronous load. A failed speculative load falls back to the serial loader. This works with the already compiled three-block bundle.
- **One-window audio lookahead.** The worker gets an immutable copy of exactly the history the serial path would consume, including partial-window padding. It returns conditioning values. The inference actor retains motion and random noise ownership. Scope and cancellation bound the lifetime of both workers.
- **Memory controls.** Overlap is limited to the tested PRO320 dimensions. iOS checks live process headroom, footprint and partition size before extra work. For example, the audio start check requires at least 2 GiB of headroom before any size-dependent reserve and a footprint below 1.25 GiB. These are engineering guardrails, not a measured iOS app limit. Sequential execution remains available in settings.
- **Less CPU work per window.** The packer copies the required FP16 speech slices directly and checks the 35 unique feature positions it uses. Tensor access maps buffers in bulk and respects strides. The color pass computes Lab once per pixel and reuses it for the transform. In isolated M5 Pro CPU checks, synthetic audio packing fell from 62.9 ms to about 0.52 ms. Correcting 28 frames fell from 195 ms to 76 ms with the same float32 output hash. These helper savings are already included in full-window measurements. [Audio measurements](apple/validation/pro320_audio_optimization.json) · [Swift runtime](ios/SoulXStudio/Sources/Runtime)
- **Reproducible comparisons.** The native Mac runner uses the app’s Swift inference sources, the same avatar caches and an optional historical reference build. It can save every frame and report per-model load, prediction, wait, memory and encoder policy. The frame comparer checks displayed pixels and consecutive-frame deltas. The fixed-avatar cache exporter and its instructions are tracked for fresh checkouts. [Runner](apple/run_mac_smoke.sh) · [Frame comparer](apple/flashhead_coreml/compare_frames.py)

### Speech defect uncovered during FPS work

The original end-to-end smoke checks could complete with finite video frames even when the speech features were invalid. On the **M5 Pro**, requests permitting GPU execution returned **1,843,200 non-finite Wav2Vec features** per case for five distinct inputs, including silence and active speech. The downstream projection hid the failure by producing the same finite context. With CPU-only encoding, the five inputs produced five distinct outputs and matched FP32 PyTorch within **0.022% NRMSE**. Injecting valid reference features into the GPU-eligible projection passed its numerical gate.

The fix checks the consumed speech features *before* projection. On this numerical failure, the runtime retries only the speech encoder once on CPU. It remembers a successful fallback for the session, shares it with the worker and records the resolved policy. Other errors and a failed CPU retry propagate to the caller. Conditioning and the transformer retain their requested compute units. This explains the one placement correction in the historical Mac speed reference. The A14’s speech backend has not been independently characterized. [Five-input diagnostic](apple/validation/pro320_audio_backend_quality.json) · [Validation tool](apple/flashhead_coreml/validate_audio_backends.py)

### Experiments that informed the defaults

| Experiment | Result | Decision |
|---|---|---|
| Three-block Q/K/V fusion, M5 Pro GPU-eligible run | 10.86 → 10.85 warm FPS when compared in isolation. Paired pixels matched. | Keep as an optional exporter experiment. |
| Six-block fusion, M5 Pro | 10.72 → 10.90 warm FPS in its paired trial. Largest partition grew from 133.80 to 267.07 MiB. Three six-block comparisons exceeded the 5% FP32 gate, up to 6.61%. All 30,412,800 compared elements matched the already-quantized three-plus-three reference. | Keep three-block partitions in the app. The isolated layout timings predate the speech correction. |
| Channels-first 1×1 convolutions, M5 Pro Automatic | Compute plan preferred the GPU for both layouts. A loaded partition took roughly 56.2 ms with conv versus 47.1 ms with linear projections. | Keep linear projections. Actual A14 placement requires its own trace. |
| FP16 storage for audio matrices with FP32 computation, M5 Pro CPU | Weight bytes fell from 377.5 MB to 188.8 MB. Accuracy passed, but warm prediction slowed from roughly 102–105 ms to 171–185 ms in the storage probe. | Keep FP32 audio weights. |
| Pinning a small prefix of DiT partitions | The preliminary Mac trials did not establish a consistent advantage. A sequential walk repeatedly misses in a small ordinary LRU. | Spend spare memory on the next partition that will actually execute. |
| Historical CUDA block skipping or direct latent feedback, RTX 4070 SUPER | Recorded experiments showed degraded lip motion, mouth detail or color drift. Decoder pruning needed teacher distillation to regain detail. | Keep all 30 PRO blocks and the pixel-to-motion feedback loop in the quality-preserving runtime. |

[Partition and layout study](apple/validation/dit_partition_trials_mac.md) · [Compute-plan probe](apple/validation/dit_conv1x1_probe_mac.json) · [Historical CUDA quality experiments](docs/research/PRO_40FPS_ITERATION_LOG_2026-09-21.md)

### What “same quality” covers

For the optimized Apple runtime, every corresponding displayed RGB pixel matched the speech-corrected reference on both four-window avatar clips. A separate 9.48-second run crossed the eight-second history rollover and ended with a partial 13-frame window. All 237 frames matched, including their temporal differences. These checks cover the rendered face, mouth, background and next-window motion on those inputs.

Three other questions still need separate evidence. Cross-platform equivalence to CUDA is affected by different precision, RNG and image resampling. The tiny decoder has known differences from the Wan teacher. Sustained phone speed, thermals, per-operation ANE placement and speech-conditioned quality on the updated A14 runtime remain unmeasured. [Full-frame hashes and longer run](apple/validation/pro320_runtime_optimization.json)

## 7. Plan from here, ranked by expected value

| Priority | Experiment | Measurement and decision gate |
|---|---|---|
| 1. Establish the updated A14 baseline when a device run is wanted | On the same iPhone, compare sequential and overlapped execution with identical PRO320 bundle, fixed avatar, seed, speech and compute policy. Test both avatars and record the resolved speech encoder policy. | Record cold first-frame delay, warm p50/p95 window time, useful FPS, per-partition load versus prediction, wait, peak sampled footprint, live headroom, fallback events and thermal state. Include repeated runs and a longer recurrence test. Require matching frame count and visible pixels where the encoder backend is equivalent. Verify audio features independently if fallback changes the backend. |
| 2. Profile inside the transformer | Measure real representative blocks at both timesteps with production hidden states. Compare compute plans and Instruments traces for automatic, CPU + GPU, and CPU + ANE. Separate dense FFN, attention, normalization, transfers and processor handoffs. | Progress-marker timing cannot choose between FFN and attention kernels. Take forward a change only if a full matched run improves latency beyond run-to-run variation and passes the numerical and frame gates. |
| 3. Optimize the demonstrated hot operation | If dense layers dominate, test supported matrix layouts or better compilation with the existing weights. If attention dominates time or memory, test exact query tiling with access to all keys. Consider reuse of audio cross-attention K/V within the two steps only after timing its small arithmetic share against extra memory traffic. | Use the production profile and compare wall time, peak memory and recurrent output. Preserve FP32-sensitive attention/norm operations unless their replacement passes the real-weight checks. No cross-step reuse of noisy self-attention K/V is assumed. |
| 4. Reduce model work if the target is interactive A14 | Train and evaluate a smaller denoiser student or a genuinely one-step distilled student. Another candidate is a lower-token profile with explicit face-detail evaluation. If tiny-VAE mouth artifacts matter more than speed, fine-tune it on SoulX latents using the full Wan decoder as teacher. | These are model-development projects. Use speech and silence, plosives, open vowels, several seeds, both avatars and held-out identities. Track mouth motion/timing, teeth and oral edges, identity, color drift and long-window stability against the existing control. Require sustained target-device timing before committing to 20 or 25 FPS. |
| 5. Improve startup separately if arbitrary portraits become a requirement | Partition the Wan reference encoder by temporal group or spatial tile with explicit causal state. Keep the fixed-avatar cache path as the measured baseline. | Validate the full reference latent, seams and finite values against original Wan. Record on-device peak footprint and initialization latency. A startup improvement only changes recurring FPS if it also changes recurring work. |

**CUDA follow-up.** Reproduce the 52.90 FPS FlashAttention2 arm on a freshly identified GPU, then test the proposed SageAttention2 plus tiny-VAE combination in the same session with matched input, seed, step count and decoder mode. Record stage times, cold and warm throughput, memory and the mouth/texture quality checks. The 67.5 FPS estimate becomes a benchmark only after this run. To compare CUDA and Core ML numerically, supply the same preprocessed tensors and actual noise values at both timesteps. Matching the integer seed alone does not align different RNG implementations.

For scheduling-only changes, use pixel equality at fixed seed and inputs as the acceptance gate. For new kernels, also keep the component-level FP32 parity comparisons. For retrained models, establish tolerances from the control’s own variability and review clips with their audio. A single endpoint image, a smaller package, or a successful conversion is insufficient evidence to promote a quality-sensitive speed change.

## 8. Reproduce and inspect

Run these from the repository root on an Apple Silicon Mac. The isolated environment uses Python 3.12, `uv`, the pinned Apple requirements and Xcode tools.

```sh
make -f apple/Makefile setup
make -f apple/Makefile fetch
make -f apple/Makefile test
make -f apple/Makefile export
make -f apple/Makefile bundle
bash ios/SoulXStudio/Tests/run.sh
```

To regenerate the two fixed-avatar caches from the exact bundled PNGs:

```sh
for avatar in indian girl; do
  dir="ios/SoulXStudio/Resources/Avatars/$avatar"
  apple/scripts/export-reference-cache.sh --models artifacts/coreml/device/pro320 \
    --image "$dir/portrait.png" --output "$dir/reference-cache" \
    --checkpoint models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth
done
```

The importable compiled profile is `artifacts/coreml/device/pro320`. Open `ios/SoulXStudio/SoulXStudio.xcodeproj` for the native app. Import the model folder or copy it to `Documents/Models/current`. Select an avatar and audio file, initialize the portrait, then generate. The exporter and runtime manifest verify component contracts. For a disconnected phone, all conversion, parity tests and Mac comparisons can run locally.

To reproduce the matched FPS and quality comparison, use a fresh output directory for each run. For an exact local reproduction, create the fixture with `ffmpeg -ss 3 -i /path/to/eng.wav -t 4.48 -ar 16000 -ac 1 speech.wav` and point the `speech` variable below at it. The recorded four-window fixture is mono 16 kHz audio spanning 3.00–7.48 s of the local MuseTalk `data/audio/eng.wav`. The longer fixture spans 3.00–12.48 s. Run with `--compute all` or `--compute cpu-gpu` consistently on both sides.

```bash
avatar=ios/SoulXStudio/Resources/Avatars/girl
speech=artifacts/coreml/fps-work/speech.wav
common=(--models artifacts/coreml/device/pro320 --portrait "$avatar/portrait.png"
        --reference-cache "$avatar/reference-cache/metadata.json"
        --audio "$speech" --chunks 4 --compute all --save-frames)

SOULX_BENCH_BASELINE=1 apple/run_mac_smoke.sh "${common[@]}" --output /tmp/soulx-ref
apple/run_mac_smoke.sh "${common[@]}" --prefetch-audio --prefetch-blocks \
  --model-timings --output /tmp/soulx-opt
.venv-coreml/bin/python apple/flashhead_coreml/compare_frames.py \
  --baseline /tmp/soulx-ref --candidate /tmp/soulx-opt
```

For a future phone comparison, the app’s `--smoke --smoke-chunks 2 --smoke-save-frames --model-timings` arguments produce frames and a stage report. Add `--prefetch-audio --prefetch-blocks` for overlapping execution. Supply matching portrait-cache metadata, source PNG and sufficiently long audio in `Documents/Smoke`. The options and deployment procedure are in the [app guide](ios/SoulXStudio/README.md) and [Apple build guide](apple/README.md).

The locally generated sample media are `artifacts/coreml/iphone-asian-active-speech/generated.mp4` from the earlier physical A14 run and `artifacts/coreml/fps-work/final-source-check/generated.mp4` from the optimized Mac run. They play at the model’s 25 FPS timebase. The [A14 sample record](apple/validation/iphone12pro_asian_sample.json) and [optimized Mac record](apple/validation/pro320_runtime_optimization.json) identify the corresponding workloads and generation timings.
