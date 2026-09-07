# 6. TensorRT partitions, experiment tooling and test coverage

[Guide index](README.md) · [Previous](05_WEBRTC_AND_SCHEDULING.md)

## What is TensorRT today

[trt_backend.py](../../soulx_rtc/trt_backend.py) installs two optional accelerators:

1. Exact-shape BF16 feed-forward islands for the Lite transformer's 30 MLP blocks.
2. An exact-shape deterministic VAE decode wrapper, including latent denormalization.

Wav2Vec, attention, timestep/modulation, recurrence/motion encoding, random sampling, color conversion and orchestration are not thereby converted to TensorRT. “TensorRT enabled” is not synonymous with whole-model export.

The FFN runtime requires batch one in the server and is incompatible with staged DiT offload. It can install a valid subset of layer artifacts; health must report the actual installed indices. A directory's name is not evidence that all 30 layers loaded.

## Artifact build path

[trt_experiment.py](../../soulx_rtc/trt_experiment.py) captures real execution inputs and exports FFN modules. It uses a small CPU trace input for ONNX export, then fixes the intended full input dimensions and builds strongly typed engines. Comparison and timing use the actual full captured shape, not the one-token trace size.

The capture hooks retain the first invocation for each FFN layer—normally the first denoising step of the first chunk from a short fixture. That is useful input realism, but not coverage of all timesteps, recurrent states, avatars or interruptions. A larger calibration/parity corpus is a planned improvement.

The builder records finite-output checks and max/mean absolute error, and times warmup and repeated execution. Present reporting does not impose a hard, corpus-level numerical/perceptual acceptance threshold. An exported or “finite” engine is not automatically visually acceptable.

Resume behavior based on existing report files must not replace artifact revalidation. Artifacts need their metadata, engine bytes, checkpoint, runtime compatibility and reference outputs checked again when reused.

[trt_vae_experiment.py](../../soulx_rtc/trt_vae_experiment.py) exports the VAE decode wrapper and compares it against a captured BF16 decode. An optional FP16 path with FP32 PixelNorm handling exists, but it is **not** a measured/certified optimization in the retained results.

## Runtime validation and execution

The backend checks exact input shape, artifact SHA-256, checkpoint hash, TensorRT version and GPU identity metadata. It validates the two expected bindings, directions, dtypes and shapes. Mismatches fail closed rather than silently falling back while reporting TRT performance.

The contexts are created without internal workspace ownership. FFN contexts use one Torch-backed uint8 arena sized for the largest serial partition. On every invocation the wrapper validates/converts contiguity and dtype, allocates the output, binds input/output addresses and enqueues execution on the current CUDA stream.

The FFN wrapper is excluded from torch compilation. Thirty blocks times four steps produces **120 Python/TRT crossings per chunk**. This motivates measuring larger islands and fixed buffers; it does not prove that all 120 crossings are expensive enough to dominate execution.

The VAE decoder's original Torch weights are moved off GPU when replaced. Its workspace is allocated lazily for the decode, synchronized/released before recurrent encoding, and rebound on the next use. This reduces overlapping peaks but conflicts with a naive whole-path CUDA graph requiring stable addresses.

Future CUDA graph capture should isolate compatible fixed-shape segments, externalize mutable inputs and preserve RNG/session state. PyTorch documents static-address/control-flow and capture restrictions in its [versioned CUDA notes](https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/docs/source/notes/cuda.rst). Do not capture the entire current dynamic service loop as if it were a pure fixed tensor function.

## What the saved microbenchmarks mean

The native portrait FFN microbenchmark reports roughly 1.154× acceleration for the tested partition workload; the square result is about 1.034×. Native VAE decode was approximately 246.66→208.62 ms in its isolated test (1.182×), with nonzero output differences.

Those are stage results. The historical ten-job whole-engine medians were approximately 27.60 FPS for optimized Torch and 28.57 FPS for TRT, under runs that were not a tightly matched alternating A/B. The requested 25% whole-model improvement was not demonstrated.

Microbenchmark captures do not measure recurrent quality drift, end-to-end request latency, IPC/encoding cost or multi-peer smoothness. [NVIDIA's performance guidance](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html) is background for profiling methodology, not evidence that an engine built with this checkout's installed version has a particular speedup. Compatibility must be checked against the installed runtime, not inferred from latest documentation.

## Benchmark module map

| Module | Measures / produces | Important exclusions or traps |
| --- | --- | --- |
| [benchmark_engine.py](../../soulx_rtc/benchmark_engine.py) | Historical square-profile engine sweeps | Not the full native-portrait media stack |
| [benchmark_musetalk.py](../../soulx_rtc/benchmark_musetalk.py) | Local MuseTalk conditioning, inference, decode and composition comparison | Composed output resized to 512²; not current native-portrait production TRT |
| [experiment.py](../../soulx_rtc/experiment.py) | Repeated useful-frame benchmarks and detailed CUDA phase events | Setup, warmup and encoding excluded; first recorded session/repeat is not a broad quality corpus |
| [benchmark_rtc.py](../../soulx_rtc/benchmark_rtc.py) | Finite peer transport and recording | Uses square recording dimensions from health size; unsuitable unchanged for a rectangular recording claim |
| [benchmark_calls.py](../../soulx_rtc/benchmark_calls.py) | Persistent turns, interruption, peer ramps, resource snapshots and cleanup | Paced wall includes gaps/idle/admission; sequential submission does not exercise already-queued turn lookahead |
| [analyze_recordings.py](../../soulx_rtc/analyze_recordings.py) | Decoded-frame differences and nominal chunk-boundary statistics | Every 24th recorded frame is not necessarily a real chunk boundary when holds/idle exist |
| [check_recorded_audio.py](../../soulx_rtc/check_recorded_audio.py) | Correlation/alignment of speech waveforms and inserted pauses | Not a perceptual lip-sync metric; codec transformations and silence selection matter |
| [replay_lab.py](../../soulx_rtc/replay_lab.py) | Source-bank manifest, hashes, boundary comparisons and persistent relay | Relays the six source videos; does not demonstrate SoulX generating their trajectories |
| [metrics.py](../../soulx_rtc/metrics.py) | Compact chunk metric structure | Interpretation still depends on source version/configuration |

### Concrete measurement corrections to implement next

The finite recorder assigns both dimensions from `health["size"]`, losing rectangular geometry. Record neural width/height and actual receiver dimensions separately; refuse a benchmark where intended and decoded sizes differ.

The current call `transport_pass` checks basic received media, monotonic timestamps, no reported transport errors/loss, one negotiation and actual H264. It does not enforce minimum useful FPS, maximum held-frame fraction, maximum audio hold or first-media latency. Add separate transport, smoothness, quality, resource and capacity verdicts.

Recorded frame index modulo 24 is not provenance. Tag every generated chunk with call/turn/epoch/chunk index and intended frame/sample/PTS ranges, then map those to sender and receiver evidence. Source-idle and held frames need explicit labels.

The historical call client waits for completion and adds `gap` before posting the next turn. Add a mode that prequeues multiple audio segments, plus mixed simultaneous/idle peers, to measure scheduler lookahead fairly.

## Automated test map

The last implementation catalog reports **34 passing automated tests**. That is a historical run, not a new GPU regression performed by this documentation audit.

| File | Coverage focus | What it does not establish |
| --- | --- | --- |
| [test_calls.py](../../tests/test_calls.py) | Persistent turn lifecycle, retry bounds, interruption, idle/audio scheduling and cleanup using controlled fixtures | Native GPU concurrency or perceptual seams |
| [test_codec.py](../../tests/test_codec.py) | Encoder factory/preset behavior and compatibility | Mobile receiver behavior or encode capacity at ten peers |
| [test_gpu_lease.py](../../tests/test_gpu_lease.py) | Advisory lock exclusion | Reservation against unrelated GPU applications |
| [test_optimizations.py](../../tests/test_optimizations.py) | Small-model optimized math/conditioning/isolation contracts | Full checkpoint recurrent quality or actual fast-attention dispatch |
| [test_recorded_audio.py](../../tests/test_recorded_audio.py) | Audio alignment/inserted-gap analysis behavior | Human-rated lip synchronization |
| [test_replay_lab.py](../../tests/test_replay_lab.py) | Asset manifest/path safety and replay-related contracts | Neural reconstruction of source assets |
| [test_rtc.py](../../tests/test_rtc.py) | Finite-session inputs, tracks, queues and lifecycle | Native GPU rendering throughput |
| [test_rtc_loopback.py](../../tests/test_rtc_loopback.py) | Actual local DTLS/H264/Opus loopback | TURN, WAN congestion and phone browser certification |
| [test_trt_contract.py](../../tests/test_trt_contract.py) | Mocked artifact/binding/profile rejection | Real engine numerical accuracy on all timesteps |

Separate GPU isolation evidence compared reordered B1 execution for two sessions/two chunks and found zero maximum pixel difference in the measured setup. This is valuable state-isolation evidence, but it is not a B2 TRT batching test or proof of indefinite recurrent determinism.

## Reproducibility hierarchy

A defensible performance artifact should identify source commit and relevant file hashes, checkpoint identity, image/audio hashes, geometry, steps, batch, precision, attention/backend identities, memory mode, engine hashes, runtime/driver/GPU, co-resident memory conditions, warmup, repetitions and every metric's numerator/denominator.

Keep failed/OOM runs alongside successful ones. Do not merge different source versions or hardware into one “best FPS” without labeling the comparison.

The executed [evidence notebook](../research/notebooks/optimization_evidence_audit.ipynb) rechecks retained JSON arithmetic and provenance on CPU. It does not rerun inference. The [validation report](../research/EVIDENCE_VALIDATION_2026-09-07.md) lists the conclusions it supports.
