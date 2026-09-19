# GPU attribution for recent runs — September 6–16, 2026

September 17 shoulder-visible source-detail extension: [six 320×576/25-FPS runs](../../benchmarks/portrait_source_detail_20260917/README.md) executed on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB**, driver 595.84, Torch 2.7.1+cu128/CUDA 12.8, INT8 weight storage/BF16 compute. Direct run snapshots show changing whole-device pre-load use of 2,487–7,705 MiB from OmniVoice and another LTX service. Therefore throughput rows are not a matched performance comparison; the fixed-setting visual matrix remains the intended evidence. An initial BF16/staged load failed under the highest co-residency and produced no sample.

September 17 source-detail extension: [four fixed-output source-resolution runs](../../benchmarks/source_detail_20260917/README.md) executed on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB**, driver 595.84, Torch 2.7.1+cu128/CUDA 12.8. Each result contains a direct device snapshot showing 2,809 MiB whole-device use before SoulX load; OmniVoice remained resident. Profiles fix 512×512/25 FPS, BF16/four steps/eager/staged and vary only effective source-crop detail 307/256/128/64. CPU landmark analysis and packaging are separate from GPU inference.

September 17 extension: [four square-resolution teeth runs](../../benchmarks/square_teeth_20260917/README.md) executed on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB**, driver 595.84, Torch 2.7.1+cu128/CUDA 12.8. Each result contains a direct device snapshot. OmniVoice remained resident; initial device use was 2,487 MiB. Profiles: 512²/1024² × 15/25 FPS, BF16, four steps, eager, staged weight offload, 8,704-MiB allocator cap. CPU packaging and media decoding are distinct from GPU generation. Both 1024 profiles completed but fail visual quality.

September 16 concurrency addendum: direct SoulX GPU inference and local WebRTC load tests ran on **NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible**, driver **595.84**, Torch **2.7.1+cu128 / CUDA 12.8**, 180-W power limit. OmniVoice PID 157695 occupied about 2,478 MiB before SoulX and was not stopped. The 200-ms `nvidia-smi` log observed 2,487–8,895 MiB whole-device use across all profiles; selected batch-five Torch telemetry peaked at 6,148 MiB reserved under a 7,168-MiB allocator cap. Five active 320×576/four-step/15-FPS streams passed video cadence over 15 turns; six failed. See [exact environment, raw runs and recording](../../benchmarks/concurrency_15fps_20260916/README.md). This is new local GPU evidence, distinct from the earlier read-only concurrency audit.

September 16 addendum: fresh Ditto 0.5/0.7 framing inference ran on **NVIDIA GeForce RTX 4070 SUPER**, 12 GB class / 12,282 MiB visible, driver 595.84, Torch 2.5.1+cu121 / CUDA 12.1, TensorRT Ampere Plus. Both runs record 2,487 MiB pre-load device allocation, workload identity unverified. See [0.7 run metadata](../../benchmarks/distance_lipsync/evidence-ditto-closer-0p7-20260916/results.json) and [report](../../benchmarks/distance_lipsync/DITTO_CLOSER_0P7_REPORT.md). CPU statistics/media checks and reused cross-model resource figures are labeled separately. The original host attribution and no-rerun statement below concern the earlier documentation audit, not this addendum.

**Local talking-head experiment host: NVIDIA GeForce RTX 4070, 12 GB
(12,282 MiB visible).** Saved September 11 evidence reports driver **570.181**.
The same model, visible memory and driver were observed during this documentation
audit. Current hardware alone is not proof of any historical run's device.

This index covers the requested recent nine-day window (September 7–15), plus
the September 6 baseline reused in those reports. Dates below are run/report
dates; Git commit dates can differ and are not substituted for execution dates.
No GPU inference was rerun to make these documentation changes.

## Runs and evidence

| Run/report date | Work | Tested GPU / execution type | Evidence and limits |
|---|---|---|---|
| September 6–7 | SoulX versus MuseTalk, native portrait, TensorRT partitions, persistent calls | RTX 4070, 12 GB | [Benchmark environment](../../benchmarks/REPORT.md), [implementation results](IMPLEMENTATION_RESULTS.md), and [test catalog](TEST_CATALOG.md) already record this GPU. Historical MuseTalk RTX 3090 results remain separate. |
| September 7 | Evidence notebook, code/architecture and optimization audits | CPU/static audit; no new GPU inference | [Evidence validation](EVIDENCE_VALIDATION_2026-09-07.md) reanalyzes retained RTX 4070 results. Notebook execution is not another GPU benchmark. |
| September 8 | Two-browser wall, Kokoro, persistent turns | RTX 4070 for SoulX; Kokoro TTS on CPU | [Browser wall report](../../benchmarks/wall/README.md) explicitly records the RTX 4070 and co-resident OmniVoice. |
| September 11 UTC (September 10 Chicago) | Ref2VA preflight; local FlashHead/MuseTalk comparison | RTX 4070, 12,282 MiB | `/workspace/experiments/ref2va-evaluation/preflight.json` records the device and timestamp. Full Ref2VA inference was **not** executed. Its publisher's RTX 4090 result was not reproduced. |
| September 11 | Low-VRAM optimization and compiled MuseTalk controls | RTX 4070, 12 GB | `/workspace/experiments/flashhead-optimization/RESULTS.md` records the device; saved MuseTalk benchmark JSON/logs provide supporting evidence. A 3.5–4 GB allocator budget is not a test on a physical 4-GB GPU. |
| September 11 | Pipeline profiling, 72 renders, avatar and ten-peer tests | RTX 4070, 12,282 MiB, driver 570.181 | [Pipeline report](PIPELINE_PROFILE_2026-09-11.md); `/workspace/experiments/flashhead-pipeline-ARsFTh/environment-final-source.json` contains `device_snapshot`. OmniVoice remained resident. |
| September 12–13 | Exact execution/serving optimizations | RTX 4070, 12,282 MiB | `/workspace/experiments/flashhead-opt-20260912/source/docs/research/EXACT_OPTIMIZATIONS_2026-09-12.md` explicitly records the GPU in its baseline contract. This older experimental report is distinct from the later media release. |
| September 13 | Media cache/IPC, single-call and ten-peer receiver validation | RTX 4070, 12 GB — retrospective local-host attribution | [Media report](MEDIA_OPTIMIZATION_2026-09-13.md) links the local experiment directory and September 11 pipeline. The retained summary does not independently certify a per-run GPU snapshot. CPU regressions/idle decode controls are not GPU throughput tests. |
| September 13 | Teeth diagnosis, 24 male-quality renders and eight receiver captures | RTX 4070, 12 GB — retrospective local-host attribution | [Male quality report](MALE_TEETH_QUALITY_2026-09-13.md) and `/workspace/experiments/flashhead-teeth-u1x4l9/ANALYSIS.md`. Device attribution follows the preserved local experiment chain, not a separate hardware capture for each render. |
| September 13 | Still-reference versus idle-motion, 12 renders | RTX 4070, 12 GB — retrospective local-host attribution | [Still-reference report](STILL_REFERENCE_QUALITY_2026-09-13.md) records the same local inputs/profile and prior male-quality experiment. No independent per-render GPU snapshot was identified. |
| September 13 | Mouth-strength sweep, 32 renders and landmark analysis | RTX 4070, 12 GB — local-host attribution; analysis renderer recorded directly | [Mouth report](MOUTH_MOVEMENT_CONTROL_2026-09-13.md). `/workspace/experiments/flashhead-mouth-3lwV6k/measured/outputs/measure.log` explicitly records `NVIDIA GeForce RTX 4070/PCIe/SSE2` and NVIDIA 570.181. That log proves the analysis renderer; it is not a separate CUDA device record for every generation run. |
| September 15 documentation audit | Git/disk checks and new Vast installer checks | Host RTX 4070; **no GPU inference/fresh install test** | [Installer documentation](https://github.com/AhmadAFS1/MuseTalk/blob/deploy/vast-talkingheads-20260915/deploy/talkingheads/README.md). Shell syntax, ShellCheck, dependency resolution and checkout/restart checks ran. No Ditto GPU result exists from this work. |
| September 16 | Character-distance and strength-0.5 interaction experiments | NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible; driver 595.84; Torch 2.7.1+cu128 / CUDA 12.8 | [Distance experiment index](../../benchmarks/distance_lipsync/README.md) and per-run `results.json` files capture the device, pre-load allocation, profile, hashes and timings. GPU generation is distinct from CPU-side MediaPipe analysis. |
| September 16 | Five-stream 15-FPS SoulX concurrency qualification | NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible; driver 595.84; Torch 2.7.1+cu128 / CUDA 12.8 | [Concurrency evidence](../../benchmarks/concurrency_15fps_20260916/README.md) captures the co-resident load, exact serving profile, raw per-peer counters, timestamp arrays, GPU samples and recorded receiver output. Five is an experimental video ceiling; four is the conservative recommendation. |

Retrospective local-host attribution means the report and local experiment
lineage identify this host, supported by saved device observations from the
same work period. It is weaker than hardware metadata captured with each run.
Do not turn that attribution into a claim of independently verified device,
driver, power limit, or co-resident memory for every individual measurement.
A sanitized extract of the historical hardware observations is retained in
[gpu_provenance_evidence.json](gpu_provenance_evidence.json).

## Other GPUs in the documentation

- **RTX 3090:** historical MuseTalk results/artifacts, and older OmniVoice load
  tests. Do not label them RTX 4070 or present them as September reruns.
- **RTX 4090:** upstream SoulX/Ref2VA claims and possible deployment hardware;
  these are not measurements of our local recent experiments.
- **RTX 5090:** upstream OmniVoice benchmark reports already identify this GPU.
  They are not benchmarks of the co-resident OmniVoice process on this host.
- **RTX 5070:** historical April OmniVoice load tests, explicitly dated and
  labeled in that repository. No new September attribution is implied.
- **A100:** upstream/tested-environment references or deployment suggestions,
  not a new local result from the work above.

## Required format for future documentation

Every run report and run summary must state, near the beginning:

1. Execution date, exact GPU model, and physical/visible VRAM.
2. Driver, framework/CUDA version, and hardware-log/artifact reference when known.
3. Model/profile, precision, resolution, warmup/timing boundaries, and relevant
   co-resident load. An allocator cap must be labeled separately from GPU VRAM.
4. Whether the result is GPU inference, a CPU/static check, reused historical
   evidence, or an external claim. Use **GPU unverified** when not established.

For mixed-hardware tables, put a GPU column beside the run/result. Do not bury
hardware only in a linked report or infer it from the machine currently open.
