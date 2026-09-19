# PRO 30-FPS implementation work — 2026-09-19

**Status: implemented and tested; experimental candidate retained, 30-FPS target not met.** This executes the cost-justified branches of the [30-FPS plan](../../docs/research/PRO_30FPS_QUANTIZATION_ANALYSIS_AND_PLAN_2026-09-18.md). The candidate improves speed but remains below real time and MuseTalk; lower relative sync scores on two utterances also prevent an unconditional quality promotion. The live server is restored and production selection is unchanged.

Hardware: **NVIDIA GeForce RTX 4070 SUPER, physical 12 GB / 12,282 MiB visible**, UUID `GPU-69894a01-4488-37ac-5fd8-878224044365`, driver 595.84. PRO uses Torch 2.7.1+cu128 / CUDA runtime 12.8. MuseTalk uses Torch 2.5.1+cu121 / runtime 12.1 on this same GPU. Before the approved pause, residents used approximately 2,478 MiB, 234 MiB, and 5,362 MiB; the last was the live SoulX worker. Source: [dated build/environment manifest](sage-sm89-build-manifest.json). The new compiler is isolated CUDA 12.8.93; the system CUDA 12.1 compiler and working Sage installation are preserved. TensorRT stage builds use 10.9.0.34 with a 512-MiB tactic workspace limit; actual execution workspace includes intermediate activations and can exceed that limit.

The user approved pausing the live SoulX server. It was stopped gracefully with no active sessions; its exact command/environment were saved privately for restoration. Fresh GPU runs below therefore exclude its worker. Final restoration used the saved command/environment and verified ready status, 512×512, four steps, 25-FPS playback, two active-call slots and zero sessions. [Restoration record](service-restoration.json). During the main sweep, the other two residents used approximately 2,478 and 234 MiB; idle device use was about 2,727 MiB. The later correction run records its changed resident load below. The exclusive lease remains enforced by [gpu_lease.py](../../soulx_rtc/gpu_lease.py). Server/UI edits being made concurrently in this checkout are outside this quantization change.

## Fresh GPU observations

All observations in this section are from the RTX 4070 SUPER described above, on 2026-09-19, with the live SoulX worker paused and the other residents retained.

| Fresh configuration | Useful generation FPS | Evidence / qualification |
| --- | ---: | --- |
| Existing accepted V2 | 10.3242 | Initial seed-50 reproduction |
| V2 with native-SM89 FP16 attention | 10.4349 | Build attribution, one seed-50 run |
| Native FP8-PV attention, original compiled BF16 decoder | 10.7308 | One seed-50 pilot; about 4% over initial reference |
| INT8 residual stages + native FP8-PV, separate workspaces | 6.7296 | Rejected full-pipeline regression |
| Same INT8 engines with shared workspace | 7.6659 | Still rejected; later windows slow to about 4.45 s |
| Fused FP16 stages + native FP8-PV | 11.8235 (seed 50), 11.7854 (seed 1) | Pilot; completed 24-process sweep below |
| Same faster PRO candidate, corrected matched MuseTalk study | **11.8111 median** | Three fresh processes; 11.8043–11.8558 |
| Native PyTorch MuseTalk V1.5 FP16, batch 8 | **42.1885 median** | Three alternating fresh processes; 42.0677–42.2052 |

The [corrected matched study](matched-musetalk-r02/schedule.json) includes audio processing, model work, compositing where applicable, and host RGB delivery. Encode/mux is separate: median delivered FPS is **11.6399 PRO versus 39.4930 MuseTalk**. Both deliver 250 frames at 320×576 with identical hashed effective float32 audio samples and fixed seed 50. PRO generates full frames while MuseTalk renders 256×256 face crops into prepared material. Model load/avatar preparation and first warmup are excluded for both; native PyTorch MuseTalk is not the fastest installed TensorRT service. Neither 30 FPS nor catching MuseTalk has been achieved.

**Measurement correction:** [first comparison r01](matched-musetalk-r01/schedule.json) is superseded, not silently deleted. Its MuseTalk generation timer stopped before BGR→RGB conversion; delivery time included that conversion. Protocol 2 moves RGB conversion inside generation, verifies exact effective audio hashes, fixes MuseTalk's random seed and snapshots both comparison scripts. Only r02 is used for final matched claims.

- Accepted V2, seed 50, 250 useful frames: **10.3242 FPS**, 24.2150 s generation. [Manifest](reference-seed50-r01/results.json). Decoder 13.0187 s, DiT 9.1274 s, motion encode 1.3751 s, audio 0.07185 s; stage totals exclude host gaps and are not a replacement for generation wall time.
- The repaired profiler records 580 nonzero-qualified aggregate rows while retaining the compiled decoder. [Profile](compiled-profile-r01/profile.json). CUDA kernel rows and CPU operator rows overlap: sum CUDA rows only for exclusive kernel attribution.
- Real input capture retained 42 post-rotary Q/K/V triplets and four public decoder latents, including later recurrent windows. [Manifest](real-inputs-r01/manifest.json). The separate stage calibration covers all six initial/recurrent signatures with at least four observations each. [Calibration](stage-calibration-r01/results.json).
- Native-SM89 FP8-PV attention versus native FP16-PV: median **1.4258×** kernel speedup over 42 cases, relative L2 median 1.46%, maximum 2.27%; all finite. [Measurements](attention-fp8-r01/results.json). Native FP16 versus the existing FP16 build is approximately unchanged (median 1.0043×); native FP8 versus existing FP16 is 1.4366×. These are separate kernel diagnostics, not video FPS or quality acceptance.
- The initial BF16 TensorRT export built but produced NaNs in all six signatures. [Failed full decode](stage-bf16-trial-r01/results.json), [direct replay](stage-bf16-direct-r01.json). Replacing ONNX's lower-bound-only BF16 `Clip` with equivalent `Max` removes the implicit upper bound and produces finite results on every rebuilt signature. [Rebuilt control](stage-bf16-build-r02/results.json). Builds now execute the real input and check every output/cache before completion; installation rejects missing execution evidence.
- The first repaired full-decode control exposed Dynamo's eight-specialization limit. That diagnostic is retained; subsequent stage trials allow 64 specializations, record the setting and inspect logs for fallback. TensorRT runtime now uses a dedicated stream with caller fences and allocator lifetime recording, avoiding its default-stream synchronization warning.
- BF16 Q/DQ stage graphs selected floating-point convolutions and were rejected by the inspector. FP16/FP32 internal graphs with BF16 final bindings quantized only five of six convolutions. Explicit FP16 engine bindings allowed **all six residual convolutions per stage, in all six signatures**, to use actual INT8 inputs/weights/tactics. [Verified INT8 build](stage-int8-build-r02/results.json), [FP16 control](stage-fp16-build-r01/results.json), [probe evidence](stage-half-boundary-probe-r01/results.json). Caller-owned caches are converted back to BF16; the decoder head and other stages remain BF16. This precision change is explicitly labeled, not described as an all-BF16 control.
- Fixed-latent FP16 stages measured 1.098–1.151× whole-decoder speedup over compiled BF16, with 0.263–0.346% decoded relative L2 and exact repeat/reset checks. [Control](stage-fp16-trial-r01/results.json). INT8 stages measured only 0.985–1.008×, with 0.643–0.674% relative L2. The second multi-model repeat/reset check failed; an isolated eight-check replay passed exactly. Both results are retained, and this is not INT8 acceptance. [Trial](stage-int8-trial-r01/results.json), [isolated replay](cache-int8-r01/results.json).
- Per-signature INT8 workspaces originally summed to 2,071.51 MiB. The runtime now allocates one 748.125-MiB arena shared by all six contexts, serializes host enqueue under a lock and GPU execution on one stream, and fences caller streams. It verifies every context's actual workspace requirement. This saves 1,323.39 MiB but does not eliminate the measured INT8 regression. A complete cause for the later-window slowdown has not been isolated; do not treat the memory repair as a successful speed qualification.

## Four-seed repeated generation and teeth review

**Fresh GPU runs on the RTX 4070 SUPER above, 2026-09-19:** all 24 scheduled fresh processes completed, with no excluded timing observations and identical runtime source/checkpoint/fixture hashes across roles. Every pair improved by at least 12.08%; the median paired speedup is **13.45%**. Overall medians are **10.3978 FPS reference versus 11.7830 FPS candidate**. These are twelve observations per role, not twelve independent people or utterances. [Complete observations and paired ratios](acceptance-r01/acceptance-summary.json).

| Seed | V2 median FPS (3 runs) | Candidate median FPS (3 runs) | Candidate range | Opening correlation | Jointly-open edge ratio |
| --- | ---: | ---: | --- | ---: | ---: |
| 50 | 10.3006 | 11.6971 | 11.6146–11.7852 | 0.9526 | 1.0854 |
| 51 | 10.4029 | 11.7807 | 11.6709–11.8112 | 0.9283 | 0.9907 |
| 0 | 10.4030 | 11.7512 | 11.6576–11.8127 | 0.9126 | 0.9333 |
| 1 | 10.2939 | 11.7914 | 11.7874–11.8102 | 0.9564 | 1.1019 |

The candidate's per-run chunk p95 is 2.3552–2.3747 s, versus 2.6593–2.6809 s for V2. All 108 measured windows per role exceed the 1.12-s native-playback deadline. Candidate whole-device peaks are **11,782–11,856 MiB**, versus 10,608 MiB for V2, with the same co-residents. Torch peak allocation drops to about 5,040–5,041 MiB but reserved memory rises to 8,666–8,676 MiB, and TensorRT allocations are additional. Faster execution therefore costs device headroom in this configuration; this is not a VRAM-reduction result.

The assistant visually inspected the four `mouth-comparison.png` sheets: [50](acceptance-r01/reviews/seed50/mouth-comparison.png), [51](acceptance-r01/reviews/seed51/mouth-comparison.png), [0](acceptance-r01/reviews/seed0/mouth-comparison.png), [1](acceptance-r01/reviews/seed1/mouth-comparison.png). Selected samples come from unsharpened pre-encode RGB and include baseline-selected opening strata, boundaries and fixed times. Tooth boundaries remain visible in the candidate's selected open-mouth frames; mouth shape and timing vary from the reference, and some bright connected tooth bands occur in both. No broad collapse to the prior LITE-like blur was apparent in these samples. This is a static review, not a claim that the full videos were watched or that flicker/anatomical errors are absent.

All 250 frames in every first-repeat pair had face detections. The opening/edge diagnostics stay inside the plan's triage thresholds, but they are not automatic dental or sync certification. Standard seeds/audio were already observed in earlier work and are regression evidence. Native full-frame and aligned mouth videos are retained for user review: [seed 50 full frame](acceptance-r01/reviews/seed50/comparison.mp4), [seed 50 mouth](acceptance-r01/reviews/seed50/mouth-comparison.mp4), [51](acceptance-r01/reviews/seed51/comparison.mp4), [0](acceptance-r01/reviews/seed0/comparison.mp4), [1](acceptance-r01/reviews/seed1/comparison.mp4).

## Implemented

For the complete stock PRO / LITE / FP8 V1 / V2 / latest component chart and the subsequent throughput analysis, see [pipeline precision and next targets](../../docs/research/PRO_PIPELINE_PRECISION_AND_THROUGHPUT_2026-09-19.md). That document reuses these measurements; it adds no new GPU run. The smaller table below records this experiment's three policies.

The candidate remains PRO, with the original checkpoint and four denoising steps:

| Pipeline portion | Existing accepted V2 | Faster candidate | Rejected INT8 candidate |
| --- | --- | --- | --- |
| 60 FFN linears | FP8 W8A8 | Same | Same |
| 120 self-attention Q/K/V/O linears | FP8 W8A8 | Same | Same |
| Self-attention core | INT8 Q/K, FP16 P/V | INT8 Q/K, FP8 P/V | Same as faster candidate |
| Cross-attention projections/core | BF16 / FlashAttention2 | Same | Same |
| Decoder residual blocks 4–6 and 8–10 | Compiled BF16 | TensorRT FP16 stages | 12 INT8 residual convolutions, FP16 remaining stage operations |
| Remaining decoder, final head | BF16 | Same | Same |
| Caller-owned causal caches | BF16 | BF16, explicit FP16 engine conversion | Same as faster candidate |
| Motion encoder / audio encoder | BF16 / FP32 | Same | Same |

FP16 is another 16-bit floating-point format; that part of the winning pilot is a compiler/layout improvement rather than an additional bit-width reduction. FP8 P/V attention is the new eight-bit conversion. No existing FP8 buffer is quantized again; all recipes start from original PRO weights.

- **Profiler repair:** aggregate device time/memory fields, field provenance, unavailable values rather than silent zero defaults, and explicit failure when GPU timings cannot be qualified. `--mode trace` retains policy-selected compilation; `--mode inventory --component decoder` is the separate eager module inventory.
- **Calibration repair:** the old convolution builder fits output scales over all captured inputs of each signature, requires at least two samples and rejects nonfinite outputs. Public latent captures include warmup windows 0/1 and later generation windows 4/8.
- **Pure fused-stage boundary:** `ResidualStage` groups contiguous residual blocks, exposes all causal cache tensors, preserves the original VAE's cache/reset ownership and checks installation/output contracts before mutation. CPU tests cover initial/recurrent behavior, reset, independent sessions and installation rollback.
- **TensorRT stage runtime/export:** explicit BF16 or FP16 engine interfaces with BF16 caller caches, all initial/recurrent signatures, INT8 Q/DQ, separate BF16/FP16 controls, a shared persistent workspace, stream fences and owned outputs. Plans validate checkpoint/VAE-source/engine/inspector hashes, versions, bindings, finite execution evidence and actual INT8 inputs plus weights/tactics. The initial nonfinite and floating-point-fallback builds are retained as failures.
- **Stage capture/build CLI:** bounded signature samples, streaming multi-sample input/output max calibration and build manifests. Selected groups default to residual blocks 4–6 and 8–10. This consolidates the proposed separate export/build scripts into `decoder_stages.py` subcommands.
- **Real attention inputs:** bounded post-rotary Q/K/V captures cover all 30 blocks across four steps by stratified selection, with later-window samples for blocks 0/14/29. A kernel runner compares explicit Sage FP16-PV and FP8-PV on the same saved activations.
- **Native attention build:** SageAttention 2.2.0 compiled for SM89 using isolated CUDA 12.8.93. Native SM89 extension and both explicit public kernel symbols import successfully with `torch.cuda.is_initialized() == False`. [Manifest and hashes](sage-sm89-build-manifest.json). Two earlier include-path failures are retained in build logs; the third build succeeded.
- **Per-role dependencies:** the paired sweep accepts separate reference/candidate `PYTHONPATH` values so the reference does not silently load a new experimental binary.

The current accepted V2 policy, original model weights and production model selection remain unchanged. Fresh MuseTalk orchestration is implemented and tested. Shared QKV quantization, cross-attention conversion and four-bit kernels are deferred: the [exclusive CUDA profile](profile-cost-summary.json) attributes approximately 2.0% to kernels containing `amax`, 4.3% to all remaining BF16 GEMMs and 16.4% to current FP8 GEMMs. These approximate categories include mixed operations; they are not standalone speed forecasts. Small projection changes cannot close the observed roughly 3.6× gap to native MuseTalk. Further decoder work or separately scoped model/architecture changes remain necessary.

The faster seed-50 pilot spends 21.1443 s generating 250 frames, including 10.9295 s measured decode, 8.4500 s DiT and 1.3793 s motion encode. Subtracting all measured decoder time gives an **illustrative 24.47-FPS Amdahl ceiling with other work unchanged**. This is arithmetic, not a measured attainable configuration; decoder CPU overhead and overlap are not separately removed. Both transformer and decoder need further gains for 30 FPS. The current evidence does not justify expecting blanket INT8 conversion to provide them.

## Recurrence, new speech and relative sync

**Fresh GPU validation on the same RTX 4070 SUPER, 2026-09-19**, using the preserved geometry/seed protocol. One pair per extended condition; these are regression checks, not another repeated throughput study. [Complete qualification evidence](qualification-summary.json).

| Condition | Frames / seed | V2 FPS | Candidate FPS | Opening correlation | Jointly-open edge ratio |
| --- | --- | ---: | ---: | ---: | ---: |
| Repeated known audio, 60-second recurrence | 1,500 / 50 | 10.3776 | 11.7424 | 0.9267 | 0.9831 |
| Two turns with a silence gap | 576 / 51 | 10.1459 | 11.5153 | 0.9404 | 1.1129 |
| New utterance B | 250 / 0 | 10.3893 | 11.8325 | 0.9604 | 1.0195 |
| New podcast segment, source 30–40 s | 250 / 1 | 10.4046 | 11.8179 | 0.9405 | 1.0273 |

The 60-second candidate's first/last ten chunks have medians 2.3496/2.3528 s, with p95 2.3744 s and whole-device peak 11,852 MiB. All extended pairs retain face detections in every frame. These finite-length tests show no large late-window timing increase or broad detail collapse in the inspected samples; they do not establish indefinite serving stability. The assistant inspected all four extended mouth sheets and native 59.5-s reference/candidate full frames. Long-clip sheets label which images come from retained native RGB PNGs versus decoded MP4; short new-speech sheets use raw RGB.

**Fixture correction:** the initially registered `podcast_sichuan_16k.wav` prefix was exactly the standard ten-second audio despite a different source-file hash. Its completed pair is retained as repeatability evidence and excluded from the count of new utterances. A distinct 30–40-s segment was registered before generation, verified against both earlier effective sample hashes, and tested in [the correction run](new-speech-correction-r01/schedule.json). [Audit and sample ranges](fixture-correction.json). Future `validation_suite.py` runs use `fixtures-correction.json` and reject duplicate effective speech before generation. The additional resident's allocation was observed at 2,876 MiB before the correction, versus the earlier 2,478 MiB; per-run load manifests are retained and this extra pair is not pooled into the standard timing sweep.

[Own/shared-crop SyncNet](extended-r01/sync-own.json) and the [corrected new-speech checks](new-speech-correction-r01/sync-own.json) show no best-lag worsening above one 25-FPS frame in either crop mode. **Scores are not uniformly preserved:** on the new 30–40-s segment, zero-lag cosine is 0.6735 reference versus 0.5890 candidate with own crops and 0.5949 with shared crops. The gap clip also declines in both modes. Crop sensitivity, overlapping windows and this nonstandard diagnostic prevent treating those values as calibrated audio-sync quality scores or official LSE-C/LSE-D. Their consistent direction warrants perceptual review; no unconditional quality pass or production promotion is claimed.

Review media: [60-second comparison](extended-r01/recurrence-review/comparison.mp4), [silence gap](extended-r01/two-turn-gap-review/comparison.mp4), [new utterance B](extended-r01/audio-b-review/comparison.mp4), [distinct podcast segment](new-speech-correction-r01/sichuan-30s-review/comparison.mp4). The [stock PRO → FP8 V1 → V2 → new candidate comparison](anchor-comparison-r01/stock-v1-v2-new.mp4) uses matching seed-50 image/audio hashes, 250 frames and native-sized panels. Stock/V1 media are **reused September 17/18 anchors**, not fresh timing controls; their same-GPU provenance and the CPU composition command are in the [manifest](anchor-comparison-r01/manifest.json).

## Repeat/reset validation and deployment boundary

The selected FP16 decoder with the final shared-workspace runtime passed **all eight exact GPU repeat/reset comparisons** on a fixed real recurrent window, with a different first-window input interleaved between resets. This verifies repeat/reset behavior for the captured shapes rather than claiming general cross-thread serving safety. [Replay manifest](cache-fp16-shared-r01/results.json).

Five-minute bounded serving and concurrency qualification are not run: the offline candidate misses both 25-FPS native playback and the 30-FPS target. The finite-length runner retains frames in host memory, so its long regression is not a bounded-memory transport test. The selected recipe remains opt-in, and production model selection is unchanged. Shared QKV quantization, selective cross projections, four-bit kernels, broader decoder coverage and training/distillation are **deferred**, not marked implemented. The report's measurements and remaining critical-path budget support that stopping decision for this cycle. This does not establish that every possible quantization kernel has been exhausted: especially four-bit compute and broader fused decoder coverage remain untested research work.

## CPU verification

Final focused suite: **56 CPU tests passed**, [log](cpu-tests-complete.log), after all runtime and fixture-audit repairs. Tests cover profiling field lookup, multi-sample calibration, exact causal-cache parity, failure-before-mutation behavior, BF16/FP16 export labeling without modifying source weights, INT8 computation proof, real-attention capture selection and late latents, effective-audio duplicate rejection, VAE source-hash rejection before CUDA allocation, plus existing regressions. ONNX export reports expected shape-specialization and Slice constant-folding warnings; engines accept only recorded exact signatures. New implementation modules pass Ruff. CPU tests are separate from the GPU evidence above.

```bash
PYTHONPATH=.pro-quant-deps:/workspace/experiments/ojin-components-deps:. \
  .venv/bin/python -m pytest \
  tests/test_pro_30fps.py tests/test_pro_quantization.py \
  tests/test_pro_quantization_v2.py tests/test_pro_vae_quantization.py \
  tests/test_pro_v2_repairs.py tests/test_optimizations.py -q
```

## Reproduction interfaces

Run from `/workspace/SoulX-FlashHead` with the GPU lease available. Do not bypass the lease with another lock filename. The commands below describe the implemented interfaces; named output directories already exist where linked above, so choose new names for reruns. The initial BF16 and INT8 build names retain rejected attempts; the validated builds are BF16 `r02`, FP16 `r01`, and INT8 `r02` with FP16 floating-point regions.

```bash
RUN_ROOT=benchmarks/pro_30fps_20260919
BASE_POLICY=benchmarks/pro_quantization_v2_20260918/policies/self_all_fp8_sage2_fp16.json
FIXTURES=benchmarks/pro_quantization_v2_20260918/fixtures.json
OLD_DEPS=.pro-sage2-deps:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.
NEW_DEPS=/workspace/experiments/pro30-deps/sage-sm89:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.

# 1. Fresh accepted baseline, then compiled profile with corrected GPU fields.
PYTHONPATH="$OLD_DEPS" .venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  --policy "$BASE_POLICY" --fixtures "$FIXTURES" --fixture-id indian150-a \
  --seed 50 --frames 250 --repeats 1 --output "$RUN_ROOT/reference-seed50-r01"
PYTHONPATH="$OLD_DEPS" .venv/bin/python benchmarks/pro_quantization_v2_20260918/profile.py \
  --policy "$BASE_POLICY" --fixtures "$FIXTURES" --fixture-id indian150-a \
  --seed 50 --frames 56 --component pipeline --mode trace \
  --output "$RUN_ROOT/compiled-profile-r01"

# 2. Real V2 latent/QKV diagnostic. Eager capture is not throughput evidence.
PYTHONPATH="$OLD_DEPS" .venv/bin/python benchmarks/pro_quantization_v2_20260918/capture.py \
  --policy "$BASE_POLICY" --fixtures "$FIXTURES" --fixture-id indian150-a \
  --seed 50 --frames 250 --split calibration --latents-only --attention-inputs \
  --max-raw-mib 3072 --output "$RUN_ROOT/real-inputs-r01"

# 3. All-signature stage calibration from the retained V2 latents.
PYTHONPATH="$OLD_DEPS" .venv/bin/python -m benchmarks.pro_30fps_20260919.decoder_stages capture \
  --captures "$RUN_ROOT/real-inputs-r01/manifest.json" \
  --groups 4,5,6 8,9,10 --max-latents 4 --max-raw-mib 2048 \
  --output "$RUN_ROOT/stage-calibration-r01"

# 4. Build floating-point control first. Diagnose any export/parser failure.
PYTHONPATH="$OLD_DEPS" .venv/bin/python -m benchmarks.pro_30fps_20260919.decoder_stages build \
  --calibration "$RUN_ROOT/stage-calibration-r01/results.json" --precision bf16 \
  --workspace-mib 512 --output "$RUN_ROOT/stage-bf16-build-r01"
# FP16 is a separately named floating-point control.
PYTHONPATH="$OLD_DEPS" .venv/bin/python -m benchmarks.pro_30fps_20260919.decoder_stages build \
  --calibration "$RUN_ROOT/stage-calibration-r01/results.json" --precision fp16 \
  --workspace-mib 512 --output "$RUN_ROOT/stage-fp16-build-r01"
# The BF16-boundary INT8 attempt failed precision inspection. The working
# engine build explicitly uses FP16 floating-point regions and interfaces.
PYTHONPATH="$OLD_DEPS" .venv/bin/python -m benchmarks.pro_30fps_20260919.decoder_stages build \
  --calibration "$RUN_ROOT/stage-calibration-r01/results.json" --precision int8 \
  --floating-precision fp16 --workspace-mib 512 --output "$RUN_ROOT/stage-int8-build-r02"

# 5. Native attention on identical real inputs. Also run OLD_DEPS / sage2_fp16
# in a separate directory for old-build versus native-build attribution.
PYTHONPATH="$NEW_DEPS" .venv/bin/python -m benchmarks.pro_30fps_20260919.attention_kernels \
  --captures "$RUN_ROOT/real-inputs-r01/manifest.json" --backend sage2 \
  --repeats 30 --output "$RUN_ROOT/native-attention-kernels-r01"
```

For a decoder policy, copy the accepted JSON into a new candidate, set `decoder.backend` to `trt_stage` or `trt_stage_compile`, `decoder.scheme` to `bf16`, `fp16_stage_control` or `int8_conservative`, and `decoder.plan` to the corresponding **complete and execution-verified** build. Experimental candidate: [FP8 attention plus FP16 stages](policies/combined_fp16_sage_fp8.json), with [preserved recipe and source identities](selected-candidate.json). Rejected performance candidate: [INT8 stages plus FP8 attention](policies/combined_int8_sage_fp8.json). Neither changes production selection.

To reproduce the selected candidate and its qualification schedule, use new output names:

```bash
RUN_ROOT=benchmarks/pro_30fps_20260919
NEW_DEPS=/workspace/experiments/pro30-deps/sage-sm89:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.
OLD_DEPS=.pro-sage2-deps:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.
PYTHONPATH="$NEW_DEPS" .venv/bin/python benchmarks/pro_quantization_v2_20260918/run.py \
  --policy "$RUN_ROOT/policies/combined_fp16_sage_fp8.json" \
  --fixtures "$RUN_ROOT/fixtures.json" --fixture-id indian150-a \
  --seed 50 --frames 250 --repeats 1 --save-raw --output "$RUN_ROOT/reproduction-new"
PYTHONPATH=.pro-quant-deps:/workspace/experiments/ojin-components-deps:. \
  .venv/bin/python benchmarks/pro_quantization_v2_20260918/sweep.py \
  --policy "$RUN_ROOT/policies/combined_fp16_sage_fp8.json" \
  --reference-policy benchmarks/pro_quantization_v2_20260918/policies/self_all_fp8_sage2_fp16.json \
  --fixtures "$RUN_ROOT/fixtures.json" --fixture-id indian150-a \
  --seeds 50 51 0 1 --frames 250 --repeats 3 --save-raw-first \
  --reference-pythonpath "$OLD_DEPS" --candidate-pythonpath "$NEW_DEPS" \
  --output "$RUN_ROOT/acceptance-new"
PYTHONPATH=.pro-quant-deps:/workspace/experiments/ojin-components-deps:. \
  .venv/bin/python -m benchmarks.pro_30fps_20260919.summarize_acceptance \
  --sweep "$RUN_ROOT/acceptance-new"
```

`schedule.json` retains every exact child command. `validation_suite.py --output NEW_DIRECTORY --standard-reviews REVIEW...` runs the preregistered 1,500/576/250/250-frame pairs, CPU reviews and own/shared-crop relative SyncNet. `compare_musetalk.py` in this experiment directory is the three-pair alternating orchestrator; its completed schedule records both interpreters and dependency paths. GPU lease and exact-shape engine checks remain mandatory. Do not use the rejected INT8 recipe as a faster default.
