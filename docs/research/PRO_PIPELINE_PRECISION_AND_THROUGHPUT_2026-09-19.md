# PRO and LITE: pipeline precision chart and next throughput targets

Updated **2026-09-19**. This is the consolidated component chart for the retained experiments. **The latest version is the rightmost video panel, “New FP8 + FP16 stages.”** It remains an experimental PRO recipe, not a production promotion or a formally accepted “V3.” The next priorities are decoder data movement/fusion and transformer execution efficiency; blanket INT8 conversion has already failed to improve this pipeline.

**Hardware for all local inference numbers below: NVIDIA GeForce RTX 4070 SUPER, 12 GB physical / 12,282 MiB visible VRAM**, GPU UUID `GPU-69894a01-4488-37ac-5fd8-878224044365`, driver **595.84**. PRO/LITE use Torch **2.7.1+cu128**, CUDA runtime **12.8**; the matched native MuseTalk uses Torch **2.5.1+cu121**, runtime **12.1**. Latest decoder engines use TensorRT **10.9.0.34**. SageAttention **2.2.0** was separately built for SM89 with CUDA **12.8.93**. Evidence: [build/environment manifest](../../benchmarks/pro_30fps_20260919/sage-sm89-build-manifest.json), [historical PRO/LITE report](../../benchmarks/pro_lite_150x_20260917/README.md), and [latest run report](../../benchmarks/pro_30fps_20260919/README.md).

**This document adds CPU source inspection and arithmetic on retained results, with no new GPU inference.** September 19 main runs paused the live SoulX worker and retained residents reporting approximately **2,478 + 234 MiB**, with about **2,727 MiB** idle whole-device use. Historical September 17/18 runs retain their own provenance in the linked reports. The later corrected new-speech pair had increased resident load and is not pooled into the main timing sweep. The prior experiment records server restoration; this document does not change serving configuration.

## 1. Which version is which?

| Version | Comparison panel | Exact retained identity | Status |
| --- | --- | --- | --- |
| Stock PRO | 1: “Stock PRO - Sep 17” | Original `Model_Pro` + Wan VAE | Historical BF16 eager anchor |
| Stock LITE | Separate PRO/LITE comparison | Original `Model_Lite` + LTX VAE | Different released model and latent geometry; not quantized PRO |
| PRO FP8 V1 | 2: “PRO FP8 V1 - Sep 18” | [Preserved PRO FP8 manifest](../../benchmarks/pro_quantization_20260918/PRO_FP8/manifest.json) | Preserved tested offline reference |
| PRO FP8 V2 | 3: “PRO FP8 V2 - Sep 19” | [self_all_fp8_sage2_fp16.json](../../benchmarks/pro_quantization_v2_20260918/policies/self_all_fp8_sage2_fp16.json) | Previous accepted offline recipe; September 19 reproduction of September 18 policy |
| Latest candidate | 4: “New FP8 + FP16 stages” | [combined_fp16_sage_fp8.json](../../benchmarks/pro_30fps_20260919/policies/combined_fp16_sage_fp8.json), [selection manifest](../../benchmarks/pro_30fps_20260919/selected-candidate.json) | Experimental; faster, with unresolved relative sync-score declines |

[Watch the four PRO versions](../../benchmarks/pro_30fps_20260919/anchor-comparison-r01/stock-v1-v2-new.mp4). V1/V2/latest are load-time recipes reconstructed from original PRO weights, with associated compiler/binary/engine settings. They are not successively requantized FP8 checkpoints. Preserve original weights and the earlier manifests when adding experiments.

## 2. Complete pipeline component chart

The chart describes the **measured offline configurations**, including stock eager LITE. It does not describe every optional LITE service optimization. “BF16” identifies principal weights/interfaces; it does not mean every reduction, normalization or accumulator uses BF16.

| Pipeline component | Stock PRO | Stock LITE | PRO FP8 V1 | PRO FP8 V2 | Latest: FP8 + FP16 stages |
| --- | --- | --- | --- | --- | --- |
| Visual checkpoint / decoder | PRO / Wan | LITE / LTX | Original PRO / Wan | Original PRO / Wan | Original PRO / Wan |
| Transformer depth / width | 30 blocks / 1,536 | 30 blocks / 1,536 | Same as PRO | Same as PRO | Same as PRO |
| FFN, 60 linear layers | BF16 | BF16 | **FP8 E4M3 W8A8** | **FP8 E4M3 W8A8** | **FP8 E4M3 W8A8** |
| Self-attention Q/K/V/O, 120 linear layers | BF16 | BF16 | BF16 | **FP8 E4M3 W8A8** | **FP8 E4M3 W8A8** |
| Self-attention core | BF16 interfaces; stock dispatcher¹ | BF16 interfaces; stock dispatcher¹ | FlashAttention2, BF16 interfaces | **Sage2: INT8 Q/K + FP16 P/V** | **Native SM89 Sage2: INT8 Q/K + FP8 P/V** |
| Audio cross-attention Q/O projections | BF16 | BF16 | BF16 | BF16 | BF16 |
| Audio cross-attention K/V projections | BF16 | BF16 | BF16; prepared once per window | BF16; prepared once per window | BF16; prepared once per window |
| Audio cross-attention core | BF16 interfaces; stock dispatcher¹ | BF16 interfaces; stock dispatcher¹ | FlashAttention2, BF16 interfaces | FlashAttention2, BF16 interfaces | FlashAttention2, BF16 interfaces |
| Audio projector, patch/time embeddings, modulation, DiT output head | Original floating-point operations; BF16 parameters | Original floating-point operations; BF16 parameters | Same precision families | Same precision families | Same precision families |
| Transformer norms / residual paths | Original floating-point operations, including FP32 RMSNorm arithmetic | Same precision families | Retained; not INT8/FP8 converted | Retained; not INT8/FP8 converted | Retained; not INT8/FP8 converted |
| Rotary position application | Stock FP64/complex arithmetic, cast back to input dtype | Stock FP64/complex arithmetic, cast back | Prepared real FP32 arithmetic, cast back | Prepared real FP32 arithmetic, cast back | Prepared real FP32 arithmetic, cast back |
| Wan decoder residual groups `upsamples[4:7]`, `[8:11]` | BF16 | Not applicable: LTX architecture | Compiled BF16 | Compiled BF16 | **Two fused TensorRT FP16 stages** |
| Remaining decoder, resampling, final image head | Wan BF16 with original floating-point exceptions | LTX BF16 with original floating-point exceptions | Compiled Wan BF16 | Compiled Wan BF16 | Remaining compiled Wan BF16; late blocks 12–14 and final head retained |
| Decoder causal cache storage | Wan BF16 | LTX decode path; no Wan cache/engine interface | Wan BF16 | Wan BF16 | **Wan BF16 owned by caller; explicit FP16 conversion at engine boundaries** |
| Reference-image VAE encoder | Wan BF16; preparation | LTX BF16; preparation | Wan BF16 | Wan BF16 | Wan BF16 |
| Motion-feedback VAE encoder | Wan BF16; per window | LTX BF16; per window | Wan BF16 | Wan BF16 | Wan BF16 |
| Wav2Vec audio encoder | FP32 | FP32 | FP32 | FP32 | FP32 |
| Color correction, history selection, RGB delivery | Original processing; output eventually FP32 / RGB uint8 | Same processing families | No new low-bit policy | No new low-bit policy | No new low-bit policy |
| Video encode / audio mux | Separate measured delivery stage | Same | Same | Same | Same |
| Main execution optimization | Eager benchmark | Eager benchmark | Compiled DiT + prepared conditioning + compiled Wan decode | Same, plus selected FP8/Sage kernels | Same, plus native FP8-PV kernel and external fused decoder engines |
| Denoising steps | 4 | 4 | 4 | 4 | 4 |

¹ **Stock attention provenance qualification:** the stock dispatcher prefers installed SageAttention, then FlashAttention3/2, then SDPA, subject to compatibility mode. The September 17 stock manifests do not identify the exact selected kernel. Therefore “stock BF16” is not proof of an entirely floating-point attention core. V1 explicitly selects FlashAttention2; V2/latest explicitly select their Sage core and pin cross-attention to FlashAttention2. Earlier shorthand tables calling stock unconditionally “FlashAttention2” were more specific than the retained evidence supports.

**Precision legend and implementation details:**

- **W8A8** means the matrix multiplication consumes eight-bit weights and eight-bit activations. These linears use `float8_e4m3fn`, dynamic activation scaling, FP32 scales, BF16 bias/output interfaces, and `_scaled_mm(..., use_fast_accum=False)`. This is actual low-precision GEMM, not eight-bit storage followed by a BF16 GEMM.
- **Q/K/V** are query/key/value tensors; **P** is the attention probability tensor. The projection precision and the attention-core precision are separate decisions. Quantized Sage cores still use floating-point scaling/softmax/accumulation and return the caller's floating-point dtype.
- V2 selects `sageattn_qk_int8_pv_fp16_cuda` with `pv_accum_dtype="fp16+fp32"`. Latest selects `sageattn_qk_int8_pv_fp8_cuda` with `pv_accum_dtype="fp32+fp16"`. Neither core is “everything INT8.”
- **BF16 and FP16 are both 16-bit. FP8 and INT8 are both 8-bit.** The winning decoder change gains from execution/fusion/layout choices with FP16 stages; it is not an eight-bit decoder. Changing an existing FP8 operation to INT8 does not halve its bit width again.
- Prepared real FP32 rotary, cached cross-attention K/V and compiled decoder execution already exist in V1 onward. They are not unimplemented next optimizations.
- There is no isolated “teeth layer.” Transformer output, decoding and recurrent motion feedback all affect tooth detail and articulation. Keeping the final image head in BF16 is a precaution, not a dental-quality guarantee.

Source of truth: [pipeline and dtype loading](../../flash_head/src/pipeline/flash_head_pipeline.py), [transformer/attention/rotary](../../flash_head/src/modules/flash_head_model.py), [LTX VAE](../../flash_head/ltx_video/ltx_vae.py), [FP8 linear implementation](../../soulx_rtc/pro_quantization.py), [V2 conversion](../../soulx_rtc/pro_quantization_v2.py), [explicit attention backends](../../soulx_rtc/pro_attention_backends.py), [Wan VAE](../../flash_head/wan/modules/vae.py), [fused decoder runtime](../../soulx_rtc/pro_vae_stage_backend.py).

### Rejected INT8 branch: keep separate from the latest candidate

| Property | Tested INT8 decoder branch |
| --- | --- |
| Policy | [combined_int8_sage_fp8.json](../../benchmarks/pro_30fps_20260919/policies/combined_int8_sage_fp8.json) |
| Transformer | Same FP8 linears and INT8-QK/FP8-PV core as latest |
| Decoder change | **12 actual INT8 residual convolutions** across blocks 4–6 and 8–10; remaining stage operations/interfaces FP16 |
| Remaining decoder / caller caches | BF16 |
| Measured generation | **6.7296 FPS**, separate workspaces; **7.6659 FPS**, shared workspace |
| Decision | Rejected for integrated slowdown; inspector confirms actual INT8 execution |

These were September 19 RTX 4070 SUPER pilot runs, not twelve-run medians. Shared workspace and stream fencing are already implemented; repairing memory use did not make INT8 faster. See the [execution report](../../benchmarks/pro_30fps_20260919/README.md) and [verified INT8 engine build](../../benchmarks/pro_30fps_20260919/stage-int8-build-r02/results.json).

## 3. LITE does much less visual work, even at BF16

At the retained **320×576, 33-frame neural window**:

| Model geometry | All PRO variants | Stock LITE |
| --- | ---: | ---: |
| VAE stride, temporal / spatial | 4 / 8×8 | 8 / 32×32 |
| Latent channels | 16 | 128 |
| Latent shape, channels × time × height × width | 16×9×72×40 | 128×5×18×10 |
| Transformer patch size | 1×2×2 | 1×1×1 |
| Tokens per denoising step | **6,480** | **900** |
| Configured motion history | 2 latent frames | 2 latent frames |
| Corresponding physical history / new frames per ordinary window | **5 / 28** | **9 / 24** |

Sources: [PRO config](../../models/SoulX-FlashHead-1_3B/Model_Pro/config.json), [LITE config](../../models/SoulX-FlashHead-1_3B/Model_Lite/config.json), [pipeline window construction](../../flash_head/src/pipeline/flash_head_pipeline.py). Token counts are derived from those shapes, not a profiler measurement. PRO has **7.2× as many tokens per step**, with a different decoder and temporal stride. This is not a predicted 7.2× runtime ratio. It explains why a BF16 LITE can be faster than FP8 PRO; quantizing PRO retains its larger visual workload.

## 4. Measured speed: historical anchors versus the latest paired studies

**Same RTX 4070 SUPER / 12 GB physical / 12,282 MiB visible.** All rows use the Indian-man **1.50×** reference at **320×576**. These are useful generated frames per second, not the MP4's 25-FPS playback rate. Do not treat this table as one simultaneous controlled sweep.

| Version | Useful generation FPS | Date / evidence / comparability |
| --- | ---: | --- |
| Stock PRO | **7.16** | Sep 17 historical eager seed-50 anchor, 250 frames; [PRO/LITE report](../../benchmarks/pro_lite_150x_20260917/README.md) |
| Stock LITE | **56.20** | Same historical eager comparison; different latent/decoder workload; not remeasured in the latest cycle |
| PRO FP8 V1 | **9.4082** | Sep 18 median in the V1-versus-V2 paired sweep; [V2 report](../../benchmarks/pro_quantization_v2_20260918/README.md) |
| PRO FP8 V2 | **10.3978** | Sep 19 main sweep, 12 fresh processes across seeds 50, 51, 0, 1 |
| Latest FP8 + FP16 stages | **11.7830** | Same Sep 19 sweep, 12 paired candidate processes; [all observations](../../benchmarks/pro_30fps_20260919/acceptance-r01/acceptance-summary.json) |
| Latest candidate, MuseTalk comparison | **11.8111** | Sep 19 corrected alternating study, three fresh processes, seed 50 |
| Native MuseTalk V1.5, FP16 batch 8 | **42.1885** | Same corrected study; [schedule and results](../../benchmarks/pro_30fps_20260919/matched-musetalk-r02/schedule.json) |

The paired sweep's median gain is **13.45%**; all twelve pairs improve. The corrected MuseTalk study includes audio/model/compositing work where applicable and host RGB delivery for both. It excludes model loading, avatar preparation and first warmup. Including encode/mux gives **11.6399 FPS PRO versus 39.4930 MuseTalk**. MuseTalk generates a 256×256 face crop and composites it; PRO generates the whole frame. This is a matched user-visible output comparison, not equal neural work or the fastest installed MuseTalk TensorRT service.

Candidate sampled whole-device peaks reach **11,782–11,856 MiB**, versus **10,608 MiB** for V2 under the same resident load: only about **426 MiB** remains at the highest observed peak. These include other processes and TensorRT allocations; Torch counters alone understate device use. The faster recipe does not demonstrate increased concurrency capacity.

Teeth remain visible in inspected static samples, but relative sync scores decline on the silence-gap and distinct later-podcast utterances in both own/shared crop checks. Full-video perceptual acceptance, serving stability and production promotion remain outstanding. [Qualification evidence](../../benchmarks/pro_30fps_20260919/qualification-summary.json), [corrected new-speech evidence](../../benchmarks/pro_30fps_20260919/new-speech-correction-r01/sync-own.json).

## 5. Where the latest candidate spends its time

Use the **single coherent 1,500-frame latest-candidate recurrence run**, not sums of unrelated medians. On the RTX 4070 SUPER it generated 60 seconds of content in **127.741769 s**, or **11.7424 FPS**. Source: [retained result](../../benchmarks/pro_30fps_20260919/extended-r01/recurrence-candidate/results.json), `runs[0].generation_s` and `runs[0].stage_seconds`.

Durations below divide that run by six: **250-frame-equivalent arithmetic, not a new short benchmark**. Stage times use GPU events; the remainder is wall time minus those stage totals, not a separately measured CPU stage.

| Stage | Seconds per 250-frame equivalent | Share of generation wall time | Implication |
| --- | ---: | ---: | --- |
| Wan decoder | **11.049357** | **51.90%** | Largest target, despite the two fused FP16 stages |
| Transformer / DiT | **8.474913** | **39.81%** | Must also improve substantially |
| Motion-feedback encode | 1.379389 | 6.48% | Secondary target; affects recurrent quality |
| Audio stage | 0.072047 | 0.34% | Very small throughput opportunity |
| Unattributed wall remainder | 0.314589 | 1.48% | Investigate if a trace shows meaningful host gaps |
| Total | **21.290295** | **100%** | **11.7424 FPS** |

Thirty FPS permits **8.333333 seconds per 250 frames**, requiring about **2.55× overall acceleration** from this long run. Matching the retained native MuseTalk median requires about **3.59×**, with a fresh matched comparison required for any eventual claim. Holding all work outside DiT/decode fixed:

```text
seconds_per_250 = 8.474913 / DiT_speedup + 11.049357 / decoder_speedup + 1.766025
useful_FPS = 250 / seconds_per_250
```

| Hypothetical DiT speedup | Hypothetical decoder speedup | Resulting useful FPS |
| ---: | ---: | ---: |
| 1× | 2× | 15.86 |
| 2× | 2× | 21.69 |
| 2× | 3× | 25.81 |
| 3× | 3× | **30.21** |
| 4× | 4× | **37.61** |

These are budget scenarios, **not speed forecasts**. Even removing all measured decoder time gives only **24.41 FPS** with everything else unchanged; removing all DiT time gives **19.51 FPS**. Equal DiT/decoder speedups need about **2.97×** for 30 FPS and **4.69×** to match 42.19 FPS. An isolated small projection conversion cannot close this gap. The earlier report's 24.47-FPS decoder-removal illustration used a different short pilot; the 24.41 figure here consistently uses the long run.

## 6. Ranked next experiments

### First: profile the exact latest recipe and isolate the quality change

Instrumentation follow-up: [pipeline latency logging](PIPELINE_LATENCY_LOGGING_2026-09-19.md) implements opt-in stage logging that preserves selected compilation, plus a separate eager module diagnostic and explicit TensorRT stream/cast timings. CPU validation is recorded there; a new full-model GPU trace remains to be run.

The retained [compiled trace](../../benchmarks/pro_30fps_20260919/compiled-profile-r01/profile.json) and [CUDA cost summary](../../benchmarks/pro_30fps_20260919/profile-cost-summary.json) describe the **V2 reference**, not the latest combined candidate. They identify hypotheses but cannot be relabeled as the latest kernel breakdown. The current candidate stage breakdown above is measured; its fine-grained operator attribution remains to be captured.

1. Run the existing profiler with the exact selected policy, native Sage dependency path and retained FP16 engine plan. Record both initial and later recurrent windows, actual kernel symbols, graph breaks, casts/copies, TRT boundaries and allocation behavior. Keep compilation enabled; the separate eager inventory is not a timing substitute.
2. Attribute **exclusive CUDA kernel time** separately from CPU operators. Do not sum nested CPU events with CUDA kernel time. Keep profiler runs separate from clean throughput runs.
3. Compare four isolated configurations on the two problematic utterances: V2; V2 + FP8 P/V only; V2 + FP16 stages only; both together. Preserve effective audio sample hashes, seed and source snapshots. Review native mouth videos plus own/shared-crop sync diagnostics. This identifies which change causes the score decline before combining another approximation.

Exit: a latest-policy cost table tied to its exact artifacts, and an attributed quality regression or an explicitly unresolved one. Do not assume the fastest combination has passed quality because static teeth are sharp.

### Priority 1: eliminate decoder copying and layout churn

**Concrete first code target:** [WanVAE_.decode](../../flash_head/wan/modules/vae.py) repeatedly concatenates the growing decoded output after each latent slice. Test collecting the nine sequential outputs and concatenating once at the end. Preserve every decoder call, cache update/reset, frame order and final output shape. Check whether compiled execution actually eliminates copies and whether holding all slices raises peak memory. This changes tensor assembly, not the model's precision.

The V2 trace spends **417.05 ms / 7.80% of exclusive CUDA time** in explicit cuDNN layout conversion. A `triton_poi_fused_cat_132` kernel accounts for another **255.28 ms / 4.77%**, but its attribution to that source-level output concatenation is **not proven**. Map compiled kernels to source before claiming saved time; the “other” category also contains padding, cloning and indexing work. These percentages are from the reference trace and cannot be added to the latest wall budget as a promised gain.

Test a **coherent compiled channels-last-3D path**, tracking layouts through convolution, residual operations, resampling and caches. Earlier V1 work tested channels-last weights and decoder compilation separately; its selected `mode="compiled"` does not also apply the channels-last conversion. Avoid merely repeating the old weight-only experiment. PyTorch documents that incompatible intermediate operations can force layout permutations; changing convolution weights alone does not ensure an end-to-end layout win. [PyTorch 2.7 Conv3d memory-format documentation](https://docs.pytorch.org/docs/2.7/generated/torch.nn.utils.convert_conv3d_weight_memory_format.html).

Also measure the current engine boundary: two stages × nine latent slices means **18 stage invocations per standard PRO decode**. Each recurrent stage passes an activation and six causal cache tensors through explicit FP16/BF16 conversions and owned outputs. Optimize allocation/copies only after profiling confirms the cost. Stable buffers need per-session ownership or appropriate buffering: reusing an output while a caller/cache still owns it corrupts recurrence. Keep BF16 caller caches in the first experiment.

Targets: [Wan decoder](../../flash_head/wan/modules/vae.py), [decoder optimization modes](../../soulx_rtc/pro_quantization.py), [stage runtime](../../soulx_rtc/pro_vae_stage_backend.py). Promote a tensor-assembly/layout change only after fixed-latent equivalence checks, recurrent checks, lower whole-decoder time and a reproducible full-pipeline gain.

### Priority 2: enlarge useful decoder fusion regions before retrying INT8

Only residual groups **4–6 and 8–10** currently use fused engines. The decoder input/middle, residual groups **0–2**, resampling at **3/7/11**, late groups **12–14** and head remain outside these engines. Let the latest profile determine which remaining region is expensive; adjacency and conversion overhead matter as much as isolated convolution FLOPs.

1. Start with a larger contiguous **floating-point** region around a measured expensive stage. Establish a finite BF16/FP16 control and whole-decoder gain before introducing INT8. Compare with the current compiled BF16 remainder, not only an eager control.
2. Extending across temporal resampling requires a new exported state contract: existing adapters support residual groups, while resampling includes initial/recurrent behavior and the `"Rep"` cache sentinel. This is not a policy-JSON-only change. Preserve the nine-slice causal order and test first slice, subsequent slices, full decode reset, back-to-back clips and independent sessions.
3. Prefer a bounded larger per-slice region initially. An unrolled whole-window engine could increase live activations beyond the roughly 426-MiB observed headroom. Keep the final head/late high-resolution path in BF16 for the first expansion; test any later conversion independently with native teeth crops.
4. Only after a fused floating-point control wins, test INT8 inside the same boundary. Require engine-inspector evidence of actual INT8 weights **and** activations/tactics, finite outputs/caches, multi-window calibration, and an integrated speed win. Reject conversion/reformat costs that erase the convolution gain.

Already completed: shared workspace, dedicated stream/caller fences, BF16 export correction and a sufficient Dynamo specialization limit. Repeating these fixes is not a new optimization. The selected FP16 engines share about **709.10 MiB** of execution workspace; the INT8 branch shares **748.125 MiB**. Lower-precision arithmetic did not imply lower execution memory.

### Priority 3: make existing FP8 transformer work cheaper

The latest DiT still consumes **39.81%** of generation wall time. Its largest linears are already eight-bit, so focus on complete operation latency: GEMM, quantization, normalization, activation, layout and output conversion together.

- Capture the actual latest activations and compare full FFNs at GEMM shapes **M=6,480, K=1,536, N=8,960**, then **M=6,480, K=8,960, N=1,536**. Include dynamic activation quantization and GELU in timing; a bare GEMM speedup is insufficient.
- Profile fusion opportunities around GELU/quantization and Q/K/V input quantization. Shared quantization or packed QKV must preserve the separate projection scaling semantics; do not silently replace three weight scales with one. Verify numerical error and full-block timing before another video sweep.
- Inspect expensive norms, layout and graph breaks exposed by the new trace. **Prepared real FP32 rotary and per-window conditioning already exist.** Do not claim their implementation as a future gain.
- Consider actual W4A4/W4A8 computation only after a compatible kernel demonstrably beats existing FP8 on these shapes including packing/dequantization/scaling. A smaller weight file or a successful four-bit load is not evidence of faster generation. No untested INT4 speedup is included in the budget.

The old V2 CUDA trace assigns **16.36%** to FP8 GEMMs, **4.33%** to all remaining BF16 GEMMs and **2.01%** to mixed kernels containing activation reductions. These are neither latest wall fractions nor exact module-family costs. They argue against prioritizing blanket cross-attention conversion or isolated `amax` removal over larger work. The original compiled INT8 FFN trial took about **6.60–6.86 ms**, versus **3.38–3.43 ms** for compiled FP8. A different INT8 kernel would need new evidence. [Retained kernel results and controls](../../benchmarks/pro_quantization_20260918/README.md).

### Secondary targets and escalation

**Motion encoder:** test compilation/layout after the larger targets. It is 6.48% of the latest wall budget; even eliminating it entirely only raises the illustrative long-run throughput to about **12.56 FPS**. Validate generated history, interruptions and cache ownership, since errors feed future windows.

**Audio/reference preparation:** Wav2Vec remains FP32, but the whole measured audio stage is only 0.34%. Reference-image encoding is primarily preparation. Neither is a credible first target for closing a 2.55× sustained-throughput gap.

**CUDA graphs/concurrency:** investigate graphs only if the latest trace shows meaningful enqueue gaps and shape/address stability can be maintained. TensorRT requires stable captured context/buffer state; graphs reduce launch overhead, not the arithmetic cost of convolutions. Extra streams can also increase memory. With current device headroom, neither graphs nor additional concurrent batches should be assumed to produce 30 FPS per stream. [TensorRT 10.x performance optimization guidance](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/performance/optimization.html).

**If execution improvements cannot meet the measured budget:** separately scope fewer denoising steps with suitable distillation, a more efficient decoder, or lower-token visual architecture trained to retain PRO detail. Simply changing the four-step recipe or using LITE is a different quality/workload experiment. The 6,480-versus-900 token gap explains why architecture changes may ultimately matter more than another format conversion. No such training or architecture speedup is established here.

## 7. Measurement and acceptance instructions for the next cycle

1. Preserve V1, accepted V2 and the latest experimental selection. Add separate named policies and output directories; derive quantized buffers from original weights. Record source, checkpoint, engine and input hashes plus exact attention symbols and decoder bindings.
2. Keep the **Indian-man 1.50×, 320×576, four steps, shift 5, two motion latents, strength 1.0** protocol. Use seeds **50, 51, 0, 1**. PRO retains five physical history frames and emits 28 new frames per ordinary window. Keep playback timestamps at 25 FPS.
3. Use fixed real latents for decoder diagnostics, all first/recurrent signatures, finite checks on every output/cache, repeat/reset and independent-session tests. Follow survivors with full videos, because a fixed-latent decoder win does not measure feedback effects.
4. Separate profiled runs from clean timing. Repeat the established paired sweep with three fresh processes per seed/role, report every observation and exclusion, and preserve equivalent resident load. Report load/warmup separately from useful generation, encode/mux, first-window latency and later-window p95. Record whole-device VRAM as well as Torch counters.
5. Use **≥5% repeatable full-pipeline gain** as the practical promotion gate for a new recipe. A **≥10% whole-decoder diagnostic gain** is a useful screening target for larger decoder work, not proof of full-pipeline acceptance. Small exact-copy removals may be retained separately if measured and useful. Any quality or cache regression overrides a speed-only pass.
6. Repeat four-seed raw mouth/full-frame reviews, the 60-second recurrence, silence gap and **two distinct effective new-speech waveforms**, including the failing later-podcast sample. Inspect native-resolution moving mouths for timing, tooth merging and flicker; do not accept edge energy as dental correctness or the relative sync diagnostic as an official calibrated score.
7. Rerun the corrected matched MuseTalk protocol only after a candidate qualifies. Thirty useful FPS and beating MuseTalk are separate goals. Ordinary 28-frame PRO windows need **≤0.9333 s** for 30-FPS generation headroom, versus the latest observed p95 around **2.37 s**. An aggregate batched FPS number does not establish this per-stream latency.

Recommended next concrete work: **latest-policy trace + quality ablation, then one-concatenation decoder assembly and compiled layout experiments, followed by a larger floating-point decoder region and measured full-FFN FP8 optimization.** The measurements will determine whether further INT8/INT4 work is justified.

## 8. Keeping this chart current

Update this document when a retained recipe changes: identify its status and manifest first, change only the component rows its actual kernels/interfaces change, and add a separately dated performance result with hardware/workload provenance. Keep historical comparisons labeled historical. Update the latest timing budget from one coherent run; never substitute a reference-policy trace for candidate attribution. Link the new run report and preserve rejected branches as evidence.
