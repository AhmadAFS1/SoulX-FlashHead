# PRO320 Apple pipeline and transformer: further optimizations, ranked (2026-11-03)

This plan came from an independent multi-agent search with skeptic verification of every candidate. It was run from the cloud session recorded in `PRO_OPTIMIZATION_SESSION_HANDOFF.md`. **No new GPU or Apple-device inference was run.** The checks were static arithmetic, repository reading and tiny CPU-only synthetic tests on an Intel Xeon container with no GPU. Apple M5 Pro and iPhone 12 Pro (A14) figures are reused historical records, labelled in each section. Time savings are estimates until measured on device.

Implemented since: rank 1's residency switch, as opt-in `--keep-models-resident` (commit `ffa5c2e`; not compiled in the cloud). It keeps the DiT partitions, decoder, motion encoder and conditioning loaded. The audio lookahead worker still reloads its speech model per context. That worker runs off the critical path (about 8 µs wait at consumption on the M5 Pro record), so its reloads cost CPU time but not window time.

---


**Evidence labels.**
- **No new runs.** This plan has no new GPU or Apple-device inference. It rests on static arithmetic, repository reading and tiny synthetic checks.
- **Where the synthetic checks ran.** CPU-only: Intel Xeon @ 2.80 GHz, 4 vCPU, no GPU and no VRAM (nvidia-smi absent), no Metal or Core ML runtime. PyTorch 2.7.0+cpu, coremltools 9.0, 1 thread, 2026-11-03. Renders were running on all four CPUs at the same time (load about 4).
- **Reused Apple M5 Pro data.** 20-core GPU, 24 GiB unified memory, no dedicated VRAM, Metal working-set budget 19,069,665,280 B. macOS 27.0.1 (26A434), 2026-11-02, Core ML `.all`. Sources: `apple/validation/pro320_runtime_optimization.json` and `apple/validation/dit_conv1x1_probe_mac.json`.
- **Reused iPhone 12 Pro (A14) data.** iPhone13,3 with 5,994,840,064 B shared memory, iOS 26.2 (23C55), older fec3bf7 runtime. Source: `apple/validation/iphone12pro_pro320_two_windows.json`.
- **Other figures.** RTX 4070 SUPER figures are historical results. Time savings are estimates, not measurements: linear scaling at 3.90 TMAC/s on the M5 Pro, or bandwidth models. The ANE 11 TOPS and A14 34 GB/s figures are external or theoretical. The warm baseline window is 1.836 s for 28 frames, which is 15.24 FPS.

| Rank | Candidate | Corrected gain (pro320) | Exactness | Interface impact | Decisive gate |
|---|---|---|---|---|---|
| 1 | Keep DiT partitions resident (M5 Pro) | 0 MAC. About 0.06–0.12 s/window near-certain (3–6%); upper bound about 0.6 s | Bit-exact | Swift only | Mac |
| 2 | FP16 self-attention SDPA on GPU | 0 MAC. 0 to about 0.28 s/window on M5 Pro (0–15%); at most 0.5% on A14 | Approximate | Exporter only | Mac |
| 3 | FFN on the A14 Neural Engine | 0 MAC. 0 to about 10 s of a 31–34 s A14 window (up to 1.5x), plausibly 3–9 s | Approximate (device kernels) | Exporter first; partition and Swift only if forced | iPhone |
| 4 | Drop dead motion rows in block 29 | 18.64 G (0.32%) with m=1; 37.27 G (0.65%) with m=2; about 4.8 / 9.6 ms on M5 Pro | Exact in real arithmetic | m=1: exporter only; m=2: multifunction model plus Swift | CPU here (exactness), then Mac (timing) |
| 5 | Cross K/V once per window | 40.8 G (0.71%); about 10 ms on M5 Pro; net negative if loaded every window | Exact | Every partition, Swift, manifest | Mac / iPhone |

## A. Exactness can be validated now on CPU here (the device decides adoption)

CPU constraint: renders hold all four CPUs. Queue any new renders behind them with 1 thread, and never load the checkpoint.

### Rank 4: Skip the dead motion rows in the last block
- **Mechanism.**
  - restoreMotion overwrites the motion latents before step 1 and before decode, and the noise draws do not depend on values. In block 29, rows [0, 220m) are therefore needed only as self-attention K/V.
  - Skip their Q/RoPE, their score and P·V rows, O plus residual, LN3 plus cross-attention, and LN2 plus the FFN. Cross-attention is block-diagonal, so the audio K/V/V̂ of frames 0..m-1 also go.
  - Emit `concat(x_in[:220m], x_live)` so the output stays [1,1980,1536]. The head is unchanged, and blocks 0–28 cannot be trimmed.
- **Savings.**
  - m=2, per step: FFN 12.11 + self Q+O 2.08 + self scores/PV 2.68 + cross Q 1.04 + P·V̂ 0.26 + QK 0.02 + audio K/V/V̂ 0.45 = 18.64 G.
  - m=2 per window (window 2 onward only): 37.27 G, 0.646% of 5.768 T or 0.666% of the post-fold 5.592 T.
  - m=1 per window (valid every window): 18.64 G, 0.32%.
  - If ffnk2/ffnk3 ships, subtract 1.51 G, giving 35.8 / 17.9 G.
  - The head saves nothing. Weight bytes are unchanged. The FFN activation shrinks by 7.9 MB per step at m=2.
  - Time: about 4.8 / 9.6 ms on M5 Pro, roughly 6% of one ~74 ms blocks_09 call.
- **Exactness.**
  - Exact in real arithmetic. A synthetic FP32 check through the real `dit.DiTBlock` gave max |diff| = 0.0 in all 8 combinations of fused/unfused, fold/no-fold and m=1/2.
  - Bitwise on device only if GEMM tiling at M=1760/1540 matches M=1980.
  - A historical CUDA note (RTX 4070 SUPER, PRO576) recorded "skip discarded tokens" as DEAD at −1.4 ms. That was a small gain, not a quality failure.
- **Interface, m=1 (ship this first).**
  - Exporter-only change to `blocks_09`; shapes stay the same.
  - With fused QKV, split Q (live rows only) from KV (all rows).
  - Add a new variant_transforms record, because validate_dit raises on unknown transforms.
  - Any comparison against a baseline bundle must mask rows [0, 220m) and the motion flow frames.
- **Interface, m=2.**
  - Export `blocks_09` as a multifunction model with `first` and `steady` functions sharing deduplicated weights.
  - Swift picks the function from motionTimes in `model(name)`, `SoulXBlockLoadRequest` and the cache key.
  - Running `steady` in window 1 silently corrupts hidden latent frame 1, which feeds the decoder's causal memory.
  - On A14, the first `steady` specialization lands in window 2 (reused record: about 61 s cold first pass vs about 11 s warm).
- **Decisive validation.**
  - CPU now:
    - Add a `deadq` row slice at index 29 in `apple/flashhead_coreml/cpu_runtime.py` (`block_forward`).
    - Render in fp32 and coreml-int8 modes, both avatars, at least 4 windows covering window 1, windows 2+ and a partial final window.
    - Require `frames.npy` and `latents.npy` to be SHA-256 identical to the baseline (`compare_frames.py`, `quality.py` PSNR inf).
    - If MKL row blocking adds ULP noise, fall back to teacher-forced (`--force-motion`) latent NRMSE far below the INT8 floor.
  - Linux conversion test in `apple/tests/test_export_variants.py`: Nq≠Nk SDPA and sliced RoPE convert, and multifunction weight.bin is about 1x.
  - Adoption needs the M5 Pro: validate_dit, per-call blocks_09 time from `run_mac_smoke.sh --model-timings`, and compare_frames pixel equality including window 1 and the history rollover. Then an A14 smoke run.

## B. Needs a Mac or iPhone (the decisive step is on device)

### Rank 1: Keep all DiT partitions (and decoder/encoder/conditioning) loaded on the M5 Pro
- **Mechanism.**
  - Today's unloads:
    - `SoulXEngine.swift:297` unloads every partition after its prediction, so each warm window does 20 loads of about 140 MB.
    - The decoder (`:256`), encoder (`:262`) and conditioning (`:363`) are also unloaded every window.
    - `SoulXAudioLookahead.swift:33` reloads the roughly 360 MiB audio model for every context.
  - The change: on macOS, gated by os_proc_available_memory and footprint, load these once per session.
- **Savings.**
  - No MAC change. Each window stops re-mapping 2.806 GB (2.61 GiB) of page-cache-mapped package bytes.
  - Near-certain floor of about 0.06–0.12 s per window (3–6%):
    - the two synchronous `blocks_00` loads, about 0.05–0.08 s;
    - the decoder load, at most about 26 ms;
    - the encoder load, at most about 14 ms.
  - Speculative remainder:
    - The in-pipeline call costs 73.9 ms against 47.06 ms in the resident probe. If that gap is load-related, 20 × 26.84 ms = 0.537 s.
    - That gives a whole-window bound of about 0.6 s, or about 21.6–22.6 FPS.
    - The gap's cause is unknown. If the report's "prefix pinning showed no consistent advantage" was a genuine pin, the per-call penalty is only a few ms.
  - Resident cost: about 1.31 GiB INT8 (up to about 2.6 GiB if FP16 copies are kept), against the 19.07 GB budget.
- **Exactness.**
  - Bit-exact: the compiled models, inputs and compute units are the same. Prior runs matched 237/237 frames across different load schedules.
  - outputBackings saves only about 1–2 ms.
  - `.fastPrediction` may change kernels, so test it as a separate arm.
- **Interface.** Swift only: a residency policy in `SoulXModelStore`, and skipping the unload in `runBlocks`. A14 cannot hold all ten (the guards are 1.25–2 GiB); the A14 analogue is to prefetch the next pass's `blocks_00` during `blocks_09`.
- **Decisive validation.** On the M5 Pro:
  - Compare `apple/run_mac_smoke.sh --model-timings --prefetch-blocks --prefetch-audio` with and without a residency flag, in A-B-B-A order on warm windows 2–4. Pass 2 of window 1 does not isolate the effect, because it mixes in first-ever compilation.
  - `compare_frames.py` must report 100% identical frames.
  - Record per-call predictionSeconds, the `blocks_00` load time and peak footprint.
  - Nothing to emulate on CPU.

### Rank 2: Run the self-attention SDPA in FP16 on the GPU
- **Mechanism.**
  - In `export_dit.convert_component`, let `fp16_safe` leave `scaled_dot_product_attention` in FP16 for the SelfAttention scope only. A tiny conversion went from 22 to 18 casts per block.
  - LayerNorm, RMSNorm and CrossAttentionWeights stay FP32.
  - q/k/v and the output are already FP16 today, so only the logits, softmax and P·V internals change.
- **Savings.**
  - No MAC change: 722.6 GMAC per window (12.04 G per block-step) leaves FP32.
  - Bytes, as unfused upper bounds: 4 casts × 18.25 MB = 73.0 MB per block-step (4.38 GB per window), plus about 1.46 GB per window from halving operands.
  - M5 Pro: 0 to about 0.28 s per window. The high end needs the FP16 SDPA to reach the fast matmul path (12.26 TFLOPS effective in the probe; unverified). If it runs on ALUs where FP16 and FP32 rates are equal, the gain is only traffic, about 15 ms (under 1%).
  - A14: at most 0.13–0.17 s (at most 0.5%). Its value there is as a prerequisite for ANE attention.
- **Exactness and headroom.**
  - Approximate, not bit-exact.
  - The worst shipped partition sits at 4.68% NRMSE against the 5% gate, so any added independent error must stay at or below about 1.8%.
  - Supporting evidence: on the M5 GPU, `blocks_00` measured 0.434% and 0.554% NRMSE. Core ML CPU_ONLY gave 0.75% and 1.59%, and the emulation 0.438% and 0.557%. So the §4 rejection of FP16 SDPA, which was measured with CPU_ONLY, may be a CPU-kernel artifact.
- **Interface.**
  - Exporter only, plus a `--compute-units` option for `validate_dit.py` (CPU_ONLY is hard-coded today).
  - Under `.all`, an FP16 SDPA becomes ANE-eligible and could split the partition between GPU and ANE. MLComputeUnits has no GPU-only setting.
- **Decisive validation.**
  - M5 Pro:
    - Measure all 10 partitions of the shipped mixed bundle under CPU_AND_GPU and ALL at production shapes; this is the GPU baseline.
    - Then the candidate. Gate: at most 5% and no worse than the GPU baseline.
    - Check that MLComputePlan stays on the GPU, take a Metal System Trace of SDPA kernel time, and finish with a matched `run_mac_smoke` scored by `quality.py`.
  - Optional CPU pre-gate: a "materialized" mode in cpu_runtime that rounds logits and P to FP16 on `blocks_06`, logging max |logit|. A "fused" mode equals today's emulation by construction and tells nothing.
  - A14: repeat the component gate on its GPU before enabling it there.

### Rank 3: Move the FFN (already FP16) to the A14 Neural Engine
- **Mechanism.** The FFN is 54.5 G per block-step (56.7%) and is already FP16 under the mixed policy. Proceed in stages:
  - (0) MLComputePlan of the current partition on the A14 under `.all` and `.cpuAndNeuralEngine`; A14 placement has never been recorded.
  - (1) Re-export the partition with a (B,C,1,S) layout and 1x1-conv FFN, INT8 per-channel unchanged, and read the A14 plan. Core ML may split GPU/ANE on its own.
  - (2) Only if the FFN stays on the GPU: explicit FFN sub-models (the attention model outputs f_in; the FFN model returns x + FFN(f_in)·e5). That is about 120 calls per window.
- **Savings.**
  - FFN work is 3.27 TMAC (6.54 TFLOP) per window.
  - Today it takes at most about 13 s of the roughly 23 s of partition intervals. That is its MAC share at about 0.50 TFLOPS effective, loads included, so it is an upper estimate.
  - At an assumed 2–5 TFLOPS on the ANE it would take 1.3–3.3 s, plus 0.1–0.5 s of call and transfer overhead.
  - Net: 0 to about 10 s of a 31.2–34.4 s warm window. A plausible 3–9 s (1.1–1.4x) needs resident FFN sub-models: about 825 MB INT8, plus about 566 MB of attention weights, on a 5.99 GB device.
  - The gain is zero if Core ML already places the FFN on the ANE, or if the sub-models reload every pass. No gain is expected on the M5 Pro (GPU preferred for every op; conv 56.2 ms vs linear 47.1 ms).
- **Exactness.** The layout rewrite is exact; a tiny check was bit-identical in FP32 and after FP16 rounding. ANE accumulation, GELU and denormal handling are approximate and undocumented.
- **Interface.** Stage 1 is exporter only, adding transposes at the layout boundary. Stage 2 changes the partition interface and Swift (two compute-unit families, prefetch and headroom rules).
- **Decisive validation.**
  - iPhone 12 Pro: the stage 0 and 1 compute plans, then the ANE latency of one resident FFN sub-model against its GPU share. Then a matched warm run with p50/p95, thermal state, footprint and saved frames scored by `quality.py`.
  - CPU pre-gate: FP16 rounding of FFN partial sums every 32/64/128 terms in `cpu_runtime.ffn_forward`, teacher-forced and closed-loop, compared with coreml-int8 and fp32.

### Rank 5: Compute the cross-attention audio K/V once per window
- **Mechanism.**
  - K = RMSNorm(cW_kᵀ+b_k) and V = cW_vᵀ+b_v depend only on the window context, which Swift passes unchanged to both passes.
  - Hoist the pre-norm FP16 k/v for all 30 blocks. Partitions take `kv` [3,2,9,32,1536] instead of `context` and drop W_k/W_v.
  - This is already in report §7 priority 3 and already done on the CUDA path (`soulx_rtc/engine.py`). It is new only as an Apple design.
- **Savings.**
  - Compute: 30 × 2 × 288 × 1536² = 40.8 G (0.707%, 0.729% post-fold). About 10.4 ms on M5 Pro; about 0.2–0.4 s on A14 (extrapolated).
  - Partition weights shrink 10.2% (−14.16 MB each), so partition loads drop 283 MB per window.
  - Offsetting costs:
    - The kv input adds 4.42 MB per call, +88.5 MB per window.
    - Live arrays take 53 MB, or 106 MB with lookahead.
    - Total cross-K/V traffic stays about the same, roughly 301 MB per window.
  - Loading a separate 141.6 MB cross_kv model every window costs about 39 ms, more than the ~10 ms saved. It pays only in one of three forms:
    - the model stays resident (+141.6 MB);
    - step-0 partitions emit kv as an extra output for step 1, using a multifunction model with shared weights;
    - the hoisted K/V comes from a model that already runs once per window.
  - Do not hoist the folded V̂ (10.6 MB of input per block).
- **Exactness.** Exact. Bitwise requires the same kernel and compute unit for the hoisted linear, so pin it; ANE placement breaks bitwise identity. Normalized K is already FP16-rounded today, so passing either pre- or post-norm FP16 is exact.
- **Interface.** Every partition and every variant (six-block, fused, ffn*), the manifest and bundle.py, the Swift runtime, and memory guards (+53–106 MB). Old bundles become incompatible. About 0.3 G overlaps Rank 4 at m=2.
- **Decisive validation.**
  - M5 Pro: A-B-B-A `run_mac_smoke --model-timings` with one of the three forms above, compare_frames pixel equality, and MLComputePlan placement.
  - A14: footprint against the 1.25 GiB guard, headroom, and the warm window.
  - The CPU test proves only the plumbing; hash-identical frames are close to a tautology. Gate this behind report §7 priority 2 profiling.

## C. Needs training
None. All five survivors are training-free: exact rewrites, or precision and placement changes.

## Merged duplicate
- **Exact hoisting pair (apple lens 4).** Folded into Ranks 4 and 5. Its 5–6 ms estimate used an unrecorded 12.3 TFLOPS rate. At the recorded 3.90 TMAC/s the combined gain is about 15–20 ms per window (0.8–1.1%).

## Rejected
- **Cross-attention query rewrite (RMSNorm on scores, gains folded into keys, b_v into the bias).**
  - Exact, but 0 net MACs: the r-multiply replaces the 1/√d multiply, and b_v is already fused into the kv bias.
  - The realistic traffic saving is about 1.1–1.8 GB per window: under 0.5% on M5 Pro and under 0.6% on A14, below noise.
  - The optional self-attention g-fold is hazardous: one real norm_q gain is −0.0072, so 1/g² ≈ 1.95e4.
- **Per-timestep modulation and gate fold into layer_norm and INT8 scales (unsound).**
  - 801 e2 and 240 e5 channels change sign between t=1000 and t=625, in every block, so weight deduplication fails (+483.7 MB).
  - Folded scales go FP16-subnormal on up to 181 of 1536 W_o channels.
  - The ceiling is at most 1.2% on M5 Pro and 0.5% on A14, likely about 0 once fused, against about +50 s on the A14 cold first window.
- **Exact bundle with block-0 reassociation through the 128-d patch.**
  - Parts (a) and (b) are Ranks 5 and 4.
  - Part (c) saves 26.3 G (0.46%), but it is not bit-exact on device, stores folded matrices in FP16 (11.1 vs 9.4 MB), and needs a 129-wide value SDPA that is likely off the fast path.
  - The total ceiling is 1.2–1.5%, at the noise floor. The patch and head merges are worth it only as free riders in another re-export.
- **Cross-Q fold into the audio keys with an RMS estimator.**
  - The real W_q·diag(γ) spectrum is flat: rank 512 leaves 12–19% of the energy and rank 960 still 1.7–2.9%. Hitting the 3e-4 to 1e-3 temperature tolerance therefore needs a rank near the 960 break-even.
  - Realistic yield is 0–0.5%, and it is approximate. Rank 5 beats it exactly.
- **FP16 SDPA plus RMSNorm on the GPU only (unsound).**
  - Its emulation evidence is circular: the emulation already accumulates in FP32 on FP16 inputs.
  - Core ML cannot pin work to the GPU alone: CPU_AND_GPU allows the CPU FP16 kernel that already failed, and `.all` makes it ANE-eligible.
  - Bytes are overstated (12.5 GB, not 15), FP16 RMSNorm risks y² overflow, and a 1.1x gate can exceed 5%.
  - Its SDPA-only core survives as Rank 2.
- **Frame-local self-attention heads and motion-frame cross-attention removal.**
  - Heads must be local at both timesteps under a 0.25x-INT8 gate (under about 0.5% off-frame mass), so at most about 5% of heads qualify: at most about 32 G (0.6%).
  - Motion cross-attention removal (53 G, 0.92%) likely fails, given the historical RTX 4070 SUPER result where cutting history from 5 to 1 broke the lips.
  - Splitting the SDPA may cost more than it saves. At most, run a CPU off-frame-mass check as a cheap kill test.