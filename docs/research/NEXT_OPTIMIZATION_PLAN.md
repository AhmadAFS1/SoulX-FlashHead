# Next optimization plan: useful throughput and smooth concurrent SoulX calls

Date: 2026-09-07. Based on source audit at `ac2c2bbb7308f87880c2ae4b01563d1e4d203c02`.

**Not all possible improvements have been implemented.** The first optimization round is delivered; the next round below is proposed work. No speedup in this plan is a promise. The immediate objective is a better quality-preserving single-call pipeline and a defensible concurrency envelope, not an unsupported ten-speaker claim.

[Architecture](../architecture/README.md) · [MuseTalk comparison](MUSETALK_OPTIMIZATION_COMPARISON.md) · [Evidence validation](EVIDENCE_VALIDATION_2026-09-07.md) · [Prior implementation status](IMPLEMENTATION_STATUS.md)

## 1. Starting point and hard capacity arithmetic

Historical native 480×832 / four-step / 25-FPS output:

- Matched optimized Torch A/B: 26.07→27.51 useful FPS, approximately 5.53%.
- Ten-job optimized Torch: 27.60 aggregate FPS.
- Ten-job FFN+VAE TensorRT: 28.57 aggregate FPS, median 87.51 seconds for 2,500 useful frames.
- True 9:16 at 576×1024 with historical staged offload: 10.88 FPS.
- Ten active RTC speakers: connected and transported, but 9,153 underrun slots. Not a capacity pass.

The Torch/TRT ten-job difference is descriptive, not a tightly matched causal A/B. These results precede some final service follow-ups. Rebaseline before claiming a new gain.

| Target | Minimum useful generation rate | Relative to 28.57 FPS |
| --- | ---: | ---: |
| 25% throughput improvement | 35.71 FPS | 1.25× |
| Two simultaneous speakers at 25 FPS | 50 FPS | 1.75× |
| Ten simultaneous speakers at 25 FPS | 250 FPS | 8.75× |
| Ten speakers plus 20% extra capacity | 300 FPS | 10.50× |

Here “20% extra capacity” means demand ×1.2, equivalent to 83.3% utilization at demand. A strict 80% utilization ceiling would instead require 312.5 FPS.

A 25% FPS increase reduces fixed-job wall time by 20%, not 25%. At that improvement, ten ten-second jobs would still require roughly 70 seconds of generation on the shared device.

Ten connected calls with source idle and one active speaker is a different operating mode. Generated-idle peers consume neural capacity too; their work cannot be ignored in admission.

## 2. Priorities and dependencies

| Order | Work package | Primary benefit | Confidence / effort | Dependency |
| --- | --- | --- | --- | --- |
| P0 | Measurement correctness and clean baseline | Trustworthy decisions | High / small–medium | None |
| P1 | Queued-turn render-ahead and continuous audio timeline | Lower avoidable turn latency/holds | High architectural relevance / large | P0 |
| P2 | Unified admission and deadline-aware B1/B2 scheduler | Fairness, predictable concurrency | High relevance; throughput gain unknown / large | P0; integrate with P1 |
| P3 | Remove dead postprocessing and plan memory lifetimes | Small latency gains, less staging | Medium–high / medium | P0 |
| P4 | DiT profiling, fused projections and larger TRT islands | Largest compute opportunity | Medium / large | P0, stable P3 memory contract |
| P5 | Motion encoder and decoder execution tuning | Lower VAE latency/VRAM | Medium / medium–large | P0, P3 |
| P6 | Audio microbatch, IPC and encoding overlap | Secondary bottlenecks at scale | Uncertain until profiled / medium | P0, P2 |
| P7 | Selective quantization and altered model profiles | Potentially larger gains with quality risk | Low–medium / research | Quality corpus and P4/P5 baselines |
| P8 | Concurrent release/migration qualification | Product decision | Required / sustained testing | Candidate winners above |

Suggested implementation sequence: P0 first; P1 and a CPU scheduler simulator next; then low-risk P3 experiments; profile-driven P4/P5; finally P2 GPU batching and P6 under actual concurrent load. P7 remains opt-in research. Every candidate keeps a rollback flag and is tested alone before combinations.

## P0 — Repair measurement and establish the next baseline

Files: [experiment.py](../../soulx_rtc/experiment.py), [benchmark_calls.py](../../soulx_rtc/benchmark_calls.py), [benchmark_rtc.py](../../soulx_rtc/benchmark_rtc.py), [analyze_recordings.py](../../soulx_rtc/analyze_recordings.py), [server.py](../../soulx_rtc/server.py).

Implement:

1. Fix the finite recorder's square-canvas assignment. Save intended neural width/height, sent width/height and actual decoded width/height. Assert all expected geometry relationships.
2. Add explicit call/turn/epoch/chunk/frame/sample provenance. Analyze real generated boundaries rather than every 24th recorded frame.
3. Split pass/fail into transport, useful throughput, per-peer smoothness, audio continuity, quality and resource cleanup. Keep raw counters even on failure.
4. Add prequeued/no-gap turn submission, simultaneous speech bursts, mixed finite/call workloads, slow receivers and cold-avatar admission fixtures. Retain deliberate-gap tests as a different scenario.
5. Hash the full active model/service/configuration surface; record actual attention, codec and engine identities and both Torch and process/device memory.
6. Use a GPU timeline profiler to separate GEMMs, attention, normalization, convolution, allocator waits, copies and Python/engine dispatch. Do not infer kernel cost solely from high-level stage labels.
7. Repeat matched alternating candidate/control runs at least five times after identical warmup, recording temperature/clocks/co-residency where available. Use ten-second clips and longer continuous audio; include multiple approved portraits and seeds.

Acceptance: CPU tests catch rectangular recording and incorrect boundary attribution; replaying unchanged JSON reproduces metrics; repeated controls establish noise before a candidate is judged. Report medians and spread; small sample p95 is descriptive, not a strong tail guarantee.

Operational constraint: arrange sufficient GPU headroom through explicit user/operator coordination. Do not stop unrelated models to manufacture a clean baseline. Record a co-resident baseline separately if that is the actual deployment requirement.

## P1 — Decouple generation from playout and preserve continuous speech

Files: [calls.py](../../soulx_rtc/calls.py), [engine.py](../../soulx_rtc/engine.py), [worker.py](../../soulx_rtc/worker.py), [test_calls.py](../../tests/test_calls.py).

Current bottleneck: one active speech turn owns both generation and playout. The next queued speech is not appended until the previous turn's video tail and useful audio finish sending. The engine may be idle or generating low-priority silence while usable next speech could be prepared.

Design:

- Separate generation cursor/turn ownership from playout cursor/turn ownership.
- Introduce a bounded, sample-accurate conditioning timeline with explicit turn sample ranges; distinguish 16-kHz model samples from 48-kHz transport samples.
- Initially support one-chunk render-ahead into an already queued next turn while preserving the old whole-chunk semantics. Then evaluate packing adjacent speech segments into continuous model chunks without forced between-turn padding.
- Keep model chunks at 24 useful frames internally. A user turn boundary may fall inside a chunk; frame/audio provenance must describe that honestly.
- Preserve the same peer, RTP clocks and audio/video gate. Mark each user turn complete at its own audible/visible boundary, not by replacing tracks.
- Bound speculative work and define interruption semantics. On barge-in, invalidate future epochs; use the existing last-sent reconditioning fallback or an explicit RNG/state ledger. Never silently claim receiver-acknowledged rollback.

Changing padding/context changes model input even with identical weights. Compare packed output to a **single concatenated continuous reference waveform**, not to independently padded old turns as if they were identical conditioning.

Tests: two/three prequeued short turns, boundaries within a chunk, useful-audio tail shorter than one packet, silence segments, interruption during speculative work, retry while prequeued, queue saturation and deletion during preparation. Assert monotonic timestamps, no duplicated/lost useful samples, bounded buffers and correct per-turn ownership.

Acceptance: no scheduler-induced empty playout queue when next speech is already available and measured compute capacity is sufficient; no audio sample loss/repetition; materially lower first-media/avoidable-hold distribution on the prequeued fixture; unchanged interruption/cleanup correctness. This improves utilization/latency, not the raw model's theoretical FLOP rate.

## P2 — One admission budget and deadline-aware chunk batching

Files: [server.py](../../soulx_rtc/server.py), [calls.py](../../soulx_rtc/calls.py), [worker.py](../../soulx_rtc/worker.py), [engine.py](../../soulx_rtc/engine.py).

Unify finite generation, speech, generated idle and reference preparation under one GPU work queue. Charge all rendering work, not just active persistent calls. Separate connected-peer limits, active-neural limits, template memory, CPU media memory and per-tenant quotas.

Estimate a job's deadline from playout origin and remaining ready media. Use measured batch/profile latency distributions to choose B1/B2 work; don't wait to fill a batch beyond the earliest feasible deadline. Prefer imminent speech over idle extension, reserve bounded preparation service, and prevent starvation.

Start with a deterministic CPU scheduling simulator using recorded chunk durations and adversarial bursts. Then build exact B2 artifacts only if memory and predicted benefit justify them. Current FFN TRT requires B1; bypassing that guard is not an implementation.

Batch motion/decoder work only if its activation memory and timing improve the full chunk. Current VAE rows are serial, so DiT batching alone may have limited aggregate benefit.

Tests: same-seed reordered B1 and B2 private-state equivalence over many chunks; mixed identities/audio lengths; finite+call competition; cold-template burst; active cancellation; stalled receiver; idle yielding; all rejected work leaves no child state.

Acceptance: no false admission beyond tested capacity, bounded worst-peer wait under feasible load, no regression in C1 latency, real aggregate useful-FPS improvement at B2 with acceptable peak memory, and no degradation in recurrent quality. Overload must reject/queue explicitly, not silently deliver frozen calls as “ten sessions supported.”

## P3 — Eliminate avoidable work and reduce staging pressure

Files: [engine.py](../../soulx_rtc/engine.py), [utils.py](../../flash_head/utils/utils.py), [trt_backend.py](../../soulx_rtc/trt_backend.py).

Candidate A: move overlap removal before **per-frame color correction**, retaining all last-nine corrected feedback frames. The first nine decoded frames are otherwise discarded. Verify exact output and feedback parity under the intended math mode. Do not skip noncausal VAE decode context.

Candidate B: skip the final motion encode only for a state explicitly declared terminal/non-appendable. At ~73 ms once per ten-second job, this is a small optimization. Persistent calls must retain recurrence.

Candidate C: preallocate/reuse output and transfer buffers; examine the ~17.5 ms reference-mode allocator-release interval. Replace allocation churn with an event-safe lifetime plan for FFN/VAE workspaces and intermediates, or smaller-workspace tactics. Preserve minimum co-resident headroom. Do not remove `empty_cache` blindly.

Candidate D: remove the inactive text/audio embedding modules after strict loading if compatibility tests allow. About 19.894 MiB BF16 is headroom, not a large compute win.

Candidate E: compare resident/compact/reference modes and selective offload against full staging. The older 576×1024 staged profile spent ~923 ms/chunk in offload+reload intervals; preventing those transfers can matter more than a small kernel change. A profile that only fits during steady state but OOMs at startup is not accepted.

Acceptance: buffer ownership tests, no use-after-reuse across cancellation, no shape/profile leakage, repeated peak-memory traces, cold-start success and unchanged long-sequence quality. Report improvements individually and end to end.

## P4 — Target the DiT, not just more small FFN engines

Files: [flash_head_model.py](../../flash_head/src/modules/flash_head_model.py), [trt_backend.py](../../soulx_rtc/trt_backend.py), [trt_experiment.py](../../soulx_rtc/trt_experiment.py).

The historical native TRT run averages ~441 ms/chunk in four DiT passes, about 55.5% of recorded phase time. It is the largest remaining compute region.

Ordered experiments:

1. Identify the active attention backend and kernel-level distribution. Measure CPU launch gaps separately from GPU execution.
2. Fuse the separate self-attention q/k/v projections into one equivalent packed linear where the backend benefits. Preserve q/k normalization order, precision, head layout and state-dict compatibility.
3. Fuse eligible RMSNorm/RoPE or adaptive normalization/residual pointwise operations. Compare compiled Torch fusion before adding custom kernels.
4. Expand FFN islands to include suitable adjacent normalization/modulation/residual operations, or export larger transformer subblocks. Preserve fast attention if replacing it loses performance.
5. Use stable input/output buffers and capture eligible fixed-shape segments with CUDA graphs. Externalize stochastic sampling/session updates and keep separate first/recurrent prefix profiles.
6. Only after correctness, test larger partitions with representative timestep/chunk captures and exact B1/B2 artifact contracts.

The current 120 FFN Python/TRT crossings per chunk are a hypothesis for profiling, not a measured 120-times-fixed-cost tax.

Acceptance: test all four timesteps, first and recurrent chunks, every approved portrait, several seeds and interruption recovery. Numerical tests must cover output finiteness and distributions; perceptual evaluation must cover accumulated drift. Compare against the actual fastest valid baseline, not an intentionally slow attention fallback.

Stop if larger export forces slower attention, increases peak VRAM enough to require staging, or wins microseconds in isolation but worsens full-call tail latency.

## P5 — Accelerate motion encoding and tune the 3-D VAE safely

Files: [ltx_vae.py](../../flash_head/ltx_video/ltx_vae.py), [causal_video_autoencoder.py](../../flash_head/ltx_video/models/autoencoders/causal_video_autoencoder.py), [vae.py](../../flash_head/ltx_video/models/autoencoders/vae.py), [trt_vae_experiment.py](../../soulx_rtc/trt_vae_experiment.py).

Motion encoding of nine frames remains ~73 ms/chunk. Export the **deterministic posterior-parameter computation**, not an opaque stochastic `sample()`; keep sampling in the session-controlled RNG path. Build separate nine-frame recurrent and any required preparation shapes. Preserve latent normalization and the uniform log-variance expansion.

For decode/encode, test convolution tactic/workspace budgets and BF16-compatible 3-D memory layouts. A `channels_last_3d` experiment must account for layout conversions, compile specialization and every intermediate; blindly changing one tensor's layout may add copies.

Repair spatial tiling only as an explicitly validated low-memory experiment. The current helper computes an invalid latent tile step for Lite. Derive actual strides and noncausal decoder halos from the block graph; compare seams and full-frame output. Temporal reuse/tiling is higher risk because the decoder is noncausal.

Acceptance: posterior mean/log-variance parity before sampling, generator-state parity afterward, bounded long-sequence drift, no edge/tile artifacts, whole-chunk speed/memory gains. A 2× motion-encode speedup saves only about 36.5 ms/chunk; its isolated upper bound is not a 2× model speedup.

## P6 — Optimize secondary costs only when they become bottlenecks

Audio: microbatch equal-length rolling Wav2Vec windows, preserving per-window normalization and feature indexing. Current ~14 ms/chunk offers limited headroom. Reusing shifted hidden states is a model-context change, not an exact cache.

IPC: profile worker serialization, parent copies and encoder queues at ten connected peers. Consider a bounded shared-memory RGB ring with explicit slot generations, reader completion and epoch cancellation. A 24-frame 480×832 chunk is ~27.42 MiB, so copies can matter even when D2H itself is small.

Encoding: measure actual libx264 CPU time, thread oversubscription, queue depth and receiver-visible quality. Test a version-safe optional hardware encoder only if available and beneficial; retain H264/Opus negotiation and rate-control/quality checks. Faster encoding does not fix insufficient neural FPS.

Overlap: pinned CPU staging and a copy stream require correct CUDA events and buffer lifetime. Benchmark serial versus overlap with the same memory headroom. Additional streams may contend on an already busy GPU.

Acceptance: CPU/RSS/queue/encode evidence at C1 and C10; no corruption on slow receiver/cancellation; end-to-end gain rather than only a faster local copy benchmark.

## P7 — Research profiles, not silent production shortcuts

Selective quantization: collect inputs across all denoising steps, recurrent states, identities, lighting and interruptions. Start with linear/MLP sensitivity studies. Keep normalization, softmax, color and posterior-sensitive operations at justified precision. BF16→FP16 is not automatically safe; INT8/FP8 needs hardware/operator support and calibration evidence.

Reduced resolution: 288×512 is a validator-compatible 9:16 candidate, not a quality-certified profile. The previous aggressive low-resolution/two-step path failed visual review. Keep native 480×832 and 576×1024 labels separate from cropped/upsampled delivery.

Reduced steps, temporal token reduction, attention caching across steps/chunks, distilled models and ROI/hybrid generation change the approximation or model/task. They require a separate quality tier or training work. Do not sell them as copying MuseTalk's exact optimizations.

A hybrid source-video compositor may recover source choreography but does not follow automatically from a whole-frame generator. Face registration, lighting, occlusions and seams need their own design and validation.

## 3. Amdahl bounds: prioritize without promising gains

Using the historical TRT profile's ~794.34 ms summed mean stage intervals:

| Hypothetical isolated change | Arithmetic whole-stage-time improvement | Caveat |
| --- | ---: | --- |
| DiT 2× faster | ~1.38× | Assumes all else unchanged; not measured |
| Decoder 2× faster | ~1.15× | Does not accelerate motion encoding |
| Motion encode 2× faster | ~1.048× | Small stage fraction |
| Remove all audio cost | ~1.018× | Physically optimistic upper bound |
| Remove all color cost | ~1.031× | Also an optimistic upper bound |

These are phase-time bounds, not predicted receiver FPS. They omit changes in overlap, launch gaps, memory tactics, tail padding and other bottlenecks. Combining hypothetical wins must recompute the full denominator; do not add percentages.

## 4. Qualification matrix and release gates

Use the same approved MuseTalk base-bank assets and recorded speech fixtures, with hashes, but acknowledge that SoulX generates different whole-frame motion.

Run:

- Native 512² control, native 480×832 product canvas, native 576×1024 true 9:16; cropped 468×832 delivery separately.
- Ten-second useful clips, long single speech, repeated short prequeued turns, deliberate pauses, silence idle, barge-in and recovery.
- B1 then verified B2; C1/C2/C4/C6/C10 simultaneous speech; ten connected with one speaker; mixed-duty bursts.
- Source/hold/generated idle separately; cold/warm avatars; declared co-resident and dedicated-headroom conditions.
- Local loopback first, then browser/mobile/WAN/TURN, packet loss/jitter and slow receiver.
- At least 30-minute soak on the **final candidate source revision**, not an earlier version.

Proposed engineering gates (product thresholds to ratify before release):

- Quality: no collapsed/gray frames, identity/pose discontinuity beyond accepted baseline, visible tiling seams or systematic lip-sync regression; blind side-by-side review across the full fixture set.
- Correctness: no cross-session state contamination, missing/duplicated useful audio, out-of-order frames, stale epoch media reentry or retained worker state after cleanup.
- Smoothness on controlled local feasible-load tests: zero counted video underruns and zero inserted speech-time audio holds after start; separately report intentional idle and unavoidable input-wait holds.
- Delivery: monotonic timestamps, actual requested geometry/H264, recorded p50/p95/max arrival gaps and first-media time; a provisional p95 interarrival target of ≤60 ms at 25 FPS is a test target, not a past pass.
- Resources: no OOM or unbounded RSS/VRAM/audio/queue growth; startup and warmup fit; headroom policy explicitly documented.
- Capacity: claimed active count ×25 **useful** FPS, with measured per-peer gates and declared headroom. Do not admit based only on an average stage rate.
- Performance: candidate paired repetitions consistently improve useful throughput or target latency beyond observed control variance; retain regressions and failure files.

## 5. Decision checkpoints

After P0/P1: decide whether queued-turn latency is acceptable at one active speaker. After P3–P5: measure whether two active speakers are even feasible. Only then spend GPU time on larger concurrent profiles.

If native quality-preserving throughput remains far below 250 FPS, the ten-speaker requirement needs different hardware allocation, a smaller/distilled model, a lower accepted quality tier, or MuseTalk. More VRAM may eliminate staging but does not itself supply the missing 8.75× compute.

Do not migrate on export success, one attractive clip, a 25-FPS wire counter or a high isolated stage FPS. Keep the current backend available and publish each experiment's source/configuration/raw evidence before promotion.
