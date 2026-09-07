# 4. Shared engine, session isolation and memory ownership

[Guide index](README.md) · [Previous](03_VAE_AND_CONTINUITY.md) · [Next: WebRTC](05_WEBRTC_AND_SCHEDULING.md)

Sources: [engine.py](../../soulx_rtc/engine.py), [worker.py](../../soulx_rtc/worker.py), [trt_backend.py](../../soulx_rtc/trt_backend.py), [gpu_lease.py](../../soulx_rtc/gpu_lease.py).

## One model owner, many private trajectories

The parent service owns HTTP, peers, clocks, queues and CPU media. One spawned worker owns the CUDA context, weights, compilation products, TRT execution contexts, cached templates and a dictionary of session states. Process-pool calls serialize access to that worker.

The worker returns opaque state IDs and CPU frame arrays. The parent mirrors only the scheduling metadata it needs, such as total frames and cursor. It does not receive or duplicate the CUDA model.

| Object | Ownership | Lifetime / isolation rule |
| --- | --- | --- |
| DiT, VAE, Wav2Vec weights | One worker | Shared read-only during generation, except controlled device moves/backend installation |
| TRT contexts/workspaces | One worker | Reused serially; not safe for concurrent arbitrary calls without additional contexts and ownership |
| Reference templates | Worker LRU, maximum eight entries | Immutable; active states keep references even after cache eviction |
| Pipeline view | Per session | Shallow view sharing weights, with private mutable inference fields |
| Motion latent and RNG | Per session | Must never be shared across peers or reset by another peer |
| Audio buffer, offset and cursor | Per session | Compacted/advanced by append and generation rules |
| Current/queued uint8 media | Parent per session | Bounded queue; epoch invalidates obsolete work |
| H264 encoder and Opus track | Per peer | Stable transport lifetime, separate from generation-turn lifetime |

The template key includes image-byte SHA-256, width, height and steps. Reference preparation uses a controlled seed so identical templates are reproducible without consuming another session's generator. Each state clones its own motion and owns its random generator.

Evicting an LRU entry is not proof that its GPU allocation has been freed: active sessions can retain it. A future multi-avatar admission controller needs to count active template leases, not just dictionary length.

## Engine API and invariants

`prepare` validates geometry, obtains/prepares a reference template, initializes private state and audio, and computes the requested finite length. `prepare_call` uses the persistent-state variant.

`append` requires the previous state to be fully generated and the cursor to be at a 24-frame boundary. It preserves the rolling eight-second audio history and aligns the newly appended waveform after the already generated timeline. Whole-chunk padding is part of that contract.

`recondition` takes recent RGB frames, converts uint8 to the model's floating-point range, encodes the required nine-frame history, samples with the state RNG, and resets the generation/audio cursor for a new continuation. It does not restore a past RNG checkpoint or prove receiver-visible equivalence.

`generate(states)` validates compatible work, prepares conditioning, runs a batched DiT, then decodes and updates each state's motion in a per-row loop. It advances generation cursors before the parent has necessarily played those frames.

The distinction between **generated** and **sent** is central. A state can be fully generated while its final chunks remain in a parent queue or on the video clock.

## Exact per-chunk execution order

1. Establish dependencies between the default CUDA stream and the engine's compute stream.
2. Prepare each state's eight-second audio window and Wav2Vec features.
3. Form per-state noise/reference tensors and batch-compatible conditioning.
4. Compute/cache the projected audio and per-block cross-attention K/V for this chunk.
5. Perform the four DiT steps, preserving each state's motion prefix and RNG sequence.
6. Apply the configured device/allocator transition before decoding.
7. For each row: decode, release any transient decoder workspace, correct color, encode the next motion prefix, select useful frames, quantize/layout-convert and copy to CPU.
8. Apply configured module reload behavior, synchronize and return frame arrays plus phase metrics.

This is one compute-owner path, not a pipeline with concurrently executing DiT and VAE on two GPU streams. The `.cpu().numpy()` handoff and final synchronization are real dependencies. A “nonblocking” flag alone would not establish correct overlapping producer/consumer lifetimes.

[PyTorch's CUDA notes](https://raw.githubusercontent.com/pytorch/pytorch/v2.7.1/docs/source/notes/cuda.rst) explain that accurate timing requires synchronization/events, and that side-stream work requires explicit dependency and lifetime handling. These are constraints for a future overlap experiment, not evidence that overlap will speed this workload.

## Memory modes: names are not interchangeable

| Mode | Behavior | Tradeoff |
| --- | --- | --- |
| Default | Resident modules; bulk color work | Simpler path, larger intermediate footprint |
| Compact | Resident modules; color processed in small frame groups | Lower color activation peak, extra loop/launch overhead |
| Reference | Small-group color; DiT temporarily off GPU for reference preparation/reconditioning and some backend installation work | Reduces preparation overlap; does not normally move DiT every chunk |
| Staged | Stage Wav2Vec and DiT between CPU/GPU around their phases; keep DiT off GPU during VAE work | Fits lower headroom, incurs large transfer latency; incompatible with installed FFN TRT path |

In the historical native TRT reference-mode profiling, the field named `dit_offload` measures an allocator-release interval around `empty_cache()`, approximately 17.5 ms/chunk. It is **not** evidence of a DiT weight transfer in that mode.

In the historical true-9:16 staged experiment, the similarly named fields really did include module movement: roughly 443 ms offload and 480 ms reload per chunk. These labels cannot be pooled without reading configuration and the generating source revision.

Moving Wav2Vec to CPU between stages was a later fallback correction. An earlier staged benchmark is not a measurement of every subsequent staging change.

## Persistent, transient and externally allocated memory

Torch allocated/reserved peaks cover Torch-managed allocations. TensorRT also owns engine/device memory that those counters do not fully capture. Conversely, a TensorRT workspace backed by a Torch uint8 tensor does appear in Torch counters. Report both allocator and whole-process/device measurements.

The native TensorRT ten-job file records roughly 2,878 MiB Torch peak but roughly 5,212 MiB process GPU memory in post-run snapshots. A snapshot is not a peak. The reported CUDA-visible total and NVML physical total can also differ; do not silently mix them in a free-memory percentage.

FFN partitions use one maximum-sized workspace across serial layer calls. The VAE decoder creates its large workspace just before use, synchronizes the stream before releasing it, and releases it before motion encoding. The historical portrait workspace is about 664 MiB, versus about 436 MiB for square decoding.

This transient policy fixed a real overlap problem but introduces allocation/synchronization costs. A future shared lifetime-planned arena must prove that no context still accesses reused storage. Simply keeping everything allocated previously caused OOM; simply deleting synchronization risks incorrect output or memory corruption.

## Host memory is a concurrency budget too

For 24 uint8 RGB frames:

| Shape | One full chunk | Two queued + one current chunk per peer |
| --- | ---: | ---: |
| 512×512 | 18 MiB | 54 MiB |
| 480×832 | 27.422 MiB | 82.266 MiB |
| 576×1024 | 40.5 MiB | 121.5 MiB |

Ten native-portrait peers can retain about **823 MiB** just in those three chunk slots. Add recent sent-frame history, IPC serialization buffers, audio, idle decoders, encoders, Python objects and worker return values. This is a planning bound, not a measured RSS peak or proof that every queue is simultaneously full.

Bounded shared-memory rings could remove copies, but require slot ownership, generation counters, cancellation epochs and slow-receiver backpressure. Dropping an obsolete chunk must never release a slot while an encoder still reads it.

## Geometry validation and shape artifacts

The engine requires both dimensions to be multiples of 32, each in the supported 256–1024 range, with area no larger than 576×1024. A shape satisfying this validator is not automatically a valid TRT artifact shape or a visually accepted profile.

A batch-two native TRT run needs matching artifacts or a different backend strategy. The current FFN TRT serving path requires batch one. The presence of a scheduler microbatch implementation does not certify real-GPU batch-two quality, memory or throughput.

## Failure ownership and remaining hardening

The checkout lease prevents duplicate cooperating model owners but does not reserve a fraction of GPU memory from other applications. This audit left the live GPU workloads untouched and did not retest fit under a new memory snapshot.

Initialization failure should prevent readiness. A runtime GPU error marks the service unavailable and propagates errors to callers; the present worker has no automatic crash-recovery/respawn contract or watchdog timeout for a hung GPU future.

Cancellation during an awaited preparation call deserves fault injection: cancelling the parent's await does not necessarily cancel the child computation, which may still insert a state. Cleanup needs a request-ownership ticket or compensating release to avoid orphan states. This is a source-derived risk, not a measured leak in the existing soak.

Health includes source hashes for the RTC Python files, but not the entire model/configuration/dependency surface. The next evidence manifest should include model source, checkpoint hash, compiled artifact identities, active attention implementation and codec versions.
