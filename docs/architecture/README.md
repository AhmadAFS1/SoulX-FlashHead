# SoulX-FlashHead: complete source walkthrough

Audit date: **2026-09-07 UTC**. Source baseline: SoulX `ac2c2bbb7308f87880c2ae4b01563d1e4d203c02`; MuseTalk comparison: `e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e`.

SoulX Lite is a recurrent, audio-conditioned **whole-frame video generator**, not a mouth-patch renderer. The fork wraps one shared GPU model in a persistent H264/Opus call service. Its model, serving and measurement layers have different contracts; this guide documents each separately.

## Start here

| Guide | Questions answered |
| --- | --- |
| [1. Entrypoints and loading](01_ENTRYPOINTS_AND_LOADING.md) | What starts the application? Which configuration, weights and model variant are actually used? |
| [2. Audio and diffusion transformer](02_AUDIO_AND_DIT.md) | What are the tensor shapes, attention operations, four denoising steps and cache invariants? |
| [3. VAE, color and continuity](03_VAE_AND_CONTINUITY.md) | How are 33 frames compressed/decoded? Why are nine frames recurrent? Can endpoints be controlled? |
| [4. Engine and memory](04_ENGINE_AND_MEMORY.md) | What is shared/private? What do the low-VRAM modes do? Where do memory and synchronization go? |
| [5. WebRTC and scheduling](05_WEBRTC_AND_SCHEDULING.md) | How do requests become persistent media, queued turns, idle, interruption and cleanup? |
| [6. TensorRT and test tooling](06_TENSORRT_AND_TEST_TOOLING.md) | What is actually exported? How are artifacts checked? What do tests and FPS numbers prove? |
| [Complete source inventory](SOURCE_INVENTORY.md) | Every tracked Python, shell, HTML, YAML and requirements file, its role, hash, lines and symbol map |
| [MuseTalk transfer analysis](../research/MUSETALK_OPTIMIZATION_COMPARISON.md) | Which previous optimizations transfer, which do not, and why? |
| [Next optimization plan](../research/NEXT_OPTIMIZATION_PLAN.md) | Ordered work packages, acceptance gates, dependencies and stop conditions |
| [Evidence validation](../research/EVIDENCE_VALIDATION_2026-09-07.md) | Recomputed results, measurement defects and limits on conclusions |

## Scope and confidence

The source inventory covers **74 tracked files / 14,600 physical lines** at the baseline: all in-repository Python, launch scripts, browser code, YAML and requirements files, including the vendored LTX and Wan code that a default ignore-aware search can miss. Active inference and service paths were traced through their callers and state changes; inactive model families and helpers are explicitly distinguished.

Installed Lite checkpoint configuration and tensor metadata were inspected to resolve actual architecture rather than assuming generic library defaults. Weight contents, installed third-party packages, compiled engines and upstream training internals are not an exhaustive audit scope. This is a source review, not a formal correctness proof or a claim that every optional path ran successfully.

MuseTalk's existing [92-document audit](../research/MUSETALK_DOC_AUDIT.md) is retained. This pass additionally revisited the actual scheduler, model wrappers, audio preparation, blending, TRT profile selection and persistent-media hot paths at the comparison commit.

**This change adds documentation and an executed CPU evidence notebook. It does not implement the proposed optimizations or produce new GPU performance measurements.** Historical test results are labeled as such; a passing old run does not certify a later source revision.

## End-to-end map

```text
HTTP / browser                    one GPU worker process
  create call -> approved still -> shared reference template + private session state
  submit audio -----------------> rolling Wav2Vec -> audio tokens + per-layer K/V
                                                    |
                         noise + reference + motion prefix
                                                    |
                              4 passes through 30-block DiT
                                                    |
                                    LTX decode: 33 RGB frames
                                                    |
                            per-frame color correction
                               /                    \
                 last 9 -> encode -> next state     drop first 9 -> 24 new frames
                                                               |
parent process <- CPU uint8 chunk <- bounded queue / epoch check
  persistent video clock -> H264 -> RTP -> receiver
  persistent audio clock -> Opus -> RTP -> receiver
```

The nine-frame feedback loop is the important difference from MuseTalk's independent face-frame work. One peer's next chunk depends on its preceding chunk, but different peers can be batched if shape, execution backend, memory and deadlines permit.

## Do not confuse these claims

- 512×512 and 480×832 are different neural workloads.
- 480×832 matches the available MuseTalk full-video plate dimensions but is not mathematically 9:16. Native 576×1024 is 9:16; 468×832 is a delivery crop.
- 25 received frames/second can include frozen repeats and idle. It is not necessarily 25 newly generated speech frames/second.
- Ten connected peers, ten admitted jobs and ten simultaneous smooth speakers are different capacities.
- Implemented TensorRT partitions do not mean the whole model is TensorRT or 25% faster.
- Stable RTP tracks and recurrent motion do not provide an exact generated first/last-frame constraint.

The current evidence still supports keeping MuseTalk in production and SoulX as a gated experimental whole-frame renderer.
