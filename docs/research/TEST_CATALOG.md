# SoulX-FlashHead test catalog

September 7 documentation-audit addendum: the [evidence validation report](EVIDENCE_VALIDATION_2026-09-07.md) and [executed CPU notebook](notebooks/optimization_evidence_audit.ipynb) recheck eight retained JSON artifacts, 26 offline run rows, FPS/phase/capacity arithmetic, call counters and selected source contracts. All five notebook code cells passed. No new GPU benchmark or runtime test-suite run is implied; the catalog below retains its original scope and chronology.

This is the complete index of the implementation and migration tests recorded in this fork as of **2026-09-07**. It is intentionally separate from the model walkthroughs: every result below names the artifact, workload, and limitation. Detailed interpretation belongs in [implementation results](IMPLEMENTATION_RESULTS.md); historical MuseTalk comparison evidence remains in [validation](VALIDATION.md) and [the 92-file audit](MUSETALK_DOC_AUDIT.md).

## Test environment and provenance

Tests ran on an RTX 4070 with 12,282 MiB visible, driver 570.181, Python 3.10, Torch 2.7.1+cu128, BF16, aiortc 1.14.0, PyAV 16.1.0 and TensorRT 10.9.0.34 where noted. An unrelated OmniVoice process remained running throughout and changed from approximately 6,150 MiB to 7,456 MiB during the work. MuseTalk was not modified. The checked-in artifacts contain hashes for the input avatar, audio, source files and raw timestamp companion where applicable.

Approved avatar input:

`/workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/idle_active_listening.mp4`

It is 480×832 at 24 FPS. Its SHA-256 is `099877cef231ce12dede03843c558d10c2fa1e9c4e054c83be595e81a00f6ae4`. The canonical first-frame PNG is checked in as `benchmarks/implementation/musetalk-idle-anchor.png`.

Useful generation FPS means newly generated frames divided by unpaced generation wall time. It excludes nine-frame overlap, tail padding, source-idle/held frames, startup, compilation, and codec drain. WebRTC wire FPS includes held and idle frames and must not be substituted for neural FPS.

## Automated tests

The final CPU suite passed **34 tests**, with four upstream Triton deprecated-autotuner warnings. The suite was run after the final audio-boundary, staged-memory, cross-peer admission and waveform-diagnostic changes.

| Test area | Coverage |
| --- | --- |
| Model optimization math | Prepared rotary path against the complex reference; cached audio projection/cross-attention conditioning; cached reference color statistics; private per-session outputs and recurrent state |
| Persistent calls | Real aiortc peer negotiation, stable audio/video tracks, one-offer contract, H264 and Opus reception, monotonically increasing PTS, retry idempotency, conflicting retry rejection, interruption epochs and cleanup |
| Scheduling | Bounded queues, active-call admission, lead ordering, compatible microbatch selection, independent outputs, generated-idle render-ahead, speech priority over synthetic idle, cross-peer idle handoff |
| Audio/media clock | Zero-idle accounting, video catch-up prevention, one-packet audio scheduling allowance, audio-hold accounting, explicit padding and stall behavior |
| TensorRT contracts | Artifact checksum, checkpoint/runtime/GPU identity, exact shapes and dtypes, exactly two bindings, workspace ownership, transient workspace release and fail-closed behavior |
| Codec | Actual H264 encoder/context detection, packet decode, bitrate/context recreation, pinned aiortc 1.14.0 factory behavior and upstream encoder path |
| Process safety | Exclusive per-checkout GPU-owner lock and release; spawned GPU worker boundary; health failure propagation |
| Audio diagnostic | Synthetic normalized-correlation test rejects low-energy silence roundoff and finds a known waveform offset |
| Static checks | JavaScript `node --check`, shell `bash -n`, Markdown local-link audit, JSON parse audit, `git diff --check`, scoped credential-pattern scan |

GPU isolation additionally ran in the service warmup: two independent sessions, two chunks, reversed batch-one execution order, maximum pixel difference **0**. Actual GPU batch-two was not validated in this constrained run. Fixed TensorRT profiles are batch one.

## Offline generation and optimization benchmarks

| Artifact | Workload/result | Status and caveat |
| --- | --- | --- |
| `square-ab.json` | 512², 4 steps, 250 frames, five alternating repeats: baseline 41.54, cached 42.03, real-rotary 43.11 useful FPS | Earlier optimization sequence; not a matched TensorRT ten-job comparison |
| `portrait-allocator-ab.json` | Native 480×832, five alternating repeats: baseline 26.07, optimized 27.51 useful FPS | Matched PyTorch A/B; approximately 5.5% gain |
| `portrait-staged-ab.json` | Native 480×832 staged PyTorch, three repeats: approximately 12.21 useful FPS | PCIe weight transfers dominate; earlier staged implementation |
| `portrait-reference-ab.json`, `portrait-reference-real-fresh.json` | Reference-only portrait preparation variants, including startup OOM and fresh optimized result around 27.6 FPS | Diagnostic/fresh-process evidence, not all rows are complete A/B tests |
| `portrait-ten-jobs.json` | Ten independent native 480×832 jobs: median 27.60 aggregate useful FPS; p95 completion 90.67 s | Aggregate sequential sweep, not ten real-time speakers |
| `portrait-9x16-staged.json` | Native 576×1024 true 9:16: median 10.88 useful FPS; 250 frames in 22.97 s | Four-step staged profile; not real time |
| `square-trt-combined-reference.json` | Full 512², 30 TRT FFNs + TRT VAE: median 44.26 useful FPS | Three fresh-process runs; no matched 25% claim |
| `portrait-trt-transient-workspace.json` | Full native 480×832 TRT: median 28.49 useful FPS | Three runs; transient VAE workspace; only about 153 MiB CUDA free in snapshot |
| `portrait-trt-ten-jobs.json` | Ten native TRT jobs: median 28.57 aggregate useful FPS; p95 completion 87.74 s | Approximately 3.5% over the earlier PyTorch sweep; separate, not matched A/B |
| `failures.json`, `square-trt-ffn.json`, `square-trt-combined.json`, `portrait-trt-combined.json` | Startup/default allocator/full-workspace OOM and failed predecessor evidence | Preserved failures; not silently replaced by successful rows |

The TRT kernel reports record all 30 FFN reload/numerical checks and isolated speedups of approximately 1.034× square and 1.154× portrait. The decoder reports approximately 1.176× square and 1.182× portrait isolated speedups with finite output and nonzero numerical deltas. These are kernel results, not additive whole-model gains.

## WebRTC tests

### Earlier native API and source-bank control

The historical migration artifacts contain the source-bank relay, native 10-second/23.05-second recordings, endpoint-handle checks and the initial finite `/sessions` tests. The source-bank relay verified 1,350/1,350 frames, continuous PTS and zero loss but performed no SoulX inference. Early native API recordings did not reliably prove H264 because codec preferences were applied after remote SDP; they remain valid transport diagnostics but are not H264-qualified.

### Verified H264 TensorRT portrait ramp

These tests used native 480×832, four steps, batch one, all 30 TRT FFNs, transient-workspace TRT VAE, and the tested `FastH264Encoder` (`libx264`, `veryfast`, two threads). Every row completed transport/cleanup checks, but underruns show that transport success is not real-time smoothness.

| Artifact | Peers/turns | Wall time | Underrun slots |
| --- | --- | ---: | ---: |
| `calls-h264-trt-c1.json` | One peer, two turns, interruption and recovery | 23.14 s | 1 |
| `calls-h264-trt-c2.json` | Two active peers, one 3-second turn each | 10.54 s | 134 |
| `calls-h264-trt-c4.json` | Four active peers | 20.35 s | 935 |
| `calls-h264-trt-c6.json` | Six active peers | 31.81 s | 2,622 |
| `calls-h264-trt-c10.json` | Ten active peers | 58.90 s | 9,153 |
| `calls-h264-trt-mixed10.json` | Ten connected, one active speaker | 18.87 s | 27 |
| `calls-generated-idle-h264.json` | One generated-idle peer, turns plus interruption/recovery | 27.58 s | 0 |

The ten-active-peer row is a decisive failure of the ten-speaker smoothness target. The mixed row is not a ten-speaker test. The one-peer recording is evidence of the call protocol, not perceptual lip-sync or invisible endpoint quality.

### Sustained call and fallback regression

`calls-generated-idle-soak30m.json` ran for 1,819.82 seconds. It completed 209 useful speech turns/210 submitted IDs, received 45,493 video frames at 24.987 wire FPS, had p95 arrival gap 44.0 ms and maximum 191.3 ms, verified H264 and monotonic PTS, and cleaned all calls/sessions/GPU states. It recorded 23 missed video slots and 10,535 held frames, so it was **not** a strict smoothness pass. Sixty samples showed server RSS 275.61–363.81 MiB, worker RSS 6,186.65 MiB and retained audio 458,240–642,560 bytes. Thirty-one native receiver snapshots and their hashes plus the compressed timestamp hash verified.

The later final staged fallback used the same 480×832/four-step asset beside the 7,456-MiB co-resident process. `calls-final-staged-c1.json` completed three turns plus interruption/recovery in 46.48 seconds with H264 and cleanup passing, but had 251 video underruns and 481,920 audio-hold samples. `calls-final-staged-c2.json` completed one turn for each of two peers in 21.77 seconds with one render slot, proving idle admission yields to queued speech; it still had 159 underrun slots. These fallback rows are not real-time results.

The recorded fast-mode waveform diagnostic is `audio-before-short.json`; it found a roughly 20-ms first-turn gap that video counters alone did not expose. `check_recorded_audio.py` is a reproducible normalized-correlation diagnostic, not a perceptual lip-sync test.

## Headroom failures and process state

After the sustained run, OmniVoice grew to approximately 7,456 MiB. A clean native TRT restart failed while deserializing a 55 MiB FFN engine allocation; a first staged restart failed on a 50 MiB Wav2Vec allocation, and a follow-up staged reload failed on a 20 MiB denoising allocation. Those attempts produced zero benchmark samples and are recorded in `portrait-trt-restart-headroom-failure.json` and `portrait-staged-restart-headroom-failure.json`. The final staged implementation starts under that load, but its transfer overhead makes it unsuitable for smooth calls. No unrelated process was stopped.

## Acceptance decision

The implementation is published, runnable, and substantially tested. The requested 25% end-to-end gain, ten simultaneous native real-time speakers, exact first/last generated frames, invisible source/generated joins, receiver-acknowledged rollback, and perceptual lip-sync gates remain unpassed. Keep MuseTalk as the production renderer; keep SoulX as an experimental whole-frame renderer until those gates are independently demonstrated.
