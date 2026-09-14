# Male-avatar teeth quality test — September 13, 2026

The requested four-condition quality test completed. The male default avatar
has smeared/merged tooth detail in **unquantized BF16 raw output as well as INT8**.
In the inspected matched samples, disabling INT8 did not produce a clear teeth
fix. Increasing resolution to 384×672 gave somewhat sharper detail in some
frames, but visible tooth defects remain. None of these profiles receives an
anatomical teeth-quality pass based on this review.

FlashHead was temporarily paused with user authorization and restored to its
exact prior launch arguments/environment afterward. The unrelated GPU service
remained running. Final health: ready, zero calls/sessions/GPU states, compiled
320×576, four steps, INT8 enabled, shared-memory transport and startup GC freezing.
No live rendering settings were changed as a consequence of the quality test.

## Inputs and controlled comparison

Evidence directory: `/workspace/experiments/flashhead-male-quality-nsgv5L`.
Fixtures and all drivers/results are retained there. `runtime-private.json`
contains launch environment secrets and must never be displayed or published.
Do not rerun `run_and_restore.py` merely to recover context.

- Avatar: certified male default idle video from the MuseTalk asset bank,
  `sample_ai_human_facetime_closeup_production_v1/certified/idle_active_listening.mp4`.
  SHA-256: `099877cef231ce12dede03843c558d10c2fa1e9c4e054c83be595e81a00f6ae4`.
- Audio: `benchmarks/comparison-10s.wav`, 160,000 samples at 16 kHz, ten seconds.
  SHA-256: `0bbc0e4d1f1e4ecaad1f425e311e8f3ced2d1011d65a30b46487389331b456b2`.
- First source frame is the reference; output idle indices 31–39 at 25 FPS
  supply the same nine initial motion frames to both precision conditions.
  `fixtures.json` fingerprints each resolution's reference and motion RGB.
- Two seeds, 50 and 51; three fresh-state renders per seed per condition.
  Total: **24 measured renders**, with raw frames and videos saved for the first
  repeat of each seed/condition (eight ten-second clips / 2,000 useful frames).
- Each state uses `prepare_call`, `recondition` on the fixed motion history,
  then `append` with identical audio. Recurrent motion remains active throughout
  generation, matching the call engine's nonterminal state semantics.
- Same compiled Lite model, four denoising steps, optimized conditioning,
  real RoPE, fused QKV, lean output processing, compact memory, and 25 output FPS.
  No optional audio/color compiler or TRT path is enabled. Each condition uses
  the same 5,120-MiB Torch allocator ceiling.
- Order: 320 BF16, 320 INT8, 384 INT8, 384 BF16. Setup/compiler warmup and
  output encoding are excluded from timing. This is a sequential controlled
  quality experiment; it is not a randomized throughput superiority study.

Each render generated 264 padded frames, with 250 retained useful frames.
Useful FPS below divides 250 by the full padded generation wall time, so padding
cost is included. Warmup, conditioning preparation, saving and encoding are
excluded. It differs slightly from a finite unpadded 250-frame benchmark.

## Throughput and memory

| Profile | Median useful FPS, six runs | Peak Torch allocation, MiB | Repeats pixel-identical per seed |
| --- | ---: | ---: | --- |
| 320×576 BF16 | 56.98 | 4,654.27 | Yes |
| 320×576 INT8 storage / BF16 compute | 53.76 | 3,318.58 | Yes |
| 384×672 BF16 | 44.28 | 4,874.65 | Yes |
| 384×672 INT8 storage / BF16 compute | 42.29 | 3,548.45 | Yes |

INT8 saved about 1,336 MiB peak Torch allocation at 320 and 1,326 MiB at 384,
while adding dequantization overhead. These counters exclude CUDA context and
external allocations, and reset after warmup; they are not process startup peaks.
The unquantized profiles exceed the live service's 3,584-MiB Torch ceiling.

All eight retained raw clips had 250 valid RGB frames, zero whole-frame
blackouts and zero identical adjacent full frames. These integrity checks do
not assess tooth shape, natural motion or phoneme-level lip-sync accuracy.

## Teeth review and numerical measurements

Precision comparisons are between matching frame indices, seeds, audio,
reference and initial motion. The manual mouth ROI is normalized XYXY
`[0.30, 0.41, 0.72, 0.57]`; it includes lips and surrounding skin. It does not
segment or track individual teeth. All 250 frames per pair contribute to metrics.

| Resolution | Seed | BF16–INT8 mouth MAE /255 | Mouth PSNR | Worst-difference frame |
| --- | ---: | ---: | ---: | ---: |
| 320×576 | 50 | 1.919 | 38.00 dB | 107 |
| 320×576 | 51 | 1.570 | 39.62 dB | 117 |
| 384×672 | 50 | 1.516 | 40.38 dB | 77 |
| 384×672 | 51 | 1.474 | 40.48 dB | 95 |

Visual review covered raw crops for both seeds at frames 12, 37, 75, 125, 175
and 225, plus high-difference frames 77, 95, 107 and 117. These include several
open-mouth poses with visible teeth. Both BF16 and INT8 contain broad or smeared
tooth bands and imperfect tooth separation. The precision differences are
subtle in these samples; a large INT8-only regression was not observed.

Higher resolution changes the latent shape and generated motion, so its tooth
appearance cannot be interpreted as a pixel-aligned precision comparison.
Some 384 frames look sharper; some poses still look unnatural. Neither disabling
INT8 nor this modest resolution increase is a demonstrated complete fix.

This is one avatar, one ten-second utterance and two seeds. Numerical differences
are not naturalness scores. Frame review supports a visible-artifact finding,
but it does not establish a blinded human preference score, phoneme accuracy,
or artifact-free motion across arbitrary speech. Subjective mouth-temporal MAD
is not used as a teeth-quality pass criterion. The full movies are retained so
the user can assess motion and the affected teeth directly.

## H264 and real WebRTC receiver checks

The production `FastH264Encoder` was exercised with its `veryfast` preset,
two threads, baseline/yuv420p and initial 1-Mbit/s target bitrate. Direct codec
roundtrips captured first-decode RGB without a second recording encode.

Separately, all eight raw clips were relayed through actual local aiortc
DTLS/SRTP/H264 peer connections. Each receiver captured its first-decoded RGB:
**250/250 frames, monotonic PTS, 40-ms PTS gaps and zero packet loss** for every
clip. Maximum observed arrival gaps ranged from about 50 to 138 ms, with larger
first-run gaps during concurrent model loading/analysis. The relay is paced
pregenerated video, and excludes inference scheduling and live audio. It does
not qualify multiple active calls or the wall's end-to-end audio synchronization.

Raw-versus-WebRTC mouth MAE was approximately 2.52–2.64 /255 across profiles.
Encoding visibly softens some detail, but the tooth problem is present before
encoding. This metric cannot assign an anatomical artifact's relative cause:
codec pixel error and quantization pixel error describe different transformations.

Receiver `.npy` files retain first-decode pixels. Receiver MP4s are subsequently
encoded at CRF 18 for convenient viewing; their audio is the original fixture
added for playback, not audio captured from the video-only relay.

## Watch and inspect

These links refer to retained local experiment files, not published Git assets:

- [Seed 50 mouth comparison](/workspace/experiments/flashhead-male-quality-nsgv5L/seed-50-mouth-comparison.mp4).
- [Seed 51 mouth comparison](/workspace/experiments/flashhead-male-quality-nsgv5L/seed-51-mouth-comparison.mp4).
- [Seed 50 full portrait comparison](/workspace/experiments/flashhead-male-quality-nsgv5L/seed-50-four-way.mp4).
- [Raw four-profile mouth crops](/workspace/experiments/flashhead-male-quality-nsgv5L/seed-50-four-way-mouth.png).
- [High-difference crops](/workspace/experiments/flashhead-male-quality-nsgv5L/seed-50-high-difference-mouth.png).
- [Raw versus WebRTC crops, 384 BF16](/workspace/experiments/flashhead-male-quality-nsgv5L/384x672-bf16/seed-50-rtc-mouth.png).
- [Quality summary](/workspace/experiments/flashhead-male-quality-nsgv5L/quality-summary.json).

## Decision and next experiment

Keep the restored live profile pending a validated quality improvement. The
test does not support attributing the main visible defect to INT8 alone.
It supports evaluating the model's image-detail limitations, reference quality,
and mouth motion at a higher resolution. A controlled 480×832 or 512-square
BF16 render of this same avatar is the next useful baseline. If it remains poor,
evaluate the higher-quality model variant or another renderer rather than
assuming further precision changes will repair the teeth. If BF16 helps a
specific case, selective quantization is still worth measuring, but no layer
policy was implemented or quality-certified here.

The shared idle cache/media transport can remain regardless of model precision.
This test makes no new migration or MuseTalk superiority claim.
