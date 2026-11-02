# Core ML implementation validation

Date: 2026-11-01 America/Chicago / 2026-11-02 UTC.

Mac: **Apple M5 Pro**, 24 GiB unified system memory, 20-core integrated GPU;
macOS 27.0.1, Xcode 27.0, Swift 6.4, Python 3.12.12, PyTorch 2.7.0,
Core ML Tools 9.0, NumPy 2.2.6. CPU correctness runs use `CPU_ONLY`; the native
Mac smoke run requests CPU + GPU. Unified memory is shared; no dedicated VRAM.
Co-resident conversion/validation load was not isolated, so these are functional
run timings, not controlled throughput benchmarks. No NVIDIA GPU was used.

Target: **iPhone 12 Pro (iPhone13,3), Apple A14 GPU**, iOS 26.2 (23C55).
ProcessInfo reports 5,994,840,064 bytes of system memory; iOS does not expose a
fixed app memory limit. GPU/Neural Engine placement must be profiled independently
of the requested Core ML compute configuration.

## Source and numerical checks

- 11 Python tests pass: six against upstream PRO modules, four against original
  real-weight support networks, and a Core ML mixed-precision regression.
- Preserved native Swift checks pass for strided tensor buffers and conversion,
  audio context indexing, image/color processing, actual audio resampling, and
  deterministic Gaussian sampling.
- All 13 DiT components and 24 numerical cases pass for each profile. Maximum
  component normalized RMSE is 4.682% for pro320 and 4.048% for pro576 against
  FP32 reference, with the declared 5% component gate. These independent random
  input checks do not establish full-video perceptual equivalence.
- All four support networks pass their stricter numerical gates for both
  profiles. Reference and audio use FP32; tiny encode/decode use FP16.

The all-FP16 DiT conversion produced about 10–12% error on the first partition.
Keeping attention and normalization FP32 corrected this; the learned dense and
convolution weights remain compressed to INT8. Audio and reference experiments
with lower precision exceeded their error gates and were rejected. Failed
experiment records are retained under `validation/` to explain these choices.

## Native Mac end-to-end run

The command-line runner compiles the same five Swift inference files used by the
app. At height 320 / width 176, it initialized a real portrait and completed two
successive windows, including color correction and motion feedback: **56 actual
generated frames**. Requested CPU + GPU on the M5 Pro; actual placement unverified.

| Operation | Measured time | Generated frames/s |
| --- | ---: | ---: |
| Portrait initialization | 9.94 s | n/a |
| First 28-frame window, including initial model loads | 17.28 s | 1.62 |
| Second 28-frame window | 5.56 s | 5.03 |

These are Mac measurements. The 25 fps playback cadence is separate. Two endpoint
images look structurally coherent; they do not certify lip sync or long-session
quality. See `validation/mac_pro320_smoke.json` for exact provenance.

## iPhone findings

The first height-576 / width-320 reference-only test requested CPU + GPU. The
process disappeared before returning an output; the final persisted memory sample
was about 3.2 GB. A Jetsam cause was not confirmed by a crash log. This profile is
not certified for the iPhone 12 Pro. Its initial reference conversion also failed
numerical gates and was subsequently replaced by FP32; that replacement has not
been shown to fit the phone at this resolution.

The iPhone profile therefore keeps the full PRO network and two-step schedule at
height 320 / width 176. The first full pro320 run with automatic compute selection also terminated with
signal 9 during the monolithic FP32 reference encoder, before denoising. This is
recorded in `validation/iphone12pro_reference320_interrupted.json`; it is not a
completed inference run.

The user subsequently selected two fixed repository avatars. Their original Wan
reference features are prepared on Mac and shipped as validated per-avatar caches.
The phone then initializes from those static features; audio conditioning, PRO
denoising, decoding, color correction and motion feedback still execute locally.
This is a deliberate change to initialization, not a claim that the full Wan
reference encoder fits the iPhone. The cached Indian avatar completed its first physical run: 28 generated frames
in 85.28 seconds (0.328 generated FPS), with automatic Core ML compute selection.
A subsequent two-window run completed 56 frames: its first window took 81.57 s,
and its next window 31.20 s (**0.897 generated FPS**). Reinstallation preceded the
run, so the first window includes initial model specialization/loading; the next
window is the observed warm measurement. Cached initialization took about 0.04 s.
The audio clip used for these functional runs contained the quiet opening of the
source recording; these runs verify execution and continuity, not lip sync.

These are physical **iPhone 12 Pro / Apple A14 / iOS 26.2** results at **width 176,
height 320**, using all 30 PRO blocks, two denoising steps, INT8 compressed dense
weights and mixed precision. Actual per-operation GPU/ANE placement is unverified.
This is a working local prototype, substantially below real-time output.
See `validation/iphone12pro_pro320_cold.json` and
`validation/iphone12pro_pro320_two_windows.json` for source records.

The **Asian avatar active-speech run** also completed 56 frames on the same A14:
first window 87.04 s (0.322 FPS), second window 34.42 s (**0.814 FPS**). Input was an
active 2.24-second speech segment (3.00–5.24 s of the retained local audio), rather
than its quiet opening. All 56 original device-generated PNG frames were retrieved
and packaged with that input audio as `artifacts/coreml/iphone-asian-active-speech/generated.mp4`.
Inspected frames show mouth changes and coherent portrait structure; this short
sample does not establish formal lip-sync quality or sustained thermal behavior.

The app is installed with both requested avatars and the complete pro320 model
bundle. It was returned to its normal interface after testing. No connected phone
is required to inspect the source, models, reports or generated sample.

## App and source isolation

Signed Release device build and simulator builds pass. The app installs under
`com.lingua.soulxstudio` and loads user-selected model profiles separately from its
binary. Simulator UI and missing-model states were checked. CUDA implementation
files, requirements and entry points are unchanged; all inference additions live
under `apple/` and `ios/`, with generated artifacts ignored by Git.

The branch is `codex/ios-coreml-inference`, based on preserved PRO source commit
`4c5c7aa72cb65045e9f26a3c9d04f08a819fb0a8`. No live GPU server deployment was changed.
