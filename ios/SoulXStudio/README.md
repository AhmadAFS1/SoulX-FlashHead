# SoulX Studio

Native iPhone host for the additive SoulX FlashHead PRO Core ML runtime. Choose a
bundled Indian or Asian avatar, initialize its reference features, import a local audio file, and
generate talking-head frames on the device. CUDA entry points remain separate.

This app requires iOS 18 or later. The iPhone 12 Pro meets the OS requirement;
whether its memory and inference throughput are sufficient for a particular
export must be established with that export on the physical phone. A successful
simulator build does not establish Neural Engine compatibility or device FPS.

## Build and install

Open `SoulXStudio.xcodeproj` in Xcode, select the **SoulXStudio** scheme and your
iPhone, and run. Automatic signing uses development team `563AETFKT9`; change
that setting if building with a different Apple developer account. The bundle
identifier is `com.lingua.soulxstudio`.

The project is reproducible from its XcodeGen definition:

```sh
xcodegen generate --spec ios/SoulXStudio/project.yml
xcodebuild -project ios/SoulXStudio/SoulXStudio.xcodeproj \
  -scheme SoulXStudio -sdk iphonesimulator -configuration Debug \
  -derivedDataPath /tmp/soulx-studio-sim CODE_SIGNING_ALLOWED=NO build
```

Run these commands from the repository root. Exported models are deliberately
outside the app bundle and source control. See the repository's `apple` directory
for conversion, model contracts, and validation tooling.

## Install models

In **Models & settings**, choose **Import model folder** and select an exported
profile folder, such as `pro320` (320 pixels high × 176 wide) or `pro576`
(576 pixels high × 320 wide). The app copies it into `Documents/Models/current`,
validates its manifest and required model files, then replaces the active profile.
The displayed dimensions come from that manifest, in width × height order.
Different profiles use the same PRO architecture and can have different spatial
resolutions; their memory and speed must be tested separately. Keep enough free
storage for both the incoming folder and existing models during replacement.

Keep multiple exported profile folders in Files or on your Mac. Import the one
you want to use; the app holds one active profile at a time. Switching profiles
clears the prior output and requires portrait initialization again. The app does
not infer a profile from its folder name or fall back to an older `pro576` folder.

For large exports, copy the folder from a Mac into the application's Documents
container using Xcode's device container tools, with the profile's files directly
inside `Documents/Models/current`, then choose **Check installed models**.
Finder file sharing and Files document access are enabled. The final layout is:

```text
Documents/
  Models/
    current/
      manifest.json
      reference.mlpackage/
      audio.mlpackage/
      conditioning.mlpackage/
      patch.mlpackage/
      blocks_00.mlpackage/
      ...
      head.mlpackage/
      decoder.mlpackage/
      encoder.mlpackage/
```

Compiled `.mlmodelc` directories are also accepted. The manifest specifies the
partition count; the runtime determines the complete required file set from it.
First use of an uncompiled package may spend time compiling it on the device.
Missing or incompatible files produce an explicit error; there is no substitute
video, remote inference, or generated-frame simulation.

## Generate a session

1. Choose **Indian** or **Asian** from the two avatar cards. Indian is selected
   initially. These are existing repository portraits; their exact source paths
   and checksums are recorded in `Resources/ASSET_SOURCES.md`. The runtime
   resizes/crops the selected source for the export's input shape.
2. Choose an audio file from Files. The runtime decodes and resamples supported
   audio to mono 16 kHz.
3. Tap **Initialize portrait** to verify and load the avatar's bundled reference
   features. These are precomputed with the original Wan reference encoder from
   that exact source PNG. The cache must match the imported profile's dimensions,
   checkpoint and tensor checksums. This avoids running the memory-heavy
   reference encoder on the iPhone. Missing or mismatched caches produce an
   explicit error; the fixed-avatar UI does not silently fall back to encoding.
4. Tap **Generate & play**. Actual generated frames and their aligned audio are
   played as each window completes. **Stop** cancels between model predictions.

The displayed **frames / second** is generated frames divided by measured
inference time, separate from the model's 25 fps output cadence. Playback follows
the audio clock and buffers if generation falls behind. The queue is bounded to
two waiting chunks to avoid accumulating a full video in RAM. The audio source is
currently loaded into memory, so use short clips for initial device testing.

Changing the portrait or compute mode requires initialization again. Each finished
or stopped stream also requires initialization for a fresh temporal state.
Active work keeps the display awake. Initialization and generation stop when the
app enters the background. Computation is local;
selecting **Neural Engine + CPU** requests those compute units from Core ML but
does not guarantee every operation runs on the Neural Engine.

Each avatar's source PNG and verified reference cache live together in the app
bundle under `Avatars/<id>/`. They provide portrait conditioning only; audio
encoding, PRO denoising, generated-frame decoding and temporal updates still run
through the real on-device runtime. Cached references are not prerecorded video.
The bundled caches currently target `pro320` (320 pixels high × 176 wide).
Using another spatial profile requires corresponding caches in the app build;
the runtime rejects a profile mismatch.

## Generation scheduling

**Overlap generation work** in Models & settings is enabled by default. On the
PRO320 profile the app can prepare one future speech window and load the next
compiled transformer partition during prediction. It checks available memory
before starting additional work. Turn the setting off for sequential execution.
The speech encoder validates its output before projection and retries on CPU if
Core ML returns non-finite features. Timings and paired quality checks are in
[the performance report](../../apple/PERFORMANCE.md).

## Code map

- `Sources/UI`: portrait stage, input controls, model installation, measurements.
- `Sources/App/StudioModel.swift`: user session lifecycle and model imports.
- `Sources/App/ChunkPlayer.swift`: bounded, audio-clock-paced chunk playback.
- `Sources/Runtime`: model loading, Core ML graph orchestration, tensor/image/audio
  processing, and optional device smoke validation.

Product and visual decisions are recorded in `PRODUCT.md` and `DESIGN.md`.

## Finite device validation

For a reproducible first-window run, place `portrait.png` and `audio.wav` in
`Documents/Smoke`, install a complete profile in `Documents/Models/current`, and launch the app with the
Xcode scheme argument `--smoke`. Optional compute arguments are `--cpu-only`,
`--cpu-gpu`, or `--cpu-ane`; otherwise the runtime requests all compute units.
The normal controls are disabled during this diagnostic run.

Use `--prefetch-blocks --prefetch-audio` to exercise overlapping work, and `--model-timings` for stage timings. Add `--smoke-chunks 2` for two windows, and `--smoke-save-frames` to save every generated PNG. Use an audio file long enough for the requested windows.

The app writes `Documents/SmokeResult/report.json` before expensive operations
and updates it after each stage. Successful inference also writes `first.png`
and `last.png` from actual output. Missing models or failed inference produce a
failed report. Inspect the report, images, memory, and device logs before claiming
the export is usable. This run measures one real inference window, not sustained
thermal behavior or continuous playback performance.

To validate only the **576-high × 320-wide** reference encoder before the full export is ready,
copy its compiled model to `Documents/Probe/reference.mlmodelc`, place the real
portrait at `Documents/Smoke/portrait.png`, and launch with `--probe reference`.
The same compute flags apply. This path requires no bundle manifest or other
models. It writes `Documents/ProbeResult/report.json` before model loading and
prediction, with OS/hardware, requested compute units, sampled memory, separate
load and prediction timing, output shape, and finite-value checks. It creates no
video frames and establishes neither end-to-end FPS nor visual/CUDA parity.
The probe takes precedence if both `--probe` and `--smoke` are supplied.
