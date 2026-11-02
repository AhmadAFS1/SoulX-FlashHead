# App integration validation — 2026-11-02 UTC

**Host GPU: Apple M5 Pro, 20 GPU cores.** The host has **24 GB unified system
memory**, with no dedicated VRAM; per-process GPU-visible allocation capacity was
not measured. Evidence: `system_profiler SPHardwareDataType SPDisplaysDataType`
on this MacBook Pro (Mac17,8). Metal 4 is reported. macOS 27.0.1 (26A434), Xcode
27.0 (27A266a), iOS/iOS Simulator SDK 27.0. Apple supplies the GPU driver with the
OS; no independent driver version was collected. Co-resident load was not
controlled: another simulator and development applications were running, and
other repository work was concurrent.

These are **compilation, simulator UI, negative-path, and static lifecycle
checks**, not SoulX GPU/ANE inference benchmarks. No NVIDIA GPU was used for
these checks. No physical iPhone performance, output quality, or CUDA parity is
established here.

| Check | Hardware / runtime | Result |
| --- | --- | --- |
| Debug simulator compilation | Apple M5 Pro host above; iOS Simulator SDK 27.0, iOS 18 deployment target; arm64 and x86_64 compile targets | Passed with integrated app and runtime. |
| Release device compilation and signing | Apple M5 Pro host above; iOS SDK 27.0, arm64/iOS 18 target; no device inference | Passed. Team `563AETFKT9`, bundle `com.lingua.soulxstudio`. |
| Empty-state visual check | Apple M5 Pro host above; **simulated** iPhone 12 Pro at 1170 × 2532, iOS 26.2 (23C54); simulator GPU, no model loaded | Launched successfully. Portrait, audio, installation action and status fit the screen; orange primary action uses black text for contrast. |
| Bundled-avatar visual check after UI update | Same Apple M5 Pro host and iPhone 12 Pro/iOS 26.2 simulator above; simulator UI rendering, no model prediction | Both approved source portraits render in the two cards. Indian is selected initially and appears in the reference stage. Audio and primary action remain visible. |
| Missing-model diagnostic launch | Same simulator and host above; `--smoke`, no model manifest, no inference attempted | Persisted failed report naming the missing manifest. No generated images or FPS result. |
| Lifecycle review | Static Swift source review on the host above; no GPU execution | Portrait changes and reinitialization clear stale output/metrics. Compute changes invalidate initialization. Busy controls prevent overlapping user sessions. Cancellation invalidates the session; playback errors cancel inference. |

Build products are local and ignored by Git:

- `build/Release-iphoneos/SoulXStudio.app`
- `/tmp/soulx-studio-sim/Build/Products/Debug-iphonesimulator/SoulXStudio.app`

Build logs: `/tmp/soulx-studio-device-build.log` and
`/tmp/soulx-studio-sim-build.log`. The only final build warning is Xcode skipping
App Intents metadata extraction because this app has no App Intents dependency.
Local visual evidence is `.impeccable/review/iphone12pro-empty-state.png`
and `.impeccable/review/iphone12pro-bundled-avatars.png` (excluded from Git by the
repository's generated-image rule). The empty-state capture predates the user's
request for the two predefined avatar cards.

The dedicated simulator was shut down after visual review to free host memory.
Settings, system audio-file picker interactions, and sustained audio/video
playback have not been exercised through UI automation. Complete-model import,
portrait encoding, generated-frame playback, memory pressure, Neural Engine
placement, sustained throughput and numerical/visual parity require the real
export and a separate physical-device validation record. The app contains no
placeholder generation path to stand in for those checks.

The final signed Release build includes both source PNGs and their `pro320`
reference caches. The cache metadata records separate **Mac CPU-only Core ML
reference predictions on Apple M5 Pro / 24 GB unified memory**, not iPhone or
GPU/ANE inference. See each `Resources/Avatars/*/reference-cache/metadata.json`
for its exact timestamp, source/model/tensor hashes and execution provenance.
