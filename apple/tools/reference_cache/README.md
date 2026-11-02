# Fixed-avatar reference features

The iOS app uses precomputed **original Wan portrait conditioning features** for
its two fixed avatars. No video frames, audio embeddings, motion history, or
DiT predictions are cached. Audio processing and PRO video generation still run
on the device. The original on-device reference encoder remains implemented,
but exceeded the tested iPhone 12 Pro memory limit.

Generate a cache from the exact bundled portrait PNG using the production
reference model and the shared Swift `SoulXImage.prepare` implementation:

```sh
apple/scripts/export-reference-cache.sh \
  --models artifacts/coreml/device/pro320 \
  --image ios/SoulXStudio/Resources/Avatars/indian/portrait.png \
  --output ios/SoulXStudio/Resources/Avatars/indian/reference-cache \
  --checkpoint models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth
```

Repeat with `girl` for the second fixed avatar. `--models` must contain the
profile's `manifest.json` and its `reference.mlmodelc` or `reference.mlpackage`.
The CLI defaults to CPU-only Core ML execution and records the actual host,
memory, runtime, source image/checkpoint/model hashes, and run date in metadata.
These are correctness exports, not GPU or iPhone throughput benchmarks.

Each directory contains `metadata.json` and `reference.f32le`, a contiguous
little-endian float32 tensor in `[1,16,9,H/8,W/8]` order. The runtime verifies the
profile, shape, byte count, original Wan checkpoint hash, source PNG hash,
tensor hash, and finite values. The prepared image tensor hash is recorded for
provenance; source-file identity is the cross-platform check because Core
Graphics resampling can vary slightly between macOS and iOS.

`initialize(image:referenceCacheURL:sourceImageURL:...)` requires the metadata
URL and exact bundled PNG URL together. With no cache URL, it runs the original
reference encoder. `--smoke` optionally reads
`Documents/Smoke/reference-cache/metadata.json` paired with
`Documents/Smoke/portrait.png` and records which initialization path was used.

Run format-integrity and numerical helper tests with
`ios/SoulXStudio/Tests/run.sh`.
