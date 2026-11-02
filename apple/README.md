# SoulX FlashHead PRO on Apple devices

This additive implementation converts the original PRO checkpoint to Core ML and
runs a native still-image-to-talking-video pipeline in **SoulX Studio**. CUDA
entry points, dependencies, quantization kernels and serving behavior are untouched.

The iPhone profile is 320 pixels high × 176 pixels wide, with 33 frames per denoising window, five history
frames, 28 new frames, and two denoising evaluations at timesteps 1000 and 625.
Output has a 25 fps timebase. Actual generation throughput must be measured; this
profile is not a promise of real-time operation on A14.

The earlier iPhone 12 Pro build on A14 / iOS 26.2 measured **0.81–0.90 generated FPS** after
initial model loading at 176×320. The optimized runtime has been measured on the
Apple M5 Pro Mac. With automatic Core ML selection, warm throughput improved
from about 9 FPS to 15 FPS with pixel-identical output on both avatars.
[Runtime benchmarks and speech validation](PERFORMANCE.md)
This is a functioning local prototype, not real-time. Both requested avatars are
installed; fixed-avatar initialization uses verified cached Wan features. See
[the validation record](VALIDATION.md) for the measurements and limitations.

## Build the models

Use a separate environment on an Apple Silicon Mac with Xcode command-line tools:

```sh
make -f apple/Makefile setup
make -f apple/Makefile fetch
make -f apple/Makefile test
make -f apple/Makefile export
make -f apple/Makefile bundle
```

Run from the repository root. `uv` manages the isolated `.venv-coreml` environment.
NumPy 2.2.6 is pinned because Core ML Tools 9's Torch frontend still uses scalar
conversions removed by newer NumPy. Source downloads use pinned revisions and
verify the PRO, Wan and tiny autoencoder checkpoint hashes. Allow at least 20 GB
of free space for source checkpoints, model packages, compiled models and caches.
Conversion of the original reference encoder can take several minutes on CPU.

The complete import folder is `artifacts/coreml/device/pro320`. `bundle.py` checks
every input and output shape/name, verifies all 30 transformer layers are present
across partitions, and writes the runtime manifest only after assembling all
required models. Exported weights are ignored by Git.

The 576-pixel-high profile is retained for experimentation on devices with more
memory. Its monolithic reference encoder did not complete the initial A14 test,
which last recorded about 3.2 GB of app footprint. Export it explicitly with:

```sh
make -f apple/Makefile export HEIGHT=576 WIDTH=320 OUTPUT=artifacts/coreml/pro576
make -f apple/Makefile bundle OUTPUT=artifacts/coreml/pro576 DEVICE_OUTPUT=artifacts/coreml/device/pro576
```

The transformer exporter supports `--resume`, `--only`, `--compression none`,
`--compression int8`, `--compression lut4`, `--compression lut6`, and exact
`--query-chunk` tiling. Alternative compression and resolution exports require
fresh numerical and device validation; they are not certified by a successful
conversion. Use a separate output folder when changing the profile or policy.

## Model boundaries and runtime

| Component | Purpose |
| --- | --- |
| `reference` | Original Wan causal encoder; prepares cached reference features for the fixed avatars |
| `audio` | Wav2Vec2 eight-second rolling audio context; includes waveform normalization and interpolation before the transformer |
| `conditioning` | PRO audio projection into nine groups of 32 context tokens |
| `patch` | Projects current latent and reference latent into 1,980 tokens (6,480 for pro576) |
| `blocks_00`… | All 30 original PRO transformer blocks, partitioned for memory management |
| `head` | Produces the 16-channel denoising flow tensor |
| `decoder` | Wan-compatible TAEHV `taew2_1` video decoder |
| `encoder` | TAEHV motion encoder; feeds back the last five color-corrected frames |

The Swift actor owns temporal state and model execution. It restores the first
one/two latent history frames before each denoising step, follows the original
stochastic schedule, corrects decoded frames in Lab color space, then updates
motion history. Audio is resampled to mono 16 kHz. A bounded playback queue follows
the audio clock and pauses during underruns. Cancellation is checked between
model calls; an individual Core ML prediction cannot be interrupted safely.

The initial decoder uses **window mode**: each 33-frame window starts with zero
TAEHV memory. It does not claim the carried-state optimization of the fastest CUDA
benchmark. The tiny VAE is an existing experimental PRO variant, not the original
full Wan decoder. Image resize interpolation and Swift's deterministic Gaussian
RNG differ from Python/Pillow/CUDA, so equal seeds do not imply pixel-identical
end-to-end videos across platforms.

## Fixed avatars and initialization

SoulX Studio bundles the user-selected **Indian-man portrait** and the repository's
`examples/girl.png` (**Asian** in the picker). Their original PNG checksums and
repository provenance are recorded in the app resources. The photo picker was
removed at the user's request.

The monolithic Wan encoder exceeded the iPhone's memory budget in both attempted
profiles. The two avatars therefore ship with original Wan reference tensors
computed once on the Mac using the validated FP32 Core ML encoder. At startup the
phone verifies the image hash, cache hash, original Wan checkpoint identity,
profile, shape and finite tensor values. These caches contain conditioning
features only: **no generated frames or pre-rendered video**. Audio conditioning,
all 30 PRO blocks, decoding, color correction and recurrent motion feedback run
on the device. The generic on-device reference encoder remains available for
diagnostics and larger devices.

Rebuild a reference cache with `apple/scripts/export-reference-cache.sh`; see
[the cache export instructions](tools/reference_cache/README.md). Use the exact bundled PNG and the matching profile. The current
bundled avatar caches are for pro320. Importing another resolution requires
matching avatar caches; mismatches are rejected.

## Quantization and Neural Engine

CUDA's mixed FP8/INT8 activation kernels cannot be imported as Core ML operators.
This path starts from the same original PRO weights, uses Apple weight compression,
and validates its arithmetic independently. Attention and normalization in the DiT stay FP32 after full-shape tests exposed errors in all-FP16 conversion. The audio network and original reference encoder use FP32 because tested reduced-precision conversions exceeded their numerical error gates. It is a separate Apple quantization
implementation, not a byte-identical execution of the RTX kernels. Weight
compression reduces package size; it does not by itself prove INT8 arithmetic on
hardware or increased inference speed.

The app defaults to Core ML's `.all` compute units and also exposes CPU/GPU and
CPU/Neural Engine configurations. Core ML decides placement. Requesting Neural
Engine support is not proof that every operation runs there; unsupported
operations may use CPU, and actual placement requires device profiling. Simulator
runs verify app behavior, not A14 Neural Engine performance.

## App and validation

See [the iOS app guide](../ios/SoulXStudio/README.md) for installation, model import,
and `--smoke` physical-device validation. Use Release builds for timing. With the signed app installed, a reliable per-file
transfer is available (use your device identifier from `xcrun devicectl list devices`):

```sh
.venv-coreml/bin/python apple/deploy_device.py --device YOUR_DEVICE_ID
```

If a transfer of this exact bundle is interrupted, add `--resume`. It checks byte
lengths before skipping files and verifies the complete transfer afterward. Omit
`--resume` after regenerating weights; equal byte lengths are not content hashes.
Models are installed into `Documents/Models/current`.

- `apple/tests/test_dit.py`: mathematical parity against upstream PRO modules.
- `apple/tests/test_support.py`: actual-checkpoint audio/Wan/tiny encoder parity.
- `flashhead_coreml.validate_dit`: Core ML versus PyTorch component checks, with
  explicit synthetic or real-checkpoint provenance.
- `SoulXSmoke`: persists stage diagnostics and first/last generated PNG frames
  from a finite on-device window.

A converted graph, a clean build and a passing component test are separate gates.
Only a completed real model run establishes that a profile fits and executes on
a particular device; sustained generation and animation quality need further
multi-window validation.
