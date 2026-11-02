#!/bin/bash
set -euo pipefail
test_dir="$(cd "$(dirname "$0")" && pwd)"
runtime_dir="$test_dir/../Sources/Runtime"
build_dir="$(mktemp -d -t soulx-swift-checks)"
trap 'rm -rf "$build_dir"' EXIT
date -u '+UTC %Y-%m-%dT%H:%M:%SZ'
echo 'Classification: CPU-only Swift helper correctness; no model/GPU/ANE/iPhone inference.'
echo 'Hardware evidence: system_profiler SPDisplaysDataType and sysctl hw.memsize. Apple unified memory is shared; no dedicated VRAM. Co-resident load unknown.'
system_profiler SPDisplaysDataType | sed -n '/Chipset Model:/p; /Total Number of Cores:/p; /Metal Support:/p'
sysctl hw.memsize
sw_vers
xcrun swiftc --version
xcrun swiftc -O "$runtime_dir/SoulXTensor.swift" "$runtime_dir/SoulXModels.swift" \
    "$runtime_dir/SoulXImage.swift" "$runtime_dir/SoulXAudio.swift" "$runtime_dir/SoulXReferenceCache.swift" \
    "$test_dir/main.swift" -o "$build_dir/check"
"$build_dir/check"
