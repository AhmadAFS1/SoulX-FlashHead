#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
runtime_dir="$repo_root/ios/SoulXStudio/Sources/Runtime"
build_dir="$repo_root/artifacts/coreml/mac-smoke-bin"
mkdir -p "$build_dir"
export SOULX_SWIFT_COMPILER="$(xcrun swiftc --version)"
export SOULX_GIT_REVISION="$(git -C "$repo_root" rev-parse HEAD)"
export SOULX_RUNTIME_ROOT="$runtime_dir"
xcrun swiftc -O -whole-module-optimization -parse-as-library \
    "$runtime_dir/SoulXEngine.swift" "$runtime_dir/SoulXModels.swift" \
    "$runtime_dir/SoulXTensor.swift" "$runtime_dir/SoulXImage.swift" \
    "$runtime_dir/SoulXAudio.swift" "$script_dir/mac_smoke.swift" \
    -o "$build_dir/soulx-mac-smoke"
exec "$build_dir/soulx-mac-smoke" "$@"
