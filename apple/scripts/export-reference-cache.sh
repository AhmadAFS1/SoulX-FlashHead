#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$script_dir/../.." && pwd)"
runtime_dir="$repo_dir/ios/SoulXStudio/Sources/Runtime"
build_dir="$(mktemp -d -t soulx-reference-export)"
trap 'rm -rf "$build_dir"' EXIT
xcrun swiftc -O "$runtime_dir/SoulXTensor.swift" "$runtime_dir/SoulXModels.swift" "$runtime_dir/SoulXAudio.swift" \
    "$runtime_dir/SoulXImage.swift" "$runtime_dir/SoulXReferenceCache.swift" \
    "$repo_dir/apple/tools/reference_cache/main.swift" -o "$build_dir/export-reference-cache"
"$build_dir/export-reference-cache" "$@"
