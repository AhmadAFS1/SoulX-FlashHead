#!/bin/bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"
relative_runtime=ios/SoulXStudio/Sources/Runtime
runtime_dir="$repo_root/$relative_runtime"
build_dir="$repo_root/artifacts/coreml/mac-smoke-bin"
sources=(SoulXEngine.swift SoulXModels.swift SoulXTensor.swift SoulXImage.swift SoulXAudio.swift SoulXReferenceCache.swift)
compiler_flags=(-O -whole-module-optimization -parse-as-library)
export SOULX_SWIFT_COMPILER="$(xcrun swiftc --version)"
export SOULX_GIT_REVISION="$(git -C "$repo_root" rev-parse HEAD)"
if [[ "${SOULX_BENCH_BASELINE:-}" == 1 ]]; then
    # The old Mac GPU audio graph emits non-finite features. Correct only its
    # encoder placement so the speed comparison uses the same valid speech.
    export SOULX_GIT_REVISION="$(git -C "$repo_root" rev-parse --verify 'fec3bf7^{commit}')"
    build_dir="$build_dir/speech-reference"
    runtime_dir="$build_dir/src"
    mkdir -p "$runtime_dir"
    for file in "${sources[@]}"; do
        git -C "$repo_root" show "$SOULX_GIT_REVISION:$relative_runtime/$file" > "$runtime_dir/$file"
    done
    python3 - "$runtime_dir/SoulXModels.swift" <<'PY'
from pathlib import Path
import sys
source = Path(sys.argv[1])
code = source.read_text()
line = "configuration.computeUnits = computeUnits"
if code.count(line) != 1:
    raise RuntimeError("Unexpected baseline Core ML configuration")
source.write_text(code.replace(line, 'configuration.computeUnits = name == "audio" ? .cpuOnly : computeUnits'))
PY
    compiler_flags+=(-D SOULX_BENCH_BASELINE)
else
    sources+=(SoulXAudioLookahead.swift)
fi
mkdir -p "$build_dir"
export SOULX_RUNTIME_ROOT="$runtime_dir"
source_paths=()
for file in "${sources[@]}"; do source_paths+=("$runtime_dir/$file"); done
xcrun swiftc "${compiler_flags[@]}" "${source_paths[@]}" "$script_dir/mac_smoke.swift" -o "$build_dir/soulx-mac-smoke"
exec "$build_dir/soulx-mac-smoke" "$@"
