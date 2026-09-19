#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"

if [[ ! -x "${python_bin}" ]]; then
  python_bin="${PYTHON_BIN:-python3}"
fi

exec "${python_bin}" "${repo_root}/scripts/rebuild_int8_decoder_engines.py" "$@"
