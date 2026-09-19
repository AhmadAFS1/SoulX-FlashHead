#!/usr/bin/env python3
"""Rebuild experimental INT8 Wan decoder engines from tracked plans/captures.

This is a small orchestration wrapper around the repository's existing
``build_decoder_engine.py`` implementation.  It deliberately creates a new
output directory and never overwrites an existing engine build.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "benchmarks" / "pro_quantization_v2_20260918"

VARIANTS = {
    "qdq-output": {
        "plan": EXPERIMENT / "runs/decoder-int8-qdq-output-r01/engine-plan.json",
        "captures": EXPERIMENT / "calibration/capture-r02/manifest.json",
    },
    "six": {
        "plan": EXPERIMENT / "runs/decoder-int8-six-target-build-r01/engine-plan.json",
        "captures": EXPERIMENT / "calibration/capture-r02/manifest.json",
    },
    "twelve": {
        "plan": EXPERIMENT / "runs/decoder-int8-twelve-target-build-r01/engine-plan.json",
        "captures": EXPERIMENT / "calibration/decoder-top12-capture-r01/manifest.json",
    },
}


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=tuple(VARIANTS),
        default="twelve",
        help="tracked plan/capture pair to rebuild (default: twelve)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="new directory for engines and diagnostics; it must not already exist",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable for the pinned Torch/TensorRT environment",
    )
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument(
        "--gpu-lock",
        default=str(ROOT / ".gpu-owner.lock"),
        help="shared GPU lease path",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the exact builder command without starting a GPU build",
    )
    args = parser.parse_args()

    if args.workspace_mib < 1:
        parser.error("--workspace-mib must be positive")

    selected = VARIANTS[args.variant]
    plan = selected["plan"]
    captures = selected["captures"]
    output = _path(args.output)
    gpu_lock = _path(args.gpu_lock)

    for required in (plan, captures):
        if not required.is_file():
            parser.error(f"required tracked input is missing: {required}")
    if output.exists():
        parser.error(f"refusing to overwrite existing output directory: {output}")

    builder = EXPERIMENT / "build_decoder_engine.py"
    command = [
        args.python,
        str(builder),
        "--plan",
        str(plan),
        "--captures",
        str(captures),
        "--precision",
        "int8",
        "--output",
        str(output),
        "--workspace-mib",
        str(args.workspace_mib),
        "--gpu-lock",
        str(gpu_lock),
    ]
    print("INT8 decoder rebuild")
    print(f"  variant:  {args.variant}")
    print(f"  plan:     {plan.relative_to(ROOT)}")
    print(f"  captures: {captures.relative_to(ROOT)}")
    print(f"  output:   {output}")
    print("  command:  " + " ".join(map(str, command)))
    if args.dry_run:
        return 0

    # The builder bootstraps imports from the repository root; keeping cwd
    # here also makes all relative paths in its diagnostics deterministic.
    completed = subprocess.run(command, cwd=ROOT)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
