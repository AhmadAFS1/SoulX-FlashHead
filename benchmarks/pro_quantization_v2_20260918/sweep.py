"""Execute an interleaved v2 reference/candidate schedule sequentially."""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_FIXTURES,
    DEFAULT_GPU_LOCK,
    ROOT,
    atomic_write_json,
    ensure_new_directory,
    relative_path,
)


def make_schedule(seeds: list[int], repeats: int) -> list[dict[str, Any]]:
    """Balance A/B and B/A pairs before observing any timing result."""
    schedule = []
    sequence = 0
    pair = 0
    for seed in seeds:
        for repeat in range(repeats):
            order = ("reference", "candidate") if (pair % 2) == 0 else ("candidate", "reference")
            for role in order:
                schedule.append({"sequence": sequence, "seed": seed, "repeat": repeat, "role": role})
                sequence += 1
            pair += 1
    return schedule


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--reference-policy", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    parser.add_argument("--reference-pythonpath", help="Explicit reference dependency search path")
    parser.add_argument("--candidate-pythonpath", help="Explicit candidate dependency search path")
    parser.add_argument("--save-raw-first", action="store_true", help="Retain lossless RGB for the first pair of each seed")
    # Forwarded verbatim to BOTH roles so a paired sweep stays comparable: an
    # arm that differs in step count or delivery on only one side is not a
    # controlled comparison.
    parser.add_argument("--sampling-steps", type=int, choices=(1, 2, 3, 4),
                        help="Forwarded to both roles. DiT time is linear in this.")
    parser.add_argument("--timestep-variant", choices=("shipped", "distilled_aligned"),
                        help="Forwarded to both roles. QUALITY-AFFECTING.")
    parser.add_argument("--lean-delivery", action="store_true",
                        help="Forwarded to both roles.")
    parser.add_argument("--skip-zero-weighted-noise", action="store_true",
                        help="Forwarded to both roles. Changes the generator stream.")
    args = parser.parse_args()
    if args.repeats < 1 or args.frames < 1:
        parser.error("frames and repeats must be positive")
    output = ensure_new_directory(args.output)
    runs = output / "runs"
    runs.mkdir()
    schedule = make_schedule(args.seeds, args.repeats)
    result: dict[str, Any] = {
        "status": "scheduled",
        "execution": "sequential subprocess orchestration; each child acquires the shared GPU lease",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_policy": relative_path(args.policy),
        "reference_policy": relative_path(args.reference_policy),
        "fixture_id": args.fixture_id,
        "sampling_steps": args.sampling_steps,
        "timestep_variant": args.timestep_variant,
        "lean_delivery": args.lean_delivery,
        "skip_zero_weighted_noise": args.skip_zero_weighted_noise,
        "frames": args.frames,
        "schedule": schedule,
        "cells": [],
    }
    atomic_write_json(output / "schedule.json", result)
    for cell in schedule:
        policy = args.reference_policy if cell["role"] == "reference" else args.policy
        destination = runs / f"{cell['sequence']:03d}-{cell['role']}-seed{cell['seed']}-r{cell['repeat'] + 1:02d}"
        command = [
            str(args.python), "benchmarks/pro_quantization_v2_20260918/run.py",
            "--policy", str(policy), "--fixtures", str(args.fixtures), "--fixture-id", args.fixture_id,
            "--output", str(destination), "--seed", str(cell["seed"]), "--frames", str(args.frames),
            "--repeats", "1", "--gpu-lock", str(args.gpu_lock),
        ]
        if args.sampling_steps is not None:
            command += ["--sampling-steps", str(args.sampling_steps)]
        if args.timestep_variant is not None:
            command += ["--timestep-variant", args.timestep_variant]
        if args.lean_delivery:
            command.append("--lean-delivery")
        if args.skip_zero_weighted_noise:
            command.append("--skip-zero-weighted-noise")
        if args.save_raw_first and cell['repeat'] == 0:
            command.append('--save-raw')
        started = datetime.now(timezone.utc).isoformat()
        environment = os.environ.copy()
        pythonpath = getattr(args, cell["role"] + "_pythonpath")
        if pythonpath is not None:
            environment["PYTHONPATH"] = pythonpath
        completed = subprocess.run(command, cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        cell_result = {
            **cell,
            "command": command,
            "started_utc": started,
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "returncode": completed.returncode,
            "output": relative_path(destination),
            "pythonpath": environment.get("PYTHONPATH"),
        }
        (output / f"{cell['sequence']:03d}.stdout.log").write_text(completed.stdout, encoding="utf-8")
        (output / f"{cell['sequence']:03d}.stderr.log").write_text(completed.stderr, encoding="utf-8")
        if (destination / "results.json").is_file():
            child = json.loads((destination / "results.json").read_text(encoding="utf-8"))
            cell_result["child_status"] = child.get("status")
        else:
            cell_result["child_status"] = "missing-results"
        result["cells"].append(cell_result)
        result["status"] = "running" if completed.returncode == 0 and cell_result["child_status"] == "complete" else "failed"
        atomic_write_json(output / "schedule.json", result)
        if result["status"] == "failed":
            raise RuntimeError(f"Sweep cell failed: {cell_result}")
    result["status"] = "complete"
    atomic_write_json(output / "schedule.json", result)


if __name__ == "__main__":
    main()
