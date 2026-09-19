"""Preregister and execute recurrent/new-utterance pairs for the selected pilot."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.pro_quantization_v2_20260918.common import (
    ROOT,
    atomic_write_json,
    ensure_new_directory,
    environment_manifest,
    sha256,
    utc_now,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--standard-reviews", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    output = ensure_new_directory(args.output).resolve()
    root = ROOT / "benchmarks/pro_30fps_20260919"
    fixtures = root / "fixtures.json"
    policies = {
        "reference": ROOT
        / "benchmarks/pro_quantization_v2_20260918/policies/self_all_fp8_sage2_fp16.json",
        "candidate": root / "policies/combined_fp16_sage_fp8.json",
    }
    dependencies = {
        "reference": ".pro-sage2-deps:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.",
        "candidate": "/workspace/experiments/pro30-deps/sage-sm89:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.",
    }
    tests = [
        ("recurrence", "indian150-a", 50, 1500),
        ("two-turn-gap", "indian150-two-turns-gap", 51, 576),
        ("audio-b", "indian150-audio-b", 0, 250),
        ("sichuan", "indian150-sichuan-prefix", 1, 250),
    ]
    result = {
        "status": "scheduled",
        "date_utc": utc_now(),
        "environment": environment_manifest(),
        "execution": "sequential paired subprocess validation; child manifests identify CPU/GPU work",
        "fixtures_sha256": sha256(fixtures),
        "policy_hashes": {k: sha256(v) for k, v in policies.items()},
        "tests": tests,
        "cells": [],
        "limitations": [
            "Offline finite-length recurrence; not five-minute bounded serving qualification."
        ],
    }
    path = output / "schedule.json"
    atomic_write_json(path, result)

    def run(command, name, environment):
        begin = utc_now()
        with (output / (name + ".log")).open("w") as log:
            process = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        result["cells"].append(
            {
                "name": name,
                "command": command,
                "begin_utc": begin,
                "end_utc": utc_now(),
                "returncode": process.returncode,
            }
        )
        result["status"] = "running" if process.returncode == 0 else "failed"
        atomic_write_json(path, result)
        if process.returncode:
            raise RuntimeError(f"Validation failed: {name}; see retained log")

    reviews = [str(p.resolve()) for p in args.standard_reviews]
    for index, (name, fixture, seed, frames) in enumerate(tests):
        for role in (
            ("reference", "candidate") if index % 2 == 0 else ("candidate", "reference")
        ):
            environment = os.environ.copy()
            environment["PYTHONPATH"] = dependencies[role]
            command = [
                sys.executable,
                str(ROOT / "benchmarks/pro_quantization_v2_20260918/run.py"),
                "--policy",
                str(policies[role]),
                "--fixtures",
                str(fixtures),
                "--fixture-id",
                fixture,
                "--seed",
                str(seed),
                "--frames",
                str(frames),
                "--repeats",
                "1",
                "--output",
                str(output / (name + "-" + role)),
            ]
            if frames <= 250:
                command.append("--save-raw")
            run(command, name + "-" + role, environment)
        review = output / (name + "-review")
        command = [
            sys.executable,
            str(ROOT / "benchmarks/pro_quantization_v2_20260918/review.py"),
            "--baseline",
            str(output / (name + "-reference")),
            "--candidate",
            str(output / (name + "-candidate")),
            "--baseline-label",
            "PRO V2 FP8",
            "--candidate-label",
            "PRO FP8 + FP16 stages",
            "--output",
            str(review),
        ]
        run(command, name + "-review", os.environ.copy())
        reviews.append(str(review))
    for mode in ("own", "shared"):
        command = [
            sys.executable,
            str(ROOT / "benchmarks/pro_quantization_v2_20260918/sync_check.py"),
            "--reviews",
            *reviews,
            "--crop-mode",
            mode,
            "--output",
            str(output / ("sync-" + mode + ".json")),
        ]
        run(command, "sync-" + mode, os.environ.copy())
    result.update(status="complete", reviews=reviews)
    atomic_write_json(path, result)


if __name__ == "__main__":
    main()
