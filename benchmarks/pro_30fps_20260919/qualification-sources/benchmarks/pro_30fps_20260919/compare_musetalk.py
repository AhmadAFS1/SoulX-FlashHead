"""Fresh alternating PRO / native MuseTalk comparison with explicit boundaries."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_FIXTURES,
    DEFAULT_GPU_LOCK,
    ROOT,
    atomic_write_json,
    ensure_new_directory,
    relative_path,
    sha256,
    snapshot_sources,
    utc_now,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--fixture-id", default="indian150-a")
    parser.add_argument("--avatar-video", type=Path, required=True)
    parser.add_argument("--pro-pythonpath", required=True)
    parser.add_argument(
        "--muse-python",
        type=Path,
        default=Path("/workspace/.venvs/musetalk_trt_stagewise/bin/python"),
    )
    parser.add_argument("--muse-pythonpath", default=str(ROOT))
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.pairs < 3:
        parser.error("Matched qualification requires at least three pairs")
    output = ensure_new_directory(args.output).resolve()
    schedule = [
        {"pair": i, "role": role}
        for i in range(args.pairs)
        for role in (("pro", "musetalk") if i % 2 == 0 else ("musetalk", "pro"))
    ]
    result = {
        "status": "scheduled",
        "date_utc": utc_now(),
        "schedule": schedule,
        "execution": "fresh sequential alternating subprocess GPU benchmarks",
        "musetalk_profile": "native PyTorch V1.5 FP16 batch 8; not tuned TensorRT serving",
        "policy": relative_path(args.policy),
        "policy_sha256": sha256(args.policy),
        "avatar_video": relative_path(args.avatar_video),
        "avatar_sha256": sha256(args.avatar_video),
        "cells": [],
        "quality_qualified": False,
        "comparison_protocol": 2,
    }
    result["source_sha256"] = snapshot_sources(
        output,
        [
            Path(__file__),
            ROOT / "benchmarks/pro_quantization_v2_20260918/compare_musetalk.py",
        ],
    )
    atomic_write_json(output / "schedule.json", result)
    last_pro = None
    pairs = {}
    for index, cell in enumerate(schedule):
        destination = output / f"{index:02d}-{cell['role']}-pair{cell['pair']}"
        environment = os.environ.copy()
        if cell["role"] == "pro":
            environment["PYTHONPATH"] = args.pro_pythonpath
            command = [
                sys.executable,
                str(ROOT / "benchmarks/pro_quantization_v2_20260918/run.py"),
                "--policy",
                str(args.policy.resolve()),
                "--seed",
                str(args.seed),
            ]
        else:
            environment["PYTHONPATH"] = args.muse_pythonpath
            command = [
                str(args.muse_python),
                str(
                    ROOT / "benchmarks/pro_quantization_v2_20260918/compare_musetalk.py"
                ),
                "--pro-run",
                str(last_pro),
                "--musetalk-avatar-video",
                str(args.avatar_video.resolve()),
                "--batch",
                "8",
            ]
        command += [
            "--fixtures",
            str(args.fixtures.resolve()),
            "--fixture-id",
            args.fixture_id,
            "--frames",
            str(args.frames),
            "--repeats",
            "1",
            "--output",
            str(destination),
            "--gpu-lock",
            str(DEFAULT_GPU_LOCK),
        ]
        started = utc_now()
        with (output / f"{index:02d}.log").open("w") as log:
            completed = subprocess.run(
                command,
                cwd=ROOT,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        record = cell | {
            "command": command,
            "started_utc": started,
            "ended_utc": utc_now(),
            "returncode": completed.returncode,
            "output": relative_path(destination),
        }
        result["cells"].append(record)
        child_path = destination / "results.json"
        child = json.loads(child_path.read_text()) if child_path.exists() else {}
        if completed.returncode or child.get("status") != "complete":
            result["status"] = "failed"
            atomic_write_json(output / "schedule.json", result)
            raise RuntimeError(f"Matched comparison failed at {record}")
        if cell["role"] == "pro":
            last_pro = destination
            generation = child["runs"][0]["generation_s"]
            delivery = generation + child["encode_mux_s"]
        else:
            generation = child["musetalk_request_summary"]["median_s"]
            delivery = child["musetalk_application_delivery"]["seconds"]
        record.update(
            generation_s=generation,
            delivery_s=delivery,
            useful_fps=args.frames / generation,
            delivered_fps=args.frames / delivery,
            gpu_uuids=[d["uuid"] for d in child["environment"]["gpu"]["devices"]],
            audio_sha256=child["fixture"]["audio"]["sha256"],
            effective_audio_sha256=child["effective_audio"]["sha256"],
        )
        pairs.setdefault(cell["pair"], {})[cell["role"]] = record
        result["status"] = "running"
        atomic_write_json(output / "schedule.json", result)
    for pair in pairs.values():
        if (
            pair["pro"]["gpu_uuids"] != pair["musetalk"]["gpu_uuids"]
            or pair["pro"]["audio_sha256"] != pair["musetalk"]["audio_sha256"]
            or pair["pro"]["effective_audio_sha256"]
            != pair["musetalk"]["effective_audio_sha256"]
        ):
            raise ValueError("Mismatched hardware or audio in comparison")
    result.update(
        status="complete",
        timing_comparable=True,
        pro_median_fps=statistics.median(
            p["pro"]["useful_fps"] for p in pairs.values()
        ),
        musetalk_median_fps=statistics.median(
            p["musetalk"]["useful_fps"] for p in pairs.values()
        ),
        paired_pro_over_musetalk=[
            p["pro"]["useful_fps"] / p["musetalk"]["useful_fps"] for p in pairs.values()
        ],
        limitations=[
            "Different native neural workloads; equal delivered canvas",
            "Native PyTorch MuseTalk; no claim against its strongest TensorRT service",
            "Avatar/model preparation excluded; audio, rendering and RGB delivery included",
            "Quality qualification is separate from this timing comparison",
        ],
    )
    atomic_write_json(output / "schedule.json", result)


if __name__ == "__main__":
    main()
