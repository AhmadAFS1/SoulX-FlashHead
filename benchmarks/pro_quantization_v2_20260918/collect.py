"""Collect only complete, comparable v2 run manifests without pooling frames."""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.pro_quantization_v2_20260918.common import atomic_write_json, relative_path, summarize


def _profile_key(run: dict[str, Any]) -> tuple[Any, ...]:
    profile = run.get("profile", {})
    return tuple(profile.get(name) for name in ("width", "height", "fps", "steps", "shift", "motion_latents", "motion_frames"))


def _row(path: Path, run: dict[str, Any]) -> dict[str, Any]:
    values = [item["useful_fps"] for item in run.get("runs", [])]
    return {
        "run": relative_path(path.parent),
        "policy": run.get("policy", {}).get("name"),
        "fixture": run.get("fixture", {}).get("id"),
        "seed": run.get("profile", {}).get("seed"),
        "profile": run.get("profile"),
        "profile_key": list(_profile_key(run)),
        "environment": run.get("environment", {}).get("gpu"),
        "repetitions": len(values),
        "useful_fps": values,
        "useful_fps_summary": summarize(values),
        "first_window_latency_s": [item.get("first_window_latency_s") for item in run.get("runs", [])],
        "chunk_distributions_s": [item.get("chunk_distribution_s") for item in run.get("runs", [])],
        "peak_allocated_mib": max((item.get("peak_allocated_mib", 0) for item in run.get("runs", [])), default=None),
        "peak_reserved_mib": max((item.get("peak_reserved_mib", 0) for item in run.get("runs", [])), default=None),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    rows, excluded = [], []
    for path in sorted(args.root.rglob("results.json")):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run.get("status") != "complete":
            excluded.append({"run": relative_path(path.parent), "status": run.get("status"), "reason": "incomplete-or-failed"})
            continue
        if run.get("execution") != "fresh local GPU inference":
            excluded.append({"run": relative_path(path.parent), "status": run.get("status"), "reason": "not-a-generation-run"})
            continue
        rows.append(_row(path, run))
    result = {
        "status": "complete",
        "execution": "CPU-only aggregation of complete fresh GPU generation manifests",
        "rows": rows,
        "excluded": excluded,
        "profile_groups": {},
    }
    for row in rows:
        key = json.dumps(row["profile_key"])
        result["profile_groups"].setdefault(key, []).append(row["run"])
    atomic_write_json(args.output, result)
    print(json.dumps({"rows": len(rows), "excluded": len(excluded), "output": str(args.output)}))


if __name__ == "__main__":
    main()