"""Validate all scheduled observations before summarizing the paired sweep."""

import argparse
import json
import statistics
from pathlib import Path

from benchmarks.pro_quantization_v2_20260918.common import (
    ROOT,
    atomic_write_json,
    sha256,
    utc_now,
)


def distribution(values):
    return {
        "n": len(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True)
    args = parser.parse_args()
    schedule = json.loads((args.sweep / "schedule.json").read_text())
    if schedule["status"] != "complete" or len(schedule["cells"]) != len(
        schedule["schedule"]
    ):
        raise ValueError("Sweep is incomplete")
    rows, paired, fingerprints = [], {}, set()
    environment = None
    for cell in schedule["cells"]:
        path = ROOT / cell["output"] / "results.json"
        result = json.loads(path.read_text())
        if (
            cell["returncode"]
            or result["status"] != "complete"
            or len(result["runs"]) != 1
        ):
            raise ValueError(f"Invalid observation: {path}")
        environment = environment or result["environment"]
        profile = {
            k: v for k, v in result["profile"].items() if k not in ("policy", "seed")
        }
        fingerprints.add(
            json.dumps(
                {
                    "profile": profile,
                    "sources": result["source_sha256"],
                    "weights": result["checkpoint_sha256"],
                    "audio": result["effective_audio"],
                    "image": result["fixture"]["reference"],
                    "gpu_uuids": [
                        v["uuid"] for v in result["environment"]["gpu"]["devices"]
                    ],
                },
                sort_keys=True,
            )
        )
        run = result["runs"][0]
        chunks = run["chunk_times_s"]
        row = {
            "role": cell["role"],
            "seed": cell["seed"],
            "repeat": cell["repeat"],
            "run": cell["output"],
            "results_sha256": sha256(path),
            "fps": run["useful_fps"],
            "generation_s": run["generation_s"],
            "encode_mux_s": result["encode_mux_s"],
            "chunk_distribution_s": run["chunk_distribution_s"],
            "missed_25fps_window_deadlines": sum(t > 28 / 25 for t in chunks),
            "windows": len(chunks),
            "peak_allocated_mib": run["peak_allocated_mib"],
            "peak_reserved_mib": run["peak_reserved_mib"],
            "device_peak_mib": run["resource_summary"]["gpu_vram_used_mib"]["max"],
            "stage_seconds": run["stage_seconds"],
        }
        key = (cell["seed"], cell["repeat"])
        if cell["role"] in paired.setdefault(key, {}):
            raise ValueError("Duplicate paired observation")
        paired[key][cell["role"]] = row
        rows.append(row)
    if len(fingerprints) != 1:
        raise ValueError(
            "Hardware, profile, sources or effective inputs changed within sweep"
        )
    ratios = []
    for (seed, repeat), pair in paired.items():
        if set(pair) != {"reference", "candidate"}:
            raise ValueError("Unpaired observation")
        ratios.append(
            {
                "seed": seed,
                "repeat": repeat,
                "candidate_over_reference": pair["candidate"]["fps"]
                / pair["reference"]["fps"],
            }
        )
    seeds = sorted({r["seed"] for r in rows})
    per_seed = {
        str(seed): {
            role: distribution(
                [r["fps"] for r in rows if r["seed"] == seed and r["role"] == role]
            )
            for role in ("reference", "candidate")
        }
        for seed in seeds
    }
    output = {
        "status": "complete",
        "date_utc": utc_now(),
        "environment": environment,
        "execution": "CPU aggregation of fresh sequential GPU observations; no new inference",
        "source": str(args.sweep / "schedule.json"),
        "excluded": [],
        "observations": rows,
        "per_seed_fps": per_seed,
        "paired_ratios": ratios,
        "overall_fps": {
            role: distribution([r["fps"] for r in rows if r["role"] == role])
            for role in ("reference", "candidate")
        },
        "paired_speedup": distribution([r["candidate_over_reference"] for r in ratios]),
        "all_pairs_at_least_5_percent_faster": all(
            r["candidate_over_reference"] >= 1.05 for r in ratios
        ),
        "all_seed_medians_at_least_30fps": all(
            v["candidate"]["median"] >= 30 for v in per_seed.values()
        ),
        "quality_qualified": False,
        "limitations": [
            "Quality and recurrence require separate review; no automatic promotion.",
            "Three pairs per seed; observed ranges are not population confidence intervals.",
            "Chunk and frame observations are dependent; not independent timing replicates.",
        ],
    }
    atomic_write_json(args.sweep / "acceptance-summary.json", output)
    print(
        json.dumps(
            {k: output[k] for k in ("overall_fps", "paired_speedup", "per_seed_fps")},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
