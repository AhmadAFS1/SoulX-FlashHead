"""Create a conservative decoder calibration plan from measured v2 artifacts."""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
from collections import defaultdict
import fnmatch
import json
from pathlib import Path
from typing import Any

import torch

from benchmarks.pro_quantization_v2_20260918.common import atomic_write_json, relative_path, sha256
from soulx_rtc.pro_vae_quantization import DecoderPlan, DecoderPlanError, DecoderTarget, fit_symmetric_scale


def load_decoder_inventory(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    rows = raw.get("modules")
    if not isinstance(rows, list):
        raise ValueError("Decoder inventory must contain a modules list")
    required = {"path", "class", "inclusive_cuda_ms", "observed_inputs", "observed_outputs"}
    if any(not isinstance(row, dict) or required - set(row) for row in rows):
        raise ValueError("Decoder inventory has an incomplete module row")
    return rows


def _spatial_area(shapes: list[list[int]]) -> int:
    return max((shape[-2] * shape[-1] for shape in shapes if len(shape) >= 4), default=0)


def protected_paths(rows: list[dict[str, Any]], patterns: list[str]) -> set[str]:
    max_area = max((_spatial_area(row["observed_outputs"]) for row in rows), default=0)
    result = set()
    for row in rows:
        path = row["path"]
        late_or_entry = path.endswith(".conv1") or ".head." in path
        full_resolution = _spatial_area(row["observed_outputs"]) == max_area
        explicit = any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)
        if late_or_entry or full_resolution or explicit:
            result.add(path)
    return result


def _captured_inputs(manifest: Path) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if raw.get("status") != "complete":
        raise ValueError("Calibration plan requires a complete capture manifest")
    entries: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    for item in raw.get("raw_tensors", []):
        if item.get("kind") != "causal_conv3d_prepared_input":
            continue
        path = manifest.parent / item["path"]
        if not path.is_file() or sha256(path) != item.get("sha256"):
            raise ValueError(f"Prepared decoder capture is missing or modified: {path}")
        entries[item["module"]].append((path, item))
    return entries


def build_plan(
    inventory: list[dict[str, Any]],
    captures: dict[str, list[tuple[Path, dict[str, Any]]]],
    *,
    name: str,
    scheme: str,
    max_targets: int,
    min_inclusive_cuda_ms: float,
    protect: list[str],
) -> DecoderPlan:
    if scheme not in ("int8_conservative", "fp8_conservative"):
        raise DecoderPlanError("Calibration plan scheme must be int8_conservative or fp8_conservative")
    protected = protected_paths(inventory, protect)
    candidates = [
        row for row in inventory
        if row["class"] == "CausalConv3d"
        and row["path"] not in protected
        and row["path"] in captures
        and float(row["inclusive_cuda_ms"]) >= min_inclusive_cuda_ms
    ]
    candidates.sort(key=lambda row: float(row["inclusive_cuda_ms"]), reverse=True)
    selected = candidates[:max_targets]
    if not selected:
        raise DecoderPlanError(
            "No captured, non-protected CausalConv3d targets satisfy the measured cost gate; do not invent a decoder plan"
        )
    quant_max = 127.0 if scheme.startswith("int8") else 448.0
    targets = []
    for row in selected:
        saved = captures[row["path"]]
        tensors = [torch.load(path, map_location="cpu", weights_only=True) for path, _ in saved]
        calibration = fit_symmetric_scale(tensors, quant_max=quant_max)
        calibration.update({
            "scale_granularity": "per_tensor",
            "quant_max": quant_max,
            "capture_count": len(saved),
            "capture_sha256": [item["sha256"] for _, item in saved],
            "source_inclusive_cuda_ms": float(row["inclusive_cuda_ms"]),
        })
        shapes = tuple(sorted({tuple(tensor.shape) for tensor in tensors}))
        targets.append(DecoderTarget(row["path"], "causal_conv3d", shapes, calibration))
    return DecoderPlan(
        name,
        scheme,
        tuple(targets),
        tuple(sorted(protected)),
        {
            "inventory_selection": "cost-ranked CausalConv3d targets with captured prepared inputs",
            "max_targets": max_targets,
            "min_inclusive_cuda_ms": min_inclusive_cuda_ms,
            "protected_patterns": protect,
        },
        {},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--scheme", choices=("int8_conservative", "fp8_conservative"), required=True)
    parser.add_argument("--name")
    parser.add_argument("--max-targets", type=int, default=4)
    parser.add_argument("--min-inclusive-cuda-ms", type=float, default=0.10)
    parser.add_argument("--protect", nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.max_targets < 1 or args.min_inclusive_cuda_ms < 0:
        parser.error("output must be new; max-targets must be positive; min-inclusive-cuda-ms cannot be negative")
    plan = build_plan(
        load_decoder_inventory(args.inventory), _captured_inputs(args.captures),
        name=args.name or args.scheme.replace("_", "-"), scheme=args.scheme,
        max_targets=args.max_targets, min_inclusive_cuda_ms=args.min_inclusive_cuda_ms,
        protect=args.protect,
    )
    output = plan.to_dict()
    output["source"] = {
        "inventory": relative_path(args.inventory), "inventory_sha256": sha256(args.inventory),
        "captures": relative_path(args.captures), "captures_sha256": sha256(args.captures),
        **dict(plan.source),
    }
    atomic_write_json(args.output, output)
    print(json.dumps({"output": str(args.output), "targets": [item.path for item in plan.targets], "protected": list(plan.protected)}))


if __name__ == "__main__":
    main()