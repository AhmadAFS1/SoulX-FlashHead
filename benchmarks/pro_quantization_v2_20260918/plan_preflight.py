#!/usr/bin/env python3
"""Check a TensorRT stage plan before acquiring the GPU.

``install_stage_plan`` validates a plan only once the model is loaded and the
GPU lease is held, and it raises a bare ValueError. On a leased box that wastes
the whole setup -- weight load, quantization, compilation -- before telling you
the plan is stale. This runs the same checks first, on CPU, in milliseconds.

It mirrors the real gate exactly:

  * ``flash_head/wan/modules/vae.py`` is the ONLY hard gate. A mismatch means
    the engines were exported from a different decoder implementation and
    ``install_stage_plan`` will refuse them. Exit code 1.
  * Every other entry in ``source_sha256`` is ADVISORY. Drift there does not
    stop the engines loading, but it does mean the run is no longer
    byte-reproducible against the plan's recorded provenance.
  * The VAE checkpoint is checked when present and reported as
    ``unavailable-locally`` when it is not, so this runs on a machine with no
    model weights.

Stdlib only -- no torch, no tensorrt, no CUDA. Importing this module must never
pull in the inference stack.

GPU/evidence note: `[CPU]` static hash check. It proves nothing about speed.

Usage:
    python benchmarks/pro_quantization_v2_20260918/plan_preflight.py \\
        benchmarks/pro_30fps_20260919/stage-fp16-build-r01/results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# The one source file install_stage_plan compares before installing engines.
ENGINE_GATING_SOURCE = "flash_head/wan/modules/vae.py"
VAE_CHECKPOINT = "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def check_plan(plan_path: Path, root: Path = ROOT) -> dict:
    """Return a structured verdict. Never raises for ordinary drift."""
    report: dict = {
        "plan": str(plan_path),
        "schema_ok": False,
        "status_ok": False,
        "engine_gate": None,
        "advisory_drift": [],
        "missing_sources": [],
        "checkpoint": None,
        "errors": [],
        "ok": False,
    }

    try:
        plan = json.loads(plan_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        report["errors"].append(f"cannot read plan: {error}")
        return report

    report["schema_ok"] = plan.get("schema_version") == 1
    report["status_ok"] = plan.get("status") == "complete"
    if not report["schema_ok"]:
        report["errors"].append(f"unsupported schema_version {plan.get('schema_version')!r}")
    if not report["status_ok"]:
        report["errors"].append(f"plan status is {plan.get('status')!r}, not 'complete'")

    sources = plan.get("source_sha256") or {}
    if not sources:
        report["errors"].append("plan carries no source_sha256 manifest")

    recorded = sources.get(ENGINE_GATING_SOURCE)
    gating_path = root / ENGINE_GATING_SOURCE
    if recorded is None:
        report["engine_gate"] = "absent-from-plan"
        report["errors"].append(f"plan does not record {ENGINE_GATING_SOURCE}")
    elif not gating_path.is_file():
        report["engine_gate"] = "source-missing"
        report["errors"].append(f"{ENGINE_GATING_SOURCE} is not in the tree")
    else:
        actual = sha256(gating_path)
        if actual == recorded:
            report["engine_gate"] = "match"
        else:
            report["engine_gate"] = "drift"
            report["errors"].append(
                f"{ENGINE_GATING_SOURCE} drifted (plan {recorded[:16]}..., "
                f"disk {actual[:16]}...); install_stage_plan will refuse these engines"
            )

    for relative, digest in sorted(sources.items()):
        if relative == ENGINE_GATING_SOURCE:
            continue
        path = root / relative
        if not path.is_file():
            report["missing_sources"].append(relative)
        elif sha256(path) != digest:
            report["advisory_drift"].append(relative)

    checkpoint = root / VAE_CHECKPOINT
    expected = plan.get("weights_sha256")
    if not checkpoint.is_file():
        report["checkpoint"] = "unavailable-locally"
    elif expected is None:
        report["checkpoint"] = "absent-from-plan"
    elif sha256(checkpoint) == expected:
        report["checkpoint"] = "match"
    else:
        report["checkpoint"] = "drift"
        report["errors"].append("VAE checkpoint does not match the plan's weights_sha256")

    # Mirror install_stage_plan's workspace validation.
    sizes = [
        record.get("workspace_bytes")
        for group in (plan.get("stages") or {}).values()
        for record in group.values()
    ]
    if not sizes or any(not isinstance(size, int) or size < 0 for size in sizes):
        report["errors"].append("plan has invalid stage workspace requirements")
    else:
        report["shared_workspace_mib"] = round(max(sizes) / (1 << 20), 3)
        report["summed_workspace_mib"] = round(sum(sizes) / (1 << 20), 3)

    report["ok"] = not report["errors"]
    return report


def format_report(report: dict) -> str:
    lines = [f"stage plan preflight: {report['plan']}"]
    lines.append(f"  engine gate ({ENGINE_GATING_SOURCE}): {report['engine_gate']}")
    lines.append(f"  vae checkpoint: {report['checkpoint']}")
    if "shared_workspace_mib" in report:
        lines.append(
            f"  workspace: {report['shared_workspace_mib']} MiB shared arena, "
            f"{report['summed_workspace_mib']} MiB summed across signatures"
        )
    if report["advisory_drift"]:
        lines.append("  [advisory] source drift (engines still load; run is not byte-reproducible):")
        lines.extend(f"      {item}" for item in report["advisory_drift"])
    if report["missing_sources"]:
        lines.append("  [advisory] plan references files absent from the tree:")
        lines.extend(f"      {item}" for item in report["missing_sources"])
    for error in report["errors"]:
        lines.append(f"  ERROR: {error}")
    lines.append("  RESULT: " + ("PASS" if report["ok"] else "FAIL"))
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    parser.add_argument("plan", type=Path, help="stage build results.json")
    parser.add_argument("--json", action="store_true", help="emit the report as JSON")
    args = parser.parse_args(argv)

    report = check_plan(args.plan.resolve())
    print(json.dumps(report, indent=2) if args.json else format_report(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
