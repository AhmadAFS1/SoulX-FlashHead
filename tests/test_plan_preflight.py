"""Guards for the stage-plan preflight. Stdlib only -- no torch, no tensorrt.

The preflight exists so a stale engine plan fails in milliseconds on CPU instead
of after a full model load on a leased GPU. These tests pin the two behaviours
that matter: vae.py is the only HARD gate, and everything else is advisory.

GPU/evidence note: `[CPU]` static checks throughout.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmarks.pro_quantization_v2_20260918.plan_preflight import (  # noqa: E402
    ENGINE_GATING_SOURCE,
    check_plan,
    main,
)

RETAINED = {
    "fp16": REPO / "benchmarks/pro_30fps_20260919/stage-fp16-build-r01/results.json",
    "int8": REPO / "benchmarks/pro_30fps_20260919/stage-int8-build-r02/results.json",
    "bf16": REPO / "benchmarks/pro_30fps_20260919/stage-bf16-build-r02/results.json",
}


def test_preflight_imports_no_inference_stack():
    """It must be runnable on a machine with no torch and no tensorrt."""
    assert "torch" not in sys.modules
    assert "tensorrt" not in sys.modules


@pytest.mark.parametrize("name", sorted(RETAINED))
def test_retained_plans_currently_pass(name):
    """A tripwire: this fails the moment anyone edits flash_head/wan/modules/vae.py.

    That is intentional. Editing the Wan VAE invalidates every retained engine,
    and the next stage-backend run must rebuild before it can mean anything.
    """
    path = RETAINED[name]
    if not path.is_file():
        pytest.skip(f"missing retained artifact: {path}")
    report = check_plan(path)
    assert report["engine_gate"] == "match", report["errors"]
    assert report["ok"], report["errors"]


def test_workspace_arithmetic_matches_the_recorded_values():
    """Independently recompute the arena sizes the plan documents cite.

    These are [CPU] recomputations of [M] build artifacts, not new measurements.
    The BF16-vs-FP16 gap is the load-bearing number: the selected FP16 engines
    need 472.5 MiB more shared arena than the BF16 engines, against roughly
    426 MiB of observed device headroom.
    """
    expected = {
        "fp16": (709.102, 1739.905),
        "int8": (748.125, 2071.510),
        "bf16": (236.602, 713.870),
    }
    seen = {}
    for name, path in RETAINED.items():
        if not path.is_file():
            pytest.skip(f"missing retained artifact: {path}")
        report = check_plan(path)
        seen[name] = (report["shared_workspace_mib"], report["summed_workspace_mib"])
        assert seen[name] == pytest.approx(expected[name], abs=0.01), name

    gap = seen["fp16"][0] - seen["bf16"][0]
    assert gap == pytest.approx(472.5, abs=0.01), (
        f"FP16-over-BF16 shared arena gap moved to {gap} MiB"
    )


def test_vae_drift_is_a_hard_failure(tmp_path):
    plan = {
        "schema_version": 1,
        "status": "complete",
        "weights_sha256": "unused",
        "source_sha256": {ENGINE_GATING_SOURCE: "0" * 64},
        "stages": {"4-5-6": {"sig": {"workspace_bytes": 1024}}},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))

    report = check_plan(path)
    assert report["engine_gate"] == "drift"
    assert not report["ok"]
    assert any("will refuse these engines" in e for e in report["errors"])
    assert main([str(path)]) == 1


def test_non_gating_drift_is_advisory_only(tmp_path):
    """Drift outside vae.py must not block a run -- the engines still load."""
    real_vae = REPO / ENGINE_GATING_SOURCE
    if not real_vae.is_file():
        pytest.skip("vae.py missing from the tree")
    from benchmarks.pro_quantization_v2_20260918.plan_preflight import sha256

    plan = {
        "schema_version": 1,
        "status": "complete",
        "weights_sha256": "unused",
        "source_sha256": {
            ENGINE_GATING_SOURCE: sha256(real_vae),
            "soulx_rtc/pro_vae_stage_backend.py": "0" * 64,
        },
        "stages": {"4-5-6": {"sig": {"workspace_bytes": 2048}}},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))

    report = check_plan(path)
    assert report["engine_gate"] == "match"
    assert report["advisory_drift"] == ["soulx_rtc/pro_vae_stage_backend.py"]
    assert report["ok"], "advisory drift must not fail the preflight"
    assert main([str(path)]) == 0


def test_incomplete_plan_is_rejected(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema_version": 1, "status": "aborted"}))
    report = check_plan(path)
    assert not report["ok"]
    assert any("status is" in e for e in report["errors"])


def test_unreadable_plan_reports_cleanly(tmp_path):
    report = check_plan(tmp_path / "does-not-exist.json")
    assert not report["ok"]
    assert any("cannot read plan" in e for e in report["errors"])


def test_missing_checkpoint_is_reported_not_failed(tmp_path):
    """Developer machines have no model weights; that must not be an error."""
    from benchmarks.pro_quantization_v2_20260918.plan_preflight import sha256

    real_vae = REPO / ENGINE_GATING_SOURCE
    if not real_vae.is_file():
        pytest.skip("vae.py missing from the tree")
    plan = {
        "schema_version": 1,
        "status": "complete",
        "weights_sha256": "whatever",
        "source_sha256": {ENGINE_GATING_SOURCE: sha256(real_vae)},
        "stages": {"g": {"s": {"workspace_bytes": 1}}},
    }
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))

    report = check_plan(path)
    assert report["checkpoint"] == "unavailable-locally"
    assert report["ok"]
