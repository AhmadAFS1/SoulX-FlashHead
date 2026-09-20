"""Tie every published FPS claim to a tree that can actually produce it.

Stdlib only -- no torch, no numpy. Runs on a CPU-only developer machine.

Each retained ``results.json`` records a ``source_sha256`` manifest of the exact
files the run executed. Nothing in the repo re-checks it, so a published number
such as "11.7424 FPS" can silently come to describe source that no longer exists
on disk. This test makes that drift loud.

It is deliberately NOT a hard failure for every file. Only ``flash_head/wan/
modules/vae.py`` gates TensorRT engine loading (``install_stage_plan`` compares
its hash before installing a stage plan), so only that file can invalidate the
engines. Everything else is reported as advisory drift: the run is no longer
byte-reproducible, which matters when comparing a new measurement against it.

GPU/evidence note: this is a ``[CPU]`` static hash check. It proves nothing
about speed and can never qualify a recipe for promotion.
"""
import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# The candidate run that produced the 11.7424 FPS / 21.290295 s budget every
# current plan document is built on.
CANDIDATE_RESULTS = (
    REPO
    / "benchmarks/pro_30fps_20260919/extended-r01/recurrence-candidate/results.json"
)

# The single file whose hash gates engine installation.
ENGINE_GATING_SOURCE = "flash_head/wan/modules/vae.py"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _drift(manifest):
    """Return (missing, drifted) relative paths against the manifest."""
    missing, drifted = [], []
    for rel, recorded in sorted(manifest.items()):
        path = REPO / rel
        if not path.is_file():
            missing.append(rel)
        elif _sha256(path) != recorded:
            drifted.append(rel)
    return missing, drifted


@pytest.fixture(scope="module")
def candidate_manifest():
    if not CANDIDATE_RESULTS.is_file():
        pytest.skip(f"missing retained artifact: {CANDIDATE_RESULTS}")
    manifest = json.loads(CANDIDATE_RESULTS.read_text()).get("source_sha256")
    if not manifest:
        pytest.skip("retained artifact has no source_sha256 manifest")
    return manifest


def test_engine_gating_source_matches_the_candidate_run(candidate_manifest):
    """vae.py must match, or the retained FP16 stage engines will not install.

    install_stage_plan compares this hash before installing a stage plan. If it
    drifts, the engines must be rebuilt before any stage-backend run.
    """
    recorded = candidate_manifest.get(ENGINE_GATING_SOURCE)
    assert recorded, f"{ENGINE_GATING_SOURCE} absent from the manifest"

    path = REPO / ENGINE_GATING_SOURCE
    assert path.is_file(), f"{ENGINE_GATING_SOURCE} is missing from the tree"

    actual = _sha256(path)
    assert actual == recorded, (
        f"{ENGINE_GATING_SOURCE} has drifted from the candidate run.\n"
        f"  recorded: {recorded}\n"
        f"  on disk:  {actual}\n"
        "The retained TensorRT stage engines will refuse to install. Rebuild "
        "the engines (capture + build) before running any stage-backend policy."
    )


def test_report_advisory_drift_against_the_candidate_run(candidate_manifest):
    """Non-gating drift is reported, not failed -- but it must be visible.

    Any file listed here means the retained 11.7424 FPS baseline was measured on
    different source than the current tree. A 4070 run must RE-BASELINE the
    current tree before its numbers are compared against that figure.
    """
    missing, drifted = _drift(candidate_manifest)
    advisory = [r for r in missing + drifted if r != ENGINE_GATING_SOURCE]

    if advisory:
        detail = "\n".join(f"    {r}" for r in sorted(advisory))
        print(
            "\n[advisory] source drift vs "
            "extended-r01/recurrence-candidate (11.7424 FPS):\n"
            f"{detail}\n"
            "  The published baseline does NOT describe this tree. Re-baseline "
            "on the 4070 with --latency-detail off before comparing.\n"
        )

    # The manifest itself must stay well-formed and complete.
    assert not missing, f"manifest references files absent from the tree: {missing}"


def test_manifest_covers_the_files_a_speed_change_would_touch(candidate_manifest):
    """A manifest that omits the hot path cannot detect a meaningful change."""
    required = {
        "flash_head/src/pipeline/flash_head_pipeline.py",
        "flash_head/src/modules/flash_head_model.py",
        "flash_head/wan/modules/vae.py",
        "soulx_rtc/pro_quantization.py",
        "soulx_rtc/pro_quantization_v2.py",
        "soulx_rtc/pro_attention_backends.py",
    }
    assert required <= set(candidate_manifest), (
        "manifest no longer covers the hot path: "
        f"{sorted(required - set(candidate_manifest))}"
    )


def test_no_cuda_or_torch_import_is_required():
    """This guard must run on a machine with no torch installed."""
    assert "torch" not in sys.modules
