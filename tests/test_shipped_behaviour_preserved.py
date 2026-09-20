"""Regression guards for defects found by adversarial review of the step change.

Every test here corresponds to a real defect that shipped in the working tree and
was caught only by review, not by the existing suite. Stdlib only -- no torch.

The defects were:
  1. Wave 1 switches were instance-only attributes, so generate() raised
     AttributeError on any pipeline built with __new__ (which an existing test does).
  2. The distilled table raised on sampling_steps=20, breaking the CFG teacher.
  3. The 2-step row was silently re-based, changing soulx_rtc.Engine(steps=2).
  4. "fast_accum" had no type check, so JSON "false" would enable it.
  5. schedules.py was absent from run.py's source manifest.
  6. sweep.py could not forward the new flags, so the paired harness could not
     run the arm at all.

GPU/evidence note: `[CPU]` static checks. None measures speed.
"""
import ast
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flash_head.src.pipeline.schedules import (  # noqa: E402
    DISTILLED_ALIGNED,
    SHIPPED,
    raw_timestep_schedule,
    resolved_timestep_schedule,
    timestep_transform,
)

PIPELINE = REPO / "flash_head/src/pipeline/flash_head_pipeline.py"
RUN_PY = REPO / "benchmarks/pro_quantization_v2_20260918/run.py"
SWEEP_PY = REPO / "benchmarks/pro_quantization_v2_20260918/sweep.py"


# ------------------------------------------------------- 1. class-level defaults


def _class_level_names(path, class_name):
    """Names assigned in the class body itself, not inside any method."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            names = set()
            for stmt in node.body:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
                elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    names.add(stmt.target.id)
            return names
    raise AssertionError(f"class {class_name} not found in {path}")


@pytest.mark.parametrize("name", [
    "lean_delivery", "verbose_timing", "skip_zero_weighted_noise",
    "resolved_timesteps", "timestep_variant",
])
def test_generate_switches_have_class_level_defaults(name):
    """generate() reads these unconditionally.

    tests/test_pipeline_latency.py builds the pipeline with
    FlashHeadPipeline.__new__(FlashHeadPipeline) and hand-sets a fixed attribute
    list. Anything generate() reads that is assigned only in __init__ raises
    AttributeError there. Class-level defaults are the contract.
    """
    assert name in _class_level_names(PIPELINE, "FlashHeadPipeline"), (
        f"{name} must be a class attribute, not assigned only in __init__"
    )


def test_class_defaults_are_all_falsy_preserving_historical_behaviour():
    """Every switch must default off, or a run changes meaning silently."""
    tree = ast.parse(PIPELINE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "FlashHeadPipeline":
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name):
                    name = stmt.targets[0].id
                    if name in ("lean_delivery", "verbose_timing", "skip_zero_weighted_noise"):
                        assert stmt.value.value is False, f"{name} must default False"
                    if name == "timestep_variant":
                        assert stmt.value.value == SHIPPED, "variant must default to shipped"
            return
    raise AssertionError("FlashHeadPipeline not found")


# ------------------------------------------------- 2/3. shipped behaviour intact


def test_teacher_step_count_does_not_raise():
    """flash_head/inference.py sets sample_steps=20 for model_type 'pretrained'.

    The shipped variant must return None for it, so the caller falls back to the
    original np.linspace. Raising here would break the teacher entirely.
    """
    assert raw_timestep_schedule(20, variant=SHIPPED) is None
    assert resolved_timestep_schedule(20, variant=SHIPPED) is None


def test_inference_still_declares_the_twenty_step_teacher():
    """If this constant ever changes, the guard above needs revisiting."""
    text = (REPO / "flash_head/inference.py").read_text()
    assert "infer_params['sample_steps'] = 20" in text


def test_shipped_two_step_schedule_is_unchanged():
    """soulx_rtc.Engine accepts steps in (2, 4); its 2-step levels must not move."""
    assert raw_timestep_schedule(2, variant=SHIPPED) == [1000.0, 500.0, 0.0]
    resolved = [round(t, 4) for t in resolved_timestep_schedule(2, variant=SHIPPED)]
    assert resolved == [1000.0, 833.3333, 0.0]


def test_shipped_four_step_schedule_is_unchanged():
    assert raw_timestep_schedule(4, variant=SHIPPED) == [1000.0, 750.0, 500.0, 250.0, 0.0]


def test_engine_still_accepts_two_and_four_steps():
    """Pins the reason the shipped 2-step row must stay put."""
    text = (REPO / "soulx_rtc/engine.py").read_text()
    assert "if steps not in (2, 4):" in text


def test_the_two_variants_actually_differ_where_intended():
    """Otherwise the opt-in is meaningless and the tests above are vacuous."""
    assert raw_timestep_schedule(2, variant=SHIPPED) != raw_timestep_schedule(
        2, variant=DISTILLED_ALIGNED
    )
    # ...and agree where they must.
    assert raw_timestep_schedule(4, variant=SHIPPED) == raw_timestep_schedule(
        4, variant=DISTILLED_ALIGNED
    )


def test_unknown_variant_is_rejected():
    with pytest.raises(ValueError, match="Unknown timestep variant"):
        raw_timestep_schedule(4, variant="whatever")


def test_the_two_timestep_transform_implementations_agree():
    """schedules.py duplicates the pipeline's transform so it imports without torch.

    Parse the pipeline's version out of source and evaluate it, rather than
    trusting that two hand-written copies stayed in sync.
    """
    text = PIPELINE.read_text()
    assert "new_t = shift * t / (1 + (shift - 1) * t)" in text, (
        "the pipeline's timestep_transform changed shape; re-verify schedules.py"
    )
    for t in (0.0, 1.0, 250.0, 500.0, 500.5, 750.0, 1000.0):
        r = t / 1000
        expected = (5.0 * r / (1 + 4.0 * r)) * 1000
        assert timestep_transform(t) == pytest.approx(expected, rel=1e-12)


# ------------------------------------------------------- 4. fast_accum typing


def test_fast_accum_type_check_exists_in_source():
    """bool("false") is True; a string must be rejected, not coerced."""
    text = (REPO / "soulx_rtc/pro_quantization_v2.py").read_text()
    assert 'if "fast_accum" in policy and not isinstance(policy["fast_accum"], bool)' in text, (
        "fast_accum must be type-validated before reaching bool() at conversion"
    )


# --------------------------------------------------------- 5/6. provenance


def test_schedules_module_is_in_the_run_source_manifest():
    """The table that defines a run's timesteps must be hashed into provenance."""
    assert 'ROOT / "flash_head/src/pipeline/schedules.py"' in RUN_PY.read_text(), (
        "schedules.py must appear in run.py's _sources()"
    )


@pytest.mark.parametrize("flag", [
    "--sampling-steps", "--timestep-variant", "--lean-delivery",
    "--skip-zero-weighted-noise",
])
def test_sweep_forwards_every_arm_flag(flag):
    """The paired sweep is the promotion protocol.

    A flag run.py accepts but sweep.py cannot forward makes that arm
    unmeasurable through the sanctioned harness.
    """
    text = SWEEP_PY.read_text()
    assert flag in text, f"sweep.py cannot forward {flag}"


def test_sweep_forwards_arm_flags_to_both_roles():
    """Forwarding to only one role would silently decontrol the comparison."""
    text = SWEEP_PY.read_text()
    # The command list is built once per cell, before the role-independent
    # append block, so a single append site covers both roles.
    assert 'policy = args.reference_policy if cell["role"] == "reference" else args.policy' in text
    append_index = text.index('if args.sampling_steps is not None:')
    role_index = text.index('policy = args.reference_policy')
    assert append_index > role_index, "flag forwarding must sit inside the per-cell loop"


def test_run_records_the_arm_settings_in_the_profile():
    """review.py compares profiles to decide pairs are comparable."""
    text = RUN_PY.read_text()
    assert '"skip_zero_weighted_noise": skip_zero_weighted_noise,' in text
    assert '"timestep_variant": timestep_variant,' in text
    assert '"steps": steps,' in text


def test_retained_results_predate_the_new_profile_keys():
    """Retained runs must stay readable; the new keys are additive only."""
    path = (
        REPO
        / "benchmarks/pro_30fps_20260919/extended-r01/recurrence-candidate/results.json"
    )
    if not path.is_file():
        pytest.skip(f"missing retained artifact: {path}")
    profile = json.loads(path.read_text())["profile"]
    assert profile["steps"] == 4
    assert "timestep_variant" not in profile, (
        "retained artifacts must not be rewritten; new keys appear only in new runs"
    )
