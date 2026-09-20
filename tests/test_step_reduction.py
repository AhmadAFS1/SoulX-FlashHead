"""Guards for the 4 -> 2 denoising-step reduction (D2). Stdlib only.

D2 is the largest single FPS lever available once ROI/crop reduction is excluded.
It is also the highest-risk item in the plan, because reducing steps changes the
input distribution reaching each trained timestep. These tests pin the mechanical
parts -- schedule membership, loop arithmetic, and the recorded budget -- so the
GPU run only has to answer the quality question.

GPU/evidence note: `[CPU]` static/arithmetic checks against retained `[M]`
artifacts. Nothing here measures speed. Passing does NOT qualify 2-step sampling;
that requires the quality arm on the recorded GPU (RTX 4070 SUPER, 12,282 MiB
visible, driver 595.84).
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from flash_head.src.pipeline.schedules import (  # noqa: E402
    DISTILLED_LEVELS,
    raw_timestep_schedule,
    resolved_timestep_schedule,
)

# Measured budget, 250-frame equivalents, from
# benchmarks/pro_30fps_20260919/extended-r01/recurrence-candidate/results.json
# (generation_s 127.741769 over 1500 frames; stage_seconds / 6).
DIT_S = 8.474913
DECODER_S = 11.049357
FIXED_S = 1.766025  # motion 1.379389 + audio 0.072047 + remainder 0.314589
BASELINE_S = DIT_S + DECODER_S + FIXED_S
TARGET_30FPS_S = 250 / 30


def fps(seconds):
    return 250 / seconds


# ------------------------------------------------------------ schedule shape


def test_two_step_schedule_has_exactly_two_forward_passes():
    """The denoise loop runs range(len(timesteps) - 1)."""
    raw = raw_timestep_schedule(2, variant="distilled_aligned")
    assert len(raw) == 3
    assert len(raw) - 1 == 2, "two steps must mean two model forward passes"


@pytest.mark.parametrize("steps,passes", [(1, 1), (2, 2), (3, 3), (4, 4)])
def test_forward_pass_count_equals_step_count(steps, passes):
    assert len(raw_timestep_schedule(steps, variant="distilled_aligned")) - 1 == passes


def test_two_step_levels_are_a_subset_of_the_four_step_levels():
    four = {round(t, 4) for t in resolved_timestep_schedule(4, variant="distilled_aligned")}
    two = {round(t, 4) for t in resolved_timestep_schedule(2, variant="distilled_aligned")}
    assert two <= four, f"2-step visits levels absent from the 4-step set: {two - four}"


def test_two_step_terminal_level_matches_the_four_step_terminal_level():
    """Both must end their last real step at 625.0, the student's final level."""
    assert round(resolved_timestep_schedule(2, variant="distilled_aligned")[-2], 4) == 625.0
    assert round(resolved_timestep_schedule(4, variant="distilled_aligned")[-2], 4) == 625.0


def test_terminal_level_is_exactly_zero_for_every_step_count():
    """skip_zero_weighted_noise keys off this being exactly 0.0, not near it."""
    for steps in (1, 2, 3, 4):
        assert resolved_timestep_schedule(steps, variant="distilled_aligned")[-1] == 0.0


def test_only_the_final_level_is_zero():
    """A mid-schedule zero would make the skip fire on a real step."""
    for steps in (1, 2, 3, 4):
        levels = resolved_timestep_schedule(steps, variant="distilled_aligned")
        assert all(level != 0.0 for level in levels[:-1]), steps


# ------------------------------------------------------------ budget arithmetic


def test_baseline_budget_reproduces_the_measured_fps():
    assert BASELINE_S == pytest.approx(21.290295, abs=1e-6)
    assert fps(BASELINE_S) == pytest.approx(11.7424, abs=1e-3)


def test_two_steps_halve_dit_and_give_the_projected_fps():
    """DiT is exactly linear in step count: 120 block executions -> 60."""
    projected = DIT_S / 2 + DECODER_S + FIXED_S
    assert DIT_S / 2 == pytest.approx(4.237457, abs=1e-6)
    assert projected == pytest.approx(17.052839, abs=1e-5)
    assert fps(projected) == pytest.approx(14.66, abs=0.01)

    gain = fps(projected) / fps(BASELINE_S) - 1
    assert gain == pytest.approx(0.2485, abs=0.001), "D2 should be about +24.85%"


def test_two_steps_alone_do_not_reach_30_fps():
    """State the shortfall explicitly so no one reads D2 as sufficient."""
    projected = DIT_S / 2 + DECODER_S + FIXED_S
    assert projected > TARGET_30FPS_S
    assert fps(projected) < 30.0
    # With 2 steps, what decoder speedup would 30 FPS still require?
    decoder_budget = TARGET_30FPS_S - DIT_S / 2 - FIXED_S
    required = DECODER_S / decoder_budget
    assert required == pytest.approx(4.74, abs=0.02), (
        "2 steps still needs a ~4.74x decoder; best plausible execution gain is 1.518x"
    )


def test_four_steps_cannot_reach_30_fps_at_any_decoder_speed():
    """The fact that makes D2 necessary rather than merely attractive."""
    floor = DIT_S + FIXED_S  # decoder entirely free
    assert floor > TARGET_30FPS_S
    assert DIT_S / TARGET_30FPS_S == pytest.approx(1.0170, abs=0.001), (
        "DiT alone should be 101.70% of the 30-FPS budget at 4 steps"
    )
    assert fps(floor) == pytest.approx(24.4118, abs=0.001)


def test_three_steps_is_the_intermediate_option():
    projected = DIT_S * 0.75 + DECODER_S + FIXED_S
    assert fps(projected) == pytest.approx(13.04, abs=0.02)


# ------------------------------------------------------------ provenance


def test_retained_candidate_run_used_four_steps():
    """The baseline this is measured against must be unambiguous."""
    path = (
        REPO
        / "benchmarks/pro_30fps_20260919/extended-r01/recurrence-candidate/results.json"
    )
    if not path.is_file():
        pytest.skip(f"missing retained artifact: {path}")
    data = json.loads(path.read_text())
    assert data["profile"]["steps"] == 4
    assert data["profile"]["shift"] == 5
    assert data["profile"]["width"] == 320 and data["profile"]["height"] == 576


def test_measured_stage_seconds_divide_into_the_budget_constants():
    path = (
        REPO
        / "benchmarks/pro_30fps_20260919/extended-r01/recurrence-candidate/results.json"
    )
    if not path.is_file():
        pytest.skip(f"missing retained artifact: {path}")
    stages = json.loads(path.read_text())["runs"][0]["stage_seconds"]
    assert stages["dit"] / 6 == pytest.approx(DIT_S, abs=1e-5)
    assert stages["vae_decode"] / 6 == pytest.approx(DECODER_S, abs=1e-5)


def test_untabulated_step_counts_still_raise():
    """No silent interpolation off the distilled level set."""
    for bad in (0, 5, 8, 20, 50):
        with pytest.raises(ValueError, match="No distilled_aligned schedule"):
            raw_timestep_schedule(bad, variant="distilled_aligned")


def test_distilled_level_constant_matches_the_four_step_schedule():
    resolved = [round(t, 4) for t in resolved_timestep_schedule(4, variant="distilled_aligned")]
    assert resolved == [round(t, 4) for t in DISTILLED_LEVELS]
