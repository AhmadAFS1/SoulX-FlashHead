"""Denoising-schedule guards. Stdlib only -- no torch, no numpy.

These run on a CPU-only developer machine with none of the inference
dependencies installed. They are CORRECTNESS guards; they say nothing about
speed and never qualify a recipe for promotion.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flash_head.src.pipeline.schedules import (  # noqa: E402
    DISTILLED_LEVELS,
    raw_timestep_schedule,
    resolved_timestep_schedule,
    timestep_transform,
)


def test_four_step_resolves_to_the_recorded_distilled_levels():
    """The retained 11.7424 FPS protocol is four steps at shift 5."""
    resolved = resolved_timestep_schedule(4, shift=5.0, variant="distilled_aligned")
    assert [round(t, 4) for t in resolved] == [1000.0, 937.5, 833.3333, 625.0, 0.0]


def test_two_step_terminates_on_the_four_step_terminal_level():
    """Two steps must end at 625.0, the level the four-step student ends on.

    The previous table used raw [1000, 500], terminating at 833.3333 and
    skipping the 625.0 level entirely.
    """
    resolved = resolved_timestep_schedule(2, shift=5.0, variant="distilled_aligned")
    assert [round(t, 4) for t in resolved] == [1000.0, 625.0, 0.0]


def test_three_step_is_a_subset_of_the_four_step_levels():
    resolved = resolved_timestep_schedule(3, shift=5.0, variant="distilled_aligned")
    assert [round(t, 4) for t in resolved] == [1000.0, 833.3333, 625.0, 0.0]


@pytest.mark.parametrize("steps", [1, 2, 3, 4])
def test_every_level_is_a_member_of_the_distilled_set(steps):
    """No schedule may place a forward pass at a level the student never saw."""
    allowed = {round(t, 4) for t in DISTILLED_LEVELS}
    for level in resolved_timestep_schedule(steps, shift=5.0, variant="distilled_aligned"):
        assert round(level, 4) in allowed, f"{steps}-step visits untrained level {level}"


def test_regression_the_old_linspace_defect_is_gone():
    """Guard the specific defect: linspace(1000, 1, 3) put a step at t=4.98.

    Raw [1000, 500.5, 1] transforms to [1000.0, 833.6109, 4.9801]. The third
    entry is a full forward pass at 0.5% noise, outside the level set, costing
    a measured 0.2347 s per 250-frame equivalent for nothing.
    """
    defect = [round(timestep_transform(t), 4) for t in (1000.0, 500.5, 1.0)]
    assert defect == [1000.0, 833.6109, 4.9801], "arithmetic drifted; revisit the table"

    resolved = [round(t, 4) for t in resolved_timestep_schedule(3, shift=5.0, variant="distilled_aligned")]
    assert 4.9801 not in resolved
    assert 833.6109 not in resolved


def test_untabulated_step_count_raises_rather_than_interpolating():
    with pytest.raises(ValueError, match="No distilled_aligned schedule"):
        raw_timestep_schedule(7, variant="distilled_aligned")


def test_raw_schedule_appends_exactly_one_terminal_zero():
    for steps in (1, 2, 3, 4):
        raw = raw_timestep_schedule(steps, variant="distilled_aligned")
        assert raw[-1] == 0.0
        assert len(raw) == steps + 1
        assert 0.0 not in raw[:-1]


def test_transform_endpoints_are_fixed_points():
    assert timestep_transform(0.0) == 0.0
    assert round(timestep_transform(1000.0), 10) == 1000.0
