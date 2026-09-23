"""Denoising-step schedules.

Pure Python, no torch/numpy import, so the schedule arithmetic can be tested on a
CPU-only developer machine without the inference dependencies installed.

Two variants exist deliberately.

``shipped`` reproduces the original behaviour EXACTLY, including the
``np.linspace`` fallback for step counts with no tabulated row. This is the
default, because existing configurations depend on it:

  * ``flash_head/inference.py`` sets ``sample_steps = 20`` for
    ``model_type == "pretrained"`` (the CFG teacher), which has no table row.
  * ``soulx_rtc/engine.py`` accepts ``steps in (2, 4)``, so the 2-step row is a
    supported RTC configuration and must keep its shipped levels.

For an untabulated count this returns ``None`` and the caller performs the
original ``np.linspace(num_timesteps, 1, steps, dtype=np.float32)``. That is
deliberate: reproducing float32 linspace rounding in pure Python risks differing
in the last bit, and the teacher path must stay bit-identical.

``distilled_aligned`` is the corrected table. Every non-terminal level is a
member of the four-step distilled level set, which at ``shift=5`` is

    t = {1000.0, 937.5, 833.3333, 625.0}

It exists because the shipped fallback does not have that property. At
``sampling_steps=3`` the fallback produces raw ``[1000, 500.5, 1]``, whose third
entry transforms to **t = 4.98** -- a full forward pass at 0.5% noise, at a level
the distilled student never saw. The aligned table also re-bases the 2-step row
from ``[1000, 500]`` to ``[1000, 250]`` so that it terminates on 625.0, the level
the four-step schedule actually ends on, rather than 833.3333.

Choosing ``distilled_aligned`` is a QUALITY-AFFECTING decision and is never the
default. Callers opt in explicitly.
"""

SHIPPED = "shipped"
DISTILLED_ALIGNED = "distilled_aligned"
VARIANTS = (SHIPPED, DISTILLED_ALIGNED)

# The original tabulated rows. Anything else fell through to np.linspace.
_SHIPPED_SCHEDULES = {
    2: [1000, 500],
    4: [1000, 750, 500, 250],
}

# Corrected rows: every non-terminal level is in the four-step distilled set.
#   1000 -> 1000.0    750 -> 937.5    500 -> 833.3333    250 -> 625.0
_DISTILLED_ALIGNED_SCHEDULES = {
    1: [1000],
    2: [1000, 250],
    3: [1000, 500, 250],
    4: [1000, 750, 500, 250],
}

# Post-shift levels the four-step distilled schedule visits, terminal included.
DISTILLED_LEVELS = (1000.0, 937.5, 833.3333333333334, 625.0, 0.0)


def timestep_transform(t, shift=5.0, num_timesteps=1000):
    """Scalar form of the pipeline's timestep shift.

    Intentionally duplicated from flash_head_pipeline.timestep_transform so this
    module stays importable without torch. test_sampling_schedule.py asserts the
    two agree; if they ever diverge, that test fails.
    """
    r = t / num_timesteps
    return (shift * r / (1 + (shift - 1) * r)) * num_timesteps


def _table(variant):
    if variant == DISTILLED_ALIGNED:
        return _DISTILLED_ALIGNED_SCHEDULES
    if variant == SHIPPED:
        return _SHIPPED_SCHEDULES
    raise ValueError(f"Unknown timestep variant {variant!r}; expected one of {VARIANTS}")


def raw_timestep_schedule(sampling_steps, num_timesteps=1000, *, variant=SHIPPED):
    """Return the raw (pre-shift) schedule including the terminal 0.0.

    Returns ``None`` for an untabulated count under ``shipped``, signalling the
    caller to use the original ``np.linspace`` fallback. Raises under
    ``distilled_aligned``, which exists precisely to refuse untrained levels.
    """
    table = _table(variant)
    if sampling_steps not in table:
        if variant == DISTILLED_ALIGNED:
            raise ValueError(
                f"No {DISTILLED_ALIGNED} schedule for sampling_steps={sampling_steps}; "
                f"available: {sorted(table)}. Add an explicit row rather than "
                f"interpolating -- the distilled student is only trained at "
                f"{DISTILLED_LEVELS[:-1]} post-shift."
            )
        return None
    raw = [float(t) for t in table[sampling_steps]]
    if num_timesteps != 1000:
        raw = [t * num_timesteps / 1000.0 for t in raw]
    return raw + [0.0]


def resolved_timestep_schedule(sampling_steps, shift=5.0, num_timesteps=1000, *, variant=SHIPPED):
    """The post-shift levels actually handed to the model, or None if untabulated."""
    raw = raw_timestep_schedule(sampling_steps, num_timesteps=num_timesteps, variant=variant)
    if raw is None:
        return None
    return [timestep_transform(t, shift=shift, num_timesteps=num_timesteps) for t in raw]
