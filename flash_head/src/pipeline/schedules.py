"""Explicit denoising-step schedules for the distilled PRO/LITE students.

Pure Python, no torch/numpy import, so the schedule arithmetic can be tested on a
CPU-only developer machine without the inference dependencies installed.

The distilled checkpoints are trained at a fixed set of post-shift noise levels.
With the default ``shift=5`` transform the four-step schedule resolves to

    t = {1000.0, 937.5, 833.3333, 625.0, 0.0}

Any reduced-step schedule must draw its non-terminal levels from that set,
otherwise a forward pass lands at a noise level the student never saw. The
previous ``np.linspace(num_timesteps, 1, sampling_steps)`` fallback did not: at
``sampling_steps=3`` it produced raw ``[1000, 500.5, 1]``, whose third entry
transforms to ``t=4.98`` -- a full forward pass at 0.5% noise.
"""

# Raw (pre-shift) timesteps per step count. The terminal 0.0 is appended by
# ``raw_timestep_schedule``; these are the levels an actual forward pass uses.
#
# Post-shift levels at shift=5.0, num_timesteps=1000:
#   1000 -> 1000.0    750 -> 937.5    500 -> 833.3333    250 -> 625.0
_RAW_SCHEDULES = {
    1: [1000],
    2: [1000, 250],
    3: [1000, 500, 250],
    4: [1000, 750, 500, 250],
}

# The post-shift levels the four-step distilled schedule visits, terminal included.
DISTILLED_LEVELS = (1000.0, 937.5, 833.3333333333334, 625.0, 0.0)


def timestep_transform(t, shift=5.0, num_timesteps=1000):
    """Scalar form of the pipeline's timestep shift. Kept in sync deliberately."""
    r = t / num_timesteps
    return (shift * r / (1 + (shift - 1) * r)) * num_timesteps


def raw_timestep_schedule(sampling_steps, num_timesteps=1000):
    """Return the raw (pre-shift) schedule including the terminal 0.0.

    Raises ValueError for step counts with no tabulated schedule rather than
    silently falling back to a linspace that leaves the distilled level set.
    """
    if sampling_steps not in _RAW_SCHEDULES:
        raise ValueError(
            f"No tabulated schedule for sampling_steps={sampling_steps}; "
            f"available: {sorted(_RAW_SCHEDULES)}. Add an explicit row rather "
            f"than interpolating -- the student is only trained at "
            f"{DISTILLED_LEVELS[:-1]} post-shift."
        )
    raw = [float(t) for t in _RAW_SCHEDULES[sampling_steps]]
    if num_timesteps != 1000:
        raw = [t * num_timesteps / 1000.0 for t in raw]
    return raw + [0.0]


def resolved_timestep_schedule(sampling_steps, shift=5.0, num_timesteps=1000):
    """The post-shift levels actually handed to the model. For results.json."""
    return [
        timestep_transform(t, shift=shift, num_timesteps=num_timesteps)
        for t in raw_timestep_schedule(sampling_steps, num_timesteps=num_timesteps)
    ]
