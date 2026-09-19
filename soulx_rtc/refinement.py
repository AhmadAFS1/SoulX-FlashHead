"""Optional clean-latent restart for experimental distilled SoulX inference."""
import torch


@torch.no_grad()
def refine_clean_latent(clean, timesteps, num_timesteps, motions, generators, denoise):
    """Re-noise a clean prediction, then predict clean at each lower timestep.

    Timesteps are already transformed to the model's time coordinates. Motion
    prefixes are fixed before every model call and in the returned tensor.
    Generators must be separate from the base sampler/motion-encoder streams.
    """
    if not timesteps:
        return clean
    if len(motions) != len(clean) or len(generators) != len(clean):
        raise ValueError("One motion prefix and refinement generator per sample required")
    values = [float(t.item()) for t in timesteps]
    if not all(0 < t < num_timesteps for t in values) or any(
            a <= b for a, b in zip(values, values[1:])):
        raise ValueError("Refinement times must strictly decrease within (0, num_timesteps)")
    result = clean
    for timestep in timesteps:
        t = (timestep / num_timesteps).to(device=clean.device, dtype=clean.dtype)
        fresh = torch.stack([torch.randn(sample.shape, dtype=clean.dtype,
            device=clean.device, generator=generator)
            for sample, generator in zip(clean, generators)])
        noisy = (1 - t) * result + t * fresh
        for index, motion in enumerate(motions):
            noisy[index, :, :motion.shape[1]] = motion
        result = noisy - denoise(noisy, timestep) * t
    for index, motion in enumerate(motions):
        result[index, :, :motion.shape[1]] = motion
    return result
