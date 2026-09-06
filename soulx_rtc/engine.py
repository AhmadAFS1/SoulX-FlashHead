"""Single-owner GPU engine. Session state is small; model weights are shared.

Only the scheduler's GPU thread may call this engine. DiT microbatches share a
shape/profile, but VAE decode/encode stays sequential to bound peak VRAM.
"""
import copy
import hashlib
import math
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np


@dataclass
class GenerationState:
    pipeline: object
    audio: np.ndarray
    total_frames: int
    cursor: int = 0


class Engine:
    fps = 25
    sample_rate = 16000
    chunk_frames = 24

    def __init__(self, size=512, steps=4, compile_model=True, fps=25):
        import torch
        import flash_head.src.pipeline.flash_head_pipeline as implementation
        from flash_head.inference import get_pipeline

        if size not in (256, 384, 512) or steps not in (2, 4):
            raise ValueError("Supported profiles: 256/384/512 pixels, 2/4 steps")
        if fps not in (15, 20, 25):
            raise ValueError("fps must be 15, 20, or 25")
        self.fps = fps
        torch.set_num_threads(4)
        implementation.COMPILE_MODEL = compile_model
        implementation.COMPILE_VAE = compile_model
        self.torch = torch
        self.size, self.steps = size, steps
        self.pipeline = get_pipeline(1, "models/SoulX-FlashHead-1_3B", "lite",
                                     "models/wav2vec2-base-960h")
        self.pipeline.audio_encoder.eval().requires_grad_(False)
        self.templates = OrderedDict()
        self.last_metrics = {}

    def prepare(self, image_path, audio, seed=42):
        torch = self.torch
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Audio must contain finite samples")
        # Content key avoids stale reference latents when an upload is replaced.
        with open(image_path, "rb") as source:
            key = (hashlib.sha256(source.read()).hexdigest(), self.size, self.steps)
        if key not in self.templates:
            p = copy.copy(self.pipeline)
            with torch.no_grad(), torch.random.fork_rng(devices=[0]):
                torch.manual_seed(0)
                p.prepare_params(image_path, (self.size, self.size), 33, 9,
                                 self.steps, seed=0, shift=5,
                                 color_correction_strength=1., use_face_crop=False)
            self.templates[key] = p
            while len(self.templates) > 8:
                self.templates.popitem(last=False)
        self.templates.move_to_end(key)
        p = copy.copy(self.templates[key])
        p.latent_motion_frames = p.ref_img_latent[:, :1].clone()
        p.generator = torch.Generator(device=p.device).manual_seed(int(seed))
        return GenerationState(p, audio, math.ceil(len(audio) * self.fps / self.sample_rate))

    def _audio(self, state):
        # Same causal rolling 8-second context as upstream streaming mode.
        end = round((state.cursor + self.chunk_frames) * self.sample_rate / self.fps)
        start = end - 8 * self.sample_rate
        window = np.zeros(8 * self.sample_rate, dtype=np.float32)
        lo, hi = max(0, start), min(end, len(state.audio))
        if hi > lo:
            window[lo - start:hi - start] = state.audio[lo:hi]
        embeddings = state.pipeline.preprocess_audio(window, sr=self.sample_rate, fps=self.fps)
        end_frame = 8 * self.fps
        torch = self.torch
        centers = torch.arange(end_frame - 33, end_frame, device=embeddings.device)[:, None]
        indices = (centers + torch.arange(-2, 3, device=embeddings.device)[None]).clamp_(0, end_frame - 1)
        return embeddings[indices][None].contiguous()

    def generate(self, states):
        """Return one uint8 RGB chunk per session, without retaining history."""
        torch = self.torch
        from flash_head.utils.utils import match_and_blend_colors_torch

        if not states or any(s.cursor >= s.total_frames for s in states):
            raise ValueError("Expected unfinished sessions")
        started = time.perf_counter()
        stamps = []
        def stamp():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            stamps.append(event)

        with torch.no_grad():
            stamp()
            contexts = torch.cat([self._audio(s) for s in states])
            p = states[0].pipeline
            shape = (p.config.out_dim, 5, p.lat_h, p.lat_w)
            def random_batch():
                return torch.stack([torch.randn(shape, device=p.device,
                    dtype=p.param_dtype, generator=s.pipeline.generator) for s in states])
            noise = random_batch()
            reference = torch.stack([s.pipeline.ref_img_latent for s in states])
            stamp()
            for i in range(len(p.timesteps) - 1):
                for j, state in enumerate(states):
                    motion = state.pipeline.latent_motion_frames
                    noise[j, :, :motion.shape[1]] = motion
                flow = p.model(x=noise, timestep=p.timesteps[i].expand(len(states)),
                               context=contexts, y=reference)
                t = (p.timesteps[i] / p.num_timesteps).to(p.param_dtype)
                nxt = (p.timesteps[i + 1] / p.num_timesteps).to(p.param_dtype)
                noise = (1 - nxt) * (noise - flow * t) + nxt * random_batch()
            stamp()
            outputs = []
            for j, state in enumerate(states):
                sp = state.pipeline
                motion = sp.latent_motion_frames
                noise[j, :, :motion.shape[1]] = motion
                videos = sp.vae.decode(noise[j])
                videos = match_and_blend_colors_torch(videos,
                    sp.original_color_reference, sp.color_correction_strength)
                posterior = sp.vae.model.encode(videos[:, :, -9:], return_dict=False)[0]
                sp.latent_motion_frames = sp.vae.normalize_latents(
                    posterior.sample(generator=sp.generator))[0]
                count = min(self.chunk_frames, state.total_frames - state.cursor)
                # Never transfer the nine overlap frames or float32 pixels to CPU.
                pixels = ((videos[0, :, 9:9 + count].float() + 1) * 127.5)
                pixels = pixels.clamp_(0, 255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
                outputs.append(pixels.cpu().numpy())
                state.cursor += count
                del videos, pixels, posterior
            stamp()
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        self.last_metrics = {
            "batch": len(states), "frames": sum(len(x) for x in outputs),
            "wall_s": elapsed,
            "audio_ms": stamps[0].elapsed_time(stamps[1]),
            "dit_ms": stamps[1].elapsed_time(stamps[2]),
            "vae_color_transfer_ms": stamps[2].elapsed_time(stamps[3]),
            "allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "reserved_mib": torch.cuda.memory_reserved() / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        }
        return outputs

    def warmup(self, image="examples/girl.png", batch_size=1):
        states = [self.prepare(image, np.zeros(64000, np.float32), 100 + i)
                  for i in range(batch_size)]
        self.generate(states)
        self.generate(states)  # Motion latent changes from one to two frames.
        self.torch.cuda.reset_peak_memory_stats()

    def validate_isolation(self):
        """Changing batch row order must not change either session's history."""
        speech = np.sin(np.arange(64000, dtype=np.float32) * .04) * .1
        forward = [self.prepare("examples/girl.png", speech * (i + 1), 71 + i) for i in range(2)]
        reverse = [self.prepare("examples/girl.png", speech * (i + 1), 71 + i) for i in range(2)][::-1]
        maximum = 0
        for _ in range(2):
            a, b = self.generate(forward), self.generate(reverse)[::-1]
            maximum = max(maximum, max(int(np.abs(x.astype(np.int16)-y.astype(np.int16)).max())
                                       for x,y in zip(a,b)))
        if maximum > 1:
            raise AssertionError(f"Batch permutation changed session output by {maximum} pixel levels")
        return {"chunks_checked": 2, "sessions": 2, "max_pixel_difference": maximum}
