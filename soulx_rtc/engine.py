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
    audio_offset: int = 0


def validate_geometry(size=512, width=None, height=None):
    if (width is None) != (height is None):
        raise ValueError("Specify both width and height")
    width, height = (size, size) if width is None else (width, height)
    if any(not isinstance(v, int) or v < 256 or v > 1024 or v % 32 for v in (width, height)):
        raise ValueError("Native width/height must be multiples of 32 between 256 and 1024")
    if width * height > 576 * 1024:
        raise ValueError("Profile exceeds the experimental 576×1024 pixel budget")
    return width, height


class Engine:
    fps = 25
    sample_rate = 16000
    chunk_frames = 24

    def __init__(self, size=512, steps=4, compile_model=True, fps=25, *,
                 width=None, height=None, optimized=False, profile=False,
                 real_rope=False, memory_mode="default", trt_ffn=None, trt_vae=None):
        import torch
        import flash_head.src.pipeline.flash_head_pipeline as implementation
        from flash_head.inference import get_pipeline

        self.width, self.height = validate_geometry(size, width, height)
        if steps not in (2, 4):
            raise ValueError("Supported denoising steps: 2/4")
        if fps not in (15, 20, 24, 25):
            raise ValueError("fps must be 15, 20, 24, or 25")
        if memory_mode not in ("default", "compact", "reference", "staged"):
            raise ValueError("memory_mode must be default, compact, reference or staged")
        self.optimized, self.profile, self.real_rope = optimized, profile, real_rope
        self.memory_mode = memory_mode
        self.fps = fps
        from .gpu_lease import acquire_gpu_lease
        self.gpu_lease = acquire_gpu_lease()
        torch.set_num_threads(4)
        implementation.COMPILE_MODEL = compile_model
        implementation.COMPILE_VAE = compile_model
        self.torch = torch
        self.size, self.steps = size, steps
        self.pipeline = get_pipeline(1, "models/SoulX-FlashHead-1_3B", "lite",
                                     "models/wav2vec2-base-960h")
        self.pipeline.audio_encoder.eval().requires_grad_(False)
        if memory_mode == "staged":
            self.pipeline.audio_encoder.to("cpu")
            torch.cuda.empty_cache()
        self.templates = OrderedDict()
        self.last_metrics = {}
        self.raw_model = getattr(self.pipeline.model, "_orig_mod", self.pipeline.model)
        trt_preparation_offload = bool(trt_ffn or trt_vae) and memory_mode == "reference"
        if trt_preparation_offload:
            # Deserialization needs temporary memory beyond steady inference.
            # Keep remaining Torch DiT weights on CPU while replacing partitions.
            self.move_dit("cpu", preparation=True)
        self.trt_vae = bool(trt_vae)
        if trt_vae:
            from .trt_backend import TRTFeedForward, file_hash
            weights = "models/SoulX-FlashHead-1_3B/VAE_LTX/diffusion_pytorch_model.safetensors"
            self.pipeline.vae.model.decoder.to("cpu")
            torch.cuda.empty_cache()
            self.pipeline.vae.decode = TRTFeedForward(trt_vae,(128,5,self.height//32,self.width//32),
                                                       file_hash(weights),lazy_workspace=True)
        self.trt_layers = []
        if trt_ffn:
            if memory_mode == "staged":
                raise ValueError("TensorRT partitions cannot use PyTorch weight offload")
            from .trt_backend import install_ffn_partitions
            self.trt_layers = install_ffn_partitions(self.raw_model,trt_ffn,self.width,self.height)
        if trt_preparation_offload:
            self.move_dit(self.pipeline.device, preparation=True)
        self.conditioning = (torch.compile(self.raw_model.prepare_conditioning)
                             if compile_model else self.raw_model.prepare_conditioning)
        self.profile_constants = {}
        self.compute_stream = torch.cuda.Stream()

    def move_dit(self, device, preparation=False):
        if self.memory_mode == "staged" or (self.memory_mode == "reference" and preparation):
            self.raw_model.to(device)
            self.torch.cuda.empty_cache()

    def constants(self, p, batch):
        key = (p.lat_h, p.lat_w, batch, self.steps, self.real_rope)
        if key not in self.profile_constants:
            from flash_head.src.modules.flash_head_model import prepare_rotary
            model = self.raw_model
            grid = (5, p.lat_h // model.patch_size[1], p.lat_w // model.patch_size[2])
            self.profile_constants[key] = (
                prepare_rotary(model.freqs.to(p.device), grid, self.real_rope),
                tuple(model.prepare_time(t.expand(batch)) for t in p.timesteps[:-1]))
        return self.profile_constants[key]

    def prepare(self, image_path, audio, seed=42):
        torch = self.torch
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Audio must contain finite samples")
        # Content key avoids stale reference latents when an upload is replaced.
        with open(image_path, "rb") as source:
            key = (hashlib.sha256(source.read()).hexdigest(), self.width, self.height, self.steps)
        if key not in self.templates:
            self.move_dit("cpu", preparation=True)
            p = copy.copy(self.pipeline)
            with torch.no_grad(), torch.random.fork_rng(devices=[0]):
                torch.manual_seed(0)
                p.prepare_params(image_path, (self.height, self.width), 33, 9,
                                 self.steps, seed=0, shift=5,
                                 color_correction_strength=1., use_face_crop=False)
                from flash_head.utils.utils import prepare_reference_color_stats
                p.reference_color_stats = prepare_reference_color_stats(p.original_color_reference)
            self.templates[key] = p
            self.move_dit(p.device, preparation=True)
            while len(self.templates) > 8:
                self.templates.popitem(last=False)
        self.templates.move_to_end(key)
        p = copy.copy(self.templates[key])
        p.latent_motion_frames = p.ref_img_latent[:, :1].clone()
        p.generator = torch.Generator(device=p.device).manual_seed(int(seed))
        return GenerationState(p, audio, math.ceil(len(audio) * self.fps / self.sample_rate))

    def _audio(self, state):
        # Same rolling 8-second window as upstream streaming mode. It reaches
        # the next chunk horizon; Wav2Vec attention is not sample-causal.
        end = round((state.cursor + self.chunk_frames) * self.sample_rate / self.fps)
        start = end - 8 * self.sample_rate
        window = np.zeros(8 * self.sample_rate, dtype=np.float32)
        lo, hi = max(state.audio_offset, start), min(end, state.audio_offset + len(state.audio))
        if hi > lo:
            window[lo - start:hi - start] = state.audio[lo - state.audio_offset:hi - state.audio_offset]
        embeddings = state.pipeline.preprocess_audio(window, sr=self.sample_rate, fps=self.fps)
        end_frame = 8 * self.fps
        torch = self.torch
        centers = torch.arange(end_frame - 33, end_frame, device=embeddings.device)[:, None]
        indices = (centers + torch.arange(-2, 3, device=embeddings.device)[None]).clamp_(0, end_frame - 1)
        return embeddings[indices][None].contiguous()

    def append(self, state, audio):
        """Append at a completed chunk boundary, retaining an eight-second history."""
        if state.cursor != state.total_frames or state.cursor % self.chunk_frames:
            raise ValueError("Append requires a drained whole-chunk generation boundary")
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        if not len(audio) or not np.isfinite(audio).all():
            raise ValueError("Audio must contain finite samples")
        start_sample = round(state.cursor * self.sample_rate / self.fps)
        end_sample = state.audio_offset + len(state.audio)
        if end_sample > start_sample:
            raise ValueError("Existing audio extends beyond the generation boundary")
        state.audio = np.concatenate((state.audio, np.zeros(start_sample-end_sample, np.float32), audio))
        state.total_frames = state.cursor + math.ceil(len(audio) * self.fps / self.sample_rate / self.chunk_frames) * self.chunk_frames
        return state.total_frames

    def prepare_call(self, path, seed=50):
        state = self.prepare(path, np.zeros(1, np.float32), seed)
        state.audio = np.zeros(0, np.float32)
        state.total_frames = 0
        return state

    def recondition(self, state, displayed_frames):
        """Explicit interruption reset to the last nine *sent* frames, not queued future."""
        torch = self.torch
        if displayed_frames.shape != (9, self.height, self.width, 3):
            raise ValueError("Reconditioning requires nine native RGB frames")
        with torch.no_grad():
            self.move_dit("cpu", preparation=True)
            pixels = torch.as_tensor(displayed_frames, device=state.pipeline.device)
            video = pixels.permute(3, 0, 1, 2)[None].to(state.pipeline.param_dtype) / 127.5 - 1
            posterior = state.pipeline.vae.model.encode(video, return_dict=False)[0]
            state.pipeline.latent_motion_frames = state.pipeline.vae.normalize_latents(
                posterior.sample(generator=state.pipeline.generator))[0]
            del pixels, video, posterior
            self.move_dit(state.pipeline.device, preparation=True)
        # Generation clock restarts; transport clock belongs to the call and never resets.
        state.audio = np.zeros(0, np.float32)
        state.audio_offset = state.cursor = state.total_frames = 0

    def generate(self, states):
        """Return one uint8 RGB chunk per session, without retaining history."""
        torch = self.torch
        from flash_head.utils.utils import match_and_blend_colors_torch

        if not states or any(s.cursor >= s.total_frames for s in states):
            raise ValueError("Expected unfinished sessions")
        started = time.perf_counter()
        stamps = []
        detail = []
        def phase(name):
            if self.profile:
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                detail.append((name, event))
        def stamp():
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            stamps.append(event)

        self.compute_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.compute_stream), torch.no_grad():
            stamp()
            phase("start")
            if self.memory_mode == "staged":
                # Keep DiT off-device during Wav2Vec too. After decode we leave
                # it on CPU until the next chunk's audio activations are gone.
                self.move_dit("cpu")
                phase("dit_offload_for_audio")
                self.pipeline.audio_encoder.to(self.pipeline.device)
                phase("audio_weights_reload")
            contexts = torch.cat([self._audio(s) for s in states])
            phase("audio")
            p = states[0].pipeline
            if self.memory_mode == "staged":
                self.pipeline.audio_encoder.to("cpu")
                torch.cuda.empty_cache()
                phase("audio_weights_offload")
                self.move_dit(p.device)
                phase("dit_reload_after_audio")
            shape = (p.config.out_dim, 5, p.lat_h, p.lat_w)
            def random_batch():
                return torch.stack([torch.randn(shape, device=p.device,
                    dtype=p.param_dtype, generator=s.pipeline.generator) for s in states])
            noise = random_batch()
            reference = torch.stack([s.pipeline.ref_img_latent for s in states])
            kwargs = {}
            times = None
            if self.optimized:
                context, kv = self.conditioning(contexts)
                rotary, times = self.constants(p, len(states))
                kwargs = dict(prepared_context=context, cross_kv=kv, rotary=rotary)
            phase("conditioning_noise_reference")
            stamp()
            for i in range(len(p.timesteps) - 1):
                for j, state in enumerate(states):
                    motion = state.pipeline.latent_motion_frames
                    noise[j, :, :motion.shape[1]] = motion
                flow = p.model(x=noise, timestep=p.timesteps[i].expand(len(states)),
                               context=contexts, y=reference,
                               **kwargs, prepared_time=None if times is None else times[i])
                t = (p.timesteps[i] / p.num_timesteps).to(p.param_dtype)
                nxt = (p.timesteps[i + 1] / p.num_timesteps).to(p.param_dtype)
                noise = (1 - nxt) * (noise - flow * t) + nxt * random_batch()
                phase(f"dit_step_{i}")
            stamp()
            # Release conditioning tensors before VAE; weights remain shared.
            kwargs.clear()
            if self.optimized:
                del context, kv, rotary
            del contexts, reference, flow
            if self.trt_layers:
                # TensorRT weights/workspace live outside the caching allocator;
                # release unused DiT allocations before the high-water VAE stage.
                torch.cuda.empty_cache()
            self.move_dit("cpu")
            phase("dit_offload")
            outputs = []
            for j, state in enumerate(states):
                sp = state.pipeline
                motion = sp.latent_motion_frames
                noise[j, :, :motion.shape[1]] = motion
                videos = sp.vae.decode(noise[j])
                phase(f"decode_{j}")
                if self.trt_vae:
                    sp.vae.decode.release_workspace()
                    phase(f"decode_workspace_release_{j}")
                if self.memory_mode in ("compact", "reference", "staged"):
                    videos = torch.cat([match_and_blend_colors_torch(videos[:, :, k:k+4],
                        sp.original_color_reference, sp.color_correction_strength,
                        reference_stats=sp.reference_color_stats if self.optimized else None)
                        for k in range(0, videos.shape[2], 4)], dim=2)
                else:
                    videos = match_and_blend_colors_torch(videos,
                        sp.original_color_reference, sp.color_correction_strength,
                        reference_stats=sp.reference_color_stats if self.optimized else None)
                phase(f"color_{j}")
                posterior = sp.vae.model.encode(videos[:, :, -9:], return_dict=False)[0]
                sp.latent_motion_frames = sp.vae.normalize_latents(
                    posterior.sample(generator=sp.generator))[0]
                phase(f"motion_encode_{j}")
                count = min(self.chunk_frames, state.total_frames - state.cursor)
                # Never transfer the nine overlap frames or float32 pixels to CPU.
                pixels = ((videos[0, :, 9:9 + count].float() + 1) * 127.5)
                pixels = pixels.clamp_(0, 255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()
                phase(f"uint8_{j}")
                outputs.append(pixels.cpu().numpy())
                phase(f"transfer_{j}")
                state.cursor += count
                # Long-lived calls retain only rolling audio history + future input.
                trim_to = max(state.audio_offset, round(state.cursor * self.sample_rate / self.fps) - 8*self.sample_rate)
                trim = min(len(state.audio), trim_to - state.audio_offset)
                state.audio = state.audio[trim:].copy()
                state.audio_offset += trim
                del videos, pixels, posterior
            if self.memory_mode != "staged":
                self.move_dit(p.device)
                phase("dit_reload")
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
            "width": self.width, "height": self.height,
            "optimized": self.optimized, "real_rope": self.real_rope,
            "trt_ffn_layers": self.trt_layers,
            "trt_vae": self.trt_vae,
            "trt_vae_workspace": "transient" if self.trt_vae else None,
            "stages_ms": {name: a.elapsed_time(b) for (_, a), (name, b) in zip(detail, detail[1:])},
        }
        return outputs

    def warmup(self, image="examples/girl.png", batch_size=1):
        states = [self.prepare(image, np.zeros(64000, np.float32), 100 + i)
                  for i in range(batch_size)]
        self.generate(states)
        self.generate(states)  # Motion latent changes from one to two frames.
        self.torch.cuda.reset_peak_memory_stats()

    def validate_isolation(self, image="examples/girl.png", batch_size=2):
        """Changing batch row order must not change either session's history."""
        speech = np.sin(np.arange(64000, dtype=np.float32) * .04) * .1
        forward = [self.prepare(image, speech * (i + 1), 71 + i) for i in range(2)]
        reverse = [self.prepare(image, speech * (i + 1), 71 + i) for i in range(2)][::-1]
        def render(states):
            return self.generate(states) if batch_size>=2 else [self.generate([s])[0] for s in states]
        maximum = 0
        for _ in range(2):
            a, b = render(forward), render(reverse)[::-1]
            maximum = max(maximum, max(int(np.abs(x.astype(np.int16)-y.astype(np.int16)).max())
                                       for x,y in zip(a,b)))
        if maximum > 1:
            raise AssertionError(f"Batch permutation changed session output by {maximum} pixel levels")
        return {"chunks_checked": 2, "sessions": 2, "batch":min(batch_size,2), "max_pixel_difference": maximum}
