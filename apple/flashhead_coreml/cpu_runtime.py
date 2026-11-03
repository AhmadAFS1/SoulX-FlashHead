"""CPU-only PyTorch reproduction of the SoulX Studio PRO window loop.

This mirrors ``ios/SoulXStudio/Sources/Runtime/SoulXEngine.swift`` for fixed
avatar caches so transformer changes can be compared as rendered video on a
machine without Core ML. It reuses the portable export modules, the Swift
SplitMix64/Box-Muller noise stream, the [1000, 625, 0] re-noising update,
window-mode tiny decoding, Lab colour transfer, five-frame motion feedback and
truncating 8-bit display conversion.

Two numerical modes are provided:

``fp32``
    Portable FP32 PyTorch with original checkpoint weights. This is the
    reference used to measure how far any Apple-style variant drifts.
``coreml-int8``
    CPU emulation of the shipped bundle's arithmetic contract: Core ML Tools'
    per-channel symmetric INT8 weights (computed with coremltools' own
    quantizer on FP16-cast weights), FP16 tensors between operations, and FP32
    layer/RMS normalisation and attention, matching ``export_dit``'s ``mixed``
    policy. FP16 rounding is applied to operation outputs; accumulation stays
    FP32. It is an emulation, not Core ML execution: no GPU, Neural Engine or
    Apple kernel ran, and device-specific accumulation order is not reproduced.
    On validate_dit's seeded cases it reaches about 0.86x (0.35-0.88x) of the
    Core ML CPU error recorded on the M5 Pro, so it is a slight lower bound.
    The tiny VAE decoder and motion encoder run in FP32 with FP16 inputs and
    outputs; the bundle computes them in FP16 (about 0.33% / 0.40% NRMSE,
    apple/validation/tiny_pro320.json). Compared arms share the same VAE, so
    transformer A/B results are unaffected; FP32-vs-emulation gaps omit it.

Neither mode measures Apple speed. Timings recorded here are CPU PyTorch wall
times on the machine that ran them.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import time
from typing import Callable

import numpy as np
import torch
from torch.nn import functional as F

from .dit import (AudioProjection, Blocks, OutputHead, Patch, ProConfig, Profile, TimeEmbedding,
                  attention, load_component, rotary_tables)

ROOT = Path(__file__).resolve().parents[2]
AVATARS = ROOT / 'ios/SoulXStudio/Resources/Avatars'
SAMPLE_RATE, FPS, USEFUL, HISTORY, WINDOW = 16000, 25, 28, 5, 33
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
SCHEDULE = (1000.0, 625.0, 0.0)
MODES = ('fp32', 'coreml-int8')


# --------------------------------------------------------------------------
# Deterministic noise: bit-compatible with SoulXNoise in SoulXTensor.swift.
# --------------------------------------------------------------------------
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)


class SoulXNoise:
    """SplitMix64 uniforms and Box-Muller normals in Swift's consumption order."""

    def __init__(self, seed: int = 42):
        self.state = np.uint64(seed)
        self.spare: np.float32 | None = None

    def uniforms(self, count: int) -> np.ndarray:
        with np.errstate(over='ignore'):
            steps = np.arange(1, count + 1, dtype=np.uint64)
            z = self.state + steps * _GOLDEN
            self.state = self.state + np.uint64(count) * _GOLDEN
            z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
            z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
            z ^= z >> np.uint64(31)
        return ((z >> np.uint64(11)).astype(np.float64) + 0.5) / 9007199254740992.0

    def normals(self, count: int) -> np.ndarray:
        out = np.empty(count, dtype=np.float32)
        start = 0
        if self.spare is not None and count:
            out[0], self.spare, start = self.spare, None, 1
        pairs = (count - start + 1) // 2
        if pairs:
            u = self.uniforms(2 * pairs).reshape(pairs, 2)
            radius = np.sqrt(-2.0 * np.log(u[:, 0]))
            angle = 2.0 * np.pi * u[:, 1]
            values = np.stack(((radius * np.cos(angle)).astype(np.float32),
                               (radius * np.sin(angle)).astype(np.float32)), axis=1).reshape(-1)
            need = count - start
            out[start:] = values[:need]
            if need % 2:
                self.spare = values[-1]
        return out


# --------------------------------------------------------------------------
# Colour: vectorised float32 port of SoulXImage.lab/rgb/statistics.
# --------------------------------------------------------------------------
def lab(rgb: np.ndarray) -> np.ndarray:
    """rgb [..., 3] float32 in [0, 1] -> Lab [..., 3]."""
    rgb = rgb.astype(np.float32)
    lin = np.where(rgb > np.float32(0.04045), np.power((rgb + np.float32(0.055)) / np.float32(1.055), np.float32(2.4)),
                   rgb / np.float32(12.92)).astype(np.float32)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = (np.float32(0.4124564) * r + np.float32(0.3575761) * g + np.float32(0.1804375) * b) / np.float32(0.95047)
    y = np.float32(0.2126729) * r + np.float32(0.7151522) * g + np.float32(0.0721750) * b
    z = (np.float32(0.0193339) * r + np.float32(0.1191920) * g + np.float32(0.9503041) * b) / np.float32(1.08883)

    def f(v):
        v = np.clip(v, np.float32(1e-8), np.float32(1)).astype(np.float32)
        return np.where(v > np.float32(0.008856), np.power(v, np.float32(1 / 3)),
                        (np.float32(903.3) * v + np.float32(16)) / np.float32(116)).astype(np.float32)

    fx, fy, fz = f(x), f(y), f(z)
    return np.stack((np.float32(116) * fy - np.float32(16), np.float32(500) * (fx - fy), np.float32(200) * (fy - fz)),
                    axis=-1).astype(np.float32)


def rgb_from_lab(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    L, A, B = values[..., 0], values[..., 1], values[..., 2]
    fy = (L + np.float32(16)) / np.float32(116)
    fx = A / np.float32(500) + fy
    fz = fy - B / np.float32(200)

    def inverse(f):
        cube = f * f * f
        return np.where(cube > np.float32(0.008856), cube, (np.float32(116) * f - np.float32(16)) / np.float32(903.3))

    x = inverse(fx) * np.float32(0.95047)
    y = np.where(L > np.float32(903.3 * 0.008856), fy * fy * fy, L / np.float32(903.3))
    z = inverse(fz) * np.float32(1.08883)

    def gamma(v):
        v = v.astype(np.float32)
        out = np.where(v > np.float32(0.0031308),
                       np.float32(1.055) * np.power(np.maximum(v, np.float32(0)), np.float32(1 / 2.4)) - np.float32(0.055),
                       np.float32(12.92) * v)
        return np.clip(out, 0, 1).astype(np.float32)

    return np.stack((gamma(np.float32(3.2404542) * x - np.float32(1.5371385) * y - np.float32(0.4985314) * z),
                     gamma(np.float32(-0.9692660) * x + np.float32(1.8760108) * y + np.float32(0.0415560) * z),
                     gamma(np.float32(0.0556434) * x - np.float32(0.2040259) * y + np.float32(1.0572252) * z)),
                    axis=-1).astype(np.float32)


def statistics(lab_pixels: np.ndarray):
    flat = lab_pixels.reshape(-1, 3).astype(np.float64)
    mean = flat.mean(axis=0)
    variance = (flat * flat).mean(axis=0) - mean * mean
    return mean.astype(np.float32), np.sqrt(np.maximum(0, variance)).astype(np.float32)


def prepare_portrait(path: Path, height: int, width: int) -> np.ndarray:
    """Aspect-fill centre crop to HxW RGB float32 [0, 1] (Pillow Lanczos).

    Swift draws into an sRGB CoreGraphics context, which colour-matches an
    embedded ICC profile (the girl portrait carries Display P3); this applies
    the same conversion before resampling. CoreGraphics' resampler is not
    reproduced bit for bit. The result only sets the reference colour
    statistics, identically in every compared arm.
    """
    import io
    from PIL import Image, ImageCms
    image = Image.open(path)
    icc = image.info.get('icc_profile')
    if icc:
        source = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        image = ImageCms.profileToProfile(image.convert('RGB'), source, ImageCms.createProfile('sRGB'),
                                          renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC, outputMode='RGB')
    image = image.convert('RGB')
    scale = max(width / image.width, height / image.height)
    size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
    image = image.resize(size, Image.Resampling.LANCZOS)
    left, top = (size[0] - width) // 2, (size[1] - height) // 2
    image = image.crop((left, top, left + width, top + height))
    return np.asarray(image, dtype=np.float32) / 255.0


def correct_frame(frame_chw: np.ndarray, mean: np.ndarray, std: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """SoulXImage.correctedFrame for one decoded frame [3, H, W] -> [3, H, W]."""
    rgb = np.clip(frame_chw.astype(np.float32), 0, 1).transpose(1, 2, 0)
    if strength <= 0:
        return rgb.transpose(2, 0, 1)
    pixels = lab(rgb)
    source_mean, source_std = statistics(pixels)
    scale = std / np.where(source_std < 1e-8, np.float32(1), source_std)
    transferred = rgb_from_lab((pixels - source_mean) * scale + mean)
    blended = rgb * np.float32(1 - strength) + transferred * np.float32(strength)
    return blended.transpose(2, 0, 1).astype(np.float32)


def display_uint8(frame_chw: np.ndarray) -> np.ndarray:
    """Swift UInt8(min(255, max(0, v * 255))) truncates toward zero."""
    return np.clip(frame_chw * np.float32(255), 0, 255).astype(np.uint8).transpose(1, 2, 0)


# --------------------------------------------------------------------------
# Audio: SoulXEngine history handling and SoulXAudio.windows.
# --------------------------------------------------------------------------
def read_audio(path: Path) -> np.ndarray:
    import soundfile as sf
    data, rate = sf.read(str(path), dtype='float32', always_2d=True)
    data = data.mean(axis=1)
    if rate != SAMPLE_RATE:
        import torchaudio.functional as AF
        data = AF.resample(torch.from_numpy(data), rate, SAMPLE_RATE).numpy()
    return data.astype(np.float32)


def audio_windows(features: torch.Tensor) -> torch.Tensor:
    """[1, 200, 12, 768] -> [1, 33, 5, 12, 768] using Swift's clamped centres."""
    index = torch.tensor([[min(199, 167 + frame + slot - 2) for slot in range(5)] for frame in range(33)])
    return features[0, index].unsqueeze(0)


# --------------------------------------------------------------------------
# Apple-style numerics.
# --------------------------------------------------------------------------
def round16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.float16).to(torch.float32)


def identity(x: torch.Tensor) -> torch.Tensor:
    return x


def coreml_int8_dequantized_reference(weight: torch.Tensor) -> torch.Tensor:
    """Core ML Tools linear_symmetric/int8/per_channel on the FP16 const weight.

    Calls coremltools' own quantizer (slow: NumPy float16 arithmetic)."""
    from coremltools.converters.mil.mil import types
    from coremltools.optimize._utils import compute_qparams_by_dtype
    w16 = weight.detach().to(torch.float16).numpy()
    blocks = [1] + [0] * (w16.ndim - 1)
    quantized, scale, _ = compute_qparams_by_dtype(w16, types.int8, 'LINEAR_SYMMETRIC', blocks)
    scale = scale.reshape([-1] + [1] * (w16.ndim - 1)).astype(np.float16)
    restored = (quantized.astype(np.float16) * scale).astype(np.float16)
    return torch.from_numpy(restored.astype(np.float32))


def _half_op(value: torch.Tensor) -> torch.Tensor:
    """NumPy float16 ufuncs compute in float32 and round the result to float16."""
    return value.to(torch.float16).to(torch.float32)


def coreml_int8_dequantized(weight: torch.Tensor) -> torch.Tensor:
    """Vectorised port of ``coreml_int8_dequantized_reference`` (tests assert equality).

    Per output channel: scale = fp16((max|w| - (-max|w|)) / 254); q =
    clip(round_half_even(fp16(w / scale)), -127, 127); w' = fp16(q * scale).
    """
    w = weight.detach().to(torch.float16).to(torch.float32)
    rows = w.reshape(w.shape[0], -1)
    peak = torch.maximum(rows.amin(1).abs(), rows.amax(1).abs())
    scale = _half_op(_half_op(peak - (-peak)) / 254.0)
    # An all-zero channel has scale 0; NumPy's 0/0 = NaN casts to int8 0.
    quantized = torch.nan_to_num(torch.round(_half_op(rows / scale[:, None])), nan=0.0).clamp_(-127, 127)
    return _half_op(quantized * scale[:, None]).reshape(weight.shape)


def apply_int8_weights(module: torch.nn.Module, threshold: int = 1024) -> int:
    """Replace learned linear/conv weights in place; returns the count replaced.

    A folded cross-attention output (``o_heads`` [heads, 128, dim]) is quantized
    with one scale per output channel over all heads, which reproduces the
    original ``W_o`` row quantization value for value (export: per_block 0,0,1).
    """
    count = 0
    for child in module.modules():
        if isinstance(child, (torch.nn.Linear, torch.nn.Conv2d)) and child.weight.numel() >= threshold:
            child.weight.data.copy_(coreml_int8_dequantized(child.weight.data))
            count += 1
        heads = getattr(child, 'o_heads', None)
        if isinstance(heads, torch.Tensor):
            h, d, dim = heads.shape
            rows = heads.data.permute(2, 0, 1).reshape(dim, h * d)
            heads.data.copy_(coreml_int8_dequantized(rows).reshape(dim, h, d).permute(1, 2, 0))
            count += 1
    return count


def round_parameters16(module: torch.nn.Module):
    """FP16 constants of FP16 ops: linear/conv biases and modulation tables.

    Normalisation weights belong to FP32 islands under the mixed policy and
    keep FP32 values.
    """
    for child in module.modules():
        if isinstance(child, (torch.nn.Linear, torch.nn.Conv2d)) and child.bias is not None:
            child.bias.data.copy_(round16(child.bias.data))
        if isinstance(getattr(child, 'o_bias', None), torch.Tensor):
            child.o_bias.data.copy_(round16(child.o_bias.data))
        modulation = getattr(child, 'modulation', None)
        if isinstance(modulation, torch.nn.Parameter):
            modulation.data.copy_(round16(modulation.data))


def linear(layer: torch.nn.Linear, x: torch.Tensor, R) -> torch.Tensor:
    weight = layer.weight if layer.weight.dtype == x.dtype else layer.weight.to(x.dtype)
    return R(F.linear(x, weight, layer.bias))


def load_blocks_lean(cfg, profile, time_modulations, checkpoint, private=True):
    """All 30 blocks without a second full-size random-initialised copy.

    ``private=False`` keeps safetensors' file-backed (reclaimable) pages for
    read-only FP32 rendering; emulation writes weights and needs a copy.
    """
    from safetensors import safe_open
    with torch.device('meta'):
        blocks = Blocks(cfg, profile, 0, cfg.num_layers, torch.zeros(2, 6, cfg.dim))
    tensors = {}
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        for key in blocks.state_dict():
            # clone(): safetensors tensors share one memory map of the 6 GB file;
            # any surviving reference would keep every touched page resident.
            tensor = source.get_tensor(key).float()
            tensors[key] = tensor.clone() if private else tensor
    blocks.load_state_dict(tensors, strict=True, assign=True)
    del tensors
    cos, sin = rotary_tables(cfg.dim // cfg.num_heads, profile.grid)
    for name, value in (('rotary_cos', cos), ('rotary_sin', sin), ('time_high', time_modulations[0:1].detach()),
                        ('time_low', time_modulations[1:2].detach())):
        blocks.register_buffer(name, value, persistent=False)
    return blocks.eval().requires_grad_(False)


def store_half(module: torch.nn.Module):
    """Keep FP16-exact emulated weights as FP16 to halve memory (upcast per use)."""
    for child in module.modules():
        if isinstance(child, torch.nn.Linear):
            child.weight.data = child.weight.data.to(torch.float16)
        heads = getattr(child, 'o_heads', None)
        if isinstance(heads, torch.Tensor):
            heads.data = heads.data.to(torch.float16)


def rms(norm, x, R):
    y = x.float()
    return R(y * torch.rsqrt((y * y).mean(dim=-1, keepdim=True) + norm.eps) * norm.weight)


def rotary16(x, cos, sin, R):
    pairs = x.reshape(*x.shape[:-1], -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((R(R(a * cos) - R(b * sin)), R(R(a * sin) + R(b * cos))), dim=-1).flatten(-2)


def ffn_forward(block, x, step, R):
    """Dense FFN, or a compensated reduced FFN when the block carries one."""
    ffn = block.ffn
    hidden = R(F.gelu(linear(ffn[0], x, R), approximate='tanh'))
    out = linear(ffn[2], hidden, R)
    bias = getattr(block, 'ffn_step_bias', None)
    if bias is not None:
        out = R(out + bias[step])
    return out


def block_forward(block, x, context, modulation, cos, sin, step, R, observe=None):
    """DiTBlock.forward with explicit Apple-mixed rounding points.

    With R=identity this is DiTBlock.forward (asserted in the tests). ``observe``
    receives named intermediate tensors for calibration statistics.
    """
    e = [R(v) for v in (block.modulation + modulation).chunk(6, dim=1)]
    b, n, dim = x.shape
    sa, ca = block.self_attn, block.cross_attn
    heads = sa.heads
    norm1 = R(F.layer_norm(x, (dim,), eps=block.norm1.eps))
    a_in = R(R(norm1 * R(1 + e[1])) + e[0])
    q = rms(sa.norm_q, linear(sa.q, a_in, R), R)
    k = rms(sa.norm_k, linear(sa.k, a_in, R), R)
    v = linear(sa.v, a_in, R)
    q = rotary16(q.reshape(b, n, heads, dim // heads), cos, sin, R).flatten(2)
    k = rotary16(k.reshape(b, n, heads, dim // heads), cos, sin, R).flatten(2)
    x = R(x + R(linear(sa.o, R(attention(q, k, v, heads)), R) * e[2]))
    times = int(context.shape[1])
    n3 = R(F.layer_norm(x, (dim,), block.norm3.weight, block.norm3.bias, block.norm3.eps))
    qc = n3.reshape(b * times, n // times, dim)
    c = context.reshape(b * times, int(context.shape[2]), dim)
    cq = rms(ca.norm_q, linear(ca.q, qc, R), R)
    ck = rms(ca.norm_k, linear(ca.k, c, R), R)
    cv = linear(ca.v, c, R)
    if ca.o_heads is None:
        x = R(x + linear(ca.o, R(attention(cq, ck, cv, heads)), R).reshape(b, n, dim))
    else:
        # Folded form (dit.CrossAttention.fold_output_projection): FP32 probabilities,
        # per-head value projection and one [rows, heads*32] x [heads*32, dim] product.
        frames, rows, keys, d = cq.shape[0], cq.shape[1], ck.shape[1], dim // heads
        qh = cq.reshape(frames, rows, heads, d).transpose(1, 2)
        kh = ck.reshape(frames, keys, heads, d).transpose(1, 2)
        vh = cv.reshape(frames, keys, heads, d).transpose(1, 2)
        probabilities = R(ca.weights(qh, kh))
        values = R(torch.matmul(vh, ca.o_heads.to(vh.dtype)))
        out = R(torch.matmul(probabilities.transpose(1, 2).reshape(frames, rows, heads * keys),
                             values.reshape(frames, heads * keys, dim)))
        x = R(x + R(out + ca.o_bias).reshape(b, n, dim))
    n2 = R(F.layer_norm(x, (dim,), eps=block.norm2.eps))
    f_in = R(R(n2 * R(1 + e[4])) + e[3])
    if observe is not None:
        observe('ffn_in', f_in)
    return R(x + R(ffn_forward(block, f_in, step, R) * e[5]))


# --------------------------------------------------------------------------
# Runtime.
# --------------------------------------------------------------------------
def acquire_load_lock():
    """Exclusive lock while a runtime loads; SOULX_CPU_LOAD_LOCK=off disables it."""
    import fcntl
    path = os.environ.get('SOULX_CPU_LOAD_LOCK', str(ROOT / 'artifacts/cpu-validation/.load.lock'))
    if path == 'off':
        return None
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, 'w')
    fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def release_load_lock(handle):
    if handle is not None:
        import fcntl
        fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def release_freed_memory():
    """Return freed heap pages to the OS (glibc only; a no-op elsewhere)."""
    import ctypes
    import gc
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except (OSError, AttributeError):
        pass


@dataclass
class WindowRecord:
    index: int
    visible: int
    seconds: dict = field(default_factory=dict)


class CpuRuntime:
    """Loads every PRO component once; renders one avatar/audio clip at a time."""

    def __init__(self, mode='coreml-int8', profile=Profile(320, 176), threads=4,
                 checkpoint_dir=ROOT / 'models/SoulX-FlashHead-1_3B/Model_Pro',
                 blocks_per_partition=3, transforms: list[Callable] | None = None):
        if mode not in MODES:
            raise ValueError(f'mode must be one of {MODES}')
        # Loading briefly holds FP32 and FP16 copies (~8.5 GB); serialise loads
        # across concurrent renders and return freed pages afterwards.
        lock = acquire_load_lock()
        torch.set_num_threads(threads)
        self.mode, self.profile, self.threads = mode, profile, threads
        self.R = round16 if mode == 'coreml-int8' else identity
        # FP16 model-interface tensors (features, latents, frames) in the bundle.
        self.io = round16 if mode == 'coreml-int8' else identity
        self.blocks_per_partition = blocks_per_partition
        checkpoint = Path(checkpoint_dir) / 'diffusion_pytorch_model.safetensors'
        self.checkpoint = checkpoint
        self.cfg = ProConfig.from_json(Path(checkpoint_dir) / 'config.json')
        with torch.inference_mode():
            timing = load_component(TimeEmbedding(self.cfg), checkpoint, 'time')
            self.time_embeddings, self.time_modulations = timing(torch.tensor([1000., 625.]))
        self.patch = load_component(Patch(self.cfg, profile), checkpoint, 'patch')
        self.conditioning = load_component(AudioProjection(self.cfg.dim), checkpoint, 'conditioning')
        self.blocks = load_blocks_lean(self.cfg, profile, self.time_modulations, checkpoint,
                                       private=mode == 'coreml-int8' or bool(transforms))
        self.head = load_component(OutputHead(self.cfg, profile, self.time_embeddings), checkpoint, 'head')
        self.transform_log = []
        for transform in transforms or []:
            self.transform_log.append(transform(self))
        if mode == 'coreml-int8':
            self.quantized_layers = sum(apply_int8_weights(m) for m in (self.patch, self.conditioning, self.blocks, self.head))
            for m in (self.patch, self.conditioning, self.blocks, self.head):
                round_parameters16(m)
            for block in self.blocks.blocks:
                if getattr(block, 'ffn_step_bias', None) is not None:
                    block.ffn_step_bias = round16(block.ffn_step_bias)
            self.time_modulations = round16(self.time_modulations)
            self.time_embeddings = round16(self.time_embeddings)
            store_half(self.blocks)
        else:
            self.quantized_layers = 0
        self.rotary = (self.R(self.blocks.rotary_cos), self.R(self.blocks.rotary_sin))
        self._load_support()
        release_load_lock(lock)
        release_freed_memory()
        self.observer: Callable | None = None

    def _load_support(self):
        from transformers import Wav2Vec2Model
        from .export_support import Audio, TinyDecode, TinyEncode, module_at
        model = Wav2Vec2Model.from_pretrained(str(ROOT / 'models/wav2vec2-base-960h'), attn_implementation='eager').eval()
        self.audio = Audio(model).eval()
        vendor = module_at('soulx_cpu_taehv', ROOT / 'models/tiny_vae/taehv/taehv.py')
        tae = vendor.TAEHV(checkpoint_path=str(ROOT / 'models/tiny_vae/taehv/taew2_1.pth'), arch_name='taew2_1').float().eval()
        self.decoder = TinyDecode(tae, vendor.MemBlock).eval()
        self.encoder = TinyEncode(tae, vendor.MemBlock).eval()

    # -- model calls ------------------------------------------------------
    def dit_blocks(self, hidden, context, step):
        R = self.R
        modulation = self.time_modulations[step:step + 1]
        cos, sin = self.rotary
        x = hidden
        for index, block in enumerate(self.blocks.blocks):
            observe = None
            if self.observer is not None:
                observe = lambda name, value, i=index: self.observer(name, i, step, value)
            x = block_forward(block, x, context, modulation, cos, sin, step, R, observe)
            if (index + 1) % self.blocks_per_partition == 0:
                x = self.io(x)  # FP16 partition interface
        return x

    def head_flow(self, hidden, step):
        R = self.R
        head = self.head
        t = self.time_embeddings[step:step + 1]
        shift, scale = (R(head.modulation) + t[:, None, :]).chunk(2, dim=1)
        shift, scale = R(shift), R(scale)
        x = R(F.layer_norm(hidden, (self.cfg.dim,), eps=head.norm.eps))
        x = linear(head.head, R(R(x * R(1 + scale)) + shift), R)
        gt, gh, gw = self.profile.grid
        x = x.reshape(gt, gh * gw, 4, self.cfg.out_dim).permute(0, 3, 2, 1)
        x = F.pixel_shuffle(x.reshape(gt, self.cfg.out_dim * 4, gh, gw), 2)
        return R(x.permute(1, 0, 2, 3).unsqueeze(0))

    def patch_hidden(self, latent, reference):
        if self.mode == 'fp32':
            return self.patch(latent, reference)
        R = self.R
        x = torch.cat((R(latent), R(reference)), dim=1)
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = R(self.patch.patch_embedding(x))
        return x.reshape(b, t, self.cfg.dim, h // 2, w // 2).permute(0, 1, 3, 4, 2).reshape(b, -1, self.cfg.dim)

    def context(self, packed):
        if self.mode == 'fp32':
            return self.conditioning(packed)
        R = self.R
        m = self.conditioning
        b = packed.shape[0]
        audio = R(packed)
        features = m.layers * m.channels
        first = audio[:, :1].reshape(b, 1, -1)
        latter = audio[:, 1:].reshape(b, 8, 4, 5, features)
        packed = torch.cat((latter[:, :, :1, :3].reshape(b, 8, 3, features),
                            latter[:, :, 1:-1, 2:3].reshape(b, 8, 2, features),
                            latter[:, :, -1:, 2:].reshape(b, 8, 3, features)), dim=2)
        first = R(F.relu(linear(m.proj1, first, R)))
        latter = R(F.relu(linear(m.proj1_vf, packed.reshape(b, 8, -1), R)))
        x = torch.cat((first, latter), dim=1)
        x = R(F.relu(linear(m.proj2, x, R)))
        x = linear(m.proj3, x, R).reshape(b, 9, m.tokens, m.dim)
        return R(m.norm(x))

    # -- clip -------------------------------------------------------------
    @torch.inference_mode()
    def render(self, avatar: str, audio_path: Path, seed=42, max_chunks=None, on_window=None, forced_motion=None):
        """Render one clip. ``forced_motion`` (a previous run's ``latents.npy``)
        teacher-forces each window's motion latents, isolating per-window
        transformer differences from closed-loop trajectory divergence."""
        profile = self.profile
        avatar_dir = AVATARS / avatar
        meta = json.loads((avatar_dir / 'reference-cache/metadata.json').read_text())
        if tuple(meta['shape']) != profile.latent_shape:
            raise ValueError('Reference cache profile mismatch')
        raw = (avatar_dir / 'reference-cache' / meta['tensor_file']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != meta['tensor_sha256']:
            raise ValueError('Reference cache hash mismatch')
        reference = torch.from_numpy(np.frombuffer(raw, dtype='<f4').copy()).reshape(profile.latent_shape)
        portrait = prepare_portrait(avatar_dir / 'portrait.png', profile.height, profile.width)
        ref_mean, ref_std = statistics(lab(portrait))
        _, _, t_lat, lh, lw = profile.latent_shape
        plane = lh * lw
        motion = reference[:, :, :1].clone()
        noise = SoulXNoise(seed)
        audio = read_audio(audio_path)
        total = math.ceil(len(audio) * FPS / SAMPLE_RATE)
        history = np.zeros(128000, dtype=np.float32)
        per_chunk = USEFUL * SAMPLES_PER_FRAME
        emitted, frames, records, latents = 0, [], [], []
        while emitted < total:
            started = time.perf_counter()
            record = WindowRecord(len(records), min(USEFUL, total - emitted))
            chunk = np.zeros(per_chunk, dtype=np.float32)
            piece = audio[emitted * SAMPLES_PER_FRAME: emitted * SAMPLES_PER_FRAME + per_chunk]
            chunk[:len(piece)] = piece
            history = np.concatenate((history[per_chunk:], chunk))
            t0 = time.perf_counter()
            features = self.io(self.audio(torch.from_numpy(history)[None]))
            if not torch.isfinite(features).all():
                raise RuntimeError('Non-finite audio features')
            context = self.context(audio_windows(features))
            record.seconds['audio_and_conditioning'] = time.perf_counter() - t0
            if forced_motion is not None:
                count = 1 if not records else 2
                motion = torch.from_numpy(np.ascontiguousarray(forced_motion[len(records), :, :count]))[None]
            latent = torch.from_numpy(noise.normals(16 * t_lat * plane)).reshape(profile.latent_shape)
            for step in range(2):
                latent[:, :, :motion.shape[2]] = motion
                t0 = time.perf_counter()
                hidden = self.patch_hidden(latent, reference)
                hidden = self.dit_blocks(hidden, context, step)
                flow = self.head_flow(hidden, step)
                record.seconds[f'dit_step{step}'] = time.perf_counter() - t0
                now, nxt = SCHEDULE[step] / 1000, SCHEDULE[step + 1] / 1000  # exact: 1, 0.625, 0
                fresh = torch.from_numpy(noise.normals(latent.numel())).reshape(latent.shape)
                latent = (1 - nxt) * (latent - now * flow) + nxt * fresh
                if not torch.isfinite(latent).all():
                    raise RuntimeError('Denoising produced non-finite values')
            latent[:, :, :motion.shape[2]] = motion
            latents.append(latent.clone())
            t0 = time.perf_counter()
            decoded = self.io(self.decoder(self.io(latent.permute(0, 2, 1, 3, 4)))).numpy()[0]
            corrected = [correct_frame(decoded[i], ref_mean, ref_std) for i in range(HISTORY, WINDOW)]
            frames.extend(display_uint8(f) for f in corrected[:record.visible])
            record.seconds['decode_and_color'] = time.perf_counter() - t0
            t0 = time.perf_counter()
            feedback = torch.from_numpy(np.stack(corrected[-HISTORY:]))[None]
            motion = self.io(self.encoder(self.io(feedback)))
            if not torch.isfinite(motion).all():
                raise RuntimeError('Motion encoder produced non-finite values')
            record.seconds['motion_encode'] = time.perf_counter() - t0
            record.seconds['window'] = time.perf_counter() - started
            records.append(record)
            emitted += record.visible
            if on_window:
                on_window(record)
            if max_chunks and len(records) >= max_chunks:
                break
        clip_audio = np.zeros(emitted * SAMPLES_PER_FRAME, dtype=np.float32)
        clip_audio[:min(len(audio), len(clip_audio))] = audio[:len(clip_audio)]
        return dict(frames=np.stack(frames), audio=clip_audio, latents=torch.cat(latents).numpy(), windows=records)


# --------------------------------------------------------------------------
# Outputs.
# --------------------------------------------------------------------------
def write_video(path: Path, frames: np.ndarray, audio: np.ndarray | None, crf=10):
    """H.264 (yuv444p is avoided for player compatibility) with optional AAC audio."""
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames.shape[1:3]
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
           '-r', str(FPS), '-i', '-']
    wav = None
    if audio is not None:
        wav = path.with_suffix('.tmp.wav')
        sf.write(str(wav), audio, SAMPLE_RATE)
        cmd += ['-i', str(wav), '-c:a', 'aac', '-b:a', '128k', '-shortest']
    cmd += ['-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(path)]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)
    if wav is not None:
        wav.unlink()


def environment(threads):
    cpu = 'unknown'
    try:
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                cpu = line.split(':', 1)[1].strip()
                break
    except OSError:
        cpu = platform.processor() or 'unknown'
    try:
        gpu = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True).stdout.strip() or 'none detected'
    except FileNotFoundError:
        gpu = 'none (nvidia-smi absent; no CUDA, Metal or Core ML device used)'
    memory = None
    try:
        memory = int(Path('/proc/meminfo').read_text().split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return {'date_utc': datetime.now(timezone.utc).isoformat(), 'execution': 'CPU-only PyTorch inference',
            'gpu_model': gpu, 'physical_or_visible_vram': 'not applicable (no GPU used)',
            'hardware_evidence': '/proc/cpuinfo, /proc/meminfo, nvidia-smi availability',
            'cpu': cpu, 'logical_cpus': os.cpu_count(), 'torch_threads': threads, 'system_memory_bytes': memory,
            'os': platform.platform(), 'torch': torch.__version__,
            'co_resident_load': 'unisolated container; other CPU jobs may share cores'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--avatar', choices=['girl', 'indian'], required=True)
    parser.add_argument('--audio', type=Path, required=True)
    parser.add_argument('--mode', choices=MODES, default='coreml-int8')
    parser.add_argument('--variant', default='baseline', help='Transformer variant from flashhead_coreml.variants')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunks', type=int)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--force-motion', type=Path, help="Teacher-force motion latents from a previous run's directory")
    args = parser.parse_args()
    from .variants import build_transforms
    transforms, description = build_transforms(args.variant)
    runtime = CpuRuntime(args.mode, threads=args.threads, transforms=transforms)
    started = time.perf_counter()
    forced = np.load(args.force_motion / 'latents.npy') if args.force_motion else None
    result = runtime.render(args.avatar, args.audio, args.seed, args.chunks,
                            on_window=lambda r: print(f'window {r.index} frames={r.visible} ' +
                                                      ' '.join(f'{k}={v:.2f}s' for k, v in r.seconds.items()), flush=True),
                            forced_motion=forced)
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / 'frames.npy', result['frames'])
    np.save(args.output / 'latents.npy', result['latents'])
    write_video(args.output / 'video.mp4', result['frames'], result['audio'])
    frames_hash = hashlib.sha256(result['frames'].tobytes()).hexdigest()
    report = {'environment': environment(args.threads), 'avatar': args.avatar, 'audio': str(args.audio.resolve()),
              'audio_sha256': hashlib.sha256(args.audio.read_bytes()).hexdigest(), 'mode': args.mode,
              'variant': args.variant, 'variant_description': description, 'transforms': runtime.transform_log,
              'seed': args.seed, 'profile': {'height': runtime.profile.height, 'width': runtime.profile.width},
              'forced_motion_from': str(args.force_motion) if args.force_motion else None,
              'quantized_layers': runtime.quantized_layers, 'frames': int(len(result['frames'])),
              'frames_sha256': frames_hash, 'elapsed_seconds': time.perf_counter() - started,
              'windows': [{'index': r.index, 'visible_frames': r.visible, 'seconds': r.seconds} for r in result['windows']],
              'scope': ('CPU emulation for quality comparison; Apple GPU/ANE speed is not measured. '
                        'coreml-int8: INT8 weights and FP16 DiT tensors emulated; tiny VAE computed in FP32 '
                        'with FP16 I/O (the bundle runs it in FP16).')}
    (args.output / 'run.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('frames', 'frames_sha256', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
