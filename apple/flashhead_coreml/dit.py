"""Portable, checkpoint-compatible SoulX FlashHead PRO transformer components.

Only standard PyTorch operations are used. CUDA/xfuser and the serving pipeline
are deliberately not imported. This is a floating point export reference; Apple
weight compression is applied after conversion, rather than exporting CUDA FP8
or INT8 GEMM kernels. Shapes are fixed per exported profile, batch size one.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ProConfig:
    dim: int = 1536
    ffn_dim: int = 8960
    num_heads: int = 12
    num_layers: int = 30
    freq_dim: int = 256
    eps: float = 1e-6
    in_dim: int = 32
    out_dim: int = 16

    @classmethod
    def from_json(cls, path):
        raw = json.loads(Path(path).read_text())
        if tuple(raw['patch_size']) != (1, 2, 2) or tuple(raw['vae_stride']) != (4, 8, 8):
            raise ValueError('Only the PRO patch/stride contract is supported')
        if raw.get('has_image_input', False):
            raise ValueError('Image cross-attention checkpoints are not supported')
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__})


@dataclass(frozen=True)
class Profile:
    height: int = 576
    width: int = 320
    frames: int = 33

    def __post_init__(self):
        if self.height < 16 or self.width < 16 or self.height % 16 or self.width % 16:
            raise ValueError('Height and width must be positive multiples of 16')
        if self.frames != 33:
            raise ValueError('The streaming contract requires 33-frame windows')

    @property
    def latent_shape(self):
        return (1, 16, 9, self.height // 8, self.width // 8)

    @property
    def grid(self):
        return (9, self.height // 16, self.width // 16)

    @property
    def tokens(self):
        return math.prod(self.grid)


def timestep_embedding(timestep, dim):
    # Upstream computes the sinusoid in float64 before casting. This is used
    # offline to form constants, so conversion never needs a float64 operator.
    freq = torch.pow(10000.0, -torch.arange(dim // 2, dtype=torch.float64) / (dim // 2))
    phase = timestep.double().reshape(-1, 1) * freq.reshape(1, -1)
    return torch.cat((phase.cos(), phase.sin()), dim=-1).float()


def rotary_tables(head_dim, grid):
    """Exactly upstream's temporal/height/width complex frequencies, as real tables."""
    t, h, w = grid
    half = head_dim // 2
    sizes = (half - 2 * (half // 3), half // 3, half // 3)
    parts = []
    for length, size in zip(grid, sizes):
        freq = 1.0 / (10000.0 ** (torch.arange(size, dtype=torch.float64) / size))
        phase = torch.arange(length, dtype=torch.float64)[:, None] * freq[None, :]
        parts.append(phase)
    phase = torch.cat((
        parts[0][:, None, None, :].expand(t, h, w, -1),
        parts[1][None, :, None, :].expand(t, h, w, -1),
        parts[2][None, None, :, :].expand(t, h, w, -1),
    ), dim=-1).reshape(1, t * h * w, 1, half)
    return phase.cos().float(), phase.sin().float()


def apply_rotary(x, cos, sin):
    pairs = x.float().reshape(*(int(v) for v in x.shape[:-1]), -1, 2)
    a, b = pairs[..., 0], pairs[..., 1]
    return torch.stack((a * cos - b * sin, a * sin + b * cos), dim=-1).flatten(-2).to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        y = x.float()
        return (y * torch.rsqrt((y * y).mean(dim=-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


def attention(q, k, v, heads, query_chunk=0):
    b, n, d = (int(v) for v in q.shape)
    q = q.reshape(b, n, heads, d // heads).transpose(1, 2)
    k = k.reshape(b, -1, heads, d // heads).transpose(1, 2)
    v = v.reshape(b, -1, heads, d // heads).transpose(1, 2)
    if query_chunk:
        # Each query slice attends to ALL keys. This is exact attention, not a
        # local-attention approximation or a separately normalized key tile.
        parts = [F.scaled_dot_product_attention(q[:, :, i:i + query_chunk], k, v)
                 for i in range(0, n, query_chunk)]
        out = torch.cat(parts, dim=2)
    else:
        out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(b, n, d)


def combine_projections(*layers):
    """One dense transform of a shared input, preserving checkpoint output rows."""
    first = layers[0]
    combined = nn.Linear(first.in_features, sum(layer.out_features for layer in layers),
                         bias=first.bias is not None, device='meta')
    combined.weight = nn.Parameter(torch.cat([layer.weight for layer in layers]), requires_grad=False)
    if first.bias is not None:
        combined.bias = nn.Parameter(torch.cat([layer.bias for layer in layers]), requires_grad=False)
    return combined


class SelfAttention(nn.Module):
    def __init__(self, cfg, query_chunk=0):
        super().__init__()
        self.heads, self.query_chunk = cfg.num_heads, query_chunk
        self.q = nn.Linear(cfg.dim, cfg.dim)
        self.k = nn.Linear(cfg.dim, cfg.dim)
        self.v = nn.Linear(cfg.dim, cfg.dim)
        self.o = nn.Linear(cfg.dim, cfg.dim)
        self.norm_q, self.norm_k = RMSNorm(cfg.dim, cfg.eps), RMSNorm(cfg.dim, cfg.eps)
        self.qkv = None

    def fuse_projections(self):
        if self.qkv is None:
            self.qkv = combine_projections(self.q, self.k, self.v)
            del self.q, self.k, self.v

    def forward(self, x, cos, sin):
        b, n, dim = (int(v) for v in x.shape)
        if self.qkv is None:
            q, k = self.norm_q(self.q(x)), self.norm_k(self.k(x))
            v = None
        else:
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            q, k = self.norm_q(q), self.norm_k(k)
        q = q.reshape(b, n, self.heads, dim // self.heads)
        k = k.reshape(b, n, self.heads, dim // self.heads)
        q = apply_rotary(q, cos, sin).flatten(2)
        k = apply_rotary(k, cos, sin).flatten(2)
        if v is None:
            v = self.v(x)
        return self.o(attention(q, k, v, self.heads, self.query_chunk))


class CrossAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.heads = cfg.num_heads
        self.q = nn.Linear(cfg.dim, cfg.dim)
        self.k = nn.Linear(cfg.dim, cfg.dim)
        self.v = nn.Linear(cfg.dim, cfg.dim)
        self.o = nn.Linear(cfg.dim, cfg.dim)
        self.norm_q, self.norm_k = RMSNorm(cfg.dim, cfg.eps), RMSNorm(cfg.dim, cfg.eps)
        self.kv = None

    def fuse_projections(self):
        if self.kv is None:
            self.kv = combine_projections(self.k, self.v)
            del self.k, self.v

    def forward(self, x, context):
        if self.kv is None:
            return self.o(attention(self.norm_q(self.q(x)), self.norm_k(self.k(context)), self.v(context), self.heads))
        k, v = self.kv(context).chunk(2, dim=-1)
        return self.o(attention(self.norm_q(self.q(x)), self.norm_k(k), v, self.heads))


class DiTBlock(nn.Module):
    def __init__(self, cfg, query_chunk=0):
        super().__init__()
        self.dim = cfg.dim
        self.self_attn = SelfAttention(cfg, query_chunk)
        self.cross_attn = CrossAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.dim, eps=cfg.eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(cfg.dim, eps=cfg.eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(cfg.dim, eps=cfg.eps)
        self.ffn = nn.Sequential(nn.Linear(cfg.dim, cfg.ffn_dim), nn.GELU(approximate='tanh'), nn.Linear(cfg.ffn_dim, cfg.dim))
        self.modulation = nn.Parameter(torch.zeros(1, 6, cfg.dim))

    def forward(self, x, context, modulation, cos, sin):
        e = (self.modulation + modulation).chunk(6, dim=1)
        x = x + self.self_attn(self.norm1(x) * (1 + e[1]) + e[0], cos, sin) * e[2]
        b, n, dim = (int(v) for v in x.shape)
        times = int(context.shape[1])
        q = self.norm3(x).reshape(b * times, n // times, dim)
        c = context.reshape(b * times, int(context.shape[2]), dim)
        x = x + self.cross_attn(q, c).reshape(b, n, dim)
        return x + self.ffn(self.norm2(x) * (1 + e[4]) + e[3]) * e[5]


class Patch(nn.Module):
    def __init__(self, cfg, profile):
        super().__init__()
        # Temporal kernel=1: flatten time into batch and use Conv2d. Weight
        # dimension 2 is squeezed during loading; arithmetic is unchanged.
        self.patch_embedding = nn.Conv2d(cfg.in_dim, cfg.dim, kernel_size=2, stride=2)
        self.dim = cfg.dim
        self.profile = profile

    def forward(self, latent, reference):
        x = torch.cat((latent, reference), dim=1)
        b, c, t, h, w = (int(v) for v in x.shape)
        x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x = self.patch_embedding(x)
        return x.reshape(b, t, self.dim, h // 2, w // 2).permute(0, 1, 3, 4, 2).reshape(b, -1, self.dim)


class AudioProjection(nn.Module):
    def __init__(self, dim=1536, intermediate=512, channels=768, layers=12, tokens=32):
        super().__init__()
        self.dim, self.intermediate = dim, intermediate
        self.channels, self.layers, self.tokens = channels, layers, tokens
        self.proj1 = nn.Linear(5 * layers * channels, intermediate)
        self.proj1_vf = nn.Linear(8 * layers * channels, intermediate)
        self.proj2 = nn.Linear(intermediate, intermediate)
        self.proj3 = nn.Linear(intermediate, tokens * dim)
        # AudioProjModel uses default LayerNorm eps=1e-5.
        self.norm = nn.LayerNorm(dim)

    def forward(self, audio):
        b = int(audio.shape[0])
        first = audio[:, :1].reshape(b, 1, -1)
        # Flatten layer/channel together so every intermediate stays at rank
        # five or below, as required by the Core ML ML Program backend.
        features = self.layers * self.channels
        latter = audio[:, 1:].reshape(b, 8, 4, 5, features)
        packed = torch.cat((latter[:, :, :1, :3].reshape(b, 8, 3, features),
                            latter[:, :, 1:-1, 2:3].reshape(b, 8, 2, features),
                            latter[:, :, -1:, 2:].reshape(b, 8, 3, features)), dim=2)
        first = F.relu(self.proj1(first))
        latter = F.relu(self.proj1_vf(packed.reshape(b, 8, -1)))
        x = torch.cat((first, latter), dim=1)
        x = F.relu(self.proj2(x))
        return self.norm(self.proj3(x).reshape(b, 9, self.tokens, self.dim))


class TimeEmbedding(nn.Module):
    """Loaded/exported offline; not a runtime model or duplicate partition weight."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.time_embedding = nn.Sequential(nn.Linear(cfg.freq_dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(cfg.dim, cfg.dim * 6))

    def forward(self, timestep):
        t = self.time_embedding(timestep_embedding(timestep, self.cfg.freq_dim))
        return t, self.time_projection(t).reshape(-1, 6, self.cfg.dim)


class Blocks(nn.Module):
    def __init__(self, cfg, profile, start, end, time_modulations, query_chunk=0):
        super().__init__()
        if not 0 <= start < end <= cfg.num_layers:
            raise ValueError('Invalid block partition')
        self.start, self.end = start, end
        self.blocks = nn.ModuleList([DiTBlock(cfg, query_chunk) for _ in range(start, end)])
        cos, sin = rotary_tables(cfg.dim // cfg.num_heads, profile.grid)
        self.register_buffer('rotary_cos', cos, persistent=False)
        self.register_buffer('rotary_sin', sin, persistent=False)
        self.register_buffer('time_high', time_modulations[0:1].detach(), persistent=False)
        self.register_buffer('time_low', time_modulations[1:2].detach(), persistent=False)

    def fuse_projections(self):
        """Call after strict checkpoint loading, before tracing and compression."""
        for block in self.blocks:
            block.self_attn.fuse_projections()
            block.cross_attn.fuse_projections()

    def forward(self, hidden, context, timestep):
        modulation = torch.where(timestep.reshape(1, 1, 1) > 800, self.time_high, self.time_low)
        x = hidden
        for block in self.blocks:
            x = block(x, context, modulation, self.rotary_cos, self.rotary_sin)
        return x


class OutputHead(nn.Module):
    def __init__(self, cfg, profile, time_embeddings):
        super().__init__()
        self.profile, self.dim, self.out_dim = profile, cfg.dim, cfg.out_dim
        self.norm = nn.LayerNorm(cfg.dim, eps=cfg.eps, elementwise_affine=False)
        self.head = nn.Linear(cfg.dim, cfg.out_dim * 4)
        self.modulation = nn.Parameter(torch.zeros(1, 2, cfg.dim))
        self.register_buffer('time_high', time_embeddings[0:1].detach(), persistent=False)
        self.register_buffer('time_low', time_embeddings[1:2].detach(), persistent=False)

    def forward(self, hidden, timestep):
        t = torch.where(timestep.reshape(1, 1) > 800, self.time_high, self.time_low)
        shift, scale = (self.modulation + t[:, None, :]).chunk(2, dim=1)
        x = self.head(self.norm(hidden) * (1 + scale) + shift)
        gt, gh, gw = self.profile.grid
        # Checkpoint output order is [patch_y, patch_x, channel]. Reorder to
        # channel-major before pixel shuffle, avoiding an unsupported rank-eight
        # unpatchify intermediate while preserving the exact voxel order.
        x = x.reshape(gt, gh * gw, 4, self.out_dim).permute(0, 3, 2, 1)
        x = F.pixel_shuffle(x.reshape(gt, self.out_dim * 4, gh, gw), 2)
        return x.permute(1, 0, 2, 3).unsqueeze(0)


def load_component(module, checkpoint, kind, start=0):
    """Read only this component's tensors, preserving strict missing-key checks."""
    from safetensors import safe_open
    tensors = {}
    with safe_open(str(checkpoint), framework='pt', device='cpu') as source:
        for key in module.state_dict():
            if kind == 'conditioning':
                original = 'audio_proj.' + key
            elif kind == 'head':
                original = 'head.' + key
            elif kind == 'blocks' and key.startswith('blocks.'):
                fields = key.split('.')
                fields[1] = str(start + int(fields[1]))
                original = '.'.join(fields)
            else:
                original = key
            tensor = source.get_tensor(original)
            if kind == 'patch' and key == 'patch_embedding.weight':
                tensor = tensor.squeeze(2)
            tensors[key] = tensor.float()
    module.load_state_dict(tensors, strict=True)
    return module.eval().requires_grad_(False)
