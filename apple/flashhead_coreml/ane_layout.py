"""Neural Engine layout for PRO block partitions (opt-in, exact in FP32).

Apple's ANE prefers channels-first ``(B, C, 1, S)`` tensors, 1x1 convolutions
instead of linear layers, reductions over the channel axis, and per-head
"split einsum" attention (``apple/ml-stable-diffusion``; the same recipe made
the MuseTalk UNet run on the M5 Pro and A14 Neural Engines in this project's
sibling repository). This module rebuilds a loaded ``dit.Blocks`` partition in
that layout without changing its Core ML interface:

* input ``hidden`` [1, N, C] and ``context`` [1, 9, 32, C] are transposed once
  at the partition boundary, and ``hidden_out`` is returned as [1, N, C];
* linear layers become 1x1 convolutions with the same weights;
* LayerNorm/RMSNorm reduce over channels and may divide their input by a
  constant first (``prescale``) with eps scaled by 1/prescale^2. That is
  exact in real arithmetic and keeps FP16 squares of residual values that
  reach |x| ~ 1e3 from overflowing (65504);
* RoPE pairs are de-interleaved by permuting the q/k projection rows (and
  their RMSNorm weights) offline. Queries and keys share the permutation, so
  every attention score is unchanged; values are not permuted;
* cross-attention scores every query against all 9 x 32 audio tokens with a
  constant block-diagonal mask (-1e4 off-frame), which equals the original
  per-frame attention after the softmax and avoids per-frame reshapes. It
  adds 2 * N * 288 * C MACs per block-step over the per-frame form (~+1.6%).

The fold variant is not combined with this layout. FFN reductions are
(``ffn_step_bias`` is honoured). Placement on the Neural Engine and FP16
accuracy must be measured on a Mac or iPhone; tests here only prove FP32
equality with ``Blocks`` and Core ML convertibility.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .dit import Blocks, DiTBlock

MASK_VALUE = -1.0e4  # exp(-1e4 + O(10)) underflows to 0 in FP16 and FP32


def conv_from_linear(layer: nn.Linear, rows=None) -> nn.Conv2d:
    weight = layer.weight.detach()
    bias = layer.bias.detach() if layer.bias is not None else None
    if rows is not None:
        weight = weight[rows]
        bias = bias[rows] if bias is not None else None
    conv = nn.Conv2d(weight.shape[1], weight.shape[0], 1, bias=bias is not None)
    conv.weight.data.copy_(weight[:, :, None, None])
    if bias is not None:
        conv.bias.data.copy_(bias)
    return conv.requires_grad_(False)


class AneLayerNorm(nn.Module):
    """LayerNorm over the channel axis of (B, C, 1, S), optional exact prescale."""

    def __init__(self, norm: nn.LayerNorm, prescale: float):
        super().__init__()
        self.prescale = float(prescale) if prescale else 0.0
        self.eps = norm.eps / (self.prescale ** 2) if self.prescale else norm.eps
        if norm.weight is not None:
            self.weight = nn.Parameter(norm.weight.detach().reshape(1, -1, 1, 1).clone(), requires_grad=False)
            self.bias = nn.Parameter(norm.bias.detach().reshape(1, -1, 1, 1).clone(), requires_grad=False)
        else:
            self.weight = self.bias = None

    def forward(self, x):
        if self.prescale:
            x = x * (1.0 / self.prescale)
        mean = x.mean(dim=1, keepdim=True)
        centred = x - mean
        y = centred * torch.rsqrt((centred * centred).mean(dim=1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight + self.bias
        return y


class AneRMSNorm(nn.Module):
    def __init__(self, norm, prescale: float, rows=None):
        super().__init__()
        self.prescale = float(prescale) if prescale else 0.0
        self.eps = norm.eps / (self.prescale ** 2) if self.prescale else norm.eps
        weight = norm.weight.detach()
        if rows is not None:
            weight = weight[rows]
        self.weight = nn.Parameter(weight.reshape(1, -1, 1, 1).clone(), requires_grad=False)

    def forward(self, x):
        if self.prescale:
            x = x * (1.0 / self.prescale)
        return x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + self.eps) * self.weight


def deinterleave_rows(dim: int, heads: int) -> torch.Tensor:
    """Per head: even channels (pair 'a') first, then odd channels (pair 'b')."""
    head_dim = dim // heads
    order = []
    for h in range(heads):
        base = h * head_dim
        order += [base + 2 * j for j in range(head_dim // 2)]
        order += [base + 2 * j + 1 for j in range(head_dim // 2)]
    return torch.tensor(order)


def split_heads_channels(x, heads):
    """(B, C, 1, S) -> list of (B, C/heads, 1, S)."""
    return list(x.chunk(heads, dim=1))


class AneBlock(nn.Module):
    def __init__(self, block: DiTBlock, prescale: float, frames: int, tokens_per_frame: int, keys_per_frame: int):
        super().__init__()
        sa, ca = block.self_attn, block.cross_attn
        if getattr(ca, 'o_heads', None) is not None:
            raise ValueError('The ANE layout does not combine with the folded cross-attention')
        if sa.qkv is not None or ca.kv is not None:
            raise ValueError('Build the ANE layout from unfused projections')
        dim = sa.q.in_features
        self.heads, self.dim = sa.heads, dim
        rows = deinterleave_rows(dim, self.heads)
        self.norm1 = AneLayerNorm(block.norm1, prescale)
        self.norm2 = AneLayerNorm(block.norm2, prescale)
        self.norm3 = AneLayerNorm(block.norm3, prescale)
        self.q, self.k = conv_from_linear(sa.q, rows), conv_from_linear(sa.k, rows)
        self.v, self.o = conv_from_linear(sa.v), conv_from_linear(sa.o)
        self.norm_q, self.norm_k = AneRMSNorm(sa.norm_q, prescale, rows), AneRMSNorm(sa.norm_k, prescale, rows)
        self.cq, self.ck = conv_from_linear(ca.q), conv_from_linear(ca.k)
        self.cv, self.co = conv_from_linear(ca.v), conv_from_linear(ca.o)
        self.cnorm_q, self.cnorm_k = AneRMSNorm(ca.norm_q, prescale), AneRMSNorm(ca.norm_k, prescale)
        self.ffn1, self.ffn2 = conv_from_linear(block.ffn[0]), conv_from_linear(block.ffn[2])
        self.register_buffer('modulation', block.modulation.detach().reshape(1, 6, dim, 1, 1).clone())
        bias = getattr(block, 'ffn_step_bias', None)
        self.register_buffer('ffn_step_bias', None if bias is None else bias.detach().reshape(2, 1, dim, 1, 1).clone())
        # Block-diagonal cross-attention mask: query frame must equal key frame.
        query_frame = torch.arange(frames).repeat_interleave(tokens_per_frame)
        key_frame = torch.arange(frames).repeat_interleave(keys_per_frame)
        mask = torch.where(key_frame[:, None] == query_frame[None, :], 0.0, MASK_VALUE)
        self.register_buffer('cross_mask', mask.reshape(1, frames * keys_per_frame, 1, frames * tokens_per_frame))

    def rotary(self, x, cos, sin):
        out = []
        for head in split_heads_channels(x, self.heads):
            a, b = head.chunk(2, dim=1)
            out.append(torch.cat((a * cos - b * sin, a * sin + b * cos), dim=1))
        return out

    def attention(self, q_heads, k, v, bias=None):
        scale = float(q_heads[0].shape[1]) ** -0.5
        k_t = k.transpose(1, 3)  # (B, S_k, 1, C)
        k_heads = list(k_t.chunk(self.heads, dim=3))
        v_heads = split_heads_channels(v, self.heads)
        out = []
        for qh, kh, vh in zip(q_heads, k_heads, v_heads):
            weights = torch.einsum('bchq,bkhc->bkhq', qh, kh) * scale  # (B, S_k, 1, S_q)
            if bias is not None:
                weights = weights + bias
            weights = weights.softmax(dim=1)
            out.append(torch.einsum('bkhq,bchk->bchq', weights, vh))
        return torch.cat(out, dim=1)

    def forward(self, x, ctx, modulation, cos, sin, high):
        e = (self.modulation + modulation).unbind(dim=1)  # six (1, C, 1, 1)
        a_in = self.norm1(x) * (1 + e[1]) + e[0]
        q = self.rotary(self.norm_q(self.q(a_in)), cos, sin)
        k = torch.cat(self.rotary(self.norm_k(self.k(a_in)), cos, sin), dim=1)
        x = x + self.o(self.attention(q, k, self.v(a_in))) * e[2]
        n3 = self.norm3(x)
        cq = split_heads_channels(self.cnorm_q(self.cq(n3)), self.heads)
        x = x + self.co(self.attention(cq, self.cnorm_k(self.ck(ctx)), self.cv(ctx), self.cross_mask))
        y = self.ffn2(F.gelu(self.ffn1(self.norm2(x) * (1 + e[4]) + e[3]), approximate='tanh'))
        if self.ffn_step_bias is not None:
            y = y + torch.where(high, self.ffn_step_bias[0], self.ffn_step_bias[1])
        return x + y * e[5]


class AneBlocks(nn.Module):
    """Drop-in replacement for a ``Blocks`` partition (same inputs and output)."""

    def __init__(self, blocks: Blocks, prescale: float = 64.0):
        super().__init__()
        self.start, self.end = blocks.start, blocks.end
        grid_t = 9
        tokens = int(blocks.rotary_cos.shape[1])
        tokens_per_frame = tokens // grid_t
        self.blocks = nn.ModuleList([AneBlock(b, prescale, grid_t, tokens_per_frame, 32) for b in blocks.blocks])
        half = int(blocks.rotary_cos.shape[-1])
        # Tables (1, half, 1, N): each pair j at every token, shared by all heads.
        self.register_buffer('rotary_cos', blocks.rotary_cos.reshape(1, tokens, half).permute(0, 2, 1)[:, :, None, :].clone())
        self.register_buffer('rotary_sin', blocks.rotary_sin.reshape(1, tokens, half).permute(0, 2, 1)[:, :, None, :].clone())
        dim = blocks.time_high.shape[-1]
        self.register_buffer('time_high', blocks.time_high.detach().reshape(1, 6, dim, 1, 1).clone())
        self.register_buffer('time_low', blocks.time_low.detach().reshape(1, 6, dim, 1, 1).clone())
        self.prescale = prescale

    def forward(self, hidden, context, timestep):
        high = timestep.reshape(1, 1, 1, 1) > 800
        modulation = torch.where(high.reshape(1, 1, 1, 1, 1), self.time_high, self.time_low)
        b, n, dim = (int(v) for v in hidden.shape)
        x = hidden.transpose(1, 2).reshape(b, dim, 1, n)
        ctx = context.reshape(b, -1, dim).transpose(1, 2).reshape(b, dim, 1, -1)
        for block in self.blocks:
            x = block(x, ctx, modulation, self.rotary_cos, self.rotary_sin, high)
        return x.reshape(b, dim, n).transpose(1, 2)


def to_ane_layout(blocks: Blocks, prescale: float = 64.0) -> AneBlocks:
    return AneBlocks(blocks, prescale).eval().requires_grad_(False)
