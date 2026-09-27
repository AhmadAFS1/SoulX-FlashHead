"""Fused INT8 GEMM for the PRO FFN's first linear (throughput plan lever C2).

One Triton kernel computes ``ffn.0``'s W8A8 GEMM and, in its epilogue, dequantises,
adds the bias, applies the FFN's GELU(tanh) and requantises to INT8 with ``ffn.2``'s
static activation scale. It writes ffn.2's INT8 input ``[M, 8960]`` directly (58 MB at
M=6480) instead of the shipped 232 MB INT32 ``torch._int_mm`` output that inductor's
``triton_poi_fused__to_copy_..._round`` kernel then reads back (~0.66 ms/call).

Numerics track the *compiled* shipped path. Inductor's fused pointwise kernel for
``Int8ComputeLinear(ffn.0) -> nn.GELU(approximate='tanh') -> Int8ComputeLinear(ffn.2)``
(with a static scale on ffn.2) computes, from the int32 accumulator, in fp32::

    y   = (acc.f32 * sa[m]) * sw[n] + bias[n].f32
    g   = (y * 0.5) * (libdevice.tanh(((y*y)*y * 0.044715 + y) * 0.7978845608028654) + 1.0)
    q   = min(max(libdevice.nearbyint(g / s2), -127.0), 127.0).to(int8)

The intermediate ``.to(bfloat16)`` round-trips of the eager path are *elided* by inductor
(its compute type for bf16 is fp32; ``emulate_precision_casts`` defaults to False), so the
epilogue below is that expression, in that order, with the same libdevice calls and the
same Triton ``/``. Differences against the compiled path can only come from FMA
contraction choices; differences against the eager path add the missing bf16 roundings.
Both are at most +/-1 LSB of the int8 output.

Weight layout: the kernel consumes ``Int8ComputeLinear.weight_int8`` as stored,
``[N, K]`` row-major (K-contiguous), and reads it as the ``[K, N]`` B operand through
strides. That is the tensor-core-native "TN" layout for INT8 ``mma.sync.m16n8k32.row.col``
-- both operands K-major -- so no transposed shared-memory path is needed (``ldmatrix.trans``
exists only for 16-bit types, so an N-contiguous INT8 B tile costs extra shuffles), and it is
the same layout the shipped cuBLAS ``cutlass_*_i16832gemm_s8_*_tn`` tactic selects. No
weight copy is prepared at install.
"""
from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# Configurations kept in the shipped autotune set: the four within ~10% of the best in the
# 22-config SM89 sweep (scratchpad c2/bench.py, uncontended) at (6480,1536,8960) and
# (4032,1536,8960). 128x128x64 / 3 stages / 4 warps won both shapes (0.89 ms and 0.66 ms
# raw); every 8-warp and every 256-wide-tile config was 20-40% slower. Kept short because
# autotuning runs once per distinct (M, N, K) at warm-up (~0.5 s per shape, plus a first-
# process Triton compile that is then cached on disk).
DEFAULT_CONFIGS = [
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=4, num_warps=4),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8}, num_stages=2, num_warps=4),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8}, num_stages=3, num_warps=4),
]


@triton.jit
def _int8_ffn0_gelu_requant_kernel(
    a_ptr, sa_ptr, w_ptr, sw_ptr, bias_ptr, s2_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    stride_om, stride_on,
    stride_sa,
    HAS_BIAS: tl.constexpr,
    EVEN_N: tl.constexpr, EVEN_K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Grouped launch order (tutorial-style) so a GROUP_M-tall band of A tiles stays hot in L2.
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mask_m = offs_m < M
    mask_n = offs_n < N

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    # B operand [BLOCK_K, BLOCK_N] read straight out of the stored [N, K] weight.
    w_ptrs = w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs, mask=mask_m[:, None], other=0)
            if EVEN_N:
                w = tl.load(w_ptrs)
            else:
                w = tl.load(w_ptrs, mask=mask_n[None, :], other=0)
        else:
            k_rem = K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=mask_m[:, None] & (offs_k[None, :] < k_rem), other=0)
            w = tl.load(w_ptrs, mask=(offs_k[:, None] < k_rem) & mask_n[None, :], other=0)
        acc = tl.dot(a, w, acc, out_dtype=tl.int32)
        a_ptrs += BLOCK_K * stride_ak
        w_ptrs += BLOCK_K * stride_wk

    # ---- epilogue: exactly inductor's fused expression for the shipped compiled path ----
    sa = tl.load(sa_ptr + offs_m * stride_sa, mask=mask_m, other=0.0).to(tl.float32)
    sw = tl.load(sw_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
    y = acc.to(tl.float32)
    y = y * sa[:, None]
    y = y * sw[None, :]
    if HAS_BIAS:
        b = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(tl.float32)
        y = y + b[None, :]
    half = y * 0.5
    cube = (y * y) * y
    inner = (y + cube * 0.044715) * 0.7978845608028654
    g = half * (libdevice.tanh(inner) + 1.0)
    s2 = tl.load(s2_ptr).to(tl.float32)
    q = libdevice.nearbyint(g / s2)
    q = tl.minimum(tl.maximum(q, -127.0), 127.0)
    out = q.to(tl.int8)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out, mask=mask_m[:, None] & mask_n[None, :])


def build_kernel(configs=None):
    """Wrap the bare kernel in autotune + shape heuristics. Exposed so benches can sweep."""
    kernel = triton.heuristics({
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
    })(_int8_ffn0_gelu_requant_kernel)
    return triton.autotune(configs=list(configs or DEFAULT_CONFIGS), key=["M", "N", "K"])(kernel)


_default_kernel = None


def _kernel():
    global _default_kernel
    if _default_kernel is None:
        _default_kernel = build_kernel(DEFAULT_CONFIGS)
    return _default_kernel


def launch_int8_ffn0_gelu_requant(a, a_scale, w, w_scale, bias, out_scale, out, kernel=None, **fixed):
    """Validate operands and launch. ``fixed`` may pin BLOCK_*/num_warps/num_stages
    (raw-kernel path used by the sweep bench); otherwise the autotuned kernel is used."""
    if a.dtype != torch.int8 or w.dtype != torch.int8:
        raise TypeError("a and w must be int8")
    if a.dim() != 2 or w.dim() != 2 or a.shape[1] != w.shape[1]:
        raise ValueError(f"shape mismatch: a {tuple(a.shape)} w {tuple(w.shape)}; need [M,K] x [N,K]")
    if a.stride(1) != 1:
        raise ValueError("a must be K-contiguous ([M, K] row-major)")
    if not (w.stride(1) == 1 or w.stride(0) == 1):
        raise ValueError("w must be contiguous along N or K")
    if a.shape[1] % 16:
        raise ValueError("K must be a multiple of 16 for the INT8 tensor-core dot")
    M, K = a.shape
    N = w.shape[0]
    sa = a_scale.reshape(-1)
    if sa.numel() not in (1, M):
        raise ValueError(f"a_scale must have 1 or M={M} elements, got {sa.numel()}")
    if sa.dtype != torch.float32 or not sa.is_contiguous():
        sa = sa.float().contiguous()
    stride_sa = 0 if sa.numel() == 1 else 1
    sw = w_scale.reshape(-1)
    if sw.numel() != N:
        raise ValueError(f"w_scale must have N={N} elements, got {sw.numel()}")
    if sw.dtype != torch.float32 or not sw.is_contiguous():
        sw = sw.float().contiguous()
    if out_scale.numel() != 1:
        raise ValueError("out_scale must be a single element (ffn.2's static_input_scale)")
    s2 = out_scale.reshape(-1)
    if s2.dtype != torch.float32:
        s2 = s2.float()
    has_bias = bias is not None
    if has_bias:
        bias = bias.reshape(-1)
        if bias.numel() != N:
            raise ValueError(f"bias must have N={N} elements, got {bias.numel()}")
        if not bias.is_contiguous():
            bias = bias.contiguous()
    else:
        bias = sw  # placeholder pointer; never read when HAS_BIAS is False
    if out.shape != (M, N) or out.dtype != torch.int8 or out.stride(1) != 1:
        raise ValueError("out must be an int8 [M, N] row-major tensor")

    def grid(meta):
        return (triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),)

    args = (a, sa, w, sw, bias, s2, out, M, N, K,
            a.stride(0), a.stride(1), w.stride(0), w.stride(1), out.stride(0), out.stride(1), stride_sa)
    if fixed:
        # Raw launch with pinned meta-parameters (bench/sweep only).
        meta = dict(fixed)
        num_warps = meta.pop("num_warps", 4)
        num_stages = meta.pop("num_stages", 3)
        meta.setdefault("GROUP_M", 8)
        meta["EVEN_N"] = N % meta["BLOCK_N"] == 0
        meta["EVEN_K"] = K % meta["BLOCK_K"] == 0
        _int8_ffn0_gelu_requant_kernel[grid(meta)](*args, HAS_BIAS=has_bias, num_warps=num_warps,
                                                   num_stages=num_stages, **meta)
    else:
        (kernel or _kernel())[grid](*args, HAS_BIAS=has_bias)
    return out


@torch.library.custom_op("soulx_pro::int8_ffn0_gelu_requant", mutates_args=(), device_types="cuda")
def int8_ffn0_gelu_requant(
    a: torch.Tensor,
    a_scale: torch.Tensor,
    w: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor],
    out_scale: torch.Tensor,
) -> torch.Tensor:
    """int8 [M,K] x int8 [N,K]^T -> dequant (a_scale[m] or scalar, w_scale[n]) + bias ->
    GELU(tanh) -> round(./out_scale) clamped to [-127,127] -> int8 [M,N].

    Registered as a custom op with a fake kernel so torch.compile keeps it in-graph
    (fullgraph=True works); autotuning happens on the first real call per (M, N, K).
    """
    out = torch.empty((a.shape[0], w.shape[0]), dtype=torch.int8, device=a.device)
    return launch_int8_ffn0_gelu_requant(a, a_scale, w, w_scale, bias, out_scale, out)


@int8_ffn0_gelu_requant.register_fake
def _int8_ffn0_gelu_requant_fake(a, a_scale, w, w_scale, bias, out_scale):
    return a.new_empty((a.shape[0], w.shape[0]), dtype=torch.int8)


def reference_int8_ffn0_gelu_requant(a, a_scale, w, w_scale, bias, out_scale, *, emulate_bf16=False):
    """Pure-torch reference of the shipped path for tests.

    ``emulate_bf16=False`` reproduces the *compiled* shipped path (fp32 throughout, as
    inductor emits it). ``emulate_bf16=True`` reproduces the *eager* shipped path, which
    rounds ffn.0's output to bf16 before GELU and GELU's output to bf16 before requant.
    Rows are processed in chunks to bound the int32 intermediate on a 12 GB part.
    """
    M, K = a.shape
    N = w.shape[0]
    sa = a_scale.reshape(-1).float()
    out = torch.empty((M, N), dtype=torch.int8, device=a.device)
    s2 = out_scale.reshape(()).float()
    chunk = 1024
    for r0 in range(0, M, chunk):
        r1 = min(M, r0 + chunk)
        acc = torch._int_mm(a[r0:r1], w.t()).float()
        sa_c = sa if sa.numel() == 1 else sa[r0:r1, None]
        y = acc * sa_c * w_scale.float()[None, :]
        if bias is not None:
            y = y + bias.float()
        if emulate_bf16:
            y = y.to(torch.bfloat16)
        g = torch.nn.functional.gelu(y, approximate="tanh")
        if emulate_bf16:
            g = g.to(torch.bfloat16)
        out[r0:r1] = (g.float() / s2).round().clamp(-127, 127).to(torch.int8)
    return out
