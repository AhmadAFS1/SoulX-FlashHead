"""Explicit, per-module attention backend selection for PRO experiments.

The production model keeps its normal global attention helper. V2 explicitly
selects self-attention and pins cross-attention to FlashAttention2 independently
of optional package availability.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
import importlib.metadata
from typing import Any, Callable

import torch
from torch import nn
from einops import rearrange


class AttentionBackendError(RuntimeError):
    """The requested explicit attention backend cannot satisfy its contract."""


@dataclass(frozen=True)
class AttentionInvocation:
    backend: str
    shape: tuple[int, ...]
    dtype: str
    output_dtype: str
    fallback: bool


def _dtype_name(value: torch.dtype) -> str:
    return str(value).removeprefix("torch.")


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int) -> tuple[int, int, int]:
    if q.ndim != 3 or k.ndim != 3:
        raise AttentionBackendError(f"Q and K must be [batch, tokens, width], got {q.shape} and {k.shape}")
    if q.shape != k.shape:
        raise AttentionBackendError(f"Q/K shapes differ: {q.shape} versus {k.shape}")
    if v.ndim not in (3, 4):
        raise AttentionBackendError(f"V must be rank 3 or 4, got {v.shape}")
    batch, tokens, width = q.shape
    if not num_heads or width % num_heads:
        raise AttentionBackendError(f"Width {width} is incompatible with {num_heads} heads")
    head_dim = width // num_heads
    expected_v_shape = (batch, tokens, num_heads, head_dim)
    if tuple(v.shape) not in (expected_v_shape, (batch, tokens, width)):
        raise AttentionBackendError(f"V shape {v.shape} is incompatible with Q shape {q.shape}")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise AttentionBackendError(f"Q/K/V dtypes differ: {q.dtype}, {k.dtype}, {v.dtype}")
    if q.dtype != torch.bfloat16:
        raise AttentionBackendError(f"Experimental attention expects BF16 inputs, got {q.dtype}")
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise AttentionBackendError("Explicit GPU attention backend received a CPU tensor")
    return batch, tokens, head_dim


class ExplicitAttentionBackend:
    """Base adapter that records each invocation for run manifests."""

    name = "unknown"

    def __init__(self, *, validate_outputs: bool = False) -> None:
        self._invocations: list[AttentionInvocation] = []
        self.validate_outputs = validate_outputs

    @property
    def invocation_manifest(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._invocations]

    def manifest(self) -> dict[str, Any]:
        return {"backend": self.name, "validate_outputs": self.validate_outputs}

    def _record(self, q: torch.Tensor, output: torch.Tensor, *, fallback: bool = False) -> None:
        if torch.compiler.is_compiling():
            return  # Python log mutation must never guard or fragment the compiled graph.
        self._invocations.append(
            AttentionInvocation(
                backend=self.name,
                shape=tuple(q.shape),
                dtype=_dtype_name(q.dtype),
                output_dtype=_dtype_name(output.dtype),
                fallback=fallback,
            )
        )

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        num_heads: int,
        *,
        owner: nn.Module,
    ) -> torch.Tensor:
        raise NotImplementedError


class FlashAttention2Backend(ExplicitAttentionBackend):
    """Direct FlashAttention2 adapter with no SDPA/Flash3/Sage fallback."""

    name = "flash_attention_2"

    def __init__(
        self,
        kernel: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        *,
        validate_outputs: bool = False,
    ) -> None:
        super().__init__(validate_outputs=validate_outputs)
        self._kernel = kernel

    def _load_kernel(self) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        if self._kernel is not None:
            return self._kernel
        try:
            flash_attn = importlib.import_module("flash_attn")
            kernel = flash_attn.flash_attn_func
        except (ImportError, AttributeError) as error:
            raise AttentionBackendError("FlashAttention2 is unavailable; refusing an SDPA fallback") from error
        self._kernel = kernel
        return kernel

    def __call__(self, q, k, v, num_heads, *, owner):
        _validate_inputs(q, k, v, num_heads)
        kernel = self._load_kernel()
        # Preserve the accepted runner's exact attention layout operations.
        q_heads = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k_heads = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v_heads = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        output = kernel(q_heads, k_heads, v_heads)
        output = rearrange(output, "b s n d -> b s (n d)", n=num_heads)
        if output.shape != q.shape or output.dtype != q.dtype:
            raise AttentionBackendError("FlashAttention2 output contract mismatch")
        if self.validate_outputs and not output.isfinite().all():
            raise AttentionBackendError("FlashAttention2 returned nonfinite output")
        self._record(q, output)
        return output


class SageAttention2Backend(ExplicitAttentionBackend):
    """Supported Ada INT8-QK/FP8-PV SageAttention2++ adapter.

    The import is delayed so the base environment never acquires experimental
    dependencies. The selected release must expose the named public kernel;
    accepting a generic ``sageattn`` symbol would silently select Sage1.
    """

    name = "sageattention2_int8_qk_fp8_pv"

    def __init__(
        self,
        kernel: Callable[..., torch.Tensor] | None = None,
        version: str | None = None,
        *,
        kernel_symbol: str = "sageattn_qk_int8_pv_fp8_cuda",
        pv_accum_dtype: str | None = "fp32+fp16",
        validate_outputs: bool = False,
    ) -> None:
        super().__init__(validate_outputs=validate_outputs)
        self._kernel = kernel
        self.version = version
        self.kernel_symbol = kernel_symbol
        self.pv_accum_dtype = pv_accum_dtype

    def manifest(self) -> dict[str, Any]:
        return {
            **super().manifest(),
            "version": self.version,
            "kernel_symbol": self.kernel_symbol,
            "pv_accum_dtype": self.pv_accum_dtype,
            "fallback": False,
        }

    def _load_kernel(self) -> Callable[..., torch.Tensor]:
        if self._kernel is not None:
            return self._kernel
        try:
            package = importlib.import_module("sageattention")
            version = str(getattr(package, "__version__", ""))
            if not version:
                version = importlib.metadata.version("sageattention")
            kernel = getattr(package, self.kernel_symbol)
        except (ImportError, AttributeError) as error:
            raise AttentionBackendError(
                f"SageAttention2 kernel {self.kernel_symbol} is unavailable; refusing a fallback"
            ) from error
        if not version.startswith("2"):
            raise AttentionBackendError(f"SageAttention2 is required, found version {version or 'unknown'}")
        self.version = version
        self._kernel = kernel
        return kernel

    def __call__(self, q, k, v, num_heads, *, owner):
        batch, tokens, head_dim = _validate_inputs(q, k, v, num_heads)
        if head_dim != 128:
            raise AttentionBackendError(f"SageAttention2 v2 policy supports head dimension 128, got {head_dim}")
        kernel = self._load_kernel()
        q_heads = q.reshape(batch, tokens, num_heads, head_dim)
        k_heads = k.reshape(batch, tokens, num_heads, head_dim)
        v_heads = v.reshape(batch, tokens, num_heads, head_dim)
        try:
            kwargs = {"tensor_layout": "NHD", "is_causal": False}
            if self.pv_accum_dtype is not None:
                kwargs["pv_accum_dtype"] = self.pv_accum_dtype
            output = kernel(q_heads, k_heads, v_heads, **kwargs)
        except TypeError as error:
            raise AttentionBackendError(
                "The installed SageAttention2 API does not support the required NHD noncausal call"
            ) from error
        if tuple(output.shape) != (batch, tokens, num_heads, head_dim):
            raise AttentionBackendError(f"SageAttention2 returned unexpected shape {output.shape}")
        if output.dtype != q.dtype:
            raise AttentionBackendError("SageAttention2 returned incorrectly typed output")
        if self.validate_outputs and not output.isfinite().all():
            raise AttentionBackendError("SageAttention2 returned nonfinite output")
        output = output.reshape_as(q)
        self._record(q, output)
        return output


class SageAttention2FP16PVBackend(SageAttention2Backend):
    """INT8 Q/K with FP16 P/V and mixed FP16/FP32 accumulation."""

    name = "sageattention2_int8_qk_fp16_pv"

    def __init__(self, kernel=None, version=None, *, validate_outputs=False):
        super().__init__(
            kernel, version,
            kernel_symbol="sageattn_qk_int8_pv_fp16_cuda",
            pv_accum_dtype="fp16+fp32",
            validate_outputs=validate_outputs,
        )


def make_attention_backend(name: str) -> ExplicitAttentionBackend:
    """Build an adapter without importing optional packages until first use."""
    if name == "flash2":
        return FlashAttention2Backend()
    if name == "sage2":
        return SageAttention2Backend()
    if name == "sage2_fp16":
        return SageAttention2FP16PVBackend()
    raise AttentionBackendError(f"Unsupported explicit attention backend: {name}")


def install_self_attention_backend(
    model: nn.Module,
    backend: ExplicitAttentionBackend,
    *,
    self_attention_type: type[nn.Module] | None = None,
) -> list[str]:
    """Attach ``backend`` only to concrete self-attention modules in ``model``."""
    if self_attention_type is None:
        from flash_head.src.modules.flash_head_model import SelfAttention

        self_attention_type = SelfAttention
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise AttentionBackendError("Model has no transformer blocks")
    installed: list[str] = []
    for index, block in enumerate(blocks):
        attention = getattr(block, "self_attn", None)
        if not isinstance(attention, self_attention_type):
            raise AttentionBackendError(f"blocks.{index}.self_attn is not the expected SelfAttention type")
        if getattr(attention, "use_usp", False):
            raise AttentionBackendError("Explicit v2 attention backends do not support sequence parallelism")
        if getattr(attention, "head_dim", None) != 128:
            raise AttentionBackendError(
                f"blocks.{index}.self_attn has unsupported head dimension {getattr(attention, 'head_dim', None)}"
            )
        if getattr(attention, "_pro_attention_backend", None) is not None:
            raise AttentionBackendError(f"blocks.{index}.self_attn already has an experimental backend")
        # The accepted Flash2 baseline must retain the original model call
        # boundary: wrapping it changes compiled numerical trajectories.
        attention._pro_attention_backend = None if isinstance(backend, FlashAttention2Backend) else backend
        installed.append(f"blocks.{index}.self_attn")
    if isinstance(backend, FlashAttention2Backend):
        pin_reference_flash2_dispatch()
    return installed


def remove_self_attention_backend(model: nn.Module) -> None:
    """Remove v2 adapters so the model returns to its ordinary attention path."""
    for block in getattr(model, "blocks", ()):
        attention = getattr(block, "self_attn", None)
        if attention is not None:
            attention._pro_attention_backend = None

def cross_attention_flash2(q, k, v, num_heads):
    kernel = importlib.import_module("flash_attn").flash_attn_func
    q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
    k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
    v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
    return rearrange(kernel(q, k, v), "b s n d -> b s (n d)", n=num_heads)


def pin_reference_flash2_dispatch():
    """Pin the original dispatcher in an isolated experiment process.

    Production never invokes this function. Preserving the original call path
    is required for bitwise reproduction of the accepted compiled baseline.
    """
    from flash_head.src.modules import flash_head_model as implementation
    package = importlib.import_module("flash_attn")
    if not callable(getattr(package, "flash_attn_func", None)):
        raise AttentionBackendError("FlashAttention2 is required; no fallback is allowed")
    implementation.flash_attn = package
    implementation.FLASH_ATTN_2_AVAILABLE = True
    implementation.FLASH_ATTN_3_AVAILABLE = False
    implementation.SAGE_ATTN_AVAILABLE = False


def pin_cross_attention_flash2(model):
    from flash_head.src.modules.flash_head_model import CrossAttention
    pin_reference_flash2_dispatch()
    paths = []
    for name, module in model.named_modules():
        if isinstance(module, CrossAttention):
            module._pro_cross_attention = None
            paths.append(name)
    return {"backend": "flash_attention_2", "installed": paths,
            "scope": "isolated offline process; original dispatcher pinned, no fallback"}
