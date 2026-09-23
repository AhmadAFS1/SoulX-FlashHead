"""Strict, transactional policy conversion for PRO quantization experiments.

This module owns model policy parsing, validation, and linear replacement.
CUDA dispatch, profiling, and media qualification live in the v2 benchmark
tools so a policy can be validated on CPU before a GPU is acquired.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn

from soulx_rtc.pro_quantization import Float8Linear, Int8ComputeLinear


SCHEMA_VERSION = 1
FP8_SCHEME = "fp8_e4m3_w8a8"
INT8_SCHEME = "int8_w8a8"
BF16_SCHEME = "bf16"
SUPPORTED_LINEAR_SCHEMES = frozenset((BF16_SCHEME, FP8_SCHEME, INT8_SCHEME))
SUPPORTED_ATTENTION_BACKENDS = frozenset(("flash2", "sage2", "sage2_fp16"))
SUPPORTED_DECODER_SCHEMES = frozenset((BF16_SCHEME, "fp16_stage_control", "int8_conservative", "fp8_conservative"))
SUPPORTED_DECODER_BACKENDS = frozenset(("torch_compile", "pytorch", "adapter", "adapter_compile", "trt_stage", "trt_stage_compile"))


class PolicyValidationError(ValueError):
    """A policy cannot be safely applied to the requested model."""


class ConversionError(RuntimeError):
    """Conversion failed after complete planning; discard the affected model."""


class _PolicyQuantizationMetadata:
    """Persistent numeric metadata that survives state-dict serialization."""

    scheme_code = 0
    scale_granularity_code = 0

    def _register_policy_metadata(self, linear: nn.Linear) -> None:
        self.register_buffer(
            "pro_quantization_v2_metadata",
            torch.tensor(
                [SCHEMA_VERSION, self.scheme_code, self.scale_granularity_code],
                dtype=torch.int32,
                device=linear.weight.device,
            ),
        )


class PolicyFloat8Linear(_PolicyQuantizationMetadata, Float8Linear):
    """The tested FP8 kernel with persistent v2 policy metadata."""

    scheme_code = 1
    scale_granularity_code = 1

    def __init__(self, linear: nn.Linear, fast_accum: bool = False) -> None:
        super().__init__(linear, fast_accum=fast_accum)
        self._register_policy_metadata(linear)


class PolicyInt8ComputeLinear(_PolicyQuantizationMetadata, Int8ComputeLinear):
    """The tested INT8 kernel with persistent v2 policy metadata."""

    scheme_code = 2
    scale_granularity_code = 2

    def __init__(self, linear: nn.Linear) -> None:
        super().__init__(linear)
        self._register_policy_metadata(linear)


@dataclass(frozen=True)
class ConversionTarget:
    """One validated, unmodified linear module selected for conversion."""

    path: str
    family: str
    scheme: str
    source_dtype: str
    target_dtype: str
    in_features: int
    out_features: int
    has_bias: bool
    original_bytes: int


@dataclass(frozen=True)
class ResolvedPolicy:
    """Serializable policy resolution that contains no model references."""

    policy: dict[str, Any]
    targets: tuple[ConversionTarget, ...]
    excluded: tuple[str, ...]
    model_geometry: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "requested_policy": self.policy,
            "targets": [asdict(target) for target in self.targets],
            "excluded": list(self.excluded),
            "model_geometry": self.model_geometry,
            "policy_sha256": hashlib.sha256(
                json.dumps(self.policy, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }


@dataclass(frozen=True)
class ConversionPlan:
    """Validated conversion plan. Building it never mutates ``model``."""

    resolved: ResolvedPolicy


def _reject_unknown_keys(value: Any, allowed: Iterable[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise PolicyValidationError(f"{where} must be an object")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise PolicyValidationError(f"Unknown keys in {where}: {unknown}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise PolicyValidationError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_policy(path: str | Path) -> dict[str, Any]:
    """Load a policy while rejecting duplicate JSON keys before interpretation."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            policy = json.load(handle, object_pairs_hook=_object_without_duplicate_keys)
    except json.JSONDecodeError as error:
        raise PolicyValidationError(f"Invalid policy JSON: {error}") from error
    validate_policy(policy)
    return policy


def _require_string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise PolicyValidationError(f"{where} must be a non-empty string")
    return value


def _require_string_list(value: Any, where: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise PolicyValidationError(f"{where} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise PolicyValidationError(f"{where} contains duplicate rules")
    return value


def validate_policy(policy: Mapping[str, Any]) -> None:
    """Validate schema and supported combinations without touching a model."""
    _reject_unknown_keys(
        policy,
        (
            "schema_version", "name", "base", "ffn", "self_attention_projections",
            "self_attention_kernel", "cross_attention_projections", "decoder", "compile",
            "prepared_conditioning", "fast_accum",
        ),
        "policy",
    )
    # "fast_accum" is deliberately absent from `required`: every retained policy
    # predates it and must keep validating, and its recorded policy_sha256 is
    # taken over the requested dict, so omitting the key preserves the hash.
    required = {
        "schema_version", "name", "base", "ffn", "self_attention_projections",
        "self_attention_kernel", "cross_attention_projections", "decoder", "compile",
        "prepared_conditioning",
    }
    missing = sorted(required - set(policy))
    if missing:
        raise PolicyValidationError(f"Missing policy keys: {missing}")
    if policy["schema_version"] != SCHEMA_VERSION:
        raise PolicyValidationError(
            f"Unsupported schema_version {policy['schema_version']!r}; expected {SCHEMA_VERSION}"
        )
    _require_string(policy["name"], "name")
    _require_string(policy["base"], "base")

    ffn = policy["ffn"]
    _reject_unknown_keys(ffn, ("scheme", "exclude"), "ffn")
    if ffn.get("scheme") not in SUPPORTED_LINEAR_SCHEMES:
        raise PolicyValidationError(f"Unsupported FFN scheme: {ffn.get('scheme')!r}")
    _require_string_list(ffn.get("exclude"), "ffn.exclude")

    projections = policy["self_attention_projections"]
    _reject_unknown_keys(projections, ("scheme", "include", "exclude", "packing"), "self_attention_projections")
    if projections.get("scheme") not in SUPPORTED_LINEAR_SCHEMES:
        raise PolicyValidationError(
            f"Unsupported self-attention projection scheme: {projections.get('scheme')!r}"
        )
    _require_string_list(projections.get("include"), "self_attention_projections.include")
    _require_string_list(projections.get("exclude"), "self_attention_projections.exclude")
    if projections.get("packing") != "separate":
        raise PolicyValidationError("Only separate self-attention projection packing is supported")

    kernel = policy["self_attention_kernel"]
    _reject_unknown_keys(kernel, ("backend",), "self_attention_kernel")
    if kernel.get("backend") not in SUPPORTED_ATTENTION_BACKENDS:
        raise PolicyValidationError(f"Unsupported self-attention backend: {kernel.get('backend')!r}")

    cross = policy["cross_attention_projections"]
    _reject_unknown_keys(cross, ("scheme", "include", "exclude"), "cross_attention_projections")
    if cross.get("scheme") not in (BF16_SCHEME, FP8_SCHEME):
        raise PolicyValidationError(
            f"Unsupported cross-attention projection scheme: {cross.get('scheme')!r}; "
            f"supported: {sorted((BF16_SCHEME, FP8_SCHEME))}"
        )
    _require_string_list(cross.get("include"), "cross_attention_projections.include")
    _require_string_list(cross.get("exclude"), "cross_attention_projections.exclude")
    if cross["scheme"] == BF16_SCHEME and (cross["include"] or cross["exclude"]):
        raise PolicyValidationError("Cross-attention rules must be empty while its scheme is bf16")
    if cross["scheme"] != BF16_SCHEME:
        # k/v feed CrossAttention.prepare_kv, which returns bare tensors once per
        # window at M=288. There is nowhere in that contract to carry an
        # activation scale, and the GEMMs are too small to pay for one.
        for rule in list(cross["include"]) + list(cross["exclude"]):
            if rule.endswith((".k", ".v")) or ".k_img" in rule or ".v_img" in rule:
                raise PolicyValidationError(
                    f"Cross-attention k/v conversion is not supported: {rule!r}. "
                    "prepare_kv returns bare BF16 tensors with no scale carrier; "
                    "convert only cross_attn.q and cross_attn.o."
                )
        if not cross["include"]:
            raise PolicyValidationError(
                "A quantized cross-attention scheme must name its include rules"
            )

    decoder = policy["decoder"]
    _reject_unknown_keys(decoder, ("scheme", "backend", "plan"), "decoder")
    if decoder.get("scheme") not in SUPPORTED_DECODER_SCHEMES:
        raise PolicyValidationError(f"Unsupported decoder scheme: {decoder.get('scheme')!r}")
    if decoder.get("backend") not in SUPPORTED_DECODER_BACKENDS:
        raise PolicyValidationError(f"Unsupported decoder backend: {decoder.get('backend')!r}")
    stage_backend = decoder["backend"] in ("trt_stage", "trt_stage_compile")
    if stage_backend and (not isinstance(decoder.get("plan"), str) or not decoder["plan"]):
        raise PolicyValidationError("Stage decoder requires an explicit engine plan")
    if stage_backend and decoder["scheme"] not in (BF16_SCHEME, "fp16_stage_control", "int8_conservative"):
        raise PolicyValidationError("Stage decoder supports only BF16, FP16 control or verified INT8")
    if decoder['scheme'] == 'fp16_stage_control' and not stage_backend:
        raise PolicyValidationError('FP16 stage control requires a stage backend')
    if decoder["scheme"] == BF16_SCHEME and decoder.get("plan") is not None and not stage_backend:
        raise PolicyValidationError("A BF16 decoder policy cannot include a quantized decoder plan")
    if decoder["scheme"] != BF16_SCHEME and decoder.get("backend") not in ("adapter", "adapter_compile", "trt_stage", "trt_stage_compile"):
        raise PolicyValidationError("Quantized decoder policies require the cache-safe adapter backend")

    compile_policy = policy["compile"]
    _reject_unknown_keys(compile_policy, ("dit", "ffn_only"), "compile")
    if not all(isinstance(compile_policy.get(key), bool) for key in ("dit", "ffn_only")):
        raise PolicyValidationError("compile.dit and compile.ffn_only must be booleans")
    if compile_policy["ffn_only"] and not compile_policy["dit"]:
        raise PolicyValidationError("compile.ffn_only requires compile.dit")
    if not isinstance(policy["prepared_conditioning"], bool):
        raise PolicyValidationError("prepared_conditioning must be a boolean")
    if "fast_accum" in policy and not isinstance(policy["fast_accum"], bool):
        # Without this, a JSON string "false" or an int 0/1 would reach
        # bool(...) at conversion time and silently change FP8 accumulate
        # numerics. bool("false") is True.
        raise PolicyValidationError(
            f"fast_accum must be a boolean, got {type(policy['fast_accum']).__name__}"
        )


def _module_bytes(module: nn.Module) -> int:
    return sum(item.numel() * item.element_size() for item in module.parameters(recurse=False))


def _get_geometry(model: nn.Module) -> dict[str, Any]:
    blocks = getattr(model, "blocks", None)
    config = getattr(model, "config", None)
    first_block = blocks[0] if blocks is not None and len(blocks) else None
    first_attention = getattr(first_block, "self_attn", None)
    first_ffn = getattr(first_block, "ffn", None)
    return {
        "block_count": len(blocks) if blocks is not None else None,
        "vae_stride": list(getattr(config, "vae_stride", ())) if config is not None else None,
        "patch_size": list(getattr(model, "patch_size", ())),
        "hidden_dim": getattr(first_attention, "dim", None),
        "head_count": getattr(first_attention, "num_heads", None),
        "head_dim": getattr(first_attention, "head_dim", None),
        "ffn_dim": getattr(first_ffn[0], "out_features", None) if first_ffn is not None else None,
    }


def validate_pro_geometry(model: nn.Module, *, strict: bool = True) -> dict[str, Any]:
    """Check the model is PRO-compatible before planning changes."""
    geometry = _get_geometry(model)
    required: dict[str, Any] = {"vae_stride": [4, 8, 8], "patch_size": [1, 2, 2]}
    if strict:
        required.update({
            "block_count": 30,
            "hidden_dim": 1536,
            "head_count": 12,
            "head_dim": 128,
            "ffn_dim": 8960,
        })
    mismatches = {
        name: {"expected": expected, "actual": geometry[name]}
        for name, expected in required.items() if geometry[name] != expected
    }
    if mismatches:
        raise PolicyValidationError(f"Model does not match required PRO geometry: {mismatches}")
    return geometry


def _matching_paths(paths: Iterable[str], patterns: Iterable[str], where: str) -> list[str]:
    all_paths = sorted(paths)
    matched: set[str] = set()
    for pattern in patterns:
        hits = [path for path in all_paths if fnmatch.fnmatchcase(path, pattern)]
        if not hits:
            raise PolicyValidationError(f"Unmatched {where} pattern: {pattern}")
        overlap = sorted(matched.intersection(hits))
        if overlap:
            raise PolicyValidationError(f"Duplicate {where} target(s): {overlap}")
        matched.update(hits)
    return sorted(matched)


def _pro_target_paths(model: nn.Module) -> tuple[list[str], list[str], list[str]]:
    ffn_paths: list[str] = []
    attention_paths: list[str] = []
    cross_paths: list[str] = []
    blocks = getattr(model, "blocks", None)
    if blocks is None:
        raise PolicyValidationError("Model has no blocks collection")
    for index, block in enumerate(blocks):
        ffn = getattr(block, "ffn", None)
        if ffn is None:
            raise PolicyValidationError(f"blocks.{index} has no ffn")
        for projection in (0, 2):
            try:
                ffn[projection]
            except (IndexError, TypeError, KeyError) as error:
                raise PolicyValidationError(f"blocks.{index}.ffn.{projection} is missing") from error
            ffn_paths.append(f"blocks.{index}.ffn.{projection}")
        attention = getattr(block, "self_attn", None)
        if attention is None:
            raise PolicyValidationError(f"blocks.{index} has no self_attn")
        if getattr(attention, "packed_qkv", None) is not None:
            raise PolicyValidationError(
                f"blocks.{index}.self_attn has packed_qkv; separate projection conversion is unsafe"
            )
        for projection in ("q", "k", "v", "o"):
            if not hasattr(attention, projection):
                raise PolicyValidationError(f"blocks.{index}.self_attn.{projection} is missing")
            attention_paths.append(f"blocks.{index}.self_attn.{projection}")
        cross = getattr(block, "cross_attn", None)
        if cross is None:
            raise PolicyValidationError(f"blocks.{index} has no cross_attn")
        if getattr(cross, "has_image_input", False):
            raise PolicyValidationError(
                f"blocks.{index}.cross_attn has image input; only audio-only blocks are supported"
            )
        # Only q/o are offered. k/v are consumed by prepare_kv, which returns
        # bare tensors and would lose the activation scale.
        for projection in ("q", "o"):
            if not hasattr(cross, projection):
                raise PolicyValidationError(f"blocks.{index}.cross_attn.{projection} is missing")
            cross_paths.append(f"blocks.{index}.cross_attn.{projection}")
    return ffn_paths, attention_paths, cross_paths


def _get_module(model: nn.Module, path: str) -> nn.Module:
    try:
        return model.get_submodule(path)
    except AttributeError as error:
        raise PolicyValidationError(f"Model path cannot be resolved: {path}") from error


def _target_dtype(scheme: str) -> str:
    if scheme == FP8_SCHEME:
        return "float8_e4m3fn"
    if scheme == INT8_SCHEME:
        return "int8"
    return "bfloat16"


def _validate_target(path: str, family: str, scheme: str, module: nn.Module) -> ConversionTarget:
    if not isinstance(module, nn.Linear):
        raise PolicyValidationError(f"Expected an unmodified nn.Linear at {path}, found {type(module).__name__}")
    if module.weight.dtype != torch.bfloat16:
        raise PolicyValidationError(f"Expected BF16 weight at {path}, found {module.weight.dtype}")
    if module.in_features % 16 or module.out_features % 16:
        raise PolicyValidationError(
            f"Unsupported GEMM alignment at {path}: {module.in_features}x{module.out_features}"
        )
    if module.bias is not None and module.bias.dtype != torch.bfloat16:
        raise PolicyValidationError(f"Expected BF16 bias at {path}, found {module.bias.dtype}")
    return ConversionTarget(
        path=path,
        family=family,
        scheme=scheme,
        source_dtype=str(module.weight.dtype).removeprefix("torch."),
        target_dtype=_target_dtype(scheme),
        in_features=module.in_features,
        out_features=module.out_features,
        has_bias=module.bias is not None,
        original_bytes=_module_bytes(module),
    )


def build_conversion_plan(
    model: nn.Module,
    policy: Mapping[str, Any],
    *,
    strict_geometry: bool = True,
) -> ConversionPlan:
    """Build a complete conversion plan before any model mutation."""
    validate_policy(policy)
    geometry = validate_pro_geometry(model, strict=strict_geometry)
    ffn_paths, attention_paths, cross_paths = _pro_target_paths(model)
    ffn = policy["ffn"]
    projections = policy["self_attention_projections"]
    cross = policy["cross_attention_projections"]

    ffn_excluded = _matching_paths(ffn_paths, ffn["exclude"], "FFN exclusion")
    selected_ffns = []
    if ffn["scheme"] != BF16_SCHEME:
        selected_ffns = [path for path in ffn_paths if path not in ffn_excluded]

    attention_excluded = _matching_paths(
        attention_paths, projections["exclude"], "self-attention exclusion"
    )
    selected_attention = _matching_paths(
        attention_paths, projections["include"], "self-attention include"
    )
    selected_attention = [path for path in selected_attention if path not in attention_excluded]
    if projections["scheme"] == BF16_SCHEME and selected_attention:
        raise PolicyValidationError("BF16 self-attention policy cannot select conversion targets")

    cross_excluded = _matching_paths(
        cross_paths, cross["exclude"], "cross-attention exclusion"
    )
    selected_cross = _matching_paths(cross_paths, cross["include"], "cross-attention include")
    selected_cross = [path for path in selected_cross if path not in cross_excluded]
    if cross["scheme"] == BF16_SCHEME and selected_cross:
        raise PolicyValidationError("BF16 cross-attention policy cannot select conversion targets")

    targets: list[ConversionTarget] = []
    for path in selected_ffns:
        targets.append(_validate_target(path, "ffn", ffn["scheme"], _get_module(model, path)))
    for path in selected_attention:
        targets.append(
            _validate_target(path, "self_attention_projection", projections["scheme"], _get_module(model, path))
        )
    for path in selected_cross:
        targets.append(
            _validate_target(path, "cross_attention_projection", cross["scheme"], _get_module(model, path))
        )
    targets.sort(key=lambda target: target.path)
    return ConversionPlan(
        ResolvedPolicy(
            policy=dict(policy),
            targets=tuple(targets),
            excluded=tuple(sorted(set(ffn_excluded + attention_excluded + cross_excluded))),
            model_geometry=geometry,
        )
    )


def _replace_module(model: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, _, child_name = path.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    if child_name.isdigit():
        parent[int(child_name)] = replacement
    else:
        setattr(parent, child_name, replacement)


def _quantized_bytes(module: nn.Module) -> int:
    return sum(item.numel() * item.element_size() for item in module.buffers(recurse=False))


def apply_conversion_plan(model: nn.Module, plan: ConversionPlan) -> dict[str, Any]:
    """Apply a fully validated plan sequentially.

    Allocation errors can occur during replacement construction. In that case
    callers must discard this model and reload original weights rather than
    continue with a partly converted instance.
    """
    converted: list[dict[str, Any]] = []
    fast_accum = bool(plan.resolved.policy.get("fast_accum", False))
    try:
        for target in plan.resolved.targets:
            source = _get_module(model, target.path)
            if not isinstance(source, nn.Linear):
                raise ConversionError(
                    f"Target changed after planning at {target.path}; refusing mixed conversion"
                )
            if target.scheme == FP8_SCHEME:
                replacement = PolicyFloat8Linear(source, fast_accum=fast_accum)
            else:
                replacement = PolicyInt8ComputeLinear(source)
            _replace_module(model, target.path, replacement)
            converted.append({
                **asdict(target),
                "scale_granularity": "tensor" if target.scheme == FP8_SCHEME else "per_output_channel",
                "bias_dtype": "bfloat16" if target.has_bias else None,
                "accumulation": (
                    ("fp32_scaled_mm_fast_accum" if fast_accum else "fp32_scaled_mm")
                    if target.scheme == FP8_SCHEME else "int32_then_fp32"
                ),
                "backend": "pytorch_native_cuda",
                "backend_version": torch.__version__,
                "group_size": None,
                "packed_layout": "none",
                "fallback": False,
                "quantized_storage_bytes": _quantized_bytes(replacement),
            })
    except Exception as error:
        raise ConversionError(
            f"Conversion failed after {len(converted)} replacements; discard this model and reload original weights"
        ) from error
    return {
        "modules": converted,
        "original_bytes": sum(item["original_bytes"] for item in converted),
        "quantized_storage_bytes": sum(item["quantized_storage_bytes"] for item in converted),
        "resolved_policy": plan.resolved.to_dict(),
    }


def resolve_and_apply(
    model: nn.Module,
    policy: Mapping[str, Any],
    *,
    strict_geometry: bool = True,
) -> tuple[ConversionPlan, dict[str, Any]]:
    """Convenience API for callers that persist the resolved plan immediately."""
    plan = build_conversion_plan(model, policy, strict_geometry=strict_geometry)
    return plan, apply_conversion_plan(model, plan)


def write_resolved_policy(plan: ConversionPlan, output: str | Path) -> None:
    """Write a deterministic resolved-policy artifact after successful planning."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan.resolved.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
