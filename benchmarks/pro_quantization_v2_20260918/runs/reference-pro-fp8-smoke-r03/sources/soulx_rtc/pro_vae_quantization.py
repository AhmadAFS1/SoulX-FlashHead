"""Cache-safe decoder compute adapters for PRO quantization experiments.

The Wan decoder dispatches by ``isinstance(CausalConv3d)`` and owns mutable
cache lists outside each convolution. This module keeps those objects in place
and replaces only their final convolution dispatch after the original causal
cache concatenation and padding have happened.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
from torch import nn
import torch.nn.functional as F


DECODER_PLAN_SCHEMA_VERSION = 1
SUPPORTED_PLAN_SCHEMES = frozenset(("bf16", "int8_conservative", "fp8_conservative"))
SUPPORTED_TARGET_KINDS = frozenset(("causal_conv3d", "conv2d"))


class DecoderPlanError(ValueError):
    """A decoder plan or adapter cannot preserve the VAE contract."""


class DecoderBackendError(RuntimeError):
    """An installed decoder backend failed its dispatch contract."""


@dataclass(frozen=True)
class DecoderTarget:
    path: str
    kind: str
    observed_shapes: tuple[tuple[int, ...], ...]
    calibration: Mapping[str, Any]


@dataclass(frozen=True)
class DecoderPlan:
    name: str
    scheme: str
    targets: tuple[DecoderTarget, ...]
    protected: tuple[str, ...]
    source: Mapping[str, Any] = field(default_factory=dict)
    engines: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": DECODER_PLAN_SCHEMA_VERSION,
            "name": self.name,
            "scheme": self.scheme,
            "targets": [asdict(target) for target in self.targets],
            "protected": list(self.protected),
            "engines": dict(self.engines),
        }


def _reject_unknown_keys(value: Any, allowed: Iterable[str], where: str) -> None:
    if not isinstance(value, Mapping):
        raise DecoderPlanError(f"{where} must be an object")
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise DecoderPlanError(f"Unknown keys in {where}: {unknown}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DecoderPlanError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def load_decoder_plan(path: str | Path, *, require_engines: bool = True) -> DecoderPlan:
    """Load a strict, calibration-bearing decoder plan."""
    try:
        with Path(path).open(encoding="utf-8") as handle:
            raw = json.load(handle, object_pairs_hook=_object_without_duplicate_keys)
    except json.JSONDecodeError as error:
        raise DecoderPlanError(f"Invalid decoder plan JSON: {error}") from error
    _reject_unknown_keys(raw, ("schema_version", "name", "scheme", "targets", "protected", "engines", "source"), "decoder plan")
    required = {"schema_version", "name", "scheme", "targets", "protected"}
    missing = sorted(required - set(raw))
    if missing:
        raise DecoderPlanError(f"Missing decoder plan keys: {missing}")
    if raw["schema_version"] != DECODER_PLAN_SCHEMA_VERSION:
        raise DecoderPlanError("Unsupported decoder plan schema version")
    if not isinstance(raw["name"], str) or not raw["name"]:
        raise DecoderPlanError("decoder plan name must be a non-empty string")
    if raw["scheme"] not in SUPPORTED_PLAN_SCHEMES:
        raise DecoderPlanError(f"Unsupported decoder scheme: {raw['scheme']!r}")
    if not isinstance(raw["targets"], list) or not raw["targets"]:
        raise DecoderPlanError("decoder plan requires one or more targets")
    if not isinstance(raw["protected"], list) or any(not isinstance(item, str) for item in raw["protected"]):
        raise DecoderPlanError("protected must be a list of paths")

    targets: list[DecoderTarget] = []
    seen: set[str] = set()
    for index, target in enumerate(raw["targets"]):
        where = f"targets[{index}]"
        _reject_unknown_keys(target, ("path", "kind", "observed_shapes", "calibration"), where)
        if set(target) != {"path", "kind", "observed_shapes", "calibration"}:
            raise DecoderPlanError(f"{where} must include path, kind, observed_shapes, and calibration")
        path_value = target["path"]
        if not isinstance(path_value, str) or not path_value or path_value in seen:
            raise DecoderPlanError(f"{where}.path must be a unique non-empty string")
        seen.add(path_value)
        if target["kind"] not in SUPPORTED_TARGET_KINDS:
            raise DecoderPlanError(f"{where}.kind is unsupported")
        if not isinstance(target["observed_shapes"], list) or not target["observed_shapes"]:
            raise DecoderPlanError(f"{where}.observed_shapes must be non-empty")
        shapes: list[tuple[int, ...]] = []
        for shape in target["observed_shapes"]:
            if not isinstance(shape, list) or len(shape) not in (4, 5) or any(
                not isinstance(dimension, int) or dimension <= 0 for dimension in shape
            ):
                raise DecoderPlanError(f"{where}.observed_shapes contains an invalid shape")
            shapes.append(tuple(shape))
        if not isinstance(target["calibration"], Mapping):
            raise DecoderPlanError(f"{where}.calibration must be an object")
        targets.append(DecoderTarget(path_value, target["kind"], tuple(shapes), dict(target["calibration"])))
    protected = tuple(raw["protected"])
    if set(protected) & seen:
        raise DecoderPlanError("A decoder path cannot be both targeted and protected")
    engines = raw.get("engines", {})
    if not isinstance(engines, Mapping) or set(engines) - seen:
        raise DecoderPlanError("engines must map only selected decoder target paths")
    for path, engine in engines.items():
        if not isinstance(engine, Mapping) or set(engine) != {"path", "sha256", "precision"}:
            raise DecoderPlanError(f"Engine record for {path} must contain path, sha256, and precision")
        if engine["precision"] not in ("int8", "fp8", "bf16"):
            raise DecoderPlanError(f"Engine record for {path} has unsupported precision")
    if require_engines and raw["scheme"] != "bf16" and set(engines) != seen:
        raise DecoderPlanError("Quantized decoder plans require one explicit engine record per target")
    source = raw.get("source", {})
    if not isinstance(source, Mapping):
        raise DecoderPlanError("source must be an object when present")
    return DecoderPlan(raw["name"], raw["scheme"], tuple(targets), protected, dict(source), dict(engines))


class DecoderComputeBackend:
    """Backend interface receiving exactly the convolution's prepared input."""

    name = "unknown"
    real_quantized_compute = False

    def __call__(self, module: nn.Module, prepared_input: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def manifest(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "real_quantized_compute": self.real_quantized_compute,
        }


class BF16ConvolutionBackend(DecoderComputeBackend):
    """Exact PyTorch convolution control using the original weight and layout."""

    name = "pytorch_bf16_control"

    def __call__(self, module: nn.Module, prepared_input: torch.Tensor) -> torch.Tensor:
        return module._conv_forward(prepared_input, module.weight, module.bias)


class EngineConvolutionBackend(DecoderComputeBackend):
    """Adapter for a separately built, explicitly quantized convolution engine."""

    real_quantized_compute = True

    def __init__(
        self,
        engine: Callable[[torch.Tensor], torch.Tensor],
        *,
        name: str,
        precision: str,
        manifest_data: Mapping[str, Any],
    ) -> None:
        if not callable(engine):
            raise DecoderPlanError("Decoder engine must be callable")
        if precision not in ("int8", "fp8"):
            raise DecoderPlanError(f"Unsupported decoder engine precision: {precision}")
        self.engine = engine
        self.name = name
        self.precision = precision
        self.manifest_data = dict(manifest_data)

    def __call__(self, module: nn.Module, prepared_input: torch.Tensor) -> torch.Tensor:
        output = self.engine(prepared_input)
        if not isinstance(output, torch.Tensor):
            raise DecoderBackendError(f"{self.name} returned {type(output).__name__}, not a tensor")
        if output.dtype != prepared_input.dtype:
            raise DecoderBackendError(
                f"{self.name} changed output dtype from {prepared_input.dtype} to {output.dtype}"
            )
        if not output.isfinite().all():
            raise DecoderBackendError(f"{self.name} returned nonfinite values")
        return output

    def manifest(self) -> dict[str, Any]:
        return {
            **super().manifest(),
            "precision": self.precision,
            "engine": self.manifest_data,
        }


def fit_symmetric_scale(
    tensors: Iterable[torch.Tensor],
    *,
    quant_max: float,
    percentile: float | None = None,
) -> dict[str, float | int]:
    """Fit a finite positive calibration scale without retaining activation dumps."""
    if quant_max <= 0:
        raise DecoderPlanError("quant_max must be positive")
    maxima: list[torch.Tensor] = []
    samples = 0
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor) or not tensor.numel():
            continue
        values = tensor.detach().float().abs().reshape(-1)
        if not torch.isfinite(values).all():
            raise DecoderPlanError("Calibration tensor contains nonfinite values")
        if percentile is None:
            maxima.append(values.max())
        else:
            if not 0 < percentile <= 100:
                raise DecoderPlanError("percentile must be in (0, 100]")
            maxima.append(torch.quantile(values, percentile / 100.0))
        samples += values.numel()
    if not maxima:
        raise DecoderPlanError("No calibration values were supplied")
    maximum = torch.stack(maxima).max().item()
    scale = max(maximum / quant_max, 1e-12)
    return {
        "algorithm": "max_abs" if percentile is None else "percentile_abs",
        "percentile": percentile,
        "samples": samples,
        "maximum": maximum,
        "scale": scale,
    }


def _default_causal_conv3d_type() -> type[nn.Module]:
    from flash_head.wan.modules.vae import CausalConv3d

    return CausalConv3d


def _decoder_root(vae: nn.Module) -> nn.Module:
    model = getattr(vae, "model", vae)
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        raise DecoderPlanError("VAE has no decoder")
    return decoder


def _module_for_path(vae: nn.Module, path: str) -> nn.Module:
    candidates = (path, f"model.{path}") if not path.startswith("model.") else (path, path[6:])
    for candidate in candidates:
        try:
            return vae.get_submodule(candidate)
        except AttributeError:
            continue
    raise DecoderPlanError(f"Decoder plan path cannot be resolved: {path}")


def _causal_forward(module: nn.Module, backend: DecoderComputeBackend, x: torch.Tensor, cache_x=None) -> torch.Tensor:
    padding = list(module._padding)
    if cache_x is not None and module._padding[4] > 0:
        cache_x = cache_x.to(x.device)
        x = torch.cat([cache_x, x], dim=2)
        padding[4] -= cache_x.shape[2]
    prepared_input = F.pad(x, padding)
    return backend(module, prepared_input)


def _install_causal_wrapper(module: nn.Module, backend: DecoderComputeBackend) -> None:
    if hasattr(module, "_pro_decoder_original_forward"):
        raise DecoderPlanError("Decoder adapter is already installed on this CausalConv3d")
    original = module.forward

    def forward(x: torch.Tensor, cache_x=None) -> torch.Tensor:
        return _causal_forward(module, backend, x, cache_x)

    module._pro_decoder_original_forward = original
    module._pro_decoder_backend = backend
    module.forward = forward


def _install_conv2d_wrapper(module: nn.Module, backend: DecoderComputeBackend) -> None:
    if hasattr(module, "_pro_decoder_original_forward"):
        raise DecoderPlanError("Decoder adapter is already installed on this Conv2d")
    original = module.forward

    def forward(x: torch.Tensor) -> torch.Tensor:
        return backend(module, x)

    module._pro_decoder_original_forward = original
    module._pro_decoder_backend = backend
    module.forward = forward


def install_decoder_adapters(
    vae: nn.Module,
    plan: DecoderPlan,
    backends: Mapping[str, DecoderComputeBackend],
    *,
    causal_conv3d_type: type[nn.Module] | None = None,
) -> dict[str, Any]:
    """Install validated adapters without replacing cache-owning VAE modules."""
    causal_conv3d_type = causal_conv3d_type or _default_causal_conv3d_type()
    _decoder_root(vae)
    planned: list[tuple[DecoderTarget, nn.Module, DecoderComputeBackend]] = []
    for target in plan.targets:
        module = _module_for_path(vae, target.path)
        backend = backends.get(target.path)
        if backend is None:
            raise DecoderPlanError(f"No backend was supplied for decoder target {target.path}")
        if isinstance(backend, nn.Module):
            raise DecoderPlanError("Decoder backends must not be modules; nested CausalConv3d changes cache counting")
        if plan.scheme != "bf16" and not backend.real_quantized_compute:
            raise DecoderPlanError(
                f"{target.path} uses {backend.name}, which is not real {plan.scheme} compute"
            )
        if target.kind == "causal_conv3d" and not isinstance(module, causal_conv3d_type):
            raise DecoderPlanError(f"Expected CausalConv3d at {target.path}, found {type(module).__name__}")
        if target.kind == "conv2d" and not isinstance(module, nn.Conv2d):
            raise DecoderPlanError(f"Expected Conv2d at {target.path}, found {type(module).__name__}")
        if hasattr(module, "_pro_decoder_original_forward"):
            raise DecoderPlanError(f"Decoder target {target.path} is already adapted")
        planned.append((target, module, backend))

    installed: list[dict[str, Any]] = []
    for target, module, backend in planned:
        if target.kind == "causal_conv3d":
            _install_causal_wrapper(module, backend)
        else:
            _install_conv2d_wrapper(module, backend)
        installed.append({
            "path": target.path,
            "kind": target.kind,
            "observed_shapes": [list(shape) for shape in target.observed_shapes],
            "calibration": dict(target.calibration),
            **backend.manifest(),
        })
    return {"scheme": plan.scheme, "plan": plan.to_dict(), "modules": installed}


def remove_decoder_adapters(vae: nn.Module) -> None:
    """Restore all adapted decoder modules to their original bound forwards."""
    for module in _decoder_root(vae).modules():
        original = getattr(module, "_pro_decoder_original_forward", None)
        if original is not None:
            module.forward = original
            delattr(module, "_pro_decoder_original_forward")
            delattr(module, "_pro_decoder_backend")