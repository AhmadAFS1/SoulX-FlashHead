"""Run real-input projection kernel checks before video-quality experiments."""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
import copy
import fnmatch
import importlib
import io
import json
from pathlib import Path
import time
from typing import Any
from collections import defaultdict

import torch
from torch import nn

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_GPU_LOCK,
    ROOT,
    atomic_write_json,
    ensure_new_directory,
    environment_manifest,
    failure_record,
    relative_path,
    sha256,
    summarize,
)
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.pro_quantization_v2 import FP8_SCHEME, PolicyFloat8Linear, load_policy


def _benchmark(module: nn.Module, value: torch.Tensor, *, batches: int, repetitions: int) -> list[float]:
    measurements = []
    with torch.inference_mode():
        for _ in range(4):
            module(value)
        torch.cuda.synchronize()
        for _ in range(batches):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repetitions):
                module(value)
            end.record()
            end.synchronize()
            measurements.append(begin.elapsed_time(end) / repetitions)
    return measurements


def _metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, Any]:
    delta = actual.float() - expected.float()
    denominator = expected.float().norm().clamp_min(1e-12)
    return {
        "shape": list(actual.shape),
        "dtype": str(actual.dtype).removeprefix("torch."),
        "finite": bool(actual.isfinite().all()),
        "max_abs": float(delta.abs().max()),
        "mean_abs": float(delta.abs().mean()),
        "relative_l2": float(delta.norm() / denominator),
    }


def _captured_inputs(manifest: Path, selected: set[str]) -> list[dict[str, Any]]:
    raw = json.loads(manifest.read_text(encoding="utf-8"))
    if raw.get("status") != "complete":
        raise ValueError(f"Capture manifest is not complete: {manifest}")
    rows = []
    for entry in raw.get("raw_tensors", []):
        if entry.get("module") in selected and entry.get("kind") in ("self_projection_input", "ffn_input"):
            source = manifest.parent / entry["path"]
            if sha256(source) != entry["sha256"]:
                raise ValueError(f"Capture tensor hash mismatch: {source}")
            rows.append({**entry, "source": source})
    rows.sort(key=lambda row: (row["module"], row.get("invocation", 0)))
    if not rows:
        raise ValueError("No bounded raw captures match the selected policy targets")
    return rows


def _load_linear(path: str, input_width: int) -> nn.Linear:
    checkpoint = ROOT / "models/SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors"
    try:
        from safetensors import safe_open
    except ImportError as error:
        raise RuntimeError("safetensors is required for kernel checks") from error
    with safe_open(checkpoint, framework="pt", device="cpu") as weights:
        weight = weights.get_tensor(f"{path}.weight")
        bias_name = f"{path}.bias"
        bias = weights.get_tensor(bias_name) if bias_name in weights.keys() else None
    if weight.ndim != 2 or weight.shape[1] != input_width:
        raise ValueError(f"Checkpoint weight shape {tuple(weight.shape)} does not match captured input for {path}")
    linear = nn.Linear(input_width, weight.shape[0], bias=bias is not None, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.copy_(weight.to(torch.bfloat16))
        if bias is not None:
            linear.bias.copy_(bias.to(torch.bfloat16))
    return linear.cuda().eval().requires_grad_(False)


def _dispatch_names(module: nn.Module, value: torch.Tensor) -> list[str]:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        with torch.inference_mode():
            module(value)
        torch.cuda.synchronize()
    names = {
        event.name for event in profiler.events()
        if any(token in event.name.lower() for token in ("scaled_mm", "gemm", "cutlass", "float8"))
    }
    return sorted(names)


def _input_checks(module: nn.Module, compiled: nn.Module, reference: nn.Module, value: torch.Tensor) -> dict[str, Any]:
    cases = {
        "zero": torch.zeros_like(value),
        "small": value * 1e-4,
        "mixed_sign": value.float().abs().mul(torch.where(
            torch.arange(value.numel(), device=value.device).reshape(value.shape) % 2 == 0, 1, -1
        )).to(value.dtype),
        "outlier": value.clone(),
    }
    cases["outlier"].reshape(-1)[0] = 100 * max(float(value.float().abs().max()), 1e-3)
    rows = {}
    with torch.inference_mode():
        for name, candidate in cases.items():
            before = candidate.clone()
            expected = reference(candidate)
            eager = module(candidate)
            compiled_actual = compiled(candidate)
            rows[name] = {
                "eager": _metrics(eager, expected),
                "compiled": _metrics(compiled_actual, expected),
                "compiled_vs_eager_max_abs": float((compiled_actual.float() - eager.float()).abs().max()),
                "input_unchanged": bool(torch.equal(candidate, before)),
            }
    return rows


def _w4a8_gate(backend_module: str | None) -> dict[str, Any]:
    result = {
        "component": "ffn-int4",
        "status": "rejected",
        "reason": "No registered W4A8 backend was supplied; FP8 remains the comparison control.",
    }
    if backend_module is None:
        return result
    try:
        backend = importlib.import_module(backend_module)
    except ImportError as error:
        result["reason"] = f"Unable to import requested W4A8 backend {backend_module}: {error}"
        return result
    metadata = getattr(backend, "BACKEND_METADATA", None)
    factory = getattr(backend, "build_w4a8_linear", None)
    if not isinstance(metadata, dict) or metadata.get("real_quantized_compute") is not True or not callable(factory):
        result["reason"] = (
            "W4A8 plugin must expose BACKEND_METADATA.real_quantized_compute=True and build_w4a8_linear; "
            "a dequantized BF16 implementation is not eligible."
        )
        return result
    result.update(status="ready", backend=metadata, factory=f"{backend_module}.build_w4a8_linear")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component", choices=("self-projections", "ffn", "ffn-int4"), required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--max-inputs", type=int, default=0)
    parser.add_argument("--w4a8-backend-module")
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    args = parser.parse_args()
    if args.batches < 5 or args.repetitions < 1 or args.max_inputs < 0:
        parser.error("batches must be at least five; repetitions must be positive; max-inputs cannot be negative")
    output = ensure_new_directory(args.output)
    result: dict[str, Any] = {
        "status": "starting",
        "execution": "fresh GPU diagnostic on captured tensors; not video generation",
        "environment": environment_manifest(),
        "component": args.component,
        "captures": relative_path(args.captures),
        "policy": relative_path(args.policy),
        "rows": [],
    }

    def save() -> None:
        atomic_write_json(output / "results.json", result)

    save()
    lease = None
    stage = "validate"
    try:
        if args.component == "ffn-int4":
            result["int4_gate"] = _w4a8_gate(args.w4a8_backend_module)
            result["status"] = "readiness_only" if result["int4_gate"]["status"] == "ready" else "rejected"
            save()
            return
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for projection kernel checks")
        policy = load_policy(args.policy)
        if args.component == "self-projections":
            if policy["self_attention_projections"]["scheme"] != FP8_SCHEME:
                raise ValueError("Self-projection kernel checks require an FP8 self-attention policy")
            selected = set(policy["self_attention_projections"]["include"])
            selected_paths = {
                path for path in (
                    f"blocks.{index}.self_attn.{projection}"
                    for index in range(30) for projection in ("q", "k", "v", "o")
                ) if any(fnmatch.fnmatchcase(path, pattern) for pattern in selected)
            }
        else:
            selected_paths = {f"blocks.{index}.ffn.{projection}" for index in range(30) for projection in (0, 2)}
        captures = _captured_inputs(args.captures, selected_paths)
        if args.max_inputs:
            captures = captures[:args.max_inputs]
        lease = acquire_gpu_lease(args.gpu_lock)
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for capture in captures:
            grouped[capture["module"]].append(capture)
        for module_path, module_captures in grouped.items():
            stage = f"module:{module_path}"
            initial = torch.load(module_captures[0]["source"], map_location="cuda", weights_only=True).to(torch.bfloat16)
            reference = _load_linear(module_path, initial.shape[-1])
            quantized = PolicyFloat8Linear(copy.deepcopy(reference)).cuda().eval().requires_grad_(False)
            compiled = torch.compile(quantized, fullgraph=True, dynamic=False)
            buffer = io.BytesIO()
            torch.save(quantized.state_dict(), buffer)
            buffer.seek(0)
            reloaded = PolicyFloat8Linear(copy.deepcopy(reference)).cuda().eval()
            reloaded.load_state_dict(torch.load(buffer, weights_only=True))
            dispatch = {
                "fp8": _dispatch_names(quantized, initial),
                "compiled_fp8": _dispatch_names(compiled, initial),
            }
            timed_shapes: set[tuple[int, ...]] = set()
            for capture in module_captures:
                stage = f"module:{module_path}:{capture['invocation']}"
                value = torch.load(capture["source"], map_location="cuda", weights_only=True).to(torch.bfloat16)
                with torch.inference_mode():
                    expected = reference(value)
                    eager = quantized(value)
                    compiled_actual = compiled(value)
                    reload_exact = torch.equal(reloaded(value), eager)
                row = {
                    "module": module_path,
                    "capture": {key: value for key, value in capture.items() if key not in ("source",)},
                    "reference": _metrics(expected, expected),
                    "eager_fp8": _metrics(eager, expected),
                    "compiled_fp8": _metrics(compiled_actual, expected),
                    "state_dict_reload_exact": reload_exact,
                    "fp8_dispatch": dispatch["fp8"],
                    "compiled_fp8_dispatch": dispatch["compiled_fp8"],
                    "input_cases": _input_checks(quantized, compiled, reference, value),
                }
                shape = tuple(value.shape)
                if shape not in timed_shapes:
                    timed_shapes.add(shape)
                    row["bf16_ms"] = _benchmark(reference, value, batches=args.batches, repetitions=args.repetitions)
                    row["bf16_summary_ms"] = summarize(row["bf16_ms"])
                    row["eager_ms"] = _benchmark(quantized, value, batches=args.batches, repetitions=args.repetitions)
                    row["compiled_ms"] = _benchmark(compiled, value, batches=args.batches, repetitions=args.repetitions)
                    row["eager_summary_ms"] = summarize(row["eager_ms"])
                    row["compiled_summary_ms"] = summarize(row["compiled_ms"])
                else:
                    row["timing_reused_from_shape"] = list(shape)
                row["status"] = "complete" if row["eager_fp8"]["finite"] and row["compiled_fp8"]["finite"] else "rejected"
                result["rows"].append(row)
                del value, expected, eager, compiled_actual
                save()
            save()
            del reloaded, compiled, quantized, reference, initial, buffer
            torch.cuda.empty_cache()
        result["status"] = "complete"
        save()
    except Exception as error:
        result.update(status="failed", failure=failure_record(stage, error))
        save()
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    main()