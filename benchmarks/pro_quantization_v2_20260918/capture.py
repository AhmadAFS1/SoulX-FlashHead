"""Capture bounded, representative v2 activations in a separate diagnostic run."""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
from dataclasses import asdict
import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_FIXTURES,
    atomic_write_json,
    relative_path,
    sha256,
)
from soulx_rtc.pro_vae_quantization import (
    BF16ConvolutionBackend,
    DecoderPlan,
    DecoderTarget,
    install_decoder_adapters,
    remove_decoder_adapters,
)


class CaptureBudgetExceeded(RuntimeError):
    """The configured bounded raw-tensor capture limit has been reached."""


class PreparedInputCaptureBackend(BF16ConvolutionBackend):
    """Record the causal module's already concatenated and padded Conv3d input."""

    def __init__(self, collector: "BoundedActivationCapture", path: str) -> None:
        self.collector = collector
        self.path = path

    def __call__(self, module: nn.Module, prepared_input: torch.Tensor) -> torch.Tensor:
        self.collector.record(self.path, "causal_conv3d_prepared_input", prepared_input)
        return super().__call__(module, prepared_input)


class BoundedActivationCapture:
    """Capture statistics for all targets and raw tensors only within a hard budget."""

    def __init__(
        self,
        manifest: Path,
        max_raw_mib: int = 512,
        decoder_representative_paths: set[str] | None = None,
        capture_linears: bool = True,
        capture_attention: bool = False,
        capture_decoder_inputs: bool = True,
    ) -> None:
        if manifest.exists():
            raise FileExistsError(f"Capture manifest already exists: {manifest}")
        if max_raw_mib <= 0:
            raise ValueError("capture raw budget must be positive")
        self.manifest = manifest
        self.raw_dir = manifest.parent / "tensors"
        self.budget = max_raw_mib * 2**20
        self.bytes_saved = 0
        self.estimated_raw_bytes = 0
        self.budget_exceeded = False
        self.records: list[dict[str, Any]] = []
        self.raw_tensors: list[dict[str, Any]] = []
        self.handles: list[Any] = []
        self.decode_owner = None
        self.original_decode = None
        self.decoder_adapters_installed = False
        self.context = {"phase": "setup", "window": None}
        self.invocations: dict[str, int] = {}
        self.representative_paths: set[str] = set()
        self._estimated_paths: set[str] = set()
        self._decoder_capture_keys = set()
        self._finalized = False
        self._result: dict[str, Any] | None = None
        self.decoder_representative_paths = decoder_representative_paths or set()
        self.capture_linears = capture_linears
        self.capture_attention = capture_attention
        self.capture_decoder_inputs = capture_decoder_inputs
        self.attention_owners = []
        self.attention_calls = {}

    def set_context(self, *, phase: str, window: int) -> None:
        self.context = {"phase": phase, "window": window}

    def _statistics(self, tensor: torch.Tensor) -> dict[str, float]:
        values = tensor.detach().float().abs().reshape(-1)
        stride = max(1, values.numel() // 65536)
        sample = values[::stride]
        return {
            "rms": float(tensor.detach().float().square().mean().sqrt()),
            "absmax": float(values.max()),
            "p999_abs": float(torch.quantile(sample, 0.999)),
        }

    def _save_raw(self, path: str, kind: str, tensor: torch.Tensor, invocation: int) -> str | None:
        if path not in self.representative_paths:
            return None
        invocation_limit = 8 if kind in ("ffn_input", "self_projection_input", "public_vae_latent") else 2
        if kind in ("causal_conv3d_prepared_input", "public_vae_latent"):
            # Include late recurrence, not duplicate early post-reset latents.
            context = (self.context["phase"], self.context["window"])
            allowed = context in (("warmup", 0), ("warmup", 1), ("generation", 4), ("generation", 8))
            key = (path, context, tuple(tensor.shape))
            if not allowed or key in self._decoder_capture_keys:
                return None
            self._decoder_capture_keys.add(key)
            invocation_limit = 8
        elif invocation >= invocation_limit:
            return None
        byte_count = tensor.numel() * tensor.element_size()
        if path not in self._estimated_paths:
            self._estimated_paths.add(path)
            self.estimated_raw_bytes += byte_count * invocation_limit
        if self.bytes_saved + byte_count > self.budget:
            self.budget_exceeded = True
            raise CaptureBudgetExceeded(
                f"Capture raw-tensor budget {self.budget} bytes is insufficient for {path}; "
                f"estimated representative storage is {self.estimated_raw_bytes} bytes"
            )
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        name = f"{path.replace('.', '_')}-{kind}-{invocation}.pt"
        destination = self.raw_dir / name
        torch.save(tensor.detach().cpu(), destination)
        self.bytes_saved += byte_count
        self.raw_tensors.append({
            "module": path,
            "kind": kind,
            "invocation": invocation,
            "phase": self.context["phase"],
            "window": self.context["window"],
            "step": invocation % 4,
            "path": str(destination.relative_to(self.manifest.parent)),
            "sha256": sha256(destination),
            "bytes": byte_count,
            "shape": list(tensor.shape),
        })
        return name

    def record(self, path: str, kind: str, tensor: torch.Tensor) -> None:
        invocation = self.invocations.get(path, 0)
        self.invocations[path] = invocation + 1
        row = {
            "module": path,
            "kind": kind,
            "invocation": invocation,
            "phase": self.context["phase"],
            "window": self.context["window"],
            "step": invocation % 4,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            **self._statistics(tensor),
        }
        raw_name = self._save_raw(path, kind, tensor, invocation)
        if raw_name:
            row["raw_tensor"] = str(Path("tensors") / raw_name)
        self.records.append(row)

    def _linear_hook(self, path: str, kind: str):
        def hook(_module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if inputs:
                self.record(path, kind, inputs[0])

        return hook

    def _decoder_paths(self, vae: nn.Module) -> tuple[str, nn.Module]:
        model = getattr(vae, "model", vae)
        decoder = getattr(model, "decoder", None)
        if decoder is None:
            raise ValueError("VAE has no decoder to capture")
        return ("model.decoder" if model is not vae else "decoder"), decoder

    def attach(self, pipeline, conversion_plan) -> None:
        if self.capture_attention:
            for index, block in enumerate(pipeline.model.blocks):
                owner = block.self_attn
                original = owner._pro_attention_backend
                if original is None:
                    raise ValueError("Post-rotary attention capture requires an explicit attention backend")
                collector = self
                class CaptureBackend:
                    def __init__(self, index, original):
                        self.index, self.original = index, original

                    def __call__(self, q, k, v, heads, *, owner):
                        collector.record_attention(self.index, q, k, v)
                        return self.original(q, k, v, heads, owner=owner)
                owner._pro_attention_backend = CaptureBackend(index, original)
                self.attention_owners.append((owner, original))
        if self.capture_linears:
            targets = list(conversion_plan.resolved.targets)
            preferred_blocks = {0, 14, 29}
            hooked_paths: set[str] = set()
            for target in targets:
                if any(target.path.startswith(f"blocks.{index}.") for index in preferred_blocks):
                    self.representative_paths.add(target.path)
            if not self.representative_paths:
                self.representative_paths.update(target.path for target in targets[:3])
            self.representative_paths.add("vae.public_decode")
            for target in targets:
                module = pipeline.model.get_submodule(target.path)
                kind = "ffn_input" if target.family == "ffn" else "self_projection_input"
                self.handles.append(module.register_forward_pre_hook(self._linear_hook(target.path, kind)))
                hooked_paths.add(target.path)

            # A BF16 reference capture is the source of truth for prospective FP8
            # self-projection kernel tests. These paths must be observed even when
            # the active reference policy leaves attention projections untouched.
            for block_index in (0, 14, 29):
                for projection in ("q", "k", "v", "o"):
                    path = f"blocks.{block_index}.self_attn.{projection}"
                    if path in hooked_paths:
                        continue
                    module = pipeline.model.get_submodule(path)
                    if not isinstance(module, nn.Linear):
                        raise ValueError(f"Expected BF16 self-attention Linear at {path} for capture")
                    self.representative_paths.add(path)
                    self.handles.append(module.register_forward_pre_hook(self._linear_hook(path, "self_projection_input")))
                    hooked_paths.add(path)

        self.representative_paths.add("vae.public_decode")
        prefix, decoder = self._decoder_paths(pipeline.vae)
        from flash_head.wan.modules.vae import CausalConv3d

        causal_targets = []
        backends = {}
        for name, module in (decoder.named_modules() if self.capture_decoder_inputs else []):
            if isinstance(module, CausalConv3d):
                path = f"{prefix}.{name}"
                causal_targets.append(DecoderTarget(path, "causal_conv3d", ((1, 1, 1, 1, 1),), {}))
                backends[path] = PreparedInputCaptureBackend(self, path)
            elif isinstance(module, nn.Conv2d):
                path = f"{prefix}.{name}"
                self.handles.append(module.register_forward_pre_hook(self._linear_hook(path, "conv2d_pre_padding_input")))
        available_decoder_paths = {target.path for target in causal_targets}
        selected_decoder_paths = self.decoder_representative_paths & available_decoder_paths
        self.representative_paths.update(selected_decoder_paths or {target.path for target in causal_targets[:3]})
        if causal_targets:
            plan = DecoderPlan("capture-bf16-control", "bf16", tuple(causal_targets), (), {})
            install_decoder_adapters(pipeline.vae, plan, backends)
            self.decoder_adapters_installed = True

        self.decode_owner = pipeline.vae
        self.original_decode = pipeline.vae.decode

        def decode(latents, *args, **kwargs):
            self.record("vae.public_decode", "public_vae_latent", latents)
            return self.original_decode(latents, *args, **kwargs)

        pipeline.vae.decode = decode

    def record_attention(self, index, q, k, v):
        invocation = self.attention_calls.get(index, 0)
        self.attention_calls[index] = invocation + 1
        step = invocation % 4
        context = (self.context['phase'], self.context['window'])
        selected = (context == ('warmup', 0) and step == index % 4) or (
            context == ('generation', 4) and index in (0, 14, 29))
        if not selected:
            return
        for role, tensor in (('q', q), ('k', k), ('v', v)):
            size = tensor.numel() * tensor.element_size()
            if self.bytes_saved + size > self.budget:
                raise CaptureBudgetExceeded('Real-attention capture budget exceeded')
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            path = self.raw_dir / f'attention-block{index}-call{invocation}-{role}.pt'
            torch.save(tensor.detach().cpu(), path)
            self.bytes_saved += size
            self.raw_tensors.append({
                'module': f'blocks.{index}.self_attn', 'kind': f'attention_{role}',
                'path': str(path.relative_to(self.manifest.parent)), 'sha256': sha256(path),
                'block': index, 'step': step, 'invocation': invocation,
                'phase': context[0], 'window': context[1], 'shape': list(tensor.shape),
                'dtype': str(tensor.dtype), 'bytes': size,
            })

    def finalize(self, *, status: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        if self._finalized:
            return self._result or {}
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        for owner, original in self.attention_owners:
            owner._pro_attention_backend = original
        self.attention_owners.clear()
        if self.decode_owner is not None and self.original_decode is not None:
            self.decode_owner.decode = self.original_decode
        if self.decoder_adapters_installed and self.decode_owner is not None:
            remove_decoder_adapters(self.decode_owner)
        result = {
            "status": status,
            "execution": "GPU activation capture diagnostic; not a throughput measurement",
            "raw_budget_bytes": self.budget,
            "raw_bytes_saved": self.bytes_saved,
            "estimated_raw_bytes": self.estimated_raw_bytes,
            "raw_budget_exceeded": self.budget_exceeded,
            "records": self.records,
            "raw_tensors": self.raw_tensors,
            "capture_contract": "All target linear pre-inputs; selected raw tensors; CausalConv3d inputs after cache concatenation and padding; Conv2d inputs before module-owned padding.",
            **(extra or {}),
        }
        atomic_write_json(self.manifest, result)
        self._finalized = True
        self._result = result
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--split", choices=("calibration", "development", "held-out"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--max-raw-mib", type=int, default=6144)
    parser.add_argument("--decoder-inventory", type=Path)
    parser.add_argument("--decoder-capture-top", type=int, default=6)
    parser.add_argument("--decoder-only", action="store_true")
    parser.add_argument("--attention-inputs", action="store_true", help="Capture stratified real post-rotary Q/K/V")
    parser.add_argument("--latents-only", action="store_true", help="Capture public latents without decoder activation hooks")
    parser.add_argument("--gpu-lock", type=Path)
    args = parser.parse_args()
    from benchmarks.pro_quantization_v2_20260918.run import run_experiment

    args.repeats = 1
    args.save_raw = False
    args.capture_manifest = args.output / "manifest.json"
    args.capture_max_raw_mib = args.max_raw_mib
    args.capture_split = args.split
    args.capture_decoder_representative_paths = set()
    args.capture_linears = not args.decoder_only
    args.capture_attention = args.attention_inputs
    args.capture_decoder_inputs = not args.latents_only
    if args.latents_only or args.attention_inputs:
        args.capture_linears = False
    if args.decoder_inventory is not None:
        from benchmarks.pro_quantization_v2_20260918.make_decoder_plan import load_decoder_inventory, protected_paths

        inventory = load_decoder_inventory(args.decoder_inventory)
        protected = protected_paths(inventory, [])
        rows = [
            row for row in inventory
            if row["class"] == "CausalConv3d" and row["path"] not in protected
        ]
        rows.sort(key=lambda row: float(row["inclusive_cuda_ms"]), reverse=True)
        args.capture_decoder_representative_paths = {row["path"] for row in rows[:args.decoder_capture_top]}
    if args.gpu_lock is None:
        from benchmarks.pro_quantization_v2_20260918.common import DEFAULT_GPU_LOCK

        args.gpu_lock = DEFAULT_GPU_LOCK
    run_experiment(args)


if __name__ == "__main__":
    main()
