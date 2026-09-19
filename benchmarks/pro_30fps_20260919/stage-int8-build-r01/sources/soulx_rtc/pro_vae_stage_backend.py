"""Opt-in Wan residual stages with explicit causal cache tensor boundaries.

The ordinary VAE owns/reset caches. A stage neither persists them nor aliases
runtime output buffers between calls. No production configuration selects this.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import torch
from torch import nn


def tensor_signature(values) -> str:
    return "__".join("x".join(map(str, value.shape)) for value in values)


class ResidualStage(nn.Module):
    """Pure residual-block group; initial call has no incoming cache tensors."""

    def __init__(self, blocks: list[nn.Module]):
        super().__init__()
        from flash_head.wan.modules.vae import ResidualBlock
        if not blocks or not all(isinstance(block, ResidualBlock) for block in blocks):
            raise ValueError("Stage must contain only contiguous Wan ResidualBlocks")
        self.blocks = nn.ModuleList(blocks)
        self.cache_count = 2 * len(blocks)
        self.observer = None

    def forward(self, x, *caches):
        if len(caches) not in (0, self.cache_count):
            raise ValueError("Stage cache count mismatch")
        from flash_head.wan.modules.vae import CausalConv3d
        updates = []
        for block_index, block in enumerate(self.blocks):
            residual = block.shortcut(x)
            for layer_index, layer in enumerate(block.residual):
                if isinstance(layer, CausalConv3d):
                    index = len(updates)
                    previous = caches[index] if caches else None
                    update = x[:, :, -2:].clone()
                    if x.shape[2] < 2 and previous is not None:
                        update = torch.cat((previous[:, :, -1:], update), dim=2)
                    updates.append(update)
                    before = x
                    x = layer(x, previous)
                    if self.observer is not None:
                        self.observer(f"blocks.{block_index}.residual.{layer_index}", before, previous, x)
                else:
                    x = layer(x)
            x = x + residual
        return (x, *updates)


class _SkipStage(nn.Module):
    def forward(self, x, feat_cache=None, feat_idx=None):
        return x


class StageAdapter(nn.Module):
    def __init__(self, stage: ResidualStage, execute: Callable):
        super().__init__()
        self.stage = stage
        # Keep runtime objects out of the module tree / state dict.
        object.__setattr__(self, "execute", execute)

    @torch.compiler.disable
    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is None or feat_idx is None:
            raise ValueError("Stage adapter requires stock causal cache ownership")
        begin = feat_idx[0]
        incoming = feat_cache[begin:begin + self.stage.cache_count]
        if len(incoming) != self.stage.cache_count:
            raise ValueError("Insufficient stage cache slots")
        if all(value is None for value in incoming):
            args = (x,)
        elif all(isinstance(value, torch.Tensor) for value in incoming):
            args = (x, *incoming)
        else:
            raise ValueError("Partially initialized residual stage cache")
        output = self.execute(*args)
        if not isinstance(output, (tuple, list)) or len(output) != self.stage.cache_count + 1:
            raise ValueError("Invalid stage output/cache contract")
        # Validate all outputs before touching caller-owned state.
        if any(value.device != x.device or value.dtype != x.dtype for value in output):
            raise ValueError("Stage output device/dtype mismatch")
        for previous, value in zip(incoming, output[1:]):
            if value.ndim != 5 or value.shape[0] != x.shape[0] or value.shape[2] not in (1, 2):
                raise ValueError("Invalid returned causal cache shape")
            if previous is not None and value.shape[1:] != previous.shape[1:]:
                # Initial one-frame caches become two-frame caches next call.
                if value.shape[1] != previous.shape[1] or value.shape[3:] != previous.shape[3:]:
                    raise ValueError("Returned cache spatial/channel mismatch")
        feat_cache[begin:begin + self.stage.cache_count] = list(output[1:])
        feat_idx[0] += self.stage.cache_count
        return output[0]


def install_stages(vae, groups, factory: Callable) -> dict:
    """Resolve all groups before replacing any module; restore on install error."""
    if hasattr(vae, "_pro_stage_originals"):
        raise ValueError("Decoder stages already installed")
    upsamples = vae.model.decoder.upsamples
    selected = set()
    resolved = []
    for group in groups:
        indices = list(group)
        if not indices or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError("Stage indices must be contiguous")
        if any(index < 0 or index >= len(upsamples) or index in selected for index in indices):
            raise ValueError("Invalid or overlapping stage indices")
        selected.update(indices)
        blocks = [upsamples[index] for index in indices]
        stage = ResidualStage(blocks).eval()
        execute = factory(indices, stage)
        resolved.append((indices, stage, execute))
    originals = {index: upsamples[index] for index in selected}
    try:
        for indices, stage, execute in resolved:
            upsamples[indices[0]] = StageAdapter(stage, execute)
            for index in indices[1:]:
                upsamples[index] = _SkipStage()
    except Exception:
        for index, module in originals.items():
            upsamples[index] = module
        raise
    vae._pro_stage_originals = originals
    return {"groups": [list(group) for group in groups], "cache_storage": "bfloat16",
            "cache_owner": "stock WanVAE decode", "fallback": False}


def remove_stages(vae):
    originals = getattr(vae, "_pro_stage_originals", None)
    if originals is not None:
        for index, module in originals.items():
            vae.model.decoder.upsamples[index] = module
        del vae._pro_stage_originals


class TensorRTStage:
    """Static bindings, owned outputs and a stream fenced to the caller.

    A dedicated stream avoids TensorRT's default-stream synchronization and
    serializes shared workspace use across interleaved caller streams.
    """

    def __init__(self, engine_path, metadata):
        import tensorrt as trt
        self.trt = trt
        self.path = Path(engine_path)
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Stage engine checksum mismatch")
        if metadata["tensorrt"] != trt.__version__ or metadata["gpu"] != torch.cuda.get_device_name():
            raise ValueError("Stage engine environment mismatch")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError("Cannot deserialize stage engine")
        self.context = self.engine.create_execution_context_without_device_memory()
        if self.context is None:
            raise RuntimeError("Cannot create stage execution context")
        self.workspace = torch.empty(max(1, self.engine.device_memory_size), dtype=torch.uint8, device="cuda")
        self.context.device_memory = self.workspace.data_ptr()
        self.stream = torch.cuda.Stream(device=self.workspace.device)
        self.inputs = metadata["inputs"]
        self.outputs = metadata["outputs"]
        if self.engine.num_io_tensors != len(self.inputs) + len(self.outputs):
            raise ValueError("Stage binding count mismatch")
        for mode, specs in ((trt.TensorIOMode.INPUT, self.inputs), (trt.TensorIOMode.OUTPUT, self.outputs)):
            for spec in specs:
                name = spec["name"]
                if (tuple(self.engine.get_tensor_shape(name)) != tuple(spec["shape"])
                        or self.engine.get_tensor_mode(name) != mode
                        or self.engine.get_tensor_dtype(name) != trt.bfloat16):
                    raise ValueError(f"Invalid BF16 stage binding {name}")

    def __call__(self, *values):
        if len(values) != len(self.inputs):
            raise ValueError("Stage binding arity mismatch")
        prepared = []
        for value, spec in zip(values, self.inputs):
            if (value.dtype != torch.bfloat16 or not value.is_cuda
                    or tuple(value.shape) != tuple(spec["shape"])
                    or value.device != self.workspace.device):
                raise ValueError("Stage binding shape/dtype/device mismatch")
            prepared.append(value.contiguous())
        outputs = [torch.empty(spec["shape"], device=self.workspace.device, dtype=torch.bfloat16) for spec in self.outputs]
        for spec, value in zip(self.inputs + self.outputs, prepared + outputs):
            if not self.context.set_tensor_address(spec["name"], value.data_ptr()):
                raise RuntimeError("Stage binding failed")
        caller = torch.cuda.current_stream(self.workspace.device)
        self.stream.wait_stream(caller)
        for value in prepared + outputs + [self.workspace]:
            value.record_stream(self.stream)
        if not self.context.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("Stage execution failed")
        caller.wait_stream(self.stream)
        return tuple(outputs)


def install_stage_plan(vae, path):
    from benchmarks.pro_quantization_v2_20260918.common import resolve_path, sha256, ROOT
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 1 or plan.get("status") != "complete":
        raise ValueError("Incomplete or unknown stage plan")
    checkpoint = ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"
    if plan["weights_sha256"] != sha256(checkpoint):
        raise ValueError("Stage weights mismatch")
    def factory(indices, stage):
        records = plan["stages"]["-".join(map(str, indices))]
        engines = {}
        for signature, record in records.items():
            if record["precision"] != plan["precision"]:
                raise ValueError("Stage precision mismatch")
            checks = record.get("build_sample_comparison", [])
            if len(checks) != stage.cache_count + 1 or not all(item.get("finite") is True for item in checks):
                raise ValueError("Stage engine lacks finite output/cache execution evidence")
            layers = resolve_path(record["layers_path"])
            if sha256(layers) != record["layers_sha256"]:
                raise ValueError("Stage precision inspector hash mismatch")
            if plan["precision"] == "int8":
                from benchmarks.pro_30fps_20260919.decoder_stages import int8_convolutions
                evidence = int8_convolutions(layers.read_text())
                if (not record.get("int8_convolutions_verified")
                        or evidence['int8_convolutions'] < stage.cache_count
                        or len(record.get('quantized_modules', [])) != stage.cache_count):
                    raise ValueError("Missing INT8 computation proof")
            engines[signature] = TensorRTStage(resolve_path(record["path"]), record)
        def execute(*values):
            signature = tensor_signature(values)
            if signature not in engines:
                raise ValueError(f"No verified stage signature {signature}")
            return engines[signature](*values)
        return execute
    result = install_stages(vae, plan["groups"], factory)
    result.update({"plan": str(path), "plan_sha256": sha256(path), "precision": plan["precision"]})
    return result
