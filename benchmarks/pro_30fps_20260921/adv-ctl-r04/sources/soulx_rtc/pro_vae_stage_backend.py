"""Opt-in Wan residual stages with explicit causal cache tensor boundaries.

The ordinary VAE owns/reset caches. A stage neither persists them nor aliases
runtime output buffers between calls. No production configuration selects this.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn

from flash_head.utils.latency import latency_scope


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
                        self.observer(
                            f"blocks.{block_index}.residual.{layer_index}",
                            before,
                            previous,
                            x,
                        )
                else:
                    x = layer(x)
            x = x + residual
        return (x, *updates)


class _SkipStage(nn.Module):
    def forward(self, x, feat_cache=None, feat_idx=None):
        return x


class StageAdapter(nn.Module):
    def __init__(self, stage: ResidualStage, execute: Callable, cache_dtype=None):
        super().__init__()
        self.stage = stage
        # cache_dtype: an ADDITIONAL dtype the causal cache tensors may legally carry,
        # on top of x's own dtype. Set when the execute backend hands caches back in its
        # binding dtype (see TensorRTStage.cache_passthrough). Left None for the eager
        # ResidualStage path, which returns bf16 like x, so that path keeps the strict
        # single-dtype contract it has always had.
        self.cache_dtype = cache_dtype
        # Keep runtime objects out of the module tree / state dict.
        object.__setattr__(self, "execute", execute)

    @torch.compiler.disable
    def forward(self, x, feat_cache=None, feat_idx=None):
        if feat_cache is None or feat_idx is None:
            raise ValueError("Stage adapter requires stock causal cache ownership")
        begin = feat_idx[0]
        incoming = feat_cache[begin : begin + self.stage.cache_count]
        if len(incoming) != self.stage.cache_count:
            raise ValueError("Insufficient stage cache slots")
        if all(value is None for value in incoming):
            args = (x,)
        elif all(isinstance(value, torch.Tensor) for value in incoming):
            args = (x, *incoming)
        else:
            raise ValueError("Partially initialized residual stage cache")
        output = self.execute(*args)
        if (
            not isinstance(output, (tuple, list))
            or len(output) != self.stage.cache_count + 1
        ):
            raise ValueError("Invalid stage output/cache contract")
        # Validate all outputs before touching caller-owned state.
        # y must always come back in x's dtype; the caches may additionally carry
        # cache_dtype when the backend passes them through un-cast. Still an exact
        # allow-list, not a relaxation to "any dtype".
        cache_dtypes = (
            (x.dtype,) if self.cache_dtype is None else (x.dtype, self.cache_dtype)
        )
        if any(value.device != x.device for value in output):
            raise ValueError("Stage output device mismatch")
        if output[0].dtype != x.dtype:
            raise ValueError("Stage output dtype mismatch")
        if any(value.dtype not in cache_dtypes for value in output[1:]):
            raise ValueError("Stage cache dtype mismatch")
        for previous, value in zip(incoming, output[1:]):
            if (
                value.ndim != 5
                or value.shape[0] != x.shape[0]
                or value.shape[2] not in (1, 2)
            ):
                raise ValueError("Invalid returned causal cache shape")
            # Initial one-frame caches become two-frame caches next call.
            if previous is not None and (
                value.shape[1] != previous.shape[1]
                or value.shape[3:] != previous.shape[3:]
            ):
                raise ValueError("Returned cache spatial/channel mismatch")
        feat_cache[begin : begin + self.stage.cache_count] = list(output[1:])
        feat_idx[0] += self.stage.cache_count
        return output[0]


def install_stages(vae, groups, factory: Callable, cache_dtype=None) -> dict:
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
        if any(
            index < 0 or index >= len(upsamples) or index in selected
            for index in indices
        ):
            raise ValueError("Invalid or overlapping stage indices")
        selected.update(indices)
        blocks = [upsamples[index] for index in indices]
        stage = ResidualStage(blocks).eval()
        execute = factory(indices, stage)
        resolved.append((indices, stage, execute))
    originals = {index: upsamples[index] for index in selected}
    try:
        for indices, stage, execute in resolved:
            upsamples[indices[0]] = StageAdapter(stage, execute, cache_dtype)
            for index in indices[1:]:
                upsamples[index] = _SkipStage()
    except Exception:
        for index, module in originals.items():
            upsamples[index] = module
        raise
    vae._pro_stage_originals = originals
    return {
        "groups": [list(group) for group in groups],
        "cache_storage": (
            "bfloat16" if cache_dtype is None else str(cache_dtype).replace("torch.", "")
        ),
        "cache_owner": "stock WanVAE decode",
        "fallback": False,
    }


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

    def __init__(self, engine_path, metadata, *, arena=None, cache_passthrough=False):
        import tensorrt as trt

        # cache_passthrough: hand the causal cache tensors back to the caller in the
        # engine's own binding dtype instead of casting them to BF16 and back on every
        # call. The cache_in/cache_out bindings are already float16 in the plan, so the
        # values already round-trip through fp16 range in both directions today -- this
        # strictly DELETES a bf16 rounding step (11 -> 8 mantissa bits), it does not add
        # one. Only the x input and the y output still cross the precision boundary.
        self.cache_passthrough = bool(cache_passthrough)

        self.trt = trt
        self.path = Path(engine_path)
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != metadata["sha256"]:
            raise ValueError("Stage engine checksum mismatch")
        if (
            metadata["tensorrt"] != trt.__version__
            or metadata["gpu"] != torch.cuda.get_device_name()
        ):
            raise ValueError("Stage engine environment mismatch")
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError("Cannot deserialize stage engine")
        self.context = self.engine.create_execution_context_without_device_memory()
        if self.context is None:
            raise RuntimeError("Cannot create stage execution context")
        required = max(1, self.engine.device_memory_size)
        if arena is None:
            arena = {
                "tensor": torch.empty(required, dtype=torch.uint8, device="cuda"),
                "stream": torch.cuda.Stream(),
                "lock": threading.Lock(),
            }
        self.workspace = arena["tensor"]
        if self.workspace.numel() < required or self.workspace.dtype != torch.uint8:
            raise ValueError(
                "Shared stage workspace is smaller than the engine requirement"
            )
        self.context.device_memory = self.workspace.data_ptr()
        self.stream = arena["stream"]
        self.enqueue_lock = arena["lock"]
        self.inputs = metadata["inputs"]
        self.outputs = metadata["outputs"]
        dtype = metadata.get("binding_dtype", "bfloat16")
        if dtype not in ("bfloat16", "float16"):
            raise ValueError("Unsupported stage binding dtype")
        self.binding_dtype = getattr(torch, dtype)
        trt_dtype = trt.bfloat16 if dtype == "bfloat16" else trt.float16
        if self.engine.num_io_tensors != len(self.inputs) + len(self.outputs):
            raise ValueError("Stage binding count mismatch")
        for mode, specs in (
            (trt.TensorIOMode.INPUT, self.inputs),
            (trt.TensorIOMode.OUTPUT, self.outputs),
        ):
            for spec in specs:
                name = spec["name"]
                if (
                    tuple(self.engine.get_tensor_shape(name)) != tuple(spec["shape"])
                    or self.engine.get_tensor_mode(name) != mode
                    or self.engine.get_tensor_dtype(name) != trt_dtype
                ):
                    raise ValueError(f"Invalid stage binding {name}")
        self._prealloc = None
        self._pp = 0
        if os.environ.get("SOULX_STAGE_PREALLOC", "0") not in ("0", "false", "False"):
            dev = self.workspace.device
            self._prealloc = [
                [torch.empty(spec["shape"], device=dev, dtype=self.binding_dtype)
                 for spec in self.outputs]
                for _ in range(2)
            ]

    def __call__(self, *values):
        if len(values) != len(self.inputs):
            raise ValueError("Stage binding arity mismatch")
        with latency_scope("trt.input_cast_contiguous", engine=str(self.path)):
            prepared = []
            for index, (value, spec) in enumerate(zip(values, self.inputs)):
                # inputs[0] is x; the rest are cache_in_*. Under cache_passthrough the
                # caches arrive already in binding dtype, so only x crosses the boundary.
                allowed = (torch.bfloat16,)
                if index > 0 and self.cache_passthrough:
                    allowed = (torch.bfloat16, self.binding_dtype)
                if (
                    value.dtype not in allowed
                    or not value.is_cuda
                    or tuple(value.shape) != tuple(spec["shape"])
                    or value.device != self.workspace.device
                ):
                    raise ValueError("Stage binding shape/dtype/device mismatch")
                # One pass, not two. `.to(dtype)` PRESERVES non-contiguous strides, so the
                # historical `.to(dtype).contiguous()` made a second full pass over x
                # (einops hands us stride (106168320, 184320, 17694720, 320, 1)).
                # Fusing the format request into the same call collapses them; when the
                # source is already contiguous and of the target dtype this returns the
                # SAME tensor object, so it is a kernel deletion rather than a cheaper
                # kernel. Verified on this build for both the identity and strided cases.
                prepared.append(
                    value.to(self.binding_dtype, memory_format=torch.contiguous_format)
                )
        with latency_scope("trt.output_allocation", gpu=False, engine=str(self.path)):
            if self._prealloc is not None:
                self._pp ^= 1
                outputs = self._prealloc[self._pp]
                for got, spec in zip(outputs, self.outputs):
                    assert tuple(got.shape) == tuple(spec["shape"])
            else:
                outputs = [
                    torch.empty(
                        spec["shape"], device=self.workspace.device, dtype=self.binding_dtype
                    )
                    for spec in self.outputs
                ]
        caller = torch.cuda.current_stream(self.workspace.device)
        with latency_scope("trt.bind_enqueue_fence", engine=str(self.path)):  # noqa: SIM117 - record lock waiting too
            with self.enqueue_lock:
                for spec, value in zip(self.inputs + self.outputs, prepared + outputs):
                    if not self.context.set_tensor_address(spec["name"], value.data_ptr()):
                        raise RuntimeError("Stage binding failed")
                self.stream.wait_stream(caller)
                _rs = prepared + [self.workspace] if self._prealloc is not None else prepared + outputs + [self.workspace]
                for value in _rs:
                    value.record_stream(self.stream)
                # Time actual engine execution on its own stream, after the
                # input dependency; the caller span also includes fencing.
                with latency_scope("trt.execute", stream=self.stream, engine=str(self.path)):
                    if not self.context.execute_async_v3(self.stream.cuda_stream):
                        raise RuntimeError("Stage execution failed")
                caller.wait_stream(self.stream)
        with latency_scope("trt.output_cache_cast_bf16", engine=str(self.path)):
            if self.cache_passthrough:
                # Only y crosses back. The cache_out_* tensors stay in binding dtype and
                # are handed straight to the caller's feat_cache, where the next call
                # consumes them with no conversion at either end.
                return (outputs[0].to(torch.bfloat16), *outputs[1:])
            return tuple(value.to(torch.bfloat16) for value in outputs)


def install_stage_plan(vae, path):
    from benchmarks.pro_quantization_v2_20260918.common import (
        ROOT,
        resolve_path,
        sha256,
    )

    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != 1 or plan.get("status") != "complete":
        raise ValueError("Incomplete or unknown stage plan")
    checkpoint = ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"
    if plan["weights_sha256"] != sha256(checkpoint):
        raise ValueError("Stage weights mismatch")
    source_path = "flash_head/wan/modules/vae.py"
    if plan.get("source_sha256", {}).get(source_path) != sha256(ROOT / source_path):
        raise ValueError(
            "Stage plan was exported from a different Wan VAE implementation"
        )
    sizes = [
        record["workspace_bytes"]
        for group in plan["stages"].values()
        for record in group.values()
    ]
    if not sizes or any(not isinstance(size, int) or size < 0 for size in sizes):
        raise ValueError("Invalid stage workspace requirements")
    arena = {
        "tensor": torch.empty(max(1, max(sizes)), dtype=torch.uint8, device="cuda"),
        "stream": torch.cuda.Stream(),
        "lock": threading.Lock(),
    }

    # Keep the causal cache in the engines' own binding dtype across calls instead of
    # casting it to BF16 and back 27 times per window. Only meaningful when the bindings
    # are float16; with bf16 bindings the cast is already a no-op. Set
    # SOULX_STAGE_CACHE_PASSTHROUGH=0 to fall back to the historical behaviour without
    # editing code -- the two paths are numerically distinguishable only by one bf16
    # rounding step that passthrough removes.
    binding_dtype = (
        torch.float16
        if plan.get("floating_precision", "bf16") == "fp16"
        else torch.bfloat16
    )
    cache_passthrough = binding_dtype is torch.float16 and os.environ.get(
        "SOULX_STAGE_CACHE_PASSTHROUGH", "1"
    ) not in ("0", "false", "False")

    def factory(indices, stage):
        records = plan["stages"]["-".join(map(str, indices))]
        engines = {}
        for signature, record in records.items():
            if record["precision"] != plan["precision"]:
                raise ValueError("Stage precision mismatch")
            floating = plan.get("floating_precision", "bf16")
            if (
                floating not in ("bf16", "fp16")
                or record.get("floating_precision", "bf16") != floating
            ):
                raise ValueError("Stage floating precision mismatch")
            if record.get("binding_dtype", "bfloat16") != (
                "float16" if floating == "fp16" else "bfloat16"
            ):
                raise ValueError("Stage floating binding mismatch")
            checks = record.get("build_sample_comparison", [])
            if len(checks) != stage.cache_count + 1 or not all(
                item.get("finite") is True for item in checks
            ):
                raise ValueError(
                    "Stage engine lacks finite output/cache execution evidence"
                )
            layers = resolve_path(record["layers_path"])
            if sha256(layers) != record["layers_sha256"]:
                raise ValueError("Stage precision inspector hash mismatch")
            if plan["precision"] == "int8":
                from benchmarks.pro_30fps_20260919.decoder_stages import (
                    int8_convolutions,
                )

                evidence = int8_convolutions(layers.read_text())
                if (
                    not record.get("int8_convolutions_verified")
                    or evidence["int8_convolutions"] < stage.cache_count
                    or len(record.get("quantized_modules", [])) != stage.cache_count
                ):
                    raise ValueError("Missing INT8 computation proof")
            engines[signature] = TensorRTStage(
                resolve_path(record["path"]),
                record,
                arena=arena,
                cache_passthrough=cache_passthrough,
            )

        def execute(*values):
            signature = tensor_signature(values)
            if signature not in engines:
                raise ValueError(f"No verified stage signature {signature}")
            return engines[signature](*values)

        return execute

    result = install_stages(
        vae, plan["groups"], factory, cache_dtype=binding_dtype if cache_passthrough else None
    )
    result.update(
        {
            "plan": str(path),
            "plan_sha256": sha256(path),
            "precision": plan["precision"],
            "workspace_bytes": max(sizes),
            "unshared_workspace_bytes": sum(sizes),
            "workspace_execution": "one shared stream and host enqueue lock; serialized engine execution",
            "floating_precision": plan.get("floating_precision", "bf16"),
            "caller_cache_dtype": (
                str(binding_dtype).replace("torch.", "")
                if cache_passthrough
                else "bfloat16"
            ),
            "cache_passthrough": cache_passthrough,
        }
    )
    return result
