"""Build and verify exact-shape explicit-Q/DQ TensorRT Wan decoder convolutions.

Only INT8 is implemented here because its ONNX Q/DQ contract is available in
the pinned TensorRT environment. FP8 is rejected until that environment proves
an explicit FP8 Conv3d parser/tactic path; it is never relabeled as INT8 or
dequantized BF16 execution.
"""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch

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
from soulx_rtc.pro_vae_quantization import DecoderPlan, DecoderPlanError, decoder_module_for_path, load_decoder_plan, shape_key


class TensorRTPreparedConv:
    """Exact-shape TensorRT engine callable with a BF16 external interface."""

    def __init__(self, engine_path: str | Path) -> None:
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError("TensorRT must be installed in the selected isolated environment") from error
        self.trt = trt
        self.path = Path(engine_path)
        self.metadata = json.loads(self.path.with_suffix(".json").read_text(encoding="utf-8"))
        if self.metadata.get("engine_sha256") != sha256(self.path):
            raise ValueError("Decoder TensorRT engine checksum mismatch")
        if self.metadata.get("tensorrt") != trt.__version__:
            raise ValueError("Decoder TensorRT engine version mismatch; rebuild in this environment")
        if self.metadata.get("gpu") != torch.cuda.get_device_name():
            raise ValueError("Decoder TensorRT engine GPU mismatch; rebuild on this target")
        layer_path = self.path.parent / self.metadata["layer_information"]
        if (sha256(layer_path) != self.metadata.get("layer_information_sha256")
                or not convolution_int8_evidence(layer_path.read_text())):
            raise ValueError("Decoder engine lacks verified convolution precision evidence")
        self.input_shape = tuple(self.metadata["input_shape"])
        self.output_shape = tuple(self.metadata["output_shape"])
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT decoder engine failed to deserialize")
        if self.engine.num_io_tensors != 2:
            raise ValueError("Decoder TensorRT engine must expose exactly input and output")
        expected_types = {"input": trt.float32, "output": trt.float32}
        expected_shapes = {"input": self.input_shape, "output": self.output_shape}
        for name in ("input", "output"):
            expected_mode = trt.TensorIOMode.INPUT if name == "input" else trt.TensorIOMode.OUTPUT
            if (
                tuple(self.engine.get_tensor_shape(name)) != expected_shapes[name]
                or self.engine.get_tensor_dtype(name) != expected_types[name]
                or self.engine.get_tensor_mode(name) != expected_mode
            ):
                raise ValueError(f"Decoder TensorRT binding mismatch for {name}")
        self.context = self.engine.create_execution_context_without_device_memory()
        if self.context is None:
            raise RuntimeError("TensorRT decoder context creation failed")
        self.workspace = None
        self.stream = torch.cuda.Stream(device=torch.cuda.current_device())

    def _workspace(self, device: torch.device) -> torch.Tensor:
        required = max(1, self.engine.device_memory_size)
        if self.workspace is None or self.workspace.device != device or self.workspace.numel() < required:
            self.workspace = torch.empty(required, device=device, dtype=torch.uint8)
            self.context.device_memory = self.workspace.data_ptr()
        return self.workspace

    def __call__(self, prepared_input: torch.Tensor) -> torch.Tensor:
        if not prepared_input.is_cuda or tuple(prepared_input.shape) != self.input_shape or prepared_input.dtype != torch.bfloat16:
            raise ValueError("Decoder TensorRT input shape/device mismatch")
        source = prepared_input.float().contiguous()
        output = torch.empty(self.output_shape, device=prepared_input.device, dtype=torch.float32)
        self._workspace(prepared_input.device)
        caller_stream = torch.cuda.current_stream(prepared_input.device)
        self.stream.wait_stream(caller_stream)
        with torch.cuda.stream(self.stream):
            for name, tensor in (("input", source), ("output", output)):
                if not self.context.set_tensor_address(name, tensor.data_ptr()):
                    raise RuntimeError(f"TensorRT binding failed for {name}")
            if not self.context.execute_async_v3(self.stream.cuda_stream):
                raise RuntimeError("TensorRT decoder execution failed")
        caller_stream.wait_stream(self.stream)
        return output.to(prepared_input.dtype)


def captured_prepared_inputs(captures: Path, target_path: str, shape: tuple[int, ...], *, device="cuda"):
    manifest = json.loads(captures.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Decoder engine build requires a complete capture manifest")
    for item in manifest.get("raw_tensors", []):
        if item.get("module") == target_path and item.get("kind") == "causal_conv3d_prepared_input":
            path = captures.parent / item["path"]
            if sha256(path) != item["sha256"]:
                raise ValueError(f"Captured tensor hash mismatch: {path}")
            value = torch.load(path, map_location="cpu", weights_only=True)
            if tuple(value.shape) == shape:
                yield value.to(device=device, dtype=torch.bfloat16)


def _captured_prepared_input(captures: Path, target_path: str, shape: tuple[int, ...]) -> torch.Tensor:
    value = next(captured_prepared_inputs(captures, target_path, shape), None)
    if value is None:
        raise ValueError(f"No captured prepared CausalConv3d input is available for {target_path}/{shape}")
    return value


def calibrate_output_scale(module, inputs) -> dict[str, Any]:
    """Streaming max-abs over every registered input of an engine signature."""
    maximum, count = 0.0, 0
    with torch.inference_mode():
        for value in inputs:
            output = module._conv_forward(value, module.weight, module.bias)
            if not bool(output.isfinite().all()):
                raise ValueError("Nonfinite output in decoder calibration")
            maximum = max(maximum, float(output.float().abs().max()))
            count += 1
    if count < 2:
        raise ValueError("Output calibration requires at least two samples per signature")
    return {"algorithm": "max_abs", "capture_count": count, "maximum": maximum,
            "scale": max(maximum / 127.0, 1e-12), "quant_max": 127}


def calibrated_input_scale(calibration):
    scale = float(calibration.get("scale", 0))
    if not np.isfinite(scale) or scale <= 0 or calibration.get("quant_max") != 127:
        raise DecoderPlanError("INT8 export requires a finite fitted scale and quant_max=127")
    if calibration.get("scale_granularity") != "per_tensor":
        raise DecoderPlanError("Only per-tensor activation calibration is supported")
    return scale


def convolution_int8_evidence(layers):
    """Require INT8 convolution input formats, never names or reformat layers."""
    data = json.loads(layers)
    records = data.get("Layers", []) if isinstance(data, dict) else data
    conv = [x for x in records if "convolution" in str(x.get("LayerType", "")).lower()]
    return bool(conv) and all(
        len(x.get("Inputs", [])) > 0 and all(
            "int8" in str(i.get("Format/Datatype", "")).lower()
            for i in x["Inputs"]
        ) for x in conv
    )


def _onnx_model(
    module,
    input_tensor: torch.Tensor,
    precision: str,
    output: Path,
    calibration: dict,
    *,
    output_scale: float,
) -> dict[str, Any]:
    try:
        import onnx
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as error:
        raise RuntimeError("onnx is required in the selected TensorRT environment") from error
    if precision != "int8":
        raise RuntimeError(
            "Explicit FP8 Wan Conv3d export has not passed this environment's parser/tactic gate; refusing a fallback."
        )
    if input_tensor.ndim not in (4, 5):
        raise ValueError(f"Only Conv2d/Conv3d inputs are supported, got {input_tensor.shape}")
    weight = module.weight.detach().float().cpu().numpy()
    bias = None if module.bias is None else module.bias.detach().float().cpu().numpy()
    input_scale = calibrated_input_scale(calibration)
    if not np.isfinite(output_scale) or output_scale <= 0:
        raise DecoderPlanError("INT8 export requires a finite positive output scale")
    weight_scale = np.maximum(np.max(np.abs(weight), axis=tuple(range(1, weight.ndim))), 1e-12) / 127.0
    input_shape = list(input_tensor.shape)
    output_shape = list(module._conv_forward(input_tensor, module.weight, module.bias).shape)
    initializers = [
        numpy_helper.from_array(weight, "weight"),
        numpy_helper.from_array(np.asarray(input_scale, dtype=np.float32), "input_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), "input_zero"),
        numpy_helper.from_array(weight_scale.astype(np.float32), "weight_scale"),
        numpy_helper.from_array(np.zeros(weight_scale.shape, dtype=np.int8), "weight_zero"),
        numpy_helper.from_array(np.asarray(output_scale, dtype=np.float32), "output_scale"),
        numpy_helper.from_array(np.asarray(0, dtype=np.int8), "output_zero"),
    ]
    if bias is not None:
        initializers.append(numpy_helper.from_array(bias, "bias"))
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "input_scale", "input_zero"], ["input_q"]),
        helper.make_node("DequantizeLinear", ["input_q", "input_scale", "input_zero"], ["input_dq"]),
        helper.make_node("QuantizeLinear", ["weight", "weight_scale", "weight_zero"], ["weight_q"], axis=0),
        helper.make_node("DequantizeLinear", ["weight_q", "weight_scale", "weight_zero"], ["weight_dq"], axis=0),
        helper.make_node(
            "Conv",
            ["input_dq", "weight_dq"] + (["bias"] if bias is not None else []),
            ["conv_output"],
            kernel_shape=list(module.kernel_size),
            strides=list(module.stride),
            dilations=list(module.dilation),
            pads=list(module.padding) + list(module.padding),
            group=module.groups,
        ),
        # A trailing Q/DQ makes the convolution itself an INT8 fusion candidate.
        # Input Q/DQ alone permits TensorRT to select a floating-point Conv.
        helper.make_node("QuantizeLinear", ["conv_output", "output_scale", "output_zero"], ["output_q"]),
        helper.make_node("DequantizeLinear", ["output_q", "output_scale", "output_zero"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "soulx_v2_explicit_int8_conv",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 19)])
    onnx.checker.check_model(model)
    onnx.save(model, output)
    return {
        "input_shape": input_shape,
        "output_shape": output_shape,
        "input_scale": input_scale,
        "output_scale": output_scale,
        "weight_scale_min": float(weight_scale.min()),
        "weight_scale_max": float(weight_scale.max()),
    }


def _build_engine(onnx_path: Path, engine_path: Path, workspace_mib: int) -> tuple[Any, str]:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "; ".join(str(parser.get_error(index)) for index in range(parser.num_errors))
        raise RuntimeError(f"TensorRT ONNX parsing failed: {errors}")
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 2**20)
    config.builder_optimization_level = 3
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT explicit-Q/DQ decoder engine build returned no engine")
    engine_path.write_bytes(bytes(serialized))
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(bytes(serialized))
    if engine is None:
        raise RuntimeError("TensorRT could not reload the just-built decoder engine")
    inspector = engine.create_engine_inspector()
    layer_information = inspector.get_engine_information(trt.LayerInformationFormat.JSON)
    return engine, layer_information


def _benchmark(function, value: torch.Tensor, repeats: int = 20) -> list[float]:
    values = []
    with torch.inference_mode():
        for _ in range(4):
            function(value)
        torch.cuda.synchronize()
        for _ in range(5):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(repeats):
                function(value)
            end.record()
            end.synchronize()
            values.append(begin.elapsed_time(end) / repeats)
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--precision", choices=("int8", "fp8"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workspace-mib", type=int, default=512)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    args = parser.parse_args()
    if args.workspace_mib < 1:
        parser.error("workspace-mib must be positive")
    output = ensure_new_directory(args.output)
    result: dict[str, Any] = {
        "status": "starting",
        "execution": "fresh GPU exact-shape TensorRT convolution engine build and diagnostic",
        "environment": environment_manifest(),
        "plan": relative_path(args.plan),
        "captures": relative_path(args.captures),
        "precision": args.precision,
        "workspace_mib": args.workspace_mib,
        "engines": [],
    }

    def save() -> None:
        atomic_write_json(output / "results.json", result)

    save()
    lease = None
    vae = None
    stage = "validate"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for TensorRT decoder engine builds")
        try:
            import tensorrt as trt
        except ImportError as error:
            raise RuntimeError("TensorRT is unavailable in this environment; use the pinned isolated environment") from error
        plan = load_decoder_plan(args.plan, require_engines=False)
        expected_scheme = f"{args.precision}_conservative"
        if plan.scheme != expected_scheme:
            raise DecoderPlanError(f"Plan scheme {plan.scheme} does not match requested {expected_scheme}")
        if args.precision != "int8":
            raise RuntimeError("FP8 decoder engine build is blocked until explicit FP8 Conv3d parser support is verified")
        lease = acquire_gpu_lease(args.gpu_lock)
        from flash_head.wan.modules import WanVAE

        vae = WanVAE(
            vae_path=str(ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"),
            dtype=torch.bfloat16, device="cuda", parallel=False,
        )
        vae.model.eval().requires_grad_(False)
        engine_records = {}
        for target in plan.targets:
            stage = f"build:{target.path}"
            if target.kind != "causal_conv3d":
                raise DecoderPlanError("This builder currently accepts only captured CausalConv3d targets")
            module = decoder_module_for_path(vae, target.path)
            engine_records[target.path] = {}
            for shape in target.observed_shapes:
                input_tensor = _captured_prepared_input(args.captures, target.path, shape)
                onnx_path = output / f"{target.path.replace('.', '_')}-{shape_key(shape)}.onnx"
                engine_path = onnx_path.with_suffix(".engine")
                output_calibration = calibrate_output_scale(
                    module, captured_prepared_inputs(args.captures, target.path, shape)
                )
                quantization = _onnx_model(
                    module, input_tensor, args.precision, onnx_path, dict(target.calibration),
                    output_scale=output_calibration["scale"],
                )
                quantization["output_calibration"] = output_calibration
                started = time.perf_counter()
                engine, layers = _build_engine(onnx_path, engine_path, args.workspace_mib)
                build_s = time.perf_counter() - started
                layer_path = engine_path.with_suffix(".layers.json")
                layer_path.write_text(layers + "\n", encoding="utf-8")
                layer_precision_evidence = convolution_int8_evidence(layers)
                if not layer_precision_evidence:
                    result["rejected_engine"] = {
                        "target": target.path, "input_shape": list(shape),
                        "engine": engine_path.name, "layer_information": layer_path.name,
                        "layer_information_sha256": sha256(layer_path),
                        "calibration": dict(target.calibration),
                        "reason": "INT8 convolution compute was not established",
                    }
                    save()
                    raise RuntimeError("Inspector does not prove INT8 convolution dispatch; engine rejected")
                metadata = {
                    "kind": "wan_prepared_causal_conv3d",
                    "target": target.path,
                    "precision": args.precision,
                    "engine_sha256": sha256(engine_path),
                    "onnx_sha256": sha256(onnx_path),
                    "capture_sha256": sha256(args.captures),
                    "weights_sha256": sha256(ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"),
                    "input_shape": quantization["input_shape"],
                    "output_shape": quantization["output_shape"],
                    "external_input_dtype": "bfloat16",
                    "engine_io_dtype": "float32",
                    "tensorrt": trt.__version__,
                    "gpu": torch.cuda.get_device_name(),
                    "torch": torch.__version__,
                    "cuda_runtime": torch.version.cuda,
                    "workspace_mib": args.workspace_mib,
                    "build_s": build_s,
                    "quantization": quantization,
                    "layer_information": layer_path.name,
                    "layer_information_sha256": sha256(layer_path),
                    "quantization_dispatch_verified": layer_precision_evidence,
                }
                metadata_path = engine_path.with_suffix(".json")
                atomic_write_json(metadata_path, metadata)
                runtime = TensorRTPreparedConv(engine_path)
                with torch.inference_mode():
                    expected = module._conv_forward(input_tensor, module.weight, module.bias)
                    actual = runtime(input_tensor)
                if not actual.isfinite().all():
                    raise RuntimeError("Decoder engine produced nonfinite output")
                delta = actual.float() - expected.float()
                record = {
                    "target": target.path,
                    "engine": engine_path.name,
                    "metadata": metadata_path.name,
                    "build_s": build_s,
                    "finite": bool(actual.isfinite().all()),
                    "max_abs": float(delta.abs().max()),
                    "mean_abs": float(delta.abs().mean()),
                    "relative_l2": float(delta.norm() / expected.float().norm().clamp_min(1e-12)),
                    "pytorch_ms": _benchmark(lambda value: module._conv_forward(value, module.weight, module.bias), input_tensor),
                    "engine_ms": _benchmark(runtime, input_tensor),
                    "engine_workspace_mib": engine.device_memory_size / 2**20,
                    "quantization_dispatch_verified": layer_precision_evidence,
                }
                record["pytorch_summary_ms"] = summarize(record["pytorch_ms"])
                record["engine_summary_ms"] = summarize(record["engine_ms"])
                result["engines"].append(record)
                engine_records[target.path][shape_key(shape)] = {
                    "path": relative_path(engine_path), "sha256": metadata["engine_sha256"], "precision": args.precision,
                }
                save()
                del runtime, actual, expected, delta, engine, input_tensor
                torch.cuda.empty_cache()
        resolved = plan.to_dict()
        resolved["engines"] = engine_records
        atomic_write_json(output / "engine-plan.json", resolved)
        result.update(status="complete", engine_plan="engine-plan.json")
        save()
    except Exception as error:
        result.update(status="failed", failure=failure_record(stage, error))
        save()
        raise
    finally:
        if lease is not None:
            lease.close()
        if vae is not None:
            del vae


if __name__ == "__main__":
    main()
