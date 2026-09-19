"""Collect a separate trace or timing diagnostic for one v2 policy run."""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
import csv
from pathlib import Path
from typing import Any

import torch

from benchmarks.pro_quantization_v2_20260918.common import DEFAULT_FIXTURES, atomic_write_json


def device_metric(event: Any, name: str) -> tuple[float | int | None, str | None]:
    """Aggregate events in Torch 2.7 have device fields, not CUDA aliases."""
    for field in (name, name.replace("device", "cuda")):
        value = getattr(event, field, None)
        if value is not None:
            return value, field
    return None, None


def aggregate_row(event: Any) -> dict[str, Any]:
    row = {
        "name": event.key, "calls": event.count,
        "device_type": str(getattr(event, "device_type", "unavailable")),
        "self_cpu_time_us": event.self_cpu_time_total,
        "cpu_time_us": event.cpu_time_total,
    }
    for label, field in (
        ("self_cuda_time_us", "self_device_time_total"),
        ("cuda_time_us", "device_time_total"),
        ("self_cuda_memory_bytes", "self_device_memory_usage"),
        ("cuda_memory_bytes", "device_memory_usage"),
    ):
        row[label], row[label + "_source"] = device_metric(event, field)
    return row


class ProfileController:
    """Own one profiler lifecycle and label only the requested pipeline component."""

    def __init__(self, output: Path, component: str, mode: str) -> None:
        if mode == "inventory" and component != "decoder":
            raise ValueError("Eager module inventory is supported only for the decoder")
        self.output = output
        self.component = component
        self.mode = mode
        self.profiler = None
        self.original = None
        self.owner = None
        self.attribute = None
        self.started = False
        self.finalized = False
        self.result: dict[str, Any] | None = None
        self.handles: list[Any] = []
        self.pending: dict[int, list[tuple[Any, dict[str, Any]]]] = {}
        self.module_events: list[tuple[Any, Any, dict[str, Any]]] = []

    def attach(self, pipeline) -> None:
        if self.mode not in ("trace", "inventory") or self.component == "pipeline":
            return
        self.owner, self.attribute = (
            (pipeline.model, "forward") if self.component == "dit" else (pipeline.vae, "decode")
        )
        self.original = getattr(self.owner, self.attribute)

        def profiled(*args, **kwargs):
            with torch.profiler.record_function(f"pro_v2/{self.component}"):
                return self.original(*args, **kwargs)

        setattr(self.owner, self.attribute, profiled)
        if self.component == "decoder" and self.mode == "inventory":
            model = getattr(pipeline.vae, "model", pipeline.vae)
            prefix = "model.decoder" if model is not pipeline.vae else "decoder"
            for name, module in model.decoder.named_modules():
                if not isinstance(module, (torch.nn.Conv2d, torch.nn.Conv3d)):
                    continue
                path = f"{prefix}.{name}"

                def pre_hook(layer, inputs, *, path=path):
                    begin = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    metadata = {
                        "path": path,
                        "class": type(layer).__name__,
                        "kernel_size": list(layer.kernel_size),
                        "stride": list(layer.stride),
                        "padding": list(layer.padding),
                        "dilation": list(layer.dilation),
                        "groups": layer.groups,
                        "input_shape": list(inputs[0].shape),
                        "input_stage": "before_causal_padding" if hasattr(layer, "_padding") else "module_input",
                    }
                    self.pending.setdefault(id(layer), []).append((begin, metadata))

                def post_hook(layer, _inputs, output):
                    begin, metadata = self.pending[id(layer)].pop(0)
                    end = torch.cuda.Event(enable_timing=True)
                    end.record()
                    metadata["output_shape"] = list(output.shape)
                    self.module_events.append((begin, end, metadata))

                self.handles.append(module.register_forward_pre_hook(pre_hook))
                self.handles.append(module.register_forward_hook(post_hook))

    def clear_events(self) -> None:
        self.pending.clear()
        self.module_events.clear()

    def start(self) -> None:
        if self.mode not in ("trace", "inventory"):
            return
        self.profiler = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )
        self.profiler.start()
        self.started = True

    def step(self) -> None:
        if self.profiler is not None:
            self.profiler.step()

    @staticmethod
    def _value(event: Any, name: str) -> float | int:
        return getattr(event, name, 0) or 0

    def finalize(self, *, status: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.finalized:
            return self.result or {}
        if self.profiler is not None:
            self.profiler.stop()
        if self.owner is not None and self.original is not None:
            setattr(self.owner, self.attribute, self.original)
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        result: dict[str, Any] = {
            "status": status,
            "component": self.component,
            "mode": self.mode,
            "execution": "GPU profiling diagnostic; not a throughput measurement",
            "decoder_execution": "eager module inventory" if self.mode == "inventory" else "policy-selected backend preserved",
        }
        if self.profiler is not None:
            trace = self.output / "trace.json"
            self.profiler.export_chrome_trace(str(trace))
            rows = [aggregate_row(event) for event in self.profiler.key_averages()]
            aggregate = self.output / "aggregated.csv"
            with aggregate.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["name"])
                writer.writeheader()
                writer.writerows(rows)
            result.update({"trace": trace.name, "aggregated": aggregate.name, "rows": len(rows)})
            result["gpu_timing_qualified"] = any((row["self_cuda_time_us"] or 0) > 0 for row in rows)
            if not result["gpu_timing_qualified"]:
                result["status"] = "failed"
                result["qualification_error"] = "No nonzero aggregate device duration; inspect CUDA trace/CUPTI availability"
        if self.component == "decoder" and self.module_events:
            torch.cuda.synchronize()
            inventory: dict[str, dict[str, Any]] = {}
            for begin, end, metadata in self.module_events:
                entry = inventory.setdefault(metadata["path"], {**metadata, "calls": 0, "inclusive_cuda_ms": 0.0, "observed_inputs": [], "observed_outputs": []})
                entry["calls"] += 1
                entry["inclusive_cuda_ms"] += begin.elapsed_time(end)
                if metadata["input_shape"] not in entry["observed_inputs"]:
                    entry["observed_inputs"].append(metadata["input_shape"])
                if metadata["output_shape"] not in entry["observed_outputs"]:
                    entry["observed_outputs"].append(metadata["output_shape"])
            rows = sorted(inventory.values(), key=lambda item: item["inclusive_cuda_ms"], reverse=True)
            inventory_path = self.output / "decoder-inventory.json"
            atomic_write_json(inventory_path, {"modules": rows, "timing_scope": "module inclusive CUDA event duration; do not sum nested totals"})
            result["decoder_inventory"] = inventory_path.name
        elif self.mode == "timing":
            result["note"] = "Timing mode uses the run manifest's CUDA-event stage and chunk distributions."
        elif self.mode == "trace":
            result["note"] = "CPU operator rows and CUDA kernel rows overlap; sum only CUDA rows for exclusive device totals."
        if extra:
            result.update(extra)
        atomic_write_json(self.output / "profile.json", result)
        self.finalized = True
        self.result = result
        if status == "complete" and result.get("gpu_timing_qualified") is False:
            raise RuntimeError(result["qualification_error"])
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--component", choices=("dit", "decoder", "pipeline"), required=True)
    parser.add_argument("--mode", choices=("trace", "timing", "inventory"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--gpu-lock", type=Path)
    args = parser.parse_args()
    if args.mode == "inventory" and args.component != "decoder":
        parser.error("inventory mode requires component=decoder")
    from benchmarks.pro_quantization_v2_20260918.common import DEFAULT_GPU_LOCK
    from benchmarks.pro_quantization_v2_20260918.run import run_experiment

    args.repeats = 1
    args.save_raw = False
    args.capture_manifest = None
    args.capture_max_raw_mib = 0
    args.profile_component = args.component
    args.profile_mode = args.mode
    args.gpu_lock = args.gpu_lock or DEFAULT_GPU_LOCK
    run_experiment(args)


if __name__ == "__main__":
    main()
