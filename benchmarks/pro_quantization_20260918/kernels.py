"""Real-input GPU FFN speed/error/dispatch/reload checks, not video acceptance."""
import argparse
import copy
import io
import json
from pathlib import Path
import time

import torch
from safetensors import safe_open

from benchmarks.pro_lite_teeth_20260917.run_variant import gpu_snapshot, process_snapshot
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.pro_quantization import Float8Linear, Int8ComputeLinear


def bench(module, x, n=20):
    with torch.inference_mode():
        for _ in range(4):
            module(x)
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(n):
            module(x)
        end.record(); end.synchronize()
    return begin.elapsed_time(end) / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    lease = acquire_gpu_lease()
    torch.set_num_threads(4)
    result = dict(execution="fresh GPU kernel benchmarks on captured PRO activations",
                  gpu=gpu_snapshot(), physical_vram_class="12 GB", torch=torch.__version__,
                  cuda=torch.version.cuda, co_residents=process_snapshot(), rows=[])
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    weights = "models/SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors"
    for index in (0, 14, 29):
        x = torch.load(args.captures / f"ffn-{index:02d}.pt", weights_only=True).cuda()
        layer = torch.nn.Sequential(torch.nn.Linear(1536,8960),
            torch.nn.GELU(approximate="tanh"), torch.nn.Linear(8960,1536)).eval().requires_grad_(False)
        with safe_open(weights, framework="pt", device="cpu") as f:
            layer.load_state_dict({k:f.get_tensor(f"blocks.{index}.ffn.{k}") for k in layer.state_dict()})
        layer = layer.to(device="cuda", dtype=torch.bfloat16)
        with torch.inference_mode():
            expected = layer(x)
        for scheme in ("bf16", "fp8", "int8"):
            candidate = copy.deepcopy(layer)
            if scheme != "bf16":
                cls = Float8Linear if scheme == "fp8" else Int8ComputeLinear
                candidate[0], candidate[2] = cls(candidate[0]), cls(candidate[2])
            for compiled in (False, True):
                row = dict(block=index, shape=list(x.shape), scheme=scheme, compiled=compiled)
                try:
                    fn = torch.compile(candidate, fullgraph=True, dynamic=False) if compiled else candidate
                    start = time.perf_counter()
                    with torch.inference_mode():
                        actual = fn(x)
                        torch.cuda.synchronize()
                        delta = actual.float() - expected.float()
                        row.update(finite=bool(actual.isfinite().all()),
                                   relative_l2=float(delta.norm() / expected.float().norm()),
                                   max_abs=float(delta.abs().max()),
                                   warmup_s=time.perf_counter()-start)
                        # Ensure runtime activation scaling is not frozen on compilation.
                        changed = fn(x * 0.37)
                        changed_reference = candidate(x * 0.37)
                        row["new_input_max_error_vs_eager"] = float((changed-changed_reference).abs().max())
                        row["new_input_changes_output"] = not torch.equal(changed, actual)
                        zero = fn(torch.zeros_like(x))
                        row["zero_input_finite"] = bool(zero.isfinite().all())
                    row["milliseconds"] = [bench(fn,x) for _ in range(3)]
                    if index == 0 and not compiled:
                        buffer = io.BytesIO()
                        torch.save(candidate.state_dict(), buffer); buffer.seek(0)
                        clone = copy.deepcopy(candidate)
                        clone.load_state_dict(torch.load(buffer, weights_only=True))
                        with torch.inference_mode():
                            row["state_dict_reload_exact"] = torch.equal(clone(x), actual)
                        del clone, buffer
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                                 torch.profiler.ProfilerActivity.CUDA]) as prof:
                            with torch.inference_mode():
                                fn(x)
                            torch.cuda.synchronize()
                        row["gemm_dispatch"] = sorted({e.name for e in prof.events() if any(
                            term in e.name.lower() for term in ("scaled_mm", "_int_mm", "gemm", "cutlass"))})
                    row["status"] = "complete"
                except Exception as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                result["rows"].append(row); save()
                print(json.dumps(row), flush=True)
            del candidate
        del layer, x, expected
        torch.cuda.empty_cache()
    result["status"] = "complete"; save()


if __name__ == "__main__":
    main()
