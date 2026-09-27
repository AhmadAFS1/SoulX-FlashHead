#!/usr/bin/env python3
"""Decoder speed at 576x320 (GPU; holds the SoulX lease). Fresh local GPU inference, RTX 4070 SUPER.

Each case is timed end to end at the pipeline's decode contract: in (16, T, 72, 40) bf16
normalised latents, out (1, 3, F, 576, 320) bf16 in [-1, 1] (so tiny-decoder cases include the
latent conversion and the NTCHW->NCTHW / [0,1]->[-1,1] output conversion).
  cold9: a full 9-latent window from an empty decoder state -> 33 frames
  warm7: the overlap-skip steady state: 7 latents against a carried decoder state -> 28 frames
CUDA events around each call, >= 3 warm-up calls, median (and min/max) of >= 10 timed calls;
peak allocated memory = torch.cuda.max_memory_allocated() during one call (TensorRT cases get
their activation memory from a torch-allocated buffer, so it is counted the same way).
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bakeoff_common as bc  # noqa: E402

TRT_DIR = bc.BK / "_trt"


def bench(fn, warmup=3, iters=12):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    frames = int(out.shape[2]) if hasattr(out, "shape") else None
    del out
    return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times), "n": iters, "warmup": warmup,
            "peak_alloc_mib": peak / 2**20, "peak_over_resident_mib": (peak - base) / 2**20, "frames_out": frames}


# ------------------------------------------------------------------------------ TensorRT
class TRTCore:
    """A TensorRT engine of TAEHVDecodeCore at a fixed T (fp16, strongly typed from the ONNX)."""

    def __init__(self, engine_path, logger):
        import tensorrt as trt
        self.trt = trt
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context_without_device_memory()
        size = self.engine.device_memory_size_v2 if hasattr(self.engine, "device_memory_size_v2") else self.engine.device_memory_size
        self.workspace = torch.empty(max(int(size), 1), dtype=torch.uint8, device="cuda")
        if hasattr(self.ctx, "set_device_memory"):
            self.ctx.set_device_memory(self.workspace.data_ptr(), int(size))
        else:
            self.ctx.device_memory = self.workspace.data_ptr()
        self.names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.out_buf = {n: torch.empty(tuple(self.engine.get_tensor_shape(n)), dtype=torch.float16, device="cuda") for n in self.outputs}
        self.workspace_mib = int(size) / 2**20

    def __call__(self, x, *states):
        vals = [x, *states]
        for n, v in zip(self.inputs, vals):
            assert v.dtype == torch.float16 and v.is_contiguous()
            self.ctx.set_tensor_address(n, v.data_ptr())
        for n in self.outputs:
            self.ctx.set_tensor_address(n, self.out_buf[n].data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return tuple(self.out_buf[n] for n in self.outputs)


def build_trt(core, t, h, w, tag, logger, workspace_mib=3072):
    import tensorrt as trt
    TRT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = TRT_DIR / f"taehv_core_{tag}_T{t}.onnx"
    eng_path = onnx_path.with_suffix(".engine")
    x = torch.zeros(t, 16, h, w, device="cuda", dtype=torch.float16)
    states = core.zero_states(t, h, w, "cuda", torch.float16)
    names_in = ["x"] + [f"s{i}" for i in range(len(states))]
    names_out = ["y"] + [f"n{i}" for i in range(len(states))]
    t0 = time.time()
    with torch.inference_mode():
        torch.onnx.export(core, (x, *states), str(onnx_path), input_names=names_in, output_names=names_out,
                          opset_version=18, dynamo=False, do_constant_folding=True)
    t_export = time.time() - t0
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 2**20)
    config.builder_optimization_level = 3
    t1 = time.time()
    ser = builder.build_serialized_network(network, config)
    if ser is None:
        raise RuntimeError("TensorRT build returned no engine")
    eng_path.write_bytes(bytes(ser))
    return {"onnx": str(onnx_path.relative_to(bc.ROOT)), "engine": str(eng_path.relative_to(bc.ROOT)),
            "onnx_bytes": onnx_path.stat().st_size, "engine_bytes": eng_path.stat().st_size,
            "export_s": t_export, "build_s": time.time() - t1, "opset": 18, "tensorrt": trt.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=bc.BK / "speed.json")
    ap.add_argument("--skip", default="", help="comma list of groups to skip: tae,compile,trt,lightvae,shipping,stock")
    ap.add_argument("--iters", type=int, default=12)
    args = ap.parse_args()
    from soulx_rtc.gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease(None)
    try:
        run(args)
    finally:
        lease.close()


def run(args):
    skip = set(filter(None, args.skip.split(",")))
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    res = {"execution": "fresh local GPU inference on RTX 4070 SUPER", "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "torch": torch.__version__, "cudnn_benchmark": True, "gpu_at_start": bc.gpu_snapshot(), "cases": {}, "notes": []}
    prev = json.loads(args.out.read_text()) if args.out.exists() else None
    if prev:
        res["cases"] = prev.get("cases", {})  # merge with an earlier partial run
        res["trt"] = prev.get("trt")
    torch.backends.cudnn.benchmark = True
    items = bc.load_latents()
    w, d = items[4] if len(items) > 4 else items[0]
    res["latent_window"] = w["file"]
    z9 = d["latent"].cuda()
    z7 = z9[:, 2:].contiguous()
    H, W = 72, 40

    def record(name, r):
        r["gpu_after"] = bc.gpu_snapshot()["compute_apps"]
        res["cases"][name] = r
        print(f"{name:55s} {r['median_ms']:9.2f} ms  (min {r['min_ms']:.2f})  peak+{r['peak_over_resident_mib']:.0f} MiB  frames {r['frames_out']}", flush=True)
        args.out.write_text(json.dumps(res, indent=1))

    with torch.inference_mode():
        # ------------------------------------------------ TAEHV family (taew2_1 == lighttaew2_1 architecture)
        if "tae" not in skip or "compile" not in skip or "trt" not in skip:
            ref_eager = {}
            for dt_name, dt in (("fp16", torch.float16), ("bf16", torch.bfloat16)):
                tae = bc.load_taehv("taew2_1", dtype=dt)
                for layout in ("nchw", "channels_last"):
                    mf = torch.channels_last if layout == "channels_last" else torch.contiguous_format
                    if layout == "channels_last":
                        tae = tae.to(memory_format=torch.channels_last)
                    ad = bc.TAEHVAdapter(tae, "norm", dt, memory_format=mf)
                    _, st = ad.raw(z9)
                    st = [s.contiguous(memory_format=mf) for s in st]
                    if "tae" not in skip:
                        record(f"taehv {dt_name} eager {layout} cold9", bench(lambda: ad.decode_window(z9), iters=args.iters))
                        record(f"taehv {dt_name} eager {layout} warm7", bench(lambda: ad.to_wan(ad.raw(z7, st)[0]), iters=args.iters))
                    if layout == "nchw":
                        ref_eager[dt_name] = ad.decode_window(z9).clone()
                    if "compile" not in skip:
                        core_c = torch.compile(bc.TAEHVDecodeCore(tae), dynamic=False)
                        adc = bc.TAEHVAdapter(tae, "norm", dt, core=core_c, memory_format=mf)
                        t0 = time.time(); o = adc.decode_window(z9); torch.cuda.synchronize(); c9 = time.time() - t0
                        diff = (o.float() - ref_eager[dt_name].float()).abs().max().item()
                        r = bench(lambda: adc.decode_window(z9), iters=args.iters); r["first_call_s"] = c9; r["maxabs_vs_eager"] = diff
                        record(f"taehv {dt_name} torch.compile {layout} cold9", r)
                        t0 = time.time(); adc.to_wan(adc.raw(z7, st)[0]); torch.cuda.synchronize(); c7 = time.time() - t0
                        r = bench(lambda: adc.to_wan(adc.raw(z7, st)[0]), iters=args.iters); r["first_call_s"] = c7
                        record(f"taehv {dt_name} torch.compile {layout} warm7", r)
                        del core_c, adc
                        torch._dynamo.reset()
                del tae
                torch.cuda.empty_cache()

            # ------------------------------------------------ TensorRT FP16
            if "trt" not in skip:
                try:
                    import tensorrt as trt
                    logger = trt.Logger(trt.Logger.WARNING)
                    tae = bc.load_taehv("taew2_1", dtype=torch.float16)
                    core = bc.TAEHVDecodeCore(tae).eval()
                    info = {}
                    for t in (9, 7):
                        info[f"T{t}"] = build_trt(core, t, H, W, "taew2_1_fp16", logger)
                        print("built", info[f"T{t}"], flush=True)
                    res["trt"] = info
                    ad = bc.TAEHVAdapter(tae, "norm", torch.float16)
                    for t, zin, label in ((9, z9, "cold9"), (7, z7, "warm7")):
                        eng = TRTCore(bc.ROOT / info[f"T{t}"]["engine"], logger)
                        adt = bc.TAEHVAdapter(tae, "norm", torch.float16, core=eng)
                        if label == "cold9":
                            o = adt.decode_window(z9)
                            info["T9"]["maxabs_vs_eager_fp16"] = (o.float() - ref_eager["fp16"].float()).abs().max().item()
                            info["T9"]["psnr_u8_vs_eager_fp16"] = bc.psnr(bc.to_u8(o), bc.to_u8(ref_eager["fp16"]))
                            fn = lambda: adt.decode_window(z9)
                        else:
                            _, st = ad.raw(z9)
                            st = [s.contiguous() for s in st]
                            e_ref = ad.to_wan(ad.raw(z7, st)[0])
                            o = adt.to_wan(adt.raw(z7, st)[0])
                            info["T7"]["maxabs_vs_eager_fp16"] = (o.float() - e_ref.float()).abs().max().item()
                            info["T7"]["psnr_u8_vs_eager_fp16"] = bc.psnr(bc.to_u8(o), bc.to_u8(e_ref))
                            fn = lambda: adt.to_wan(adt.raw(z7, st)[0])
                        r = bench(fn, iters=args.iters)
                        r["trt_workspace_mib"] = eng.workspace_mib
                        record(f"taehv fp16 TensorRT {label}", r)
                        del eng, adt
                    res["trt"] = info
                    del tae, core
                except Exception as e:
                    import traceback
                    res["trt_error"] = f"{type(e).__name__}: {e}"
                    res["notes"].append("TensorRT failed: " + traceback.format_exc()[-1500:])
                    print("TRT failed", e, flush=True)
                torch.cuda.empty_cache()

        # ------------------------------------------------ Wan-architecture decoders
        def wan_cases(label, model, scale, compiled: bool):
            cached = torch.compile(model.cached_decode, dynamic=False) if compiled else model.cached_decode
            mode = "torch.compile" if compiled else "eager"

            def cold():
                model.clear_cache()
                out = cached(z9.unsqueeze(0), scale).clamp_(-1, 1)
                return out

            t0 = time.time(); cold(); torch.cuda.synchronize(); first = time.time() - t0
            r = bench(cold, iters=args.iters if not label.startswith("stock") else 10); r["first_call_s"] = first
            record(f"{label} {mode} cold9", r)
            model.clear_cache()
            cached(z9.unsqueeze(0), scale)
            state = model._feat_map

            def warm():
                model._feat_map = state
                model._conv_idx = [0]
                return cached(z7.unsqueeze(0), scale).clamp_(-1, 1)

            t0 = time.time(); warm(); torch.cuda.synchronize(); first = time.time() - t0
            r = bench(warm, iters=args.iters if not label.startswith("stock") else 10); r["first_call_s"] = first
            record(f"{label} {mode} warm7", r)
            model.clear_cache()
            if compiled:
                torch._dynamo.reset()

        if "lightvae" not in skip:
            lv, _ = bc.load_lightvae()
            scale = list(bc.wan_scale("cuda", torch.bfloat16))
            wan_cases("lightvaew2_1 bf16", lv, scale, False)
            wan_cases("lightvaew2_1 bf16", lv, scale, True)
            del lv
            torch.cuda.empty_cache()
        if "shipping" not in skip:
            ship = bc.load_shipping()
            wan_cases("shipping (pruned+ft4) bf16", ship.model, ship.scale, False)
            wan_cases("shipping (pruned+ft4) bf16", ship.model, ship.scale, True)
            del ship
            torch.cuda.empty_cache()
        if "stock" not in skip:
            stock = bc.load_stock()
            wan_cases("stock Wan 2.1 bf16", stock.model, stock.scale, False)
            del stock
            torch.cuda.empty_cache()
    res["gpu_at_end"] = bc.gpu_snapshot()
    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
