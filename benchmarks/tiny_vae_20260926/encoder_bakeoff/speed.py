#!/usr/bin/env python3
"""Motion re-encode speed at 576x320 (GPU; holds the SoulX lease). Fresh local GPU inference, RTX 4070 SUPER.

One timed call = the pipeline's motion-encode contract end to end: in cond_frame
(1, 3, 5, 576, 320) bf16 in [-1, 1] (a real one: window 4 of indian150-a decoded by the stock
decoder, lean-trimmed, colour-corrected, trailing 5), out (16, 2, 72, 40) bf16 DiT-normalised
latents. Tiny-encoder cases include their input conversion (NCTHW -> NTCHW, [-1,1] -> [0,1],
end padding to 8 frames) and output conversion (layout, lighttaew2_1's normalisation, bf16).
CUDA events around each call, 3 warm-up calls (after any compile), median/min/max of 20 timed
calls; peak allocated memory over resident during one call. Run once per cuDNN benchmark
setting (the harness leaves torch.backends.cudnn.benchmark at its default, False).
TensorRT FP16 (10.3, borrowed read-only via decoder_bakeoff/_trt10_path, same as the decoder
bake-off): ONNX opset 18 export of TAEHVEncodeCore at T=8, strongly typed fp16 engine.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_common as ec  # noqa: E402

bc = ec.bc
TRT_DIR = ec.EB / "_trt"


def _load_decoder_speed():
    spec = importlib.util.spec_from_file_location("decoder_speed", bc.BK / "speed.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bench(fn, warmup=3, iters=20):
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
    shape = list(out.shape)
    del out
    return {"median_ms": statistics.median(times), "min_ms": min(times), "max_ms": max(times), "n": iters,
            "warmup": warmup, "peak_over_resident_mib": (peak - base) / 2**20, "out_shape": shape}


def build_trt_encoder(core, t, h, w, tag, logger, workspace_mib=2048):
    import tensorrt as trt
    TRT_DIR.mkdir(parents=True, exist_ok=True)
    onnx_path = TRT_DIR / f"taehv_enc_{tag}_T{t}.onnx"
    eng_path = onnx_path.with_suffix(".engine")
    x = torch.zeros(t, 3, h, w, device="cuda", dtype=torch.float16)
    t0 = time.time()
    with torch.inference_mode():
        torch.onnx.export(core, (x,), str(onnx_path), input_names=["x"], output_names=["z"],
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
    return {"onnx_sha256": bc.sha256(onnx_path), "engine_sha256": bc.sha256(eng_path),
            "onnx_bytes": onnx_path.stat().st_size, "engine_bytes": eng_path.stat().st_size,
            "export_s": t_export, "build_s": time.time() - t1, "opset": 18, "tensorrt": trt.__version__,
            "engine_path": str(eng_path), "onnx_path": str(onnx_path)}


class TRTEncoderCore:
    def __init__(self, eng):
        self.eng = eng

    def __call__(self, x):
        return self.eng(x)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ec.EB / "speed.json")
    ap.add_argument("--cudnn-benchmark", type=int, choices=(0, 1), required=True)
    ap.add_argument("--skip", default="", help="comma list: stock,stockcompile,lightvae,tae,compile,trt")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    from soulx_rtc.gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease(None)
    try:
        with torch.inference_mode():
            run(args)
    finally:
        lease.close()


def run(args):
    skip = set(filter(None, args.skip.split(",")))
    torch.backends.cudnn.benchmark = bool(args.cudnn_benchmark)
    setting = f"cudnn_benchmark={int(args.cudnn_benchmark)}"
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
    res = json.loads(args.out.read_text()) if args.out.exists() else {"settings": {}}
    res.update({"execution": "fresh local GPU inference on RTX 4070 SUPER", "torch": torch.__version__})
    cur = res["settings"].setdefault(setting, {"cases": {}, "notes": []})
    cur["date_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    cur["gpu_at_start"] = bc.gpu_snapshot()

    items = bc.load_latents()
    w, d = items[4]
    stock = bc.load_stock()
    ref_img = ec.load_reference()
    clip = ec.cond_frames(stock.decode(d["latent"].cuda()), ref_img)
    cur["clip"] = {"from_window": w["file"], "shape": list(clip.shape), "dtype": str(clip.dtype)}
    ref = stock.encode(clip).float()
    std = ref.transpose(0, 1).reshape(2, 16, -1).std(-1)  # per slot/channel, this clip

    def nrmse(e):  # vs the eager stock Wan encode of the same clip, mean over slots/channels
        dd = (e.float() - ref).transpose(0, 1).reshape(2, 16, -1)
        return (dd.pow(2).mean(-1).sqrt() / std).mean().item()

    def record(name, r, out=None, ref_out=None):
        r["gpu_after"] = bc.gpu_snapshot()["compute_apps"]
        if out is not None:
            r["nrmse_vs_wan_eager_this_clip"] = nrmse(out)
            if ref_out is not None:
                r["maxabs_vs_eager_same_model"] = (out.float() - ref_out.float()).abs().max().item()
        cur["cases"][name] = r
        print(f"[{setting}] {name:52s} {r['median_ms']:8.2f} ms (min {r['min_ms']:.2f} max {r['max_ms']:.2f}) "
              f"peak+{r['peak_over_resident_mib']:.0f} MiB  {r.get('nrmse_vs_wan_eager_this_clip', float('nan')):.4f}", flush=True)
        args.out.write_text(json.dumps(res, indent=1))

    def first_call(fn):
        t0 = time.time()
        o = fn()
        torch.cuda.synchronize()
        return o, time.time() - t0

    # ------------------------------------------------ stock Wan 2.1 encoder (the shipping motion encoder)
    if "stock" not in skip:
        record("stock Wan 2.1 bf16 eager", bench(lambda: stock.encode(clip), iters=args.iters), stock.encode(clip))
    if "stockcompile" not in skip:
        enc_c = torch.compile(stock.encode, dynamic=False)  # == run.py --compile-vae-encode
        o, fc = first_call(lambda: enc_c(clip))
        r = bench(lambda: enc_c(clip), iters=args.iters)
        r["first_call_s"] = fc
        record("stock Wan 2.1 bf16 torch.compile (harness --compile-vae-encode)", r, o, ref)
        del enc_c
        torch._dynamo.reset()
    del stock
    torch.cuda.empty_cache()

    # ------------------------------------------------ LightVAE encoder
    if "lightvae" not in skip:
        lv, _ = bc.load_lightvae()
        lenc = ec.WanArchEncoder(lv, list(bc.wan_scale("cuda", torch.bfloat16)), "pm1")
        e_ref = lenc(clip)
        record("lightvaew2_1 bf16 eager", bench(lambda: lenc(clip), iters=args.iters), e_ref)

        def lcall(c):
            return lenc(c)
        lc = torch.compile(lcall, dynamic=False)
        o, fc = first_call(lambda: lc(clip))
        r = bench(lambda: lc(clip), iters=args.iters)
        r["first_call_s"] = fc
        record("lightvaew2_1 bf16 torch.compile", r, o, e_ref)
        del lv, lenc, lc
        torch._dynamo.reset()
        torch.cuda.empty_cache()

    # ------------------------------------------------ TAEHV encoders
    eager_ref = {}
    if "tae" not in skip or "compile" not in skip:
        for name, out_conv, dts in (("taew2_1", "asis", (("fp16", torch.float16), ("bf16", torch.bfloat16))),
                                    ("lighttaew2_1", "raw2norm", (("fp16", torch.float16),))):
            for dt_name, dt in dts:
                tae = bc.load_taehv(name, dtype=dt)
                for layout in ("nchw", "channels_last"):
                    mf = torch.channels_last if layout == "channels_last" else torch.contiguous_format
                    if layout == "channels_last":
                        tae = tae.to(memory_format=torch.channels_last)
                    enc = ec.TAEHVEncoder(tae, 0, "01", out_conv, dt, memory_format=mf)
                    e = enc(clip)
                    if layout == "nchw":
                        eager_ref[(name, dt_name)] = e.clone()
                    if "tae" not in skip:
                        record(f"{name} {dt_name} eager {layout}", bench(lambda: enc(clip), iters=args.iters), e,
                               eager_ref[(name, dt_name)])
                    if "compile" not in skip and name == "taew2_1":
                        def call(c, _enc=enc):
                            return _enc(c)
                        fc_fn = torch.compile(call, dynamic=False)
                        o, fc = first_call(lambda: fc_fn(clip))
                        r = bench(lambda: fc_fn(clip), iters=args.iters)
                        r["first_call_s"] = fc
                        record(f"{name} {dt_name} torch.compile {layout}", r, o, eager_ref[(name, dt_name)])
                        del fc_fn
                        torch._dynamo.reset()
                del tae
                torch.cuda.empty_cache()

    # ------------------------------------------------ TensorRT FP16 (taew2_1; lighttaew2_1 is the same network)
    if "trt" not in skip:
        try:
            import tensorrt as trt
            dspeed = _load_decoder_speed()
            logger = trt.Logger(trt.Logger.WARNING)
            tae = bc.load_taehv("taew2_1", dtype=torch.float16)
            core = ec.TAEHVEncodeCore(tae).eval()
            info = res.get("trt") or build_trt_encoder(core, 8, 576, 320, "taew2_1_fp16", logger)
            res["trt"] = info
            eng = dspeed.TRTCore(info["engine_path"], logger)
            enc_t = ec.TAEHVEncoder(tae, 0, "01", "asis", torch.float16, core=TRTEncoderCore(eng))
            if ("taew2_1", "fp16") not in eager_ref:
                eager_ref[("taew2_1", "fp16")] = ec.TAEHVEncoder(tae, 0, "01", "asis", torch.float16)(clip)
            o = enc_t(clip)
            r = bench(lambda: enc_t(clip), iters=args.iters)
            r["trt_workspace_mib"] = eng.workspace_mib
            record("taew2_1 fp16 TensorRT (default stream)", r, o, eager_ref[("taew2_1", "fp16")])
            side = torch.cuda.Stream()
            with torch.cuda.stream(side):
                o = enc_t(clip)
                r = bench(lambda: enc_t(clip), iters=args.iters)
                r["trt_workspace_mib"] = eng.workspace_mib
                r["stream"] = "non-default torch.cuda.Stream"
                record("taew2_1 fp16 TensorRT (side stream)", r, o, eager_ref[("taew2_1", "fp16")])
            del eng, enc_t, tae, core
        except Exception as e:
            import traceback
            cur["notes"].append("TensorRT failed: " + traceback.format_exc()[-1500:])
            print("TRT failed", e, flush=True)
        torch.cuda.empty_cache()

    cur["gpu_at_end"] = bc.gpu_snapshot()
    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
