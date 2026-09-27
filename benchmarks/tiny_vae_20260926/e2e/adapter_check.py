"""GPU check of soulx_rtc/pro_tiny_vae.py before it goes into the harness (2026-09-26).

For one fixture's 9 saved DiT latent windows (benchmarks/tiny_vae_20260926/latents, fresh SoulX
DiT outputs from the decoder bake-off) it checks, per backend (eager / compile / tensorrt):
  * decoder, stream mode: frame count 33 every window, bf16, range [-1,1], and max-abs difference
    against the bake-off's independent reference implementation (bakeoff_common.TAEHVAdapter
    .stream_decode, eager fp16) -- eager must be bit-identical;
  * decoder, window mode vs TAEHVAdapter.decode_window;
  * encoder on a real 5-frame clip vs encoder_common.TAEHVEncoder (front=0, "01", "asis");
  * lightvaew2_1 decode/encode shapes (eager only);
  * single-call timings with CUDA events (median of 10 after 3 warm-up calls).
Holds the SoulX GPU lease. Writes adapter_check.json next to this file.
usage: PYTHONPATH=benchmarks/tiny_vae_20260926/decoder_bakeoff/_trt10_path:.pro-quant-deps:. \
       .venv/bin/python benchmarks/tiny_vae_20260926/e2e/adapter_check.py [--backends eager,compile,tensorrt]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE.parent / "decoder_bakeoff"))
sys.path.insert(0, str(HERE.parent / "encoder_bakeoff"))


def bench(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        times.append(a.elapsed_time(b))
    return round(statistics.median(times), 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backends", default="eager,compile,tensorrt")
    ap.add_argument("--fixture", default="indian150-a")
    args = ap.parse_args()
    from soulx_rtc.gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease(None)
    try:
        run(args)
    finally:
        lease.close()


def run(args):
    import bakeoff_common as bc
    import encoder_common as ec
    from soulx_rtc import pro_tiny_vae as ptv
    from flash_head.wan.modules import WanVAE

    torch.backends.cudnn.benchmark = False  # harness default; the adapter scopes its own
    out = {"label": "fresh local GPU inference on RTX 4070 SUPER (adapter unit check, not a harness run)",
           "gpu": bc.gpu_snapshot(), "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "fixture": args.fixture, "checks": {}}
    items = [(w, d) for w, d in bc.load_latents() if w["fixture"] == args.fixture]
    items.sort(key=lambda x: x[0]["window"])
    zs = [d["latent"].cuda() for _, d in items]
    wan = WanVAE(vae_path=str(bc.STOCK_VAE), dtype=torch.bfloat16, device="cuda")
    pipe = SimpleNamespace(vae=wan)
    orig_decode, orig_encode = wan.decode, wan.encode

    with torch.no_grad():
        # independent reference (bake-off code, eager fp16)
        tae_ref = bc.load_taehv("taew2_1")
        ref = bc.TAEHVAdapter(tae_ref, "norm", torch.float16)
        ref.stream_reset()
        ref_stream = [ref.stream_decode(z).clone() for z in zs]
        ref_window = ref.decode_window(zs[4]).clone()
        # a real motion clip: Wan decode of window 4 -> lean trim -> colour correction -> last 5
        videos = orig_decode(zs[4])
        clip = ec.cond_frames(videos, ec.load_reference()).contiguous()
        ref_enc = ec.TAEHVEncoder(tae_ref, front=0, rng="01", out="asis")(clip)
        wan_enc = orig_encode(clip)
        del tae_ref, ref, videos
        torch.cuda.empty_cache()

        for backend in [b for b in args.backends.split(",") if b]:
            rec = {}
            wan.decode, wan.encode = orig_decode, orig_encode
            for attr in ("_pro_tiny_decoder", "_pro_tiny_encoder"):
                if hasattr(wan, attr):
                    delattr(wan, attr)
            t0 = time.perf_counter()
            dec = ptv.install_tiny_decoder(pipe, "taew2_1", mode="stream", backend=backend, history_frames=5,
                                           model_frames=33, resolution=(576, 320))
            rec["decoder_install_s"] = round(time.perf_counter() - t0, 2)
            outs = [dec(z) for z in zs]
            torch.cuda.synchronize()
            rec["stream_frames"] = dec.state_manifest()["frames_per_window"]
            rec["stream_latents_decoded"] = dec.state_manifest()["latents_decoded_per_window"]
            rec["dtype_ok"] = all(o.dtype == torch.bfloat16 and o.is_contiguous() for o in outs)
            rec["range"] = [min(float(o.min()) for o in outs), max(float(o.max()) for o in outs)]
            rec["stream_maxabs_vs_bakeoff_ref"] = max(float((o.float() - r.float()).abs().max()) for o, r in zip(outs, ref_stream))
            rec["stream_psnr_u8_vs_bakeoff_ref_min"] = min(bc.psnr(bc.to_u8(o), bc.to_u8(r)) for o, r in zip(outs, ref_stream))
            # timings on the harness-shaped calls: cold (reset + window 0) and warm (window 4 after state)
            def cold():
                dec.reset(); return dec(zs[0])
            rec["cold9_ms"] = bench(cold)
            dec.reset(); dec(zs[0])
            state = [s.clone() for s in dec.state]; tail = dec.tail.clone()
            def warm():
                dec.state = [s.clone() for s in state]; dec.tail = tail; dec.windows = 1
                return dec(zs[4])
            rec["warm7_ms_incl_state_clone"] = bench(warm)
            # window mode
            wan.decode = orig_decode; delattr(wan, "_pro_tiny_decoder")
            decw = ptv.install_tiny_decoder(pipe, "taew2_1", mode="window", backend=backend, history_frames=5,
                                            model_frames=33, resolution=(576, 320))
            ow = decw(zs[4])
            rec["window_frames"] = int(ow.shape[2])
            rec["window_maxabs_vs_bakeoff_ref"] = float((ow.float() - ref_window.float()).abs().max())
            rec["window9_ms"] = bench(lambda: decw(zs[4]))
            wan.decode = orig_decode; delattr(wan, "_pro_tiny_decoder")
            del dec, decw, outs
            torch.cuda.empty_cache()
            # encoder
            enc = ptv.install_tiny_encoder(pipe, "taew2_1", backend=backend, clip_frames=5, resolution=(576, 320))
            e = enc(clip)
            rec["encoder_shape"] = list(e.shape)
            rec["encoder_dtype"] = str(e.dtype)
            rec["encoder_maxabs_vs_bakeoff_ref"] = float((e.float() - ref_enc.float()).abs().max())
            std = wan_enc.float().std(dim=(2, 3), keepdim=True)
            rec["encoder_nrmse_vs_wan_per_slot"] = [round(float(((e.float() - wan_enc.float())[:, s] ** 2).mean().sqrt()
                                                              / wan_enc.float()[:, s].std()), 4) for s in range(2)]
            rec["encode_ms"] = bench(lambda: enc(clip))
            try:
                enc(clip[:, :, :1])
                rec["encoder_refuses_other_lengths"] = False
            except ValueError:
                rec["encoder_refuses_other_lengths"] = True
            wan.encode = orig_encode; delattr(wan, "_pro_tiny_encoder")
            del enc
            torch.cuda.empty_cache()
            rec["peak_alloc_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
            out["checks"][f"taew2_1/{backend}"] = rec
            print(backend, json.dumps(rec), flush=True)
            torch._dynamo.reset()

        # lighttaew2_1 convention path (raw latents) and lightvaew2_1 (Wan protocol): eager smoke
        for name in ("lighttaew2_1", "lightvaew2_1"):
            rec = {}
            wan.decode, wan.encode = orig_decode, orig_encode
            dec = ptv.install_tiny_decoder(pipe, name, mode="stream", backend="eager", history_frames=5,
                                           model_frames=33, resolution=(576, 320))
            o = [dec(z) for z in zs[:3]]
            ref_stock = orig_decode(zs[1])
            rec["stream_frames"] = dec.state_manifest()["frames_per_window"]
            rec["psnr_u8_window1_vs_stock_wan"] = bc.psnr(bc.to_u8(o[1][:, :, 5:]), bc.to_u8(ref_stock[:, :, 5:]))
            enc = ptv.install_tiny_encoder(pipe, name, backend="eager", clip_frames=5, resolution=(576, 320))
            e = enc(clip)
            rec["encoder_shape"] = list(e.shape)
            rec["encoder_nrmse_vs_wan_per_slot"] = [round(float(((e.float() - wan_enc.float())[:, s] ** 2).mean().sqrt()
                                                              / wan_enc.float()[:, s].std()), 4) for s in range(2)]
            for attr in ("_pro_tiny_decoder", "_pro_tiny_encoder"):
                delattr(wan, attr)
            del dec, enc, o
            torch.cuda.empty_cache()
            out["checks"][f"{name}/eager"] = rec
            print(name, json.dumps(rec), flush=True)
    out["gpu_after"] = bc.gpu_snapshot()
    (HERE / "adapter_check.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
