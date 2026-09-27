#!/usr/bin/env python3
"""Decoder quality bake-off at 576x320 on SoulX's own DiT latents (GPU; holds the SoulX lease).

For every latent window in benchmarks/tiny_vae_20260926/latents/index.json:
  reference = STOCK Wan 2.1 decoder, bf16 eager, stock weights: 9 latents -> 33 frames
  shipping  = stock + ft4 distilled weights + block skip [9,10,13,14], bf16 eager
  tiny      = taew2_1 / lighttaew2_1 (fp16, and bf16 as a variant), lightvaew2_1 (bf16)
Convention search (tiny candidates): latent normalised vs un-normalised (z*std+mean), output
range [0,1]->[-1,1] vs taken as [-1,1], NTCHW vs NCTHW layout, temporal offset / trim of the
36 raw TAEHV frames (offsets -3..+2 around the upstream 'drop first 3'), LightVAE with vs without
the Wan un-normalisation, LightVAE temporal offsets -2..+2.
Window mode: every window decoded from scratch (what the non-overlap-skip pipeline does).
Stream mode: the overlap-skip analogue per fixture sequence (window 0 cold; windows 1+ decode
the trailing 7 latents with the carried decoder state; metrics on the 28 delivered frames).

Execution label: fresh local GPU inference on the RTX 4070 SUPER.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bakeoff_common as bc  # noqa: E402

TAE_OFFSETS = [-3, -2, -1, 0, 1, 2]
LV_OFFSETS = [-2, -1, 0, 1, 2]


def aligned_psnr(cand_u8, ref_u8, offset):
    """cand frame (t + offset) vs ref frame t over the valid overlap."""
    n_c, n_r = cand_u8.shape[0], ref_u8.shape[0]
    t0, t1 = max(0, -offset), min(n_r, n_c - offset)
    c, r = cand_u8[t0 + offset:t1 + offset], ref_u8[t0:t1]
    m = (bc.MOUTH[0], bc.MOUTH[1])
    return {"frames": int(t1 - t0), "psnr_full": bc.psnr(c, r), "psnr_mouth": bc.psnr(c[:, m[0], m[1]], r[:, m[0], m[1]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=bc.BK / "quality.json")
    ap.add_argument("--visual", default="indian150-a:4,tts-open-vowels:6", help="fixture:window pairs for PNG grids")
    ap.add_argument("--video", default="indian150-a:4", help="fixture:window for the side-by-side mp4")
    args = ap.parse_args()

    from soulx_rtc.gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease(None)
    try:
        run(args)
    finally:
        lease.close()


def run(args):
    torch.backends.cudnn.benchmark = False
    t_start = time.time()
    res = {"execution": "fresh local GPU inference on RTX 4070 SUPER", "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "torch": torch.__version__, "gpu_at_start": bc.gpu_snapshot(), "notes": []}
    stock = bc.load_stock()
    ship = bc.load_shipping()
    res["shipping_manifest"] = ship._bakeoff_manifest
    lv, lv_load = bc.load_lightvae()
    res["lightvae_load"] = lv_load
    scale = stock.scale  # [mean, inv_std] bf16, exactly the pipeline's
    zero_scale = [torch.zeros_like(scale[0]), torch.ones_like(scale[1])]
    taes = {n: bc.load_taehv(n, dtype=torch.float16) for n in ("taew2_1", "lighttaew2_1")}
    taes_bf16 = {n: bc.load_taehv(n, dtype=torch.bfloat16) for n in ("taew2_1", "lighttaew2_1")}
    tae_fp32 = bc.load_taehv("taew2_1", dtype=torch.float32)
    ad = {(n, c): bc.TAEHVAdapter(taes[n], c, torch.float16) for n in taes for c in ("norm", "unnorm")}
    # expected conventions (CPU probe); verified below from the GPU search and flagged if different
    chosen = {"taew2_1": "norm", "lighttaew2_1": "unnorm"}
    ad_bf16 = {n: bc.TAEHVAdapter(taes_bf16[n], chosen[n], torch.bfloat16) for n in taes_bf16}
    ad_fp32 = bc.TAEHVAdapter(tae_fp32, "norm", torch.float32)

    streams = {
        "stock": bc.WanStream(stock.model, scale),
        "shipping": bc.WanStream(ship.model, scale),
        "lightvaew2_1": bc.WanStream(lv, scale),
        "taew2_1": bc.TAEHVAdapter(taes["taew2_1"], chosen["taew2_1"], torch.float16),
        "lighttaew2_1": bc.TAEHVAdapter(taes["lighttaew2_1"], chosen["lighttaew2_1"], torch.float16),
    }

    visual_keys = {tuple([p.split(":")[0], int(p.split(":")[1])]) for p in args.visual.split(",") if p}
    video_key = tuple([args.video.split(":")[0], int(args.video.split(":")[1])])
    keep_frames = {}

    items = bc.load_latents(args.limit)
    rows = {k: [] for k in ("shipping", "taew2_1", "lighttaew2_1", "lightvaew2_1",
                            "taew2_1_bf16", "lighttaew2_1_bf16")}
    stream_rows = {k: [] for k in streams}
    conv_rows = []  # per-window convention/alignment search
    per_window = []
    prev = None
    with torch.inference_mode():
        # layout check: TAEHV is NTCHW; the Wan NCTHW layout is refused by the first conv
        z0 = items[0][1]["latent"].cuda()
        try:
            taes["taew2_1"].decode_video(z0.unsqueeze(0).half(), parallel=True, show_progress_bar=False)
            res["layout_ncthw_into_taehv"] = "ran (unexpected)"
        except Exception as e:  # expected
            res["layout_ncthw_into_taehv"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        for w, d in items:
            key = (w["fixture"], w["window"])
            z = d["latent"].cuda()
            t0 = time.time()
            ref = stock.decode(z)  # (1,3,33,H,W) bf16, clamped
            r = bc.to_u8(ref)
            if prev is None or prev[0] != w["fixture"]:
                sanity = (ref[0, :, ::8, ::8, ::8].float() - d["harness_out_strided"].cuda().float()).abs().max().item()
                res.setdefault("sanity_vs_harness_decode_maxabs", {})[w["fixture"]] = sanity
            win = {"fixture": w["fixture"], "window": w["window"], "file": w["file"]}
            frames = {"reference (stock Wan, bf16)": r} if key in visual_keys or key == video_key else None

            # ---------------- shipping decoder, window mode
            s = bc.to_u8(ship.decode(z))
            rows["shipping"].append(bc.metrics(s, r))
            if frames is not None:
                frames["shipping (pruned+ft4, bf16)"] = s

            # ---------------- TAEHV family: convention / alignment search
            cr = {"fixture": w["fixture"], "window": w["window"]}
            for n in taes:
                for c in ("norm", "unnorm"):
                    raw, _ = ad[(n, c)].raw(z)  # (36,3,H,W) [0,1]
                    raw_u8 = bc.to_u8(bc.TAEHVAdapter.to_wan(raw))
                    for o in TAE_OFFSETS:
                        cr[f"{n}|{c}|off{o:+d}"] = aligned_psnr(raw_u8, r, 3 + o)
                    cand = raw_u8[3:]
                    m = bc.metrics(cand, r)
                    cr[f"{n}|{c}|full"] = m
                    if c == chosen[n]:
                        rows[n].append(m)
                        if frames is not None:
                            frames[f"{n} (fp16)"] = cand
                        # wrong output range: [0,1] frames read as if already [-1,1]
                        wrong = bc.to_u8(raw[3:].permute(1, 0, 2, 3).unsqueeze(0).to(torch.bfloat16))
                        cr[f"{n}|{c}|out_as_pm1"] = {"psnr_full": bc.psnr(wrong, r)}
                    del raw, raw_u8, cand
                # bf16 variant with the chosen convention
                mb = bc.metrics(bc.to_u8(ad_bf16[n].decode_window(z)), r)
                rows[f"{n}_bf16"].append(mb)
            if len(per_window) < 3:  # fp32 vs fp16 precision check on the first windows
                f32 = bc.to_u8(ad_fp32.decode_window(z))
                f16 = bc.to_u8(ad[("taew2_1", "norm")].decode_window(z))
                cr["taew2_1_fp32_vs_fp16_psnr"] = bc.psnr(f16, f32)
                cr["taew2_1_fp32_vs_ref_psnr"] = bc.psnr(f32, r)
                del f32, f16

            # ---------------- LightVAE (SoulX WanVAE_(dim=24))
            lvo = lv.decode(z.unsqueeze(0), scale).clamp_(-1, 1)
            lv_u8 = bc.to_u8(lvo)
            rows["lightvaew2_1"].append(bc.metrics(lv_u8, r))
            for o in LV_OFFSETS:
                cr[f"lightvaew2_1|scale|off{o:+d}"] = aligned_psnr(lv_u8, r, o)
            lvw = bc.to_u8(lv.decode(z.unsqueeze(0), zero_scale).clamp_(-1, 1))
            cr["lightvaew2_1|no_unnorm|off+0"] = {"psnr_full": bc.psnr(lvw, r)}
            if frames is not None:
                frames["lightvaew2_1 (bf16)"] = lv_u8
            del lvo, lvw
            conv_rows.append(cr)

            # ---------------- stream mode (overlap-skip analogue), in window order per fixture
            if w["window"] == 0:
                for st in streams.values():
                    st.stream_reset()
            else:
                assert prev == (w["fixture"], w["window"] - 1), f"non-consecutive windows {prev} -> {key}"
            for n, st in streams.items():
                out = st.stream_decode(z)
                assert out.shape[2] == 33, (n, out.shape)
                stream_rows[n].append({**bc.metrics(bc.to_u8(out)[bc.HIST:], r[bc.HIST:]), "window": w["window"], "fixture": w["fixture"]})
                del out
            prev = key
            win["seconds"] = time.time() - t0
            per_window.append(win)
            if frames is not None:
                keep_frames[key] = {k: v.cpu() for k, v in frames.items()}
            print(f"[{len(per_window)}/{len(items)}] {w['file']} ship {rows['shipping'][-1]['psnr_full']:.2f} "
                  f"tae {rows['taew2_1'][-1]['psnr_full']:.2f} ltae {rows['lighttaew2_1'][-1]['psnr_full']:.2f} "
                  f"lvae {rows['lightvaew2_1'][-1]['psnr_full']:.2f} ({win['seconds']:.1f}s)", flush=True)
            del ref, r, s, z
    res["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20

    # ---------------- convention summary (mean over windows)
    def mean_of(k, field="psnr_full"):
        v = [c[k][field] for c in conv_rows if k in c]
        return sum(v) / len(v) if v else None
    conv_summary = {}
    for n in taes:
        for c in ("norm", "unnorm"):
            conv_summary[f"{n} latent={c}"] = {
                "psnr_full_mean": mean_of(f"{n}|{c}|full"), "psnr_mouth_mean": mean_of(f"{n}|{c}|full", "psnr_mouth"),
                "offsets": {o: {"psnr_full_mean": mean_of(f"{n}|{c}|off{o:+d}"), "psnr_mouth_mean": mean_of(f"{n}|{c}|off{o:+d}", "psnr_mouth"),
                                "frames": conv_rows[0][f"{n}|{c}|off{o:+d}"]["frames"]} for o in TAE_OFFSETS},
                "colour_offset_rgb_mean": [sum(cr[f"{n}|{c}|full"]["colour_offset_rgb"][i] for cr in conv_rows) / len(conv_rows) for i in range(3)],
                "mae_full_mean": mean_of(f"{n}|{c}|full", "mae_full"),
            }
        conv_summary[f"{n} output read as [-1,1] (wrong range)"] = {"psnr_full_mean": mean_of(f"{n}|{chosen[n]}|out_as_pm1")}
    conv_summary["lightvaew2_1 decode(z, wan scale)"] = {
        "offsets": {o: {"psnr_full_mean": mean_of(f"lightvaew2_1|scale|off{o:+d}"), "psnr_mouth_mean": mean_of(f"lightvaew2_1|scale|off{o:+d}", "psnr_mouth"),
                        "frames": conv_rows[0][f"lightvaew2_1|scale|off{o:+d}"]["frames"]} for o in LV_OFFSETS}}
    conv_summary["lightvaew2_1 decode(z, no un-normalisation)"] = {"psnr_full_mean": mean_of("lightvaew2_1|no_unnorm|off+0")}
    best = {}
    for n in taes:
        cands = [(conv_summary[f"{n} latent={c}"]["offsets"][o]["psnr_full_mean"], c, o) for c in ("norm", "unnorm") for o in TAE_OFFSETS
                 if conv_summary[f"{n} latent={c}"]["offsets"][o]["frames"] == 33]
        p, c, o = max(cands)
        best[n] = {"latent": c, "offset": o, "trim": f"drop first {3 + o}, last {-o} of 36 raw frames", "psnr_full_mean": p,
                   "matches_assumed": c == chosen[n] and o == 0}
    lv_c = [(conv_summary["lightvaew2_1 decode(z, wan scale)"]["offsets"][o]["psnr_full_mean"], o) for o in LV_OFFSETS
            if conv_summary["lightvaew2_1 decode(z, wan scale)"]["offsets"][o]["frames"] == 33]
    best["lightvaew2_1"] = {"latent": "normalised, decode(z, wan scale)", "offset": max(lv_c)[1], "psnr_full_mean": max(lv_c)[0]}
    precision = [{k: cr[k] for k in cr if k.startswith("taew2_1_fp32")} for cr in conv_rows if "taew2_1_fp32_vs_fp16_psnr" in cr]

    res.update({
        "windows": per_window,
        "window_mode": {k: bc.aggregate(v) for k, v in rows.items()},
        "window_mode_rows": rows,
        "stream_mode": {k: bc.aggregate(v) for k, v in stream_rows.items()},
        "stream_mode_rows": stream_rows,
        "convention_summary": conv_summary,
        "convention_chosen": best,
        "precision_check_taew2_1": precision,
        "convention_rows": conv_rows,
        "gpu_at_end": bc.gpu_snapshot(),
        "wall_s": time.time() - t_start,
    })
    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out)
    if keep_frames:
        import visuals
        vis = visuals.make(keep_frames, video_key, rows_by_key={(p["fixture"], p["window"]): i for i, p in enumerate(per_window)}, window_rows=rows)
        res["visuals"] = vis
        args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({"convention_chosen": best, "window_mode": {k: {m: v[m] for m in ("psnr_full", "psnr_mouth", "sharp_mouth")} for k, v in res["window_mode"].items()}}, indent=1))


if __name__ == "__main__":
    main()
