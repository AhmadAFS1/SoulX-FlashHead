#!/usr/bin/env python3
"""Motion re-encode bake-off, quality half (GPU; holds the SoulX lease).

Inputs: the 36 SoulX DiT latent windows of benchmarks/tiny_vae_20260926/latents/ (4 fixtures x
windows 0-8, seed 50). Each is decoded with the STOCK Wan decoder (bf16 eager; reproduces the
in-loop decode bit-exactly per the decoder bake-off), lean-trimmed, colour-corrected against the
reference and cut to the trailing 5 frames -- the pipeline's cond_frame, (1,3,5,576,320) bf16.
Clip construction is validated against the dump itself: window w+1's injected motion latents
(latent[w+1][:, :2]) ARE the in-loop (torch.compiled) Wan encode of window w's cond_frame.

Reference encoder: WanVAE.encode (stock weights, bf16 eager) -> (16, 2, 72, 40) normalised.
Candidates: taew2_1, lighttaew2_1 (TAEHV encoder), lightvaew2_1 (WanVAE_(dim=24)), under a
convention search (input range, frame padding/alignment, output normalisation, precision).

Baselines on the same clips (single step, same windows):
  * noise floor: eager Wan encode vs the in-loop compiled Wan encode (the dump).
  * decoder-induced perturbation: Wan encode of the cond_frame produced by the shipping pruned
    decoder / by the taew2_1 decoder, vs Wan encode of the stock-decoder cond_frame.
  * the REJECTED latent-feedback arms: last2 = the DiT's trailing 2 latents (latent[w][:, 7:9]);
    last2-fix0 = [Wan encode of cond_frame frame 0, latent[w][:, 8]].

Metrics per latent slot (0 = keyframe latent of frame 0 alone, 1 = frames 1-4), vs the Wan encode:
  nRMSE per channel (RMSE / std of the Wan latent channel over all clips), cosine similarity
  (flattened slot), per-channel Pearson, mean offset per channel (systematic = mean over clips),
  and pixel-space: STOCK Wan decode of the candidate's 2 latents (-> 5 frames) vs the input clip
  (round-trip PSNR full + mouth) and vs the Wan round trip (PSNR, mean RGB offset /255).

Execution label: fresh local GPU inference on the RTX 4070 SUPER.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_common as ec  # noqa: E402

bc = ec.bc


# ------------------------------------------------------------------------------ latent metrics
class LatAcc:
    """Accumulates latent-space error of candidate vs reference over clips, per slot."""

    def __init__(self):
        self.rows = []  # per clip: dict of per-slot stats

    @staticmethod
    def clip_stats(c, r):
        c, r = c.float(), r.float()
        assert c.shape == r.shape, (c.shape, r.shape)
        out = []
        for s in range(r.shape[1]):
            cs, rs = c[:, s], r[:, s]
            d = cs - rs
            rc = rs - rs.mean((1, 2), keepdim=True)
            cc = cs - cs.mean((1, 2), keepdim=True)
            pear = (rc * cc).sum((1, 2)) / (rc.flatten(1).norm(dim=1) * cc.flatten(1).norm(dim=1) + 1e-12)
            out.append({
                "mse_c": d.pow(2).mean((1, 2)).tolist(),
                "off_c": d.mean((1, 2)).tolist(),
                "pear_c": pear.tolist(),
                "cos": F.cosine_similarity(cs.flatten(), rs.flatten(), dim=0).item(),
                "rel_l2": (d.norm() / rs.norm()).item(),
                "maxabs": d.abs().max().item(),
            })
        return out

    def add(self, c, r, meta=None):
        st = self.clip_stats(c, r)
        self.rows.append({"meta": meta or {}, "slots": st})
        return st


def aggregate_lat(rows, ref_std):
    """ref_std[s] = per-channel std (16) of the reference latent over all clips for slot s."""
    n = len(rows)
    out = {"n_clips": n, "slots": []}
    for s in range(len(ref_std)):
        mse = torch.tensor([r["slots"][s]["mse_c"] for r in rows])  # (n,16)
        off = torch.tensor([r["slots"][s]["off_c"] for r in rows])
        pear = torch.tensor([r["slots"][s]["pear_c"] for r in rows])
        cos = torch.tensor([r["slots"][s]["cos"] for r in rows])
        rel = torch.tensor([r["slots"][s]["rel_l2"] for r in rows])
        sd = torch.tensor(ref_std[s])
        nrmse_c = mse.mean(0).sqrt() / sd
        off_mean = off.mean(0)
        off_sd = off.std(0) if n > 1 else torch.zeros(16)
        tstat = off_mean / (off_sd / math.sqrt(n) + 1e-12)
        sign_cons = ((off > 0).float().mean(0) - 0.5).abs() * 2  # 1 = every clip same sign
        out["slots"].append({
            "nrmse_per_channel": nrmse_c.tolist(),
            "nrmse_mean": nrmse_c.mean().item(),
            "nrmse_max": nrmse_c.max().item(),
            "nrmse_pooled": (mse.mean(0).sum().sqrt() / sd.pow(2).sum().sqrt()).item(),
            "cos_mean": cos.mean().item(), "cos_min": cos.min().item(),
            "pearson_mean": pear.mean().item(), "pearson_min_channel": pear.mean(0).min().item(),
            "rel_l2_mean": rel.mean().item(), "rel_l2_max": rel.max().item(),
            "offset_mean_per_channel": off_mean.tolist(),
            "offset_rms_over_channels": off_mean.pow(2).mean().sqrt().item(),
            "offset_maxabs_channel": off_mean.abs().max().item(),
            "offset_maxabs_channel_idx": int(off_mean.abs().argmax().item()),
            "offset_over_std_rms": (off_mean / sd).pow(2).mean().sqrt().item(),
            "offset_tstat_per_channel": tstat.tolist(),
            "offset_sign_consistency_mean": sign_cons.mean().item(),
            "per_clip_offset_rms_mean": off.pow(2).mean(1).sqrt().mean().item(),
        })
    s0, s1 = out["slots"][0], out["slots"][1] if len(out["slots"]) > 1 else out["slots"][0]
    out["nrmse_mean_both"] = (s0["nrmse_mean"] + s1["nrmse_mean"]) / 2
    return out


# ------------------------------------------------------------------------------ pixel metrics
def pix_stats(dec_u8, target_u8):
    """dec/target (5,H,W,3) uint8. PSNR full/mouth; mean RGB offset per slot group (frame 0 / 1-4)."""
    d0 = dec_u8[:1].float() - target_u8[:1].float()
    d1 = dec_u8[1:].float() - target_u8[1:].float()
    m = bc.MOUTH
    return {
        "psnr_full": bc.psnr(dec_u8, target_u8),
        "psnr_mouth": bc.psnr(dec_u8[:, m[0], m[1]], target_u8[:, m[0], m[1]]),
        "psnr_full_f0": bc.psnr(dec_u8[:1], target_u8[:1]),
        "psnr_full_f14": bc.psnr(dec_u8[1:], target_u8[1:]),
        "rgb_off_f0": d0.mean((0, 1, 2)).tolist(),
        "rgb_off_f14": d1.mean((0, 1, 2)).tolist(),
        "rgb_off_all": (dec_u8.float() - target_u8.float()).mean((0, 1, 2)).tolist(),
    }


def aggregate_pix(rows):
    if not rows:
        return {}
    out = {"n_clips": len(rows)}
    for k in ("psnr_full", "psnr_mouth", "psnr_full_f0", "psnr_full_f14"):
        v = [r[k] for r in rows]
        out[k] = {"mean": sum(v) / len(v), "worst": min(v)}
    for k in ("rgb_off_f0", "rgb_off_f14", "rgb_off_all"):
        v = torch.tensor([r[k] for r in rows])
        out[k] = {"mean": v.mean(0).tolist(), "worst_abs": v.abs().max(0).values.tolist(),
                  "mean_absmax": v.mean(0).abs().max().item()}
    return out


# ------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ec.EB / "quality.json")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from soulx_rtc.gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease(None)
    try:
        with torch.inference_mode():
            run(args)
    finally:
        lease.close()


def run(args):
    t_start = time.time()
    res = {"execution": "fresh local GPU inference on RTX 4070 SUPER",
           "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "torch": torch.__version__, "gpu_at_start": bc.gpu_snapshot(), "notes": []}
    torch.manual_seed(0)
    items = bc.load_latents(args.limit)
    key = {(w["fixture"], w["window"]): i for i, (w, _) in enumerate(items)}
    ref_img = ec.load_reference()
    res["reference_image"] = {"path": str(ec.REF_IMAGE.relative_to(ec.ROOT)), "sha256": bc.sha256(ec.REF_IMAGE),
                              "tensor_shape": list(ref_img.shape), "dtype": str(ref_img.dtype)}

    stock = bc.load_stock()
    ship = bc.load_shipping()
    res["shipping_decoder"] = ship._bakeoff_manifest
    tae_dec = bc.load_taehv("taew2_1", dtype=torch.float16)
    tae_dec_ad = bc.TAEHVAdapter(tae_dec, "norm", torch.float16)

    # ---------------------------------------------------------------- phase 1: clips
    clips = {"stock": [], "ship": [], "taedec": []}
    val = LatAcc()
    ref_lat = []
    for i, (w, d) in enumerate(items):
        z = d["latent"].cuda()
        clips["stock"].append(ec.cond_frames(stock.decode(z), ref_img))
        clips["ship"].append(ec.cond_frames(ship.decode(z), ref_img))
        clips["taedec"].append(ec.cond_frames(tae_dec_ad.decode_window(z), ref_img))
        r = stock.encode(clips["stock"][-1])
        assert r.shape == (16, 2, 72, 40) and r.dtype == torch.bfloat16, (r.shape, r.dtype)
        ref_lat.append(r)
        nxt = key.get((w["fixture"], w["window"] + 1))
        if nxt is not None:
            inloop = items[nxt][1]["latent"][:, :2].cuda()
            val.add(r, inloop, {"fixture": w["fixture"], "window": w["window"]})
        print(f"clip {i:2d} {w['file']}", flush=True)
    del ship
    torch.cuda.empty_cache()
    n = len(items)
    res["clips"] = {"n": n, "shape": list(clips["stock"][0].shape), "dtype": str(clips["stock"][0].dtype),
                    "range": [min(c.min().item() for c in clips["stock"]), max(c.max().item() for c in clips["stock"])],
                    "fixtures": sorted({w["fixture"] for w, _ in items})}

    # per-channel std of the Wan reference latent over all clips, per slot
    R = torch.stack([r.float() for r in ref_lat])  # (n,16,2,h,w)
    ref_std = [R[:, :, s].transpose(0, 1).reshape(16, -1).std(1).tolist() for s in range(2)]
    ref_mean = [R[:, :, s].transpose(0, 1).reshape(16, -1).mean(1).tolist() for s in range(2)]
    res["wan_latent_stats"] = {"per_channel_std": ref_std, "per_channel_mean": ref_mean,
                               "global_std": [R[:, :, s].std().item() for s in range(2)]}
    del R
    # validation: eager Wan encode of my clip vs the in-loop (compiled) Wan encode in the dump
    res["validation_vs_inloop"] = {
        "what": "eager Wan encode of the reconstructed cond_frame(w) vs latent[w+1][:, :2] injected in-loop by the compiled encoder",
        "agg": aggregate_lat(val.rows, ref_std),
        "maxabs_max": max(max(s["maxabs"] for s in r["slots"]) for r in val.rows) if val.rows else None,
        "rel_l2_max": max(max(s["rel_l2"] for s in r["slots"]) for r in val.rows) if val.rows else None,
    }
    print("validation:", json.dumps({k: res["validation_vs_inloop"][k] for k in ("maxabs_max", "rel_l2_max")}), flush=True)

    # ---------------------------------------------------------------- phase 2: convention search
    tae = {"taew2_1": bc.load_taehv("taew2_1", dtype=torch.float16),
           "lighttaew2_1": bc.load_taehv("lighttaew2_1", dtype=torch.float16)}
    tae32 = {"taew2_1": bc.load_taehv("taew2_1", dtype=torch.float32),
             "lighttaew2_1": bc.load_taehv("lighttaew2_1", dtype=torch.float32)}
    tae_bf = bc.load_taehv("taew2_1", dtype=torch.bfloat16)
    lv, lv_load = bc.load_lightvae()
    res["lightvae_load"] = lv_load
    wscale = list(bc.wan_scale("cuda", torch.bfloat16))

    # core == upstream encode_video (end pad, [0,1])
    x0 = (clips["stock"][0][0].transpose(0, 1).float() + 1) * 0.5
    up = tae32["taew2_1"].encode_video(x0[None], parallel=True, show_progress_bar=False)[0]
    mine = ec.TAEHVEncoder(tae32["taew2_1"], 0, "01", "asis", torch.float32).core(
        torch.cat([x0, x0[-1:].expand(3, -1, -1, -1)], 0))
    res["core_vs_upstream_encode_video_maxabs_fp32"] = (up - mine).abs().max().item()
    up_seq = tae32["taew2_1"].encode_video(x0[None], parallel=False, show_progress_bar=False)[0]
    res["upstream_parallel_vs_sequential_maxabs_fp32"] = (up - up_seq).abs().max().item()
    print("core vs upstream", res["core_vs_upstream_encode_video_maxabs_fp32"], "par vs seq", res["upstream_parallel_vs_sequential_maxabs_fp32"], flush=True)

    variants = {}
    for name in ("taew2_1", "lighttaew2_1"):
        for front in range(4):
            for out in ("asis", "raw2norm"):
                variants[f"{name} fp16 front{front} in01 {out}"] = ec.TAEHVEncoder(tae[name], front, "01", out)
        for out in ("asis", "raw2norm"):
            variants[f"{name} fp16 front0 inpm1 {out}"] = ec.TAEHVEncoder(tae[name], 0, "pm1", out)
    variants["taew2_1 fp32 front0 in01 asis"] = ec.TAEHVEncoder(tae32["taew2_1"], 0, "01", "asis", torch.float32)
    variants["taew2_1 bf16 front0 in01 asis"] = ec.TAEHVEncoder(tae_bf, 0, "01", "asis", torch.bfloat16)
    variants["lighttaew2_1 fp32 front0 in01 raw2norm"] = ec.TAEHVEncoder(tae32["lighttaew2_1"], 0, "01", "raw2norm", torch.float32)
    variants["lightvaew2_1 bf16 inpm1 wan-scale"] = ec.WanArchEncoder(lv, wscale, "pm1")
    variants["lightvaew2_1 bf16 inpm1 no-scale"] = ec.WanArchEncoder(lv, None, "pm1")
    variants["lightvaew2_1 bf16 in01 wan-scale"] = ec.WanArchEncoder(lv, wscale, "01")

    conv_acc = {k: LatAcc() for k in variants}
    cand_lat = {k: [] for k in variants}
    for i in range(n):
        for k, enc in variants.items():
            e = enc(clips["stock"][i])
            assert e.shape == (16, 2, 72, 40), (k, e.shape)
            conv_acc[k].add(e, ref_lat[i])
            cand_lat[k].append(e.to(torch.bfloat16))
    res["convention_search"] = {k: aggregate_lat(a.rows, ref_std) for k, a in conv_acc.items()}
    for k, a in res["convention_search"].items():
        print(f"{k:45s} nRMSE s0 {a['slots'][0]['nrmse_mean']:.3f} s1 {a['slots'][1]['nrmse_mean']:.3f} "
              f"cos {a['slots'][0]['cos_mean']:.4f}/{a['slots'][1]['cos_mean']:.4f}", flush=True)

    def best(prefix):
        ks = [k for k in variants if k.startswith(prefix + " ")]
        return min(ks, key=lambda k: res["convention_search"][k]["nrmse_mean_both"])

    chosen = {"taew2_1": best("taew2_1 fp16"), "lighttaew2_1": best("lighttaew2_1 fp16"),
              "lightvaew2_1": best("lightvaew2_1 bf16")}
    res["chosen_conventions"] = chosen
    print("chosen", chosen, flush=True)

    # ---------------------------------------------------------------- phase 3: baselines + round trip
    arms = {}  # name -> list of (16,2,h,w) latents, one per clip
    for c, k in chosen.items():
        arms[f"{c} ({k})"] = cand_lat[k]
    # show the pixel consequence of the Wan-style alignment guess too
    tae_front3 = "taew2_1 fp16 front3 in01 asis"
    if tae_front3 != chosen["taew2_1"]:
        arms[f"taew2_1 alt ({tae_front3})"] = cand_lat[tae_front3]
    wan_ship = [stock.encode(c) for c in clips["ship"]]
    wan_taedec = [stock.encode(c) for c in clips["taedec"]]
    arms["baseline: Wan encode of shipping-decoder cond_frame"] = wan_ship
    arms["baseline: Wan encode of taew2_1-decoder cond_frame"] = wan_taedec
    enc_tae = variants[chosen["taew2_1"]]
    enc_lv = variants[chosen["lightvaew2_1"]]
    arms["combo: taew2_1 encode of taew2_1-decoder cond_frame"] = [enc_tae(c) for c in clips["taedec"]]
    # the encoder's own error on the frames each real arm would feed it, vs the Wan encode of THE SAME clip
    arm_ref = {}  # arm -> (reference latents, reference label); default = Wan encode of the stock clip
    k = "encoder-only on shipping frames: taew2_1 encode of shipping-decoder cond_frame (vs Wan encode of same clip)"
    arms[k] = [enc_tae(c) for c in clips["ship"]]
    arm_ref[k] = (wan_ship, "Wan encode of the shipping-decoder cond_frame")
    k = "encoder-only on shipping frames: lightvaew2_1 encode of shipping-decoder cond_frame (vs Wan encode of same clip)"
    arms[k] = [enc_lv(c) for c in clips["ship"]]
    arm_ref[k] = (wan_ship, "Wan encode of the shipping-decoder cond_frame")
    k = "taew2_1 encode of taew2_1-decoder cond_frame (vs Wan encode of same clip)"
    arms[k] = arms["combo: taew2_1 encode of taew2_1-decoder cond_frame"]
    arm_ref[k] = (wan_taedec, "Wan encode of the taew2_1-decoder cond_frame")
    fb_last2, fb_fix0 = [], []
    for i, (w, d) in enumerate(items):
        z = d["latent"].cuda()
        fb_last2.append(z[:, -2:].contiguous())
        f0 = stock.encode(clips["stock"][i][:, :, :1])  # (16,1,h,w), the fix0 keyframe encode
        fb_fix0.append(torch.cat([f0, z[:, -1:]], 1).contiguous())
    arms["REJECTED latent feedback last2 (DiT latents 7,8)"] = fb_last2
    arms["REJECTED latent feedback last2-fix0 (Wan enc f0 + DiT latent 8)"] = fb_fix0
    arm_lat = {k: LatAcc() for k in arms}
    arm_pix_in = {k: [] for k in arms}   # vs the arm's own input clip (round trip)
    arm_pix_ref = {k: [] for k in arms}  # vs the Wan round trip of the stock clip (what the DiT sees, in pixels)
    wan_rt_rows = []
    own_clip = {k: ("ship" if "shipping-decoder" in k else "taedec" if "taew2_1-decoder" in k else "stock") for k in arms}
    for i in range(n):
        ref_dec = bc.to_u8(stock.decode(ref_lat[i]))  # (5,H,W,3)
        clip_u8 = {k: bc.to_u8(v[i]) for k, v in clips.items()}
        wan_rt_rows.append(pix_stats(ref_dec, clip_u8["stock"]))
        alt_dec = {}
        for k, lats in arms.items():
            e = lats[i]
            if k in arm_ref:
                rl, rlab = arm_ref[k]
                if rlab not in alt_dec:
                    alt_dec[rlab] = bc.to_u8(stock.decode(rl[i]))
                r_lat, r_dec = rl[i], alt_dec[rlab]
            else:
                r_lat, r_dec = ref_lat[i], ref_dec
            arm_lat[k].add(e, r_lat, {"fixture": items[i][0]["fixture"], "window": items[i][0]["window"]})
            dec = bc.to_u8(stock.decode(e))
            arm_pix_in[k].append(pix_stats(dec, clip_u8[own_clip[k]]))
            arm_pix_ref[k].append(pix_stats(dec, r_dec))
        # the Wan round trip of the other clips vs themselves, for the decoder baselines' own reference
        print(f"round trip {i:2d}", flush=True)
    res["wan_roundtrip_vs_clip"] = aggregate_pix(wan_rt_rows)
    res["arms"] = {}
    for k in arms:
        res["arms"][k] = {"latent_vs_wan_encode": aggregate_lat(arm_lat[k].rows, ref_std),
                          "roundtrip_vs_own_input_clip": aggregate_pix(arm_pix_in[k]),
                          "pixels_vs_wan_roundtrip": aggregate_pix(arm_pix_ref[k]),
                          "input_clip": own_clip[k],
                          "reference": arm_ref[k][1] if k in arm_ref else "Wan encode of the stock-decoder cond_frame"}
    res["per_clip"] = {k: [{"meta": r["meta"], "slots": [{kk: (round(v, 5) if isinstance(v, float) else [round(x, 5) for x in v])
                                                           for kk, v in s.items()} for s in r["slots"]]}
                           for r in arm_lat[k].rows] for k in arms}
    res["per_clip_pixels_vs_wan_roundtrip"] = {k: arm_pix_ref[k] for k in arms}
    res["elapsed_s"] = time.time() - t_start
    res["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2**20
    res["gpu_at_end"] = bc.gpu_snapshot()
    args.out.write_text(json.dumps(res, indent=1))
    print("wrote", args.out, f"{res['elapsed_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
