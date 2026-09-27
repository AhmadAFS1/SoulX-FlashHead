#!/usr/bin/env python3
"""CPU-only empirical latent-convention probe for tiny / light Wan 2.1 autoencoders.

Hardware: CPU only (CUDA hidden). NOT GPU inference; no speed numbers are produced.

Why: the MuseTalk/TAESD precedent on this machine showed that the latent scaling convention
must be resolved empirically (MAE 0.26 with the wrong convention, 0.015 with the right one).

Data: 9 consecutive frames (1 + 4k, the Wan causal grouping) of the 576x320 reference run
benchmarks/pro_30fps_20260921/s30-reuse/video.mp4 (SoulX PRO output; mp4-decoded because
every raw.npy on disk has been deleted), a 192x192 face crop (rows 128:320, cols 64:256,
eyes to chin, mouth open). The crop keeps the CPU cost small; every decoder sees the same
crop, and metrics use the 160x160 interior (16 px border dropped) to avoid crop-edge effects.

Reference: SoulX's own stock Wan VAE (flash_head/wan/modules/vae.py, Wan2.1_VAE.pth, fp32).
  z      = WanVAE_.encode(x, scale)   -- NORMALISED latents, the space the DiT works in
  ref    = WanVAE_.decode(z, scale)   -- what the full decoder makes of them
Candidates decode z under both conventions:
  "normalised"   : feed z directly
  "unnormalised" : feed z * std + mean (what WanVAE_.decode computes internally)
and their encoders are compared with z / z_raw under two frame alignments:
  "frontpad3" : prepend 3 copies of frame 0 (so latent 0 <-> frame 0 alone, like Wan)
  "endpad"    : upstream encode_video default (pads the END to a multiple of 4)

Run from the repo root (PYTHONPATH as in load_smoke.py):
  .venv/bin/python benchmarks/tiny_vae_20260926/convention_probe_cpu.py --json <out.json>
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import argparse
import gc
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_smoke import (LIGHTTAE, LIGHTVAE, STOCK_VAE, TAEW2_1, WAN_CFG, TAEHV,  # noqa: E402
                        load_state, wan_mean_std, wan_vae)

VIDEO = ROOT / "benchmarks/pro_30fps_20260921/s30-reuse/video.mp4"


def read_frames(start: int, count: int, rows=(128, 320), cols=(64, 256)) -> torch.Tensor:
    cap = cv2.VideoCapture(str(VIDEO))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(count):
        ok, bgr = cap.read()
        if not ok:
            raise RuntimeError("short read")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[rows[0]:rows[1], cols[0]:cols[1]]
        frames.append(torch.from_numpy(rgb.copy()).permute(2, 0, 1))
    cap.release()
    return torch.stack(frames).float().div_(255)  # (T, 3, H, W) in [0, 1]


def interior(t: torch.Tensor, b: int = 16) -> torch.Tensor:
    return t[..., b:-b, b:-b]


def psnr01(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = (interior(a) - interior(b)).pow(2).mean().item()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def mae255(a, b) -> float:
    return (interior(a) - interior(b)).abs().mean().item() * 255


def mean_rgb_shift255(a, b) -> list[float]:
    # a, b: (T, 3, H, W) in [0, 1]
    return [round(v * 255, 2) for v in (interior(a) - interior(b)).mean(dim=(0, 2, 3)).tolist()]


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b).norm() / b.norm()).item()


def per_slot_rel(e: torch.Tensor, target: torch.Tensor) -> list[float]:
    # e, target: (1, 16, T, h, w)
    return [round(rel(e[:, :, i], target[:, :, i]), 3) for i in range(target.shape[2])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=30, help="first frame (0-based) of the 9")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    assert not torch.cuda.is_available()

    mean, std = wan_mean_std()
    scale = [mean, 1.0 / std]
    ms = mean.view(1, 16, 1, 1, 1), std.view(1, 16, 1, 1, 1)

    x01 = read_frames(args.start, 9)                       # (9, 3, 192, 192) in [0, 1]
    x = x01.mul(2).sub(1).permute(1, 0, 2, 3)[None]        # (1, 3, 9, H, W) in [-1, 1], NCTHW
    res: dict = {"hardware": "CPU only (CUDA hidden); not GPU inference",
                 "video": str(VIDEO.relative_to(ROOT)), "frames": [args.start, args.start + 8],
                 "crop_rows_cols": [[128, 320], [64, 256]], "metric_region": "160x160 interior"}

    # ---------------- stock Wan VAE reference (fp32, mmap'd weights) ----------------
    sd = torch.load(STOCK_VAE, map_location="cpu", weights_only=True, mmap=True)
    with torch.device("meta"):
        stock = wan_vae.WanVAE_(dim=96, **WAN_CFG)
    stock.load_state_dict(sd, assign=True)
    stock.eval()
    with torch.no_grad():
        z = stock.encode(x, scale)                          # (1, 16, 3, 24, 24) normalised
        z5 = stock.encode(x[:, :, -5:], scale)              # (1, 16, 2, 24, 24) motion analogue
        ref = stock.decode(z, scale).clamp_(-1, 1)          # (1, 3, 9, H, W)
    del stock, sd
    gc.collect()
    z_raw = z * ms[1] + ms[0]
    z5_raw = z5 * ms[1] + ms[0]
    ref01 = ref[0].permute(1, 0, 2, 3).add(1).div(2)        # (9, 3, H, W) in [0, 1]
    res["latent_stats"] = {
        "z_normalised_mean_std": [round(z.mean().item(), 3), round(z.std().item(), 3)],
        "z_raw_mean_std": [round(z_raw.mean().item(), 3), round(z_raw.std().item(), 3)],
    }
    res["stock_roundtrip_vs_source"] = {"psnr_db": round(psnr01(ref01, x01), 2),
                                        "mae_255": round(mae255(ref01, x01), 2)}
    print("latent stats", res["latent_stats"])
    print("stock Wan encode->decode vs source frames:", res["stock_roundtrip_vs_source"])

    # ---------------- TAEHV family ----------------
    for name, path in (("taew2_1", TAEW2_1), ("lighttaew2_1", LIGHTTAE)):
        t = TAEHV(checkpoint_path=str(path)).float().eval()
        entry = {"decode": {}, "encode": {}}
        with torch.no_grad():
            for conv, inp in (("normalised", z), ("unnormalised", z_raw)):
                out = t.decode_video(inp.transpose(1, 2).contiguous(), parallel=True,
                                     show_progress_bar=False)[0]      # (9, 3, H, W) [0, 1]
                entry["decode"][conv] = {
                    "psnr_vs_stock_decode_db": round(psnr01(out, ref01), 2),
                    "mae_vs_stock_decode_255": round(mae255(out, ref01), 2),
                    "psnr_vs_source_db": round(psnr01(out, x01), 2),
                    "mean_rgb_shift_vs_stock_255": mean_rgb_shift255(out, ref01),
                    "frames": out.shape[0],
                }
            v = x01[None]                                             # (1, 9, 3, H, W)
            front = torch.cat([v[:, :1].expand(-1, 3, -1, -1, -1), v], 1)
            for align, frames in (("frontpad3", front), ("endpad", v)):
                e = t.encode_video(frames, parallel=True, show_progress_bar=False).transpose(1, 2)
                entry["encode"][align] = {
                    "rel_err_vs_normalised_per_slot": per_slot_rel(e, z),
                    "rel_err_vs_unnormalised_per_slot": per_slot_rel(e, z_raw),
                }
            m5 = x01[None, -5:]
            for align, frames in (("motion5_frontpad3", torch.cat([m5[:, :1].expand(-1, 3, -1, -1, -1), m5], 1)),
                                  ("motion5_endpad", m5)):
                e5 = t.encode_video(frames, parallel=True, show_progress_bar=False).transpose(1, 2)
                entry["encode"][align] = {
                    "rel_err_vs_normalised_per_slot": per_slot_rel(e5, z5),
                    "rel_err_vs_unnormalised_per_slot": per_slot_rel(e5, z5_raw),
                }
        res[name] = entry
        print(f"== {name}")
        for k, d in entry.items():
            for kk, vv in d.items():
                print(f"  {k} {kk}: {vv}")
        del t
        gc.collect()

    # ---------------- LightVAE via SoulX's WanVAE_(dim=24) ----------------
    lv = wan_vae.WanVAE_(dim=24, **WAN_CFG).eval()
    lv.load_state_dict(load_state(LIGHTVAE), strict=True)
    with torch.no_grad():
        out = lv.decode(z, scale).clamp_(-1, 1)[0].permute(1, 0, 2, 3).add(1).div(2)
        e = lv.encode(x, scale)
        e5 = lv.encode(x[:, :, -5:], scale)
        rt = lv.decode(e, scale).clamp_(-1, 1)[0].permute(1, 0, 2, 3).add(1).div(2)
    res["lightvaew2_1"] = {
        "decode_normalised": {"psnr_vs_stock_decode_db": round(psnr01(out, ref01), 2),
                              "mae_vs_stock_decode_255": round(mae255(out, ref01), 2),
                              "psnr_vs_source_db": round(psnr01(out, x01), 2),
                              "mean_rgb_shift_vs_stock_255": mean_rgb_shift255(out, ref01)},
        "encode_rel_err_vs_normalised_per_slot": per_slot_rel(e, z),
        "encode_motion5_rel_err_vs_normalised_per_slot": per_slot_rel(e5, z5),
        "own_roundtrip_vs_source_psnr_db": round(psnr01(rt, x01), 2),
    }
    print("== lightvaew2_1")
    for k, v in res["lightvaew2_1"].items():
        print(f"  {k}: {v}")

    if args.json:
        args.json.write_text(json.dumps(res, indent=2))
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
