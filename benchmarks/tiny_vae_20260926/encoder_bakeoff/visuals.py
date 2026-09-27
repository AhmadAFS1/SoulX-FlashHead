#!/usr/bin/env python3
"""Labelled grid of what the DiT is conditioned on: the STOCK Wan decode of each arm's 2 motion
latents (5 frames), for two cond_frame clips. Rows: full frame 4 (slot 1), mouth crop x2 of
frame 0 (slot 0) and frame 4 (slot 1), and |arm - Wan round trip| x8 on the frame-4 mouth crop.
GPU; holds the SoulX lease. Fresh local GPU inference on the RTX 4070 SUPER."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoder_common as ec  # noqa: E402

bc = ec.bc
OUT = ec.EB / "visual"
PICKS = [("indian150-a", 4), ("tts-open-vowels", 6)]


def font(sz):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"):
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def main():
    from soulx_rtc.gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease(None)
    try:
        with torch.inference_mode():
            run()
    finally:
        lease.close()


def run():
    OUT.mkdir(exist_ok=True)
    items = {(w["fixture"], w["window"]): d for w, d in bc.load_latents()}
    stock = bc.load_stock()
    ref_img = ec.load_reference()
    tae = bc.load_taehv("taew2_1", dtype=torch.float16)
    ltae = bc.load_taehv("lighttaew2_1", dtype=torch.float16)
    lv, _ = bc.load_lightvae()
    wscale = list(bc.wan_scale("cuda", torch.bfloat16))
    encs = [("Wan 2.1 encoder (shipping)", stock.encode),
            ("taew2_1 encoder", ec.TAEHVEncoder(tae, 0, "01", "asis")),
            ("lightvaew2_1 encoder", ec.WanArchEncoder(lv, wscale, "pm1")),
            ("lighttaew2_1 encoder", ec.TAEHVEncoder(ltae, 0, "01", "raw2norm")),
            ("taew2_1, Wan-style front pad (wrong)", ec.TAEHVEncoder(tae, 3, "01", "asis"))]
    m = bc.MOUTH
    f_lab, f_small = font(14), font(13)
    for fixture, win in PICKS:
        z = items[(fixture, win)]["latent"].cuda()
        clip = ec.cond_frames(stock.decode(z), ref_img)
        cols = [("input cond_frame (stock decode, colour-corrected)", bc.to_u8(clip).cpu().numpy())]
        for name, enc in encs:
            cols.append((name + " -> Wan decode", bc.to_u8(stock.decode(enc(clip))).cpu().numpy()))
        f0 = stock.encode(clip[:, :, :1])
        fb = torch.cat([f0, z[:, -1:]], 1)
        cols.append(("REJECTED latent feedback last2-fix0 -> Wan decode", bc.to_u8(stock.decode(fb)).cpu().numpy()))
        wan_rt = cols[1][1]
        W, H = 320, 576
        mh, mw = (m[0].stop - m[0].start) * 2, (m[1].stop - m[1].start) * 2
        head = 62
        canvas = Image.new("RGB", (W * len(cols), head + H + 3 * mh + 3 * 22), (20, 20, 20))
        dr = ImageDraw.Draw(canvas)
        for j, (name, fr) in enumerate(cols):
            x = j * W
            words, lines, cur = name.split(), [], ""
            for wd in words:
                if len(cur) + len(wd) + 1 > 30:
                    lines.append(cur)
                    cur = wd
                else:
                    cur = (cur + " " + wd).strip()
            lines.append(cur)
            for k, ln in enumerate(lines[:3]):
                dr.text((x + 6, 3 + 19 * k), ln, fill=(255, 255, 255), font=f_lab)
            canvas.paste(Image.fromarray(fr[4]), (x, head))
            y = head + H
            for lab, img in (("mouth, frame 0 (slot 0)", fr[0][m[0], m[1]]),
                             ("mouth, frame 4 (slot 1)", fr[4][m[0], m[1]]),
                             ("|diff vs Wan round trip| x8, frame 4",
                              np.clip(np.abs(fr[4][m[0], m[1]].astype(np.int16) - wan_rt[4][m[0], m[1]].astype(np.int16)) * 8, 0, 255).astype(np.uint8))):
                dr.text((x + 6, y + 3), lab, fill=(200, 200, 200), font=f_small)
                canvas.paste(Image.fromarray(img).resize((mw, mh), Image.NEAREST), (x, y + 22))
                y += mh + 22
        path = OUT / f"motion_latents_{fixture}_w{win}.png"
        canvas.save(path, optimize=True)
        print("wrote", path, path.stat().st_size)


if __name__ == "__main__":
    main()
