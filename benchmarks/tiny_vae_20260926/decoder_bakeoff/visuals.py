"""PNG grids (full frame + 4x mouth crop per decoder) and a labelled side-by-side mp4 (CPU)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import bakeoff_common as bc

OUT = bc.BK / "visual"


def _label(img, text, scale=0.55, h=26):
    bar = np.full((h, img.shape[1], 3), 24, np.uint8)
    cv2.putText(bar, text, (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, scale, (235, 235, 235), 1, cv2.LINE_AA)
    return np.concatenate([bar, img], 0)


def _mouth(frame, k):
    m = frame[bc.MOUTH[0], bc.MOUTH[1]]
    return cv2.resize(m, (m.shape[1] * k, m.shape[0] * k), interpolation=cv2.INTER_NEAREST)


def pick_frame(ref):
    """Delivered frame (5..32) with the darkest mouth interior = widest open mouth."""
    lum = ref[bc.HIST:, 225:262, 130:200].astype(np.float32).mean(axis=(1, 2, 3))
    return bc.HIST + int(lum.argmin())


def grid(frames: dict, t: int, title: str, psnr_mouth: dict):
    cols = []
    for name, v in frames.items():
        f = v[t]
        full = np.full((576, 640, 3), 16, np.uint8)
        full[:, 160:480] = f
        crop = _mouth(f, 4)  # 384 x 640
        txt = name if name not in psnr_mouth else f"{name}  mouth PSNR {psnr_mouth[name]:.2f} dB"
        cols.append(_label(np.concatenate([full, crop], 0), txt, 0.5))
    img = np.concatenate([np.concatenate([c, np.full((c.shape[0], 4, 3), 60, np.uint8)], 1) for c in cols], 1)
    return _label(img, title, 0.7, 34)


def mouth_zoom(frames: dict, t: int, title: str):
    tiles = [_label(_mouth(v[t], 3), n, 0.5) for n, v in frames.items()]
    blank = np.zeros_like(tiles[0])
    while len(tiles) % 3:
        tiles.append(blank)
    rows = [np.concatenate(tiles[i:i + 3], 1) for i in range(0, len(tiles), 3)]
    return _label(np.concatenate(rows, 0), title, 0.6, 30)


def make(keep_frames, video_key, rows_by_key, window_rows):
    OUT.mkdir(parents=True, exist_ok=True)
    made = []
    name_to_row = {"shipping (pruned+ft4, bf16)": "shipping", "taew2_1 (fp16)": "taew2_1",
                   "lighttaew2_1 (fp16)": "lighttaew2_1", "lightvaew2_1 (bf16)": "lightvaew2_1"}
    for key, fr in keep_frames.items():
        frames = {k: v.numpy() for k, v in fr.items()}
        ref = frames["reference (stock Wan, bf16)"]
        t = pick_frame(ref)
        i = rows_by_key[key]
        pm = {n: window_rows[r][i]["psnr_mouth"] for n, r in name_to_row.items()}
        title = (f"{key[0]} window {key[1]}, frame {t}/32 (delivered frames are 5..32). Top: full 576x320 frame; "
                 f"bottom: mouth crop rows 193:289 cols 86:246 at 4x nearest. Fresh local GPU inference, RTX 4070 SUPER")
        g = grid(frames, t, title, pm)
        p = OUT / f"grid_{key[0]}_w{key[1]}_f{t}.png"
        cv2.imwrite(str(p), cv2.cvtColor(g, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        z = mouth_zoom(frames, t, f"{key[0]} w{key[1]} f{t}: mouth crop at 3x")
        pz = OUT / f"mouth3x_{key[0]}_w{key[1]}_f{t}.png"
        cv2.imwrite(str(pz), cv2.cvtColor(z, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 9])
        made += [str(p.relative_to(bc.ROOT)), str(pz.relative_to(bc.ROOT))]
        if key == video_key:
            made.append(video(frames, key))
    return made


def video(frames, key, fps=10):
    import imageio.v2 as imageio
    p = OUT / f"sbs_{key[0]}_w{key[1]}_33f_{fps}fps.mp4"
    writer = imageio.get_writer(str(p), fps=fps, codec="libx264", quality=None,
                                ffmpeg_params=["-crf", "16", "-pix_fmt", "yuv420p", "-preset", "slow"])
    names = list(frames)
    n = frames[names[0]].shape[0]
    for t in range(n):
        cols = []
        for name in names:
            f = frames[name][t]
            m = _mouth(f, 2)  # 192 x 320
            cols.append(_label(np.concatenate([f, m], 0), name.split(" (")[0], 0.45, 22))
        img = np.concatenate(cols, 1)
        img = _label(img, f"{key[0]} window {key[1]} frame {t:2d}/32 ({'history' if t < bc.HIST else 'delivered'}); "
                          f"playback {fps} fps (0.4x); fresh local GPU inference, RTX 4070 SUPER", 0.5, 26)
        h, w = img.shape[:2]
        img = np.pad(img, ((0, h % 2), (0, w % 2), (0, 0)))
        writer.append_data(img)
    writer.close()
    return str(p.relative_to(bc.ROOT))
