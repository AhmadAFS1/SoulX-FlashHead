"""Side-by-side sample recordings: Wan 2.1 VAE decoder vs taew2_1 on IDENTICAL latents.

The pipeline runs in benchmarks/pro_30fps_20260922/tae-* differ in more than the decoder:
the motion re-encode feeds decoded pixels back into the DiT, so a decoder swap also changes
every later window's latents. For a clean look at the decoder alone, this script takes the
DiT latents dumped from one real 576x320 run per fixture (benchmarks/tiny_vae_20260926/latents,
9 windows each, no overlap-skip) and decodes every window twice:

  left   stock Wan 2.1 VAE decoder (full, bf16, eager) -- the model's own decoder
  right  taew2_1 (TAEHV, fp16, eager), convention resolved in the decoder bake-off:
         DiT-normalised latent as is, NTCHW, output [0,1], first 3 raw frames trimmed

Each window gives 33 frames and delivers frames 5..32, exactly as the pipeline trims the 5
history frames (flash_head_pipeline.py, videos[:, :, motion_frames_num:]); 9 x 28 = 252 ->
the first 250 frames, 25 fps, muxed with the fixture's audio.

Outputs per fixture (benchmarks/tiny_vae_20260926/samples/):
  <fx>-wan21-vs-taew21-full.mp4    full frame, side by side
  <fx>-wan21-vs-taew21-mouth.mp4   mouth crop (the review's rows 193:289, cols 86:246), 4x
  <fx>-wan21-vs-taew21-stills.png  mouth crops at 4 frames, pixel-exact 4x (nearest)
  metrics.json                     PSNR of taew2_1 vs Wan 2.1 over the delivered frames
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
LAT = ROOT / "benchmarks/tiny_vae_20260926/latents"
OUT = ROOT / "benchmarks/tiny_vae_20260926/samples"
FIXTURES = json.loads((ROOT / "benchmarks/pro_30fps_20260920/tts-fixtures/fixtures.json").read_text())
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
HISTORY, WINDOWS, FRAMES, FPS = 5, 9, 250, 25
MOUTH = (slice(193, 289), slice(86, 246))  # rows, cols
LEFT = ("Wan 2.1 VAE decoder", "stock, full - the model's own decoder")
RIGHT = ("taew2_1 tiny decoder", "same latents - ~12x faster decode")


def audio_path(fixture: str) -> Path:
    items = FIXTURES if isinstance(FIXTURES, list) else FIXTURES.get("fixtures", FIXTURES)
    items = items if isinstance(items, list) else list(items.values())
    for it in items:
        if it.get("id") == fixture:
            return ROOT / it["audio"]["path"]
    raise KeyError(fixture)


def to_u8(frames: torch.Tensor) -> np.ndarray:
    """(F,3,H,W) in [0,1] -> (F,H,W,3) uint8 (same rounding as the harness)."""
    return (frames.float().clamp(0, 1).mul(255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())


def header(width: int, title: str, sub: str, colour, scale: float = 1.0) -> np.ndarray:
    h = int(58 * scale)
    img = Image.new("RGB", (width, h), (0, 0, 0))
    d = ImageDraw.Draw(img)
    d.text((10, int(6 * scale)), title, font=ImageFont.truetype(FONT, int(19 * scale)), fill=colour)
    d.text((10, int(32 * scale)), sub, font=ImageFont.truetype(FONT, int(14 * scale)), fill=(170, 170, 170))
    return np.asarray(img)


def encode(frames: np.ndarray, audio: Path, path: Path, crf: int) -> None:
    n, h, w, _ = frames.shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(FPS), "-i", "-", "-i", str(audio), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
           "-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
           "-shortest", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.stdin.write(np.ascontiguousarray(frames).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255.0 ** 2 / max(mse, 1e-12)))


def main(fixtures: list[str]) -> None:
    from soulx_rtc.gpu_lease import acquire_gpu_lease
    from soulx_rtc.pro_tiny_vae import TAEHVDecodeCore, load_tiny_model, resolve_tiny_vae
    from flash_head.wan.modules import WanVAE

    OUT.mkdir(parents=True, exist_ok=True)
    lease = acquire_gpu_lease(None)
    metrics = {}
    try:
        wan = WanVAE(vae_path=str(ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"),
                     dtype=torch.bfloat16, device="cuda")
        info = resolve_tiny_vae("taew2_1")
        tae, taehv = load_tiny_model(info)
        core = TAEHVDecodeCore(tae, taehv.MemBlock).eval()
        for fx in fixtures:
            left, right = [], []
            for w in range(WINDOWS):
                z = torch.load(LAT / f"{fx}-s50-w{w}.pt", map_location="cpu", weights_only=False)["latent"]
                z = z.to("cuda", torch.bfloat16)  # (16, 9, 72, 40), DiT-normalised
                with torch.no_grad():
                    a = wan.decode(z)  # (1,3,33,576,320) in [-1,1]
                    a = a[0].permute(1, 0, 2, 3).add(1).div(2)  # (33,3,H,W) [0,1]
                    x = z.permute(1, 0, 2, 3).to(torch.float16).contiguous()
                    b = core(x, *core.zero_states(z.shape[2], z.shape[3], "cuda", torch.float16))[0][core.trim:]
                if a.shape[0] != 33 or b.shape[0] != 33:
                    raise RuntimeError(f"frame count {a.shape[0]} / {b.shape[0]} != 33")
                left.append(to_u8(a[HISTORY:]))
                right.append(to_u8(b[HISTORY:]))
            A = np.concatenate(left)[:FRAMES]
            B = np.concatenate(right)[:FRAMES]
            n, H, W, _ = A.shape
            m = {"frames": int(n), "psnr_full_db": psnr(A, B),
                 "psnr_mouth_db": psnr(A[:, MOUTH[0], MOUTH[1]], B[:, MOUTH[0], MOUTH[1]])}
            metrics[fx] = m
            print(fx, json.dumps(m), flush=True)

            # full frame, side by side
            gap = np.zeros((H, 8, 3), np.uint8)
            hl = header(W, *LEFT, (255, 255, 255))
            hr = header(W, *RIGHT, (255, 210, 74))
            top = np.concatenate([hl, np.zeros((hl.shape[0], 8, 3), np.uint8), hr], 1)
            full = np.stack([np.concatenate([top, np.concatenate([A[i], gap, B[i]], 1)], 0) for i in range(n)])
            encode(full, audio_path(fx), OUT / f"{fx}-wan21-vs-taew21-full.mp4", crf=14)

            # mouth crop, 4x (lanczos), stacked vertically
            def zoom(img):
                crop = Image.fromarray(img[MOUTH[0], MOUTH[1]])
                return np.asarray(crop.resize((crop.width * 4, crop.height * 4), Image.LANCZOS))
            mh = header(640, f"1. {LEFT[0]}", "mouth crop 4x", (255, 255, 255))
            th = header(640, f"2. {RIGHT[0]}", "mouth crop 4x, same latents", (255, 210, 74))
            mouth = np.stack([np.concatenate([mh, zoom(A[i]), th, zoom(B[i])], 0) for i in range(n)])
            encode(mouth, audio_path(fx), OUT / f"{fx}-wan21-vs-taew21-mouth.mp4", crf=12)

            # stills: 4 frames with the widest open mouth (largest dark area in the lip region), nearest 4x
            dark = [(A[i, 225:275, 120:210].mean(-1) < 70).sum() for i in range(n)]
            picks = sorted(np.argsort(dark)[::-1][:1].tolist() + [n // 5, n // 2, (4 * n) // 5])
            rows = []
            for i in picks:
                pa = Image.fromarray(A[i, MOUTH[0], MOUTH[1]]).resize((640, 384), Image.NEAREST)
                pb = Image.fromarray(B[i, MOUTH[0], MOUTH[1]]).resize((640, 384), Image.NEAREST)
                lab = header(1288, f"frame {i}   (left: {LEFT[0]}   |   right: {RIGHT[0]})", "pixel-exact 4x", (255, 255, 255), 0.9)
                rows.append(np.concatenate([lab, np.concatenate([np.asarray(pa), np.zeros((384, 8, 3), np.uint8), np.asarray(pb)], 1)], 0))
            Image.fromarray(np.concatenate(rows, 0)).save(OUT / f"{fx}-wan21-vs-taew21-stills.png", optimize=True)
            m["still_frames"] = [int(i) for i in picks]
        (OUT / "metrics.json").write_text(json.dumps({
            "what": "taew2_1 vs stock Wan 2.1 decoder on identical DiT latents; delivered frames only",
            "hardware": "fresh local GPU inference, NVIDIA GeForce RTX 4070 SUPER",
            "per_fixture": metrics}, indent=2) + "\n")
    finally:
        lease.close()


if __name__ == "__main__":
    main(sys.argv[1:] or ["indian150-a", "tts-plosives"])
