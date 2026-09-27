#!/usr/bin/env python3
"""CPU-only smoke test: tiny / light autoencoders on Wan 2.1 latents for SoulX-FlashHead PRO.

Hardware: CPU only (CUDA is hidden via CUDA_VISIBLE_DEVICES=""). Nothing here is GPU
inference and no number printed here is a speed measurement: the MAC counts are a static
analysis at production shapes on the meta device, and every tensor value comes from random
inputs through the real weights.

Candidates (weights under models/tiny_vae/, see README-level facts in the report):
  * taew2_1        madebyollin/taehv, TAEHV architecture, fp16 checkpoint
  * lighttaew2_1   lightx2v/Autoencoders, same TAEHV architecture, fp32, distilled from taew2_1
  * lightvaew2_1   lightx2v/Autoencoders, Wan 2.1 VAE with every width x0.25 (pruning 75%),
                   instantiated with SoulX's OWN flash_head/wan/modules/vae.py WanVAE_(dim=24)

Layouts:
  * TAEHV: NTCHW, pixels in [0, 1] (decode output clamped to [0, 1]; encode input [0, 1]).
  * Wan (stock and LightVAE): NCTHW, pixels in [-1, 1]; WanVAE_.decode(z, scale) takes the
    NORMALISED latent (what the DiT produces) and un-normalises internally with
    z / scale[1] + scale[0] = z * std + mean.

Run from the repo root:
  export PYTHONPATH=/workspace/experiments/pro30-deps/sage-sm89-stream:.pro-quant-deps:/workspace/experiments/ojin-components-deps:.
  .venv/bin/python benchmarks/tiny_vae_20260926/load_smoke.py [--json out.json]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only, before torch is imported

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
TINY = ROOT / "models" / "tiny_vae"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TINY / "taehv"))  # upstream taehv.py (downloaded, unmodified)

from taehv import TAEHV, MemBlock, StreamingTAEHV, TGrow  # noqa: E402

from flash_head.wan.modules import vae as wan_vae  # noqa: E402  (READ-ONLY module)

STOCK_VAE = ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"
TAEW2_1 = TINY / "taehv" / "taew2_1.pth"
LIGHTTAE = TINY / "lightx2v" / "lighttaew2_1.safetensors"
LIGHTVAE = TINY / "lightx2v" / "lightvaew2_1.safetensors"

# SoulX's _video_vae config (flash_head/wan/modules/vae.py:934-942) with dim overridable.
WAN_CFG = dict(z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
               temperal_downsample=[False, True, True], dropout=0.0)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wan_mean_std() -> tuple[torch.Tensor, torch.Tensor]:
    """Read the 16 mean/std values from WanVAE.__init__ in SoulX's vae.py (not retyped)."""
    tree = ast.parse((ROOT / "flash_head/wan/modules/vae.py").read_text())
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "WanVAE":
            for sub in ast.walk(node):
                if isinstance(sub, ast.Assign) and isinstance(sub.targets[0], ast.Name) \
                        and sub.targets[0].id in ("mean", "std"):
                    found[sub.targets[0].id] = torch.tensor(ast.literal_eval(sub.value))
    return found["mean"], found["std"]


def rng(t: torch.Tensor) -> str:
    return f"min {t.min().item():+.4f} max {t.max().item():+.4f} mean {t.mean().item():+.4f}"


def load_state(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    return torch.load(path, map_location="cpu", weights_only=True)


# ---------------------------------------------------------------------------------------
# Window-by-window TAEHV decode with carried state, time-parallel inside each window.
# Upstream decode_video(parallel=True) zero-pads every MemBlock's "past" at the start of
# the call; here the past of the first timestep is the previous window's last timestep at
# that block (the exact quantity StreamingTAEHV keeps in `memory[i]`). The decoder has no
# TPool, and TGrow/Upsample/conv are per-timestep, so the MemBlock inputs are the only state.
# ---------------------------------------------------------------------------------------
def taehv_decode_window(taehv: TAEHV, x_ntchw: torch.Tensor, state: dict | None):
    """x: (N, T, 16, h, w) latents. Returns (frames NTCHW in [0,1], new_state).

    state None = cold start: output is trimmed by frames_to_trim (4T - 3 frames), exactly
    decode_video. Warm state: every latent yields t_upscale (4) frames, no trim.
    """
    n, t = x_ntchw.shape[:2]
    x = x_ntchw.reshape(n * t, *x_ntchw.shape[2:])
    new_state = {}
    for i, block in enumerate(taehv.decoder):
        if isinstance(block, MemBlock):
            nt, c, h, w = x.shape
            tt = nt // n
            cur = x.reshape(n, tt, c, h, w)
            first = state[i] if state is not None else torch.zeros_like(cur[:, :1])
            past = torch.cat([first, cur[:, :-1]], 1).reshape(x.shape)
            new_state[i] = cur[:, -1:].clone()
            x = block(x, past)
        else:
            x = block(x)
    nt = x.shape[0]
    out = x.view(n, nt // n, *x.shape[1:])
    out = taehv.postprocess_output_frames(out)
    if state is None:
        out = out[:, taehv.frames_to_trim:]
    return out, new_state


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


# ---------------------------------------------------------------------------------------
# Static conv/attention MAC count on the meta device (no memory, no compute).
# ---------------------------------------------------------------------------------------
class MacCounter:
    def __init__(self, model: nn.Module):
        self.total = 0
        self.by_prefix: dict[str, int] = {}
        self.handles = []
        for name, mod in model.named_modules():
            if isinstance(mod, (nn.Conv2d, nn.Conv3d)):
                self.handles.append(mod.register_forward_hook(self._conv_hook(name)))
            elif isinstance(mod, wan_vae.AttentionBlock):
                self.handles.append(mod.register_forward_hook(self._attn_hook(name)))

    def _add(self, name, macs):
        self.total += macs
        self.by_prefix[name] = self.by_prefix.get(name, 0) + macs

    def _conv_hook(self, name):
        def hook(mod, inp, out):
            k = 1
            for s in mod.weight.shape[2:]:
                k *= s
            self._add(name, out.numel() * mod.weight.shape[1] * k)
        return hook

    def _attn_hook(self, name):
        def hook(mod, inp, out):
            b, c, t, h, w = inp[0].shape
            self._add(name + ".sdpa", 2 * b * t * (h * w) ** 2 * c)  # QK^T and AV
        return hook

    def macs_under(self, prefixes) -> int:
        return sum(v for k, v in self.by_prefix.items() if any(k.startswith(p) for p in prefixes))

    def close(self):
        for h in self.handles:
            h.remove()


def wan_decoder_macs(dim: int, lat_h: int, lat_w: int, skip_blocks=()) -> dict:
    """First window (9 fresh latents -> 33 frames) and steady state under the shipping
    overlap-skip shim (2 latents warm the cache, then 7 latents -> 28 frames)."""
    with torch.device("meta"):
        m = wan_vae.WanVAE_(dim=dim, **WAN_CFG)
        z = torch.zeros(1, 16, 9, lat_h, lat_w)
        scale = [torch.zeros(16), torch.ones(16)]
    res = {}
    skip_prefixes = [f"decoder.upsamples.{i}." for i in skip_blocks]
    for label, warm, fresh in (("first_window_9lat_33f", 0, 9), ("steady_7lat_28f", 2, 7)):
        m.clear_cache()
        if warm:
            with torch.no_grad():
                m.cached_decode(z[:, :, :warm], scale)
            m._conv_idx = [0]
        c = MacCounter(m)
        with torch.no_grad():
            m.cached_decode(z[:, :, warm:warm + fresh], scale)
        c.close()
        res[label] = c.total - c.macs_under(skip_prefixes)
    return res


def wan_encoder_macs(dim: int, frames: int, height: int, width: int) -> int:
    with torch.device("meta"):
        m = wan_vae.WanVAE_(dim=dim, **WAN_CFG)
        x = torch.zeros(1, 3, frames, height, width)
        scale = [torch.zeros(16), torch.ones(16)]
    c = MacCounter(m)
    with torch.no_grad():
        m.encode(x, scale)
    c.close()
    return c.total


def taehv_macs(lat_h: int, lat_w: int) -> dict:
    with torch.device("meta"):
        t = TAEHV(checkpoint_path=None, arch_name="taew2_1")
    res = {}
    for label, n_lat in (("decode_first_window_9lat", 9), ("decode_steady_7lat_28f", 7)):
        with torch.device("meta"):
            x = torch.zeros(1, n_lat, 16, lat_h, lat_w)
        c = MacCounter(t.decoder)
        with torch.no_grad():
            taehv_decode_window(t, x, None)  # MACs identical warm/cold
        c.close()
        res[label] = c.total
    with torch.device("meta"):
        frames = torch.zeros(1, 8, 3, lat_h * 8, lat_w * 8)  # 5 motion frames end-padded to 8
    c = MacCounter(t.encoder)
    with torch.no_grad():
        t.encode_video(frames, parallel=True, show_progress_bar=False)
    c.close()
    res["encode_motion_8f_to_2lat"] = c.total
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None, help="optional results JSON path")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    out: dict = {"hardware": "CPU only (AMD Ryzen 9 7950X host; CUDA hidden). No GPU inference.",
                 "torch": torch.__version__}
    assert not torch.cuda.is_available()

    print("== files (models/tiny_vae) ==")
    files = {}
    for p in sorted(TINY.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            files[str(p.relative_to(ROOT))] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
            print(f"  {p.relative_to(ROOT)}  {p.stat().st_size:>10} B  sha256 {files[str(p.relative_to(ROOT))]['sha256']}")
    out["files"] = files

    mean, std = wan_mean_std()
    scale = [mean, 1.0 / std]
    out["wan_mean"], out["wan_std"] = mean.tolist(), std.tolist()

    # The test latent, as the task asks: NCTHW (1, 16, 3, 8, 8), N(0,1) like normalised DiT latents.
    z = torch.randn(1, 16, 3, 8, 8)
    z_ntchw = z.transpose(1, 2).contiguous()  # (1, 3, 16, 8, 8) for TAEHV
    z_raw = z * std.view(1, 16, 1, 1, 1) + mean.view(1, 16, 1, 1, 1)  # un-normalised

    # ---------------------------------------------------------------- TAEHV family
    tae = {}
    for name, path in (("taew2_1", TAEW2_1), ("lighttaew2_1", LIGHTTAE)):
        sd = load_state(path)
        m = TAEHV(checkpoint_path=str(path)).eval()  # arch guessed from filename: 16 ch, patch 1
        info = {
            "checkpoint_dtype": sorted({str(v.dtype) for v in sd.values()}),
            "tensors": len(sd),
            "params_total": count_params(m), "params_encoder": count_params(m.encoder),
            "params_decoder": count_params(m.decoder),
            "latent_channels": m.latent_channels, "patch_size": m.patch_size,
            "t_downscale": m.t_downscale, "t_upscale": m.t_upscale,
            "frames_to_trim": m.frames_to_trim,
        }
        with torch.no_grad():
            dec_par = m.decode_video(z_ntchw, parallel=True, show_progress_bar=False)
            dec_seq = m.decode_video(z_ntchw, parallel=False, show_progress_bar=False)
            dec_raw_in = m.decode_video(z_raw.transpose(1, 2).contiguous(), parallel=True,
                                        show_progress_bar=False)
            # Window-by-window with carried state == one call over all latents?
            a, st = taehv_decode_window(m, z_ntchw[:, :1], None)
            b, st = taehv_decode_window(m, z_ntchw[:, 1:], st)
            chunked = torch.cat([a, b], 1)
            # Official streaming wrapper, one latent at a time.
            s = StreamingTAEHV(m)
            frames = []
            for i in range(z_ntchw.shape[1]):
                f = s.decode(z_ntchw[:, i:i + 1])
                while f is not None:
                    frames.append(f)
                    f = s.decode()
            streamed = torch.cat(frames, 1)
            # Encode: 9 frames (Wan-style 1 + 4k) in [0, 1], NTCHW. Both paddings give 3 latents; only endpad is aligned with Wan (see convention probe).
            px = dec_par.clone()
            enc_endpad = m.encode_video(px, parallel=True, show_progress_bar=False)
            front = torch.cat([px[:, :1].expand(-1, 3, -1, -1, -1), px], 1)
            enc_frontpad = m.encode_video(front, parallel=True, show_progress_bar=False)
            # SoulX motion re-encode analogue: 5 frames -> 2 latents. Upstream end padding
            # (encode_video default) is the alignment that matches Wan's latents; Wan-style
            # front padding is WRONG for slots >= 1 (convention_probe_cpu_s*.json).
            enc_motion = m.encode_video(px[:, -5:], parallel=True, show_progress_bar=False)
        info.update({
            "decode_in_shape_NTCHW": list(z_ntchw.shape),
            "decode_out_shape_NTCHW": list(dec_par.shape), "decode_out_range": rng(dec_par),
            "decode_raw_frames_before_trim": dec_par.shape[1] + m.frames_to_trim,
            "parallel_vs_sequential_maxabs": (dec_par - dec_seq).abs().max().item(),
            "chunked_carried_state_vs_full_maxabs": (chunked - dec_par).abs().max().item(),
            "chunk_frame_counts": [a.shape[1], b.shape[1]],
            "streaming_vs_full_maxabs": (streamed - dec_par).abs().max().item(),
            "decode_of_unnormalised_input_range": rng(dec_raw_in),
            "encode_9f_endpad_shape": list(enc_endpad.shape),
            "encode_9f_frontpad3_shape": list(enc_frontpad.shape),
            "encode_5f_motion_endpad_shape": list(enc_motion.shape),
            "encode_out_range": rng(enc_frontpad),
        })
        tae[name] = info
        print(f"== {name} ({path.relative_to(ROOT)}) ==")
        for k, v in info.items():
            print(f"  {k}: {v}")
    ta = load_state(TAEW2_1)
    tl = load_state(LIGHTTAE)
    same_keys = set(ta) == set(tl) and all(ta[k].shape == tl[k].shape for k in ta)
    enc_same = all(torch.equal(ta[k].float(), tl[k].float()) for k in ta if k.startswith("encoder."))
    dec_diff = max((ta[k].float() - tl[k].float()).abs().max().item() for k in ta if k.startswith("decoder."))
    print(f"== taew2_1 vs lighttaew2_1 state dicts: same keys/shapes {same_keys}; "
          f"encoder weights identical (after fp16->fp32) {enc_same}; decoder max |dw| {dec_diff:.4f}")
    tae["taew2_1_vs_lighttaew2_1"] = {"same_keys_and_shapes": same_keys,
                                      "encoder_weights_identical": enc_same,
                                      "decoder_max_abs_weight_diff": dec_diff}
    out["taehv"] = tae

    # ---------------------------------------------------------------- LightVAE
    print("== lightvaew2_1 in SoulX's own WanVAE_ (flash_head/wan/modules/vae.py) ==")
    lsd = load_state(LIGHTVAE)
    lv = wan_vae.WanVAE_(dim=24, **WAN_CFG).eval()
    strict = lv.load_state_dict(lsd, strict=True)
    info = {"instantiated_as": "WanVAE_(dim=24, z_dim=16, dim_mult=[1,2,4,4], num_res_blocks=2, "
                               "temperal_downsample=[False,True,True])",
            "strict_load": str(strict),
            "params_total": count_params(lv), "params_encoder": count_params(lv.encoder) + count_params(lv.conv1),
            "params_decoder": count_params(lv.decoder) + count_params(lv.conv2),
            "checkpoint_dtype": sorted({str(v.dtype) for v in lsd.values()})}
    # What run.py --vae-weights does: load into the stock-width (dim=96) model, strict=False.
    with torch.device("meta"):
        stock_meta = wan_vae.WanVAE_(dim=96, **WAN_CFG)
    try:
        stock_meta.load_state_dict(lsd, strict=False, assign=True)
        info["run_py_vae_weights_path"] = "loaded (unexpected)"
    except RuntimeError as e:
        info["run_py_vae_weights_path"] = "RuntimeError size mismatch: " + str(e).splitlines()[1].strip()[:140]
    with torch.no_grad():
        y = lv.decode(z, scale).clamp_(-1, 1)  # NORMALISED latent in, as WanVAE.decode does
        lv.clear_cache()
        e = lv.encode(y, scale)
    info.update({"decode_in_shape_NCTHW": list(z.shape), "decode_out_shape_NCTHW": list(y.shape),
                 "decode_out_range": rng(y), "encode_9f_shape": list(e.shape), "encode_out_range": rng(e)})
    for k, v in info.items():
        print(f"  {k}: {v}")
    out["lightvaew2_1"] = info

    # ---------------------------------------------------------------- static MACs
    print("== static conv+attention MACs at 576x320 (latent 72x40), meta device ==")
    lat_h, lat_w = 72, 40
    macs = {
        "stock_wan_decoder": wan_decoder_macs(96, lat_h, lat_w),
        "shipping_pruned_decoder_skip_9_10_13_14": wan_decoder_macs(96, lat_h, lat_w, (9, 10, 13, 14)),
        "lightvae_decoder": wan_decoder_macs(24, lat_h, lat_w),
        "taehv_family": taehv_macs(lat_h, lat_w),
        "stock_wan_encoder_5f": wan_encoder_macs(96, 5, 576, 320),
        "lightvae_encoder_5f": wan_encoder_macs(24, 5, 576, 320),
    }
    for k, v in macs.items():
        if isinstance(v, dict):
            print("  " + k + ": " + ", ".join(f"{a} {b / 1e9:,.0f} GMAC" for a, b in v.items()))
        else:
            print(f"  {k}: {v / 1e9:,.0f} GMAC")
    out["static_macs"] = macs

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
