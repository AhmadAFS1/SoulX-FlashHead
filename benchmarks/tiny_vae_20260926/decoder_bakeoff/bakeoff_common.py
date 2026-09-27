"""Shared pieces of the 576x320 tiny-decoder bake-off (2026-09-26).

Decoders compared (all take SoulX DiT latents: (16, T, 72, 40) bf16, NORMALISED by the Wan
mean/std, exactly what flash_head_pipeline.py hands vae.decode):
  stock     WanVAE (flash_head/wan/modules/vae.py, READ-ONLY) with the stock Wan2.1_VAE.pth, bf16 eager
  shipping  same + distilled benchmarks/pro_30fps_20260922/distill/vae_skip9-10-13-14_ft4.pth
            + soulx_rtc.pro_decoder_ops.install_decoder_block_skip(vae, [9,10,13,14]), bf16 eager
            (the shipping build runs this module layout as FP16 TensorRT spans; here it is eager)
  taew2_1       TAEHV (models/tiny_vae/taehv/taehv.py, madebyollin) + taew2_1.pth, fp16
  lighttaew2_1  TAEHV architecture + lightx2v lighttaew2_1.safetensors, fp16
  lightvaew2_1  SoulX's own WanVAE_(dim=24) + lightx2v lightvaew2_1.safetensors, bf16

Every decoder returns (1, 3, F, 576, 320) in [-1, 1] (NCTHW), cast to bf16 like the pipeline's
decode output. Delivery conversion for metrics is the pipeline's lean path: float32,
((x + 1) / 2).clip(0, 1) * 255, truncating cast to uint8 (flash_head_pipeline.py:516-532),
without colour correction (this measures the decoder itself).
"""
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[3]
BK = Path(__file__).resolve().parent
TINY = ROOT / "models" / "tiny_vae"
LATENTS = ROOT / "benchmarks" / "tiny_vae_20260926" / "latents"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TINY / "taehv"))

from taehv import TAEHV, MemBlock  # noqa: E402  (upstream, unmodified)

STOCK_VAE = ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth"
SHIP_WEIGHTS = ROOT / "benchmarks/pro_30fps_20260922/distill/vae_skip9-10-13-14_ft4.pth"
SHIP_SKIP = [9, 10, 13, 14]
TAEW2_1 = TINY / "taehv" / "taew2_1.pth"
LIGHTTAE = TINY / "lightx2v" / "lighttaew2_1.safetensors"
LIGHTVAE = TINY / "lightx2v" / "lightvaew2_1.safetensors"
WAN_CFG = dict(z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
               temperal_downsample=[False, True, True], dropout=0.0)
MOUTH = (slice(193, 289), slice(86, 246))  # rows, cols of a (576, 320) frame
HIST = 5  # motion_frames_num: frames 0..4 of every window are history, 5..32 are delivered


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def gpu_snapshot() -> dict:
    try:
        q = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu,driver_version",
                                     "--format=csv,noheader"], text=True).strip()
        apps = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                                        "--format=csv,noheader"], text=True).strip()
        return {"gpu": q, "compute_apps": apps.splitlines()}
    except Exception as e:  # pragma: no cover
        return {"error": str(e)}


# ------------------------------------------------------------------------------ loaders
def load_stock(device="cuda"):
    from flash_head.wan.modules import WanVAE
    return WanVAE(vae_path=str(STOCK_VAE), dtype=torch.bfloat16, device=device)


def load_shipping(device="cuda"):
    """Stock WanVAE + ft4 weights (run.py --vae-weights semantics: strict=False) + block skip."""
    from soulx_rtc.pro_decoder_ops import install_decoder_block_skip
    vae = load_stock(device)
    override = torch.load(SHIP_WEIGHTS, map_location="cpu", weights_only=True)
    missing, unexpected = vae.model.load_state_dict(override, strict=False)
    assert not unexpected, unexpected[:5]
    rep = install_decoder_block_skip(vae, SHIP_SKIP)
    vae._bakeoff_manifest = {"weights": str(SHIP_WEIGHTS.relative_to(ROOT)), "sha256": sha256(SHIP_WEIGHTS),
                             "missing_keys": len(missing), "loaded_keys": len(override),
                             "block_skip": rep["decoder_block_skip"]["skipped"]}
    return vae


def load_lightvae(device="cuda", dtype=torch.bfloat16):
    from safetensors.torch import load_file
    from flash_head.wan.modules import vae as wan_vae
    m = wan_vae.WanVAE_(dim=24, **WAN_CFG)
    res = m.load_state_dict(load_file(str(LIGHTVAE)), strict=True)
    return m.eval().requires_grad_(False).to(device=device, dtype=dtype), str(res)


def load_taehv(which: str, device="cuda", dtype=torch.float16):
    path = {"taew2_1": TAEW2_1, "lighttaew2_1": LIGHTTAE}[which]
    m = TAEHV(checkpoint_path=str(path), arch_name="taew2_1")
    return m.eval().requires_grad_(False).to(device=device, dtype=dtype)


def wan_scale(device="cuda", dtype=torch.bfloat16):
    """[mean, 1/std] exactly as WanVAE builds it (vae.py:1007-1009), in the requested dtype."""
    from flash_head.wan.modules import WanVAE
    # Read the numbers without loading weights: the list literals live in WanVAE.__init__.
    import ast
    import inspect
    src = inspect.getsource(WanVAE.__init__)
    found = {}
    for node in ast.walk(ast.parse("class _X:\n" + "\n".join("    " + line for line in src.splitlines()))):
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in ("mean", "std"):
            found[node.targets[0].id] = ast.literal_eval(node.value)
    mean = torch.tensor(found["mean"], dtype=dtype, device=device)
    inv_std = 1.0 / torch.tensor(found["std"], dtype=dtype, device=device)
    return mean, inv_std


# ------------------------------------------------------------------------------ TAEHV core
class TAEHVDecodeCore(nn.Module):
    """Time-parallel TAEHV decode of one window with explicit per-MemBlock state.

    forward(x (T,16,h,w), *states) -> (frames (4T,3,8h,8w) in [0,1], *new_states)
    states[k] is the previous timestep's input of the k-th MemBlock ((1,C,h_k,w_k)); zeros
    reproduce upstream decode_video(parallel=True) exactly (it zero-pads the memory); the
    previous window's new_states continue the stream exactly (== StreamingTAEHV, checked on
    CPU in load_smoke.py to 7.5e-7). A plain static graph: Conv2d/ReLU/tanh/nearest
    Upsample/cat/reshape, so it can be torch.compiled or exported to ONNX/TensorRT.
    """

    def __init__(self, tae: TAEHV):
        super().__init__()
        self.decoder = tae.decoder
        self.mem_index = [i for i, b in enumerate(self.decoder) if isinstance(b, MemBlock)]
        self.trim = tae.frames_to_trim

    def state_shapes(self, t: int, h: int, w: int):
        """(1, C, h_k, w_k) per MemBlock: C = conv[0].in_channels / 2, spatial x2 per preceding Upsample."""
        shapes, up = [], 1
        for b in self.decoder:
            if isinstance(b, nn.Upsample):
                up *= int(b.scale_factor)
            elif isinstance(b, MemBlock):
                shapes.append((1, b.conv[0].in_channels // 2, h * up, w * up))
        return shapes

    def zero_states(self, t, h, w, device, dtype, memory_format=torch.contiguous_format):
        return [torch.zeros(s, device=device, dtype=dtype).contiguous(memory_format=memory_format)
                for s in self.state_shapes(t, h, w)]

    def forward(self, x, *states):
        new = []
        k = 0
        for b in self.decoder:
            if isinstance(b, MemBlock):
                past = torch.cat([states[k], x[:-1]], 0)
                new.append(x[-1:])
                x = b(x, past)
                k += 1
            else:
                x = b(x)
        return (x.clamp(0, 1), *new)


class TAEHVAdapter:
    """SoulX decode contract around TAEHVDecodeCore.

    conv: "norm" feeds the DiT latent as is; "unnorm" feeds z*std+mean (what WanVAE_.decode
    computes before conv2). Output is (1,3,F,H,W) bf16 in [-1,1].
    """

    def __init__(self, tae: TAEHV, conv: str, dtype=torch.float16, core=None, memory_format=torch.contiguous_format):
        self.core = core if core is not None else TAEHVDecodeCore(tae)
        self.base = TAEHVDecodeCore(tae)
        self.conv = conv
        self.dtype = dtype
        self.mean, self.inv_std = wan_scale("cuda", torch.float32)
        self.memory_format = memory_format
        self.state = None

    def prep(self, zs):
        z = zs.float()
        if self.conv == "unnorm":
            z = z / self.inv_std.view(16, 1, 1, 1) + self.mean.view(16, 1, 1, 1)
        x = z.permute(1, 0, 2, 3).to(self.dtype)  # (T,16,h,w)
        return x.contiguous(memory_format=self.memory_format)

    def raw(self, zs, states=None):
        """Untrimmed frames (4T,3,H,W) in [0,1] + new states."""
        x = self.prep(zs)
        if states is None:
            states = self.base.zero_states(x.shape[0], x.shape[2], x.shape[3], x.device, x.dtype, self.memory_format)
        out = self.core(x, *states)
        return out[0], list(out[1:])

    @staticmethod
    def to_wan(frames):  # (F,3,H,W) [0,1] -> (1,3,F,H,W) bf16 [-1,1]
        return frames.permute(1, 0, 2, 3).unsqueeze(0).mul(2).sub(1).clamp(-1, 1).to(torch.bfloat16).contiguous()

    def decode_window(self, zs, trim_start=None):
        """Cold 9-latent window: 4T raw frames, trimmed to 4T-3 (upstream trims the first 3)."""
        f, _ = self.raw(zs)
        s = self.base.trim if trim_start is None else trim_start
        return self.to_wan(f[s:s + f.shape[0] - self.base.trim])

    # stream mode (overlap-skip analogue): window 0 cold full, windows 1+ decode the trailing
    # `keep` latents with the carried state and prepend the previous 5-frame tail.
    def stream_reset(self):
        self.state = None
        self.tail = None

    def stream_decode(self, zs, keep=7):
        if self.state is None:
            f, st = self.raw(zs)
            out = self.to_wan(f[self.base.trim:])
        else:
            f, st = self.raw(zs[:, -keep:], self.state)
            out = torch.cat([self.tail, self.to_wan(f)], 2)
        self.state = [s.clone() for s in st]
        self.tail = out[:, :, -HIST:].clone()
        return out


class WanStream:
    """Overlap-skip analogue for a Wan-architecture model (soulx_rtc/pro_decoder_ops.py:243-368),
    eager: window 0 = clear_cache + cached_decode(all 9); windows 1+ restore the persisted
    _feat_map and cached_decode the trailing 7 latents; prepend the 5-frame tail."""

    def __init__(self, model, scale, cached=None):
        self.model, self.scale = model, scale
        self.cached = cached or model.cached_decode
        self.stream_reset()

    def stream_reset(self):
        self.map, self.tail, self.keep = None, None, None

    def stream_decode(self, zs):
        m = self.model
        if self.map is None:
            m.clear_cache()
            out = self.cached(zs.unsqueeze(0), self.scale).clamp_(-1, 1)
            self.keep = (int(out.shape[2]) - HIST) // 4
        else:
            m._feat_map = self.map
            m._conv_idx = [0]
            fresh = self.cached(zs[:, -self.keep:].unsqueeze(0), self.scale).clamp_(-1, 1)
            out = torch.cat([self.tail, fresh], 2)
        self.map = m._feat_map
        self.tail = out[:, :, -HIST:].detach().clone()
        m.clear_cache()  # hand the model back clean for window-mode decodes (new list; ours survives)
        return out.to(torch.bfloat16)


# ------------------------------------------------------------------------------ metrics
def to_u8(x):
    """(1,3,F,H,W) [-1,1] -> (F,H,W,3) uint8, the pipeline's lean delivery arithmetic."""
    f = x[0].to(torch.bfloat16).to(torch.float32)
    return (((f + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255).contiguous().to(torch.uint8)


def _luma(u):
    f = u.float()
    return 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def _grad(l):
    return ((l[:, :, 1:] - l[:, :, :-1]).abs().mean() + (l[:, 1:, :] - l[:, :-1, :]).abs().mean()).item()


def psnr(c, r):
    mse = ((c.float() - r.float()) ** 2).mean().item()
    return 99.0 if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def metrics(c, r):
    """c, r: (F,H,W,3) uint8 on the GPU. Returns the bake-off metric dict."""
    cf, rf = c.float(), r.float()
    cm, rm = c[:, MOUTH[0], MOUTH[1]], r[:, MOUTH[0], MOUTH[1]]
    lc, lr = _luma(c), _luma(r)
    fl_c = (cf[1:] - cf[:-1]).abs().mean().item()
    fl_r = (rf[1:] - rf[:-1]).abs().mean().item()
    off = (cf - rf).mean(dim=(0, 1, 2)).tolist()
    return {
        "psnr_full": psnr(c, r),
        "psnr_mouth": psnr(cm, rm),
        "mae_full": (cf - rf).abs().mean().item(),
        "sharp_full": _grad(lc) / _grad(lr),
        "sharp_mouth": _grad(_luma(cm)) / _grad(_luma(rm)),
        "flicker": fl_c / fl_r if fl_r > 0 else float("nan"),
        "flicker_ref_abs": fl_r,
        "colour_offset_rgb": off,
    }


def aggregate(rows):
    """rows: list of metric dicts -> mean and worst window per metric."""
    if not rows:
        return {}
    out = {"n_windows": len(rows)}
    for k in ("psnr_full", "psnr_mouth", "mae_full", "sharp_full", "sharp_mouth", "flicker"):
        v = [r[k] for r in rows]
        mean = sum(v) / len(v)
        if k.startswith("psnr"):
            worst = min(v)
        elif k == "mae_full":
            worst = max(v)
        else:  # ratio: worst = farthest from 1
            worst = max(v, key=lambda a: abs(a - 1))
        out[k] = {"mean": mean, "worst": worst}
    offs = [r["colour_offset_rgb"] for r in rows]
    out["colour_offset_rgb"] = {
        "mean": [sum(o[i] for o in offs) / len(offs) for i in range(3)],
        "worst_abs": [max(abs(o[i]) for o in offs) for i in range(3)],
    }
    return out


def load_latents(limit=None):
    idx = json.loads((LATENTS / "index.json").read_text())["windows"]
    items = []
    for w in idx[: limit or None]:
        d = torch.load(LATENTS / w["file"], map_location="cpu", weights_only=False)
        items.append((w, d))
    return items
