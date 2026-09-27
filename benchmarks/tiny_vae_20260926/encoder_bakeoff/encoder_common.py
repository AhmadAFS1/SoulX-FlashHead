"""Shared pieces of the 576x320 motion re-encode (tiny-encoder) bake-off (2026-09-26).

What is being replaced: the per-window motion re-encode, flash_head_pipeline.py:502-504,
    self.latent_motion_frames = self.vae.encode(cond_frame)
with cond_frame = the trailing motion_frames_num = 5 frames of the window AFTER the lean trim
(videos[:, :, 5:], :481-482) and the colour correction against the reference image
(match_and_blend_colors_torch(videos, original_color_reference, 1.0), :483-485), shape
(1, 3, 5, 576, 320) bf16 in [-1, 1] NCTHW. WanVAE.encode returns (16, 2, 72, 40) bf16 latents
NORMALISED by the Wan mean/std (vae.py:791-796): slot 0 = frame 0 alone (cleared cache),
slot 1 = frames 1-4. The next window injects them as noise[:, :2] (:396, :449).

Candidates (all return the same contract: (16, 2, 72, 40) bf16, DiT-normalised):
  taew2_1       TAEHV encoder (models/tiny_vae/taehv/taehv.py, unmodified) + taew2_1.pth
  lighttaew2_1  same architecture + lightx2v lighttaew2_1.safetensors
  lightvaew2_1  SoulX's own WanVAE_(dim=24) + lightx2v lightvaew2_1.safetensors (Wan protocol)
Conventions are resolved empirically in quality.py (input range, frame padding/alignment,
output normalisation); nothing is assumed from READMEs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

EB = Path(__file__).resolve().parent
sys.path.insert(0, str(EB.parent / "decoder_bakeoff"))
import bakeoff_common as bc  # noqa: E402  (decoder bake-off helpers: loaders, wan_scale, metrics)
from taehv import MemBlock  # noqa: E402  (upstream, unmodified; path set by bakeoff_common)

ROOT = bc.ROOT
REF_IMAGE = ROOT / "benchmarks/pro_lite_150x_20260917/reference-150x.png"  # every fixture's reference
RESOLUTION = (576, 320)  # run.py:127 (H, W)
HIST = 5                 # motion_frames_num (run.py:91)


# ------------------------------------------------------------------------------ pipeline pieces
def load_reference(device="cuda", dtype=torch.bfloat16):
    """original_color_reference exactly as prepare_params builds it (flash_head_pipeline.py:244-247,
    use_face_crop=False -> Image.open().convert('RGB'))."""
    from PIL import Image
    from flash_head.utils.utils import resize_and_centercrop
    pil = Image.open(REF_IMAGE).convert("RGB")
    t = resize_and_centercrop(pil, RESOLUTION).to(device, dtype=dtype)  # 1 C 1 H W
    return (t / 255 - 0.5) * 2


def cond_frames(videos, reference):
    """The pipeline's cond_frame from a decoded (1,3,33,H,W) bf16 window: lean trim, colour
    correction (strength 1.0, bf16, no cached stats -- as generate() calls it), trailing 5."""
    from flash_head.utils.utils import match_and_blend_colors_torch
    v = videos[:, :, HIST:]
    v = match_and_blend_colors_torch(v, reference, 1.0)
    return v[:, :, -HIST:].contiguous()


# ------------------------------------------------------------------------------ TAEHV encoder
class TAEHVEncodeCore(nn.Module):
    """Time-parallel TAEHV encode of one clip, N = 1: (T, 3, H, W) in [0, 1] -> (T/4, 16, H/8, W/8).

    Identical to upstream apply_model_with_memblocks_parallel for a single video (zero memory at
    t = 0, one-step memory per MemBlock, TPool = channel-stack of consecutive timesteps + 1x1
    conv); checked against TAEHV.encode_video in quality.py. The motion re-encode is stateless
    (WanVAE.encode clears its cache per call), so no state crosses calls. A static graph of
    Conv2d/ReLU/cat/reshape, so it can be torch.compiled or exported to ONNX/TensorRT.
    """

    def __init__(self, tae):
        super().__init__()
        self.encoder = tae.encoder

    def forward(self, x):
        for b in self.encoder:
            if isinstance(b, MemBlock):
                past = torch.cat([torch.zeros_like(x[:1]), x[:-1]], 0)
                x = b(x, past)
            else:
                x = b(x)
        return x


class TAEHVEncoder:
    """SoulX motion-encode contract around a TAEHV encoder.

    front: how many copies of frame 0 are PREPENDED (0..3); the rest of the padding to a
           multiple of 4 is copies of the last frame APPENDED (front=0 == upstream encode_video,
           front=3 == Wan-style "frame 0 alone in slot 0").
    rng:   "01" feeds (x + 1) / 2 (TAEHV README: [0, 1]); "pm1" feeds x in [-1, 1] as is.
    out:   "asis" returns the encoder output as the DiT latent; "raw2norm" treats it as an
           un-normalised Wan latent and applies (e - mean) * inv_std with WanVAE's bf16 scale.
    """

    def __init__(self, tae, front=0, rng="01", out="asis", dtype=torch.float16, core=None,
                 memory_format=torch.contiguous_format):
        self.core = core if core is not None else TAEHVEncodeCore(tae)
        self.front, self.rng, self.out, self.dtype = front, rng, out, dtype
        mean, inv_std = bc.wan_scale("cuda", torch.bfloat16)  # WanVAE keeps them in bf16
        self.mean = mean.float().view(16, 1, 1, 1)
        self.inv_std = inv_std.float().view(16, 1, 1, 1)
        self.memory_format = memory_format

    def prep(self, clip):
        x = clip[0].transpose(0, 1).to(self.dtype)  # (T, 3, H, W)
        if self.rng == "01":
            x = (x + 1) * 0.5
        t = x.shape[0]
        n_pad = (-(t + self.front)) % 4
        parts = []
        if self.front:
            parts.append(x[:1].expand(self.front, -1, -1, -1))
        parts.append(x)
        if n_pad:
            parts.append(x[-1:].expand(n_pad, -1, -1, -1))
        x = torch.cat(parts, 0) if len(parts) > 1 else x
        return x.contiguous(memory_format=self.memory_format)

    def post(self, e):
        e = e.transpose(0, 1)  # (16, L, h, w)
        if self.out == "raw2norm":
            e = (e.float() - self.mean) * self.inv_std
        return e.to(torch.bfloat16).contiguous()

    def __call__(self, clip):
        return self.post(self.core(self.prep(clip)))


class WanArchEncoder:
    """A Wan-architecture encoder (stock WanVAE_ or LightVAE WanVAE_(dim=24)) with the pipeline's
    call: model.encode(video, scale).squeeze(0) (vae.py:1296-1298). scale=None -> raw mu."""

    def __init__(self, model, scale, rng="pm1"):
        self.model, self.scale, self.rng = model, scale, rng

    def __call__(self, clip):
        x = clip if self.rng == "pm1" else (clip + 1) * 0.5
        s = self.scale if self.scale is not None else [0.0, 1.0]
        return self.model.encode(x, s).squeeze(0)
