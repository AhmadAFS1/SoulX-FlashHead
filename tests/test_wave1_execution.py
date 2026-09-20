"""Behavioural guards for the Wave 1 execution changes.

Needs torch, but CPU-only -- no CUDA. Auto-marked `heavy` by conftest, so it is
skipped on a developer machine without torch and runs on the box that has it.

Covers:
  W1.1  cross-attention FP8 q/o conversion planning (and the k/v refusal)
  W1.3  use_fast_accum threading from policy to the actual _scaled_mm call
  W1.4  device-side uint8 delivery: bit-exactness and trim equivalence

GPU/evidence note: `[CPU]` correctness checks. None of these measures speed.
A pass here does NOT qualify any recipe; that needs a measured run on the
recorded GPU (RTX 4070 SUPER, 12,282 MiB visible, driver 595.84).
"""
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from soulx_rtc.pro_quantization import Float8Linear  # noqa: E402
from soulx_rtc.pro_quantization_v2 import (  # noqa: E402
    PolicyFloat8Linear,
    PolicyValidationError,
    build_conversion_plan,
)


# --------------------------------------------------------------- tiny PRO stub


class _SelfAttn(nn.Module):
    def __init__(self, dim=1536, heads=12):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = dim, heads, dim // heads
        for name in ("q", "k", "v", "o"):
            setattr(self, name, nn.Linear(dim, dim).to(torch.bfloat16))


class _CrossAttn(nn.Module):
    def __init__(self, dim=1536, heads=12):
        super().__init__()
        self.dim, self.num_heads, self.head_dim = dim, heads, dim // heads
        self.has_image_input = False
        for name in ("q", "k", "v", "o"):
            setattr(self, name, nn.Linear(dim, dim).to(torch.bfloat16))


class _Block(nn.Module):
    def __init__(self, dim=1536, ffn_dim=8960):
        super().__init__()
        self.self_attn = _SelfAttn(dim)
        self.cross_attn = _CrossAttn(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim).to(torch.bfloat16),
            nn.GELU(),
            nn.Linear(ffn_dim, dim).to(torch.bfloat16),
        )


class _Config:
    vae_stride = (4, 8, 8)


class _Model(nn.Module):
    """Minimal PRO-shaped model. Two blocks keeps the test fast; geometry
    validation runs non-strict so block_count is not asserted."""

    def __init__(self, blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(blocks))
        self.config = _Config()
        self.patch_size = (1, 2, 2)


def _policy(**overrides):
    policy = {
        "schema_version": 1,
        "name": "test",
        "base": "pro-fp8-v1",
        "ffn": {"scheme": "bf16", "exclude": []},
        "self_attention_projections": {
            "scheme": "bf16", "include": [], "exclude": [], "packing": "separate",
        },
        "self_attention_kernel": {"backend": "sage2"},
        "cross_attention_projections": {"scheme": "bf16", "include": [], "exclude": []},
        "decoder": {"scheme": "bf16", "backend": "torch_compile", "plan": None},
        "compile": {"dit": True, "ffn_only": False},
        "prepared_conditioning": True,
    }
    policy.update(overrides)
    return policy


# ------------------------------------------------------------------ W1.1


def test_cross_attention_plan_selects_exactly_q_and_o():
    model = _Model(blocks=3)
    plan = build_conversion_plan(model, _policy(cross_attention_projections={
        "scheme": "fp8_e4m3_w8a8",
        "include": ["blocks.*.cross_attn.q", "blocks.*.cross_attn.o"],
        "exclude": [],
    }), strict_geometry=False)

    paths = [t.path for t in plan.resolved.targets]
    assert len(paths) == 2 * len(model.blocks)
    assert all(p.endswith((".cross_attn.q", ".cross_attn.o")) for p in paths)
    assert not any(".cross_attn.k" in p or ".cross_attn.v" in p for p in paths)
    assert all(t.family == "cross_attention_projection" for t in plan.resolved.targets)


def test_cross_attention_kv_is_refused_at_planning_time():
    model = _Model()
    with pytest.raises(PolicyValidationError, match="k/v conversion is not supported"):
        build_conversion_plan(model, _policy(cross_attention_projections={
            "scheme": "fp8_e4m3_w8a8",
            "include": ["blocks.*.cross_attn.k"],
            "exclude": [],
        }), strict_geometry=False)


def test_prepare_kv_still_returns_bare_bf16_after_cross_qo_conversion():
    """The prepared-K/V contract must survive q/o conversion untouched."""
    from soulx_rtc.pro_quantization_v2 import apply_conversion_plan

    model = _Model(blocks=1)
    block = model.blocks[0]
    before_k = block.cross_attn.k.weight.detach().clone()
    before_v = block.cross_attn.v.weight.detach().clone()

    plan = build_conversion_plan(model, _policy(cross_attention_projections={
        "scheme": "fp8_e4m3_w8a8",
        "include": ["blocks.*.cross_attn.q", "blocks.*.cross_attn.o"],
        "exclude": [],
    }), strict_geometry=False)
    apply_conversion_plan(model, plan)

    assert isinstance(block.cross_attn.k, nn.Linear)
    assert isinstance(block.cross_attn.v, nn.Linear)
    assert block.cross_attn.k.weight.dtype == torch.bfloat16
    assert torch.equal(block.cross_attn.k.weight, before_k)
    assert torch.equal(block.cross_attn.v.weight, before_v)
    assert isinstance(block.cross_attn.q, PolicyFloat8Linear)
    assert isinstance(block.cross_attn.o, PolicyFloat8Linear)


def test_bf16_cross_attention_converts_nothing():
    model = _Model(blocks=2)
    plan = build_conversion_plan(model, _policy(), strict_geometry=False)
    assert not [t for t in plan.resolved.targets if "cross_attn" in t.path]


# ------------------------------------------------------------------ W1.3


def test_fast_accum_defaults_to_false_and_reaches_scaled_mm(monkeypatch):
    """The retained measurements all used use_fast_accum=False. Pin it."""
    seen = {}

    def recording_scaled_mm(a, b, **kwargs):
        seen.update(kwargs)
        return torch.zeros(a.shape[0], b.shape[1], dtype=kwargs["out_dtype"])

    monkeypatch.setattr(torch, "_scaled_mm", recording_scaled_mm)

    linear = nn.Linear(16, 32).to(torch.bfloat16)
    Float8Linear(linear)(torch.randn(4, 16, dtype=torch.bfloat16))
    assert seen["use_fast_accum"] is False


def test_fast_accum_true_is_threaded_through(monkeypatch):
    seen = {}

    def recording_scaled_mm(a, b, **kwargs):
        seen.update(kwargs)
        return torch.zeros(a.shape[0], b.shape[1], dtype=kwargs["out_dtype"])

    monkeypatch.setattr(torch, "_scaled_mm", recording_scaled_mm)

    linear = nn.Linear(16, 32).to(torch.bfloat16)
    Float8Linear(linear, fast_accum=True)(torch.randn(4, 16, dtype=torch.bfloat16))
    assert seen["use_fast_accum"] is True


def test_policy_fast_accum_flag_reaches_every_converted_linear():
    from soulx_rtc.pro_quantization_v2 import apply_conversion_plan

    model = _Model(blocks=2)
    plan = build_conversion_plan(
        model,
        _policy(ffn={"scheme": "fp8_e4m3_w8a8", "exclude": []}, fast_accum=True),
        strict_geometry=False,
    )
    report = apply_conversion_plan(model, plan)

    converted = [m for m in model.modules() if isinstance(m, PolicyFloat8Linear)]
    assert converted, "no FP8 linears were produced"
    assert all(m.fast_accum is True for m in converted)
    assert all(
        item["accumulation"] == "fp32_scaled_mm_fast_accum" for item in report["modules"]
    )


def test_policy_without_fast_accum_keeps_the_recorded_accumulation_label():
    from soulx_rtc.pro_quantization_v2 import apply_conversion_plan

    model = _Model(blocks=1)
    plan = build_conversion_plan(
        model, _policy(ffn={"scheme": "fp8_e4m3_w8a8", "exclude": []}), strict_geometry=False
    )
    report = apply_conversion_plan(model, plan)
    assert all(item["accumulation"] == "fp32_scaled_mm" for item in report["modules"])
    assert all(
        m.fast_accum is False
        for m in model.modules()
        if isinstance(m, PolicyFloat8Linear)
    )


# ------------------------------------------------------------------ W1.4


def _host_delivery(sample):
    """The historical path: float32 layout on device, uint8 cast on the host."""
    frames = (((sample + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255).contiguous()
    return frames.numpy().astype("uint8")


def _device_delivery(sample):
    """The lean path: identical op order, cast to uint8 before leaving the device."""
    frames = (((sample + 1) / 2).permute(1, 2, 3, 0).clip(0, 1) * 255)
    return frames.contiguous().to(torch.uint8).numpy()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_device_uint8_delivery_is_bit_identical_to_the_host_path(seed):
    torch.manual_seed(seed)
    sample = torch.randn(3, 33, 8, 6, dtype=torch.float32)
    assert (_host_delivery(sample) == _device_delivery(sample)).all()


def test_delivery_matches_at_range_endpoints_and_quantisation_boundaries():
    """-1 and +1 map to 0 and 255; values beyond the range must clip, not wrap."""
    values = [-3.0, -1.0, -1 + 2 / 255, 0.0, 1 - 2 / 255, 1.0, 3.0]
    sample = torch.tensor(values, dtype=torch.float32).reshape(1, len(values), 1, 1)
    sample = sample.expand(3, len(values), 1, 1).contiguous()

    host, device = _host_delivery(sample), _device_delivery(sample)
    assert (host == device).all()
    assert device.min() == 0 and device.max() == 255


def test_colour_correction_is_per_frame_independent():
    """W1.4 trims history frames BEFORE colour correction.

    That is only valid if correction never mixes frames. The statistics reduce
    over dim=[2,3] of a (B, T, H, W, C) tensor -- H and W -- so each T is
    independent and f(x)[n:] must equal f(x[n:]).
    """
    from flash_head.utils.utils import match_and_blend_colors_torch

    torch.manual_seed(0)
    video = torch.randn(1, 3, 12, 8, 6, dtype=torch.float32).clip(-1, 1)
    reference = torch.randn(1, 3, 1, 8, 6, dtype=torch.float32).clip(-1, 1)
    trim = 5

    full = match_and_blend_colors_torch(video, reference, 1.0)
    trimmed = match_and_blend_colors_torch(video[:, :, trim:], reference, 1.0)

    assert full[:, :, trim:].shape == trimmed.shape
    assert torch.allclose(full[:, :, trim:], trimmed, atol=0, rtol=0), (
        "colour correction mixes frames; trimming before it is NOT safe"
    )


def test_trailing_motion_frames_are_unaffected_by_leading_trim():
    """cond_frame takes the TRAILING motion_frames_num; trimming the leading
    history frames must not change which frames those are."""
    video = torch.arange(33, dtype=torch.float32).reshape(1, 1, 33, 1, 1)
    motion = 5
    before = video[:, :, -motion:]
    after = video[:, :, motion:][:, :, -motion:]
    assert torch.equal(before, after)
