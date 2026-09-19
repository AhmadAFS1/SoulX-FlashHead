"""CPU policy contracts; real GEMM/error/reload checks live in the GPU benchmark."""
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from soulx_rtc.pro_quantization import Float8Linear, Int8ComputeLinear, quantize_pro_ffns


def model():
    m = nn.Module()
    m.config = SimpleNamespace(vae_stride=(4, 8, 8))
    m.patch_size = (1, 2, 2)
    m.blocks = nn.ModuleList([nn.Module(), nn.Module()])
    for block in m.blocks:
        block.ffn = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        block.cross_attn = nn.Linear(32, 32)
    m.head = nn.Linear(32, 16)
    return m.to(torch.bfloat16)


def test_disabled_policy_preserves_weights_and_module_identity():
    m = model()
    before = {name: value.clone() for name, value in m.state_dict().items()}
    linear = m.blocks[0].ffn[0]
    report = quantize_pro_ffns(m, "none")
    assert not report["modules"]
    assert m.blocks[0].ffn[0] is linear
    assert all(torch.equal(m.state_dict()[k], v) for k, v in before.items())


@pytest.mark.parametrize("scheme,cls", [("fp8", Float8Linear), ("int8", Int8ComputeLinear)])
def test_policy_protects_exclusions_head_and_audio_attention(scheme, cls):
    m = model()
    protected = m.blocks[0].ffn[2]
    head, audio = m.head, m.blocks[0].cross_attn
    report = quantize_pro_ffns(m, scheme, ["blocks.0.ffn.2"])
    assert len(report["modules"]) == 3
    assert isinstance(m.blocks[1].ffn[0], cls)
    assert m.blocks[0].ffn[2] is protected
    assert m.head is head and m.blocks[0].cross_attn is audio
    assert report["quantized_bytes"] < report["original_bytes"]


def test_wrong_geometry_and_unknown_exclusion_fail_before_changes():
    m = model()
    m.config.vae_stride = (8, 32, 32)
    with pytest.raises(ValueError, match="PRO geometry"):
        quantize_pro_ffns(m, "fp8")
    m.config.vae_stride = (4, 8, 8)
    with pytest.raises(ValueError, match="Unknown FFN"):
        quantize_pro_ffns(m, "fp8", ["head"])
    assert isinstance(m.blocks[0].ffn[0], nn.Linear)


@pytest.mark.parametrize("cls", [Float8Linear, Int8ComputeLinear])
def test_zero_weight_scales_are_finite_and_positive(cls):
    linear = nn.Linear(32, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        linear.weight.zero_()
    converted = cls(linear)
    assert torch.isfinite(converted.weight_scale).all()
    assert (converted.weight_scale > 0).all()
