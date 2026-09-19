"""CPU-only contracts for strict v2 PRO quantization policies."""
from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from soulx_rtc.pro_quantization import Float8Linear
from soulx_rtc.pro_quantization_v2 import (
    BF16_SCHEME,
    FP8_SCHEME,
    PolicyValidationError,
    PolicyFloat8Linear,
    apply_conversion_plan,
    build_conversion_plan,
    load_policy,
    validate_policy,
)
from soulx_rtc.pro_attention_backends import (
    AttentionBackendError,
    ExplicitAttentionBackend,
    install_self_attention_backend,
    make_attention_backend,
    remove_self_attention_backend,
)
from benchmarks.pro_quantization_v2_20260918.sweep import make_schedule


def model():
    result = nn.Module()
    result.config = SimpleNamespace(vae_stride=(4, 8, 8))
    result.patch_size = (1, 2, 2)
    result.blocks = nn.ModuleList([nn.Module(), nn.Module()])
    for block in result.blocks:
        block.ffn = nn.Sequential(nn.Linear(32, 64), nn.GELU(), nn.Linear(64, 32))
        block.self_attn = nn.Module()
        block.self_attn.packed_qkv = None
        for name in ("q", "k", "v", "o"):
            setattr(block.self_attn, name, nn.Linear(32, 32))
        block.cross_attn = nn.Linear(32, 32)
    result.head = nn.Linear(32, 16)
    return result.to(torch.bfloat16)


def policy(*, attention_scheme=BF16_SCHEME, include=None, exclude=None):
    return {
        "schema_version": 1,
        "name": "test-policy",
        "base": "pro-fp8-v1",
        "ffn": {"scheme": FP8_SCHEME, "exclude": []},
        "self_attention_projections": {
            "scheme": attention_scheme,
            "include": [] if include is None else include,
            "exclude": [] if exclude is None else exclude,
            "packing": "separate",
        },
        "self_attention_kernel": {"backend": "flash2"},
        "cross_attention_projections": {"scheme": BF16_SCHEME, "include": [], "exclude": []},
        "decoder": {"scheme": BF16_SCHEME, "backend": "torch_compile", "plan": None},
        "compile": {"dit": True, "ffn_only": False},
        "prepared_conditioning": True,
    }


def test_invalid_later_target_causes_no_partial_mutation():
    candidate = model()
    originals = [block.ffn[0] for block in candidate.blocks]
    candidate.blocks[1].ffn[2] = nn.Linear(31, 32, dtype=torch.bfloat16)
    before = {name: tensor.clone() for name, tensor in candidate.state_dict().items()}

    with pytest.raises(PolicyValidationError, match="alignment"):
        build_conversion_plan(candidate, policy(), strict_geometry=False)

    assert all(block.ffn[0] is original for block, original in zip(candidate.blocks, originals))
    assert all(torch.equal(candidate.state_dict()[name], tensor) for name, tensor in before.items())


def test_exact_self_attention_coverage_and_exclusion_precedence():
    candidate = model()
    selected = ["blocks.*.self_attn.[qkv]", "blocks.0.self_attn.o"]
    plan = build_conversion_plan(
        candidate,
        policy(attention_scheme=FP8_SCHEME, include=selected, exclude=["blocks.1.self_attn.k"]),
        strict_geometry=False,
    )
    paths = [target.path for target in plan.resolved.targets]
    assert paths == [
        "blocks.0.ffn.0", "blocks.0.ffn.2", "blocks.0.self_attn.k", "blocks.0.self_attn.o",
        "blocks.0.self_attn.q", "blocks.0.self_attn.v", "blocks.1.ffn.0", "blocks.1.ffn.2",
        "blocks.1.self_attn.q", "blocks.1.self_attn.v",
    ]
    report = apply_conversion_plan(candidate, plan)
    assert isinstance(candidate.blocks[0].self_attn.q, Float8Linear)
    assert isinstance(candidate.blocks[0].self_attn.q, PolicyFloat8Linear)
    assert isinstance(candidate.blocks[1].self_attn.v, Float8Linear)
    assert isinstance(candidate.blocks[1].self_attn.k, nn.Linear)
    assert report["modules"][-1]["fallback"] is False
    assert report["modules"][0]["backend_version"] == torch.__version__
    assert "pro_quantization_v2_metadata" in candidate.blocks[0].ffn[0].state_dict()


def test_unknown_or_unmatched_rules_fail_before_conversion():
    candidate = model()
    unknown_key = policy()
    unknown_key["ffn"]["typo"] = True
    with pytest.raises(PolicyValidationError, match="Unknown keys"):
        validate_policy(unknown_key)

    unmatched = policy(attention_scheme=FP8_SCHEME, include=["blocks.*.self_attn.not_real"])
    with pytest.raises(PolicyValidationError, match="Unmatched"):
        build_conversion_plan(candidate, unmatched, strict_geometry=False)
    assert isinstance(candidate.blocks[0].ffn[0], nn.Linear)


def test_overlapping_rules_and_unknown_exclusions_fail_before_conversion():
    candidate = model()
    overlapping = policy(
        attention_scheme=FP8_SCHEME,
        include=["blocks.*.self_attn.q", "blocks.0.self_attn.q"],
    )
    with pytest.raises(PolicyValidationError, match="Duplicate self-attention include"):
        build_conversion_plan(candidate, overlapping, strict_geometry=False)
    bad_exclusion = policy()
    bad_exclusion["ffn"]["exclude"] = ["blocks.*.ffn.9"]
    with pytest.raises(PolicyValidationError, match="Unmatched FFN exclusion"):
        build_conversion_plan(candidate, bad_exclusion, strict_geometry=False)
    assert isinstance(candidate.blocks[0].ffn[0], nn.Linear)


def test_duplicate_json_keys_and_empty_attention_include_are_strict(tmp_path):
    source = tmp_path / "duplicate.json"
    source.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(PolicyValidationError, match="Duplicate JSON key"):
        load_policy(source)
    plan = build_conversion_plan(model(), policy(attention_scheme=FP8_SCHEME, include=[]), strict_geometry=False)
    assert all(target.family == "ffn" for target in plan.resolved.targets)


def test_bf16_attention_policy_cannot_silently_select_modules():
    with pytest.raises(PolicyValidationError, match="BF16 self-attention"):
        build_conversion_plan(
            model(), policy(include=["blocks.*.self_attn.q"]), strict_geometry=False
        )


def test_packed_qkv_and_double_conversion_are_rejected():
    candidate = model()
    candidate.blocks[0].self_attn.packed_qkv = nn.Linear(32, 96, dtype=torch.bfloat16)
    with pytest.raises(PolicyValidationError, match="packed_qkv"):
        build_conversion_plan(candidate, policy(), strict_geometry=False)

    candidate = model()
    plan = build_conversion_plan(candidate, policy(), strict_geometry=False)
    apply_conversion_plan(candidate, plan)
    with pytest.raises(PolicyValidationError, match="unmodified nn.Linear"):
        build_conversion_plan(candidate, policy(), strict_geometry=False)


def test_policy_validation_rejects_conflicts_and_preserves_input_data():
    candidate = policy(attention_scheme=FP8_SCHEME, include=[])
    original = deepcopy(candidate)
    candidate["compile"] = {"dit": False, "ffn_only": True}
    with pytest.raises(PolicyValidationError, match="ffn_only"):
        validate_policy(candidate)
    assert original["name"] == "test-policy"


class FakeSelfAttention(nn.Module):
    def __init__(self, head_dim=128):
        super().__init__()
        self.head_dim = head_dim
        self.use_usp = False
        self._pro_attention_backend = None


class FakeBackend(ExplicitAttentionBackend):
    name = "test"

    def __call__(self, q, k, v, num_heads, *, owner):
        return q


def test_explicit_attention_backend_only_attaches_to_self_attention():
    candidate = nn.Module()
    candidate.blocks = nn.ModuleList([nn.Module(), nn.Module()])
    for block in candidate.blocks:
        block.self_attn = FakeSelfAttention()
        block.cross_attn = nn.Linear(32, 32)
    backend = FakeBackend()
    installed = install_self_attention_backend(
        candidate, backend, self_attention_type=FakeSelfAttention
    )
    assert installed == ["blocks.0.self_attn", "blocks.1.self_attn"]
    assert all(block.self_attn._pro_attention_backend is backend for block in candidate.blocks)
    assert all(not hasattr(block.cross_attn, "_pro_attention_backend") for block in candidate.blocks)
    remove_self_attention_backend(candidate)
    assert all(block.self_attn._pro_attention_backend is None for block in candidate.blocks)


def test_attention_installer_rejects_unsupported_head_geometry():
    candidate = nn.Module()
    candidate.blocks = nn.ModuleList([nn.Module()])
    candidate.blocks[0].self_attn = FakeSelfAttention(head_dim=64)
    with pytest.raises(AttentionBackendError, match="head dimension"):
        install_self_attention_backend(
            candidate, FakeBackend(), self_attention_type=FakeSelfAttention
        )


def test_sweep_schedule_balances_reference_candidate_order():
    schedule = make_schedule([50, 51], 1)
    assert [item["role"] for item in schedule] == ["reference", "candidate", "candidate", "reference"]
    assert [item["sequence"] for item in schedule] == [0, 1, 2, 3]

def test_sage2_fp16_backend_selects_exact_public_kernel_without_importing_package():
    backend = make_attention_backend("sage2_fp16")
    manifest = backend.manifest()
    assert manifest["backend"] == "sageattention2_int8_qk_fp16_pv"
    assert manifest["kernel_symbol"] == "sageattn_qk_int8_pv_fp16_cuda"
    assert manifest["pv_accum_dtype"] == "fp16+fp32"
    assert manifest["fallback"] is False
