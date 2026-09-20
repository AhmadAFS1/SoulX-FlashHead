"""Execute the real `validate_policy` on a CPU-only machine with no torch.

`soulx_rtc.pro_quantization_v2` imports torch at module scope, but `validate_policy`
itself is pure-Python schema logic that touches no tensor. A minimal stub lets the
ACTUAL validator run here rather than a reimplementation of it, which is what
test_policy_schema_contract.py has to settle for.

The stub supplies only what module scope needs: `nn.Module`/`nn.Linear` as base
classes and a few dtype sentinels. If the module ever grows real tensor work at
import time this test will fail loudly rather than silently pass on a fake.

GPU/evidence note: `[CPU]` static validation. Gates correctness, never promotion.
"""
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

POLICY_DIR = REPO / "benchmarks/pro_30fps_20260919/policies"
CANDIDATE = POLICY_DIR / "combined_fp16_sage_fp8.json"


@pytest.fixture(scope="module")
def validator():
    """Import the real validate_policy behind a minimal torch stub."""
    if "torch" in sys.modules:  # a real torch is present; use it
        from soulx_rtc.pro_quantization_v2 import PolicyValidationError, validate_policy
        return validate_policy, PolicyValidationError

    torch = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")

    class Module:
        def __init__(self, *a, **k):
            pass

        def register_buffer(self, *a, **k):
            pass

    class Linear(Module):
        pass

    nn.Module, nn.Linear = Module, Linear
    torch.nn = nn
    torch.bfloat16 = "bfloat16"
    torch.int32 = "int32"
    torch.float8_e4m3fn = "float8_e4m3fn"
    torch.int8 = "int8"
    torch.__version__ = "stub"
    torch.tensor = lambda *a, **k: None

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    try:
        from soulx_rtc.pro_quantization_v2 import PolicyValidationError, validate_policy
    finally:
        sys.modules.pop("torch", None)
        sys.modules.pop("torch.nn", None)
    return validate_policy, PolicyValidationError


def _mutated(**changes):
    policy = json.loads(CANDIDATE.read_text())
    policy.update(changes)
    return policy


@pytest.mark.parametrize(
    "name", sorted(p.name for p in POLICY_DIR.glob("*.json")) if POLICY_DIR.is_dir() else []
)
def test_every_policy_in_the_tree_validates(validator, name):
    validate_policy, _ = validator
    validate_policy(json.loads((POLICY_DIR / name).read_text()))


def test_cross_attention_q_and_o_are_accepted(validator):
    validate_policy, _ = validator
    validate_policy(_mutated(cross_attention_projections={
        "scheme": "fp8_e4m3_w8a8",
        "include": ["blocks.*.cross_attn.q", "blocks.*.cross_attn.o"],
        "exclude": [],
    }))


@pytest.mark.parametrize("rule", [
    "blocks.*.cross_attn.k",
    "blocks.*.cross_attn.v",
    "blocks.0.cross_attn.k",
    "blocks.*.cross_attn.k_img",
])
def test_cross_attention_k_and_v_are_rejected(validator, rule):
    """prepare_kv returns bare BF16 tensors -- there is no scale carrier."""
    validate_policy, PolicyValidationError = validator
    with pytest.raises(PolicyValidationError, match="k/v conversion is not supported"):
        validate_policy(_mutated(cross_attention_projections={
            "scheme": "fp8_e4m3_w8a8", "include": [rule], "exclude": [],
        }))


def test_quantized_cross_attention_without_include_is_rejected(validator):
    validate_policy, PolicyValidationError = validator
    with pytest.raises(PolicyValidationError, match="must name its include rules"):
        validate_policy(_mutated(cross_attention_projections={
            "scheme": "fp8_e4m3_w8a8", "include": [], "exclude": [],
        }))


def test_int8_cross_attention_is_rejected(validator):
    """INT8 lost decisively on this pipeline; do not let a policy select it here."""
    validate_policy, PolicyValidationError = validator
    with pytest.raises(PolicyValidationError, match="Unsupported cross-attention"):
        validate_policy(_mutated(cross_attention_projections={
            "scheme": "int8_w8a8", "include": ["blocks.*.cross_attn.q"], "exclude": [],
        }))


def test_bf16_cross_attention_with_rules_is_still_rejected(validator):
    validate_policy, PolicyValidationError = validator
    with pytest.raises(PolicyValidationError, match="must be empty"):
        validate_policy(_mutated(cross_attention_projections={
            "scheme": "bf16", "include": ["blocks.*.cross_attn.q"], "exclude": [],
        }))


def test_fast_accum_is_optional_and_boolean_typed(validator):
    validate_policy, _ = validator
    # Absent: every retained policy must keep validating, hash unchanged.
    policy = json.loads(CANDIDATE.read_text())
    assert "fast_accum" not in policy
    validate_policy(policy)
    # Present: accepted.
    validate_policy(_mutated(fast_accum=True))
    validate_policy(_mutated(fast_accum=False))


def test_unknown_top_level_keys_are_still_rejected(validator):
    """The schema stays closed; adding fast_accum must not open it."""
    validate_policy, PolicyValidationError = validator
    with pytest.raises(PolicyValidationError, match="Unknown keys in policy"):
        validate_policy(_mutated(turbo=True))
