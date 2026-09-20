"""Policy-schema contract guards. Stdlib only -- no torch, no numpy.

`validate_policy` closes every level with `_reject_unknown_keys`, so any schema
edit can silently invalidate the retained policies that every published number
is keyed to. Nothing tested that before.

These checks reimplement only the *schema* rules, deliberately, so they run on a
CPU-only developer machine where `soulx_rtc.pro_quantization_v2` cannot be
imported (it imports torch at module scope). The behavioural half of the same
contract lives in test_cross_attention_fp8.py behind the `heavy` marker.

GPU/evidence note: `[CPU]` static checks. They gate correctness, never promotion.
"""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

POLICY_DIRS = [
    REPO / "benchmarks/pro_30fps_20260919/policies",
    REPO / "benchmarks/pro_quantization_v2_20260918/policies",
]

# Mirrors validate_policy's `required` set. "fast_accum" is intentionally absent.
REQUIRED_KEYS = {
    "schema_version", "name", "base", "ffn", "self_attention_projections",
    "self_attention_kernel", "cross_attention_projections", "decoder", "compile",
    "prepared_conditioning",
}
ALLOWED_KEYS = REQUIRED_KEYS | {"fast_accum"}


def _policies():
    found = []
    for directory in POLICY_DIRS:
        if directory.is_dir():
            found.extend(sorted(directory.glob("*.json")))
    return found


def test_policy_directories_exist():
    assert _policies(), f"no policy JSON found under {[str(d) for d in POLICY_DIRS]}"


@pytest.mark.parametrize("path", _policies(), ids=lambda p: p.name)
def test_every_retained_policy_still_satisfies_the_schema(path):
    """Adding an optional key must not invalidate any recorded policy."""
    policy = json.loads(path.read_text())
    missing = sorted(REQUIRED_KEYS - set(policy))
    assert not missing, f"{path.name} is missing required keys: {missing}"
    unknown = sorted(set(policy) - ALLOWED_KEYS)
    assert not unknown, f"{path.name} has keys the validator would reject: {unknown}"


@pytest.mark.parametrize("path", _policies(), ids=lambda p: p.name)
def test_cross_attention_never_selects_k_or_v(path):
    """prepare_kv returns bare BF16 tensors; k/v have nowhere to carry a scale."""
    cross = json.loads(path.read_text())["cross_attention_projections"]
    rules = list(cross.get("include", [])) + list(cross.get("exclude", []))
    offenders = [r for r in rules if r.endswith((".k", ".v")) or ".k_img" in r or ".v_img" in r]
    assert not offenders, f"{path.name} selects cross-attention k/v: {offenders}"


@pytest.mark.parametrize("path", _policies(), ids=lambda p: p.name)
def test_bf16_cross_attention_carries_no_rules(path):
    cross = json.loads(path.read_text())["cross_attention_projections"]
    if cross.get("scheme") == "bf16":
        assert not cross.get("include") and not cross.get("exclude"), (
            f"{path.name}: a bf16 cross-attention scheme must carry empty rules"
        )


@pytest.mark.parametrize("path", _policies(), ids=lambda p: p.name)
def test_quantized_cross_attention_names_its_targets(path):
    cross = json.loads(path.read_text())["cross_attention_projections"]
    if cross.get("scheme") != "bf16":
        assert cross.get("include"), (
            f"{path.name}: a quantized cross-attention scheme must name include rules"
        )


def test_the_retained_candidate_policy_is_untouched():
    """The selected 11.7830 FPS candidate must keep its exact recorded content.

    New experiments are separate policy files. Editing this one in place would
    silently re-key every published measurement that cites it.
    """
    path = REPO / "benchmarks/pro_30fps_20260919/policies/combined_fp16_sage_fp8.json"
    policy = json.loads(path.read_text())
    assert policy["name"] == "pro30-stage-fp16-native-sage-fp8"
    assert policy["cross_attention_projections"] == {
        "scheme": "bf16", "include": [], "exclude": [],
    }
    assert "fast_accum" not in policy, (
        "adding fast_accum to the retained candidate would change its policy_sha256"
    )
    assert policy["decoder"]["scheme"] == "fp16_stage_control"
    assert policy["self_attention_kernel"]["backend"] == "sage2"


def test_new_experiment_policies_differ_from_the_candidate_in_the_intended_way():
    directory = REPO / "benchmarks/pro_30fps_20260919/policies"
    base = json.loads((directory / "combined_fp16_sage_fp8.json").read_text())

    crossqo = json.loads((directory / "combined_fp16_sage_fp8_crossqo.json").read_text())
    differing = {k for k in set(base) | set(crossqo) if base.get(k) != crossqo.get(k)}
    assert differing == {"name", "cross_attention_projections"}, differing
    assert crossqo["cross_attention_projections"]["include"] == [
        "blocks.*.cross_attn.q", "blocks.*.cross_attn.o",
    ]

    fastaccum = json.loads((directory / "combined_fp16_sage_fp8_fastaccum.json").read_text())
    differing = {k for k in set(base) | set(fastaccum) if base.get(k) != fastaccum.get(k)}
    assert differing == {"name", "fast_accum"}, differing
    assert fastaccum["fast_accum"] is True
