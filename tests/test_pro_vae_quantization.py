"""CPU contracts for the cache-safe Wan decoder adapter."""
from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from benchmarks.pro_quantization_v2_20260918.capture import BoundedActivationCapture, CaptureBudgetExceeded
from benchmarks.pro_quantization_v2_20260918.make_decoder_plan import build_plan, protected_paths
from flash_head.wan.modules.vae import CausalConv3d, count_conv3d
from soulx_rtc.pro_vae_quantization import (
    BF16ConvolutionBackend,
    DecoderPlan,
    DecoderPlanError,
    DecoderTarget,
    decoder_module_for_path,
    fit_symmetric_scale,
    install_decoder_adapters,
    load_decoder_plan,
    remove_decoder_adapters,
)


class TinyVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(CausalConv3d(2, 2, 3, padding=1), nn.SiLU())


class WrappedTinyVAE:
    def __init__(self):
        self.model = TinyVAE()


def plan(path="decoder.0", scheme="bf16"):
    return DecoderPlan(
        "tiny",
        scheme,
        (DecoderTarget(path, "causal_conv3d", ((1, 2, 1, 4, 4),), {"scale": 1.0}),),
        (),
        {},
    )


def test_bf16_adapter_preserves_causal_module_identity_and_cache_math():
    torch.manual_seed(3)
    vae = TinyVAE().eval()
    layer = vae.decoder[0]
    x = torch.randn(1, 2, 1, 4, 4)
    cache = torch.randn(1, 2, 2, 4, 4)
    expected = layer(x, cache)
    original_count = count_conv3d(vae.decoder)

    report = install_decoder_adapters(vae, plan(), {"decoder.0": BF16ConvolutionBackend()})
    actual = layer(x, cache)
    assert isinstance(layer, CausalConv3d)
    assert count_conv3d(vae.decoder) == original_count
    assert torch.equal(actual, expected)
    assert report["modules"][0]["backend"] == "pytorch_bf16_control"

    remove_decoder_adapters(vae)
    assert torch.equal(layer(x, cache), expected)


def test_decoder_path_resolution_handles_the_non_module_wan_style_wrapper():
    wrapped = WrappedTinyVAE()
    assert decoder_module_for_path(wrapped, "model.decoder.0") is wrapped.model.decoder[0]


def test_adapter_validation_happens_before_any_module_is_wrapped():
    vae = TinyVAE()
    invalid = DecoderPlan(
        "bad",
        "bf16",
        (
            DecoderTarget("decoder.0", "causal_conv3d", ((1, 2, 1, 4, 4),), {}),
            DecoderTarget("decoder.1", "causal_conv3d", ((1, 2, 1, 4, 4),), {}),
        ),
        (),
        {},
    )
    with pytest.raises(DecoderPlanError, match="Expected CausalConv3d"):
        install_decoder_adapters(
            vae,
            invalid,
            {"decoder.0": BF16ConvolutionBackend(), "decoder.1": BF16ConvolutionBackend()},
        )
    assert not hasattr(vae.decoder[0], "_pro_decoder_original_forward")


def test_quantized_plan_refuses_dequantized_bf16_backend():
    vae = TinyVAE()
    with pytest.raises(DecoderPlanError, match="not real"):
        install_decoder_adapters(
            vae,
            plan(scheme="int8_conservative"),
            {"decoder.0": BF16ConvolutionBackend()},
        )


def test_calibration_scales_are_positive_and_finite():
    fitted = fit_symmetric_scale([torch.zeros(4), torch.tensor([-2.0, 1.0])], quant_max=127)
    assert fitted["scale"] > 0
    assert fitted["maximum"] == 2.0


def test_capture_budget_is_bounded_and_writes_a_failure_manifest(tmp_path):
    collector = BoundedActivationCapture(tmp_path / "manifest.json", max_raw_mib=1)
    collector.representative_paths.add("projection")
    tensor = torch.zeros(1, 1, 300_000, dtype=torch.bfloat16)
    collector.record("projection", "self_projection_input", tensor)
    with pytest.raises(CaptureBudgetExceeded):
        collector.record("projection", "self_projection_input", tensor)
    result = collector.finalize(status="failed")
    assert result["raw_budget_exceeded"] is True
    assert result["estimated_raw_bytes"] > result["raw_budget_bytes"]


def test_capture_represents_eight_step_window_limit_per_projection(tmp_path):
    collector = BoundedActivationCapture(tmp_path / "manifest.json", max_raw_mib=1)
    collector.representative_paths.add("projection")
    tensor = torch.zeros(1, 1, 10, dtype=torch.bfloat16)
    for _ in range(10):
        collector.record("projection", "self_projection_input", tensor)
    assert len(collector.raw_tensors) == 8
    assert collector.estimated_raw_bytes == tensor.numel() * tensor.element_size() * 8


def test_quantized_decoder_plan_requires_engine_for_every_target(tmp_path):
    path = tmp_path / "plan.json"
    payload = {
        "schema_version": 1,
        "name": "int8",
        "scheme": "int8_conservative",
        "targets": [{
            "path": "decoder.0", "kind": "causal_conv3d",
            "observed_shapes": [[1, 2, 1, 4, 4]], "calibration": {},
        }],
        "protected": [],
        "engines": {},
    }
    path.write_text(json.dumps(payload))
    with pytest.raises(DecoderPlanError, match="one explicit engine"):
        load_decoder_plan(path)
    assert load_decoder_plan(path, require_engines=False).engines == {}
    payload["engines"] = {
        "decoder.0": {"path": "decoder.engine", "sha256": "digest", "precision": "int8"}
    }
    path.write_text(json.dumps(payload))
    assert load_decoder_plan(path).engines["decoder.0"]["1x2x1x4x4"]["precision"] == "int8"


def test_measured_decoder_plan_protects_entry_and_full_resolution_paths(tmp_path):
    captures = {}
    for path in ("model.decoder.middle.0.residual.2", "model.decoder.upsamples.9.residual.2"):
        tensor_path = tmp_path / f"{path.replace('.', '_')}.pt"
        torch.save(torch.ones(1, 2, 1, 4, 4), tensor_path)
        captures[path] = [(tensor_path, {"sha256": __import__("hashlib").sha256(tensor_path.read_bytes()).hexdigest()})]
    inventory = [
        {"path": "model.decoder.conv1", "class": "CausalConv3d", "inclusive_cuda_ms": 5.0, "observed_inputs": [[1, 4, 1, 9, 5]], "observed_outputs": [[1, 512, 1, 9, 5]]},
        {"path": "model.decoder.middle.0.residual.2", "class": "CausalConv3d", "inclusive_cuda_ms": 4.0, "observed_inputs": [[1, 512, 1, 9, 5]], "observed_outputs": [[1, 512, 1, 9, 5]]},
        {"path": "model.decoder.upsamples.9.residual.2", "class": "CausalConv3d", "inclusive_cuda_ms": 3.0, "observed_inputs": [[1, 128, 1, 72, 40]], "observed_outputs": [[1, 128, 1, 72, 40]]},
        {"path": "model.decoder.head.2", "class": "CausalConv3d", "inclusive_cuda_ms": 2.0, "observed_inputs": [[1, 128, 1, 72, 40]], "observed_outputs": [[1, 3, 1, 72, 40]]},
    ]
    protected = protected_paths(inventory, [])
    assert "model.decoder.conv1" in protected
    assert "model.decoder.head.2" in protected
    plan = build_plan(inventory, captures, name="test", scheme="int8_conservative", max_targets=2, min_inclusive_cuda_ms=0, protect=[])
    assert [target.path for target in plan.targets] == ["model.decoder.middle.0.residual.2"]
    payload = plan.to_dict() | {"source": {"inventory": "decoder-inventory.json"}}
    plan_path = tmp_path / "calibration-plan.json"
    plan_path.write_text(json.dumps(payload))
    reloaded = load_decoder_plan(plan_path, require_engines=False)
    assert reloaded.source["inventory"] == "decoder-inventory.json"