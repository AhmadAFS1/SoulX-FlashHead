from types import SimpleNamespace

import pytest
import torch

from benchmarks.pro_quantization_v2_20260918.build_decoder_engine import (
    calibrate_output_scale,
)
from benchmarks.pro_quantization_v2_20260918.profile import aggregate_row, device_metric
from flash_head.wan.modules.vae import ResidualBlock
from soulx_rtc.pro_vae_stage_backend import (
    ResidualStage,
    StageAdapter,
    install_stages,
    remove_stages,
)


def test_profiler_uses_aggregate_device_fields_and_preserves_missing():
    event = SimpleNamespace(
        key="kernel",
        count=2,
        self_cpu_time_total=1,
        cpu_time_total=2,
        self_device_time_total=42,
        device_time_total=45,
    )
    row = aggregate_row(event)
    assert row["self_cuda_time_us"] == 42
    assert row["self_cuda_time_us_source"] == "self_device_time_total"
    assert row["self_cuda_memory_bytes"] is None
    assert device_metric(
        SimpleNamespace(self_cuda_time_total=17), "self_device_time_total"
    ) == (17, "self_cuda_time_total")
    assert device_metric(
        SimpleNamespace(self_device_time_total=0, self_cuda_time_total=99),
        "self_device_time_total",
    ) == (0, "self_device_time_total")


def test_output_scale_includes_later_outliers_and_rejects_missing_coverage():
    conv = torch.nn.Conv3d(1, 1, 1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(2)
    values = [torch.ones(1, 1, 1, 2, 2), torch.full((1, 1, 1, 2, 2), 12.0)]
    calibration = calibrate_output_scale(conv, iter(values))
    assert calibration["capture_count"] == 2
    assert calibration["scale"] == pytest.approx(24 / 127)
    with pytest.raises(ValueError, match="at least two"):
        calibrate_output_scale(conv, values[:1])
    with pytest.raises(ValueError, match="Nonfinite"):
        calibrate_output_scale(conv, [values[0], values[1] * float("nan")])


def test_stage_matches_stock_initial_recurrent_and_independent_sessions():
    torch.manual_seed(9)
    blocks = [ResidualBlock(2, 2).eval(), ResidualBlock(2, 2).eval()]
    stage = ResidualStage(blocks).eval()
    adapter = StageAdapter(stage, stage)
    native = [[None] * 4 for _ in range(2)]
    converted = [[None] * 4 for _ in range(2)]
    with torch.inference_mode():
        for temporal in (1, 1, 4, 4):
            for session in (0, 1):
                x = torch.randn(1, 2, temporal, 4, 4)
                expected = x
                cursor = [0]
                for block in blocks:
                    expected = block(expected, native[session], cursor)
                candidate_cursor = [0]
                actual = adapter(x, converted[session], candidate_cursor)
                assert candidate_cursor == cursor == [4]
                assert torch.equal(expected, actual)
                assert all(
                    torch.equal(a, b)
                    for a, b in zip(native[session], converted[session])
                )
        x = torch.randn(1, 2, 1, 4, 4)
        assert torch.equal(adapter(x, [None] * 4, [0]), adapter(x, [None] * 4, [0]))


def test_stage_refuses_partial_state_before_mutation_and_output_arity():
    stage = ResidualStage([ResidualBlock(2, 2).eval()])
    x = torch.zeros(1, 2, 1, 4, 4)
    incoming = [None, x]
    cursor = [0]
    with pytest.raises(ValueError, match="Partially"):
        StageAdapter(stage, stage)(x, incoming, cursor)
    assert incoming[0] is None and cursor == [0]
    with pytest.raises(ValueError, match="contract"):
        StageAdapter(stage, lambda *args: (x,))(x, [None, None], cursor)
    assert cursor == [0]


def test_stage_install_is_atomic_and_restorable():
    modules = torch.nn.ModuleList(
        [ResidualBlock(2, 2), ResidualBlock(2, 2), torch.nn.Identity()]
    )
    vae = SimpleNamespace(
        model=SimpleNamespace(decoder=SimpleNamespace(upsamples=modules))
    )
    originals = list(modules)
    with pytest.raises(ValueError, match="overlapping"):
        install_stages(vae, [[0, 1], [1]], lambda indices, stage: stage)
    assert list(modules) == originals
    install_stages(vae, [[0, 1]], lambda indices, stage: stage)
    assert isinstance(modules[0], StageAdapter)
    remove_stages(vae)
    assert list(modules) == originals


def test_stage_int8_export_has_bf16_cache_bindings_and_all_convolution_qdq(tmp_path):
    onnx = pytest.importorskip("onnx")
    from benchmarks.pro_30fps_20260919.decoder_stages import export_stage

    stage = ResidualStage([ResidualBlock(2, 2).eval().to(torch.bfloat16)])
    values = (torch.ones(1, 2, 1, 4, 4, dtype=torch.bfloat16),)
    scales = {
        f"blocks.0.residual.{i}": {"input_max": 4.0, "output_max": 5.0, "count": 2}
        for i in (2, 6)
    }
    path = tmp_path / "stage.onnx"
    metadata = export_stage(stage, values, path, "int8", scales)
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    assert len(metadata["quantized_modules"]) == 2
    assert len([n for n in graph.graph.node if n.op_type == "QuantizeLinear"]) == 6
    assert not any(n.op_type == "Clip" for n in graph.graph.node)
    assert all(
        v.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16
        for v in list(graph.graph.input) + list(graph.graph.output)
    )
    with pytest.raises(ValueError, match="Incomplete calibration"):
        export_stage(stage, values, tmp_path / "bad.onnx", "int8", {})


def test_real_attention_capture_covers_all_blocks_and_steps_with_explicit_context(
    tmp_path,
):
    from benchmarks.pro_quantization_v2_20260918.capture import (
        BoundedActivationCapture,
        CaptureBudgetExceeded,
    )

    capture = BoundedActivationCapture(
        tmp_path / "manifest.json", max_raw_mib=1, capture_attention=True
    )
    capture.set_context(phase="warmup", window=0)
    tensor = torch.zeros(1, 2, 4, dtype=torch.bfloat16)
    for step in range(4):
        for block in range(30):
            capture.record_attention(block, tensor, tensor, tensor)
    assert len(capture.raw_tensors) == 90
    assert {r["block"] for r in capture.raw_tensors} == set(range(30))
    assert {r["step"] for r in capture.raw_tensors} == {0, 1, 2, 3}
    capture.set_context(phase="generation", window=4)
    for step in range(4):
        for block in range(30):
            capture.record_attention(block, tensor, tensor, tensor)
    assert len(capture.raw_tensors) == 126
    assert {r["block"] for r in capture.raw_tensors if r["phase"] == "generation"} == {
        0,
        14,
        29,
    }
    capture.budget = capture.bytes_saved
    with pytest.raises(CaptureBudgetExceeded):
        capture.record_attention(0, tensor, tensor, tensor)


def test_public_latents_include_late_windows_instead_of_only_warmup_repeats(tmp_path):
    from benchmarks.pro_quantization_v2_20260918.capture import BoundedActivationCapture

    capture = BoundedActivationCapture(tmp_path / "manifest.json", max_raw_mib=1)
    capture.representative_paths.add("vae.public_decode")
    for phase, windows in [("warmup", range(2)), ("generation", range(9))]:
        for window in windows:
            capture.set_context(phase=phase, window=window)
            capture.record(
                "vae.public_decode", "public_vae_latent", torch.ones(1, 2, 3, 4)
            )
    assert [(row["phase"], row["window"]) for row in capture.raw_tensors] == [
        ("warmup", 0),
        ("warmup", 1),
        ("generation", 4),
        ("generation", 8),
    ]


def test_stage_precision_evidence_rejects_float_convolution_with_quantized_io():
    import json

    from benchmarks.pro_30fps_20260919.decoder_stages import int8_convolutions

    floating = {
        "Name": "false-int8",
        "LayerType": "CaskConvolution",
        "Inputs": [{"Format/Datatype": "Int8"}],
        "Weights": {"Type": "Float"},
        "TacticName": "f32f32_tf32f32",
    }
    integer = dict(
        floating, Name="actual-int8", Weights={"Type": "Int8"}, TacticName="i8i8_i8i32"
    )
    result = int8_convolutions(json.dumps({"Layers": [floating, integer]}))
    assert result["convolutions"] == 2
    assert result["quantized_names"] == ["actual-int8"]


def test_fp16_stage_export_is_explicit_and_preserves_source_weights(tmp_path):
    onnx = pytest.importorskip("onnx")
    from benchmarks.pro_30fps_20260919.decoder_stages import export_stage

    stage = ResidualStage([ResidualBlock(2, 2).eval().to(torch.bfloat16)])
    original = {k: v.clone() for k, v in stage.state_dict().items()}
    values = (torch.ones(1, 2, 1, 4, 4, dtype=torch.bfloat16),)
    path = tmp_path / "half.onnx"
    metadata = export_stage(stage, values, path, "fp16", {})
    assert metadata["binding_dtype"] == "float16"
    assert metadata["floating_precision"] == "fp16"
    assert metadata["caller_cache_dtype"] == "bfloat16"
    model = onnx.load(path)
    assert all(
        v.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16
        for v in list(model.graph.input) + list(model.graph.output)
    )
    assert all(torch.equal(v, original[k]) for k, v in stage.state_dict().items())


def test_stage_plan_rejects_changed_vae_source_before_cuda_allocation(
    tmp_path, monkeypatch
):
    import json

    from benchmarks.pro_quantization_v2_20260918 import common
    from soulx_rtc.pro_vae_stage_backend import install_stage_plan

    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "weights_sha256": "weights",
                "source_sha256": {"flash_head/wan/modules/vae.py": "outdated-source"},
            }
        )
    )
    monkeypatch.setattr(
        common,
        "sha256",
        lambda path: "weights" if path.suffix == ".pth" else "current-source",
    )

    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("Unverified engines must be rejected before allocation")

    monkeypatch.setattr(torch, "empty", forbidden_allocation)
    with pytest.raises(ValueError, match="different Wan VAE implementation"):
        install_stage_plan(object(), plan)


def test_new_speech_requires_distinct_effective_samples():
    import numpy as np

    from benchmarks.pro_30fps_20260919.validation_suite import require_distinct_audio

    samples = np.arange(32, dtype=np.float32)
    with pytest.raises(ValueError, match="Duplicate effective speech"):
        require_distinct_audio(
            {"standard": samples, "different_file_same_segment": samples.copy()}
        )
    hashes = require_distinct_audio({"standard": samples, "later_segment": samples + 1})
    assert len(set(hashes.values())) == 2
