import json
from types import SimpleNamespace

import pytest
import torch

from flash_head.utils.latency import (
    LatencyRecorder,
    PipelineInstrumentation,
    latency_scope,
)


def read_events(path):
    return [json.loads(line) for line in (path / "latency-events.jsonl").read_text().splitlines()]


def test_disabled_logging_never_touches_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled path touched CUDA")
    monkeypatch.setattr(torch.cuda, "Event", forbidden)
    monkeypatch.setattr(torch.cuda, "current_stream", forbidden)
    with latency_scope("disabled"):
        assert torch.equal(torch.ones(2), torch.ones(2))


def test_nested_cpu_events_phases_failure_and_bounded_buffer(tmp_path):
    recorder = LatencyRecorder(tmp_path, cuda=False, max_events=3)
    with recorder.activate():
        recorder.set_context(phase="warmup", window=0)
        with latency_scope("outer"):
            with latency_scope("child"):
                pass
            with pytest.raises(RuntimeError, match="all spans"):
                recorder.flush()
        with pytest.raises(ValueError), latency_scope("failed"):
            raise ValueError("test")
        with latency_scope("overflow"):
            pass
    recorder.flush()
    rows = read_events(tmp_path)
    assert len(rows) == 3
    assert rows[1]["parent_id"] == rows[0]["id"]
    assert rows[2]["status"] == "failed"
    assert rows[2]["error_type"] == "ValueError"
    assert all(row["cuda_elapsed_ms"] is None for row in rows)
    with recorder.activate():
        recorder.set_context(phase="generation", repeat=1, window=3)
        with latency_scope("outer"):
            pass
    recorder.flush(status="complete")
    summary = json.loads((tmp_path / "latency-summary.json").read_text())
    assert summary["events"] == 4 and summary["dropped_events"] == 1
    assert not summary["complete_coverage"]
    assert {r["phase"] for r in summary["aggregates"]} == {"warmup", "generation"}
    assert len(read_events(tmp_path)) == 4  # flush appends without duplicating


def test_nested_activations_restore_outer_recorder(tmp_path):
    a = LatencyRecorder(tmp_path / "a", cuda=False)
    b = LatencyRecorder(tmp_path / "b", cuda=False)
    with a.activate():
        with b.activate(), latency_scope("b"):
            pass
        with latency_scope("a"):
            pass
    a.flush(); b.flush()
    assert [r["name"] for r in read_events(tmp_path / "a")] == ["a"]
    assert [r["name"] for r in read_events(tmp_path / "b")] == ["b"]


def test_cuda_events_use_selected_stream_and_only_flush_synchronizes(tmp_path, monkeypatch):
    calls = []
    caller = SimpleNamespace(cuda_stream=5)
    engine = SimpleNamespace(cuda_stream=9)

    class Event:
        def __init__(self, **kwargs):
            pass
        def record(self, stream):
            calls.append(("record", stream.cuda_stream))
        def elapsed_time(self, end):
            assert calls[-1][0] == "sync"
            return 2.5

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: caller)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append(("sync", str(device))))
    recorder = LatencyRecorder(tmp_path)
    with recorder.activate(), latency_scope("caller"), latency_scope("engine", stream=engine):
        pass
    assert calls == [("record", 5), ("record", 9), ("record", 9), ("record", 5)]
    recorder.flush()
    assert calls[-1] == ("sync", "cuda")
    assert [r["cuda_stream"] for r in read_events(tmp_path)] == [5, 9]
    assert all(r["cuda_elapsed_ms"] == 2.5 for r in read_events(tmp_path))


def make_pipeline():
    class VAE:
        def __init__(self):
            self.model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU())
        def encode(self, value):
            return self.model(value)
        def decode(self, value):
            return self.model(value)
    return SimpleNamespace(
        model=torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.ReLU()),
        audio_encoder=torch.nn.Sequential(torch.nn.Linear(3, 3)),
        vae=VAE(), generate=lambda value: value, preprocess_audio=lambda value: value,
    )


def test_module_instrumentation_preserves_outputs_rng_and_restores_descriptors(tmp_path):
    pipeline = make_pipeline()
    value = torch.randn(2, 3)
    expected = pipeline.model(value)
    random_state = torch.get_rng_state().clone()
    recorder = LatencyRecorder(tmp_path, cuda=False, mode="modules")
    wrappers = PipelineInstrumentation()
    wrappers.attach(pipeline, modules=True)
    with recorder.activate():
        actual = pipeline.model(value)
        pipeline.vae.decode(value)
        pipeline.audio_encoder(value)
    wrappers.close()
    recorder.flush()
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), random_state)
    assert "forward" not in vars(pipeline.model)
    assert "forward" not in vars(pipeline.model[0])
    names = {r["name"] for r in read_events(tmp_path)}
    assert {"dit.forward", "dit.0", "dit.1", "vae.decode", "vae.model.0", "audio.encoder.0"} <= names
    assert torch.equal(pipeline.model(value), expected)


def test_wrapped_exception_restores_stack_and_method(tmp_path):
    class Broken:
        def forward(self):
            raise KeyError("expected")
    broken = Broken()
    recorder = LatencyRecorder(tmp_path, cuda=False)
    wrappers = PipelineInstrumentation()
    wrappers.wrap(broken, "forward", "broken")
    with recorder.activate(), pytest.raises(KeyError):
        broken.forward()
    wrappers.close()
    recorder.flush(status="failed")
    assert read_events(tmp_path)[0]["status"] == "failed"
    assert "forward" not in vars(broken)


def test_stage_wrappers_preserve_compiled_graph_and_module_mode_refuses_it(tmp_path):
    pipeline = make_pipeline()
    compilations = []
    def backend(graph, example_inputs):
        compilations.append(graph)
        return graph.forward
    pipeline.model.forward = torch.compile(pipeline.model.forward, backend=backend, fullgraph=True)
    x = torch.randn(2, 3)
    expected = pipeline.model(x)
    wrappers = PipelineInstrumentation()
    with pytest.raises(ValueError, match="eager"):
        wrappers.attach(pipeline, modules=True)
    wrappers.attach(pipeline)
    recorder = LatencyRecorder(tmp_path, cuda=False)
    with recorder.activate():
        assert torch.equal(expected, pipeline.model(x))
        assert torch.equal(expected, pipeline.model(x))
    wrappers.close()
    recorder.flush()
    assert len(compilations) == 1


def test_real_generate_cpu_fixture_logging_is_output_and_history_neutral(tmp_path, monkeypatch):
    # Exercise the actual pipeline method, without loading weights or a GPU.
    from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)

    class Model(torch.nn.Module):
        def forward(self, x, **kwargs):
            return x * 0.1
    class VAE:
        def decode(self, x):
            return x.unsqueeze(0)
        def encode(self, x):
            return x[0].clone()

    p = FlashHeadPipeline.__new__(FlashHeadPipeline)
    p.config = SimpleNamespace(out_dim=3, vae_stride=[1, 1, 1])
    p.frame_num = 5
    p.lat_h = p.lat_w = 2
    p.param_dtype = torch.float32
    p.device = "cpu"
    p.generator = torch.Generator().manual_seed(50)
    p.num_timesteps = 1000
    p.timesteps = [torch.tensor([v]) for v in (1000, 500, 0)]
    p.model_type = "pro"
    p.model = Model()
    p.vae = VAE()
    p.rank = 1
    p.motion_frames_num = 2
    p.color_correction_strength = 0
    p.ref_img_latent = torch.zeros(3, 5, 2, 2)
    p.latent_motion_frames = torch.zeros(3, 1, 2, 2)
    original_history = p.latent_motion_frames.clone()
    expected = p.generate(torch.zeros(1))
    expected_history = p.latent_motion_frames.clone()
    expected_rng = p.generator.get_state().clone()
    p.latent_motion_frames = original_history
    p.generator.manual_seed(50)
    recorder = LatencyRecorder(tmp_path, cuda=False)
    with recorder.activate():
        actual = p.generate(torch.zeros(1))
    recorder.flush()
    assert torch.equal(actual, expected)
    assert torch.equal(p.latent_motion_frames, expected_history)
    assert torch.equal(p.generator.get_state(), expected_rng)
    names = {r["name"] for r in read_events(tmp_path)}
    assert {"denoise.model", "denoise.update", "decode.total", "motion.encode",
            "postprocess.output_fp32", "postprocess.color_correction"} <= names


@pytest.mark.parametrize("backend,expected", [
    ("trt_stage_compile", "trt_stage"),
    ("adapter_compile", "adapter"),
    ("torch_compile", "pytorch"),
])
def test_eager_diagnostic_overrides_compilation_without_mutating_policy(monkeypatch, backend, expected):
    from benchmarks.pro_quantization_v2_20260918 import decoder_install, run
    seen = []
    monkeypatch.setattr(decoder_install, "install_quantized_decoder",
                        lambda vae, decoder: seen.append(dict(decoder)) or {"installed": True})
    policy = {"decoder": {"backend": backend,
                          "scheme": "bf16" if backend == "torch_compile" else "fp16_stage_control"}}
    result = run._install_decoder_policy(SimpleNamespace(vae=object()), policy, eager_diagnostic=True)
    assert policy["decoder"]["backend"] == backend
    if expected == "pytorch":
        assert result["compiled_modules"] == [] and not seen
    else:
        assert seen[0]["backend"] == expected
