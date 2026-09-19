"""CPU regressions for the V2 review findings; no GPU performance claims."""
import json
import sys
from types import SimpleNamespace
import pytest
import torch
from soulx_rtc.pro_vae_quantization import (
    DecoderPlanError, DecoderBackendError, load_decoder_plan, MultiShapeEngine, shape_key,
)
from benchmarks.pro_quantization_v2_20260918.build_decoder_engine import (
    calibrated_input_scale, convolution_int8_evidence,
)


def test_engine_coverage_and_source_roundtrip(tmp_path):
    shapes = [[1, 2, 3, 4, 4], [1, 2, 6, 4, 4]]
    artifact = {'path': 'a.engine', 'sha256': 'digest', 'precision': 'int8'}
    payload = dict(schema_version=2, name='test', scheme='int8_conservative',
                   targets=[dict(path='decoder.0', kind='causal_conv3d', observed_shapes=shapes, calibration={})],
                   protected=[], source={'captures': 'source.json'},
                   engines={'decoder.0': {shape_key(shapes[0]): artifact}})
    path = tmp_path / 'plan.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(DecoderPlanError, match='coverage'):
        load_decoder_plan(path)
    payload['engines']['decoder.0'][shape_key(shapes[1])] = artifact
    path.write_text(json.dumps(payload))
    assert load_decoder_plan(path).to_dict()['source'] == payload['source']
    payload['schema_version'] = 1
    payload['engines']['decoder.0'] = artifact
    path.write_text(json.dumps(payload))
    with pytest.raises(DecoderPlanError, match='Legacy'):
        load_decoder_plan(path)


def test_shape_dispatch_and_unknown_shape():
    dispatch = MultiShapeEngine({'1x2x3x4x4': lambda x: x + 1, '1x2x6x4x4': lambda x: x + 2})
    assert dispatch(torch.zeros(1, 2, 3, 4, 4)).mean() == 1
    assert dispatch(torch.zeros(1, 2, 6, 4, 4)).mean() == 2
    with pytest.raises(DecoderBackendError, match='No verified'):
        dispatch(torch.zeros(1, 2, 9, 4, 4))


def test_calibration_uses_fitted_scale_and_rejects_invalid():
    record = dict(scale=8 / 127, quant_max=127, scale_granularity='per_tensor')
    assert calibrated_input_scale(record) == 8 / 127
    for bad in (0, -1, float('nan'), float('inf')):
        with pytest.raises(DecoderPlanError):
            calibrated_input_scale(record | {'scale': bad})


def test_precision_proof_does_not_accept_names_or_reformats():
    assert not convolution_int8_evidence(json.dumps({'Layers': [{'Name': 'int8_convolution', 'LayerType': 'Reformat'}]}))
    conv = {'LayerType': 'CaskConvolution', 'Inputs': [{'Format/Datatype': 'Row major Int8'}]}
    assert convolution_int8_evidence(json.dumps({'Layers': [conv]}))
    conv['Inputs'][0]['Format/Datatype'] = 'FP16'
    assert not convolution_int8_evidence(json.dumps({'Layers': [conv]}))


def test_quantized_trial_uses_shared_installer(monkeypatch):
    from benchmarks.pro_quantization_v2_20260918 import decoder_trial, decoder_install
    calls = []
    monkeypatch.setattr(decoder_install, 'install_quantized_decoder', lambda vae, decoder: calls.append((vae, decoder)) or {'installed': True})
    decoder = {'scheme': 'int8_conservative', 'plan': 'verified.json'}
    assert decoder_trial._configure_candidate('vae', {'decoder': decoder}, False) == {'installed': True}
    assert calls == [('vae', decoder)]


def test_cross_attention_is_pinned_even_when_sage_available(monkeypatch):
    from flash_head.src.modules import flash_head_model as m
    from soulx_rtc.pro_attention_backends import pin_cross_attention_flash2
    called = []
    monkeypatch.setattr(m, 'SAGE_ATTN_AVAILABLE', True)
    monkeypatch.setattr(m, 'FLASH_ATTN_2_AVAILABLE', m.FLASH_ATTN_2_AVAILABLE)
    monkeypatch.setattr(m, 'FLASH_ATTN_3_AVAILABLE', m.FLASH_ATTN_3_AVAILABLE)
    monkeypatch.setattr(m, 'flash_attn', getattr(m, 'flash_attn', None), raising=False)
    monkeypatch.setattr(m, 'sageattn', lambda *a: pytest.fail('Sage must not execute'), raising=False)
    monkeypatch.setitem(sys.modules, 'flash_attn', SimpleNamespace(flash_attn_func=lambda q, k, v: called.append(k.shape[1]) or q))
    cross = m.CrossAttention(8, 2)
    pin_cross_attention_flash2(cross)
    assert cross(torch.randn(1, 3, 8), torch.randn(1, 5, 8)).shape == (1, 3, 8)
    assert called == [5]


def test_decoder_capture_includes_later_windows(tmp_path):
    from benchmarks.pro_quantization_v2_20260918.capture import BoundedActivationCapture
    collector = BoundedActivationCapture(tmp_path / 'manifest.json', max_raw_mib=1)
    collector.representative_paths.add('decoder')
    for phase, window in [('warmup', 0), ('warmup', 1), ('generation', 0), ('generation', 4), ('generation', 8)]:
        collector.set_context(phase=phase, window=window)
        for temporal in (3, 6, 6):
            collector.record('decoder', 'causal_conv3d_prepared_input', torch.zeros(1, 2, temporal, 4, 4))
    assert len(collector.raw_tensors) == 8
    assert {(r['phase'], r['window']) for r in collector.raw_tensors} == {('warmup', 0), ('warmup', 1), ('generation', 4), ('generation', 8)}


def test_onnx_export_uses_later_capture_scale(tmp_path):
    onnx = pytest.importorskip('onnx')
    from benchmarks.pro_quantization_v2_20260918.build_decoder_engine import _onnx_model
    from soulx_rtc.pro_vae_quantization import fit_symmetric_scale
    first = torch.ones(1, 2, 3, 4, 4, dtype=torch.bfloat16)
    fitted = fit_symmetric_scale([first, first * 8], quant_max=127)
    fitted['scale_granularity'] = 'per_tensor'
    fitted['quant_max'] = 127
    conv = torch.nn.Conv3d(2, 2, 3).to(torch.bfloat16).eval()
    path = tmp_path / 'conv.onnx'
    metadata = _onnx_model(conv, first, 'int8', path, fitted, output_scale=4 / 127)
    constants = {x.name: onnx.numpy_helper.to_array(x) for x in onnx.load(path).graph.initializer}
    assert float(constants['input_scale']) == pytest.approx(8 / 127)
    assert metadata['input_scale'] == 8 / 127
    assert metadata['output_scale'] == 4 / 127
    assert {'output_scale', 'output_zero'} <= constants.keys()


def test_compiled_attention_does_not_mutate_logging(monkeypatch):
    from soulx_rtc.pro_attention_backends import ExplicitAttentionBackend
    backend = ExplicitAttentionBackend()
    monkeypatch.setattr(torch.compiler, 'is_compiling', lambda: True)
    backend._record(torch.zeros(1), torch.zeros(1))
    assert backend.invocation_manifest == []


def test_quantized_backend_rejects_wrong_output_shape():
    from soulx_rtc.pro_vae_quantization import EngineConvolutionBackend
    conv = torch.nn.Conv3d(2, 2, 3)
    backend = EngineConvolutionBackend(lambda x: x, name='fake', precision='int8', manifest_data={})
    with pytest.raises(DecoderBackendError, match='shape/device'):
        backend(conv, torch.zeros(1, 2, 3, 4, 4))


def test_reference_backend_retains_original_model_call_boundary(monkeypatch):
    from soulx_rtc import pro_attention_backends as adapters
    from tests.test_pro_quantization_v2 import FakeSelfAttention
    model = torch.nn.Module()
    model.blocks = torch.nn.ModuleList([torch.nn.Module()])
    model.blocks[0].self_attn = FakeSelfAttention()
    pinned = []
    monkeypatch.setattr(adapters, 'pin_reference_flash2_dispatch', lambda: pinned.append(True))
    adapters.install_self_attention_backend(model, adapters.FlashAttention2Backend(), self_attention_type=FakeSelfAttention)
    assert pinned == [True]
    assert model.blocks[0].self_attn._pro_attention_backend is None
