"""Shared fail-closed decoder installation for video and fixed-latent trials."""
from .common import ROOT, resolve_path, sha256
from soulx_rtc.pro_vae_quantization import (
    load_decoder_plan, shape_key, MultiShapeEngine, EngineConvolutionBackend,
    install_decoder_adapters,
)


def install_quantized_decoder(vae, decoder):
    if decoder['backend'] in ('trt_stage', 'trt_stage_compile'):
        import json
        from soulx_rtc.pro_vae_stage_backend import install_stage_plan
        path = resolve_path(decoder['plan'])
        data = json.loads(path.read_text())
        expected = 'bf16' if decoder['scheme'] == 'bf16' else 'int8'
        if data.get('precision') != expected:
            raise ValueError('Stage precision and policy mismatch')
        report = install_stage_plan(vae, path)
        if decoder['backend'] == 'trt_stage_compile':
            from soulx_rtc.pro_quantization import optimize_wan_vae
            report['compile'] = optimize_wan_vae(vae, 'compiled')
        return report
    from .build_decoder_engine import TensorRTPreparedConv
    plan = load_decoder_plan(resolve_path(decoder['plan']))
    if plan.scheme != decoder['scheme']:
        raise ValueError('Decoder scheme does not match engine plan')
    precision = plan.scheme.split('_', 1)[0]
    weights_hash = sha256(ROOT / "models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth")
    backends = {}
    for target in plan.targets:
        runtimes, metadata = {}, {}
        for shape in target.observed_shapes:
            key = shape_key(shape)
            artifact = plan.engines[target.path][key]
            path = resolve_path(artifact['path'])
            if sha256(path) != artifact['sha256']:
                raise ValueError(f'Decoder engine hash mismatch: {target.path}/{key}')
            runtime = TensorRTPreparedConv(path)
            data = runtime.metadata
            if (data.get('weights_sha256') != weights_hash
                    or data.get('quantization', {}).get('input_scale') != target.calibration.get('scale')
                    or data.get('target') != target.path or data.get('precision') != precision
                    or tuple(data.get('input_shape', ())) != shape
                    or data.get('external_input_dtype') != 'bfloat16'
                    or data.get('quantization_dispatch_verified') is not True):
                raise ValueError(f'Decoder engine contract mismatch: {target.path}/{key}')
            runtimes[key], metadata[key] = runtime, data
        backends[target.path] = EngineConvolutionBackend(
            MultiShapeEngine(runtimes), name='tensorrt_explicit_qdq', precision=precision,
            manifest_data=metadata,
        )
    report = install_decoder_adapters(vae, plan, backends)
    if decoder['backend'] == 'adapter_compile':
        from soulx_rtc.pro_quantization import optimize_wan_vae
        report['compile'] = optimize_wan_vae(vae, 'compiled')
    return report
