"""Export fixed-profile PRO components with bounded checkpoint memory.

Example: PYTHONPATH=apple python -m flashhead_coreml.export_dit --checkpoint
models/SoulX-FlashHead-1_3B/Model_Pro --output artifacts/coreml/pro576

Export does not certify ANE placement. The manifest records the source hash,
compression, fixed two-step schedule and exact model I/O for device validation.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .dit import ProConfig, Profile, Patch, AudioProjection, TimeEmbedding, Blocks, OutputHead, load_component


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for data in iter(lambda: f.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def save_manifest(path, manifest):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2) + '\n')
    temporary.replace(path)


def convert_component(module, inputs, output_name, destination, compression='none', precision='mixed'):
    import coremltools as ct
    from coremltools.converters.mil.mil.scope import ScopeSource
    module = module.eval().requires_grad_(False)
    examples = tuple(inputs.values())
    with torch.inference_mode():
        traced = torch.jit.trace(module, examples, check_trace=False)
    def fp16_safe(op):
        # The trained PRO attention is numerically sensitive. Preserve the
        # upstream float32 normalization reductions, including the square and
        # reciprocal square root, rather than allowing the converter to rewrite
        # explicit torch.float() islands to half precision.
        modules = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_TYPE, [])
        return op.op_type not in ('layer_norm', 'scaled_dot_product_attention') and not any(name in ('RMSNorm', 'LayerNorm') for name in modules)
    model = ct.convert(
        traced, convert_to='mlprogram',
        inputs=[ct.TensorType(name=n, shape=tuple(x.shape), dtype=np.float16) for n, x in inputs.items()],
        outputs=[ct.TensorType(name=output_name, dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=(ct.precision.FLOAT32 if precision == 'float32' else ct.precision.FLOAT16
                           if precision == 'float16' else ct.transform.FP16ComputePrecision(op_selector=fp16_safe)),
        compute_units=ct.ComputeUnit.ALL,
        skip_model_load=True,
    )
    if compression != 'none':
        import coremltools.optimize.coreml as cto
        if compression == 'int8':
            weight_config = cto.OpLinearQuantizerConfig(mode='linear_symmetric', dtype='int8', granularity='per_channel', weight_threshold=1024)
        else:
            bits = int(compression.replace('lut', ''))
            weight_config = cto.OpPalettizerConfig(mode='kmeans', nbits=bits, weight_threshold=1024)
        # A global compression policy also quantizes RoPE and timestep tables.
        # Compress learned matrix/convolution weights only; preserve all other
        # constants, normalization parameters and biases in FP16.
        config = cto.OptimizationConfig()
        def select_weights(block):
            for op in block.operations:
                weight = op.inputs.get('weight') if op.op_type in ('linear', 'conv') else None
                if weight is not None and weight.op.op_type == 'const' and weight.rank >= 2:
                    config.set_op_name(weight.op.name, weight_config)
                for child in op.blocks:
                    select_weights(child)
        for function in model._mil_program.functions.values():
            select_weights(function)
        if compression == 'int8':
            model = cto.linear_quantize_weights(model, config=config)
        else:
            model = cto.palettize_weights(model, config=config)
    model.short_description = 'SoulX FlashHead PRO on-device component; fixed 33-frame/two-step profile'
    model.user_defined_metadata['source_execution'] = f'Portable FP32 PyTorch reference, Core ML {precision}; mixed retains FP32 normalization and attention; no CUDA kernels'
    model.user_defined_metadata['compression'] = compression
    model.save(str(destination))
    spec = model.get_spec()
    def shape(feature):
        return list(feature.type.multiArrayType.shape)
    return {'file': destination.name,
            'inputs': {f.name: {'shape': shape(f), 'dtype': 'float16'} for f in spec.description.input},
            'outputs': {f.name: {'shape': shape(f), 'dtype': 'float16'} for f in spec.description.output}}


def export(args):
    if not 1 <= args.blocks_per_chunk <= 30 or args.query_chunk < 0:
        raise ValueError('Invalid partition or query chunk size')
    torch.set_num_threads(args.threads)
    checkpoint_dir = Path(args.checkpoint)
    checkpoint = checkpoint_dir / 'diffusion_pytorch_model.safetensors'
    cfg = ProConfig.from_json(checkpoint_dir / 'config.json')
    profile = Profile(args.height, args.width)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    timing = load_component(TimeEmbedding(cfg), checkpoint, 'time')
    with torch.inference_mode():
        t, tm = timing(torch.tensor([1000., 625.]))
    del timing
    manifest_path = out / 'dit_manifest.json'
    manifest = {'schema_version': 1, 'height': profile.height, 'width': profile.width,
                'conversion_environment': {'date_utc': datetime.now(timezone.utc).isoformat(),
                                           'gpu_model': 'none used (CPU-only tracing and conversion)',
                                           'physical_or_visible_vram': 'not applicable to CPU-only conversion',
                                           'evidence': 'CPU PyTorch tensors; Core ML skip_model_load=True',
                                           'driver': 'not applicable (CPU only)', 'torch': torch.__version__,
                                           'co_resident_load': 'not measured; export_seconds is not an inference benchmark'},
                'frames': 33, 'history_frames': 5, 'new_frames': 28, 'fps': 25,
                'latent_shape': list(profile.latent_shape), 'schedule': [1000, 625, 0],
                'source_checkpoint_sha256': file_hash(checkpoint),
                'precision': args.precision, 'compression': args.compression,
                'precision_policy': {'mixed': 'mixed: float32 normalization and attention; float16 matrix projections',
                                     'float16': 'float16 throughout (diagnostic; failed the full-profile numerical gate)',
                                     'float32': 'float32 throughout with float16 I/O'}[args.precision],
                'blocks_per_chunk': args.blocks_per_chunk, 'query_chunk': args.query_chunk,
                'compression_scope': 'learned linear and convolution weights only',
                'quantization_note': 'Apple weight compression of original PRO weights; CUDA activation quantization is not reproduced',
                'models': {}, 'execution': 'CPU tracing and Core ML conversion on Mac, no iPhone benchmark'}
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ('height', 'width', 'source_checkpoint_sha256', 'compression', 'precision',
                    'blocks_per_chunk', 'query_chunk', 'compression_scope', 'precision_policy'):
            if previous.get(key) != manifest[key]:
                raise ValueError(f'Resume manifest mismatch: {key}')
        manifest['models'] = previous['models']
    latent = lambda: torch.zeros(profile.latent_shape)
    hidden = lambda: torch.zeros(1, profile.tokens, cfg.dim)
    context = lambda: torch.zeros(1, 9, 32, cfg.dim)
    timestep = lambda: torch.tensor([1000.])
    components = [('patch', lambda: load_component(Patch(cfg, profile), checkpoint, 'patch'),
                   lambda: {'latent': latent(), 'reference': latent()}, 'hidden'),
                  ('conditioning', lambda: load_component(AudioProjection(cfg.dim), checkpoint, 'conditioning'),
                   lambda: {'audio': torch.zeros(1, 33, 5, 12, 768)}, 'context')]
    for index, start in enumerate(range(0, cfg.num_layers, args.blocks_per_chunk)):
        end = min(start + args.blocks_per_chunk, cfg.num_layers)
        components.append((f'blocks_{index:02d}',
                           lambda s=start, e=end: load_component(Blocks(cfg, profile, s, e, tm, args.query_chunk), checkpoint, 'blocks', s),
                           lambda: {'hidden': hidden(), 'context': context(), 'timestep': timestep()}, 'hidden_out'))
    components.append(('head', lambda: load_component(OutputHead(cfg, profile, t), checkpoint, 'head'),
                       lambda: {'hidden': hidden(), 'timestep': timestep()}, 'flow'))
    selected = {name.strip() for name in args.only.split(',')} if args.only else None
    if selected is not None:
        unknown = selected - {component[0] for component in components}
        if unknown:
            raise ValueError(f'Unknown components: {sorted(unknown)}')
    for name, factory, inputs_factory, output_name in components:
        if selected is not None and name not in selected:
            continue
        destination = out / (name + '.mlpackage')
        if args.resume and name in manifest['models'] and destination.exists():
            print(f'Reusing {name}', flush=True)
            continue
        started = time.monotonic()
        print(f'Exporting {name}', flush=True)
        module, inputs = factory(), inputs_factory()
        entry = convert_component(module, inputs, output_name, destination, args.compression, args.precision)
        entry['export_seconds'] = time.monotonic() - started
        manifest['models'][name] = entry
        save_manifest(manifest_path, manifest)
        del module, inputs
        gc.collect()
        print(f'Saved {destination} ({entry["export_seconds"]:.1f}s)', flush=True)
    manifest['complete'] = all(name in manifest['models'] and (out / (name + '.mlpackage')).exists()
                               for name, _, _, _ in components)
    save_manifest(manifest_path, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='models/SoulX-FlashHead-1_3B/Model_Pro')
    parser.add_argument('--output', default='artifacts/coreml/pro576')
    parser.add_argument('--height', type=int, default=576)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--blocks-per-chunk', type=int, default=3)
    parser.add_argument('--query-chunk', type=int, default=0, help='Exact query tiling; 0 uses fused SDPA')
    parser.add_argument('--compression', choices=['none', 'int8', 'lut6', 'lut4'], default='int8')
    parser.add_argument('--precision', choices=['mixed', 'float16', 'float32'], default='mixed', help='mixed retains FP32 normalization and attention; I/O stays FP16')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--only', help='Comma-separated component names')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.blocks_per_chunk <= 30 or args.query_chunk < 0:
        parser.error('Invalid partition or query chunk size')
    export(args)


if __name__ == '__main__':
    main()
