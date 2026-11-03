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


NORM_SCOPES = ('RMSNorm', 'LayerNorm', 'AneLayerNorm', 'AneRMSNorm')

PRECISION_POLICY = {
    'mixed': 'mixed: float32 normalization and attention; float16 matrix projections',
    'mixed-gpu': 'mixed-gpu: float32 normalization; float16 attention and matrix projections',
    'float16': 'float16 throughout (standard layout failed the full-profile gate; use with --layout ane prescaled norms)',
    'float32': 'float32 throughout with float16 I/O'}
# Earlier wordings of the same policies, accepted when resuming an export.
LEGACY_POLICY = {'float16': ('float16 throughout (diagnostic; failed the full-profile numerical gate)',)}


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
        # CrossAttentionWeights is the explicit FP32 softmax of the folded
        # cross-attention (dit.CrossAttention.fold_output_projection).
        return op.op_type not in ('layer_norm', 'scaled_dot_product_attention') and not any(
            name in NORM_SCOPES + ('CrossAttentionWeights',) for name in modules)
    def fp16_attention(op):
        # 'mixed-gpu': normalisation stays FP32; attention runs in FP16.
        modules = op.scopes.get(ScopeSource.TORCHSCRIPT_MODULE_TYPE, [])
        return op.op_type != 'layer_norm' and not any(name in NORM_SCOPES for name in modules)
    model = ct.convert(
        traced, convert_to='mlprogram',
        inputs=[ct.TensorType(name=n, shape=tuple(x.shape), dtype=np.float16) for n, x in inputs.items()],
        outputs=[ct.TensorType(name=output_name, dtype=np.float16)],
        minimum_deployment_target=ct.target.iOS18,
        compute_precision=(ct.precision.FLOAT32 if precision == 'float32' else ct.precision.FLOAT16
                           if precision == 'float16' else ct.transform.FP16ComputePrecision(op_selector=fp16_attention)
                           if precision == 'mixed-gpu' else ct.transform.FP16ComputePrecision(op_selector=fp16_safe)),
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
        # A folded cross-attention output is a const [heads, 128, dim] matmul
        # operand. One INT8 scale per output channel (axis 2) over all heads
        # reproduces the original W_o row quantization exactly.
        folded_config = (cto.OpLinearQuantizerConfig(mode='linear_symmetric', dtype='int8', granularity='per_block',
                                                     block_size=[0, 0, 1], weight_threshold=1024)
                         if compression == 'int8' else weight_config)
        def select_weights(block):
            for op in block.operations:
                weight = op.inputs.get('weight') if op.op_type in ('linear', 'conv') else None
                if weight is not None and weight.op.op_type == 'const' and weight.rank >= 2:
                    config.set_op_name(weight.op.name, weight_config)
                operand = op.inputs.get('y') if op.op_type == 'matmul' else None
                if operand is not None and operand.op.op_type == 'const' and operand.rank == 3:
                    config.set_op_name(operand.op.name, folded_config)
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


class _PartitionView:
    """Expose one partition to runtime-style variant transforms."""
    def __init__(self, blocks):
        self.blocks = blocks

    def fold_cross_output(self):
        self.blocks.fold_cross_output()


def fits_unchanged(manifest, name, transforms=()):
    """A reused partition must match the current FFN selection and fit files."""
    from .ffn_pruning import _file_sha256, normalise_ratios
    records = [r for r in manifest.get('variant_transforms', {}).get(name, [])
               if r.get('transform') == 'calibrated_ffn_reduction']
    wanted = [t for t in transforms if getattr(t, 'ffn_ratio', None) is not None]
    if len(records) != len(wanted):
        print(f'FFN selection for {name} changed; exporting it again', flush=True)
        return False
    for record, transform in zip(records, wanted):
        layers = list(range(*record['layers']))
        want = normalise_ratios(transform.ffn_ratio, layers)
        have = normalise_ratios(record['removed_fraction'], layers)
        if str(transform.fit_dir) != record['fit_dir'] or any(
                abs(want.get(layer, 0.0) - have.get(layer, 0.0)) > 1e-9 for layer in layers):
            print(f'FFN selection for {name} changed; exporting it again', flush=True)
            return False
        for layer, expected in zip(layers, record['fit_sha256']):
            if expected is None:
                continue  # layer kept dense
            path = Path(record['fit_dir']) / f'block_{layer:02d}.pt'
            if not path.exists() or _file_sha256(path) != expected:
                print(f'FFN fit for {name} changed; exporting it again', flush=True)
                return False
    return True


def apply_variant(transform, blocks):
    from .ffn_pruning import reduce_partition
    reduce = getattr(transform, 'ffn_ratio', None)
    if reduce is not None:
        return reduce_partition(blocks, transform.ffn_ratio, transform.fit_dir)
    return transform(_PartitionView(blocks))


def export(args):
    if not 1 <= args.blocks_per_chunk <= 30 or args.query_chunk < 0:
        raise ValueError('Invalid partition or query chunk size')
    torch.set_num_threads(args.threads)
    fuse_qkv = getattr(args, 'fuse_qkv', False)
    layout = getattr(args, 'layout', 'standard') or 'standard'
    prescale = float(getattr(args, 'norm_prescale', 64.0) or 0.0)
    qk_prescale = float(getattr(args, 'qk_norm_prescale', 8.0) or 0.0)
    if layout == 'ane' and fuse_qkv:
        raise ValueError('The ANE layout is built from unfused projections; drop --fuse-qkv')
    if layout == 'ane' and args.precision not in ('float16', 'float32'):
        # FP32 islands cannot run on the Neural Engine and would split every
        # block between compute units; float32 is for parity debugging only.
        raise ValueError('--layout ane needs --precision float16 (or float32 for debugging), not ' + args.precision)
    if layout == 'ane' and args.query_chunk:
        raise ValueError('--layout ane does not implement --query-chunk; use 0')
    variant = getattr(args, 'variant', 'baseline') or 'baseline'
    if layout == 'ane' and 'fold' in variant.split('+'):
        raise ValueError('The ANE layout does not combine with the fold variant')
    from .variants import build_transforms
    transforms, variant_description = build_transforms(variant)
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
                'precision_policy': PRECISION_POLICY[args.precision],
                'layout': layout, 'norm_prescale': prescale if layout == 'ane' else None,
                'qk_norm_prescale': qk_prescale if layout == 'ane' else None,
                'component_precision': ({'blocks': args.precision, 'patch': 'mixed', 'conditioning': 'mixed', 'head': 'mixed',
                                         'note': 'ANE layout applies to block partitions only; the other components keep the '
                                                 'standard layout and its gated mixed policy'}
                                        if layout == 'ane' else None),
                'blocks_per_chunk': args.blocks_per_chunk, 'query_chunk': args.query_chunk,
                'fuse_qkv': fuse_qkv,
                'variant': variant, 'variant_description': variant_description, 'variant_transforms': {},
                'compression_scope': ('learned linear and convolution weights only' if 'fold' not in variant.split('+') else
                                      'learned linear and convolution weights, plus folded cross-attention output operands'),
                'quantization_note': 'Apple weight compression of original PRO weights; CUDA activation quantization is not reproduced',
                'models': {}, 'execution': 'CPU tracing and Core ML conversion on Mac, no iPhone benchmark'}
    if args.resume and manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        for key in ('height', 'width', 'source_checkpoint_sha256', 'compression', 'precision',
                    'blocks_per_chunk', 'query_chunk', 'compression_scope'):
            if previous.get(key) != manifest[key]:
                raise ValueError(f'Resume manifest mismatch: {key}')
        # The policy text is descriptive and has been reworded; 'precision'
        # above is the binding value. Accept earlier wordings of the same one.
        if previous.get('precision_policy') not in (manifest['precision_policy'],) + LEGACY_POLICY.get(args.precision, ()):
            raise ValueError('Resume manifest mismatch: precision_policy')
        if previous.get('fuse_qkv', False) != fuse_qkv:
            raise ValueError('Resume manifest mismatch: fuse_qkv')
        if previous.get('variant', 'baseline') != variant:
            raise ValueError('Resume manifest mismatch: variant')
        if previous.get('layout', 'standard') != layout or previous.get('norm_prescale') != manifest['norm_prescale']:
            raise ValueError('Resume manifest mismatch: layout')
        if layout == 'ane' and previous.get('qk_norm_prescale', prescale) != qk_prescale:
            raise ValueError('Resume manifest mismatch: qk_norm_prescale (export into a new directory)')
        manifest['variant_transforms'] = previous.get('variant_transforms', {})
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
        precision = args.precision if layout != 'ane' or name.startswith('blocks_') else 'mixed'
        previous_precision = manifest['models'].get(name, {}).get('precision', args.precision)
        if (args.resume and name in manifest['models'] and destination.exists() and previous_precision == precision
                and fits_unchanged(manifest, name, transforms)):
            print(f'Reusing {name}', flush=True)
            continue
        started = time.monotonic()
        print(f'Exporting {name}', flush=True)
        module, inputs = factory(), inputs_factory()
        if isinstance(module, Blocks):
            # Variant transforms act on whole blocks after strict loading and
            # before compression; FFN reduction keeps the partition interface.
            applied = [apply_variant(transform, module) for transform in transforms]
            if applied:
                manifest['variant_transforms'][name] = applied
        if fuse_qkv and isinstance(module, Blocks):
            module.fuse_projections()
        if layout == 'ane' and isinstance(module, Blocks):
            from .ane_layout import to_ane_layout
            module = to_ane_layout(module, prescale, qk_prescale)
        entry = convert_component(module, inputs, output_name, destination, args.compression, precision)
        entry['precision'] = precision
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
    parser.add_argument('--fuse-qkv', action='store_true', help='Combine self-attention Q/K/V and cross-attention K/V projections after loading checkpoint rows')
    parser.add_argument('--variant', default='baseline', help="Transformer variant, e.g. 'fold', 'ffn25+fold' (see flashhead_coreml.variants)")
    parser.add_argument('--compression', choices=['none', 'int8', 'lut6', 'lut4'], default='int8')
    parser.add_argument('--precision', choices=['mixed', 'mixed-gpu', 'float16', 'float32'], default='mixed',
                        help='mixed retains FP32 normalization and attention; mixed-gpu keeps only normalization FP32; I/O stays FP16')
    parser.add_argument('--layout', choices=['standard', 'ane'], default='standard',
                        help="ane: channels-first 1x1-conv block partitions for the Neural Engine; needs --precision float16 "
                             "(patch, conditioning and head keep the standard layout and the mixed policy)")
    parser.add_argument('--norm-prescale', type=float, default=64.0,
                        help='ANE layout: divide LayerNorm inputs by this constant before squaring, with eps scaled to match. '
                             'Exact in real arithmetic and FP32; in FP16 it avoids overflow but rounds differently. 0 disables')
    parser.add_argument('--qk-norm-prescale', type=float, default=8.0,
                        help='ANE layout: the same for the q/k RMSNorms, whose inputs are smaller; a large value would push '
                             'their mean squares into FP16 subnormals. 0 disables')
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--only', help='Comma-separated component names')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.blocks_per_chunk <= 30 or args.query_chunk < 0:
        parser.error('Invalid partition or query chunk size')
    export(args)


if __name__ == '__main__':
    main()
