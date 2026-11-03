"""Reproducible CPU-only Core ML DiT parity checks.

Without --checkpoint this exports and runs a small synthetic architecture.
With --checkpoint it validates existing production packages from --models.
Neither mode measures GPU, Neural Engine, iPhone FPS, or perceptual quality.
"""
import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import torch

from .dit import ProConfig, Profile, Patch, AudioProjection, TimeEmbedding, Blocks, OutputHead, load_component
from .export_dit import convert_component


def environment():
    import coremltools as ct
    def sysctl(name):
        return subprocess.check_output(['sysctl', '-n', name], text=True).strip()
    return {'date_utc': datetime.now(timezone.utc).isoformat(),
            'execution': 'CPU-only Core ML prediction and PyTorch reference; no GPU inference',
            'gpu_model': 'none used', 'physical_or_visible_vram': 'not applicable to CPU-only validation',
            'hardware_evidence': 'Core ML ComputeUnit.CPU_ONLY; torch tensors on CPU; sysctl',
            'cpu': sysctl('machdep.cpu.brand_string'), 'physical_memory_bytes': int(sysctl('hw.memsize')),
            'os': platform.platform(), 'driver': 'not applicable (CPU only)',
            'runtime': {'torch': torch.__version__, 'coremltools': ct.__version__},
            'co_resident_load': 'Other conversion jobs may be running; timings are not benchmarked'}


def errors(actual, expected):
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    delta = actual - expected
    rmse = float(np.sqrt(np.mean(delta * delta)))
    scale = max(float(np.sqrt(np.mean(expected * expected))), 1e-12)
    return {'finite': bool(np.isfinite(actual).all()), 'max_abs_error': float(np.max(np.abs(delta))),
            'rmse': rmse, 'normalized_rmse': rmse / scale}


def validate(args):
    import coremltools as ct
    torch.set_num_threads(args.threads)
    torch.manual_seed(41)
    synthetic = args.checkpoint is None
    cfg = ProConfig(dim=24, ffn_dim=40, num_heads=3, num_layers=2, freq_dim=16) if synthetic else ProConfig.from_json(Path(args.checkpoint) / 'config.json')
    profile = Profile(32, 32) if synthetic else Profile(args.height, args.width)
    checkpoint = None if synthetic else Path(args.checkpoint) / 'diffusion_pytorch_model.safetensors'
    def load(module, kind, start=0):
        return module.eval().requires_grad_(False) if synthetic else load_component(module, checkpoint, kind, start)
    timing = load(TimeEmbedding(cfg), 'time')
    with torch.inference_mode():
        time_embeddings, time_modulations = timing(torch.tensor([1000., 625.]))
    del timing
    models = Path(args.models)
    models.mkdir(parents=True, exist_ok=True)
    manifest_path = models / 'dit_manifest.json'
    export_manifest = {} if synthetic else json.loads(manifest_path.read_text())
    baseline_directory = Path(args.compare_blocks) if getattr(args, 'compare_blocks', None) else None
    baseline_chunk = None
    if baseline_directory is not None:
        if synthetic:
            raise ValueError('--compare-blocks requires the production checkpoint')
        baseline_manifest = json.loads((baseline_directory / 'dit_manifest.json').read_text())
        for key in ('height', 'width', 'source_checkpoint_sha256', 'schedule'):
            if baseline_manifest[key] != export_manifest[key]:
                raise ValueError(f'Comparison baseline mismatch: {key}')
        baseline_chunk = baseline_manifest['blocks_per_chunk']
        if baseline_chunk < 1 or args.blocks_per_chunk % baseline_chunk:
            raise ValueError('Candidate partition boundaries must align with baseline partitions')
    if not synthetic:
        for key in ('height', 'width', 'blocks_per_chunk', 'query_chunk'):
            if export_manifest[key] != getattr(args, key):
                raise ValueError(f'Validation/export configuration mismatch: {key}')
        # An ANE-layout export is scored against the standard FP32 partition,
        # which it equals in FP32 (tests/test_ane_layout.py), so the gate
        # measures conversion and precision error only.
    random = lambda shape: torch.randn(shape)
    components = [('patch', lambda: load(Patch(cfg, profile), 'patch'),
                   lambda: {'latent': random(profile.latent_shape), 'reference': random(profile.latent_shape)}, 'hidden'),
                  ('conditioning', lambda: load(AudioProjection(cfg.dim, intermediate=16, channels=8, layers=3) if synthetic else AudioProjection(cfg.dim), 'conditioning'),
                   lambda: {'audio': random((1, 33, 5, 3, 8) if synthetic else (1, 33, 5, 12, 768))}, 'context')]
    for index, start in enumerate(range(0, cfg.num_layers, args.blocks_per_chunk)):
        end = min(start + args.blocks_per_chunk, cfg.num_layers)
        components.append((f'blocks_{index:02d}',
                           lambda s=start, e=end: load(Blocks(cfg, profile, s, e, time_modulations, args.query_chunk), 'blocks', s),
                           lambda: {'hidden': random((1, profile.tokens, cfg.dim)), 'context': random((1, 9, 32, cfg.dim)), 'timestep': torch.tensor([1000.])}, 'hidden_out'))
    components.append(('head', lambda: load(OutputHead(cfg, profile, time_embeddings), 'head'),
                       lambda: {'hidden': random((1, profile.tokens, cfg.dim)), 'timestep': torch.tensor([1000.])}, 'flow'))
    selected = {name.strip() for name in args.only.split(',')} if args.only else {c[0] for c in components}
    unknown = selected - {c[0] for c in components}
    if unknown:
        raise ValueError(f'Unknown components: {sorted(unknown)}')
    report = {'environment': environment(), 'workload': 'Synthetic small architecture' if synthetic else 'Original PRO checkpoint, random inputs, fixed profile',
              'height': profile.height, 'width': profile.width, 'schedule': [1000, 625, 0],
              'scope': 'Numerical component checks only; not end-to-end animation quality or target-device feasibility',
              'source_checkpoint': None if synthetic else str(checkpoint.resolve()),
              'source_checkpoint_sha256': export_manifest.get('source_checkpoint_sha256'),
              'compression': export_manifest.get('compression', 'none'),
              'precision': export_manifest.get('precision', 'mixed'),
              'precision_policy': export_manifest.get('precision_policy'),
              'fuse_qkv': export_manifest.get('fuse_qkv', False),
              'variant': export_manifest.get('variant', 'baseline'),
              'layout': export_manifest.get('layout', 'standard'),
              'comparison_baseline': str(baseline_directory.resolve()) if baseline_directory else None,
              'models_directory': str(models.resolve()),
              'selected_components': sorted(selected),
              'input_distribution': 'Deterministic independent standard normal inputs, rounded to FP16; not real image/audio activations',
              'models': {}}
    report_path = models / 'dit_validation.json'
    limit = .01 if synthetic else .05
    for name, factory, inputs_factory, output in components:
        if name not in selected:
            continue
        torch.manual_seed(41 + sum(name.encode('utf-8')))
        module, inputs = factory(), inputs_factory()
        if not synthetic and isinstance(module, Blocks) and export_manifest.get('variant', 'baseline') != 'baseline':
            # Score the conversion of the exported variant against the same
            # variant in FP32 PyTorch, not against the dense baseline blocks.
            from .ffn_pruning import reduce_partition
            records = export_manifest.get('variant_transforms', {}).get(name, [])
            if not records:
                raise ValueError(f'{name}: variant export lacks variant_transforms records')
            for record in records:
                if record['transform'] == 'calibrated_ffn_reduction':
                    from .ffn_pruning import _file_sha256
                    for layer, expected in zip(range(*record['layers']), record['fit_sha256']):
                        path = Path(record['fit_dir']) / f'block_{layer:02d}.pt'
                        if expected is not None and (not path.exists() or _file_sha256(path) != expected):
                            raise ValueError(f'{name}: {path} differs from the fit used at export')
                    rebuilt = reduce_partition(module, record['removed_fraction'], record['fit_dir'])
                    if rebuilt['ffn_widths'] != record['ffn_widths']:
                        raise ValueError(f'{name}: rebuilt FFN widths differ from the export')
                elif record['transform'] == 'fold_cross_attention_output':
                    module.fold_cross_output()
                else:
                    raise ValueError(f'{name}: unknown variant transform {record["transform"]}')
        destination = models / (name + '.mlpackage')
        if synthetic:
            convert_component(module, inputs, output, destination, 'none')
        elif not destination.exists():
            raise FileNotFoundError(destination)
        print(f'Validating {name} with CPU_ONLY', flush=True)
        mlmodel = ct.models.MLModel(str(destination), compute_units=ct.ComputeUnit.CPU_ONLY)
        baseline_models = []
        if baseline_directory is not None and name.startswith('blocks_'):
            start = int(name.split('_')[1]) * args.blocks_per_chunk
            end = min(start + args.blocks_per_chunk, cfg.num_layers)
            for layer in range(start, end, baseline_chunk):
                path = baseline_directory / f'blocks_{layer // baseline_chunk:02d}.mlpackage'
                baseline_models.append(ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY))
        cases = []
        for step in ([1000., 625.] if 'timestep' in inputs else [None]):
            if step is not None:
                inputs['timestep'] = torch.tensor([step])
            arrays = {key: value.numpy().astype(np.float16) for key, value in inputs.items()}
            # Give both implementations identical rounded input values.
            with torch.inference_mode():
                expected = module(*(torch.from_numpy(arrays[key]).float() for key in inputs)).numpy()
            predicted = np.asarray(mlmodel.predict(arrays)[output])
            result = errors(predicted, expected)
            if baseline_models:
                baseline_hidden = arrays['hidden']
                for baseline_model in baseline_models:
                    baseline_hidden = baseline_model.predict(dict(arrays, hidden=baseline_hidden))[output]
                delta = errors(predicted, baseline_hidden)
                delta['identical_elements'] = int(np.count_nonzero(predicted == baseline_hidden))
                delta['elements'] = int(predicted.size)
                result['baseline_blocks_delta'] = delta
                del baseline_model, baseline_hidden
            result['timestep'] = step
            result['passed'] = result['finite'] and result['normalized_rmse'] <= limit
            cases.append(result)
            print(json.dumps(result), flush=True)
        report['models'][name] = {'cases': cases, 'normalized_rmse_limit': limit}
        report['passed'] = all(case['passed'] for model in report['models'].values() for case in model['cases'])
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        del module, inputs, mlmodel, baseline_models, expected, predicted
        gc.collect()
    if not report['passed']:
        raise RuntimeError(f'Numerical validation failed; inspect {report_path}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--models', default='artifacts/coreml/smoke_dit')
    parser.add_argument('--height', type=int, default=576)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--blocks-per-chunk', type=int, default=3)
    parser.add_argument('--query-chunk', type=int, default=0)
    parser.add_argument('--compare-blocks', help='Also compare each candidate block partition to aligned partitions in this baseline export directory, using the same inputs as the FP32 gate')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--only')
    args = parser.parse_args()
    if not 1 <= args.blocks_per_chunk <= 30 or args.query_chunk < 0:
        parser.error('Invalid partition or query chunk size')
    validate(args)


if __name__ == '__main__':
    main()
