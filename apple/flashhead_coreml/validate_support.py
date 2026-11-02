"""Full-shape numerical support-model checks using real weights and CPU_ONLY.

This verifies conversion against the export wrappers. It is neither a GPU or
iPhone speed benchmark nor an end-to-end perceptual quality evaluation.
"""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import coremltools as ct
import numpy as np
import torch

from .export_support import ROOT, Audio, TinyDecode, TinyEncode, Reference, module_at


def environment():
    def command(*args):
        return subprocess.check_output(args, text=True).strip()
    graphics = json.loads(command('system_profiler', 'SPDisplaysDataType', '-json'))['SPDisplaysDataType']
    return {
        'date_utc': datetime.now(timezone.utc).isoformat(),
        'execution': 'CPU-only Core ML prediction and CPU PyTorch reference; no GPU or ANE inference',
        'gpu_model_present': [device.get('sppci_model', 'unverified') for device in graphics],
        'gpu_used': False,
        'physical_system_memory_bytes': int(command('sysctl', '-n', 'hw.memsize')),
        'physical_vram': 'Apple unified memory shared with system RAM; no dedicated VRAM',
        'visible_vram': 'Not allocated or queried; CPU-only validation',
        'hardware_evidence': 'system_profiler SPDisplaysDataType; sysctl hw.memsize; Core ML ComputeUnit.CPU_ONLY; all torch tensors on CPU',
        'os': platform.platform(),
        'driver_runtime': 'macOS-provided Core ML runtime; GPU driver not used',
        'runtime': {'torch': torch.__version__, 'coremltools': ct.__version__, 'numpy': np.__version__, 'python': platform.python_version()},
        'co_resident_load': 'Other Core ML export/validation jobs may be running; timings are not performance benchmarks',
    }


def digest_tree(path):
    digest = hashlib.sha256()
    paths = sorted(p for p in path.rglob('*') if p.is_file()) if path.is_dir() else [path]
    for item in paths:
        if path.is_dir():
            digest.update(str(item.relative_to(path)).encode())
        with item.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest()


def errors(actual, expected):
    actual, expected = actual.astype(np.float64), expected.astype(np.float64)
    if not (np.isfinite(actual).all() and np.isfinite(expected).all()):
        return {'finite': False, 'passed': False, 'output_shape': list(actual.shape),
                'actual_nonfinite_count': int(np.count_nonzero(~np.isfinite(actual))),
                'expected_nonfinite_count': int(np.count_nonzero(~np.isfinite(expected)))}
    delta = actual - expected
    rmse = float(np.sqrt(np.mean(delta * delta)))
    expected_rms = float(np.sqrt(np.mean(expected * expected)))
    cosine = float(np.dot(actual.ravel(), expected.ravel()) /
                   max(np.linalg.norm(actual) * np.linalg.norm(expected), 1e-12))
    result = {'finite': bool(np.isfinite(actual).all() and np.isfinite(expected).all()),
              'normalized_rmse': rmse / max(expected_rms, 1e-12),
              'rmse': rmse, 'expected_rms': expected_rms, 'cosine_similarity': cosine,
              'max_abs_error': float(np.max(np.abs(delta))),
              'p99_abs_error': float(np.percentile(np.abs(delta), 99)),
              'output_shape': list(actual.shape)}
    result['passed'] = result['finite'] and result['normalized_rmse'] <= .01 and cosine >= .999 and result['max_abs_error'] <= .05
    return result


def factory(name, height, width):
    if name == 'audio':
        from transformers import Wav2Vec2Model
        checkpoint = ROOT/'models/wav2vec2-base-960h'
        model = Wav2Vec2Model.from_pretrained(str(checkpoint), attn_implementation='eager').float().eval()
        return Audio(model), torch.randn(1, 128000) * .1, 'waveform', 'features', checkpoint
    if name == 'reference':
        checkpoint = ROOT/'models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth'
        vendor = module_at('soulx_validation_wan', ROOT/'flash_head/wan/modules/vae.py')
        model = vendor.WanVAE(vae_path=str(checkpoint), dtype=torch.float32, device='cpu')
        return Reference(model, vendor.count_conv3d), torch.rand(1, 3, 1, height, width) * 2 - 1, 'image', 'reference', checkpoint
    checkpoint = ROOT/'models/tiny_vae/taehv/taew2_1.pth'
    vendor = module_at('soulx_validation_taehv', checkpoint.parent/'taehv.py')
    model = vendor.TAEHV(checkpoint_path=str(checkpoint), arch_name='taew2_1').float().eval()
    if name == 'encoder':
        return TinyEncode(model, vendor.MemBlock), torch.rand(1, 5, 3, height, width), 'frames', 'motion', checkpoint
    return TinyDecode(model, vendor.MemBlock), torch.randn(1, 9, 16, height//8, width//8), 'latent', 'frames', checkpoint


def validate(args):
    torch.set_num_threads(args.threads)
    report = {'environment': environment(), 'workload': 'Original real checkpoint weights; fixed full-shape random inputs; one case per component',
              'height': args.height, 'width': args.width,
              'scope': 'Conversion versus FP32 export wrapper, identical rounded inputs; does not establish end-to-end CUDA or visual parity',
              'gates': {'normalized_rmse_max': .01, 'cosine_similarity_min': .999, 'max_abs_error_max': .05},
              'models': {}, 'passed': False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    def save():
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    save()
    for name in args.components.split(','):
        if name not in ('audio', 'encoder', 'decoder', 'reference'):
            raise ValueError(f'Unknown component {name}')
        seed = 32001 + sum(name.encode())
        torch.manual_seed(seed)
        package = args.models/(name + '.mlpackage')
        print(f'VALIDATING {name}: CPU_ONLY, {args.height}x{args.width}, real weights', flush=True)
        module, inputs, input_name, output_name, checkpoint = factory(name, args.height, args.width)
        module = module.eval().requires_grad_(False)
        arrays = inputs.numpy().astype(np.float32 if name in ('audio', 'reference') else np.float16)
        start = time.monotonic()
        with torch.inference_mode():
            expected = module(torch.from_numpy(arrays).float()).numpy()
        torch_seconds = time.monotonic() - start
        del module, inputs
        gc.collect()
        start = time.monotonic()
        converted = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.CPU_ONLY)
        load_seconds = time.monotonic() - start
        start = time.monotonic()
        actual = np.asarray(converted.predict({input_name: arrays})[output_name])
        prediction_seconds = time.monotonic() - start
        result = errors(actual, expected)
        result.update({'seed': seed, 'input_shape': list(arrays.shape), 'input_dtype': str(arrays.dtype),
                       'torch_cpu_seconds': torch_seconds, 'coreml_load_seconds': load_seconds,
                       'coreml_cpu_prediction_seconds': prediction_seconds,
                       'timing_scope': 'One correctness run under unisolated co-resident load; not benchmark throughput',
                       'checkpoint': str(checkpoint), 'checkpoint_sha256': digest_tree(checkpoint),
                       'checkpoint_digest_kind': 'path-and-content tree SHA-256' if checkpoint.is_dir() else 'file SHA-256',
                       'package': str(package), 'package_tree_sha256': digest_tree(package)})
        report['models'][name] = result
        report['passed'] = all(value['passed'] for value in report['models'].values())
        save()
        print(json.dumps(result, indent=2), flush=True)
        del converted, actual, expected, arrays
        gc.collect()
    if not report['passed']:
        raise RuntimeError(f'Support conversion gate failed; inspect {args.report}')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, default=ROOT/'artifacts/coreml/pro576')
    parser.add_argument('--report', type=Path, default=ROOT/'apple/validation/support_coreml.json')
    parser.add_argument('--components', default='audio,encoder,decoder')
    parser.add_argument('--height', type=int, default=576)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--threads', type=int, default=4)
    validate(parser.parse_args())


if __name__ == '__main__':
    main()
