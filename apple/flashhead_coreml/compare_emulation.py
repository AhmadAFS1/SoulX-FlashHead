"""Compare the CPU Core ML emulation with the recorded Core ML component errors.

``validate_dit`` measured each production Core ML partition (CPU_ONLY on an
Apple M5 Pro) against FP32 PyTorch using seeded random inputs. This tool rebuilds
the same seeded modules and inputs, runs ``cpu_runtime``'s INT8/FP16 emulation,
and reports both normalised errors side by side. Agreement in magnitude supports
using the emulation for A/B rendering; it does not make it Core ML.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .cpu_runtime import (ROOT, apply_int8_weights, block_forward, environment, identity, round16,
                          round_parameters16)
from .dit import Blocks, OutputHead, Patch, ProConfig, Profile, TimeEmbedding, load_component
from .validate_dit import errors


def emulate_blocks(module, hidden, context, step, R):
    x = hidden
    cos, sin = R(module.rotary_cos), R(module.rotary_sin)
    modulation = R((module.time_high if step == 0 else module.time_low))
    for block in module.blocks:
        x = block_forward(block, x, context, modulation, cos, sin, step, R)
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT / 'models/SoulX-FlashHead-1_3B/Model_Pro'))
    parser.add_argument('--recorded', default=str(ROOT / 'apple/validation/pro320_dit.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    recorded = json.loads(Path(args.recorded).read_text())
    cfg = ProConfig.from_json(Path(args.checkpoint) / 'config.json')
    profile = Profile(recorded['height'], recorded['width'])
    checkpoint = Path(args.checkpoint) / 'diffusion_pytorch_model.safetensors'
    with torch.inference_mode():
        _, time_modulations = load_component(TimeEmbedding(cfg), checkpoint, 'time')(torch.tensor([1000., 625.]))
    rows = []
    for index in range(10):
        name = f'blocks_{index:02d}'
        # Same construction order and seeds as validate_dit.validate().
        torch.manual_seed(41 + sum(name.encode('utf-8')))
        module = load_component(Blocks(cfg, profile, 3 * index, 3 * index + 3, time_modulations), checkpoint, 'blocks', 3 * index)
        hidden = torch.randn((1, profile.tokens, cfg.dim))
        context = torch.randn((1, 9, 32, cfg.dim))
        with torch.inference_mode():
            expected = {}
            for step, t in enumerate((1000., 625.)):
                h, c = round16(hidden), round16(context)
                expected[step] = module(h, c, torch.tensor([t])).numpy()
                reference = emulate_blocks(module, h, c, step, identity).numpy()
                np.testing.assert_allclose(reference, expected[step], rtol=1e-4, atol=1e-3)
            apply_int8_weights(module)
            round_parameters16(module)
            for step, t in enumerate((1000., 625.)):
                h, c = round16(hidden), round16(context)
                emulated = round16(emulate_blocks(module, h, c, step, round16)).numpy()
                ours = errors(emulated, expected[step])
                theirs = next(case for case in recorded['models'][name]['cases'] if case['timestep'] == t)
                rows.append({'component': name, 'timestep': t,
                             'emulation_normalized_rmse': ours['normalized_rmse'],
                             'recorded_coreml_cpu_normalized_rmse': theirs['normalized_rmse'],
                             'emulation_max_abs_error': ours['max_abs_error'],
                             'recorded_coreml_cpu_max_abs_error': theirs['max_abs_error']})
                print(json.dumps(rows[-1]), flush=True)
        del module
    ratio = [r['emulation_normalized_rmse'] / r['recorded_coreml_cpu_normalized_rmse'] for r in rows]
    report = {'environment': environment(args.threads),
              'recorded_environment': recorded['environment'],
              'scope': 'Emulation fidelity check on validate_dit seeded random inputs; not real activations',
              'rows': rows,
              'ratio_emulation_to_recorded': {'min': min(ratio), 'median': float(np.median(ratio)), 'max': max(ratio)}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report['ratio_emulation_to_recorded']))


if __name__ == '__main__':
    main()
