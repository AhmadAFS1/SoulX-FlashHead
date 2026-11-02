"""CPU wall time of the 30 PRO blocks, baseline vs a transformer variant.

Each layer is loaded alone, timed in FP32 PyTorch at the production token
count (1,980 at pro320) for both timesteps, transformed in place, and timed
again with interleaved repeats. This is a CPU-only proxy for how removing
multiply-accumulates changes dense execution time. It is not an Apple GPU or
Neural Engine measurement and says nothing about Core ML kernel selection.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch

from .cpu_runtime import ROOT, environment
from .dit import Blocks, ProConfig, Profile, TimeEmbedding, load_component
from .variants import build_transforms


class _View:
    def __init__(self, blocks):
        self.blocks = blocks


def macs_per_block_step(tokens, dim, ffn, heads, keys=32, frames=9, folded=False):
    cross_o = 2 * tokens * dim * dim
    folded_cost = frames * heads * keys * (dim // heads) * dim + tokens * heads * keys * dim
    return {
        'ffn': 2 * tokens * dim * ffn,
        'self_qkvo': 4 * tokens * dim * dim,
        'self_attention': 2 * tokens * tokens * dim,
        'cross_q': tokens * dim * dim,
        'cross_o_and_av': folded_cost if folded else cross_o // 2 + tokens * keys * dim,
        'cross_kv': 2 * frames * keys * dim * dim,
        'cross_scores': tokens * keys * dim,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', required=True)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    checkpoint_dir = ROOT / 'models/SoulX-FlashHead-1_3B/Model_Pro'
    checkpoint = checkpoint_dir / 'diffusion_pytorch_model.safetensors'
    cfg = ProConfig.from_json(checkpoint_dir / 'config.json')
    profile = Profile(320, 176)
    with torch.inference_mode():
        _, tm = load_component(TimeEmbedding(cfg), checkpoint, 'time')(torch.tensor([1000., 625.]))
    transforms, description = build_transforms(args.variant)
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(1, profile.tokens, cfg.dim, generator=generator)
    context = torch.randn(1, 9, 32, cfg.dim, generator=generator)
    rows, totals = [], {'baseline': 0.0, 'variant': 0.0}
    for layer in range(cfg.num_layers):
        module = load_component(Blocks(cfg, profile, layer, layer + 1, tm), checkpoint, 'blocks', layer)
        variant = load_component(Blocks(cfg, profile, layer, layer + 1, tm), checkpoint, 'blocks', layer)
        for transform in transforms:
            ratio = getattr(transform, 'ffn_ratio', None)
            if ratio is not None:
                from .ffn_pruning import reduce_partition
                reduce_partition(variant, ratio, transform.fit_dir)
            else:
                transform(_View(variant))
        times = {'baseline': [], 'variant': []}
        with torch.inference_mode():
            for t in (1000., 625.):
                timestep = torch.tensor([t])
                module(hidden, context, timestep), variant(hidden, context, timestep)  # warm-up
                for _ in range(args.repeats):
                    for name, m in (('baseline', module), ('variant', variant)):
                        started = time.perf_counter()
                        m(hidden, context, timestep)
                        times[name].append(time.perf_counter() - started)
        row = {'layer': layer, 'ffn_width': variant.blocks[0].ffn[0].out_features,
               **{f'{k}_median_seconds': statistics.median(v) for k, v in times.items()}}
        for k in totals:
            totals[k] += 2 * row[f'{k}_median_seconds']  # two timesteps per window, median per call
        rows.append(row)
        print(json.dumps(row), flush=True)
        del module, variant
    widths = [r['ffn_width'] for r in rows]
    folded = 'fold' in args.variant.split('+')
    base = macs_per_block_step(profile.tokens, cfg.dim, cfg.ffn_dim, cfg.num_heads)
    total_base = 2 * cfg.num_layers * sum(base.values())
    total_variant = 2 * sum(sum(macs_per_block_step(profile.tokens, cfg.dim, w, cfg.num_heads, folded=folded).values())
                            for w in widths)
    report = {'environment': environment(args.threads), 'variant': args.variant, 'description': description,
              'profile': {'height': profile.height, 'width': profile.width, 'tokens': profile.tokens},
              'scope': 'CPU-only FP32 PyTorch block timing; not Core ML, Apple GPU or Neural Engine',
              'per_window_block_seconds': totals,
              'cpu_speedup': totals['baseline'] / totals['variant'],
              'static_macs_per_window': {'baseline': total_base, 'variant': total_variant,
                                         'reduction': 1 - total_variant / total_base},
              'layers': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('per_window_block_seconds', 'cpu_speedup', 'static_macs_per_window')}))


if __name__ == '__main__':
    main()
