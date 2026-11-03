"""Summarise CPU A/B renders: metrics tables, floors and comparison videos.

Expects ``run-cpu-ab.sh`` output under ROOT/<arm>/<clip>/. The INT8 baseline
arm is the reference. Floors: ``fp32-baseline`` (how far the shipped INT8/FP16
emulation already is from FP32) and ``coreml-int8-baseline-seed43`` (how much
the closed loop changes with only the noise seed). Teacher-forced arms
(``*-forced``) reuse the baseline's motion latents per window.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from . import quality

BASELINE = 'coreml-int8-baseline'
LABELS = {
    'coreml-int8-baseline': 'Shipped INT8 PRO\n(baseline)',
    'coreml-int8-fold': 'Exact cross-attn fold\n-3.0% MACs',
    'coreml-int8-ffnk2_fold': 'Fold + adaptive FFN\n-4.6% MACs',
    'coreml-int8-ffn12.5_fold': 'Uniform FFN -12.5%\n(negative control)',
    'fp32-baseline': 'FP32 reference\n(no quantization)',
    'coreml-int8-baseline-seed43': 'Baseline, seed 43\n(seed floor)',
}


def clean(value):
    if isinstance(value, float) and math.isinf(value):
        return 'identical'
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items() if k not in ('timing_environment',)}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def arms(root: Path):
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('artifacts/cpu-validation/renders'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--videos', type=Path)
    args = parser.parse_args()
    root = args.root
    cache, report = {}, {'clips': {}}
    for clip in sorted(p.name for p in (root / BASELINE).iterdir() if (p / 'run.json').exists()):
        base = root / BASELINE / clip
        entry = {'baseline': clean(quality.run_summary(base, cache)), 'arms': {}}
        for arm in arms(root):
            run = root / arm / clip
            if arm == BASELINE or not (run / 'run.json').exists():
                continue
            entry['arms'][arm] = {'summary': clean(quality.run_summary(run, cache)),
                                  'vs_baseline': clean(quality.compare(base, run, cache))}
            print(clip, arm, json.dumps(entry['arms'][arm]['vs_baseline']['landmarks']), flush=True)
        report['clips'][clip] = entry
        if args.videos:
            panels = [(LABELS[a], root / a / clip) for a in ('fp32-baseline', BASELINE, 'coreml-int8-fold', 'coreml-int8-ffnk2_fold')
                      if (root / a / clip / 'run.json').exists()]
            if len(panels) >= 2:
                index = [label for label, _ in panels].index(LABELS[BASELINE])
                quality.comparison_video(args.videos / f'{clip}-ab.mp4', panels, diff_pair=(index, len(panels) - 1),
                                         title=f'{clip}: CPU-emulated Core ML INT8, seed 42 (Intel Xeon CPU, no GPU)')
            negative = root / 'coreml-int8-ffn12.5_fold' / clip
            if (negative / 'run.json').exists():
                quality.comparison_video(args.videos / f'{clip}-negative-control.mp4',
                                         [(LABELS[BASELINE], base), (LABELS['coreml-int8-ffn12.5_fold'], negative)],
                                         title=f'{clip}: what a uniform 12.5% FFN cut does (rejected)')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1) + '\n')


if __name__ == '__main__':
    main()
