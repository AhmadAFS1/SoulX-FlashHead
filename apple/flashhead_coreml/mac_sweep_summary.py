"""Summarise ``mac-speed-sweep.sh`` runs: warm FPS, speed-up and quality.

Warm FPS follows apple/PERFORMANCE.md: frames after the first window divided
by their summed generation time (``inference_seconds`` per window in each
run's ``timings.json``), pooled over repeats. Quality compares each arm's
first repeat with the baseline's first repeat: identical frames, frame PSNR,
and the MediaPipe lip-opening correlation / oral edge ratio used by the CPU
A/B tools (``quality.py``). Hardware is whatever Mac ran the sweep; each run's
``timings.json`` records it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_frames(run: Path):
    from PIL import Image
    files = sorted(run.glob('frame-*.png'))
    return np.stack([np.asarray(Image.open(f).convert('RGB')) for f in files]) if files else None


def warm(runs):
    frames = seconds = 0.0
    for run in runs:
        data = json.loads((run / 'timings.json').read_text())
        for window in data.get('windows', [])[1:]:
            frames += window['frames']
            seconds += window['inference_seconds']
    return frames / seconds if seconds else float('nan')


def main():
    from . import quality
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--baseline', default='baseline')
    args = parser.parse_args()
    arms = {}
    for run in sorted(args.sweep.glob('*-r*')):
        if run.is_dir() and (run / 'timings.json').exists():
            arms.setdefault(run.name.rsplit('-r', 1)[0], []).append(run)
    if args.baseline not in arms:
        raise SystemExit(f'No baseline runs named {args.baseline!r} in {args.sweep}')
    base_fps = warm(arms[args.baseline])
    base_frames = load_frames(sorted(arms[args.baseline])[0])
    base_rows = quality.landmarks(base_frames) if base_frames is not None else None
    hardware = json.loads((sorted(arms[args.baseline])[0] / 'timings.json').read_text()).get('hardware', {})
    print(f"Hardware: {hardware.get('gpu_model', 'see timings.json')}  (warm FPS = frames after window 1 / generation time)")
    print(f"{'arm':22s} {'runs':>4s} {'warm FPS':>9s} {'speed-up':>9s} {'identical':>10s} {'min PSNR':>9s} {'lip corr':>9s} {'edge':>6s}")
    summary = {}
    for name, runs in arms.items():
        fps = warm(runs)
        row = {'runs': len(runs), 'warm_fps': fps, 'speedup': fps / base_fps}
        frames = load_frames(sorted(runs)[0])
        if frames is not None and base_frames is not None and frames.shape == base_frames.shape:
            same = int(sum(np.array_equal(a, b) for a, b in zip(frames, base_frames)))
            psnrs = [quality.psnr(a, b) for a, b in zip(frames, base_frames)]
            finite = [p for p in psnrs if np.isfinite(p)]
            paired = quality.paired(base_rows, quality.landmarks(frames)) if name != args.baseline else {}
            row.update(identical=f'{same}/{len(frames)}', min_psnr=min(finite) if finite else float('inf'),
                       lip_corr=paired.get('opening_correlation'), edge=paired.get('median_edge_ratio_on_both_open'))
        summary[name] = row
        fmt = lambda v, f: (f.format(v) if isinstance(v, (int, float)) and v is not None and np.isfinite(v) else
                            ('inf' if isinstance(v, float) else (v or '-')))
        print(f"{name:22s} {row['runs']:4d} {fmt(fps, '{:9.2f}'):>9s} {fmt(row['speedup'], '{:8.2f}x'):>9s} "
              f"{row.get('identical', '-'):>10s} {fmt(row.get('min_psnr'), '{:8.1f}'):>9s} "
              f"{fmt(row.get('lip_corr'), '{:9.3f}'):>9s} {fmt(row.get('edge'), '{:6.3f}'):>6s}")
    (args.sweep / 'summary.json').write_text(json.dumps({'hardware': hardware, 'arms': summary}, indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
