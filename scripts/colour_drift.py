"""Per-window mean-RGB drift of a run's raw frames against its own window 0.

usage: colour_drift.py <raw.npy> [<raw.npy> ...]

raw.npy is what run.py --save-raw writes: (T, H, W, 3) uint8. Each 28-frame window's mean
colour is compared with window 0; the slope and the largest drift expose the slow colour creep
that the recurrent motion re-encode can introduce and that a single-window comparison misses.
Every file is measured against itself, so the figure does not depend on any other run.
"""
import sys

import numpy as np

W = 28  # useful frames per window at 576x320


def drift(path):
    a = np.load(path, mmap_mode="r")
    n = a.shape[0] // W
    means = np.stack([a[i * W:(i + 1) * W].astype(np.float32).mean(axis=(0, 1, 2)) for i in range(n)])
    rel = means - means[0]
    slope = np.polyfit(np.arange(n), rel, 1)[0]
    return rel, slope, n


if len(sys.argv) < 2:
    sys.exit(__doc__)
for path in sys.argv[1:]:
    rel, slope, n = drift(path)
    print(f"=== {path} ({n} windows) ===")
    print("  win : dR      dG      dB     (vs window 0)")
    for i in range(n):
        print(f"  {i:3d} : {rel[i, 0]:+7.3f} {rel[i, 1]:+7.3f} {rel[i, 2]:+7.3f}")
    print(f"  slope/window: R{slope[0]:+.4f} G{slope[1]:+.4f} B{slope[2]:+.4f}  "
          f"|max drift| {np.abs(rel).max():.3f}/255")
    print()
