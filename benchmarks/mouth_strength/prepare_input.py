"""Prepare the existing male idle fixture without GPU inference."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from soulx_rtc.idle_cache import IdleCache

parser = argparse.ArgumentParser()
parser.add_argument('--idle', required=True)
parser.add_argument('--output', required=True, type=Path)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=False)
cache = IdleCache()
clip = cache.acquire(args.idle, 320, 576)
assert clip is not None
try:
    Image.fromarray(clip.frames[0]).save(args.output/'anchor.png')
    motion = np.stack([clip.at(i,25) for i in range(31,40)])
    np.save(args.output/'motion.npy', motion)
    (args.output/'provenance.json').write_text(json.dumps(dict(
        source=args.idle, source_sha256=hashlib.sha256(Path(args.idle).read_bytes()).hexdigest(),
        output_indices=list(range(31,40)), fps=25, width=320, height=576,
        motion_sha256=hashlib.sha256(motion.tobytes()).hexdigest()), indent=2))
finally:
    cache.release(clip)
