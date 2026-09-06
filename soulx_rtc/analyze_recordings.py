"""Decode receiver recordings and quantify visible chunk/clip boundaries.

RGB MAE is an unaligned diagnostic, not a perceptual or lip-sync quality score.
The recordings have already passed through lossy H264 and a recorder encode.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import av
import numpy as np
from PIL import Image

from .replay_lab import mae


def inspect(path, anchor):
    first = last = previous = None
    diffs, pts, boundaries = [], [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        width, height, fps = stream.width, stream.height, float(stream.average_rate)
        for index, frame in enumerate(container.decode(stream)):
            rgb = frame.to_ndarray(format="rgb24")
            if first is None:
                first = rgb.copy()
            if previous is not None:
                delta = mae(previous, rgb)
                diffs.append(delta)
                if index % 24 == 0:
                    boundaries.append({"frame": index, "time_s": index / fps, "rgb_mae": delta})
            pts.append(float(frame.pts * frame.time_base))
            previous = last = rgb
    if first is None:
        raise ValueError(f"No video frames: {path}")
    # Same PIL BILINEAR resize/center crop as the conditioning image path.
    scale = max(height / anchor.height, width / anchor.width)
    ref = anchor.resize((math.ceil(anchor.width * scale), math.ceil(anchor.height * scale)), Image.Resampling.BILINEAR)
    left, top = round((ref.width - width) / 2), round((ref.height - height) / 2)
    ref = np.asarray(ref.crop((left, top, left + width, top + height)))
    with av.open(str(path)) as container:
        audio = container.streams.audio[0]
        audio_samples = sum(frame.samples for frame in container.decode(audio))
        audio_rate = audio.rate
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    row = dict(file=path.name, sha256=digest.hexdigest(), frames=len(pts),
               width=width, height=height, fps=fps, video_duration_s=len(pts)/fps,
               audio_decoded_samples=audio_samples, audio_rate=audio_rate,
               pts_monotonic=all(b > a for a, b in zip(pts, pts[1:])),
               adjacent_rgb_mae_median=float(np.median(diffs)),
               adjacent_rgb_mae_p95=float(np.percentile(diffs, 95)),
               adjacent_rgb_mae_max=max(diffs), chunk_boundaries=boundaries,
               first_to_conditioning_rgb_mae=mae(first, ref),
               last_to_conditioning_rgb_mae=mae(last, ref))
    return row, (first, last)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", required=True)
    parser.add_argument("--videos", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    with Image.open(args.anchor) as image:
        anchor = image.convert("RGB")
    rows, ends = [], []
    for name in args.videos:
        row, endpoints = inspect(Path(name), anchor)
        rows.append(row)
        ends.append(endpoints)
    seams = [dict(source=rows[i]["file"], target=rows[j]["file"],
                  rgb_mae=mae(ends[i][1], ends[j][0]),
                  exact=np.array_equal(ends[i][1], ends[j][0]))
             for i in range(len(rows)) for j in range(len(rows)) if i != j]
    result = {"method": __doc__, "recordings": rows, "independent_clip_seams": seams}
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
