"""Compare every displayed pixel and consecutive-frame change in paired renders."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def psnr(rmse):
    return 20 * math.log10(255 / rmse) if rmse else None


def compare(baseline, candidate, roi=None):
    source = sorted(baseline.glob("frame-*.png"))
    other = sorted(candidate.glob("frame-*.png"))
    if not source or [p.name for p in source] != [p.name for p in other]:
        raise ValueError("Paired runs must contain matching frame names")
    maximum = worst_rmse = worst_roi_rmse = worst_motion_rmse = 0.0
    exact = 0
    previous = None
    digests = [hashlib.sha256(), hashlib.sha256()]
    for reference, result in zip(source, other):
        with Image.open(reference) as image:
            before = np.array(image.convert("RGB"))
        with Image.open(result) as image:
            after = np.array(image.convert("RGB"))
        if before.shape != after.shape:
            raise ValueError(f"Shape mismatch for {reference.name}")
        for digest, pixels in zip(digests, (before, after)):
            digest.update(pixels.tobytes())
        delta = after.astype(np.float64) - before
        exact += int(not np.any(delta))
        maximum = max(maximum, float(np.max(np.abs(delta))))
        worst_rmse = max(worst_rmse, float(np.sqrt(np.mean(delta ** 2))))
        if roi is not None:
            left, top, right, bottom = roi
            h, w = before.shape[:2]
            if not (0 <= left < right <= w and 0 <= top < bottom <= h):
                raise ValueError("ROI must be inside the frame")
            worst_roi_rmse = max(worst_roi_rmse, float(np.sqrt(np.mean(delta[top:bottom, left:right] ** 2))))
        if previous is not None:
            worst_motion_rmse = max(worst_motion_rmse, float(np.sqrt(np.mean((delta - previous) ** 2))))
        previous = delta
    return {"frames": len(source), "exact_frames": exact, "max_channel_error_u8": maximum,
            "worst_frame_rmse_u8": worst_rmse, "worst_frame_psnr_db": psnr(worst_rmse),
            "roi_xyxy": roi, "worst_roi_rmse_u8": worst_roi_rmse if roi else None,
            "worst_temporal_delta_rmse_u8": worst_motion_rmse,
            "baseline_rgb_sha256": digests[0].hexdigest(), "candidate_rgb_sha256": digests[1].hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--roi", nargs=4, type=int, metavar=("LEFT", "TOP", "RIGHT", "BOTTOM"))
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.candidate, args.roi), indent=2))


if __name__ == "__main__":
    main()
