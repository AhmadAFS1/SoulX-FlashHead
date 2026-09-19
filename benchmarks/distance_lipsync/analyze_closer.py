"""Landmark comparison for baseline close versus two tighter Indian-male clips."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from analyze import correlation, landmarks_for_video


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--baseline", type=Path,
                        default=Path("benchmarks/distance_lipsync/evidence-indian-male-20260916/close-seed-50.mp4"))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    paths = {
        "close-100": args.baseline,
        "closer-125": args.directory / "closer-125-seed-50.mp4",
        "closest-150": args.directory / "closest-150-seed-50.mp4",
    }
    rows, series = [], {}
    for label, path in paths.items():
        opening, face_width, eye_span = landmarks_for_video(path)
        series[label] = opening
        values = np.array([v for v in opening if v is not None])
        widths = np.array([v for v in face_width if v is not None])
        eyes = np.array([v for v in eye_span if v is not None])
        changes = np.array([abs(b - a) for a, b in zip(opening, opening[1:]) if a is not None and b is not None])
        rows.append({
            "label": label, "path": str(path), "frames": len(opening), "detected": len(values),
            "detection_rate": len(values) / max(len(opening), 1),
            "mean_face_width_px": float(widths.mean()), "mean_eye_span_px": float(eyes.mean()),
            "mean_opening": float(values.mean()), "p95_opening": float(np.percentile(values, 95)),
            "opening_sd": float(values.std()), "mean_abs_frame_change": float(changes.mean()),
            "opening_series": opening,
        })

    comparisons = []
    baseline = series["close-100"]
    for label in ("closer-125", "closest-150"):
        lag_scores = {lag: correlation(baseline, series[label], lag) for lag in range(-5, 6)}
        valid = {lag: score for lag, score in lag_scores.items() if score is not None}
        best_lag = max(valid, key=valid.get)
        comparisons.append({
            "label": label,
            "zero_lag_correlation_vs_close": correlation(baseline, series[label], 0),
            "best_lag_frames": best_lag, "best_lag_ms": best_lag * 40,
            "best_lag_correlation": valid[best_lag], "lag_scan_frames": lag_scores,
        })

    output = {
        "metric_definition": "MediaPipe inner-lip distance 13/14 divided by eye-corner distance 33/263.",
        "comparison_definition": f"Pearson correlation against {args.baseline}; lag scan ±5 frames.",
        "limitations": "One avatar, utterance, and seed. Landmark motion proxy, not phoneme accuracy or perceived audiovisual sync.",
        "rows": rows, "comparisons": comparisons,
    }
    (args.directory / "mouth-motion.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "opening_series"}), flush=True)
    for row in comparisons:
        print(json.dumps({k: v for k, v in row.items() if k != "lag_scan_frames"}), flush=True)


if __name__ == "__main__":
    main()
