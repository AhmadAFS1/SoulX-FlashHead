"""Measure pose behavior and attenuation effects for strength-0.5 clips."""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from analyze import correlation, landmarks_for_video


LABELS = ("close-100", "closer-125", "closest-150")
UNMODIFIED = {
    "close-100": Path("benchmarks/distance_lipsync/evidence-indian-male-20260916/close-seed-50.mp4"),
    "closer-125": Path("benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/closer-125-seed-50.mp4"),
    "closest-150": Path("benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/closest-150-seed-50.mp4"),
}


def summarize(label, path, strength):
    opening, face_width, eye_span = landmarks_for_video(path)
    values = np.array([value for value in opening if value is not None])
    widths = np.array([value for value in face_width if value is not None])
    eyes = np.array([value for value in eye_span if value is not None])
    changes = np.array([abs(b - a) for a, b in zip(opening, opening[1:]) if a is not None and b is not None])
    return {
        "label": label, "strength": strength, "path": str(path), "frames": len(opening), "detected": len(values),
        "detection_rate": len(values) / max(len(opening), 1),
        "mean_face_width_px": float(widths.mean()), "mean_eye_span_px": float(eyes.mean()),
        "mean_opening": float(values.mean()), "p95_opening": float(np.percentile(values, 95)),
        "opening_sd": float(values.std()), "mean_abs_frame_change": float(changes.mean()),
        "opening_series": opening,
    }


def comparison(reference, candidate):
    lag_scores = {lag: correlation(reference, candidate, lag) for lag in range(-5, 6)}
    valid = {lag: score for lag, score in lag_scores.items() if score is not None}
    best_lag = max(valid, key=valid.get)
    return {"zero_lag_correlation": correlation(reference, candidate, 0),
            "best_lag_frames": best_lag, "best_lag_ms": best_lag * 40,
            "best_lag_correlation": valid[best_lag], "lag_scan_frames": lag_scores}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    rows, series = [], {}
    for label in LABELS:
        path = args.directory / f"{label}-strength-0p5-seed-50.mp4"
        row = summarize(label, path, 0.5)
        rows.append(row)
        series[(label, 0.5)] = row["opening_series"]
        control = summarize(label, UNMODIFIED[label], 1.0)
        series[(label, 1.0)] = control["opening_series"]

    pose_comparisons = []
    for label in LABELS[1:]:
        result = comparison(series[("close-100", 0.5)], series[(label, 0.5)])
        result["label"] = label
        pose_comparisons.append(result)

    strength_comparisons = []
    for row in rows:
        label = row["label"]
        control_values = np.array([v for v in series[(label, 1.0)] if v is not None])
        result = comparison(series[(label, 1.0)], series[(label, 0.5)])
        result.update({
            "label": label,
            "mean_opening_change_pct": (row["mean_opening"] / control_values.mean() - 1) * 100,
            "p95_opening_change_pct": (row["p95_opening"] / np.percentile(control_values, 95) - 1) * 100,
        })
        strength_comparisons.append(result)

    output = {
        "metric_definition": "MediaPipe inner-lip distance 13/14 divided by eye-corner distance 33/263.",
        "pose_comparison_definition": "Within strength 0.5, compare 1.25x/1.50x trajectories to 1.00x.",
        "strength_comparison_definition": "Within each pose, compare strength 0.5 against its prior unmodified strength-1 clip.",
        "limitations": "One avatar, utterance, and seed; experimental retraced hook; landmark proxy rather than phoneme accuracy.",
        "rows": rows, "pose_comparisons": pose_comparisons, "strength_comparisons": strength_comparisons,
    }
    (args.directory / "mouth-motion.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "opening_series"}), flush=True)
    print("POSE", json.dumps([{k: v for k, v in row.items() if k != "lag_scan_frames"} for row in pose_comparisons]), flush=True)
    print("STRENGTH", json.dumps([{k: v for k, v in row.items() if k != "lag_scan_frames"} for row in strength_comparisons]), flush=True)


if __name__ == "__main__":
    main()
