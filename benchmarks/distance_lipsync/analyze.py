"""Measure face scale and mouth-motion preservation in distance experiment clips.

These are landmark-based output diagnostics, not phoneme-level lip-sync scores.
"""
import argparse
import json
import re
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def landmarks_for_video(path):
    capture = cv2.VideoCapture(str(path))
    opening, face_width, eye_span = [], [], []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=.5, min_tracking_confidence=.5) as mesh:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prediction = mesh.process(rgb)
            if not prediction.multi_face_landmarks:
                opening.append(None); face_width.append(None); eye_span.append(None)
                continue
            marks = prediction.multi_face_landmarks[0].landmark
            def point(index):
                return np.array([marks[index].x * rgb.shape[1], marks[index].y * rgb.shape[0]])
            eyes = np.linalg.norm(point(33) - point(263))
            eye_span.append(float(eyes))
            face_width.append(float(np.linalg.norm(point(234) - point(454))))
            opening.append(float(np.linalg.norm(point(13) - point(14)) / max(eyes, 1e-6)))
    capture.release()
    return opening, face_width, eye_span


def finite_pairs(a, b, lag=0):
    if lag > 0:
        a, b = a[:-lag], b[lag:]
    elif lag < 0:
        a, b = a[-lag:], b[:lag]
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 3:
        return np.array([]), np.array([])
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def correlation(a, b, lag=0):
    x, y = finite_pairs(a, b, lag)
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    cv2.setNumThreads(1)
    rows, series = [], {}
    pattern = re.compile(r"(close|medium|far)-seed-(\d+)\.mp4$")
    for path in sorted(root.glob("*-seed-*.mp4")):
        match = pattern.match(path.name)
        if not match:
            continue
        distance, seed = match.group(1), int(match.group(2))
        opening, face_width, eye_span = landmarks_for_video(path)
        series[(distance, seed)] = opening
        values = np.array([value for value in opening if value is not None])
        widths = np.array([value for value in face_width if value is not None])
        eyes = np.array([value for value in eye_span if value is not None])
        changes = np.abs(np.diff(values)) if len(values) > 1 else np.array([])
        rows.append({
            "distance": distance, "seed": seed, "frames": len(opening), "detected": len(values),
            "detection_rate": len(values) / max(len(opening), 1),
            "mean_face_width_px": float(widths.mean()) if len(widths) else None,
            "mean_eye_span_px": float(eyes.mean()) if len(eyes) else None,
            "mean_opening": float(values.mean()) if len(values) else None,
            "p95_opening": float(np.percentile(values, 95)) if len(values) else None,
            "opening_sd": float(values.std()) if len(values) else None,
            "mean_abs_frame_change": float(changes.mean()) if len(changes) else None,
            "opening_series": opening,
        })

    comparisons = []
    for seed in sorted({seed for _, seed in series}):
        baseline = series.get(("close", seed))
        if baseline is None:
            continue
        for distance in ("medium", "far"):
            candidate = series.get((distance, seed))
            if candidate is None:
                continue
            lag_scores = {lag: correlation(baseline, candidate, lag) for lag in range(-5, 6)}
            valid = {lag: score for lag, score in lag_scores.items() if score is not None}
            best_lag = max(valid, key=valid.get) if valid else None
            comparisons.append({
                "seed": seed, "distance": distance,
                "zero_lag_correlation_vs_close": correlation(baseline, candidate, 0),
                "best_lag_frames": best_lag,
                "best_lag_ms": None if best_lag is None else best_lag * 40,
                "best_lag_correlation": None if best_lag is None else valid[best_lag],
                "lag_scan_frames": lag_scores,
            })

    output = {
        "metric_definition": "MediaPipe inner-lip distance (landmarks 13/14) divided by eye-corner distance (33/263).",
        "comparison_definition": "Pearson correlation of opening trajectories versus the close reference for the same seed; lag scan ±5 frames at 25 FPS.",
        "limitations": "Automated 2D landmark proxy on H264 clips. It measures output motion preservation, not phoneme accuracy or perceived audiovisual sync.",
        "rows": rows, "comparisons": comparisons,
    }
    (root / "mouth-motion.json").write_text(json.dumps(output, indent=2) + "\n")
    for row in rows:
        print(json.dumps({k: v for k, v in row.items() if k != "opening_series"}), flush=True)
    for row in comparisons:
        print(json.dumps({k: v for k, v in row.items() if k != "lag_scan_frames"}), flush=True)


if __name__ == "__main__":
    main()
