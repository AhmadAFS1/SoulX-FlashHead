"""CPU MediaPipe mouth-opening proxies on saved individual clips.

Not an anatomical ground truth or a lip-sync score. Uses decoded H264 frames.
"""
import argparse
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    rows = []
    cv2.setNumThreads(1)
    for path in sorted(root.glob('*-strength-*.mp4')):
        cap = cv2.VideoCapture(str(path))
        series = []
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=.5, min_tracking_confidence=.5) as mesh:
            while True:
                ok, rgb = cap.read()
                if not ok:
                    break
                prediction = mesh.process(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
                if not prediction.multi_face_landmarks:
                    series.append(None)
                    continue
                marks = prediction.multi_face_landmarks[0].landmark
                def point(i):
                    return np.array([marks[i].x*rgb.shape[1], marks[i].y*rgb.shape[0]])
                eye_span = np.linalg.norm(point(33)-point(263))
                series.append(float(np.linalg.norm(point(13)-point(14))/max(eye_span,1e-6)))
        cap.release()
        values = np.array([v for v in series if v is not None])
        changes = [abs(b-a) for a,b in zip(series,series[1:]) if a is not None and b is not None]
        row = dict(file=path.name, frames=len(series), detected=len(values),
            mean_opening=float(values.mean()) if len(values) else None,
            p95_opening=float(np.percentile(values,95)) if len(values) else None,
            mean_absolute_opening_change=float(np.mean(changes)) if changes else None,
            opening_series=series)
        rows.append(row)
        print(json.dumps({k:v for k,v in row.items() if k!='opening_series'}),flush=True)
    (root/'mouth-metrics.json').write_text(json.dumps(dict(rows=rows,
        definition='Distance between inner lip landmarks 13/14 divided by eye-corner distance 33/263 in pixels',
        limitations='Automated 2D landmark proxy on decoded clips; affected by pose/tracking/blur; not lip-sync accuracy.'),indent=2))


if __name__ == '__main__':
    main()
