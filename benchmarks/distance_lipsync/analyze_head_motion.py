"""CPU 2D head-motion proxies from existing SoulX clips; not calibrated 3D pose."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


def span(values):
    return float(np.percentile(values, 95) - np.percentile(values, 5))


def analyze(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    records = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=.5, min_tracking_confidence=.5) as mesh:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            prediction = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if not prediction.multi_face_landmarks:
                records.append(None)
                continue
            lm = prediction.multi_face_landmarks[0].landmark
            def point(i):
                return np.array([lm[i].x * frame.shape[1], lm[i].y * frame.shape[0]])
            left, right, nose = point(33), point(263), point(1)
            mid = (left + right) / 2
            eye = right - left
            size = np.linalg.norm(eye)
            axis = eye / size
            down = np.array([-axis[1], axis[0]])
            offset = (nose - mid) / size
            records.append([*mid, size, np.degrees(np.arctan2(eye[1], eye[0])),
                            np.dot(offset, axis), np.dot(offset, down)])
    cap.release()
    assert records and all(row is not None for row in records), 'Missing detections; do not compress time'
    values = np.array(records)
    size = float(np.median(values[:, 2]))
    summaries = []
    for start, end in [(0, len(values)), *[(i, min(i+round(fps),len(values))) for i in range(0,len(values),round(fps))]]:
        v = values[start:end]
        summaries.append(dict(start_s=start/fps, end_s=end/fps,
            eye_mid_x_span_eye_units=span(v[:, 0])/size,
            eye_mid_y_span_eye_units=span(v[:, 1])/size,
            roll_span_degrees=span(v[:, 3]), yaw_proxy_span=span(v[:, 4]),
            pitch_proxy_span=span(v[:, 5]), eye_scale_span_fraction=span(v[:, 2])/size))
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frames=len(records), detected=len(records), fps=fps, median_eye_span_px=size,
        full_clip=summaries[0], one_second_windows=summaries[1:],
        series_columns=['eye_mid_x_px','eye_mid_y_px','eye_span_px','roll_degrees','yaw_proxy','pitch_proxy'],
        series=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    cv2.setNumThreads(1)
    rows=[]
    for strength in (1.0, 0.5):
        for label in ('close-100','closer-125','closest-150'):
            if strength == .5:
                path=root/'evidence-indian-male-closer-strength-0p5-20260916'/f'{label}-strength-0p5-seed-50.mp4'
            elif label == 'close-100':
                path=root/'evidence-indian-male-20260916/close-seed-50.mp4'
            else:
                path=root/'evidence-indian-male-closer-20260916'/f'{label}-seed-50.mp4'
            row=dict(strength=strength,label=label,**analyze(path))
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('series','one_second_windows')}),flush=True)
    result=dict(date_utc=datetime.now(timezone.utc).isoformat(),
        execution='CPU MediaPipe landmark inference and NumPy; OpenGL context may initialize. No new SoulX GPU generation.',
        input_gpu='NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible; driver 595.84; Torch 2.7.1+cu128 / CUDA 12.8. Per-input generation results.json is controlling provenance.',
        definitions='All spans are p95 minus p5. Translation normalized by each clip median eye span. Roll from eye-corner line. Yaw/pitch are nose-to-eye-midpoint projections in a roll-aligned eye-span coordinate system, NOT calibrated yaw/pitch angles.',
        limitations='One portrait/audio/seed; 2D face deformation/landmark error can affect proxies. Normalization removes display zoom but not all perspective effects. No word alignment or causal head-motion control is established. Strength 1 was unhooked; compilation differs from the 0.5 intervention.',
        rows=rows)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    main()
