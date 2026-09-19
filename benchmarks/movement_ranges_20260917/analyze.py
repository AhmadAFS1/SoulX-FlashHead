"""CPU landmark diagnostics of RTX 4070 SUPER outputs; not a lip-sync score."""
import argparse
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parent


def span(x):
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def analyze(row):
    cap = cv2.VideoCapture(str(ROOT / row['name'] / 'video.mp4'))
    series = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as mesh:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            prediction = mesh.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if not prediction.multi_face_landmarks:
                series.append(None); continue
            points = np.array([(p.x * bgr.shape[1], p.y * bgr.shape[0])
                               for p in prediction.multi_face_landmarks[0].landmark])
            eye = points[263] - points[33]
            size = np.linalg.norm(eye)
            if size < 1:
                series.append(None); continue
            mid = (points[33] + points[263]) / 2
            opening = np.linalg.norm(points[13] - points[14]) / size
            axis = eye / size
            nose_offset = (points[1] - mid) / size
            series.append([float(opening), float(np.degrees(np.arctan2(eye[1], eye[0]))),
                           float(mid[0]), float(mid[1]), float(size),
                           float(nose_offset @ axis), float(nose_offset @ np.array([-axis[1], axis[0]]))])
    cap.release()
    assert len(series) == 250, (row['name'], len(series))
    valid = np.array([r for r in series if r is not None])
    metrics = {}
    if len(valid):
        scale = float(np.median(valid[:, 4]))
        adjacent = [abs(series[i][0] - series[i-1][0]) for i in range(1, len(series))
                    if series[i] is not None and series[i-1] is not None]
        metrics = dict(mouth_opening_mean=float(valid[:, 0].mean()),
                       mouth_opening_p95=float(np.percentile(valid[:, 0], 95)),
                       roll_span_degrees=span(valid[:, 1]),
                       horizontal_span_eye_units=span(valid[:, 2]) / scale,
                       vertical_span_eye_units=span(valid[:, 3]) / scale,
                       yaw_proxy_span=span(valid[:, 5]), pitch_proxy_span=span(valid[:, 6]),
                       mouth_adjacent_change_mean=float(np.mean(adjacent)) if adjacent else None)
    return dict(name=row['name'], seed=row['seed'], shift=row['shift'], history=row['history'],
                detected_frames=len(valid), total_frames=len(series), metrics=metrics, series=series)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    output = ROOT / 'diagnostics.json'
    data = dict(execution='CPU MediaPipe/XNNPACK landmarks and NumPy on retained NVIDIA GeForce RTX 4070 SUPER GPU outputs. MediaPipe may initialize an OpenGL context.',
                input_gpu='RTX 4070 SUPER, 12282 MiB visible, driver 595.84, Torch 2.7.1+cu128/CUDA 12.8; controlling provenance in results.json',
                definitions='Mouth opening = inner-lip landmarks 13/14 distance divided by eye-corner 33/263 distance. Motion spans = p95 minus p5. Translation normalized by median eye span. Roll is image-plane eye-line angle. Yaw/pitch are 2D nose-offset proxies, not calibrated angles.',
                limitations='No lip-sync, dental quality or naturalness score. Missing detections retain their frame slots; adjacent changes exclude gaps. Blur and face deformation can distort landmarks. Low motion caused by rendering failure is not improvement.',
                columns=['mouth_opening_eye_units', 'roll_degrees', 'eye_mid_x_px', 'eye_mid_y_px', 'eye_span_px', 'yaw_proxy', 'pitch_proxy'], rows=[])
    if output.exists():
        data = json.loads(output.read_text())
        data.pop('generation_status', None)
    done = {row['name'] for row in data['rows']}
    while True:
        try:
            manifest = json.loads((ROOT / 'results.json').read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if not args.watch:
                raise
            time.sleep(2); continue
        for row in manifest['runs']:
            if row['name'] in done:
                continue
            result = analyze(row)
            data['rows'].append(result); done.add(row['name'])
            output.write_text(json.dumps(data, indent=2) + '\n')
            print(json.dumps({k:v for k,v in result.items() if k != 'series'}), flush=True)
        if not args.watch or manifest['status'] in ['complete', 'failed']:
            data['generation_status'] = manifest['status']
            break
        time.sleep(2)
    bases = {r['seed']:r for r in data['rows'] if r['shift'] == 5 and r['history'] == 2}
    for row in data['rows']:
        base = bases.get(row['seed'])
        if base is None or not row['metrics'] or not base['metrics']:
            continue
        row['relative_to_same_seed_baseline'] = {}
        for key in ['mouth_opening_mean', 'mouth_opening_p95', 'roll_span_degrees']:
            denominator = base['metrics'][key]
            row['relative_to_same_seed_baseline'][key] = row['metrics'][key] / denominator if denominator > 1e-8 else None
        pairs = [(a[0], b[0]) for a,b in zip(row['series'], base['series']) if a is not None and b is not None]
        if len(pairs) > 2:
            values = np.array(pairs)
            if np.all(values.std(axis=0) > 1e-8):
                row['opening_correlation_vs_baseline'] = float(np.corrcoef(values.T)[0,1])
    output.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
