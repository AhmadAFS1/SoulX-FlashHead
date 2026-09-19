"""CPU landmark and mouth-crop diagnostics for shoulder-visible source detail."""
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
WIDTHS = [1280, 640, 320, 256, 128, 64]
TIMES = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5]


def main():
    cv2.setNumThreads(1)
    sheet = Image.new("RGB", (7 * 192, len(WIDTHS) * 150), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    rows, series = [], {}
    for row, width in enumerate(WIDTHS):
        path = ROOT / f"int8-detail-{width}" / "video.mp4"
        capture = cv2.VideoCapture(str(path)); fps = capture.get(cv2.CAP_PROP_FPS)
        openings, mouth_widths, edge_energy = [], [], []
        samples = set(); index = 0
        with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                refine_landmarks=True, min_detection_confidence=.5,
                min_tracking_confidence=.5) as mesh:
            while True:
                ok, bgr = capture.read()
                if not ok: break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                prediction = mesh.process(rgb)
                if not prediction.multi_face_landmarks:
                    index += 1; continue
                points = np.array([(mark.x * rgb.shape[1], mark.y * rgb.shape[0])
                                   for mark in prediction.multi_face_landmarks[0].landmark])
                eye_span = max(np.linalg.norm(points[33] - points[263]), 1e-6)
                openings.append(float(np.linalg.norm(points[13] - points[14]) / eye_span))
                mouth_widths.append(float(np.linalg.norm(points[61] - points[291])))
                center = points[[13, 14]].mean(axis=0)
                x, y = int(round(center[0])) - 48, int(round(center[1])) - 26
                crop = rgb[max(0, y):y + 52, max(0, x):x + 96]
                edge_energy.append(float(cv2.Laplacian(
                    cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()))
                timestamp = index / fps
                for column, target in enumerate(TIMES):
                    if column not in samples and abs(timestamp - target) <= .5 / fps + 1e-6:
                        samples.add(column)
                        sheet.paste(Image.fromarray(crop).resize((192, 104), Image.Resampling.NEAREST),
                                    (column * 192, row * 150 + 42))
                        draw.text((column * 192 + 4, row * 150 + 3),
                                  f"source width {width}px", fill="black")
                        draw.text((column * 192 + 4, row * 150 + 20),
                                  f"audio {timestamp:.2f}s", fill="black")
                index += 1
        capture.release(); series[width] = openings
        rows.append({
            "effective_source_width_px": width, "decoded_frames": index,
            "detected_frames": len(openings),
            "median_mouth_width_px": float(np.median(mouth_widths)),
            "median_normalized_opening": float(np.median(openings)),
            "median_whole_mouth_laplacian_variance": float(np.median(edge_energy)),
        })
    for item in rows:
        width = item["effective_source_width_px"]
        item["opening_trajectory_correlation_vs_320"] = float(
            np.corrcoef(series[320], series[width])[0, 1])
    sheet.save(ROOT / "mouth-comparison.png")
    (ROOT / "diagnostics.json").write_text(json.dumps({
        "execution": "CPU MediaPipe/XNNPACK analysis of retained GPU outputs; no new SoulX inference.",
        "metric_limitations": "Laplacian variance covers the whole mouth crop and includes lips, beard, pose and compression; it is not a validated tooth-quality score.",
        "rows": rows}, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
