"""CPU landmark and conservative sharpness diagnostics for the PRO/LITE A/B."""
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
VARIANTS = ["lite", "pro"]
TIMES = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5]


def crop_at_mouth(rgb, points):
    center = points[[13, 14]].mean(axis=0)
    x, y = int(round(center[0])) - 48, int(round(center[1])) - 26
    crop = np.zeros((52, 96, 3), dtype=np.uint8)
    src = rgb[max(0, y):min(rgb.shape[0], y + 52), max(0, x):min(rgb.shape[1], x + 96)]
    dx, dy = max(0, -x), max(0, -y)
    crop[dy:dy + src.shape[0], dx:dx + src.shape[1]] = src
    return crop


def analyze_video(variant):
    path = ROOT / variant / "video.mp4"
    capture = cv2.VideoCapture(str(path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    values, openings = [], []
    index = 0
    with mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=.5, min_tracking_confidence=.5) as mesh:
        while True:
            ok, bgr = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prediction = mesh.process(rgb)
            if prediction.multi_face_landmarks:
                points = np.array([(mark.x * rgb.shape[1], mark.y * rgb.shape[0])
                                   for mark in prediction.multi_face_landmarks[0].landmark])
                eye_span = max(np.linalg.norm(points[33] - points[263]), 1e-6)
                opening_px = float(np.linalg.norm(points[13] - points[14]))
                opening_norm = opening_px / eye_span
                crop = crop_at_mouth(rgb, points)
                gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

                # The central oral strip reduces beard/skin influence, but still
                # contains lips and varies with pose. It remains an exploratory
                # sharpness measurement rather than a dental-accuracy score.
                cx = crop.shape[1] // 2
                half_width = max(8, int(round(np.linalg.norm(points[61] - points[291]) * .32)))
                cy = crop.shape[0] // 2
                half_height = max(3, int(round(opening_px * .75)))
                oral = gray[max(0, cy-half_height):min(gray.shape[0], cy+half_height+1),
                            max(0, cx-half_width):min(gray.shape[1], cx+half_width+1)]
                values.append({
                    "frame": index, "time_s": index / fps,
                    "normalized_opening": opening_norm, "opening_px": opening_px,
                    "mouth_width_px": float(np.linalg.norm(points[61] - points[291])),
                    "whole_mouth_laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    "central_oral_laplacian_variance": float(cv2.Laplacian(oral, cv2.CV_64F).var()),
                    "central_oral_mean_luma": float(oral.mean()),
                })
                openings.append(opening_norm)
            index += 1
    capture.release()
    open_values = [v for v in values if v["opening_px"] >= 4.0]
    return {
        "variant": variant, "path": str(path.relative_to(ROOT.parent.parent)),
        "fps": fps, "decoded_frames": index, "detected_frames": len(values),
        "median_mouth_width_px": float(np.median([v["mouth_width_px"] for v in values])),
        "median_normalized_opening": float(np.median(openings)),
        "p95_normalized_opening": float(np.percentile(openings, 95)),
        "median_whole_mouth_laplacian_variance": float(np.median([v["whole_mouth_laplacian_variance"] for v in values])),
        "open_frame_count_ge_4px": len(open_values),
        "open_frame_median_central_oral_laplacian_variance": float(np.median([v["central_oral_laplacian_variance"] for v in open_values])),
        "open_frame_median_central_oral_mean_luma": float(np.median([v["central_oral_mean_luma"] for v in open_values])),
        "samples": values,
    }, openings


def make_raw_contact_sheet():
    sheet = Image.new("RGB", (len(TIMES) * 192, len(VARIANTS) * 150), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    for row, variant in enumerate(VARIANTS):
        for column, timestamp in enumerate(TIMES):
            rgb = np.array(Image.open(ROOT / variant / f"frame-{timestamp:.1f}s.png").convert("RGB"))
            with mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                    refine_landmarks=True, min_detection_confidence=.5) as mesh:
                prediction = mesh.process(rgb)
            if not prediction.multi_face_landmarks:
                continue
            points = np.array([(mark.x * rgb.shape[1], mark.y * rgb.shape[0])
                               for mark in prediction.multi_face_landmarks[0].landmark])
            crop = crop_at_mouth(rgb, points)
            sheet.paste(Image.fromarray(crop).resize((192, 104), Image.Resampling.NEAREST),
                        (column * 192, row * 150 + 42))
            draw.text((column * 192 + 4, row * 150 + 3), variant.upper(), fill="black")
            draw.text((column * 192 + 4, row * 150 + 20), f"raw RGB {timestamp:.1f}s", fill="black")
    sheet.save(ROOT / "mouth-comparison.png")


def main():
    cv2.setNumThreads(1)
    rows, trajectories = [], {}
    for variant in VARIANTS:
        row, trajectory = analyze_video(variant)
        rows.append(row); trajectories[variant] = trajectory
    # The released variants retain different frame counts for the same 10 s
    # request (their recurrent chunk contracts differ). Compare the shared
    # prefix so the correlation remains descriptive rather than failing.
    shared = min(len(trajectories["lite"]), len(trajectories["pro"]))
    rows[1]["opening_trajectory_correlation_vs_lite"] = float(
        np.corrcoef(trajectories["lite"][:shared], trajectories["pro"][:shared])[0, 1])
    rows[1]["opening_trajectory_correlation_frames"] = shared
    make_raw_contact_sheet()
    (ROOT / "diagnostics.json").write_text(json.dumps({
        "execution": "CPU MediaPipe/XNNPACK analysis of fresh retained RTX 4070 SUPER outputs; no additional SoulX inference.",
        "limitations": (
            "Laplacian variance is an edge-energy measure, not a validated tooth-blur or dental-accuracy score. "
            "Whole-mouth values include lips, beard, pose and H264. The central oral strip reduces but does not "
            "remove those confounds. Variant-specific mouth poses make same-time sharpness comparisons imperfect."),
        "rows": rows,
    }, indent=2) + "\n")
    print(json.dumps([{k: v for k, v in row.items() if k != "samples"} for row in rows], indent=2))


if __name__ == "__main__":
    main()
