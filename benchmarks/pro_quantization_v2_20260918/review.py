"""Create a labeled, deterministic paired quality review for two v2 runs."""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
import json
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from benchmarks.pro_quantization_20260918.review import inspect, probe
from benchmarks.pro_quantization_v2_20260918.common import atomic_write_json, ensure_new_directory, relative_path


def _load_run(directory: Path) -> dict[str, Any]:
    result_path = directory / "results.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Run has no results manifest: {directory}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "complete" or result.get("execution") != "fresh local GPU inference":
        raise ValueError(f"Review requires a complete fresh generation run: {directory}")
    if not (directory / "video.mp4").is_file():
        raise FileNotFoundError(f"Run has no video: {directory}")
    return result


def _profile_key(run: dict[str, Any]) -> tuple[Any, ...]:
    profile = run["profile"]
    return tuple(profile.get(name) for name in ("width", "height", "fps", "frames", "steps", "shift", "motion_latents"))


def _select_frames(frames: list[dict[str, Any]], fps: int) -> dict[str, Any]:
    detected = [item for item in frames if item.get("detected")]
    if not detected:
        raise ValueError("Baseline face analysis detected no faces; cannot select mouth strata")
    by_opening = sorted(detected, key=lambda item: item["opening_norm"])
    closed = by_opening[0]["frame"]
    partial = by_opening[len(by_opening) // 2]["frame"]
    open_frame = by_opening[-1]["frame"]
    frame_count = len(frames)
    boundaries = sorted({0, min(frame_count - 1, 28), min(frame_count - 1, 56), frame_count - 1})
    if frame_count > 250:
        boundaries = sorted(set(boundaries + [
            (frame_count // 2 // 28) * 28,
            ((frame_count - 1) // 28) * 28,
        ]))
    even = sorted({min(frame_count - 1, round(second * fps)) for second in (0.5, 2.5, 4.5, 6.5, 8.5)})
    even += [round(second * fps) for second in (19.5, 29.5, 39.5, 49.5, 59.5) if round(second * fps) < frame_count]
    indices = sorted(set((closed, partial, open_frame, *boundaries, *even)))
    return {
        "rule": "baseline FaceMesh opening_norm min/median/max plus first, recurrent chunk boundaries, final frame, and fixed timeline samples",
        "strata": {"closed": closed, "partial": partial, "open": open_frame},
        "chunk_boundaries": boundaries,
        "timeline": even,
        "indices": indices,
    }


def _video_frame(directory: Path, index: int) -> Image.Image:
    capture = cv2.VideoCapture(str(directory / "video.mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Cannot decode frame {index} from {directory}")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def _raw_frame(directory: Path, index: int) -> tuple[Image.Image, str]:
    raw = directory / "raw.npy"
    if raw.is_file():
        pixels = np.load(raw, mmap_mode="r")
        if index >= len(pixels):
            raise IndexError(f"Raw frame index {index} is outside {raw}")
        return Image.fromarray(np.asarray(pixels[index])), "raw_rgb"
    manifest = json.loads((directory / 'results.json').read_text())
    for sample in manifest.get('samples', []):
        if sample.get('frame') == index and sample.get('source') == 'raw_rgb':
            with Image.open(directory / sample['path']) as image:
                return image.copy(), 'raw_rgb_sample'
    return _video_frame(directory, index), "decoded_mp4"


def _crop(image: Image.Image, row: dict[str, Any] | None) -> Image.Image | None:
    if row is None or "mouth_center" not in row:
        return None
    center = np.asarray(row["mouth_center"], dtype=int)
    cx, cy = center.tolist()
    return image.crop((max(0, cx - 48), max(0, cy - 26), cx + 48, cy + 26))


def _escape_drawtext(label: str) -> str:
    return label.replace("\\", "\\\\").replace("'", "\\'").replace(":", "\\:")


def _paired_metrics(baseline: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = [
        (left, right) for left, right in zip(baseline, candidate)
        if left.get("detected") and right.get("detected")
    ]
    opened = [(left, right) for left, right in pairs if left["opening_px"] >= 4 and right["opening_px"] >= 4]
    if len(pairs) < 2:
        return {"detected_pairs": len(pairs), "both_open_pairs": len(opened), "insufficient_face_pairs": True}
    opening = np.corrcoef(
        [left["opening_norm"] for left, _ in pairs], [right["opening_norm"] for _, right in pairs]
    )[0, 1]
    return {
        "detected_pairs": len(pairs),
        "both_open_pairs": len(opened),
        "opening_correlation": float(opening) if np.isfinite(opening) else None,
        "median_mouth_center_distance_px": float(np.median([
            np.linalg.norm(np.asarray(left["mouth_center"]) - np.asarray(right["mouth_center"]))
            for left, right in pairs
        ])),
        "median_edge_ratio_on_both_open": float(np.median([
            right["oral_edge_energy"] / max(left["oral_edge_energy"], 1e-6) for left, right in opened
        ])) if opened else None,
    }


def _make_sheet(output: Path, records: list[dict[str, Any]], labels: tuple[str, str]) -> None:
    columns = len(records)
    sheet = Image.new("RGB", (columns * 192, 300), "white")
    draw = ImageDraw.Draw(sheet)
    for column, record in enumerate(records):
        for row, (role, label) in enumerate(zip(("baseline", "candidate"), labels)):
            crop = record[role]["crop"]
            if crop is not None:
                sheet.paste(crop.resize((192, 104), Image.Resampling.NEAREST), (column * 192, row * 150 + 42))
            draw.text((column * 192 + 4, row * 150 + 3), label, fill="black")
            draw.text((column * 192 + 4, row * 150 + 20), f"frame {record['frame']} {record[role]['source']}", fill="black")
    sheet.save(output / "mouth-comparison.png")


def _make_mouth_video(
    output: Path,
    baseline_directory: Path,
    candidate_directory: Path,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    fps: int,
) -> dict[str, Any]:
    """Create a fixed-size synchronized mouth view from each video's own boxes."""
    dimensions = (384, 104)
    silent = output / "mouth-comparison.silent.mp4"
    target = output / "mouth-comparison.mp4"
    process = subprocess.Popen([
        "ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "384x104",
        "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264", "-crf", "0", "-preset", "fast",
        "-pix_fmt", "yuv420p", str(silent),
    ], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    baseline = cv2.VideoCapture(str(baseline_directory / "video.mp4"))
    candidate = cv2.VideoCapture(str(candidate_directory / "video.mp4"))
    missing = {"baseline": 0, "candidate": 0}
    try:
        for baseline_row, candidate_row in zip(baseline_rows, candidate_rows):
            left_ok, left_bgr = baseline.read()
            right_ok, right_bgr = candidate.read()
            if not left_ok or not right_ok:
                raise RuntimeError("Paired video ended before review analysis")
            canvas = Image.new("RGB", dimensions, "black")
            for column, (role, bgr, row) in enumerate((
                ("baseline", left_bgr, baseline_row), ("candidate", right_bgr, candidate_row)
            )):
                image = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                mouth = _crop(image, row)
                if mouth is None:
                    missing[role] += 1
                else:
                    canvas.paste(mouth.resize((192, 104), Image.Resampling.NEAREST), (column * 192, 0))
            if process.stdin is None:
                raise RuntimeError("FFmpeg mouth-video stdin is unavailable")
            process.stdin.write(np.asarray(canvas).tobytes())
    finally:
        baseline.release()
        candidate.release()
        if process.stdin is not None:
            process.stdin.close()
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr is not None else ""
    if process.wait() != 0:
        raise RuntimeError(f"FFmpeg mouth-video encoding failed: {stderr}")
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(silent), "-i", str(baseline_directory / "video.mp4"),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", str(target),
    ], check=True)
    silent.unlink(missing_ok=True)
    return {"path": target.name, "dimensions": list(dimensions), "missing_face_boxes": missing, "media": probe(target)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = ensure_new_directory(args.output)
    cv2.setNumThreads(1)
    baseline_run, candidate_run = _load_run(args.baseline), _load_run(args.candidate)
    if _profile_key(baseline_run) != _profile_key(candidate_run):
        raise ValueError("Baseline and candidate profiles differ; a paired review would be misleading")
    baseline_analysis = inspect(args.baseline)
    candidate_analysis = inspect(args.candidate)
    if baseline_analysis["decoded_frames"] != candidate_analysis["decoded_frames"]:
        raise ValueError("Baseline and candidate videos contain different frame counts")
    selection = _select_frames(baseline_analysis["frames"], baseline_run["profile"]["fps"])
    atomic_write_json(output / "selection.json", selection)
    samples_dir = output / "samples"
    samples_dir.mkdir()
    records = []
    for frame in selection["indices"]:
        record = {"frame": frame}
        for role, directory, analysis in (
            ("baseline", args.baseline, baseline_analysis), ("candidate", args.candidate, candidate_analysis)
        ):
            image, source = _raw_frame(directory, frame)
            row = analysis["frames"][frame] if frame < len(analysis["frames"]) else None
            crop = _crop(image, row)
            full_name = f"{role}-frame-{frame:04d}.png"
            image.save(samples_dir / full_name)
            crop_name = None
            if crop is not None:
                crop_name = f"{role}-mouth-{frame:04d}.png"
                crop.save(samples_dir / crop_name)
            record[role] = {"source": source, "full": str(Path("samples") / full_name), "crop": crop, "crop_path": crop_name}
        records.append(record)
    _make_sheet(output, records, (args.baseline_label, args.candidate_label))
    for record in records:
        for role in ("baseline", "candidate"):
            record[role].pop("crop")
    labels = (_escape_drawtext(args.baseline_label), _escape_drawtext(args.candidate_label))
    comparison = output / "comparison.mp4"
    filter_graph = (
        f"[0:v]drawtext=text='{labels[0]}':x=10:y=12:fontsize=20:fontcolor=white:borderw=2:bordercolor=black[v0];"
        f"[1:v]drawtext=text='{labels[1]}':x=10:y=12:fontsize=20:fontcolor=white:borderw=2:bordercolor=black[v1];"
        "[v0][v1]hstack=inputs=2[v]"
    )
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(args.baseline / "video.mp4"), "-i", str(args.candidate / "video.mp4"),
        "-filter_complex", filter_graph, "-map", "[v]", "-map", "0:a:0", "-c:v", "libx264", "-crf", "16",
        "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "copy", str(comparison),
    ], check=True)
    mouth_comparison = _make_mouth_video(
        output, args.baseline, args.candidate, baseline_analysis["frames"], candidate_analysis["frames"],
        baseline_run["profile"]["fps"],
    )
    result = {
        "status": "complete",
        "execution": "CPU FFmpeg/MediaPipe paired review; no new generation",
        "baseline": {"label": args.baseline_label, "run": relative_path(args.baseline), "analysis": baseline_analysis, "media": probe(args.baseline / "video.mp4")},
        "candidate": {"label": args.candidate_label, "run": relative_path(args.candidate), "analysis": candidate_analysis, "media": probe(args.candidate / "video.mp4")},
        "selection": selection,
        "sample_records": records,
        "paired": _paired_metrics(baseline_analysis["frames"], candidate_analysis["frames"]),
        "comparison": {"path": comparison.name, "media": probe(comparison)},
        "mouth_comparison": mouth_comparison,
        "limitations": "Face landmarks and edge energy are diagnostics, not dental correctness or audio-sync certification.",
    }
    atomic_write_json(output / "review.json", result)
    print(json.dumps({"output": str(output), "paired": result["paired"]}, indent=2))


if __name__ == "__main__":
    main()
