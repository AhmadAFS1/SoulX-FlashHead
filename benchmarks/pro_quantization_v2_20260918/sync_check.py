"""Run a manifest-driven relative SyncNet diagnostic on v2 paired reviews.

This is intentionally not an official LSE-C/LSE-D implementation. It reports
only same-fixture, per-window cosine diagnostics and retains crop coverage.
"""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import cv2
import librosa
import numpy as np
import torch
import yaml

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_GPU_LOCK,
    ROOT,
    atomic_write_json,
    environment_manifest,
    relative_path,
    resolve_path,
    sha256,
)
from soulx_rtc.gpu_lease import acquire_gpu_lease


def _box(row: dict[str, Any] | None) -> tuple[int, int, int, int] | None:
    values = None if row is None else row.get("face_box")
    if not isinstance(values, list) or len(values) != 4:
        return None
    x1, y1, x2, y2 = (int(value) for value in values)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _valid_starts(boxes: list[tuple[int, int, int, int] | None], mel_columns: int, fps: int, offsets: list[int], stride: int) -> list[int]:
    starts = []
    for start in range(0, max(0, len(boxes) - 15), stride):
        if not all(boxes[index] is not None for index in range(start, start + 16)):
            continue
        valid_audio = all(
            start + offset >= 0 and int(80 * (start + offset) / fps) + 52 <= mel_columns
            for offset in offsets
        )
        if valid_audio:
            starts.append(start)
    return starts


def _crop_frames(video: Path, boxes: list[tuple[int, int, int, int] | None]) -> tuple[list[np.ndarray | None], int]:
    capture = cv2.VideoCapture(str(video))
    frames: list[np.ndarray | None] = []
    available = 0
    for box in boxes:
        ok, bgr = capture.read()
        if not ok:
            frames.append(None)
            continue
        available += 1
        if box is None:
            frames.append(None)
            continue
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(bgr.shape[1], x2), min(bgr.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            frames.append(None)
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(rgb[y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4))
    capture.release()
    return frames, available


def _load_audio(run: dict[str, Any]) -> tuple[np.ndarray, int, str]:
    fixture = run["fixture"]
    source = Path(fixture["audio"]["path"])
    samples, rate = librosa.load(source, sr=fixture["sample_rate"], mono=True)
    expected = int(run.get("effective_audio", {}).get("samples", len(samples)))
    if expected > len(samples):
        samples = np.tile(samples, int(np.ceil(expected / len(samples))))
    samples = samples[:expected]
    expected_hash = run.get("effective_audio", {}).get("sha256")
    actual_hash = __import__("hashlib").sha256(np.asarray(samples, dtype=np.float32).tobytes()).hexdigest()
    if expected_hash and expected_hash != actual_hash:
        raise ValueError("Run effective audio does not reconstruct from its fixture manifest")
    return samples, rate, actual_hash


def _embeddings(model, frames: list[np.ndarray | None], starts: list[int]) -> torch.Tensor:
    result = []
    for start in starts:
        clip = np.stack(frames[start:start + 16])
        tensor = torch.from_numpy(clip).permute(0, 3, 1, 2).float() / 127.5 - 1
        tensor = tensor.reshape(1, 48, 256, 256)[:, :, 128:, :].cuda()
        result.append(model.get_image_embed(tensor).cpu()[0])
    return torch.stack(result)


def _audio_embeddings(model, mel: np.ndarray, starts: list[int], offsets: list[int], fps: int) -> dict[int, torch.Tensor]:
    keys = sorted({start + offset for start in starts for offset in offsets})
    chunks = np.stack([mel[:, int(80 * key / fps):int(80 * key / fps) + 52] for key in keys])[:, None]
    result = []
    for offset in range(0, len(chunks), 8):
        result.append(model.get_audio_embed(torch.from_numpy(chunks[offset:offset + 8]).float().cuda()).cpu())
    return dict(zip(keys, torch.cat(result)))


def _scores(vision: torch.Tensor, audio: dict[int, torch.Tensor], starts: list[int], offsets: list[int]) -> dict[str, Any]:
    rows = {}
    for offset in offsets:
        values = (vision * torch.stack([audio[start + offset] for start in starts])).sum(1).numpy()
        rows[str(offset)] = {"mean": float(values.mean()), "per_window": values.tolist()}
    best = max(offsets, key=lambda offset: rows[str(offset)]["mean"])
    zero = rows.get("0")
    shuffled = (vision * torch.stack([audio[start] for start in starts]).roll(len(starts) // 2, 0)).sum(1)
    return {
        "best_lag_frames": best,
        "best_mean": rows[str(best)]["mean"],
        "zero_lag_mean": None if zero is None else zero["mean"],
        "shuffled_mean": float(shuffled.mean()),
        "scores": rows,
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--crop-mode", choices=("own", "shared"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--offsets", type=int, nargs="+", default=list(range(-5, 6)))
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    args = parser.parse_args()
    if args.output.exists() or not args.offsets or args.stride < 1:
        parser.error("output must be new; offsets must be non-empty; stride must be positive")
    result: dict[str, Any] = {
        "status": "starting",
        "execution": "fresh GPU relative SyncNet diagnostic; no video generation",
        "environment": environment_manifest(),
        "crop_mode": args.crop_mode,
        "offsets": args.offsets,
        "reviews": {},
        "limitations": "Relative same-audio diagnostic only; not official LSE-C/LSE-D. Face crop coverage and overlapping-window dependence are retained.",
    }

    def save() -> None:
        atomic_write_json(args.output, result)

    save()
    lease = None
    stage = "load"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for SyncNet diagnostics")
        lease = acquire_gpu_lease(args.gpu_lock)
        muse = Path("/workspace/MuseTalk")
        if not muse.is_dir():
            raise FileNotFoundError("MuseTalk SyncNet source is unavailable at /workspace/MuseTalk")
        sys.path.insert(0, str(muse))
        from musetalk.data.audio import melspectrogram
        from musetalk.models.syncnet import SyncNet

        config = yaml.safe_load((muse / "configs/training/syncnet.yaml").read_text(encoding="utf-8"))["model"]
        model = SyncNet(config)
        checkpoint = muse / "models/syncnet/latentsync_syncnet.pt"
        weights = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(weights["state_dict"], strict=True)
        model = model.cuda().eval()
        result["checkpoint"] = {"path": str(checkpoint), "sha256": sha256(checkpoint)}
        for review_path in args.reviews:
            stage = f"review:{review_path.name}"
            review = json.loads((review_path / "review.json").read_text(encoding="utf-8"))
            baseline_dir = resolve_path(review["baseline"]["run"])
            candidate_dir = resolve_path(review["candidate"]["run"])
            baseline_run = json.loads((baseline_dir / "results.json").read_text(encoding="utf-8"))
            candidate_run = json.loads((candidate_dir / "results.json").read_text(encoding="utf-8"))
            audio, rate, audio_hash = _load_audio(baseline_run)
            _, candidate_rate, candidate_hash = _load_audio(candidate_run)
            if rate != candidate_rate or audio_hash != candidate_hash:
                raise ValueError("Paired SyncNet review requires identical effective audio")
            fps = baseline_run["profile"]["fps"]
            if fps != candidate_run["profile"]["fps"]:
                raise ValueError("Paired SyncNet review requires matching playback FPS")
            mel = melspectrogram(audio)
            shared = [_box(row) for row in review["baseline"]["analysis"]["frames"]]
            own = {
                "baseline": shared,
                "candidate": [_box(row) for row in review["candidate"]["analysis"]["frames"]],
            }
            pair = {"audio_sha256": audio_hash, "sample_rate": rate, "fps": fps, "variants": {}}
            for role, directory in (("baseline", baseline_dir), ("candidate", candidate_dir)):
                boxes = own[role] if args.crop_mode == "own" else shared
                starts = _valid_starts(boxes, mel.shape[1], fps, args.offsets, args.stride)
                coverage = sum(box is not None for box in boxes)
                variant = {"face_box_coverage": coverage / max(1, len(boxes)), "valid_window_starts": starts}
                if not starts:
                    variant.update(status="invalid", reason="no valid 16-frame face/audio windows")
                else:
                    frames, decoded = _crop_frames(directory / "video.mp4", boxes)
                    if decoded < len(boxes):
                        variant.update(status="invalid", reason="video shorter than review analysis")
                    else:
                        vision = _embeddings(model, frames, starts)
                        audio_embeddings = _audio_embeddings(model, mel, starts, args.offsets, fps)
                        variant.update(status="complete", decoded_frames=decoded, **_scores(vision, audio_embeddings, starts, args.offsets))
                pair["variants"][role] = variant
            result["reviews"][review_path.name] = pair
            save()
        result["status"] = "complete"
        save()
    except Exception as error:
        result.update(status="failed", failure={"stage": stage, "type": type(error).__name__, "message": str(error)})
        save()
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    main()