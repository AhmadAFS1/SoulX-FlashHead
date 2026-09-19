"""Verify v2 media, hashes, manifests, and report references without generation."""
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

from benchmarks.pro_quantization_v2_20260918.common import atomic_write_json, relative_path, sha256


def _ffprobe(path: Path) -> dict[str, Any]:
    completed = subprocess.run([
        "ffprobe", "-v", "error", "-count_frames", "-show_entries",
        "stream=codec_type,width,height,avg_frame_rate,duration,nb_read_frames", "-of", "json", str(path),
    ], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if completed.returncode:
        raise RuntimeError(f"ffprobe failed for {path}: {completed.stderr}")
    return json.loads(completed.stdout)


def _full_decode(path: Path) -> str | None:
    completed = subprocess.run(["ffmpeg", "-v", "error", "-xerror", "-i", str(path), "-f", "null", "-"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return None if completed.returncode == 0 else completed.stderr.strip()


def _verify_run(path: Path, run: dict[str, Any]) -> list[str]:
    errors = []
    required = ("requested-policy.json", "resolved-policy.json", "fixtures-resolved.json", "source-manifest.json", "video.mp4")
    for name in required:
        if not (path / name).is_file():
            errors.append(f"missing:{name}")
    if errors:
        return errors
    video = path / "video.mp4"
    try:
        media = _ffprobe(video)
        video_streams = [stream for stream in media.get("streams", []) if stream.get("codec_type") == "video"]
        audio_streams = [stream for stream in media.get("streams", []) if stream.get("codec_type") == "audio"]
        if len(video_streams) != 1 or len(audio_streams) < 1:
            errors.append("video-or-audio-stream-missing")
        else:
            stream = video_streams[0]
            profile = run["profile"]
            if (stream.get("width"), stream.get("height")) != (profile["width"], profile["height"]):
                errors.append("video-dimensions-mismatch")
            frame_count = int(stream.get("nb_read_frames") or -1)
            if frame_count != profile["frames"]:
                errors.append(f"frame-count-mismatch:{frame_count}!={profile['frames']}")
            if stream.get("avg_frame_rate") != f"{profile['fps']}/1":
                errors.append(f"fps-mismatch:{stream.get('avg_frame_rate')}")
        decode_error = _full_decode(video)
        if decode_error:
            errors.append(f"ffmpeg-decode-failed:{decode_error}")
    except Exception as error:
        errors.append(f"media-check:{type(error).__name__}:{error}")
    fixture = run.get("fixture", {})
    for field in ("reference", "audio"):
        identity = fixture.get(field, {})
        source = Path(identity.get("path", ""))
        if not source.is_file() or sha256(source) != identity.get("sha256"):
            errors.append(f"fixture-hash-mismatch:{field}")
    for name, expected in run.get("source_sha256", {}).items():
        snapshot = path / "sources" / name
        if not snapshot.is_file() or sha256(snapshot) != expected:
            errors.append(f"source-snapshot-mismatch:{name}")
    for sample in run.get("samples", []):
        if sample.get("source") != "raw_rgb" or not (path / sample.get("path", "")).is_file():
            errors.append(f"sample-provenance-invalid:{sample}")
    if run.get("decoder", {}).get("scheme") not in (None, "bf16"):
        for module in run["decoder"].get("modules", []):
            engine = module.get("engine", {})
            records = [engine] if "quantization_dispatch_verified" in engine else list(engine.values())
            if not records or any(record.get("quantization_dispatch_verified") is not True for record in records):
                errors.append(f"decoder-engine-dispatch-unverified:{module.get('path')}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "verification.json"
    reports = []
    for manifest in sorted(args.root.rglob("results.json")):
        run = json.loads(manifest.read_text(encoding="utf-8"))
        if run.get("status") != "complete" or run.get("execution") != "fresh local GPU inference":
            reports.append({"run": relative_path(manifest.parent), "status": "excluded", "reason": run.get("status")})
            continue
        errors = _verify_run(manifest.parent, run)
        reports.append({"run": relative_path(manifest.parent), "status": "valid" if not errors else "invalid", "errors": errors})
    result = {
        "status": "complete",
        "execution": "CPU-only artifact verification; no generation",
        "runs": reports,
        "valid": sum(item["status"] == "valid" for item in reports),
        "invalid": sum(item["status"] == "invalid" for item in reports),
        "excluded": sum(item["status"] == "excluded" for item in reports),
    }
    atomic_write_json(output, result)
    print(json.dumps(result, indent=2))
    if result["invalid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()