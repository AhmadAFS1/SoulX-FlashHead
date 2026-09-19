"""Isolate source-detail resolution at fixed 512-square/25-FPS generation."""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from soulx_rtc.engine import Engine
from soulx_rtc.experiment import record
from soulx_rtc.server import decode_audio


SOURCE = Path("/workspace/benchmarks/same-avatar/shared.png")
CROP = (38, 64, 345, 371)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detail", type=int, choices=[64, 128, 256, 307], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    result = {
        "status": "starting",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "fresh local GPU inference",
        "gpu_snapshot": subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
            "--format=csv"], text=True),
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "profile": {
            "width": 512, "height": 512, "fps": 25, "steps": 4,
            "seed": 50, "precision": "BF16", "memory_mode": "staged",
            "compiled": False, "conditioning": "stock", "seconds": 10,
            "allocator_cap_mib": 8704,
        },
        "source": str(SOURCE), "source_crop_xyxy": list(CROP),
        "effective_source_detail_px": args.detail,
        "reference_preparation": (
            "Crop the same 307x307 region; downsample to detail x detail with Lanczos "
            "unless detail=307; then upscale to 512x512 with Lanczos and save lossless PNG."),
        "co_resident_load": "OmniVoice left resident; exact initial whole-device use is in gpu_snapshot.",
    }

    def save():
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
        crop = Image.open(SOURCE).convert("RGB").crop(CROP)
        if args.detail != 307:
            crop = crop.resize((args.detail, args.detail), Image.Resampling.LANCZOS)
        reference = crop.resize((512, 512), Image.Resampling.LANCZOS)
        reference_path = args.output / "reference.png"
        reference.save(reference_path)
        result["reference_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()

        engine = Engine(width=512, height=512, steps=4, fps=25, compile_model=False,
                        optimized=True, real_rope=True, lean=True, fused_qkv=True,
                        memory_mode="staged", cuda_memory_mib=8704)
        result["status"] = "warmup"; save()
        engine.warmup(str(reference_path))
        audio = decode_audio("benchmarks/comparison-10s.wav", 10)
        state = engine.prepare_call(str(reference_path), 50)
        engine.append(state, audio)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        result["status"] = "generating"; save()
        started = time.perf_counter(); chunks = []
        while state.cursor < state.total_frames:
            chunks.append(engine.generate([state])[0])
            print(json.dumps({"detail": args.detail, "frames": state.cursor,
                              "total": state.total_frames}), flush=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        frames = np.concatenate(chunks)[:250]
        result.update(
            wall_s=elapsed, useful_fps=len(frames) / elapsed, frames=len(frames),
            peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
            raw_sha256=hashlib.sha256(frames.tobytes()).hexdigest())
        record(args.output / "video.mp4", [frames], audio, 25)
        for timestamp in (0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5):
            Image.fromarray(frames[round(timestamp * 25)]).save(
                args.output / f"frame-{timestamp:.1f}s.png")
        result["status"] = "complete"; save()
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        save()
        raise


if __name__ == "__main__":
    main()
