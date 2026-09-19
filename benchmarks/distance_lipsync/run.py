"""Controlled face-scale experiment for SoulX FlashHead Lite.

"Distance" is operationalized as apparent subject/face scale in the reference
frame. The script keeps identity, audio, seed, native output geometry, model,
and inference settings fixed. It never starts or stops a serving process.
"""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from soulx_rtc.engine import Engine
from soulx_rtc.experiment import record
from soulx_rtc.server import decode_audio


LEVELS = {"close": 1.0, "medium": 0.72, "far": 0.50}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fitted_portrait(source, width, height):
    image = Image.open(source).convert("RGB")
    scale = max(width / image.width, height / image.height)
    resized = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.BILINEAR)
    left = (resized.width - width) // 2
    top = (resized.height - height) // 2
    return resized.crop((left, top, left + width, top + height))


def make_references(source, output, width, height):
    base = fitted_portrait(source, width, height)
    background = base.filter(ImageFilter.GaussianBlur(radius=24))
    paths = {}
    for label, scale in LEVELS.items():
        path = output / f"reference-{label}.png"
        if scale == 1:
            frame = base
        else:
            frame = background.copy()
            subject = base.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
            # Feather all four boundaries so the framing change does not create a hard rectangle.
            mask = Image.new("L", subject.size, 255)
            feather = max(3, round(min(subject.size) * 0.035))
            mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
            frame.paste(subject, ((width - subject.width) // 2, (height - subject.height) // 2), mask)
        frame.save(path)
        paths[label] = path
    return paths


def gpu_metadata():
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=10)
    name, total, used, driver = [part.strip() for part in query.stdout.splitlines()[0].split(",")]
    return {
        "model": name,
        "visible_vram_mib": float(total),
        "used_vram_before_model_load_mib": float(used),
        "driver": driver,
        "evidence": "nvidia-smi query at experiment start",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="examples/girl.png")
    parser.add_argument("--audio", default="benchmarks/comparison-10s.wav")
    parser.add_argument("--seconds", type=float, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[50, 51])
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    width, height, fps = 320, 576, 25
    references = make_references(args.source, output, width, height)
    audio = decode_audio(args.audio, args.seconds)
    results = {
        "status": "initializing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does apparent character distance (face scale) affect SoulX FlashHead lip-sync output?",
        "distance_definition": "Controlled apparent scale of one fitted portrait: close=1.00, medium=0.72, far=0.50.",
        "gpu": gpu_metadata(),
        "co_resident_load": "Existing non-experiment GPU allocation is included in used_vram_before_model_load_mib; no service was stopped.",
        "profile": {"model": "SoulX-FlashHead Lite", "width": width, "height": height, "fps": fps,
                    "steps": 4, "optimized": True, "real_rope": True, "lean": True,
                    "fused_qkv": True, "compiled": True, "memory_mode": "compact",
                    "int8_weight_storage": True, "torch_allocator_cap_mib": 3584},
        "inputs": {
            "source": {"path": args.source, "sha256": sha256(args.source)},
            "audio": {"path": args.audio, "sha256": sha256(args.audio), "seconds": args.seconds},
            "references": {label: {"path": str(path), "sha256": sha256(path), "scale": LEVELS[label]}
                           for label, path in references.items()},
        },
        "seeds": args.seeds,
        "rows": [],
    }

    def save():
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    save()
    engine = Engine(width=width, height=height, steps=4, fps=fps, optimized=True,
                    real_rope=True, lean=True, fused_qkv=True, compile_model=True,
                    memory_mode="compact", int8_weights=True, cuda_memory_mib=3584)
    engine.warmup(str(references["close"]))
    results["status"] = "measuring"
    save()

    for seed in args.seeds:
        for label in LEVELS:
            state = engine.prepare_call(str(references[label]), seed)
            engine.append(state, audio)
            engine.torch.cuda.synchronize()
            engine.torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            chunks = []
            while state.cursor < state.total_frames:
                chunks.append(engine.generate([state])[0])
            engine.torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            frames = np.concatenate(chunks)[:round(args.seconds * fps)]
            video = output / f"{label}-seed-{seed}.mp4"
            record(video, [frames], audio, fps)
            row = {
                "distance": label,
                "scale": LEVELS[label],
                "seed": seed,
                "frames": int(len(frames)),
                "wall_s": elapsed,
                "useful_fps": len(frames) / elapsed,
                "peak_torch_mib": engine.torch.cuda.max_memory_allocated() / 2**20,
                "raw_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
                "video": str(video),
                "video_sha256": sha256(video),
            }
            results["rows"].append(row)
            save()
            print(json.dumps(row), flush=True)

    results["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
