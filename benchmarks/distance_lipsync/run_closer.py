"""Generate two tighter Indian-male framings against the exact prior close clip."""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from run import fitted_portrait, gpu_metadata, sha256
from soulx_rtc.engine import Engine
from soulx_rtc.experiment import record
from soulx_rtc.server import decode_audio


LEVELS = {"closer-125": 1.25, "closest-150": 1.50}


def zoom_reference(source, width, height, scale):
    base = fitted_portrait(source, width, height)
    enlarged = base.resize((round(width * scale), round(height * scale)), Image.Resampling.LANCZOS)
    left = (enlarged.width - width) // 2
    top = (enlarged.height - height) // 2
    return enlarged.crop((left, top, left + width, top + height))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", default="/workspace/benchmarks/same-avatar/shared.png")
    parser.add_argument("--audio", default="benchmarks/comparison-10s.wav")
    parser.add_argument("--baseline", default="benchmarks/distance_lipsync/evidence-indian-male-20260916/close-seed-50.mp4")
    parser.add_argument("--seed", type=int, default=50)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    width, height, fps, seconds = 320, 576, 25, 10
    references = {}
    for label, scale in LEVELS.items():
        path = output / f"reference-{label}.png"
        zoom_reference(args.source, width, height, scale).save(path)
        references[label] = path

    audio = decode_audio(args.audio, seconds)
    results = {
        "status": "initializing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Does zooming closer than the prior Indian-male close framing further improve SoulX mouth motion?",
        "baseline": {"path": args.baseline, "sha256": sha256(args.baseline), "scale": 1.0, "seed": args.seed},
        "gpu": gpu_metadata(),
        "profile": {"model": "SoulX-FlashHead Lite", "width": width, "height": height, "fps": fps,
                    "steps": 4, "optimized": True, "real_rope": True, "lean": True,
                    "fused_qkv": True, "compiled": True, "memory_mode": "compact",
                    "int8_weight_storage": True, "torch_allocator_cap_mib": 3584},
        "inputs": {
            "source": {"path": args.source, "sha256": sha256(args.source)},
            "audio": {"path": args.audio, "sha256": sha256(args.audio), "seconds": seconds},
            "references": {label: {"path": str(path), "sha256": sha256(path), "scale": LEVELS[label]}
                           for label, path in references.items()},
        },
        "rows": [],
    }

    def save():
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    save()
    engine = Engine(width=width, height=height, steps=4, fps=fps, optimized=True,
                    real_rope=True, lean=True, fused_qkv=True, compile_model=True,
                    memory_mode="compact", int8_weights=True, cuda_memory_mib=3584)
    engine.warmup(str(references["closer-125"]))
    results["status"] = "measuring"
    save()

    for label, scale in LEVELS.items():
        state = engine.prepare_call(str(references[label]), args.seed)
        engine.append(state, audio)
        engine.torch.cuda.synchronize()
        engine.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        chunks = []
        while state.cursor < state.total_frames:
            chunks.append(engine.generate([state])[0])
        engine.torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        frames = np.concatenate(chunks)[:seconds * fps]
        video = output / f"{label}-seed-{args.seed}.mp4"
        record(video, [frames], audio, fps)
        row = {"label": label, "scale": scale, "seed": args.seed, "frames": len(frames),
               "wall_s": elapsed, "useful_fps": len(frames) / elapsed,
               "peak_torch_mib": engine.torch.cuda.max_memory_allocated() / 2**20,
               "raw_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
               "video": str(video), "video_sha256": sha256(video)}
        results["rows"].append(row)
        save()
        print(json.dumps(row), flush=True)

    results["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
