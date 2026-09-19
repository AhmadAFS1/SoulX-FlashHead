"""Render close/closer/closest Indian-male clips at audio-conditioning strength 0.5.

The intervention is the existing experimental forward hook that multiplies the
output of every Lite audio cross-attention block. It is process-local and does
not change serving code, defaults, or checkpoint files.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from run import gpu_metadata, sha256
from soulx_rtc.engine import Engine
from soulx_rtc.experiment import record
from soulx_rtc.server import decode_audio


REFERENCES = {
    "close-100": Path("benchmarks/distance_lipsync/evidence-indian-male-20260916/reference-close.png"),
    "closer-125": Path("benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/reference-closer-125.png"),
    "closest-150": Path("benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/reference-closest-150.png"),
}
UNMODIFIED_RESULTS = {
    "close-100": Path("benchmarks/distance_lipsync/evidence-indian-male-20260916/results.json"),
    "closer-125": Path("benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/results.json"),
    "closest-150": Path("benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/results.json"),
}


def unmodified_raw_hash(label):
    data = json.loads(UNMODIFIED_RESULTS[label].read_text())
    if label == "close-100":
        row = next(row for row in data["rows"] if row["distance"] == "close" and row["seed"] == 50)
    else:
        row = next(row for row in data["rows"] if row["label"] == label and row["seed"] == 50)
    return row["raw_sha256"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audio", default="benchmarks/comparison-10s.wav")
    parser.add_argument("--seed", type=int, default=50)
    parser.add_argument("--strength", type=float, default=0.5)
    args = parser.parse_args()
    if args.strength != 0.5:
        parser.error("This evidence runner is fixed to the requested strength 0.5")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    width, height, fps, seconds = 320, 576, 25, 10
    audio = decode_audio(args.audio, seconds)
    results = {
        "status": "initializing",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question": "How do close 1.00x, closer 1.25x, and closest 1.50x behave with audio-conditioning strength 0.5?",
        "intervention": "Multiply every Lite audio cross-attention block output by a shared scalar 0.5.",
        "intervention_scope": "Process-local experimental hooks only; production code/defaults/checkpoints unchanged.",
        "strength": args.strength,
        "seed": args.seed,
        "gpu": gpu_metadata(),
        "profile": {"model": "SoulX-FlashHead Lite", "width": width, "height": height, "fps": fps,
                    "steps": 4, "optimized": True, "real_rope": True, "lean": True,
                    "fused_qkv": True, "compiled": True, "memory_mode": "compact",
                    "int8_weight_storage": True, "torch_allocator_cap_mib": 3584},
        "inputs": {
            "audio": {"path": args.audio, "sha256": sha256(args.audio), "seconds": seconds},
            "references": {label: {"path": str(path), "sha256": sha256(path)} for label, path in REFERENCES.items()},
        },
        "unmodified_raw_hashes": {label: unmodified_raw_hash(label) for label in REFERENCES},
        "rows": [],
    }

    def save():
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")

    save()
    engine = Engine(width=width, height=height, steps=4, fps=fps, optimized=True,
                    real_rope=True, lean=True, fused_qkv=True, compile_model=True,
                    memory_mode="compact", int8_weights=True, cuda_memory_mib=3584)
    torch = engine.torch
    gain = torch.tensor(args.strength, device=engine.pipeline.device, dtype=engine.pipeline.param_dtype)

    def attenuate(module, inputs, output_value):
        return output_value * gain

    hooks = [block.cross_attn.register_forward_hook(attenuate) for block in engine.raw_model.blocks]
    torch._dynamo.reset()
    results["hook_count"] = len(hooks)
    engine.warmup(str(REFERENCES["close-100"]))
    results["status"] = "measuring"
    save()

    for label, reference in REFERENCES.items():
        state = engine.prepare_call(str(reference), args.seed)
        engine.append(state, audio)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        chunks = []
        while state.cursor < state.total_frames:
            chunks.append(engine.generate([state])[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        frames = np.concatenate(chunks)[:seconds * fps]
        video = output / f"{label}-strength-0p5-seed-{args.seed}.mp4"
        record(video, [frames], audio, fps)
        raw_hash = hashlib.sha256(frames.tobytes()).hexdigest()
        row = {"label": label, "strength": args.strength, "seed": args.seed, "frames": len(frames),
               "wall_s": elapsed, "useful_fps": len(frames) / elapsed,
               "peak_torch_mib": torch.cuda.max_memory_allocated() / 2**20,
               "raw_sha256": raw_hash,
               "differs_from_unmodified": raw_hash != results["unmodified_raw_hashes"][label],
               "video": str(video), "video_sha256": sha256(video)}
        if not row["differs_from_unmodified"]:
            raise RuntimeError(f"Strength hook did not change {label}; compiled hook capture failed")
        results["rows"].append(row)
        save()
        print(json.dumps(row), flush=True)

    results["status"] = "complete"
    save()
    for hook in hooks:
        hook.remove()


if __name__ == "__main__":
    main()
