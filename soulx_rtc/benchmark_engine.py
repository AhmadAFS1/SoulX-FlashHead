"""Independent-session generation throughput; no encoding/playback shortcuts."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .engine import Engine


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--sessions", type=int, nargs="+", default=[1, 10])
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--output", default="benchmarks/engine.json")
    args = ap.parse_args()
    engine = Engine(args.size, args.steps, not args.eager, args.fps)
    print("WARMUP", flush=True)
    t0 = time.perf_counter()
    engine.warmup(batch_size=args.batch)
    warmup = time.perf_counter() - t0
    audio, sr = sf.read("examples/podcast_sichuan_16k.wav", dtype="float32")
    assert sr == 16000
    audio = audio[:round(args.seconds * sr)]
    results = []
    for n in args.sessions:
        states = [engine.prepare("examples/girl.png", np.roll(audio, i * 640), 42 + i)
                  for i in range(n)]
        engine.torch.cuda.reset_peak_memory_stats()
        ticks, first, completed, hashes = [], {}, {}, {}
        t0 = time.perf_counter()
        while any(s.cursor < s.total_frames for s in states):
            for offset in range(0, n, args.batch):
                active = [(i, s) for i, s in enumerate(states)
                          if offset <= i < offset + args.batch and s.cursor < s.total_frames]
                if not active:
                    continue
                frames = engine.generate([s for _, s in active])
                ticks.append(engine.last_metrics.copy())
                for (i, state), chunk in zip(active, frames):
                    if i not in first:
                        import hashlib
                        first[i] = time.perf_counter() - t0
                        hashes[i] = hashlib.sha256(chunk.tobytes()).hexdigest()
                    if state.cursor == state.total_frames:
                        completed[i] = time.perf_counter() - t0
        elapsed = time.perf_counter() - t0
        frames = sum(s.total_frames for s in states)
        row = dict(sessions=n, batch=args.batch, size=args.size, steps=args.steps, fps=args.fps,
                   seconds=args.seconds, warmup_s=warmup, wall_s=elapsed,
                   generated_frames=frames, aggregate_fps=frames / elapsed,
                   realtime_session_capacity=frames / elapsed / args.fps,
                   first_chunk_s=first, completion_s=completed,
                   distinct_first_chunk_hashes=len(set(hashes.values())), chunks=ticks)
        results.append(row)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2))
        print(json.dumps({k:v for k,v in row.items() if k != "chunks"}), flush=True)


if __name__ == "__main__":
    main()
