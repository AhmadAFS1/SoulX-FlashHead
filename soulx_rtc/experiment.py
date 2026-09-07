"""Matched native-resolution GPU A/B experiment. Encoding is outside timed work."""
import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

from .engine import Engine
from .server import decode_audio


@contextlib.contextmanager
def record_failure(result,save,stage):
    try:
        yield
    except Exception as exc:
        result.update(status="failed",failure=dict(stage=stage,type=type(exc).__name__,message=str(exc)))
        save()
        raise


def record(path, chunks, audio, fps):
    from scipy.signal import resample_poly
    with av.open(str(path), "w") as out:
        video = out.add_stream("libx264", rate=fps)
        video.width, video.height = chunks[0].shape[2], chunks[0].shape[1]
        video.pix_fmt = "yuv420p"
        video.options = {"crf": "18", "preset": "veryfast"}
        sound = out.add_stream("aac", rate=48000)
        sound.layout = "mono"
        for i, rgb in enumerate(frame for chunk in chunks for frame in chunk):
            frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in video.encode(frame):
                out.mux(packet)
        pcm = (resample_poly(audio, 3, 1).clip(-1, 1)*32767).astype(np.int16)
        for offset in range(0, len(pcm), 960):
            samples = pcm[offset:offset+960]
            frame = av.AudioFrame.from_ndarray(samples[None], format="s16", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = 48000, offset, Fraction(1, 48000)
            for packet in sound.encode(frame):
                out.mux(packet)
        for stream in (video, sound):
            for packet in stream.encode():
                out.mux(packet)


def memory_snapshot(torch):
    """Outside timed work; includes TRT allocations, but is not a peak sampler."""
    free,total=torch.cuda.mem_get_info()
    snapshot=dict(device_free_mib=free/2**20,device_total_mib=total/2**20,
                  note="Post-run snapshot, not peak; Torch peak counters exclude TRT-owned allocations")
    try:
        query=subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=5,check=True)
        snapshot["process_device_mib"]=sum(float(line.split(",")[1]) for line in query.stdout.splitlines()
            if int(line.split(",")[0])==os.getpid())
    except (OSError,ValueError,subprocess.SubprocessError):
        snapshot["process_device_mib"]=None
    return snapshot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seconds", type=float, default=10)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--image", required=True)
    ap.add_argument("--audio", required=True)
    ap.add_argument("--modes", nargs="+", choices=["baseline", "cached", "real"], default=["baseline", "cached"])
    ap.add_argument("--memory-mode", choices=["default", "compact", "reference", "staged"], default="default")
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--trt-ffn")
    ap.add_argument("--trt-vae")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    if args.batch < 1 or args.sessions < 1 or args.repeats < 1 or not 0 < args.seconds <= 30:
        ap.error("Require positive batch/sessions/repeats and seconds <=30")
    if args.trt_ffn and (args.batch!=1 or args.memory_mode=="staged"):
        ap.error("TensorRT FFN requires batch one and no staged weight offload")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        ap.error("Output exists; choose a fresh evidence path")
    audio = decode_audio(args.audio, args.seconds)
    result = dict(config=vars(args), inputs={k: dict(file=Path(p).name,
        sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
        for k, p in (("image", args.image), ("audio", args.audio))},
        rows=[], quality=[], warmup_s={},
        metric="Unpaced useful native frames; setup/warmup/encoding excluded; 4-step unless configured otherwise")
    result["code_sha256"] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (Path(__file__), Path(__file__).with_name("engine.py"),
         Path("flash_head/src/modules/flash_head_model.py"), Path("flash_head/utils/utils.py"))}
    def save():
        output.write_text(json.dumps(result, indent=2)+"\n")
    save()
    with record_failure(result,save,"model_load"):
        engine = Engine(width=args.width, height=args.height, steps=args.steps,
                        fps=args.fps, compile_model=not args.eager, profile=True,
                        memory_mode=args.memory_mode,trt_ffn=args.trt_ffn,trt_vae=args.trt_vae)
    print("MODEL_READY", flush=True)
    warmed, reference = set(), None
    for repeat in range(args.repeats):
        order = args.modes if repeat % 2 == 0 else args.modes[::-1]
        for mode in order:
            engine.optimized, engine.real_rope = mode != "baseline", mode == "real"
            if mode not in warmed:
                started = time.perf_counter()
                with record_failure(result,save,f"warmup_{mode}"):
                    engine.warmup(image=args.image, batch_size=min(args.batch, args.sessions))
                result["warmup_s"][mode] = time.perf_counter()-started
                warmed.add(mode)
                print(f"WARMED {mode} {result['warmup_s'][mode]:.2f}s", flush=True)
            with record_failure(result,save,f"prepare_{mode}_{repeat}"):
                states = [engine.prepare(args.image, audio, args.seed+i) for i in range(args.sessions)]
            engine.torch.cuda.reset_peak_memory_stats()
            chunks, metrics, first, completion = [], [], {}, {}
            started = time.perf_counter()
            while any(s.cursor < s.total_frames for s in states):
                for offset in range(0, len(states), args.batch):
                    selected = [(i, s) for i, s in enumerate(states)
                                if offset <= i < offset+args.batch and s.cursor < s.total_frames]
                    if not selected:
                        continue
                    with record_failure(result,save,f"generate_{mode}_{repeat}"):
                        generated = engine.generate([s for _, s in selected])
                    metrics.append(engine.last_metrics.copy())
                    for (i, s), chunk in zip(selected, generated):
                        first.setdefault(i, time.perf_counter()-started)
                        if s.cursor == s.total_frames:
                            completion[i] = time.perf_counter()-started
                        if i == 0 and repeat == 0:
                            chunks.append(chunk)
            elapsed = time.perf_counter()-started
            total = sum(s.total_frames for s in states)
            row = dict(mode=mode, repeat=repeat, useful_frames=total, wall_s=elapsed,
                       aggregate_fps=total/elapsed, first_chunk_s=first, completion_s=completion,
                       peak_allocated_mib=engine.torch.cuda.max_memory_allocated()/2**20,
                       peak_reserved_mib=engine.torch.cuda.max_memory_reserved()/2**20,
                       device_memory_snapshot=memory_snapshot(engine.torch),
                       chunks=metrics)
            result["rows"].append(row)
            print(json.dumps({k:v for k,v in row.items() if k != "chunks"}), flush=True)
            if chunks:
                digest = hashlib.sha256()
                for chunk in chunks:
                    digest.update(chunk.tobytes())
                quality = dict(mode=mode, sha256=digest.hexdigest(), frames=sum(len(c) for c in chunks))
                if mode == "baseline":
                    reference = chunks
                elif reference is not None:
                    errors = [np.abs(a.astype(np.int16)-b.astype(np.int16)) for a,b in zip(reference,chunks)]
                    quality.update(pixel_mae=float(sum(e.sum(dtype=np.float64) for e in errors)/sum(e.size for e in errors)),
                                   max_pixel_error=max(int(e.max()) for e in errors),
                                   per_chunk_mae=[float(e.mean()) for e in errors])
                    del errors
                result["quality"].append(quality)
                if args.record:
                    record(output.with_name(output.stem+f"-{mode}.mp4"), chunks, audio, args.fps)
            save()
    result["summary"] = {mode: dict(
        median_fps=float(np.median([r["aggregate_fps"] for r in result["rows"] if r["mode"]==mode])),
        p95_wall_s=float(np.percentile([r["wall_s"] for r in result["rows"] if r["mode"]==mode],95)))
        for mode in args.modes}
    result["status"] = "completed"
    save()
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
