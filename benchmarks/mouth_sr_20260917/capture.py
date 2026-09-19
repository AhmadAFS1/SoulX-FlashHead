"""Fresh RTX 4070 SUPER baseline capture before video encoding or enhancement."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from soulx_rtc.engine import Engine
from soulx_rtc.server import decode_audio

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, choices=[320, 512], default=320)
    parser.add_argument('--seed', type=int, default=50)
    args = parser.parse_args()
    out = ROOT/f'{args.size}-seed{args.seed}'
    out.mkdir(exist_ok=False)
    square = args.size == 512
    ref = ('benchmarks/square_teeth_20260917/512-25/reference.png' if square else
           'benchmarks/smile_reference_320_20260917/neutral-reference.png')
    audio_path = 'benchmarks/comparison-10s.wav'
    result = dict(status='starting', execution='Fresh local GPU inference',
        date_utc=datetime.now(timezone.utc).isoformat(),
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv'], text=True),
        coresident=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv'], text=True),
        torch=torch.__version__, cuda_runtime=torch.version.cuda,
        physical_vram='12 GB class; nvidia-smi visible capacity in gpu field',
        profile=dict(width=args.size, height=512 if square else 576, seed=args.seed,
            steps=4, shift=5, history=2, strength=1, fps=25, seconds=10,
            int8_weights=not square, compute='BF16', compile=False,
            memory_mode='staged' if square else 'compact', allocator_cap_mib=8704 if square else 3584),
        reference=ref, reference_sha256=hashlib.sha256(Path(ref).read_bytes()).hexdigest(),
        audio=audio_path, audio_sha256=hashlib.sha256(Path(audio_path).read_bytes()).hexdigest())
    def save():
        (out/'capture.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    try:
        engine = Engine(width=args.size, height=512 if square else 576, steps=4, fps=25,
            compile_model=False, optimized=True, real_rope=True, lean=True, fused_qkv=True,
            memory_mode='staged' if square else 'compact', int8_weights=not square,
            cuda_memory_mib=8704 if square else 3584)
        engine.warmup(ref)
        audio = decode_audio(audio_path, 10)
        state = engine.prepare_call(ref, args.seed); engine.append(state, audio)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter(); chunks = []
        while state.cursor < state.total_frames:
            chunks.append(engine.generate([state])[0])
        torch.cuda.synchronize()
        result['generation_s'] = time.perf_counter()-start
        frames = np.concatenate(chunks)[:250]
        result.update(status='complete', frames=len(frames),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
            raw_sha256=hashlib.sha256(frames.tobytes()).hexdigest())
        np.save(out/'raw.npy', frames)
        print(json.dumps(result), flush=True)
    except Exception as exc:
        result.update(status='failed', error=repr(exc));raise
    finally:
        save()


if __name__ == '__main__':
    main()
