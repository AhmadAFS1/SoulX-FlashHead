"""Fresh RTX 4070 SUPER strength sweep; offline process-local hooks only."""
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

ROOT = Path(__file__).resolve().parent
SOURCE = Path('benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/reference-closer-125.png')
AUDIO = Path('benchmarks/comparison-10s.wav')
LEVELS = [1.0, 0.9, 0.75, 0.5]
TIMES = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5]


def snapshot():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv'], text=True)


def main():
    if (ROOT / 'results.json').exists():
        raise RuntimeError('Results already exist; preserve them and use a new directory.')
    metadata = dict(status='starting', date_utc=datetime.now(timezone.utc).isoformat(),
                    execution='Fresh local GPU inference', gpu_snapshot=snapshot(),
                    torch=torch.__version__, cuda=torch.version.cuda,
                    co_resident_load=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True),
                    profile=dict(width=320, height=576, fps=25, steps=4, seconds=10,
                                 precision='INT8 weight storage / BF16 compute', memory_mode='compact',
                                 compiled=False, allocator_cap_mib=3584, shift=5, motion_frames_latent_num=2),
                    source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
                    audio=str(AUDIO), audio_sha256=hashlib.sha256(AUDIO.read_bytes()).hexdigest(),
                    levels=LEVELS, seeds=[50, 51], runs=[])

    def save():
        (ROOT / 'results.json').write_text(json.dumps(metadata, indent=2) + '\n')

    save()
    Image.open(SOURCE).save(ROOT / 'reference.png')
    engine = Engine(width=320, height=576, steps=4, fps=25, compile_model=False,
                    optimized=True, real_rope=True, lean=True, fused_qkv=True,
                    memory_mode='compact', int8_weights=True, cuda_memory_mib=3584)
    engine.warmup(str(SOURCE))
    audio = decode_audio(str(AUDIO), 10)

    def render(seed):
        state = engine.prepare_call(str(SOURCE), seed)
        engine.append(state, audio)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        chunks = []
        while state.cursor < state.total_frames:
            chunks.append(engine.generate([state])[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        frames = np.concatenate(chunks)[:250]
        return frames, dict(wall_s=elapsed, useful_fps=250 / elapsed,
                            raw_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                            frames=len(frames), peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)

    metadata['status'] = 'control'; save()
    baseline, baseline_stats = render(50)
    record(ROOT / 'unhooked-seed50.mp4', [baseline], audio, 25)
    metadata['unhooked_control'] = baseline_stats
    gain = torch.ones((), device=engine.pipeline.device, dtype=engine.pipeline.param_dtype)
    hooks = [block.cross_attn.register_forward_hook(lambda module, inputs, output: output * gain)
             for block in engine.raw_model.blocks]
    metadata['hook_count'] = len(hooks)
    assert len(hooks) == 30
    try:
        for seed in metadata['seeds']:
            for level in LEVELS:
                gain.fill_(level)
                folder = ROOT / f'seed{seed}-strength{level:g}'
                folder.mkdir(exist_ok=False)
                metadata['status'] = f'generating seed={seed} strength={level:g}'; save()
                before = snapshot()
                frames, stats = render(seed)
                if seed == 50 and level == 1:
                    delta = np.abs(frames.astype(np.int16) - baseline.astype(np.int16))
                    metadata['hook_identity_control'] = dict(exact=bool(np.array_equal(frames, baseline)),
                                                            mae=float(delta.mean()), max_error=int(delta.max()))
                    del baseline
                record(folder / 'video.mp4', [frames], audio, 25)
                for t in TIMES:
                    Image.fromarray(frames[round(t * 25)]).save(folder / f'frame-{t:.1f}s.png')
                row = dict(seed=seed, strength=level, directory=folder.name, gpu_snapshot=before, **stats)
                (folder / 'results.json').write_text(json.dumps(row, indent=2) + '\n')
                metadata['runs'].append(row); save()
                print(json.dumps(row), flush=True)
        for seed in metadata['seeds']:
            hashes = [r['raw_sha256'] for r in metadata['runs'] if r['seed'] == seed]
            assert len(set(hashes)) == len(LEVELS), 'Strength changes did not affect output.'
        metadata['status'] = 'complete'; save()
    except Exception as exc:
        metadata.update(status='failed', error=repr(exc)); save()
        raise
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == '__main__':
    main()
