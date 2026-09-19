"""Offline seed, schedule-shift and motion-history sweeps on RTX 4070 SUPER.

History changes require an experiment-only adaptation of the fixed-history engine.
Keep a generated copy of that method for inspection; never modify the service.
"""
import hashlib
import inspect
import json
import subprocess
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import soulx_rtc.engine as engine_module
from soulx_rtc.engine import Engine
from soulx_rtc.experiment import record
from soulx_rtc.server import decode_audio
from flash_head.src.pipeline.flash_head_pipeline import timestep_transform

ROOT = Path(__file__).resolve().parent
SOURCE = Path('benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/reference-closer-125.png')
AUDIO = Path('benchmarks/comparison-10s.wav')


def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv'], text=True)


def main():
    destination = ROOT / 'other-parameters'
    destination.mkdir(exist_ok=False)
    strength = json.loads((ROOT / 'results.json').read_text())
    assert strength['status'] == 'complete'
    data = dict(status='starting', date_utc=datetime.now(timezone.utc).isoformat(),
                execution='Fresh GPU inference on NVIDIA GeForce RTX 4070 SUPER; per-run nvidia-smi evidence',
                gpu_snapshot=gpu(), torch=torch.__version__, cuda=torch.version.cuda,
                co_resident_load=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True),
                profile=strength['profile'], source_sha256=strength['source_sha256'],
                audio_sha256=strength['audio_sha256'], runs=[])

    def save():
        (destination / 'results.json').write_text(json.dumps(data, indent=2) + '\n')

    save()
    engine = Engine(width=320, height=576, steps=4, fps=25, compile_model=False,
                    optimized=True, real_rope=True, lean=True, fused_qkv=True,
                    memory_mode='compact', int8_weights=True, cuda_memory_mib=3584)
    engine.warmup(str(SOURCE))
    audio = decode_audio(str(AUDIO), 10)

    def render(name, seed=50, shift=5, history=2):
        folder = destination / name
        folder.mkdir(exist_ok=False)
        state = engine.prepare_call(str(SOURCE), seed)
        state.pipeline.timesteps = [timestep_transform(torch.tensor([t], device=state.pipeline.device), shift=shift)
                                    for t in [1000, 750, 500, 250, 0]]
        state.pipeline.motion_frames_num = 8 * (history - 1) + 1
        engine.motion_frames = state.pipeline.motion_frames_num
        engine.chunk_frames = 33 - engine.motion_frames
        # Cached timestep embeddings are keyed by geometry, so explicitly clear
        # them before each schedule change in this offline experiment.
        engine.profile_constants.clear()
        engine.append(state, audio)
        data['status'] = f'generating {name}'; save()
        before = gpu()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter(); chunks = []
        while state.cursor < state.total_frames:
            chunks.append(engine.generate([state])[0])
            assert state.pipeline.latent_motion_frames.shape[1] == history
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        frames = np.concatenate(chunks)[:250]
        row = dict(name=name, seed=seed, strength=1, shift=shift, history_latents=history,
                   overlap_frames=engine.motion_frames, useful_frames_per_chunk=engine.chunk_frames,
                   gpu_snapshot=before, frames=len(frames), wall_s=elapsed, useful_fps=250 / elapsed,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   raw_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                   effective_timesteps=[float(t.item()) for t in state.pipeline.timesteps])
        record(folder / 'video.mp4', [frames], audio, 25)
        for t in [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5]:
            Image.fromarray(frames[round(t * 25)]).save(folder / f'frame-{t:.1f}s.png')
        data['runs'].append(row); save()
        print(json.dumps(row), flush=True)
        return row

    try:
        baseline = render('baseline')
        data['baseline_matches_strength_control'] = baseline['raw_sha256'] == strength['unhooked_control']['raw_sha256']
        assert data['baseline_matches_strength_control'], 'Cross-process baseline changed; investigate before reusing it.'
        for seed in [52, 53]:
            render(f'seed{seed}', seed=seed)
        for shift in [1, 3, 7]:
            render(f'shift{shift}', shift=shift)

        original = textwrap.dedent(inspect.getsource(Engine.generate))
        patched = original
        replacements = {
            'output_offset = 9': 'output_offset = self.motion_frames',
            'if self.lean:': 'if False:  # Retain full decoded context for 17-frame history.',
            'videos[:, :, -9:]': 'videos[:, :, -self.motion_frames:]',
        }
        for old, new in replacements.items():
            assert patched.count(old) == 1, (old, patched.count(old))
            patched = patched.replace(old, new)
        (destination / 'generated_history_method.py').write_text(patched)
        scope = dict(vars(engine_module))
        exec(compile(patched, str(destination / 'generated_history_method.py'), 'exec'), scope)
        engine.generate = scope['generate'].__get__(engine, Engine)
        data['history_adaptation'] = dict(original_sha256=hashlib.sha256(original.encode()).hexdigest(),
                                         replacements=replacements,
                                         note='Keep 33-frame model window; change overlap, motion re-encoding length, output offset and chunk advance together. Out-of-training-history experiment.')
        check = render('history2-control', history=2)
        data['history_adapter_identity'] = check['raw_sha256'] == baseline['raw_sha256']
        assert data['history_adapter_identity'], 'History adapter baseline differs; investigate before sweeping.'
        for history in [1, 3]:
            render(f'history{history}', history=history)
        data['status'] = 'complete'; save()
    except Exception as exc:
        data.update(status='failed', error=repr(exc)); save()
        raise


if __name__ == '__main__':
    main()
