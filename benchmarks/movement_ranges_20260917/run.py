"""Wide offline movement sweeps; RTX 4070 SUPER provenance saved per render."""
import argparse
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
SEEDS = [0, 1, 7, 42, 50, 51, 99, 1234]
REPEAT_SEEDS = [7, 50, 99]
SHIFTS = [0.5, 1, 2, 3, 5, 7, 10, 15]
HISTORIES = [1, 2, 3, 4]
TIMES = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5]


def run_name(seed, shift=5, history=2):
    return f'seed{seed}-shift{shift:g}-history{history}'


def gpu():
    return subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv'], text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (ROOT / 'results.json').exists() and not args.resume:
        raise RuntimeError('Use a fresh directory; results already exist.')
    data = dict(status='starting', date_utc=datetime.now(timezone.utc).isoformat(),
        execution='Fresh local GPU inference', gpu_snapshot=gpu(),
        torch=torch.__version__, cuda=torch.version.cuda,
        co_resident_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], text=True),
        profile=dict(width=320, height=576, fps=25, seconds=10, steps=4, strength=1,
                     memory_mode='compact', precision='INT8 weight storage / BF16 compute',
                     compiled=False, allocator_cap_mib=3584),
        inputs={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [SOURCE, AUDIO]},
        code_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     [Path(__file__), Path(engine_module.__file__), Path('flash_head/src/modules/flash_head_model.py')]},
        seeds=SEEDS, repeat_seeds=REPEAT_SEEDS, shifts=SHIFTS, histories=HISTORIES,
        runs=[], controls=[])

    if args.resume:
        prior = (ROOT / 'results.json').read_text()
        (ROOT / 'results-before-resume.json').write_text(prior)
        data = json.loads(prior)
        assert data['status'] == 'failed'
        data.setdefault('failed_attempts', []).append(dict(status=data['status'], error=data.pop('error')))
        data['resume'] = dict(date_utc=datetime.now(timezone.utc).isoformat(), gpu_snapshot=gpu(),
                              runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                              note='History 4 exceeded 3584 MiB allocator cap; retry at 4608 MiB, retaining 3584 MiB for other settings.')

    def save():
        (ROOT / 'results.json').write_text(json.dumps(data, indent=2) + '\n')

    save()
    Image.open(SOURCE).save(ROOT / 'reference.png')
    engine = Engine(width=320, height=576, steps=4, fps=25, compile_model=False,
                    optimized=True, real_rope=True, lean=True, fused_qkv=True,
                    memory_mode='compact', int8_weights=True, cuda_memory_mib=3584)
    engine.warmup(str(SOURCE))
    audio = decode_audio(str(AUDIO), 10)

    def render(seed, shift=5, history=2, control=False):
        name = run_name(seed, shift, history) + (('-resume-control' if args.resume else '-stock-control') if control else '')
        existing = next((r for r in data['runs'] if r['name'] == name), None)
        if existing is not None and not control:
            return existing
        folder = ROOT / name
        if folder.exists() and args.resume:
            assert not (folder / 'results.json').exists()
            folder.rename(ROOT / f'{name}-failed-attempt1')
        folder.mkdir(exist_ok=False)
        cap_mib = 4608 if history == 4 else 3584
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(cap_mib * 2**20 / torch.cuda.get_device_properties(0).total_memory)
        state = engine.prepare_call(str(SOURCE), seed)
        state.pipeline.timesteps = [timestep_transform(torch.tensor([t], device=state.pipeline.device), shift=shift)
                                    for t in [1000, 750, 500, 250, 0]]
        engine.motion_frames = 8 * (history - 1) + 1
        engine.chunk_frames = 33 - engine.motion_frames
        state.pipeline.motion_frames_num = engine.motion_frames
        engine.profile_constants.clear()
        engine.append(state, audio)
        data['status'] = f'generating {name}'; save()
        before = gpu()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter(); chunks = []; boundaries = []
        while state.cursor < state.total_frames:
            chunk = engine.generate([state])[0]
            assert chunk.shape == (engine.chunk_frames, 576, 320, 3)
            assert state.pipeline.latent_motion_frames.shape[1] == history
            chunks.append(chunk)
            boundaries.append(state.cursor)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        frames = np.concatenate(chunks)[:250]
        row = dict(name=name, seed=seed, shift=shift, history=history, strength=1,
                   allocator_cap_mib=cap_mib,
                   overlap_frames=engine.motion_frames, emitted_frames_per_chunk=engine.chunk_frames,
                   chunk_boundaries=[v for v in boundaries if v < 250],
                   gpu_snapshot=before, wall_s=elapsed, useful_fps=250 / elapsed,
                   peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                   frames=len(frames), raw_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),
                   effective_timesteps=[float(t.item()) for t in state.pipeline.timesteps])
        record(folder / 'video.mp4', [frames], audio, 25)
        for t in TIMES:
            Image.fromarray(frames[round(t * 25)]).save(folder / f'frame-{t:.1f}s.png')
        (folder / 'results.json').write_text(json.dumps(row, indent=2) + '\n')
        data['controls' if control else 'runs'].append(row); save()
        print(json.dumps(dict(completed=len(data['runs']), control=control, **row)), flush=True)
        return row

    try:
        if args.resume:
            originals = {r['seed']:r for r in data['controls'] if r['name'].endswith('-stock-control')}
            resumed_control = render(7, control=True)
            assert resumed_control['raw_sha256'] == originals[7]['raw_sha256'], 'Resumed baseline changed.'
            data['resume']['baseline_identity'] = True; save()
        else:
            originals = {seed: render(seed, control=True) for seed in REPEAT_SEEDS}
        original_source = textwrap.dedent(inspect.getsource(Engine.generate))
        patched = original_source
        replacements = {
            'output_offset = 9': 'output_offset = self.motion_frames',
            'if self.lean:': 'if False:  # Keep full context for histories longer than emitted chunks.',
            'videos[:, :, -9:]': 'videos[:, :, -self.motion_frames:]',
        }
        for old, new in replacements.items():
            assert patched.count(old) == 1, (old, patched.count(old))
            patched = patched.replace(old, new)
        (ROOT / 'generated_history_method.py').write_text(patched)
        scope = dict(vars(engine_module))
        exec(compile(patched, str(ROOT / 'generated_history_method.py'), 'exec'), scope)
        engine.generate = scope['generate'].__get__(engine, Engine)
        data['history_adapter'] = dict(replacements=replacements,
                                      original_method_sha256=hashlib.sha256(original_source.encode()).hexdigest(),
                                      baseline_identity={})
        for seed in REPEAT_SEEDS:
            check = render(seed)
            same = check['raw_sha256'] == originals[seed]['raw_sha256']
            data['history_adapter']['baseline_identity'][str(seed)] = same; save()
            assert same, 'History adaptation changed default outputs.'
        for seed in SEEDS:
            if seed not in REPEAT_SEEDS:
                render(seed)
        for seed in REPEAT_SEEDS:
            for shift in SHIFTS:
                if shift != 5:
                    render(seed, shift=shift)
        for seed in REPEAT_SEEDS:
            for history in HISTORIES:
                if history != 2:
                    render(seed, history=history)
        assert len(data['runs']) == 38
        data['status'] = 'complete'; save()
    except Exception as exc:
        data.update(status='failed', error=repr(exc)); save()
        raise


if __name__ == '__main__':
    main()
