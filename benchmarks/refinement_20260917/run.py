"""Fresh RTX 4070 SUPER GPU comparison; each arm has identical reference/audio."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from PIL import Image
import torch
from soulx_rtc.engine import Engine
from soulx_rtc.experiment import record
from soulx_rtc.server import decode_audio
from flash_head.src.pipeline.flash_head_pipeline import timestep_transform

ROOT = Path(__file__).resolve().parent
REFERENCE = Path('benchmarks/smile_reference_320_20260917/neutral-reference.png')
AUDIO = Path('benchmarks/comparison-10s.wav')
ARMS = [('baseline4', ()), ('continuous6', ()), ('refine100', (100, 50)),
        ('refine50', (50, 25))]


def main():
    extra = '--extra' in sys.argv
    if not extra and (ROOT/'results.json').exists():
        raise FileExistsError('Preserve existing results; use a fresh artifact directory')
    result = dict(status='starting', date_utc=datetime.now(timezone.utc).isoformat(),
        execution='Fresh local GPU inference',
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv'], text=True),
        coresident=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv'], text=True),
        physical_vram='12 GB class; visible capacity recorded by nvidia-smi',
        torch=torch.__version__, cuda_runtime=torch.version.cuda,
        profile=dict(width=320, height=576, fps=25, seconds=10,
            precision='INT8 weight storage / BF16 compute', memory_mode='compact',
            allocator_cap_mib=3584, compile=False, color_correction=1.0),
        reference=str(REFERENCE), reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        audio=str(AUDIO), audio_sha256=hashlib.sha256(AUDIO.read_bytes()).hexdigest(), runs=[])
    if extra:
        result = json.loads((ROOT/'results.json').read_text())
        assert result['status'] == 'complete'
        result['extra_started_utc'] = datetime.now(timezone.utc).isoformat()
        result['extra_gpu'] = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv'], text=True)
        result['status'] = 'extra_running'
    def save():
        (ROOT/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    engine = Engine(width=320, height=576, steps=4, fps=25, compile_model=False,
        optimized=True, real_rope=True, lean=True, fused_qkv=True,
        memory_mode='compact', int8_weights=True, cuda_memory_mib=3584)
    engine.warmup(str(REFERENCE))
    audio = decode_audio(str(AUDIO), 10)
    hashes = {(r['seed'], r['arm']): r['raw_sha256'] for r in result['runs']}
    rng_hashes = {(r['seed'], r['arm']): r['base_rng_sha256'] for r in result['runs']}
    try:
        for seed in (50, 51):
            arms = ([('refine250', (250, 125)), ('refine500', (500, 250))] if extra
                    else ARMS + ([('restored4', ())] if seed == 50 else []))
            for name, refinement in arms:
                output = ROOT/f'seed{seed}'/name
                output.mkdir(parents=True, exist_ok=False)
                engine.refinement_timesteps = refinement
                state = engine.prepare_call(str(REFERENCE), seed)
                if name == 'continuous6':
                    state.pipeline.timesteps = [timestep_transform(torch.tensor([float(t)], device='cuda'), shift=5)
                        for t in np.linspace(1000, 0, 7)]
                engine.profile_constants.clear()
                engine.append(state, audio)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter(); chunks = []
                while state.cursor < state.total_frames:
                    chunks.append(engine.generate([state])[0])
                torch.cuda.synchronize(); elapsed = time.perf_counter() - start
                frames = np.concatenate(chunks)[:250]
                digest = hashlib.sha256(frames.tobytes()).hexdigest()
                rng_digest = hashlib.sha256(state.pipeline.generator.get_state().numpy().tobytes()).hexdigest()
                hashes[(seed, name)] = digest; rng_hashes[(seed, name)] = rng_digest
                row = dict(seed=seed, arm=name, frames=len(frames), wall_s=elapsed,
                    useful_fps=len(frames)/elapsed, raw_sha256=digest, base_rng_sha256=rng_digest,
                    model_evaluations_per_chunk=6 if name == 'continuous6' or refinement else 4,
                    base_transformed_times=[float(t.item()) for t in state.pipeline.timesteps],
                    refinement_raw_times=refinement,
                    refinement_transformed_times=[float(t.item()) for t in getattr(state.pipeline, 'refinement_times', [])],
                    peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
                record(output/'video.mp4', [frames], audio, 25)
                for timestamp in (.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5):
                    Image.fromarray(frames[round(timestamp*25)]).save(output/f'frame-{timestamp:.1f}s.png')
                result['runs'].append(row); save()
                print(json.dumps(row), flush=True)
        result['disabled_restoration_identical'] = hashes[(50, 'baseline4')] == hashes[(50, 'restored4')]
        result['refinement_preserves_base_rng'] = all(value == rng_hashes[(seed, 'baseline4')]
            for (seed, name), value in rng_hashes.items() if name.startswith('refine'))
        assert result['disabled_restoration_identical']
        assert result['refinement_preserves_base_rng']
        result['status'] = 'complete'
    except Exception as exc:
        result.update(status='failed', error=repr(exc)); raise
    finally:
        save()


if __name__ == '__main__':
    main()
