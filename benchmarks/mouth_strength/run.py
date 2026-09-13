"""Offline Lite audio-residual sweep; never operates on server processes.

Run from the repository with its venv. --output must be a new directory.
The forward hooks exist only in this experiment process; deployed code is unchanged.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from soulx_rtc.engine import Engine
from soulx_rtc.experiment import record
from soulx_rtc.server import decode_audio


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--male-input', required=True, help='Directory with anchor.png and motion.npy')
    parser.add_argument('--audio', default='benchmarks/comparison-10s.wav')
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(exist_ok=False, parents=True)
    levels = [1.0, 0.9, 0.75, 0.5]
    result = dict(status='initializing', levels=levels, seeds=[50, 51], rows=[],
                  intervention='Multiply output of every Lite audio cross-attention by a shared scalar',
                  audio_sha256=hashlib.sha256(Path(args.audio).read_bytes()).hexdigest())

    def save():
        (root / 'results.json').write_text(json.dumps(result, indent=2))

    save()
    engine = Engine(width=320, height=576, steps=4, fps=25, optimized=True,
                    real_rope=True, lean=True, fused_qkv=True, compile_model=True,
                    memory_mode='compact', int8_weights=True, cuda_memory_mib=3584)
    torch = engine.torch
    gain = torch.ones((), device=engine.pipeline.device, dtype=engine.pipeline.param_dtype)

    def attenuate(module, inputs, output):
        return output * gain

    audio = decode_audio(args.audio, 10)
    inputs = {'male': str(Path(args.male_input) / 'anchor.png'), 'girl': 'examples/girl.png'}
    motion = np.load(Path(args.male_input) / 'motion.npy', allow_pickle=False)
    result['portraits'] = {k: dict(path=v, sha256=hashlib.sha256(Path(v).read_bytes()).hexdigest())
                          for k, v in inputs.items()}
    result['male_motion_sha256'] = hashlib.sha256(motion.tobytes()).hexdigest()
    result['profile'] = dict(width=320, height=576, steps=4, fps=25, int8_storage=True,
                             compiled=True, torch_cap_mib=3584)

    def prepare(avatar, seed):
        state = engine.prepare_call(inputs[avatar], seed)
        if avatar == 'male':
            engine.recondition(state, motion)
        engine.append(state, audio)
        return state

    def render(avatar, seed):
        state = prepare(avatar, seed)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        chunks = []
        while state.cursor < state.total_frames:
            chunks.append(engine.generate([state])[0])
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        raw = np.concatenate(chunks)[:250]
        return raw, dict(wall_s=elapsed, useful_fps=250 / elapsed, padded_frames=state.cursor,
                         peak_torch_mib=torch.cuda.max_memory_allocated() / 2**20,
                         raw_sha256=hashlib.sha256(raw.tobytes()).hexdigest())

    # Establish baseline without hooks, then verify strength=1 preserves output.
    engine.warmup(inputs['male'])
    for avatar in inputs:
        state = prepare(avatar, 50)
        engine.generate([state])
        del state
    reference, _ = render('male', 50)
    reference_hash = hashlib.sha256(reference.tobytes()).hexdigest()
    record(root / 'male-seed-50-unmodified.mp4', [reference], audio, 25)
    hooks = [b.cross_attn.register_forward_hook(attenuate) for b in engine.raw_model.blocks]
    # The unhooked baseline was already compiled. Dynamo does not necessarily
    # guard subsequent hook registration, so explicitly retrace with hooks.
    torch._dynamo.reset()
    result['hook_count'] = len(hooks)
    for level in levels:
        gain.fill_(level)
        state = prepare('male', 50)
        engine.generate([state])
        del state
    gain.fill_(1)
    check, _ = render('male', 50)
    result['identity_control_exact'] = hashlib.sha256(check.tobytes()).hexdigest() == reference_hash
    delta = np.abs(check.astype(np.int16)-reference.astype(np.int16))
    result['identity_control_mae_255'] = float(delta.mean())
    result['identity_control_max_error_255'] = int(delta.max())
    result['identity_control_note'] = 'Retraced hooked graph may change rounding; retain unmodified control separately.'
    reference_hash = hashlib.sha256(check.tobytes()).hexdigest()
    print('BASELINE_CONTROL', json.dumps({k:v for k,v in result.items() if k.startswith('identity_')}),flush=True)
    del check, reference, delta
    torch.cuda.empty_cache()
    result['status'] = 'measuring'
    save()
    print('WARMED', flush=True)
    for avatar in inputs:
        for seed in [50, 51]:
            contact = Image.new('RGB', (320*4, 576*5))
            for repeat in range(2):
                for level in (levels if repeat == 0 else levels[::-1]):
                    gain.fill_(level)
                    raw, row = render(avatar, seed)
                    if avatar == 'male' and seed == 50 and level != 1.0:
                        if row['raw_sha256'] == reference_hash:
                            raise RuntimeError('Attenuation did not affect output; check compiled hook capture')
                    row.update(avatar=avatar, seed=seed, repeat=repeat, strength=level,
                               frames=len(raw), finite=bool(np.isfinite(raw).all()))
                    result['rows'].append(row)
                    save()
                    print(json.dumps(row), flush=True)
                    if repeat == 0:
                        name = f'{avatar}-seed-{seed}-strength-{level:g}'
                        record(root / f'{name}.mp4', [raw], audio, 25)
                        for j, frame in enumerate([12, 37, 75, 125, 175]):
                            tile = Image.fromarray(raw[frame])
                            draw = ImageDraw.Draw(tile)
                            draw.rectangle((0, 0, 320, 22), fill='black')
                            draw.text((4, 4), f'strength={level:g} frame={frame}', fill='white')
                            contact.paste(tile, (levels.index(level)*320, j*576))
                    del raw
            contact.save(root / f'{avatar}-seed-{seed}-contact.jpg')
    result['status'] = 'complete'
    save()
    for hook in hooks:
        hook.remove()


if __name__ == '__main__':
    main()
