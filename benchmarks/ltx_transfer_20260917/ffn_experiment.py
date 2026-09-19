"""Opt-in LTX-inspired token chunking; no service defaults are modified.

Run from the repo with PYTHONPATH=. .venv/bin/python <this file> --output DIR.
Only Linear/GELU/Linear FFNs are supported. No LTX weights are transferred.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import types

import numpy as np
from PIL import Image
import torch
from torch import nn


@contextmanager
def chunk_ffns(model, tokens=256):
    if tokens < 1:
        raise ValueError('tokens must be positive')
    saved = []
    try:
        for block in model.blocks:
            ffn = block.ffn
            if (type(ffn) is not nn.Sequential or len(ffn) != 3
                    or [type(layer) for layer in ffn] != [nn.Linear, nn.GELU, nn.Linear]):
                raise ValueError('Only stock Linear/GELU/Linear FFNs are supported')
            original = ffn.forward
            saved.append((ffn, ffn.__dict__.get('forward'), 'forward' in ffn.__dict__))

            def forward(self, x, original=original):
                if torch.is_grad_enabled():
                    raise RuntimeError('Chunk experiment requires inference/no-grad mode')
                if x.ndim != 3:
                    raise ValueError('Expected batch, tokens, channels')
                if x.shape[1] <= tokens:
                    return original(x)
                # Never overwrite the input: it may be referenced elsewhere.
                result = x.new_empty((*x.shape[:-1], self[-1].out_features))
                for start in range(0, x.shape[1], tokens):
                    result[:, start:start + tokens] = original(x[:, start:start + tokens])
                return result

            ffn.forward = types.MethodType(forward, ffn)
        yield
    finally:
        for ffn, previous, had_override in reversed(saved):
            if had_override:
                ffn.forward = previous
            else:
                del ffn.forward


def cpu_check():
    torch.manual_seed(17)
    ffn = nn.Sequential(nn.Linear(13, 39), nn.GELU(approximate='tanh'), nn.Linear(39, 13)).eval()
    model = types.SimpleNamespace(blocks=[types.SimpleNamespace(ffn=ffn)])
    with torch.inference_mode():
        for shape in [(2, 17, 13), (1, 1, 13), (2, 4, 13)]:
            x = torch.randn(shape); before = x.clone(); expected = ffn(x)
            with chunk_ffns(model, 4):
                actual = ffn(x)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            assert torch.equal(x, before)
            assert 'forward' not in ffn.__dict__
        try:
            with chunk_ffns(model, 4):
                raise RuntimeError('restoration check')
        except RuntimeError:
            pass
        assert 'forward' not in ffn.__dict__
    return 'passed: batch, tail, short sequence, input preservation, restoration'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokens', type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(date_utc=datetime.now(timezone.utc).isoformat(),
        hardware=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version', '--format=csv'], text=True),
        coresident=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv'], text=True),
        torch=torch.__version__, cuda=torch.version.cuda, cpu_check=cpu_check(),
        workload='GPU inference: 512 square, 25 FPS, 4 steps, BF16, staged, seed 50, 2 seconds; no compile',
        runs=[], status='starting')
    def save():
        (args.output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    save()
    from soulx_rtc.engine import Engine
    from soulx_rtc.server import decode_audio
    from soulx_rtc.experiment import record
    engine = Engine(size=512, steps=4, fps=25, compile_model=False,
        optimized=True, real_rope=True, lean=True, fused_qkv=True,
        memory_mode='staged', cuda_memory_mib=8704)
    reference = '/workspace/SoulX-FlashHead/benchmarks/square_teeth_20260917/512-25/reference.png'
    # Existing square reference is reused verbatim; do not alter other runs.
    result['reference'] = reference
    audio = decode_audio('benchmarks/comparison-10s.wav', 2)
    engine.warmup(reference)
    outputs = {}
    def run(label):
        state = engine.prepare_call(reference, 50)
        engine.append(state, audio)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter(); frames = []
        while state.cursor < state.total_frames:
            frames.append(engine.generate([state])[0])
        torch.cuda.synchronize(); elapsed = time.perf_counter()-start
        frames = np.concatenate(frames)[:50]
        result['runs'].append(dict(label=label, wall_s=elapsed, useful_fps=len(frames)/elapsed,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20))
        record(args.output/f'{label}.mp4', [frames], audio, 25)
        Image.fromarray(frames[37]).save(args.output/f'{label}-1.48s.png')
        outputs[label] = frames
        save()
    try:
        run('baseline')
        with chunk_ffns(engine.raw_model, args.tokens):
            run('chunked')
        run('restored')
        for label in ('chunked', 'restored'):
            delta = np.abs(outputs[label].astype(np.int16)-outputs['baseline'].astype(np.int16))
            result[label+'_vs_baseline'] = dict(max_pixel_difference=int(delta.max()),
                mean_absolute_difference=float(delta.mean()), identical=bool(not delta.any()))
        result['status'] = 'complete'
    except Exception as exc:
        result.update(status='failed', error=repr(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
