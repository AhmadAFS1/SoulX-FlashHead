"""Calibrated structured FFN reduction for the PRO transformer.

The PRO FFN (1536 -> 8960 -> 1536, GELU-tanh) holds 56.7% of the
transformer's multiply-accumulates. This module removes whole hidden units, so
every exported partition keeps its dense, fixed-shape matrices and the Swift
runtime's model interface is unchanged. No retraining is used.

Method (per block, independently):

1. ``capture``: render calibration clips with the FP32 portable model and store
   a token subsample of each block's FFN input at both timesteps.
2. ``fit``: form the hidden activations ``a = gelu(W1 f + b1)``, centre them per
   timestep and accumulate their Gram matrix ``G``. Units are removed greedily
   by the Optimal Brain Surgeon criterion for whole input columns of the second
   projection, ``||W2[:, j]||^2 / [G^-1]_jj``, i.e. the least-squares output
   error of dropping unit ``j`` after optimally refitting the units kept.
3. The kept columns of ``W2`` are then refitted by damped least squares, and the
   removed units' mean contribution is restored with one bias per timestep
   (two vectors of 1,536 values per block, selected like the time modulation).

The objective is the FFN output on real activations; it is a layer-local
approximation that is validated end to end by rendered video.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIT = ROOT / 'artifacts/cpu-validation/ffn-fit'


# --------------------------------------------------------------------------
# 1. Capture calibration activations.
# --------------------------------------------------------------------------
def capture(args):
    from .cpu_runtime import CpuRuntime
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    runtime = CpuRuntime('fp32', threads=args.threads)
    generator = np.random.default_rng(args.seed)
    buffers = {}

    def observer(name, block, step, value):
        if name != 'ffn_in':
            return
        tokens = value.shape[1]
        keep = np.sort(generator.choice(tokens, size=int(tokens * args.fraction), replace=False))
        buffers.setdefault((block, step), []).append(value[0, keep].to(torch.float16).numpy())

    runtime.observer = observer
    manifest = {'clips': [], 'fraction': args.fraction, 'seed': args.seed, 'render_seed': args.render_seed,
                'mode': 'fp32', 'tokens_per_block_step': 0}
    for clip, spec in enumerate(args.clip):
        avatar, audio = spec.split(':', 1)
        started = time.perf_counter()
        result = runtime.render(avatar, Path(audio), seed=args.render_seed, max_chunks=args.chunks,
                                on_window=lambda r: print(f'{avatar} window {r.index} {r.seconds["window"]:.1f}s', flush=True))
        manifest['clips'].append({'avatar': avatar, 'audio': audio, 'windows': len(result['windows']),
                                  'seconds': time.perf_counter() - started})
        # Persist per clip so a long capture never holds every activation in RAM.
        for (block, step), parts in buffers.items():
            np.save(out / f'ffn_in_b{block:02d}_s{step}_c{clip}.npy', np.concatenate(parts))
        manifest['tokens_per_block_step'] += int(sum(p.shape[0] for p in buffers[(0, 0)]))
        buffers.clear()
        (out / 'capture.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest))


# --------------------------------------------------------------------------
# 2. Fit: OBS removal order and least-squares refit.
# --------------------------------------------------------------------------
def hidden_activations(f, w1, b1, chunk=8192):
    for start in range(0, f.shape[0], chunk):
        yield F.gelu(F.linear(f[start:start + chunk], w1, b1), approximate='tanh')


def load_inputs(capture_dir: Path, block: int, holdout_every=10):
    """FFN inputs per timestep, split into fit rows and every tenth row held out."""
    fit_rows, held_rows = [], []
    for step in (0, 1):
        files = sorted(capture_dir.glob(f'ffn_in_b{block:02d}_s{step}_c*.npy'))
        f = torch.from_numpy(np.concatenate([np.load(path) for path in files])).float()
        mask = torch.arange(f.shape[0]) % holdout_every == holdout_every - 1
        fit_rows.append(f[~mask])
        held_rows.append(f[mask])
    return fit_rows, held_rows


def block_statistics(fit_rows, w1, b1):
    """Per-timestep means and the pooled per-timestep-centred Gram (float64).

    Two passes: means first, then centred products, so near-constant units do
    not lose their (small) variance to cancellation.
    """
    n = w1.shape[0]
    means, counts = [], []
    for f in fit_rows:
        total = torch.zeros(n, dtype=torch.float64)
        for a in hidden_activations(f, w1, b1):
            total += a.sum(0, dtype=torch.float64)
        means.append(total / f.shape[0])
        counts.append(int(f.shape[0]))
    gram = torch.zeros(n, n, dtype=torch.float64)
    for f, mean in zip(fit_rows, means):
        partial = torch.zeros(n, n, dtype=torch.float32)
        centre = mean.float()
        for a in hidden_activations(f, w1, b1):
            a = a - centre
            partial.addmm_(a.T, a)
        gram += partial.double()
    return gram, torch.stack(means), counts


def obs_order(gram, w2, remove, damp=0.01, batch=1):
    """Greedy OBS removal of ``remove`` whole columns of w2 (D x n).

    ``batch`` columns are removed per update (the lowest current scores), with
    the exact block form of the downdate; batch=1 is the classic one-at-a-time
    procedure. Returns the removal order and the cumulative predicted output
    error (sum over calibration tokens, unweighted).
    """
    n = gram.shape[0]
    h = gram.clone()
    h.diagonal().add_(damp * h.diagonal().mean())
    hinv = torch.cholesky_inverse(torch.linalg.cholesky(h))
    del h
    w = w2.double().clone()
    alive = torch.ones(n, dtype=torch.bool)
    order, errors = [], []
    cumulative = 0.0
    while len(order) < remove:
        take = min(batch, remove - len(order))
        score = (w * w).sum(0) / hinv.diagonal()
        score[~alive] = float('inf')
        chosen = torch.topk(score, take, largest=False).indices
        block = hinv[chosen][:, chosen]
        block_inv = torch.linalg.inv(block)
        ws = w[:, chosen]
        cumulative += float(torch.einsum('di,ij,dj->', ws, block_inv, ws))
        rows = block_inv @ hinv[chosen]            # (take x n)
        w -= ws @ rows
        hinv -= hinv[:, chosen] @ rows
        alive[chosen] = False
        w[:, chosen] = 0
        order.extend(int(j) for j in chosen)
        errors.extend([cumulative] * take)
    return order, errors


def refit(gram, means, w2, b2, kept, damp=0.01):
    """Least-squares W2 on kept units plus one compensating bias per timestep."""
    g = gram.double()
    h = g[kept][:, kept].clone()
    h.diagonal().add_(damp * g.diagonal().mean())
    rhs = g[kept] @ w2.double().T           # (k x D) = G_K: W2^T
    solution = torch.cholesky_solve(rhs, torch.linalg.cholesky(h))  # (k x D)
    w2_new = solution.T.contiguous()       # (D x k)
    # Exact mean restoration for each timestep: W2 mu_t - W2' mu_t,K.
    step_bias = means @ w2.double().T - means[:, kept] @ solution
    return w2_new.float(), b2.float(), step_bias.float()


def ffn_output(f, w1, b1, w2, b2, step_bias=None, step=0):
    out = []
    for a in hidden_activations(f, w1, b1):
        y = F.linear(a, w2, b2)
        if step_bias is not None:
            y = y + step_bias[step]
        out.append(y)
    return torch.cat(out)


def holdout_errors(held_rows, means, w1, b1, w2, b2, variants):
    """Relative squared FFN-output error on held-out tokens for each variant.

    Normalised by the held-out output variance about each timestep's
    calibration mean, so 0.01 means 1% of the signal energy is lost.
    """
    from .cpu_runtime import coreml_int8_dequantized
    w1q, w2q = coreml_int8_dequantized(w1), coreml_int8_dequantized(w2)
    num = {name: 0.0 for name in ['int8_dense'] + list(variants)}
    den = 0.0
    for step, f in enumerate(held_rows):
        reference = ffn_output(f, w1, b1, w2, b2)
        centre = means[step].float() @ w2.T + b2
        den += float(((reference - centre) ** 2).sum())
        num['int8_dense'] += float(((ffn_output(f, w1q, b1, w2q, b2) - reference) ** 2).sum())
        for name, (kept, w2_new, step_bias, quantize) in variants.items():
            a_w1, a_w2 = w1[kept], w2_new
            if quantize:
                a_w1, a_w2 = coreml_int8_dequantized(a_w1), coreml_int8_dequantized(a_w2)
            y = ffn_output(f, a_w1, b1[kept], a_w2, b2, step_bias, step)
            num[name] += float(((y - reference) ** 2).sum())
    return {name: value / den for name, value in num.items()}


def fit(args):
    from .dit import Blocks, ProConfig, Profile, TimeEmbedding, load_component
    torch.set_num_threads(args.threads)
    capture_dir = Path(args.capture)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = ROOT / 'models/SoulX-FlashHead-1_3B/Model_Pro'
    checkpoint = checkpoint_dir / 'diffusion_pytorch_model.safetensors'
    cfg = ProConfig.from_json(checkpoint_dir / 'config.json')
    ratios = [float(r) for r in args.ratios.split(',')]
    remove = int(round(max(ratios) * cfg.ffn_dim))
    with torch.inference_mode():
        _, tm = load_component(TimeEmbedding(cfg), checkpoint, 'time')(torch.tensor([1000., 625.]))
    report_path = out / 'fit.json'
    report = json.loads(report_path.read_text()) if args.resume and report_path.exists() else {
        'capture': json.loads((capture_dir / 'capture.json').read_text()), 'damp': args.damp,
        'obs_batch': args.batch, 'ratios': ratios, 'holdout': 'every tenth captured token per timestep',
        'blocks': {}}
    layers = range(cfg.num_layers) if args.blocks is None else [int(b) for b in args.blocks.split(',')]
    for block in layers:
        if args.resume and str(block) in report['blocks'] and (out / f'block_{block:02d}.pt').exists():
            continue
        started = time.perf_counter()
        module = load_component(Blocks(cfg, Profile(320, 176), block, block + 1, tm), checkpoint, 'blocks', block)
        ffn = module.blocks[0].ffn
        w1, b1, w2, b2 = ffn[0].weight, ffn[0].bias, ffn[2].weight, ffn[2].bias
        with torch.inference_mode():
            fit_rows, held_rows = load_inputs(capture_dir, block)
            gram, means, counts = block_statistics(fit_rows, w1, b1)
            energy = float(torch.einsum('dn,nm,dm->', w2.double(), gram, w2.double()))
            order, _ = obs_order(gram, w2, remove, args.damp, args.batch)
            entry = {'fit_tokens': counts, 'holdout_tokens': [int(f.shape[0]) for f in held_rows],
                     'order': order, 'predicted_relative_error': {}}
            fitted, variants = {}, {}
            for ratio in ratios:
                removed = set(order[:int(round(ratio * cfg.ffn_dim))])
                kept = torch.tensor([j for j in range(cfg.ffn_dim) if j not in removed])
                w2_new, _, step_bias = refit(gram, means, w2, b2, kept, args.damp)
                entry['predicted_relative_error'][str(ratio)] = predicted_error(gram, w2, kept, w2_new) / energy
                fitted[str(ratio)] = {'kept': kept.to(torch.int32), 'w2': w2_new, 'step_bias': step_bias}
                variants[f'prune_{ratio}'] = (kept, w2_new, step_bias, False)
                variants[f'prune_{ratio}_int8'] = (kept, w2_new, step_bias, True)
            entry['holdout_relative_error'] = holdout_errors(held_rows, means, w1, b1, w2, b2, variants)
        torch.save(fitted, out / f'block_{block:02d}.pt')
        entry['seconds'] = time.perf_counter() - started
        report['blocks'][str(block)] = entry
        report_path.write_text(json.dumps(report) + '\n')
        print(json.dumps({'block': block, 'seconds': round(entry['seconds'], 1),
                          'holdout': {k: round(v, 5) for k, v in entry['holdout_relative_error'].items()}}),
              flush=True)
        del module, gram, fit_rows, held_rows


def predicted_error(gram, w2, kept, w2_new):
    """Centred calibration error of the refitted FFN, ||A W2^T - A_K W2'^T||^2."""
    g = gram.double()
    w = w2.double()
    wk = torch.zeros_like(w)
    wk[:, kept] = w2_new.double()
    delta = w - wk
    return float(torch.einsum('dn,nm,dm->', delta, g, delta))


# --------------------------------------------------------------------------
# 3. Apply a fitted reduction to a runtime or an exporter module.
# --------------------------------------------------------------------------
def reduce_block(block, fitted):
    """Replace a DiTBlock's FFN with the reduced, refitted one (in place)."""
    from torch import nn
    kept = fitted['kept'].long()
    first, second = block.ffn[0], block.ffn[2]
    reduced_first = nn.Linear(first.in_features, len(kept))
    reduced_first.weight.data.copy_(first.weight.data[kept])
    reduced_first.bias.data.copy_(first.bias.data[kept])
    reduced_second = nn.Linear(len(kept), second.out_features)
    reduced_second.weight.data.copy_(fitted['w2'])
    reduced_second.bias.data.copy_(second.bias.data)
    block.ffn[0], block.ffn[2] = reduced_first.requires_grad_(False), reduced_second.requires_grad_(False)
    block.ffn_step_bias = fitted['step_bias'].clone()  # [2, D]: t=1000, t=625
    return len(kept)


def parse_variant(name: str):
    """'ffn25' -> 0.25 of hidden units removed in every block."""
    digits = name[3:]
    if not digits.isdigit():
        raise ValueError(f'Unknown FFN variant {name}')
    return int(digits) / 100.0


def reduce_partition(blocks, ratio: float, fit_dir: Path = DEFAULT_FIT):
    """Apply a fitted reduction to a ``Blocks`` partition (global layer ids)."""
    widths, hashes = [], []
    for offset, block in enumerate(blocks.blocks):
        layer = blocks.start + offset
        path = Path(fit_dir) / f'block_{layer:02d}.pt'
        fitted = torch.load(path)[str(ratio)]
        widths.append(reduce_block(block, fitted))
        hashes.append(_file_sha256(path))
    return {'transform': 'calibrated_ffn_reduction', 'removed_fraction': ratio, 'layers': [blocks.start, blocks.end],
            'ffn_widths': widths, 'fit_dir': str(fit_dir), 'fit_sha256': hashes}


def _file_sha256(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        for data in iter(lambda: handle.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def variant_transform(name: str, fit_dir: Path = DEFAULT_FIT):
    ratio = parse_variant(name)

    def transform(runtime):
        return reduce_partition(runtime.blocks, ratio, fit_dir)

    transform.ffn_ratio, transform.fit_dir = ratio, fit_dir
    return [transform], (f'FFN hidden width reduced by {ratio:.0%} in every block '
                         '(OBS order, least-squares refit, per-timestep bias)')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    c = sub.add_parser('capture')
    c.add_argument('--clip', action='append', required=True, help='avatar:audio.wav (repeatable)')
    c.add_argument('--chunks', type=int)
    c.add_argument('--fraction', type=float, default=0.5)
    c.add_argument('--seed', type=int, default=7)
    c.add_argument('--render-seed', type=int, default=1234)
    c.add_argument('--threads', type=int, default=4)
    c.add_argument('--output', default=str(ROOT / 'artifacts/cpu-validation/ffn-capture'))
    f = sub.add_parser('fit')
    f.add_argument('--capture', default=str(ROOT / 'artifacts/cpu-validation/ffn-capture'))
    f.add_argument('--output', default=str(DEFAULT_FIT))
    f.add_argument('--ratios', default='0.125,0.25,0.375')
    f.add_argument('--damp', type=float, default=0.01)
    f.add_argument('--batch', type=int, default=32, help='OBS columns removed per downdate')
    f.add_argument('--blocks', help='Comma-separated layer ids (default: all)')
    f.add_argument('--resume', action='store_true')
    f.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    {'capture': capture, 'fit': fit}[args.command](args)


if __name__ == '__main__':
    main()
