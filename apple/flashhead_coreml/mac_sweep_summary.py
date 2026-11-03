"""Summarise ``mac-speed-sweep.sh`` runs: warm FPS, speed-up, gates and quality.

Warm FPS follows apple/PERFORMANCE.md. It is the number of frames after the
first window divided by their summed generation time (``inference_seconds``
per window in each run's ``timings.json``), pooled over the passing repeats
of an arm. Only runs whose report says ``status == "passed"`` and
``inference_completed`` count. Failed or partial runs are listed as FAILED
(k/n) and are never averaged in.

Quality compares an arm's first passing run that saved frames with the
baseline's first passing run that saved frames. It reports identical frames,
the minimum finite frame PSNR, and the MediaPipe lip-opening correlation and
oral edge ratio used by the CPU A/B tools (``quality.py``). ``repeat same``
compares an arm's first two passing runs with each other. That is the
run-to-run determinism floor on this Mac.

Gates are read from ``gates.jsonl``, written by the sweep script. Each line
is one ``validate_dit`` run: an NRMSE check per partition against FP32
PyTorch, on random inputs, with the compute units requested. A cpu-ane gate
does not prove Neural Engine placement.

Hardware comes from each run's ``timings.json``. It is reported, never
inferred, and a sweep that mixes GPU models is flagged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def finite(value):
    return value is not None and isinstance(value, (int, float)) and math.isfinite(value)


def clean(value):
    """JSON-safe copy: non-finite floats become None."""
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def fmt(value, spec, missing='n/a'):
    return spec.format(value) if finite(value) else missing


def read_report(run: Path):
    try:
        return json.loads((run / 'timings.json').read_text())
    except (OSError, json.JSONDecodeError):
        return None


def passed(report) -> bool:
    return bool(report) and report.get('status') == 'passed' and bool(report.get('inference_completed'))


def warm(reports):
    """Pooled warm FPS, per-run warm FPS and warm window count."""
    frames = seconds = 0.0
    per_run, windows = [], 0
    for report in reports:
        run_frames = run_seconds = 0.0
        for window in report.get('windows', [])[1:]:
            run_frames += window['frames']
            run_seconds += window['inference_seconds']
            windows += 1
        if run_seconds:
            per_run.append(run_frames / run_seconds)
        frames += run_frames
        seconds += run_seconds
    return (frames / seconds if seconds else float('nan')), per_run, windows


def load_frames(run: Path):
    from PIL import Image
    files = sorted(run.glob('frame-*.png'))
    return np.stack([np.asarray(Image.open(f).convert('RGB')) for f in files]) if files else None


def gate_results(sweep: Path):
    """bundle -> list of gate records from gates.jsonl (latest per compute/partitioning)."""
    path = sweep / 'gates.jsonl'
    gates: dict[str, dict] = {}
    if not path.exists():
        return {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        log = Path(record.get('log', ''))
        worst, cases = None, 0
        if log.exists():
            for text in log.read_text(errors='replace').splitlines():
                if text.startswith('{') and 'normalized_rmse' in text:
                    try:
                        case = json.loads(text)
                    except json.JSONDecodeError:
                        continue
                    cases += 1
                    if finite(case.get('normalized_rmse')):
                        worst = max(worst or 0.0, case['normalized_rmse'])
        record['cases'] = cases
        record['max_normalized_rmse'] = worst
        gates.setdefault(record['bundle'], {})[record['compute']] = record
    return {bundle: list(by_compute.values()) for bundle, by_compute in gates.items()}


def gate_text(records):
    if not records:
        return '-'
    parts = []
    for r in records:
        short = {'cpu': 'cpu', 'cpu-gpu': 'gpu', 'cpu-ane': 'ane', 'all': 'all'}.get(r['compute'], r['compute'])
        parts.append(f"{short}:{'ok' if r.get('exit_status') == 0 else 'FAIL'}")
    return ','.join(parts)


def main():
    from . import quality
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--sweep', type=Path, required=True)
    parser.add_argument('--baseline', default='baseline')
    parser.add_argument('--no-quality', action='store_true', help='Skip frame loading and landmarks (timings only)')
    args = parser.parse_args()

    arms: dict[str, list] = {}
    for run in sorted(args.sweep.glob('*-r*')):
        if run.is_dir():
            arm, _, repeat = run.name.rpartition('-r')
            if repeat.isdigit():
                arms.setdefault(arm, []).append((int(repeat), run, read_report(run)))
    for runs in arms.values():
        runs.sort(key=lambda item: item[0])
    if args.baseline not in arms:
        raise SystemExit(f'No baseline runs named {args.baseline!r} in {args.sweep}')
    base_ok = [(r, p, rep) for r, p, rep in arms[args.baseline] if passed(rep)]
    if not base_ok:
        raise SystemExit(f'The baseline has no passing run in {args.sweep}; see {args.baseline}-r*.log')
    base_fps, _, base_windows = warm([rep for _, _, rep in base_ok])
    if not base_windows:
        raise SystemExit('The baseline runs have fewer than 2 windows; warm FPS needs CHUNKS >= 2')

    gpu_models = sorted({str(rep.get('hardware', {}).get('gpu_model', 'unverified'))
                         for runs in arms.values() for _, _, rep in runs if passed(rep)})
    hardware = base_ok[0][2].get('hardware', {})
    memory = hardware.get('physical_system_memory_bytes')
    print('Mac Core ML sweep (GPU/Neural Engine inference requested; per-operation placement unverified)')
    print(f"  GPU model: {', '.join(gpu_models)}   (evidence: {hardware.get('evidence', 'timings.json')})")
    if len(gpu_models) > 1:
        print('  WARNING: runs in this sweep report different GPU models; do not compare their FPS')
    print(f"  Memory: {memory / 2**30:.0f} GiB unified (physical_system_memory_bytes)" if finite(memory) else '  Memory: unverified')
    print(f"  Runtime: {hardware.get('driver_runtime', 'unverified')}")
    print(f"  Date: {base_ok[0][2].get('date_utc', 'unverified')}   Co-resident load: {hardware.get('co_resident_load', 'unknown')}")
    print(f"  Workload: {base_ok[0][2].get('workload', 'see timings.json')}; "
          f"{base_ok[0][2].get('requested_chunks', '?')} windows per run")
    print('  warm FPS = frames after window 1 / generation time, pooled over passing repeats')
    print()

    gates = gate_results(args.sweep)
    base_frames = None
    base_rows = None
    base_frame_run = None
    if not args.no_quality:
        for _, path, _ in base_ok:
            base_frames = load_frames(path)
            if base_frames is not None:
                base_frame_run = path
                base_rows = quality.landmarks(base_frames)
                break

    header = (f"{'arm':22s} {'bundle':18s} {'dit':8s} {'runs':>6s} {'warm FPS':>9s} {'range':>13s} {'speed-up':>9s} "
              f"{'gates':16s} {'identical':>10s} {'min PSNR':>9s} {'lip corr':>9s} {'edge':>6s} {'repeat same':>11s}")
    print(header)
    summary = {}
    order = [args.baseline] + sorted(name for name in arms if name != args.baseline)
    for name in order:
        runs = arms[name]
        ok = [(r, p, rep) for r, p, rep in runs if passed(rep)]
        first = (ok or runs)[0][2] or {}
        bundle = Path(first.get('models_directory', '?')).name
        dit = first.get('dit_compute_units', 'inherit')
        if dit == 'inherit':
            dit = first.get('requested_compute_units', '?')
        row = {'bundle': bundle, 'dit_compute_units': dit, 'runs_total': len(runs), 'runs_passed': len(ok),
               'failed_runs': [str(p) for _, p, rep in runs if not passed(rep)],
               'flags': {k: first.get(k) for k in ('models_resident', 'parallel_frames', 'gpu_low_precision_accumulation',
                                                   'fast_prediction_specialization', 'audio_lookahead', 'partition_preload')},
               'gates': gates.get(bundle, [])}
        fps, per_run, windows = warm([rep for _, _, rep in ok])
        row.update(warm_fps=fps, warm_fps_per_run=per_run, warm_windows=windows,
                   speedup=fps / base_fps if finite(fps) else float('nan'),
                   warm_ms_per_frame=1000.0 / fps if finite(fps) and fps else float('nan'))
        if len(ok) < len(runs):
            runs_text = f'FAIL {len(runs) - len(ok)}/{len(runs)}'
        else:
            runs_text = f'{len(ok)}/{len(runs)}'
        reason = None
        if args.no_quality:
            reason = 'quality skipped (--no-quality)'
        elif base_frames is None:
            reason = 'baseline saved no frames'
        elif not ok:
            reason = 'no passing run'
        else:
            frames = frame_run = None
            for _, path, _ in ok:
                frames = load_frames(path)
                if frames is not None:
                    frame_run = path
                    break
            if frames is None:
                reason = 'no passing run saved frames'
            elif frames.shape != base_frames.shape:
                reason = f'frame shape {frames.shape} differs from baseline {base_frames.shape}'
            else:
                same = int(sum(np.array_equal(a, b) for a, b in zip(frames, base_frames)))
                psnrs = [quality.psnr(a, b) for a, b in zip(frames, base_frames)]
                finite_psnr = [p for p in psnrs if math.isfinite(p)]
                pair = {} if frame_run == base_frame_run else quality.paired(base_rows, quality.landmarks(frames))
                row.update(identical=f'{same}/{len(frames)}', identical_frames=same, compared_frames=len(frames),
                           compared_runs=[str(base_frame_run), str(frame_run)],
                           min_psnr=min(finite_psnr) if finite_psnr else None,
                           lip_corr=pair.get('opening_correlation'), edge=pair.get('median_edge_ratio_on_both_open'),
                           mouth_center_px=pair.get('median_mouth_center_distance_px'))
                if len(ok) > 1:
                    second = load_frames(ok[1][1])
                    if second is not None and second.shape == frames.shape:
                        repeat_same = int(sum(np.array_equal(a, b) for a, b in zip(frames, second)))
                        row['repeat_identical'] = f'{repeat_same}/{len(frames)}'
        if reason:
            row['quality_missing'] = reason
        if any(g.get('exit_status') != 0 for g in row['gates']):
            row['gate_failed'] = True
            reason = 'GATE FAILED: do not use this timing' + (f'; {reason}' if reason else '')
        summary[name] = row
        identical = row.get('identical', 'n/a')
        min_psnr = 'inf' if 'identical' in row and row.get('min_psnr') is None else fmt(row.get('min_psnr'), '{:8.1f}')
        spread = f'{min(per_run):.2f}-{max(per_run):.2f}' if per_run else 'n/a'
        print(f"{name:22s} {bundle:18s} {str(dit):8s} {runs_text:>6s} {fmt(fps, '{:9.2f}'):>9s} {spread:>13s} "
              f"{fmt(row['speedup'], '{:8.2f}x'):>9s} {gate_text(row['gates']):16s} {identical:>10s} {min_psnr:>9s} "
              f"{fmt(row.get('lip_corr'), '{:9.3f}'):>9s} {fmt(row.get('edge'), '{:6.3f}'):>6s} "
              f"{row.get('repeat_identical', 'n/a'):>11s}" + (f'   ({reason})' if reason else ''))
    print()
    print('Gates: ok/FAIL per validate_dit run (NRMSE <= 0.05 vs FP32 per partition); "-" = not gated in this sweep.')
    print('Baseline bundle pro320 reuses the gate from its original export. lip corr/edge are blank for the baseline itself.')
    (args.sweep / 'summary.json').write_text(json.dumps(clean({
        'gpu_models': gpu_models, 'hardware': hardware,
        'classification': 'Mac Core ML inference; GPU/Neural Engine requested, per-operation placement unverified',
        'baseline': args.baseline, 'baseline_warm_fps': base_fps, 'arms': summary}), indent=1, default=str) + '\n')


if __name__ == '__main__':
    main()
