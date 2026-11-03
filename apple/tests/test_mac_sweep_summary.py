"""Sweep summary bookkeeping on synthetic timings.json files (no Mac, no Core ML)."""
import json
import math
import sys

import numpy as np
import pytest

from flashhead_coreml import mac_sweep_summary as summary


def write_run(root, name, status='passed', completed=True, windows=((28, 9.0), (28, 2.0), (28, 2.0)), models='pro320',
              gpu='Apple M5 Pro', dit='inherit'):
    run = root / name
    run.mkdir(parents=True)
    report = {'status': status, 'inference_completed': completed, 'models_directory': f'artifacts/coreml/device/{models}',
              'dit_compute_units': dit, 'requested_compute_units': 'all', 'requested_chunks': len(windows),
              'hardware': {'gpu_model': gpu, 'physical_system_memory_bytes': 48 * 2**30},
              'windows': [{'frames': f, 'inference_seconds': s} for f, s in windows]}
    (run / 'timings.json').write_text(json.dumps(report))
    return run


def run_main(monkeypatch, sweep, *extra):
    monkeypatch.setattr(sys, 'argv', ['mac_sweep_summary', '--sweep', str(sweep), '--no-quality', *extra])
    summary.main()
    return json.loads((sweep / 'summary.json').read_text())


def test_failed_and_partial_runs_are_not_averaged(tmp_path, monkeypatch, capsys):
    write_run(tmp_path, 'baseline-r1')
    write_run(tmp_path, 'baseline-r2', windows=((28, 9.0), (28, 2.0)))
    write_run(tmp_path, 'fast-r1', windows=((28, 5.0), (28, 1.0)), models='pro320-fold')
    write_run(tmp_path, 'fast-r2', status='failed', completed=False, windows=((28, 5.0), (28, 100.0)), models='pro320-fold')
    write_run(tmp_path, 'broken-r1', status='running', completed=False, windows=((28, 5.0),), models='pro320-ane')
    (tmp_path / 'gates.jsonl').write_text(json.dumps({'bundle': 'pro320-fold', 'compute': 'cpu', 'blocks_per_chunk': 3,
                                                      'exit_status': 0, 'log': str(tmp_path / 'missing.log')}) + '\n')
    data = run_main(monkeypatch, tmp_path)
    out = capsys.readouterr().out
    assert data['baseline_warm_fps'] == pytest.approx(84 / 6.0)
    fast = data['arms']['fast']
    assert fast['runs_passed'] == 1 and fast['runs_total'] == 2 and fast['warm_fps'] == pytest.approx(28.0)
    assert fast['speedup'] == pytest.approx(28.0 / 14.0)
    assert fast['gates'][0]['exit_status'] == 0
    broken = data['arms']['broken']
    assert broken['warm_fps'] is None and broken['speedup'] is None  # JSON-safe, never NaN
    assert 'FAIL 1/2' in out and 'cpu:ok' in out
    broken_line = next(line for line in out.splitlines() if line.startswith('broken'))
    assert 'inf' not in broken_line and 'n/a' in broken_line and 'FAIL 1/1' in broken_line
    assert 'Apple M5 Pro' in out
    assert math.isfinite(data['arms']['baseline']['warm_fps'])


def test_baseline_without_passing_run_or_warm_window_stops(tmp_path, monkeypatch):
    write_run(tmp_path, 'baseline-r1', status='failed', completed=False)
    with pytest.raises(SystemExit, match='no passing run'):
        run_main(monkeypatch, tmp_path)
    other = tmp_path / 'one-window'
    write_run(other, 'baseline-r1', windows=((28, 9.0),))
    with pytest.raises(SystemExit, match='fewer than 2 windows'):
        run_main(monkeypatch, other)


def test_mixed_gpu_models_are_flagged(tmp_path, monkeypatch, capsys):
    write_run(tmp_path, 'baseline-r1')
    write_run(tmp_path, 'other-r1', gpu='Apple M4')
    data = run_main(monkeypatch, tmp_path)
    assert data['gpu_models'] == ['Apple M4', 'Apple M5 Pro']
    assert 'WARNING' in capsys.readouterr().out


def test_gate_log_nrmse_is_parsed(tmp_path):
    log = tmp_path / 'validate.log'
    log.write_text('Validating blocks_00 with ALL\n' + json.dumps({'normalized_rmse': 0.012, 'finite': True}) + '\n'
                   + json.dumps({'normalized_rmse': 0.031, 'finite': True}) + '\n')
    (tmp_path / 'gates.jsonl').write_text(json.dumps({'bundle': 'b', 'compute': 'all', 'blocks_per_chunk': 3,
                                                      'exit_status': 1, 'log': str(log)}) + '\n')
    gates = summary.gate_results(tmp_path)
    assert gates['b'][0]['cases'] == 2 and gates['b'][0]['max_normalized_rmse'] == pytest.approx(0.031)
    assert summary.gate_text(gates['b']) == 'all:FAIL'


def test_clean_replaces_non_finite_values():
    assert summary.clean({'a': float('nan'), 'b': [float('inf'), 1.0], 'c': np.float32(2.0)}) == {'a': None, 'b': [None, 1.0], 'c': 2.0}


def test_quality_uses_first_passing_run_with_frames(tmp_path, monkeypatch):
    from PIL import Image
    from flashhead_coreml import quality
    monkeypatch.setattr(quality, 'landmarks', lambda frames: [{}] * len(frames))
    monkeypatch.setattr(quality, 'paired', lambda a, b: {'opening_correlation': 0.99, 'median_edge_ratio_on_both_open': 1.0})
    rng = np.random.default_rng(0)
    frames = rng.integers(0, 255, (3, 8, 8, 3), dtype=np.uint8)
    def save(run, images):
        for i, image in enumerate(images):
            Image.fromarray(image).save(run / f'frame-{i:06d}.png')
    save(write_run(tmp_path, 'baseline-r1'), frames)
    save(write_run(tmp_path, 'baseline-r2'), frames)
    write_run(tmp_path, 'arm-r1', status='failed', completed=False)  # failed repeat must not be compared
    changed = frames.copy()
    changed[1, 0, 0, 0] ^= 1
    save(write_run(tmp_path, 'arm-r2'), changed)
    monkeypatch.setattr(sys, 'argv', ['mac_sweep_summary', '--sweep', str(tmp_path)])
    summary.main()
    data = json.loads((tmp_path / 'summary.json').read_text())
    assert data['arms']['baseline']['repeat_identical'] == '3/3'
    arm = data['arms']['arm']
    assert arm['identical'] == '2/3' and arm['compared_runs'][1].endswith('arm-r2') and arm['lip_corr'] == 0.99
    assert arm['min_psnr'] > 50
