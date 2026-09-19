"""Measure explicit attention kernels on retained, real post-rotary Q/K/V."""
import argparse
import importlib
import json
from pathlib import Path
import statistics

import torch

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_GPU_LOCK, ROOT, atomic_write_json, ensure_new_directory,
    environment_manifest, failure_record, sha256, snapshot_sources,
    utc_now,
)
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.pro_attention_backends import make_attention_backend


def measure(backend, values, repeats):
    events = []
    for _ in range(8):
        backend(*values, 12, owner=None)
    torch.cuda.synchronize()
    for _ in range(repeats):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        result = backend(*values, 12, owner=None)
        end.record()
        events.append((begin, end))
    torch.cuda.synchronize()
    times = [begin.elapsed_time(end) for begin, end in events]
    return result, {'milliseconds': times, 'median_ms': statistics.median(times)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=['sage2_fp16','sage2'], required=True)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--gpu-lock', type=Path, default=DEFAULT_GPU_LOCK)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error('Use at least three observations per case')
    output = ensure_new_directory(args.output)
    source = json.loads(args.captures.read_text())
    if source['status'] != 'complete':
        raise ValueError('Requires a complete real-attention capture')
    cases = {}
    for row in source['raw_tensors']:
        if row['kind'] in ('attention_q','attention_k','attention_v'):
            key = (row['block'], row['invocation'])
            cases.setdefault(key, {})[row['kind'][-1]] = row
    if not cases or any(set(case) != {'q','k','v'} for case in cases.values()):
        raise ValueError('Incomplete Q/K/V capture triplets')
    result = {'status':'starting', 'date_utc':utc_now(), 'execution':'fresh GPU kernel-only diagnostic on real activations',
              'environment':environment_manifest(), 'capture_sha256':sha256(args.captures), 'cases':[]}
    result['source_sha256'] = snapshot_sources(output, [Path(__file__), ROOT/'soulx_rtc/pro_attention_backends.py'])
    lease = None
    try:
        lease = acquire_gpu_lease(args.gpu_lock)
        package = importlib.import_module('sageattention')
        result['package'] = {'path': package.__file__, 'binaries': {
            str(p):sha256(p) for p in Path(package.__file__).parent.glob('*.so')}}
        reference = make_attention_backend('sage2_fp16')
        candidate = make_attention_backend(args.backend)
        with torch.inference_mode():
            for index, (key, case) in enumerate(sorted(cases.items())):
                values = []
                for role in ('q','k','v'):
                    row = case[role]; path = args.captures.parent/row['path']
                    if sha256(path) != row['sha256']: raise ValueError('Attention input hash mismatch')
                    values.append(torch.load(path, map_location='cuda', weights_only=True))
                observations = {}
                for name, backend in ([('reference',reference),('candidate',candidate)] if index%2 == 0 else [('candidate',candidate),('reference',reference)]):
                    observations[name] = measure(backend,values,args.repeats)
                expected, rt = observations['reference']; actual, ct = observations['candidate']
                result['cases'].append({'block':key[0],'invocation':key[1],'step':case['q']['step'],
                    'shape':list(values[0].shape), 'reference':rt,'candidate':ct,
                    'speedup':rt['median_ms']/ct['median_ms'], 'finite':bool(actual.isfinite().all()),
                    'relative_l2':float((actual.float()-expected.float()).norm()/expected.float().norm().clamp_min(1e-12)),
                    'max_abs':float((actual.float()-expected.float()).abs().max())})
                atomic_write_json(output/'results.json',result)
        result.update(status='complete',reference_backend=reference.manifest(),candidate_backend=candidate.manifest())
    except Exception as error:
        result['status']='failed'; result['failure']=failure_record('attention_kernel',error)
        raise
    finally:
        atomic_write_json(output/'results.json',result)
        if lease is not None:lease.close()


if __name__ == '__main__':main()
