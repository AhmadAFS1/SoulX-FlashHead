"""CPU aggregation of retained run manifests; never mixes GPU profiles."""
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
rows=[]
for path in sorted(ROOT.glob('*/results.json')):
    run=json.loads(path.read_text())
    if run.get('status')!='complete':
        continue
    rows.append(dict(name=path.parent.name, gpu=run['gpu'], profile=run['profile'],
        repetitions=len(run['runs']),
        median_useful_fps=statistics.median(r['useful_fps'] for r in run['runs']),
        min_useful_fps=min(r['useful_fps'] for r in run['runs']),
        max_useful_fps=max(r['useful_fps'] for r in run['runs']),
        peak_torch_allocated_mib=max(r['peak_allocated_mib'] for r in run['runs']),
        peak_torch_reserved_mib=max(r['peak_reserved_mib'] for r in run['runs']),
        peak_sampled_device_mib=max(r['resource_summary']['gpu_vram_used_mib']['max'] for r in run['runs']),
        warmup_s=run['warmup_s'],
        mean_stage_seconds_per_chunk={stage:sum(r['stage_seconds'].get(stage,0) for r in run['runs'])/
                                     sum(len(r['chunk_times_s']) for r in run['runs'])
                                     for stage in ('audio','dit','vae_decode','motion_encode')}))
result=dict(execution='CPU aggregation of fresh retained GPU measurements',
            gpu='NVIDIA GeForce RTX 4070 SUPER, physical 12 GB class / 12,282 MiB visible',
            driver='595.84',torch='2.7.1+cu128',cuda_runtime='12.8',rows=rows)
(ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
for row in rows:
    print(f"{row['name']}: {row['median_useful_fps']:.3f} FPS; {row['peak_torch_allocated_mib']:.0f} MiB Torch; {row['peak_sampled_device_mib']:.0f} MiB device")
