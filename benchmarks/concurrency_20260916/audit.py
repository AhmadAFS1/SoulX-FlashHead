"""Reconcile retained capacity evidence. CPU/file audit only; no service calls."""
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MUSE = ROOT.parent / 'MuseTalk'


def source(path):
    repo = ROOT if path.is_relative_to(ROOT) else MUSE
    return dict(repo=repo.name, path=str(path.relative_to(repo)),
        revision=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'],text=True).strip(),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    engine=[]
    for name in ('engine-512-4-b1.json','engine-512-4-b2.json'):
        p=ROOT/'benchmarks'/name
        for r in json.loads(p.read_text()):
            engine.append(dict(source=source(p),gpu='RTX 4070, 12282 MiB; historical report attribution',
                **{k:r[k] for k in ('sessions','batch','size','steps','generated_frames','wall_s','aggregate_fps')}))
    for name in ('portrait-ten-jobs.json','portrait-trt-ten-jobs.json','square-trt-combined-reference.json'):
        p=ROOT/'benchmarks/implementation'/name
        d=json.loads(p.read_text())
        engine.append(dict(source=source(p),gpu='RTX 4070, 12282 MiB; historical report attribution',
            config=d['config'],summary=d['summary']))
    p=ROOT/'benchmarks/distance_lipsync/evidence-indian-male-closer-strength-0p5-20260916/results.json'
    d=json.loads(p.read_text())
    current=dict(source=source(p),gpu=d['gpu'],profile=d['profile'],rows=d['rows'])
    calls=[]
    for label in ('c1','c2','c4','c6','c10','mixed10'):
        p=ROOT/f'benchmarks/implementation/calls-h264-trt-{label}.json'
        d=json.loads(p.read_text())
        calls.append(dict(source=source(p),label=label,gpu='RTX 4070, 12282 MiB; historical report attribution',
            **{k:d.get(k) for k in ('config','wall_s','completed_turns','all_transport_pass','total_underrun_frames','cleanup_pass')}))
    muse=[]
    for name,gpu in (
        ('load_test_webrtc_v100_baseline_pytorch_20_20_batch4_ramp1_6.json','Tesla V100-SXM2-32GB, 32768 MB reported'),
        ('load_test_webrtc_report_v100_20_20_4_5_6_8streams_4_8_16_libx264_20260523.json','Tesla V100-SXM2-32GB, 32768 MB reported'),
        ('load_test_webrtc_4090_gpt_moving_avatar_20_20_4_5_6_8streams_8_12_libx264_20260523.json','RTX 4090, 24564 MiB')):
        p=MUSE/name
        rows=json.loads(p.read_text())
        for r in rows:
            interval=r['avg_segment_interval_s']
            # Do not assign full-stage capacity from survivor-only means.
            r['derived_aggregate_fps_proxy']=(r['concurrency']/interval
                if interval>0 and r['failed']==0 else None)
        muse.append(dict(source=source(p),gpu=gpu,rows=rows))
    docs=[]
    for repo,name,start,end in (
        (MUSE,'current_cross_server_throughput_findings.md',502,525),
        (MUSE,'current_unet_trt_throughput_findings_2026-05-29.md',1,36),
        (MUSE,'current_unet_trt_throughput_findings_2026-05-29.md',98,121),
        (MUSE,'docs/webrtc_load_test_findings_2026-06-07.md',1,50),
        (MUSE,'load_test_webrtc_rtx6000ada_int8_trt_unet_split8_20fps_20260608.md',1,59),
        (MUSE,'docs/webrtc_generation_optimization_results_2026-07-03.md',30,80),
        (ROOT,'docs/research/PIPELINE_PROFILE_2026-09-11.md',26,46),
        (ROOT,'docs/research/PIPELINE_PROFILE_2026-09-11.md',127,190)):
        p=repo/name
        docs.append(dict(source=source(p),start_line=start,end_line=end,
            excerpt='\n'.join(p.read_text().splitlines()[start-1:end])))
    fps=next(r['useful_fps'] for r in current['rows'] if r['label']=='closer-125')
    scenarios=[]
    for n in (1,2,3,4,6,8,10):
        for target in (20,25):
            demand=n*target
            scenarios.append(dict(active_speakers=n,target_fps=target,
                required_fps_no_headroom=demand,required_fps_at_80pct_utilization=demand/.8,
                speedup_over_current_no_headroom=demand/fps,
                speedup_over_current_with_headroom=demand/.8/fps))
    unavailable=[ROOT.parent/'experiments/flashhead-pipeline-ARsFTh/profile-320.json',
        MUSE/'tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_trt_unet_split8_20_20_4_8_12_16streams_8_16_libx264_20260608.json',
        MUSE/'tmp/load_tests/load_test_webrtc_3090_int8_5stage_trt_unet_split8_20_20_4_6_8streams_8_16_buckets_batch8_300w_20260529.json']
    result=dict(date_utc=datetime.now(timezone.utc).isoformat(),execution='CPU/file audit; NO new GPU throughput or WebRTC load test',
        methodology='Useful generated FPS differs from wire FPS. MuseTalk C / mean interval is an approximate cadence proxy, not exact sum of per-stream FPS. Sources have different GPUs, avatars, geometry, precision, pacing and timing boundaries; no matched cross-model ranking.',
        current_soulx=current,historical_soulx_engine=engine,historical_soulx_calls=calls,
        musetalk_raw_summaries=muse,documentary_evidence=docs,
        source_availability=[dict(path=str(p),exists=p.exists()) for p in unavailable],
        illustrative_turn_taking=dict(connected=10,speaking_fraction=.3,
            expected_active=10*.3,
            probability_two_or_more_if_independent=1-sum(math.comb(10,k)*.3**k*.7**(10-k) for k in (0,1)),
            warning='Independent binomial illustration only; no user traffic data or correlation model was measured.'),
        illustrative_sizing=dict(baseline_fps=fps,utilization_budget=.8,scenarios=scenarios,
            warning='Arithmetic planning scenarios only. Not a concurrency validation, extrapolation to another GPU, or SLA.'))
    out=Path(__file__).with_name('evidence.json')
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(out)
    print('Historical records and sizing arithmetic captured; no inference run.')


if __name__ == '__main__':
    main()
