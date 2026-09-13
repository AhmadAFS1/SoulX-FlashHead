"""Recorded transport/obvious corruption checks, not a perceptual quality oracle."""
import argparse,json
from pathlib import Path
import av
import numpy as np

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('report',type=Path)
ap.add_argument('--peer',type=int,default=0)
ap.add_argument('--output',required=True,type=Path)
args=ap.parse_args()
if args.output.exists():raise ValueError('Fresh evidence required')
report=json.loads(args.report.read_text())
peer=next(p for p in report['peers'] if p['index']==args.peer)
fps=report['health_before']['fps']
intervals=[(t['start_frame']/fps,t['start_frame']/fps+t['useful_audio_samples']/48000)
           for t in peer['server']['turns'] if t['status']=='complete' and t['start_frame'] is not None]
video=args.report.with_name(args.report.stem+f'-peer{args.peer}.mp4')
times=[];jumps=[];black=0;longest=run=0;previous=None
with av.open(str(video)) as src:
    assert len(src.streams.audio)==len(src.streams.video)==1
    for frame in src.decode(video=0):
        rgb=frame.to_ndarray(format='rgb24');t=float(frame.pts*frame.time_base)
        assert [frame.width,frame.height]==[report['health_before']['width'],report['health_before']['height']]
        black += int(rgb.mean()<2)
        if previous is not None:
            error=np.abs(rgb.astype(np.int16)-previous.astype(np.int16))
            jumps.append(dict(pts=t,mad=float(error.mean())))
            speaking=any(a<=t<b for a,b in intervals)
            run=run+1 if speaking and error.max()==0 else 0
            longest=max(longest,run)
        previous=rgb;times.append(t)
gaps=np.diff(times)
result=dict(video=str(video),peer=args.peer,frames=len(times),
    pts_monotonic=bool(len(gaps) and (gaps>0).all()),
    presentation_gap_max_s=float(gaps.max()),
    arrival_gap_max_s=peer['arrival_gap_max_s'],arrival_gap_p95_s=peer['arrival_gap_p95_s'],
    wire_fps=peer['wire_fps'],black_frames=black,
    max_identical_speaking_run_frames=longest,
    largest_frame_changes=sorted(jumps,key=lambda r:r['mad'],reverse=True)[:12],
    transport_pass=peer['transport_pass'],underruns=peer['server']['underrun_frames'],
    missed_slots=peer['server']['missed_video_slots'],
    note='Checks obvious whole-frame blackouts/cuts and exact repeated frames during completed speech; inspect footage for identity, mouth and blend artifacts.')
result['structural_video_pass']=bool(result['pts_monotonic'] and not black and longest<=5
    and max((r['mad'] for r in jumps),default=0)<40)
result['strict_pacing_pass']=bool(peer['transport_pass'] and peer['wire_fps']>=24.5
    and peer['server']['underrun_frames']==0 and result['presentation_gap_max_s']<=.08001
    and peer['arrival_gap_max_s']<=.15)
args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
