"""Exact cached/legacy RGB comparison at delivery size, including loop/rewind."""
import argparse,json
from pathlib import Path
import numpy as np
from soulx_rtc.idle_cache import IdleCache
from soulx_rtc.calls import IdleVideo

ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output',required=True,type=Path)
ap.add_argument('--video',default='/workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/idle_active_listening.mp4')
args=ap.parse_args()
if args.output.exists():raise ValueError('Fresh evidence required')
cache=IdleCache();clip=cache.acquire(args.video,320,576)
assert clip is not None
old=IdleVideo(args.video,320,576,25,clip.frames[0])
indices=list(range(300))+[0,5,255]
maximum=0
try:
    for i in indices:
        a,b=old.next(i),clip.at(i,25)
        maximum=max(maximum,int(np.abs(a.astype(np.int16)-b.astype(np.int16)).max()))
finally:old.close();cache.release(clip)
result=dict(compared_frames=len(indices),source_frames=len(clip.frames),max_rgb_error=maximum,
            cache=cache.stats(),passed=maximum==0)
args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
raise SystemExit(0 if result['passed'] else 1)
