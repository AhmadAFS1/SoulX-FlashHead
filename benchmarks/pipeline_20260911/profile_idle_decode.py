"""CPU-only idle decoding controls; no GPU model or service mutation."""
import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path
import av
import numpy as np
from soulx_rtc.calls import IdleVideo

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',required=True,type=Path)
    ap.add_argument('--video',default='/workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/idle_active_listening.mp4')
    args=ap.parse_args()
    if args.output.exists(): raise ValueError('Fresh evidence required')
    result=dict(video=args.video,sha256=hashlib.sha256(Path(args.video).read_bytes()).hexdigest(),
                geometry=[320,576],decoder_threads=[])
    with av.open(args.video) as src:
        anchor=next(src.decode(video=0)).reformat(width=320,height=576,format='rgb24').to_ndarray()
    for count in [0,1,2,4]:
        idle=IdleVideo(args.video,320,576,25,anchor)
        idle.container=av.open(args.video)
        stream=idle.container.streams.video[0];stream.thread_count=count
        idle.source_fps=float(stream.average_rate);idle.frames=idle.container.decode(video=0)
        times=[]
        try:
            for i in range(125):
                started=time.perf_counter();idle.next(i)
                elapsed=(time.perf_counter()-started)*1000
                if i>=5:times.append(elapsed)
        finally: idle.close()
        result['decoder_threads'].append(dict(threads=count,frames=len(times),
            mean_ms=statistics.mean(times),median_ms=statistics.median(times),max_ms=max(times)))
    times=[];exact=[]
    with av.open(args.video) as src:
        frames=src.decode(video=0);reformatter=av.video.reformatter.VideoReformatter()
        for i in range(60):
            a=time.perf_counter();frame=next(frames);b=time.perf_counter()
            rgb=frame.reformat(width=320,height=576,format='rgb24').to_ndarray();c=time.perf_counter()
            candidate=reformatter.reformat(frame,width=320,height=576,format='rgb24').to_ndarray();d=time.perf_counter()
            if i>=5:times.append([(b-a)*1000,(c-b)*1000,(d-c)*1000])
            exact.append(np.array_equal(rgb,candidate))
    result['split']=dict(decode_mean_ms=statistics.mean(t[0] for t in times),
        frame_reformat_mean_ms=statistics.mean(t[1] for t in times),
        cached_reformatter_mean_ms=statistics.mean(t[2] for t in times),
        exact_frames=sum(exact),compared_frames=len(exact))
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
