"""Matched VAE convolution candidates and one warmed operator profile."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from soulx_rtc.engine import Engine
from soulx_rtc.server import decode_audio
from soulx_rtc.experiment import record
from flash_head.utils.utils import match_and_blend_colors_torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--width',type=int,default=320)
    ap.add_argument('--height',type=int,default=576)
    ap.add_argument('--budget',type=int,default=3584)
    ap.add_argument('--variants',nargs='+',default=['baseline','precise_color','cudnn_autotune','channels_last','dit_graph'])
    ap.add_argument('--reset-for-layout',action='store_true',help='Invalidate compiled assumptions before changing weight strides')
    args=ap.parse_args()
    out=Path(args.output)
    if out.exists(): raise ValueError('Fresh evidence path required')
    image='/workspace/SoulX-FlashHead/examples/girl.png'
    audio=decode_audio('/workspace/SoulX-FlashHead/benchmarks/comparison-10s.wav',10)
    result=dict(config=vars(args),rows=[],quality=[],warmup=[],
                image_sha256=hashlib.sha256(Path(image).read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ['soulx_rtc/engine.py','flash_head/src/pipeline/flash_head_pipeline.py','soulx_rtc/compact_weights.py']})
    def save(): out.write_text(json.dumps(result,indent=2))
    engine=Engine(width=args.width,height=args.height,steps=4,optimized=True,real_rope=True,
                  memory_mode='compact',lean=True,fused_qkv=True,int8_weights=True,
                  cuda_memory_mib=args.budget,profile=True)
    def render():
        state=engine.prepare(image,audio,50);state.terminal=True
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        chunks=[];metrics=[];start=time.perf_counter()
        while state.cursor<state.total_frames:
            chunks.append(engine.generate([state])[0]);metrics.append(engine.last_metrics.copy())
        wall=time.perf_counter()-start
        return np.concatenate(chunks),dict(fps=250/wall,wall_s=wall,chunks=metrics,
                                            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    baseline=None
    try:
        for variant in args.variants:
            engine.color_match=match_and_blend_colors_torch
            engine.dit_graph=False
            if variant=='precise_color':
                engine.color_match=torch.compile(match_and_blend_colors_torch,fullgraph=True,
                    options={'emulate_precision_casts':True})
            if variant=='cudnn_autotune':
                torch.backends.cudnn.benchmark=True
                torch.backends.cudnn.benchmark_limit=10
            if variant=='channels_last':
                if args.reset_for_layout:
                    torch._dynamo.reset()
                engine.pipeline.vae.model.to(memory_format=torch.channels_last_3d)
            if variant=='dit_graph':
                torch.backends.cudnn.benchmark=False
                engine.pipeline.vae.model.to(memory_format=torch.contiguous_format)
                engine.dit_graph=True
            start=time.perf_counter();render()
            result['warmup'].append(dict(variant=variant,seconds=time.perf_counter()-start))
            for repeat in range(3):
                rgb,row=render();row.update(variant=variant,repeat=repeat)
                result['rows'].append(row);save()
                print(json.dumps({k:v for k,v in row.items() if k!='chunks'}),flush=True)
                if repeat==0 or variant=='baseline':
                    if baseline is None: baseline=rgb
                    error=rgb.astype(np.float32)-baseline.astype(np.float32)
                    mse=float((error**2).mean())
                    result['quality'].append(dict(variant=variant,repeat=repeat,mae=float(np.abs(error).mean()),
                        psnr_db=10*math.log10(255**2/mse) if mse else None,
                        max_error=float(np.abs(error).max())))
                    if repeat==0:
                        record(out.with_name(out.stem+'-'+variant+'.mp4'),[rgb],audio,25)
                    save()
            if variant=='baseline':
                state=engine.prepare(image,audio,50)
                engine.generate([state])
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.CUDA]) as prof:
                    engine.generate([state])
                (out.with_suffix('.operators.txt')).write_text(prof.key_averages().table(
                    sort_by='self_cuda_time_total',row_limit=40))
                prof.export_chrome_trace(str(out.with_suffix('.trace.json.gz')))
        result['status']='completed'
    except Exception as exc:
        result.update(status='failed',failure=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally: save()

if __name__=='__main__': main()
