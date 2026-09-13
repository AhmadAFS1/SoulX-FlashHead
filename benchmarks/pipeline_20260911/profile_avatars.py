"""One-owner matched profiling and independent color/audio compilation trials."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from soulx_rtc.engine import Engine
from soulx_rtc.experiment import record, memory_snapshot
from soulx_rtc.server import decode_audio
from flash_head.utils.utils import match_and_blend_colors_torch


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',required=True)
    ap.add_argument('--width',type=int,default=320)
    ap.add_argument('--height',type=int,default=576)
    ap.add_argument('--budget',type=int,default=3584)
    ap.add_argument('--steps',type=int,choices=[2,4],default=4)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--variants',nargs='+',default=['baseline','color','audio','both'])
    ap.add_argument('--avatars',nargs='+',default=['girl','man','yongen','idle'])
    args=ap.parse_args()
    out=Path(args.output)
    if out.exists():
        raise ValueError('Use fresh evidence')
    root=Path('/workspace/SoulX-FlashHead')
    paths=dict(girl=root/'examples/girl.png',man=Path('/workspace/MuseTalk/assets/demo/man/man.png'),
               yongen=Path('/workspace/MuseTalk/assets/demo/yongen/yongen.jpeg'),
               idle=root/'benchmarks/implementation/musetalk-idle-anchor.png')
    audio=decode_audio(str(root/'benchmarks/comparison-10s.wav'),10)
    report=dict(config=vars(args),rows=[],quality=[],preparations=[],warmup=[],
                inputs={k:dict(path=str(paths[k]),sha256=hashlib.sha256(paths[k].read_bytes()).hexdigest()) for k in args.avatars},
                audio_sha256=hashlib.sha256((root/'benchmarks/comparison-10s.wav').read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
                    ['soulx_rtc/engine.py','flash_head/src/pipeline/flash_head_pipeline.py','soulx_rtc/compact_weights.py']},
                timing='Useful 250 native frames; setup, warmup and video encoding excluded; profiled phase values are GPU event intervals')
    def save():
        out.write_text(json.dumps(report,indent=2))
    save()
    engine=Engine(width=args.width,height=args.height,steps=args.steps,fps=25,compile_model=True,
                  optimized=True,real_rope=True,memory_mode='compact',lean=True,fused_qkv=True,
                  int8_weights=True,cuda_memory_mib=args.budget,profile=True)
    original_audio=engine.pipeline.audio_encoder.forward
    compiled_audio=torch.compile(original_audio,fullgraph=True)
    compiled_color=torch.compile(match_and_blend_colors_torch,fullgraph=True)
    baseline={}
    def render(avatar):
        torch.cuda.synchronize()
        begin=time.perf_counter()
        state=engine.prepare(str(paths[avatar]),audio,50)
        state.terminal=True
        torch.cuda.synchronize()
        report['preparations'].append(dict(avatar=avatar,wall_ms=(time.perf_counter()-begin)*1000,
                                           **engine.last_preparation))
        torch.cuda.reset_peak_memory_stats()
        chunks=[]; metrics=[]
        begin=time.perf_counter()
        while state.cursor<state.total_frames:
            chunks.append(engine.generate([state])[0])
            metrics.append(engine.last_metrics.copy())
        elapsed=time.perf_counter()-begin
        return chunks,dict(wall_s=elapsed,fps=sum(len(c) for c in chunks)/elapsed,
                            first_chunk_ms=metrics[0]['wall_s']*1000,chunks=metrics,
                            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
    try:
        for variant in args.variants:
            engine.dit_graph=variant=='graph'
            engine.color_match=compiled_color if variant in ['color','both'] else match_and_blend_colors_torch
            engine.pipeline.audio_encoder.forward=compiled_audio if variant in ['audio','both'] else original_audio
            engine.compile_color=variant in ['color','both'];engine.compile_audio=variant in ['audio','both']
            start=time.perf_counter()
            # Exercise first and continuing motion, final short chunk and terminal path.
            render(args.avatars[0])
            report['warmup'].append(dict(variant=variant,seconds=time.perf_counter()-start))
            torch.cuda.empty_cache()
            save()
            for avatar in args.avatars:
                for repeat in range(args.repeats):
                    chunks,row=render(avatar)
                    row.update(variant=variant,avatar=avatar,repeat=repeat)
                    report['rows'].append(row)
                    save()
                    print(json.dumps({k:row[k] for k in ['variant','avatar','repeat','fps','first_chunk_ms','peak_allocated_mib']}),flush=True)
                    if repeat==0:
                        rgb=np.concatenate(chunks)
                        if variant=='baseline':
                            baseline[avatar]=rgb
                        if avatar in baseline:
                            error=rgb.astype(np.float32)-baseline[avatar].astype(np.float32)
                            mse=float(np.mean(error**2));mae=float(np.mean(np.abs(error)))
                            report['quality'].append(dict(variant=variant,avatar=avatar,mae=mae,
                                psnr_db=10*math.log10(255**2/mse) if mse else None,
                                max_error=float(np.abs(error).max())))
                        if variant in ['baseline','both']:
                            record(out.with_name(out.stem+f'-{variant}-{avatar}.mp4'),chunks,audio,25)
                        save()
                report.setdefault('memory',[]).append(dict(variant=variant,avatar=avatar,**memory_snapshot(torch)))
        report['status']='completed'
    except Exception as exc:
        report.update(status='failed',failure=dict(type=type(exc).__name__,message=str(exc)))
        raise
    finally:
        save()

if __name__=='__main__':
    main()
