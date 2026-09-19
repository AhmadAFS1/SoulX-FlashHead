"""Isolated square-resolution experiment, including opt-in 1024-square override."""
import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import soulx_rtc.engine as engine_module
from soulx_rtc.experiment import record
from soulx_rtc.server import decode_audio

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--size',type=int,choices=[512,1024],required=True)
    ap.add_argument('--fps',type=int,choices=[15,25],required=True)
    ap.add_argument('--output',type=Path,required=True)
    a=ap.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
    result={'status':'starting','date_utc':datetime.now(timezone.utc).isoformat(),
      'gpu_snapshot':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv'],text=True),
      'torch':torch.__version__,'cuda':torch.version.cuda,
      'profile':{'size':a.size,'fps':a.fps,'steps':4,'seed':50,'precision':'BF16','memory_mode':'staged','compiled':False,'conditioning':'stock','seconds':10,'allocator_cap_mib':8704},
      'source':'/workspace/benchmarks/same-avatar/shared.png','source_crop_xyxy':[38,64,345,371],
      'note':'Same square crop at both resolutions. Process-local geometry override only; no serving cap changes. OmniVoice remains resident.'}
    def save(): (a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    try:
        src=Path(result['source']); result['source_sha256']=hashlib.sha256(src.read_bytes()).hexdigest()
        ref=a.output/'reference.png'
        Image.open(src).convert('RGB').crop(tuple(result['source_crop_xyxy'])).resize((a.size,a.size),Image.Resampling.LANCZOS).save(ref)
        original=engine_module.validate_geometry
        def geometry(size,width=None,height=None):
            if (width,height)==(1024,1024): return width,height
            return original(size,width,height)
        engine_module.validate_geometry=geometry
        engine=engine_module.Engine(width=a.size,height=a.size,steps=4,fps=a.fps,compile_model=False,
          optimized=True,real_rope=True,lean=True,fused_qkv=True,memory_mode='staged',cuda_memory_mib=8704)
        result['status']='warmup';save(); engine.warmup(str(ref))
        audio=decode_audio('benchmarks/comparison-10s.wav',10)
        state=engine.prepare_call(str(ref),50);engine.append(state,audio)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
        result['status']='generating';save(); start=time.perf_counter();chunks=[]
        while state.cursor<state.total_frames:
            chunks.append(engine.generate([state])[0])
            print(json.dumps({'size':a.size,'fps':a.fps,'frames':state.cursor,'total':state.total_frames}),flush=True)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-start
        frames=np.concatenate(chunks)[:10*a.fps]
        result.update(wall_s=elapsed,useful_fps=len(frames)/elapsed,frames=len(frames),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,raw_sha256=hashlib.sha256(frames.tobytes()).hexdigest())
        record(a.output/'video.mp4',[frames],audio,a.fps)
        # Save native lossless RGB frames at fixed utterance times for dental inspection.
        for t in [0.5,1.5,2.5,3.5,4.5,6.5,8.5]:
            i=round(t*a.fps);Image.fromarray(frames[i]).save(a.output/f'frame-{t:.1f}s.png')
        result['status']='complete';save()
    except Exception as exc:
        result.update(status='failed',error_type=type(exc).__name__,error=str(exc));save();raise

if __name__=='__main__':main()
