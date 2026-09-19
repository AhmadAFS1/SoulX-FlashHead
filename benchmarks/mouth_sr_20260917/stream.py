"""One-session GPU integration: Engine.generate -> mouth SR -> delivery frames."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
import numpy as np
import torch
from soulx_rtc.engine import Engine
from soulx_rtc.mouth_sr import SRVGGUpscaler, MouthEnhancer
from soulx_rtc.experiment import record
from soulx_rtc.server import decode_audio

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'stream-native'


def main():
    OUT.mkdir(exist_ok=False)
    result=dict(status='starting',execution='Fresh combined SoulX + native mouth SR GPU inference',
        date_utc=datetime.now(timezone.utc).isoformat(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv'],text=True),
        coresident=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True),
        physical_vram='12 GB class; visible capacity recorded by nvidia-smi',
        torch=torch.__version__,cuda_runtime=torch.version.cuda,
        profile='320x576, 25 FPS, 10 seconds, seed 50, steps 4, shift 5, history 2, strength 1; INT8/BF16 compact eager; 3584 MiB allocator cap; SR FP16 strength .65')
    def save():(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    ref='benchmarks/smile_reference_320_20260917/neutral-reference.png'
    engine=Engine(width=320,height=576,steps=4,fps=25,compile_model=False,
        optimized=True,real_rope=True,lean=True,fused_qkv=True,
        memory_mode='compact',int8_weights=True,cuda_memory_mib=3584)
    engine.warmup(ref)
    upscaler=SRVGGUpscaler('models/mouth-sr/realesr-general-x4v3.pth')
    upscaler.upscale(np.zeros((96,160,3),np.uint8),1)
    enhancer=MouthEnhancer(upscaler,scale=1,strength=.65)
    audio=decode_audio('benchmarks/comparison-10s.wav',10)
    state=engine.prepare_call(ref,50);engine.append(state,audio)
    raw=[];output=[];timings=[]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
    try:
        while state.cursor<state.total_frames:
            a=time.perf_counter();chunk=engine.generate([state])[0];b=time.perf_counter()
            # Delivery-only: engine has already prepared its next motion prefix.
            enhanced=enhancer.process_chunk(chunk);c=time.perf_counter()
            raw.append(chunk);output.append(enhanced)
            timings.append(dict(generation_s=b-a,enhancement_s=c-b,frames=len(chunk)))
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        raw=np.concatenate(raw)[:250];frames=np.concatenate(output)[:250]
        capture=json.loads((ROOT/'320-seed50/capture.json').read_text())
        original_hash=hashlib.sha256(raw.tobytes()).hexdigest()
        result.update(status='complete',wall_s=elapsed,useful_fps=250/elapsed,
            frames=250,raw_sha256=original_hash,
            source_matches_unenhanced_capture=original_hash==capture['raw_sha256'],
            output_sha256=hashlib.sha256(frames.tobytes()).hexdigest(),chunks=timings,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        assert result['source_matches_unenhanced_capture']
        record(OUT/'video.mp4',[frames],audio,25)
        print(json.dumps(result),flush=True)
    except Exception as exc:
        result.update(status='failed',error=repr(exc));raise
    finally:
        enhancer.close();save()


if __name__=='__main__':main()
