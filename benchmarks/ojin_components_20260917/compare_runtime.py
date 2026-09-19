"""Isolate TensorRT versus PyTorch using the same native-2x weights/inputs."""
from datetime import datetime,timezone
import json
from pathlib import Path
import time
import cv2
import numpy as np
import torch
import tensorrt as trt
from soulx_rtc.srvgg_trt import load_native2x,Native2xTRT,digest
from soulx_rtc.gpu_lease import acquire_gpu_lease
from benchmarks.roi_hybrid_analysis_20260917.benchmark import snapshot,processes

ROOT=Path(__file__).resolve().parent/'matched-runtime'


@torch.inference_mode()
def main():
    assert not (ROOT/'same-weights-runtime.json').exists()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    source=ROOT/'soulx-LITE-lite-only-512-to-512-seed50/raw.npy'
    frames=np.load(source,mmap_mode='r')[::10].copy()
    result=dict(date_utc=datetime.now(timezone.utc).isoformat(),gpu=snapshot(),processes=processes(),
        physical_vram='12 GB class; visible capacity in GPU snapshot',torch=torch.__version__,
        cuda=torch.version.cuda,tensorrt=trt.__version__,input_sha256=digest(source),
        execution='Standalone GPU SR with H2D/D2H and quantization; no Lite, tracking or encoding in timing',runs=[])
    with acquire_gpu_lease():
        model=load_native2x('models/ojin-components/2xNomosUni_compact_multijpg_ldl.safetensors').cuda().half()
        engine=Native2xTRT('models/ojin-components/native2x-srvgg-512-trt10.16-cu13.engine')
        def pytorch(rgb):
            x=torch.from_numpy(rgb).permute(2,0,1)[None].cuda().half()/255
            output=model(x).float().clamp(0,1)[0].permute(1,2,0).cpu().numpy()
            return (output*255).round().clip(0,255).astype(np.uint8)
        for rgb in frames[:4]:pytorch(rgb);engine.upscale(rgb)
        for repeat in range(3):
            order=[('pytorch-fp16',pytorch),('tensorrt-fp16',engine.upscale)]
            if repeat%2:order.reverse()
            for name,fn in order:
                torch.cuda.synchronize();start=time.perf_counter()
                for rgb in frames:fn(rgb)
                torch.cuda.synchronize();elapsed=time.perf_counter()-start
                result['runs'].append(dict(repeat=repeat,mode=name,frames=len(frames),seconds=elapsed,fps=len(frames)/elapsed))
        errors=[]
        for rgb in frames:
            a=pytorch(rgb).astype(np.float32);b=engine.upscale(rgb).astype(np.float32)
            err=np.abs(a-b)
            errors.append(dict(mae_8bit=float(err.mean()),max_8bit=float(err.max())))
        result.update(status='complete',pixel_errors=errors)
        (ROOT/'same-weights-runtime.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
