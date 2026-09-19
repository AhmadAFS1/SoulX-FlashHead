"""Fresh Lite512 -> Tasks FaceLandmarker -> native2x TRT1024, refiner omitted.

Public substitute SR weights and local orchestration. NOT Ojin service parity.
"""
from collections import deque
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
import subprocess
import psutil
import cv2
import librosa
import mediapipe as mp
import numpy as np
from PIL import Image
import torch
import tensorrt as trt

from flash_head.src.pipeline import flash_head_pipeline as pm
from flash_head.inference import get_audio_embedding, run_pipeline
from soulx_rtc.face_landmarker import FaceLandmarkerTracker
from soulx_rtc.srvgg_trt import Native2xTRT, digest
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.experiment import record
from benchmarks.roi_hybrid_analysis_20260917.benchmark import snapshot, processes

ROOT=Path(__file__).resolve().parent
SOURCE=Path('benchmarks/square_teeth_20260917/512-25/reference.png')
AUDIO=Path('benchmarks/comparison-10s.wav')
TASK=Path('models/ojin-components/face_landmarker.task')
ENGINE=Path('models/ojin-components/native2x-srvgg-512-fp16.engine')
TIMES=(.5,1.5,2.5,3.5,4.5,6.5,8.5)


class Monitor:
    """Jittered sampling avoids the previous fixed-half-second phase lock."""
    def __enter__(self):
        self.stop=threading.Event();self.samples=[];self.started=time.perf_counter()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();return self
    def run(self):
        process=psutil.Process();process.cpu_percent(None);psutil.cpu_percent(None)
        while not self.stop.wait(.137 if len(self.samples)%2 else .193):
            row=dict(elapsed_s=time.perf_counter()-self.started,process_rss_mib=process.memory_info().rss/2**20,
                process_cpu_percent=process.cpu_percent(None),host_cpu_percent=psutil.cpu_percent(None),
                host_ram_used_mib=psutil.virtual_memory().used/2**20)
            try:
                values=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'],text=True,timeout=3).strip().split(',')
                row.update(gpu_percent=float(values[0]),device_vram_mib=float(values[1]))
            except Exception as e:row['error']=repr(e)
            self.samples.append(row)
    def __exit__(self,*_):self.stop.set();self.thread.join()


@torch.inference_mode()
def main(root=ROOT, engine=ENGINE):
    root=Path(root).resolve();root.mkdir(parents=True,exist_ok=True)
    assert not (root/'results.json').exists(),'Preserve previous evidence'
    torch.set_num_threads(4);cv2.setNumThreads(1)
    audio,_=librosa.load(AUDIO,sr=16000,mono=True)
    chunks=np.pad(audio,(0,11*15360-len(audio))).reshape(-1,15360)
    result=dict(status='starting',date_utc=datetime.now(timezone.utc).isoformat(),gpu=snapshot(),
        physical_vram='12 GB class; nvidia-smi visible capacity in gpu field',
        processes=processes(),torch=torch.__version__,cuda=torch.version.cuda,tensorrt=trt.__version__,
        mediapipe=mp.__version__,numpy=np.__version__,
        profile=dict(steps=4,shift=5,history=2,color_correction=1,fps=25,frames=250,
            dtype='BF16',compile=False,source=str(SOURCE),source_sha256=digest(SOURCE),
            audio=str(AUDIO),audio_sha256=digest(AUDIO),task_sha256=digest(TASK),engine_sha256=digest(engine),
            generation=[512,512],delivery=[1024,1024],
            refiner='omitted, unavailable',
            upscaler='native 2x 16-conv SRVGG FP16 TensorRT; public NomosUni substitute, NOT Ojin weights',
            tracker='Tasks FaceLandmarker, float16/1 task, VIDEO mode, CPU delegate',
            tracker_note='Mouth coordinates recorded; no mouth refinement applied without its weights'),runs=[])
    def save():(root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    with acquire_gpu_lease():
        try:
            pm.COMPILE_MODEL=pm.COMPILE_VAE=pm.USE_PARALLEL_VAE=False
            pipeline=pm.FlashHeadPipeline('models/SoulX-FlashHead-1_3B','lite',
                'models/wav2vec2-base-960h',param_dtype=torch.bfloat16)
            upscaler=Native2xTRT(engine)
            def prepare(seed):
                torch.manual_seed(seed)
                pipeline.prepare_params(str(SOURCE),(512,512),33,9,4,seed=seed,shift=5,
                    color_correction_strength=1,use_face_crop=False)
            prepare(50)
            embedding=get_audio_embedding(pipeline,np.zeros(128000,np.float32),167,200)
            warm=run_pipeline(pipeline,embedding)[9:].cpu().numpy().astype(np.uint8)
            for frame in warm[:4]:upscaler.upscale(frame)
            tracker=FaceLandmarkerTracker(TASK)
            for frame in warm:tracker(frame)
            tracker.close();del warm;torch.cuda.synchronize();torch.cuda.empty_cache()
            for seed in (50,51):
                for mode in ('lite-only','facelandmarker-full-native2x-TRT'):
                    prepare(seed);tracker=FaceLandmarkerTracker(TASK) if mode!='lite-only' else None
                    cache=deque([0.]*128000,maxlen=128000)
                    raw_chunks=[];output_chunks=[];tracking=[];chunk_times=[]
                    stages=dict(generation=0.,face_landmarker=0.,upscale=0.)
                    torch.cuda.reset_peak_memory_stats()
                    with Monitor() as monitor:
                        start=time.perf_counter();count=0
                        for segment in chunks:
                            chunk_start=time.perf_counter();t=chunk_start;cache.extend(segment.tolist())
                            embedding=get_audio_embedding(pipeline,np.asarray(cache,np.float32),167,200)
                            raw=run_pipeline(pipeline,embedding)[9:].cpu().numpy().astype(np.uint8)
                            torch.cuda.synchronize();stages['generation']+=time.perf_counter()-t
                            raw=raw[:250-count];count+=len(raw);raw_chunks.append(raw)
                            if tracker:
                                for frame in raw:
                                    t=time.perf_counter();points=tracker(frame);stages['face_landmarker']+=time.perf_counter()-t
                                    tracking.append(None if points is None else points.tolist())
                                    t=time.perf_counter();out=upscaler.upscale(frame);torch.cuda.synchronize()
                                    stages['upscale']+=time.perf_counter()-t;output_chunks.append(out)
                            chunk_times.append(time.perf_counter()-chunk_start)
                        elapsed=time.perf_counter()-start
                    row=dict(seed=seed,mode=mode,frames=count,wall_s=elapsed,fps=count/elapsed,
                        stages_s=stages,chunk_s=chunk_times,samples=monitor.samples,
                        torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                        torch_memory_note='Excludes TensorRT-owned memory; device samples include it',
                        gpu_after=snapshot(),processes_after=processes())
                    raw=np.concatenate(raw_chunks);row['raw_sha256']=hashlib.sha256(raw.tobytes()).hexdigest()
                    name=f'soulx-LITE-{mode}-512-to-{1024 if tracker else 512}-seed{seed}'
                    folder=root/name;folder.mkdir(exist_ok=False)
                    if tracker:
                        frames=np.stack(output_chunks);tracker.close()
                        row['detected']=sum(p is not None for p in tracking)
                        (folder/'mouth-landmarks.json').write_text(json.dumps(tracking)+'\n')
                    else:
                        frames=raw;np.save(folder/'raw.npy',raw)
                    row['output_sha256']=hashlib.sha256(frames.tobytes()).hexdigest()
                    path=folder/(name+'.mp4');record(path,[frames],audio,25)
                    row['video']=str(path.relative_to(root));row['video_sha256']=digest(path)
                    for t in TIMES:Image.fromarray(frames[round(t*25)]).save(folder/f'frame-{t:.1f}s.png')
                    result['runs'].append(row);save()
                    print(json.dumps({k:v for k,v in row.items() if k in ['mode','seed','fps','wall_s','stages_s','detected']}),flush=True)
                    del frames,raw,raw_chunks,output_chunks
            result['raw_parity_per_seed']={str(seed):len({x['raw_sha256'] for x in result['runs'] if x['seed']==seed})==1 for seed in (50,51)}
            assert all(result['raw_parity_per_seed'].values())
            result['status']='complete';save()
        except Exception as e:result.update(status='failed',error=repr(e));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--engine',type=Path,default=ENGINE)
    args=parser.parse_args();main(args.output,args.engine)
