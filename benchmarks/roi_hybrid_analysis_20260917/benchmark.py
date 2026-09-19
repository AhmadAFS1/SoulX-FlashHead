"""Diagnostic only: time live SoulX frames plus uncached MuseTalk face processing.

No service integration. FP16 MuseTalk 1.5 and BF16 SoulX Lite share one GPU.
FaceMesh/geometric blending is an experimental adapter, not the MuseTalk
production DWPose/face-parsing pipeline. No quality or streaming claim.
"""
import argparse
from collections import deque
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import cv2
import librosa
import mediapipe as mp
import numpy as np
import psutil
import torch

ROOT = Path(__file__).resolve().parents[2]
MUSE = Path('/workspace/MuseTalk')
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(MUSE))
from soulx_rtc.gpu_lease import acquire_gpu_lease
from flash_head.src.pipeline import flash_head_pipeline as pm
from flash_head.inference import get_audio_embedding, run_pipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from musetalk.models.unet import PositionalEncoding
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel


def snapshot():
    return subprocess.check_output(['nvidia-smi',
        '--query-gpu=name,memory.total,memory.used,driver_version,utilization.gpu',
        '--format=csv,noheader'], text=True).strip()


def processes():
    return subprocess.check_output(['nvidia-smi',
        '--query-compute-apps=pid,process_name,used_memory',
        '--format=csv,noheader'], text=True).strip()


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8*1024**2), b''): h.update(b)
    return h.hexdigest()


@contextmanager
def timed(stages, name):
    torch.cuda.synchronize()
    start = time.perf_counter()
    yield
    torch.cuda.synchronize()
    stages[name] = stages.get(name, 0) + time.perf_counter() - start


class Monitor:
    def __init__(self):
        self.stop = threading.Event()
        self.samples = []

    def __enter__(self):
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        return self

    def run(self):
        p = psutil.Process()
        p.cpu_percent(None); psutil.cpu_percent(None)
        while not self.stop.wait(.5):
            row = dict(elapsed_s=time.perf_counter()-self.started,
                process_rss_mib=p.memory_info().rss/2**20,
                process_cpu_percent=p.cpu_percent(None),
                host_cpu_percent=psutil.cpu_percent(None),
                host_ram_used_mib=psutil.virtual_memory().used/2**20)
            try:
                vals = subprocess.check_output(['nvidia-smi',
                    '--query-gpu=utilization.gpu,memory.used',
                    '--format=csv,noheader,nounits'], text=True, timeout=3).strip().split(',')
                row.update(gpu_percent=float(vals[0]), device_vram_mib=float(vals[1]))
            except Exception as e: row['error'] = str(e)
            self.samples.append(row)

    def __exit__(self, *_):
        self.stop.set(); self.thread.join()


class MouthPass:
    def __init__(self):
        self.vae = AutoencoderKL.from_pretrained(str(MUSE/'models/sd-vae'),
            torch_dtype=torch.float16, local_files_only=True).eval().requires_grad_(False).cuda()
        config = json.loads((MUSE/'models/musetalkV15/musetalk.json').read_text())
        self.net = UNet2DConditionModel(**config)
        self.net.load_state_dict(torch.load(MUSE/'models/musetalkV15/unet.pth',
            map_location='cpu', weights_only=True), strict=True)
        self.net = self.net.eval().requires_grad_(False).half().cuda()
        self.pe = PositionalEncoding().half().cuda().eval()
        self.whisper = WhisperModel.from_pretrained(str(MUSE/'models/whisper'),
            torch_dtype=torch.float16, local_files_only=True).eval().requires_grad_(False).cuda()
        self.processor = AudioProcessor(str(MUSE/'models/whisper'))
        self.generator = torch.Generator(device='cuda').manual_seed(123)
        self.timestep = torch.tensor([0], device='cuda')

    def audio(self, path):
        features, n = self.processor.get_audio_feature(str(path), weight_dtype=torch.float16)
        return self.processor.get_whisper_chunk(features, 'cuda', torch.float16, self.whisper, n, fps=25)

    def reset(self):
        self.mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=.5, min_tracking_confidence=.5)
        self.generator.manual_seed(123)
        self.last_box = None
        self.detected = 0

    def crop(self, rgb):
        result = self.mesh.process(rgb)
        if not result.multi_face_landmarks:
            raise RuntimeError('Face tracking failed; do not hide a skipped workload in throughput')
        self.detected += 1
        pts = np.array([(p.x*rgb.shape[1], p.y*rgb.shape[0])
            for p in result.multi_face_landmarks[0].landmark])
        oval = pts[list(set(i for pair in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in pair))]
        lo, hi = oval.min(0), oval.max(0)
        center = (lo+hi)/2
        side = max(hi-lo)*1.12
        box = np.array([*(center-side/2), *(center+side/2)])
        if self.last_box is not None: box = .65*box + .35*self.last_box
        self.last_box = box
        x0,y0,x1,y1 = np.round(box).astype(int)
        x0,y0 = max(0,x0),max(0,y0)
        x1,y1 = min(rgb.shape[1],x1),min(rgb.shape[0],y1)
        return cv2.resize(rgb[y0:y1,x0:x1], (256,256), interpolation=cv2.INTER_LANCZOS4), (x0,y0,x1,y1)

    def apply(self, frames, audio, batch, stages):
        outputs = []
        for start in range(0,len(frames),batch):
            subset = frames[start:start+batch]
            with timed(stages, 'track_crop_upload'):
                crops_boxes = [self.crop(rgb) for rgb in subset]
                crops = np.stack([x[0] for x in crops_boxes])
                pixels = torch.from_numpy(crops).permute(0,3,1,2).cuda().half()/127.5-1
                masked = pixels.clone(); masked[:,:,128:] = -1
            with timed(stages, 'new_face_vae_encode'):
                zmask = self.vae.encode(masked).latent_dist.sample(generator=self.generator)*self.vae.config.scaling_factor
                zfull = self.vae.encode(pixels).latent_dist.sample(generator=self.generator)*self.vae.config.scaling_factor
                z = torch.cat([zmask,zfull],1)
            with timed(stages, 'musetalk_unet'):
                cond = self.pe(audio[start:start+len(subset)].cuda().half())
                pred = self.net(z,self.timestep,encoder_hidden_states=cond).sample
            with timed(stages, 'musetalk_decode_transfer'):
                decoded = self.vae.decode(pred/self.vae.config.scaling_factor).sample
                restored = ((decoded/2+.5).clamp(0,1)*255).round().byte().permute(0,2,3,1).cpu().numpy()
            with timed(stages, 'blend'):
                for rgb, patch, (_, box) in zip(subset, restored, crops_boxes):
                    x0,y0,x1,y1 = box; h,w=y1-y0,x1-x0
                    patch = cv2.resize(patch,(w,h),interpolation=cv2.INTER_LINEAR)
                    mask = np.zeros((h,w),np.float32)
                    cv2.ellipse(mask,(w//2,int(h*.72)),(max(1,int(w*.43)),max(1,int(h*.25))),0,0,360,1,-1)
                    mask = cv2.GaussianBlur(mask,(0,0),max(1,w*.025))[:,:,None]
                    out=rgb.copy()
                    out[y0:y1,x0:x1]=(rgb[y0:y1,x0:x1]*(1-mask)+patch*mask).round().clip(0,255).astype(np.uint8)
                    outputs.append(out)
        return np.stack(outputs)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--face-batches',type=int,nargs='+',default=[8,4])
    parser.add_argument('--record',action='store_true',help='Retain the first measured video per mode outside timing')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    os.chdir(ROOT); lease=acquire_gpu_lease()
    torch.set_num_threads(4); cv2.setNumThreads(1)
    reference=ROOT/'benchmarks/pro_lite_150x_20260917/reference-150x.png'
    audio_path=ROOT/'benchmarks/comparison-10s.wav'
    data=dict(status='starting',date_utc=datetime.now(timezone.utc).isoformat(),
        gpu_before=snapshot(),processes_before=processes(),torch=torch.__version__,cuda=torch.version.cuda,
        execution='diagnostic fresh inference; serial stages on one GPU; no service changes',
        profile=dict(width=320,height=576,fps=25,frames=250,seed=50,steps=4,
            soulx='BF16 eager official pipeline',musetalk='1.5 FP16 eager',face=256,
            tracking='CPU FaceMesh',composition='geometric lower-face feather mask',
            audio='full 10-second known waveform; MuseTalk features precomputed and timed separately'),
        inputs={str(p):sha(p) for p in [reference,audio_path]},runs=[])
    def save(): (args.output/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    save()
    try:
        pm.COMPILE_MODEL=pm.COMPILE_VAE=pm.USE_PARALLEL_VAE=False
        pipeline=pm.FlashHeadPipeline(str(ROOT/'models/SoulX-FlashHead-1_3B'),'lite',
            str(ROOT/'models/wav2vec2-base-960h'),param_dtype=torch.bfloat16)
        audio,_=librosa.load(audio_path,sr=16000,mono=True)
        slices=np.pad(audio,(0,11*15360-len(audio))).reshape(-1,15360)
        def prepare():
            torch.manual_seed(50)
            pipeline.prepare_params(str(reference),(576,320),33,9,4,seed=50,shift=5,color_correction_strength=1,use_face_crop=False)
        recorded=set()
        def run(mode,mouth=None,prompts=None,batch=8):
            prepare()
            if mouth: mouth.reset()
            cache=deque([0.]*128000,maxlen=128000)
            stages={}; chunk_times=[]; raw=[]; enhanced=[]; count=0
            torch.cuda.reset_peak_memory_stats()
            with Monitor() as monitor:
                start=time.perf_counter()
                for segment in slices:
                    chunk_start=time.perf_counter()
                    with timed(stages,'soulx_generate_audio_decode_reencode'):
                        cache.extend(segment.tolist())
                        embedding=get_audio_embedding(pipeline,np.asarray(cache,dtype=np.float32),167,200)
                        frames=run_pipeline(pipeline,embedding)[9:].cpu().numpy().astype(np.uint8)
                    frames=frames[:250-count]; raw.append(frames)
                    if mouth: enhanced.append(mouth.apply(frames,prompts[count:count+len(frames)],batch,stages))
                    count+=len(frames)
                    chunk_times.append(time.perf_counter()-chunk_start)
                torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            row=dict(mode=mode,batch=batch if mouth else None,elapsed_s=elapsed,fps=count/elapsed,
                frames=count,stage_s=stages,chunk_s=chunk_times,first_chunk_s=chunk_times[0],
                peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                gpu_after=snapshot(),processes_after=processes(),samples=monitor.samples,
                raw_sha256=hashlib.sha256(np.concatenate(raw).tobytes()).hexdigest())
            if mouth:
                row['tracked_frames']=mouth.detected; mouth.mesh.close()
                row['enhanced_sha256']=hashlib.sha256(np.concatenate(enhanced).tobytes()).hexdigest()
            if args.record and 'warmup' not in mode and mode not in recorded:
                from soulx_rtc.experiment import record
                from PIL import Image
                label = 'soulx-LITE-indian-man-1.50x-baseline' if mouth is None else f'soulx-LITE-plus-MuseTalk-indian-man-1.50x-face-batch{batch}'
                folder=args.output/label
                folder.mkdir(exist_ok=False)
                pixels=np.concatenate(enhanced if mouth else raw)
                record(folder/f'{label}.mp4',[pixels],audio,25)
                for timestamp in (.5,1.5,2.5,3.5,4.5,6.5,8.5):
                    Image.fromarray(pixels[round(timestamp*25)]).save(folder/f'frame-{timestamp:.1f}s.png')
                row['recorded_video']=str(folder/f'{label}.mp4')
                row['video_sha256']=sha(folder/f'{label}.mp4')
                recorded.add(mode)
            print(json.dumps({k:v for k,v in row.items() if k in ('mode','elapsed_s','fps','stage_s','gpu_after')}),flush=True)
            return row
        data['status']='baseline'; save()
        data['warmup_baseline']=run('baseline_warmup')
        for _ in range(args.repeats): data['runs'].append(run('lite_only')); save()
        torch.cuda.empty_cache()
        data['status']='loading_musetalk'; save()
        mouth=MouthPass(); data['gpu_both_loaded']=snapshot(); save()
        torch.cuda.empty_cache()
        audio_times={}
        with timed(audio_times,'audio_prepare_cold_s'): prompts=mouth.audio(audio_path)
        with timed(audio_times,'audio_prepare_warm_s'): prompts=mouth.audio(audio_path)
        data['musetalk_audio']=audio_times
        data['status']='hybrid'; save()
        for batch in args.face_batches:
            data[f'warmup_hybrid_b{batch}']=run(f'hybrid_b{batch}_warmup',mouth,prompts,batch); save()
            for _ in range(args.repeats): data['runs'].append(run(f'hybrid_b{batch}',mouth,prompts,batch)); save()
        data['raw_soulx_parity']=len({r['raw_sha256'] for r in data['runs']})==1
        data['status']='complete'; save()
    except Exception as e:
        data.update(status='failed',error=repr(e),gpu_failure=snapshot()); save(); raise
    finally: lease.close()


if __name__=='__main__': main()
