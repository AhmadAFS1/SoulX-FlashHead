"""Single visual generator: Lite face crops with fixed/aligned CPU composition.

Exploratory quality/throughput experiment. No MuseTalk or SR weights loaded.
The body/background plate is static, so this changes the motion contract.
"""
from collections import deque
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import librosa
import mediapipe as mp
import numpy as np
from PIL import Image
import torch

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
from benchmarks.roi_hybrid_analysis_20260917.benchmark import Monitor, snapshot, processes, sha
from flash_head.src.pipeline import flash_head_pipeline as pm
from flash_head.inference import get_audio_embedding,run_pipeline
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.experiment import record

ROOT=Path(__file__).resolve().parent
SOURCE=REPO/'benchmarks/distance_lipsync/evidence-indian-male-20260916/reference-close.png'
AUDIO=REPO/'benchmarks/comparison-10s.wav'
BOX=(16,48,304,336)  # Native 288-square head/neck context inside the normal plate.
ANCHORS=[33,133,362,263,168,6]  # Eyes and upper nose; exclude moving mouth/jaw.
TIMES=(.5,1.5,2.5,3.5,4.5,6.5,8.5)


def landmarks(mesh,rgb):
    pred=mesh.process(rgb)
    if not pred.multi_face_landmarks:return None
    return np.array([(p.x*rgb.shape[1],p.y*rgb.shape[0]) for p in pred.multi_face_landmarks[0].landmark],np.float32)


class Composer:
    def __init__(self,plate):
        self.plate=plate
        self.mesh=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True)
        self.reference=landmarks(self.mesh,plate)
        assert self.reference is not None
        self.mesh.close()
        self.mesh=mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True)
        yy,xx=np.mgrid[:288,:288]
        self.fixed_alpha=np.clip(np.minimum.reduce([xx,yy,287-xx,287-yy])/20,0,1).astype(np.float32)[:,:,None]
        mask=np.zeros(plate.shape[:2],np.float32)
        ids=sorted(set(i for pair in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in pair))
        hull=cv2.convexHull(np.round(self.reference[ids]).astype(np.int32))
        cv2.fillConvexPoly(mask,hull,1)
        mask=cv2.dilate(mask,np.ones((7,7),np.uint8))
        mask=cv2.GaussianBlur(mask,(19,19),3)
        self.face_alpha=mask[:,:,None]
        self.rows=[]

    def fixed(self,rgb):
        patch=cv2.resize(rgb,(288,288),interpolation=cv2.INTER_AREA if rgb.shape[0]>=288 else cv2.INTER_CUBIC)
        x0,y0,x1,y1=BOX;out=self.plate.copy()
        out[y0:y1,x0:x1]=(patch*self.fixed_alpha+out[y0:y1,x0:x1]*(1-self.fixed_alpha)).round().clip(0,255).astype(np.uint8)
        return out

    def aligned(self,rgb):
        pts=landmarks(self.mesh,rgb)
        if pts is None:
            self.rows.append(dict(detected=False));return self.plate.copy()
        transform,_=cv2.estimateAffinePartial2D(pts[ANCHORS],self.reference[ANCHORS],method=cv2.LMEDS)
        if transform is None:raise RuntimeError('No stable similarity transform')
        warped=cv2.warpAffine(rgb,transform,(self.plate.shape[1],self.plate.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
        moved=cv2.transform(pts[None],transform)[0]
        # Diagnostic motion disagreement before alignment in plate coordinates.
        mapped=pts*288/rgb.shape[0]+np.array(BOX[:2])
        self.rows.append(dict(detected=True,
            upper_anchor_rms_before_px=float(np.sqrt(np.mean(np.sum((mapped[ANCHORS]-self.reference[ANCHORS])**2,axis=1)))),
            upper_anchor_rms_after_px=float(np.sqrt(np.mean(np.sum((moved[ANCHORS]-self.reference[ANCHORS])**2,axis=1)))),
            normalized_opening=float(np.linalg.norm(pts[13]-pts[14])/np.linalg.norm(pts[33]-pts[263])),
            transform=transform.tolist()))
        out=(warped*self.face_alpha+self.plate*(1-self.face_alpha)).round().clip(0,255).astype(np.uint8)
        assert np.array_equal(out[self.face_alpha[:,:,0]==0],self.plate[self.face_alpha[:,:,0]==0])
        return out

    def close(self):self.mesh.close()


def save_video(folder,name,frames,audio):
    folder.mkdir(exist_ok=False)
    path=folder/(name+'.mp4');record(path,[frames],audio,25)
    for t in TIMES:Image.fromarray(frames[round(t*25)]).save(folder/f'frame-{t:.1f}s.png')
    return dict(path=str(path.relative_to(REPO)),video_sha256=sha(path),raw_rgb_sha256=hashlib.sha256(frames.tobytes()).hexdigest())


@torch.inference_mode()
def main(output=ROOT):
    root=Path(output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    assert not (root/'results.json').exists(),'Choose a fresh experiment directory with --output'
    torch.set_num_threads(4);cv2.setNumThreads(1)
    plate=np.array(Image.open(SOURCE).convert('RGB'))
    Image.fromarray(plate).save(root/'reference-normal-distance.png')
    audio,_=librosa.load(AUDIO,sr=16000,mono=True)
    chunks=np.pad(audio,(0,11*15360-len(audio))).reshape(-1,15360)
    result=dict(status='starting',date_utc=datetime.now(timezone.utc).isoformat(),
        gpu=snapshot(),processes=processes(),torch=torch.__version__,cuda=torch.version.cuda,
        cpu_logical_count=__import__('os').cpu_count(),
        source=str(SOURCE),source_sha256=sha(SOURCE),audio_sha256=sha(AUDIO),
        crop_xyxy=BOX,profile=dict(steps=4,fps=25,frames=250,dtype='BF16',compile=False,
            visual_generator='same single SoulX Lite weights in all arms',plate='static normal-distance portrait'),runs=[])
    def save():(root/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    with acquire_gpu_lease():
        try:
            pm.COMPILE_MODEL=pm.COMPILE_VAE=pm.USE_PARALLEL_VAE=False
            pipeline=pm.FlashHeadPipeline(str(REPO/'models/SoulX-FlashHead-1_3B'),'lite',
                str(REPO/'models/wav2vec2-base-960h'),param_dtype=torch.bfloat16)
            for size in (0,256,384,512):
                width,height=(320,576) if size==0 else (size,size)
                reference=SOURCE if size==0 else root/f'reference-face-crop-{size}.png'
                if size:Image.fromarray(plate).crop(BOX).resize((size,size),Image.Resampling.LANCZOS).save(reference)
                def prepare(seed):
                    torch.manual_seed(seed)
                    pipeline.prepare_params(str(reference),(height,width),33,9,4,seed=seed,shift=5,
                        color_correction_strength=1,use_face_crop=False)
                prepare(50)
                # Shape-specific warmup, reset again before each measured trajectory.
                embedding=get_audio_embedding(pipeline,np.zeros(128000,np.float32),167,200)
                run_pipeline(pipeline,embedding);torch.cuda.synchronize();torch.cuda.empty_cache()
                for seed in (50,51):
                    prepare(seed)
                    cache=deque([0.]*128000,maxlen=128000)
                    original=[];fixed=[];aligned=[];stages=dict(generation=0.,fixed_composition=0.,aligned_composition=0.)
                    composer=Composer(plate) if size else None
                    torch.cuda.reset_peak_memory_stats()
                    chunk_times=[]
                    with Monitor() as monitor:
                        start=time.perf_counter();count=0
                        for segment in chunks:
                            t=time.perf_counter();cache.extend(segment.tolist())
                            embedding=get_audio_embedding(pipeline,np.asarray(cache,np.float32),167,200)
                            raw=run_pipeline(pipeline,embedding)[9:].cpu().numpy().astype(np.uint8)
                            torch.cuda.synchronize();dt=time.perf_counter()-t
                            stages['generation']+=dt;chunk_times.append(dt)
                            raw=raw[:250-count];count+=len(raw);original.append(raw)
                            if composer:
                                t=time.perf_counter();fixed.extend(composer.fixed(f) for f in raw)
                                stages['fixed_composition']+=time.perf_counter()-t
                                t=time.perf_counter();aligned.extend(composer.aligned(f) for f in raw)
                                stages['aligned_composition']+=time.perf_counter()-t
                        elapsed=time.perf_counter()-start
                    label=f'full-frame-seed{seed}' if size==0 else f'face-crop{size}-seed{seed}'
                    row=dict(variant=label,size=[width,height],seed=seed,status='complete',
                        reference=str(reference),stages_s=stages,diagnostic_two_compositor_wall_s=elapsed,
                        generation_fps=250/stages['generation'],
                        fixed_branch_fps=250/(stages['generation']+stages['fixed_composition']),
                        aligned_branch_fps=250/(stages['generation']+stages['aligned_composition']),
                        fps_note='Branch FPS from measured generation plus that measured compositor; excludes alternative branch, loading, warmup, setup and encoding',
                        chunk_generation_s=chunk_times,samples=monitor.samples,
                        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                        gpu_after=snapshot(),processes_after=processes())
                    folder=root/label;folder.mkdir(exist_ok=False)
                    row['native']=save_video(folder/'native',f'soulx-LITE-{label}-native',np.concatenate(original),audio)
                    if composer:
                        row['fixed']=save_video(folder/'fixed',f'soulx-LITE-{label}-fixed-paste-normal-distance',np.stack(fixed),audio)
                        row['aligned']=save_video(folder/'aligned',f'soulx-LITE-{label}-aligned-face-normal-distance',np.stack(aligned),audio)
                        row['alignment']=composer.rows;row['outside_aligned_face_unchanged']=True;composer.close()
                    result['runs'].append(row);save()
                    print(json.dumps({k:v for k,v in row.items() if k in ['variant','generation_fps','fixed_branch_fps','aligned_branch_fps','peak_allocated_mib']}),flush=True)
            result['status']='complete';save()
        except Exception as e:
            result.update(status='failed',error=repr(e));save();raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT,help='Fresh output directory within the repository')
    main(parser.parse_args().output)
