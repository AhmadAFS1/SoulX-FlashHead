"""Relative same-audio SyncNet diagnostic using matched baseline face boxes.

This is not standard LSE-C/LSE-D, an absolute quality score, or dental validation.
"""
import argparse
import json
from pathlib import Path
import sys

import cv2
import librosa
import numpy as np
import torch
import yaml

from benchmarks.pro_lite_teeth_20260917.run_variant import gpu_snapshot, process_snapshot, sha256
from soulx_rtc.gpu_lease import acquire_gpu_lease


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--reviews',type=Path,nargs='+',required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--own-boxes',action='store_true')
    args=ap.parse_args()
    lease=acquire_gpu_lease(); torch.set_num_threads(4); cv2.setNumThreads(1)
    muse=Path('/workspace/MuseTalk'); sys.path.insert(0,str(muse))
    from musetalk.models.syncnet import SyncNet
    from musetalk.data.audio import melspectrogram
    config=yaml.safe_load((muse/'configs/training/syncnet.yaml').read_text())['model']
    model=SyncNet(config)
    checkpoint=muse/'models/syncnet/latentsync_syncnet.pt'
    weights=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model.load_state_dict(weights['state_dict'],strict=True);del weights
    model=model.cuda().eval()
    starts=list(range(25,211,5)); offsets=list(range(-5,6))
    wav,_=librosa.load('benchmarks/pro_lite_150x_20260917/audio.wav',sr=16000)
    mel=melspectrogram(wav)
    keys=sorted({s+o for s in starts for o in offsets})
    chunks=np.stack([mel[:,int(80*k/25):int(80*k/25)+52] for k in keys])[:,None]
    audio=[]
    for i in range(0,len(chunks),8):
        audio.append(model.get_audio_embed(torch.from_numpy(chunks[i:i+8]).float().cuda()).cpu())
    audio=dict(zip(keys,torch.cat(audio)))
    result=dict(execution='Fresh GPU relative SyncNet evaluation, no video generation',
                gpu=gpu_snapshot(),physical_vram_class='12 GB',torch=torch.__version__,cuda=torch.version.cuda,
                processes=process_snapshot(),checkpoint_sha256=sha256(checkpoint),
                method=('Per-output FaceMesh boxes' if args.own_boxes else 'Baseline FaceMesh boxes reused within each pair')+'; RGB 256x256 lower half; 16 frames, 80x52 mel; cosine similarity; positive lag pairs video with later audio.',
                limitations='Relative diagnostic only; not official LSE-C/LSE-D. Overlapping windows; same identity/audio. Face crops can confound scores.',reviews={})
    for review in args.reviews:
        info=json.loads((review/'review.json').read_text())
        boxes=[r['face_box'] for r in info['inputs']['stock']['analysis']['frames']]
        pair={}
        for name in ('stock','candidate'):
            if args.own_boxes:
                boxes=[r['face_box'] for r in info['inputs'][name]['analysis']['frames']]
            path=Path(info['inputs'][name]['path'])/'video.mp4'
            cap=cv2.VideoCapture(str(path)); frames=[]
            for x1,y1,x2,y2 in boxes:
                ok,bgr=cap.read();assert ok
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(rgb[y1:y2,x1:x2],(256,256),interpolation=cv2.INTER_LANCZOS4))
            cap.release()
            frames=np.stack(frames)
            vision=[]
            for s in starts:
                clip=torch.from_numpy(frames[s:s+16]).permute(0,3,1,2).float()/127.5-1
                clip=clip.reshape(1,48,256,256)[:,:,128:,:].cuda()
                vision.append(model.get_image_embed(clip).cpu()[0])
            vision=torch.stack(vision)
            scores={}
            for offset in offsets:
                ae=torch.stack([audio[s+offset] for s in starts])
                per=(vision*ae).sum(1).numpy()
                scores[str(offset)]=dict(mean=float(per.mean()),per_window=per.tolist())
            best=max(offsets,key=lambda o:scores[str(o)]['mean'])
            shuffled=(vision*torch.stack([audio[s] for s in starts]).roll(len(starts)//2,0)).sum(1)
            pair[name]=dict(best_lag_frames=best,best_mean=scores[str(best)]['mean'],
                            zero_lag_mean=scores['0']['mean'],shuffled_mean=float(shuffled.mean()),scores=scores)
        result['reviews'][review.name]=pair
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(review.name,{k:{x:v for x,v in r.items() if x!='scores'} for k,r in pair.items()},flush=True)
    result['status']='complete';args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
