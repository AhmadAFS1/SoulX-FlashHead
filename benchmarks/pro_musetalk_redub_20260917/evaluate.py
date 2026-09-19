"""Controlled relative SyncNet assessment; not standard LSE or certification."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone

import cv2
import numpy as np
import torch
import yaml
import librosa

ROOT=Path(__file__).resolve().parent
MUSE=Path('/workspace/MuseTalk');sys.path.insert(0,str(MUSE))
from musetalk.models.syncnet import SyncNet
from musetalk.data.audio import melspectrogram


def decode(path):
    cap=cv2.VideoCapture(str(path));out=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        out.append(cv2.cvtColor(f,cv2.COLOR_BGR2RGB))
    cap.release();assert len(out)==250,(path,len(out))
    return out


@torch.no_grad()
def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    config=yaml.safe_load((MUSE/'configs/training/syncnet.yaml').read_text())['model']
    model=SyncNet(config)
    ckpt=MUSE/'models/syncnet/latentsync_syncnet.pt'
    weights=torch.load(ckpt,map_location='cpu',weights_only=False)
    model.load_state_dict(weights['state_dict'],strict=True)
    del weights
    model=model.cuda().eval()
    starts=list(range(25,211,5)) # 1.00 through 8.40 s, each with 0.64 s context.
    offsets=list(range(-10,11))
    audio_emb={}
    for tag in ['A','B']:
        wav,_=librosa.load(ROOT/f'audio-{tag}.wav',sr=16000)
        mel=melspectrogram(wav)
        keys=sorted(set(s+o for s in starts for o in offsets))
        chunks=np.stack([mel[:,int(80*k/25):int(80*k/25)+52] for k in keys])[:,None]
        embs=[]
        for i in range(0,len(chunks),8):
            embs.append(model.get_audio_embed(torch.from_numpy(chunks[i:i+8]).float().cuda()).cpu())
        audio_emb[tag]=dict(zip(keys,torch.cat(embs)))
    videos={'pro-base':(ROOT/'pro-base.mp4','pro-base'),
            'pro-redub-A':(ROOT/'musetalk/pro-redub-A.mp4','pro-base'),
            'pro-redub-B':(ROOT/'musetalk/pro-redub-B.mp4','pro-base'),
            'silent-redub-B':(ROOT/'musetalk/silent-redub-B.mp4','silent-control-base')}
    data=dict(date_utc=datetime.now(timezone.utc).isoformat(),
        execution='Fresh GPU LatentSync SyncNet evaluation; not video generation',
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv'],text=True),
        co_resident=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True),
        torch=torch.__version__,cuda=torch.version.cuda,
        checkpoint=str(ckpt),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),
        method='Native source DWPose boxes + 10 pixel margin, RGB 256x256, normalized [-1,1], lower half; 16 frames at 25 FPS; repo mel 80x52. Cosine similarity, higher is better. Positive lag pairs video with later audio.',
        limitations='Relative diagnostic only, not official LSE-C/LSE-D, no absolute acceptance threshold; windows overlap. First/final blend regions excluded. Source boxes reused for aligned outputs.',
        starts=starts,offsets=offsets,runs={})
    for name,(path,base) in videos.items():
        frames=decode(path)
        coords=json.loads((ROOT/f'musetalk/{base}/coords.json').read_text())
        crops=np.stack([cv2.resize(f[y1:y2,x1:x2],(256,256),interpolation=cv2.INTER_LANCZOS4)
                        for f,(x1,y1,x2,y2) in zip(frames,coords)])
        embeddings=[]
        for s in starts:
            clip=torch.from_numpy(crops[s:s+16]).permute(0,3,1,2).float()/127.5-1
            clip=clip.reshape(1,48,256,256)[:,:,128:,:].cuda()
            embeddings.append(model.get_image_embed(clip).cpu()[0])
        vision=torch.stack(embeddings)
        result={}
        for tag in ['A','B']:
            scores={}
            for offset in offsets:
                audio=torch.stack([audio_emb[tag][s+offset] for s in starts])
                vals=(vision*audio).sum(1).numpy()
                scores[str(offset)]=dict(mean=float(vals.mean()),per_window=vals.tolist())
            best=max(offsets,key=lambda k:scores[str(k)]['mean'])
            correct_audio=torch.stack([audio_emb[tag][s] for s in starts])
            shuffled=(vision*correct_audio.roll(len(starts)//2,0)).sum(1)
            result[tag]=dict(zero_lag_mean=scores['0']['mean'],best_lag_frames=best,
                best_mean=scores[str(best)]['mean'],shuffled_mean=float(shuffled.mean()),scores=scores)
        data['runs'][name]=result
        (ROOT/'syncnet-results.json').write_text(json.dumps(data,indent=2)+'\n')
        print(name,json.dumps({k:{j:v for j,v in r.items() if j!='scores'} for k,r in result.items()}),flush=True)
    data['status']='complete'
    (ROOT/'syncnet-results.json').write_text(json.dumps(data,indent=2)+'\n')


if __name__=='__main__':main()
