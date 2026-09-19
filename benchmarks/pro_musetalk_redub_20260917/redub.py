"""Offline MuseTalk 1.5 diagnostic using native DWPose/S3FD/BiSeNet preparation."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
MUSE=Path('/workspace/MuseTalk')
sys.path[:0]=[str(MUSE),str(MUSE/'scripts'),str(MUSE/'musetalk/utils')]
from runtime_cpu_tuning import apply_cpu_tuning_runtime
apply_cpu_tuning_runtime('pro-redub-diagnostic')
from musetalk.utils.preprocessing import get_landmark_and_bbox,coord_placeholder
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.blending import get_image_prepare_material,get_image_blending
from musetalk.utils.utils import load_all_model,datagen
from musetalk.utils.audio_processor import AudioProcessor
from transformers import WhisperModel


def snapshot():
    return subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv'],text=True)


def write_video(path,frames,audio):
    silent=path.with_suffix('.silent.mp4')
    p=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','bgr24',
        '-s','320x576','-r','25','-i','pipe:0','-an','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p',str(silent)],stdin=subprocess.PIPE)
    p.communicate(np.asarray(frames).tobytes());assert p.returncode==0
    subprocess.run(['ffmpeg','-v','error','-n','-i',str(silent),'-i',str(audio),
        '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t','10',
        '-movflags','+faststart',str(path)],check=True)


@torch.no_grad()
def main():
    output=ROOT/'musetalk';output.mkdir(exist_ok=True)
    if list(output.glob('*-redub-*.mp4')):
        raise RuntimeError('Completed render exists; use a fresh experiment directory.')
    torch.manual_seed(123);torch.set_num_threads(4)
    metadata=dict(date_utc=datetime.now(timezone.utc).isoformat(),execution='Fresh GPU inference',
        gpu=snapshot(),torch=torch.__version__,cuda=torch.version.cuda,
        co_resident=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv'],text=True),
        profile=dict(model='MuseTalk V1.5',precision='FP16',width=320,height=576,fps=25,batch=8,
                     bbox_shift=0,extra_margin=10,parsing_mode='jaw',left_cheek_width=90,right_cheek_width=90,
                     preprocessing='native DWPose + S3FD, BiSeNet cached masks; stock VAE+UNet+Whisper',
                     boundary_postprocessing='none; raw MuseTalk output retained'),runs=[])
    def save(): (output/'results.json').write_text(json.dumps(metadata,indent=2)+'\n')
    save()
    vae,unet,pe=load_all_model(unet_config=str(MUSE/'models/musetalkV15/musetalk.json'),device='cuda')
    vae.vae=vae.vae.half().eval();vae.runtime_dtype=torch.float16
    unet.model=unet.model.half().eval();pe=pe.half().cuda().eval()
    whisper=WhisperModel.from_pretrained(str(MUSE/'models/whisper')).half().cuda().eval()
    processor=AudioProcessor(str(MUSE/'models/whisper'))
    parser=FaceParsing(left_cheek_width=90,right_cheek_width=90)
    audio={}
    for tag in ['A','B']:
        features,n=processor.get_audio_feature(str(ROOT/f'audio-{tag}.wav'))
        audio[tag]=processor.get_whisper_chunk(features,'cuda',torch.float16,whisper,n,fps=25,
            audio_padding_length_left=2,audio_padding_length_right=2)
        assert len(audio[tag])>=250
        audio[tag]=audio[tag][:250]
    for base,tags in [('pro-base',['A','B']),('silent-control-base',['B'])]:
        folder=output/base;folder.mkdir(exist_ok=True)
        if len(list(folder.glob('*.png')))!=250:
            subprocess.run(['ffmpeg','-v','error','-n','-i',str(ROOT/f'{base}.mp4'),str(folder/'%08d.png')],check=True)
        files=sorted(folder.glob('*.png'));assert len(files)==250
        coords,frames=get_landmark_and_bbox([str(p) for p in files],0)
        assert len(coords)==250 and all(tuple(c)!=coord_placeholder for c in coords)
        masks=[];crop_boxes=[];latents=[];updated=[]
        torch.manual_seed(123)
        for idx,(frame,box) in enumerate(zip(frames,coords)):
            x1,y1,x2,y2=map(int,box);y2=min(y2+10,frame.shape[0]);box=[x1,y1,x2,y2]
            crop=cv2.resize(frame[y1:y2,x1:x2],(256,256),interpolation=cv2.INTER_LANCZOS4)
            latents.append(vae.get_latents_for_unet(crop))
            mask,crop_box=get_image_prepare_material(frame,box,fp=parser,mode='jaw')
            assert mask is not None and np.count_nonzero(mask)>0
            masks.append(mask);crop_boxes.append(crop_box);updated.append(box)
        (folder/'coords.json').write_text(json.dumps(updated)+'\n')
        for tag in tags:
            name=('pro' if base=='pro-base' else 'silent')+f'-redub-{tag}'
            recon=[];started=time.perf_counter()
            for whisper_batch,latent_batch in datagen(audio[tag],latents,batch_size=8,device='cuda'):
                predicted=unet.model(latent_batch.half(),torch.tensor([0],device='cuda'),
                    encoder_hidden_states=pe(whisper_batch.to(device='cuda',dtype=torch.float16))).sample
                recon.extend(vae.decode_latents(predicted))
            out=[]
            for i,face in enumerate(recon):
                x1,y1,x2,y2=updated[i]
                face=cv2.resize(face.astype(np.uint8),(x2-x1,y2-y1))
                out.append(get_image_blending(frames[i],face,updated[i],masks[i],crop_boxes[i]))
            assert len(out)==250
            target=output/f'{name}.mp4'
            write_video(target,out,ROOT/f'audio-{tag}.wav')
            row=dict(name=name,base=base,audio=tag,frames=250,face_detections=250,
                     render_and_encode_seconds=time.perf_counter()-started,
                     first_last_rgb_equal=bool(np.array_equal(out[0],out[-1])),
                     source_sha256=hashlib.sha256((ROOT/f'{base}.mp4').read_bytes()).hexdigest(),
                     audio_sha256=hashlib.sha256((ROOT/f'audio-{tag}.wav').read_bytes()).hexdigest())
            metadata['runs'].append(row);save();print(json.dumps(row),flush=True)
    metadata['status']='complete';save()


if __name__=='__main__':main()
