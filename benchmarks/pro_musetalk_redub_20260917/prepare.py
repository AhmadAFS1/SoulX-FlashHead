"""CPU preparation/anchoring for the PRO -> MuseTalk redub diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image
from flash_head.utils.utils import resize_and_centercrop

ROOT = Path(__file__).resolve().parent
SILENT = ROOT.parent/'closed_mouth_silence_20260917/silence-seed51.mp4'


def decode(path):
    cap=cv2.VideoCapture(str(path)); frames=[]
    while True:
        ok,bgr=cap.read()
        if not ok: break
        frames.append(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    cap.release()
    return np.asarray(frames)


def encode(path, frames):
    proc=subprocess.Popen(['ffmpeg','-v','error','-n','-f','rawvideo','-pix_fmt','rgb24',
        '-s','320x576','-r','25','-i','pipe:0','-an','-c:v','libx264','-qp','0',
        '-g','1','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    proc.communicate(frames.tobytes()); assert proc.returncode==0


def anchor(frames, ref):
    out=frames.copy()
    # Six exact shared frames, then a six-frame cosine blend at each end.
    for j in range(12):
        weight=0.0 if j<6 else .5-.5*np.cos(np.pi*(j-5)/7)
        for i in (j,len(out)-1-j):
            out[i]=np.rint(weight*frames[i]+(1-weight)*ref).astype(np.uint8)
    return out


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['inputs','anchor']);a=ap.parse_args()
    if a.stage=='inputs':
        rgb=decode(SILENT)[0]
        Image.fromarray(rgb).save(ROOT/'reference-from-silence-seed51.png')
        ref=resize_and_centercrop(Image.fromarray(rgb),(576,320))[0,:,0].permute(1,2,0).numpy()
        Image.fromarray(ref).save(ROOT/'reference-320x576.png')
        sources={'A':ROOT.parent/'comparison-10s.wav',
                 'B':Path('/workspace/MuseTalk/generated/webrtc_quality/2026-08-08/kokoro_duration_matrix/multi_sentence.wav')}
        provenance={}
        for key,path in sources.items():
            subprocess.run(['ffmpeg','-v','error','-n','-i',str(path),'-af',
                'atrim=start=0:end=9,asetpts=PTS-STARTPTS,afade=t=out:st=8.95:d=0.05,adelay=500,apad=whole_dur=10',
                '-t','10','-ar','16000','-ac','1',str(ROOT/f'audio-{key}.wav')],check=True)
            provenance[key]={'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        (ROOT/'inputs.json').write_text(json.dumps(dict(execution='CPU media preparation, no GPU inference',
            silent_video=str(SILENT),silent_sha256=hashlib.sha256(SILENT.read_bytes()).hexdigest(),
            reference='First decoded frame of silence-seed51, then SoulX standard resize/center-crop to 320x576',
            audio='First 9 seconds of each recording, 0.5 s silent handles each end, 50 ms fade at speech cut',sources=provenance),indent=2)+'\n')
    else:
        ref=np.array(Image.open(ROOT/'reference-320x576.png'))
        pro=decode(ROOT/'pro/video.mp4');assert len(pro)==250
        silent=decode(SILENT)
        control=np.array([np.asarray(resize_and_centercrop(Image.fromarray(silent[min(round(i*24/25),len(silent)-1)]),(576,320))[0,:,0].permute(1,2,0)) for i in range(250)])
        checks={}
        for name,frames in [('pro-base',pro),('silent-control-base',control)]:
            path=ROOT/f'{name}.mp4';encode(path,anchor(frames,ref))
            decoded=decode(path);assert len(decoded)==250
            exact=all(np.array_equal(decoded[0],decoded[i]) for i in [*range(6),*range(244,250)])
            assert exact
            checks[name]=dict(frames=len(decoded),first_last_equal=True,all_12_handle_frames_equal=exact,
                decoded_anchor_sha256=hashlib.sha256(decoded[0].tobytes()).hexdigest())
        assert checks['pro-base']['decoded_anchor_sha256']==checks['silent-control-base']['decoded_anchor_sha256']
        (ROOT/'anchor-validation.json').write_text(json.dumps(dict(execution='CPU processing',
            method='6 exact handle frames and 6 cosine-blend frames per end; all-intra lossless H264 yuv420p',
            control_tail='8 s silent clip held for final 2 s; primary quality scoring excludes final handle region',clips=checks),indent=2)+'\n')
        tasks={name:dict(video_path=str(ROOT/f'{base}.mp4'),audio_path=str(ROOT/f'audio-{audio}.wav'),result_name=f'{name}.mp4')
               for name,base,audio in [('pro-redub-B','pro-base','B'),('silent-redub-B','silent-control-base','B'),('pro-redub-A','pro-base','A')]}
        (ROOT/'musetalk-tasks.json').write_text(json.dumps(tasks,indent=2)+'\n')


if __name__=='__main__': main()
