"""CPU packaging: labeled video, native PNG mouth samples, full media validation."""
import argparse
import json
from pathlib import Path
import subprocess
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

TIMES=(.5,1.5,2.5,3.5,4.5,6.5,8.5)


def package(videos,labels,prefix):
    prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
    command=['ffmpeg','-y','-v','error']
    filters=[]
    for i,(path,label) in enumerate(zip(videos,labels)):
        command+=['-i',str(path)]
        filters.append(f"[{i}:v]drawtext=text='{label}':x=8:y=12:fontsize=17:fontcolor=white:borderw=2:bordercolor=black[v{i}]")
    filters.append(''.join(f'[v{i}]' for i in range(len(videos)))+f'hstack=inputs={len(videos)}[out]')
    out=prefix.parent/(prefix.name+'.mp4')
    command+=['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]',
        '-map','0:a:0','-c:v','libx264','-crf','18','-preset','veryfast','-pix_fmt','yuv420p',
        '-c:a','copy','-t','10','-movflags','+faststart',str(out)]
    subprocess.run(command,check=True)
    mouth=Image.new('RGB',(192*len(TIMES),150*len(videos)),'white');d=ImageDraw.Draw(mouth)
    full=Image.new('RGB',(160*len(TIMES),315*len(videos)),'white');f=ImageDraw.Draw(full)
    detections=[]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True) as mesh:
        for row,(path,label) in enumerate(zip(videos,labels)):
            cap=cv2.VideoCapture(str(path))
            for col,t in enumerate(TIMES):
                png=Path(path).parent/f'frame-{t:.1f}s.png'
                if png.exists(): rgb=np.array(Image.open(png).convert('RGB')); source='native PNG'
                else:
                    cap.set(cv2.CAP_PROP_POS_MSEC,t*1000);ok,bgr=cap.read();assert ok
                    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);source='decoded H264'
                f.text((col*160+3,row*315+3),f'{label} {t}s',fill='black')
                full.paste(Image.fromarray(rgb).resize((160,288)),(col*160,row*315+26))
                result=mesh.process(rgb);assert result.multi_face_landmarks
                pts=np.array([(p.x*rgb.shape[1],p.y*rgb.shape[0]) for p in result.multi_face_landmarks[0].landmark])
                cx,cy=np.round(pts[[13,14]].mean(0)).astype(int)
                crop=Image.fromarray(rgb).crop((cx-48,cy-26,cx+48,cy+26))
                mouth.paste(crop.resize((192,104),Image.Resampling.NEAREST),(col*192,row*150+42))
                d.text((col*192+3,row*150+3),f'{label}\n{t}s {source}',fill='black')
                detections.append(dict(label=label,time=t,source=source,opening_px=float(np.linalg.norm(pts[13]-pts[14]))))
            cap.release()
    mouth.save(prefix.parent/(prefix.name+'-mouths.png'));full.save(prefix.parent/(prefix.name+'-frames.png'))
    validation={}
    for path in [*map(Path,videos),out]:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
            'stream=codec_type,width,height,avg_frame_rate,duration,nb_read_frames','-of','json',str(path)],text=True))
        video=next(s for s in info['streams'] if s['codec_type']=='video')
        assert int(video['nb_read_frames'])==250 and video['avg_frame_rate']=='25/1'
        assert any(s['codec_type']=='audio' for s in info['streams'])
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
        validation[str(path)]=info
    (prefix.parent/(prefix.name+'-validation.json')).write_text(json.dumps(dict(
        execution='CPU packaging and media checks; no new GPU inference',media=validation,samples=detections),indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--video',action='append',required=True)
    ap.add_argument('--label',action='append',required=True);ap.add_argument('--prefix',required=True)
    a=ap.parse_args();assert len(a.video)==len(a.label);package(a.video,a.label,a.prefix)
