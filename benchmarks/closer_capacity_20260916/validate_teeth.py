"""CPU comparison of retained RTX 4070 SUPER videos; no new GPU inference."""
import json
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
CLIPS = [
    ('25fps stock', ROOT/'benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/closer-125-seed-50.mp4', 0),
    ('25fps strength 0.5', ROOT/'benchmarks/distance_lipsync/evidence-indian-male-closer-strength-0p5-20260916/closer-125-strength-0p5-seed-50.mp4', 0),
    ('15fps receiver', OUT/'closer125-c5-peer0.mp4', 1.8265),
]
TIMES = [0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5]

def main():
    cv2.setNumThreads(1)
    sheet=Image.new('RGB',(7*192,3*154),'#eeeeee')
    draw=ImageDraw.Draw(sheet)
    rows=[]
    for row,(label,path,offset) in enumerate(CLIPS):
        cap=cv2.VideoCapture(str(path)); fps=cap.get(cv2.CAP_PROP_FPS)
        values=[]; samples=[]; index=0
        with mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True) as mesh:
            while True:
                ok,bgr=cap.read()
                if not ok: break
                t=index/fps-offset; index+=1
                if not 0<=t<10: continue
                rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB)
                prediction=mesh.process(rgb)
                if not prediction.multi_face_landmarks: continue
                pts=np.array([(p.x*rgb.shape[1],p.y*rgb.shape[0]) for p in prediction.multi_face_landmarks[0].landmark])
                cx,cy=pts[[13,14]].mean(axis=0)
                # Fixed native-pixel crop; avoids making resampling sharpen one clip.
                x=int(round(cx))-48; y=int(round(cy))-26
                crop=rgb[max(0,y):y+52,max(0,x):x+96]
                gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
                values.append({'t':t,'laplacian_variance':float(cv2.Laplacian(gray,cv2.CV_64F).var()),'mouth_width_px':float(np.linalg.norm(pts[61]-pts[291])), 'opening_px':float(np.linalg.norm(pts[13]-pts[14]))})
                for col,target in enumerate(TIMES):
                    if abs(t-target)<=.5/fps+1e-6 and col not in samples:
                        samples.append(col)
                        sheet.paste(Image.fromarray(crop).resize((192,104),Image.Resampling.NEAREST),(col*192,row*154+42))
                        draw.text((col*192+4,row*154+3),label,fill='black')
                        draw.text((col*192+4,row*154+20),f'audio {t:.2f}s',fill='black')
        cap.release()
        rows.append({'label':label,'path':str(path.relative_to(ROOT)),'fps':fps,'audio_offset_s':offset,'frames_detected':len(values),'median_laplacian_variance':float(np.median([v['laplacian_variance'] for v in values])),'median_mouth_width_px':float(np.median([v['mouth_width_px'] for v in values])),'samples':values})
    sheet.save(OUT/'teeth-comparison-20260917.png')
    (OUT/'teeth-diagnostics-20260917.json').write_text(json.dumps({'execution':'CPU analysis of retained RTX 4070 SUPER videos, 2026-09-17','limitations':'Whole-mouth edge energy is not a tooth-detail/accuracy metric; pose, lip opening, facial hair and compression confound it. Audio aligned using retained receiver waveform offset; earlier offline clips begin at zero.','rows':rows},indent=2)+'\n')
    print(json.dumps([{k:v for k,v in r.items() if k!='samples'} for r in rows],indent=2))

if __name__=='__main__': main()
