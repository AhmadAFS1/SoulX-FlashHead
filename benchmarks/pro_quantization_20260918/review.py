"""CPU media/landmark review and matched, labeled PRO comparison artifacts."""
import argparse
import json
from pathlib import Path
import subprocess

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw


def inspect(directory):
    video = directory / "video.mp4"
    cap = cv2.VideoCapture(str(video))
    rows, crops = [], []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=.5, min_tracking_confidence=.5) as mesh:
        index = 0
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prediction = mesh.process(rgb)
            row = dict(frame=index, detected=bool(prediction.multi_face_landmarks))
            if prediction.multi_face_landmarks:
                points = np.array([(p.x*rgb.shape[1],p.y*rgb.shape[0]) for p in prediction.multi_face_landmarks[0].landmark])
                opening = np.linalg.norm(points[13]-points[14])
                center = points[[13,14]].mean(0)
                width = np.linalg.norm(points[61]-points[291])
                cx, cy = np.round(center).astype(int)
                oral = cv2.cvtColor(rgb[max(0,cy-max(3,round(opening*.75))):cy+max(3,round(opening*.75))+1,
                    max(0,cx-max(8,round(width*.32))):cx+max(8,round(width*.32))+1],cv2.COLOR_RGB2GRAY)
                row.update(opening_px=float(opening), opening_norm=float(opening/max(np.linalg.norm(points[33]-points[263]),1e-6)),
                           oral_edge_energy=float(cv2.Laplacian(oral,cv2.CV_64F).var()),
                           mouth_center=center.tolist(), mouth_width=float(width))
                # Consistent square face boxes for an optional relative sync diagnostic.
                lo, hi = points.min(0), points.max(0)
                size = max(hi-lo)*1.12; midpoint=(lo+hi)/2
                row['face_box']=[max(0,int(midpoint[0]-size/2)),max(0,int(midpoint[1]-size/2)),
                                 min(rgb.shape[1],int(midpoint[0]+size/2)),min(rgb.shape[0],int(midpoint[1]+size/2))]
            rows.append(row); index += 1
    cap.release()
    detected=[r for r in rows if r['detected']]
    opened=[r for r in detected if r['opening_px']>=4]
    return dict(decoded_frames=len(rows), detected_frames=len(detected), open_frames=len(opened),
                median_open_oral_edge_energy=float(np.median([r['oral_edge_energy'] for r in opened])) if opened else None,
                median_opening_norm=float(np.median([r['opening_norm'] for r in detected])), frames=rows)


def probe(path):
    return json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
        'stream=codec_type,width,height,avg_frame_rate,duration,nb_read_frames','-of','json',str(path)],text=True))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--baseline',type=Path,required=True)
    ap.add_argument('--candidate',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--label',default='Optimized PRO FP8')
    args=ap.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    cv2.setNumThreads(1)
    result=dict(execution='CPU FFmpeg/MediaPipe analysis of fresh GPU outputs; no new generation',
                gpu_source='NVIDIA GeForce RTX 4070 SUPER, 12 GB class / 12,282 MiB; see input results.json',
                limitations='Edge energy measures contrast/detail, not dental correctness; landmark motion agreement is not lip-sync certification.', inputs={})
    for name,directory in [('stock',args.baseline),('candidate',args.candidate)]:
        result['inputs'][name]=dict(path=str(directory),media=probe(directory/'video.mp4'),
                                   generation=json.loads((directory/'results.json').read_text()),analysis=inspect(directory))
    a,b=[result['inputs'][n]['analysis'] for n in ('stock','candidate')]
    assert a['decoded_frames']==b['decoded_frames']
    paired=[(x,y) for x,y in zip(a['frames'],b['frames']) if x['detected'] and y['detected']]
    opened=[(x,y) for x,y in paired if x['opening_px']>=4 and y['opening_px']>=4]
    result['paired']=dict(frames=len(paired),both_open_frames=len(opened),
        opening_correlation=float(np.corrcoef([x['opening_norm'] for x,y in paired],[y['opening_norm'] for x,y in paired])[0,1]),
        median_mouth_center_distance_px=float(np.median([np.linalg.norm(np.array(x['mouth_center'])-y['mouth_center']) for x,y in paired])),
        median_edge_ratio_on_both_open=float(np.median([y['oral_edge_energy']/max(x['oral_edge_energy'],1e-6) for x,y in opened])) if opened else None)
    comparison=args.output/'comparison.mp4'
    labels=['Stock PRO BF16',args.label]
    filters=';'.join(f"[{i}:v]drawtext=text='{label}':x=10:y=12:fontsize=20:fontcolor=white:borderw=2:bordercolor=black[v{i}]" for i,label in enumerate(labels))
    filters+=';[v0][v1]hstack=inputs=2[v]'
    subprocess.run(['ffmpeg','-v','error','-y','-i',str(args.baseline/'video.mp4'),'-i',str(args.candidate/'video.mp4'),
        '-filter_complex',filters,'-map','[v]','-map','0:a:0','-c:v','libx264','-crf','16','-preset','fast',
        '-pix_fmt','yuv420p','-c:a','copy',str(comparison)],check=True)
    result['comparison']=probe(comparison)
    times=([.5,9.5,19.5,29.5,39.5,49.5,59.5] if a["decoded_frames"] >= 1500 else [.5,1.5,2.5,3.5,4.5,6.5,8.5])
    sheet=Image.new('RGB',(len(times)*192,300),'white');draw=ImageDraw.Draw(sheet)
    full=Image.new('RGB',(len(times)*160,626),'white');fd=ImageDraw.Draw(full)
    for row,(name,directory,label) in enumerate([('stock',args.baseline,labels[0]),('candidate',args.candidate,labels[1])]):
        samples=result['inputs'][name]['analysis']['frames']
        for col,t in enumerate(times):
            path=directory/f'frame-{t:.1f}s.png'
            if not path.exists(): continue
            image=Image.open(path).convert('RGB')
            center=samples[round(t*25)].get('mouth_center')
            if center is not None:
                cx,cy=np.round(center).astype(int)
                crop=image.crop((cx-48,cy-26,cx+48,cy+26)).resize((192,104),Image.Resampling.NEAREST)
                sheet.paste(crop,(col*192,row*150+42))
            draw.text((col*192+4,row*150+3),label,fill='black')
            draw.text((col*192+4,row*150+20),f'Raw RGB {t:.1f}s',fill='black')
            full.paste(image.resize((160,288),Image.Resampling.LANCZOS),(col*160,row*313+25))
            fd.text((col*160+3,row*313+5),f'{name} {t:.1f}s',fill='black')
    sheet.save(args.output/'mouth-comparison.png');full.save(args.output/'frames-comparison.png')
    (args.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'paired':result['paired'],'summary':{n:{k:v for k,v in result['inputs'][n]['analysis'].items() if k!='frames'} for n in ('stock','candidate')}},indent=2))


if __name__=='__main__':main()
