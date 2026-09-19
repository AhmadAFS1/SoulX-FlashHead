"""CPU media review/packaging and per-frame lip-motion comparison."""
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image,ImageDraw
from prepare import decode,anchor,encode

ROOT=Path(__file__).resolve().parent


def main():
    cv2.setNumThreads(1)
    names=['pro-base','pro-redub-B','silent-redub-B']
    paths=[ROOT/'pro-base.mp4',ROOT/'musetalk/pro-redub-B.mp4',ROOT/'musetalk/silent-redub-B.mp4']
    clips={name:decode(path) for name,path in zip(names,paths)}
    stats={}; landmarks={}
    for name,frames in clips.items():
        values=[];marks=[]
        with mp.solutions.face_mesh.FaceMesh(max_num_faces=1,refine_landmarks=True,
                min_detection_confidence=.5,min_tracking_confidence=.5) as mesh:
            for rgb in frames:
                result=mesh.process(rgb)
                if not result.multi_face_landmarks:
                    values.append(None);marks.append(None);continue
                pts=np.array([(p.x*rgb.shape[1],p.y*rgb.shape[0]) for p in result.multi_face_landmarks[0].landmark])
                values.append(float(np.linalg.norm(pts[13]-pts[14])/np.linalg.norm(pts[33]-pts[263])))
                marks.append(pts)
        stats[name]=dict(frames=len(frames),detected=sum(v is not None for v in values),opening=values)
        landmarks[name]=marks
    region=slice(25,225)
    correlations={}
    for a,b in [('pro-base','pro-redub-B'),('pro-redub-B','silent-redub-B'),('pro-base','silent-redub-B')]:
        pairs=[(x,y) for x,y in zip(stats[a]['opening'][region],stats[b]['opening'][region]) if x is not None and y is not None]
        correlations[a+' versus '+b]=float(np.corrcoef(np.asarray(pairs).T)[0,1])
    opening=np.array(stats['pro-base']['opening'],float)-np.array(stats['pro-redub-B']['opening'],float)
    selected=[25,50,75,100,125,150,175,200]
    selected+=list(25+np.argsort(np.abs(opening[25:225]))[-4:])
    selected=sorted(set(int(i) for i in selected))
    sheet=Image.new('RGB',(192*len(selected),160*3),'white');draw=ImageDraw.Draw(sheet)
    for row,name in enumerate(names):
        for col,i in enumerate(selected):
            rgb=clips[name][i];pts=landmarks[name][i]
            if pts is None:continue
            cx,cy=np.round(pts[[13,14]].mean(0)).astype(int)
            crop=Image.fromarray(rgb).crop((cx-40,cy-23,cx+40,cy+23)).resize((192,110),Image.Resampling.NEAREST)
            sheet.paste(crop,(col*192,row*160+48))
            draw.text((col*192+3,row*160+3),f'{name}\n{i/25:.2f}s gap {stats[name]["opening"][i]:.3f}',fill='black')
    sheet.save(ROOT/'mouth-comparison.png')
    full=Image.new('RGB',(160*8,312*3),'white');draw=ImageDraw.Draw(full)
    for row,name in enumerate(names):
        for col,i in enumerate([25,50,75,100,125,150,175,200]):
            full.paste(Image.fromarray(clips[name][i]).resize((160,288)),(col*160,row*312+24))
            draw.text((col*160+3,row*312+3),f'{name} {i/25:.0f}s',fill='black')
    full.save(ROOT/'frame-comparison.jpg')
    cmd=['ffmpeg','-v','error','-n']
    for path in paths:cmd+=['-i',str(path)]
    cmd+=['-i',str(ROOT/'audio-B.wav')]
    labels=['PRO source - old speech A','MuseTalk redub - speech B','Silent base + MuseTalk B']
    filters=[f"[{i}:v]drawtext=text='{label}':x=6:y=10:fontsize=14:fontcolor=white:borderw=2:bordercolor=black[v{i}]" for i,label in enumerate(labels)]
    filters+=['[v0][v1][v2]hstack=inputs=3[out]']
    cmd+=['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]','-map','3:a:0',
          '-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-c:a','aac','-t','10','-movflags','+faststart',str(ROOT/'comparison-with-audio-B.mp4')]
    if not (ROOT/'comparison-with-audio-B.mp4').exists():
        subprocess.run(cmd,check=True)
    # Package the finished redub with the same anchor, outside active speech.
    ref=np.array(Image.open(ROOT/'reference-320x576.png'))
    anchored=ROOT/'pro-redub-B-anchored-silent.mp4'
    if not anchored.exists():
        encode(anchored,anchor(clips['pro-redub-B'],ref))
    finished=ROOT/'pro-redub-B-anchored.mp4'
    if not finished.exists():
        subprocess.run(['ffmpeg','-v','error','-n','-i',str(anchored),'-i',str(ROOT/'audio-B.wav'),
            '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-movflags','+faststart',str(finished)],check=True)
    final=decode(finished);base=clips['pro-base']
    assert np.array_equal(final[0],final[-1]) and np.array_equal(final[0],base[0])
    validation={}
    for path in [*paths,ROOT/'pro/video.mp4',ROOT/'comparison-with-audio-B.mp4',finished]:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
            'stream=codec_type,width,height,avg_frame_rate,nb_read_frames,duration','-of','json',str(path)],text=True))
        v=next(s for s in probe['streams'] if s['codec_type']=='video')
        assert int(v['nb_read_frames'])==250 and v['avg_frame_rate']=='25/1'
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
        validation[str(path.relative_to(ROOT))]=probe
    result=dict(execution='CPU MediaPipe landmarks, media packaging and full decode; no new video GPU inference',
        opening_definition='Inner lip distance / eye span; proxy, not phoneme or lip-sync ground truth',
        correlations_1_to_9_seconds=correlations,frames=stats,selected_review_frames=selected,
        final_packaging=dict(first_last_decoded_rgb_equal=True,same_anchor_as_pro_base=True,
            caveat='Raw MuseTalk output does not force identical endpoints. Only the packaged copy has six-frame handles and six-frame blends; all mouth/sync measurements use raw output.'),media=validation)
    (ROOT/'review-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(correlations),flush=True)


if __name__=='__main__':main()
