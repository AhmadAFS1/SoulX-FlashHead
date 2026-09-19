"""CPU-only labeled comparison, native mouth samples and media validation."""
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
TIMES=(.5,1.5,2.5,3.5,4.5,6.5,7.5)
NAMES=['LTX23-distant-indian-man-bicubic2-480x832-to-960x1664',
       'LTX23-distant-indian-man-legacy-general4x-resized2x-480x832-to-960x1664',
       'LTX23-distant-indian-man-FaceLandmarker-native2x-TRT-480x832-to-960x1664']
LABELS=['LTX source + bicubic 2x','LTX source + earlier 4x SR resized 2x',
        'LTX source + FaceLandmarker + native 2x TRT; no refiner']
PREFIX='LTX23-distant-indian-man-bicubic-vs-oldSR-vs-FaceLandmarker-native2xTRT-no-refiner'


def main():
    videos=[ROOT/name/(name+'.mp4') for name in NAMES]
    out=ROOT/(PREFIX+'.mp4')
    cmd=['ffmpeg','-y','-v','error']
    filters=[]
    for i,(path,label) in enumerate(zip(videos,LABELS)):
        cmd+=['-i',str(path)]
        filters.append(f"[{i}:v]drawtext=text='{label}':x=12:y=14:fontsize=27:fontcolor=white:borderw=2:bordercolor=black[v{i}]")
    filters.append('[v0][v1][v2]hstack=inputs=3[out]')
    cmd+=['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]','-map','0:a:0?',
        '-c:v','libx264','-threads','4','-crf','18','-preset','veryfast','-pix_fmt','yuv420p',
        '-c:a','copy','-movflags','+faststart',str(out)]
    subprocess.run(cmd,check=True)
    landmarks=json.loads((ROOT/NAMES[-1]/'mouth-landmarks.json').read_text())
    mouths=Image.new('RGB',(224*len(TIMES),175*3),'white');draw=ImageDraw.Draw(mouths)
    full=Image.new('RGB',(144*len(TIMES),278*3),'white');fd=ImageDraw.Draw(full)
    for row,(name,label) in enumerate(zip(NAMES,LABELS)):
        for col,t in enumerate(TIMES):
            image=Image.open(ROOT/name/f'frame-{t:.1f}s.png').convert('RGB')
            points=np.asarray(landmarks[round(t*24)])*2
            cx,cy=np.round((points.min(0)+points.max(0))/2).astype(int)
            mouths.paste(image.crop((cx-112,cy-62,cx+112,cy+62)),(col*224,row*175+48))
            draw.text((col*224+3,row*175+3),f'{label}\n{t}s pre-encode PNG',fill='black')
            full.paste(image.resize((144,250)),(col*144,row*278+28))
            fd.text((col*144+3,row*278+3),f'{label[:20]} {t}s',fill='black')
    mouths.save(ROOT/(PREFIX+'-mouths.png'));full.save(ROOT/(PREFIX+'-frames.png'))
    validation={'execution':'CPU comparison packaging and full media decode; no GPU inference','media':{}}
    for path in [*videos,out]:
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
            'stream=codec_type,width,height,avg_frame_rate,duration,nb_read_frames','-of','json',str(path)],text=True))
        video=next(s for s in info['streams'] if s['codec_type']=='video')
        assert int(video['nb_read_frames'])==201 and video['avg_frame_rate']=='24/1'
        assert any(s['codec_type']=='audio' for s in info['streams'])
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
        validation['media'][str(path)]=info
    (ROOT/'media-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print(out)


if __name__=='__main__':main()
