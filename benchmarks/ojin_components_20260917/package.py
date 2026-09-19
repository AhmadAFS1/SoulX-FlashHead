"""CPU packaging/media validation of three equal-1024 delivery treatments."""
import json
from pathlib import Path
import subprocess
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent/'matched-runtime'
TIMES=(.5,1.5,2.5,3.5,4.5,6.5,8.5)


def main():
    validation=dict(execution='CPU packaging and media validation; see parent README/results for GPU run provenance',media={})
    for seed in (50,51):
        names=[f'soulx-LITE-bicubic2-512-to-1024-seed{seed}',
               f'soulx-LITE-legacy-general4x-resized2x-512-to-1024-seed{seed}',
               f'soulx-LITE-facelandmarker-full-native2x-TRT-512-to-1024-seed{seed}']
        labels=['LITE bicubic 2x','Earlier public SR 4x resized to 2x','Native 2x TRT + FaceLandmarker; NO mouth refiner']
        videos=[ROOT/n/(n+'.mp4') for n in names]
        prefix=ROOT/f'soulx-LITE-bicubic-vs-oldSR-vs-native2xTRT-FaceLandmarker-512to1024-seed{seed}'
        out=prefix.with_suffix('.mp4')
        cmd=['ffmpeg','-y','-v','error']
        filters=[]
        for i,(path,label) in enumerate(zip(videos,labels)):
            cmd+=['-i',str(path)]
            filters.append(f"[{i}:v]drawtext=text='{label}':x=12:y=12:fontsize=26:fontcolor=white:borderw=2:bordercolor=black[v{i}]")
        filters.append('[v0][v1][v2]hstack=inputs=3[out]')
        cmd+=['-filter_complex_threads','1','-filter_complex',';'.join(filters),'-map','[out]','-map','0:a:0',
            '-c:v','libx264','-threads','4','-crf','18','-preset','veryfast','-pix_fmt','yuv420p',
            '-c:a','copy','-t','10','-movflags','+faststart',str(out)]
        subprocess.run(cmd,check=True)
        tracked=json.loads((ROOT/names[-1]/'mouth-landmarks.json').read_text())
        mouths=Image.new('RGB',(224*len(TIMES),175*3),'white');d=ImageDraw.Draw(mouths)
        full=Image.new('RGB',(256*len(TIMES),284*3),'white');fd=ImageDraw.Draw(full)
        for row,name in enumerate(names):
            for col,t in enumerate(TIMES):
                im=Image.open(ROOT/name/f'frame-{t:.1f}s.png').convert('RGB')
                pts=np.asarray(tracked[round(t*25)])*2
                cx,cy=np.round((pts.min(axis=0)+pts.max(axis=0))/2).astype(int)
                mouths.paste(im.crop((cx-112,cy-62,cx+112,cy+62)),(col*224,row*175+48))
                d.text((col*224+3,row*175+3),f'{["Bicubic 2x","Earlier 4x SR -> 2x","Native 2x TRT (public weights)"][row]}\n{t}s; pre-encode PNG',fill='black')
                full.paste(im.resize((256,256)),(col*256,row*284+28))
                fd.text((col*256+3,row*284+3),f'{["Bicubic","Earlier SR","Native 2x TRT"][row]} {t}s',fill='black')
        mouths.save(prefix.parent/(prefix.name+'-mouths.png'))
        full.save(prefix.parent/(prefix.name+'-frames.png'))
        print('PACKAGED',out,flush=True)
    for path in ROOT.rglob('*.mp4'):
        info=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries',
            'stream=codec_type,width,height,avg_frame_rate,duration,nb_read_frames','-of','json',str(path)],text=True))
        v=next(s for s in info['streams'] if s['codec_type']=='video')
        assert int(v['nb_read_frames'])==250 and v['avg_frame_rate']=='25/1'
        assert any(s['codec_type']=='audio' for s in info['streams'])
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
        validation['media'][str(path)]=info
    (ROOT/'media-validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('Validated',len(validation['media']),'videos',flush=True)


if __name__=='__main__':main()
