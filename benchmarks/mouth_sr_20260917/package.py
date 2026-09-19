"""CPU-only comparison packaging for RTX 4070 SUPER SoulX/SR experiments."""
import argparse
import json
from pathlib import Path

def named_comparison(path):
    """Resolve descriptive comparison names from the shared rename manifest."""
    import json
    path = Path(path).resolve()
    benchmark_root = next(p for p in Path(__file__).resolve().parents if p.name == 'benchmarks')
    repo = benchmark_root.parent
    mapping = json.loads((benchmark_root / 'comparison-video-renames.json').read_text())
    return repo / mapping.get(str(path.relative_to(repo)), str(path.relative_to(repo)))
import subprocess
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ARMS = ['original','bicubic2','full_sr2','mouth_sr_native','mouth_sr2']
LABELS = ['Original','Bicubic 2x control','Full-frame SR 2x','Mouth SR native','Mouth SR 2x']
TIMES = [.5,1.5,2.5,3.5,4.5,6.5,8.5]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture',required=True);args=parser.parse_args()
    root=ROOT/args.capture
    result=json.loads((root/'enhancement.json').read_text());assert result['status']=='complete'
    w,h=result['capture']['profile']['width'],result['capture']['profile']['height']
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    sheet=Image.new('RGB',(7*256,5*200),'#141923');draw=ImageDraw.Draw(sheet)
    centers=[]
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True) as mesh:
        for t in TIMES:
            im=np.array(Image.open(root/'original'/f'frame-{t:.1f}s.png'))
            pred=mesh.process(im);assert pred.multi_face_landmarks
            marks=pred.multi_face_landmarks[0].landmark
            centers.append((round((marks[13].x+marks[14].x)*w/2),round((marks[13].y+marks[14].y)*h/2)))
    for row,(arm,label) in enumerate(zip(ARMS,LABELS)):
        scale=1 if arm in ('original','mouth_sr_native') else 2
        for col,(t,(x,y)) in enumerate(zip(TIMES,centers)):
            im=Image.open(root/arm/f'frame-{t:.1f}s.png')
            crop=im.crop(((x-64)*scale,(y-40)*scale,(x+64)*scale,(y+40)*scale))
            sheet.paste(crop.resize((256,160),Image.Resampling.NEAREST),(col*256,row*200+40))
            draw.text((col*256+4,row*200+10),f'{label} / {t}s',font=font,fill='white')
    sheet.save(root/'mouth-comparison.png')
    # Fixed overview crop centers from the first saved sample; sheets track each time.
    x,y=centers[0];inputs=[];filters=[]
    for i,(arm,label) in enumerate(zip(ARMS,LABELS)):
        scale=1 if arm in ('original','mouth_sr_native') else 2
        inputs+=['-i',str(root/arm/'video.mp4')]
        panel_h=round(h*320/w)//2*2
        filters += [f'[{i}:v]setpts=PTS-STARTPTS,split=2[f{i}][m{i}]',
            f"[f{i}]scale=320:{panel_h}:flags=lanczos,pad=320:{panel_h+40}:0:40:color=0x141923,drawtext=text='{label}':fontcolor=white:fontsize=17:x=8:y=10[top{i}]",
            f'[m{i}]crop={128*scale}:{80*scale}:{(x-64)*scale}:{(y-40)*scale},scale=320:200:flags=neighbor[bottom{i}]',
            f'[top{i}][bottom{i}]vstack=inputs=2[p{i}]']
    filters+=['[p0][p1][p2][p3][p4]hstack=inputs=5[out]']
    subprocess.run(['ffmpeg','-y','-v','error',*inputs,'-filter_complex',';'.join(filters),
        '-map','[out]','-map','0:a:0','-c:v','libx264','-crf','16','-preset','medium',
        '-pix_fmt','yuv420p','-c:a','copy','-t','10','-movflags','+faststart',str(named_comparison(root/'comparison.mp4'))],check=True)
    media=[]
    for path in sorted(root.glob('*/video.mp4'))+[named_comparison(root/'comparison.mp4')]:
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(path)]))
        video=next(s for s in probe['streams'] if s['codec_type']=='video')
        assert int(video['nb_read_frames'])==250 and video['avg_frame_rate']=='25/1'
        assert any(s['codec_type']=='audio' for s in probe['streams'])
        if path.name=='video.mp4':
            scale=1 if path.parent.name in ('original','mouth_sr_native') else 2
            assert (video['width'],video['height'])==(w*scale,h*scale)
        subprocess.run(['ffmpeg','-v','error','-i',str(path),'-f','null','-'],check=True)
        media.append(dict(path=str(path),streams=probe['streams']))
    (root/'media-validation.json').write_text(json.dumps(media,indent=2)+'\n')
    print('Validated',args.capture,len(media),'videos',flush=True)


if __name__=='__main__':main()
