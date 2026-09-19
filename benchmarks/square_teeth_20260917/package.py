"""CPU packaging of retained RTX 4070 SUPER square-resolution inference."""
import json
from pathlib import Path
import subprocess
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parent
NAMES=['512-15','512-25','1024-15','1024-25']
TIMES=[0.5,1.5,2.5,3.5,4.5,6.5,8.5]

def main():
    sheet=Image.new('RGB',(7*200,4*225),'white');draw=ImageDraw.Draw(sheet)
    rows=[]
    for r,name in enumerate(NAMES):
        folder=ROOT/name;data=json.loads((folder/'results.json').read_text())
        assert data['status']=='complete',data
        probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_entries','stream=codec_type,width,height,avg_frame_rate,nb_read_frames','-of','json',str(folder/'video.mp4')],text=True))
        vs=next(s for s in probe['streams'] if s['codec_type']=='video')
        assert int(vs['nb_read_frames'])==data['frames']
        rows.append({'name':name,'generation':data,'decoded_video':vs})
        for c,t in enumerate(TIMES):
            im=Image.open(folder/f'frame-{t:.1f}s.png')
            sheet.paste(im.resize((200,200),Image.Resampling.LANCZOS),(c*200,r*225+25))
            draw.text((c*200+5,r*225+5),f'{name} / {t:.1f}s',fill='black')
    sheet.save(ROOT/'frames-comparison.png')
    (ROOT/'summary.json').write_text(json.dumps({'execution':'Fresh GPU generation on RTX 4070 SUPER 12282 MiB, driver 595.84, Torch 2.7.1+cu128/CUDA 12.8, 2026-09-17; CPU packaging and full-frame decode validation.','rows':rows},indent=2)+'\n')
    for r in rows: print(r['name'],{k:r['generation'][k] for k in ['wall_s','useful_fps','peak_allocated_mib','peak_reserved_mib']})

if __name__=='__main__':main()
