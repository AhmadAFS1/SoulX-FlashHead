"""CPU packaging of paired RTX 4070 SUPER inference; see per-run GPU evidence."""
import json, subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parent
NAMES=['neutral','smiling']
LABELS=['Neutral reference','Smiling reference']
TIMES=[0.5,1.5,2.5,3.5,4.5,6.5,8.5]
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
refs=Image.new('RGB',(640,620),'#141923'); draw=ImageDraw.Draw(refs)
mouths=Image.new('RGB',(7*256,2*200),'#141923'); md=ImageDraw.Draw(mouths)
frames=Image.new('RGB',(7*160,2*318),'#141923'); fd=ImageDraw.Draw(frames)
rows=[]
for row,(name,label) in enumerate(zip(NAMES,LABELS)):
    result=json.loads((ROOT/name/'results.json').read_text()); assert result['status']=='complete'
    refs.paste(Image.open(ROOT/name/'reference.png'),(row*320,44));draw.text((row*320+10,12),label,font=font,fill='white')
    for col,t in enumerate(TIMES):
        im=Image.open(ROOT/name/f'frame-{t:.1f}s.png')
        mouths.paste(im.crop((96,208,224,288)).resize((256,160),Image.Resampling.NEAREST),(col*256,row*200+40))
        md.text((col*256+5,row*200+10),f'{name} / {t:.1f}s',font=font,fill='white')
        frames.paste(im.resize((160,288)),(col*160,row*318+30));fd.text((col*160+4,row*318+5),f'{name} {t:.1f}s',fill='white')
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(ROOT/name/'video.mp4')]))
    v=next(s for s in probe['streams'] if s['codec_type']=='video')
    assert (v['width'],v['height'],v['avg_frame_rate'],int(v['nb_read_frames']))==(320,576,'25/1',250)
    rows.append({'name':name,'run':result,'media':v})
assert rows[0]['run']['profile']==rows[1]['run']['profile']
assert rows[0]['run']['audio_sha256']==rows[1]['run']['audio_sha256']
refs.save(ROOT/'references.png');mouths.save(ROOT/'mouth-comparison.png');frames.save(ROOT/'frames-comparison.png')
filters=[]
for i,label in enumerate(LABELS):
    filters.extend([f'[{i}:v]setpts=PTS-STARTPTS,split=2[f{i}][m{i}]',f'[f{i}]pad=320:624:0:48:color=0x141923,drawtext=text=\'{label} | 320 x 576\':fontcolor=white:fontsize=17:x=10:y=16[top{i}]',f'[m{i}]crop=128:80:96:208,scale=320:200:flags=neighbor,pad=320:232:0:32:color=0x141923,drawtext=text=\'Mouth detail (2.5x)\':fontcolor=white:fontsize=16:x=10:y=8[bottom{i}]',f'[top{i}][bottom{i}]vstack=inputs=2[p{i}]'])
filters.append('[p0][p1]hstack=inputs=2[out]')
subprocess.run(['ffmpeg','-y','-v','error','-i',str(ROOT/'neutral/video.mp4'),'-i',str(ROOT/'smiling/video.mp4'),'-filter_complex',';'.join(filters),'-map','[out]','-map','0:a:0','-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p','-c:a','copy','-t','10','-movflags','+faststart',str(ROOT/'soulx-LITE-neutral-vs-smiling-reference-indian-man-1.25x-seed50.mp4')],check=True)
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames','-show_streams','-of','json',str(ROOT/'soulx-LITE-neutral-vs-smiling-reference-indian-man-1.25x-seed50.mp4')]))
v=next(s for s in probe['streams'] if s['codec_type']=='video');assert int(v['nb_read_frames'])==250
assert any(s['codec_type']=='audio' for s in probe['streams'])
subprocess.run(['ffmpeg','-v','error','-i',str(ROOT/'soulx-LITE-neutral-vs-smiling-reference-indian-man-1.25x-seed50.mp4'),'-f','null','-'],check=True)
(ROOT/'summary.json').write_text(json.dumps({'execution':'Fresh SoulXFlash GPU inference on NVIDIA GeForce RTX 4070 SUPER, 12282 MiB visible VRAM; CPU packaging and media validation. Imagegen edit hardware unverified.','rows':rows,'comparison_media':probe},indent=2)+'\n')
print('Validated paired 250-frame 320x576 outputs and comparison with audio.')
