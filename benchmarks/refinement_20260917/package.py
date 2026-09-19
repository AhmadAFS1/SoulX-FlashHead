"""CPU-only packaging/validation of the recorded RTX 4070 SUPER GPU runs."""
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
import sys
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
ARMS = ['baseline4', 'continuous6', 'refine100', 'refine50']
LABELS = ['Normal 4 steps', 'Continuous 6 steps', '4 + refine 100/50', '4 + refine 50/25']
SUFFIX = ''
if '--strong' in sys.argv:
    ARMS = ['baseline4', 'continuous6', 'refine250', 'refine500']
    LABELS = ['Normal 4 steps', 'Continuous 6 steps', '4 + refine 250/125', '4 + refine 500/250']
    SUFFIX = '-strong'
TIMES = [.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5]
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
results = json.loads((ROOT/'results.json').read_text())
assert results['status'] == 'complete'
media = []
for seed in (50, 51):
    sheet = Image.new('RGB', (7*256, 4*200), '#141923')
    draw = ImageDraw.Draw(sheet)
    for row, (arm, label) in enumerate(zip(ARMS, LABELS)):
        for col, timestamp in enumerate(TIMES):
            im = Image.open(ROOT/f'seed{seed}'/arm/f'frame-{timestamp:.1f}s.png')
            sheet.paste(im.crop((96, 208, 224, 288)).resize((256, 160), Image.Resampling.NEAREST),
                        (col*256, row*200+40))
            draw.text((col*256+4, row*200+10), f'{label} / {timestamp}s', font=FONT, fill='white')
    sheet.save(ROOT/f'mouths-seed{seed}{SUFFIX}.png')
    inputs = []; filters = []
    for i, (arm, label) in enumerate(zip(ARMS, LABELS)):
        inputs += ['-i', str(ROOT/f'seed{seed}'/arm/'video.mp4')]
        filters += [f'[{i}:v]setpts=PTS-STARTPTS,split=2[f{i}][m{i}]',
            f"[f{i}]pad=320:624:0:48:color=0x141923,drawtext=text='{label}':fontcolor=white:fontsize=18:x=10:y=16[top{i}]",
            f'[m{i}]crop=128:80:96:208,scale=320:200:flags=neighbor[bottom{i}]',
            f'[top{i}][bottom{i}]vstack=inputs=2[p{i}]']
    filters += ['[p0][p1][p2][p3]hstack=inputs=4[out]']
    subprocess.run(['ffmpeg', '-y', '-v', 'error', *inputs, '-filter_complex', ';'.join(filters),
        '-map', '[out]', '-map', '0:a:0', '-c:v', 'libx264', '-crf', '16', '-preset', 'medium',
        '-pix_fmt', 'yuv420p', '-c:a', 'copy', '-t', '10', '-movflags', '+faststart',
        str(named_comparison(ROOT/f'comparison-seed{seed}{SUFFIX}.mp4'))], check=True)
for path in sorted(ROOT.glob('seed*/*/video.mp4')) + sorted(ROOT.glob('soulx-LITE-4steps-*.mp4')):
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_streams', '-of', 'json', str(path)]))
    video = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    assert int(video['nb_read_frames']) == 250 and video['avg_frame_rate'] == '25/1'
    assert any(s['codec_type'] == 'audio' for s in probe['streams'])
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'null', '-'], check=True)
    media.append(dict(path=str(path), streams=probe['streams']))
(ROOT/'media-validation.json').write_text(json.dumps(media, indent=2)+'\n')
cards = ''.join(f'<h2>{path.stem}</h2><video controls preload="metadata" src="{path.name}"></video>'
    f'<p><a href="mouths-{path.stem.rsplit("-", 1)[-1]}{"-strong" if "strong-refinement" in path.name else ""}.png">Full-size mouth crops</a></p>'
    for path in sorted(ROOT.glob('soulx-LITE-4steps-*.mp4')))
(ROOT/'review.html').write_text('''<!doctype html><meta charset="utf-8"><title>SoulX refinement comparison</title>
<style>body{background:#141923;color:#eee;font:18px system-ui;max-width:1400px;margin:24px auto;padding:16px}video{width:100%}a{color:#9cf}</style>
<h1>SoulX clean-latent refinement experiment</h1>
<p>Fresh NVIDIA GeForce RTX 4070 SUPER inference, 12 GB class / 12,282 MiB visible, CUDA 12.8, 2026-09-17.
Each clip uses the same portrait and audio, 320×576, 25 FPS. Lower panels are fixed mouth crops enlarged without sharpening.
Refinement labels are raw pre-shift timesteps. See README.md for runtime, limitations and findings.</p>''' + cards)
print(f'Validated {len(media)} videos; generated two synchronized comparisons and crop sheets.')
