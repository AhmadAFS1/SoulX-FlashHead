"""CPU packaging and decode checks of fresh RTX 4070 SUPER strength runs."""
import json
import subprocess
from pathlib import Path

def named_comparison(path):
    """Resolve descriptive comparison names from the shared rename manifest."""
    import json
    path = Path(path).resolve()
    benchmark_root = next(p for p in Path(__file__).resolve().parents if p.name == 'benchmarks')
    repo = benchmark_root.parent
    mapping = json.loads((benchmark_root / 'comparison-video-renames.json').read_text())
    return repo / mapping.get(str(path.relative_to(repo)), str(path.relative_to(repo)))

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent


def probe(path):
    streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_entries', 'stream=codec_type,width,height,avg_frame_rate,nb_read_frames,duration',
        '-of', 'json', str(path)]))['streams']
    video = next(s for s in streams if s['codec_type'] == 'video')
    assert int(video['nb_read_frames']) == 250 and video['avg_frame_rate'] == '25/1'
    assert any(s['codec_type'] == 'audio' for s in streams)
    return streams


def main():
    data = json.loads((ROOT / 'results.json').read_text())
    assert data['status'] == 'complete'
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    validation = dict(execution='CPU ffprobe frame counting and complete ffmpeg decode; GPU inference provenance in results.json', videos={})
    for seed in data['seeds']:
        rows = [r for r in data['runs'] if r['seed'] == seed]
        command = ['ffmpeg', '-y', '-v', 'error']
        filters = []
        sheet = Image.new('RGB', (7 * 256, 4 * 200), '#141923')
        draw = ImageDraw.Draw(sheet)
        for i, row in enumerate(rows):
            folder = ROOT / row['directory']
            streams = probe(folder / 'video.mp4')
            video = next(s for s in streams if s['codec_type'] == 'video')
            assert (video['width'], video['height']) == (320, 576)
            validation['videos'][row['directory']] = streams
            command += ['-i', str(folder / 'video.mp4')]
            label = f"Strength {row['strength']:g} | seed {seed}"
            filters += [f'[{i}:v]setpts=PTS-STARTPTS,split=2[f{i}][m{i}]',
                f"[f{i}]pad=320:624:0:48:color=0x141923,drawtext=text='{label}':fontcolor=white:fontsize=19:x=10:y=16[top{i}]",
                f"[m{i}]crop=128:80:96:208,scale=320:200:flags=neighbor,pad=320:232:0:32:color=0x141923,drawtext=text='Mouth detail (2.5x)':fontcolor=white:fontsize=16:x=10:y=8[bottom{i}]",
                f'[top{i}][bottom{i}]vstack=inputs=2[p{i}]']
            for j, t in enumerate([0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5]):
                im = Image.open(folder / f'frame-{t:.1f}s.png')
                sheet.paste(im.crop((96, 208, 224, 288)).resize((256, 160), Image.Resampling.NEAREST), (j * 256, i * 200 + 40))
                draw.text((j * 256 + 5, i * 200 + 10), f"strength {row['strength']:g} / {t:.1f}s", font=font, fill='white')
        sheet.save(ROOT / f'mouths-seed{seed}.png')
        filters.append(''.join(f'[p{i}]' for i in range(4)) + 'hstack=inputs=4[out]')
        output = named_comparison(ROOT / f'comparison-seed{seed}.mp4')
        command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-map', '0:a:0',
                    '-c:v', 'libx264', '-crf', '16', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                    '-c:a', 'copy', '-t', '10', '-movflags', '+faststart', str(output)]
        subprocess.run(command, check=True)
        validation['videos'][output.name] = probe(output)
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True)
    (ROOT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print('Validated eight individual videos and two four-way comparisons with audio.', flush=True)


if __name__ == '__main__':
    main()
