"""CPU packaging of RTX 4070 SUPER seed/shift/history experiments."""
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
from package import probe

ROOT = Path(__file__).resolve().parent


def main():
    data = json.loads((ROOT / 'other-parameters/results.json').read_text())
    assert data['status'] == 'complete'
    assert data['baseline_matches_strength_control'] and data['history_adapter_identity']
    groups = {
        'seeds': [(f'Seed {s}', ROOT / (f'seed{s}-strength1' if s in [50, 51] else f'other-parameters/seed{s}')) for s in [50, 51, 52, 53]],
        'shift': [(f'Shift {s}', ROOT / ('other-parameters/baseline' if s == 5 else f'other-parameters/shift{s}')) for s in [1, 3, 5, 7]],
        'history': [(f'History {s} latents', ROOT / ('other-parameters/history2-control' if s == 2 else f'other-parameters/history{s}')) for s in [1, 2, 3]],
    }
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    checks = {}
    for name, rows in groups.items():
        command = ['ffmpeg', '-y', '-v', 'error']
        filters = []
        sheet = Image.new('RGB', (7 * 160, len(rows) * 320), '#141923')
        draw = ImageDraw.Draw(sheet)
        for i, (label, folder) in enumerate(rows):
            checks[str(folder.relative_to(ROOT))] = probe(folder / 'video.mp4')
            command += ['-i', str(folder / 'video.mp4')]
            filters += [f'[{i}:v]setpts=PTS-STARTPTS,split=2[f{i}][m{i}]',
                f"[f{i}]pad=320:624:0:48:color=0x141923,drawtext=text='{label}':fontcolor=white:fontsize=19:x=10:y=16[top{i}]",
                f"[m{i}]crop=128:80:96:208,scale=320:200:flags=neighbor,pad=320:232:0:32:color=0x141923,drawtext=text='Mouth detail (2.5x)':fontcolor=white:fontsize=16:x=10:y=8[bottom{i}]",
                f'[top{i}][bottom{i}]vstack=inputs=2[p{i}]']
            for j, t in enumerate([0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5]):
                im = Image.open(folder / f'frame-{t:.1f}s.png')
                sheet.paste(im.resize((160, 288), Image.Resampling.LANCZOS), (j * 160, i * 320 + 32))
                draw.text((j * 160 + 4, i * 320 + 4), f'{label} / {t:.1f}s', fill='white')
        sheet.save(ROOT / f'frames-{name}.png')
        filters.append(''.join(f'[p{i}]' for i in range(len(rows))) + f'hstack=inputs={len(rows)}[out]')
        output = named_comparison(ROOT / f'comparison-{name}.mp4')
        command += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-map', '0:a:0',
                    '-c:v', 'libx264', '-crf', '16', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                    '-c:a', 'copy', '-t', '10', '-movflags', '+faststart', str(output)]
        subprocess.run(command, check=True)
        checks[output.name] = probe(output)
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True)
    (ROOT / 'other-parameters/validation.json').write_text(json.dumps(dict(execution='CPU packaging and media validation', videos=checks), indent=2) + '\n')
    print('Validated seed, shift and history comparisons.', flush=True)


if __name__ == '__main__':
    main()
