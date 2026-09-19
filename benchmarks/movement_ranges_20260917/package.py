"""CPU comparison-video assembly and validation of RTX 4070 SUPER experiments."""
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

from PIL import Image, ImageDraw
from run import ROOT, SEEDS, REPEAT_SEEDS, TIMES, run_name


def probe(path, expected_frames=250):
    streams = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
        '-show_entries', 'stream=codec_type,width,height,avg_frame_rate,nb_read_frames,duration',
        '-of', 'json', str(path)]))['streams']
    video = next(s for s in streams if s['codec_type'] == 'video')
    assert int(video['nb_read_frames']) == expected_frames and video['avg_frame_rate'] == '25/1', (path, video)
    assert any(s['codec_type'] == 'audio' for s in streams), path
    return streams


def make_segment(name, rows, title):
    destination = ROOT / 'comparisons'
    destination.mkdir(exist_ok=True)
    command = ['ffmpeg', '-y', '-v', 'error', '-threads', '2']
    filters = []
    sheet = Image.new('RGB', (7 * 160, len(rows) * 320), '#141923')
    draw = ImageDraw.Draw(sheet)
    for i, (label, folder_name) in enumerate(rows):
        folder = ROOT / folder_name
        command += ['-i', str(folder / 'video.mp4')]
        filters += [f'[{i}:v]setpts=PTS-STARTPTS,split=2[f{i}][m{i}]',
            f"[f{i}]pad=320:624:0:48:color=0x141923,drawtext=text='{label}':fontcolor=white:fontsize=19:x=10:y=16[top{i}]",
            f"[m{i}]crop=128:80:96:208,scale=320:200:flags=neighbor,pad=320:232:0:32:color=0x141923,drawtext=text='Mouth detail (2.5x)':fontcolor=white:fontsize=16:x=10:y=8[bottom{i}]",
            f'[top{i}][bottom{i}]vstack=inputs=2[p{i}]']
        for j, t in enumerate(TIMES):
            im = Image.open(folder / f'frame-{t:.1f}s.png')
            sheet.paste(im.resize((160, 288), Image.Resampling.LANCZOS), (j * 160, i * 320 + 32))
            draw.text((j * 160 + 4, i * 320 + 4), f'{label} / {t:.1f}s', fill='white')
    sheet.save(destination / f'{name}-frames.png')
    filters.append(''.join(f'[p{i}]' for i in range(len(rows))) +
                   f"hstack=inputs={len(rows)},pad=iw:896:0:40:color=0x141923,drawtext=text='{title}':fontcolor=white:fontsize=21:x=12:y=10[out]")
    output = named_comparison(destination / f'{name}.mp4')
    command += ['-filter_complex_threads', '2', '-filter_complex', ';'.join(filters), '-map', '[out]', '-map', '0:a:0',
                '-c:v', 'libx264', '-threads', '2', '-crf', '17', '-preset', 'fast', '-pix_fmt', 'yuv420p',
                '-c:a', 'copy', '-t', '10', '-movflags', '+faststart', str(output)]
    subprocess.run(command, check=True)
    return output


def concatenate(name, segments):
    listing = ROOT / 'comparisons' / f'{name}-concat.txt'
    listing.write_text(''.join(f"file '{p.name}'\nduration 10.0\n" for p in segments))
    output = named_comparison(ROOT / f'{name}.mp4')
    # Join video streams exactly; attach a repeated copy of the original audio
    # once so AAC padding at each clip boundary cannot accumulate drift.
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
        '-stream_loop', str(len(segments) - 1), '-i', 'benchmarks/comparison-10s.wav',
        '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'copy', '-c:a', 'aac',
        '-t', str(10 * len(segments)), '-movflags', '+faststart', str(output)], check=True)
    streams = probe(output, 250 * len(segments))
    subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 'null', '-'], check=True)
    return streams


def main():
    data = json.loads((ROOT / 'results.json').read_text())
    assert data['status'] == 'complete'
    assert all(data['history_adapter']['baseline_identity'].values())
    assert len({r['raw_sha256'] for r in data['runs']}) == 38, 'Repeated output suggests an ineffective setting.'
    validation = dict(execution='CPU ffprobe frame counts and full ffmpeg comparison decode; input GPU provenance in results.json', individual={}, comparisons={})
    for row in data['runs']:
        streams = probe(ROOT / row['name'] / 'video.mp4')
        video = next(s for s in streams if s['codec_type'] == 'video')
        assert (video['width'], video['height']) == (320, 576)
        validation['individual'][row['name']] = streams
    groups = {
        'shift-low': [0.5, 1, 2, 5],
        'shift-mid': [3, 5, 7, 10],
        'shift-high': [5, 7, 10, 15],
    }
    for group, shifts in groups.items():
        segments = []
        for seed in REPEAT_SEEDS:
            rows = [(f'Shift {s:g}' + (' (default)' if s == 5 else ''), run_name(seed, shift=s)) for s in shifts]
            title = f'{group} | seed {seed} | history 2 | strength 1 | 320 x 576'
            segment = make_segment(f'{group}-seed{seed}', rows, title)
            segments.append(segment)
        validation['comparisons'][group] = concatenate(group, segments)
        print(f'Packaged {group}', flush=True)
    segments = []
    for seed in REPEAT_SEEDS:
        rows = [(f'History {h}' + (' (default)' if h == 2 else ''), run_name(seed, history=h)) for h in [1, 2, 3, 4]]
        segments.append(make_segment(f'history-seed{seed}', rows,
            f'Motion history | seed {seed} | shift 5 | strength 1 | 320 x 576'))
    validation['comparisons']['history'] = concatenate('history', segments)
    segments = []
    for group, seeds in enumerate([SEEDS[:4], SEEDS[4:]], 1):
        rows = [(f'Seed {seed}', run_name(seed)) for seed in seeds]
        segments.append(make_segment(f'seeds-group{group}', rows,
            f'Seed variation | group {group} of 2 | shift 5 | history 2 | strength 1'))
    validation['comparisons']['seeds'] = concatenate('seeds', segments)
    (ROOT / 'validation.json').write_text(json.dumps(validation, indent=2) + '\n')
    print('Validated all 38 runs and five overview videos.', flush=True)


if __name__ == '__main__':
    main()
