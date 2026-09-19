"""CPU comparison/packaging of retained Ditto 0.5 and 0.7 framing runs."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

from analyze import correlation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('baseline', type=Path)
    parser.add_argument('candidate', type=Path)
    args = parser.parse_args()
    runs = [json.loads((p / 'results.json').read_text()) for p in (args.baseline, args.candidate)]
    for key in ('gpu', 'torch', 'cuda', 'seed', 'fps', 'fade_type', 'fade_out_frames',
                'config_sha256', 'ditto_revision', 'audio_sha256', 'references'):
        assert runs[0][key] == runs[1][key], f'Mismatched control: {key}'
    metrics = [json.loads((p / 'mouth-motion.json').read_text()) for p in (args.baseline, args.candidate)]
    labels = ['close-100', 'closer-125', 'closest-150']
    comparisons = []
    for label in labels:
        pair = [next(r for r in m['rows'] if r['label'] == label) for m in metrics]
        for end, name in ((250, 'all_frames'), (235, 'before_final_15_frame_fade')):
            series = [r['opening_series'][:end] for r in pair]
            assert all(len(s) == end and all(v is not None for v in s) for s in series)
            means = [float(np.mean(s)) for s in series]
            p95s = [float(np.percentile(s, 95)) for s in series]
            comparisons.append(dict(label=label, window=name, frames=end,
                mean_opening=means, p95_opening=p95s,
                mean_change_pct=100 * (means[1] / means[0] - 1),
                p95_change_pct=100 * (p95s[1] / p95s[0] - 1),
                zero_lag_correlation=correlation(*series, 0)))
    cmd = ['ffmpeg', '-v', 'error', '-y']
    filters = []
    index = 0
    for directory, run in zip((args.baseline, args.candidate), runs):
        for label, zoom in zip(labels, ('1.00', '1.25', '1.50')):
            cmd += ['-i', str(directory / f'{label}-seed-50.mp4')]
            filters.append(f"[{index}:v]pad=iw:ih+36:0:36:black,drawtext=text='Ditto {zoom}x - alpha {run['vad_alpha']:g}':x=6:y=9:fontsize=16:fontcolor=white[v{index}]")
            index += 1
    filters += ['[v0][v1][v2]hstack=inputs=3[top]', '[v3][v4][v5]hstack=inputs=3[bottom]',
                '[top][bottom]vstack=inputs=2[v]']
    output = args.candidate / 'ditto-alpha-0p5-vs-0p7.mp4'
    subprocess.run(cmd + ['-filter_complex', ';'.join(filters), '-map', '[v]', '-map', '0:a',
        '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', '-c:a', 'copy', str(output)], check=True)
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', str(output), '-vf',
        'select=eq(n\\,75)', '-frames:v', '1', str(args.candidate / 'alpha-comparison-frame75.jpg')], check=True)
    validation = []
    for video in sorted(args.candidate.glob('*.mp4')):
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True)
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames',
            '-show_streams', '-show_format', '-of', 'json', str(video)], text=True))
        v = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        assert int(v['nb_read_frames']) == 250
        assert v['avg_frame_rate'] == '25/1'
        a = next((s for s in probe['streams'] if s['codec_type'] == 'audio'), None)
        if not video.name.endswith('.tmp.mp4'):
            assert a is not None and abs(float(a['duration']) - 10) < 0.05
        validation.append(dict(video=video.name, sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
            frames=int(v['nb_read_frames']), width=v['width'], height=v['height'],
            video_duration=v['duration'], audio_duration=a['duration'] if a else None,
            full_decode='pass'))
    report = dict(gpu_inference_provenance=[r['gpu'] for r in runs],
        analysis='CPU NumPy statistics and FFmpeg; no new GPU inference in this script',
        alpha_order=[r['vad_alpha'] for r in runs], matched_controls='pass',
        limits='One avatar/audio/seed; no replicate determinism test or phoneme scoring. Correlation is motion agreement, not audio sync.',
        comparisons=comparisons, validation=validation)
    (args.candidate / 'alpha-comparison.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
