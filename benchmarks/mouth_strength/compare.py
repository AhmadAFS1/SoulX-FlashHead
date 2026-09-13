"""Compose labeled existing renders and summarize repeatability/throughput."""
import argparse
import json
import statistics
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    root = args.directory
    data = json.loads((root/'results.json').read_text())
    assert data['status'] == 'complete'
    summaries = []
    for avatar in ['male', 'girl']:
        for strength in data['levels']:
            rows = [r for r in data['rows'] if r['avatar'] == avatar and r['strength'] == strength]
            assert len(rows) == 4 and all(r['frames'] == 250 and r['finite'] for r in rows)
            exact = all(len({r['raw_sha256'] for r in rows if r['seed'] == s}) == 1 for s in data['seeds'])
            summaries.append(dict(avatar=avatar, strength=strength,
                median_useful_fps=statistics.median(r['useful_fps'] for r in rows),
                max_torch_mib=max(r['peak_torch_mib'] for r in rows), repeats_exact=exact))
        for seed in data['seeds']:
            cmd = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-filter_complex_threads', '1']
            filters = []
            for i, level in enumerate(data['levels']):
                cmd += ['-threads', '1', '-i', str(root/f'{avatar}-seed-{seed}-strength-{level:g}.mp4')]
                filters.append(f"[{i}:v]drawbox=x=0:y=0:w=iw:h=32:color=black:t=fill,"
                               f"drawtext=text='Audio strength {level:g}':x=8:y=8:fontsize=18:fontcolor=white[v{i}]")
            filters.append('[v0][v1][v2][v3]hstack=inputs=4[out]')
            cmd += ['-filter_complex', ';'.join(filters), '-map', '[out]', '-map', '0:a',
                    '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-preset', 'veryfast',
                    '-c:a', 'copy', '-movflags', '+faststart', str(root/f'{avatar}-seed-{seed}-comparison.mp4')]
            subprocess.run(cmd, check=True)
            # Fully decode each composite and inspect actual frame count/duration.
            subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(root/f'{avatar}-seed-{seed}-comparison.mp4'),
                            '-f','null','-'],check=True)
            info = json.loads(subprocess.check_output(['ffprobe','-v','error','-count_frames',
                '-select_streams','v:0','-show_entries','stream=width,height,nb_read_frames,duration',
                '-of','json',str(root/f'{avatar}-seed-{seed}-comparison.mp4')]))['streams'][0]
            assert int(info['nb_read_frames']) == 250 and info['width'] == 1280 and info['height'] == 576
    (root/'summary.json').write_text(json.dumps(dict(rows=summaries, identity_control_exact=data['identity_control_exact'],
        identity_control_mae_255=data['identity_control_mae_255'],
        comparisons_decoded=True, note='Offline raw-generation comparisons; no WebRTC or objective lip-sync scoring.'), indent=2))
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
