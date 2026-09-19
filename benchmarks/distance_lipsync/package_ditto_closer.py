"""Build labeled Ditto comparison and validate full video decoding."""
import subprocess
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

parser=argparse.ArgumentParser()
parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent/'evidence-ditto-closer-0p5-20260916')
args=parser.parse_args()
root=args.directory
alpha=json.loads((root/'results.json').read_text())['vad_alpha']
labels=['close-100','closer-125','closest-150']
cmd=['ffmpeg','-v','error','-y']
for label in labels:
    cmd += ['-i',str(root/f'{label}-seed-50.mp4')]
filters=[]
for i,zoom in enumerate(['1.00','1.25','1.50']):
    filters.append(f"[{i}:v]pad=iw:ih+36:0:36:black,drawtext=text='Ditto {zoom}x - vad_alpha {alpha:g}':x=6:y=9:fontsize=16:fontcolor=white[v{i}]")
filters.append('[v0][v1][v2]hstack=inputs=3[v]')
out=named_comparison(root/f'ditto-three-framings-{str(alpha).replace(".", "p")}.mp4')
subprocess.run(cmd+['-filter_complex',';'.join(filters),'-map','[v]','-map','0:a',
    '-c:v','libx264','-crf','18','-preset','veryfast','-c:a','copy',str(out)],check=True)
subprocess.run(['ffmpeg','-v','error','-y','-i',str(out),'-vf',
    "select='eq(n,25)+eq(n,75)+eq(n,125)+eq(n,175)+eq(n,225)',tile=1x5",
    '-frames:v','1',str(root/'contact.jpg')],check=True)
for video in root.glob('*.mp4'):
    subprocess.run(['ffmpeg','-v','error','-i',str(video),'-f','null','-'],check=True)
print(out)
