"""CPU-only evidence inventory for a proposed custom mouth refiner; no training."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import urllib.request

import av
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
BANK = Path('/workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1')
RECORDING = Path('benchmarks/concurrency_15fps_20260916/bf16-batch5-c5-recorded-peer0.mp4')


def main():
    evidence = dict(date_utc=datetime.now(timezone.utc).isoformat(),
        execution='CPU media inspection and public repository metadata audit; no GPU inference or training',
        gpu_snapshot=subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,memory.used,driver_version',
                                             '--format=csv,noheader'], text=True).strip(),
        residents=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                           '--format=csv,noheader'], text=True).strip(),
        exact_soulx_recording=str(RECORDING),
        recording_sha256=hashlib.sha256(RECORDING.read_bytes()).hexdigest(),
        assets=[], public_repositories={})
    for path in sorted((BANK/'certified').glob('*.mp4')):
        probe=json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-show_entries',
            'format=duration:stream=codec_type,width,height,avg_frame_rate,nb_frames', '-of', 'json', str(path)], text=True))
        evidence['assets'].append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), probe=probe))
    for repo in ['TKing-Su/OrthoNet-io', 'ojinai/kit-example']:
        def fetch(suffix):
            req=urllib.request.Request('https://api.github.com/repos/'+repo+suffix,
                                      headers={'User-Agent':'SoulX-mouth-refiner-feasibility'})
            with urllib.request.urlopen(req, timeout=20) as response:
                return json.load(response)
        meta=fetch(''); branch=meta['default_branch']; tree=fetch('/git/trees/'+branch+'?recursive=1')
        evidence['public_repositories'][repo]=dict(branch=branch, commit=tree['sha'],
            paths=[entry['path'] for entry in tree['tree']], releases=[r['tag_name'] for r in fetch('/releases')])
    times=(1.,3.,5.,7.,9.,11.)
    sheet=Image.new('RGB', (240*len(times), 444*2), 'white'); draw=ImageDraw.Draw(sheet)
    for row, filename in enumerate(['speaking_direct_v14_subtle.mp4','speaking_direct_v15_reference_paced.mp4']):
        wanted={round(t*24):col for col,t in enumerate(times)}
        with av.open(str(BANK/'certified'/filename)) as source:
            for index, frame in enumerate(source.decode(video=0)):
                if index in wanted:
                    col=wanted[index]; image=frame.to_image()
                    sheet.paste(image.resize((240,416)), (col*240,row*444+28))
                    draw.text((col*240+4,row*444+3),f'{filename.split("_")[2]} source @ {times[col]:.0f}s', fill='black')
    sheet.save(ROOT/'existing-character-speaking-bank-samples.jpg', quality=95)
    evidence['sample_times_s']=list(times)
    evidence['capacity_arithmetic']=dict(historical_generator_fps=77.36,
        historical_profile='320x576 BF16 batch5, four steps, RTX 4070 SUPER; not a fresh refiner run',
        five_streams_15fps_serial_extra_ms_per_output=(1/75-1/77.36)*1000,
        note='Illustrative serial arithmetic only; CPU/GPU overlap and changed batching require measurement')
    (ROOT/'audit.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({k:v for k,v in evidence.items() if k not in ['assets','public_repositories']},indent=2))


if __name__=='__main__':
    main()
