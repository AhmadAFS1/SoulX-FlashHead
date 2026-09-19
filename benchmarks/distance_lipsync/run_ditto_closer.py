"""Ditto framing replication; run with /workspace/.venvs/ditto/bin/python."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ditto', type=Path, default=Path('/workspace/ditto-talkinghead'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--vad-alpha', type=float, default=0.5)
    args = ap.parse_args()
    if not 0 <= args.vad_alpha <= 1:
        ap.error('--vad-alpha must lie between 0 and 1')
    root = Path(__file__).resolve().parent
    repo = root.parents[1]
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.ditto))
    from inference import StreamSDK, seed_everything
    refs = {
        'close-100': root/'evidence-indian-male-20260916/reference-close.png',
        'closer-125': root/'evidence-indian-male-closer-20260916/reference-closer-125.png',
        'closest-150': root/'evidence-indian-male-closer-20260916/reference-closest-150.png',
    }
    audio_path = repo/'benchmarks/comparison-10s.wav'
    audio, _ = librosa.load(str(audio_path), sr=16000)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    cfg = args.ditto/'checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl'
    models = args.ditto/'checkpoints/ditto_trt_Ampere_Plus'
    result = dict(date=datetime.now(timezone.utc).isoformat(), status='running',
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv,noheader'],text=True).strip(),
        torch=torch.__version__, cuda=torch.version.cuda, seed=50, fps=25,
        vad_alpha=args.vad_alpha, fade_type='s', fade_out_frames=15,
        config=str(cfg), config_sha256=digest(cfg), models=str(models),
        ditto_revision=subprocess.check_output(['git','-C',str(args.ditto),'rev-parse','HEAD'],text=True).strip(),
        audio_sha256=digest(audio_path), references={k:dict(path=str(v),sha256=digest(v)) for k,v in refs.items()}, rows=[])
    def save():
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    seed_everything(50)
    sdk = StreamSDK(str(cfg), str(models))
    for label, ref in refs.items():
        seed_everything(50)
        output = args.output/f'{label}-seed-50.mp4'
        sdk.setup(str(ref),str(output),fade_type='s',
            fade_out_keys=('exp','pitch','yaw','roll','t'),overall_ctrl_info={'vad_alpha':args.vad_alpha})
        sdk.setup_Nd(N_d=int(np.ceil(len(audio)/16000*25)),fade_out=15)
        start=time.perf_counter()
        sdk.audio2motion_queue.put(sdk.wav2feat.wav2feat(audio))
        sdk.close()
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(output)+'.tmp.mp4','-i',str(audio_path),
            '-map','0:v','-map','1:a','-c:v','copy','-c:a','aac',str(output)],check=True)
        result['rows'].append(dict(label=label,video=output.name,sha256=digest(output),seconds=time.perf_counter()-start))
        save()
        print(result['rows'][-1],flush=True)
    result['status']='complete'
    save()


if __name__=='__main__':
    main()
