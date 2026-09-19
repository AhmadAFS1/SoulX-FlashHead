"""Diagnostic only: same MuseTalk portrait, actual zero-waveform audio.

GPU hardware/runtime are recorded from this run; CPU landmarks are proxies.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from PIL import Image, ImageDraw

from soulx_rtc.engine import Engine
from soulx_rtc.experiment import record

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/source/sample_ai_human_facetime_v1.png')


def query(args):
    return subprocess.check_output(['nvidia-smi', *args], text=True)


def measure(frames):
    rows = []
    with mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
            refine_landmarks=True, min_detection_confidence=.5, min_tracking_confidence=.5) as mesh:
        for rgb in frames:
            result = mesh.process(rgb)
            if not result.multi_face_landmarks:
                rows.append(None)
                continue
            marks = result.multi_face_landmarks[0].landmark
            def p(i):
                return np.array([marks[i].x * rgb.shape[1], marks[i].y * rgb.shape[0]])
            eye = p(263) - p(33)
            rows.append(dict(opening=float(np.linalg.norm(p(13)-p(14))/np.linalg.norm(eye)),
                             roll=float(np.degrees(np.arctan2(eye[1], eye[0]))),
                             eye_mid=((p(263)+p(33))/2).tolist()))
    return rows


def main():
    output = ROOT / 'results.json'
    if output.exists():
        raise RuntimeError('Existing results; use a new directory.')
    cv2.setNumThreads(1)
    data = dict(date_utc=datetime.now(timezone.utc).isoformat(),
        execution='Fresh local GPU inference; CPU MediaPipe analysis of generated RGB frames',
        gpu=query(['--query-gpu=name,memory.total,memory.used,driver_version','--format=csv']),
        co_resident=query(['--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        torch=torch.__version__, cuda=torch.version.cuda,
        source=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        profile=dict(width=480,height=832,fps=24,seconds=8,steps=4,shift=5,history=2,
                     audio='128000 exact zero float32 waveform samples, encoded through Wav2Vec',
                     compiled=False,int8_weights=True,compute='BF16',allocator_cap_mib=6144),
        limitation='Lip gap and roll are 2D landmark proxies, not proof of physical closure or 3D pose. Two seeds cannot establish a general guarantee.',
        runs=[])
    def save():
        output.write_text(json.dumps(data,indent=2)+'\n')
    save()
    try:
        engine = Engine(width=480,height=832,steps=4,fps=24,compile_model=False,
                        optimized=True,real_rope=True,lean=True,fused_qkv=True,
                        memory_mode='compact',int8_weights=True,cuda_memory_mib=6144)
        audio = np.zeros(128000,np.float32)
        reference = np.array(Image.open(SOURCE).convert('RGB').resize((480,832)))
        data['reference_landmarks'] = measure([reference])[0]
        for seed in [50,51]:
            state = engine.prepare(str(SOURCE),audio,seed)
            chunks = []
            while state.cursor < state.total_frames:
                chunks.append(engine.generate([state])[0])
            frames = np.concatenate(chunks)[:192]
            path = ROOT / f'silence-seed{seed}.mp4'
            record(path,[frames],audio,24)
            series = measure(frames)
            valid = [(i,r) for i,r in enumerate(series) if r is not None]
            gaps = np.array([r['opening'] for _,r in valid])
            rolls = np.array([r['roll'] for _,r in valid])
            worst = sorted(valid,key=lambda v:v[1]['opening'],reverse=True)[:4]
            indices = sorted(set([0,24,48,72,96,120,144,168,191]+[i for i,_ in worst]))
            sheet = Image.new('RGB',(240*5,444*((len(indices)+4)//5)), 'white')
            draw = ImageDraw.Draw(sheet)
            for j,i in enumerate(indices):
                x,y = (j%5)*240,(j//5)*444
                sheet.paste(Image.fromarray(frames[i]).resize((240,416)),(x,y+28))
                label = f'{i/24:.2f}s gap={series[i]["opening"]:.4f}' if series[i] else f'{i/24:.2f}s no detection'
                draw.text((x+4,y+5),label,fill='black')
            sheet.save(ROOT/f'silence-seed{seed}-frames.jpg')
            row = dict(seed=seed,video=str(path),frames=len(frames),detected=len(valid),
                       lip_gap_mean=float(gaps.mean()),lip_gap_max=float(gaps.max()),
                       lip_gap_p95=float(np.percentile(gaps,95)),
                       roll_range_p95_p5=float(np.percentile(rolls,95)-np.percentile(rolls,5)),
                       worst_frames=[i for i,_ in worst],series=series)
            data['runs'].append(row)
            save()
            print(json.dumps({k:v for k,v in row.items() if k!='series'}),flush=True)
        data['status']='complete'
        save()
    except Exception as exc:
        data['status']='failed'
        data['error']=repr(exc)
        save()
        raise


if __name__ == '__main__':
    main()
