"""Replay matched raw RGB for legacy SR and FaceMesh controls; not fresh Lite FPS."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import cv2
import librosa
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
import torch
from soulx_rtc.experiment import record
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.mouth_sr import SRVGGUpscaler, MouthEnhancer
from soulx_rtc.srvgg_trt import digest
from benchmarks.roi_hybrid_analysis_20260917.benchmark import snapshot, processes

ROOT=Path(__file__).resolve().parent/'matched-runtime'
TIMES=(.5,1.5,2.5,3.5,4.5,6.5,8.5)


def main():
    assert not (ROOT/'controls.json').exists()
    torch.set_num_threads(4);cv2.setNumThreads(1)
    audio,_=librosa.load('benchmarks/comparison-10s.wav',sr=16000)
    result=dict(date_utc=datetime.now(timezone.utc).isoformat(),gpu=snapshot(),
        physical_vram='12 GB class; visible capacity in snapshot',processes=processes(),
        torch=torch.__version__,cuda=torch.version.cuda,mediapipe=mp.__version__,
        execution='Replay GPU SR / CPU FaceMesh control on saved matched raw RGB; Lite is NOT resident',runs=[])
    with acquire_gpu_lease():
        model=SRVGGUpscaler('models/mouth-sr/realesr-general-x4v3.pth')
        model.upscale(np.zeros((512,512,3),np.uint8),2)
        for seed in (50,51):
            raw_path=ROOT/f'soulx-LITE-lite-only-512-to-512-seed{seed}'/'raw.npy'
            frames=np.load(raw_path,mmap_mode='r')
            landmarks_path=ROOT/f'soulx-LITE-facelandmarker-full-native2x-TRT-512-to-1024-seed{seed}'/'mouth-landmarks.json'
            actual=json.loads(landmarks_path.read_text())
            tracker=MouthEnhancer(None)  # FaceMesh only; no synthesis or blend.
            detections=[];times=[];distances=[]
            for i,frame in enumerate(frames):
                t=time.perf_counter();points=tracker._detect(np.array(frame));times.append(time.perf_counter()-t)
                detections.append(None if points is None else points.tolist())
                if points is not None and actual[i] is not None:
                    distances.append(float(np.linalg.norm(points-np.asarray(actual[i]),axis=1).mean()))
            tracker.close()
            tracking=dict(seed=seed,facemesh_detected=sum(p is not None for p in detections),
                facelandmarker_detected=sum(p is not None for p in actual),facemesh_total_s=sum(times),
                landmark_disagreement_mean_px=float(np.mean(distances)),
                landmark_disagreement_p95_px=float(np.percentile(distances,95)),
                note='Disagreement is not an accuracy score; no landmark ground truth')
            (ROOT/f'facemesh-control-seed{seed}.json').write_text(json.dumps(dict(summary=tracking,points=detections),indent=2)+'\n')
            sheet=Image.new('RGB',(512*len(TIMES),512),'white')
            for col,ts in enumerate(TIMES):
                idx=round(ts*25);img=Image.fromarray(frames[idx]);draw=ImageDraw.Draw(img)
                for points,color in [(detections[idx],'cyan'),(actual[idx],'yellow')]:
                    if points:
                        for x,y in points:draw.ellipse((x-1,y-1,x+1,y+1),fill=color)
                draw.text((8,8),f'{ts}s cyan FaceMesh / yellow FaceLandmarker',fill='white',stroke_width=1,stroke_fill='black')
                sheet.paste(img,(col*512,0))
            sheet.save(ROOT/f'facemesh-vs-FaceLandmarker-seed{seed}.png')
            for mode in ('bicubic2','legacy-general4x-resized2x'):
                name=f'soulx-LITE-{mode}-512-to-1024-seed{seed}'
                folder=ROOT/name;folder.mkdir(exist_ok=False);outputs=[];durations=[]
                torch.cuda.reset_peak_memory_stats()
                for frame in frames:
                    frame=np.array(frame);torch.cuda.synchronize();t=time.perf_counter()
                    out=cv2.resize(frame,(1024,1024),interpolation=cv2.INTER_CUBIC) if mode=='bicubic2' else model.upscale(frame,2)
                    torch.cuda.synchronize();durations.append(time.perf_counter()-t);outputs.append(out)
                outputs=np.stack(outputs)
                path=folder/(name+'.mp4');record(path,[outputs],audio,25)
                for ts in TIMES:Image.fromarray(outputs[round(ts*25)]).save(folder/f'frame-{ts:.1f}s.png')
                result['runs'].append(dict(seed=seed,mode=mode,processing_s=sum(durations),
                    fps=250/sum(durations),video=str(path.relative_to(ROOT)),video_sha256=digest(path),
                    raw_input_file_sha256=digest(raw_path),torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                    tracker_comparison=tracking))
                del outputs
                (ROOT/'controls.json').write_text(json.dumps(result,indent=2)+'\n')
                print(seed,mode,250/sum(durations),flush=True)
    result['status']='complete';(ROOT/'controls.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
