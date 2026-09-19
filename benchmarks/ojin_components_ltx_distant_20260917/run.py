"""Postprocess the existing distant LTX Indian-man clip with disclosed components."""
from datetime import datetime,timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import subprocess
import time

import av
import cv2
import numpy as np
from PIL import Image
import torch
import tensorrt as trt

from benchmarks.ojin_components_20260917.run import Monitor
from benchmarks.roi_hybrid_analysis_20260917.benchmark import snapshot,processes
from soulx_rtc.face_landmarker import FaceLandmarkerTracker
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.mouth_sr import SRVGGUpscaler
from soulx_rtc.srvgg_trt import Native2xTRT,digest

ROOT=Path(__file__).resolve().parent
SOURCE=Path('/workspace/LTX-2.3/lumatalk_completed_videos_20260701T025914Z/remake/generated_clips/11_indian_man_speaking.mp4')
TASK=Path('models/ojin-components/face_landmarker.task')
ENGINE=Path('models/ojin-components/native2x-srvgg-480x832-trt10.16-cu13.engine')
OLD_WEIGHTS=Path('models/mouth-sr/realesr-general-x4v3.pth')
TIMES=(.5,1.5,2.5,3.5,4.5,6.5,7.5)


class Recorder:
    def __init__(self,path):
        self.path=path;self.tmp=path.with_name(path.stem+'-silent.mp4')
        self.container=av.open(str(self.tmp),'w')
        self.stream=self.container.add_stream('libx264',rate=24)
        self.stream.width,self.stream.height=960,1664
        self.stream.pix_fmt='yuv420p';self.stream.options={'crf':'18','preset':'veryfast','threads':'4'}
    def write(self,rgb,index):
        frame=av.VideoFrame.from_ndarray(rgb,format='rgb24');frame.pts=index;frame.time_base=Fraction(1,24)
        for packet in self.stream.encode(frame):self.container.mux(packet)
    def close(self):
        for packet in self.stream.encode():self.container.mux(packet)
        self.container.close()
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(self.tmp),'-i',str(SOURCE),
            '-map','0:v:0','-map','1:a:0?','-c:v','copy','-c:a','aac','-shortest','-movflags','+faststart',str(self.path)],check=True)
        self.tmp.unlink()


def frames():
    with av.open(str(SOURCE)) as source:
        for frame in source.decode(video=0):yield frame.to_ndarray(format='rgb24')


def main():
    assert not (ROOT/'results.json').exists(),'Preserve prior evidence'
    import mediapipe as mp
    cv2.setNumThreads(1);torch.set_num_threads(4)
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries',
        'format=duration:stream=codec_type,width,height,avg_frame_rate,nb_frames','-of','json',str(SOURCE)],text=True))
    result=dict(status='starting',date_utc=datetime.now(timezone.utc).isoformat(),
        execution='Postprocessing of an existing LTX-made H.264/AAC video; no new LTX or SoulX generation',
        source=str(SOURCE),source_sha256=digest(SOURCE),source_probe=probe,
        gpu=snapshot(),physical_vram='12 GB class; visible capacity in GPU snapshot',processes=processes(),
        torch=torch.__version__,cuda=torch.version.cuda,tensorrt=trt.__version__,mediapipe=mp.__version__,
        task=str(TASK),task_sha256=digest(TASK),engine=str(ENGINE),engine_sha256=digest(ENGINE),
        native2x_weights='models/ojin-components/2xNomosUni_compact_multijpg_ldl.safetensors',
        native2x_weights_identity='Public Philip Hofmann substitute; NOT Ojin checkpoint',
        proprietary_mouth_refiner='unavailable and omitted',runs=[])
    def save():(ROOT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    with acquire_gpu_lease():
        old=SRVGGUpscaler(OLD_WEIGHTS)
        native=Native2xTRT(ENGINE)
        sample=next(frames());old.upscale(sample,2);native.upscale(sample)
        for mode in ('bicubic2','legacy-general4x-resized2x','FaceLandmarker-native2x-TRT'):
            name=f'LTX23-distant-indian-man-{mode}-480x832-to-960x1664'
            folder=ROOT/name;folder.mkdir(exist_ok=False)
            recorder=Recorder(folder/(name+'.mp4'))
            tracker=FaceLandmarkerTracker(TASK,fps=24) if mode.startswith('Face') else None
            detections=[];widths=[];latencies=[];count=0;output_hash=hashlib.sha256()
            torch.cuda.reset_peak_memory_stats()
            with Monitor() as monitor:
                start=time.perf_counter()
                for index,rgb in enumerate(frames()):
                    torch.cuda.synchronize();t=time.perf_counter();points=None
                    if mode=='bicubic2':out=cv2.resize(rgb,(960,1664),interpolation=cv2.INTER_CUBIC)
                    elif mode.startswith('legacy'):out=old.upscale(rgb,2)
                    else:
                        points=tracker(rgb);out=native.upscale(rgb)
                    torch.cuda.synchronize();latencies.append(time.perf_counter()-t)
                    if tracker:
                        detections.append(None if points is None else points.tolist())
                        if points is not None:widths.append(float(np.ptp(points[:,0])))
                    output_hash.update(out.tobytes());recorder.write(out,index);count+=1
                    for timestamp in TIMES:
                        if index==round(timestamp*24):Image.fromarray(out).save(folder/f'frame-{timestamp:.1f}s.png')
                elapsed=time.perf_counter()-start
            recorder.close()
            if tracker:tracker.close();(folder/'mouth-landmarks.json').write_text(json.dumps(detections)+'\n')
            row=dict(mode=mode,frames=count,processing_s=sum(latencies),processing_fps=count/sum(latencies),
                processing_wall_with_encoding_s=elapsed,fps_note='Per-frame decode-to-output processing excludes H264 encoding; wall includes synchronous recorder writes',
                output_raw_sha256=output_hash.hexdigest(),video=str((folder/(name+'.mp4')).relative_to(ROOT)),
                video_sha256=digest(folder/(name+'.mp4')),torch_peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                torch_peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
                gpu_after=snapshot(),samples=monitor.samples)
            if tracker:row.update(detected=sum(p is not None for p in detections),mouth_width_px_mean=float(np.mean(widths)),mouth_width_px_p05=float(np.percentile(widths,5)),mouth_width_px_p95=float(np.percentile(widths,95)))
            result['runs'].append(row);save();print(json.dumps({k:v for k,v in row.items() if k not in ('samples',)}),flush=True)
        result['status']='complete';save()


if __name__=='__main__':main()
