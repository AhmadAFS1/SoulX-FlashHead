"""Matched GPU SR / CPU ROI postprocessing of fresh lossless SoulX RGB captures."""
import argparse
from datetime import datetime, timezone
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
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.mouth_sr import SRVGGUpscaler, MouthEnhancer

ROOT = Path(__file__).resolve().parent
WEIGHTS = Path('models/mouth-sr/realesr-general-x4v3.pth')
TIMES = (.5,1.5,2.5,3.5,4.5,6.5,8.5)


class Recorder:
    def __init__(self, directory, width, height):
        self.directory = directory
        self.tmp = directory/'silent.mp4'
        self.container = av.open(str(self.tmp), 'w')
        self.stream = self.container.add_stream('libx264', rate=25)
        self.stream.width, self.stream.height = width, height
        self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf':'18', 'preset':'veryfast', 'threads':'2'}
    def write(self, rgb, index):
        frame = av.VideoFrame.from_ndarray(rgb, format='rgb24')
        frame.pts, frame.time_base = index, Fraction(1,25)
        for packet in self.stream.encode(frame): self.container.mux(packet)
    def close(self):
        for packet in self.stream.encode(): self.container.mux(packet)
        self.container.close()
        subprocess.run(['ffmpeg','-y','-v','error','-i',str(self.tmp),'-i','benchmarks/comparison-10s.wav',
            '-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-t','10',
            '-movflags','+faststart',str(self.directory/'video.mp4')], check=True)
        self.tmp.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture', required=True)
    args = parser.parse_args()
    root = ROOT/args.capture
    if (root/'enhancement.json').exists(): raise FileExistsError('Preserve existing results')
    frames = np.load(root/'raw.npy', mmap_mode='r')
    capture = json.loads((root/'capture.json').read_text())
    assert capture['status'] == 'complete'
    cv2.setNumThreads(1); torch.set_num_threads(4)
    lease = acquire_gpu_lease()
    result = dict(status='starting', date_utc=datetime.now(timezone.utc).isoformat(),
        execution='GPU SR with CPU MediaPipe tracking/composition on raw pre-encoding RGB',
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv'],text=True),
        coresident=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True),
        physical_vram='12 GB class; visible capacity from nvidia-smi',
        torch=torch.__version__, cuda_runtime=torch.version.cuda,
        model='Official general-x4v3 SRVGG, native 4x, area downsample to requested output; not Ojin weights',
        weights=str(WEIGHTS), weights_sha256=hashlib.sha256(WEIGHTS.read_bytes()).hexdigest(),
        capture=capture, runs=[])
    def save(): (root/'enhancement.json').write_text(json.dumps(result,indent=2)+'\n')
    save()
    model = SRVGGUpscaler(WEIGHTS)
    model.upscale(np.array(frames[0]),2)
    model.upscale(np.array(frames[0,:128,:128]),1)
    for arm in ('original','bicubic2','full_sr2','mouth_sr_native','mouth_sr2'):
        out = root/arm; out.mkdir(exist_ok=False)
        scale = 1 if arm in ('original','mouth_sr_native') else 2
        h,w = frames.shape[1:3]
        recorder = Recorder(out,w*scale,h*scale)
        enhancer = MouthEnhancer(model, scale=scale, strength=.65) if arm.startswith('mouth') else None
        rows=[];latencies=[]; digest=hashlib.sha256()
        torch.cuda.reset_peak_memory_stats()
        try:
            for index, raw in enumerate(frames):
                raw=np.array(raw)
                torch.cuda.synchronize();start=time.perf_counter()
                meta={'detected':False,'applied':False}
                if arm == 'original': output=raw.copy()
                elif arm == 'bicubic2': output=cv2.resize(raw,(w*2,h*2),interpolation=cv2.INTER_CUBIC)
                elif arm == 'full_sr2': output=model.upscale(raw,2)
                else: output,meta=enhancer.process_frame(raw)
                torch.cuda.synchronize();latencies.append(time.perf_counter()-start)
                digest.update(output.tobytes());rows.append(meta)
                for timestamp in TIMES:
                    if index == round(timestamp*25):
                        Image.fromarray(output).save(out/f'frame-{timestamp:.1f}s.png')
                recorder.write(output,index)
                if index % 50 == 0:print(args.capture,arm,index,flush=True)
            recorder.close()
        finally:
            if enhancer:enhancer.close()
        row=dict(arm=arm,frames=len(frames),scale=scale,processing_s=sum(latencies),
            mean_ms=1000*np.mean(latencies),p95_ms=1000*np.percentile(latencies,95),
            fps=len(frames)/sum(latencies),applied=sum(x['applied'] for x in rows),
            detected=sum(x['detected'] for x in rows),raw_sha256=digest.hexdigest(),
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        (out/'tracking.json').write_text(json.dumps(rows)+'\n')
        result['runs'].append(row);save();print(json.dumps(row),flush=True)
    result['status']='complete';save();lease.close()


if __name__ == '__main__':main()
