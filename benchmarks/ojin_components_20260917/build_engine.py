"""Build and validate a public native-2x SRVGG engine on this actual GPU."""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time
import numpy as np
import onnx
from PIL import Image
import torch
import tensorrt as trt
from soulx_rtc.srvgg_trt import load_native2x, Native2xTRT, digest
from soulx_rtc.gpu_lease import acquire_gpu_lease
from benchmarks.roi_hybrid_analysis_20260917.benchmark import snapshot, processes

ROOT=Path(__file__).resolve().parent
WEIGHTS=Path('models/ojin-components/2xNomosUni_compact_multijpg_ldl.safetensors')
ENGINE=Path('models/ojin-components/native2x-srvgg-512-fp16.engine')


def main(root=ROOT, engine=ENGINE, width=512, height=512, validation_inputs=()):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    engine=Path(engine)
    if engine.exists():raise FileExistsError('Preserve the validated engine')
    with acquire_gpu_lease():
        torch.set_num_threads(4)
        record=dict(date_utc=datetime.now(timezone.utc).isoformat(),gpu=snapshot(),
            physical_vram='12 GB class; visible bytes in nvidia-smi snapshot',
            gpu_name=torch.cuda.get_device_name(),processes=processes(),torch=torch.__version__,
            cuda=torch.version.cuda,tensorrt=trt.__version__,onnx=onnx.__version__,
            weights=str(WEIGHTS),weights_sha256=digest(WEIGHTS),
            identity='Public Philip Hofmann 2xNomosUni substitute; NOT Ojin checkpoint/engine',
            input_shape=[1,3,height,width],precision='FP16 enabled; FP32 input/output',
            status='building')
        (root/'build.json').write_text(json.dumps(record,indent=2)+'\n')
        model=load_native2x(WEIGHTS)
        onnx_path=engine.with_suffix('.onnx')
        torch.onnx.export(model,torch.zeros(1,3,height,width),str(onnx_path),
            input_names=['input'],output_names=['output'],opset_version=17,dynamo=False)
        onnx.checker.check_model(onnx.load(str(onnx_path)))
        logger=trt.Logger(trt.Logger.WARNING)
        builder=trt.Builder(logger)
        network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(network,logger)
        if not parser.parse(onnx_path.read_bytes()):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config=builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,512*1024**2)
        config.builder_optimization_level=3
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        start=time.perf_counter();serialized=builder.build_serialized_network(network,config)
        if serialized is None:raise RuntimeError('TensorRT build failed')
        record['build_s']=time.perf_counter()-start
        engine.write_bytes(bytes(serialized))
        record['engine_sha256']=digest(engine);record['onnx_sha256']=digest(onnx_path)
        engine.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        runtime=Native2xTRT(engine)
        inspector=runtime.engine.create_engine_inspector()
        (root/'engine-layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        model=model.cuda().float();checks=[]
        paths=list(map(Path,validation_inputs))
        if not paths and (width,height)==(512,512):
            paths=[Path('benchmarks/square_teeth_20260917/512-25/reference.png')]
            paths += [Path('benchmarks/mouth_sr_20260917/512-seed50/original')/f'frame-{t}s.png' for t in ['1.5','2.5','6.5']]
        with torch.inference_mode():
            for p in paths:
                rgb=np.array(Image.open(p).convert('RGB'))
                x=torch.from_numpy(rgb).permute(2,0,1)[None].cuda().float()/255
                reference=model(x).clamp(0,1)
                actual=runtime.tensor(x).clamp(0,1)
                error=(reference-actual).abs();mse=error.square().mean().item()
                row=dict(input=str(p),mae=error.mean().item(),max_error=error.max().item(),
                    psnr_db=-10*np.log10(max(mse,1e-20)),finite=bool(torch.isfinite(actual).all()))
                assert row['finite'] and row['psnr_db']>45 and row['mae']<.002,row
                checks.append(row)
        record.update(status='complete',validation=checks,gpu_after=snapshot(),
            engine_device_memory_bytes=runtime.engine.device_memory_size)
        engine.with_suffix('.json').write_text(json.dumps(record,indent=2)+'\n')
        (root/'build.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT)
    parser.add_argument('--engine',type=Path,default=ENGINE)
    parser.add_argument('--width',type=int,default=512)
    parser.add_argument('--height',type=int,default=512)
    parser.add_argument('--validation-input',action='append',default=[])
    args=parser.parse_args();main(args.output,args.engine,args.width,args.height,args.validation_input)
