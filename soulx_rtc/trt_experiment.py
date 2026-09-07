"""Capture real FFN inputs, export ONNX, build/reload TRT and compare to compiled Torch."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .trt_backend import TRTFeedForward, file_hash


def capture(args):
    from .engine import Engine
    from .server import decode_audio
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new capture directory; never mix activation provenance")
    engine = Engine(width=args.width,height=args.height,compile_model=False,optimized=True,
                    real_rope=True,memory_mode=args.memory_mode)
    state = engine.prepare(args.image,decode_audio(args.audio,3),50)
    output.mkdir(parents=True,exist_ok=True)
    handles = []
    captured = set()
    decode = engine.pipeline.vae.decode
    def capture_decode(latents):
        if not (output/"vae-input.pt").exists():
            torch.save(latents.detach().cpu(),output/"vae-input.pt")
        return decode(latents)
    engine.pipeline.vae.decode = capture_decode
    def hook(index):
        def save(module, inputs):
            if index not in captured:
                torch.save(inputs[0].detach().cpu(),output/f"ffn-{index:02d}-input.pt")
                captured.add(index)
        return save
    for index in args.layers:
        handles.append(engine.raw_model.blocks[index].ffn.register_forward_pre_hook(hook(index)))
    try:
        engine.generate([state])
    finally:
        engine.pipeline.vae.decode = decode
        for handle in handles:
            handle.remove()
    (output/"capture.json").write_text(json.dumps(dict(width=args.width,height=args.height,
        layers=sorted(captured),image_sha256=file_hash(args.image),audio_sha256=file_hash(args.audio),
        seed=50,fps=engine.fps,steps=engine.steps,batch=1,
        source="actual first-chunk native SoulX FFN activations"),indent=2)+"\n")
    print("CAPTURED",sorted(captured),flush=True)


def benchmark(function, x, repeats=50):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.no_grad():
        for _ in range(10):
            function(x)
        torch.cuda.synchronize()
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            function(x)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)/repeats


def build(args):
    import tensorrt as trt
    import onnx
    from safetensors import safe_open
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    weights = Path("models/SoulX-FlashHead-1_3B/Model_Lite/diffusion_pytorch_model.safetensors")
    digest = file_hash(weights)
    dtype = {"bf16":torch.bfloat16,"fp16":torch.float16}[args.precision]
    reports = json.loads((output/"report.json").read_text()) if args.resume and (output/"report.json").exists() else []
    for index in args.layers:
        if args.resume and any(r["layer"]==index for r in reports):
            print(f"RESUME: layer {index} already validated",flush=True)
            continue
        source = Path(args.inputs)/f"ffn-{index:02d}-input.pt"
        x = torch.load(source,map_location="cpu",weights_only=True)
        if x.ndim!=3 or x.shape[-1]!=1536 or not torch.isfinite(x).all():
            raise ValueError("Invalid captured FFN input")
        layer = torch.nn.Sequential(torch.nn.Linear(1536,8960),torch.nn.GELU(approximate="tanh"),
                                    torch.nn.Linear(8960,1536)).eval().requires_grad_(False)
        with safe_open(weights,framework="pt",device="cpu") as checkpoint:
            layer.load_state_dict({key:checkpoint.get_tensor(f"blocks.{index}.ffn.{key}")
                                   for key in layer.state_dict()})
        layer = layer.to(dtype=dtype)
        onnx_path = output/f"ffn-{index:02d}.onnx"
        engine_path = onnx_path.with_suffix(".engine")
        if engine_path.exists():
            raise ValueError(f"Engine already exists: {engine_path}")
        # FFN is token-independent. Trace one token on CPU, then specialize the
        # declared sequence dimension; real full-shape numerical testing follows.
        torch.onnx.export(layer,x[:, :1].to(dtype),str(onnx_path),input_names=["input"],output_names=["output"],
                          dynamic_axes={"input":{1:"tokens"},"output":{1:"tokens"}},
                          opset_version=17,dynamo=False,do_constant_folding=True)
        graph = onnx.load(str(onnx_path))
        for value in list(graph.graph.input)+list(graph.graph.output):
            value.type.tensor_type.shape.dim[1].dim_value = x.shape[1]
        onnx.checker.check_model(graph)
        onnx.save(graph,str(onnx_path))
        del graph
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        network = builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network,logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError("ONNX parse failed: "+"; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,args.workspace_mib*2**20)
        config.builder_optimization_level=3
        cache_path=output/f"timing-{args.precision}-w{args.workspace_mib}.cache"
        timing_cache=config.create_timing_cache(cache_path.read_bytes() if cache_path.exists() else b"")
        if not config.set_timing_cache(timing_cache,ignore_mismatch=False):
            raise RuntimeError("TensorRT timing cache is incompatible with this target")
        started = time.perf_counter()
        serialized = builder.build_serialized_network(network,config)
        if serialized is None:
            raise RuntimeError("TensorRT build failed")
        cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))
        engine_path.write_bytes(bytes(serialized))
        metadata = dict(layer=index,shape=list(x.shape),precision=args.precision,
            gpu=torch.cuda.get_device_name(),compute_capability=list(torch.cuda.get_device_capability()),
            torch=torch.__version__,cuda=torch.version.cuda,tensorrt=trt.__version__,
            engine_sha256=file_hash(engine_path),weights_sha256=digest,input_sha256=file_hash(source),
            build_s=time.perf_counter()-started,workspace_mib=args.workspace_mib)
        engine_path.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
        del serialized,config,parser,network,builder,timing_cache
        runner = TRTFeedForward(engine_path,x.shape,digest)
        x = x.cuda()
        reference = layer.cuda()
        compiled = torch.compile(reference)
        with torch.no_grad():
            expected = reference(x.to(dtype)).float()
            actual = runner(x).float()
        delta = (expected-actual).abs()
        report = dict(layer=index,precision=args.precision,shape=list(x.shape),
            max_abs=float(delta.max()),mean_abs=float(delta.mean()),finite=bool(torch.isfinite(actual).all()),
            compiled_torch_ms=benchmark(compiled,x.to(dtype)),trt_ms=benchmark(runner,x))
        report["speedup"] = report["compiled_torch_ms"]/report["trt_ms"]
        reports.append(report)
        print(json.dumps(report),flush=True)
        (output/"report.json").write_text(json.dumps(reports,indent=2)+"\n")
        del runner,reference,compiled,expected,actual,delta,layer,x
        torch.compiler.reset()
        torch.cuda.empty_cache()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["capture","build"])
    ap.add_argument("--width",type=int,default=512)
    ap.add_argument("--height",type=int,default=512)
    ap.add_argument("--memory-mode",default="default",choices=["default","compact","reference","staged"])
    ap.add_argument("--layers",type=int,nargs="+",default=[0])
    ap.add_argument("--precision",choices=["bf16","fp16"],default="bf16")
    ap.add_argument("--workspace-mib",type=int,default=256)
    ap.add_argument("--image",default="benchmarks/implementation/musetalk-idle-anchor.png")
    ap.add_argument("--audio",default="/workspace/MuseTalk/generated/webrtc_quality/2026-08-08/kokoro_duration_matrix/multi_sentence.wav")
    ap.add_argument("--inputs",default=".trt-experiment/captures")
    ap.add_argument("--output",required=True)
    ap.add_argument("--resume",action="store_true")
    args=ap.parse_args()
    if any(not 0<=n<30 for n in args.layers):
        ap.error("Layer indices must be 0–29")
    if args.action=="capture":
        capture(args)
    else:
        from .gpu_lease import acquire_gpu_lease
        with acquire_gpu_lease():
            build(args)


if __name__=="__main__":
    main()
