"""Exact-shape BF16 LTX VAE decoder export/build/reload from a real latent."""
import argparse
import json
import time
from pathlib import Path

import torch

from .trt_backend import TRTFeedForward, file_hash
from .trt_experiment import benchmark


class VaeDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, latent):
        z = latent.unsqueeze(0)
        z = z*self.model.std_of_means.to(z.dtype).view(1,-1,1,1,1)+self.model.mean_of_means.to(z.dtype).view(1,-1,1,1,1)
        return self.model.decode(z,return_dict=False,target_shape=z.shape)[0]


class FP32PixelNorm(torch.nn.Module):
    def __init__(self, dim, eps):
        super().__init__()
        self.dim,self.eps=dim,eps

    def forward(self,x):
        value=x.float()
        return (value/torch.sqrt(value.square().mean(self.dim,keepdim=True)+self.eps)).to(x.dtype)


def main():
    import tensorrt as trt
    from flash_head.ltx_video.ltx_vae import LtxVAE
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",default=".trt-experiment/captures/vae-input.pt")
    ap.add_argument("--output",required=True)
    ap.add_argument("--workspace-mib",type=int,default=256)
    ap.add_argument("--precision",choices=["bf16","fp16"],default="bf16")
    args=ap.parse_args()
    folder=Path(args.output)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/"vae.engine"
    report_path=folder/"report.json"
    if path.exists() or report_path.exists():
        ap.error("Use a new output directory")
    report=dict(status="running",stage="load",source="actual SoulX denoised latent",input_sha256=file_hash(args.input))
    def save():
        report_path.write_text(json.dumps(report,indent=2)+"\n")
    save()
    from .gpu_lease import acquire_gpu_lease
    lease=None
    try:
        lease=acquire_gpu_lease()
        torch.set_num_threads(4)
        latent=torch.load(args.input,map_location="cuda",weights_only=True)
        if latent.ndim!=4 or latent.shape[:2]!=(128,5) or not torch.isfinite(latent).all():
            raise ValueError("Expected finite native Lite latent [128,5,H/32,W/32]")
        model=LtxVAE("models/SoulX-FlashHead-1_3B/VAE_LTX").model
        core=VaeDecoder(model).eval().requires_grad_(False)
        report.update(stage="torch_reference",input_shape=list(latent.shape))
        save()
        with torch.no_grad():
            expected=core(latent).cpu()
        compiled=torch.compile(core)
        report["compiled_torch_ms"]=benchmark(compiled,latent,repeats=20)
        report["output_shape"]=list(expected.shape)
        report["stage"]="onnx_export"
        save()
        if args.precision=="fp16":
            from flash_head.ltx_video.models.autoencoders.pixel_norm import PixelNorm
            # Numerical comparison remains against the ORIGINAL BF16 decoder.
            core.half()
            for name,module in list(core.named_modules()):
                if isinstance(module,PixelNorm):
                    parent,_,attribute=name.rpartition(".")
                    setattr(core.get_submodule(parent),attribute,FP32PixelNorm(module.dim,module.eps))
            latent=latent.half()
            report["precision_note"]="FP16 weights/activations with explicit FP32 pixel-norm reductions; compared to original BF16 outputs"
        # Export only the tensor decoder; RNG, motion encoder, scheduler and
        # color correction remain in the existing PyTorch owner.
        torch.onnx.export(core,latent,str(folder/"vae.onnx"),opset_version=17,dynamo=False,
                          input_names=["input"],output_names=["output"],do_constant_folding=True)
        core.cpu()
        del model,compiled,core
        torch.compiler.reset()
        torch.cuda.empty_cache()
        report["stage"]="trt_build"
        save()
        logger=trt.Logger(trt.Logger.WARNING)
        builder=trt.Builder(logger)
        network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser=trt.OnnxParser(network,logger)
        if not parser.parse_from_file(str(folder/"vae.onnx")):
            raise RuntimeError("ONNX parser: "+"; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config=builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,args.workspace_mib*2**20)
        config.builder_optimization_level=3
        started=time.perf_counter()
        serialized=builder.build_serialized_network(network,config)
        if serialized is None:
            raise RuntimeError("TensorRT decoder build returned no engine")
        path.write_bytes(bytes(serialized))
        weights="models/SoulX-FlashHead-1_3B/VAE_LTX/diffusion_pytorch_model.safetensors"
        metadata=dict(kind="vae_decoder",shape=list(latent.shape),output_shape=list(expected.shape),
            precision=args.precision,engine_sha256=file_hash(path),weights_sha256=file_hash(weights),
            input_sha256=file_hash(args.input),tensorrt=trt.__version__,gpu=torch.cuda.get_device_name(),
            torch=torch.__version__,cuda=torch.version.cuda,workspace_mib=args.workspace_mib,
            build_s=time.perf_counter()-started)
        path.with_suffix(".json").write_text(json.dumps(metadata,indent=2)+"\n")
        del serialized,config,parser,network,builder
        report["stage"]="reload_and_compare"
        save()
        runner=TRTFeedForward(path,latent.shape,metadata["weights_sha256"])
        with torch.no_grad():
            actual=runner(latent).cpu().float()
        delta=(actual-expected.float()).abs()
        report.update(trt_ms=benchmark(runner,latent,repeats=20),max_abs=float(delta.max()),
                      mean_abs=float(delta.mean()),finite=bool(torch.isfinite(actual).all()),
                      device_workspace_mib=runner.engine.device_memory_size/2**20)
        report["speedup"]=report["compiled_torch_ms"]/report["trt_ms"]
        report.update(status="completed",stage="done",note="Kernel validation only; recurrent full-video quality still required")
        save()
        print(json.dumps(report),flush=True)
    except Exception as exc:
        report.update(status="failed",error_type=type(exc).__name__,error=str(exc))
        save()
        raise
    finally:
        if lease is not None:
            lease.close()


if __name__=="__main__":
    main()
