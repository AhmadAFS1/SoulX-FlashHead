"""Isolated Wan decoder trials at unchanged precision on identical PRO latents."""
import argparse
import json
from pathlib import Path
import time
import torch

from benchmarks.pro_lite_teeth_20260917.run_variant import gpu_snapshot, process_snapshot
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.pro_quantization import optimize_wan_vae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latents", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=("channels_last", "pointwise", "compiled"), required=True)
    args = ap.parse_args()
    lease = acquire_gpu_lease()
    torch.set_num_threads(4)
    from flash_head.wan.modules import WanVAE
    result = dict(execution="fresh GPU Wan decode; identical captured latents", gpu=gpu_snapshot(),
                  physical_vram_class="12 GB", co_residents=process_snapshot(),
                  torch=torch.__version__, cuda=torch.version.cuda, mode=args.mode)
    def save():
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    save()
    try:
        vae = WanVAE(vae_path="models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth",
                     dtype=torch.bfloat16, device="cuda", parallel=False)
        x = torch.load(args.latents, weights_only=True).cuda()
        def measure():
            with torch.inference_mode():
                vae.decode(x)
                torch.cuda.synchronize()
                seconds = []
                for _ in range(3):
                    start = time.perf_counter()
                    out = vae.decode(x)
                    torch.cuda.synchronize()
                    seconds.append(time.perf_counter()-start)
            return out, seconds
        expected, result["baseline_seconds"] = measure()
        result["policy"] = optimize_wan_vae(vae, args.mode)
        start = time.perf_counter()
        actual, result["candidate_seconds"] = measure()
        result["candidate_warmup_and_measure_s"] = time.perf_counter()-start
        delta = actual.float()-expected.float()
        result.update(status="complete", shape=list(actual.shape), finite=bool(actual.isfinite().all()),
                      mean_abs=float(delta.abs().mean()), max_abs=float(delta.abs().max()),
                      relative_l2=float(delta.norm()/expected.float().norm()),
                      peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20)
        save(); print(json.dumps(result),flush=True)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        save(); raise


if __name__ == "__main__":
    main()
