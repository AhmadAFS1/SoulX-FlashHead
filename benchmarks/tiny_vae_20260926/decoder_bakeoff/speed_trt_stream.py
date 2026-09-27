#!/usr/bin/env python3
"""Follow-up: TensorRT taew2_1 engines timed on a NON-default CUDA stream (TensorRT 10.3 warns that
enqueueV3 on the default stream adds cudaStreamSynchronize calls). Reuses the engines speed.py
built under _trt/. Fresh local GPU inference, RTX 4070 SUPER. Merges into speed.json."""
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bakeoff_common as bc  # noqa: E402
import speed  # noqa: E402


def main():
    from soulx_rtc.gpu_lease import acquire_gpu_lease
    lease = acquire_gpu_lease(None)
    try:
        import tensorrt as trt
        out = bc.BK / "speed.json"
        res = json.loads(out.read_text())
        logger = trt.Logger(trt.Logger.WARNING)
        w, d = bc.load_latents()[4]
        z9 = d["latent"].cuda()
        z7 = z9[:, 2:].contiguous()
        tae = bc.load_taehv("taew2_1", dtype=torch.float16)
        ad = bc.TAEHVAdapter(tae, "norm", torch.float16)
        side = torch.cuda.Stream()
        with torch.inference_mode(), torch.cuda.stream(side):
            _, st = ad.raw(z9)
            st = [s.contiguous() for s in st]
            for t, label in ((9, "cold9"), (7, "warm7")):
                eng = speed.TRTCore(bc.ROOT / res["trt"][f"T{t}"]["engine"], logger)
                adt = bc.TAEHVAdapter(tae, "norm", torch.float16, core=eng)
                fn = (lambda: adt.decode_window(z9)) if label == "cold9" else (lambda: adt.to_wan(adt.raw(z7, st)[0]))
                r = speed.bench(fn, iters=20)
                r["trt_workspace_mib"] = eng.workspace_mib
                r["stream"] = "non-default torch.cuda.Stream"
                r["gpu_after"] = bc.gpu_snapshot()["compute_apps"]
                res["cases"][f"taehv fp16 TensorRT {label} (side stream)"] = r
                print(label, f"{r['median_ms']:.2f} ms", flush=True)
                del eng, adt
        res["trt_stream_followup_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        out.write_text(json.dumps(res, indent=1))
    finally:
        lease.close()


if __name__ == "__main__":
    main()
