"""Compare decoder repeat/reset behavior with controls on a fixed real window."""

import json
from pathlib import Path

import torch

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_GPU_LOCK,
    environment_manifest,
    snapshot_sources,
)
from benchmarks.pro_quantization_v2_20260918.decoder_trial import (
    _captured_latents,
    _configure_candidate,
    _load_vae,
    _metrics,
)
from soulx_rtc.gpu_lease import acquire_gpu_lease


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "status": "starting",
        "environment": environment_manifest(),
        "execution": "fresh GPU decoder repeat/reset diagnostic",
        "policy": str(args.policy),
    }
    result["sources"] = snapshot_sources(
        args.output, [Path(__file__), Path("soulx_rtc/pro_vae_stage_backend.py")]
    )
    lease = acquire_gpu_lease(DEFAULT_GPU_LOCK)
    torch._dynamo.config.recompile_limit = 64
    try:
        vae = _load_vae()
        result["configuration"] = _configure_candidate(
            vae, json.loads(args.policy.read_text()), False
        )
        paths = _captured_latents(args.captures)
        values = [
            torch.load(path, map_location="cuda", weights_only=True)
            for path in paths[:2]
        ]
        rows = []
        with torch.inference_mode():
            for _ in range(4):
                vae.decode(values[1])
            first = vae.decode(values[1]).cpu()
            for i in range(4):
                repeated = vae.decode(values[1]).cpu()
                rows.append(
                    {
                        "kind": "repeat",
                        "index": i,
                        "exact": torch.equal(first, repeated),
                        **_metrics(repeated, first),
                    }
                )
                vae.decode(values[0])
                restored = vae.decode(values[1]).cpu()
                rows.append(
                    {
                        "kind": "reset",
                        "index": i,
                        "exact": torch.equal(first, restored),
                        **_metrics(restored, first),
                    }
                )
        result.update(
            status="complete",
            observations=rows,
            all_exact=all(row["exact"] for row in rows),
        )
    except Exception as error:
        result.update(status="failed", error=str(error))
        raise
    finally:
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
        lease.close()


if __name__ == "__main__":
    main()
