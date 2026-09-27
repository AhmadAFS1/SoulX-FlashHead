#!/usr/bin/env python3
"""Strip a SOULX_DUMP_DIR decode dump to latents only (CPU; no GPU use).

Each decode-NNN.pt holds {"args": [latent (16,9,72,40) bf16], "out": frames (1,3,33,576,320) bf16}.
run.py's warmup runs 2 windows before reset(), so decode-000/001 are warmup and decode-002.. are
the measured windows 0..8. For each measured window this writes
  latents/<fixture>-s<seed>-w<k>.pt = {"latent": bf16 (16,9,72,40), "harness_out_strided":
      out[0, :, ::8, ::8, ::8] (bf16, 3x5x72x40, a checksum of the in-loop stock decode),
      "provenance": {...}}
and appends to latents/index.json.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--fixture", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--rc", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    files = sorted(args.dump.glob("decode-*.pt"))
    status = None
    try:
        status = json.loads((args.run / "results.json").read_text()).get("status")
    except Exception:
        pass
    index_path = args.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {"windows": []}
    index["windows"] = [w for w in index["windows"] if not (w["fixture"] == args.fixture and w["seed"] == args.seed)]
    kept = 0
    for f in files:
        call = int(f.stem.split("-")[1])
        if call < 2:
            continue  # warmup
        window = call - 2
        d = torch.load(f, map_location="cpu", weights_only=False)
        lat = d["args"][0].contiguous()
        out = d["out"]
        assert tuple(lat.shape) == (16, 9, 72, 40), lat.shape
        assert tuple(out.shape) == (1, 3, 33, 576, 320), out.shape
        name = f"{args.fixture}-s{args.seed}-w{window}.pt"
        prov = {"fixture": args.fixture, "seed": args.seed, "window": window, "dump_call": call,
                "source": "SoulX-FlashHead PRO DiT latents (decode input), run.py SOULX_DUMP_DIR, "
                          "policy decoder_bakeoff/policies/dump_flash2_eagerdec.json, no overlap-skip",
                "harness_rc": args.rc, "harness_status": status,
                "latent_sha256": hashlib.sha256(lat.view(torch.int16).numpy().tobytes()).hexdigest(),
                "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        torch.save({"latent": lat, "harness_out_strided": out[0, :, ::8, ::8, ::8].contiguous(),
                    "provenance": prov}, args.out / name)
        index["windows"].append({**prov, "file": name})
        kept += 1
    index["windows"].sort(key=lambda w: (w["fixture"], w["seed"], w["window"]))
    index_path.write_text(json.dumps(index, indent=1))
    print(f"kept {kept} measured windows from {len(files)} decode dumps ({args.fixture} s{args.seed}, status {status})")
    return 0 if kept else 2


if __name__ == "__main__":
    raise SystemExit(main())
