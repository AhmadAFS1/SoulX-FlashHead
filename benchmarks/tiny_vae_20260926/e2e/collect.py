"""Collect the end-to-end tiny-VAE arms into e2e/results.json (CPU only, reads run outputs).

Every arm: fresh local GPU inference on the RTX 4070 SUPER (12,282 MiB visible), 2026-09-26,
576x320, 250 frames, 2 steps, flash2 self-attention (the SageAttention SM89 build is missing).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "benchmarks/pro_30fps_20260922"
E2E = Path(__file__).resolve().parent

ARMS = [
    ("tae-ctl-r01", "control: shipping VAE (pruned+distilled Wan decoder, FP16 TRT spans, overlap-skip; compiled Wan encoder)"),
    ("tae-dec-r01", "T1: taew2_1 decoder (stream, torch.compile) + compiled Wan motion encoder"),
    ("tae-decenc-r01", "T2: taew2_1 decoder + taew2_1 motion encoder (torch.compile)"),
    ("tae-decenc-trt-r01", "T3: taew2_1 decoder + encoder, TensorRT FP16"),
    ("tae-light-decenc-r01", "T4: lighttaew2_1 decoder + encoder (torch.compile)"),
    ("tae-enc-r01", "T5: shipping decoder + taew2_1 motion encoder (torch.compile)"),
    ("tae-decenc-trt-window-r01", "T6: as T3 but window-mode decode (all 9 latents from zero state each window)"),
    ("tae-ctl-r02", "control repeat on the EDITED run.py (default-path check; warm caches)"),
    ("tae-ctl-r03-origrunpy", "control repeat on the ORIGINAL run.py (warm caches; noise floor)"),
]
EXTRA_FIXTURES = ("tts-plosives", "tts-open-vowels")


def _drift(name):
    path = E2E / "logs" / f"drift-{name}.log"
    if not path.is_file():
        return None
    text = path.read_text()
    vals = re.findall(r"\|max drift\| ([0-9.]+)/255", text)
    slopes = re.findall(r"slope/window: R([+-][0-9.]+) G([+-][0-9.]+) B([+-][0-9.]+)", text)
    if len(vals) < 2:
        return None
    return {"control_max_drift_255": float(vals[0]), "candidate_max_drift_255": float(vals[1]),
            "candidate_slope_rgb_per_window": [float(x) for x in slopes[1]]}


def _review(dirname):
    path = RUNS / dirname / "review.json"
    if not path.is_file():
        return None
    p = json.loads(path.read_text())["paired"]
    return {"opening_correlation": round(p["opening_correlation"], 4),
            "oral_edge_ratio": round(p["median_edge_ratio_on_both_open"], 4),
            "mouth_center_distance_px": round(p["median_mouth_center_distance_px"], 2),
            "both_open_pairs": p["both_open_pairs"]}


def complete(name):
    path = RUNS / name / "results.json"
    return path.is_file() and json.loads(path.read_text()).get("status") == "complete"


def row(name, label):
    d = json.loads((RUNS / name / "results.json").read_text())
    r = d["runs"][0]
    res = json.loads((RUNS / name / "resources-0.json").read_text())
    ct = r["chunk_times_s"]
    tv = d.get("tiny_vae", {})
    out = {
        "run": name, "arm": label, "fixture": d["fixture"]["id"], "seed": d["profile"]["seed"],
        "status": d["status"], "execution": d["execution"],
        "label": "fresh local GPU inference on RTX 4070 SUPER",
        "policy": d["policy"]["path"],
        "attention": d["attention_backend"]["requested"],
        "useful_fps": round(r["useful_fps"], 3),
        "steady_fps_windows_1_8": round(28 * (len(ct) - 1) / sum(ct[1:]), 3),
        "generation_s": round(r["generation_s"], 4),
        "window_p50_s": round(r["chunk_distribution_s"]["p50"], 4),
        "window0_s": round(ct[0], 4),
        "stage_seconds": {k: round(v, 4) for k, v in r["stage_seconds"].items()},
        "stage_ms_per_window": {k: round(1000 * v / len(ct), 1) for k, v in r["stage_seconds"].items()},
        "nvidia_smi_peak_mib": max(x["gpu_vram_used_mib"] for x in res),
        "torch_peak_allocated_mib": round(r["peak_allocated_mib"]),
        "torch_peak_reserved_mib": round(r["peak_reserved_mib"]),
        "warmup_s": round(d["warmup_s"], 2),
        "raw_rgb_sha256": d["raw_rgb_sha256"],
        "tiny_vae": {role: {k: v.get(k) for k in ("name", "sha256", "decoder_latent_convention" if role == "decoder"
                                                  else "encoder_latent_convention", "backend", "decode_mode", "state")
                            if k in v} for role, v in tv.items()} or None,
        "gate_vs_reference_s30_reuse": _review(f"review-{name}"),
        "paired_vs_same_session_control": _review(f"review-{name}-vs-ctl"),
        "colour_drift": _drift(name),
    }
    return out


def main():
    rows = [row(n, l) for n, l in ARMS if complete(n)]
    for fx in EXTRA_FIXTURES:
        for prefix, label in (("tae-ctl", "control"), ("tae-decenc-trt", "T3 taew2_1 TRT dec+enc"),
                              ("tae-light-decenc", "T4 lighttaew2_1 dec+enc")):
            name = f"{prefix}-{fx}-r01"
            if complete(name):
                rows.append(row(name, f"{label} ({fx})"))
    ctl = {r["fixture"]: r for r in rows if r["run"].startswith("tae-ctl") and r["run"].endswith("r01")}
    for r in rows:
        c = ctl.get(r["fixture"])
        if c is not None:
            r["fps_vs_same_session_control"] = round(r["useful_fps"] / c["useful_fps"], 3)
    ref = json.loads((RUNS / "tae-ctl-r01/results.json").read_text())
    header = {
        "hardware": "NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM (nvidia-smi / run manifest), driver 595.84, "
                    "torch 2.7.1+cu128, CUDA runtime 12.8; GPU UUID " + ref["environment"]["gpu"]["devices"][0]["uuid"],
        "label": "fresh local GPU inference on RTX 4070 SUPER, 2026-09-26; single tenant during every measured run (lease held)",
        "profile": "576x320, 250 frames, 2 steps distilled_aligned, seed 50, static INT8 ffn.2 scales, DiT CUDA graph, fused INT8 FFN, lean delivery",
        "attention": "flash2 in EVERY arm: the SageAttention-2 SM89 build the shipping policy needs is missing and could not be rebuilt; "
                     "the shipping 2026-09-22 figure (30.22 FPS, sage2) is reused, not re-run",
        "tensorrt": "Wan span engines: TensorRT 10.9.0.34 runtime restored to .restored-deps-20260926; tiny-VAE engines: TensorRT 10.3.0 "
                    "borrowed read-only via decoder_bakeoff/_trt10_path",
        "gate_reference": "benchmarks/pro_30fps_20260921/s30-reuse (19.62 FPS, indian150-a seed 50); accepted band corr 0.94-0.97, "
                          "edge ratio 0.95-1.07, mouth distance <= 3.4 px",
        "colour_drift": "per-run max |mean-RGB drift vs its own window 0| over 9 windows (/255); s30-reuse/raw.npy is deleted, so the "
                        "reference's 1.48 is the historical figure",
    }
    (E2E / "results.json").write_text(json.dumps({"header": header, "rows": rows}, indent=1))
    for r in rows:
        g = r["gate_vs_reference_s30_reuse"] or {}
        p = r["paired_vs_same_session_control"] or {}
        dr = (r["colour_drift"] or {}).get("candidate_max_drift_255")
        print(f"{r['run']:34s} fps {r['useful_fps']:7.3f} x{r.get('fps_vs_same_session_control', 0):.2f} p50 {r['window_p50_s']:.3f} "
              f"dec {r['stage_seconds'].get('vae_decode', 0):.3f} enc {r['stage_seconds'].get('motion_encode', 0):.3f} "
              f"dit {r['stage_seconds'].get('dit', 0):.3f} vram {r['nvidia_smi_peak_mib']} | ref: {g.get('opening_correlation')} "
              f"{g.get('oral_edge_ratio')} {g.get('mouth_center_distance_px')} | ctl: {p.get('opening_correlation')} "
              f"{p.get('oral_edge_ratio')} {p.get('mouth_center_distance_px')} | drift {dr}")


if __name__ == "__main__":
    main()
