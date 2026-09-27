#!/usr/bin/env python3
"""Aggregate quality.json + speed.json into results.json and results.md (CPU only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
BK = Path(__file__).resolve().parent
ROOT = BK.parents[2]

SHIP_RUN = ROOT / "benchmarks/pro_30fps_20260922/final-v4-lean-r01/results.json"
LABEL = "fresh local GPU inference on RTX 4070 SUPER"
NAMES = {
    "shipping": "shipping decoder (skip 9,10,13,14 + ft4), bf16 eager",
    "taew2_1": "taew2_1 (TAEHV), fp16",
    "taew2_1_bf16": "taew2_1 (TAEHV), bf16",
    "lighttaew2_1": "lighttaew2_1 (LightTAE), fp16",
    "lighttaew2_1_bf16": "lighttaew2_1 (LightTAE), bf16",
    "lightvaew2_1": "lightvaew2_1 (LightVAE, WanVAE_ dim=24), bf16",
    "stock": "stock Wan decoder, stream (overlap-skip analogue)",
}



HARDWARE = """**Hardware / environment (every GPU number in this file).** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM
(nvidia-smi query recorded in quality.json / speed.json), driver 595.84, torch 2.7.1+cu128 (CUDA runtime 12.8, cuDNN 9.7.1),
TensorRT 10.3.0 python bindings (borrowed read-only from /workspace/.venvs/musetalk_trt_stagewise through the symlinks in
decoder_bakeoff/_trt10_path/). Runs on 2026-09-26: latent dumps 20:41-20:47 UTC, quality 20:47-20:51 UTC, speed 20:51-20:58 UTC.
Co-resident load: none during the quality and speed runs (nvidia-smi listed only this process at the start and after every speed
case); another project's ComfyUI/MuseTalk jobs used the GPU before and between runs, and one of them made the first dump attempt
fail with an out-of-memory error, so that attempt was re-run.
The 2026-09-22 shipping-build numbers are **reused** from benchmarks/pro_30fps_20260922/final-v4-lean-r01/results.json (same GPU model,
fresh local GPU inference at the time); they were not re-run.
"""

SUMMARY = """## Answer and best candidate

* **The SD TAESD decoder itself cannot be used.** It decodes 4-channel single-image SD latents. SoulX uses Wan 2.1 latents
  (16 channels, 8x spatial, 4x causal temporal). The Wan 2.1 version from the same author, **taew2_1 (TAEHV)**, does fit: it takes the
  DiT's normalised latents directly, gives exactly 33 frames per 9-latent window with the upstream 3-frame trim, and can stream across
  windows with carried state (28 frames per 7 latents).
* **Best candidate: taew2_1, fp16, as a TensorRT FP16 engine** (torch.compile of the same core is 9-12 ms slower per call).
  * Speed: 36 ms for the steady-state 7-latent decode and 46 ms for the cold 9-latent decode. The shipping decoder takes 434 ms per
    window in the harness (reused figure), so this is about 12x faster.
  * Arithmetic projection, not measured: about 53 useful FPS instead of 30.2, because the DiT and the motion encode stay as they are.
    5x end to end is out of reach from the decoder alone.
* **The trade is fidelity.** Measured against the stock decoder:
  * PSNR: taew2_1 is 39.6 dB full frame and 35.4 dB on the mouth. The shipping decoder is 49.0 and 44.9 dB, so taew2_1 is about
    9.5 dB lower on both.
  * Sharpness: taew2_1 keeps mouth sharpness at 0.98x the reference; the shipping decoder is at 0.96x.
  * Temporal flicker: taew2_1 is 1.00x the reference.
  * Colour: the offset is at most 0.43/255 per channel in any window.
  * On the PNGs the losses are fine texture: stubble looks grainier and gaps between teeth are softer. Lip contours and teeth are
    preserved.
* **The other candidates are worse:**
  * **lighttaew2_1:** loses about 0.6-0.9 dB, is smoother (flicker 0.83x) and has a 0.7/255 colour shift.
  * **lightvaew2_1:** its PSNR looks similar, but only because it is visibly blurry (sharpness 0.82x, flicker 0.77x). It is also about 4-5x
    slower than taew2_1 (175 ms compiled against 36-45 ms).
* **Not measured yet:** the end-to-end effect. The Wan encoder re-encodes the tiny decoder's colour-corrected output as the next
  window's conditioning, so the lip-sync, edge-ratio and colour-drift gates and a labelled full-run video are still needed before
  shipping.
"""

VISUAL_NOTES = """
## What the PNGs show (inspected by eye)

* **indian150-a w4 f24** (lips parted, a sliver of teeth):
  * **Shipping:** indistinguishable from the reference at 4x.
  * **taew2_1:** same lip outline, teeth sliver and mouth-corner shape. The moustache and stubble texture is slightly grainier,
    with faint blotchy or blocky micro-texture, and the skin is a touch lighter. There is no smearing.
  * **lighttaew2_1:** lips a little more saturated and pink, skin smoother, stubble detail reduced.
  * **lightvaew2_1:** clearly soft. Stubble is smeared, lip edges are blurred, and there is a slight purple cast in the mouth
    opening.
* **tts-open-vowels w6 f7** (open mouth, upper teeth visible):
  * **Reference and shipping:** crisp teeth with visible gaps between them.
  * **taew2_1:** the teeth row is there, but the individual gaps are less distinct. Lip edges stay sharp, and the stubble is
    grainier.
  * **lighttaew2_1:** teeth softer and skin smoother.
  * **lightvaew2_1:** softest of all. The teeth are still readable, but the lip contour is blurred.
* **Across both frames:** apart from taew2_1's faint blotchy micro-texture, no candidate shows a regular 8x8 latent-grid
  pattern, colour banding, or a visible global colour shift at 1x.
* **mp4 spot check:** I did not watch the mp4 as a video. I looked at frame 20 and at a strip of mouth crops over delivered frames
  12-19 extracted from it (indian150-a w4, smile with teeth).
  * **taew2_1:** keeps the structure of individual teeth, but with speckled dark gaps and grain, strongest on frame 12. The grain
    looked stable from frame to frame in that strip, with no obvious shimmer.
  * **lightvaew2_1:** merges the teeth into a soft white band.
  * **lighttaew2_1:** in between.
  * **Shipping:** matches the reference.
"""

CAVEATS = """
## Caveats and what was not measured

* **Missing inputs:** the 4 "real latent" windows (real-inputs-r01/tensors) no longer exist, so none were used.
* **The shipping build cannot currently be reproduced on this machine.**
  * `/workspace/experiments/pro30-deps` (the SM89 SageAttention-2 build) and `/workspace/experiments/ojin-components-deps`
    (TensorRT 10.9) are gone.
  * scratchpad/hires_dump.sh would therefore fail: its policy needs sage2 attention and the trt_stage_compile decoder.
  * The latents were dumped instead with decoder_bakeoff/dump_latents.sh and policies/dump_flash2_eagerdec.json. That policy keeps
    the same DiT quantisation but uses FlashAttention-2 and the stock bf16 eager decoder in the loop, with no overlap-skip.
  * The latents are genuine SoulX DiT outputs, but with flash2 instead of sage2 attention.
* **Shipping decoder numbers:** here it runs in eager bf16. The shipping build runs the same module layout as FP16 TensorRT
  spans, so its in-build fidelity can differ slightly from the rows above.
* **Metric definitions:**
  * Metrics are on uint8 frames from the pipeline's lean-delivery arithmetic, without colour correction. The pipeline's
    per-frame colour matching would largely remove the small global colour offsets.
  * "Sharpness" is the mean absolute luma gradient ratio. It is not review.py's oral edge ratio.
* **The motion re-encode stays on the Wan encoder in this study.** A tiny decoder changes the pixels that are re-encoded into the
  next window's DiT conditioning. That feedback effect needs an end-to-end harness run with the gates.
* **Stream mode:** the stream-mode rows emulate overlap-skip with each decoder's own carried state. The stock decoder itself scores
  48.3 dB mean and 41.5 dB worst in that mode, because the carried cache differs from the re-encoded history. The shipping decoder
  scores 42.4 dB. So part of every stream-mode gap is the overlap-skip mechanism, not the decoder.
* **TensorRT memory:** the peak-memory column for TensorRT excludes the engine's activation workspace (1,721 MiB for cold9 and
  1,339 MiB for warm7, allocated through torch at context creation).
"""


def f(x, d=2):
    return "n/a" if x is None else f"{x:.{d}f}"


def qrow(name, a):
    c = a["colour_offset_rgb"]
    return (f"| {NAMES.get(name, name)} | {a['n_windows']} | {f(a['psnr_full']['mean'])} / {f(a['psnr_full']['worst'])} | "
            f"{f(a['psnr_mouth']['mean'])} / {f(a['psnr_mouth']['worst'])} | {f(a['sharp_full']['mean'], 3)} / {f(a['sharp_full']['worst'], 3)} | "
            f"{f(a['sharp_mouth']['mean'], 3)} / {f(a['sharp_mouth']['worst'], 3)} | {f(a['flicker']['mean'], 3)} / {f(a['flicker']['worst'], 3)} | "
            f"{', '.join(f(v, 2) for v in c['mean'])} / {', '.join(f(v, 2) for v in c['worst_abs'])} |")


QHEAD = ("| decoder | windows | PSNR full dB (mean / worst) | PSNR mouth dB (mean / worst) | sharpness full (mean / worst) | "
         "sharpness mouth (mean / worst) | flicker ratio (mean / worst) | colour offset R,G,B /255 (mean / worst abs) |\n"
         "|---|---:|---:|---:|---:|---:|---:|---:|")


def main():
    q = json.loads((BK / "quality.json").read_text())
    s = json.loads((BK / "speed.json").read_text()) if (BK / "speed.json").exists() else {"cases": {}}
    ship = json.loads(SHIP_RUN.read_text())
    sr = ship["runs"][0]
    ship_decode_ms = sr["stage_seconds"]["vae_decode"] / 9 * 1000

    # projection (arithmetic on the 2026-09-22 shipping stage totals; NOT a measurement)
    other = sr["generation_s"] - sum(sr["stage_seconds"].values())
    non_decode = sr["generation_s"] - sr["stage_seconds"]["vae_decode"]

    def project(cold_ms, warm_ms):
        total = non_decode + (cold_ms + 8 * warm_ms) / 1000
        return {"generation_s": total, "useful_fps": 250 / total}

    cases = s.get("cases", {})
    proj = {}
    for label, cold, warm in (("taew2_1 fp16 TensorRT", "taehv fp16 TensorRT cold9", "taehv fp16 TensorRT warm7"),
                              ("taew2_1 fp16 torch.compile", "taehv fp16 torch.compile nchw cold9", "taehv fp16 torch.compile nchw warm7"),
                              ("taew2_1 fp16 eager channels_last", "taehv fp16 eager channels_last cold9", "taehv fp16 eager channels_last warm7"),
                              ("lightvaew2_1 bf16 torch.compile", "lightvaew2_1 bf16 torch.compile cold9", "lightvaew2_1 bf16 torch.compile warm7")):
        if cold in cases and warm in cases:
            proj[label] = project(cases[cold]["median_ms"], cases[warm]["median_ms"])
    results = {
        "execution": LABEL,
        "date_utc": q["date_utc"],
        "inputs": {
            "windows": q["windows"],
            "n_windows": len(q["windows"]),
            "fixtures": sorted({w["fixture"] for w in q["windows"]}),
            "note": ("The 4 'real latent' windows listed in benchmarks/pro_30fps_20260919/real-inputs-r01/manifest.json "
                     "no longer exist (its tensors/ directory is empty), so all windows here are fresh DiT latents from "
                     "decoder_bakeoff/dump_latents.sh (fixtures indian150-a, tts-plosives, tts-open-vowels, tts-sibilants; seed 50; "
                     "measured windows 0-8 of each run)."),
        },
        "sanity_reference_vs_in_loop_decode_maxabs": q.get("sanity_vs_harness_decode_maxabs"),
        "conventions": {"chosen": q["convention_chosen"], "search_mean_over_windows": q["convention_summary"],
                        "layout_NCTHW_into_TAEHV": q.get("layout_ncthw_into_taehv"),
                        "precision_check_taew2_1_fp32_vs_fp16": q.get("precision_check_taew2_1")},
        "quality_window_mode": q["window_mode"],
        "quality_stream_mode_delivered_frames": q["stream_mode"],
        "speed": s,
        "shipping_harness_reference": {
            "run": str(SHIP_RUN.relative_to(ROOT)), "execution": ship["execution"], "date_utc": ship["date_utc"],
            "useful_fps": sr["useful_fps"], "stage_seconds": sr["stage_seconds"],
            "vae_decode_ms_per_window": ship_decode_ms,
            "note": "FP16 TensorRT spans + torch.compile, overlap-skip (window 0 cold 9 latents, windows 1-8 warm 7 latents), reused not re-run",
        },
        "projection_not_measured": {
            "method": "shipping 2026-09-22 generation_s minus its vae_decode total, plus (cold9 + 8 x warm7) median of the candidate decoder",
            "non_decode_s": non_decode, "other_s": other, "by_candidate": proj,
        },
        "visuals": q.get("visuals"),
    }
    (BK / "results.json").write_text(json.dumps(results, indent=1))

    md = []
    md.append("# Tiny-decoder bake-off at 576x320 on SoulX DiT latents (2026-09-26)\n")
    md.append(HARDWARE)
    md.append(f"All numbers measured here are **{LABEL}**, unless marked as reused or projected.\n")
    md.append(SUMMARY)
    md.append(f"Inputs: {len(q['windows'])} latent windows (16x9x72x40 bf16) from {', '.join(results['inputs']['fixtures'])} "
              f"(seed 50, measured windows 0-8). {results['inputs']['note']}\n")
    md.append(f"Reference: stock Wan 2.1 decoder, bf16 eager, stock weights; it reproduces the in-loop decode of the dump run "
              f"bit-exactly (max abs diff on a strided checksum: {q.get('sanity_vs_harness_decode_maxabs')}).\n")
    md.append("## Conventions (resolved empirically, mean PSNR full frame over all windows)\n")
    md.append("| candidate | setting | PSNR full dB | PSNR mouth dB | frames |\n|---|---|---:|---:|---:|")
    cs = q["convention_summary"]
    for k, v in cs.items():
        if "offsets" in v:
            for o, x in v["offsets"].items():
                md.append(f"| {k} | temporal offset {o} | {f(x['psnr_full_mean'])} | {f(x.get('psnr_mouth_mean'))} | {x['frames']} |")
        else:
            md.append(f"| {k} | offset 0 | {f(v.get('psnr_full_mean'))} | {f(v.get('psnr_mouth_mean'))} | 33 |")
    md.append(f"\nNCTHW (Wan layout) into TAEHV: {q.get('layout_ncthw_into_taehv')}\n")
    md.append("Chosen: " + "; ".join(f"{k}: {v}" for k, v in q["convention_chosen"].items()) + "\n")
    md.append("## Quality, window mode (every window decoded from scratch, all 33 frames)\n")
    md.append("Ratios are candidate/reference (1.0 = same as the stock decoder); worst = lowest PSNR, ratio farthest from 1, largest |offset|.\n")
    md.append(QHEAD)
    for k, a in q["window_mode"].items():
        md.append(qrow(k, a))
    md.append("\n## Quality, stream mode (overlap-skip analogue; 28 delivered frames per window)\n")
    md.append(QHEAD)
    for k, a in q["stream_mode"].items():
        md.append(qrow(k if k != "stock" else "stock", a))
    md.append("\n## Speed (median of timed calls, CUDA events; decode contract in/out included)\n")
    md.append("| case | median ms | min ms | max ms | peak alloc over resident MiB | frames | first call s |\n|---|---:|---:|---:|---:|---:|---:|")
    for k, r in cases.items():
        md.append(f"| {k} | {f(r['median_ms'])} | {f(r['min_ms'])} | {f(r['max_ms'])} | {f(r['peak_over_resident_mib'], 0)} | {r['frames_out']} | {f(r.get('first_call_s'), 1)} |")
    md.append(f"\nShipping harness decode (reused, {ship['execution']}, {ship['date_utc'][:10]}): {ship_decode_ms:.1f} ms/window average "
              f"(FP16 TensorRT spans, overlap-skip; 1 cold + 8 warm windows), {sr['useful_fps']:.2f} useful FPS.\n")
    if proj:
        md.append("## End-to-end projection (arithmetic, NOT measured)\n")
        md.append(f"Shipping generation {sr['generation_s']:.3f} s minus its decode total {sr['stage_seconds']['vae_decode']:.3f} s = {non_decode:.3f} s, "
                  "plus the candidate's cold9 + 8 x warm7 medians:\n")
        md.append("| decoder | projected generation s | projected useful FPS |\n|---|---:|---:|")
        md.append(f"| shipping (measured 2026-09-22) | {sr['generation_s']:.3f} | {sr['useful_fps']:.2f} |")
        for k, v in proj.items():
            md.append(f"| {k} | {v['generation_s']:.3f} | {v['useful_fps']:.2f} |")
    if s.get("trt"):
        md.append("\n## TensorRT build\n")
        md.append("```\n" + json.dumps(s["trt"], indent=1) + "\n```")
    if s.get("trt_error"):
        md.append(f"\nTensorRT error: {s['trt_error']}\n")
    if q.get("visuals"):
        md.append("\n## Visuals\n")
        for p in q["visuals"]:
            md.append(f"- `{p}`")
    md.append(VISUAL_NOTES)
    md.append(CAVEATS)
    (BK / "results.md").write_text("\n".join(md) + "\n")
    print("wrote results.json, results.md")


if __name__ == "__main__":
    main()
