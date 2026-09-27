#!/usr/bin/env python3
"""Writes results.json + results.md from quality.json and speed.json (CPU only, no GPU)."""
from __future__ import annotations

import json
from pathlib import Path

EB = Path(__file__).resolve().parent
Q = json.loads((EB / "quality.json").read_text())
S = json.loads((EB / "speed.json").read_text())
LABEL = "fresh local GPU inference on RTX 4070 SUPER"

# ------------------------------------------------------------------ shipping reuse (final-v4-lean-r01, 2026-09-22)
SHIP = {"generation_s": 8.2725, "useful_fps": 30.2206, "motion_encode_s": 1.05255, "vae_decode_s": 3.90554,
        "motion_encode_ms_per_window": 116.95, "windows": 9, "frames": 250,
        "label": "reused: fresh local GPU inference on RTX 4070 SUPER, 2026-09-22 (final-v4-lean-r01), not re-run"}
DEC_TRT = {"cold9_ms": 46.29, "warm7_ms": 36.11, "label": "reused from decoder_bakeoff/speed.json (same session, same GPU)"}

ARMS = [
    ("taew2_1 (taew2_1 fp16 front0 in01 asis)", "taew2_1 encoder", "candidate"),
    ("lightvaew2_1 (lightvaew2_1 bf16 inpm1 wan-scale)", "lightvaew2_1 encoder", "candidate"),
    ("lighttaew2_1 (lighttaew2_1 fp16 front0 in01 raw2norm)", "lighttaew2_1 encoder", "candidate"),
    ("taew2_1 alt (taew2_1 fp16 front3 in01 asis)", "taew2_1, Wan-style front pad (wrong alignment)", "convention check"),
    ("encoder-only on shipping frames: taew2_1 encode of shipping-decoder cond_frame (vs Wan encode of same clip)",
     "taew2_1 encoder on shipping-decoder frames (vs Wan encode of same clip)", "candidate, arm-realistic input"),
    ("encoder-only on shipping frames: lightvaew2_1 encode of shipping-decoder cond_frame (vs Wan encode of same clip)",
     "lightvaew2_1 encoder on shipping-decoder frames (vs Wan encode of same clip)", "candidate, arm-realistic input"),
    ("taew2_1 encode of taew2_1-decoder cond_frame (vs Wan encode of same clip)",
     "taew2_1 encoder on taew2_1-decoder frames (vs Wan encode of same clip)", "candidate, arm-realistic input"),
    ("baseline: Wan encode of shipping-decoder cond_frame", "Wan encoder on shipping-decoder frames (perturbation the shipping build already carries)", "baseline (accepted)"),
    ("baseline: Wan encode of taew2_1-decoder cond_frame", "Wan encoder on taew2_1-decoder frames (perturbation a tiny decoder brings)", "baseline"),
    ("combo: taew2_1 encode of taew2_1-decoder cond_frame", "taew2_1 decoder + taew2_1 encoder (vs shipping-path Wan encode of stock frames)", "baseline"),
    ("REJECTED latent feedback last2-fix0 (Wan enc f0 + DiT latent 8)", "REJECTED latent feedback last2-fix0", "baseline (rejected, drift 2.77/255)"),
    ("REJECTED latent feedback last2 (DiT latents 7,8)", "REJECTED latent feedback last2", "baseline (rejected, drift 2.56/255)"),
]


def same_sign_frac(arm_key, slot, ch=None):
    """Fraction of clips whose per-clip mean offset has the sign of the across-clip mean (one channel, or mean over channels)."""
    rows = Q["per_clip"][arm_key]
    chans = [ch] if ch is not None else list(range(16))
    fr = []
    for c in chans:
        v = [r["slots"][slot]["off_c"][c] for r in rows]
        m = sum(v) / len(v)
        fr.append(sum(1 for x in v if (x > 0) == (m > 0)) / len(v))
    return sum(fr) / len(fr)


def f(x, n=3):
    return f"{x:.{n}f}"


def arm_row(key, label, kind):
    a = Q["arms"][key]
    L = a["latent_vs_wan_encode"]["slots"]
    P = a["pixels_vs_wan_roundtrip"]
    R = a["roundtrip_vs_own_input_clip"]
    return {
        "arm": label, "kind": kind, "clips": a["latent_vs_wan_encode"]["n_clips"], "input_clip": a["input_clip"],
        "reference": a["reference"],
        "slot0_nrmse_mean": L[0]["nrmse_mean"], "slot0_nrmse_max_channel": L[0]["nrmse_max"],
        "slot1_nrmse_mean": L[1]["nrmse_mean"], "slot1_nrmse_max_channel": L[1]["nrmse_max"],
        "slot0_cos_mean": L[0]["cos_mean"], "slot0_cos_min": L[0]["cos_min"],
        "slot1_cos_mean": L[1]["cos_mean"], "slot1_cos_min": L[1]["cos_min"],
        "slot0_pearson_mean": L[0]["pearson_mean"], "slot1_pearson_mean": L[1]["pearson_mean"],
        "slot0_offset_rms": L[0]["offset_rms_over_channels"], "slot0_offset_maxabs": L[0]["offset_maxabs_channel"],
        "slot0_offset_maxabs_channel": L[0]["offset_maxabs_channel_idx"],
        "slot1_offset_rms": L[1]["offset_rms_over_channels"], "slot1_offset_maxabs": L[1]["offset_maxabs_channel"],
        "slot1_offset_maxabs_channel": L[1]["offset_maxabs_channel_idx"],
        "slot0_offset_sign_consistency": L[0]["offset_sign_consistency_mean"],
        "slot1_offset_sign_consistency": L[1]["offset_sign_consistency_mean"],
        "slot0_offset_per_channel": L[0]["offset_mean_per_channel"], "slot1_offset_per_channel": L[1]["offset_mean_per_channel"],
        "pix_vs_wan_rt_psnr_full": P["psnr_full"]["mean"], "pix_vs_wan_rt_psnr_full_worst": P["psnr_full"]["worst"],
        "pix_vs_wan_rt_psnr_mouth": P["psnr_mouth"]["mean"], "pix_vs_wan_rt_psnr_mouth_worst": P["psnr_mouth"]["worst"],
        "pix_vs_wan_rt_rgb_offset_frame0": P["rgb_off_f0"]["mean"], "pix_vs_wan_rt_rgb_offset_frames1_4": P["rgb_off_f14"]["mean"],
        "pix_vs_wan_rt_rgb_offset_mean_absmax": P["rgb_off_all"]["mean_absmax"],
        "pix_vs_wan_rt_rgb_offset_worst_abs": P["rgb_off_all"]["worst_abs"],
        "roundtrip_vs_input_psnr_full": R["psnr_full"]["mean"], "roundtrip_vs_input_psnr_full_worst": R["psnr_full"]["worst"],
        "roundtrip_vs_input_psnr_mouth": R["psnr_mouth"]["mean"], "roundtrip_vs_input_psnr_mouth_worst": R["psnr_mouth"]["worst"],
        "label": LABEL,
    }


def speed_rows():
    s0, s1 = S["settings"]["cudnn_benchmark=0"]["cases"], S["settings"]["cudnn_benchmark=1"]["cases"]
    rows = []
    for k, r in s0.items():
        r1 = s1.get(k, {})
        rows.append({"case": k, "median_ms_cudnn_bench_off": r["median_ms"], "min_ms_off": r["min_ms"], "max_ms_off": r["max_ms"],
                     "median_ms_cudnn_bench_on": r1.get("median_ms"), "n": r["n"], "warmup": r["warmup"],
                     "peak_over_resident_mib": r["peak_over_resident_mib"], "trt_workspace_mib": r.get("trt_workspace_mib"),
                     "first_call_s": r.get("first_call_s"), "nrmse_vs_wan_eager_this_clip": r.get("nrmse_vs_wan_eager_this_clip"),
                     "maxabs_vs_eager_same_model": r.get("maxabs_vs_eager_same_model"), "label": LABEL})
    return rows


def project(enc_ms, dec=False):
    g = SHIP["generation_s"] - SHIP["motion_encode_s"] + SHIP["windows"] * enc_ms / 1000
    if dec:
        g = g - SHIP["vae_decode_s"] + (DEC_TRT["cold9_ms"] + 8 * DEC_TRT["warm7_ms"]) / 1000
    return {"generation_s": g, "useful_fps": SHIP["frames"] / g}


def main():
    arms = [arm_row(*a) for a in ARMS]
    sp = speed_rows()
    spd = {r["case"]: r for r in sp}
    t_trt = spd["taew2_1 fp16 TensorRT (side stream)"]["median_ms_cudnn_bench_off"]
    t_cmp = spd["taew2_1 fp16 torch.compile nchw"]["median_ms_cudnn_bench_off"]
    t_eag = spd["taew2_1 fp16 eager nchw"]["median_ms_cudnn_bench_off"]
    t_lv = spd["lightvaew2_1 bf16 torch.compile"]["median_ms_cudnn_bench_off"]
    t_wan = spd["stock Wan 2.1 bf16 torch.compile (harness --compile-vae-encode)"]["median_ms_cudnn_bench_off"]
    proj = {
        "label": "ARITHMETIC PROJECTION on the reused 2026-09-22 shipping stage totals, NOT measured",
        "formula": "generation_s = 8.2725 - 1.05255 (motion_encode) + 9 * encode_ms [- 3.90554 + (46.29 + 8*36.11)/1000 for the taew2_1 TRT decoder]",
        "encoder_only_taew2_1_trt": project(t_trt), "encoder_only_taew2_1_compile": project(t_cmp),
        "encoder_only_taew2_1_eager": project(t_eag), "encoder_only_lightvaew2_1_compile": project(t_lv),
        "taew2_1_decoder_trt_plus_taew2_1_encoder_trt": project(t_trt, dec=True),
        "taew2_1_decoder_trt_only (decoder bake-off projection)": {"generation_s": SHIP["generation_s"] - SHIP["vae_decode_s"] + (DEC_TRT["cold9_ms"] + 8 * DEC_TRT["warm7_ms"]) / 1000},
    }
    proj["taew2_1_decoder_trt_only (decoder bake-off projection)"]["useful_fps"] = 250 / proj["taew2_1_decoder_trt_only (decoder bake-off projection)"]["generation_s"]

    val = Q["validation_vs_inloop"]
    conv = {k: {"slot0_nrmse": v["slots"][0]["nrmse_mean"], "slot1_nrmse": v["slots"][1]["nrmse_mean"],
                "slot0_cos": v["slots"][0]["cos_mean"], "slot1_cos": v["slots"][1]["cos_mean"]}
            for k, v in Q["convention_search"].items()}
    out = {
        "execution": LABEL,
        "hardware": {"gpu": "NVIDIA GeForce RTX 4070 SUPER", "vram_visible_mib": 12282, "evidence": "nvidia-smi (quality.json gpu_at_start, speed.json)",
                     "driver": "595.84", "torch": "2.7.1+cu128", "cuda_runtime": "12.8", "cudnn": "9.7.1 (90701)",
                     "tensorrt": "10.3.0 (borrowed read-only from /workspace/.venvs/musetalk_trt_stagewise via decoder_bakeoff/_trt10_path)",
                     "co_resident_load": "none: every nvidia-smi snapshot during the runs lists only this bake-off's own lease-holding process"},
        "dates_utc": {"quality": Q["date_utc"], "speed": {k: v["date_utc"] for k, v in S["settings"].items()}},
        "inputs": {"clips": Q["clips"], "latents": "benchmarks/tiny_vae_20260926/latents (36 SoulX DiT windows, 4 fixtures, seed 50; flash2-attention dump variant)",
                   "clip_construction": "stock Wan decode (bf16 eager) -> videos[:, :, 5:] -> match_and_blend_colors_torch(ref, 1.0) in bf16 -> trailing 5 frames",
                   "validation_vs_inloop_encode": {"what": val["what"], "rel_l2_max": val["rel_l2_max"], "maxabs_max": val["maxabs_max"],
                                                   "slot0_nrmse": val["agg"]["slots"][0]["nrmse_mean"], "slot1_nrmse": val["agg"]["slots"][1]["nrmse_mean"],
                                                   "slot0_cos": val["agg"]["slots"][0]["cos_mean"], "slot1_cos": val["agg"]["slots"][1]["cos_mean"]},
                   "reference_image": Q["reference_image"]},
        "conventions_resolved": {
            "taew2_1": "input (x+1)/2 in [0,1], NCTHW->NTCHW, fp16; pad the END with 3 copies of the last frame (upstream encode_video) -> 8 frames -> 2 latents; output used AS IS (already DiT-normalised); (L,16,h,w)->(16,L,h,w) bf16",
            "lighttaew2_1": "same input/padding as taew2_1; output is UN-normalised -> (e - mean) * inv_std with WanVAE's bf16 scale",
            "lightvaew2_1": "WanVAE_(dim=24).encode(clip in [-1,1], wan scale).squeeze(0): exactly the Wan protocol",
            "core_vs_upstream_encode_video_maxabs_fp32": Q["core_vs_upstream_encode_video_maxabs_fp32"],
            "search": conv,
        },
        "wan_roundtrip_vs_clip": Q["wan_roundtrip_vs_clip"],
        "wan_latent_per_channel_std": Q["wan_latent_stats"]["per_channel_std"],
        "arms": arms,
        "speed": sp,
        "trt_engine": {k: v for k, v in S["trt"].items() if k not in ("engine_path", "onnx_path")} | {"files": "deleted after timing (sha256 recorded)"},
        "shipping_reused": SHIP,
        "projection": proj,
        "best_candidate": "taew2_1 encoder (TAEHV, fp16, end-padding, [0,1] input, output as-is), TensorRT FP16 or torch.compile",
    }
    (EB / "results.json").write_text(json.dumps(out, indent=1))
    write_md(out, spd, t_trt, t_cmp, t_eag, t_lv, t_wan)
    print("wrote results.json, results.md")


def write_md(out, spd, t_trt, t_cmp, t_eag, t_lv, t_wan):
    A = {r["arm"]: r for r in out["arms"]}
    v = out["inputs"]["validation_vs_inloop_encode"]
    P = out["projection"]
    L = []
    w = L.append
    w("# Motion re-encode bake-off: tiny encoders vs the Wan 2.1 encoder at 576x320 (2026-09-26)\n")
    w("**Hardware:** NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM (nvidia-smi), driver 595.84, torch 2.7.1+cu128, "
      "CUDA runtime 12.8, cuDNN 9.7.1. TensorRT 10.3.0 was borrowed read-only from `/workspace/.venvs/musetalk_trt_stagewise` "
      "(the same symlinks the decoder bake-off used).\n")
    w("**Labels:**\n- Every number measured here is fresh local GPU inference on the RTX 4070 SUPER, 2026-09-26 "
      f"(quality {out['dates_utc']['quality']}, speed {out['dates_utc']['speed']['cudnn_benchmark=0']} / {out['dates_utc']['speed']['cudnn_benchmark=1']}).\n"
      "- Shipping-build numbers are reused from `final-v4-lean-r01` (2026-09-22), not re-run.\n"
      "- The FPS projections are arithmetic on those reused totals. They are not measurements.\n")
    w("**Co-resident load:** none. Every nvidia-smi snapshot taken during the runs lists only this bake-off's own process, which held the SoulX lease.\n")
    w("**Workload:** the per-window motion re-encode, `flash_head_pipeline.py:502-504`. Input: the trailing 5 colour-corrected frames of a window, "
      "(1,3,5,576,320) bf16 in [-1,1]. Output: 2 DiT-normalised latents, (16,2,72,40) bf16. Slot 0 is frame 0 encoded alone; slot 1 covers frames 1-4.\n")

    w("## 1. Inputs, and how they were validated\n")
    w(f"- **Clips:** {out['inputs']['clips']['n']} cond_frame clips, one per window, from 4 fixtures (indian150-a, tts-plosives, "
      "tts-open-vowels, tts-sibilants; seed 50; windows 0-8). They were built from the saved DiT latents in "
      "`benchmarks/tiny_vae_20260926/latents/` exactly as the pipeline builds them:\n"
      "  1. Stock Wan decode, bf16 eager (the decoder bake-off showed this reproduces the in-loop decode bit-exactly).\n"
      "  2. Lean trim `[:, :, 5:]`.\n"
      "  3. `match_and_blend_colors_torch` against `reference-150x.png` at strength 1.0, in bf16.\n"
      "  4. Keep the trailing 5 frames.\n")
    w(f"- **Validation against the dump:** the motion latents that window w+1 injects in-loop (`latent[w+1][:, :2]`) are the "
      f"torch.compiled Wan encode of window w's cond_frame. On the 32 window pairs where both exist, my eager Wan encode of the "
      f"reconstructed clip matches them to rel L2 at most {v['rel_l2_max']:.4f} and max abs {v['maxabs_max']:.3f} "
      f"(nRMSE {v['slot0_nrmse']:.4f} / {v['slot1_nrmse']:.4f}, cosine {v['slot0_cos']:.5f} / {v['slot1_cos']:.5f}). "
      "So the clips are the pipeline's clips, and this difference is the eager-vs-compiled noise floor.\n")
    w("- **Reference:** `WanVAE.encode` with stock weights, bf16 eager. The shipping ft4 weights have a bf16-identical encoder (checked on CPU by the earlier VAE-contract analysis, reused), so this is also the shipping encoder.\n")

    w("## 2. Conventions, resolved empirically\n")
    w("Error is nRMSE against the Wan encode, averaged over 36 clips and 16 channels, for slot 0 / slot 1.\n")
    w("| variant | slot 0 nRMSE | slot 1 nRMSE | cos slot 0 / 1 |\n|---|---:|---:|---|")
    for k, c in out["conventions_resolved"]["search"].items():
        w(f"| {k} | {f(c['slot0_nrmse'])} | {f(c['slot1_nrmse'])} | {f(c['slot0_cos'],4)} / {f(c['slot1_cos'],4)} |")
    w("")
    w("`frontK` means K copies of frame 0 are prepended and the rest of the padding to 8 frames is copies of the last frame. "
      "front0 is upstream `encode_video` (end padding); front3 is the Wan-style guess that frame 0 sits alone in slot 0. "
      "`in01` feeds (x+1)/2; `inpm1` feeds x as is. `asis` uses the output directly; `raw2norm` applies (e-mean)*inv_std.\n")
    w("**Resolved conventions**, which match the decoder bake-off's decoder conventions:\n"
      "- **taew2_1:** input in [0,1]; end padding (front0); output used as is, because it is already DiT-normalised.\n"
      "- **lighttaew2_1:** input in [0,1]; end padding; output is un-normalised and needs (e-mean)*inv_std.\n"
      "- **lightvaew2_1:** the Wan protocol unchanged, i.e. input in [-1,1] and `encode(x, wan_scale)`.\n")
    w("**Wrong conventions fail badly:**\n"
      "- A wrong output normalisation gives nRMSE 0.85-1.9.\n"
      "- Input in [-1,1] gives 3-13.\n"
      "- The Wan-style front pad leaves slot 0 unchanged but raises slot 1 from 0.089 to 0.29. That is a temporal lag of the lip state (see the visual).\n")
    w(f"**Precision and implementation checks** (all on taew2_1):\n"
      "- fp16, fp32 and bf16 give the same error.\n"
      f"- My static encode core matches upstream `encode_video` exactly (max abs {out['conventions_resolved']['core_vs_upstream_encode_video_maxabs_fp32']}, fp32).\n"
      "- TensorRT FP16 matches eager fp16 to a max abs of 0.016.\n")

    w("## 3. Latent error vs the Wan encoder, per slot (36 clips)\n")
    w("How to read the columns:\n"
      "- **nRMSE** is RMSE divided by that channel's std of the Wan latent over all clips (slot 0 std 0.21-0.62, slot 1 std 0.35-0.92).\n"
      "- **cos** is cosine similarity over the flattened slot.\n"
      "- **offset** is the systematic per-channel mean offset, averaged over clips. It is given as the RMS over the 16 channels and the largest channel (channel number in brackets).\n")
    w("| arm | slot 0 nRMSE | slot 1 nRMSE | cos slot 0 / 1 (min) | slot 0 offset RMS / max | slot 1 offset RMS / max |\n|---|---:|---:|---|---|---|")
    for r in out["arms"]:
        w(f"| {r['arm']} | {f(r['slot0_nrmse_mean'])} | {f(r['slot1_nrmse_mean'])} | {f(r['slot0_cos_mean'],4)} / {f(r['slot1_cos_mean'],4)} "
          f"({f(r['slot0_cos_min'],4)} / {f(r['slot1_cos_min'],4)}) | {f(r['slot0_offset_rms'],4)} / {f(r['slot0_offset_maxabs'],3)} (ch {r['slot0_offset_maxabs_channel']}) "
          f"| {f(r['slot1_offset_rms'],4)} / {f(r['slot1_offset_maxabs'],3)} (ch {r['slot1_offset_maxabs_channel']}) |")
    w("\nThe error is uniform across fixtures: taew2_1 scores 0.135-0.139 on slot 0 and 0.088-0.091 on slot 1 in every fixture. Windows 0 and 8 are no worse than the middle windows.\n")

    w("## 4. Pixel view of what the DiT is conditioned on\n")
    w("Each arm's 2 latents were decoded with the stock Wan decoder into 5 frames.\n"
      "- **vs Wan round trip** compares them with the decode of the Wan encoder's latents. This is what the DiT sees differently.\n"
      "- **round trip vs input** compares them with the clip that was encoded.\n"
      f"- For scale, the Wan encoder's own round trip scores {f(out['wan_roundtrip_vs_clip']['psnr_full']['mean'],2)} dB full frame / "
      f"{f(out['wan_roundtrip_vs_clip']['psnr_mouth']['mean'],2)} dB mouth against the input.\n")
    w("| arm | vs Wan round trip: PSNR full / mouth (worst) | mean RGB offset /255, frame 0 | mean RGB offset /255, frames 1-4 | round trip vs input: PSNR full / mouth |\n|---|---|---|---|---|")
    for r in out["arms"]:
        o0 = ", ".join(f"{x:+.2f}" for x in r["pix_vs_wan_rt_rgb_offset_frame0"])
        o1 = ", ".join(f"{x:+.2f}" for x in r["pix_vs_wan_rt_rgb_offset_frames1_4"])
        w(f"| {r['arm']} | {f(r['pix_vs_wan_rt_psnr_full'],2)} / {f(r['pix_vs_wan_rt_psnr_mouth'],2)} ({f(r['pix_vs_wan_rt_psnr_full_worst'],2)} / {f(r['pix_vs_wan_rt_psnr_mouth_worst'],2)}) "
          f"| {o0} | {o1} | {f(r['roundtrip_vs_input_psnr_full'],2)} / {f(r['roundtrip_vs_input_psnr_mouth'],2)} |")
    w("")

    t = A["taew2_1 encoder"]
    fx = A["REJECTED latent feedback last2-fix0"]
    sh = A["Wan encoder on shipping-decoder frames (perturbation the shipping build already carries)"]
    td = A["Wan encoder on taew2_1-decoder frames (perturbation a tiny decoder brings)"]
    w("## 5. What the DiT will see, and the drift question\n")
    w("The 2 motion latents replace `noise[:, :2]` at every denoising step of the next window. Under the shipping overlap-skip decoder they are never decoded, so only the DiT sees them.\n")
    w("**Why the latent-feedback arms drifted.** They fed the DiT's own output latents back into the next window. Its slot 1 differs from the Wan encode of the colour-corrected frames by:\n"
      f"- a systematic offset of RMS {f(fx['slot1_offset_rms'],3)} over channels, up to {f(fx['slot1_offset_maxabs'],3)} on channel {fx['slot1_offset_maxabs_channel']}, with channel {fx['slot1_offset_maxabs_channel']} keeping the sign of its mean in {100*same_sign_frac('REJECTED latent feedback last2-fix0 (Wan enc f0 + DiT latent 8)', 1, fx['slot1_offset_maxabs_channel']):.0f}% of clips (all channels: {100*same_sign_frac('REJECTED latent feedback last2-fix0 (Wan enc f0 + DiT latent 8)', 1):.0f}%);\n"
      f"- nRMSE {f(fx['slot1_nrmse_mean'])}.\n\n"
      f"In pixels that is {', '.join(f'{x:+.2f}' for x in fx['pix_vs_wan_rt_rgb_offset_frames1_4'])} /255 (RGB). "
      "Nothing re-anchors those latents to colour-corrected pixels, so the offset compounds from window to window. That matches the measured drift of 2.56-2.77/255 (slope R +0.34/window).\n")
    w("**How the taew2_1 encoder compares on the same windows:**\n"
      f"- **Systematic offset:** RMS {f(t['slot1_offset_rms'],4)} on slot 1, **{fx['slot1_offset_rms']/t['slot1_offset_rms']:.1f}x smaller** than latent feedback; the largest channel is {f(t['slot1_offset_maxabs'],3)}, {fx['slot1_offset_maxabs']/t['slot1_offset_maxabs']:.1f}x smaller.\n"
      f"- **Slot 0:** RMS {f(t['slot0_offset_rms'],4)}, its largest term. Channel 13 sits at +{f(t['slot0_offset_per_channel'][13],3)} ({t['slot0_offset_per_channel'][13]/out['wan_latent_per_channel_std'][0][13]:.2f} of that channel's std) and has the same sign in {100*same_sign_frac('taew2_1 (taew2_1 fp16 front0 in01 asis)', 0, 13):.0f}% of clips. That is still {fx['slot1_offset_rms']/t['slot0_offset_rms']:.1f}x below latent feedback's slot 1.\n"
      f"- **In pixels:** at most {f(t['pix_vs_wan_rt_rgb_offset_mean_absmax'],2)}/255 mean RGB offset, against {f(fx['pix_vs_wan_rt_rgb_offset_mean_absmax'],2)}/255 for last2-fix0.\n")
    w("**Why its offset should not compound.** The tiny encoder still encodes the colour-corrected pixels every window, so the pixel colour-correction loop stays closed. Its bias is re-applied to freshly corrected frames each window, which gives a bounded constant offset, not a random walk.\n")
    w("**Calibration against perturbations the DiT already copes with:**\n"
      f"- **The shipping build:** swapping the stock decoder for the shipping pruned decoder changes the Wan-encoded motion latents by nRMSE {f(sh['slot0_nrmse_mean'])} / {f(sh['slot1_nrmse_mean'])} (offset RMS {f(sh['slot0_offset_rms'],4)} / {f(sh['slot1_offset_rms'],4)}). "
      f"The shipping build already carries that perturbation, and its lip-sync and colour gates are in band (reused: correlation 0.970, mouth distance 2.2 px, colour drift 1.32/255; its edge ratio of 0.907 is below the band). The taew2_1 encoder's error is the same size: {f(t['slot0_nrmse_mean'])} / {f(t['slot1_nrmse_mean'])}. "
      "On the shipping decoder's own frames, the input of an encoder-only arm, it is again the same (0.133 / 0.090).\n"
      f"- **A tiny decoder:** the taew2_1 decoder recommended by the decoder bake-off perturbs the Wan-encoded motion latents about twice as much, {f(td['slot0_nrmse_mean'])} / {f(td['slot1_nrmse_mean'])}. "
      "Adding the taew2_1 encoder on top barely raises that (combo 0.324 / 0.194), and on taew2_1-decoded frames its own error is smaller still (0.108 / 0.087). "
      "In a full tiny-VAE arm the decoder, not the encoder, is the dominant risk.\n")
    w("**Visual check** (`visual/motion_latents_*.png`, inspected by eye):\n"
      "- taew2_1, lightvaew2_1 and lighttaew2_1 round trips cannot be told apart from the Wan round trip at 1x or at the 2x mouth crop. Their x8 difference maps show only low-level texture noise.\n"
      "- In the open-vowel clip the lips close between frame 0 and frame 4. The Wan-style front pad keeps the mouth open at frame 4, a lag that would hurt lip sync.\n"
      "- last2-fix0 shows a visible colour and contrast shift: darker skin, and pinker, more saturated lips.\n")

    w("## 6. Speed of one re-encode call at 576x320\n")
    w("Method: the full contract from bf16 [-1,1] NCTHW in to (16,2,72,40) bf16 DiT-normalised out, including the tiny encoders' layout, range, padding and normalisation conversions. "
      "Input is a real clip (indian150-a window 4). Timing uses CUDA events, 3 warm-up calls and 20 timed calls, and reports the median.\n")
    w("| case | median ms (cuDNN benchmark off = harness default) | median ms (benchmark on) | peak over resident MiB | nRMSE vs Wan eager (this clip) |\n|---|---:|---:|---:|---:|")
    for k, r in spd.items():
        ws = f" + {r['trt_workspace_mib']:.0f} workspace" if r.get("trt_workspace_mib") else ""
        w(f"| {k} | {f(r['median_ms_cudnn_bench_off'],2)} | {f(r['median_ms_cudnn_bench_on'],2) if r['median_ms_cudnn_bench_on'] else '-'} | {r['peak_over_resident_mib']:.0f}{ws} | {f(r['nrmse_vs_wan_eager_this_clip'],4)} |")
    w(f"| shipping harness motion_encode stage (reused, final-v4-lean-r01) | {SHIP['motion_encode_ms_per_window']:.1f} ms/window | | | |")
    w("")
    w("Notes on the speed table:\n"
      f"- **Calibration:** the stock Wan encoder, compiled the way `--compile-vae-encode` compiles it, measures {t_wan:.1f} ms here, against the harness's 117.0 ms/window, so these standalone timings carry over.\n"
      f"- **taew2_1:** {t_trt:.2f} ms as a TensorRT FP16 engine, {t_cmp:.2f} ms with torch.compile, {t_eag:.2f} ms eager. That is **{t_wan/t_trt:.0f}x** faster than the shipping encoder and saves about {t_wan-t_trt:.0f} ms per window.\n"
      "- **TensorRT build:** 7.6 s; the engine uses a 382 MiB workspace.\n"
      "- **Other encoders:** lighttaew2_1 has the same architecture and speed. lightvaew2_1 needs 19 ms compiled.\n"
      "- **Settings:** channels_last hurts eager and makes no difference compiled. The cuDNN benchmark setting changes the fp16 and Wan timings by 4% or less; bf16 taew2_1 compiled goes from 7.68 to 6.64 ms.\n")

    w("## 7. Projection (arithmetic on the reused 2026-09-22 stage totals, NOT measured)\n")
    w(f"Formula: `{P['formula']}`.\n")
    w("| arm | projected generation s | projected useful FPS |\n|---|---:|---:|")
    w(f"| shipping (reused, measured 2026-09-22) | 8.273 | 30.22 |")
    for k in ("encoder_only_taew2_1_trt", "encoder_only_taew2_1_compile", "encoder_only_taew2_1_eager", "encoder_only_lightvaew2_1_compile",
              "taew2_1_decoder_trt_only (decoder bake-off projection)", "taew2_1_decoder_trt_plus_taew2_1_encoder_trt"):
        w(f"| {k} | {P[k]['generation_s']:.3f} | {P[k]['useful_fps']:.1f} |")
    w("\nThe encoder alone buys about +14% (30.2 to 34.3 FPS). With the taew2_1 decoder as well the projection is about 67 FPS (2.2x). "
      "That is still not 5x: the DiT, at about 343 ms per window, is untouched and becomes about 83% of the remaining time.\n")

    w("## 8. Recommendation\n")
    w("**The taew2_1 encoder is safe to test end to end, and it is the one to test.** Summary of the evidence:\n"
      "- **Latent error:** equal to the perturbation the shipping build already carries, and smaller than what the tiny decoder introduces.\n"
      "- **Offsets:** 4-10x smaller than the latent-feedback offsets that caused drift.\n"
      "- **Colour loop:** it keeps the colour-correction loop closed.\n"
      "- **Speed:** 18x faster than the shipping encoder.\n")
    w("**Contract** (fp16):\n"
      "```\n"
      "x = clip[0].transpose(0, 1).half()\n"
      "x = (x + 1) / 2\n"
      "x = cat([x, x[-1:].expand(3, ...)])   # end-pad to 8 frames\n"
      "z = encoder(x)                        # (2, 16, 72, 40)\n"
      "z = z.transpose(0, 1).bfloat16()      # (16, 2, 72, 40)\n"
      "```\n"
      "- The encoder is stateless per call, so it needs no reset. It can be a TensorRT FP16 engine, or torch.compile at +0.2 ms.\n"
      "- Do NOT use the Wan-style front pad.\n")
    w("**Order of end-to-end arms** (each gated with review.py, colour drift over 9 windows against a regenerated control raw.npy, and a labelled video):\n"
      "1. Encoder-only: the shipping build plus `--tiny-encoder taew2_1`. This isolates the encoder. Watch the colour-drift step at window 1, which is the first window conditioned on tiny-encoded latents, and the channel-13 slot-0 bias.\n"
      "2. The taew2_1 stream decoder with the Wan encoder.\n"
      "3. Both.\n")
    w("**The other candidates:**\n"
      "- **lightvaew2_1:** second choice. Its latent error is similar (0.155 / 0.096), but it shows a -0.5/255 green offset and is 3x slower (19 ms).\n"
      "- **lighttaew2_1:** worst on every metric (0.191 / 0.114, up to -0.65/255). Not recommended.\n")
    w("**Constraints for the integration:**\n"
      "- The reference-image encode must stay on the Wan encoder.\n"
      "- `--latent-feedback` must stay off.\n")

    w("## 9. Caveats\n")
    w("- **Single-step only.** Every measurement here is one window: the latents the next window would receive. The closed-loop effects over 9 windows (the DiT's response, lip sync, colour drift, mouth edge ratio) need the end-to-end harness arm and cannot be inferred from this bake-off.\n"
      "- **Latent source.** The latents come from the decoder bake-off's dump, which used flash2 attention and the stock eager decoder in the loop, because the sage2 and TensorRT 10.9 dependency directories are missing. They are real SoulX DiT outputs but not bit-identical to the shipping sage2 build. The shipping build cannot be reproduced on this machine until those dependencies are restored.\n"
      "- **Clip source.** The primary clips come from the stock decoder in window mode. The shipping build produces cond_frame through the pruned decoder in overlap-skip stream mode. The shipping-decoder arm here used window mode.\n"
      "- **TensorRT.** TensorRT 10.3 was borrowed from another venv. Production needs a proper install. The engine and ONNX files were deleted after timing; their sha256 are in speed.json.\n"
      "- **Colour-drift gate.** It still needs a regenerated control raw.npy, because every raw.npy on disk was deleted.\n")
    (EB / "results.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
