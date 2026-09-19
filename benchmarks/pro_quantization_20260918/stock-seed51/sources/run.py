"""Matched stock/quantized PRO experiment; never imports the live Engine.

GPU hardware, runtime, resident load and all input identities are saved per run.
Warmup/capture is excluded from generation timing; each repeat resets history/RNG.
"""
import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import sys
import shutil
from pathlib import Path
import threading
import time

import librosa
import numpy as np
from PIL import Image
import torch

from benchmarks.pro_lite_teeth_20260917.run_variant import (
    gpu_snapshot, process_snapshot, sha256, monitor_generation, summarize_samples,
)
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.pro_quantization import quantize_pro_ffns, optimize_wan_vae

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "benchmarks/pro_lite_150x_20260917"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--precision", choices=("none", "fp8", "int8"), default="none")
    ap.add_argument("--vae", choices=("none", "channels_last", "pointwise", "compiled"), default="none")
    ap.add_argument("--compile-ffn", action="store_true")
    ap.add_argument("--attention", choices=("flash2", "sage1"), default="flash2")
    ap.add_argument("--optimized-dit", action="store_true")
    ap.add_argument("--compile-dit", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--frames", type=int, default=250)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--save-raw", action="store_true")
    args = ap.parse_args()
    if args.frames < 1 or args.repeats < 1:
        ap.error("frames and repeats must be positive")
    os.chdir(ROOT)
    lease = acquire_gpu_lease()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    result = dict(status="starting", date_utc=datetime.now(timezone.utc).isoformat(),
                  execution="fresh local GPU inference", gpu=gpu_snapshot(),
                  processes_before_load=process_snapshot(), torch=torch.__version__,
                  cuda_runtime=torch.version.cuda, physical_vram_class="12 GB",
                  profile=dict(width=320, height=576, fps=25, frames=args.frames,
                               steps=4, seed=args.seed, shift=5, motion_latents=2,
                               motion_frames=5, new_frames_per_chunk=28, strength=1.0,
                               framing="1.50x", precision=args.precision,
                               vae=args.vae, compile_ffn=args.compile_ffn, attention=args.attention,
                               optimized_dit=args.optimized_dit, compile_dit=args.compile_dit),
                  reference=dict(path=str(FIXTURE / "reference-150x.png"),
                                 sha256=sha256(FIXTURE / "reference-150x.png")),
                  audio=dict(path=str(FIXTURE / "audio.wav"), sha256=sha256(FIXTURE / "audio.wav")),
                  source_sha256={str(p): sha256(p) for p in (
                      Path(__file__), ROOT / "soulx_rtc/pro_quantization.py",
                      ROOT / "flash_head/src/pipeline/flash_head_pipeline.py",
                      ROOT / "flash_head/src/modules/flash_head_model.py")})

    def save():
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    save()
    (args.output / "sources").mkdir()
    for source in result['source_sha256']:
        shutil.copyfile(source, args.output / "sources" / Path(source).name)
    try:
        from flash_head.src.pipeline import flash_head_pipeline as implementation
        implementation.COMPILE_MODEL = False
        implementation.COMPILE_VAE = False
        implementation.USE_PARALLEL_VAE = False
        from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline
        from flash_head.inference import get_audio_embedding, run_pipeline
        from soulx_rtc.experiment import record
        from flash_head.src.modules import flash_head_model

        # Make the baseline independent of whether optional attention is installed.
        flash_head_model.SAGE_ATTN_AVAILABLE = False
        flash_head_model.FLASH_ATTN_3_AVAILABLE = False
        result["attention_backend"] = "flash_attention_2" if flash_head_model.FLASH_ATTN_2_AVAILABLE else "sdpa"
        if args.attention == "sage1":
            sys.path.insert(0, str(ROOT / ".pro-quant-deps"))
            from sageattention import sageattn
            reference_attention = flash_head_model.flash_attention
            check = {}
            result["attention_backend"] = "sageattention_1.0.6_self_only_flash2_cross"
            result["attention_first_input_check"] = check
            def attention(q, k, v, num_heads, compatibility_mode=False):
                if compatibility_mode or q.shape[1] != 6480 or k.shape[1] != 6480:
                    return reference_attention(q, k, v, num_heads, compatibility_mode)
                shape = (q.shape[0], q.shape[1], num_heads, q.shape[2] // num_heads)
                # Sage 1 smooths K in place; clone to preserve caller-owned tensors.
                out = sageattn(q.reshape(shape), k.reshape(shape).clone(), v.reshape(shape),
                               tensor_layout="NHD", is_causal=False).reshape_as(q)
                if not check:
                    expected = reference_attention(q, k, v, num_heads)
                    delta = out.float() - expected.float()
                    check.update(shape=list(q.shape), relative_l2=float(delta.norm()/expected.float().norm()),
                                 max_abs=float(delta.abs().max()), finite=bool(out.isfinite().all()))
                return out
            flash_head_model.flash_attention = attention
        weights = ROOT / "models/SoulX-FlashHead-1_3B"
        result["checkpoint_sha256"] = {
            name: sha256(weights / name) for name in (
                "Model_Pro/diffusion_pytorch_model.safetensors", "Model_Pro/config.json",
                "VAE_Wan/Wan2.1_VAE.pth")}
        result["status"] = "loading"; save()
        pipeline = FlashHeadPipeline(str(weights), "pro", str(ROOT / "models/wav2vec2-base-960h"),
                                     device="cuda", param_dtype=torch.bfloat16, use_usp=False)
        pipeline.audio_encoder.eval().requires_grad_(False)
        result["quantization"] = quantize_pro_ffns(pipeline.model, args.precision, args.exclude)
        if args.compile_ffn:
            for block in pipeline.model.blocks:
                block.ffn = torch.compile(block.ffn, fullgraph=True, dynamic=False)
        result["vae_optimization"] = optimize_wan_vae(pipeline.vae, args.vae)
        torch.cuda.empty_cache()
        pipeline.prepare_params(str(FIXTURE / "reference-150x.png"), (576, 320), 33, 5, 4,
                                seed=args.seed, shift=5, color_correction_strength=1.0,
                                use_face_crop=False)
        if args.compile_dit and not args.optimized_dit:
            raise ValueError("compile-dit requires optimized-dit to avoid complex rotary operations")
        if args.optimized_dit:
            from flash_head.src.modules.flash_head_model import prepare_rotary
            model = pipeline.model
            model.freqs = model.freqs.to(pipeline.device)
            grid = ((pipeline.frame_num-1)//pipeline.config.vae_stride[0]+1,
                    pipeline.lat_h//model.patch_size[1], pipeline.lat_w//model.patch_size[2])
            with torch.no_grad():
                rotary = prepare_rotary(model.freqs, grid, True)
                times = tuple(model.prepare_time(t) for t in pipeline.timesteps[:-1])
            base_forward = model.forward
            if args.compile_dit:
                base_forward = torch.compile(base_forward, dynamic=False, fullgraph=False)
            chunk = {}
            base_generate = pipeline.generate
            def prepared_forward(*a, **kw):
                index = chunk['step']; chunk['step'] += 1
                return base_forward(*a, **kw, prepared_context=chunk['context'],
                                    cross_kv=chunk['kv'], rotary=rotary, prepared_time=times[index])
            @torch.no_grad()
            def prepared_generate(audio_embedding):
                chunk['context'],chunk['kv'] = model.prepare_conditioning(audio_embedding)
                chunk['step'] = 0
                try:
                    return base_generate(audio_embedding)
                finally:
                    chunk.clear()
            model.forward = prepared_forward
            pipeline.generate = prepared_generate
        audio, _ = librosa.load(FIXTURE / "audio.wav", sr=16000, mono=True)
        samples = math.ceil(args.frames * 16000 / 25)
        # Long tests repeat the same known audio; explicitly marked in metadata.
        audio = np.tile(audio, math.ceil(samples / len(audio)))[:samples]
        result["audio"]["repeated_for_long_test"] = samples > 160000
        slice_samples = 28 * 16000 // 25
        padded = np.pad(audio, (0, (-len(audio)) % slice_samples))
        slices = padded.reshape(-1, slice_samples)
        events = []

        def instrument(owner, attribute, stage):
            original = getattr(owner, attribute)
            def wrapped(*a, **kw):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                out = original(*a, **kw)
                end.record()
                events.append((stage, start, end))
                return out
            setattr(owner, attribute, wrapped)

        instrument(pipeline.model, "forward", "dit")
        instrument(pipeline.vae, "decode", "vae_decode")
        instrument(pipeline.vae, "encode", "motion_encode")
        instrument(pipeline, "preprocess_audio", "audio")
        hooks = []
        capture_stats = []
        counters = defaultdict(int)
        if args.capture:
            cap = args.output / "captures"
            cap.mkdir()
            def hook_for(i):
                def hook(module, inputs):
                    x = inputs[0].detach()
                    invocation = counters[i]; counters[i] += 1
                    xf = x.float()
                    capture_stats.append(dict(block=i, invocation=invocation,
                        chunk=invocation // 4, step=invocation % 4, shape=list(x.shape),
                        rms=float(xf.square().mean().sqrt()), absmax=float(xf.abs().max())))
                    if i in (0, 14, 29) and invocation == 0:
                        torch.save(x.cpu(), cap / f"ffn-{i:02d}.pt")
                return hook
            for i, block in enumerate(pipeline.model.blocks):
                hooks.append(block.ffn.register_forward_pre_hook(hook_for(i)))
            decode = pipeline.vae.decode
            def capture_decode(z):
                if not (cap / "vae-input.pt").exists():
                    torch.save(z.detach().cpu(), cap / "vae-input.pt")
                return decode(z)
            pipeline.vae.decode = capture_decode

        def reset():
            pipeline.reset_person_name(pipeline.person_name)
            pipeline.generator.manual_seed(args.seed)
            events.clear()
            return deque([0.0] * 128000, maxlen=128000)

        result["status"] = "warming"; save()
        warm = time.perf_counter()
        cache = reset()
        for index in range(2):
            cache.extend(slices[index % len(slices)].tolist())
            embedding = get_audio_embedding(pipeline, np.asarray(cache, dtype=np.float32), 167, 200)
            run_pipeline(pipeline, embedding)
        torch.cuda.synchronize()
        result["warmup_s"] = time.perf_counter() - warm
        for handle in hooks:
            handle.remove()
        if args.capture:
            pipeline.vae.decode = decode
            (cap / "activation-statistics.json").write_text(json.dumps(capture_stats, indent=2) + "\n")
        result["runs"] = []
        for repeat in range(args.repeats):
            cache = reset()
            generated, chunk_times, stage_totals = [], [], defaultdict(float)
            torch.cuda.reset_peak_memory_stats()
            result["status"] = f"generating_repeat_{repeat}"; save()
            resources, stop = [], threading.Event()
            started = time.perf_counter()
            monitor = threading.Thread(target=monitor_generation, args=(stop, resources, started), daemon=True)
            monitor.start()
            try:
                for index, audio_slice in enumerate(slices):
                    chunk_start = time.perf_counter()
                    cache.extend(audio_slice.tolist())
                    embedding = get_audio_embedding(pipeline, np.asarray(cache, dtype=np.float32), 167, 200)
                    frames = run_pipeline(pipeline, embedding)[5:].cpu().numpy().astype(np.uint8)
                    torch.cuda.synchronize()
                    chunk_times.append(time.perf_counter() - chunk_start)
                    if repeat == 0:
                        generated.append(frames)
                    for stage, begin, end in events:
                        stage_totals[stage] += begin.elapsed_time(end) / 1000
                    events.clear()
                generation_s = time.perf_counter() - started
            finally:
                stop.set(); monitor.join()
            row = dict(repeat=repeat, generation_s=generation_s, useful_fps=args.frames / generation_s,
                       chunk_times_s=chunk_times, stage_seconds=dict(stage_totals),
                       peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                       resource_summary=summarize_samples(resources))
            result["runs"].append(row)
            (args.output / f"resources-{repeat}.json").write_text(json.dumps(resources) + "\n")
            if repeat == 0:
                rgb = np.concatenate(generated, axis=0)[:args.frames]
                assert len(rgb) == args.frames
                result["raw_rgb_sha256"] = hashlib.sha256(rgb.tobytes()).hexdigest()
                record(args.output / "video.mp4", [rgb], audio, 25)
                if args.save_raw:
                    np.save(args.output / "raw.npy", rgb)
                for timestamp in (0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 9.5, 19.5, 29.5, 39.5, 49.5, 59.5):
                    if round(timestamp * 25) < len(rgb):
                        Image.fromarray(rgb[round(timestamp * 25)]).save(args.output / f"frame-{timestamp:.1f}s.png")
                del rgb, generated
            save()
        result.update(status="complete", gpu_after=gpu_snapshot(), processes_after=process_snapshot())
        save()
        print(json.dumps({"output":str(args.output), "runs":result["runs"]}), flush=True)
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc), gpu_at_failure=gpu_snapshot())
        save()
        raise


if __name__ == "__main__":
    main()
