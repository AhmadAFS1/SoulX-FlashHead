"""Run one official SoulX FlashHead variant for the controlled teeth A/B."""
import argparse
import hashlib
import json
import os
import subprocess
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import librosa
import numpy as np
from PIL import Image
import psutil
import torch


ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models/SoulX-FlashHead-1_3B"
WAV2VEC = ROOT / "models/wav2vec2-base-960h"
REFERENCE = ROOT / "benchmarks/distance_lipsync/evidence-indian-male-closer-20260916/reference-closer-125.png"
AUDIO = ROOT / "benchmarks/comparison-10s.wav"
WIDTH, HEIGHT, FPS, STEPS, SEED, FRAMES = 320, 576, 25, 4, 50, 250


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot():
    return subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,driver_version,utilization.gpu",
        "--format=csv,noheader",
    ], text=True).strip()


def process_snapshot():
    query = subprocess.run([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader",
    ], text=True, capture_output=True, check=True)
    return query.stdout.strip().splitlines()


def monitor_generation(stop, samples, started, interval_s=0.5):
    """Sample whole-device and process resources during the denoising loop."""
    process = psutil.Process(os.getpid())
    process.cpu_percent(None)
    psutil.cpu_percent(None)
    while not stop.is_set():
        virtual = psutil.virtual_memory()
        sample = {
            "elapsed_s": time.perf_counter() - started,
            "system_cpu_percent": psutil.cpu_percent(None),
            "process_cpu_percent": process.cpu_percent(None),
            "system_ram_used_mib": virtual.used / 2**20,
            "system_ram_percent": virtual.percent,
            "process_rss_mib": process.memory_info().rss / 2**20,
            "torch_allocated_mib": torch.cuda.memory_allocated() / 2**20,
            "torch_reserved_mib": torch.cuda.memory_reserved() / 2**20,
        }
        try:
            gpu = subprocess.check_output([
                "nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ], text=True).strip().split(",")
            sample.update({
                "gpu_utilization_percent": float(gpu[0]),
                "gpu_vram_used_mib": float(gpu[1]),
                "gpu_vram_total_mib": float(gpu[2]),
            })
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            sample["gpu_sample_error"] = str(exc)
        samples.append(sample)
        stop.wait(interval_s)


def summarize_samples(samples):
    metrics = (
        "gpu_utilization_percent", "gpu_vram_used_mib", "system_cpu_percent",
        "process_cpu_percent", "system_ram_used_mib", "system_ram_percent",
        "process_rss_mib", "torch_allocated_mib", "torch_reserved_mib",
    )
    summary = {"sample_count": len(samples)}
    for metric in metrics:
        values = [sample[metric] for sample in samples if metric in sample]
        if values:
            summary[metric] = {
                "mean": sum(values) / len(values), "min": min(values), "max": max(values),
            }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["lite", "pro"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--framing", default="1.25x")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    reference = args.reference.resolve()

    model_dir = CHECKPOINT / ("Model_Lite" if args.variant == "lite" else "Model_Pro")
    vae_files = ([CHECKPOINT / "VAE_LTX/config.json", CHECKPOINT / "VAE_LTX/diffusion_pytorch_model.safetensors"]
                 if args.variant == "lite" else [CHECKPOINT / "VAE_Wan/Wan2.1_VAE.pth"])
    result = {
        "status": "starting",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "execution": "fresh local GPU inference",
        "variant": args.variant,
        "gpu_before_load": gpu_snapshot(),
        "processes_before_load": process_snapshot(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "profile": {
            "width": WIDTH, "height": HEIGHT, "fps": FPS, "steps": STEPS,
            "seed": SEED, "frames": FRAMES, "precision": "BF16",
            "compile_model": False, "compile_vae": False,
            "audio_mode": "official streaming window", "color_correction_strength": 1.0,
        },
        "reference": {"path": str(reference), "sha256": sha256(reference), "framing": args.framing},
        "audio": {"path": str(AUDIO), "sha256": sha256(AUDIO), "sample_rate": 16000},
        "checkpoint": {
            "model_config": str(model_dir / "config.json"),
            "model_config_sha256": sha256(model_dir / "config.json"),
            "model_weights": str(model_dir / "diffusion_pytorch_model.safetensors"),
            "model_weights_bytes": (model_dir / "diffusion_pytorch_model.safetensors").stat().st_size,
            "model_weights_sha256": sha256(model_dir / "diffusion_pytorch_model.safetensors"),
            "vae_files": [{"path": str(p), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in vae_files],
        },
        "note": "Allocator peaks are process-local Torch values; whole-device snapshots include co-resident processes.",
    }

    def save():
        (args.output / "results.json").write_text(json.dumps(result, indent=2) + "\n")

    save()
    try:
        # Disable the upstream compile globals before pipeline construction. This
        # keeps both variants eager and avoids variant-specific compile caches.
        from flash_head.src.pipeline import flash_head_pipeline as pipeline_module
        pipeline_module.COMPILE_MODEL = False
        pipeline_module.COMPILE_VAE = False
        pipeline_module.USE_PARALLEL_VAE = False
        from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline
        from flash_head.inference import get_audio_embedding, run_pipeline
        from soulx_rtc.experiment import record

        result["status"] = "loading_model"; save()
        load_started = time.perf_counter()
        pipeline = FlashHeadPipeline(
            checkpoint_dir=str(CHECKPOINT), model_type=args.variant,
            wav2vec_dir=str(WAV2VEC), device="cuda", param_dtype=torch.bfloat16,
            use_usp=False,
        )
        torch.cuda.synchronize()
        result["model_load_s"] = time.perf_counter() - load_started
        result["gpu_after_load"] = gpu_snapshot()

        motion_latents = 2
        motion_frames = (motion_latents - 1) * pipeline.config.vae_stride[0] + 1
        slice_len = 33 - motion_frames
        result["architecture"] = {
            "vae_stride": list(pipeline.config.vae_stride),
            "patch_size": list(pipeline.config.patch_size),
            "latent_channels": pipeline.config.out_dim,
            "motion_frames": motion_frames,
            "new_frames_per_chunk": slice_len,
            "vae_family": "LTX" if args.variant == "lite" else "Wan2.1",
        }

        result["status"] = "preparing_reference"; save()
        prep_started = time.perf_counter()
        pipeline.prepare_params(
            cond_image_path_or_dir=str(reference), target_size=(HEIGHT, WIDTH),
            frame_num=33, motion_frames_num=motion_frames, sampling_steps=STEPS,
            seed=SEED, shift=5.0, color_correction_strength=1.0, use_face_crop=False,
        )
        torch.cuda.synchronize()
        result["reference_prepare_s"] = time.perf_counter() - prep_started

        audio, _ = librosa.load(AUDIO, sr=16000, mono=True)
        audio = audio[:160000]
        if len(audio) < 160000:
            audio = np.pad(audio, (0, 160000 - len(audio)))
        samples_per_slice = slice_len * 16000 // FPS
        remainder = len(audio) % samples_per_slice
        padded = np.pad(audio, (0, samples_per_slice - remainder)) if remainder else audio
        slices = padded.reshape(-1, samples_per_slice)
        cache = deque([0.0] * (8 * 16000), maxlen=8 * 16000)
        audio_end_idx, audio_start_idx = 8 * FPS, 8 * FPS - 33

        torch.cuda.reset_peak_memory_stats()
        result["status"] = "generating"; save()
        generated = []
        chunk_times = []
        generation_started = time.perf_counter()
        resource_samples = []
        monitor_stop = threading.Event()
        monitor = threading.Thread(
            target=monitor_generation,
            args=(monitor_stop, resource_samples, generation_started),
            name="generation-resource-monitor", daemon=True,
        )
        monitor.start()
        try:
            for index, audio_slice in enumerate(slices):
                if sum(len(chunk) for chunk in generated) >= FRAMES:
                    break
                cache.extend(audio_slice.tolist())
                started = time.perf_counter()
                embedding = get_audio_embedding(
                    pipeline, np.asarray(cache, dtype=np.float32), audio_start_idx, audio_end_idx)
                frames = run_pipeline(pipeline, embedding)[motion_frames:].cpu().numpy().astype(np.uint8)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                generated.append(frames)
                chunk_times.append(elapsed)
                print(json.dumps({"variant": args.variant, "chunk": index, "frames": sum(len(c) for c in generated), "seconds": elapsed}), flush=True)
        finally:
            monitor_stop.set()
            monitor.join()
        torch.cuda.synchronize()
        generation_s = time.perf_counter() - generation_started
        frames = np.concatenate(generated, axis=0)[:FRAMES]
        if len(frames) != FRAMES:
            raise RuntimeError(f"Expected {FRAMES} frames, got {len(frames)}")

        result.update({
            "generation_s": generation_s,
            "useful_fps": FRAMES / generation_s,
            "chunk_times_s": chunk_times,
            "resource_sampling": {
                "interval_s": 0.5,
                "scope": "generation loop only",
                "gpu_metrics_scope": "whole device, including co-resident processes",
                "cpu_note": "system CPU is host-normalized; process CPU can exceed 100% across cores",
                "samples": resource_samples,
                "summary": summarize_samples(resource_samples),
            },
            "peak_allocated_mib_generation": torch.cuda.max_memory_allocated() / 2**20,
            "peak_reserved_mib_generation": torch.cuda.max_memory_reserved() / 2**20,
            "raw_rgb_sha256": hashlib.sha256(frames.tobytes()).hexdigest(),
            "gpu_after_generation": gpu_snapshot(),
            "processes_after_generation": process_snapshot(),
        })
        video_path = args.output / "video.mp4"
        record(video_path, [frames], audio, FPS)
        result["video_sha256"] = sha256(video_path)
        for timestamp in (0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5):
            Image.fromarray(frames[round(timestamp * FPS)]).save(args.output / f"frame-{timestamp:.1f}s.png")
        result["status"] = "complete"; save()
    except Exception as exc:
        result.update({
            "status": "failed", "error_type": type(exc).__name__, "error": str(exc),
            "gpu_at_failure": gpu_snapshot(), "processes_at_failure": process_snapshot(),
        })
        save()
        raise


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
