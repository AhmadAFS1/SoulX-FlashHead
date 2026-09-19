"""Run one immutable, policy-driven PRO quantization v2 experiment.

Generation timing includes audio preparation, recurrence, RGB transfer, and
final padded generation. Media encoding and post-run evaluation are reported
separately so they cannot disappear from delivery measurements by accident.
"""
from __future__ import annotations

try:
    from .script_bootstrap import bootstrap_script_path
except ImportError:
    from script_bootstrap import bootstrap_script_path

bootstrap_script_path(__file__)

import argparse
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

import librosa
import numpy as np
from PIL import Image
import torch

from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_FIXTURES,
    DEFAULT_GPU_LOCK,
    ROOT,
    atomic_write_json,
    ensure_new_directory,
    environment_manifest,
    failure_record,
    gpu_details,
    gpu_processes,
    relative_path,
    resolve_fixture,
    resolve_path,
    sha256,
    snapshot_sources,
    summarize,
)
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.pro_attention_backends import (
    install_self_attention_backend,
    make_attention_backend,
    remove_self_attention_backend,
)
from soulx_rtc.pro_quantization import optimize_wan_vae
from soulx_rtc.pro_quantization_v2 import (
    BF16_SCHEME,
    apply_conversion_plan,
    build_conversion_plan,
    load_policy,
    write_resolved_policy,
)
from soulx_rtc.pro_vae_quantization import (
    EngineConvolutionBackend,
    install_decoder_adapters,
    load_decoder_plan,
    remove_decoder_adapters,
)


class Tee:
    """Mirror runner output into a retained artifact without hiding the terminal."""

    def __init__(self, original, destination) -> None:
        self.original = original
        self.destination = destination

    def write(self, text: str) -> int:
        self.original.write(text)
        return self.destination.write(text)

    def flush(self) -> None:
        self.original.flush()
        self.destination.flush()


WINDOW_HISTORY_FRAMES = 5
WINDOW_USEFUL_FRAMES = 28
WINDOW_MODEL_FRAMES = WINDOW_HISTORY_FRAMES + WINDOW_USEFUL_FRAMES


def _fixed_profile(frames: int, seed: int, policy: dict[str, Any]) -> dict[str, Any]:
    return {
        "width": 320,
        "height": 576,
        "fps": 25,
        "frames": frames,
        "steps": 4,
        "seed": seed,
        "shift": 5,
        "motion_latents": 2,
        "motion_frames": WINDOW_HISTORY_FRAMES,
        "new_frames_per_chunk": WINDOW_USEFUL_FRAMES,
        "model_frames_per_chunk": WINDOW_MODEL_FRAMES,
        "strength": 1.0,
        "framing": "1.50x",
        "policy": policy["name"],
        "base": policy["base"],
    }


def _sources() -> list[Path]:
    return [
        Path(__file__),
        ROOT / "benchmarks/pro_quantization_v2_20260918/common.py",
        ROOT / "soulx_rtc/pro_quantization_v2.py",
        ROOT / "soulx_rtc/pro_attention_backends.py",
        ROOT / "soulx_rtc/pro_vae_quantization.py",
        ROOT / "soulx_rtc/gpu_lease.py",
        ROOT / "flash_head/src/modules/flash_head_model.py",
        ROOT / "flash_head/src/pipeline/flash_head_pipeline.py",
        ROOT / "flash_head/wan/modules/vae.py",
    ]


def _install_prepared_generation(pipeline, *, compile_dit: bool) -> dict[str, Any]:
    """Install sequential-only prepared conditioning after final conversion/layout."""
    from flash_head.src.modules.flash_head_model import prepare_rotary

    model = pipeline.model
    model.freqs = model.freqs.to(pipeline.device)
    grid = (
        (pipeline.frame_num - 1) // pipeline.config.vae_stride[0] + 1,
        pipeline.lat_h // model.patch_size[1],
        pipeline.lat_w // model.patch_size[2],
    )
    with torch.no_grad():
        rotary = prepare_rotary(model.freqs, grid, True)
        times = tuple(model.prepare_time(t) for t in pipeline.timesteps[:-1])
    base_forward: Callable[..., torch.Tensor] = model.forward
    if compile_dit:
        base_forward = torch.compile(base_forward, dynamic=False, fullgraph=False)
    base_generate = pipeline.generate
    state: dict[str, Any] = {}

    def prepared_forward(*args, **kwargs):
        index = state["step"]
        state["step"] += 1
        return base_forward(
            *args,
            **kwargs,
            prepared_context=state["context"],
            cross_kv=state["cross_kv"],
            rotary=rotary,
            prepared_time=times[index],
        )

    @torch.no_grad()
    def prepared_generate(audio_embedding):
        state["context"], state["cross_kv"] = model.prepare_conditioning(audio_embedding)
        state["step"] = 0
        try:
            return base_generate(audio_embedding)
        finally:
            state.clear()

    model.forward = prepared_forward
    pipeline.generate = prepared_generate
    return {
        "enabled": True,
        "compile_dit": compile_dit,
        "rotary_shape": list(rotary.shape),
        "time_steps": len(times),
        "concurrency": "sequential offline only; mutable state is reset after each generated chunk",
    }


def _instrument(owner, attribute: str, stage: str, events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]]) -> None:
    original = getattr(owner, attribute)

    def wrapped(*args, **kwargs):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        value = original(*args, **kwargs)
        end.record()
        events.append((stage, begin, end))
        return value

    setattr(owner, attribute, wrapped)


def _audio_slices(fixture: dict[str, Any], frames: int) -> tuple[np.ndarray, np.ndarray]:
    if fixture["sample_rate"] != 16000:
        raise ValueError("The current offline PRO audio embedding path requires a 16 kHz fixture")
    if fixture["fps"] != 25:
        raise ValueError("The fixed PRO v2 contract requires 25 FPS fixtures")
    audio, rate = librosa.load(fixture["audio"]["path"], sr=fixture["sample_rate"], mono=True)
    if rate != fixture["sample_rate"]:
        raise ValueError("Fixture audio was not decoded at the requested sample rate")
    requested_samples = math.ceil(frames * fixture["sample_rate"] / fixture["fps"])
    if requested_samples > len(audio):
        audio = np.tile(audio, math.ceil(requested_samples / len(audio)))
    effective_audio = np.asarray(audio[:requested_samples], dtype=np.float32)
    samples_per_chunk = WINDOW_USEFUL_FRAMES * fixture["sample_rate"] // fixture["fps"]
    if samples_per_chunk * fixture["fps"] != WINDOW_USEFUL_FRAMES * fixture["sample_rate"]:
        raise ValueError("The fixed chunk profile needs an integral audio sample count")
    padded = np.pad(effective_audio, (0, (-len(effective_audio)) % samples_per_chunk))
    return effective_audio, padded.reshape(-1, samples_per_chunk)


def _save_frames(output: Path, rgb: np.ndarray, fps: int) -> list[dict[str, float | int | str]]:
    saved = []
    for timestamp in (0.5, 1.5, 2.5, 3.5, 4.5, 6.5, 8.5, 9.5, 19.5, 29.5, 39.5, 49.5, 59.5):
        frame = round(timestamp * fps)
        if frame < len(rgb):
            name = f"frame-{timestamp:.1f}s.png"
            Image.fromarray(rgb[frame]).save(output / name)
            saved.append({"frame": frame, "time_s": timestamp, "path": name, "source": "raw_rgb"})
    return saved


def _install_decoder_policy(pipeline, policy: dict[str, Any], *, capture: bool = False) -> dict[str, Any]:
    decoder = policy["decoder"]
    if decoder["scheme"] == BF16_SCHEME:
        if decoder["backend"] == "torch_compile":
            if capture:
                return {
                    "mode": "none",
                    "precision_changed": False,
                    "compiled_modules": [],
                    "capture_override": "decoder compilation disabled to preserve module-level capture",
                }
            return optimize_wan_vae(pipeline.vae, "compiled")
        if decoder["backend"] == "pytorch":
            return {"mode": "none", "precision_changed": False, "compiled_modules": []}
        raise ValueError("A BF16 adapter requires an explicit decoder plan and is not a default decoder path")

    from benchmarks.pro_quantization_v2_20260918.decoder_install import install_quantized_decoder
    return install_quantized_decoder(pipeline.vae, decoder)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    if args.frames < 1 or args.repeats < 1:
        raise ValueError("frames and repeats must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for a PRO quantization experiment")
    policy_path = resolve_path(args.policy)
    policy = load_policy(policy_path)
    fixture = resolve_fixture(args.fixtures, args.fixture_id)
    requested_capture_split = getattr(args, "capture_split", None)
    if requested_capture_split is not None:
        declared = fixture["split"].replace("-and-", "-").split("-")
        if requested_capture_split not in declared:
            raise ValueError(
                f"Fixture {fixture['id']} is assigned to {fixture['split']!r}, not capture split {requested_capture_split!r}"
            )
    output = ensure_new_directory(args.output)
    capture_collector = None
    profile_controller = None
    if getattr(args, "capture_manifest", None) is not None:
        from benchmarks.pro_quantization_v2_20260918.capture import BoundedActivationCapture

        capture_collector = BoundedActivationCapture(
            Path(args.capture_manifest),
            getattr(args, "capture_max_raw_mib", 512),
            getattr(args, "capture_decoder_representative_paths", set()),
        )
    if getattr(args, "profile_component", None) is not None:
        if capture_collector is not None:
            raise ValueError("Capture and profiling are separate diagnostics and cannot share a run")
        from benchmarks.pro_quantization_v2_20260918.profile import ProfileController

        profile_controller = ProfileController(
            output, args.profile_component, getattr(args, "profile_mode", "timing")
        )
    requested_policy_path = output / "requested-policy.json"
    requested_policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "fixtures-resolved.json").write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profile = _fixed_profile(args.frames, args.seed, policy)
    result: dict[str, Any] = {
        "status": "starting",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "execution": (
            "fresh GPU activation capture diagnostic; timing is not a performance claim"
            if capture_collector is not None else (
                "fresh GPU profiling diagnostic; timing is not a performance claim"
                if profile_controller is not None else "fresh local GPU inference"
            )
        ),
        "environment": environment_manifest(),
        "profile": profile,
        "fixture": fixture,
        "policy": {"path": relative_path(policy_path), "sha256": sha256(policy_path), "name": policy["name"]},
        "generation_metric": "wall-clock audio preparation through RGB transfer; includes final padded generation; excludes encode/mux",
        "stages": {},
    }
    if requested_capture_split is not None:
        result["capture_split"] = requested_capture_split

    def save(status: str | None = None) -> None:
        if status is not None:
            result["status"] = status
        atomic_write_json(output / "results.json", result)

    source_manifest = snapshot_sources(output, _sources())
    result["source_sha256"] = source_manifest
    atomic_write_json(output / "source-manifest.json", source_manifest)
    save()
    lease = None
    pipeline = None
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (output / "stderr.log").open("w", encoding="utf-8") as stderr:
        sys.stdout, sys.stderr = Tee(old_stdout, stdout), Tee(old_stderr, stderr)
        stage = "acquire_gpu_lease"
        try:
            lease = acquire_gpu_lease(args.gpu_lock)
            torch.set_num_threads(4)
            torch.manual_seed(0)
            from flash_head.src.pipeline import flash_head_pipeline as implementation

            implementation.COMPILE_MODEL = False
            implementation.COMPILE_VAE = False
            implementation.USE_PARALLEL_VAE = False
            from flash_head.src.pipeline.flash_head_pipeline import FlashHeadPipeline

            from soulx_rtc.pro_attention_backends import pin_cross_attention_flash2

            stage = "load"
            save("loading")
            weights = ROOT / "models/SoulX-FlashHead-1_3B"
            result["checkpoint_sha256"] = {
                name: sha256(weights / name)
                for name in (
                    "Model_Pro/diffusion_pytorch_model.safetensors",
                    "Model_Pro/config.json",
                    "VAE_Wan/Wan2.1_VAE.pth",
                )
            }
            pipeline = FlashHeadPipeline(
                str(weights), "pro", str(ROOT / "models/wav2vec2-base-960h"),
                device="cuda", param_dtype=torch.bfloat16, use_usp=False,
            )
            pipeline.audio_encoder.eval().requires_grad_(False)

            stage = "validate_and_convert"
            plan = build_conversion_plan(pipeline.model, policy, strict_geometry=True)
            write_resolved_policy(plan, output / "resolved-policy.json")
            result["quantization"] = apply_conversion_plan(pipeline.model, plan)
            result["stages"][stage] = "complete"
            save("converted")

            stage = "install_attention_backend"
            attention = make_attention_backend(policy["self_attention_kernel"]["backend"])
            result["attention_backend"] = {
                "requested": policy["self_attention_kernel"]["backend"],
                "installed": install_self_attention_backend(pipeline.model, attention),
                "cross_attention": pin_cross_attention_flash2(pipeline.model),
                "fallback_allowed": False,
            }
            result["stages"][stage] = "complete"

            stage = "install_decoder_policy"
            decoder_trace = (
                profile_controller is not None
                and profile_controller.mode == "trace"
                and profile_controller.component == "decoder"
            )
            result["decoder"] = _install_decoder_policy(
                pipeline, policy, capture=capture_collector is not None or decoder_trace
            )
            result["stages"][stage] = "complete"

            stage = "prepare"
            pipeline.prepare_params(
                fixture["reference"]["path"], (576, 320), WINDOW_MODEL_FRAMES, WINDOW_HISTORY_FRAMES, 4,
                seed=args.seed, shift=5, color_correction_strength=1.0, use_face_crop=False,
            )
            if policy["prepared_conditioning"]:
                result["prepared_conditioning"] = _install_prepared_generation(
                    pipeline, compile_dit=policy["compile"]["dit"] and not policy["compile"]["ffn_only"] and capture_collector is None
                )
            elif policy["compile"]["dit"]:
                raise ValueError("Full DiT compilation requires prepared conditioning in the v2 runner")
            if policy["compile"]["ffn_only"] and capture_collector is None:
                for block in pipeline.model.blocks:
                    block.ffn = torch.compile(block.ffn, fullgraph=True, dynamic=False)
            if capture_collector is not None:
                capture_collector.attach(pipeline, plan)
            if profile_controller is not None:
                profile_controller.attach(pipeline)
            torch.cuda.empty_cache()
            result["stages"][stage] = "complete"
            save("warming")

            from benchmarks.pro_lite_teeth_20260917.run_variant import monitor_generation, summarize_samples
            from flash_head.inference import get_audio_embedding, run_pipeline

            effective_audio, slices = _audio_slices(fixture, args.frames)
            result["effective_audio"] = {
                "sample_rate": fixture["sample_rate"],
                "samples": int(len(effective_audio)),
                "sha256": hashlib.sha256(effective_audio.tobytes()).hexdigest(),
                "source_sha256": fixture["audio"]["sha256"],
                "repeated": len(effective_audio) > len(librosa.load(fixture["audio"]["path"], sr=fixture["sample_rate"], mono=True)[0]),
                "padding_samples_included_in_generation": int(slices.size - len(effective_audio)),
            }
            events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
            _instrument(pipeline.model, "forward", "dit", events)
            _instrument(pipeline.vae, "decode", "vae_decode", events)
            _instrument(pipeline.vae, "encode", "motion_encode", events)
            _instrument(pipeline, "preprocess_audio", "audio", events)

            def reset() -> deque[float]:
                pipeline.reset_person_name(pipeline.person_name)
                pipeline.generator.manual_seed(args.seed)
                events.clear()
                return deque([0.0] * 128000, maxlen=128000)

            stage = "warmup"
            warm_start = time.perf_counter()
            audio_history = reset()
            for index in range(2):
                if capture_collector is not None:
                    capture_collector.set_context(phase="warmup", window=index)
                audio_history.extend(slices[index % len(slices)].tolist())
                embedding = get_audio_embedding(pipeline, np.asarray(audio_history, dtype=np.float32), 167, 200)
                run_pipeline(pipeline, embedding)
            torch.cuda.synchronize()
            result["warmup_s"] = time.perf_counter() - warm_start
            result["prepare_warmup_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            result["prepare_warmup_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            events.clear()
            if profile_controller is not None:
                profile_controller.clear_events()
                profile_controller.start()
            save("running")

            result["runs"] = []
            saved_rgb = None
            for repeat in range(args.repeats):
                stage = f"generation_repeat_{repeat}"
                audio_history = reset()
                generated: list[np.ndarray] = []
                chunk_times: list[float] = []
                stage_totals: defaultdict[str, float] = defaultdict(float)
                torch.cuda.reset_peak_memory_stats()
                resources: list[dict[str, Any]] = []
                stop = threading.Event()
                start = time.perf_counter()
                monitor = threading.Thread(
                    target=monitor_generation, args=(stop, resources, start), daemon=True
                )
                monitor.start()
                try:
                    for window, audio_slice in enumerate(slices):
                        if capture_collector is not None:
                            capture_collector.set_context(phase="generation", window=window)
                        chunk_start = time.perf_counter()
                        audio_history.extend(audio_slice.tolist())
                        embedding = get_audio_embedding(pipeline, np.asarray(audio_history, dtype=np.float32), 167, 200)
                        window = run_pipeline(pipeline, embedding)[WINDOW_HISTORY_FRAMES:].cpu()
                        torch.cuda.synchronize()
                        chunk_times.append(time.perf_counter() - chunk_start)
                        if not bool(window.isfinite().all()):
                            raise RuntimeError(f"Generated window {window} contains nonfinite RGB values")
                        frames = window.numpy().astype(np.uint8)
                        if repeat == 0:
                            generated.append(frames)
                        for event_stage, begin, end in events:
                            stage_totals[event_stage] += begin.elapsed_time(end) / 1000
                        events.clear()
                        if profile_controller is not None:
                            profile_controller.step()
                finally:
                    stop.set()
                    monitor.join()
                generation_s = time.perf_counter() - start
                row = {
                    "repeat": repeat,
                    "generation_s": generation_s,
                    "useful_fps": args.frames / generation_s,
                    "first_window_latency_s": chunk_times[0],
                    "chunk_times_s": chunk_times,
                    "chunk_distribution_s": summarize(chunk_times),
                    "stage_seconds": dict(stage_totals),
                    "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                    "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                    "resource_summary": summarize_samples(resources),
                }
                result["runs"].append(row)
                atomic_write_json(output / f"resources-{repeat}.json", resources)
                if repeat == 0:
                    saved_rgb = np.concatenate(generated, axis=0)[:args.frames]
                    if len(saved_rgb) != args.frames:
                        raise RuntimeError(f"Expected {args.frames} useful frames, received {len(saved_rgb)}")
                    if not np.isfinite(saved_rgb).all():
                        raise RuntimeError("Generated RGB contains nonfinite values")
                    result["raw_rgb_sha256"] = hashlib.sha256(saved_rgb.tobytes()).hexdigest()
                save("running")

            if saved_rgb is None:
                raise RuntimeError("No RGB output was retained from the first repeat")
            stage = "encode_mux"
            from soulx_rtc.experiment import record

            encoding_start = time.perf_counter()
            record(output / "video.mp4", [saved_rgb], effective_audio, fixture["fps"])
            result["encode_mux_s"] = time.perf_counter() - encoding_start
            result["samples"] = _save_frames(output, saved_rgb, fixture["fps"])
            if args.save_raw:
                np.save(output / "raw.npy", saved_rgb)
            result["attention_backend"]["invocations"] = attention.invocation_manifest
            if capture_collector is not None:
                result["capture"] = capture_collector.finalize(status="complete")
            if profile_controller is not None:
                result["profile_artifacts"] = profile_controller.finalize(status="complete")
            result["environment_after"] = {
                "gpu": gpu_details(), "resident_gpu_processes": gpu_processes(),
            }
            result["stages"][stage] = "complete"
            save("complete")
            print(json.dumps({"output": str(output), "runs": result["runs"]}, indent=2), flush=True)
            return result
        except Exception as error:
            if capture_collector is not None:
                result["capture"] = capture_collector.finalize(
                    status="failed", extra={"error_type": type(error).__name__, "error": str(error)}
                )
            if profile_controller is not None:
                result["profile_artifacts"] = profile_controller.finalize(
                    status="failed", extra={"error_type": type(error).__name__, "error": str(error)}
                )
            result["failure"] = failure_record(stage, error)
            save("failed")
            raise
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr
            if pipeline is not None:
                cleanup_errors = []
                try:
                    remove_self_attention_backend(pipeline.model)
                except Exception as error:
                    cleanup_errors.append(f"attention:{type(error).__name__}:{error}")
                try:
                    remove_decoder_adapters(pipeline.vae)
                except Exception as error:
                    cleanup_errors.append(f"decoder:{type(error).__name__}:{error}")
                if cleanup_errors:
                    result["cleanup_errors"] = cleanup_errors
                    atomic_write_json(output / "results.json", result)
                del pipeline
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if lease is not None:
                lease.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, default=DEFAULT_FIXTURES)
    parser.add_argument("--fixture-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--frames", type=int, default=250)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--save-raw", action="store_true")
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--capture-max-raw-mib", type=int, default=512)
    parser.add_argument("--profile-component", choices=("dit", "decoder", "pipeline"))
    parser.add_argument("--profile-mode", choices=("trace", "timing"), default="timing")
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()