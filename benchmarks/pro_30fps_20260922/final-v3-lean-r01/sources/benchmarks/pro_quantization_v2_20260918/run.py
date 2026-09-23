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
from contextlib import nullcontext
from collections import defaultdict, deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Callable

import librosa
import numpy as np
from PIL import Image
import torch
from flash_head.utils.latency import LatencyRecorder, PipelineInstrumentation, latency_scope

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


def _set_history_frames(value: int) -> dict:
    """Override the motion-history window length for one run.

    Fewer history frames means fewer tokens reaching the DiT and fewer latent frames to
    decode, so it is the largest non-resolution lever left. It also CHANGES WINDOW
    GEOMETRY, which is the class of change that shipped a 4-frame lip-sync shift earlier
    in this work -- caught only because opening_correlation read 0.044 at lag 0 against
    0.958 at lag -4. Any arm using this must check per-window frame counts and run the
    articulation gate, not just FPS.

    The stock value is 5 pixel frames = 2 latent frames (1 + 4//4). Legal smaller values
    are those satisfying (value - 1) %% 4 == 0, i.e. 1, because the VAE expands one
    latent frame to 4 pixel frames after the first.
    """
    global WINDOW_HISTORY_FRAMES, WINDOW_MODEL_FRAMES
    if value != WINDOW_HISTORY_FRAMES and (value - 1) % 4 != 0:
        raise ValueError(
            f"history frames must satisfy (n-1) %% 4 == 0 to align with the VAE's 4x "
            f"temporal expansion; got {value}"
        )
    before = (WINDOW_HISTORY_FRAMES, WINDOW_MODEL_FRAMES)
    WINDOW_HISTORY_FRAMES = int(value)
    WINDOW_MODEL_FRAMES = WINDOW_HISTORY_FRAMES + WINDOW_USEFUL_FRAMES
    return {"history_frames": [before[0], WINDOW_HISTORY_FRAMES],
            "model_frames": [before[1], WINDOW_MODEL_FRAMES],
            "useful_frames": WINDOW_USEFUL_FRAMES}


# Output resolution (H, W). 576x320 is the trained/shipped size; every retained TensorRT
# stage engine bakes its spatial dims into its signature, so a different resolution needs
# a fresh capture + build. Overridden by --resolution.
RESOLUTION = (576, 320)


def _set_resolution(height: int, width: int) -> dict:
    global RESOLUTION
    if height % 16 or width % 16:
        # 8x VAE stride then a 2x DiT patch: latent dims must be even, so pixels must
        # be multiples of 16 or the patchify silently drops a row/column.
        raise ValueError(f"resolution must be multiples of 16, got {height}x{width}")
    before = RESOLUTION
    RESOLUTION = (int(height), int(width))
    return {"before": list(before), "after": list(RESOLUTION),
            "latent": [RESOLUTION[0] // 8, RESOLUTION[1] // 8],
            "area_vs_576x320": round(RESOLUTION[0] * RESOLUTION[1] / (576 * 320), 4)}


def _fixed_profile(frames: int, seed: int, policy: dict[str, Any], steps: int = 4,
                   skip_zero_weighted_noise: bool = False,
                   timestep_variant: str = "shipped") -> dict[str, Any]:
    return {
        "width": RESOLUTION[1],
        "height": RESOLUTION[0],
        "fps": 25,
        "frames": frames,
        "steps": steps,
        "skip_zero_weighted_noise": skip_zero_weighted_noise,
        "timestep_variant": timestep_variant,
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
        ROOT / "soulx_rtc/pro_quantization.py",
        ROOT / "benchmarks/pro_quantization_v2_20260918/decoder_install.py",
        ROOT / "benchmarks/pro_quantization_v2_20260918/build_decoder_engine.py",
        ROOT / "soulx_rtc/pro_quantization_v2.py",
        ROOT / "soulx_rtc/pro_attention_backends.py",
        ROOT / "soulx_rtc/pro_vae_quantization.py",
        ROOT / "soulx_rtc/pro_vae_stage_backend.py",
        ROOT / "soulx_rtc/pro_stage_onnx.py",
        # Runtime decoder op rewrites (sub-pixel Resample, channels-last head,
        # overlap-skip). These change generated pixels, so a run's source manifest is
        # incomplete without them.
        ROOT / "soulx_rtc/pro_decoder_ops.py",
        # Multi-session scheduler: changes how windows are issued (per-session CUDA
        # streams) and therefore what a throughput number means.
        ROOT / "benchmarks/pro_quantization_v2_20260918/sessions.py",
        ROOT / "benchmarks/pro_quantization_v2_20260918/profile.py",
        ROOT / "soulx_rtc/gpu_lease.py",
        ROOT / "flash_head/src/modules/flash_head_model.py",
        ROOT / "flash_head/src/pipeline/flash_head_pipeline.py",
        ROOT / "flash_head/src/pipeline/schedules.py",
        ROOT / "flash_head/inference.py",
        ROOT / "flash_head/utils/latency.py",
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


def _install_decoder_policy(pipeline, policy: dict[str, Any], *, capture: bool = False, eager_diagnostic: bool = False) -> dict[str, Any]:
    decoder = policy["decoder"]
    if eager_diagnostic:
        decoder = dict(decoder)
        if decoder["backend"] == "trt_stage_compile":
            decoder["backend"] = "trt_stage"
        elif decoder["backend"] == "torch_compile":
            decoder["backend"] = "pytorch"
        elif decoder["backend"] == "adapter_compile":
            decoder["backend"] = "adapter"
    if decoder["backend"] in ("trt_stage", "trt_stage_compile"):
        if capture:
            raise ValueError("Capture/inventory requires the unmodified decoder, not installed stages")
        from benchmarks.pro_quantization_v2_20260918.decoder_install import install_quantized_decoder
        return install_quantized_decoder(pipeline.vae, decoder)
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
    resolution_override = None
    if getattr(args, "resolution", None) is not None:
        resolution_override = _set_resolution(*args.resolution)
    history_override = None
    if getattr(args, "history_frames", None) is not None:
        # Must run before _fixed_profile and before prepare_params, both of which read
        # the module-level window constants.
        history_override = _set_history_frames(args.history_frames)
    policy_path = resolve_path(args.policy)
    policy = load_policy(policy_path)
    if policy['decoder']['backend'] == 'trt_stage_compile':
        torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 64)
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
    latency_mode = getattr(args, "latency_detail", "off")
    if latency_mode not in ("off", "stages", "modules"):
        raise ValueError("Unknown latency detail mode")
    if getattr(args, "latency_max_events", 50000) < 1:
        raise ValueError("latency-max-events must be positive")
    if latency_mode != "off" and getattr(args, "capture_manifest", None) is not None:
        raise ValueError("Latency logging and activation capture require separate runs")
    if latency_mode != "off" and getattr(args, "profile_mode", None) == "inventory":
        raise ValueError("Use latency modules mode instead of combining eager inventories")
    eager_diagnostic = latency_mode == "modules"
    lean_delivery = bool(getattr(args, "lean_delivery", False))
    sessions = int(getattr(args, "sessions", 1))
    if sessions < 1:
        raise ValueError("sessions must be positive")
    # --force-scheduler runs even a single session through the multi-session scheduler
    # (own CUDA stream, pinned non-blocking delivery) to bisect scheduler effects.
    multi_session = sessions > 1 or bool(getattr(args, "force_scheduler", False))
    if multi_session and (
        latency_mode != "off"
        or getattr(args, "capture_manifest", None) is not None
        or getattr(args, "profile_component", None) is not None
        or getattr(args, "calibrate_int8", False)
    ):
        raise ValueError("Multi-session mode is a throughput measurement; diagnostics are single-session")
    if lean_delivery and getattr(args, "capture_manifest", None) is not None:
        raise ValueError("Activation capture expects the historical float32 delivery")
    sampling_steps = int(getattr(args, "sampling_steps", 4))
    skip_zero_weighted_noise = bool(getattr(args, "skip_zero_weighted_noise", False))
    timestep_variant = str(getattr(args, "timestep_variant", "shipped"))
    # Fail here rather than after the GPU lease and model load: raw_timestep_schedule
    # refuses untabulated counts, and the prepared-conditioning path precomputes one
    # time embedding per step, so a bad count must not reach either.
    from flash_head.src.pipeline.schedules import raw_timestep_schedule

    raw_timestep_schedule(sampling_steps, variant=timestep_variant)
    if getattr(args, "capture_manifest", None) is not None:
        from benchmarks.pro_quantization_v2_20260918.capture import BoundedActivationCapture

        capture_collector = BoundedActivationCapture(
            Path(args.capture_manifest),
            getattr(args, "capture_max_raw_mib", 512),
            getattr(args, "capture_decoder_representative_paths", set()),
            capture_linears=getattr(args, "capture_linears", True),
            capture_attention=getattr(args, "capture_attention", False),
            capture_decoder_inputs=getattr(args, "capture_decoder_inputs", True),
        )
    if getattr(args, "profile_component", None) is not None:
        if capture_collector is not None:
            raise ValueError("Capture and profiling are separate diagnostics and cannot share a run")
        from benchmarks.pro_quantization_v2_20260918.profile import ProfileController

        profile_controller = ProfileController(
            output, args.profile_component, getattr(args, "profile_mode", "timing")
        )
        profile_controller.eager_diagnostic = eager_diagnostic
    requested_policy_path = output / "requested-policy.json"
    requested_policy_path.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "fixtures-resolved.json").write_text(json.dumps(fixture, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    profile = _fixed_profile(args.frames, args.seed, policy, steps=sampling_steps,
                             skip_zero_weighted_noise=skip_zero_weighted_noise,
                             timestep_variant=timestep_variant)
    result: dict[str, Any] = {
        "status": "starting",
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "execution": (
            "fresh GPU activation capture diagnostic; timing is not a performance claim"
            if capture_collector is not None else (
                "fresh GPU profiling diagnostic; timing is not a performance claim"
                if profile_controller is not None or latency_mode != "off" else "fresh local GPU inference"
            )
        ),
        "environment": environment_manifest(),
        "dynamo_recompile_limit": torch._dynamo.config.recompile_limit,
        "profile": profile,
        "window_geometry_override": history_override,
        "resolution_override": resolution_override,
        "fixture": fixture,
        "policy": {"path": relative_path(policy_path), "sha256": sha256(policy_path), "name": policy["name"]},
        "generation_metric": (
            "wall-clock audio preparation through RGB transfer; includes final padded generation; excludes encode/mux"
            + ("; AGGREGATE over concurrent sessions (useful_fps = sessions x frames / wall)" if multi_session else "")
        ),
        "sessions": {
            "count": sessions,
            "seed_stride": int(getattr(args, "session_seed_stride", 0)),
            "mode": "one CUDA stream per session, round-robin issue, one window in flight per session" if multi_session else "single stream",
        },
        "delivery": {
            "mode": "device_uint8_trimmed" if lean_delivery else "host_float32_then_astype",
            "lean_delivery": lean_delivery,
            "d2h_dtype": "uint8" if lean_delivery else "float32",
            "d2h_frames": "useful only" if lean_delivery else "useful plus history",
            "finite_check": "device, pre-cast" if lean_delivery else "host, pre-astype",
        },
        "stages": {},
    }
    latency = None
    instrumentation = PipelineInstrumentation()
    if latency_mode != "off":
        latency = LatencyRecorder(
            output, mode=latency_mode, max_events=getattr(args, "latency_max_events", 50000),
            provenance={"environment": result["environment"], "profile": profile,
                        "policy": result["policy"], "date_utc": result["date_utc"]},
        )
        result["latency_diagnostic"] = {
            "mode": latency_mode,
            "pytorch_execution": "eager override; not policy throughput" if eager_diagnostic else "policy-selected compilation preserved",
            "tensorrt_execution": "policy-selected engines preserved; internals remain fused",
            "performance_claim": False,
        }
    if requested_capture_split is not None:
        result["capture_split"] = requested_capture_split

    def save(status: str | None = None) -> None:
        if status is not None:
            result["status"] = status
        atomic_write_json(output / "results.json", result)

    source_manifest = snapshot_sources(output, _sources())
    result["source_sha256"] = source_manifest
    if latency is not None:
        latency.provenance["source_sha256"] = source_manifest
    atomic_write_json(output / "source-manifest.json", source_manifest)
    save()
    lease = None
    pipeline = None
    old_stdout, old_stderr = sys.stdout, sys.stderr
    with (output / "stdout.log").open("w", encoding="utf-8") as stdout, (output / "stderr.log").open("w", encoding="utf-8") as stderr:
        sys.stdout, sys.stderr = Tee(old_stdout, stdout), Tee(old_stderr, stderr)
        stage = "acquire_gpu_lease"
        latency_activation = latency.activate() if latency is not None else nullcontext()
        latency_activation.__enter__()
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
            if latency is not None:
                latency.set_context(phase="setup")
            with latency_scope("setup.load_pipeline"):
                pipeline = FlashHeadPipeline(
                    str(weights), "pro", str(ROOT / "models/wav2vec2-base-960h"),
                    device="cuda", param_dtype=torch.bfloat16, use_usp=False,
                )
            pipeline.audio_encoder.eval().requires_grad_(False)
            pipeline.lean_delivery = lean_delivery
            pipeline.skip_zero_weighted_noise = skip_zero_weighted_noise
            pipeline.timestep_variant = timestep_variant

            stage = "validate_and_convert"
            plan = build_conversion_plan(pipeline.model, policy, strict_geometry=True)
            write_resolved_policy(plan, output / "resolved-policy.json")
            with latency_scope("setup.quantize"):
                result["quantization"] = apply_conversion_plan(pipeline.model, plan)
            if getattr(args, "calibrate_int8", False):
                from soulx_rtc.pro_quantization import set_int8_calibration

                result["int8_calibration"] = {
                    "sites": set_int8_calibration(pipeline.model, True),
                    "note": "observed activation amax per INT8 site; not a throughput run",
                }
            if getattr(args, "static_int8_scales", None):
                from soulx_rtc.pro_quantization import apply_static_int8_scales

                amax = json.loads(Path(args.static_int8_scales).read_text())
                result["int8_static_scales"] = apply_static_int8_scales(
                    pipeline.model,
                    amax.get("amax", amax),
                    margin=args.static_int8_margin,
                    only=(args.static_int8_only or None),
                )
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
                and profile_controller.mode == "inventory"
                and profile_controller.component == "decoder"
            )
            with latency_scope("setup.install_decoder"):
                result["decoder"] = _install_decoder_policy(
                    pipeline, policy, capture=capture_collector is not None or decoder_trace,
                    eager_diagnostic=eager_diagnostic,
                )
            # Decoder op rewrites go in immediately after the decoder policy, before the
            # reference prepare warms anything: torch.compile traces lazily, so swapping
            # modules here is seen by the first trace instead of forcing a recompile.
            if getattr(args, "subpixel_resample", False):
                from soulx_rtc.pro_decoder_ops import install_subpixel_resample

                with latency_scope("setup.install_subpixel_resample"):
                    result["decoder_ops"] = install_subpixel_resample(pipeline.vae)
            if getattr(args, "channels_last_head", False):
                from soulx_rtc.pro_decoder_ops import install_channels_last_head

                with latency_scope("setup.install_channels_last_head"):
                    result.setdefault("decoder_ops", {}).update(
                        install_channels_last_head(pipeline.vae)
                    )
            result["stages"][stage] = "complete"

            stage = "prepare"
            with latency_scope("setup.prepare_reference"):
                pipeline.prepare_params(
                    fixture["reference"]["path"], RESOLUTION, WINDOW_MODEL_FRAMES, WINDOW_HISTORY_FRAMES,
                    sampling_steps,
                    seed=args.seed, shift=5, color_correction_strength=1.0, use_face_crop=False,
                )
            result["sampling_schedule"] = {
                "sampling_steps": sampling_steps,
                "timestep_variant": timestep_variant,
                "forward_passes_per_window": sampling_steps,
                "resolved_post_shift_timesteps": list(
                    getattr(pipeline, "resolved_timesteps", [])
                ),
                "shift": 5,
                "skip_zero_weighted_noise": skip_zero_weighted_noise,
                "seed_comparability": (
                    "seeds are NOT comparable across step counts: a fresh full randn is "
                    "drawn per step, so the generator stream differs. Compare distributions."
                ),
            }
            if policy["prepared_conditioning"]:
                result["prepared_conditioning"] = _install_prepared_generation(
                    pipeline, compile_dit=policy["compile"]["dit"] and not policy["compile"]["ffn_only"] and capture_collector is None and not eager_diagnostic
                )
            elif policy["compile"]["dit"]:
                raise ValueError("Full DiT compilation requires prepared conditioning in the v2 runner")
            if policy["compile"]["ffn_only"] and capture_collector is None and not eager_diagnostic:
                for block in pipeline.model.blocks:
                    block.ffn = torch.compile(block.ffn, fullgraph=True, dynamic=False)
            if capture_collector is not None:
                capture_collector.attach(pipeline, plan)
            if profile_controller is not None:
                profile_controller.attach(pipeline)
            if latency is not None:
                instrumentation.attach(pipeline, modules=eager_diagnostic)
                latency.flush()
            torch.cuda.empty_cache()
            result["stages"][stage] = "complete"
            save("warming")

            from benchmarks.pro_lite_teeth_20260917.run_variant import monitor_generation, summarize_samples
            from flash_head.inference import get_audio_embedding, run_pipeline

            with latency_scope("setup.read_audio", gpu=False):
                effective_audio, slices = _audio_slices(fixture, args.frames)
            result["effective_audio"] = {
                "sample_rate": fixture["sample_rate"],
                "samples": int(len(effective_audio)),
                "sha256": hashlib.sha256(effective_audio.tobytes()).hexdigest(),
                "source_sha256": fixture["audio"]["sha256"],
                "repeated": len(effective_audio) > len(librosa.load(fixture["audio"]["path"], sr=fixture["sample_rate"], mono=True)[0]),
                "padding_samples_included_in_generation": int(slices.size - len(effective_audio)),
            }
            # These two USED to be mutually exclusive. Overlap-skip must mutate _feat_map
            # between windows, and a torch.compile'd encode guards on that structure, so
            # every window missed its guard. Measured over 250 frames:
            #   wrap encode + compiled                 : motion_encode 12.454 s,  9.8361 FPS
            #   restore cache inside decode + compiled :                2.611 s, 16.9637 FPS
            #   restore cache inside decode, uncompiled:                1.415 s, 18.4298 FPS
            # install_overlap_skip now neutralises _feat_map to a stable list of Nones on
            # ENTRY to encode -- the exact structure the stock path always shows it -- so
            # the guard holds and both land together:
            #   neutralising shim + compiled           :                1.116 s, 19.2643 FPS
            # encode() rebuilds its own _enc_feat_map regardless, so this changes no
            # numerics: oral edge ratio 1.0259, opening_correlation 0.9633, colour drift
            # below control. --force-encode-compile is kept only to re-measure the old
            # collision; it is no longer needed for the combination to be safe.
            if getattr(args, "compile_vae_encode", False):
                # Must run BEFORE _instrument wraps vae.encode: instrumenting first
                # would hand torch.compile the CUDA-event wrapper instead of the
                # encoder, which both breaks the stage timing and graph-breaks.
                import torch as _torch
                pipeline.vae.encode = _torch.compile(pipeline.vae.encode, dynamic=False)
                result["vae_encode_compiled"] = True

            if getattr(args, "overlap_skip", False):
                # AFTER the encode compile on purpose: the shim must wrap the compiled
                # callable, not be traced by it -- its save/restore of _feat_map is
                # exactly the kind of global mutation that would graph-break.
                from soulx_rtc.pro_decoder_ops import install_overlap_skip

                result.setdefault("decoder_ops", {}).update(
                    install_overlap_skip(pipeline)
                )

            events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
            dump_dir = os.environ.get("SOULX_DUMP_DIR")
            if dump_dir:
                # Bisection aid: save every call's inputs/outputs of the four stage
                # boundaries so two harness modes can be diffed stage by stage.
                Path(dump_dir).mkdir(parents=True, exist_ok=True)
                counters: dict[str, int] = {}

                def _dump(owner, attribute, name):
                    original = getattr(owner, attribute)

                    def wrapped(*a, **k):
                        value = original(*a, **k)
                        index = counters.get(name, 0)
                        counters[name] = index + 1
                        torch.cuda.synchronize()
                        payload = {
                            "args": [x.detach().to("cpu") if isinstance(x, torch.Tensor) else x for x in a],
                            "kwargs": {kk: (v.detach().to("cpu") if isinstance(v, torch.Tensor) else v) for kk, v in k.items()},
                            "out": value.detach().to("cpu") if isinstance(value, torch.Tensor) else value,
                        }
                        torch.save(payload, Path(dump_dir) / f"{name}-{index:03d}.pt")
                        return value

                    setattr(owner, attribute, wrapped)

                _dump(pipeline.model, "forward", "forward")
                _dump(pipeline.vae, "decode", "decode")
                _dump(pipeline.vae, "encode", "encode")
                _dump(pipeline, "preprocess_audio", "audio")
            _instrument(pipeline.model, "forward", "dit", events)
            _instrument(pipeline.vae, "decode", "vae_decode", events)
            _instrument(pipeline.vae, "encode", "motion_encode", events)
            _instrument(pipeline, "preprocess_audio", "audio", events)

            def reset() -> deque[float]:
                pipeline.reset_person_name(pipeline.person_name)
                pipeline.generator.manual_seed(args.seed)
                if getattr(args, "overlap_skip", False):
                    # The decoder cache persists ACROSS windows by design, but must not
                    # persist across generations. reset() runs before the warmup pass and
                    # again before the measured pass; leaving the cache populated makes
                    # the measured window 0 decode against the warmup's temporal context.
                    from soulx_rtc.pro_decoder_ops import reset_overlap_skip

                    reset_overlap_skip(pipeline)
                events.clear()
                return deque([0.0] * 128000, maxlen=128000)

            stage = "warmup"
            warm_start = time.perf_counter()
            audio_history = reset()
            for index in range(0 if multi_session else 2):
                if latency is not None:
                    latency.set_context(phase="warmup", window=index)
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
            if latency is not None:
                latency.flush()
            if profile_controller is not None:
                profile_controller.clear_events()
                profile_controller.start()
            save("running")

            result["runs"] = []
            saved_rgb = None
            if multi_session:
                from benchmarks.pro_quantization_v2_20260918.sessions import SessionScheduler

                scheduler = SessionScheduler(
                    pipeline, events, get_audio_embedding, run_pipeline,
                    sessions=sessions, seed=args.seed,
                    seed_stride=int(getattr(args, "session_seed_stride", 0)),
                    lean_delivery=lean_delivery, history_frames=WINDOW_HISTORY_FRAMES,
                    overlap_skip=bool(getattr(args, "overlap_skip", False)),
                )
                # Warm every session's stream: the caching allocator keeps free blocks
                # per stream, so a default-stream warmup would leave the measured
                # streams paying cudaMalloc on their first windows.
                stage = "warmup"
                warm_start = time.perf_counter()
                scheduler.reset()
                scheduler.run(slices, windows=2)
                torch.cuda.synchronize()
                result["warmup_s"] = time.perf_counter() - warm_start
                result["prepare_warmup_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
                result["prepare_warmup_peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
                events.clear()
                save("running")
                for repeat in range(args.repeats):
                    stage = f"generation_repeat_{repeat}"
                    scheduler.reset()
                    torch.cuda.reset_peak_memory_stats()
                    resources = []
                    stop = threading.Event()
                    start = time.perf_counter()
                    monitor = threading.Thread(
                        target=monitor_generation, args=(stop, resources, start), daemon=True
                    )
                    monitor.start()
                    try:
                        generation_s = scheduler.run(slices)
                    finally:
                        stop.set()
                        monitor.join()
                    row = scheduler.row(repeat, generation_s, args.frames, summarize)
                    row.update({
                        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
                        "resource_summary": summarize_samples(resources),
                    })
                    result["runs"].append(row)
                    atomic_write_json(output / f"resources-{repeat}.json", resources)
                    if repeat == 0:
                        outputs = scheduler.outputs(args.frames)
                        saved_rgb = outputs[0]
                        result["session_outputs"] = scheduler.session_outputs(outputs)
                        result["raw_rgb_sha256"] = hashlib.sha256(saved_rgb.tobytes()).hexdigest()
                        if args.save_raw:
                            for index, rgb in enumerate(outputs[1:], start=1):
                                np.save(output / f"raw-s{index}.npy", rgb)
                        if getattr(args, "overlap_skip", False):
                            result.setdefault("decoder_ops", {})["overlap_skip_sessions"] = (
                                scheduler.overlap_skip_states()
                            )
                    save("running")
                scheduler.close()
            for repeat in range(0 if multi_session else args.repeats):
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
                        window_index = window
                        if latency is not None:
                            latency.set_context(phase="generation", repeat=repeat, window=window_index)
                        if capture_collector is not None:
                            capture_collector.set_context(phase="generation", window=window)
                        chunk_start = time.perf_counter()
                        with latency_scope("pipeline.window"):
                            with latency_scope("audio.history", gpu=False):
                                audio_history.extend(audio_slice.tolist())
                                audio_array = np.asarray(audio_history, dtype=np.float32)
                            embedding = get_audio_embedding(pipeline, audio_array, 167, 200)
                            rgb_device = run_pipeline(pipeline, embedding)
                            with latency_scope("delivery.trim_d2h"):
                                if lean_delivery:
                                    # Already trimmed, already uint8, already
                                    # finite-checked on device. One quarter of
                                    # the bytes cross the bus.
                                    window = rgb_device.cpu()
                                else:
                                    window = rgb_device[WINDOW_HISTORY_FRAMES:].cpu()
                            del rgb_device
                        torch.cuda.synchronize()
                        chunk_times.append(time.perf_counter() - chunk_start)
                        with latency_scope("delivery.validate_uint8", gpu=False):
                            if lean_delivery:
                                frames = window.numpy()
                            else:
                                if not bool(window.isfinite().all()):
                                    raise RuntimeError(f"Generated window {window_index} contains nonfinite RGB values")
                                frames = window.numpy().astype(np.uint8)
                        if repeat == 0:
                            generated.append(frames)
                        for event_stage, begin, end in events:
                            stage_totals[event_stage] += begin.elapsed_time(end) / 1000
                        events.clear()
                        if profile_controller is not None:
                            profile_controller.step()
                    generation_s = time.perf_counter() - start
                    if getattr(args, "overlap_skip", False):
                        from soulx_rtc.pro_decoder_ops import overlap_skip_state

                        result.setdefault("decoder_ops", {}).setdefault(
                            "overlap_skip", {}
                        ).update(overlap_skip_state(pipeline))
                    if getattr(args, "calibrate_int8", False):
                        from soulx_rtc.pro_quantization import (
                            collect_int8_amax,
                            set_int8_calibration,
                        )

                        result["int8_calibration"]["amax"] = collect_int8_amax(pipeline.model)
                        set_int8_calibration(pipeline.model, False)
                finally:
                    stop.set()
                    monitor.join()
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
                if latency is not None:
                    latency.flush()
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
            if latency is not None:
                latency.set_context(phase="delivery")
            with latency_scope("delivery.encode_mux", gpu=False):
                record(output / "video.mp4", [saved_rgb], effective_audio, fixture["fps"])
            result["encode_mux_s"] = time.perf_counter() - encoding_start
            result["samples"] = _save_frames(output, saved_rgb, fixture["fps"])
            if args.save_raw:
                np.save(output / "raw.npy", saved_rgb)
            result["attention_backend"]["implementation"] = attention.manifest()
            result["attention_backend"]["invocations"] = attention.invocation_manifest
            if capture_collector is not None:
                result["capture"] = capture_collector.finalize(status="complete")
            if profile_controller is not None:
                result["profile_artifacts"] = profile_controller.finalize(status="complete")
            if latency is not None:
                result["latency_artifacts"] = latency.flush(status="complete")
            result["environment_after"] = {
                "gpu": gpu_details(), "resident_gpu_processes": gpu_processes(),
            }
            result["stages"][stage] = "complete"
            save("complete")
            print(json.dumps({"output": str(output), "runs": result["runs"]}, indent=2), flush=True)
            return result
        except Exception as error:
            if latency is not None:
                try:
                    result["latency_artifacts"] = latency.flush(status="failed")
                except Exception as logging_error:
                    result["latency_logging_error"] = f"{type(logging_error).__name__}: {logging_error}"
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
            instrumentation.close()
            latency_activation.__exit__(None, None, None)
            sys.stdout, sys.stderr = old_stdout, old_stderr
            if pipeline is not None:
                cleanup_errors = []
                try:
                    remove_self_attention_backend(pipeline.model)
                except Exception as error:
                    cleanup_errors.append(f"attention:{type(error).__name__}:{error}")
                try:
                    from soulx_rtc.pro_vae_stage_backend import remove_stages
                    remove_stages(pipeline.vae)
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
    parser.add_argument("--capture-linears", dest="capture_linears", action="store_true", default=True)
    parser.add_argument("--no-capture-linears", dest="capture_linears", action="store_false",
                        help="Capture only decoder inputs (public VAE latents), not every DiT linear. "
                             "Linear capture alone is ~3.8 GB per representative block; decoder-only "
                             "capture is what a stage-engine calibration actually needs.")
    parser.add_argument("--profile-component", choices=("dit", "decoder", "pipeline"))
    parser.add_argument("--profile-mode", choices=("trace", "timing", "inventory"), default="timing")
    parser.add_argument("--latency-detail", choices=("off", "stages", "modules"), default="off",
                        help="Bounded inclusive latency logs; modules explicitly disables PyTorch compilation")
    parser.add_argument("--latency-max-events", type=int, default=50000)
    parser.add_argument("--lean-delivery", action="store_true",
                        help="Trim history frames and convert to uint8 on device before the D2H copy. "
                             "Quarters the per-window transfer; raw_rgb_sha256 stays comparable.")
    parser.add_argument("--resolution", type=int, nargs=2, metavar=("H", "W"), default=None,
                        help="Output resolution, default 576 320. Multiples of 16. Any other value "
                             "needs its own TensorRT stage plan (capture + build) -- the shipped "
                             "engines bake 576x320 into every signature.")
    parser.add_argument("--history-frames", type=int, default=None,
                        help="Override the motion-history window (default 5). Fewer history frames "
                             "means fewer DiT tokens and fewer latents to decode. CHANGES WINDOW "
                             "GEOMETRY -- run the articulation gate, not just FPS.")
    parser.add_argument("--force-encode-compile", action="store_true",
                        help="Allow --compile-vae-encode alongside --overlap-skip. Only safe once "
                             "the overlap-skip encode shim neutralises _feat_map on entry; without "
                             "it the compiled encoder's guard misses every window.")
    parser.add_argument("--overlap-skip", action="store_true",
                        help="Persist the decoder causal cache across windows and stop re-decoding "
                             "the motion overlap. Changes cross-window temporal context: gate on "
                             "per-window mean RGB drift, not a single window.")
    parser.add_argument("--calibrate-int8", action="store_true",
                        help="Record the observed activation amax at every INT8 site. Diagnostic "
                             "run: the extra reduction makes its timings unusable as throughput.")
    parser.add_argument("--static-int8-scales", default=None,
                        help="JSON of per-site activation amax (from --calibrate-int8). Installs a "
                             "static scale so the quantize is one DRAM pass instead of two.")
    parser.add_argument("--static-int8-margin", type=float, default=1.25,
                        help="Headroom over the calibrated amax. Activations above it SATURATE, so "
                             "this is the only guard against out-of-distribution clipping.")
    parser.add_argument("--static-int8-only", nargs="*", default=["ffn.2"],
                        help="Restrict static scales to sites containing these substrings. Defaults "
                             "to the 8960-wide ffn.2 input, the only site whose second read misses L2.")
    parser.add_argument("--subpixel-resample", action="store_true",
                        help="Rewrite the 2D Resample as an algebraically exact sub-pixel "
                             "convolution (conv at half resolution + PixelShuffle). Deletes the "
                             "upsampled intermediate. Gated by an fp64 identity assert at install.")
    parser.add_argument("--channels-last-head", action="store_true",
                        help="Hold the decoder head's CausalConv3d weight in channels_last_3d. "
                             "Bit-identical, but a win ONLY inside the compiled region -- eager it "
                             "is a ~17%% regression.")
    parser.add_argument("--compile-vae-encode", action="store_true",
                        help="torch.compile the motion-feedback VAE encoder. flash_head_pipeline sets "
                             "COMPILE_VAE=True as its module default but this harness disables it, and the "
                             "decoder install only ever compiles vae.decode. Costs extra warmup.")
    parser.add_argument("--skip-zero-weighted-noise", action="store_true",
                        help="Skip the terminal-step randn that is multiplied by exactly zero. "
                             "Changes the generator stream, so results are not bitwise comparable.")
    parser.add_argument("--timestep-variant", choices=("shipped", "distilled_aligned"), default="shipped",
                        help="shipped reproduces the original table exactly (2-step ends at 833.33). "
                             "distilled_aligned re-bases 2-step to end at 625.0, the level the 4-step "
                             "distilled schedule ends on. QUALITY-AFFECTING.")
    parser.add_argument("--sampling-steps", type=int, default=4, choices=(1, 2, 3, 4),
                        help="Denoising forward passes per window. DiT time is exactly linear in "
                             "this. 4 is the measured protocol; 2 halves DiT. QUALITY-AFFECTING: "
                             "seeds are not comparable across step counts.")
    parser.add_argument("--sessions", type=int, default=1,
                        help="Concurrent generation sessions sharing the weights/engines, one CUDA stream each; "
                             "useful_fps becomes the AGGREGATE across sessions")
    parser.add_argument("--force-scheduler", action="store_true",
                        help="Run through the multi-session scheduler even with --sessions 1 (bisection aid)")
    parser.add_argument("--session-seed-stride", type=int, default=0,
                        help="Seed offset per session (0 = identical seeds, which makes session outputs comparable "
                             "as an isolation check)")
    parser.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
