"""Capture calibration samples for Wan VAE **encoder** stages, then build FP16 engines.

Mirror of ``benchmarks/pro_30fps_20260919/decoder_stages.py`` for the per-window motion
re-encode (``vae.encode(cond_frame)``, ``cond_frame`` = ``(1, 3, 5, H, W)`` bf16). Groups
are comma-separated ``encoder.downsamples`` indices, contiguous, optionally led by
``conv1`` (``"conv1,0,1,2"`` ``"3,4,5"`` ``"6,7,8"``); each is exported as a
``soulx_rtc.pro_vae_encoder_stage.EncoderSpanStage`` through the decoder's ONNX pipeline
(``torch.onnx.export`` opset 18 -> fp16 conversion -> ``pro_stage_onnx.clean``) and built
with the strongly-typed TensorRT builder of ``build_decoder_engine._build_engine``.

Calibration inputs (``capture``):

* ``--inputs PATH ...`` -- ``.pt`` files, directories of them, or a JSON manifest listing
  them. A file holds either a pixel tensor ``(1, 3, T, H, W)`` / ``(3, T, H, W)`` in
  [-1, 1] (the trailing ``--frames`` frames are used, so a whole decoded clip works), or a
  harness ``SOULX_DUMP_DIR`` payload (``{"args", "kwargs", "out"}``): an ``encode-*.pt``
  dump's ``args[0]`` IS the production ``cond_frame``; a ``decode-*.pt`` dump's ``out`` is
  the teacher clip whose trailing frames are used.
* ``--synthetic --captures MANIFEST`` -- run the (stock or ``--synthetic-skip-blocks``
  pruned) decoder on the registered public VAE latents of a capture manifest
  (``benchmarks/pro_30fps_20260919/real-inputs-r01/manifest.json``: 16x9x72x40 -> 33
  frames at 576x320) and take ``--synthetic-windows`` trailing 5-frame windows of every
  clip. That is the production input distribution up to colour correction.
* ``--reference-frames N`` -- additionally encode ``frame0.repeat(N)`` of the first two
  inputs, the way ``prepare_params`` encodes the reference image (``frame_num`` = 33 for
  the PRO window), so the plan also covers the steady-state signature that only the
  one-time reference encode produces.

The eager stage path is self-checked against the stock encoder on every capture input
(``torch.equal`` on the latents), samples are saved once per signature and counted, and
every plan records ``target: "encoder"`` so ``install_stage_plan`` and
``install_encoder_plan`` cannot be handed each other's plans.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from benchmarks.pro_30fps_20260919.decoder_stages import export_stage
from benchmarks.pro_quantization_v2_20260918.build_decoder_engine import _build_engine
from benchmarks.pro_quantization_v2_20260918.common import (
    DEFAULT_GPU_LOCK,
    ROOT,
    atomic_write_json,
    ensure_new_directory,
    environment_manifest,
    failure_record,
    relative_path,
    sha256,
    snapshot_sources,
    utc_now,
)
from benchmarks.pro_quantization_v2_20260918.decoder_trial import (
    _captured_latents,
    _load_vae,
    _metrics,
)
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.pro_vae_encoder_stage import (
    PLAN_TARGET,
    STOCK_VAE_WEIGHTS,
    describe_encoder_span,
    encoder_group_key,
    encoder_span_cache_count,
    install_encoder_stages,
    make_encoder_stage,
    parse_encoder_group,
    remove_encoder_stages,
    tensor_signature,
)

DEFAULT_GROUPS = ["conv1,0,1,2", "3,4,5", "6,7,8"]
MOTION_FRAMES = 5  # pipeline.motion_frames_num for the PRO window (run.py WINDOW_HISTORY_FRAMES)
REFERENCE_FRAMES = 33  # pipeline.frame_num (run.py WINDOW_MODEL_FRAMES): prepare_params' reference encode


# --------------------------------------------------------------------------------------
# weights
# --------------------------------------------------------------------------------------


def vae_weights_path(args) -> Path:
    value = getattr(args, "vae_weights", None)
    return Path(value) if value else ROOT / STOCK_VAE_WEIGHTS


def load_vae_weights(vae, path, result=None) -> dict:
    """Load a (fine-tuned) state_dict into ``vae.model`` (strict=False; unexpected keys and
    shape mismatches refused). The encoder weights of the decoder fine-tunes are the stock
    ones, but the plan's sha gate compares the whole file, so the build must load exactly
    the file the runtime will present."""
    path = Path(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    for wrapper in ("state_dict", "model", "module"):
        if isinstance(state, dict) and wrapper in state and isinstance(state[wrapper], dict) and not any(
            isinstance(v, torch.Tensor) for v in state.values()
        ):
            state = state[wrapper]
            break
    if not isinstance(state, dict) or not state or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError(f"{path} is not a tensor state_dict")
    reference = vae.model.state_dict()
    unexpected = sorted(set(state) - set(reference))
    if unexpected:
        raise ValueError(f"--vae-weights {path} has keys the Wan VAE does not have (first: {unexpected[:5]})")
    mismatch = [k for k, v in state.items() if tuple(v.shape) != tuple(reference[k].shape)]
    if mismatch:
        raise ValueError(f"--vae-weights {path}: shape mismatch for {mismatch[:5]}")
    with torch.no_grad():
        missing, unexpected = vae.model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"--vae-weights {path}: unexpected keys {list(unexpected)[:5]}")
    record = {
        "path": relative_path(path),
        "sha256": sha256(path),
        "loaded_keys": len(state),
        "missing_keys": len(missing),
        "encoder_keys_loaded": sum(k.startswith("encoder.") for k in state),
    }
    if result is not None:
        result["vae_weights"] = record
    return record


def _apply_vae_weights(vae, args, result) -> None:
    value = getattr(args, "vae_weights", None)
    if value:
        load_vae_weights(vae, value, result)


# --------------------------------------------------------------------------------------
# calibration inputs
# --------------------------------------------------------------------------------------


def _as_pixels(value, frames: int, origin: str) -> torch.Tensor:
    """``(1, 3, T>=frames, H, W)`` bf16 from a saved tensor or a harness dump payload."""
    if isinstance(value, dict):
        args = value.get("args")
        candidate = None
        if isinstance(args, (list, tuple)) and args and isinstance(args[0], torch.Tensor) and args[0].ndim == 5 and args[0].shape[1] == 3:
            candidate = args[0]  # encode dump: the production cond_frame itself
        elif isinstance(value.get("out"), torch.Tensor):
            candidate = value["out"]  # decode dump: the teacher clip
        if candidate is None:
            raise ValueError(f"{origin}: dump payload has neither a pixel args[0] nor an 'out' tensor")
        value = candidate
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{origin}: expected a tensor or a dump payload, got {type(value).__name__}")
    if value.ndim == 4:
        value = value.unsqueeze(0)
    if value.ndim != 5 or value.shape[0] != 1 or value.shape[1] != 3:
        raise ValueError(f"{origin}: expected (1, 3, T, H, W) pixels, got {tuple(value.shape)}")
    if value.shape[2] < frames:
        raise ValueError(f"{origin}: {value.shape[2]} frames < {frames}")
    value = value[:, :, -frames:].to(torch.bfloat16).contiguous()
    if not torch.isfinite(value).all():
        raise ValueError(f"{origin}: nonfinite pixels")
    low, high = float(value.min()), float(value.max())
    if low < -1.01 or high > 1.01:
        raise ValueError(f"{origin}: pixels outside [-1, 1] ({low:.3f}..{high:.3f}); cond_frame is the clamped decoder output")
    return value


def _expand_inputs(entries) -> list[Path]:
    paths: list[Path] = []
    for entry in entries:
        entry = Path(entry)
        if entry.is_dir():
            found = sorted(p for p in entry.rglob("*.pt") if p.is_file())
            if not found:
                raise ValueError(f"--inputs {entry}: no .pt files")
            paths.extend(found)
        elif entry.suffix == ".json":
            data = json.loads(entry.read_text())
            if isinstance(data, dict):
                data = data.get("cond_frames") or data.get("inputs") or data.get("paths")
            if not isinstance(data, list):
                raise ValueError(f"--inputs {entry}: manifest must be a list or hold 'cond_frames'/'inputs'/'paths'")
            for item in data:
                item = item["path"] if isinstance(item, dict) else item
                path = Path(item)
                paths.append(path if path.is_absolute() else entry.parent / path)
        elif entry.is_file():
            paths.append(entry)
        else:
            raise FileNotFoundError(entry)
    return paths


def load_calibration_inputs(entries, frames: int, limit: int | None = None) -> list[tuple[str, torch.Tensor, dict]]:
    """``[(name, cond_frame bf16 on cuda, provenance record)]`` from ``--inputs``."""
    loaded = []
    for path in _expand_inputs(entries)[: limit or None]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        pixels = _as_pixels(payload, frames, str(path))
        loaded.append((path.stem, pixels.cuda(), {"path": relative_path(path), "sha256": sha256(path), "shape": list(pixels.shape)}))
    if not loaded:
        raise ValueError("No calibration inputs")
    return loaded


def synthetic_inputs(vae, manifest: Path, frames: int, windows: int, skip_blocks, limit: int | None, result) -> list[tuple[str, torch.Tensor, dict]]:
    """Decode the registered latents (stock eager decoder, optionally with the production
    block skips) and take ``windows`` trailing ``frames``-frame windows of every clip."""
    paths = _captured_latents(manifest)[: limit or None]
    skip_blocks = sorted({int(v) for v in (skip_blocks or [])})
    if skip_blocks:
        from soulx_rtc.pro_decoder_ops import install_decoder_block_skip

        result["synthetic_decoder_ops"] = install_decoder_block_skip(vae, skip_blocks)
    loaded = []
    try:
        with torch.inference_mode():
            for path in paths:
                latent = torch.load(path, map_location="cuda", weights_only=True).to(torch.bfloat16)
                clip = vae.decode(latent)  # (1, 3, 1 + 4 (T - 1), H, W) in [-1, 1]
                total = int(clip.shape[2])
                if total < frames * windows:
                    raise ValueError(f"{path}: {total} decoded frames < {frames} x {windows}")
                for window in range(windows):
                    end = total - window * frames
                    pixels = clip[:, :, end - frames : end].to(torch.bfloat16).contiguous()
                    loaded.append((
                        f"{path.stem}-w{window}",
                        pixels,
                        {"latent": relative_path(path), "sha256": sha256(path), "frames": [end - frames, end],
                         "shape": list(pixels.shape), "decoder_skip_blocks": skip_blocks},
                    ))
                del clip, latent
                torch.cuda.empty_cache()
    finally:
        if skip_blocks:
            from soulx_rtc.pro_decoder_ops import remove_decoder_block_skip

            remove_decoder_block_skip(vae)
    return loaded


# --------------------------------------------------------------------------------------
# capture
# --------------------------------------------------------------------------------------


def capture_samples(vae, inputs, groups, output: Path, result: dict, *, reference_frames: int = 0, max_bytes: int) -> None:
    """Install eager observing stages, run ``vae.encode`` on every input (and the optional
    reference-style encodes), record one sample tuple per signature plus per-conv maxima."""
    groups = [parse_encoder_group(value) for value in groups]
    result.update(
        target=PLAN_TARGET,
        groups=groups,
        spans={encoder_group_key(group): describe_encoder_span(vae, group) for group in groups},
        cache_slots={encoder_group_key(group): encoder_span_cache_count(vae, group) for group in groups},
        samples={},
        calibration={},
        source_inputs=[],
        reference_frames=reference_frames,
        retained_bytes=0,
        max_bytes=max_bytes,
        eager_self_check=[],
    )
    with torch.inference_mode():
        references = [vae.encode(pixels) for _, pixels, _ in inputs]

    def factory(group, stage):
        key = encoder_group_key(group)
        result["samples"][key] = {}
        result["calibration"][key] = {}

        def execute(*values):
            signature = tensor_signature(values)
            records = result["samples"][key]
            if signature not in records:
                count = sum(value.numel() * value.element_size() for value in values)
                if count + result["retained_bytes"] > result["max_bytes"]:
                    raise RuntimeError("Encoder stage capture byte budget exceeded; signature coverage incomplete")
                path = output / f"stage-{key}-sample-{len(records)}.pt"
                torch.save(tuple(value.detach().cpu() for value in values), path)
                records[signature] = {"path": path.name, "sha256": sha256(path), "count": 0,
                                      "shapes": [list(value.shape) for value in values]}
                result["retained_bytes"] += count
            records[signature]["count"] += 1
            scales = result["calibration"][key].setdefault(signature, {})

            def observe(name, before, previous, after):
                maximum = float(before.abs().max())
                if previous is not None:
                    maximum = max(maximum, float(previous.abs().max()))
                out_max = float(after.abs().max())
                if not all(torch.isfinite(torch.tensor(v)) for v in (maximum, out_max)):
                    raise ValueError("Nonfinite encoder stage calibration")
                item = scales.setdefault(name, {"input_max": 0.0, "output_max": 0.0, "count": 0})
                item["input_max"] = max(item["input_max"], maximum)
                item["output_max"] = max(item["output_max"], out_max)
                item["count"] += 1

            stage.observer = observe
            try:
                return stage(*values)
            finally:
                stage.observer = None

        return execute

    install_encoder_stages(vae, groups, factory)
    try:
        with torch.inference_mode():
            for (name, pixels, record), reference in zip(inputs, references):
                out = vae.encode(pixels)
                identical = bool(torch.equal(out, reference))
                result["eager_self_check"].append({"input": name, "identical": identical})
                if not identical:
                    raise RuntimeError(f"Eager encoder stage path differs from the stock encoder on {name}")
                result["source_inputs"].append({"name": name, **record})
                atomic_write_json(output / "results.json", result)
            if reference_frames:
                for name, pixels, _ in inputs[:2]:
                    clip = pixels[:, :, :1].repeat(1, 1, reference_frames, 1, 1)
                    vae.encode(clip)
                    result["source_inputs"].append({"name": f"{name}-reference{reference_frames}", "shape": list(clip.shape),
                                                    "synthetic_reference_encode": True})
                    del clip
        for samples in result["samples"].values():
            if any(item["count"] < 2 for item in samples.values()):
                raise RuntimeError("Fewer than two calibration samples for an encoder stage signature")
    finally:
        remove_encoder_stages(vae)


def capture(args, output: Path, result: dict) -> None:
    vae = _load_vae()
    _apply_vae_weights(vae, args, result)
    frames = int(args.frames)
    if args.synthetic:
        if args.captures is None:
            raise ValueError("--synthetic needs --captures MANIFEST (registered public VAE latents)")
        inputs = synthetic_inputs(vae, args.captures, frames, int(args.synthetic_windows), args.synthetic_skip_blocks,
                                  args.max_inputs, result)
        result["source_capture"] = {"path": relative_path(args.captures), "sha256": sha256(args.captures)}
    else:
        if not args.inputs:
            raise ValueError("capture needs --inputs PATH ... or --synthetic --captures MANIFEST")
        inputs = load_calibration_inputs(args.inputs, frames, args.max_inputs)
    if len(inputs) < 2:
        raise ValueError("Encoder stage calibration needs at least two inputs")
    result["frames_per_input"] = frames
    result["input_count"] = len(inputs)
    capture_samples(vae, inputs, args.groups, output, result,
                    reference_frames=int(args.reference_frames or 0), max_bytes=args.max_raw_mib * 2**20)


# --------------------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------------------


def build(args, output: Path, result: dict) -> None:
    import tensorrt as trt

    source = json.loads(args.calibration.read_text())
    if source.get("status") != "complete" or source.get("target") != PLAN_TARGET:
        raise ValueError("Incomplete calibration or not an encoder stage calibration")
    if source["weights_sha256"] != result["weights_sha256"]:
        raise ValueError("Calibration was captured from different VAE weights than this build loads")
    if args.precision not in ("fp16", "bf16"):
        raise ValueError("Encoder stages are floating-point only (fp16 / bf16)")
    vae = _load_vae()
    _apply_vae_weights(vae, args, result)
    only = {encoder_group_key(group) for group in (args.only or [])}
    floating = "fp16" if args.precision == "fp16" else args.floating_precision
    result.update(
        schema_version=1,
        target=PLAN_TARGET,
        groups=[],
        spans={},
        calibration_groups=[parse_encoder_group(group) for group in source["groups"]],
        precision=args.precision,
        floating_precision=floating,
        stages={},
        calibration={"path": relative_path(args.calibration), "sha256": sha256(args.calibration)},
        graph_rewrite=args.graph_rewrite,
        reference_frames=source.get("reference_frames", 0),
    )
    for source_group in source["groups"]:
        group = parse_encoder_group(source_group)
        key = encoder_group_key(group)
        if only and key not in only:
            continue
        live_span = describe_encoder_span(vae, group)
        if source["spans"].get(key) != live_span:
            raise ValueError(f"Calibration group {key} was captured with modules {source['spans'].get(key)} but this build's encoder has {live_span}")
        stage = make_encoder_stage(vae, group).eval()
        if stage.cache_count != encoder_span_cache_count(vae, group):
            raise ValueError("Stage cache count does not match the group's cache slots")
        result["groups"].append(group)
        result["spans"][key] = live_span
        result["stages"][key] = {}
        for i, (signature, sample) in enumerate(source["samples"][key].items()):
            path = args.calibration.parent / sample["path"]
            if sha256(path) != sample["sha256"]:
                raise ValueError("Encoder stage sample hash mismatch")
            values = tuple(v.cuda() for v in torch.load(path, map_location="cpu", weights_only=True))
            if tensor_signature(values) != signature or [list(v.shape) for v in values] != sample["shapes"]:
                raise ValueError("Encoder stage sample signature/shapes disagree with the calibration record")
            if len(values) not in (1, stage.cache_count + 1):
                raise ValueError("Sample does not match the stage's cache count")
            scales = source["calibration"][key][signature]
            onnx_path = output / f"encoder-{key}-{i}.onnx"
            metadata = export_stage(stage, values, onnx_path, args.precision, scales, floating, graph_rewrite=args.graph_rewrite)
            engine_path = onnx_path.with_suffix(".engine")
            begin = time.perf_counter()
            engine, layers = _build_engine(onnx_path, engine_path, args.workspace_mib)
            layers_path = engine_path.with_suffix(".layers.json")
            layers_path.write_text(layers + "\n")
            record = metadata | {
                "path": relative_path(engine_path),
                "sha256": sha256(engine_path),
                "precision": args.precision,
                "onnx_sha256": sha256(onnx_path),
                "layers_path": relative_path(layers_path),
                "layers_sha256": sha256(layers_path),
                "int8_convolutions_verified": False,
                "tensorrt": trt.__version__,
                "gpu": torch.cuda.get_device_name(),
                "build_s": time.perf_counter() - begin,
                "workspace_bytes": engine.device_memory_size,
                "calibration_input_max": max([v["input_max"] for v in scales.values()] or [0.0]),
                "calibration_count": sample["count"],
            }
            from soulx_rtc.pro_vae_stage_backend import TensorRTStage

            runtime = TensorRTStage(engine_path, record)
            with torch.inference_mode():
                expected = stage(*values)
                actual = runtime(*values)
                torch.cuda.synchronize()
                record["build_sample_comparison"] = [_metrics(a, b) for a, b in zip(actual, expected)]
            result["stages"][key][signature] = record
            atomic_write_json(output / "results.json", result)
            if not all(row["finite"] for row in record["build_sample_comparison"]):
                raise RuntimeError("Built encoder stage produced nonfinite output/cache on its calibration input")
            del runtime, expected, actual, engine, values
            torch.cuda.empty_cache()
    if not result["stages"]:
        raise ValueError("--only selected none of the calibration groups")
    if not args.keep_onnx:
        for onnx_path in output.glob("*.onnx"):
            onnx_path.unlink()


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("--inputs", type=Path, nargs="*", default=None, metavar="PATH",
                     help=".pt tensors (1,3,T>=5,H,W) in [-1,1], directories of them, JSON manifests, or harness "
                          "SOULX_DUMP_DIR encode-*/decode-*.pt payloads; the trailing --frames frames are encoded.")
    cap.add_argument("--synthetic", action="store_true",
                     help="Decode the public VAE latents of --captures with the eager decoder and use trailing "
                          "5-frame windows of the clips as cond_frames (production distribution up to colour correction).")
    cap.add_argument("--captures", type=Path, default=None, metavar="MANIFEST",
                     help="Capture manifest with registered public_vae_latent tensors (for --synthetic), e.g. "
                          "benchmarks/pro_30fps_20260919/real-inputs-r01/manifest.json (576x320).")
    cap.add_argument("--synthetic-windows", type=int, default=2, help="Trailing 5-frame windows per decoded clip (default 2).")
    cap.add_argument("--synthetic-skip-blocks", type=int, nargs="*", default=None, metavar="INDEX",
                     help="Decoder upsamples[] blocks bypassed while producing synthetic frames (the production "
                          "pruned decoder: 9 10 13 14). Encoder weights are unaffected; this only shapes the inputs.")
    cap.add_argument("--groups", nargs="+", default=DEFAULT_GROUPS,
                     help='Contiguous encoder.downsamples index groups, optionally led by conv1, e.g. "conv1,0,1,2" "3,4,5" "6,7,8".')
    cap.add_argument("--frames", type=int, default=MOTION_FRAMES, help="Pixel frames per cond_frame (pipeline.motion_frames_num, 5).")
    cap.add_argument("--reference-frames", type=int, default=0, metavar="N",
                     help=f"Also encode frame0.repeat(N) of the first two inputs like prepare_params' reference encode "
                          f"(PRO frame_num = {REFERENCE_FRAMES}) so the plan covers its steady-state signature too. "
                          "0 (default) = motion-encode signatures only; then install_encoder_plan must run after "
                          "prepare_params or with unknown_signature='eager'.")
    cap.add_argument("--max-inputs", type=int, default=None)
    cap.add_argument("--max-raw-mib", type=int, default=1024)
    bld = commands.add_parser("build")
    bld.add_argument("--calibration", type=Path, required=True, help="capture output results.json")
    bld.add_argument("--precision", choices=["bf16", "fp16"], required=True)
    bld.add_argument("--floating-precision", choices=["bf16", "fp16"], default="bf16")
    bld.add_argument("--workspace-mib", type=int, default=2048)
    bld.add_argument("--graph-rewrite", choices=["none", "clean", "norm-conv"], default="none",
                     help="ONNX rewrites before the TensorRT build (soulx_rtc/pro_stage_onnx.py; norm-conv recommended)")
    bld.add_argument("--only", nargs="*", default=None, metavar="GROUP", help="Build only these calibration groups (probing).")
    bld.add_argument("--keep-onnx", action="store_true", help="Keep the exported .onnx files next to the engines (disk!).")
    for sub in (cap, bld):
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--gpu-lock", type=Path, default=DEFAULT_GPU_LOCK)
        sub.add_argument("--vae-weights", type=Path, default=None, metavar="PATH",
                         help="State_dict loaded into vae.model after the stock VAE is built. Its sha256 becomes "
                              "weights_sha256 (whole file, even though the encoder weights are stock), so calibration, "
                              "build and install_encoder_plan(vae_weights=PATH) must all present the same file.")
    args = parser.parse_args()
    weights = vae_weights_path(args)
    if not weights.is_file():
        parser.error(f"VAE weights file not found: {weights}")
    if args.command == "build" and args.precision == "bf16" and args.floating_precision != "bf16":
        parser.error("Use --precision fp16 for an explicitly labeled FP16 build")
    output = ensure_new_directory(args.output)
    result = {
        "status": "starting",
        "date_utc": utc_now(),
        "execution": "fresh GPU encoder stage diagnostic; not end-to-end video throughput",
        "environment": environment_manifest(),
        "arguments": {k: str(v) for k, v in vars(args).items()},
        "weights_sha256": sha256(weights),
        "weights_path": relative_path(weights),
        "target": PLAN_TARGET,
    }
    result["source_sha256"] = snapshot_sources(
        output,
        [
            Path(__file__),
            ROOT / "soulx_rtc/pro_vae_encoder_stage.py",
            ROOT / "soulx_rtc/pro_vae_stage_backend.py",
            ROOT / "soulx_rtc/pro_stage_onnx.py",
            ROOT / "flash_head/wan/modules/vae.py",
            ROOT / "benchmarks/pro_30fps_20260919/decoder_stages.py",
            ROOT / "benchmarks/pro_quantization_v2_20260918/build_decoder_engine.py",
        ],
    )
    atomic_write_json(output / "results.json", result)
    lease = None
    try:
        lease = acquire_gpu_lease(args.gpu_lock)
        (capture if args.command == "capture" else build)(args, output, result)
        result["status"] = "complete"
    except Exception as error:
        result["status"] = "failed"
        result["failure"] = failure_record(args.command, error)
        raise
    finally:
        atomic_write_json(output / "results.json", result)
        if lease is not None:
            lease.close()


if __name__ == "__main__":
    main()
