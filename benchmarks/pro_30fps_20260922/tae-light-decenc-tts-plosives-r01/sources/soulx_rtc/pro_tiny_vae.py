"""Tiny / light autoencoders in place of the Wan 2.1 VAE for SoulX-FlashHead PRO (576x320).

WHY THIS EXISTS. At 576x320 the shipping build spends ~434 ms/window in the Wan decoder and
~117 ms/window in the per-window motion re-encode (2026-09-22, RTX 4070 SUPER), i.e. 60% of
the window. SD's TAESD cannot decode these latents (4 channels, per-image 2D). SoulX latents are
Wan 2.1: 16 channels, 8x spatial, 4x causal temporal (latent 0 -> 1 frame, every later latent ->
4 frames, 9 latents -> 33 frames). The same author's video counterpart, TAEHV ``taew2_1``, is
trained for exactly this latent space; LightX2V retrained it (``lighttaew2_1``) and also ships
a 4x-narrower Wan VAE (``lightvaew2_1``). The CPU/GPU bake-offs in
benchmarks/tiny_vae_20260926/{decoder,encoder}_bakeoff resolved each one's conventions
EMPIRICALLY (the wrong convention costs ~20 dB):

  name          arch       decoder input                encoder output              weights
  taew2_1       TAEHV      DiT-normalised latent as is  DiT-normalised as is        fp16 .pth (MIT)
  lighttaew2_1  TAEHV      z * std + mean (raw Wan)     raw Wan -> (e-mean)*inv_std fp32 safetensors (Apache-2.0)
  lightvaew2_1  WanVAE_(dim=24)  Wan protocol: decode(z, scale) / encode(x, scale)      fp32 safetensors (Apache-2.0)

TAEHV layout is NTCHW, pixels in [0, 1]. Decoding T latents gives 4T raw frames, of which the
first 3 are dropped (upstream frames_to_trim) -> 4T-3 = 33 for T=9, the Wan count. Its encoder
pads the END with copies of the last frame to a multiple of 4 (upstream ``encode_video``); the
GPU bake-off showed this is the correctly aligned grouping for SoulX's 5-frame motion clip
(Wan-style front padding delays the lip state: slot-1 nRMSE 0.29 vs 0.089).

CONTRACTS REPRODUCED (flash_head_pipeline.py):
  decode(zs):  zs (16, T, 72, 40) bf16, DiT-normalised, NOT modified in place
               -> (1, 3, F, 576, 320) bf16, clamped to [-1, 1], NCTHW, contiguous; F = 33 always.
  encode(v):   v (1, 3, t, 576, 320) bf16 in [-1, 1] (the trailing 5 colour-corrected frames)
               -> (16, 1 + (t-1)//4, 72, 40) bf16, DiT-normalised (batch dim squeezed).
The reference-image encode (prepare_params, 33 copies of the reference frame -> the DiT's y) is
never routed here: the harness installs the tiny encoder only after prepare, so y stays on the
Wan encoder.

PER-WINDOW PROTOCOL (decode_mode):
  "stream" (default): the tiny model's NATIVE streaming state, the analogue of the shipping
     overlap-skip. TAEHV's decoder has no temporal pooling; its only state is each MemBlock's
     previous-timestep input (9 tensors, ~31 MB fp16 at 72x40). Window 0 decodes all 9 latents
     cold (zero memory, == upstream decode_video(parallel=True)); windows 1+ decode only the
     trailing ``keep`` = (33-5)//4 = 7 fresh latents with the carried state (7 -> 28 frames) and
     prepend the previous window's 5-frame tail, keeping the 33-frame contract. The CPU probe
     showed window-by-window decoding with carried state equals one full-sequence decode to
     7.5e-7 (load_smoke.py), so this is exact, and it skips re-decoding the 2 overlap latents.
     Same semantics as overlap-skip: the carried state is the tiny decoder's own pre-correction
     state, not a re-decode of the colour-corrected motion latents -- gate colour drift.
  "window": every window decodes all 9 latents from zero state (the non-overlap-skip Wan path),
     so the colour-corrected motion latents also seed the decoder. 9/7 of the decode work.

BACKENDS: "eager" (fp16 modules), "compile" (torch.compile of the static time-parallel core,
two shapes for the decoder: T=9 cold and T=7 warm), "tensorrt" (FP16 engines of the same cores,
built once from ONNX at install and cached under benchmarks/tiny_vae_20260926/engines/; needs a
TensorRT with the builder, e.g. benchmarks/tiny_vae_20260926/decoder_bakeoff/_trt10_path).

ROLLBACK: nothing here runs unless run.py gets --tiny-vae-decoder / --tiny-vae-encoder. The Wan
VAE stays loaded (its encoder is needed for the reference latent, and WanVAE_.clear_cache counts
the decoder's convolutions on every encode).
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
TINY_DIR = ROOT / "models" / "tiny_vae"
TAEHV_SOURCE = TINY_DIR / "taehv" / "taehv.py"
ENGINE_DIR = ROOT / "benchmarks" / "tiny_vae_20260926" / "engines"

# The recommended policy for a tiny-decoder arm: decoder {backend: pytorch, scheme: bf16} makes
# _install_decoder_policy a no-op (no TensorRT arena/engines, no Wan decoder compile).
TINY_DECODER_POLICY_HINT = "benchmarks/tiny_vae_20260926/policies/tiny_v4_flash2.json"

BACKENDS = ("eager", "compile", "tensorrt")
DECODE_MODES = ("stream", "window")


@dataclass(frozen=True)
class CatalogEntry:
    name: str
    arch: str               # "taehv" | "wan_dim24"
    default_path: Path
    decoder_latent: str     # "dit_normalised" | "wan_raw" | "wan_protocol"
    encoder_latent: str     # "dit_normalised" | "wan_raw" | "wan_protocol"
    weights_dtype: str
    licence: str
    source: str
    evidence: str


CATALOG: dict[str, CatalogEntry] = {
    "taew2_1": CatalogEntry(
        "taew2_1", "taehv", TINY_DIR / "taehv" / "taew2_1.pth", "dit_normalised", "dit_normalised",
        "fp16", "MIT", "github.com/madebyollin/taehv commit 011dfc2",
        "GPU decoder bake-off 39.63 dB vs stock Wan (un-normalised input 19.04 dB); encoder nRMSE 0.138/0.089",
    ),
    "lighttaew2_1": CatalogEntry(
        "lighttaew2_1", "taehv", TINY_DIR / "lightx2v" / "lighttaew2_1.safetensors", "wan_raw", "wan_raw",
        "fp32", "Apache-2.0", "huggingface.co/lightx2v/Autoencoders sha 02cbfd1",
        "GPU decoder bake-off 39.04 dB with z*std+mean (normalised input 20.25 dB); encoder raw output",
    ),
    "lightvaew2_1": CatalogEntry(
        "lightvaew2_1", "wan_dim24", TINY_DIR / "lightx2v" / "lightvaew2_1.safetensors", "wan_protocol", "wan_protocol",
        "fp32", "Apache-2.0", "huggingface.co/lightx2v/Autoencoders sha 02cbfd1",
        "loads strict into flash_head WanVAE_(dim=24); GPU decoder bake-off 40.30 dB but mouth sharpness 0.835x",
    ),
}

LATENT_CONVENTION_TEXT = {
    "dit_normalised": "DiT-normalised latent (z - mean) * inv_std used as is (TAEHV: no latent scale/shift)",
    "wan_raw": "raw Wan latent: decoder input z / inv_std + mean; encoder output (e - mean) * inv_std",
    "wan_protocol": "WanVAE_ protocol: decode(z, [mean, inv_std]) / encode(x, [mean, inv_std]) un/normalise internally",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_tiny_vae(spec: str) -> dict[str, Any]:
    """``NAME`` or ``NAME:PATH`` (a checkpoint with NAME's architecture and convention, e.g. a
    SoulX fine-tune of taew2_1), or a bare PATH whose file name contains a catalog name.
    The architecture and latent convention always come from the catalog, never from guessing."""
    if not isinstance(spec, str) or not spec:
        raise ValueError("tiny VAE spec must be a non-empty string")
    name, path = spec, None
    if ":" in spec:
        name, raw_path = spec.split(":", 1)
        path = Path(raw_path)
    elif spec not in CATALOG:
        candidate = Path(spec)
        stem = candidate.name
        # longest match first: "lighttaew2_1" contains "taew2_1"
        matches = sorted((n for n in CATALOG if n in stem), key=len, reverse=True)
        if not matches:
            raise ValueError(
                f"Unknown tiny VAE {spec!r}: use one of {sorted(CATALOG)} or NAME:PATH "
                "(the architecture and latent convention must be named, not guessed)"
            )
        name, path = matches[0], candidate
    if name not in CATALOG:
        raise ValueError(f"Unknown tiny VAE {name!r}; expected one of {sorted(CATALOG)}")
    entry = CATALOG[name]
    path = Path(path) if path is not None else entry.default_path
    if not path.is_absolute():
        path = (ROOT / path) if not path.exists() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tiny VAE weights not found: {path}")
    return {
        "name": entry.name,
        "arch": entry.arch,
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "abs_path": str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "weights_dtype": entry.weights_dtype,
        "licence": entry.licence,
        "source": entry.source,
        "decoder_latent_convention": entry.decoder_latent,
        "encoder_latent_convention": entry.encoder_latent,
        "evidence": entry.evidence,
    }


# ----------------------------------------------------------------------------- flag validation
def tiny_vae_requested(args) -> bool:
    return bool(getattr(args, "tiny_vae_decoder", None) or getattr(args, "tiny_vae_encoder", None))


def validate_tiny_vae_args(args, policy: dict[str, Any]) -> None:
    """Refuse combinations that would silently measure or run the wrong thing. CPU only; runs
    before the GPU lease. No-op (and no requirement) when neither tiny flag is given, except that
    the tiny-only knobs are refused on their own."""
    decoder = getattr(args, "tiny_vae_decoder", None)
    encoder = getattr(args, "tiny_vae_encoder", None)
    backend = getattr(args, "tiny_vae_backend", None)
    mode = getattr(args, "tiny_vae_decode_mode", None)
    if not decoder and not encoder:
        if backend is not None or mode is not None:
            raise ValueError("--tiny-vae-backend / --tiny-vae-decode-mode need --tiny-vae-decoder or --tiny-vae-encoder")
        return
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"--tiny-vae-backend must be one of {BACKENDS}")
    if mode is not None:
        if mode not in DECODE_MODES:
            raise ValueError(f"--tiny-vae-decode-mode must be one of {DECODE_MODES}")
        if not decoder:
            raise ValueError("--tiny-vae-decode-mode needs --tiny-vae-decoder")
    errors: list[str] = []
    if int(getattr(args, "sessions", 1)) > 1 or getattr(args, "force_scheduler", False):
        errors.append("--sessions > 1 / --force-scheduler: the tiny VAE's stream state is single-session")
    if getattr(args, "capture_manifest", None) is not None:
        errors.append("--capture-manifest records Wan decoder activations; not meaningful with a tiny VAE")
    for spec in (decoder, encoder):
        if spec and backend == "tensorrt":
            entry = CATALOG.get(spec.split(":", 1)[0])
            if entry is not None and entry.arch != "taehv":
                errors.append(f"--tiny-vae-backend tensorrt is implemented for TAEHV models only, not {entry.name}")
    if decoder:
        dec = policy.get("decoder", {})
        if dec.get("backend") != "pytorch" or dec.get("scheme") != "bf16" or dec.get("plan") is not None:
            errors.append(
                "a tiny decoder needs a policy whose decoder is {backend: pytorch, scheme: bf16} with no plan "
                f"(got {dec}); a TensorRT/compiled Wan decoder policy would allocate engines or compile a decoder "
                f"that never runs. Use {TINY_DECODER_POLICY_HINT}"
            )
        if getattr(args, "overlap_skip", False):
            errors.append("--overlap-skip hard-wires the Wan cached_decode; the tiny decoder has its own "
                          "carried-state equivalent (--tiny-vae-decode-mode stream, the default)")
        for flag, attr in (("--skip-decoder-blocks", "skip_decoder_blocks"), ("--subpixel-resample", "subpixel_resample"),
                           ("--channels-last-head", "channels_last_head"), ("--vae-weights", "vae_weights")):
            if getattr(args, attr, None):
                errors.append(f"{flag} rewrites/reloads the Wan decoder, which a tiny decoder never runs")
        if getattr(args, "profile_component", None) == "decoder":
            errors.append("--profile-component decoder profiles the Wan decoder; not meaningful with a tiny decoder")
    if encoder:
        if getattr(args, "latent_feedback", "off") != "off":
            errors.append("--latent-feedback replaces the motion re-encode itself (last2-fix0 would route a "
                          "1-frame encode through the tiny encoder); pick one lever")
        if getattr(args, "compile_vae_encode", False):
            errors.append("--compile-vae-encode compiles the Wan motion encoder the tiny encoder replaces; "
                          "use --tiny-vae-backend compile to compile the tiny encoder")
    if errors:
        raise ValueError("Incompatible tiny-VAE flags:\n  - " + "\n  - ".join(errors))


# ----------------------------------------------------------------------------- model loading
def _import_taehv():
    # Registered in sys.modules: Dynamo re-imports a traced function's module by name.
    name = "soulx_vendor_taehv"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, TAEHV_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)  # upstream file, unmodified
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def load_tiny_model(info: dict[str, Any], device="cuda"):
    """Returns (module, taehv_module_or_None). TAEHV in fp16, LightVAE in bf16, eval, no grad."""
    if info["arch"] == "taehv":
        taehv = _import_taehv()
        # arch_name pinned: the architecture is the taew2_1 layout for both TAEHV checkpoints
        model = taehv.TAEHV(checkpoint_path=info["abs_path"], arch_name="taew2_1")
        return model.eval().requires_grad_(False).to(device=device, dtype=torch.float16), taehv
    if info["arch"] == "wan_dim24":
        from safetensors.torch import load_file
        from flash_head.wan.modules import vae as wan_vae

        model = wan_vae.WanVAE_(dim=24, z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
                                temperal_downsample=[False, True, True], dropout=0.0)
        path = info["abs_path"]
        state = load_file(path) if path.endswith(".safetensors") else torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        return model.eval().requires_grad_(False).to(device=device, dtype=torch.bfloat16), None
    raise ValueError(f"unknown tiny VAE arch {info['arch']!r}")


# ----------------------------------------------------------------------------- TAEHV static cores
class TAEHVDecodeCore(nn.Module):
    """Time-parallel TAEHV decode of one window with explicit per-MemBlock state.

    forward(x (T,16,h,w), *states) -> (frames (4T,3,8h,8w) in [0,1], *new_states)
    states[k] is the previous timestep's input of the k-th MemBlock, (1, C_k, h_k, w_k). Zeros
    reproduce upstream decode_video(parallel=True) (it zero-pads the memory); the previous
    window's new_states continue the stream exactly. Plain Conv2d/ReLU/tanh/Upsample/cat/reshape,
    so torch.compile and ONNX/TensorRT see one static graph per T.
    (Same core as benchmarks/tiny_vae_20260926/decoder_bakeoff/bakeoff_common.py.)
    """

    def __init__(self, tae, memblock_type):
        super().__init__()
        self.decoder = tae.decoder
        self._memblock = memblock_type
        self.trim = int(tae.frames_to_trim)
        self.t_upscale = int(tae.t_upscale)

    def state_shapes(self, h: int, w: int):
        shapes, up = [], 1
        for block in self.decoder:
            if isinstance(block, nn.Upsample):
                up *= int(block.scale_factor)
            elif isinstance(block, self._memblock):
                shapes.append((1, block.conv[0].in_channels // 2, h * up, w * up))
        return shapes

    def zero_states(self, h, w, device, dtype):
        return [torch.zeros(s, device=device, dtype=dtype) for s in self.state_shapes(h, w)]

    def forward(self, x, *states):
        new = []
        k = 0
        for block in self.decoder:
            if isinstance(block, self._memblock):
                past = torch.cat([states[k], x[:-1]], 0)
                new.append(x[-1:])
                x = block(x, past)
                k += 1
            else:
                x = block(x)
        return (x.clamp(0, 1), *new)


class TAEHVEncodeCore(nn.Module):
    """Time-parallel TAEHV encode of one clip: (T, 3, H, W) in [0,1], T % 4 == 0 -> (T/4, 16, H/8, W/8).
    Zero memory at t=0 (== upstream encode_video(parallel=True) for one video); stateless across calls,
    like WanVAE.encode which clears its cache every call."""

    def __init__(self, tae, memblock_type):
        super().__init__()
        self.encoder = tae.encoder
        self._memblock = memblock_type
        self.t_downscale = int(tae.t_downscale)

    def forward(self, x):
        for block in self.encoder:
            if isinstance(block, self._memblock):
                past = torch.cat([torch.zeros_like(x[:1]), x[:-1]], 0)
                x = block(x, past)
            else:
                x = block(x)
        return x


# ----------------------------------------------------------------------------- TensorRT
class _SharedDeviceMemory:
    """One torch-owned workspace for every tiny-VAE engine context: they run strictly in order
    on one stream, so they never need their activation memory at the same time."""

    def __init__(self):
        self.buffer = None
        self.contexts = []

    def attach(self, context, size: int):
        self.contexts.append((context, int(size)))
        need = max(s for _, s in self.contexts)
        if self.buffer is None or self.buffer.numel() < need:
            self.buffer = torch.empty(max(need, 1), dtype=torch.uint8, device="cuda")
            for ctx, _ in self.contexts:
                self._point(ctx)
        else:
            self._point(context)

    def _point(self, context):
        if hasattr(context, "set_device_memory"):
            context.set_device_memory(self.buffer.data_ptr(), int(self.buffer.numel()))
        else:  # older TensorRT 10.x
            context.device_memory = self.buffer.data_ptr()

    @property
    def mib(self) -> float:
        return 0.0 if self.buffer is None else self.buffer.numel() / 2**20


class TRTModule:
    """A static-shape FP16 TensorRT engine: fp16 contiguous CUDA tensors in, preallocated fp16
    outputs (reused across calls: callers must consume/clone before the next call)."""

    def __init__(self, engine_path: Path, logger, shared: _SharedDeviceMemory):
        import tensorrt as trt

        self.trt = trt
        self.runtime = trt.Runtime(logger)
        self.engine = self.runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(f"TensorRT could not deserialize {engine_path}")
        self.context = self.engine.create_execution_context_without_device_memory()
        size = self.engine.device_memory_size_v2 if hasattr(self.engine, "device_memory_size_v2") else self.engine.device_memory_size
        shared.attach(self.context, int(size))
        self.device_memory_mib = int(size) / 2**20
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.buffers = {n: torch.empty(tuple(self.engine.get_tensor_shape(n)), dtype=torch.float16, device="cuda")
                        for n in self.outputs}
        for n in self.outputs:
            self.context.set_tensor_address(n, self.buffers[n].data_ptr())
        self.input_shapes = [tuple(self.engine.get_tensor_shape(n)) for n in self.inputs]

    def __call__(self, *tensors):
        if len(tensors) != len(self.inputs):
            raise ValueError(f"engine expects {len(self.inputs)} inputs, got {len(tensors)}")
        for name, shape, tensor in zip(self.inputs, self.input_shapes, tensors):
            if tensor.dtype != torch.float16 or not tensor.is_contiguous() or tuple(tensor.shape) != shape:
                raise ValueError(f"engine input {name}: need fp16 contiguous {shape}, got {tensor.dtype} {tuple(tensor.shape)}")
            self.context.set_tensor_address(name, tensor.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        outs = tuple(self.buffers[n] for n in self.outputs)
        return outs if len(outs) > 1 else outs[0]


def _gpu_tag() -> str:
    name = torch.cuda.get_device_name().replace(" ", "_")
    return "".join(c for c in name if c.isalnum() or c in "_-")


def build_or_load_engine(core: nn.Module, example: tuple, n_outputs: int, tag: str, info: dict, logger,
                         shared: _SharedDeviceMemory, workspace_mib: int = 3072) -> tuple[TRTModule, dict]:
    """Export ``core`` at the example's static shapes to ONNX (opset 18), build a strongly typed
    FP16 engine, and cache it by weights sha256 + shapes + TensorRT version + GPU name."""
    import tensorrt as trt

    shape_key = "_".join("x".join(str(d) for d in t.shape) for t in example[:1])
    key = f"{info['name']}-{info['sha256'][:12]}-{tag}-{shape_key}-trt{trt.__version__}-{_gpu_tag()}"
    ENGINE_DIR.mkdir(parents=True, exist_ok=True)
    engine_path = ENGINE_DIR / f"{key}.engine"
    record: dict[str, Any] = {"engine": str(engine_path.relative_to(ROOT)), "tensorrt": trt.__version__, "cached": engine_path.is_file()}
    if not engine_path.is_file():
        onnx_path = engine_path.with_suffix(".onnx")
        n_in, n_out = len(example), int(n_outputs)
        t0 = time.perf_counter()
        with torch.inference_mode():
            torch.onnx.export(core, example, str(onnx_path), input_names=[f"i{k}" for k in range(n_in)],
                              output_names=[f"o{k}" for k in range(n_out)], opset_version=18, dynamo=False,
                              do_constant_folding=True)
        record["export_s"] = time.perf_counter() - t0
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, logger)
        if not parser.parse_from_file(str(onnx_path)):
            raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 2**20)
        config.builder_optimization_level = 3
        t1 = time.perf_counter()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT build returned no engine")
        engine_path.write_bytes(bytes(serialized))
        record["build_s"] = time.perf_counter() - t1
        onnx_path.unlink(missing_ok=True)
        for extra in ENGINE_DIR.glob(onnx_path.stem + "*.data"):
            extra.unlink(missing_ok=True)
        del builder, network, parser, config, serialized
        torch.cuda.empty_cache()
    record["engine_sha256"] = sha256(engine_path)
    record["engine_bytes"] = engine_path.stat().st_size
    module = TRTModule(engine_path, logger, shared)
    record["device_memory_mib"] = module.device_memory_mib
    return module, record


@contextlib.contextmanager
def _cudnn_benchmark(enabled: bool):
    """Scoped cudnn.benchmark: the TAEHV convolutions pick their algorithm per shape without
    changing the global flag the rest of the pipeline (Wan encoder, DiT) runs under."""
    if not enabled:
        yield
        return
    previous = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = True
    try:
        yield
    finally:
        torch.backends.cudnn.benchmark = previous


# ----------------------------------------------------------------------------- decoder
class TinyDecoder:
    """Replaces ``WanVAE.decode`` (see module docstring for the contract and modes)."""

    def __init__(self, info: dict, model, taehv_module, scale, *, mode: str, backend: str,
                 history_frames: int, cold_latents: int, latent_hw: tuple[int, int],
                 cudnn_benchmark: bool = True, trt_logger=None, shared_memory: _SharedDeviceMemory | None = None):
        if mode not in DECODE_MODES:
            raise ValueError(f"decode mode must be one of {DECODE_MODES}")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.info, self.model, self.mode, self.backend = info, model, mode, backend
        self.arch = info["arch"]
        self.convention = info["decoder_latent_convention"]
        self.history = int(history_frames)
        self.cold_latents = int(cold_latents)
        self.cold_frames = 1 + 4 * (self.cold_latents - 1)
        self.keep = (self.cold_frames - self.history) // 4
        if (self.cold_frames - self.history) % 4:
            raise ValueError("useful frames per window must be a multiple of 4 for a carried-state decode")
        self.scale = scale  # [mean, inv_std] as WanVAE builds them (bf16)
        self.mean = scale[0].float().view(-1, 1, 1, 1)
        self.inv_std = scale[1].float().view(-1, 1, 1, 1)
        self.cudnn_benchmark = bool(cudnn_benchmark) and backend != "tensorrt"
        self.engines: dict[str, Any] = {}
        self.calls = 0
        h, w = latent_hw
        if self.arch == "taehv":
            core = TAEHVDecodeCore(model, taehv_module.MemBlock).eval()
            self._trim = core.trim
            self._zero_states = core.zero_states(h, w, "cuda", torch.float16)
            if backend == "eager":
                self._cores = {"cold": core, "warm": core}
            elif backend == "compile":
                compiled = torch.compile(core, dynamic=False, fullgraph=True)
                self._cores = {"cold": compiled, "warm": compiled}
            else:
                shared = shared_memory or _SharedDeviceMemory()
                self._cores = {}
                shapes = [("cold", self.cold_latents)] + ([("warm", self.keep)] if mode == "stream" else [])
                for label, t in shapes:
                    example = (torch.zeros(t, 16, h, w, device="cuda", dtype=torch.float16), *self._zero_states)
                    self._cores[label], self.engines[label] = build_or_load_engine(
                        core, example, 1 + len(self._zero_states), f"dec{label}", info, trt_logger, shared)
                self._shared = shared
        elif self.arch == "wan_dim24":
            if backend == "tensorrt":
                raise ValueError("tensorrt backend is implemented for TAEHV only")
            cached = model.cached_decode
            decode = model.decode
            if backend == "compile":
                cached = torch.compile(cached, dynamic=False)
                decode = torch.compile(decode, dynamic=False)
            self._wan_cached, self._wan_decode = cached, decode
        else:
            raise ValueError(f"unknown arch {self.arch}")
        self.reset()

    # -- state
    def reset(self):
        self.state = None
        self.tail = None
        self.windows = 0
        self.frames_per_window: list[int] = []
        self.latents_decoded_per_window: list[int] = []
        self.latents_skipped = 0

    # -- helpers
    def _prep(self, zs):
        z = zs
        if self.convention == "wan_raw":
            z = zs.float() / self.inv_std + self.mean
        return z.permute(1, 0, 2, 3).to(torch.float16).contiguous()  # (T,16,h,w)

    @staticmethod
    def _to_wan(frames):  # (F,3,H,W) fp16 [0,1] -> (1,3,F,H,W) bf16 [-1,1]
        return frames.permute(1, 0, 2, 3).unsqueeze(0).mul(2).sub(1).clamp_(-1, 1).to(torch.bfloat16).contiguous()

    def _run(self, label, x, states):
        out = self._cores[label](x, *states)
        return out[0], list(out[1:])

    def _taehv(self, zs):
        stream = self.mode == "stream"
        if not stream or self.state is None:
            x = self._prep(zs)
            if x.shape[0] != self.cold_latents:
                raise ValueError(f"tiny decoder built for {self.cold_latents}-latent windows, got {x.shape[0]}")
            frames, new = self._run("cold", x, self._zero_states)
            out = self._to_wan(frames[self._trim:])
            decoded = int(zs.shape[1])
        else:
            x = self._prep(zs[:, -self.keep:])
            frames, new = self._run("warm", x, self.state)
            out = torch.cat([self.tail, self._to_wan(frames)], dim=2)
            decoded = self.keep
            self.latents_skipped += int(zs.shape[1]) - self.keep
        if stream:
            # clone: TensorRT reuses its output buffers, and a view would pin the window tensor
            self.state = [s.clone() for s in new]
            self.tail = out[:, :, -self.history:].clone()
        return out, decoded

    def _wan24(self, zs):
        model = self.model
        if self.mode == "window":
            return self._wan_decode(zs.unsqueeze(0), self.scale).clamp_(-1, 1), int(zs.shape[1])
        if self.state is None:
            model.clear_cache()
            out = self._wan_cached(zs.unsqueeze(0), self.scale).clamp_(-1, 1)
            decoded = int(zs.shape[1])
        else:
            model._feat_map = self.state
            model._conv_idx = [0]
            fresh = self._wan_cached(zs[:, -self.keep:].unsqueeze(0), self.scale).clamp_(-1, 1)
            out = torch.cat([self.tail, fresh], dim=2)
            decoded = self.keep
            self.latents_skipped += int(zs.shape[1]) - self.keep
        self.state = model._feat_map
        self.tail = out[:, :, -self.history:].detach().clone()
        return out.to(torch.bfloat16), decoded

    def __call__(self, zs):
        if zs.dim() != 4 or zs.shape[0] != 16:
            raise ValueError(f"tiny decoder expects (16, T, h, w) latents, got {tuple(zs.shape)}")
        with _cudnn_benchmark(self.cudnn_benchmark):
            out, decoded = (self._taehv if self.arch == "taehv" else self._wan24)(zs)
        frames = int(out.shape[2])
        # A window with a different frame count shifts every later frame against the audio and
        # the harness only checks the total; catch it where the count is decided.
        if frames != self.cold_frames:
            raise ValueError(f"tiny decoder emitted {frames} frames for window {self.windows}, expected {self.cold_frames}")
        self.windows += 1
        self.calls += 1
        self.frames_per_window.append(frames)
        self.latents_decoded_per_window.append(decoded)
        return out

    def manifest(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in self.info.items() if k != "abs_path"},
            "role": "decoder",
            "latent_convention": LATENT_CONVENTION_TEXT[self.convention],
            "decode_mode": self.mode,
            "backend": self.backend,
            "cudnn_benchmark_scoped": self.cudnn_benchmark,
            "history_frames": self.history,
            "cold_latents": self.cold_latents,
            "keep_latents_warm": self.keep if self.mode == "stream" else None,
            "frames_per_window_contract": self.cold_frames,
            "output": "(1,3,F,H,W) bf16 clamp[-1,1] NCTHW; TAEHV [0,1] mapped x*2-1; first 3 raw frames trimmed on cold decode",
            "engines": self.engines or None,
            "tensorrt_shared_device_memory_mib": getattr(getattr(self, "_shared", None), "mib", None),
        }

    def state_manifest(self) -> dict[str, Any]:
        return {"windows": self.windows, "frames_per_window": list(self.frames_per_window),
                "latents_decoded_per_window": list(self.latents_decoded_per_window),
                "latents_skipped": self.latents_skipped}


# ----------------------------------------------------------------------------- encoder
class TinyEncoder:
    """Replaces ``WanVAE.encode`` for the per-window motion re-encode only."""

    def __init__(self, info: dict, model, taehv_module, scale, *, backend: str, clip_frames: int,
                 image_hw: tuple[int, int], cudnn_benchmark: bool = True, trt_logger=None,
                 shared_memory: _SharedDeviceMemory | None = None):
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}")
        self.info, self.model, self.backend = info, model, backend
        self.arch = info["arch"]
        self.convention = info["encoder_latent_convention"]
        self.scale = scale
        self.mean = scale[0].float().view(-1, 1, 1, 1)
        self.inv_std = scale[1].float().view(-1, 1, 1, 1)
        self.clip_frames = int(clip_frames)
        self.cudnn_benchmark = bool(cudnn_benchmark) and backend != "tensorrt"
        self.engines: dict[str, Any] = {}
        self.calls = 0
        self.frames_seen: list[int] = []
        if self.arch == "taehv":
            core = TAEHVEncodeCore(model, taehv_module.MemBlock).eval()
            self._t_down = core.t_downscale
            if backend == "eager":
                self._core = core
            elif backend == "compile":
                self._core = torch.compile(core, dynamic=False, fullgraph=True)
            else:
                shared = shared_memory or _SharedDeviceMemory()
                padded = self.clip_frames + (-self.clip_frames) % self._t_down
                example = (torch.zeros(padded, 3, *image_hw, device="cuda", dtype=torch.float16),)
                self._core, self.engines["clip"] = build_or_load_engine(core, example, 1, "enc", info, trt_logger, shared)
                self._shared = shared
        elif self.arch == "wan_dim24":
            if backend == "tensorrt":
                raise ValueError("tensorrt backend is implemented for TAEHV only")
            encode = model.encode
            self._wan_encode = torch.compile(encode, dynamic=False) if backend == "compile" else encode
        else:
            raise ValueError(f"unknown arch {self.arch}")

    def _taehv(self, video):
        x = video[0].transpose(0, 1).to(torch.float16)  # (t,3,H,W)
        x = (x + 1) * 0.5
        t = x.shape[0]
        pad = (-t) % self._t_down
        if pad:  # upstream encode_video: append copies of the LAST frame
            x = torch.cat([x, x[-1:].expand(pad, -1, -1, -1)], 0)
        x = x.contiguous()
        e = self._core(x)  # (L,16,h,w)
        e = e.transpose(0, 1)
        if self.convention == "wan_raw":
            e = (e.float() - self.mean) * self.inv_std
        return e.to(torch.bfloat16).contiguous()

    def _wan24(self, video):
        return self._wan_encode(video.to(torch.bfloat16), self.scale).squeeze(0)

    def __call__(self, video, *args, **kwargs):
        if args or kwargs:
            raise ValueError("tiny encoder takes the video only (no distributed split arguments)")
        if video.dim() != 5 or video.shape[0] != 1 or video.shape[1] != 3:
            raise ValueError(f"tiny encoder expects (1, 3, t, H, W), got {tuple(video.shape)}")
        t = int(video.shape[2])
        if t != self.clip_frames:
            raise ValueError(f"tiny encoder is installed for the {self.clip_frames}-frame motion clip only, got {t} frames "
                             "(the reference-image encode must stay on the Wan encoder)")
        with _cudnn_benchmark(self.cudnn_benchmark):
            out = (self._taehv if self.arch == "taehv" else self._wan24)(video)
        expected = 1 + (t - 1) // 4
        if tuple(out.shape[:2]) != (16, expected):
            raise ValueError(f"tiny encoder returned {tuple(out.shape)}, expected (16, {expected}, h, w) like WanVAE.encode")
        self.calls += 1
        return out

    def manifest(self) -> dict[str, Any]:
        return {
            **{k: v for k, v in self.info.items() if k != "abs_path"},
            "role": "motion encoder (per-window re-encode only; reference latent stays on Wan)",
            "latent_convention": LATENT_CONVENTION_TEXT[self.convention],
            "frame_alignment": "TAEHV: (x+1)/2, append copies of the last frame to a multiple of 4 (upstream encode_video)"
                               if self.arch == "taehv" else "Wan protocol (frame 0 alone, then chunks of 4)",
            "backend": self.backend,
            "cudnn_benchmark_scoped": self.cudnn_benchmark,
            "clip_frames": self.clip_frames,
            "engines": self.engines or None,
        }


# ----------------------------------------------------------------------------- install
# Decoder and encoder engines run strictly in order on one stream: one workspace serves both.
_SHARED_TRT_MEMORY = _SharedDeviceMemory()


def _trt_logger(backend):
    if backend != "tensorrt":
        return None
    import tensorrt as trt

    return trt.Logger(trt.Logger.WARNING)


def install_tiny_decoder(pipeline, spec: str, *, mode: str = "stream", backend: str = "eager",
                         history_frames: int, model_frames: int, resolution: tuple[int, int]) -> TinyDecoder:
    """Set ``pipeline.vae.decode`` to a TinyDecoder. Call after the decoder policy (which must be
    the no-op pytorch/bf16 one) and before dumps/instrumentation wrap vae.decode."""
    vae = pipeline.vae
    if hasattr(vae, "_pro_overlap_skip"):
        raise ValueError("overlap-skip is installed; it and a tiny decoder both replace vae.decode")
    if hasattr(vae, "_pro_tiny_decoder"):
        raise ValueError("a tiny decoder is already installed")
    info = resolve_tiny_vae(spec)
    model, taehv = load_tiny_model(info)
    cold_latents = (int(model_frames) - 1) // 4 + 1
    decoder = TinyDecoder(info, model, taehv, vae.scale, mode=mode, backend=backend, history_frames=history_frames,
                          cold_latents=cold_latents, latent_hw=(resolution[0] // 8, resolution[1] // 8),
                          trt_logger=_trt_logger(backend), shared_memory=_SHARED_TRT_MEMORY)
    vae._pro_tiny_decoder = decoder
    vae.decode = decoder
    return decoder


def install_tiny_encoder(pipeline, spec: str, *, backend: str = "eager", clip_frames: int,
                         resolution: tuple[int, int]) -> TinyEncoder:
    """Set ``pipeline.vae.encode`` to a TinyEncoder. Call AFTER prepare_params (so the reference
    latent is Wan-encoded) and BEFORE overlap-skip/dumps/instrumentation wrap vae.encode."""
    vae = pipeline.vae
    if hasattr(vae, "_pro_overlap_skip"):
        raise ValueError("install the tiny encoder before overlap-skip, whose shim must wrap it")
    if hasattr(vae, "_pro_tiny_encoder"):
        raise ValueError("a tiny encoder is already installed")
    info = resolve_tiny_vae(spec)
    model, taehv = load_tiny_model(info)
    encoder = TinyEncoder(info, model, taehv, vae.scale, backend=backend, clip_frames=clip_frames,
                          image_hw=tuple(resolution), trt_logger=_trt_logger(backend),
                          shared_memory=_SHARED_TRT_MEMORY)
    vae._pro_tiny_encoder = encoder
    vae.encode = encoder
    return encoder


def tiny_sources() -> list[Path]:
    """Extra files for run.py's source manifest when a tiny VAE is active."""
    return [Path(__file__), TAEHV_SOURCE]
