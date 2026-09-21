"""Opt-in PRO FFN compute quantization, pinned/tested against Torch 2.7.1 CUDA.

These experimental modules use real FP8/INT8 GEMMs, with BF16 interfaces.
They are deliberately not selected by the production Engine. Apply after the
ordinary checkpoint load/device/dtype conversion and before compilation.
"""
import torch
from torch import nn


class Float8Linear(nn.Module):
    """Tensorwise E4M3 weights and dynamically scaled E4M3 activations.

    ``fast_accum`` selects cuBLASLt's fast-accumulate tier. It defaults to False,
    which is the setting every retained measurement used. It is exposed so the
    two readings of the SM89 FP8 path can be told apart by measurement rather
    than argument: either the Ada FP8/FP16-accumulate tier is genuinely twice
    the FP32-accumulate tier, or the flag is a no-op on the ``m16n8k32`` path
    whose accumulator is architecturally f32. Turning it on is a numerical
    change on a path already showing 4.79% relative L2 at block 14, so it needs
    its own quality arm, not just a speed pass.
    """

    def __init__(self, linear, fast_accum: bool = False):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.fast_accum = bool(fast_accum)
        with torch.no_grad():
            weight = linear.weight.detach().float()
            scale = weight.abs().amax().clamp_min(1e-12) / 448.0
            quantized = (weight / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        self.register_buffer("weight_fp8", quantized.contiguous())
        self.register_buffer("weight_scale", scale.float())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        scale = flat.abs().amax().float().clamp_min(1e-12) / 448.0
        quantized = (flat.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        output = torch._scaled_mm(
            quantized, self.weight_fp8.t(), scale_a=scale,
            scale_b=self.weight_scale, bias=self.bias,
            out_dtype=x.dtype, use_fast_accum=self.fast_accum,
        )
        return output.reshape(*shape[:-1], self.out_features)


class Int8ComputeLinear(nn.Module):
    """Per-token activation / per-output-channel weight INT8 GEMM."""

    def __init__(self, linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        with torch.no_grad():
            weight = linear.weight.detach().float()
            scale = weight.abs().amax(dim=1).clamp_min(1e-12) / 127.0
            quantized = (weight / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        # Store [N, K] row-major and transpose at call time, so the operand handed
        # to torch._int_mm is [K, N] with stride (1, N-major source) -- i.e. the
        # K-contiguous "TN" layout cuBLAS needs to select the INT8 tensor-core
        # tactic (cutlass_*_tensorop_i16832gemm_s8_*_tn_*).
        #
        # The previous form, quantized.t().contiguous(), materialised a
        # row-major [K, N] buffer, which is N-contiguous from the GEMM's point of
        # view and silently fell back to the non-tensorop ampere_igemm_int8_*_nn
        # kernel. Measured on this SM89 part at the real FFN shapes
        # (M=6480): 1536->8960 2.932 -> 0.982 ms (2.99x), 8960->1536
        # 2.770 -> 0.826 ms (3.35x). Numerics are unchanged -- torch._int_mm is
        # exact integer arithmetic and the same values are multiplied either way.
        self.register_buffer("weight_int8", quantized.contiguous())
        self.register_buffer("weight_scale", scale.float())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        # Optional static activation scale. When set, forward skips the per-row amax
        # reduction -- which is what forces the producer's output to be read from DRAM
        # TWICE (once to reduce, once to quantize). At the FFN's 8960-wide site that
        # buffer is 232 MB of INT32, so the second read costs ~0.54 ms/block-forward.
        # A scalar scale collapses it to one pass. This IS a numerical change: per-row
        # adaptivity is lost and any activation above the calibrated ceiling saturates
        # at +/-127. Opt-in, and gated on the oral-edge-ratio review like any other
        # numerical change.
        self.register_buffer("static_input_scale", None)
        self.register_buffer("observed_amax", torch.zeros((), dtype=torch.float32))
        self.calibrating = False

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        if self.calibrating:
            with torch.no_grad():
                # torch.maximum, not .item(): a host sync here would distort the very
                # timings the calibration run is meant to leave untouched.
                self.observed_amax.copy_(
                    torch.maximum(self.observed_amax, flat.abs().amax().float())
                )
        if self.static_input_scale is not None:
            scale = self.static_input_scale
        else:
            scale = flat.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / 127.0
        quantized = (flat.float() / scale).round().clamp(-127, 127).to(torch.int8)
        output = torch._int_mm(quantized, self.weight_int8.t()).float()
        output = output * scale * self.weight_scale[None, :]
        if self.bias is not None:
            output = output + self.bias.float()
        return output.to(x.dtype).reshape(*shape[:-1], self.out_features)


def quantize_pro_ffns(model, precision="none", exclude=()):
    """Replace only the named PRO FFN linears; return an inspectable manifest."""
    if precision not in ("none", "fp8", "int8"):
        raise ValueError("precision must be none, fp8 or int8")
    if tuple(model.config.vae_stride) != (4, 8, 8) or tuple(model.patch_size) != (1, 2, 2):
        raise ValueError("This policy is validated only for PRO geometry")
    exclude = set(exclude)
    valid = {f"blocks.{i}.ffn.{j}" for i in range(len(model.blocks)) for j in (0, 2)}
    if exclude - valid:
        raise ValueError(f"Unknown FFN exclusions: {sorted(exclude - valid)}")
    report = dict(precision=precision, backend="pytorch_native_cuda", modules=[],
                  excluded=sorted(exclude), original_bytes=0, quantized_bytes=0,
                  interface_dtype="bfloat16", torch=torch.__version__)
    if precision == "none":
        return report
    cls = Float8Linear if precision == "fp8" else Int8ComputeLinear
    for i, block in enumerate(model.blocks):
        for j in (0, 2):
            name = f"blocks.{i}.ffn.{j}"
            if name in exclude:
                continue
            linear = block.ffn[j]
            if not isinstance(linear, nn.Linear) or linear.weight.dtype != torch.bfloat16:
                raise ValueError(f"Expected an unmodified BF16 Linear at {name}")
            if linear.in_features % 16 or linear.out_features % 16:
                raise ValueError(f"Unsupported GEMM alignment at {name}")
            converted = cls(linear)
            report["original_bytes"] += sum(p.numel() * p.element_size() for p in linear.parameters())
            report["quantized_bytes"] += sum(b.numel() * b.element_size() for b in converted.buffers())
            block.ffn[j] = converted
            report["modules"].append(name)
    return report


def optimize_wan_vae(vae, mode="none"):
    """Independent precision-preserving layout/pointwise decoder experiment."""
    if mode not in ("none", "channels_last", "pointwise", "compiled", "compiled_all"):
        raise ValueError("Unknown Wan optimization mode")
    report = {"mode": mode, "compiled_modules": [], "precision_changed": False}
    if mode == "none":
        return report
    if mode in ("channels_last", "pointwise"):
        torch.nn.utils.convert_conv3d_weight_memory_format(vae.model.decoder, torch.channels_last_3d)
    if mode == "pointwise":
        for name, module in vae.model.decoder.named_modules():
            if type(module).__name__ == "RMS_norm":
                module.forward = torch.compile(module.forward, fullgraph=True, dynamic=False)
                report["compiled_modules"].append(name)
    if mode in ("compiled", "compiled_all"):
        vae.decode = torch.compile(vae.decode, dynamic=False)
        report["compiled_modules"].append("vae.decode")
    if mode == "compiled_all":
        # The motion-feedback encoder runs once per window on motion_frames_num
        # frames and is 1.3764 s / 6.53% of the measured 21.0951 s baseline, in
        # BF16 and uncompiled: flash_head_pipeline sets COMPILE_VAE = True as the
        # module default but the benchmark harness disables it, and the "compiled"
        # mode above only ever restored vae.decode.
        #
        # Kept as a distinct mode rather than folded into "compiled" so that
        # decoder_trial.py's compiled-BF16 *control* keeps its exact historical
        # meaning -- those trials compare decode only, and silently adding an
        # encode compile would change their warmup, not their measurement.
        vae.encode = torch.compile(vae.encode, dynamic=False)
        report["compiled_modules"].append("vae.encode")
    return report


def set_int8_calibration(model, enabled: bool) -> int:
    """Turn observed-amax collection on or off for every INT8 linear in ``model``."""
    count = 0
    for module in model.modules():
        if isinstance(module, Int8ComputeLinear):
            module.calibrating = bool(enabled)
            if enabled:
                with torch.no_grad():
                    module.observed_amax.zero_()
            count += 1
    return count


def collect_int8_amax(model) -> dict:
    """Read back the per-site observed activation amax after a calibration run."""
    return {
        name: float(module.observed_amax.item())
        for name, module in model.named_modules()
        if isinstance(module, Int8ComputeLinear)
    }


def apply_static_int8_scales(model, amax: dict, margin: float = 1.25, only=None) -> dict:
    """Install static activation scales derived from a calibration run.

    ``margin`` is headroom over the largest activation magnitude seen during
    calibration. It is not cosmetic: the calibration fixture is one utterance, and any
    activation beyond ``margin * observed`` saturates instead of scaling, so the margin
    is the only thing standing between an out-of-distribution input and clipping. The
    default is deliberately generous -- the cost of margin is ~log2(margin) bits of
    resolution, the cost of being wrong is clipped activations inside an autoregressive
    loop.

    ``only`` optionally restricts installation to sites whose name contains one of the
    given substrings, so the wide FFN site can be converted without touching the
    narrow projections (where the second read already fits in L2 and is free).
    """
    applied = {}
    for name, module in model.named_modules():
        if not isinstance(module, Int8ComputeLinear):
            continue
        if only is not None and not any(token in name for token in only):
            continue
        observed = float(amax.get(name, 0.0))
        if observed <= 0.0:
            continue
        scale = observed * float(margin) / 127.0
        with torch.no_grad():
            module.static_input_scale = torch.tensor(
                scale, dtype=torch.float32, device=module.weight_scale.device
            )
        applied[name] = {"observed_amax": observed, "margin": margin, "scale": scale}
    return applied
