"""Opt-in PRO FFN compute quantization, pinned/tested against Torch 2.7.1 CUDA.

These experimental modules use real FP8/INT8 GEMMs, with BF16 interfaces.
They are deliberately not selected by the production Engine. Apply after the
ordinary checkpoint load/device/dtype conversion and before compilation.
"""
import torch
from torch import nn


class Float8Linear(nn.Module):
    """Tensorwise E4M3 weights and dynamically scaled E4M3 activations."""

    def __init__(self, linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
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
            out_dtype=x.dtype, use_fast_accum=False,
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
        self.register_buffer("weight_int8_t", quantized.t().contiguous())
        self.register_buffer("weight_scale", scale.float())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, self.in_features)
        scale = flat.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-12) / 127.0
        quantized = (flat.float() / scale).round().clamp(-127, 127).to(torch.int8)
        output = torch._int_mm(quantized, self.weight_int8_t).float()
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
    if mode not in ("none", "channels_last", "pointwise", "compiled"):
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
    if mode == "compiled":
        vae.decode = torch.compile(vae.decode, dynamic=False)
        report["compiled_modules"].append("vae.decode")
    return report
