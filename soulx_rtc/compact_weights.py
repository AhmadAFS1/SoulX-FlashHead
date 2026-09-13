"""Opt-in INT8 weight storage with BF16/FP32 computation, not integer GEMM.

Per-output-channel symmetric quantization reduces resident DiT weights. The
dequantized matrix lives for one linear call. Keep normalization, conditioning
outside blocks, and the output head at the reference precision.
"""
import torch
from torch import nn
from torch.nn import functional as F


class Int8StorageLinear(nn.Module):
    def __init__(self, linear):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        with torch.no_grad():
            weight = linear.weight.detach().float()
            scale = weight.abs().amax(dim=1, keepdim=True) / 127
            scale = torch.where(scale > 0, scale, torch.ones_like(scale))
            # Use the stored scale while rounding, including BF16 rounding error.
            scale = scale.to(linear.weight.dtype)
            quantized = (weight / scale.float()).round().clamp(-127, 127).to(torch.int8)
        self.register_buffer('weight_int8', quantized)
        self.register_buffer('scale', scale)
        self.register_buffer('bias', None if linear.bias is None else linear.bias.detach().clone())

    def forward(self, x):
        weight = self.weight_int8.to(x.dtype) * self.scale.to(x.dtype)
        bias = None if self.bias is None else self.bias.to(x.dtype)
        return F.linear(x, weight, bias)


def quantize_block_linears(blocks):
    """Replace only block linears after strict checkpoint loading; idempotent."""
    report = {'layers': 0, 'original_weight_bytes': 0, 'stored_weight_bytes': 0}
    def visit(parent):
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear):
                converted = Int8StorageLinear(child)
                report['layers'] += 1
                report['original_weight_bytes'] += child.weight.numel() * child.weight.element_size()
                report['stored_weight_bytes'] += sum(t.numel() * t.element_size()
                                                    for t in (converted.weight_int8, converted.scale))
                setattr(parent, name, converted)
            else:
                visit(child)
    for block in blocks:
        visit(block)
    return report
