"""Core ML conversion checks for transformer variants (runs on Linux or macOS).

Only conversion and the saved ML Program are inspected: no prediction, GPU,
Neural Engine or performance measurement is involved.
"""
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(importlib.util.find_spec('coremltools') is None, reason='coremltools required')

from flashhead_coreml import cpu_runtime as cr
from flashhead_coreml import ffn_pruning as fp
from flashhead_coreml.dit import Blocks, ProConfig, Profile
from flashhead_coreml.export_dit import convert_component

CFG = ProConfig(dim=64, ffn_dim=128, num_heads=2, num_layers=1, freq_dim=16)
PROFILE = Profile(32, 32)


def load_program(path):
    import coremltools as ct
    from coremltools.converters.mil.frontend.milproto.load import load
    spec = ct.models.MLModel(str(path), skip_model_load=True).get_spec()
    return load(model_spec=spec, specification_version=spec.specificationVersion,
                file_weights_dir=str(Path(path) / 'Data/com.apple.CoreML/weights'))


def ops(program):
    found = []
    def walk(block):
        for op in block.operations:
            found.append(op)
            for child in op.blocks:
                walk(child)
    for function in program.functions.values():
        walk(function)
    return found


@torch.no_grad()
def test_folded_reduced_partition_converts_with_fp32_softmax_and_exact_int8_operand():
    torch.manual_seed(5)
    blocks = Blocks(CFG, PROFILE, 0, 1, torch.randn(2, 6, CFG.dim) * 0.1).eval().requires_grad_(False)
    for parameter in blocks.parameters():
        parameter.data.normal_(0, 0.1)
    original_o = cr.coreml_int8_dequantized(blocks.blocks[0].cross_attn.o.weight.data)
    fp.reduce_block(blocks.blocks[0], {'kept': torch.arange(0, CFG.ffn_dim, 2, dtype=torch.int32),
                                       'w2': torch.randn(CFG.dim, CFG.ffn_dim // 2) * 0.1,
                                       'step_bias': torch.randn(2, CFG.dim) * 0.1})
    blocks.fold_cross_output()
    inputs = {'hidden': torch.zeros(1, PROFILE.tokens, CFG.dim), 'context': torch.zeros(1, 9, 32, CFG.dim),
              'timestep': torch.tensor([1000.])}
    with tempfile.TemporaryDirectory() as directory:
        destination = Path(directory) / 'blocks.mlpackage'
        entry = convert_component(blocks, inputs, 'hidden_out', destination, 'int8', 'mixed')
        assert entry['outputs']['hidden_out']['shape'] == [1, PROFILE.tokens, CFG.dim]
        program = load_program(destination)
    every = ops(program)
    softmax = [op for op in every if op.op_type == 'softmax']
    assert softmax and all(op.outputs[0].dtype.__name__ == 'fp32' for op in softmax)
    dequant = [op for op in every if op.op_type == 'constexpr_blockwise_shift_scale']
    folded = [op for op in dequant if op.inputs['data'].val.ndim == 3]
    assert len(folded) == 1
    data, scale = folded[0].inputs['data'].val, folded[0].inputs['scale'].val
    assert scale.shape == (1, 1, CFG.dim)
    restored = (data.astype(np.float16) * scale.astype(np.float16)).astype(np.float32)
    rows = torch.from_numpy(restored).permute(2, 0, 1).reshape(CFG.dim, CFG.dim)
    assert torch.equal(rows, original_o)
    # Reduced FFN projections are compressed like every other learned matrix.
    shapes = sorted(tuple(op.inputs['data'].val.shape) for op in dequant if op.inputs['data'].val.ndim == 2)
    assert (CFG.ffn_dim // 2, CFG.dim) in shapes and (CFG.dim, CFG.ffn_dim // 2) in shapes
