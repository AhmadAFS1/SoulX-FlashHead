"""The Neural Engine layout must equal the reference partition in FP32 and convert.

CPU-only checks (no Core ML prediction, no ANE placement or speed claim).
"""
import importlib.util
from pathlib import Path
import tempfile

import numpy as np
import pytest
import torch

from flashhead_coreml import ffn_pruning as fp
from flashhead_coreml.ane_layout import to_ane_layout
from flashhead_coreml.dit import Blocks, ProConfig, Profile

CFG = ProConfig(dim=64, ffn_dim=128, num_heads=2, num_layers=2, freq_dim=16)
PROFILE = Profile(32, 32)  # grid (9, 2, 2): 4 tokens per frame


def partition(seed=0):
    torch.manual_seed(seed)
    blocks = Blocks(CFG, PROFILE, 0, 2, torch.randn(2, 6, CFG.dim) * 0.2).eval().requires_grad_(False)
    for parameter in blocks.parameters():
        parameter.data.normal_(0, 0.2)
    return blocks


@pytest.mark.parametrize('prescale', [0.0, 64.0])
@torch.no_grad()
def test_ane_layout_equals_reference_partition(prescale):
    blocks = partition()
    hidden = torch.randn(1, PROFILE.tokens, CFG.dim) * 30  # residual-scale values
    context = torch.randn(1, 9, 32, CFG.dim)
    ane = to_ane_layout(blocks, prescale)
    for t in (1000., 625.):
        expected = blocks(hidden, context, torch.tensor([t]))
        torch.testing.assert_close(ane(hidden, context, torch.tensor([t])), expected, rtol=1e-4, atol=1e-3)


@torch.no_grad()
def test_ane_layout_keeps_reduced_ffn_and_step_bias():
    blocks = partition(1)
    for block in blocks.blocks:
        fp.reduce_block(block, {'kept': torch.arange(0, CFG.ffn_dim, 2, dtype=torch.int32),
                                'w2': torch.randn(CFG.dim, CFG.ffn_dim // 2) * 0.1,
                                'step_bias': torch.randn(2, CFG.dim)})
    hidden, context = torch.randn(1, PROFILE.tokens, CFG.dim), torch.randn(1, 9, 32, CFG.dim)
    ane = to_ane_layout(blocks, 64.0)
    for t in (1000., 625.):
        torch.testing.assert_close(ane(hidden, context, torch.tensor([t])), blocks(hidden, context, torch.tensor([t])),
                                   rtol=1e-4, atol=1e-3)


def test_ane_layout_refuses_folded_cross_attention():
    blocks = partition()
    blocks.fold_cross_output()
    with pytest.raises(ValueError):
        to_ane_layout(blocks)


@pytest.mark.skipif(importlib.util.find_spec('coremltools') is None, reason='coremltools required')
@torch.no_grad()
def test_ane_layout_converts_fp16_int8_with_conv_weights():
    from flashhead_coreml.export_dit import convert_component
    ane = to_ane_layout(partition(), 64.0)
    inputs = {'hidden': torch.zeros(1, PROFILE.tokens, CFG.dim), 'context': torch.zeros(1, 9, 32, CFG.dim),
              'timestep': torch.tensor([1000.])}
    with tempfile.TemporaryDirectory() as directory:
        entry = convert_component(ane, inputs, 'hidden_out', Path(directory) / 'blocks.mlpackage', 'int8', 'float16')
    assert entry['outputs']['hidden_out']['shape'] == [1, PROFILE.tokens, CFG.dim]
    assert entry['inputs']['hidden']['shape'] == [1, PROFILE.tokens, CFG.dim]
