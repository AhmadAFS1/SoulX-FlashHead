"""CPU checks for the A/B renderer, the exact cross-attention fold and the
calibrated FFN reduction. Synthetic shapes; no GPU, Core ML runtime or model
weights are needed (coremltools is used only for its quantizer and MIL passes).
"""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashhead_coreml import cpu_runtime as cr
from flashhead_coreml import ffn_pruning as fp
from flashhead_coreml.dit import Blocks, DiTBlock, ProConfig, Profile, rotary_tables

torch.set_num_threads(2)
CFG = ProConfig(dim=24, ffn_dim=40, num_heads=3, num_layers=2, freq_dim=16)
GRID = (9, 2, 2)


def block(seed=0):
    torch.manual_seed(seed)
    b = DiTBlock(CFG).eval().requires_grad_(False)
    for parameter in b.parameters():
        parameter.data.normal_(0, 0.2)
    return b


def inputs(seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, math.prod(GRID), CFG.dim, generator=g)
    context = torch.randn(1, 9, 32, CFG.dim, generator=g)
    modulation = torch.randn(1, 6, CFG.dim, generator=g) * 0.1
    cos, sin = rotary_tables(CFG.dim // CFG.num_heads, GRID)
    return x, context, modulation, cos, sin


@torch.no_grad()
def test_emulated_block_matches_module_in_fp32():
    b = block()
    x, context, modulation, cos, sin = inputs()
    expected = b(x, context, modulation, cos, sin)
    actual = cr.block_forward(b, x, context, modulation, cos, sin, 0, cr.identity)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_cross_attention_fold_is_exact():
    b = block()
    x, context, modulation, cos, sin = inputs()
    expected = b(x, context, modulation, cos, sin)
    b.cross_attn.fold_output_projection()
    assert not hasattr(b.cross_attn, 'o')
    torch.testing.assert_close(b(x, context, modulation, cos, sin), expected, rtol=1e-5, atol=1e-5)
    emulated = cr.block_forward(b, x, context, modulation, cos, sin, 0, cr.identity)
    torch.testing.assert_close(emulated, expected, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_folded_int8_weights_equal_original_row_quantization():
    b = block()
    original = cr.coreml_int8_dequantized(b.cross_attn.o.weight.data)
    b.cross_attn.fold_output_projection()
    cr.apply_int8_weights(b.cross_attn)
    h, d, dim = b.cross_attn.o_heads.shape
    rows = b.cross_attn.o_heads.permute(2, 0, 1).reshape(dim, h * d)
    assert torch.equal(rows, original)


@torch.no_grad()
def test_step_bias_selection_matches_runtime():
    b = block()
    b.ffn_step_bias = torch.randn(2, CFG.dim)
    x, context, modulation, cos, sin = inputs()
    for step, high in ((0, True), (1, False)):
        module = b(x, context, modulation, cos, sin, torch.tensor([[high]]))
        emulated = cr.block_forward(b, x, context, modulation, cos, sin, step, cr.identity)
        torch.testing.assert_close(emulated, module, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_blocks_partition_threads_timestep_to_step_bias():
    profile = Profile(32, 32)
    tm = torch.randn(2, 6, CFG.dim) * 0.1
    blocks = Blocks(CFG, profile, 0, 2, tm).eval()
    for parameter in blocks.parameters():
        parameter.data.normal_(0, 0.2)
    hidden = torch.randn(1, profile.tokens, CFG.dim)
    context = torch.randn(1, 9, 32, CFG.dim)
    reference = {t: blocks(hidden, context, torch.tensor([t])) for t in (1000., 625.)}
    for b in blocks.blocks:
        b.ffn_step_bias = torch.zeros(2, CFG.dim)
    for t in (1000., 625.):
        torch.testing.assert_close(blocks(hidden, context, torch.tensor([t])), reference[t])
    # One block: the bias enters as bias[row] * e5, row 0 at t=1000 and row 1 at t=625.
    single = Blocks(CFG, profile, 0, 1, tm).eval()
    single.blocks[0].load_state_dict(blocks.blocks[0].state_dict(), strict=False)
    single.blocks[0].ffn_step_bias = None
    plain = {t: single(hidden, context, torch.tensor([t])) for t in (1000., 625.)}
    bias = torch.randn(2, CFG.dim)
    single.blocks[0].ffn_step_bias = bias
    for row, (t, table) in enumerate(((1000., tm[0]), (625., tm[1]))):
        e5 = (single.blocks[0].modulation + table)[0, 5]
        torch.testing.assert_close(single(hidden, context, torch.tensor([t])) - plain[t],
                                   (bias[row] * e5).expand_as(plain[t]), rtol=1e-4, atol=1e-5)


def test_noise_matches_scalar_swift_transcription():
    def scalar(seed, count):
        state, spare, out = seed, None, []
        mask = (1 << 64) - 1
        def uniform():
            nonlocal state
            state = (state + 0x9E3779B97F4A7C15) & mask
            z = state
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
            z ^= z >> 31
            return ((z >> 11) + 0.5) / 9007199254740992.0
        for _ in range(count):
            if spare is not None:
                out.append(spare); spare = None; continue
            r, a = math.sqrt(-2 * math.log(uniform())), 2 * math.pi * uniform()
            spare = np.float32(r * math.sin(a))
            out.append(np.float32(r * math.cos(a)))
        return np.array(out, dtype=np.float32)
    noise = cr.SoulXNoise(42)
    # Odd counts exercise the cached spare value across calls.
    values = np.concatenate([noise.normals(5), noise.normals(8), noise.normals(3)])
    np.testing.assert_array_equal(values, scalar(42, 16))


def test_int8_emulation_matches_coremltools_quantizer():
    ct = pytest.importorskip('coremltools')
    import coremltools.optimize.coreml as cto
    torch.manual_seed(3)
    model = torch.nn.Sequential(torch.nn.Linear(64, 48)).eval()
    traced = torch.jit.trace(model, torch.zeros(1, 64))
    mlmodel = ct.convert(traced, convert_to='mlprogram', inputs=[ct.TensorType(name='x', shape=(1, 64), dtype=np.float16)],
                         minimum_deployment_target=ct.target.iOS18, compute_precision=ct.precision.FLOAT16,
                         skip_model_load=True)
    config = cto.OptimizationConfig(global_config=cto.OpLinearQuantizerConfig(
        mode='linear_symmetric', dtype='int8', granularity='per_channel', weight_threshold=1024))
    quantized = cto.linear_quantize_weights(mlmodel, config=config)
    ops = [op for f in quantized._mil_program.functions.values() for op in f.operations
           if op.op_type == 'constexpr_blockwise_shift_scale']
    assert len(ops) == 1
    data, scale = ops[0].inputs['data'].val, ops[0].inputs['scale'].val
    restored = (data.astype(np.float16) * scale.astype(np.float16)).astype(np.float32)
    np.testing.assert_array_equal(cr.coreml_int8_dequantized(model[0].weight.data).numpy(), restored)


def test_obs_refit_recovers_least_squares_and_means():
    torch.manual_seed(0)
    n, d, tokens = 12, 5, 400
    basis = torch.randn(tokens, 4)
    a = torch.cat([basis @ torch.randn(4, 8), torch.randn(tokens, 4)], dim=1)
    a = a + 0.01 * torch.randn(tokens, n)  # eight nearly redundant units
    steps = torch.arange(tokens) % 2
    a = a + steps[:, None] * 0.3
    w2, b2 = torch.randn(d, n), torch.randn(d)
    means = torch.stack([a[steps == s].mean(0) for s in (0, 1)]).double()
    gram = sum(((a[steps == s] - means[s].float()).T @ (a[steps == s] - means[s].float())).double() for s in (0, 1))
    order, errors = fp.obs_order(gram, w2, 4, damp=1e-8)
    assert len(set(order)) == 4 and all(e2 >= e1 for e1, e2 in zip(errors, errors[1:]))
    kept = torch.tensor(sorted(set(range(n)) - set(order)))
    w2_new, _, step_bias = fp.refit(gram, means, w2, b2, kept, damp=1e-10)
    target = a @ w2.T
    approx = a[:, kept] @ w2_new.T + step_bias[steps]
    # Least squares with per-step intercepts on the centred design.
    design = torch.cat([a[:, kept], (steps == 0).float()[:, None], (steps == 1).float()[:, None]], dim=1).double()
    solution = torch.linalg.lstsq(design, target.double()).solution
    torch.testing.assert_close(approx.double(), design @ solution, rtol=1e-3, atol=1e-3)
    # The four removed units come from the redundant group.
    assert all(j < 8 for j in order)
    # Per-timestep mean output is restored exactly.
    for s in (0, 1):
        torch.testing.assert_close(approx[steps == s].mean(0), target[steps == s].mean(0), rtol=1e-4, atol=1e-4)
    # Keeping every unit reproduces the original projection.
    full, _, zero_bias = fp.refit(gram, means, w2, b2, torch.arange(n), damp=1e-10)
    torch.testing.assert_close(full, w2, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(zero_bias, torch.zeros_like(zero_bias), rtol=0, atol=1e-3)


@torch.no_grad()
def test_variant_names_select_fitted_ratios():
    assert fp.parse_variant('ffn25') == 0.25 and fp.parse_variant('ffn12.5') == 0.125
    assert fp.fitted_ratio({'0.125': 'a', '0.375': 'b'}, fp.parse_variant('ffn37.5'), 'fit') == 'b'
    with pytest.raises(KeyError):
        fp.fitted_ratio({'0.125': 'a'}, 0.25, 'fit')


@torch.no_grad()
def test_reduce_block_keeps_shapes_and_interface():
    b = block()
    x, context, modulation, cos, sin = inputs()
    fitted = {'kept': torch.arange(0, CFG.ffn_dim, 2, dtype=torch.int32),
              'w2': torch.randn(CFG.dim, CFG.ffn_dim // 2), 'step_bias': torch.randn(2, CFG.dim)}
    assert fp.reduce_block(b, fitted) == CFG.ffn_dim // 2
    assert b.ffn[0].out_features == CFG.ffn_dim // 2 and b.ffn[2].in_features == CFG.ffn_dim // 2
    out = b(x, context, modulation, cos, sin, torch.tensor([[True]]))
    assert out.shape == x.shape and torch.isfinite(out).all()


def test_colour_round_trip_and_truncation():
    rgb = np.random.default_rng(0).random((8, 8, 3), dtype=np.float32)
    np.testing.assert_allclose(cr.rgb_from_lab(cr.lab(rgb)), rgb, atol=2e-4)
    frame = np.full((3, 2, 2), 0.9999, dtype=np.float32)
    assert cr.display_uint8(frame).max() == 254  # Swift UInt8(Float) truncates


def test_audio_window_centres():
    features = torch.arange(200, dtype=torch.float32).reshape(1, 200, 1, 1).expand(1, 200, 12, 768)
    windows = cr.audio_windows(features)
    assert windows.shape == (1, 33, 5, 12, 768)
    assert windows[0, 0, :, 0, 0].tolist() == [165, 166, 167, 168, 169]
    assert windows[0, 32, :, 0, 0].tolist() == [197, 198, 199, 199, 199]


def test_batched_obs_close_to_one_at_a_time():
    torch.manual_seed(1)
    n, d, tokens = 48, 6, 2000
    a = torch.randn(tokens, 8) @ torch.randn(8, n) + 0.05 * torch.randn(tokens, n)
    gram = (a.T @ a).double()
    means = torch.zeros(2, n, dtype=torch.float64)
    w2, b2 = torch.randn(d, n), torch.zeros(d)
    def error(order):
        kept = torch.tensor(sorted(set(range(n)) - set(order)))
        return fp.predicted_error(gram, w2, kept, fp.refit(gram, means, w2, b2, kept, damp=1e-6)[0])
    single, _ = fp.obs_order(gram, w2, 24, damp=1e-6, batch=1)
    batched, _ = fp.obs_order(gram, w2, 24, damp=1e-6, batch=4)
    assert len(set(batched)) == 24
    assert error(batched) <= 1.25 * error(single) + 1e-9
