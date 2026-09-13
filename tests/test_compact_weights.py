import torch
from torch import nn
from soulx_rtc.compact_weights import Int8StorageLinear, quantize_block_linears


def test_quantization_error_is_bounded_and_zero_rows_remain_exact():
    torch.manual_seed(37)
    original = nn.Linear(128, 64)
    with torch.no_grad():
        original.weight[0].zero_()
    candidate = Int8StorageLinear(original)
    x = torch.randn(2, 7, 128)
    expected, actual = original(x), candidate(x)
    bound = x.abs().sum(-1, keepdim=True) * candidate.scale.T / 2 + 1e-6
    assert torch.all((actual - expected).abs() <= bound)
    assert torch.equal(actual[..., 0], expected[..., 0])
    assert torch.isfinite(actual).all()
    assert candidate.weight_int8.dtype == torch.int8


def test_nested_conversion_preserves_norm_and_head_and_is_idempotent():
    blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(32), nn.Linear(32, 96), nn.GELU(), nn.Linear(96, 32, bias=False))])
    head = nn.Linear(32, 4)
    head_before = head.weight.detach().clone()
    report = quantize_block_linears(blocks)
    assert report['layers'] == 2
    assert report['stored_weight_bytes'] < report['original_weight_bytes'] / 2
    assert isinstance(blocks[0][0], nn.LayerNorm)
    assert torch.equal(head.weight, head_before)
    assert quantize_block_linears(blocks)['layers'] == 0
    assert blocks[0](torch.randn(2, 32)).shape == (2, 32)


def test_bf16_storage_and_serialization():
    original = nn.Linear(32, 64, bias=False).to(torch.bfloat16)
    candidate = Int8StorageLinear(original)
    clone = Int8StorageLinear(original)
    clone.load_state_dict(candidate.state_dict())
    x = torch.randn(2, 32, dtype=torch.bfloat16)
    assert torch.equal(candidate(x), clone(x))
    assert candidate(x).dtype == torch.bfloat16
    assert torch.isfinite(candidate(x)).all()
