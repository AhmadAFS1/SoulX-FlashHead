import numpy as np
import pytest

from soulx_rtc.engine import Engine, GenerationState, validate_geometry


def test_native_geometry():
    assert validate_geometry(width=480, height=832) == (480, 832)
    assert validate_geometry(width=576, height=1024) == (576, 1024)
    assert validate_geometry(width=288, height=512) == (288, 512)
    for kwargs in ({"width": 480}, {"width": 480, "height": 853},
                   {"width": 1024, "height": 1024}, {"size": 0}):
        with pytest.raises(ValueError):
            validate_geometry(**kwargs)


def test_prepared_rotary_preserves_complex_baseline():
    import torch
    from flash_head.src.modules.flash_head_model import (
        apply_prepared_rotary, prepare_rotary, precompute_freqs_cis_3d, rope_apply)
    torch.manual_seed(32)
    for grid in ((5, 2, 3), (5, 3, 2)):
        x = torch.randn(2, np.prod(grid), 4, 24)
        freqs = precompute_freqs_cis_3d(24)
        expected = rope_apply(x, freqs, grid)
        torch.testing.assert_close(apply_prepared_rotary(x, prepare_rotary(freqs, grid)),
                                   expected, rtol=0, atol=0)
        torch.testing.assert_close(apply_prepared_rotary(x, prepare_rotary(freqs, grid, True)),
                                   expected, rtol=1e-5, atol=1e-6)


def test_cached_color_and_temporal_tiles():
    import torch
    from flash_head.utils.utils import match_and_blend_colors_torch, prepare_reference_color_stats
    torch.manual_seed(13)
    source, reference = torch.rand(2, 3, 9, 32, 48)*2-1, torch.rand(2, 3, 1, 32, 48)*2-1
    stats = prepare_reference_color_stats(reference)
    expected = match_and_blend_colors_torch(source, reference, 1.)
    cached = match_and_blend_colors_torch(source, reference, 1., stats)
    tiled = torch.cat([match_and_blend_colors_torch(source[:, :, i:i+4], reference, 1., stats)
                       for i in range(0, 9, 4)], dim=2)
    torch.testing.assert_close(cached, expected, rtol=0, atol=0)
    torch.testing.assert_close(tiled, expected, rtol=0, atol=0)


def test_cached_cross_attention_matches_batch_and_separate(monkeypatch):
    import torch
    import flash_head.src.modules.flash_head_model as model
    attention = model.flash_attention
    monkeypatch.setattr(model, "flash_attention", lambda q,k,v,num_heads: attention(q,k,v,num_heads,True))
    torch.manual_seed(17)
    block = model.DiTAudioBlock(False, 96, 4, 192).eval()
    x, ctx, t = torch.randn(2, 30, 96), torch.randn(2, 5, 32, 96), torch.randn(2, 6, 96)
    freq = model.precompute_freqs_cis_3d(24)
    grid = (5, 2, 3)
    with torch.no_grad():
        kv = block.cross_attn.prepare_kv(ctx.flatten(0, 1))
        cached = block(x, ctx, t, freq, grid, kv, model.prepare_rotary(freq, grid))
        baseline = block(x, ctx, t, freq, grid)
        separate = torch.cat([block(x[i:i+1], ctx[i:i+1], t[i:i+1], freq, grid)
                              for i in range(2)])
    torch.testing.assert_close(cached, baseline, rtol=0, atol=0)
    torch.testing.assert_close(cached, separate, rtol=1e-4, atol=2e-5)


def test_append_preserves_rolling_audio_and_chunk_alignment():
    engine = object.__new__(Engine)
    engine.fps = 25
    state = GenerationState(None, np.ones(12000, np.float32), 24, 24)
    assert engine.append(state, np.full(16000, 2, np.float32)) == 72
    assert np.all(state.audio[:12000] == 1)
    assert np.all(state.audio[12000:15360] == 0)
    assert np.all(state.audio[15360:] == 2)
    with pytest.raises(ValueError):
        engine.append(state, np.ones(100, np.float32))


def test_full_small_dit_cached_matches_uncached(monkeypatch):
    import torch
    import flash_head.src.modules.flash_head_model as model
    attention = model.flash_attention
    monkeypatch.setattr(model,"flash_attention",lambda q,k,v,num_heads:attention(q,k,v,num_heads,True))
    class TinyAudio(torch.nn.Module):
        def __init__(self,**kwargs):
            super().__init__()
            self.proj=torch.nn.Linear(768,96)
        def forward(self,first,latter):
            x=torch.cat((first.mean((2,3)),latter.mean((2,3))),1)
            return self.proj(x).unsqueeze(2).expand(-1,-1,32,-1).contiguous()
    monkeypatch.setattr(model,"AudioProjModel",TinyAudio)
    torch.manual_seed(101)
    net=model.WanModelAudioProject(dim=96,in_dim=8,ffn_dim=192,out_dim=4,text_dim=16,
        freq_dim=16,eps=1e-6,vae_stride=(8,32,32),patch_size=(1,1,1),num_heads=4,
        num_layers=2,has_image_input=False).eval()
    x,y=torch.randn(2,4,5,2,3),torch.randn(2,4,5,2,3)
    ctx=torch.randn(2,33,5,12,768)
    t=torch.tensor([750.,500.])
    with torch.no_grad():
        baseline=net(x,t,ctx,y)
        projected,kv=net.prepare_conditioning(ctx)
        actual=net(x,t,ctx,y,prepared_context=projected,cross_kv=kv,
                   rotary=model.prepare_rotary(net.freqs,(5,2,3)),prepared_time=net.prepare_time(t))
    torch.testing.assert_close(actual,baseline,rtol=0,atol=0)
