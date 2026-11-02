"""CPU-only parity tests against the repository's original PRO implementation.

GPU: none; no GPU or iPhone inference timings. CUDA/distributed imports are
stubbed solely for loading the upstream module in these CPU tests.
"""
import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import torch

from flashhead_coreml.dit import (ProConfig, Profile, DiTBlock, Patch, AudioProjection,
                                  OutputHead, rotary_tables, timestep_embedding,
                                  attention)


def load_original():
    names = ['xfuser', 'xfuser.core', 'xfuser.core.distributed', 'xfuser.core.long_ctx_attention']
    stubs = {name: types.ModuleType(name) for name in names}
    dist = stubs['xfuser.core.distributed']
    dist.get_sequence_parallel_rank = lambda: 0
    dist.get_sequence_parallel_world_size = lambda: 1
    dist.get_sp_group = lambda: None
    stubs['xfuser.core.long_ctx_attention'].xFuserLongContextAttention = object
    filename = Path(__file__).resolve().parents[2] / 'flash_head/src/modules/flash_head_model.py'
    spec = importlib.util.spec_from_file_location('original_flashhead_for_test', filename)
    module = importlib.util.module_from_spec(spec)
    previous = {name: sys.modules.get(name) for name in names}
    sys.modules.update(stubs)
    try:
        spec.loader.exec_module(module)
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module


class PortableDiTTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.original = load_original()
        cls.cfg = ProConfig(dim=24, ffn_dim=40, num_heads=3, num_layers=2, freq_dim=16)
        cls.profile = Profile(32, 32)

    def test_block_matches_original_including_rope_audio_and_modulation(self):
        torch.manual_seed(17)
        c = self.cfg
        original = self.original.DiTAudioBlock(False, c.dim, c.num_heads, c.ffn_dim, c.eps).eval()
        portable = DiTBlock(c).eval()
        portable.load_state_dict(original.state_dict())
        x = torch.randn(1, self.profile.tokens, c.dim)
        audio = torch.randn(1, 9, 32, c.dim)
        modulation = torch.randn(1, 6, c.dim)
        freq = self.original.precompute_freqs_cis_3d(c.dim // c.num_heads)
        rotary = self.original.prepare_rotary(freq, self.profile.grid, real=True)
        cos, sin = rotary_tables(c.dim // c.num_heads, self.profile.grid)
        with torch.no_grad():
            expected = original(x, audio, modulation, freq, self.profile.grid, rotary=rotary)
            actual = portable(x, audio, modulation, cos, sin)
            portable.self_attn.fuse_projections()
            portable.cross_attn.fuse_projections()
            fused = portable(x, audio, modulation, cos, sin)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(fused, expected, atol=2e-6, rtol=2e-5)

    def test_query_tiling_preserves_all_keys(self):
        torch.manual_seed(18)
        q = torch.randn(2, 17, 24)
        k, v = torch.randn(2, 29, 24), torch.randn(2, 29, 24)
        torch.testing.assert_close(attention(q, k, v, 3, query_chunk=5), attention(q, k, v, 3), atol=5e-7, rtol=1e-5)

    def test_patch_conv2d_matches_original_conv3d_and_token_order(self):
        torch.manual_seed(19)
        portable = Patch(self.cfg, self.profile)
        conv = torch.nn.Conv3d(32, 24, (1, 2, 2), stride=(1, 2, 2))
        conv.weight.data.copy_(portable.patch_embedding.weight.unsqueeze(2))
        conv.bias.data.copy_(portable.patch_embedding.bias)
        x, y = torch.randn(self.profile.latent_shape), torch.randn(self.profile.latent_shape)
        expected = conv(torch.cat((x, y), dim=1)).permute(0, 2, 3, 4, 1).reshape(1, -1, 24)
        torch.testing.assert_close(portable(x, y), expected)

    def test_audio_projection_matches_original_window_packing(self):
        torch.manual_seed(20)
        c, d, layers = 8, 24, 3
        portable = AudioProjection(d, intermediate=16, channels=c, layers=layers)
        original = self.original.AudioProjModel(seq_len=5, seq_len_vf=8, blocks=layers, channels=c,
                                              intermediate_dim=16, output_dim=d, context_tokens=32, norm_output_audio=True)
        original.load_state_dict(portable.state_dict())
        audio = torch.randn(1, 33, 5, layers, c)
        groups = audio[:, 1:].reshape(1, 8, 4, 5, layers, c)
        selected = torch.stack([torch.cat((group[0, :3], group[1, 2:3], group[2, 2:3], group[3, 2:]), dim=0)
                                for group in groups[0]], dim=0)[None]
        with torch.no_grad():
            expected = original(audio[:, :1], selected)
            actual = portable(audio)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)

    def test_head_and_unpatchification_match_original(self):
        torch.manual_seed(21)
        c = self.cfg
        times = torch.randn(2, c.dim)
        portable = OutputHead(c, self.profile, times)
        original = self.original.Head(c.dim, 16, (1, 2, 2), c.eps)
        original.load_state_dict(portable.state_dict())
        hidden = torch.randn(1, self.profile.tokens, c.dim)
        for step, index in [(1000., 0), (625., 1)]:
            out = original(hidden, times[index:index+1])
            expected = out.reshape(1, 9, 2, 2, 1, 2, 2, 16).permute(0, 7, 1, 4, 2, 5, 3, 6).reshape(1, 16, 9, 4, 4)
            torch.testing.assert_close(portable(hidden, torch.tensor([step])), expected)

    def test_timestep_matches_original(self):
        t = torch.tensor([1000., 625.])
        torch.testing.assert_close(timestep_embedding(t, 256), self.original.sinusoidal_embedding_1d(256, t))


if __name__ == '__main__':
    unittest.main()
