"""CPU checks for the tiny-VAE harness integration (soulx_rtc/pro_tiny_vae.py, run.py --tiny-vae-*).

No GPU: flag validation, catalog resolution, and the carried-state TAEHV decode core on random
weights at a 4x4 latent (window-by-window decoding with carried MemBlock state must equal one
full-sequence decode -- the property the "stream" decode mode relies on).
"""
from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest
import torch

from soulx_rtc import pro_tiny_vae as ptv

ROOT = Path(__file__).resolve().parents[1]
TINY_POLICY = json.loads((ROOT / "benchmarks/tiny_vae_20260926/policies/tiny_v4_flash2.json").read_text())
SHIP_POLICY = json.loads((ROOT / "benchmarks/pro_30fps_20260920/policies/final_v4.json").read_text())


def _args(**overrides):
    base = dict(tiny_vae_decoder=None, tiny_vae_encoder=None, tiny_vae_backend=None, tiny_vae_decode_mode=None,
                sessions=1, force_scheduler=False, capture_manifest=None, overlap_skip=False,
                skip_decoder_blocks=None, subpixel_resample=False, channels_last_head=False, vae_weights=None,
                profile_component=None, latent_feedback="off", compile_vae_encode=False)
    base.update(overrides)
    return Namespace(**base)


def test_default_path_is_a_no_op():
    ptv.validate_tiny_vae_args(_args(), SHIP_POLICY)  # nothing requested, shipping policy: fine
    assert not ptv.tiny_vae_requested(_args())


def test_tiny_knobs_alone_are_refused():
    with pytest.raises(ValueError, match="need --tiny-vae-decoder"):
        ptv.validate_tiny_vae_args(_args(tiny_vae_backend="compile"), SHIP_POLICY)
    with pytest.raises(ValueError, match="need --tiny-vae-decoder"):
        ptv.validate_tiny_vae_args(_args(tiny_vae_decode_mode="window"), SHIP_POLICY)


def test_decoder_arm_valid_combination():
    ptv.validate_tiny_vae_args(_args(tiny_vae_decoder="taew2_1", compile_vae_encode=True), TINY_POLICY)
    ptv.validate_tiny_vae_args(_args(tiny_vae_decoder="taew2_1", tiny_vae_encoder="taew2_1",
                                     tiny_vae_backend="tensorrt"), TINY_POLICY)


@pytest.mark.parametrize("override, message", [
    (dict(overlap_skip=True), "--overlap-skip"),
    (dict(skip_decoder_blocks=[9, 10]), "--skip-decoder-blocks"),
    (dict(subpixel_resample=True), "--subpixel-resample"),
    (dict(channels_last_head=True), "--channels-last-head"),
    (dict(vae_weights="x.pth"), "--vae-weights"),
    (dict(sessions=2), "--sessions"),
    (dict(force_scheduler=True), "--force-scheduler"),
    (dict(capture_manifest="m.json"), "--capture-manifest"),
    (dict(profile_component="decoder"), "--profile-component decoder"),
])
def test_decoder_refusals(override, message):
    with pytest.raises(ValueError, match=message):
        ptv.validate_tiny_vae_args(_args(tiny_vae_decoder="taew2_1", **override), TINY_POLICY)


def test_decoder_refuses_wan_tensorrt_policy():
    with pytest.raises(ValueError, match="backend: pytorch, scheme: bf16"):
        ptv.validate_tiny_vae_args(_args(tiny_vae_decoder="taew2_1"), SHIP_POLICY)


def test_encoder_refusals_and_shipping_decoder_allowed():
    # encoder-only arm on the shipping Wan decoder (TensorRT policy + overlap-skip) is allowed
    ptv.validate_tiny_vae_args(_args(tiny_vae_encoder="taew2_1", overlap_skip=True, skip_decoder_blocks=[9]), SHIP_POLICY)
    with pytest.raises(ValueError, match="--compile-vae-encode"):
        ptv.validate_tiny_vae_args(_args(tiny_vae_encoder="taew2_1", compile_vae_encode=True), SHIP_POLICY)
    with pytest.raises(ValueError, match="--latent-feedback"):
        ptv.validate_tiny_vae_args(_args(tiny_vae_encoder="taew2_1", latent_feedback="last2-fix0"), SHIP_POLICY)


def test_tensorrt_backend_only_for_taehv():
    with pytest.raises(ValueError, match="TAEHV models only"):
        ptv.validate_tiny_vae_args(_args(tiny_vae_decoder="lightvaew2_1", tiny_vae_backend="tensorrt"), TINY_POLICY)


def test_resolution_names_paths_and_conventions():
    info = ptv.resolve_tiny_vae("taew2_1")
    assert info["arch"] == "taehv" and info["decoder_latent_convention"] == "dit_normalised"
    assert len(info["sha256"]) == 64
    light = ptv.resolve_tiny_vae("models/tiny_vae/lightx2v/lighttaew2_1.safetensors")
    assert light["name"] == "lighttaew2_1" and light["decoder_latent_convention"] == "wan_raw"
    named = ptv.resolve_tiny_vae("taew2_1:models/tiny_vae/taehv/taew2_1.pth")
    assert named["sha256"] == info["sha256"]
    assert ptv.resolve_tiny_vae("lightvaew2_1")["arch"] == "wan_dim24"
    with pytest.raises(ValueError, match="Unknown tiny VAE"):
        ptv.resolve_tiny_vae("taesd")
    with pytest.raises(FileNotFoundError):
        ptv.resolve_tiny_vae("taew2_1:/nonexistent/x.pth")


def test_run_py_default_sources_unchanged():
    from benchmarks.pro_quantization_v2_20260918 import run as harness

    assert not any("pro_tiny_vae" in str(p) or "taehv" in str(p) for p in harness._sources())


def test_taehv_core_carried_state_equals_full_sequence():
    torch.manual_seed(0)
    taehv = ptv._import_taehv()
    tae = taehv.TAEHV(checkpoint_path=None, arch_name="taew2_1").eval()
    core = ptv.TAEHVDecodeCore(tae, taehv.MemBlock)
    z = torch.randn(16, 16, 4, 4)  # 16 latents at 4x4
    with torch.no_grad():
        full = core(z, *core.zero_states(4, 4, "cpu", torch.float32))[0]
        a, *state = core(z[:9], *core.zero_states(4, 4, "cpu", torch.float32))
        b = core(z[9:], *[s.clone() for s in state])[0]
        upstream = tae.decode_video(z.unsqueeze(0), parallel=True, show_progress_bar=False)[0]
    assert full.shape[0] == 64 and a.shape[0] == 36 and b.shape[0] == 28
    assert torch.allclose(torch.cat([a, b]), full, atol=1e-5)
    # cold decode trimmed by frames_to_trim (3) is upstream decode_video
    assert torch.allclose(full[core.trim:], upstream, atol=1e-5)


def test_taehv_encode_core_matches_upstream():
    torch.manual_seed(0)
    taehv = ptv._import_taehv()
    tae = taehv.TAEHV(checkpoint_path=None, arch_name="taew2_1").eval()
    core = ptv.TAEHVEncodeCore(tae, taehv.MemBlock)
    x = torch.rand(5, 3, 32, 32)
    padded = torch.cat([x, x[-1:].expand(3, -1, -1, -1)])  # append copies of the last frame
    with torch.no_grad():
        ours = core(padded)
        upstream = tae.encode_video(x.unsqueeze(0), parallel=True, show_progress_bar=False)[0]
    assert ours.shape == (2, 16, 4, 4)
    assert torch.allclose(ours, upstream, atol=1e-5)
