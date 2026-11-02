"""CPU parity against upstream support networks, with downloaded real weights.

These are correctness checks, not GPU or iPhone performance measurements.
"""
import sys
from pathlib import Path
import pytest
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flashhead_coreml.export_support import ROOT, Audio, TinyDecode, TinyEncode, Reference, module_at

torch.set_num_threads(4)

@pytest.fixture(scope='module')
def tiny():
    path = ROOT/'models/tiny_vae/taehv'
    if not (path/'taew2_1.pth').exists(): pytest.skip('Run flashhead_coreml.fetch_models first')
    vendor = module_at('soulx_test_taehv', path/'taehv.py')
    return vendor, vendor.TAEHV(str(path/'taew2_1.pth'), arch_name='taew2_1').float().eval()

@torch.no_grad()
def test_tiny_decoder_real_weights(tiny):
    vendor, model = tiny
    latent = torch.randn(1,9,16,8,4)
    expected = model.decode_video(latent, show_progress_bar=False)
    torch.testing.assert_close(TinyDecode(model,vendor.MemBlock)(latent), expected, rtol=0, atol=0)

@torch.no_grad()
def test_tiny_encoder_tail_padding_real_weights(tiny):
    vendor, model = tiny
    frames = torch.rand(1,5,3,64,32)
    expected = model.encode_video(frames, show_progress_bar=False).permute(0,2,1,3,4)
    torch.testing.assert_close(TinyEncode(model,vendor.MemBlock)(frames), expected, rtol=0, atol=0)

@torch.no_grad()
def test_audio_against_transformers_layer_outputs():
    path = ROOT/'models/wav2vec2-base-960h'
    if not (path/'model.safetensors').exists(): pytest.skip('Run flashhead_coreml.fetch_models first')
    from transformers import Wav2Vec2Model
    model = Wav2Vec2Model.from_pretrained(str(path), attn_implementation='eager').eval()
    waveform = torch.randn(1,128000)*0.1
    x = (waveform-waveform.mean(-1,keepdim=True))/torch.sqrt(waveform.var(-1,unbiased=False,keepdim=True)+1e-7)
    x = F.interpolate(model.feature_extractor(x),size=200,mode='linear',align_corners=True).transpose(1,2)
    x = model.feature_projection(x)[0]
    states=[]
    hooks=[layer.register_forward_hook(lambda m,a,o: states.append(o[0] if isinstance(o,tuple) else o)) for layer in model.encoder.layers]
    model.encoder(x)
    for hook in hooks: hook.remove()
    expected = torch.stack(states, dim=2)
    torch.testing.assert_close(Audio(model)(waveform), expected, rtol=2e-4, atol=3e-5)

@torch.no_grad()
def test_reference_real_wan_causal_encode():
    path=ROOT/'models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth'
    if not path.exists(): pytest.skip('Run flashhead_coreml.fetch_models first')
    vendor=module_at('soulx_test_wan', ROOT/'flash_head/wan/modules/vae.py')
    vae=vendor.WanVAE(vae_path=str(path),dtype=torch.float32,device='cpu')
    image=torch.rand(1,3,1,32,32)*2-1
    expected=vae.model.encode(image.repeat(1,1,33,1,1),vae.scale)
    actual=Reference(vae,vendor.count_conv3d)(image)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
