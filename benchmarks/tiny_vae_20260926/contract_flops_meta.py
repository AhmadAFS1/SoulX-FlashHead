"""Static FLOP count (meta device, no weights, no GPU) of the VAE candidates at 576x320."""
import sys, torch
from torch.utils.flop_counter import FlopCounterMode
sys.path.insert(0, '/workspace/SoulX-FlashHead')
sys.path.insert(0, '/workspace/SoulX-FlashHead/models/tiny_vae/taehv')
from flash_head.wan.modules.vae import WanVAE_, count_conv3d
from soulx_rtc.pro_decoder_ops import SkippedResidualBlock
import taehv as T

LH, LW = 72, 40
CFG = dict(z_dim=16, dim_mult=[1, 2, 4, 4], num_res_blocks=2, attn_scales=[],
           temperal_downsample=[False, True, True], dropout=0.0)

def flops(fn):
    fc = FlopCounterMode(display=False)
    with fc, torch.no_grad():
        out = fn()
    return fc.get_total_flops() / 1e12, out

def wan(dim, skip=()):
    with torch.device('meta'):
        m = WanVAE_(dim=dim, **CFG).eval().requires_grad_(False)
    for i in skip:
        m.decoder.upsamples[i] = SkippedResidualBlock(i, 2, m.decoder.upsamples[i])
    return m

def wan_decode(m, n_lat, warm):
    """warm=True: steady-state window (cache already holds 2-frame entries) -> decode n_lat latents
    after priming with 2 latents, counting only the n_lat."""
    m.clear_cache()
    z = torch.randn(1, 16, 2 + n_lat if warm else n_lat, LH, LW, device='meta')
    x = m.conv2(z)
    total, frames = 0.0, 0
    for i in range(x.shape[2]):
        m._conv_idx = [0]
        f, out = flops(lambda: m.decoder(x[:, :, i:i + 1], feat_cache=m._feat_map, feat_idx=m._conv_idx))
        if not warm or i >= 2:
            total += f; frames += out.shape[2]
    return total, frames

def wan_encode(m, t):
    x = torch.randn(1, 3, t, 8 * LH, 8 * LW, device='meta')
    f, mu = flops(lambda: m.encode(x, [0.0, 1.0]))
    return f, tuple(mu.shape)

torch.set_grad_enabled(False)
rows = []
for name, dim, skip in (("Wan2.1 full (dim 96)", 96, ()), ("Wan2.1 shipping pruned 9,10,13,14", 96, (9, 10, 13, 14)),
                        ("LightVAE lightvaew2_1 (dim 24)", 24, ())):
    m = wan(dim, skip)
    d9, f9 = wan_decode(m, 9, warm=False)
    d7, f7 = wan_decode(m, 7, warm=True)
    e5, s5 = wan_encode(m, 5)
    rows.append((name, d9, f9, d7, f7, e5, s5))

tae = T.TAEHV(checkpoint_path=None)
tae = tae.to("meta").eval().requires_grad_(False)
z9 = torch.randn(1, 9, 16, LH, LW, device='meta')
d9, out9 = flops(lambda: tae.decode_video(z9, parallel=True, show_progress_bar=False))
z7 = torch.randn(1, 7, 16, LH, LW, device='meta')
d7raw, out7 = flops(lambda: T.apply_model_with_memblocks_parallel(tae.decoder, z7, False))
x8 = torch.randn(1, 8, 3, 8 * LH, 8 * LW, device='meta')
e8, lat = flops(lambda: tae.encode_video(x8, parallel=True, show_progress_bar=False))
rows.append(("TAEHV taew2_1 / lighttaew2_1 arch", d9, out9.shape[1], d7raw, out7.shape[1], e8, tuple(lat.shape)))

print(f"{'model':38s} {'dec 9 lat cold':>15s} {'frames':>6s} {'dec 7 lat warm':>15s} {'frames':>6s} {'enc motion':>11s}  latent")
for name, d9, f9, d7, f7, e5, s5 in rows:
    print(f"{name:38s} {d9:12.3f} TF {f9:6d} {d7:12.3f} TF {f7:6d} {e5:8.3f} TF  {s5}")

# Checkpoint-layout compatibility (headers only, no weights executed).
import json as _json, struct as _struct, hashlib as _hl, os as _os
def _header(f):
    with open(f, 'rb') as fh:
        n = _struct.unpack('<Q', fh.read(8))[0]; h = _json.loads(fh.read(n))
    h.pop('__metadata__', None); return {k: tuple(v['shape']) for k, v in h.items()}
R = '/workspace/SoulX-FlashHead/'
light = _header(R + 'models/tiny_vae/lightx2v/lightvaew2_1.safetensors')
with torch.device('meta'):
    m24 = WanVAE_(dim=24, **CFG)
sd24 = {k: tuple(v.shape) for k, v in m24.state_dict().items()}
ltae = _header(R + 'models/tiny_vae/lightx2v/lighttaew2_1.safetensors')
tae_sd = torch.load(R + 'models/tiny_vae/taehv/taew2_1.pth', map_location='cpu', weights_only=True)
compat = {
    "lightvaew2_1_equals_WanVAE__dim24": set(sd24) == set(light) and all(sd24[k] == light[k] for k in light),
    "lighttaew2_1_equals_taew2_1_layout": set(ltae) == set(tae_sd) and all(ltae[k] == tuple(tae_sd[k].shape) for k in ltae),
}
out = {
    "execution": "CPU-only static analysis: torch FlopCounterMode on meta tensors (no weights, no GPU); not a timing",
    "resolution": [576, 320], "latent_hw": [LH, LW],
    "columns": ["decode_9_latents_cold_TFLOP", "frames", "decode_7_latents_warm_TFLOP", "frames", "motion_encode_TFLOP", "latent_shape"],
    "rows": {name: [round(d9, 3), int(f9), round(d7, 3), int(f7), round(e5, 3), list(s5)] for name, d9, f9, d7, f7, e5, s5 in rows},
    "checkpoint_layout": compat,
}
print(_json.dumps(compat))
dest = _os.environ.get("FLOPS_JSON")
if dest:
    with open(dest, 'w') as fh:
        _json.dump(out, fh, indent=2)
