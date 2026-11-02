"""Apple-only audio, original Wan still encoder and Wan-compatible tiny VAE exports.

All networks use real checkpoint weights. Runtime is independent of CUDA.
"""
from pathlib import Path
import argparse, gc, importlib.util, json, sys, time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import coremltools as ct

ROOT = Path(__file__).resolve().parents[2]

def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

class Audio(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, waveform):
        x = (waveform - waveform.mean(-1, keepdim=True)) / torch.sqrt(waveform.var(-1, unbiased=False, keepdim=True) + 1e-7)
        x = self.model.feature_extractor(x)
        x = F.interpolate(x, size=200, mode='linear', align_corners=True).transpose(1, 2)
        x = self.model.feature_projection(x)[0]
        encoder = self.model.encoder
        x = encoder.layer_norm(x + encoder.pos_conv_embed(x))
        states = []
        for layer in encoder.layers:
            # Explicit eval-mode transformer traversal avoids Transformers 5's
            # hook-based hidden-state collector (not traceable).
            attn = layer.attention
            batch, length, width = (int(d) for d in x.shape)
            heads = attn.num_heads
            q = attn.q_proj(x).reshape(batch,length,heads,width//heads).transpose(1,2)
            k = attn.k_proj(x).reshape(batch,length,heads,width//heads).transpose(1,2)
            v = attn.v_proj(x).reshape(batch,length,heads,width//heads).transpose(1,2)
            y = F.scaled_dot_product_attention(q,k,v).transpose(1,2).reshape(batch,length,width)
            x = layer.layer_norm(x + attn.out_proj(y))
            x = layer.final_layer_norm(x + layer.feed_forward(x))
            states.append(x)
        return torch.stack(states, dim=2)

class TinyDecode(nn.Module):
    def __init__(self, tae, memtype):
        super().__init__()
        self.decoder = tae.decoder
        self.memtype = memtype
        self.trim = int(tae.frames_to_trim)
    def forward(self, latent):
        x = latent[0]
        for block in self.decoder:
            if isinstance(block, self.memtype):
                x = block(x, torch.cat((torch.zeros_like(x[:1]), x[:-1]), dim=0))
            else:
                x = block(x)
        return x[self.trim:].clamp(0, 1).unsqueeze(0)

class TinyEncode(nn.Module):
    def __init__(self, tae, memtype):
        super().__init__()
        self.encoder = tae.encoder
        self.memtype = memtype
    def forward(self, frames):
        x = frames[0]
        x = torch.cat((x, x[-1:].expand(3, -1, -1, -1)), dim=0)
        for block in self.encoder:
            if isinstance(block, self.memtype):
                x = block(x, torch.cat((torch.zeros_like(x[:1]), x[:-1]), dim=0))
            else:
                x = block(x)
        return x.permute(1, 0, 2, 3).unsqueeze(0)

class Reference(nn.Module):
    def __init__(self, vae, count):
        super().__init__()
        self.encoder = vae.model.encoder
        self.projection = vae.model.conv1
        self.cache_count = count(self.encoder)
        self.register_buffer('mean', vae.mean.reshape(1,16,1,1,1))
        self.register_buffer('inv_std', vae.inv_std.reshape(1,16,1,1,1))
    def forward(self, image):
        # Original Wan reference is 33 copies of the still, with causal feature
        # caches shared over the first frame and eight following four-frame groups.
        cache = [None] * self.cache_count
        outputs = []
        for i in range(9):
            frames = image if i == 0 else image.repeat(1,1,4,1,1)
            outputs.append(self.encoder(frames, feat_cache=cache, feat_idx=[0]))
        mean = self.projection(torch.cat(outputs, dim=2))[:, :16]
        return (mean - self.mean) * self.inv_std


def export(module, example, input_name, output_name, target, fp32_input=False,
           compute_precision=ct.precision.FLOAT16):
    module = module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    started = time.time()
    with torch.no_grad():
        traced = module if isinstance(module, torch.jit.ScriptModule) else torch.jit.trace(module, example, check_trace=False)
    # Preserve expensive traces for diagnostics if the downstream converter fails.
    if input_name == 'image':
        traced.save(str(target.with_suffix('.torchscript.pt')))
    converted = ct.convert(traced, convert_to='mlprogram',
        minimum_deployment_target=ct.target.iOS18,
        inputs=[ct.TensorType(name=input_name, shape=example.shape, dtype=np.float32 if fp32_input else np.float16)],
        outputs=[ct.TensorType(name=output_name, dtype=np.float16)],
        compute_precision=compute_precision, skip_model_load=True)
    converted.author = 'SoulX FlashHead Apple inference'
    converted.short_description = 'PRO Core ML component; original weights; fixed 33-frame temporal contract'
    converted.user_defined_metadata['soulx_compute_precision'] = (
        str(compute_precision) if isinstance(compute_precision, ct.precision)
        else 'fp16_with_fp32_attention_and_normalization')
    converted.save(str(target))
    print(f'EXPORTED {target} in {time.time()-started:.1f}s', flush=True)
    del converted, traced
    gc.collect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--component', choices=['audio','reference','decoder','encoder','all'], default='all')
    parser.add_argument('--height', type=int, default=576)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/coreml/pro576')
    parser.add_argument('--reference-trace', type=Path, help='Reuse a saved reference TorchScript trace with the same height/width; avoids tracing again')
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.component in ('audio','all'):
        from transformers import Wav2Vec2Model
        model = Wav2Vec2Model.from_pretrained(str(ROOT/'models/wav2vec2-base-960h'), attn_implementation='eager').eval()
        # CPU_ONLY full-shape validation found 3.17% normalized error with
        # blanket FP16 and 2.29% with attention/norms retained in FP32. Full
        # FP32 reaches 0.021% including the FP16 output cast. Keep this smaller
        # network accurate; activation/weight compression needs a new gate.
        export(Audio(model), torch.zeros(1,128000), 'waveform','features', args.output/'audio.mlpackage', True,
               compute_precision=ct.precision.FLOAT32)
        del model
        gc.collect()
    if args.component in ('decoder','encoder','all'):
        vendor = module_at('soulx_apple_taehv', ROOT/'models/tiny_vae/taehv/taehv.py')
        tae = vendor.TAEHV(checkpoint_path=str(ROOT/'models/tiny_vae/taehv/taew2_1.pth'), arch_name='taew2_1').float().eval()
        if args.component in ('decoder','all'):
            export(TinyDecode(tae,vendor.MemBlock), torch.zeros(1,9,16,args.height//8,args.width//8),'latent','frames',args.output/'decoder.mlpackage')
        if args.component in ('encoder','all'):
            export(TinyEncode(tae,vendor.MemBlock), torch.zeros(1,5,3,args.height,args.width),'frames','motion',args.output/'encoder.mlpackage')
        del tae
        gc.collect()
    if args.component in ('reference','all'):
        if args.reference_trace:
            reference = torch.jit.load(str(args.reference_trace), map_location='cpu').eval()
        else:
            vendor = module_at('soulx_apple_wan', ROOT/'flash_head/wan/modules/vae.py')
            vae = vendor.WanVAE(vae_path=str(ROOT/'models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth'), dtype=torch.float32,device='cpu')
            reference = Reference(vae,vendor.count_conv3d)
            del vae
            gc.collect()
        export(reference,torch.zeros(1,3,1,args.height,args.width),'image','reference',args.output/'reference.mlpackage',True,
               # Wan's full-size FP16 and mixed norm/attention FP32 variants
               # failed the numerical gate. Full FP32 passes; initialization
               # runs once and its weights are released before the DiT.
               compute_precision=ct.precision.FLOAT32)

if __name__=='__main__': main()
