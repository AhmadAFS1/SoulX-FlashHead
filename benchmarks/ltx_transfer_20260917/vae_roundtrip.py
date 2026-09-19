"""Diagnostic only: pass a sharp LTX sample through SoulX's existing VAE."""
import json, subprocess, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from flash_head.ltx_video.ltx_vae import LtxVAE
from soulx_rtc.gpu_lease import acquire_gpu_lease

root=Path(__file__).parent
lease=acquire_gpu_lease()
torch.set_num_threads(4)
report={'date_utc':datetime.now(timezone.utc).isoformat(),
 'hardware':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version','--format=csv'],text=True),
 'coresident':subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv'],text=True),
 'torch':torch.__version__,'cuda':torch.version.cuda,
 'workload':'GPU VAE-only encode/decode, BF16, 9 repeated 480x832 RGB frames, posterior sample seed 50, no DiT/audio/color correction',
 'source':'LTX fork committed Q4 sample at 5 seconds; historical source run was RTX 5060 Ti per fork; not regenerated here'}
with torch.inference_mode():
 vae=LtxVAE('models/SoulX-FlashHead-1_3B/VAE_LTX')
 rgb=np.asarray(Image.open(root/'q4-source-5s.png').convert('RGB')).copy()
 video=torch.from_numpy(rgb).permute(2,0,1).unsqueeze(0).unsqueeze(2).repeat(1,1,9,1,1).to('cuda',torch.bfloat16)/127.5-1
 torch.manual_seed(50);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
 latent=vae.encode(video);recon=vae.decode(latent)
 torch.cuda.synchronize()
 report.update(wall_s=time.perf_counter()-start,latent_shape=list(latent.shape),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
 out=((recon[0,:,4].float().clamp(-1,1)+1)*127.5).round().to(torch.uint8).permute(1,2,0).cpu().numpy()
 Image.fromarray(out).save(root/'q4-through-soulx-vae.png')
 report['rgb_mae']=float(np.abs(out.astype(np.int16)-rgb.astype(np.int16)).mean())
(root/'vae-roundtrip.json').write_text(json.dumps(report,indent=2)+'\n')
