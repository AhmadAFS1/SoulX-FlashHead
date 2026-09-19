"""CPU-only checkpoint metadata inventory and Graphviz architecture rendering."""
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import struct
import subprocess

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def main():
    import torch
    paths = [
        'models/SoulX-FlashHead-1_3B/Model_Lite/diffusion_pytorch_model.safetensors',
        'models/SoulX-FlashHead-1_3B/Model_Pro/diffusion_pytorch_model.safetensors',
        'models/SoulX-FlashHead-1_3B/VAE_LTX/diffusion_pytorch_model.safetensors',
        'models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth',
        'models/wav2vec2-base-960h/pytorch_model.bin',
        'models/mouth-sr/realesr-general-x4v3.pth',
        'models/ojin-components/2xNomosUni_compact_multijpg_ldl.safetensors',
    ]
    rows = []
    for relative in paths:
        path = REPO / relative
        if path.suffix == '.safetensors':
            with path.open('rb') as source:
                size = struct.unpack('<Q', source.read(8))[0]
                header = json.loads(source.read(size))
            tensors = [v for k,v in header.items() if k != '__metadata__']
            count = sum(math.prod(v['shape']) for v in tensors)
            dtypes = sorted({v['dtype'] for v in tensors})
        else:
            data = torch.load(path, map_location='meta', weights_only=True, mmap=True)
            for key in ('params_ema', 'params', 'state_dict'):
                if key in data:
                    data = data[key]
                    break
            tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
            count = sum(v.numel() for v in tensors)
            dtypes = sorted({str(v.dtype) for v in tensors})
        rows.append(dict(path=relative, stored_tensor_elements=count,
                         tensor_count=len(tensors), dtypes=dtypes, file_bytes=path.stat().st_size,
                         hypothetical_all_bf16_weight_bytes=count*2))
    evidence = dict(date_utc=datetime.now(timezone.utc).isoformat(),
        execution='Static source audit, CPU-only metadata reads and Graphviz rendering; no model inference',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        source_note='Local modifications are present; diagram describes inspected working tree',
        torch=torch.__version__, compiled_cuda_runtime=torch.version.cuda,
        gpu_snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used,driver_version',
                                              '--format=csv,noheader'],text=True).strip(),
        resident_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                                    '--format=csv,noheader'],text=True).strip(),
        count_note='Stored tensor elements, including inactive modules/buffers; not runtime active-parameter or VRAM measurement',
        weights=rows)
    (ROOT/'weight-inventory.json').write_text(json.dumps(evidence,indent=2)+'\n')
    for path in sorted(ROOT.glob('*.dot')):
        for extension in ('svg','png','pdf'):
            subprocess.run(['dot','-T'+extension,str(path),'-o',str(path.with_suffix('.'+extension))],check=True)
    print(json.dumps(rows,indent=2))


if __name__ == '__main__':
    main()
