"""Fetch pinned source weights without installing or changing CUDA dependencies."""
import hashlib
import json
from pathlib import Path
import urllib.request
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]
REPOSITORIES = {
    'Soul-AILab/SoulX-FlashHead-1_3B': ('5fbd2e44ba08ab2deb28bd55cff2c15523e51ba5', [
        'Model_Pro/config.json', 'Model_Pro/diffusion_pytorch_model.safetensors', 'VAE_Wan/Wan2.1_VAE.pth']),
    'facebook/wav2vec2-base-960h': ('22aad52d435eb6dbaf354bdad9b0da84ce7d6156', [
        'config.json', 'preprocessor_config.json', 'model.safetensors']),
}
EXPECTED = {
    'Model_Pro/diffusion_pytorch_model.safetensors': 'e47e61b9023ea1aac60c0c0fff077289bc6cca443c71907e5a76a411480af250',
    'VAE_Wan/Wan2.1_VAE.pth': '38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981',
    'taew2_1.pth': 'd26151e76cdc2c9424bef988de874b33d9a53f30ef3060cd556c429c469c797e',
}

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()

def record(path, repo, revision, name):
    sha = digest(path)
    if name in EXPECTED and sha != EXPECTED[name]:
        raise ValueError(f'Checkpoint hash mismatch: {path}')
    print(f'VERIFIED {path.name} {sha}', flush=True)
    return dict(repository=repo, revision=revision, file=name, sha256=sha, bytes=path.stat().st_size)

def main():
    records = []
    for repo, (revision, names) in REPOSITORIES.items():
        for name in names:
            path = Path(hf_hub_download(repo, name, revision=revision, local_dir=ROOT/'models'/repo.split('/')[-1]))
            records.append(record(path, repo, revision, name))
    vendor = ROOT/'models/tiny_vae/taehv'
    vendor.mkdir(parents=True, exist_ok=True)
    for name in ('taehv.py', 'taew2_1.pth', 'LICENSE'):
        path = vendor/name
        if not path.exists():
            temporary = path.with_suffix(path.suffix+'.download')
            urllib.request.urlretrieve(f'https://raw.githubusercontent.com/madebyollin/taehv/011dfc2/{name}', temporary)
            temporary.replace(path)
        records.append(record(path, 'madebyollin/taehv', '011dfc2', name))
    (ROOT/'models/apple-sources.json').write_text(json.dumps(records, indent=2)+'\n')

if __name__ == '__main__': main()
