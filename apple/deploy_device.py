"""Copy a compiled model bundle to SoulX Studio with resumable file verification.

Requires the signed app to be installed. Copy one file per connection because a
single multi-gigabyte directory transfer can disconnect partway through.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

BUNDLE_ID='com.lingua.soulxstudio'

def run(args, **kwargs):
    return subprocess.run(['xcrun','devicectl',*args],check=True,**kwargs)

def inventory(device):
    with tempfile.TemporaryDirectory() as temporary:
        output=Path(temporary)/'files.json'
        result=subprocess.run(['xcrun','devicectl','device','info','files','--device',device,
            '--domain-type','appDataContainer','--domain-identifier',BUNDLE_ID,
            '--subdirectory','Documents/Models/current','--json-output',str(output),'--quiet'],capture_output=True,text=True)
        if result.returncode: return {}
        return {item['relativePath']:item['metadata']['size'] for item in json.loads(output.read_text())['result']['files'] if not item['resources']['isDirectory']}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',required=True)
    parser.add_argument('--models',type=Path,default=Path('artifacts/coreml/device/pro320'))
    parser.add_argument('--resume',action='store_true',help='Skip matching byte lengths after an interrupted copy of this exact bundle; omit after regenerating weights')
    args=parser.parse_args()
    if not (args.models/'manifest.json').exists(): parser.error('A complete bundled model folder is required')
    files=sorted([p for p in args.models.rglob('*') if p.is_file()],key=lambda p:(p.name=='manifest.json',str(p)))
    existing=inventory(args.device) if args.resume else {}
    # Create missing parent directories without replacing existing model files.
    with tempfile.TemporaryDirectory() as temporary:
        skeleton=Path(temporary)/'current'
        skeleton.mkdir()
        for source in files:
            (skeleton/source.relative_to(args.models).parent).mkdir(parents=True,exist_ok=True)
        run(['device','copy','to','--device',args.device,'--source',str(skeleton),
             '--destination','Documents/Models/current','--domain-type','appDataContainer',
             '--domain-identifier',BUNDLE_ID,'--timeout','600','--quiet'])
    for source in files:
        relative=str(source.relative_to(args.models))
        if args.resume and source.name!='manifest.json' and existing.get(relative)==source.stat().st_size:
            continue
        print(f'COPY {relative} ({source.stat().st_size} bytes)',flush=True)
        for attempt in range(3):
            try:
                run(['device','copy','to','--device',args.device,'--source',str(source),
                     '--destination','Documents/Models/current/'+relative,'--domain-type','appDataContainer',
                     '--domain-identifier',BUNDLE_ID,'--timeout','600','--quiet'],timeout=620)
                break
            except (subprocess.CalledProcessError,subprocess.TimeoutExpired):
                if attempt==2: raise
                print(f'Retrying {relative}',flush=True)
    copied=inventory(args.device)
    invalid=[str(p.relative_to(args.models)) for p in files if copied.get(str(p.relative_to(args.models)))!=p.stat().st_size]
    if invalid: raise RuntimeError(f'Incomplete file sizes after transfer: {invalid}')
    print(f'COMPLETE: {len(files)} model files match byte lengths. Model load and numerical validation remain separate checks.',flush=True)

if __name__=='__main__': main()
