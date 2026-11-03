"""Verify the complete model contract and assemble an iOS import folder.

Run after both exporters. Never mark an incomplete partition set as ready.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import coremltools as ct


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,default=Path('artifacts/coreml/pro576'))
    parser.add_argument('--output',type=Path,default=Path('artifacts/coreml/device/pro576'))
    parser.add_argument('--compile',action='store_true',help='Compile on Mac for faster installation; iOS can also compile packages once')
    args=parser.parse_args()
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must be different folders')
    source=json.loads((args.input/'dit_manifest.json').read_text())
    if not source.get('complete'):
        raise ValueError('The transformer export manifest is incomplete')
    count=(30+source['blocks_per_chunk']-1)//source['blocks_per_chunk']
    height,width=source['height'],source['width']
    latent=[1,16,9,height//8,width//8]
    names=['reference','audio','patch','conditioning']+[f'blocks_{i:02}' for i in range(count)]+['head','decoder','encoder']
    expected={
        'reference':({'image':[1,3,1,height,width]},'reference',latent),
        'audio':({'waveform':[1,128000]},'features',[1,200,12,768]),
        'patch':({'latent':latent,'reference':latent},'hidden',[1,9*(height//16)*(width//16),1536]),
        'conditioning':({'audio':[1,33,5,12,768]},'context',[1,9,32,1536]),
        'head':({'hidden':[1,9*(height//16)*(width//16),1536],'timestep':[1]},'flow',latent),
        'decoder':({'latent':[1,9,16,height//8,width//8]},'frames',[1,33,3,height,width]),
        'encoder':({'frames':[1,5,3,height,width]},'motion',[1,16,2,height//8,width//8]),
    }
    hidden=[1,9*(height//16)*(width//16),1536]
    for i in range(count): expected[f'blocks_{i:02}']=({'hidden':hidden,'context':[1,9,32,1536],'timestep':[1]},'hidden_out',hidden)
    precision_metadata={}
    for name in names:
        path=args.input/(name+'.mlpackage')
        if not path.exists(): raise FileNotFoundError(f'Missing required component: {path}')
        model=ct.models.MLModel(str(path),skip_model_load=True)
        precision_metadata[name]=dict(model.user_defined_metadata)
        spec=model.get_spec()
        inputs={v.name:list(v.type.multiArrayType.shape) for v in spec.description.input}
        outputs={v.name:list(v.type.multiArrayType.shape) for v in spec.description.output}
        exp_inputs,exp_name,exp_shape=expected[name]
        if inputs != exp_inputs or outputs != {exp_name:exp_shape}:
            raise ValueError(f'{name} contract mismatch: {inputs}, {outputs}')
    args.output.mkdir(parents=True,exist_ok=True)
    # An interrupted rebuild must not advertise a complete installation.
    (args.output/'manifest.json').unlink(missing_ok=True)
    records={}
    for name in names:
        path=args.input/(name+'.mlpackage')
        # Never let an obsolete compiled model shadow a newly copied package.
        for extension in ('.mlmodelc', '.mlpackage', '.mlmodel'):
            old = args.output/(name+extension)
            if old.is_dir(): shutil.rmtree(old)
            elif old.exists(): old.unlink()
        if args.compile:
            subprocess.run(['xcrun','coremlcompiler','compile',str(path),str(args.output)],check=True)
        else:
            shutil.copytree(path,args.output/path.name,dirs_exist_ok=True)
        records[name]={'source':path.name,'metadata':precision_metadata[name]}
        print('BUNDLED',name,flush=True)
    manifest=dict(schema_version=1,profile=f'pro{height}',height=height,width=width,block_count=count,
        frames_per_window=33,history_frames=5,fps=25,sample_rate=16000,sampling_steps=[1000,625,0],
        color_correction_strength=1.0,decoder_mode='window',compression=source['compression'],
        source_checkpoint_sha256=source['source_checkpoint_sha256'],models=records,
        dit_precision=source['precision'],transformer_variant=source.get('variant','baseline'),
        warning='Apple weight compression with component-specific mixed FP16/FP32 execution; inspect model metadata. CUDA FP8/INT8 activation rounding is not reproduced. ANE placement and real-time throughput require device profiling.')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    shutil.copy2(args.input/'dit_manifest.json',args.output/'dit_manifest.json')
    print(f'Complete model bundle: {args.output}',flush=True)

if __name__=='__main__':main()
