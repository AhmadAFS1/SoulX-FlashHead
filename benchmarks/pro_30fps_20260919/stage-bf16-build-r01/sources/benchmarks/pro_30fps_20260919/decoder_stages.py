"""Capture/calibrate Wan residual stages, then build independent BF16/INT8 engines."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from benchmarks.pro_quantization_v2_20260918.common import (
    ROOT, DEFAULT_GPU_LOCK, atomic_write_json, ensure_new_directory,
    environment_manifest, sha256, relative_path, failure_record, snapshot_sources,
    utc_now,
)
from benchmarks.pro_quantization_v2_20260918.decoder_trial import _load_vae, _captured_latents
from benchmarks.pro_quantization_v2_20260918.build_decoder_engine import _build_engine
from soulx_rtc.gpu_lease import acquire_gpu_lease
from soulx_rtc.pro_vae_stage_backend import install_stages, remove_stages, tensor_signature, ResidualStage


def int8_convolutions(layers):
    records = json.loads(layers).get("Layers", [])
    convs = [item for item in records if "convolution" in str(item.get("LayerType", "")).lower()]
    quantized = [item for item in convs if item.get("Inputs") and all(
        "int8" in str(binding.get("Format/Datatype", "")).lower() for binding in item["Inputs"])]
    return {"convolutions": len(convs), "int8_convolutions": len(quantized),
            "quantized_names": [item.get("Name") for item in quantized]}


def capture(args, output, result):
    paths = _captured_latents(args.captures)[:args.max_latents]
    if len(paths) < 2:
        raise ValueError("Stage calibration needs at least two registered latent windows")
    vae = _load_vae()
    groups = [list(map(int, value.split(','))) for value in args.groups]
    result.update(groups=groups, samples={}, calibration={}, source_latents=[],
                  source_capture={"path": relative_path(args.captures), "sha256": sha256(args.captures)},
                  retained_bytes=0, max_bytes=args.max_raw_mib * 2**20)

    def factory(indices, stage):
        key = '-'.join(map(str, indices))
        result['samples'][key] = {}
        result['calibration'][key] = {}
        def execute(*values):
            signature = tensor_signature(values)
            records = result['samples'][key]
            if signature not in records:
                count = sum(value.numel() * value.element_size() for value in values)
                if count + result['retained_bytes'] > result['max_bytes']:
                    raise RuntimeError('Stage capture byte budget exceeded; signature coverage incomplete')
                path = output / f'stage-{key}-sample-{len(records)}.pt'
                torch.save(tuple(value.detach().cpu() for value in values), path)
                records[signature] = {'path': path.name, 'sha256': sha256(path), 'count': 0,
                                      'shapes': [list(value.shape) for value in values]}
                result['retained_bytes'] += count
            records[signature]['count'] += 1
            scales = result['calibration'][key].setdefault(signature, {})
            def observe(name, before, previous, after):
                maximum = float(before.abs().max())
                if previous is not None:
                    maximum = max(maximum, float(previous.abs().max()))
                out_max = float(after.abs().max())
                if not all(torch.isfinite(torch.tensor(v)) for v in (maximum, out_max)):
                    raise ValueError('Nonfinite stage calibration')
                item = scales.setdefault(name, {'input_max': 0., 'output_max': 0., 'count': 0})
                item['input_max'] = max(item['input_max'], maximum)
                item['output_max'] = max(item['output_max'], out_max)
                item['count'] += 1
            stage.observer = observe
            try:
                return stage(*values)
            finally:
                stage.observer = None
        return execute
    install_stages(vae, groups, factory)
    try:
        with torch.inference_mode():
            for path in paths:
                latent = torch.load(path, map_location='cuda', weights_only=True).to(torch.bfloat16)
                result['source_latents'].append({'path': relative_path(path), 'sha256': sha256(path)})
                vae.decode(latent)
                atomic_write_json(output/'results.json', result)
        for samples in result['samples'].values():
            if any(item['count'] < 2 for item in samples.values()):
                raise RuntimeError('Fewer than two calibration samples for a stage signature')
    finally:
        remove_stages(vae)


def export_stage(stage, values, path, precision, scales):
    import onnx
    import numpy as np
    from onnx import helper, numpy_helper, TensorProto
    inputs = ['x'] + [f'cache_in_{i}' for i in range(len(values)-1)]
    outputs = ['y'] + [f'cache_out_{i}' for i in range(stage.cache_count)]
    with torch.inference_mode():
        actual = stage(*values)
        torch.onnx.export(stage, values, str(path), input_names=inputs, output_names=outputs,
                          opset_version=18, dynamo=False, do_constant_folding=True)
    model = onnx.load(path)
    converted = []
    if precision == 'int8':
        nodes = []
        for node in model.graph.node:
            if node.op_type != 'Conv' or '.residual.' not in node.input[1]:
                nodes.append(node)
                continue
            weight_name = node.input[1]
            name = weight_name.removesuffix('.weight')
            if name not in scales or scales[name]['count'] < 2:
                raise ValueError(f'Incomplete calibration for {name}')
            prefix = name.replace('.', '_')
            weight = stage.get_parameter(weight_name).detach().float().cpu().numpy()
            bias = stage.get_parameter(name+'.bias').detach().float().cpu().numpy()
            ws = np.maximum(np.abs(weight).max(axis=tuple(range(1, weight.ndim))), 1e-12)/127.
            iq, oq = max(scales[name]['input_max']/127.,1e-12), max(scales[name]['output_max']/127.,1e-12)
            for suffix, array in [('weight',weight),('bias',bias),('ws',ws.astype('float32')),
                                  ('wz',np.zeros(ws.shape,dtype='int8')),('is',np.array(iq,dtype='float32')),
                                  ('os',np.array(oq,dtype='float32')),('z',np.array(0,dtype='int8'))]:
                model.graph.initializer.append(numpy_helper.from_array(array,prefix+'_'+suffix))
            original_output = node.output[0]
            nodes.extend([
                helper.make_node('Cast',[node.input[0]],[prefix+'_fp32'],to=TensorProto.FLOAT),
                helper.make_node('QuantizeLinear',[prefix+'_fp32',prefix+'_is',prefix+'_z'],[prefix+'_iq']),
                helper.make_node('DequantizeLinear',[prefix+'_iq',prefix+'_is',prefix+'_z'],[prefix+'_idq']),
                helper.make_node('QuantizeLinear',[prefix+'_weight',prefix+'_ws',prefix+'_wz'],[prefix+'_wq'],axis=0),
                helper.make_node('DequantizeLinear',[prefix+'_wq',prefix+'_ws',prefix+'_wz'],[prefix+'_wdq'],axis=0),
            ])
            node.input[:] = [prefix+'_idq',prefix+'_wdq',prefix+'_bias']
            node.output[:] = [prefix+'_conv']
            nodes.append(node)
            nodes.extend([
                helper.make_node('QuantizeLinear',[prefix+'_conv',prefix+'_os',prefix+'_z'],[prefix+'_oq']),
                helper.make_node('DequantizeLinear',[prefix+'_oq',prefix+'_os',prefix+'_z'],[prefix+'_odq']),
                helper.make_node('Cast',[prefix+'_odq'],[original_output],to=TensorProto.BFLOAT16),
            ])
            converted.append(name)
        del model.graph.node[:]
        model.graph.node.extend(nodes)
        if len(converted) != stage.cache_count:
            raise ValueError('INT8 export did not cover every declared residual convolution')
    elif precision != 'bf16':
        raise ValueError('Unsupported stage precision')
    onnx.checker.check_model(model)
    onnx.save(model,path)
    return {'inputs':[{'name':n,'shape':list(v.shape)} for n,v in zip(inputs,values)],
            'outputs':[{'name':n,'shape':list(v.shape)} for n,v in zip(outputs,actual)],
            'quantized_modules':converted}


def build(args, output, result):
    import tensorrt as trt
    source = json.loads(args.calibration.read_text())
    if source.get('status') != 'complete' or source['weights_sha256'] != result['weights_sha256']:
        raise ValueError('Incomplete or incompatible stage calibration')
    vae = _load_vae()
    result.update(schema_version=1,groups=source['groups'],precision=args.precision,stages={},
                  calibration={'path':relative_path(args.calibration),'sha256':sha256(args.calibration)})
    for indices in source['groups']:
        key = '-'.join(map(str,indices))
        stage = ResidualStage([vae.model.decoder.upsamples[i] for i in indices]).eval()
        result['stages'][key] = {}
        for i,(signature,sample) in enumerate(source['samples'][key].items()):
            path = args.calibration.parent/sample['path']
            if sha256(path)!=sample['sha256']:raise ValueError('Stage sample hash mismatch')
            values=tuple(v.cuda() for v in torch.load(path,map_location='cpu',weights_only=True))
            if tensor_signature(values)!=signature:raise ValueError('Stage sample signature mismatch')
            onnx_path=output/f'stage-{key}-{i}.onnx'
            metadata=export_stage(stage,values,onnx_path,args.precision,source['calibration'][key][signature])
            engine_path=onnx_path.with_suffix('.engine')
            begin=time.perf_counter()
            engine,layers=_build_engine(onnx_path,engine_path,args.workspace_mib)
            evidence=int8_convolutions(layers)
            layers_path=engine_path.with_suffix('.layers.json');layers_path.write_text(layers+'\n')
            if args.precision=='int8' and evidence['int8_convolutions']<len(metadata['quantized_modules']):
                raise RuntimeError('Stage inspector does not prove all requested INT8 convolutions')
            result['stages'][key][signature]=metadata|{
                'path':relative_path(engine_path),'sha256':sha256(engine_path),'precision':args.precision,
                'onnx_sha256':sha256(onnx_path),'layers_path':relative_path(layers_path),
                'layers_sha256':sha256(layers_path),'precision_evidence':evidence,
                'int8_convolutions_verified':args.precision=='int8',
                'tensorrt':trt.__version__,'gpu':torch.cuda.get_device_name(),
                'build_s':time.perf_counter()-begin,'workspace_bytes':engine.device_memory_size,
            }
            del engine,values
            torch.cuda.empty_cache()
            atomic_write_json(output/'results.json',result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest='command',required=True)
    cap=commands.add_parser('capture');cap.add_argument('--captures',type=Path,required=True)
    cap.add_argument('--groups',nargs='+',default=['4,5,6','8,9,10'])
    cap.add_argument('--max-latents',type=int,default=4);cap.add_argument('--max-raw-mib',type=int,default=2048)
    bld=commands.add_parser('build');bld.add_argument('--calibration',type=Path,required=True)
    bld.add_argument('--precision',choices=['bf16','int8'],required=True)
    bld.add_argument('--workspace-mib',type=int,default=512)
    for sub in (cap,bld):
        sub.add_argument('--output',type=Path,required=True);sub.add_argument('--gpu-lock',type=Path,default=DEFAULT_GPU_LOCK)
    args=parser.parse_args();output=ensure_new_directory(args.output)
    result={'status':'starting','date_utc':utc_now(),'execution':'fresh GPU stage diagnostic; not end-to-end video throughput',
            'environment':environment_manifest(),'arguments':{k:str(v) for k,v in vars(args).items()},
            'weights_sha256':sha256(ROOT/'models/SoulX-FlashHead-1_3B/VAE_Wan/Wan2.1_VAE.pth')}
    result['source_sha256']=snapshot_sources(output,[Path(__file__),ROOT/'soulx_rtc/pro_vae_stage_backend.py',
        ROOT/'flash_head/wan/modules/vae.py',ROOT/'benchmarks/pro_quantization_v2_20260918/build_decoder_engine.py'])
    atomic_write_json(output/'results.json',result)
    lease=None
    try:
        lease=acquire_gpu_lease(args.gpu_lock)
        (capture if args.command=='capture' else build)(args,output,result)
        result['status']='complete'
    except Exception as error:
        result['status']='failed';result['failure']=failure_record(args.command,error)
        raise
    finally:
        atomic_write_json(output/'results.json',result)
        if lease is not None:lease.close()


if __name__=='__main__':main()
