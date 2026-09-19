import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper,numpy_helper,TensorProto
from benchmarks.pro_quantization_v2_20260918.build_decoder_engine import _build_engine
from benchmarks.pro_30fps_20260919.decoder_stages import int8_convolutions
from benchmarks.pro_quantization_v2_20260918.common import DEFAULT_GPU_LOCK,environment_manifest
from soulx_rtc.gpu_lease import acquire_gpu_lease
r=Path('benchmarks/pro_30fps_20260919');out=r/'stage-precision-probes-r01';out.mkdir();lease=acquire_gpu_lease(DEFAULT_GPU_LOCK)
results={'environment':environment_manifest(),'variants':{}}
for name,dtype in [('fp16',TensorProto.FLOAT16),('fp32',TensorProto.FLOAT)]:
 m=onnx.load(r/'stage-int8-build-r01/stage-4-5-6-0.onnx')
 def convert(t):
  if t.data_type==TensorProto.BFLOAT16:
   assert t.raw_data
   a=(np.frombuffer(t.raw_data,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(t.dims)
   t.CopyFrom(numpy_helper.from_array(a.astype(np.float16 if dtype==TensorProto.FLOAT16 else np.float32),t.name))
 for t in m.graph.initializer:convert(t)
 for n in m.graph.node:
  for a in n.attribute:
   if a.type==onnx.AttributeProto.TENSOR:convert(a.t)
   if n.op_type=='Cast' and a.name=='to' and a.i==TensorProto.BFLOAT16:a.i=dtype
 for v in m.graph.value_info:
  if v.type.tensor_type.elem_type==TensorProto.BFLOAT16:v.type.tensor_type.elem_type=dtype
 before=[];after=[]
 for v in m.graph.input:
  old=v.name;new=old+'_internal'
  for n in m.graph.node:
   for i,k in enumerate(n.input):
    if k==old:n.input[i]=new
  before.append(helper.make_node('Cast',[old],[new],to=dtype))
 for v in m.graph.output:
  old=v.name;new=old+'_internal'
  for n in m.graph.node:
   for i,k in enumerate(n.input):
    if k==old:n.input[i]=new
   for i,k in enumerate(n.output):
    if k==old:n.output[i]=new
  after.append(helper.make_node('Cast',[new],[old],to=TensorProto.BFLOAT16))
 nodes=before+list(m.graph.node)+after;del m.graph.node[:];m.graph.node.extend(nodes);onnx.checker.check_model(m)
 p=out/(name+'.onnx');onnx.save(m,p)
 try:
  engine,layers=_build_engine(p,p.with_suffix('.engine'),512);p.with_suffix('.layers.json').write_text(layers);results['variants'][name]=int8_convolutions(layers);del engine
 except Exception as error:results['variants'][name]={'error':str(error)}
 (out/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(name,results['variants'][name],flush=True)
lease.close()
