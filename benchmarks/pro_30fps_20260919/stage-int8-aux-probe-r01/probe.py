import json
from pathlib import Path
import onnx
from onnx import helper,TensorProto
from benchmarks.pro_quantization_v2_20260918.build_decoder_engine import _build_engine
from benchmarks.pro_30fps_20260919.decoder_stages import int8_convolutions
from benchmarks.pro_quantization_v2_20260918.common import DEFAULT_GPU_LOCK,environment_manifest
from soulx_rtc.gpu_lease import acquire_gpu_lease
r=Path('benchmarks/pro_30fps_20260919');out=r/'stage-int8-aux-probe-r01';out.mkdir();lease=acquire_gpu_lease(DEFAULT_GPU_LOCK)
m=onnx.load(r/'stage-int8-build-r01/stage-4-5-6-0.onnx')
for n in m.graph.node:
 if n.op_type=='QuantizeLinear' and n.output[0].endswith('_oq'):
  m.graph.output.append(helper.make_tensor_value_info(n.output[0],TensorProto.INT8,[1,384,1,144,80]))
onnx.checker.check_model(m);p=out/'stage.onnx';onnx.save(m,p)
engine,layers=_build_engine(p,p.with_suffix('.engine'),512);p.with_suffix('.layers.json').write_text(layers);e=int8_convolutions(layers);(out/'results.json').write_text(json.dumps({'environment':environment_manifest(),'evidence':e},indent=2)+'\n');print(e,flush=True);lease.close()
