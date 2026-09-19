"""Native 2x SRVGG TensorRT adapter. Public substitute weights, not Ojin's engine.

Only deserialize locally built engines with their matching metadata. No silent
PyTorch fallback. Fixed-shape engines must be rebuilt for a different GPU/runtime.
"""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from safetensors.torch import load_file
from .vendor.srvgg import SRVGGNetCompact


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_native2x(weights):
    state = load_file(str(weights))
    # This pinned public checkpoint has 16 body convs, 64 features, PReLU.
    model = SRVGGNetCompact(num_feat=64, num_conv=16, upscale=2)
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


class Native2xTRT:
    def __init__(self, path):
        import tensorrt as trt
        path = Path(path)
        meta = json.loads(path.with_suffix('.json').read_text())
        if digest(path) != meta['engine_sha256']:
            raise ValueError('TensorRT engine checksum mismatch')
        if trt.__version__ != meta['tensorrt'] or torch.cuda.get_device_name() != meta['gpu_name']:
            raise ValueError('Rebuild engine for this GPU and TensorRT version')
        self.meta = meta
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError('Engine deserialization failed')
        self.shape = tuple(meta['input_shape'])
        self.out_shape = (1, 3, self.shape[2]*2, self.shape[3]*2)
        if self.engine.num_io_tensors != 2:
            raise ValueError('Unexpected engine interface')
        for name,shape,mode in [('input',self.shape,trt.TensorIOMode.INPUT),
                                ('output',self.out_shape,trt.TensorIOMode.OUTPUT)]:
            if (tuple(self.engine.get_tensor_shape(name)) != shape
                or self.engine.get_tensor_dtype(name) != trt.float32
                or self.engine.get_tensor_mode(name) != mode):
                raise ValueError('Binding differs from engine metadata')
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError('Failed to allocate TensorRT execution context')

    @torch.inference_mode()
    def tensor(self, x):
        if tuple(x.shape) != self.shape or not x.is_cuda or x.device.index not in (None,torch.cuda.current_device()):
            raise ValueError('Unexpected input shape/device')
        x = x.float().contiguous()
        out = torch.empty(self.out_shape, dtype=torch.float32, device=x.device)
        for name,t in [('input',x),('output',out)]:
            if not self.context.set_tensor_address(name,t.data_ptr()):
                raise RuntimeError('TensorRT binding failed')
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError('TensorRT execution failed')
        return out

    @torch.inference_mode()
    def upscale(self, rgb, scale=2):
        if scale not in (1,2):
            raise ValueError('Native 2x model supports 1x/2x delivery only')
        if rgb.dtype != np.uint8 or rgb.shape != (self.shape[2],self.shape[3],3):
            raise ValueError('Unexpected RGB geometry/dtype')
        x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2,0,1)[None].cuda().float()/255
        out = self.tensor(x).clamp(0,1)[0].permute(1,2,0).cpu().numpy()
        if scale == 1:
            out = cv2.resize(out,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_AREA)
        return (out*255).round().clip(0,255).astype(np.uint8)
