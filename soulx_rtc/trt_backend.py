"""Opt-in, exact-shape TensorRT FFN partitions. No silent PyTorch fallback.

Engine files are trusted local executable artifacts, never accepted from HTTP.
Weights and engines are deliberately excluded from Git.
"""
import hashlib
import json
from pathlib import Path

import torch


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


class TRTFeedForward(torch.nn.Module):
    def __init__(self, path, expected_shape, weight_digest=None, allocate_workspace=True,
                 lazy_workspace=False):
        super().__init__()
        import tensorrt as trt
        self.trt = trt
        path = Path(path)
        metadata = json.loads(path.with_suffix(".json").read_text())
        if metadata["shape"] != list(expected_shape):
            raise ValueError("TensorRT shape/profile mismatch")
        if metadata["engine_sha256"] != file_hash(path):
            raise ValueError("TensorRT artifact checksum mismatch")
        if weight_digest is not None and metadata["weights_sha256"] != weight_digest:
            raise ValueError("TensorRT model weights mismatch")
        if metadata["tensorrt"] != trt.__version__ or metadata["gpu"] != torch.cuda.get_device_name():
            raise ValueError("TensorRT runtime/device mismatch; rebuild on this target")
        self.dtype = {"bf16":torch.bfloat16,"fp16":torch.float16}[metadata["precision"]]
        self.shape = tuple(expected_shape)
        self.output_shape = tuple(metadata.get("output_shape",expected_shape))
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(path.read_bytes())
        if self.engine is None:
            raise RuntimeError("TensorRT engine failed to deserialize")
        if self.engine.num_io_tensors != 2:
            raise ValueError("TensorRT engine must expose exactly input/output")
        expected_dtype = {"bf16":trt.bfloat16,"fp16":trt.float16}[metadata["precision"]]
        for name, shape, mode in (("input",self.shape,trt.TensorIOMode.INPUT),
                                  ("output",self.output_shape,trt.TensorIOMode.OUTPUT)):
            if (tuple(self.engine.get_tensor_shape(name)) != shape or
                    self.engine.get_tensor_dtype(name) != expected_dtype or
                    self.engine.get_tensor_mode(name) != mode):
                raise ValueError("TensorRT actual binding differs from artifact metadata")
        self.context = self.engine.create_execution_context_without_device_memory()
        if self.context is None:
            raise RuntimeError("TensorRT execution context failed")
        self.metadata = metadata
        self.workspace = None
        self.lazy_workspace = lazy_workspace
        if allocate_workspace and not lazy_workspace:
            self.set_workspace(torch.empty(max(1,self.engine.device_memory_size),device="cuda",dtype=torch.uint8))

    def set_workspace(self, workspace):
        if (not workspace.is_cuda or workspace.dtype != torch.uint8 or not workspace.is_contiguous()
                or workspace.numel() < self.engine.device_memory_size):
            raise ValueError("TensorRT shared workspace too small")
        self.workspace = workspace
        self.context.device_memory = workspace.data_ptr()

    def release_workspace(self):
        if not self.lazy_workspace:
            raise RuntimeError("Only a transient workspace may be released")
        # TensorRT must have finished using the pointer before returning it to
        # Torch's pool. The next execute always binds a newly owned workspace.
        if self.workspace is not None:
            torch.cuda.current_stream().synchronize()
            self.workspace = None

    @torch.compiler.disable
    def forward(self, x):
        if not x.is_cuda or tuple(x.shape)!=self.shape:
            raise ValueError("TensorRT input shape/device mismatch")
        if self.workspace is None:
            if not self.lazy_workspace:
                raise RuntimeError("TensorRT workspace not assigned")
            self.set_workspace(torch.empty(max(1,self.engine.device_memory_size),device=x.device,dtype=torch.uint8))
        if x.device != self.workspace.device:
            raise ValueError("TensorRT input and workspace devices differ")
        original_dtype = x.dtype
        input_tensor = x.to(self.dtype).contiguous()
        output = torch.empty(self.output_shape,device=x.device,dtype=self.dtype)
        for name,tensor in (("input",input_tensor),("output",output)):
            if not self.context.set_tensor_address(name,tensor.data_ptr()):
                raise RuntimeError("TensorRT tensor binding failed")
        if not self.context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return output.to(original_dtype)


def install_ffn_partitions(model, directory, width, height, batch=1):
    directory = Path(directory)
    weights = Path("models/SoulX-FlashHead-1_3B/Model_Lite/diffusion_pytorch_model.safetensors")
    digest = file_hash(weights)
    shape = (batch,5*(height//32)*(width//32),model.dim)
    paths = sorted(directory.glob("ffn-*.engine"))
    if not paths:
        raise ValueError("No TensorRT FFN engines in configured directory")
    installed = []
    # Free original FFN device weights before allocating replacement engines.
    # Any failure aborts startup; it never silently falls back to mixed weights.
    entries = []
    for path in paths:
        metadata = json.loads(path.with_suffix(".json").read_text())
        index = metadata["layer"]
        if not isinstance(index,int) or not 0<=index<len(model.blocks) or index in installed:
            raise ValueError("Invalid or duplicate TensorRT layer")
        entries.append((index,path))
        installed.append(index)
    for index,_ in entries:
        model.blocks[index].ffn.to("cpu")
    torch.cuda.empty_cache()
    candidates = [(index,TRTFeedForward(path,shape,digest,allocate_workspace=False)) for index,path in entries]
    workspace = torch.empty(max(1,max(p.engine.device_memory_size for _,p in candidates)),device="cuda",dtype=torch.uint8)
    for index,partition in candidates:
        partition.set_workspace(workspace)
        model.blocks[index].ffn = partition
    torch.cuda.empty_cache()
    return installed
