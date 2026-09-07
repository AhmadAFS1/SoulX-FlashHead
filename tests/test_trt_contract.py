import json
import sys
from types import SimpleNamespace

import pytest

from soulx_rtc.trt_backend import TRTFeedForward, file_hash


def test_trt_artifact_rejects_mismatches_before_execution(tmp_path,monkeypatch):
    import torch
    monkeypatch.setitem(sys.modules,"tensorrt",SimpleNamespace(__version__="10.9.0.34"))
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda:"test GPU")
    path=tmp_path/"ffn-00.engine"
    path.write_bytes(b"deliberately not executable")
    metadata=dict(shape=[1,1280,1536],engine_sha256=file_hash(path),weights_sha256="weights",
                  tensorrt="10.9.0.34",gpu="test GPU",precision="bf16")
    def write(**updates):
        path.with_suffix(".json").write_text(json.dumps(metadata|updates))
    write()
    with pytest.raises(ValueError,match="shape"):
        TRTFeedForward(path,(1,1950,1536))
    write(engine_sha256="bad")
    with pytest.raises(ValueError,match="checksum"):
        TRTFeedForward(path,(1,1280,1536))
    write()
    with pytest.raises(ValueError,match="weights"):
        TRTFeedForward(path,(1,1280,1536),"different-weights")
    write(tensorrt="different-version")
    with pytest.raises(ValueError,match="runtime/device"):
        TRTFeedForward(path,(1,1280,1536))
    write(gpu="another GPU")
    with pytest.raises(ValueError,match="runtime/device"):
        TRTFeedForward(path,(1,1280,1536))


def test_trt_validates_actual_engine_binding_before_workspace(tmp_path,monkeypatch):
    import torch
    path=tmp_path/"decoder.engine"
    path.write_bytes(b"mock engine")
    path.with_suffix(".json").write_text(json.dumps(dict(shape=[1,4,8],
        engine_sha256=file_hash(path),tensorrt="test",gpu="test GPU",precision="bf16")))
    class Logger:
        WARNING=0
        def __init__(self,*args):
            pass
    engine=SimpleNamespace(num_io_tensors=2,get_tensor_shape=lambda name:(1,99,99))
    trt=SimpleNamespace(__version__="test",bfloat16="bf16",float16="fp16",Logger=Logger,
        TensorIOMode=SimpleNamespace(INPUT="in",OUTPUT="out"),
        Runtime=lambda _:SimpleNamespace(deserialize_cuda_engine=lambda _:engine))
    monkeypatch.setitem(sys.modules,"tensorrt",trt)
    monkeypatch.setattr(torch.cuda,"get_device_name",lambda:"test GPU")
    with pytest.raises(ValueError,match="actual binding"):
        TRTFeedForward(path,(1,4,8))


def test_transient_workspace_waits_before_release(monkeypatch):
    import torch
    runner=object.__new__(TRTFeedForward)
    torch.nn.Module.__init__(runner)
    runner.lazy_workspace=True
    runner.workspace=object()
    events=[]
    monkeypatch.setattr(torch.cuda,"current_stream",lambda:SimpleNamespace(synchronize=lambda:events.append("done")))
    runner.release_workspace()
    assert events==["done"] and runner.workspace is None
    runner.lazy_workspace=False
    with pytest.raises(RuntimeError,match="transient"):
        runner.release_workspace()
