"""Opt-in fixed-shape DiT graph; one owner, external RNG and session state.

Returned storage is borrowed until the next replay. The engine consumes it
before another replay. Never use this wrapper concurrently from two streams.
"""
import torch


def tree_map(fn, value):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, tuple):
        return tuple(tree_map(fn, v) for v in value)
    if isinstance(value, list):
        return [tree_map(fn, v) for v in value]
    if isinstance(value, dict):
        return {k: tree_map(fn, v) for k, v in value.items()}
    return value


def copy_tree(target, source):
    if isinstance(target, torch.Tensor):
        if (target.shape != source.shape or target.dtype != source.dtype
                or target.device != source.device):
            raise ValueError("CUDA graph input contract changed")
        target.copy_(source)
    elif isinstance(target, dict):
        if target.keys() != source.keys():
            raise ValueError("CUDA graph keyword contract changed")
        for key in target:
            copy_tree(target[key], source[key])
    elif isinstance(target, (tuple, list)):
        if len(target) != len(source):
            raise ValueError("CUDA graph sequence contract changed")
        for a, b in zip(target, source):
            copy_tree(a, b)
    elif target != source:
        raise ValueError("CUDA graph constant changed")


class DenoiseGraph:
    def __init__(self, model, x, timestep, kwargs):
        self.inputs = tree_map(lambda t: t.clone(), (x, timestep, kwargs))
        self.device = x.device
        self.stream = torch.cuda.current_stream(x.device).cuda_stream
        a, t, options = self.inputs
        # Finish compile/allocator initialization outside capture. The engine
        # already runs on a side stream; inference has no stochastic operations.
        for _ in range(3):
            model(x=a, timestep=t, **options)
        torch.cuda.synchronize(x.device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = model(x=a, timestep=t, **options)

    def __call__(self, x, timestep, kwargs):
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise ValueError("DiT graph must remain on its owning compute stream")
        copy_tree(self.inputs, (x, timestep, kwargs))
        self.graph.replay()
        return self.output
