"""GPU process boundary: CUDA tensors never cross the WebRTC process boundary.

Keeping only a thread for inference lets Python RTP/codec callbacks contend
with model graph-break/kernel-dispatch code for the GIL. A spawned process
isolates the hot path while retaining one model and bounded uint8 IPC chunks.
"""
import asyncio
import multiprocessing
import secrets
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .engine import Engine

_engine = None
_states = {}


@dataclass
class RemoteState:
    id: str
    total_frames: int
    cursor: int = 0


def initialize(size, steps, compiled, fps, batch, validate):
    global _engine
    _engine = Engine(size, steps, compiled, fps)
    for n in range(1, batch + 1):
        _engine.warmup(batch_size=n)
    isolation = _engine.validate_isolation() if validate else None
    return isolation


def prepare(path, audio, seed):
    sid = secrets.token_urlsafe(18)
    state = _engine.prepare(path, audio, seed)
    _states[sid] = state
    return RemoteState(sid, state.total_frames)


def generate(ids):
    states = [_states[sid] for sid in ids]
    chunks = _engine.generate(states)
    return chunks, _engine.last_metrics.copy(), [s.cursor for s in states]


def release(sid):
    _states.pop(sid, None)


class GPUProcess:
    def __init__(self):
        self.executor = ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context("spawn"))

    async def call(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    async def start(self, args):
        return await self.call(initialize, args.size, args.steps, not args.eager,
                               args.fps, args.batch, args.validate_isolation)

    async def prepare(self, path, audio, seed):
        return await self.call(prepare, path, audio, seed)

    async def generate(self, states):
        chunks, metrics, cursors = await self.call(generate, [s.id for s in states])
        for s, cursor in zip(states, cursors):
            s.cursor = cursor
        return chunks, metrics

    async def release(self, state):
        await self.call(release, state.id)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
