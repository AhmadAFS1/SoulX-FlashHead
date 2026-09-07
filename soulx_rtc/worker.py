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


def initialize(size, steps, compiled, fps, batch, validate, options=None, idle_video=None):
    import tempfile
    import numpy as np
    import av
    from pathlib import Path
    from PIL import Image
    global _engine
    _engine = Engine(size, steps, compiled, fps, **(options or {}))
    with tempfile.TemporaryDirectory(prefix="soulx-warmup-") as folder:
        image="examples/girl.png"
        if idle_video:
            with av.open(idle_video) as src:
                anchor=next(src.decode(video=0)).reformat(width=_engine.width,height=_engine.height,format="rgb24").to_ndarray()
            image=str(Path(folder)/"anchor.png")
            Image.fromarray(anchor).save(image)
        else:
            from flash_head.utils.utils import resize_and_centercrop
            with Image.open(image) as src:
                anchor=resize_and_centercrop(src.convert("RGB"),(_engine.height,_engine.width))[0,:,0].permute(1,2,0).numpy()
        for n in range(1, batch + 1):
            _engine.warmup(image=image,batch_size=n)
        # Interrupt/source-idle reconditioning has a distinct encoder layout.
        # Complete its cold compilation before accepting the first caller.
        state=_engine.prepare_call(image,100)
        _engine.recondition(state,np.repeat(anchor[None],9,axis=0))
        _engine.append(state,np.zeros(16000,np.float32))
        _engine.generate([state])
        isolation = _engine.validate_isolation(image=image,batch_size=min(batch,2)) if validate else None
    return isolation


def prepare(path, audio, seed):
    sid = secrets.token_urlsafe(18)
    state = _engine.prepare(path, audio, seed)
    _states[sid] = state
    return RemoteState(sid, state.total_frames)


def generate(ids):
    from .metrics import process_rss_mib
    states = [_states[sid] for sid in ids]
    chunks = _engine.generate(states)
    metrics = _engine.last_metrics.copy()
    metrics.update(resident_states=len(_states), resident_templates=len(_engine.templates),
                   worker_rss_mib=process_rss_mib(),
                   retained_audio_bytes=sum(s.audio.nbytes for s in _states.values()))
    return chunks, metrics, [s.cursor for s in states]


def release(sid):
    _states.pop(sid, None)


def prepare_call(path, seed):
    import numpy as np
    remote = prepare(path, np.zeros(1, np.float32), seed)
    state = _states[remote.id]
    state.audio = np.zeros(0, np.float32)
    state.total_frames = remote.total_frames = 0
    return remote


def append(sid, audio):
    return _engine.append(_states[sid], audio)


def recondition(sid, displayed):
    _engine.recondition(_states[sid], displayed)


class GPUProcess:
    def __init__(self):
        self.executor = ProcessPoolExecutor(max_workers=1,
            mp_context=multiprocessing.get_context("spawn"))
        self.states = set()

    async def call(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    async def start(self, args):
        options = {name: getattr(args, name) for name in
                   ("width", "height", "optimized", "profile", "real_rope", "memory_mode", "trt_ffn", "trt_vae")
                   if hasattr(args, name)}
        return await self.call(initialize, args.size, args.steps, not args.eager,
                               args.fps, args.batch, args.validate_isolation, options,
                               getattr(args,"idle_video",None))

    async def prepare(self, path, audio, seed):
        state = await self.call(prepare, path, audio, seed)
        self.states.add(state.id)
        return state

    async def prepare_call(self, path, seed):
        state = await self.call(prepare_call, path, seed)
        self.states.add(state.id)
        return state

    async def append(self, state, audio):
        state.total_frames = await self.call(append, state.id, audio)

    async def recondition(self, state, displayed):
        await self.call(recondition, state.id, displayed)
        state.cursor = state.total_frames = 0

    async def generate(self, states):
        chunks, metrics, cursors = await self.call(generate, [s.id for s in states])
        for s, cursor in zip(states, cursors):
            s.cursor = cursor
        return chunks, metrics

    async def release(self, state):
        await self.call(release, state.id)
        self.states.discard(state.id)

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
