"""Bounded, shared CPU RGB clips. Active leases cannot be evicted.

One serialized loader reserves bytes before allocating immutable frames. Playback
is index-only; it neither decodes nor seeks on the event loop. No GPU allocations.
"""
import hashlib
import io
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass
class IdleClip:
    frames: tuple
    fps: float
    nbytes: int
    users: int = 1

    def at(self, index, output_fps):
        return self.frames[math.floor(index*self.fps/output_fps) % len(self.frames)]


class IdleCache:
    def __init__(self, max_bytes=512*2**20):
        if max_bytes < 0:
            raise ValueError('Idle cache budget must be nonnegative')
        self.max_bytes = int(max_bytes)
        self.entries = OrderedDict()
        self.lock, self.loader = threading.Lock(), threading.Lock()
        self.used = self.reserved = self.hits = self.misses = self.fallbacks = 0

    def stats(self):
        with self.lock:
            return dict(bytes=self.used, reserved_bytes=self.reserved, limit_bytes=self.max_bytes,
                        entries=len(self.entries), leases=sum(c.users for c in self.entries.values()),
                        hits=self.hits, misses=self.misses, fallbacks=self.fallbacks)

    def _reserve(self, count):
        with self.lock:
            while self.used+self.reserved+count > self.max_bytes:
                key = next((k for k,c in self.entries.items() if c.users == 0), None)
                if key is None:
                    return False
                self.used -= self.entries.pop(key).nbytes
            self.reserved += count
            return True

    def acquire(self, path, width, height):
        # Bound compressed input too; large/un-cacheable approved assets use the
        # existing bounded decoder. Read once so key and decoded bytes agree.
        with self.loader:
            if not self.max_bytes or Path(path).stat().st_size > 64*2**20:
                with self.lock: self.fallbacks += 1
                return None
            with open(path,'rb') as source:
                data=source.read(64*2**20+1)
            if len(data)>64*2**20:
                with self.lock: self.fallbacks += 1
                return None
            key=(hashlib.sha256(data).digest(),width,height,'rgb24-v1')
            with self.lock:
                if key in self.entries:
                    clip=self.entries[key];clip.users += 1
                    self.entries.move_to_end(key);self.hits += 1
                    return clip
                self.misses += 1
            frames=[]
            try:
                with av.open(io.BytesIO(data)) as source:
                    fps=float(source.streams.video[0].average_rate)
                    if not math.isfinite(fps) or fps<=0:
                        raise ValueError('Invalid idle source frame rate')
                    for frame in source.decode(video=0):
                        if not self._reserve(width*height*3):
                            with self.lock: self.fallbacks += 1
                            return None
                        rgb=frame.reformat(width=width,height=height,format='rgb24').to_ndarray()
                        # bytes-backed arrays cannot accidentally be made writable.
                        frames.append(np.frombuffer(rgb.tobytes(),np.uint8).reshape(height,width,3))
                if not frames:
                    raise ValueError('Empty idle video')
                with self.lock:
                    clip=IdleClip(tuple(frames),fps,self.reserved)
                    self.used += self.reserved;self.reserved=0
                    self.entries[key]=clip
                    return clip
            finally:
                with self.lock: self.reserved=0

    def release(self, clip):
        if clip is not None:
            with self.lock:
                if clip.users<=0:
                    raise RuntimeError('Idle cache lease released twice')
                clip.users -= 1
