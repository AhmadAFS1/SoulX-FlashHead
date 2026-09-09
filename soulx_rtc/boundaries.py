"""Sender-side RGB boundaries; not a receiver display acknowledgement."""
import hashlib
import math
from collections import deque

import numpy as np


def digest(rgb):
    return hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()


class Boundaries:
    def __init__(self, frames=4):
        if frames < 2:
            raise ValueError("A boundary needs at least two endpoint frames")
        self.frames = frames
        self.pending = None
        self.events = deque(maxlen=128)

    def begin(self, kind, anchor, turn_id, epoch, frames=None):
        self.cancel()
        self.pending = dict(kind=kind, anchor=anchor.copy(), turn_id=turn_id,
                            epoch=epoch, index=0, frames=frames or self.frames)

    def cancel(self):
        if self.pending:
            p = self.pending
            self.events.append(dict(kind=p['kind'], turn_id=p['turn_id'],
                                    epoch=p['epoch'], status='cancelled', frames=p['index']))
        self.pending = None

    def apply(self, target, epoch, pts):
        p = self.pending
        if p is None:
            return target
        if epoch != p['epoch']:
            self.cancel()
            return target
        i, n = p['index'], p['frames']
        alpha = (1-math.cos(math.pi*i/(n-1)))/2
        if i == 0:
            output = p['anchor'].copy()
        elif i == n-1:
            output = target.copy()
        else:
            output = np.rint(p['anchor'].astype(np.float32)*(1-alpha)
                             + target.astype(np.float32)*alpha).astype(np.uint8)
        event = dict(kind=p['kind'], turn_id=p['turn_id'], epoch=epoch,
                     index=i, frames=n, pts=pts, alpha=alpha, status='frame')
        # Hash only endpoints: bounded telemetry, no per-frame full-image history.
        if i == 0 or i == n-1:
            expected = p['anchor'] if i == 0 else target
            event.update(expected_sha256=digest(expected), output_sha256=digest(output),
                         exact=bool(np.array_equal(output, expected)))
        self.events.append(event)
        p['index'] += 1
        if p['index'] == n:
            self.pending = None
        return output
