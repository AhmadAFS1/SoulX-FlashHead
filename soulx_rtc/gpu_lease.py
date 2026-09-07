"""Prevent duplicate GPU owners from this checkout; unrelated apps are untouched."""
import fcntl
from pathlib import Path


def acquire_gpu_lease(path=None):
    path=Path(path) if path else Path(__file__).resolve().parents[1]/".gpu-owner.lock"
    handle=path.open("a")
    try:
        fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise RuntimeError("Another SoulX GPU owner from this checkout is running; stop it before loading a second model") from exc
    return handle  # Closing the file (including process exit) releases the lease.
