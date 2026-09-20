"""Shared pytest configuration.

Two jobs:

1. Put the repo root on ``sys.path`` so ``flash_head``/``soulx_rtc``/``benchmarks``
   import regardless of the invocation directory.
2. Skip collection of test modules whose third-party imports are not installed.

(2) matters because the inference dependencies (torch, av, aiortc, soundfile,
numpy) are absent on a CPU-only developer machine. A missing module raises at
COLLECTION time, before any marker is consulted, so ``-m 'not heavy'`` cannot
rescue it -- the whole run errors out. Skipping by probing what is actually
importable keeps the stdlib-only guards runnable everywhere and the full suite
runnable on the box that has the dependencies.

Nothing here weakens a test. A skipped module is reported as skipped.

GPU/evidence note: every test in this suite is a `[CPU]` correctness or static
check. None of them measures speed, and passing them never qualifies a recipe
for promotion -- that requires a measured run on the recorded GPU.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Third-party modules a test file may import at module scope.
_OPTIONAL = (
    "torch", "numpy", "av", "aiohttp", "aiortc", "soundfile", "cv2",
    "einops", "transformers", "onnx", "tensorrt", "mediapipe", "PIL",
    "scipy", "yaml", "loguru",
)


def _available(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


_MISSING = frozenset(name for name in _OPTIONAL if not _available(name))


_FIRST_PARTY = ("soulx_rtc", "flash_head", "benchmarks")

# Cache of first-party module -> the optional dependency that blocked its import.
_BLOCKED_BY = {}


def _module_scope_imports(path):
    """Optional dependencies a test file needs, directly or transitively.

    Direct imports are read from column-0 import lines. First-party imports are
    resolved by actually importing them and catching ModuleNotFoundError, which
    is the only reliable way to see a dependency one level down -- find_spec
    does not execute a module, so it cannot tell that soulx_rtc.avatars needs
    numpy.
    """
    names = set()
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return names
    for line in text.splitlines():
        if line.startswith("import "):
            head = line[len("import "):]
        elif line.startswith("from "):
            head = line[len("from "):]
        else:
            continue  # indented imports are deferred; they are the file's problem
        target = head.split()[0].strip("(,")
        root = target.split(".")[0]
        if root in _OPTIONAL:
            names.add(root)
        elif root in _FIRST_PARTY:
            blocker = _first_party_blocker(target)
            if blocker:
                names.add(blocker)
    return names


def _first_party_blocker(dotted):
    """Return the optional dependency that prevents importing ``dotted``, if any."""
    if dotted in _BLOCKED_BY:
        return _BLOCKED_BY[dotted]
    blocker = None
    try:
        importlib.import_module(dotted)
    except ModuleNotFoundError as error:
        missing = (error.name or "").split(".")[0]
        blocker = missing if missing in _OPTIONAL else None
    except Exception:
        # Any other import-time failure is the test module's own problem; let
        # collection surface it rather than silently hiding the file.
        blocker = None
    _BLOCKED_BY[dotted] = blocker
    return blocker


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "heavy: needs torch/media/network dependencies; not part of the "
        "stdlib-only guard set that runs on a CPU-only developer machine",
    )


def pytest_collection_modifyitems(config, items):
    """Mark anything from a module that needed an optional dependency."""
    for item in items:
        path = Path(str(getattr(item, "fspath", "")))
        if path.name.startswith("test_") and _module_scope_imports(path):
            item.add_marker(pytest.mark.heavy)


def pytest_ignore_collect(collection_path, config):
    """Skip modules whose module-scope third-party imports are unavailable."""
    path = Path(str(collection_path))
    if path.suffix != ".py" or not path.name.startswith("test_"):
        return None
    needed = _module_scope_imports(path)
    unavailable = needed & _MISSING
    if unavailable:
        return True
    return None


def pytest_report_header(config):
    if _MISSING:
        return (
            f"soulx: skipping modules needing {sorted(_MISSING)} "
            "(CPU-only machine; run the full suite where they are installed)"
        )
    return "soulx: all optional dependencies present; full suite collectable"
