"""Make direct v2 CLI execution safe from sibling-module shadowing."""
from __future__ import annotations

from pathlib import Path
import sys


def bootstrap_script_path(script_file: str) -> None:
    """Prioritize the repository root before importing heavyweight dependencies.

    ``python benchmarks/.../profile.py`` otherwise puts this directory first,
    causing ``import profile`` inside Python's ``cProfile`` to import the v2
    command instead of the standard library module.
    """
    script_directory = Path(script_file).resolve().parent
    repository_root = script_directory.parents[1]
    retained = []
    for entry in sys.path:
        candidate = Path(entry or ".").resolve()
        if candidate not in (script_directory, repository_root):
            retained.append(entry)
    sys.path[:] = [str(repository_root), *retained]