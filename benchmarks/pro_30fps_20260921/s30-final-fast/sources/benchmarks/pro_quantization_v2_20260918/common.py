"""Shared, side-effect-conscious utilities for PRO quantization v2 tools."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import tempfile
import traceback
from typing import Any, Iterable, Mapping

import torch


ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
DEFAULT_FIXTURES = EXPERIMENT_ROOT / "fixtures.json"
DEFAULT_GPU_LOCK = ROOT / ".gpu-owner.lock"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: str | Path) -> str:
    value = Path(path).resolve()
    try:
        return str(value.relative_to(ROOT))
    except ValueError:
        return str(value)


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def atomic_write_json(path: str | Path, value: Mapping[str, Any] | list[Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=target.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, target)


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def ensure_new_directory(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=False)
    return target


def file_identity(path: str | Path) -> dict[str, Any]:
    value = Path(path)
    if not value.is_file():
        raise FileNotFoundError(f"Required file is missing: {value}")
    return {"path": relative_path(value), "sha256": sha256(value), "bytes": value.stat().st_size}


def snapshot_sources(output: str | Path, sources: Iterable[str | Path]) -> dict[str, str]:
    root = Path(output) / "sources"
    manifest: dict[str, str] = {}
    for source in sorted({Path(item).resolve() for item in sources}):
        if not source.is_file():
            raise FileNotFoundError(f"Cannot snapshot missing source: {source}")
        relative = relative_path(source)
        if Path(relative).is_absolute():
            raise ValueError(f"Source is outside the repository and cannot be snapshotted: {source}")
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        manifest[relative] = sha256(source)
    return manifest


def command_output(args: list[str]) -> tuple[int | None, str]:
    try:
        completed = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    except FileNotFoundError:
        return None, "unavailable"
    return completed.returncode, completed.stdout.strip()


def git_state() -> dict[str, Any]:
    commit_code, commit = command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    status_code, status = command_output(["git", "-C", str(ROOT), "status", "--short"])
    return {
        "commit": commit if commit_code == 0 else None,
        "dirty": bool(status) if status_code == 0 else None,
        "status": status.splitlines() if status_code == 0 else [status],
    }


def gpu_details() -> dict[str, Any]:
    code, text = command_output([
        "nvidia-smi",
        "--query-gpu=uuid,name,compute_cap,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ])
    if code != 0:
        return {"available": False, "error": text}
    rows = []
    for line in text.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) == 6:
            rows.append({
                "uuid": values[0],
                "name": values[1],
                "compute_capability": values[2],
                "visible_vram_mib": int(values[3]),
                "used_vram_mib": int(values[4]),
                "driver": values[5],
            })
    return {"available": True, "devices": rows}


def gpu_processes() -> list[dict[str, str]]:
    code, text = command_output([
        "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"
    ])
    if code not in (0, None):
        return [{"error": text}]
    if not text or text == "No running processes found":
        return []
    rows = []
    for line in text.splitlines():
        values = [item.strip() for item in line.split(",", 2)]
        if len(values) == 3:
            rows.append({"pid": values[0], "process": values[1], "used_memory_mib": values[2]})
    return rows


def package_inventory() -> dict[str, str]:
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages[name.lower()] = distribution.version
    return dict(sorted(packages.items()))


def _ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return None


def environment_overrides() -> dict[str, str]:
    allowed_prefixes = ("CUDA_", "TORCH", "TRITON", "NCCL", "PYTHON")
    forbidden_fragments = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "API_KEY", "CREDENTIAL")
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if key.startswith(allowed_prefixes) and not any(fragment in key.upper() for fragment in forbidden_fragments)
    }


def compiler_cache_state() -> dict[str, Any]:
    candidates = {
        "torchinductor": Path(os.environ.get("TORCHINDUCTOR_CACHE_DIR", "~/.cache/torch/inductor")).expanduser(),
        "triton": Path(os.environ.get("TRITON_CACHE_DIR", "~/.triton/cache")).expanduser(),
    }
    return {
        name: {"path": str(path), "existed_before_run": path.exists()}
        for name, path in candidates.items()
    }


def attention_backend_availability() -> dict[str, bool]:
    return {
        "flash_attention_2": importlib.util.find_spec("flash_attn") is not None,
        "sageattention": importlib.util.find_spec("sageattention") is not None,
    }


def environment_manifest() -> dict[str, Any]:
    nvcc_code, nvcc = command_output(["nvcc", "--version"])
    disk = shutil.disk_usage(ROOT)
    return {
        "date_utc": utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "git": git_state(),
        "gpu": gpu_details(),
        "resident_gpu_processes": gpu_processes(),
        "physical_vram_class": "unverified in this run",
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_compiler": nvcc if nvcc_code == 0 else "unavailable",
        "attention_backend_availability": attention_backend_availability(),
        "available_disk": {"path": str(ROOT), "free_bytes": disk.free, "total_bytes": disk.total},
        "physical_ram_bytes": _ram_bytes(),
        "environment_overrides": environment_overrides(),
        "compiler_caches": compiler_cache_state(),
        "packages": package_inventory(),
    }


def load_fixtures(path: str | Path) -> dict[str, Any]:
    raw = read_json(resolve_path(path))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1 or not isinstance(raw.get("fixtures"), list):
        raise ValueError("fixtures.json must contain schema_version 1 and a fixtures list")
    entries: dict[str, Any] = {}
    required = {"id", "reference", "audio", "sample_rate", "fps", "split", "framing"}
    for entry in raw["fixtures"]:
        if not isinstance(entry, Mapping) or set(entry) - (required | {"license", "provenance", "notes"}):
            raise ValueError("Fixture contains unsupported keys")
        if required - set(entry):
            raise ValueError(f"Fixture is missing keys: {sorted(required - set(entry))}")
        identifier = entry["id"]
        if not isinstance(identifier, str) or not identifier or identifier in entries:
            raise ValueError("Fixture IDs must be unique non-empty strings")
        for field in ("reference", "audio"):
            identity = entry[field]
            if not isinstance(identity, Mapping) or set(identity) != {"path", "sha256"}:
                raise ValueError(f"Fixture {identifier} {field} must contain path and sha256")
            actual = resolve_path(identity["path"])
            if sha256(actual) != identity["sha256"]:
                raise ValueError(f"Fixture hash mismatch for {identifier} {field}: {actual}")
        if not isinstance(entry["sample_rate"], int) or entry["sample_rate"] <= 0:
            raise ValueError(f"Fixture {identifier} has invalid sample_rate")
        if not isinstance(entry["fps"], int) or entry["fps"] <= 0:
            raise ValueError(f"Fixture {identifier} has invalid fps")
        entries[identifier] = dict(entry)
    return {"source": relative_path(resolve_path(path)), "raw": dict(raw), "entries": entries}


def resolve_fixture(path: str | Path, fixture_id: str) -> dict[str, Any]:
    fixtures = load_fixtures(path)
    try:
        fixture = fixtures["entries"][fixture_id]
    except KeyError as error:
        raise ValueError(f"Unknown fixture ID: {fixture_id}") from error
    resolved = dict(fixture)
    for field in ("reference", "audio"):
        resolved[field] = dict(resolved[field])
        resolved[field]["path"] = str(resolve_path(resolved[field]["path"]))
    resolved["fixtures_source"] = fixtures["source"]
    return resolved


def summarize(values: Iterable[float]) -> dict[str, float | int | None]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {"count": 0, "p50": None, "p95": None, "p99": None, "max": None, "min": None}

    def percentile(percent: float) -> float:
        index = (len(ordered) - 1) * percent
        lower, upper = int(index), min(int(index) + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(ordered), "min": ordered[0], "p50": percentile(0.50),
        "p95": percentile(0.95), "p99": percentile(0.99), "max": ordered[-1],
    }


def failure_record(stage: str, error: BaseException) -> dict[str, Any]:
    return {
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "gpu": gpu_details(),
        "resident_gpu_processes": gpu_processes(),
    }