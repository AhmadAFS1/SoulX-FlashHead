"""Opt-in, bounded latency diagnostics; no CUDA work when not activated.

CUDA events describe inclusive elapsed time on a stream, including waits and
host launch gaps. They are not exclusive kernel durations. Use the accompanying
Torch trace for kernel attribution; never add parent and child durations.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import torch

_ACTIVE = ContextVar("flash_head_latency", default=None)


def latency_scope(name, *, gpu=True, stream=None, **metadata):
    recorder = _ACTIVE.get()
    if recorder is None:
        return nullcontext()
    return recorder.span(name, gpu=gpu, stream=stream, **metadata)


class LatencyRecorder:
    """One diagnostic run; flush only at boundaries outside active spans.

    Metadata contains shapes/dtypes and run identifiers, never tensor contents.
    A bounded buffer drops additional spans explicitly instead of consuming
    unbounded event memory. CPU-only mode is useful for functional tests.
    """

    def __init__(self, output, *, cuda=True, device="cuda", max_events=50000,
                 mode="stages", provenance=None):
        if max_events < 1:
            raise ValueError("max_events must be positive")
        if mode not in ("stages", "modules"):
            raise ValueError("Unknown latency detail mode")
        self.output = Path(output)
        if any((self.output / name).exists() for name in ("latency-events.jsonl", "latency-summary.json")):
            raise FileExistsError("Use a new directory for each latency recording")
        self.cuda = cuda
        self.device = torch.device(device) if cuda else None
        self.max_events = max_events
        self.mode = mode
        self.provenance = provenance or {}
        self._stack = ContextVar(f"latency_stack_{id(self)}", default=())
        self._context = ContextVar(f"latency_context_{id(self)}", default=None)
        self._pending = []
        self._next_id = 0
        self._dropped = 0
        self._written = 0
        self._origin = time.perf_counter()
        self._aggregate = defaultdict(lambda: {
            "calls": 0, "errors": 0, "host_total_ms": 0.0,
            "host_max_ms": 0.0, "cuda_calls": 0,
            "cuda_elapsed_total_ms": 0.0, "cuda_elapsed_max_ms": 0.0,
        })

    @contextmanager
    def activate(self):
        token = _ACTIVE.set(self)
        try:
            yield self
        finally:
            _ACTIVE.reset(token)

    def set_context(self, **values):
        if self._stack.get():
            raise RuntimeError("Cannot change run context inside an active span")
        self._context.set(dict(values))

    @contextmanager
    def span(self, name, *, gpu=True, stream=None, **metadata):
        if len(self._pending) >= self.max_events:
            self._dropped += 1
            yield
            return
        stack = self._stack.get()
        ident = self._next_id
        self._next_id += 1
        row = {
            "id": ident, "parent_id": stack[-1] if stack else None,
            "name": name, "context": dict(self._context.get() or {}),
            "metadata": metadata, "status": "complete",
        }
        # Reserve before entering children, so nested calls cannot exceed cap.
        pending = [row, None, None]
        self._pending.append(pending)
        token = self._stack.set((*stack, ident))
        begin = end = None
        started = time.perf_counter()
        row["host_start_ms"] = (started - self._origin) * 1000
        try:
            if self.cuda and gpu:
                stream = stream if stream is not None else torch.cuda.current_stream(self.device)
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record(stream)
                row["cuda_stream"] = int(stream.cuda_stream)
            with torch.profiler.record_function(f"soulx/{name}"):
                yield
        except BaseException as error:
            row["status"] = "failed"
            row["error_type"] = type(error).__name__
            raise
        finally:
            row["host_ms"] = (time.perf_counter() - started) * 1000
            try:
                if begin is not None:
                    end.record(stream)
                    pending[1:] = [begin, end]
            finally:
                self._stack.reset(token)

    def flush(self, *, status="running"):
        if self._stack.get():
            raise RuntimeError("Flush requires all spans to have ended")
        if any(begin is not None for _, begin, _ in self._pending):
            # One synchronization per explicit flush, never per span.
            torch.cuda.synchronize(self.device)
        self.output.mkdir(parents=True, exist_ok=True)
        rows = []
        for row, begin, end in self._pending:
            row["cuda_elapsed_ms"] = begin.elapsed_time(end) if begin is not None else None
            key = (row["context"].get("phase", "unspecified"), row["name"])
            entry = self._aggregate[key]
            entry["calls"] += 1
            entry["errors"] += row["status"] != "complete"
            entry["host_total_ms"] += row["host_ms"]
            entry["host_max_ms"] = max(entry["host_max_ms"], row["host_ms"])
            if row["cuda_elapsed_ms"] is not None:
                entry["cuda_calls"] += 1
                entry["cuda_elapsed_total_ms"] += row["cuda_elapsed_ms"]
                entry["cuda_elapsed_max_ms"] = max(entry["cuda_elapsed_max_ms"], row["cuda_elapsed_ms"])
            rows.append(json.dumps(row, sort_keys=True))
        with (self.output / "latency-events.jsonl").open("a", encoding="utf-8") as handle:
            if rows:
                handle.write("\n".join(rows) + "\n")
        self._written += len(rows)
        self._pending.clear()
        aggregates = []
        for (phase, name), item in sorted(self._aggregate.items()):
            aggregates.append({
                "phase": phase, "name": name, **item,
                "cuda_elapsed_total_ms": item["cuda_elapsed_total_ms"] if item["cuda_calls"] else None,
                "cuda_elapsed_max_ms": item["cuda_elapsed_max_ms"] if item["cuda_calls"] else None,
                "host_mean_ms": item["host_total_ms"] / item["calls"],
                "cuda_elapsed_mean_ms": (
                    item["cuda_elapsed_total_ms"] / item["cuda_calls"]
                    if item["cuda_calls"] else None
                ),
            })
        result = {
            "schema_version": 1, "status": status, "mode": self.mode,
            "execution": "GPU latency diagnostic" if self.cuda else "CPU-only latency diagnostic",
            "provenance": self.provenance,
            "events": self._written, "dropped_events": self._dropped,
            "complete_coverage": self._dropped == 0,
            "timing_scope": "Inclusive host wall and CUDA stream elapsed times; nested/overlapping spans must not be summed. CUDA intervals include waits and host gaps, not just kernels.",
            "performance_claim": False,
            "aggregates": aggregates,
        }
        target = self.output / "latency-summary.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(target)
        return {"events": "latency-events.jsonl", "summary": target.name,
                "mode": self.mode, "dropped_events": self._dropped}


class PipelineInstrumentation:
    """Reversible wrappers outside compiled callables; eager detail is explicit."""

    def __init__(self):
        self._originals = []
        self._seen = set()

    def wrap(self, owner, attribute, name):
        if (id(owner), attribute) in self._seen:
            return
        original = getattr(owner, attribute, None)
        if not callable(original):
            return
        had_instance_value = attribute in vars(owner)

        @wraps(original)
        def measured(*args, **kwargs):
            if _ACTIVE.get() is None:
                return original(*args, **kwargs)
            tensors = [value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor)][:4]
            metadata = {"inputs": [{"shape": list(v.shape), "dtype": str(v.dtype),
                                     "device": str(v.device)} for v in tensors]}
            with latency_scope(name, **metadata):
                return original(*args, **kwargs)

        setattr(owner, attribute, measured)
        self._originals.append((owner, attribute, original, had_instance_value))
        self._seen.add((id(owner), attribute))

    def attach(self, pipeline, *, modules=False):
        roots = [("dit", pipeline.model), ("audio.encoder", pipeline.audio_encoder),
                 ("vae.model", getattr(pipeline.vae, "model", pipeline.vae))]
        if modules:
            for _, root in roots:
                for module in root.modules():
                    if hasattr(module, "_orig_mod") or hasattr(module.forward, "_torchdynamo_orig_callable"):
                        raise ValueError("Module diagnostics require eager PyTorch modules")
            for method in (pipeline.vae.encode, pipeline.vae.decode):
                if hasattr(method, "_torchdynamo_orig_callable"):
                    raise ValueError("Module diagnostics require eager VAE entrypoints")
        try:
            for owner, attr, name in [
                (pipeline, "preprocess_audio", "audio.total"),
                (pipeline, "generate", "generation.window"),
                (pipeline.model, "forward", "dit.forward"),
                (pipeline.model, "prepare_conditioning", "dit.prepare_conditioning"),
                (pipeline.vae, "encode", "vae.encode"),
                (pipeline.vae, "decode", "vae.decode"),
                (pipeline.audio_encoder, "forward", "audio.encoder"),
            ]:
                self.wrap(owner, attr, name)
            if modules:
                for prefix, root in roots:
                    for path, module in root.named_modules():
                        if path:
                            self.wrap(module, "forward", f"{prefix}.{path}")
                        if getattr(module, "_pro_attention_backend", None) is not None:
                            self.wrap(module, "_pro_attention_backend", f"{prefix}.{path}.attention_core")
                        if getattr(module, "_pro_cross_attention", None) is not None:
                            self.wrap(module, "_pro_cross_attention", f"{prefix}.{path}.attention_core")
        except BaseException:
            self.close()
            raise

    def close(self):
        for owner, attribute, original, had_value in reversed(self._originals):
            if had_value:
                setattr(owner, attribute, original)
            else:
                delattr(owner, attribute)
        self._originals.clear()
        self._seen.clear()
