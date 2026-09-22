"""Multi-session (concurrent CUDA stream) generation for the PRO throughput harness.

Why this exists
---------------
At 448x256 every GPU-bound component of the single-stream PRO pipeline scales with
pixel area and the TensorRT stage engines sit at their FP16 floor
(docs/research/PRO_40FPS_ITERATION_LOG_2026-09-21.md, iteration 1). A saturating
matmul load run alongside the pipeline kept 75% of its solo rate while the pipeline
kept 48% -- a sum of 1.23 (iteration 3) -- so roughly a quarter of a second workload
runs in the gaps of the first. The product goal is aggregate FPS across concurrent
WebRTC talking-head streams, so the structural lever is to run N generation sessions
against ONE set of weights and engines, each on its own CUDA stream, and let session
B's DiT fill the gaps of session A's decode.

What is shared and what is per session
--------------------------------------
Shared, read-only during generation: DiT weights (INT8/FP8), VAE weights, the
TensorRT stage engines with their single execution context and workspace, the audio
encoder, and the reference latents / colour reference (same person in every session
here). Sharing one TensorRT execution context across streams is safe ONLY because
``pro_vae_stage_backend.TensorRTStage`` already executes every engine call on one
dedicated engine stream fenced to the caller stream, which serialises workspace use;
its output ping-pong pools are keyed per caller stream for the same reason.

Per session, swapped onto the single pipeline object around each window's issue:

* ``pipeline.generator``            -- the noise stream (a CUDA generator)
* ``pipeline.latent_motion_frames`` -- motion feedback latents, written by generate()
* the overlap-skip decoder cache    -- ``pro_decoder_ops.set_overlap_skip_session``
* the rolling audio history, the CUDA stream, pinned host output buffers, and the
  CUDA timing events that the harness's stage instrumentation records.

Scheduling
----------
Each session owns a CUDA stream that carries its audio embedding, DiT steps, colour
correction, motion encode and device-to-host copy. **All sessions' decodes run on one
shared decode stream, one decode in flight at a time.** The first version gave each
session its own decode as well and ran out of VRAM in warmup (11.2 GiB in use, 9.56
GiB allocated): the caching allocator keeps free blocks per stream, so two decode
streams meant two full sets of decoder activations, two engine output pools, and --
because every TensorRT stage call ``record_stream``s its input/output buffers on the
engine stream -- two windows' worth of queued engine buffers pinned until the GPU
reached them. The decodes were serialised on the engine stream anyway, so sharing the
decode stream costs no overlap that mattered: session B's DiT still runs while session
A decodes, and A's colour/encode/copy tail runs while B decodes.

The host issues windows round-robin. A session's next window is issued only after
its previous window's completion event has fired, and a decode is issued only after
the previous decode (from either session) has completed on the GPU; while the host
waits for that, the GPU is running the other session's DiT. The GPU work of different
sessions is otherwise ordered only by the engine-stream fence.

Host-blocking calls that would have serialised the streams were removed from the
path: the eight ``torch.cuda.synchronize()`` timing fences in
``FlashHeadPipeline.generate`` are gated on ``verbose_timing`` (they only ever
bracketed ``time.time()`` prints), and each window's device-to-host copy is a
``non_blocking`` copy into pinned memory followed by an event instead of ``.cpu()``.
The harness's own instrumentation records CUDA events on the current stream, so the
per-stage seconds remain valid per session; they overlap across sessions and are
therefore NOT additive with wall time.

Correctness check built in
--------------------------
With ``seed_stride == 0`` every session uses the same seed and the same audio, so
their outputs must agree up to kernel nondeterminism. ``session_outputs[i]`` records
the max/mean absolute difference against session 0; a large value means state leaked
between sessions (generator, motion latents, or decoder cache).
"""
from __future__ import annotations

from collections import defaultdict, deque
import hashlib
import time
from typing import Any, Callable

import numpy as np
import torch

AUDIO_HISTORY_SAMPLES = 128000  # 8 s at 16 kHz, matching run.py's deque


class Session:
    def __init__(self, key: str, seed: int, pipeline) -> None:
        self.key = str(key)
        self.seed = int(seed)
        self.stream = torch.cuda.Stream()
        self.generator = torch.Generator(device=pipeline.device)
        self.latent_motion_frames: torch.Tensor | None = None
        self.audio_history: deque[float] = deque()
        self.frames: list[np.ndarray] = []
        self.chunk_times: list[float] = []
        self.stage_totals: defaultdict[str, float] = defaultdict(float)
        self.pending: tuple | None = None
        self.windows = 0

    def reset(self, pipeline) -> None:
        # Mirrors FlashHeadPipeline.reset_person_name for the per-session fields.
        self.generator.manual_seed(self.seed)
        # torch.cuda.Stream() streams are non-blocking: they do NOT synchronise with the
        # default stream. Anything this session reads must therefore be produced on its
        # own stream or ordered after the producing stream. The first version cloned
        # the reference latent on the default stream and read it here a few hundred
        # microseconds later; the DiT was seeded with whatever the block held, and the
        # first window came out non-finite.
        producer = torch.cuda.current_stream(pipeline.device)
        with torch.cuda.stream(self.stream):
            self.stream.wait_stream(producer)
            self.latent_motion_frames = pipeline.ref_img_latent[:, :1].clone()
        self.audio_history = deque([0.0] * AUDIO_HISTORY_SAMPLES, maxlen=AUDIO_HISTORY_SAMPLES)
        self.frames = []
        self.chunk_times = []
        self.stage_totals = defaultdict(float)
        self.pending = None
        self.windows = 0


class SessionScheduler:
    def __init__(
        self,
        pipeline,
        events: list,
        get_audio_embedding: Callable,
        run_pipeline: Callable,
        *,
        sessions: int,
        seed: int,
        seed_stride: int,
        lean_delivery: bool,
        history_frames: int,
        overlap_skip: bool,
        audio_window: tuple[int, int] = (167, 200),
    ) -> None:
        if sessions < 1:
            raise ValueError("sessions must be positive")
        self.pipeline = pipeline
        self.events = events
        self.get_audio_embedding = get_audio_embedding
        self.run_pipeline = run_pipeline
        self.lean_delivery = bool(lean_delivery)
        self.history_frames = int(history_frames)
        self.overlap_skip = bool(overlap_skip)
        self.audio_window = tuple(audio_window)
        self.sessions = [
            Session(str(index), seed + index * seed_stride, pipeline) for index in range(sessions)
        ]
        # Shared decode stream; see the module docstring for why decodes are not
        # per-session. `vae.decode` at this point is the harness's instrumented wrapper
        # around the overlap-skip shim; wrapping it keeps the stage timing events on the
        # decode stream, where the decode actually runs.
        self.decode_stream = torch.cuda.Stream()
        self._decode_done: torch.cuda.Event | None = None
        self.decode_wait_s = 0.0
        self._orig_decode = pipeline.vae.decode
        pipeline.vae.decode = self._decode_on_shared_stream
        # Everything the sessions share (weights, engines, reference latents, prepared
        # rotary/time embeddings) was produced on the default stream during setup.
        # Drain it once so no session stream can observe a half-written shared tensor.
        torch.cuda.synchronize()

    def _decode_on_shared_stream(self, zs):
        caller = torch.cuda.current_stream(zs.device)
        if self._decode_done is not None:
            # One decode in flight. The host blocks here while the GPU runs the
            # previous decode AND this session's already-issued DiT concurrently.
            started = time.perf_counter()
            self._decode_done.synchronize()
            self.decode_wait_s += time.perf_counter() - started
        stream = self.decode_stream
        stream.wait_stream(caller)
        with torch.cuda.stream(stream):
            out = self._orig_decode(zs)
            done = torch.cuda.Event()
            done.record(stream)
        caller.wait_stream(stream)
        # `out` was allocated on the decode stream and is consumed on the caller's
        # stream; without this the allocator could hand its block to the next decode
        # while the caller is still reading it.
        out.record_stream(caller)
        self._decode_done = done
        return out

    def close(self) -> None:
        from soulx_rtc.pro_vae_stage_backend import set_output_pool_key

        set_output_pool_key("0")
        self.pipeline.vae.decode = self._orig_decode

    # -- per-generation reset ---------------------------------------------------------
    def reset(self) -> None:
        pipeline = self.pipeline
        pipeline.reset_person_name(pipeline.person_name)
        for session in self.sessions:
            session.reset(pipeline)
            if self.overlap_skip:
                from soulx_rtc.pro_decoder_ops import reset_overlap_skip, set_overlap_skip_session

                set_overlap_skip_session(pipeline, session.key)
                reset_overlap_skip(pipeline)
        self.events.clear()
        self.decode_wait_s = 0.0

    # -- one window -------------------------------------------------------------------
    def _issue(self, session: Session, audio_slice: np.ndarray, window: int) -> None:
        pipeline = self.pipeline
        with torch.cuda.stream(session.stream):
            pipeline.generator = session.generator
            pipeline.latent_motion_frames = session.latent_motion_frames
            if self.overlap_skip:
                from soulx_rtc.pro_decoder_ops import set_overlap_skip_session

                set_overlap_skip_session(pipeline, session.key)
            from soulx_rtc.pro_vae_stage_backend import set_output_pool_key

            # The engine output pools double as the persisted decoder cache under
            # overlap-skip; they must be per session (see pro_vae_stage_backend).
            set_output_pool_key(session.key)
            started = time.perf_counter()
            session.audio_history.extend(audio_slice.tolist())
            audio_array = np.asarray(session.audio_history, dtype=np.float32)
            embedding = self.get_audio_embedding(pipeline, audio_array, *self.audio_window)
            rgb_device = self.run_pipeline(pipeline, embedding)
            window_device = rgb_device if self.lean_delivery else rgb_device[self.history_frames:]
            # Pinned + non_blocking so the copy is ordered on this session's stream and
            # the host does not block; `.cpu()` would synchronise this stream and stall
            # the issue of the other session's window.
            host = torch.empty(window_device.shape, dtype=window_device.dtype, pin_memory=True)
            host.copy_(window_device, non_blocking=True)
            done = torch.cuda.Event()
            done.record(session.stream)
            # generate() replaced latent_motion_frames with this window's encode output.
            session.latent_motion_frames = pipeline.latent_motion_frames
            issued = list(self.events)
            self.events.clear()
            # rgb_device/window_device must stay alive until the copy has completed.
            session.pending = (done, host, window_device, rgb_device, window, started, issued)

    def _complete(self, session: Session) -> None:
        done, host, window_device, rgb_device, window, started, issued = session.pending
        done.synchronize()
        session.chunk_times.append(time.perf_counter() - started)
        if self.lean_delivery:
            frames = np.array(host.numpy(), copy=True)
        else:
            if not bool(host.isfinite().all()):
                # Distinguish a real non-finite pixel from a stale pinned read.
                device_finite = bool(torch.isfinite(window_device).all())
                raise RuntimeError(
                    f"Session {session.key} window {window} contains nonfinite RGB values "
                    f"(device tensor finite: {device_finite})"
                )
            frames = host.numpy().astype(np.uint8)
        session.frames.append(frames)
        for stage, begin, end in issued:
            session.stage_totals[stage] += begin.elapsed_time(end) / 1000
        session.pending = None
        session.windows += 1
        del window_device, rgb_device, host

    # -- the schedule -----------------------------------------------------------------
    def run(self, slices: np.ndarray, windows: int | None = None) -> float:
        count = len(slices) if windows is None else int(windows)
        started = time.perf_counter()
        for window in range(count):
            for session in self.sessions:
                if session.pending is not None:
                    self._complete(session)
                self._issue(session, slices[window % len(slices)], window)
        for session in self.sessions:
            if session.pending is not None:
                self._complete(session)
        return time.perf_counter() - started

    # -- reporting --------------------------------------------------------------------
    def outputs(self, frames: int) -> list[np.ndarray]:
        result = []
        for session in self.sessions:
            rgb = np.concatenate(session.frames, axis=0)[:frames]
            if len(rgb) != frames:
                raise RuntimeError(
                    f"Session {session.key}: expected {frames} useful frames, received {len(rgb)}"
                )
            if not np.isfinite(rgb).all():
                raise RuntimeError(f"Session {session.key}: generated RGB contains nonfinite values")
            result.append(rgb)
        return result

    def session_outputs(self, outputs: list[np.ndarray]) -> list[dict[str, Any]]:
        reference = outputs[0].astype(np.int16)
        rows = []
        for session, rgb in zip(self.sessions, outputs):
            diff = np.abs(rgb.astype(np.int16) - reference)
            rows.append({
                "session": session.key,
                "seed": session.seed,
                "frames": int(len(rgb)),
                "raw_rgb_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
                "max_abs_diff_vs_session0": int(diff.max()),
                "mean_abs_diff_vs_session0": float(diff.mean()),
                "identical_seed": session.seed == self.sessions[0].seed,
            })
        return rows

    def overlap_skip_states(self) -> dict[str, Any]:
        if not self.overlap_skip:
            return {}
        from soulx_rtc.pro_decoder_ops import overlap_skip_state, set_overlap_skip_session

        states = {}
        for session in self.sessions:
            set_overlap_skip_session(self.pipeline, session.key)
            states[session.key] = overlap_skip_state(self.pipeline)
        return states

    def row(self, repeat: int, generation_s: float, frames: int, summarize: Callable) -> dict[str, Any]:
        total_frames = frames * len(self.sessions)
        stage_sum: defaultdict[str, float] = defaultdict(float)
        for session in self.sessions:
            for stage, seconds in session.stage_totals.items():
                stage_sum[stage] += seconds
        return {
            "repeat": repeat,
            "generation_s": generation_s,
            "sessions": len(self.sessions),
            "frames_per_session": frames,
            "frames_total": total_frames,
            # Aggregate across sessions: the product metric for concurrent streams.
            "useful_fps": total_frames / generation_s,
            "useful_fps_per_session": frames / generation_s,
            "first_window_latency_s": max(s.chunk_times[0] for s in self.sessions),
            "chunk_times_s": [s.chunk_times for s in self.sessions],
            "chunk_distribution_s": summarize([t for s in self.sessions for t in s.chunk_times]),
            # Sum over sessions of per-stream CUDA event spans. Sessions overlap in
            # time, so these are NOT additive with generation_s.
            "stage_seconds": dict(stage_sum),
            "stage_seconds_per_session": [dict(s.stage_totals) for s in self.sessions],
            "stage_seconds_note": (
                "per-session CUDA event spans; DiT/audio/encode on the session's stream, "
                "vae_decode on the shared decode stream; sessions overlap, so the sums "
                "exceed wall time by design"
            ),
            # Host time spent blocked waiting for the previous decode before issuing the
            # next one. Large values are fine (the GPU is busy); they are not idle time.
            "host_decode_wait_s": self.decode_wait_s,
        }
