"""Run with python -m soulx_rtc.server (one process per GPU)."""
import argparse
import asyncio
import contextlib
import io
import json
import logging
import math
import os
import secrets
import tempfile
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from aiohttp import web
from aiortc import (MediaStreamTrack, RTCConfiguration, RTCIceServer,
                    RTCPeerConnection, RTCRtpSender, RTCSessionDescription)
from aiortc.mediastreams import MediaStreamError
from PIL import Image
from scipy.signal import resample_poly

from .engine import Engine

LOG = logging.getLogger("soulx_rtc")


@dataclass
class Session:
    id: str
    state: object
    audio48: np.ndarray
    fps: int = 25
    startup_delay: float = .25
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=2))
    pc: object = None
    connected: bool = False
    closed: bool = False
    error: str = ""
    created: float = field(default_factory=time.monotonic)
    activated: float = 0.
    last_activity: float = field(default_factory=time.monotonic)
    origin: object = None
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    video_sent: int = 0
    audio_sent: int = 0
    video_drain_sent: int = 0
    audio_drain_samples: int = 0
    first_chunk_s: object = None
    first_video_s: object = None
    generation_done_s: object = None
    stalls_s: float = 0.
    gpu_wall_s: float = 0.
    last_scheduled: float = 0.

    def metrics(self):
        from .codec import sender_encoder_info
        return dict(id=self.id, fps=self.fps, startup_delay_s=self.startup_delay, frames=self.state.total_frames,
                    video_encoders=sender_encoder_info(self.pc),
                    generated_frames=self.state.cursor, video_sent=self.video_sent,
                    audio_samples_sent=self.audio_sent, queued_chunks=self.queue.qsize(),
                    video_drain_sent=self.video_drain_sent,
                    audio_drain_samples=self.audio_drain_samples,
                    connected=self.connected, closed=self.closed, error=self.error,
                    first_chunk_s=self.first_chunk_s, first_video_s=self.first_video_s,
                    generation_done_s=self.generation_done_s,
                    playback_stall_s=self.stalls_s, gpu_wall_s=self.gpu_wall_s)


class VideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, session):
        super().__init__()
        self.session = session
        self.frames = deque()
        self.last_frame = None

    async def recv(self):
        s = self.session
        if s.closed or s.error:
            raise MediaStreamError
        if s.video_sent >= s.state.total_frames:
            # aiortc's receiver releases a video frame on the next RTP timestamp.
            # One explicitly accounted tail frame flushes the final useful frame.
            if s.video_drain_sent or self.last_frame is None:
                raise MediaStreamError
            await asyncio.sleep(max(0, s.origin + s.video_sent / s.fps - time.monotonic()))
            frame = av.VideoFrame.from_ndarray(self.last_frame, format="rgb24")
            frame.pts = round(s.video_sent * 90000 / s.fps)
            frame.time_base = Fraction(1, 90000)
            s.video_drain_sent += 1
            return frame
        if not self.frames:
            # Wait for generated content, never count repeated images as throughput.
            while not s.closed and not s.error:
                try:
                    chunk = await asyncio.wait_for(s.queue.get(), .25)
                    self.frames.extend(chunk)
                    break
                except asyncio.TimeoutError:
                    continue
            if not self.frames:
                raise MediaStreamError
        now = time.monotonic()
        if s.origin is None:
            s.origin = now + s.startup_delay
        target = s.origin + s.video_sent / s.fps
        # Keep A/V on one clock. Starvation slows both instead of audio running ahead.
        if now > target + 1 / s.fps:
            pause = now - target
            s.origin += pause
            s.stalls_s += pause
            target = now
        await asyncio.sleep(max(0, target - time.monotonic()))
        self.last_frame = self.frames.popleft()
        frame = av.VideoFrame.from_ndarray(self.last_frame, format="rgb24")
        frame.pts = round(s.video_sent * 90000 / s.fps)
        frame.time_base = Fraction(1, 90000)
        if s.first_video_s is None:
            s.first_video_s = time.monotonic() - s.activated
        s.video_sent += 1
        s.last_activity = time.monotonic()
        s.changed.set()
        return frame


class AudioTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self, session):
        super().__init__()
        self.session = session

    async def recv(self):
        s = self.session
        if s.closed or s.error:
            raise MediaStreamError
        if s.audio_sent >= len(s.audio48):
            # Flush the receiver's Opus prefetch without counting silence as work.
            if s.audio_drain_samples >= 5 * 960:
                raise MediaStreamError
            pts = s.audio_sent + s.audio_drain_samples
            await asyncio.sleep(max(0, s.origin + pts / 48000 - time.monotonic()))
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 960), np.int16), format="s16", layout="mono")
            frame.sample_rate = 48000
            frame.time_base = Fraction(1, 48000)
            frame.pts = pts
            s.audio_drain_samples += 960
            return frame
        required_video = min(s.state.total_frames, s.audio_sent * s.fps // 48000 + 1)
        while s.origin is None or s.video_sent < required_video:
            s.changed.clear()
            if s.closed or s.error:
                raise MediaStreamError
            try:
                await asyncio.wait_for(s.changed.wait(), .25)
            except asyncio.TimeoutError:
                pass
        await asyncio.sleep(max(0, s.origin + s.audio_sent / 48000 - time.monotonic()))
        samples = s.audio48[s.audio_sent:s.audio_sent + 960]
        if len(samples) < 960:
            samples = np.pad(samples, (0, 960 - len(samples)))
        frame = av.AudioFrame.from_ndarray(samples[None, :], format="s16", layout="mono")
        frame.sample_rate = 48000
        frame.time_base = Fraction(1, 48000)
        frame.pts = s.audio_sent
        s.audio_sent += 960
        s.last_activity = time.monotonic()
        return frame


def decode_audio(data, seconds):
    source = io.BytesIO(data) if isinstance(data, bytes) else data
    with sf.SoundFile(source) as f:
        if f.samplerate > 192000 or f.channels > 2 or f.samplerate < 8000:
            raise ValueError("Use mono/stereo audio at 8–192 kHz")
        audio = f.read(min(f.frames, math.ceil(seconds * f.samplerate)), dtype="float32", always_2d=True)
        sr = f.samplerate
    audio = audio.mean(axis=1)
    if not len(audio) or not np.isfinite(audio).all():
        raise ValueError("Empty or non-finite audio")
    divisor = math.gcd(sr, 16000)
    return np.clip(resample_poly(audio, 16000 // divisor, sr // divisor), -1, 1).astype(np.float32)


class Service:
    def __init__(self, args):
        self.args = args
        self.sessions = {}
        self.rotation = deque()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="soulx-gpu")
        self.ready = False
        self.engine = None
        self.worker = None
        self.task = None
        self.chunks = deque(maxlen=1000)
        self.pending = 0
        self.isolation = None
        self.token = os.environ.get("SOULX_API_TOKEN", "")
        self.ice = json.loads(os.environ.get("SOULX_ICE_SERVERS", "[]"))
        self.idle_asset = None
        import hashlib
        self.source_sha256 = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path("soulx_rtc").glob("*.py"))}
        from .calls import CallService
        self.calls = CallService(self)

    async def gpu(self, function, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, function, *args)

    def mark_unhealthy(self, exc):
        self.ready = False
        for item in list(self.sessions.values())+list(self.calls.items.values()):
            item.error = type(exc).__name__
            item.changed.set()

    async def startup(self, app):
        idle_video = getattr(self.args,"idle_video",None)
        if idle_video:
            import hashlib
            path = Path(idle_video)
            self.idle_asset = dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        from .worker import GPUProcess
        self.worker = GPUProcess()
        self.isolation = await self.worker.start(self.args)
        LOG.info("Session isolation check: %s", self.isolation)
        self.ready = True
        self.task = asyncio.create_task(self.schedule())
        LOG.info("READY size=%s steps=%s batch=%s max_sessions=%s",
                 self.args.size, self.args.steps, self.args.batch, self.args.max_sessions)

    async def close_session(self, s):
        s.closed = True
        s.connected = False
        s.changed.set()
        if s.pc:
            await s.pc.close()
        self.sessions.pop(s.id, None)
        with contextlib.suppress(ValueError):
            self.rotation.remove(s.id)
        while not s.queue.empty():
            s.queue.get_nowait()
        if self.worker:
            await self.worker.release(s.state)

    async def cleanup(self, app):
        await self.calls.cleanup()
        if self.task:
            self.task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.task
        await asyncio.gather(*(self.close_session(s) for s in list(self.sessions.values())))
        self.executor.shutdown(wait=True, cancel_futures=True)
        if self.worker:
            self.worker.close()

    async def schedule(self):
        while True:
            if self.worker and not self.ready:
                await asyncio.sleep(.05)
                continue
            selected = []
            for _ in range(len(self.rotation)):
                sid = self.rotation[0]
                self.rotation.rotate(-1)
                s = self.sessions.get(sid)
                if s is None:
                    continue
                if time.monotonic() - s.last_activity > 60:
                    await self.close_session(s)
                    continue
                if (s.connected and not s.closed and not s.error and
                        s.state.cursor < s.state.total_frames and not s.queue.full()):
                    selected.append(s)
            if not selected:
                call_work = await self.calls.schedule_once()
                if not call_work:
                    await asyncio.sleep(.005)
                continue
            # Stable tie order retains round-robin fairness; generated lead orders
            # eligible peers. No waiting to fill a batch near a playback deadline.
            selected.sort(key=lambda s: ((s.state.cursor-s.video_sent)/s.fps,
                                         s.last_scheduled))
            selected = selected[:self.args.batch]
            for s in selected:
                s.last_scheduled = time.monotonic()
            try:
                if self.worker:
                    chunks, metrics = await self.worker.generate([s.state for s in selected])
                else:  # Lightweight in-process engine for deterministic CPU tests.
                    chunks = await self.gpu(self.engine.generate, [s.state for s in selected])
                    metrics = self.engine.last_metrics.copy()
                self.chunks.append(metrics)
                for s, chunk in zip(selected, chunks):
                    if s.closed:
                        continue
                    now = time.monotonic()
                    if s.first_chunk_s is None:
                        s.first_chunk_s = now - s.activated
                    s.gpu_wall_s += metrics["wall_s"] / len(selected)
                    if s.state.cursor == s.state.total_frames:
                        s.generation_done_s = now - s.activated
                    s.queue.put_nowait(chunk)
                await self.calls.schedule_once()
            except Exception as exc:
                LOG.exception("Generation failed")
                self.mark_unhealthy(exc)

    async def create(self, request):
        if not self.ready:
            raise web.HTTPServiceUnavailable(text="GPU worker is not ready")
        if len(self.sessions) + len(self.calls.items) + self.pending >= self.args.max_sessions:
            raise web.HTTPTooManyRequests(text="Session capacity reached")
        self.pending += 1  # Reserve before the first await; concurrent uploads cannot over-admit.
        try:
            payload, fields = {}, {}
            if request.content_type.startswith("multipart/"):
                reader = await request.multipart()
                size = 0
                async for part in reader:
                    data = bytearray()
                    while block := await part.read_chunk():
                        size += len(block)
                        if size > 20 * 1024**2:
                            raise web.HTTPRequestEntityTooLarge(max_size=20 * 1024**2, actual_size=size)
                        data.extend(block)
                    if part.name in ("audio", "image"):
                        payload[part.name] = bytes(data)
                    elif part.name in ("seconds", "seed"):
                        fields[part.name] = data.decode("utf-8")
            else:
                fields = await request.json() if request.can_read_body else {}
            seconds, seed = float(fields.get("seconds", 10)), int(fields.get("seed", 42))
            if not math.isfinite(seconds) or not 0 < seconds <= 30:
                raise ValueError("seconds must be >0 and <=30")
            if not 0<=seed<2**63:
                raise ValueError("Seed must be an integer from 0 through 2**63-1")
            audio = await asyncio.to_thread(decode_audio,
                payload.get("audio", "examples/podcast_sichuan_16k.wav"), seconds)
            with tempfile.TemporaryDirectory(prefix="soulx-upload-") as folder:
                image_path = "examples/girl.png"
                if "image" in payload:
                    with Image.open(io.BytesIO(payload["image"])) as img:
                        if img.width * img.height > 4096**2:
                            raise ValueError("Image exceeds 16 megapixels")
                        image_path = str(Path(folder) / "avatar.png")
                        img = img.convert("RGB")
                        img.thumbnail((1024, 1024))
                        img.save(image_path)
                try:
                    state = (await self.worker.prepare(image_path, audio, seed) if self.worker
                             else await self.gpu(self.engine.prepare, image_path, audio, seed))
                except RuntimeError as exc:
                    self.mark_unhealthy(exc)
                    raise web.HTTPServiceUnavailable(text="GPU preparation failed; worker restart required") from exc
            audio48 = (resample_poly(audio, 3, 1).clip(-1, 1) * 32767).astype(np.int16)
            sid = secrets.token_urlsafe(18)
            session = Session(sid, state, audio48, fps=self.args.fps)
            self.sessions[sid] = session
            self.rotation.append(sid)
            return web.json_response(session.metrics(), status=201)
        except (ValueError, TypeError, AttributeError, OSError, RuntimeError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        finally:
            self.pending -= 1

    def get(self, request):
        s = self.sessions.get(request.match_info["sid"])
        if s is None:
            raise web.HTTPNotFound(text="Unknown session")
        return s

    async def offer(self, request):
        s = self.get(request)
        if s.pc is not None:
            raise web.HTTPConflict(text="Session already has a peer")
        params = await request.json()
        if params.get("type") != "offer":
            raise web.HTTPBadRequest(text="Expected an SDP offer")
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(**i) for i in self.ice]))
        s.pc = pc  # Reserve before awaits.
        @pc.on("connectionstatechange")
        async def changed():
            if pc.connectionState == "connected":
                s.connected = True
                s.activated = time.monotonic()
                s.last_activity = s.activated
            elif pc.connectionState in ("failed", "closed") and not s.closed:
                await self.close_session(s)
        try:
            pc.addTrack(VideoTrack(s))
            pc.addTrack(AudioTrack(s))
            # H264 is browser-friendly; software encoding stays outside the GPU worker.
            codecs = [c for c in RTCRtpSender.getCapabilities("video").codecs
                      if c.mimeType.lower() == "video/h264"]
            for transceiver in pc.getTransceivers():
                if transceiver.kind == "video":
                    transceiver.setCodecPreferences(codecs)
            await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type="offer"))
            await asyncio.wait_for(pc.setLocalDescription(await pc.createAnswer()), timeout=20)
            return web.json_response({"sdp": pc.localDescription.sdp, "type": "answer"})
        except Exception:
            await self.close_session(s)
            raise

    async def stats(self, request):
        return web.json_response(self.get(request).metrics())

    async def delete(self, request):
        await self.close_session(self.get(request))
        return web.Response(status=204)

    async def health(self, request):
        from .metrics import process_rss_mib
        return web.json_response(dict(ready=self.ready, size=self.args.size, steps=self.args.steps,
            server_rss_mib=process_rss_mib(),
            fps=self.args.fps,
            width=getattr(self.args, "width", None) or self.args.size,
            height=getattr(self.args, "height", None) or self.args.size,
            optimized=getattr(self.args, "optimized", False),
            real_rope=getattr(self.args, "real_rope", False),
            memory_mode=getattr(self.args, "memory_mode", "default"),
            idle_policy=getattr(self.args,"idle_policy","source"), idle_asset=self.idle_asset,
            max_active_calls=getattr(self.args,"max_active_calls",1),
            trt_ffn=bool(getattr(self.args,"trt_ffn",None)), trt_vae=bool(getattr(self.args,"trt_vae",None)),
            h264_preset=getattr(self.args,"h264_preset","upstream"),
            source_sha256=self.source_sha256,
            calls=len(self.calls.items),
            gpu_states=len(self.worker.states) if self.worker else None,
            batch=self.args.batch, sessions=len(self.sessions), max_sessions=self.args.max_sessions,
            queued_limit_chunks_per_session=2, session_isolation_test=self.isolation,
            recent_chunks=list(self.chunks)[-10:]))

    async def config(self, request):
        return web.json_response({"iceServers": self.ice})


def make_app(service):
    @web.middleware
    async def auth(request, handler):
        if request.path != "/" and service.token:
            supplied = request.headers.get("Authorization", "")
            if not secrets.compare_digest(supplied, "Bearer " + service.token):
                raise web.HTTPUnauthorized(text="Bearer token required")
        return await handler(request)
    app = web.Application(middlewares=[auth], client_max_size=20 * 1024**2)
    async def index(request):
        return web.FileResponse(Path(__file__).with_name("index.html"))
    app.add_routes([web.get("/", index), web.get("/health", service.health),
        web.get("/config", service.config), web.post("/sessions", service.create),
        web.get("/sessions/{sid}", service.stats), web.delete("/sessions/{sid}", service.delete),
        web.post("/sessions/{sid}/offer", service.offer)])
    app.add_routes(service.calls.routes())
    app.on_startup.append(service.startup)
    app.on_cleanup.append(service.cleanup)
    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--size", type=int, choices=[256, 384, 512], default=512)
    ap.add_argument("--width", type=int)
    ap.add_argument("--height", type=int)
    ap.add_argument("--optimized", action="store_true", help="Cache chunk conditioning and profile/reference constants")
    ap.add_argument("--real-rope", action="store_true", help="Experimental FP32 real rotary math (requires --optimized)")
    ap.add_argument("--profile", action="store_true", help="Detailed CUDA event timings")
    ap.add_argument("--trt-ffn", help="Trusted exact-shape TensorRT FFN artifact directory (experimental)")
    ap.add_argument("--trt-vae", help="Trusted exact-shape TensorRT VAE decoder engine (experimental)")
    ap.add_argument("--memory-mode", choices=["default", "compact", "reference", "staged"], default="default")
    ap.add_argument("--idle-video", help="Approved local idle asset for persistent calls; clients cannot select filesystem paths")
    ap.add_argument("--idle-policy", choices=["source", "hold", "generate"], default="source",
                    help="Source replay saves GPU; generated silence preserves recurrence but spends GPU while idle")
    ap.add_argument("--max-active-calls", type=int, default=1,
                    help="Bound simultaneously rendering calls independently from connected peers")
    ap.add_argument("--steps", type=int, choices=[2, 4], default=4)
    ap.add_argument("--fps", type=int, choices=[15, 20, 24, 25], default=25)
    ap.add_argument("--batch", type=int, choices=[1, 2, 4], default=1)
    ap.add_argument("--max-sessions", type=int, default=10)
    ap.add_argument("--eager", action="store_true")
    ap.add_argument("--validate-isolation", action="store_true")
    ap.add_argument("--access-log", action="store_true", help="Log every HTTP request (off for polling/load tests)")
    ap.add_argument("--h264-preset", choices=["upstream","veryfast"], default="upstream",
                    help="Optional pinned low-latency CPU H264 encoder; not neural FPS")
    args = ap.parse_args()
    from .engine import validate_geometry
    try:
        args.width, args.height = validate_geometry(args.size, args.width, args.height)
        if args.real_rope and not args.optimized:
            raise ValueError("--real-rope requires --optimized")
        if args.trt_ffn and (args.batch != 1 or args.memory_mode == "staged"):
            raise ValueError("TensorRT FFN profiles require batch one and no weight offload")
        if not 1 <= args.max_active_calls <= args.max_sessions:
            raise ValueError("max-active-calls must be between 1 and max-sessions")
    except ValueError as exc:
        ap.error(str(exc))
    if args.host not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("SOULX_API_TOKEN"):
        ap.error("Set SOULX_API_TOKEN before exposing this GPU service beyond localhost")
    logging.basicConfig(level=logging.INFO)
    if args.h264_preset != "upstream":
        from .codec import install_encoder_factory
        install_encoder_factory(args.fps,args.h264_preset)
    if (args.size, args.steps, args.fps) == (256, 2, 15):
        LOG.warning("STRESS TEST ONLY: 256/2/15 passed media delivery but failed visual quality (facial artifacts)")
    web.run_app(make_app(Service(args)), host=args.host, port=args.port,
                access_log=logging.getLogger("aiohttp.access") if args.access_log else None)


if __name__ == "__main__":
    main()
