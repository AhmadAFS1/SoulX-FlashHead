"""Persistent local-call API with bounded turns, shared GPU and stable RTP clocks.

Idle media is explicitly not generated FPS. Reconditioning uses last sent RGB,
not a claimed receiver acknowledgement. Visual live/idle seams need review.
"""
import asyncio
import contextlib
import hashlib
import io
import json
import math
import secrets
import tempfile
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import soundfile as sf
from aiohttp import web
from aiortc import RTCConfiguration, RTCIceServer, RTCPeerConnection, RTCRtpSender, RTCSessionDescription
from aiortc.mediastreams import AudioStreamTrack, VideoStreamTrack, MediaStreamError
from PIL import Image
from scipy.signal import resample_poly


@dataclass
class Turn:
    id: str
    digest: str
    audio: np.ndarray
    audio48: np.ndarray
    frames: int
    status: str = "queued"
    start_frame: object = None
    start_sample: object = None
    video_sent: int = 0
    audio_sent: int = 0
    accepted: float = field(default_factory=time.monotonic)
    first_media_s: object = None
    finished_s: object = None
    useful_samples: int = 0
    synthetic_idle: bool = False

    def __post_init__(self):
        self.useful_samples = len(self.audio48)

    def release_audio(self):
        self.audio = np.zeros(0, np.float32)
        self.audio48 = np.zeros(0, np.int16)

    def summary(self):
        return dict(id=self.id, status=self.status, frames=self.frames,
                    synthetic_idle=self.synthetic_idle,
                    useful_audio_samples=self.useful_samples, video_sent=self.video_sent,
                    audio_sent=self.audio_sent, start_frame=self.start_frame,
                    start_sample=self.start_sample, first_media_s=self.first_media_s,
                    finished_s=self.finished_s)


class IdleVideo:
    """One decoder per peer; no unbounded whole-video RGB cache."""
    def __init__(self, path, width, height, output_fps, anchor):
        self.path, self.width, self.height = path, width, height
        self.output_fps, self.anchor = output_fps, anchor
        self.container = self.frames = None
        self.position = -1
        self.last = anchor
        self.source_fps = output_fps
        self.lock = threading.Lock()

    def next(self, index):
        with self.lock:
            return self._next(index)

    def _next(self, index):
        if not self.path:
            return self.anchor
        if self.container is None:
            self.container = av.open(self.path)
            self.source_fps = float(self.container.streams.video[0].average_rate)
            self.frames = self.container.decode(video=0)
        desired = math.floor(index * self.source_fps / self.output_fps)
        while self.position < desired:
            frame = next(self.frames, None)
            if frame is None:
                self.container.seek(0)
                self.frames = self.container.decode(video=0)
                frame = next(self.frames)
            self.last = frame.reformat(width=self.width, height=self.height, format="rgb24").to_ndarray()
            self.position += 1
        return self.last

    def close(self):
        with self.lock:
            if self.container:
                self.container.close()
                self.container = None


@dataclass
class Call:
    id: str
    state: object
    fps: int
    idle: IdleVideo
    pc: object = None
    connected: bool = False
    closed: bool = False
    error: str = ""
    created: float = field(default_factory=time.monotonic)
    origin: object = None
    epoch: int = 0
    active: object = None
    queue: deque = field(default_factory=deque)
    turns: OrderedDict = field(default_factory=OrderedDict)
    chunks: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=2))
    current_frames: deque = field(default_factory=deque)
    sent_history: deque = field(default_factory=lambda: deque(maxlen=9))
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    control_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    video_sent: int = 0
    video_clock: int = 0
    missed_video_slots: int = 0
    audio_sent: int = 0
    generated_frames: int = 0
    audio_hold_samples: int = 0
    idle_frames: int = 0
    held_frames: int = 0
    underrun_frames: int = 0
    last_scheduled: float = 0.
    idle_index: int = 0
    negotiations: int = 0
    idle_policy: str = "source"
    generated_idle_frames: int = 0
    disconnected_at: object = None

    def metrics(self):
        from .codec import sender_encoder_info
        return dict(id=self.id, fps=self.fps, width=self.idle.width, height=self.idle.height,
                    connected=self.connected, closed=self.closed, epoch=self.epoch, error=self.error,
                    active_turn=self.active.id if self.active else None,
                    queued_turns=len(self.queue), queued_chunks=self.chunks.qsize(),
                    video_sent=self.video_sent, audio_samples_sent=self.audio_sent,
                    video_clock_slots=self.video_clock, missed_video_slots=self.missed_video_slots,
                    generated_frames=self.generated_frames, idle_frames=self.idle_frames,
                    generated_idle_frames=self.generated_idle_frames, idle_policy=self.idle_policy,
                    generated_frame_note="Includes new whole-chunk tail padding and neural idle; excludes held/source-idle frames",
                    completed_useful_video_frames=sum(math.ceil(t.useful_samples*self.fps/48000)
                        for t in self.turns.values() if t.status=="complete"),
                    held_frames=self.held_frames, underrun_frames=self.underrun_frames,
                    audio_hold_samples=self.audio_hold_samples,
                    audio_scheduling_lead_samples=960,
                    negotiations=self.negotiations, track_replacements=0,
                    video_encoders=sender_encoder_info(self.pc),
                    total_turns=len(self.turns), completed_turns=sum(t.status=="complete" for t in self.turns.values()),
                    turns=[t.summary() for t in list(self.turns.values())[-64:]],
                    continuity="stable transport; generated endpoint/receiver-playout equivalence not guaranteed")

    def finish_if_drained(self):
        turn = self.active
        if turn and turn.video_sent >= turn.frames and turn.audio_sent >= turn.useful_samples:
            turn.status = "complete"
            turn.finished_s = time.monotonic()-turn.accepted
            turn.release_audio()
            self.active = None
            self.changed.set()


class CallVideoTrack(VideoStreamTrack):
    def __init__(self, call):
        super().__init__()
        self.call = call

    async def recv(self):
        c = self.call
        if c.closed or c.error:
            raise MediaStreamError
        if c.origin is None:
            c.origin = time.monotonic()+.1
        # RTP follows elapsed call time, not a backlog of old idle frames.
        # Late encoding may miss transport slots; do not emit a catch-up burst
        # or make a new turn wait for video to catch the already-running audio.
        elapsed_slot=max(0,math.floor((time.monotonic()-c.origin)*c.fps))
        if elapsed_slot>c.video_clock:
            missed=elapsed_slot-c.video_clock
            c.missed_video_slots+=missed
            if c.active and c.active.start_frame is not None:
                c.underrun_frames+=missed
            c.video_clock=elapsed_slot
        await asyncio.sleep(max(0, c.origin+c.video_clock/c.fps-time.monotonic()))
        turn = c.active
        if turn and not c.current_frames and not c.chunks.empty():
            epoch, chunk = c.chunks.get_nowait()
            if epoch == c.epoch:
                c.current_frames.extend(chunk)
        rgb = None
        if turn and turn.video_sent < turn.frames and c.current_frames:
            if turn.start_frame is None:
                # Anchor both senders to a future shared 20-ms boundary. Ensure
                # audio cannot have already passed it, even after event-loop lag.
                start = max(c.video_clock+math.ceil(.12*c.fps), math.ceil(c.audio_sent*c.fps/48000))
                turn.start_frame = start
                turn.start_sample = math.ceil(start*48000/c.fps/960)*960
                turn.status = "armed"
            if c.video_clock >= turn.start_frame:
                rgb = c.current_frames.popleft()
                turn.video_sent += 1
                turn.status = "speaking" if turn.audio_sent < turn.useful_samples else "draining"
                if turn.first_media_s is None:
                    turn.first_media_s = time.monotonic()-turn.accepted
        if rgb is None:
            if turn and turn.start_frame is not None and c.video_clock >= turn.start_frame and turn.video_sent < turn.frames:
                # Keep wire clock monotonic, account repetitions, and hold audio.
                c.underrun_frames += 1
                rgb = c.sent_history[-1] if c.sent_history else c.idle.anchor
                c.held_frames += 1
            elif turn and turn.start_frame is not None and c.video_clock >= turn.start_frame:
                # Last speech image remains until the final audio packet is sent.
                rgb = c.sent_history[-1] if c.sent_history else c.idle.anchor
                c.held_frames += 1
            elif c.idle.path:
                epoch = c.epoch
                rgb = await asyncio.to_thread(c.idle.next, c.idle_index)
                if epoch != c.epoch:
                    rgb = c.sent_history[-1] if c.sent_history else c.idle.anchor
                c.idle_index += 1
                c.idle_frames += 1
            else:
                rgb = c.sent_history[-1] if c.sent_history else c.idle.anchor
                c.held_frames += 1
        c.sent_history.append(rgb)
        frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
        frame.pts, frame.time_base = round(c.video_clock*90000/c.fps), Fraction(1,90000)
        c.video_clock += 1
        c.video_sent += 1
        c.changed.set()
        c.finish_if_drained()
        return frame


class CallAudioTrack(AudioStreamTrack):
    def __init__(self, call):
        super().__init__()
        self.call = call

    async def recv(self):
        c = self.call
        while c.origin is None:
            if c.closed or c.error:
                raise MediaStreamError
            await asyncio.sleep(.005)
        if c.closed or c.error:
            raise MediaStreamError
        await asyncio.sleep(max(0,c.origin+c.audio_sent/48000-time.monotonic()))
        samples = np.zeros(960, np.int16)
        t = c.active
        if t and t.start_sample is not None and c.audio_sent >= t.start_sample:
            # Do not let speech run ahead when generation starves. Silence is
            # counted only as transport, never useful samples or generated FPS.
            # Audio and video callbacks race at a shared 40-ms boundary. Permit
            # at most one 20-ms packet of lead after video has begun, otherwise
            # a healthy stream can insert a spurious silence packet. The bound
            # still stops speech advancing indefinitely through a video stall.
            video_horizon = round(t.video_sent*48000/c.fps)+(960 if t.video_sent else 0)
            requested = min(960, t.useful_samples-t.audio_sent)
            available = min(requested, max(0, video_horizon-t.audio_sent))
            if not t.synthetic_idle:
                c.audio_hold_samples += requested-available
            if available > 0:
                if not t.synthetic_idle:
                    samples[:available] = t.audio48[t.audio_sent:t.audio_sent+available]
                t.audio_sent += available
        frame = av.AudioFrame.from_ndarray(samples[None], format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = 48000, c.audio_sent, Fraction(1,48000)
        c.audio_sent += 960
        c.finish_if_drained()
        return frame


class CallService:
    def __init__(self, service):
        self.service = service
        self.items = {}
        self.pending = 0

    def routes(self):
        return [web.post("/calls",self.create), web.get("/calls/{cid}",self.stats),
                web.post("/calls/{cid}/offer",self.offer), web.post("/calls/{cid}/turns",self.append),
                web.post("/calls/{cid}/interrupt",self.interrupt), web.delete("/calls/{cid}",self.delete)]

    def get(self, request):
        call = self.items.get(request.match_info["cid"])
        if call is None:
            raise web.HTTPNotFound(text="Unknown call")
        return call

    async def create(self, request):
        service = self.service
        if not service.ready:
            raise web.HTTPServiceUnavailable(text="GPU worker is not ready")
        if len(self.items)+len(service.sessions)+service.pending >= service.args.max_sessions:
            raise web.HTTPTooManyRequests(text="Connected peer capacity reached")
        service.pending += 1
        try:
            params = await request.json() if request.can_read_body else {}
            if not isinstance(params,dict):
                raise ValueError("Expected a JSON object")
            seed = int(params.get("seed",50))
            if not 0<=seed<2**63:
                raise ValueError("Seed must be an integer from 0 through 2**63-1")
            width = getattr(service.args,"width",None) or service.args.size
            height = getattr(service.args,"height",None) or service.args.size
            path = getattr(service.args,"idle_video",None)
            if path:
                def first():
                    with av.open(path) as src:
                        return next(src.decode(video=0)).reformat(width=width,height=height,format="rgb24").to_ndarray()
                anchor = await asyncio.to_thread(first)
            else:
                with Image.open("examples/girl.png") as image:
                    from flash_head.utils.utils import resize_and_centercrop
                    anchor = resize_and_centercrop(image.convert("RGB"),(height,width))[0,:,0].permute(1,2,0).numpy()
            with tempfile.TemporaryDirectory(prefix="soulx-call-") as folder:
                image = str(Path(folder)/"anchor.png")
                Image.fromarray(anchor).save(image)
                try:
                    state = (await service.worker.prepare_call(image,seed) if service.worker else
                             await service.gpu(service.engine.prepare_call,image,seed))
                except RuntimeError as exc:
                    service.mark_unhealthy(exc)
                    raise web.HTTPServiceUnavailable(text="GPU preparation failed; worker restart required") from exc
            policy=getattr(service.args,"idle_policy","source")
            call = Call(secrets.token_urlsafe(18),state,service.args.fps,
                        IdleVideo(path if policy=="source" else None,width,height,service.args.fps,anchor),
                        idle_policy=policy)
            self.items[call.id] = call
            return web.json_response(call.metrics(),status=201)
        except (ValueError, TypeError, OSError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        finally:
            service.pending -= 1

    async def offer(self, request):
        c, service = self.get(request), self.service
        if c.pc is not None:
            raise web.HTTPConflict(text="Call already negotiated")
        params = await request.json()
        if params.get("type") != "offer":
            raise web.HTTPBadRequest(text="Expected offer")
        pc = c.pc = RTCPeerConnection(RTCConfiguration(iceServers=[RTCIceServer(**i) for i in service.ice]))
        @pc.on("connectionstatechange")
        async def changed():
            c.connected = pc.connectionState == "connected"
            if c.connected:
                c.disconnected_at = None
            elif c.disconnected_at is None:
                c.disconnected_at = time.monotonic()
            if pc.connectionState in ("closed","failed") and not c.closed:
                await self.close(c)
        try:
            pc.addTrack(CallVideoTrack(c))
            pc.addTrack(CallAudioTrack(c))
            codecs = [x for x in RTCRtpSender.getCapabilities("video").codecs if x.mimeType.lower()=="video/h264"]
            for t in pc.getTransceivers():
                if t.kind == "video":
                    t.setCodecPreferences(codecs)
            # aiortc computes common codecs while applying the remote offer.
            # Preferences set afterwards do not retroactively force H264.
            await pc.setRemoteDescription(RTCSessionDescription(**params))
            await asyncio.wait_for(pc.setLocalDescription(await pc.createAnswer()),20)
            c.negotiations += 1
            return web.json_response(dict(sdp=pc.localDescription.sdp,type="answer"))
        except Exception:
            await self.close(c)
            raise

    async def append(self, request):
        from .server import decode_audio
        if not self.service.ready:
            raise web.HTTPServiceUnavailable(text="GPU worker is not ready")
        c = self.get(request)
        if c.closed or c.error:
            raise web.HTTPConflict(text="Call is not healthy")
        turn_id = request.headers.get("X-Turn-ID","")
        if not turn_id or len(turn_id)>128 or turn_id.startswith("_idle:"):
            raise web.HTTPBadRequest(text="X-Turn-ID required (1–128 characters)")
        data = await request.read()
        digest = hashlib.sha256(data).hexdigest()
        async with c.control_lock:
            if turn_id in c.turns:
                existing = c.turns[turn_id]
                if existing.digest != digest:
                    raise web.HTTPConflict(text="Turn ID already belongs to different audio")
                return web.json_response(existing.summary())
            if len(c.queue)>=4:
                raise web.HTTPTooManyRequests(text="Turn queue full")
            if len(c.turns)>=2048:
                # Do not evict idempotency keys and permit accidental replay.
                raise web.HTTPTooManyRequests(text="Call turn-history limit reached; start a new call")
            try:
                with sf.SoundFile(io.BytesIO(data)) as source:
                    if source.frames > 30*source.samplerate:
                        raise ValueError("Each turn must be <=30 seconds; split longer audio explicitly")
                audio = await asyncio.to_thread(decode_audio,data,30)
            except (ValueError, RuntimeError, OSError) as exc:
                raise web.HTTPBadRequest(text=str(exc)) from exc
            if c.closed:
                raise web.HTTPGone(text="Call closed during upload")
            pcm = (resample_poly(audio,3,1).clip(-1,1)*32767).astype(np.int16)
            frames = math.ceil(len(audio)*c.fps/16000/24)*24
            turn = Turn(turn_id,digest,audio,pcm,frames)
            c.turns[turn_id] = turn
            c.queue.append(turn)
            return web.json_response(turn.summary(),status=202)

    async def schedule_once(self):
        service = self.service
        for c in list(self.items.values()):
            if not c.connected and time.monotonic()-(c.disconnected_at or c.created)>60:
                await self.close(c)
        eligible = [c for c in self.items.values() if c.connected and not c.closed and not c.error
                    and (c.active is not None or c.queue or c.idle_policy=="generate") and not c.chunks.full() and not c.lock.locked()]
        waiting_speech = any(c.queue for c in self.items.values()
                             if c.connected and not c.closed and not c.error)
        def extend_idle(c):
            # Do not let an infinite synthetic turn monopolize the rendering
            # admission slot while another peer has actual speech queued.
            # Already-generated idle drains normally; RTP tracks stay unchanged.
            return (c.active and c.active.synthetic_idle and not waiting_speech and c.chunks.qsize()<1
                    and c.state.cursor==c.state.total_frames)
        eligible = [c for c in eligible if c.active is None or c.state.cursor<c.state.total_frames or extend_idle(c)]
        if not eligible:
            return False
        active_count = sum(c.active is not None for c in self.items.values())
        limit = getattr(service.args,"max_active_calls",1)
        eligible.sort(key=lambda c: (0 if c.queue or (c.active and not c.active.synthetic_idle) else 1,
                    len(c.current_frames)+c.chunks.qsize()*24,c.last_scheduled))
        selected = []
        admissions = max(0,limit-active_count)
        for c in eligible:
            if c.active is None:
                if not admissions:
                    continue
                admissions -= 1
            selected.append(c)
            if len(selected) >= getattr(service.args,"batch",1):
                break
        if not selected:
            return False
        async with contextlib.AsyncExitStack() as stack:
            epochs = {}
            for c in selected:
                await stack.enter_async_context(c.lock)
                epochs[c.id] = c.epoch
            try:
                rendering = []
                for c in selected:
                    epoch = epochs[c.id]
                    if extend_idle(c):
                        # Render one silence chunk ahead while the previous one
                        # is playing; do not stop/re-arm the track every 24 frames.
                        # Reserve counters before awaiting so playback cannot
                        # complete this synthetic turn during the append RPC.
                        turn=c.active
                        turn.frames+=24
                        turn.useful_samples+=round(24*48000/c.fps)
                        silence=np.zeros(round(24*16000/c.fps),np.float32)
                        if service.worker:
                            await service.worker.append(c.state,silence)
                        else:
                            await service.gpu(service.engine.append,c.state,silence)
                        if epoch != c.epoch or c.closed:
                            continue
                    if c.active is None:
                        if c.queue:
                            turn = c.queue.popleft()
                        else:
                            samples=round(24*16000/c.fps)
                            turn=Turn(f"_idle:{c.epoch}:{c.generated_idle_frames}","silence",
                                      np.zeros(samples,np.float32),np.zeros(samples*3,np.int16),24,synthetic_idle=True)
                        c.active = turn
                        turn.status = "preparing"
                        # Idle playback changes visible motion; re-encode its latest
                        # sent history only when no previous chunk remains in flight.
                        if c.idle.path and len(c.sent_history)==9:
                            displayed = np.stack(c.sent_history)
                            if service.worker:
                                await service.worker.recondition(c.state,displayed)
                            else:
                                await service.gpu(service.engine.recondition,c.state,displayed)
                        if epoch != c.epoch or c.closed:
                            continue
                        if service.worker:
                            await service.worker.append(c.state,turn.audio)
                        else:
                            await service.gpu(service.engine.append,c.state,turn.audio)
                        turn.audio = np.zeros(0, np.float32)
                    if epoch != c.epoch or c.closed:
                        continue
                    rendering.append(c)
                rendering = [c for c in rendering if epochs[c.id]==c.epoch and not c.closed]
                if not rendering:
                    return True
                for c in rendering:
                    c.last_scheduled = time.monotonic()
                if service.worker:
                    chunks, metrics = await service.worker.generate([c.state for c in rendering])
                else:
                    chunks = await service.gpu(service.engine.generate,[c.state for c in rendering])
                    metrics = service.engine.last_metrics.copy()
                service.chunks.append(metrics)
                for c, chunk in zip(rendering,chunks):
                    epoch = epochs[c.id]
                    if epoch == c.epoch and not c.closed:
                        c.generated_frames += len(chunk)
                        if c.active and c.active.synthetic_idle:
                            c.generated_idle_frames += len(chunk)
                        c.chunks.put_nowait((epoch,chunk))
            except Exception as exc:
                service.mark_unhealthy(exc)
                import logging
                logging.getLogger(__name__).exception("Call generation failed")
        return True

    async def interrupt(self, request):
        c = self.get(request)
        async with c.control_lock:
            c.epoch += 1
            for t in list(c.queue)+([c.active] if c.active else []):
                t.status, t.finished_s = "interrupted",time.monotonic()-t.accepted
                t.release_audio()
            c.queue.clear()
            c.active = None
            c.current_frames.clear()
            while not c.chunks.empty():
                c.chunks.get_nowait()
            c.changed.set()
            async with c.lock:
                if c.closed:
                    raise web.HTTPGone(text="Call closed")
                displayed = list(c.sent_history) or [c.idle.anchor]
                displayed = np.stack(([displayed[0]]*9+displayed)[-9:])
                try:
                    if self.service.worker:
                        await self.service.worker.recondition(c.state,displayed)
                    else:
                        await self.service.gpu(self.service.engine.recondition,c.state,displayed)
                except RuntimeError as exc:
                    self.service.mark_unhealthy(exc)
                    raise web.HTTPServiceUnavailable(text="GPU reconditioning failed; worker restart required") from exc
            return web.json_response(c.metrics())

    async def stats(self, request):
        return web.json_response(self.get(request).metrics())

    async def close(self, c):
        if c.closed:
            return
        c.closed, c.connected = True, False
        c.epoch += 1
        c.changed.set()
        self.items.pop(c.id,None)
        if c.pc:
            await c.pc.close()
        async with c.lock:
            if self.service.worker:
                await self.service.worker.release(c.state)
        c.idle.close()
        c.queue.clear()
        for turn in c.turns.values():
            turn.release_audio()
        c.turns.clear()
        c.active = None
        c.current_frames.clear()
        c.sent_history.clear()
        while not c.chunks.empty():
            c.chunks.get_nowait()

    async def delete(self, request):
        c = self.items.get(request.match_info["cid"])
        if c:
            await self.close(c)
        return web.Response(status=204)

    async def cleanup(self):
        await asyncio.gather(*(self.close(c) for c in list(self.items.values())))
