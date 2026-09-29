"""MuseTalk-compatible WebRTC groups for the browser wall.

A group names one independent persistent call per wall tile. It adds no media
path: each tile's player page negotiates its own call, and group speech queues
the same idempotent turn on every call through CallService. Group records hold
only call IDs, seeds and configuration; the calls own all GPU and media state.
Routes and response fields follow MuseTalk's /webrtc/groups API so the same
wall workflow and client scripts apply.
"""
import asyncio
import json
import os
import secrets
import shutil
import subprocess
import time

from aiohttp import web

from .codec import sender_encoder_info

MAX_GROUP_PEERS = 12
GROUP_TTL_S = 3600  # Kept this long after the last live call, for Reconnect.
MAX_GROUPS = 64


def call_status(c):
    """One word per wall tile: turn state while speaking, else peer state."""
    if c is None:
        return "missing"
    if c.error:
        return "error"
    if c.closed:
        return "closed"
    turn = c.active
    if turn is not None and not turn.synthetic_idle:
        return turn.status  # preparing / armed / speaking / draining
    if c.queue:
        return "queued"
    if c.returning_idle:
        return "returning"
    return getattr(c.pc, "connectionState", None) or "created"


def gpu_index():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0].strip()
    return visible or "0"


def sample_gpu(index):
    """nvidia-smi snapshot with MuseTalk's /stats/gpu-live field names."""
    if shutil.which("nvidia-smi") is None:
        return dict(available=False, reason="nvidia-smi not found", gpu_index=index)
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(index),
             "--query-gpu=name,utilization.gpu,utilization.memory,memory.used,memory.total,"
             "temperature.gpu,power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return dict(available=False, reason=str(exc), gpu_index=index)
    rows = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode or not rows:
        return dict(available=False, gpu_index=index,
                    reason=result.stderr.strip() or f"nvidia-smi exited with {result.returncode}")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 7:
        return dict(available=False, reason=f"unexpected nvidia-smi row: {rows[0]}", gpu_index=index)

    def number(value):
        try:
            return float(value)
        except ValueError:
            return None  # "N/A" and "[Not Supported]"
    used, total = number(parts[3]), number(parts[4])
    return dict(available=True, gpu_index=index, name=parts[0], ts=round(time.time(), 3),
                gpu_util_pct=number(parts[1]), memory_util_pct=number(parts[2]),
                memory_used_mb=used, memory_total_mb=total,
                memory_used_gb=round(used/1024, 2) if used is not None else None,
                memory_total_gb=round(total/1024, 2) if total is not None else None,
                temperature_c=number(parts[5]), power_draw_w=number(parts[6]))


class GroupService:
    def __init__(self, service):
        self.service = service
        self.calls = service.calls
        self.groups = {}
        self.gpu_sample = None
        self.gpu_lock = asyncio.Lock()

    def routes(self):
        return [web.post("/webrtc/groups/create", self.create),
                web.get("/webrtc/groups/{gid}", self.get),
                web.delete("/webrtc/groups/{gid}", self.delete),
                web.post("/webrtc/groups/{gid}/stream", self.stream),
                web.post("/webrtc/groups/{gid}/interrupt", self.interrupt),
                web.post("/webrtc/groups/{gid}/reconnect", self.reconnect),
                web.get("/stats/gpu-live", self.gpu_live)]

    def live(self, group):
        return any(slot["call_id"] in self.calls.items for slot in group["slots"])

    def prune(self):
        now = time.monotonic()
        for gid, group in list(self.groups.items()):
            if not self.live(group) and now-group["touched"] > GROUP_TTL_S:
                del self.groups[gid]
        if len(self.groups) >= MAX_GROUPS:
            dead = [gid for gid, group in self.groups.items() if not self.live(group)]
            if not dead:
                raise web.HTTPTooManyRequests(text="Too many live groups; delete one first")
            del self.groups[min(dead, key=lambda gid: self.groups[gid]["touched"])]

    def lookup(self, request):
        group = self.groups.get(request.match_info["gid"])
        if group is None:
            raise web.HTTPNotFound(text="Group not found")
        group["touched"] = time.monotonic()
        return group

    def config(self, count, seed):
        args = self.service.args
        return dict(count=count, first_seed=seed, fps=args.fps, playback_fps=args.fps,
                    width=getattr(args, "width", None) or args.size,
                    height=getattr(args, "height", None) or args.size,
                    steps=args.steps, batch_size=args.batch,
                    max_active_calls=getattr(args, "max_active_calls", 1),
                    idle_policy=getattr(args, "idle_policy", "source"),
                    ice_transport_policy=self.service.ice_transport_policy)

    def session(self, group, slot):
        cid = slot["call_id"]
        c = self.calls.items.get(cid)
        entry = dict(session_id=cid, index=slot["index"], seed=slot["seed"], status=call_status(c),
                     player_url=f"/webrtc/player/{cid}", avatar_id=group["avatar_id"],
                     replacements=slot["replacements"], connected=False, active_stream=None,
                     error=None, stats=None, last_turn=None)
        if c is None:
            return entry
        # Call.turns holds only submitted speech; synthetic idle never enters it.
        last = next(reversed(c.turns.values()), None)
        entry.update(
            connected=c.connected, error=c.error or None,
            active_stream=c.active.id if c.active and not c.active.synthetic_idle else None,
            last_turn=last.summary() if last else None,
            stats=dict(peer=getattr(c.pc, "connectionState", None), queued_turns=len(c.queue),
                       queued_chunks=c.chunks.qsize(), video_sent=c.video_sent,
                       generated_frames=c.generated_frames, idle_frames=c.idle_frames,
                       held_frames=c.held_frames, underrun_frames=c.underrun_frames,
                       missed_video_slots=c.missed_video_slots,
                       audio_hold_samples=c.audio_hold_samples, total_turns=len(c.turns),
                       completed_turns=sum(t.status == "complete" for t in c.turns.values()),
                       negotiations=c.negotiations, video_encoders=sender_encoder_info(c.pc)))
        return entry

    def response(self, group):
        gid = group["group_id"]
        sessions = [self.session(group, slot) for slot in group["slots"]]
        return dict(group_id=gid, avatar_id=group["avatar_id"], avatar_name=group["avatar_name"],
                    created_at=group["created_at"], count=len(sessions), config=group["config"],
                    session_ids=[s["session_id"] for s in sessions],
                    session_statuses={s["session_id"]: s["status"] for s in sessions},
                    sessions=sessions, wall_url=f"/webrtc/groups/{gid}/wall",
                    stream_all_url=f"/webrtc/groups/{gid}/stream")

    @staticmethod
    def targets(group, raw):
        """All slots, or the subset named by a JSON list / comma-separated string."""
        if raw in (None, "", []):
            return list(group["slots"])
        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw.lstrip().startswith("[") else raw.split(",")
            except ValueError as exc:
                raise web.HTTPBadRequest(text="session_ids must be a list of IDs") from exc
        if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
            raise web.HTTPBadRequest(text="session_ids must be a list of IDs")
        wanted = {item.strip() for item in raw if item.strip()}
        slots = [slot for slot in group["slots"] if slot["call_id"] in wanted]
        if len(slots) != len(wanted):
            raise web.HTTPBadRequest(text="Unknown session_id for this group")
        return slots

    async def create(self, request):
        params = dict(request.query)
        if request.can_read_body and request.content_type == "application/json":
            try:
                body = await request.json()
            except ValueError as exc:
                raise web.HTTPBadRequest(text="Expected a JSON object") from exc
            if not isinstance(body, dict):
                raise web.HTTPBadRequest(text="Expected a JSON object")
            params.update(body)
        service = self.service
        limit = min(MAX_GROUP_PEERS, service.args.max_sessions)
        try:
            count, seed = int(params.get("count", 2)), int(params.get("seed", 50))
            avatar = self.calls.avatars.resolve(str(params.get("avatar_id", "default")))
        except (ValueError, TypeError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        if not 1 <= count <= limit:
            raise web.HTTPBadRequest(text=f"count must be between 1 and {limit}")
        if not 0 <= seed <= 2**63-count:
            raise web.HTTPBadRequest(text="seed must be an integer from 0 through 2**63-count")
        if not service.ready:
            raise web.HTTPServiceUnavailable(text="GPU worker is not ready")
        free = service.args.max_sessions-len(self.calls.items)-len(service.sessions)-service.pending
        if count > free:
            raise web.HTTPTooManyRequests(
                text=f"{max(free, 0)} of {service.args.max_sessions} peer slots free; "
                     "delete a group or choose fewer peers")
        self.prune()
        created = []
        try:
            for index in range(count):
                created.append(await self.calls.create_call(
                    dict(seed=seed+index, avatar_id=avatar["id"])))
        except BaseException:
            # Creation is atomic for the caller: never leak the calls already prepared.
            await asyncio.gather(*(self.calls.close(c) for c in created), return_exceptions=True)
            raise
        gid = secrets.token_hex(5)
        self.groups[gid] = group = dict(
            group_id=gid, avatar_id=avatar["id"], avatar_name=avatar["name"],
            created_at=time.time(), touched=time.monotonic(), lock=asyncio.Lock(),
            slots=[dict(index=i, seed=seed+i, call_id=c.id, replacements=0)
                   for i, c in enumerate(created)],
            config=self.config(count, seed))
        return web.json_response(self.response(group), status=201)

    async def get(self, request):
        return web.json_response(self.response(self.lookup(request)))

    async def stream(self, request):
        group = self.lookup(request)
        form = await request.post()
        upload = form.get("audio_file")
        if not hasattr(upload, "file"):
            raise web.HTTPBadRequest(text="audio_file is required")
        data = upload.file.read()
        if not data:
            raise web.HTTPBadRequest(text="audio_file is empty")
        turn_id = str(form.get("turn_id") or f"group-{group['group_id']}-{secrets.token_hex(6)}")
        self.calls.check_turn_id(turn_id)
        slots = self.targets(group, form.get("session_ids"))

        async def one(slot):
            cid = slot["call_id"]
            c = self.calls.items.get(cid)
            if c is None:
                return dict(session_id=cid, ok=False, status="missing",
                            detail="Call ended; use Reconnect peers")
            try:
                summary, code = await self.calls.append_turn(c, turn_id, data)
            except web.HTTPException as exc:
                return dict(session_id=cid, ok=False, status="error", code=exc.status, detail=exc.text)
            # 200 is an idempotent replay: that call already owns this turn.
            return dict(session_id=cid, ok=True, status="accepted" if code == 202 else "duplicate",
                        turn=summary)
        results = await asyncio.gather(*(one(slot) for slot in slots))
        started = sum(r["ok"] for r in results)
        return web.json_response(dict(group_id=group["group_id"], turn_id=turn_id, started=started,
                                      failed=len(results)-started, results=results))

    async def interrupt(self, request):
        group = self.lookup(request)
        raw = None
        if request.can_read_body:
            try:
                body = await request.json()
            except ValueError as exc:
                raise web.HTTPBadRequest(text="Expected a JSON object") from exc
            raw = body.get("session_ids") if isinstance(body, dict) else None
        slots = self.targets(group, raw)

        async def one(slot):
            cid = slot["call_id"]
            c = self.calls.items.get(cid)
            if c is None:
                return dict(session_id=cid, ok=False, status="missing", detail="Call ended")
            try:
                await self.calls.interrupt_call(c)
            except web.HTTPException as exc:
                return dict(session_id=cid, ok=False, status="error", code=exc.status, detail=exc.text)
            return dict(session_id=cid, ok=True, status="interrupted")
        results = await asyncio.gather(*(one(slot) for slot in slots))
        interrupted = sum(r["ok"] for r in results)
        return web.json_response(dict(group_id=group["group_id"], interrupted=interrupted,
                                      failed=len(results)-interrupted, results=results))

    async def reconnect(self, request):
        """Replace calls that ended (closed peer, page reload) with fresh ones, same seed."""
        group = self.lookup(request)
        results = []
        async with group["lock"]:
            for slot in group["slots"]:
                previous = slot["call_id"]
                c = self.calls.items.get(previous)
                if c is not None and not c.closed:
                    results.append(dict(session_id=previous, ok=True, status="kept"))
                    continue
                try:
                    call = await self.calls.create_call(dict(seed=slot["seed"], avatar_id=group["avatar_id"]))
                except web.HTTPException as exc:
                    results.append(dict(session_id=previous, ok=False, status="error",
                                        code=exc.status, detail=exc.text))
                    continue
                if self.groups.get(group["group_id"]) is not group:  # Deleted meanwhile.
                    await self.calls.close(call)
                    raise web.HTTPNotFound(text="Group not found")
                slot["call_id"] = call.id
                slot["replacements"] += 1
                results.append(dict(session_id=call.id, previous_session_id=previous, ok=True,
                                    status="replaced"))
        body = self.response(group)
        body["reconnect"] = results
        return web.json_response(body)

    async def delete(self, request):
        group = self.groups.pop(request.match_info["gid"], None)
        if group is None:
            raise web.HTTPNotFound(text="Group not found")
        deleted, errors = [], []
        async with group["lock"]:
            for slot in group["slots"]:
                c = self.calls.items.get(slot["call_id"])
                try:
                    if c is not None:
                        await self.calls.close(c)
                    deleted.append(slot["call_id"])
                except Exception as exc:  # Report; keep releasing the other calls.
                    errors.append(dict(session_id=slot["call_id"], detail=str(exc)))
        return web.json_response(dict(group_id=group["group_id"], deleted_sessions=deleted,
                                      errors=errors))

    async def gpu_live(self, request):
        async with self.gpu_lock:
            # Walls poll every two seconds; one nvidia-smi per second serves them all.
            if self.gpu_sample is None or time.monotonic()-self.gpu_sample[0] >= 1:
                self.gpu_sample = time.monotonic(), await asyncio.to_thread(sample_gpu, gpu_index())
        return web.json_response(self.gpu_sample[1], headers={"Cache-Control": "no-store"})
