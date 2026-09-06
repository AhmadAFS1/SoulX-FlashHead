"""CPU-only source-bank continuity control; this does NOT run SoulX inference.

Replays the exact certified MuseTalk files through one persistent H264/Opus
peer. Decoding is bounded to one frame at a time. No source assets are changed.
"""
import argparse
import asyncio
import hashlib
import json
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from aiortc import (AudioStreamTrack, RTCConfiguration, RTCPeerConnection,
                    RTCRtpSender, VideoStreamTrack)
from aiortc.mediastreams import MediaStreamError


def source_paths(manifest, musetalk_root):
    config = json.loads(Path(manifest).read_text())
    root = Path(musetalk_root).resolve()
    bank = (root / config["asset_bundle"] / "certified").resolve()
    bank.relative_to(root)
    names = []
    for pose in config["poses"].values():
        for item in [pose] + pose.get("variants", []):
            if item["asset_file"] not in names:
                names.append(item["asset_file"])
    paths = [(bank / name).resolve() for name in names]
    for path in paths:
        path.relative_to(bank)
        if not path.is_file():
            raise FileNotFoundError(path)
    if not paths:
        raise ValueError("Manifest has no source clips")
    return paths


def frame_hash(rgb):
    return hashlib.sha256(rgb.tobytes()).hexdigest()


def mae(a, b):
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def audit_sources(paths):
    rows, endpoints = [], []
    offset = 0
    for path in paths:
        first = []
        last = deque(maxlen=6)
        count = 0
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.average_rate is None or stream.average_rate <= 0:
                raise ValueError(f"Missing/invalid frame rate: {path}")
            rate = Fraction(stream.average_rate)
            shape = [stream.width, stream.height]
            for frame in container.decode(stream):
                rgb = frame.to_ndarray(format="rgb24")
                if count < 6:
                    first.append(frame_hash(rgb))
                last.append(frame_hash(rgb))
                if count == 0:
                    opening = rgb
                closing = rgb
                count += 1
        if count < 6:
            raise ValueError(f"Need at least six frames: {path}")
        with path.open("rb") as source:
            hasher = hashlib.sha256()
            for block in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(block)
            digest = hasher.hexdigest()
        rows.append(dict(file=path.name, sha256=digest, frames=count,
                         width=shape[0], height=shape[1], fps=str(rate),
                         start_frame=offset, first_six_sha256=first,
                         last_six_sha256=list(last)))
        endpoints.append((opening, closing))
        offset += count
    if not rows:
        raise ValueError("Need at least one source clip")
    shapes = {(r["width"], r["height"], r["fps"]) for r in rows}
    if len(shapes) != 1:
        raise ValueError("Replay requires identical geometry and cadence")
    transitions = [dict(source=rows[i]["file"], target=rows[j]["file"],
                        rgb_mae=mae(endpoints[i][1], endpoints[j][0]),
                        exact=np.array_equal(endpoints[i][1], endpoints[j][0]))
                   for i in range(len(rows)) for j in range(len(rows)) if i != j]
    return rows, transitions


class MediaClock:
    def __init__(self):
        self.epoch = None

    async def pace(self, media_time):
        if self.epoch is None:
            self.epoch = time.monotonic() + .1
        await asyncio.sleep(max(0, self.epoch + media_time - time.monotonic()))


def decoded_frames(paths):
    for path in paths:
        with av.open(str(path)) as container:
            yield from container.decode(video=0)


class BankVideoTrack(VideoStreamTrack):
    def __init__(self, paths, fps, clock):
        super().__init__()
        self.frames = decoded_frames(paths)
        self.fps, self.clock, self.index = fps, clock, 0
        self.last, self.drained = None, False

    async def recv(self):
        if self.drained:
            raise MediaStreamError
        frame = await asyncio.to_thread(next, self.frames, None)
        if frame is None:
            if self.last is None:
                raise MediaStreamError
            # One new timestamp releases the previous H264 access unit.
            frame = self.last
            self.drained = True
        else:
            self.last = frame
        await self.clock.pace(self.index / self.fps)
        frame.pts = round(self.index * 90000 / self.fps)
        frame.time_base = Fraction(1, 90000)
        self.index += 1
        return frame


class ClockedSilenceTrack(AudioStreamTrack):
    def __init__(self, seconds, clock):
        super().__init__()
        self.clock, self.position = clock, 0
        self.limit = round((seconds + .1) * 48000)

    async def recv(self):
        if self.position >= self.limit:
            raise MediaStreamError
        await self.clock.pace(self.position / 48000)
        frame = av.AudioFrame(format="s16", layout="stereo", samples=960)
        frame.planes[0].update(bytes(frame.planes[0].buffer_size))
        frame.sample_rate = 48000
        frame.pts, frame.time_base = self.position, Fraction(1, 48000)
        self.position += 960
        return frame


async def replay(paths, output, record=True):
    sources, transitions = await asyncio.to_thread(audit_sources, paths)
    fps = Fraction(sources[0]["fps"])
    count = sum(s["frames"] for s in sources)
    duration = float(count / fps)
    clock = MediaClock()
    sender, receiver = [RTCPeerConnection(RTCConfiguration(iceServers=[])) for _ in range(2)]
    video = BankVideoTrack(paths, fps, clock)
    sender.addTrack(video)
    sender.addTrack(ClockedSilenceTrack(duration, clock))
    h264 = [c for c in RTCRtpSender.getCapabilities("video").codecs if c.mimeType == "video/H264"]
    if not h264:
        raise RuntimeError("H264 encoder unavailable")
    for transceiver in sender.getTransceivers():
        if transceiver.kind == "video":
            transceiver.setCodecPreferences(h264)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    container, streams = None, {}
    if record:
        container = av.open(str(output.with_suffix(".mp4")), "w")
        streams["video"] = container.add_stream("libx264", rate=fps)
        streams["video"].width = sources[0]["width"]
        streams["video"].height = sources[0]["height"]
        streams["video"].pix_fmt = "yuv420p"
        streams["video"].options = {"crf": "18", "preset": "veryfast"}
        streams["audio"] = container.add_stream("aac", rate=48000)
        streams["audio"].layout = "stereo"
    pts, arrivals, audio_pts, boundaries, errors, tasks = [], [], [], {}, [], []
    boundary_indices = {s["start_frame"] for s in sources[1:]}
    seen_ssrc = {"audio": set(), "video": set()}
    audio_samples = 0
    previous = None
    started = time.monotonic()

    async def consume(track):
        nonlocal previous, audio_samples
        try:
            while True:
                frame = await track.recv()
                timestamp = float(frame.pts * frame.time_base)
                if timestamp >= duration - 1e-7:
                    continue  # Ignore explicit codec/jitter-buffer drain tail.
                if track.kind == "video":
                    index = round(timestamp * fps)
                    rgb = frame.to_ndarray(format="rgb24")
                    if index in boundary_indices and previous is not None:
                        boundaries[index] = dict(time_s=timestamp, rgb_mae=mae(previous, rgb))
                    previous = rgb
                    pts.append(timestamp)
                    arrivals.append(time.monotonic() - started)
                else:
                    audio_pts.append(timestamp)
                    audio_samples += min(frame.samples, max(0, round(duration * 48000) - round(timestamp * 48000)))
                if container:
                    for packet in streams[track.kind].encode(frame):
                        container.mux(packet)
        except MediaStreamError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            errors.append(f"{track.kind}: {type(exc).__name__}: {exc}")

    @receiver.on("track")
    def on_track(track):
        tasks.append(asyncio.create_task(consume(track)))

    try:
        await sender.setLocalDescription(await sender.createOffer())
        await receiver.setRemoteDescription(sender.localDescription)
        await receiver.setLocalDescription(await receiver.createAnswer())
        await sender.setRemoteDescription(receiver.localDescription)
        deadline = time.monotonic() + duration + 15
        while time.monotonic() < deadline:
            for stat in (await receiver.getStats()).values():
                if stat.type == "inbound-rtp":
                    seen_ssrc[stat.kind].add(stat.ssrc)
            if len(pts) >= count and audio_samples >= round(duration * 48000):
                break
            if errors:
                break
            await asyncio.sleep(.1)
        stats = [vars(s) for s in (await receiver.getStats()).values() if s.type == "inbound-rtp"]
        result = dict(mode="source_asset_relay_control_no_soulx_inference", sources=sources,
                      all_ordered_source_transitions=transitions, width=sources[0]["width"],
                      height=sources[0]["height"], fps=float(fps), duration_s=duration,
                      expected_frames=count, received_frames=len(pts), audio_samples=audio_samples,
                      expected_audio_samples=round(duration * 48000),
                      peer_connections=1, negotiations=1, video_track_replacements=0,
                      pts_monotonic=all(b > a for a, b in zip(pts, pts[1:])),
                      unique_timestamps=len(set(pts)),
                      max_pts_step_error_s=max((abs(b-a-float(1/fps)) for a,b in zip(pts,pts[1:])), default=0),
                      first_video_s=arrivals[0] if arrivals else None,
                      max_arrival_gap_s=max((b-a for a,b in zip(arrivals,arrivals[1:])), default=0),
                      first_audio_pts_s=audio_pts[0] if audio_pts else None,
                      last_audio_pts_s=audio_pts[-1] if audio_pts else None,
                      ssrcs={k: sorted(v) for k,v in seen_ssrc.items()}, inbound_rtp=stats,
                      receiver_boundaries=boundaries, errors=errors)
        result["passed_transport_control"] = (len(pts) == count and len(set(pts)) == count
            and result["pts_monotonic"] and result["max_pts_step_error_s"] < 1e-4
            and audio_samples == round(duration * 48000) and len(boundaries) == len(sources)-1
            and all(len(v) == 1 for v in seen_ssrc.values()) and not errors
            and all(s["packetsLost"] == 0 for s in stats))
        output.write_text(json.dumps(result, indent=2, default=str) + "\n")
        print(json.dumps({k:v for k,v in result.items() if k not in ("sources", "all_ordered_source_transitions", "inbound_rtp")}, indent=2))
        return result
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await asyncio.gather(sender.close(), receiver.close())
        video.frames.close()
        if container:
            for stream in streams.values():
                for packet in stream.encode():
                    container.mux(packet)
            container.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--musetalk-root", default="/workspace/MuseTalk")
    parser.add_argument("--manifest", default="/workspace/MuseTalk/configs/pose_test/sample_ai_human_ltx23_facetime_closeup_production_v1.json")
    parser.add_argument("--output", default="benchmarks/migration/base-bank-webrtc.json")
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(replay(source_paths(args.manifest, args.musetalk_root), args.output, not args.no_record))
    if not result["passed_transport_control"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
