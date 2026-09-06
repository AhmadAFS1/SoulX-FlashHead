"""Receive actual DTLS/SRTP media from N peers; report stalls and frame counts."""
import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import aiohttp
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError


def input_provenance(path):
    """Record only the fixture name/hash, never API credentials or environment."""
    if not path:
        return None
    path = Path(path)
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


async def run(args):
    headers = {"Authorization": "Bearer " + os.environ["SOULX_API_TOKEN"]} if os.environ.get("SOULX_API_TOKEN") else {}
    peers, sessions, tasks, rows, recordings = [], [], [], [], []
    async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=300)) as client:
        async def api(path, method="GET", **kwargs):
            async with client.request(method, args.url + path, **kwargs) as response:
                response.raise_for_status()
                return await response.json() if response.status != 204 else None
        health = await api("/health")
        start = time.monotonic()
        barrier = asyncio.Event()
        async def peer(index):
            if args.images or args.audio:
                form = aiohttp.FormData()
                form.add_field("seconds", str(args.seconds))
                form.add_field("seed", str(args.seed + index))
                if args.images:
                    portrait = Path(args.images[index % len(args.images)])
                    form.add_field("image", portrait.read_bytes(), filename=portrait.name)
                if args.audio:
                    audio = Path(args.audio)
                    form.add_field("audio", audio.read_bytes(), filename=audio.name)
                session = await api("/sessions", "POST", data=form)
            else:
                session = await api("/sessions", "POST", json={"seconds": args.seconds, "seed": args.seed + index})
            sid = session["id"]
            sessions.append(sid)
            pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            peers.append(pc)
            row = {"index": index, "id": sid, "expected_video": session["frames"],
                   "video_frames": 0, "audio_frames": 0, "audio_samples": 0,
                   "audio_packet_samples": 0, "seed": args.seed + index,
                   "input_image": input_provenance(args.images[index % len(args.images)]) if args.images else None,
                   "input_audio": input_provenance(args.audio),
                   "first_video_s": None, "first_audio_s": None,
                   "video_pts": [], "video_arrivals": [], "errors": []}
            rows.append(row)
            recording = None
            if args.record and index == 0:
                import av
                path = Path(args.output).with_suffix(".mp4")
                path.parent.mkdir(parents=True, exist_ok=True)
                container = av.open(str(path), "w")
                vs = container.add_stream("libx264", rate=health["fps"])
                vs.width = vs.height = health["size"]
                vs.pix_fmt = "yuv420p"
                vs.options = {"preset": "veryfast", "crf": "18"}
                aus = container.add_stream("aac", rate=48000)
                aus.layout = "stereo"
                recording = (container, {"video": vs, "audio": aus})
                recordings.append(recording)
            async def consume(track):
                try:
                    while True:
                        frame = await track.recv()
                        elapsed = time.monotonic() - start
                        # Explicit transport tail packets only drain codec/jitter
                        # buffers. They are never productive frames/audio.
                        media_time = float(frame.pts * frame.time_base)
                        limit = row["expected_video"] / health["fps"] if track.kind == "video" else args.seconds
                        if media_time >= limit - 1e-7:
                            continue
                        if recording:
                            container, streams = recording
                            for packet in streams[track.kind].encode(frame):
                                container.mux(packet)
                        if track.kind == "video":
                            row["video_frames"] += 1
                            row["video_pts"].append(float(frame.pts * frame.time_base))
                            row["video_arrivals"].append(elapsed)
                            if row["first_video_s"] is None:
                                row["first_video_s"] = elapsed
                                pixels = frame.to_ndarray(format="rgb24")
                                row["first_frame_hash"] = hashlib.sha256(pixels.tobytes()).hexdigest()
                                if args.save_frame:
                                    from PIL import Image
                                    path = Path(args.output).with_suffix(f".peer{index}.jpg")
                                    path.parent.mkdir(parents=True, exist_ok=True)
                                    Image.fromarray(pixels).save(path)
                        else:
                            row["audio_frames"] += 1
                            row["audio_packet_samples"] += frame.samples
                            row["audio_samples"] += min(frame.samples, max(0,
                                round(args.seconds * 48000) - round(media_time * 48000)))
                            if row["first_audio_s"] is None:
                                row["first_audio_s"] = elapsed
                except asyncio.CancelledError:
                    raise
                except MediaStreamError:
                    pass
                except Exception as exc:
                    row["errors"].append(type(exc).__name__)
            @pc.on("track")
            def on_track(track):
                tasks.append(asyncio.create_task(consume(track)))
            pc.addTransceiver("video", direction="recvonly")
            pc.addTransceiver("audio", direction="recvonly")
            await pc.setLocalDescription(await pc.createOffer())
            await barrier.wait()
            answer = await api(f"/sessions/{sid}/offer", "POST", json={"sdp": pc.localDescription.sdp, "type": "offer"})
            await pc.setRemoteDescription(RTCSessionDescription(**answer))
        setup = [asyncio.create_task(peer(i)) for i in range(args.sessions)]
        try:
            deadline = time.monotonic() + 180
            while len(sessions) < args.sessions:
                for task in setup:
                    if task.done() and task.exception():
                        raise task.exception()
                if time.monotonic() > deadline:
                    raise TimeoutError("Session preparation timed out")
                await asyncio.sleep(.05)
            # Align offers so all clients contend for GPU work together.
            start = time.monotonic()
            barrier.set()
            await asyncio.gather(*setup)
            deadline = time.monotonic() + args.timeout
            while time.monotonic() < deadline:
                status = await asyncio.gather(*(api(f"/sessions/{s}") for s in sessions))
                if all(s["video_sent"] == s["frames"] for s in status):
                    await asyncio.sleep(1)  # Drain packets already sent.
                    break
                if any(s["error"] for s in status):
                    break
                await asyncio.sleep(.25)
            elapsed = time.monotonic() - start
            status = {s: await api(f"/sessions/{s}") for s in sessions}
            for row in rows:
                row["server"] = status[row["id"]]
                gaps = [b - a for a,b in zip(row["video_arrivals"], row["video_arrivals"][1:])]
                pts = row["video_pts"]
                row["max_video_gap_s"] = max(gaps, default=0)
                row["video_gaps_over_200ms"] = sum(g > .2 for g in gaps)
                row["unique_video_timestamps"] = len(set(pts))
                row["pts_monotonic"] = all(b > a for a,b in zip(pts, pts[1:]))
                row["pts_step_max_error_s"] = max((abs(b-a-1/health["fps"]) for a,b in zip(pts,pts[1:])), default=0)
            for pc, row in zip(peers, rows):
                stats = await pc.getStats()
                row["rtp_stats"] = [vars(s) for s in stats.values() if s.type == "inbound-rtp"]
            report = dict(profile={k: health[k] for k in ("size", "steps", "batch", "fps")},
                          sessions=args.sessions, clip_seconds=args.seconds, wall_s=elapsed,
                          received_frames=sum(r["video_frames"] for r in rows), peers=rows)
            report["received_aggregate_fps"] = report["received_frames"] / elapsed
            report["metric_note"] = "Paced receiver wall includes negotiation and drain; not unpaced model FPS. Audio samples exclude final packet padding."
            path = Path(args.output)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2, default=str))
            print(json.dumps({k:v for k,v in report.items() if k != "peers"}, indent=2))
            for r in rows:
                print(json.dumps({k:r[k] for k in ("index","video_frames","audio_samples","first_video_s","max_video_gap_s","unique_video_timestamps","server")}))
        finally:
            for task in setup + tasks:
                task.cancel()
            await asyncio.gather(*(pc.close() for pc in peers), return_exceptions=True)
            await asyncio.gather(*(api(f"/sessions/{s}", "DELETE") for s in sessions), return_exceptions=True)
            await asyncio.gather(*setup, *tasks, return_exceptions=True)
            for container, streams in recordings:
                for stream in streams.values():
                    for packet in stream.encode():
                        container.mux(packet)
                container.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--sessions", type=int, default=1)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--output", default="benchmarks/rtc.json")
    p.add_argument("--save-frame", action="store_true")
    p.add_argument("--record", action="store_true", help="Record peer 0; use separately from performance runs")
    p.add_argument("--images", nargs="+", help="Alternate uploaded portraits across independent sessions")
    p.add_argument("--audio", help="Upload this audio instead of the bundled sample")
    p.add_argument("--seed", type=int, default=50, help="First session seed; incremented per peer")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
