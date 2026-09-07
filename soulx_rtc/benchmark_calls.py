"""Actual persistent-peer benchmark. Wire/idle/held FPS is not generated FPS."""
import argparse
import asyncio
import hashlib
import gzip
import io
import json
import math
import os
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaRecorder
from aiortc.mediastreams import MediaStreamError

from .server import decode_audio


async def run(args):
    client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Use a fresh evidence path")
    output.parent.mkdir(parents=True,exist_ok=True)
    headers = {"Authorization":"Bearer "+os.environ["SOULX_API_TOKEN"]} if os.environ.get("SOULX_API_TOKEN") else {}
    fixtures = []
    for path in args.audio:
        audio = decode_audio(path,args.audio_seconds)
        stream = io.BytesIO()
        sf.write(stream,audio,16000,format="WAV",subtype="PCM_16")
        fixtures.append((stream.getvalue(),dict(file=Path(path).name,
            source_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            upload_sha256=hashlib.sha256(stream.getvalue()).hexdigest(),samples_16k=len(audio))))
    peers, ids, tasks, recorders, rows, samples = [],[],[],[],[],[]
    started = time.monotonic()
    async with aiohttp.ClientSession(headers=headers,timeout=aiohttp.ClientTimeout(total=args.timeout+120)) as client:
        async def api(path,method="GET",**kwargs):
            async with client.request(method,args.url+path,**kwargs) as response:
                if response.status>=400:
                    raise RuntimeError(f"HTTP {response.status}: {await response.text()}")
                return None if response.status==204 else await response.json()
        health = await api("/health")
        async def create(index):
            c = await api("/calls","POST",json={"seed":50+index})
            ids.append(c["id"])
            pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
            peers.append(pc)
            row = dict(index=index,id=c["id"],video_pts=[],audio_pts=[],video_arrivals=[],audio_samples=0,errors=[])
            row["snapshots"]=[]
            next_snapshot=0.
            rows.append(row)
            recorder = MediaRecorder(str(output.with_name(output.stem+"-peer0.mp4"))) if args.record and index==0 else None
            if recorder:
                recorders.append(recorder)
            # Recorder and metrics both need media; relay duplicates frames safely.
            from aiortc.contrib.media import MediaRelay
            relay = MediaRelay()
            async def consume(track):
                nonlocal next_snapshot
                try:
                    while True:
                        frame=await track.recv()
                        row[track.kind+"_pts"].append(float(frame.pts*frame.time_base))
                        if track.kind=="video":
                            row["video_arrivals"].append(time.monotonic()-started)
                            pts=float(frame.pts*frame.time_base)
                            if args.snapshots_every and index==0 and pts>=next_snapshot:
                                next_snapshot=pts+args.snapshots_every
                                path=output.with_name(output.stem+f"-snapshot-{len(row['snapshots']):04d}.jpg")
                                rgb=frame.to_ndarray(format="rgb24")
                                def save_snapshot():
                                    from PIL import Image
                                    Image.fromarray(rgb).save(path,quality=92)
                                    return hashlib.sha256(path.read_bytes()).hexdigest()
                                digest=await asyncio.to_thread(save_snapshot)
                                row["snapshots"].append(dict(file=path.name,pts_s=pts,sha256=digest))
                        else:
                            row["audio_samples"]+=frame.samples
                except MediaStreamError:
                    pass
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    row["errors"].append(type(exc).__name__)
            @pc.on("track")
            def track(t):
                tasks.append(asyncio.create_task(consume(relay.subscribe(t))))
                if recorder:
                    recorder.addTrack(relay.subscribe(t))
            pc.addTransceiver("video",direction="recvonly")
            pc.addTransceiver("audio",direction="recvonly")
            await pc.setLocalDescription(await pc.createOffer())
            answer=await api(f"/calls/{c['id']}/offer","POST",json={"sdp":pc.localDescription.sdp,"type":"offer"})
            await pc.setRemoteDescription(RTCSessionDescription(**answer))
            if recorder:
                await recorder.start()
            row["pc"]=pc
            return row
        try:
            await asyncio.gather(*(create(i) for i in range(args.sessions)))
            await asyncio.sleep(args.idle_seconds)
            call_started=time.monotonic()
            async def turns(row):
                base=f"/calls/{row['id']}"
                index = 0
                while (time.monotonic()-call_started < args.duration_seconds if args.duration_seconds else index < args.turns):
                    data,_=fixtures[index%len(fixtures)]
                    turn_id=f"turn-{index}"
                    await api(base+"/turns","POST",data=data,headers={"X-Turn-ID":turn_id,"Content-Type":"audio/wav"})
                    # Real retry must not create a second playback.
                    retry=await api(base+"/turns","POST",data=data,headers={"X-Turn-ID":turn_id,"Content-Type":"audio/wav"})
                    row.setdefault("retry_ids",[]).append(retry["id"])
                    deadline=time.monotonic()+args.timeout
                    while time.monotonic()<deadline:
                        status=await api(base)
                        if status["error"]:
                            row["errors"].append(status["error"])
                            break
                        turn=next(t for t in status["turns"] if t["id"]==turn_id)
                        if turn["status"]=="complete":
                            break
                        await asyncio.sleep(.1)
                    else:
                        row["errors"].append("turn_timeout")
                        break
                    index += 1
                    if args.duration_seconds or index<args.turns:
                        await asyncio.sleep(args.gap)
                if args.interrupt and not row["errors"]:
                    data,_=fixtures[-1]
                    await api(base+"/turns","POST",data=data,headers={"X-Turn-ID":"interrupt-fixture"})
                    deadline=time.monotonic()+args.timeout
                    while time.monotonic()<deadline:
                        status=await api(base)
                        interrupted=next(t for t in status["turns"] if t["id"]=="interrupt-fixture")
                        if status["error"]:
                            raise RuntimeError(status["error"])
                        if interrupted["video_sent"]>=math.ceil(args.interrupt_after*health["fps"]):
                            break
                        await asyncio.sleep(.05)
                    else:
                        raise TimeoutError("Interrupt fixture did not begin playback")
                    row["interrupt_before"]=interrupted
                    row["interrupt_after"]=await api(base+"/interrupt","POST",json={})
                    if args.resume_after_interrupt:
                        data,_=fixtures[0]
                        await api(base+"/turns","POST",data=data,headers={"X-Turn-ID":"recovery"})
                        deadline=time.monotonic()+args.timeout
                        while time.monotonic()<deadline:
                            status=await api(base)
                            recovery=next(t for t in status["turns"] if t["id"]=="recovery")
                            if status["error"]:
                                raise RuntimeError(status["error"])
                            if recovery["status"]=="complete":
                                break
                            await asyncio.sleep(.1)
                        else:
                            raise TimeoutError("Post-interrupt recovery did not complete")
                row["call_wall_s"]=time.monotonic()-call_started
                row["server"]=await api(base)
            async def progress():
                while True:
                    await asyncio.sleep(30)
                    snapshots=await asyncio.gather(*(api(f"/calls/{r['id']}") for r in rows))
                    sample=dict(elapsed_s=time.monotonic()-call_started,
                        calls=[{k:s[k] for k in ("completed_turns","queued_chunks","queued_turns","underrun_frames")} for s in snapshots],
                        health=await api("/health"))
                    samples.append(sample)
                    print(json.dumps(dict(elapsed_s=round(time.monotonic()-call_started,1),
                        completed_turns=sum(s["completed_turns"] for s in snapshots),
                        underruns=sum(s["underrun_frames"] for s in snapshots))),flush=True)
            progress_task=asyncio.create_task(progress())
            try:
                speakers=len(rows) if args.speakers is None else args.speakers
                if speakers:
                    await asyncio.gather(*(turns(row) for row in rows[:speakers]))
                else:
                    await asyncio.sleep(args.duration_seconds)
            finally:
                progress_task.cancel()
                await asyncio.gather(progress_task,return_exceptions=True)
            await asyncio.sleep(args.idle_seconds)
            elapsed=time.monotonic()-call_started
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            for row in rows:
                if "server" not in row:
                    row["server"]=await api(f"/calls/{row['id']}")
                    row["call_wall_s"]=elapsed
                pc=row.pop("pc")
                stats=[vars(s) for s in (await pc.getStats()).values() if s.type=="inbound-rtp"]
                row["inbound_rtp"]=stats
                row["video_frames"]=len(row["video_pts"])
                row["pts_monotonic"]={k:all(b>a for a,b in zip(row[k+"_pts"],row[k+"_pts"][1:])) for k in ("video","audio")}
                gaps=np.diff(row["video_arrivals"])
                row["arrival_gap_p95_s"]=float(np.percentile(gaps,95)) if len(gaps) else None
                row["arrival_gap_max_s"]=float(gaps.max()) if len(gaps) else None
                row["wire_fps"]=(len(row["video_pts"])-1)/(row["video_arrivals"][-1]-row["video_arrivals"][0]) if len(gaps) else 0
                encoders=row["server"].get("video_encoders",[])
                row["h264_verified"]=(len(encoders)==1 and encoders[0]["codec"]=="libx264"
                    and encoders[0]["encoder"] in ("H264Encoder","FastH264Encoder"))
                row["transport_pass"]=(all(row["pts_monotonic"].values()) and all(len(row[k+"_pts"])>1 for k in ("audio","video"))
                    and len(stats)==2 and all(s["packetsLost"]==0 for s in stats) and not row["errors"]
                    and row["server"]["negotiations"]==1 and row["h264_verified"])
            result=dict(health_before=health,config=vars(args),inputs=[x[1] for x in fixtures],
                client_sha256=client_sha256,
                wall_s=elapsed,peers=rows,resource_samples=samples,health_after=await api("/health"),
                metric_note="Paced call wall includes idle, gaps and admission wait; wire FPS includes held/idle frames and is NOT generated FPS.",
                completed_turns=sum(r["server"]["completed_turns"] for r in rows),
                all_transport_pass=all(r["transport_pass"] for r in rows),
                total_underrun_frames=sum(r["server"]["underrun_frames"] for r in rows))
            if args.compact_evidence:
                raw=[dict(index=r["index"],**{k:r.pop(k) for k in
                    ("video_pts","audio_pts","video_arrivals")}) for r in rows]
                compressed=gzip.compress(json.dumps(raw,separators=(",",":")).encode(),mtime=0)
                raw_path=output.with_suffix(".timestamps.json.gz")
                raw_path.write_bytes(compressed)
                result["raw_timestamps"]=dict(file=raw_path.name,sha256=hashlib.sha256(compressed).hexdigest())
            output.write_text(json.dumps(result,indent=2,default=str)+"\n")
            print(json.dumps({k:result[k] for k in ("wall_s","metric_note","completed_turns",
                                                   "all_transport_pass","total_underrun_frames")}),flush=True)
            if not result["all_transport_pass"]:
                raise RuntimeError("Persistent transport checks failed; inspect saved evidence")
        except Exception as exc:
            if not output.exists():
                partial=[{k:v for k,v in row.items() if k!="pc"} for row in rows]
                output.write_text(json.dumps(dict(status="failed",error_type=type(exc).__name__,
                    error=str(exc),config=vars(args),health_before=health,peers=partial),
                    indent=2,default=str)+"\n")
            raise
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            for recorder in recorders:
                await recorder.stop()
            await asyncio.gather(*(pc.close() for pc in peers),return_exceptions=True)
            await asyncio.gather(*(api(f"/calls/{cid}","DELETE") for cid in ids),return_exceptions=True)
        # Successful completion also verifies that all admitted peer states were
        # actually released; before-cleanup stats alone cannot establish that.
        result["health_after_cleanup"]=await api("/health")
        result["cleanup_pass"]=all(result["health_after_cleanup"].get(k)==health.get(k)
                                   for k in ("calls","sessions","gpu_states"))
        output.write_text(json.dumps(result,indent=2,default=str)+"\n")
        if not result["cleanup_pass"]:
            raise RuntimeError("Peer/GPU-state counts did not return to the pre-test baseline; inspect evidence")


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url",default="http://127.0.0.1:8765")
    ap.add_argument("--sessions",type=int,default=1)
    ap.add_argument("--speakers",type=int,help="Number of peers sending turns; other connected peers only receive idle")
    ap.add_argument("--turns",type=int,default=2)
    ap.add_argument("--duration-seconds",type=float,default=0,
                    help="Soak: repeat turns until this duration is reached, overriding --turns")
    ap.add_argument("--audio",nargs="+",required=True)
    ap.add_argument("--audio-seconds",type=float,default=3)
    ap.add_argument("--gap",type=float,default=2)
    ap.add_argument("--idle-seconds",type=float,default=1)
    ap.add_argument("--timeout",type=float,default=180)
    ap.add_argument("--record",action="store_true")
    ap.add_argument("--compact-evidence",action="store_true",help="Store full timestamp arrays in a companion gzip file")
    ap.add_argument("--snapshots-every",type=float,default=0,
                    help="Save one native receiver image from peer zero at this interval; zero disables")
    ap.add_argument("--interrupt",action="store_true")
    ap.add_argument("--interrupt-after",type=float,default=.6,
                    help="Interrupt after this much generated playback, not just wall time after upload")
    ap.add_argument("--resume-after-interrupt",action="store_true")
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    if args.sessions<1 or args.turns<1 or args.duration_seconds<0 or args.snapshots_every<0 or not 0<args.audio_seconds<=30:
        ap.error("Require positive sessions/turns and audio-seconds <=30")
    if (not all(math.isfinite(v) for v in (args.duration_seconds,args.gap,args.idle_seconds,
            args.timeout,args.snapshots_every,args.interrupt_after)) or min(args.gap,args.idle_seconds)<0 or args.timeout<=0):
        ap.error("Times must be finite, gaps nonnegative and timeout positive")
    if args.interrupt and not 0<=args.interrupt_after<args.audio_seconds:
        ap.error("interrupt-after must be nonnegative and shorter than the audio fixture")
    if args.resume_after_interrupt and not args.interrupt:
        ap.error("resume-after-interrupt requires --interrupt")
    if args.speakers is not None and (not 0<=args.speakers<=args.sessions or (args.speakers==0 and not args.duration_seconds)):
        ap.error("speakers must be 0..sessions; zero speakers requires a positive duration")
    asyncio.run(run(args))


if __name__=="__main__":
    main()
