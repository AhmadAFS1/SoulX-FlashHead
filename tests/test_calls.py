import asyncio
import io
import time
from types import SimpleNamespace

import aiohttp
import av
import numpy as np
import soundfile as sf
import pytest
from aiohttp import web
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError

from soulx_rtc.calls import Call, IdleVideo, Turn, CallVideoTrack, CallAudioTrack
from soulx_rtc.server import Service, make_app


class FakeEngine:
    last_metrics = {"wall_s": .001}
    reconditions = 0

    def prepare_call(self, path, seed):
        return SimpleNamespace(cursor=0, total_frames=0)

    def append(self, state, audio):
        state.total_frames = state.cursor + int(np.ceil(len(audio)*25/16000/24))*24

    def recondition(self, state, displayed):
        assert displayed.shape == (9, 64, 64, 3)
        state.cursor = state.total_frames = 0
        self.reconditions += 1

    def generate(self, states):
        for state in states:
            state.cursor += 24
        return [np.full((24,64,64,3),120,np.uint8) for _ in states]


def wav(seconds=.2, amplitude=.05):
    data = io.BytesIO()
    sf.write(data, np.sin(np.arange(round(seconds*16000))*.05)*amplitude,16000,format="WAV")
    return data.getvalue()


@pytest.mark.parametrize("fast_codec",[False,True])
def test_persistent_real_peer_turns_retry_interrupt_cleanup(monkeypatch,fast_codec):
    monkeypatch.delenv("SOULX_API_TOKEN",raising=False)
    if fast_codec:
        import aiortc.rtcrtpsender
        from soulx_rtc.codec import install_encoder_factory
        monkeypatch.setattr(aiortc.rtcrtpsender,"get_encoder",aiortc.rtcrtpsender.get_encoder)
        install_encoder_factory(25,"veryfast")
    async def run():
        service = Service(SimpleNamespace(batch=1,max_sessions=1,size=64,width=64,height=64,
                                         steps=4,fps=25,max_active_calls=1,idle_video=None))
        service.engine = FakeEngine()
        async def startup(app):
            service.ready = True
            service.task = asyncio.create_task(service.schedule())
        service.startup = startup
        runner = web.AppRunner(make_app(service))
        await runner.setup()
        site = web.TCPSite(runner,"127.0.0.1",0)
        await site.start()
        url = "http://127.0.0.1:"+str(site._server.sockets[0].getsockname()[1])
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        pts, tasks = {"audio":[],"video":[]},[]
        async def consume(track):
            try:
                while True:
                    f = await track.recv()
                    pts[track.kind].append(float(f.pts*f.time_base))
            except MediaStreamError:
                pass
        @pc.on("track")
        def track(t):
            tasks.append(asyncio.create_task(consume(t)))
        try:
            async with aiohttp.ClientSession() as client:
                async with client.post(url+"/calls",json={"seed":2**100}) as r:
                    assert r.status==400 and service.ready
                async with client.post(url+"/calls",json={"seed":51}) as r:
                    assert r.status==201,await r.text()
                    cid=(await r.json())["id"]
                base=url+"/calls/"+cid
                async with client.post(url+"/sessions",json={"seconds":1}) as r:
                    assert r.status==429
                pc.addTransceiver("video",direction="recvonly")
                pc.addTransceiver("audio",direction="recvonly")
                await pc.setLocalDescription(await pc.createOffer())
                offer={"type":"offer","sdp":pc.localDescription.sdp}
                async with client.post(base+"/offer",json=offer) as r:
                    assert r.status==200,await r.text()
                    await pc.setRemoteDescription(RTCSessionDescription(**await r.json()))
                async with client.post(base+"/offer",json=offer) as r:
                    assert r.status==409
                for name in ("one","two"):
                    async with client.post(base+"/turns",data=wav(),headers={"X-Turn-ID":name}) as r:
                        assert r.status==202,await r.text()
                async with client.post(base+"/turns",data=wav(),headers={"X-Turn-ID":"one"}) as r:
                    assert r.status==200
                async with client.post(base+"/turns",data=wav(amplitude=.1),headers={"X-Turn-ID":"one"}) as r:
                    assert r.status==409
                c=service.calls.items[cid]
                deadline=time.monotonic()+8
                while time.monotonic()<deadline and c.turns["two"].status!="complete":
                    assert not c.error,c.error
                    await asyncio.sleep(.05)
                assert c.turns["one"].status==c.turns["two"].status=="complete"
                assert all(t.audio_sent==3200*3 for t in c.turns.values())
                assert all(len(t.audio)==len(t.audio48)==0 for t in c.turns.values())
                assert c.negotiations==1 and c.generated_frames==48
                assert c.metrics()["video_encoders"][0]["encoder"]==("FastH264Encoder" if fast_codec else "H264Encoder")
                async with client.post(base+"/turns",data=wav(2),headers={"X-Turn-ID":"cancel"}) as r:
                    assert r.status==202
                await asyncio.sleep(.25)
                sent_before=c.video_sent
                async with client.post(base+"/interrupt",json={}) as r:
                    assert r.status==200,await r.text()
                assert c.epoch==1 and c.turns["cancel"].status=="interrupted"
                assert service.engine.reconditions==1 and not c.current_frames and c.chunks.empty()
                await asyncio.sleep(.15)
                assert c.video_sent>sent_before
                assert all(len(v)>20 and all(b>a for a,b in zip(v,v[1:])) for v in pts.values())
                stats=[s for s in (await pc.getStats()).values() if s.type=="inbound-rtp"]
                assert len(stats)==2 and all(s.packetsLost==0 for s in stats)
                for _ in range(2):
                    async with client.delete(base) as r:
                        assert r.status==204
                assert not service.calls.items
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks,return_exceptions=True)
            await pc.close()
            await runner.cleanup()
    asyncio.run(run())


def test_no_idle_switch_before_audio_drain():
    async def run():
        idle=IdleVideo(None,64,64,25,np.zeros((64,64,3),np.uint8))
        c=Call("test",None,25,idle)
        c.sent_history.append(np.full((64,64,3),99,np.uint8))
        c.active=Turn("t","hash",np.ones(16000,np.float32),np.ones(48000,np.int16),24,
                      start_frame=0,start_sample=0,video_sent=24,audio_sent=47040)
        frame=await CallVideoTrack(c).recv()
        assert np.all(frame.to_ndarray(format="rgb24")==99)
        assert c.active is not None
        c.active.audio_sent=48000
        c.finish_if_drained()
        assert c.active is None
    asyncio.run(run())


def test_idle_clock_does_not_build_a_catchup_backlog():
    async def run():
        c=Call("clock",None,25,IdleVideo(None,64,64,25,np.zeros((64,64,3),np.uint8)))
        c.origin=time.monotonic()-2
        c.audio_sent=96000
        track=CallVideoTrack(c)
        first=await track.recv()
        assert first.pts>=180000 and c.video_sent==1
        assert c.missed_video_slots>=50
        c.active=Turn("t","digest",np.ones(100,np.float32),np.ones(300,np.int16),24)
        c.chunks.put_nowait((0,np.ones((24,64,64,3),np.uint8)))
        await track.recv()
        assert c.active.start_frame<=c.video_clock+3
    asyncio.run(run())


def test_audio_boundary_race_has_one_packet_allowance_but_stalls_are_counted():
    async def run():
        c=Call("audio-clock",None,25,IdleVideo(None,64,64,25,np.zeros((64,64,3),np.uint8)))
        c.origin=time.monotonic()-.2
        c.audio_sent=1920
        c.active=Turn("speech","digest",np.ones(16000,np.float32),np.ones(48000,np.int16),48,
                      start_frame=0,start_sample=0,video_sent=1,audio_sent=1920)
        track=CallAudioTrack(c)
        packet=await track.recv()
        assert np.all(packet.to_ndarray()==1) and c.active.audio_sent==2880
        assert c.audio_hold_samples==0
        packet=await track.recv()  # Video still has not advanced: exhaust allowance.
        assert not packet.to_ndarray().any() and c.active.audio_sent==2880
        assert c.audio_hold_samples==960
        c.active.video_sent=2
        packet=await track.recv()
        assert np.all(packet.to_ndarray()==1) and c.active.audio_sent==3840
    asyncio.run(run())


def test_generated_silence_keeps_state_without_filling_turn_history():
    async def run():
        service=Service(SimpleNamespace(batch=1,max_active_calls=1))
        service.engine=FakeEngine()
        idle=IdleVideo(None,64,64,25,np.zeros((64,64,3),np.uint8))
        c=Call("generated-idle",SimpleNamespace(cursor=0,total_frames=0),25,idle,
               connected=True,idle_policy="generate")
        service.calls.items[c.id]=c
        try:
            assert await service.calls.schedule_once()
            assert c.active.synthetic_idle and not c.turns
            assert c.generated_idle_frames==24 and c.state.cursor==24
            # Simulate fully drained first chunk without resetting model state.
            c.chunks.get_nowait()
            c.active.video_sent=24
            c.active.audio_sent=c.active.useful_samples
            c.finish_if_drained()
            turn=Turn("speech","digest",np.ones(3200,np.float32),np.ones(9600,np.int16),24)
            c.queue.append(turn)
            c.turns[turn.id]=turn
            assert await service.calls.schedule_once()
            assert c.active is turn and not c.active.synthetic_idle
            assert c.state.cursor==48 and c.generated_idle_frames==24
            assert service.engine.reconditions==0
        finally:
            await service.cleanup(None)
    asyncio.run(run())


def test_call_microbatch_respects_active_admission_and_private_outputs():
    async def run():
        service=Service(SimpleNamespace(batch=2,max_active_calls=2))
        service.engine=FakeEngine()
        for i in range(3):
            c=Call(str(i),SimpleNamespace(cursor=0,total_frames=0),25,
                   IdleVideo(None,64,64,25,np.zeros((64,64,3),np.uint8)),connected=True)
            turn=Turn(str(i),"digest",np.ones(3200,np.float32),np.ones(9600,np.int16),24)
            c.queue.append(turn)
            c.turns[turn.id]=turn
            service.calls.items[c.id]=c
        try:
            assert await service.calls.schedule_once()
            calls=list(service.calls.items.values())
            assert [c.state.cursor for c in calls]==[24,24,0]
            assert [c.chunks.qsize() for c in calls]==[1,1,0]
            assert calls[0].active is not calls[1].active
            assert not await service.calls.schedule_once()  # active turns not drained
            for c in calls[:2]:
                c.chunks.get_nowait()
                c.active.video_sent=24
                c.active.audio_sent=c.active.useful_samples
                c.finish_if_drained()
            assert await service.calls.schedule_once()
            assert calls[2].state.cursor==24
        finally:
            await service.cleanup(None)
    asyncio.run(run())


def test_shared_worker_failure_marks_all_calls_unhealthy():
    service=Service(SimpleNamespace(batch=1,max_active_calls=1))
    service.ready=True
    for i in range(2):
        service.calls.items[str(i)]=Call(str(i),None,25,
            IdleVideo(None,64,64,25,np.zeros((64,64,3),np.uint8)))
    service.mark_unhealthy(RuntimeError("simulated GPU failure"))
    assert not service.ready
    assert all(c.error=="RuntimeError" and c.changed.is_set() for c in service.calls.items.values())
    service.executor.shutdown()


def test_generated_idle_yields_admission_to_another_peers_speech():
    async def run():
        service=Service(SimpleNamespace(batch=1,max_active_calls=1))
        service.engine=FakeEngine()
        for i in range(2):
            service.calls.items[str(i)]=Call(str(i),SimpleNamespace(cursor=0,total_frames=0),25,
                IdleVideo(None,64,64,25,np.zeros((64,64,3),np.uint8)),
                connected=True,idle_policy="generate")
        idle,speaker=list(service.calls.items.values())
        try:
            assert await service.calls.schedule_once()
            assert idle.active.synthetic_idle and speaker.active is None
            idle.chunks.get_nowait()  # The initial chunk is now being played.
            idle.active.video_sent=1
            turn=Turn("speech","digest",np.ones(3200,np.float32),np.ones(9600,np.int16),24)
            speaker.queue.append(turn)
            speaker.turns[turn.id]=turn
            assert not await service.calls.schedule_once()  # Stop idle extension.
            assert idle.state.cursor==idle.active.frames==24
            idle.active.video_sent=24
            idle.active.audio_sent=idle.active.useful_samples
            idle.finish_if_drained()
            assert await service.calls.schedule_once()
            assert speaker.active is turn and speaker.state.cursor==24
            assert idle.active is None and idle.state.cursor==24
        finally:
            await service.cleanup(None)
    asyncio.run(run())


def test_generated_idle_renders_ahead_without_rearming_or_unbounded_queue():
    async def run():
        service=Service(SimpleNamespace(batch=1,max_active_calls=1))
        service.engine=FakeEngine()
        c=Call("ahead",SimpleNamespace(cursor=0,total_frames=0),25,
               IdleVideo(None,64,64,25,np.zeros((64,64,3),np.uint8)),
               connected=True,idle_policy="generate")
        service.calls.items[c.id]=c
        try:
            assert await service.calls.schedule_once()
            original=c.active
            c.chunks.get_nowait()
            original.video_sent,original.audio_sent=1,960
            original.start_frame,original.start_sample=3,5760
            assert await service.calls.schedule_once()
            assert c.active is original and original.frames==48 and c.state.cursor==48
            assert original.start_frame==3 and c.chunks.qsize()==1
            assert not await service.calls.schedule_once()
            assert not c.turns and c.generated_idle_frames==48
            # Logical generated silence can pass the original zero-array length
            # without allocating an ever-growing waveform or losing its clock.
            c.origin=time.monotonic()-3
            c.audio_sent=48000
            original.video_sent,original.audio_sent=48,48000
            frame=await CallAudioTrack(c).recv()
            assert original.audio_sent==48960 and not frame.to_ndarray().any()
        finally:
            await service.cleanup(None)
    asyncio.run(run())
