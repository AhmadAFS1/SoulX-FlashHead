import asyncio
from types import SimpleNamespace

import av
from aiohttp import web

from soulx_rtc.benchmark_calls import run
from soulx_rtc.server import Service, make_app
from soulx_rtc.verify_boundaries import verify
from test_calls import FakeEngine, wav


def test_recorded_call_contains_portrait_video_and_audio(tmp_path,monkeypatch):
    monkeypatch.delenv('SOULX_API_TOKEN',raising=False)
    audio=tmp_path/'speech.wav'
    audio.write_bytes(wav())
    async def scenario():
        service=Service(SimpleNamespace(batch=1,max_sessions=1,size=64,width=64,height=64,
                                       steps=4,fps=25,max_active_calls=1,idle_video=None))
        service.engine=FakeEngine()
        async def startup(app):
            service.ready=True
            service.task=asyncio.create_task(service.schedule())
        service.startup=startup
        runner=web.AppRunner(make_app(service))
        await runner.setup()
        site=web.TCPSite(runner,'127.0.0.1',0)
        await site.start()
        args=SimpleNamespace(url='http://127.0.0.1:'+str(site._server.sockets[0].getsockname()[1]),
            sessions=1,speakers=1,turns=1,duration_seconds=0,audio=[str(audio)],audio_seconds=.2,
            gap=0,idle_seconds=.3,timeout=10,record=True,compact_evidence=False,
            snapshots_every=0,interrupt=False,resume_after_interrupt=False,
            output=str(tmp_path/'result.json'))
        try:
            await run(args)
        finally:
            await runner.cleanup()
    asyncio.run(scenario())
    import json
    report=json.loads((tmp_path/'result.json').read_text())
    assert verify(report)['passed']
    report['peers'][0]['boundary_final']['boundary_events'][0]['exact']=False
    assert not verify(report)['passed']
    with av.open(str(tmp_path/'result-peer0.mp4')) as video:
        assert len(video.streams.video)==len(video.streams.audio)==1
        assert next(video.decode(video=0)).width==64
