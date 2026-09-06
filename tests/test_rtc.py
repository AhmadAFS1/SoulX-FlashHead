import asyncio
import io
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from soulx_rtc.server import AudioTrack, Session, VideoTrack, decode_audio


def test_audio_decode_resamples_and_limits():
    data = io.BytesIO()
    sf.write(data, np.zeros((44100, 2), np.float32), 44100, format="WAV")
    out = decode_audio(data.getvalue(), .5)
    assert out.shape == (8000,)
    assert out.dtype == np.float32


def test_audio_rejects_empty():
    data = io.BytesIO()
    sf.write(data, np.zeros(0, np.float32), 16000, format="WAV")
    with pytest.raises(ValueError):
        decode_audio(data.getvalue(), 10)


def test_tracks_monotonic_and_bounded():
    async def run():
        s = Session("test", SimpleNamespace(total_frames=5, cursor=5), np.zeros(9600, np.int16))
        s.activated = s.created
        s.queue.put_nowait(np.zeros((5, 32, 32, 3), np.uint8))
        video, audio = VideoTrack(s), AudioTrack(s)
        async def videos():
            return [await video.recv() for _ in range(5)]
        async def audios():
            return [await audio.recv() for _ in range(10)]
        v, a = await asyncio.gather(videos(), audios())
        assert [f.pts for f in v] == [0, 3600, 7200, 10800, 14400]
        assert [f.pts for f in a] == list(range(0, 9600, 960))
        assert s.queue.maxsize == 2
        assert s.stalls_s < .04
    asyncio.run(run())


def test_cancel_unblocks_waiting_tracks():
    async def run():
        s = Session("test", SimpleNamespace(total_frames=5, cursor=0), np.zeros(9600, np.int16))
        video, audio = VideoTrack(s), AudioTrack(s)
        tasks = [asyncio.create_task(video.recv()), asyncio.create_task(audio.recv())]
        await asyncio.sleep(.02)
        s.closed = True
        s.changed.set()
        results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 1)
        assert all(isinstance(r, Exception) for r in results)
    asyncio.run(run())


def test_rope_batch_equals_separate():
    import torch
    from flash_head.src.modules.flash_head_model import rope_apply, precompute_freqs_cis_3d
    x = torch.randn(2, 20, 4, 24)
    freq = precompute_freqs_cis_3d(24)
    batch = rope_apply(x, freq, (5, 2, 2))
    singles = torch.cat([rope_apply(row[None], freq, (5, 2, 2)) for row in x])
    torch.testing.assert_close(batch, singles, rtol=0, atol=0)
    assert not torch.equal(batch[0], batch[1])


def test_dit_audio_block_preserves_batch(monkeypatch):
    import torch
    import flash_head.src.modules.flash_head_model as model
    attention = model.flash_attention
    monkeypatch.setattr(model, "flash_attention", lambda q,k,v,num_heads: attention(q,k,v,num_heads,True))
    torch.manual_seed(17)
    block = model.DiTAudioBlock(has_image_input=False, dim=96, ffn_dim=192, num_heads=4).eval()
    x, context, t = torch.randn(2, 20, 96), torch.randn(2, 5, 32, 96), torch.randn(2, 6, 96)
    freq = model.precompute_freqs_cis_3d(24)
    with torch.no_grad():
        batch = block(x, context, t, freq, (5, 2, 2))
        separate = torch.cat([block(x[i:i+1], context[i:i+1], t[i:i+1], freq, (5, 2, 2)) for i in range(2)])
    torch.testing.assert_close(batch, separate, rtol=1e-4, atol=2e-5)


def test_full_queue_does_not_block_other_sessions():
    from soulx_rtc.server import Service
    class FakeEngine:
        last_metrics = {"wall_s": .001}
        def generate(self, states):
            for state in states:
                state.cursor += 1
            return [np.zeros((1, 32, 32, 3), np.uint8) for _ in states]
    async def run():
        service = Service(SimpleNamespace(batch=1))
        service.engine = FakeEngine()
        stalled = Session("stalled", SimpleNamespace(total_frames=5, cursor=0), np.zeros(1, np.int16))
        active = Session("active", SimpleNamespace(total_frames=1, cursor=0), np.zeros(1, np.int16))
        for s in (stalled, active):
            s.connected = True
            s.activated = s.created
            service.sessions[s.id] = s
            service.rotation.append(s.id)
        for _ in range(2):
            stalled.queue.put_nowait(np.zeros((1, 32, 32, 3), np.uint8))
        service.task = asyncio.create_task(service.schedule())
        await asyncio.wait_for(active.queue.get(), 1)
        assert active.state.cursor == 1 and stalled.state.cursor == 0
        await service.cleanup(None)
    asyncio.run(run())


def test_http_auth_and_capacity(monkeypatch):
    import aiohttp
    from aiohttp import web
    from soulx_rtc.server import Service, make_app
    class FakeEngine:
        def prepare(self, path, audio, seed):
            return SimpleNamespace(total_frames=25, cursor=0)
    async def run():
        monkeypatch.setenv("SOULX_API_TOKEN", "test-only-token")
        service = Service(SimpleNamespace(batch=1, max_sessions=1, size=512, steps=4, fps=25))
        service.engine = FakeEngine()
        async def startup(app):
            service.ready = True
        service.startup = startup
        runner = web.AppRunner(make_app(service))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
        try:
            async with aiohttp.ClientSession() as client:
                async with client.get(url + "/health") as r:
                    assert r.status == 401
                headers = {"Authorization": "Bearer test-only-token"}
                async with client.post(url + "/sessions", json={"seconds": 1}, headers=headers) as r:
                    assert r.status == 201
                    sid = (await r.json())["id"]
                async with client.post(url + "/sessions", json={"seconds": 1}, headers=headers) as r:
                    assert r.status == 429
                async with client.delete(url + "/sessions/" + sid, headers=headers) as r:
                    assert r.status == 204
                assert not service.sessions and service.pending == 0
        finally:
            await runner.cleanup()
    asyncio.run(run())
