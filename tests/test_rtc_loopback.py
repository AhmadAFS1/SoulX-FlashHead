"""CPU-only smoke test through real ICE, DTLS, SRTP, H264 and Opus."""
import asyncio
import time
from types import SimpleNamespace

import aiohttp
import numpy as np
from aiohttp import web
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from soulx_rtc.server import Service, make_app


def test_real_media_loopback(monkeypatch):
    monkeypatch.delenv("SOULX_API_TOKEN", raising=False)
    class FakeEngine:
        last_metrics = {"wall_s": .001}
        def prepare(self, path, audio, seed):
            return SimpleNamespace(total_frames=10, cursor=0)
        def generate(self, states):
            chunks = []
            for s in states:
                s.cursor = 10
                chunks.append(np.stack([np.full((64, 64, 3), i * 20, np.uint8) for i in range(10)]))
            return chunks
    async def run():
        service = Service(SimpleNamespace(batch=1, max_sessions=2, size=64, steps=0, fps=25))
        service.engine = FakeEngine()
        async def startup(app):
            service.ready = True
            service.task = asyncio.create_task(service.schedule())
        service.startup = startup
        runner = web.AppRunner(make_app(service))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        url = "http://127.0.0.1:" + str(site._server.sockets[0].getsockname()[1])
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        count, tasks = {"video": 0, "audio": 0}, []
        async def read(track):
            while True:
                await track.recv()
                count[track.kind] += 1
        @pc.on("track")
        def track(t):
            tasks.append(asyncio.create_task(read(t)))
        try:
            async with aiohttp.ClientSession() as client:
                async with client.post(url + "/sessions", json={"seconds": .4}) as r:
                    assert r.status == 201
                    sid = (await r.json())["id"]
                pc.addTransceiver("video", direction="recvonly")
                pc.addTransceiver("audio", direction="recvonly")
                await pc.setLocalDescription(await pc.createOffer())
                async with client.post(url + f"/sessions/{sid}/offer", json={
                        "sdp": pc.localDescription.sdp, "type": "offer"}) as r:
                    assert r.status == 200
                    await pc.setRemoteDescription(RTCSessionDescription(**await r.json()))
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline and (count["video"] < 10 or count["audio"] < 20):
                    await asyncio.sleep(.05)
                assert count["video"] >= 10 and count["audio"] >= 20, count
                assert service.sessions[sid].video_sent == 10
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await pc.close()
            await runner.cleanup()
    asyncio.run(run())
