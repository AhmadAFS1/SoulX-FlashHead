"""Wall/TTS HTTP contracts without loading neural model weights."""
import asyncio
import io
import json
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from aiohttp.test_utils import TestClient, TestServer

from soulx_rtc.server import Service, make_app
from soulx_rtc.tts import KokoroService, SAMPLE_RATE


def fake_pipeline(monkeypatch, samples):
    class Pipeline:
        def __init__(self, **kwargs):
            self.model = kwargs.get('model')
        def __call__(self, *args, **kwargs):
            yield SimpleNamespace(audio=samples)
    monkeypatch.setitem(sys.modules, 'kokoro', SimpleNamespace(KPipeline=Pipeline))


@pytest.mark.parametrize('params', [None, [], {}, {'text': 2}, {'text': ' '},
    {'text': 'x'*2001}, {'text': 'hello', 'voice': '../voice'},
    {'text': 'hello', 'speed': float('nan')}, {'text': 'hello', 'speed': None},
    {'text': 'hello', 'language_code': 'b', 'voice': 'af_heart'}])
def test_invalid_tts_request(params):
    with pytest.raises((ValueError, TypeError)):
        KokoroService.validate(params)


def test_synthesis_wav_duration_and_warm_model(monkeypatch):
    fake_pipeline(monkeypatch, np.ones(2400, np.float32)*.1)
    tts = KokoroService()
    for cold in ('1', '0'):
        data, headers = tts.synthesize('Hello', 'af_heart', 'a', 1)
        with sf.SoundFile(io.BytesIO(data)) as audio:
            assert audio.samplerate == SAMPLE_RATE and audio.frames == 2400
        assert headers['X-Kokoro-Cold-Start'] == cold
        assert headers['X-Kokoro-Audio-Seconds'] == '0.100'


@pytest.mark.parametrize('samples,match', [(np.zeros(30*SAMPLE_RATE+1), '30 seconds'),
    (np.array([np.nan]), 'non-finite'), (np.zeros(0), 'no speech')])
def test_synthesis_rejects_invalid_output(monkeypatch, samples, match):
    fake_pipeline(monkeypatch, samples)
    with pytest.raises((ValueError, RuntimeError), match=match):
        KokoroService().synthesize('Hello', 'af_heart', 'a', 1)


def test_wall_auth_tts_busy_and_responsive_status(monkeypatch):
    monkeypatch.setenv('SOULX_API_TOKEN', 'wall-test-token')
    async def run():
        service = Service(SimpleNamespace(batch=1, max_sessions=3, size=64, steps=4, fps=25))
        async def startup(app):
            service.ready = True
        service.startup = startup
        entered, release = threading.Event(), threading.Event()
        def slow_synthesis(*args):
            entered.set()
            assert release.wait(5)
            return b'RIFF-test', {'X-Kokoro-Audio-Seconds': '0.1'}
        service.tts.synthesize = slow_synthesis
        async with TestClient(TestServer(make_app(service))) as client:
            for path in ('/', '/webrtc/wall', '/webrtc/wall.js'):
                async with client.get(path) as response:
                    assert response.status == 200
            for path in ('/health', '/config', '/webrtc/tts/kokoro/status'):
                assert (await client.get(path)).status == 401
            assert (await client.post('/webrtc/tts/kokoro', json={'text': 'hello'})).status == 401
            headers = {'Authorization': 'Bearer wall-test-token'}
            assert (await client.post('/webrtc/tts/kokoro', json=[], headers=headers)).status == 400
            first = asyncio.create_task(client.post('/webrtc/tts/kokoro', json={'text': 'hello'}, headers=headers))
            try:
                assert await asyncio.to_thread(entered.wait, 3)
                response = await asyncio.wait_for(client.get('/webrtc/tts/kokoro/status', headers=headers), 1)
                assert (await response.json())['busy']
                assert (await client.get('/health', headers=headers)).status == 200
                second = await client.post('/webrtc/tts/kokoro', json={'text': 'hello'}, headers=headers)
                assert second.status == 429
                assert second.headers['Retry-After'] == '2'
            finally:
                release.set()
            response = await first
            assert response.status == 200 and response.content_type == 'audio/wav'
            assert await response.read() == b'RIFF-test'
    asyncio.run(run())


def test_anonymous_split_browser_and_server_ice(monkeypatch):
    browser = '[{"urls":["turn:public.example:50685?transport=tcp"],"username":"web","credential":"secret"}]'
    server = '[{"urls":["turn:127.0.0.1:1455?transport=tcp"],"username":"web","credential":"secret"}]'
    monkeypatch.setenv('SOULX_API_TOKEN', 'ignored-in-anonymous-mode')
    monkeypatch.setenv('SOULX_BROWSER_ICE_SERVERS', browser)
    monkeypatch.setenv('SOULX_SERVER_ICE_SERVERS', server)
    monkeypatch.setenv('SOULX_ICE_TRANSPORT_POLICY', 'relay')

    async def run():
        service = Service(SimpleNamespace(batch=1, max_sessions=1, size=64, steps=4,
                                          fps=25, allow_anonymous=True))
        async def startup(app):
            service.ready = True
        service.startup = startup
        async with TestClient(TestServer(make_app(service))) as client:
            response = await client.get('/config')
            assert response.status == 200
            config = await response.json()
            assert config == {
                'iceServers': json.loads(browser),
                'iceTransportPolicy': 'relay',
                'authRequired': False,
            }
            assert service.server_ice == json.loads(server)

    try:
        asyncio.run(run())
    finally:
        from soulx_rtc.server import _apply_ice_transport_policy
        _apply_ice_transport_policy('all')
