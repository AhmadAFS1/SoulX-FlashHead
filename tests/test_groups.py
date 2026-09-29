"""MuseTalk-compatible wall groups over persistent calls. [CPU] contracts only.

A deterministic fake engine replaces neural generation; one aiortc peer checks
that group speech reaches a real negotiated call. Nothing here measures speed.
"""
import asyncio
import io
import time
from types import SimpleNamespace

import aiohttp
import numpy as np
import pytest
import soundfile as sf
from aiohttp import web
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError
from PIL import Image

from soulx_rtc import groups as groups_module
from soulx_rtc.server import Service, make_app

TOKEN = 'group-test-token'
AUTH = {'Authorization': 'Bearer ' + TOKEN}


class FakeEngine:
    last_metrics = {'wall_s': .001, 'frames': 24, 'batch': 1}

    def __init__(self, fail_on=None):
        self.prepared, self.fail_on = 0, fail_on

    def prepare_call(self, path, seed):
        self.prepared += 1
        if self.prepared == self.fail_on:
            raise OSError('prepare failed for test')
        return SimpleNamespace(cursor=0, total_frames=0, seed=seed)

    def append(self, state, audio):
        state.total_frames = state.cursor + int(np.ceil(len(audio)*25/16000/24))*24

    def recondition(self, state, displayed):
        state.cursor = state.total_frames = 0

    def generate(self, states):
        for state in states:
            state.cursor += 24
        return [np.full((24, 64, 64, 3), 90, np.uint8) for _ in states]


def wav(seconds=.2, amplitude=.05):
    data = io.BytesIO()
    sf.write(data, np.sin(np.arange(round(seconds*16000))*.05)*amplitude, 16000, format='WAV')
    return data.getvalue()


def audio_form(data, **fields):
    form = aiohttp.FormData()
    form.add_field('audio_file', data, filename='turn.wav', content_type='audio/wav')
    for key, value in fields.items():
        form.add_field(key, value)
    return form


async def serve(service):
    runner = web.AppRunner(make_app(service))
    await runner.setup()
    site = web.TCPSite(runner, '127.0.0.1', 0)
    await site.start()
    return runner, 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])


def make_service(monkeypatch, tmp_path, max_sessions=4, engine=None, token=TOKEN):
    if token:
        monkeypatch.setenv('SOULX_API_TOKEN', token)
    else:
        monkeypatch.delenv('SOULX_API_TOKEN', raising=False)
    portrait = tmp_path/'portrait.png'
    Image.new('RGB', (96, 96), (120, 90, 60)).save(portrait)
    service = Service(SimpleNamespace(batch=2, max_sessions=max_sessions, size=64, width=64, height=64,
        steps=4, fps=25, max_active_calls=2, idle_video=None, idle_policy='hold'))
    service.engine = engine or FakeEngine()
    service.calls.avatars.entries['fixture'] = dict(id='fixture', name='Fixture', path=str(portrait), kind='image')

    async def startup(app):
        service.ready = True
        service.task = asyncio.create_task(service.schedule())
    service.startup = startup
    return service


def test_pages_are_public_and_apis_keep_the_token(monkeypatch, tmp_path):
    async def run():
        service = make_service(monkeypatch, tmp_path)
        runner, url = await serve(service)
        try:
            async with aiohttp.ClientSession() as client:
                for path in ('/', '/clip', '/webrtc/wall', '/webrtc/lab', '/webrtc/wall.js',
                             '/webrtc/player/AbC_-1', '/webrtc/groups/0a1b2c3d4e/wall'):
                    async with client.get(url+path) as r:
                        assert r.status == 200, path
                        assert r.headers['Cache-Control'].startswith('no-store')
                        body = await r.text()
                        assert TOKEN not in body
                async with client.get(url+'/webrtc/wall') as r:
                    assert 'FlashHead WebRTC latency wall' in await r.text()
                async with client.get(url+'/webrtc/player/x') as r:
                    assert 'webrtc-token-request' in await r.text()
                for method, path in (('POST', '/webrtc/groups/create'), ('GET', '/webrtc/groups/abc'),
                                     ('DELETE', '/webrtc/groups/abc'), ('POST', '/webrtc/groups/abc/stream'),
                                     ('POST', '/webrtc/groups/abc/reconnect'), ('GET', '/stats/gpu-live'),
                                     ('GET', '/webrtc/player/a/b'), ('POST', '/webrtc/player/abc'),
                                     ('POST', '/webrtc/wall')):
                    async with client.request(method, url+path) as r:
                        assert r.status == 401, (method, path)
        finally:
            await runner.cleanup()
    asyncio.run(run())


def test_group_lifecycle_idempotent_speech_subset_interrupt_reconnect(monkeypatch, tmp_path):
    async def run():
        service = make_service(monkeypatch, tmp_path)
        runner, url = await serve(service)
        try:
            async with aiohttp.ClientSession(headers=AUTH) as client:
                for query in ('count=0', 'count=5', 'count=x', 'count=2&avatar_id=../../etc/passwd', 'count=2&seed=-1'):
                    async with client.post(url+'/webrtc/groups/create?'+query) as r:
                        assert r.status in (400, 429), query
                assert not service.calls.items and service.pending == 0
                async with client.post(url+'/webrtc/groups/create?count=3&seed=7&avatar_id=fixture') as r:
                    assert r.status == 201, await r.text()
                    group = await r.json()
                gid, ids = group['group_id'], group['session_ids']
                assert group['count'] == 3 and len(set(ids)) == 3
                assert [s['seed'] for s in group['sessions']] == [7, 8, 9]
                assert [service.calls.items[i].state.seed for i in ids] == [7, 8, 9]
                assert group['wall_url'] == f'/webrtc/groups/{gid}/wall'
                assert all(s['player_url'] == f"/webrtc/player/{s['session_id']}" for s in group['sessions'])
                assert all(s['status'] == 'created' for s in group['sessions'])
                assert {k: group['config'][k] for k in ('count', 'first_seed', 'fps', 'width', 'height', 'batch_size')} == \
                    dict(count=3, first_seed=7, fps=25, width=64, height=64, batch_size=2)
                # Two free slots remain of four; a three-peer group is refused without leaks.
                async with client.post(url+'/webrtc/groups/create?count=3&avatar_id=fixture') as r:
                    assert r.status == 429
                assert len(service.calls.items) == 3 and service.pending == 0

                async with client.post(url+f'/webrtc/groups/{gid}/stream', data=audio_form(wav(), turn_id='t1')) as r:
                    body = await r.json()
                    assert r.status == 200 and body['started'] == 3 and body['failed'] == 0
                    assert {x['status'] for x in body['results']} == {'accepted'}
                # A retried request (lost response) never queues the same speech twice.
                async with client.post(url+f'/webrtc/groups/{gid}/stream', data=audio_form(wav(), turn_id='t1')) as r:
                    body = await r.json()
                    assert body['started'] == 3 and {x['status'] for x in body['results']} == {'duplicate'}
                assert all(list(service.calls.items[i].turns) == ['t1'] for i in ids)
                async with client.post(url+f'/webrtc/groups/{gid}/stream', data=audio_form(wav(amplitude=.1), turn_id='t1')) as r:
                    body = await r.json()
                    assert body['failed'] == 3 and {x['code'] for x in body['results']} == {409}
                async with client.post(url+f'/webrtc/groups/{gid}/stream',
                                       data=audio_form(wav(), turn_id='t2', session_ids=ids[1])) as r:
                    body = await r.json()
                    assert body['started'] == 1 and body['results'][0]['session_id'] == ids[1]
                assert [len(service.calls.items[i].turns) for i in ids] == [1, 2, 1]
                async with client.post(url+f'/webrtc/groups/{gid}/stream', data=audio_form(wav(), session_ids='nope')) as r:
                    assert r.status == 400
                async with client.post(url+f'/webrtc/groups/{gid}/stream', data=aiohttp.FormData({'turn_id': 'x'})) as r:
                    assert r.status == 400
                async with client.post(url+f'/webrtc/groups/{gid}/stream', data=audio_form(wav(), turn_id='_idle:x')) as r:
                    assert r.status == 400

                async with client.get(url+f'/webrtc/groups/{gid}') as r:
                    group = await r.json()
                assert group['sessions'][1]['last_turn']['id'] == 't2'
                assert all(s['status'] == 'queued' for s in group['sessions'])  # No peers connected yet.
                async with client.post(url+f'/webrtc/groups/{gid}/interrupt', json={'session_ids': [ids[0]]}) as r:
                    assert (await r.json())['interrupted'] == 1
                assert [service.calls.items[i].epoch for i in ids] == [1, 0, 0]
                async with client.post(url+f'/webrtc/groups/{gid}/interrupt') as r:
                    assert (await r.json())['interrupted'] == 3
                assert all(not service.calls.items[i].queue for i in ids)

                # A tile whose call ended (closed peer / page reload) is replaced, same seed.
                await service.calls.close(service.calls.items[ids[2]])
                async with client.get(url+f'/webrtc/groups/{gid}') as r:
                    assert (await r.json())['sessions'][2]['status'] == 'missing'
                async with client.post(url+f'/webrtc/groups/{gid}/stream', data=audio_form(wav(), turn_id='t3')) as r:
                    body = await r.json()
                    assert body['started'] == 2 and body['results'][2]['status'] == 'missing'
                async with client.post(url+f'/webrtc/groups/{gid}/reconnect') as r:
                    body = await r.json()
                assert [x['status'] for x in body['reconnect']] == ['kept', 'kept', 'replaced']
                new_id = body['sessions'][2]['session_id']
                assert new_id != ids[2] and body['reconnect'][2]['previous_session_id'] == ids[2]
                assert service.calls.items[new_id].state.seed == 9 and body['sessions'][2]['replacements'] == 1

                async with client.get(url+'/stats/gpu-live') as r:
                    assert r.status == 200 and 'available' in await r.json()
                async with client.delete(url+f'/webrtc/groups/{gid}') as r:
                    body = await r.json()
                    assert sorted(body['deleted_sessions']) == sorted(ids[:2]+[new_id]) and not body['errors']
                assert not service.calls.items
                for method in ('GET', 'DELETE'):
                    async with client.request(method, url+f'/webrtc/groups/{gid}') as r:
                        assert r.status == 404
        finally:
            await runner.cleanup()
    asyncio.run(run())


def test_group_creation_is_atomic_when_a_call_fails(monkeypatch, tmp_path):
    async def run():
        service = make_service(monkeypatch, tmp_path, engine=FakeEngine(fail_on=2))
        runner, url = await serve(service)
        try:
            async with aiohttp.ClientSession(headers=AUTH) as client:
                async with client.post(url+'/webrtc/groups/create', json={'count': 3, 'avatar_id': 'fixture'}) as r:
                    assert r.status == 400 and 'prepare failed' in await r.text()
            assert not service.calls.items and service.pending == 0 and not service.groups.groups
        finally:
            await runner.cleanup()
    asyncio.run(run())


def test_gpu_live_is_sampled_at_most_once_per_second(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(groups_module, 'sample_gpu', lambda index: calls.append(index) or
                        dict(available=True, gpu_util_pct=40., memory_used_gb=3.5))

    async def run():
        service = make_service(monkeypatch, tmp_path)
        runner, url = await serve(service)
        try:
            async with aiohttp.ClientSession(headers=AUTH) as client:
                bodies = await asyncio.gather(*(client.get(url+'/stats/gpu-live') for _ in range(4)))
                assert [(await b.json())['gpu_util_pct'] for b in bodies] == [40.]*4
        finally:
            await runner.cleanup()
    asyncio.run(run())
    assert len(calls) == 1


def test_group_speech_reaches_a_negotiated_peer(monkeypatch, tmp_path):
    async def run():
        service = make_service(monkeypatch, tmp_path, token=None)
        runner, url = await serve(service)
        pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        frames, tasks = {'audio': 0, 'video': 0}, []

        async def consume(track):
            try:
                while True:
                    await track.recv()
                    frames[track.kind] += 1
            except MediaStreamError:
                pass

        @pc.on('track')
        def track(t):
            tasks.append(asyncio.create_task(consume(t)))
        try:
            async with aiohttp.ClientSession() as client:
                async with client.post(url+'/webrtc/groups/create?count=2&avatar_id=fixture') as r:
                    group = await r.json()
                gid, first = group['group_id'], group['session_ids'][0]
                pc.addTransceiver('video', direction='recvonly')
                pc.addTransceiver('audio', direction='recvonly')
                await pc.setLocalDescription(await pc.createOffer())
                async with client.post(url+f'/calls/{first}/offer',
                                       json={'type': 'offer', 'sdp': pc.localDescription.sdp}) as r:
                    await pc.setRemoteDescription(RTCSessionDescription(**await r.json()))
                deadline = time.monotonic()+8
                while time.monotonic() < deadline and not service.calls.items[first].connected:
                    await asyncio.sleep(.05)
                async with client.post(url+f'/webrtc/groups/{gid}/stream',
                                       data=audio_form(wav(), turn_id='speech', session_ids=first)) as r:
                    assert (await r.json())['started'] == 1
                c = service.calls.items[first]
                while time.monotonic() < deadline and c.turns['speech'].status != 'complete':
                    assert not c.error, c.error
                    await asyncio.sleep(.05)
                assert c.turns['speech'].status == 'complete' and c.turns['speech'].first_media_s is not None
                async with client.get(url+f'/webrtc/groups/{gid}') as r:
                    session = (await r.json())['sessions'][0]
                assert session['connected'] and session['stats']['generated_frames'] == 24
                assert session['last_turn']['status'] == 'complete'
                assert frames['video'] > 10 and frames['audio'] > 10
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await pc.close()
            await runner.cleanup()
    asyncio.run(run())
