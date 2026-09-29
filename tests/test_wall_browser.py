"""Opt-in Chromium test: real ICE/H264/Opus, deterministic CPU frame/TTS fixtures.

SOULX_BROWSER_TEST=1 .venv/bin/python -m pytest -q tests/test_wall_browser.py
Install playwright and its Chromium browser first. This tests the MuseTalk-style
wall, its per-tile player frames and the group API, not SoulX inference speed or
generated visual quality.
"""
import asyncio
import io
import os
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from aiohttp import web
from PIL import Image

from soulx_rtc.server import Service, make_app

pytestmark = pytest.mark.skipif(os.getenv('SOULX_BROWSER_TEST') != '1', reason='Opt-in Chromium smoke test')

PLAYING = """() => { const v = document.querySelector('video');
  return !!v && v.videoWidth > 0 && v.currentTime > 1 && !v.paused; }"""


class WallEngine:
    last_metrics = {'wall_s': .001, 'frames': 24, 'batch': 1}
    def prepare_call(self, path, seed):
        return SimpleNamespace(cursor=0, total_frames=0, seed=seed)
    def append(self, state, audio):
        state.total_frames = state.cursor + int(np.ceil(len(audio)*25/16000/24))*24
    def generate(self, states):
        chunks = []
        for state in states:
            state.cursor += 24
            chunks.append(np.full((24,64,64,3), state.seed % 255, np.uint8))
        return chunks
    def recondition(self, state, displayed):
        state.cursor = state.total_frames = 0


async def frames_playing(page, count, timeout=15):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        frames = page.frames[1:]
        if len(frames) == count:
            try:
                if all(await asyncio.gather(*(f.evaluate(PLAYING) for f in frames))):
                    return frames
            except Exception:  # A frame is still navigating.
                pass
        await asyncio.sleep(.25)
    raise AssertionError(f'{count} player frames did not start playing')


async def muted(page):
    return [await f.evaluate('document.querySelector("video").muted') for f in page.frames[1:]]


def test_browser_wall_lifecycle(monkeypatch, tmp_path):
    playwright = pytest.importorskip('playwright.async_api')
    monkeypatch.setenv('SOULX_API_TOKEN', 'browser-test-token')
    portrait = tmp_path/'portrait.png'
    Image.new('RGB', (96, 96), (140, 100, 80)).save(portrait)
    async def run():
        service = Service(SimpleNamespace(batch=2, max_sessions=4, size=64, width=64, height=64,
            steps=4, fps=25, max_active_calls=3, idle_video=None, idle_policy='hold'))
        service.engine = WallEngine()
        service.calls.avatars.entries['fixture'] = dict(id='fixture', name='Fixture', path=str(portrait), kind='image')
        async def startup(app):
            service.ready = True
            service.task = asyncio.create_task(service.schedule())
        service.startup = startup
        calls = []
        def synthesize(*args):
            calls.append(args)
            output = io.BytesIO()
            sf.write(output, .1*np.sin(np.arange(9600)*.08), 24000, format='WAV')
            return output.getvalue(), {'X-Kokoro-Synthesis-Ms': '10', 'X-Kokoro-Audio-Seconds': '0.4',
                'X-Kokoro-Real-Time-Factor': '0.025', 'X-Kokoro-Cold-Start': '0'}
        service.tts.synthesize = synthesize
        runner = web.AppRunner(make_app(service))
        await runner.setup()
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        url = 'http://127.0.0.1:' + str(site._server.sockets[0].getsockname()[1])
        try:
            async with playwright.async_playwright() as pw:
                browser = await pw.chromium.launch(headless=True, args=['--no-sandbox'])
                page = await browser.new_page(viewport={'width': 1440, 'height': 1100})
                errors = []
                page.on('pageerror', lambda error: errors.append(str(error)))
                await page.goto(url + '/webrtc/wall')
                await page.wait_for_function("!document.querySelector('#tokenRow').hidden")
                # Type the token, then click straight away: the field's blur-time
                # "change" must not swallow that click.
                await page.fill('#token', 'browser-test-token')
                await page.click('#refreshBtn')
                await page.wait_for_function("document.querySelector('#renderFps').value === '25'")
                assert await page.input_value('#resolution') == '64×64'
                await page.select_option('#avatarId', 'fixture')
                await page.fill('#count', '3')
                # One click creates the group, synthesizes once and speaks to all.
                await page.click('#kokoroBtn')
                await page.wait_for_function("document.querySelector('#status').textContent.startsWith('Started 3; failed 0')", timeout=30000)
                await frames_playing(page, 3)
                assert len(calls) == 1 and len(service.calls.items) == 3
                ids = list(service.calls.items)
                assert sorted(c.state.seed for c in service.calls.items.values()) == [50, 51, 52]
                assert await muted(page) == [False, True, True]
                await page.locator('[data-role=audio]').nth(2).click()
                await page.wait_for_function("document.querySelectorAll('.card')[2].classList.contains('audible')")
                await asyncio.sleep(.3)
                assert await muted(page) == [True, True, False]
                await page.wait_for_function("[...document.querySelectorAll('.card-stats')].every(e => e.textContent.includes('H264/opus'))", timeout=10000)

                # A lost HTTP response after server acceptance must not replay speech.
                failed_once = False
                async def lose_response(route):
                    nonlocal failed_once
                    if not failed_once:
                        failed_once = True
                        response = await route.fetch()
                        assert response.status == 200
                        await route.abort()
                    else:
                        await route.continue_()
                await page.route('**/webrtc/groups/*/stream', lose_response)
                await page.click('#kokoroBtn')
                await page.wait_for_function("!document.querySelector('#retryBtn').disabled")
                assert all(len(c.turns) == 2 for c in service.calls.items.values())
                await page.click('#retryBtn')
                await page.wait_for_function("document.querySelector('#status').textContent.includes('Retries accepted')")
                assert all(len(c.turns) == 2 for c in service.calls.items.values())
                await page.unroute('**/webrtc/groups/*/stream')
                await page.locator('[data-role=speak]').first.click()
                await page.wait_for_function("document.querySelector('#status').textContent.startsWith('Started 1; failed 0')")
                assert [len(service.calls.items[i].turns) for i in ids] == [3, 2, 2]
                assert list(service.calls.items) == ids
                assert all(c.negotiations == 1 for c in service.calls.items.values())
                await page.click('#interruptBtn')
                await page.wait_for_function("document.querySelector('#status').textContent.includes('Speech interrupted')")
                assert all(c.epoch == 1 for c in service.calls.items.values())

                await page.click('#debugBtn')
                await page.wait_for_function("document.querySelector('#debugBtn').textContent === 'Stats: docked'")
                await asyncio.sleep(.3)
                assert not any(await asyncio.gather(*(f.evaluate("document.body.classList.contains('debug-off')")
                                                      for f in page.frames[1:])))
                await page.screenshot(path=str(tmp_path / 'wall-desktop.png'), full_page=True)
                await page.set_viewport_size({'width': 390, 'height': 844})
                assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                await page.screenshot(path=str(tmp_path / 'wall-mobile.png'), full_page=True)
                await page.set_viewport_size({'width': 1440, 'height': 1100})
                await page.click('#debugBtn')
                await page.click('#debugBtn')

                # Opening the shared group URL unloads the old frames; their calls end
                # and Reconnect replaces them with fresh calls that keep each seed.
                wall_url = await page.evaluate('currentGroup.wall_url')
                await page.goto(url + wall_url)
                await page.fill('#token', 'browser-test-token')
                await page.dispatch_event('#token', 'change')
                await page.wait_for_function("document.querySelector('#status').textContent.includes('Reconnect peers replaces them')", timeout=15000)
                assert not service.calls.items
                await page.click('#reconnectBtn')
                await page.wait_for_function("document.querySelector('#status').textContent.startsWith('Reconnected: 0 kept, 3 replaced')", timeout=15000)
                await frames_playing(page, 3)
                assert sorted(c.state.seed for c in service.calls.items.values()) == [50, 51, 52]
                assert not set(service.calls.items) & set(ids)

                await page.click('#deleteBtn')
                await page.wait_for_function("document.querySelectorAll('.card').length === 0")
                assert not service.calls.items and not service.pending
                assert await page.evaluate('location.pathname') == '/webrtc/wall'
                assert not errors, errors
                await browser.close()
        finally:
            await runner.cleanup()
    asyncio.run(run())
