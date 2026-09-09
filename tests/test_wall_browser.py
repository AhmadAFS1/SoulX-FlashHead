"""Opt-in Chromium test: real ICE/H264/Opus, deterministic CPU frame/TTS fixtures.

SOULX_BROWSER_TEST=1 .venv/bin/python -m pytest -q tests/test_wall_browser.py
Install playwright and its Chromium browser first. This tests the UI/transport,
not SoulX inference speed or generated visual quality.
"""
import asyncio
import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from aiohttp import web

from soulx_rtc.server import Service, make_app

pytestmark = pytest.mark.skipif(os.getenv('SOULX_BROWSER_TEST') != '1', reason='Opt-in Chromium smoke test')


class WallEngine:
    last_metrics = {'wall_s': .001}
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


def test_browser_wall_lifecycle(monkeypatch, tmp_path):
    playwright = pytest.importorskip('playwright.async_api')
    monkeypatch.setenv('SOULX_API_TOKEN', 'browser-test-token')
    async def run():
        service = Service(SimpleNamespace(batch=2, max_sessions=3, size=64, width=64, height=64,
            steps=4, fps=25, max_active_calls=3, idle_video=None, idle_policy='hold'))
        service.engine = WallEngine()
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
                await page.fill('#token', 'browser-test-token')
                await page.fill('#count', '3')
                await page.click('#refresh')
                await page.wait_for_function("document.querySelector('#profile').textContent.includes('GPU ready')")
                await page.click('#speak')
                await page.wait_for_function("document.querySelector('#status').textContent.includes('Speech accepted by 3/3')", timeout=30000)
                await page.wait_for_function("[...document.querySelectorAll('video')].length === 3 && [...document.querySelectorAll('video')].every(v => v.videoWidth > 0 && v.currentTime > 1)", timeout=15000)
                assert len(calls) == 1 and len(service.calls.items) == 3
                ids = list(service.calls.items)
                assert len({c.state.seed for c in service.calls.items.values()}) == 3
                await page.wait_for_function("[...document.querySelectorAll('.stats')].every(e => e.textContent.includes('H264') && e.textContent.includes('opus'))")
                assert await page.locator('video').evaluate_all('(videos) => videos.every(v => v.muted)')
                await page.locator('[data-action=listen]').nth(1).click()
                assert await page.locator('video').evaluate_all('(videos) => videos.map(v => v.muted)') == [True, False, True]
                await page.locator('[data-action=listen]').nth(2).click()
                assert await page.locator('video').evaluate_all('(videos) => videos.map(v => v.muted)') == [True, True, False]
                # A lost HTTP response after server acceptance must not replay speech.
                failed_once = False
                async def lose_response(route):
                    nonlocal failed_once
                    if not failed_once:
                        failed_once = True
                        response = await route.fetch()
                        assert response.status == 202
                        await route.abort()
                    else:
                        await route.continue_()
                await page.route('**/calls/*/turns', lose_response)
                await page.click('#speak')
                await page.wait_for_function("!document.querySelector('#retry').disabled")
                assert all(len(c.turns) == 2 for c in service.calls.items.values())
                await page.click('#retry')
                await page.wait_for_function("document.querySelector('#status').textContent.includes('Retries accepted')")
                assert all(len(c.turns) == 2 for c in service.calls.items.values())
                await page.unroute('**/calls/*/turns')
                await page.locator('[data-action=speak]').first.click()
                await page.wait_for_function("document.querySelector('#status').textContent.includes('Speech accepted by 1/1')")
                assert [len(c.turns) for c in service.calls.items.values()] == [3, 2, 2]
                assert list(service.calls.items) == ids
                assert all(c.negotiations == 1 for c in service.calls.items.values())
                await page.click('#interrupt')
                await page.wait_for_function("document.querySelector('#status').textContent.includes('Speech interrupted')")
                assert all(c.epoch == 1 for c in service.calls.items.values())
                await page.screenshot(path=str(tmp_path / 'wall-desktop.png'), full_page=True)
                await page.set_viewport_size({'width': 390, 'height': 844})
                assert await page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
                await page.screenshot(path=str(tmp_path / 'wall-mobile.png'), full_page=True)
                await page.click('#stop')
                await page.wait_for_function("document.querySelectorAll('.tile').length === 0")
                assert not service.calls.items
                # Stop while create responses are still pending: no server orphan.
                pending = asyncio.Event()
                gate = asyncio.Event()
                async def delayed_create(route):
                    response = await route.fetch()
                    pending.set()
                    await gate.wait()
                    await route.fulfill(response=response)
                await page.route('**/calls', delayed_create)
                await page.click('#create')
                await asyncio.wait_for(pending.wait(), 10)
                await page.click('#stop')
                gate.set()
                await page.wait_for_function("!document.querySelector('#create').disabled", timeout=15000)
                assert not service.calls.items and not service.pending
                assert not errors, errors
                await browser.close()
        finally:
            await runner.cleanup()
    asyncio.run(run())
