"""Optional live GPU/Kokoro browser probe; requires a running localhost server.

Run from the checkout with .venv/bin/python benchmarks/wall/check_live.py.
Writes fresh evidence under /tmp by default; SOULX_WALL_OUTPUT overrides it.
"""
import asyncio, json, os, time
from pathlib import Path
import aiohttp
from playwright.async_api import async_playwright

async def main():
    target='http://127.0.0.1:8765'
    output=Path(os.environ.get('SOULX_WALL_OUTPUT', '/tmp/soulx-wall-live-' + time.strftime('%Y%m%d-%H%M%S')))
    output.mkdir(parents=True, exist_ok=False)
    async with async_playwright() as pw:
        browser=await pw.chromium.launch(headless=True,args=['--no-sandbox'])
        page=await browser.new_page(viewport={'width':1440,'height':1300})
        errors=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        try:
            await page.goto(target+'/webrtc/wall')
            await page.fill('#count','2')
            await page.fill('#text','Hello! We are testing two live connections with Kokoro speech.')
            await page.click('#speak')
            await page.wait_for_function("document.querySelector('#status').textContent.includes('Speech accepted by 2/2')",timeout=180000)
            print('Both peers accepted real Kokoro audio',flush=True)
            await page.wait_for_function("[...document.querySelectorAll('.tile pre')].length === 2 && [...document.querySelectorAll('.tile pre')].every(e => JSON.parse(e.textContent || '{}').completed_turns >= 1)",timeout=180000)
            await page.locator('[data-action=listen]').first.click()
            await page.screenshot(path=str(output/'live-two-peers.png'),full_page=True)
            first=await page.locator('.tile pre').evaluate_all('(nodes) => nodes.map(n=>JSON.parse(n.textContent))')
            await page.fill('#text','This second turn keeps the same connection alive.')
            await page.locator('[data-action=speak]').nth(1).click()
            await page.wait_for_function("document.querySelector('#status').textContent.includes('Speech accepted by 1/1')",timeout=120000)
            await page.wait_for_function("JSON.parse(document.querySelectorAll('.tile pre')[1].textContent || '{}').completed_turns >= 2",timeout=180000)
            second=await page.locator('.tile pre').evaluate_all('(nodes) => nodes.map(n=>JSON.parse(n.textContent))')
            stats=await page.locator('.stats').all_text_contents()
            assert [x['id'] for x in first] == [x['id'] for x in second]
            assert all(x['negotiations']==1 for x in second)
            assert all('H264' in s and 'opus' in s for s in stats), stats
            assert not errors,errors
            async with aiohttp.ClientSession() as client:
                health=await (await client.get(target+'/health')).json()
            report={'checked_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'health':health,
                    'tts':await page.locator('#ttsMetric').inner_text(),'first_turn':first,'after_second_turn':second,'browser_stats':stats,'page_errors':errors}
            await page.screenshot(path=str(output/'live-second-turn.png'),full_page=True)
            await page.set_viewport_size({'width':390,'height':844})
            assert await page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            await page.screenshot(path=str(output/'live-mobile.png'),full_page=True)
            await page.click('#stop')
            await page.wait_for_function("document.querySelectorAll('.tile').length === 0")
            async with aiohttp.ClientSession() as client:
                report['after_cleanup']=await (await client.get(target+'/health')).json()
            assert report['after_cleanup']['calls']==0 and report['after_cleanup']['gpu_states']==0
            (output/'live-check.json').write_text(json.dumps(report,indent=2))
            print(json.dumps({'passed':True,'calls':len(second),'completed_turns':[x['completed_turns'] for x in second], 'browser_stats':stats,'cleanup_calls':report['after_cleanup']['calls']}),flush=True)
        finally:
            try:
                await page.click('#stop',timeout=3000)
            except Exception:
                pass
            await browser.close()

asyncio.run(main())
