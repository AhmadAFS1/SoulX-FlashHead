"""Lazy CPU Kokoro synthesis for the browser wall, adapted from MuseTalk.

One cached model, one in-flight synthesis, and at most 30 seconds of output.
The HTTP thread never runs model loading or inference.
"""
import asyncio
import importlib.util
import io
import logging
import math
import time

import numpy as np
import soundfile as sf
from aiohttp import web

LOG = logging.getLogger(__name__)
SAMPLE_RATE = 24000
VOICES = ("af_heart", "af_bella", "af_sarah", "am_adam", "am_michael",
          "bf_emma", "bf_isabella", "bm_george", "bm_lewis")


class KokoroService:
    def __init__(self):
        self.pipelines = {}
        self.loaded = ()
        self.task = None

    def routes(self):
        return [web.get("/webrtc/tts/kokoro/status", self.status),
                web.post("/webrtc/tts/kokoro", self.create)]

    async def status(self, request):
        return web.json_response(dict(
            available=importlib.util.find_spec("kokoro") is not None,
            busy=self.task is not None and not self.task.done(),
            model="hexgrad/Kokoro-82M", device="cpu", sample_rate=SAMPLE_RATE,
            voices=VOICES, max_audio_seconds=30, max_text_characters=2000,
            loaded_languages=self.loaded), headers={"Cache-Control": "no-store"})

    @staticmethod
    def validate(params):
        if not isinstance(params, dict):
            raise ValueError("Expected a JSON object")
        text = params.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError("text must contain 1–2000 characters")
        voice = params.get("voice", "af_heart")
        if voice not in VOICES:
            raise ValueError("Choose a voice listed by /webrtc/tts/kokoro/status")
        language = params.get("language_code", voice[0])
        if language != voice[0]:
            raise ValueError("language_code must match the voice: a (US) or b (UK)")
        speed = float(params.get("speed", 1))
        if not math.isfinite(speed) or not .5 <= speed <= 2:
            raise ValueError("speed must be between 0.5 and 2.0")
        return text.strip(), voice, language, speed

    def synthesize(self, text, voice, language, speed):
        try:
            from kokoro import KPipeline
        except ImportError as exc:
            raise RuntimeError("Kokoro is not installed. Run .venv/bin/python -m pip install -r requirements-tts.txt") from exc
        started = time.monotonic()
        cold = language not in self.pipelines
        if cold:
            model = next(iter(self.pipelines.values())).model if self.pipelines else True
            self.pipelines[language] = KPipeline(lang_code=language, device="cpu", model=model,
                                                  repo_id="hexgrad/Kokoro-82M")
            self.loaded = tuple(self.pipelines)
        chunks, count = [], 0
        for result in self.pipelines[language](text, voice=voice, speed=speed):
            audio = result.audio
            if audio is None:
                continue
            if hasattr(audio, "detach"):
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            count += len(audio)
            if count > 30 * SAMPLE_RATE:
                raise ValueError("Speech exceeds 30 seconds. Shorten the text or increase speed; no audio was sent.")
            if not np.isfinite(audio).all():
                raise RuntimeError("Kokoro produced non-finite audio")
            chunks.append(audio)
        if not count:
            raise ValueError("Kokoro produced no speech. Try text containing spoken words.")
        output = io.BytesIO()
        sf.write(output, np.concatenate(chunks), SAMPLE_RATE, format="WAV", subtype="PCM_16")
        elapsed, duration = time.monotonic() - started, count / SAMPLE_RATE
        return output.getvalue(), {
            "X-Kokoro-Synthesis-Ms": f"{elapsed * 1000:.1f}",
            "X-Kokoro-Audio-Seconds": f"{duration:.3f}",
            "X-Kokoro-Real-Time-Factor": f"{elapsed / duration:.4f}",
            "X-Kokoro-Cold-Start": "1" if cold else "0",
            "X-Kokoro-Device": "cpu", "Cache-Control": "no-store",
            "Content-Disposition": 'inline; filename="kokoro_test.wav"',
        }

    async def create(self, request):
        try:
            params = self.validate(await request.json())
        except (ValueError, TypeError) as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        if self.task is not None and not self.task.done():
            raise web.HTTPTooManyRequests(text="Kokoro is busy; retry when synthesis finishes",
                                          headers={"Retry-After": "2"})
        self.task = asyncio.create_task(asyncio.to_thread(self.synthesize, *params))
        # Retain admission if the browser disconnects during inference.
        self.task.add_done_callback(lambda task: task.exception() if not task.cancelled() else None)
        try:
            data, headers = await asyncio.shield(self.task)
        except ValueError as exc:
            raise web.HTTPBadRequest(text=str(exc)) from exc
        except Exception as exc:
            LOG.exception("Kokoro synthesis failed")
            raise web.HTTPServiceUnavailable(text=f"Kokoro synthesis failed: {exc}") from exc
        return web.Response(body=data, content_type="audio/wav", headers=headers)

    async def cleanup(self):
        if self.task is not None:
            await asyncio.gather(self.task, return_exceptions=True)
