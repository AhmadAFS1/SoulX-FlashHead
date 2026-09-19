"""Generate a talking-avatar MP4 (with audio) from a WAV file — against your LOCAL container.

Usage:
    python main.py [INPUT.wav] [OUTPUT.mp4]

Reads mono 16-bit PCM speech audio, drives the avatar running in your local
avatar-service container, and writes a lip-synced MP4 with the original audio muxed in.

This is the Ojin python-sdk example `01-speech-to-video-mp4-generator` with ONE change:
it points at your container (`OJIN_WS_URL`) instead of Ojin's hosted service. No Ojin
account and no network access are required — everything runs on your machine.
"""

import asyncio
import os
import pathlib
import sys
import wave

from mp4_writer import Mp4Writer

from ojin.stv import OjinSTVClient, QueueOutput, STVEvent, STVVideoFrame

FPS = 25

# The container's proxy. Inside docker compose this is the `avatar` service; on the host
# it's localhost. The hosted-service default (wss://models.ojin.ai/realtime) is NOT used.
WS_URL = os.environ.get("OJIN_WS_URL", "ws://localhost:8000/realtime")
# The avatar to drive = a filename in the container's /avatars folder (see ../../avatars).
CONFIG_ID = os.environ.get("OJIN_CONFIG_ID", "sample.png")
# The container runs in local mode and accepts any key; this is a placeholder, not a secret.
API_KEY = os.environ.get("OJIN_API_KEY", "local")


def read_mono_wav(path: pathlib.Path) -> tuple[bytes, int]:
    """Read a mono 16-bit PCM WAV as (pcm_bytes, sample_rate), or exit with a hint."""
    if not path.exists():
        sys.exit(
            f"\n  No audio file at '{path}'.\n"
            "  Pass one:  python main.py myvoice.wav\n"
            "  Need a WAV? Convert anything with ffmpeg:\n"
            f"    ffmpeg -i input.mp3 -ac 1 -ar 16000 {path}\n"
        )
    with wave.open(str(path), "rb") as wav:
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            sys.exit(
                f"\n  '{path}' must be MONO 16-bit PCM. Convert it:\n"
                f"    ffmpeg -i '{path}' -ac 1 -ar 16000 mono.wav\n"
            )
        return wav.readframes(wav.getnframes()), wav.getframerate()


async def open_session(
    startup_timeout: float,
) -> tuple[OjinSTVClient, asyncio.Event, dict]:
    """Open a ready session, retrying while the container is still warming up.

    A cold container reports "no inference capacity" for a short window even after it starts
    accepting connections (the proxy re-checks the model backend on an interval, so there is a
    gap where it still believes the backend is down). That is transient and the server asks us
    to retry, so treat it as "not ready yet" rather than a failure.
    """
    deadline = asyncio.get_event_loop().time() + startup_timeout
    attempt = 0
    while True:
        attempt += 1
        ready, done, error = asyncio.Event(), asyncio.Event(), {}

        def on_error(message: str = "", **_: object) -> None:
            """Capture a fatal error and unblock the waiters."""
            error["message"] = message
            ready.set()
            done.set()

        # This is an offline render, so use an effectively unbounded video buffer:
        # we never want frames dropped (the default QueueOutput caps at 60).
        client = OjinSTVClient(
            api_key=API_KEY,
            config_id=CONFIG_ID,
            ws_url=WS_URL,
            output=QueueOutput(max_video=10**9),
        )
        client.add_listener(STVEvent.SESSION_READY, lambda **_: ready.set())
        client.add_listener(STVEvent.BOT_STOPPED_SPEAKING, lambda **_: done.set())
        client.add_listener(STVEvent.ERROR, on_error)

        try:
            await client.connect_with_retry()
            await client.start()  # may log "already connected" — connect_with_retry did it
            await asyncio.wait_for(ready.wait(), timeout=60)
        except asyncio.TimeoutError:
            error.setdefault("message", "timed out waiting for the session to become ready")
        except Exception as exc:  # noqa: BLE001 — surfaced below with a readable hint
            error.setdefault("message", str(exc))

        if not error:
            return client, done, error

        message = str(error.get("message", ""))
        await client.close()

        # Everything about a starting container looks like an error for a while: the proxy
        # isn't listening yet (connection refused), then it is but reports no capacity until
        # the model has loaded. Both are transient, so keep waiting. Only a bad avatar name or
        # a rejected key is worth failing fast on — retrying those never helps.
        fatal = any(
            marker in message.upper()
            for marker in ("CONFIG_NOT_FOUND", "AUTH_FAILED", "MODEL_NOT_FOUND", "UNAUTHORIZED")
        )
        if not fatal and asyncio.get_event_loop().time() < deadline:
            if attempt == 1:
                print("  waiting for the avatar service to be ready...", flush=True)
            await asyncio.sleep(5)
            continue

        if fatal:
            sys.exit(
                f"\n  The service rejected the session: {message}\n"
                f"  Check that '{CONFIG_ID}' exists in the avatars/ folder mounted into the\n"
                "  container (docker compose exec avatar ls /avatars).\n"
            )
        sys.exit(
            "\n  The avatar service never became ready.\n"
            f"  Last error from {WS_URL}: {message}\n"
            "  Check it is running and finished loading:  docker compose logs -f avatar\n"
        )


async def render(pcm: bytes, rate: int, wav_path: pathlib.Path, out: pathlib.Path) -> None:
    """Drive the avatar with `pcm` and write its frames + `wav_path` audio to `out`."""
    seconds = len(pcm) / (rate * 2)  # mono 16-bit, for the finish timeout
    writer: Mp4Writer | None = None

    # Generous: covers a full cold start (model load ~2 min) plus the capacity-ready gap.
    client, done, error = await open_session(startup_timeout=420)
    try:
        await client.say(pcm, sample_rate=rate, num_channels=1)

        async def consume() -> None:
            """Feed each returned RGB frame to the MP4 writer (built on the first one)."""
            nonlocal writer
            async for frame in client.output_stream():
                if (
                    isinstance(frame, STVVideoFrame)
                    and frame.rgb is not None
                    and frame.frame_type != 0  # speech frames only
                ):
                    if writer is None:
                        # Size the MP4 to whatever the server sends (first frame wins).
                        # With the 2x upscaler active that is 1024x1024.
                        print(f"  frame size: {frame.width}x{frame.height}")
                        writer = Mp4Writer(
                            out, frame.width, frame.height, fps=FPS, audio_wav=wav_path
                        )
                    writer.write(frame.rgb)
                    # In a terminal, redraw one line. Under `docker compose up` there is no TTY
                    # and \r doesn't overwrite, so log periodically instead of once per frame.
                    if sys.stdout.isatty():
                        print(f"\r  rendering... {writer.frames} frames", end="", flush=True)
                    elif writer.frames % 25 == 0:
                        print(f"  rendering... {writer.frames} frames", flush=True)

        consumer = asyncio.create_task(consume())
        try:
            # The avatar plays in real time, so it finishes ~`seconds` after we send
            # it; allow generous margin, then finalize whatever we captured.
            await asyncio.wait_for(done.wait(), timeout=seconds + 60)
        except asyncio.TimeoutError:
            print(
                "\n  No end-of-speech from the server in time — finalizing what was\n"
                "  rendered (the audio may be too short or too quiet to drive speech)."
            )
        await asyncio.sleep(0.5)  # let the final frames drain
        consumer.cancel()
    finally:
        await client.close()
        if writer is not None:
            writer.close()

    if error:
        sys.exit(f"\n  Stopped: {error['message']}\n")
    if writer is None:
        sys.exit(
            "\n  No video frames were produced — the audio may be too short or too\n"
            "  quiet to drive speech. Try a longer, louder clip.\n"
        )
    secs = writer.frames / FPS
    print(f"\n  Done -> {out}  ({writer.frames} frames, {secs:.1f}s, audio included)\n")


def main() -> None:
    """Render the MP4 against the local container."""
    here = pathlib.Path(__file__).parent
    in_path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else here / "sample_audio_16k.wav")
    out_path = pathlib.Path(sys.argv[2] if len(sys.argv) > 2 else os.environ.get("OUT_MP4", "out/avatar.mp4"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pcm, rate = read_mono_wav(in_path)

    seconds = len(pcm) / (rate * 2)
    print(f"  Driving avatar '{CONFIG_ID}' at {WS_URL} with {seconds:.1f}s of audio...")
    asyncio.run(render(pcm, rate, in_path, out_path))


if __name__ == "__main__":
    main()
