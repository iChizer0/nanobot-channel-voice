"""No-op audio backend for headless hosts: capture yields paced silence so the loop
runs without ever producing an utterance, playback discards. Keeps the channel alive
instead of crashing the gateway.
"""

from __future__ import annotations

import asyncio

from nanobot_channel_voice.audio.base import (
    CaptureSource,
    PlaybackSink,
    PlaybackStream,
    frame_bytes,
)


class NullCapture(CaptureSource):
    def __init__(self, sample_rate: int, frame_ms: int):
        self._frame = b"\x00" * frame_bytes(sample_rate, frame_ms)
        self._frame_s = frame_ms / 1000.0
        self._running = False

    async def start(self) -> None:
        self._running = True

    async def read_frame(self) -> bytes:
        if not self._running:
            return b""
        await asyncio.sleep(self._frame_s)
        return self._frame

    async def stop(self) -> None:
        self._running = False


class NullPlayback(PlaybackSink):
    async def play_wav(self, wav_bytes: bytes) -> bool:
        return True

    async def abort(self) -> None:
        pass

    async def open_stream(self, rate: int) -> PlaybackStream:
        return _NullPlaybackStream()


class _NullPlaybackStream(PlaybackStream):
    async def write(self, pcm: bytes) -> None:
        await asyncio.sleep(0)  # discard, but yield so the loop still interleaves

    async def drain(self) -> None:
        pass

    async def kill(self) -> None:
        pass
