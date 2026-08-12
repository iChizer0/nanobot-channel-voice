"""The TTS worker's synthesis backlog gate.

Playback paces the device; the gate paces SYNTHESIS: chunk N+1 is held while
more than _SYNTH_BACKLOG_MS of audio is already accepted-but-unplayed, so a
barge-in cannot flush minutes of pre-rendered audio. Driven with a fake TTS and
a stubbed ``backlog_ms``: no audio device or wall-clock pacing involved.
"""

from __future__ import annotations

import asyncio

import nanobot_channel_voice.backend.local as local_mod
from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.local import LocalBackend, _Turn
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.vad.base import Vad


class _SilentVad(Vad):
    def is_speech(self, frame: bytes) -> bool:
        return False


class _FakeTts:
    output_rate = 16000

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
        self.calls.append(text)
        return b"\x01\x00" * 160  # 10 ms; sub-threshold, so trim_lead_silence returns it whole


def _build() -> tuple[LocalBackend, _FakeTts, AudioSink]:
    cfg = VoiceConfig.model_validate({})
    sink = AudioSink(NullPlayback(), mode="stream")
    tts = _FakeTts()

    async def transcribe(pcm: bytes) -> str:
        return ""

    async def publish(text: str, token: str) -> None:
        pass

    async def interrupt() -> None:
        pass

    backend = LocalBackend(
        cfg, vad=_SilentVad(), tts=tts, sink=sink,
        transcribe=transcribe, publish_text=publish, interrupt=interrupt,
    )

    async def swallow(event) -> None:
        pass

    backend._on_event = swallow
    return backend, tts, sink


async def _until(pred, timeout=2.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while not pred():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition not reached")
        await asyncio.sleep(0.005)


def test_gate_holds_then_releases_when_backlog_drains(monkeypatch):
    monkeypatch.setattr(local_mod, "_BACKLOG_POLL_S", 0.01)
    backend, tts, sink = _build()
    backlog = [0.0]
    sink.backlog_ms = lambda: backlog[0]  # type: ignore[method-assign]

    async def scenario():
        backend._cur_turn = _Turn("t")
        worker = asyncio.create_task(backend._tts_worker())
        try:
            backend._tts_enqueue("first chunk.")
            await _until(lambda: len(tts.calls) == 1)  # first chunk never gated
            await backend._tts_queue.join()

            backlog[0] = 10_000.0
            backend._tts_enqueue("second chunk.")
            await asyncio.sleep(0.08)
            assert tts.calls == ["first chunk."]  # held while backlog is high
            assert backend._metrics.counters.get("tts_backlog_gated") == 1

            backlog[0] = 0.0
            await asyncio.wait_for(backend._tts_queue.join(), 2.0)
            assert tts.calls == ["first chunk.", "second chunk."]
        finally:
            worker.cancel()

    asyncio.run(scenario())


def test_barge_in_epoch_bump_releases_gate_and_drops_chunk(monkeypatch):
    monkeypatch.setattr(local_mod, "_BACKLOG_POLL_S", 0.01)
    backend, tts, sink = _build()
    backlog = [0.0]
    sink.backlog_ms = lambda: backlog[0]  # type: ignore[method-assign]

    async def scenario():
        backend._cur_turn = _Turn("t")
        worker = asyncio.create_task(backend._tts_worker())
        try:
            backend._tts_enqueue("first chunk.")
            await _until(lambda: len(tts.calls) == 1)
            await backend._tts_queue.join()

            backlog[0] = 10_000.0
            backend._tts_enqueue("stale tail.")
            await asyncio.sleep(0.05)
            assert len(tts.calls) == 1  # gated

            await sink.flush()  # barge-in: epoch bump must release the gate...
            await asyncio.wait_for(backend._tts_queue.join(), 2.0)
            assert tts.calls == ["first chunk."]  # ...and the chunk is dropped unsynthesized
        finally:
            worker.cancel()

    asyncio.run(scenario())


def test_gate_never_delays_a_turns_first_chunk(monkeypatch):
    monkeypatch.setattr(local_mod, "_BACKLOG_POLL_S", 0.5)  # a poll would blow the timeout
    backend, tts, sink = _build()
    sink.backlog_ms = lambda: 10_000.0  # type: ignore[method-assign]

    async def scenario():
        backend._cur_turn = _Turn("t")
        worker = asyncio.create_task(backend._tts_worker())
        try:
            backend._tts_enqueue("first chunk.")
            await asyncio.wait_for(backend._tts_queue.join(), 0.4)
            assert tts.calls == ["first chunk."]
        finally:
            worker.cancel()

    asyncio.run(scenario())
