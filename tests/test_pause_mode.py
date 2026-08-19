"""Pause-then-confirm (bargeIn.mode="pause"): the sink's pause gate and the
backend's pause-instead-of-duck branch."""

from __future__ import annotations

import asyncio

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import OutputAudio, VoiceState
from nanobot_channel_voice.backend.local import LocalBackend
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.vad.base import Vad


class _SilentVad(Vad):
    def __init__(self) -> None:
        self.floor_scales: list[float] = []

    def is_speech(self, frame: bytes) -> bool:
        return False

    def scale_floor(self, factor: float) -> None:
        self.floor_scales.append(factor)


def _pcm_item(sink: AudioSink, ms: int = 200) -> OutputAudio:
    return OutputAudio(pcm=b"\x01\x00" * (16 * ms), rate=16000, epoch=sink.epoch)


# ---- sink pause gate --------------------------------------------------------

def test_pause_stalls_the_writer_and_resume_continues():
    async def scenario():
        sink = AudioSink(NullPlayback(), mode="stream")
        sink.configure_pause(True)
        await sink.start()
        try:
            sink.pause(True)
            assert sink.paused
            sink.enqueue(_pcm_item(sink))
            await asyncio.sleep(0.05)
            assert sink.backlog_ms() > 0  # nothing consumed while paused
            assert not sink.starved()     # paused is deliberately silent, not starved
            sink.pause(False)
            await asyncio.wait_for(sink.wait_idle(), 3.0)  # writer resumed and finished
        finally:
            await sink.stop()

    asyncio.run(scenario())


def test_flush_releases_a_paused_writer():
    async def scenario():
        sink = AudioSink(NullPlayback(), mode="stream")
        sink.configure_pause(True)
        await sink.start()
        try:
            sink.enqueue(_pcm_item(sink, ms=400))
            await asyncio.sleep(0.02)  # writer is mid-chunk
            sink.pause(True)
            await sink.flush()  # must not wedge on the stalled writer...
            assert not sink.paused  # ...and ends the candidate: gate re-opened
            await asyncio.wait_for(sink.wait_idle(), 2.0)
        finally:
            await sink.stop()

    asyncio.run(scenario())


def test_drain_stream_waits_for_the_verdict_instead_of_unpausing():
    """A paused drain must NOT force the gate open (that would play the turn's
    final chunk at full level into the open mic mid-confirm); it waits until the
    candidate resolves: here, a release."""
    async def scenario():
        sink = AudioSink(NullPlayback(), mode="stream")
        sink.configure_pause(True)
        await sink.start()
        try:
            sink.pause(True)
            sink.enqueue(_pcm_item(sink))
            drain = asyncio.create_task(sink.drain_stream())
            await asyncio.sleep(0.1)
            assert not drain.done()  # still paused: the drain is waiting
            assert sink.paused       # ...and did NOT force the gate open
            sink.pause(False)        # verdict: false alarm, release
            await asyncio.wait_for(drain, 3.0)
        finally:
            await sink.stop()

    asyncio.run(scenario())


def test_flush_resolves_a_paused_drain():
    """The other verdict: a kill's flush opens the gate with a dead epoch."""
    async def scenario():
        sink = AudioSink(NullPlayback(), mode="stream")
        sink.configure_pause(True)
        await sink.start()
        try:
            sink.pause(True)
            sink.enqueue(_pcm_item(sink))
            drain = asyncio.create_task(sink.drain_stream())
            await asyncio.sleep(0.05)
            assert not drain.done()
            await sink.flush()
            await asyncio.wait_for(drain, 3.0)
        finally:
            await sink.stop()

    asyncio.run(scenario())


def test_pause_freezes_the_playout_clocks():
    """played_ms/backlog_ms must freeze at the pause edge: a wall-clock model
    would drain the echo filter's hold on audio nobody heard."""
    async def scenario():
        sink = AudioSink(NullPlayback(), mode="stream")
        sink.configure_pause(True)
        await sink.start()
        try:
            sink.enqueue(_pcm_item(sink, ms=400))
            await asyncio.sleep(0.05)  # some blocks written, stream open
            sink.pause(True)
            await asyncio.sleep(0.02)  # let the writer reach the gate
            played_at_pause = sink.played_ms()
            backlog_at_pause = sink.backlog_ms()
            await asyncio.sleep(0.15)
            assert sink.played_ms() == played_at_pause    # frozen, not draining
            assert sink.backlog_ms() == backlog_at_pause
            sink.pause(False)
            await asyncio.wait_for(sink.wait_idle(), 3.0)
        finally:
            await sink.stop()

    asyncio.run(scenario())


def test_pause_release_splices_the_span_out_of_the_clock():
    """A released pause must not read as elapsed playout: without the clock splice
    starved_ms/played_ms count the silence until the next write re-anchors."""
    async def scenario():
        sink = AudioSink(NullPlayback(), mode="stream")
        sink.configure_pause(True)
        await sink.start()
        try:
            sink.enqueue(_pcm_item(sink, ms=400))
            await asyncio.sleep(0.05)
            sink.pause(True)
            await asyncio.sleep(0.02)  # writer at the gate
            played_at_pause = sink.played_ms()
            await asyncio.sleep(0.2)
            sink.pause(False)
            assert sink.starved_ms() == 0.0  # the pause was deliberate silence
            assert sink.played_ms() <= played_at_pause + 20
            await asyncio.wait_for(sink.wait_idle(), 3.0)
        finally:
            await sink.stop()

    asyncio.run(scenario())


def test_pause_is_a_noop_in_blob_mode():
    sink = AudioSink(NullPlayback(), mode="blob")
    sink.pause(True)
    assert not sink.paused


# ---- backend mode branch ----------------------------------------------------

def _backend(mode: str) -> tuple[LocalBackend, AudioSink, _SilentVad]:
    cfg = VoiceConfig.model_validate({
        "aec": "soft", "duckDb": -12.0, "bargeIn": {"mode": mode},
    })
    sink = AudioSink(NullPlayback(), mode="stream")
    vad = _SilentVad()

    async def transcribe(pcm: bytes) -> str:
        return ""

    async def publish(text: str, token: str, notes: tuple[str, ...] = ()) -> None:
        pass

    async def interrupt() -> None:
        pass

    backend = LocalBackend(
        cfg, vad=vad, tts=None, sink=sink,
        transcribe=transcribe, publish_text=publish, interrupt=interrupt,
    )
    return backend, sink, vad


def test_pause_mode_engages_pause_not_gain_or_floor():
    backend, sink, vad = _backend("pause")
    backend._turn = VoiceState.SPEAKING
    assert backend._duck_armed()  # armed by pause capability, duckDb irrelevant
    backend._engage_duck(suspect=False)
    assert sink.paused
    assert sink._gain_target == 1.0   # no envelope duck
    assert vad.floor_scales == []     # floor untouched: the leak stops entirely
    backend._release_duck("echo")
    assert not sink.paused
    assert backend._metrics.counters.get("barge_in_false_resume.echo") == 1


def test_duck_mode_still_ducks():
    backend, sink, vad = _backend("duck")
    backend._turn = VoiceState.SPEAKING
    backend._engage_duck(suspect=False)
    assert not sink.paused
    assert sink._gain_target < 1.0
    assert vad.floor_scales  # floor stepped with the leak
    backend._clear_duck()
    assert sink._gain_target == 1.0


def test_pause_mode_arms_even_with_duck_db_off():
    cfg = VoiceConfig.model_validate({
        "aec": "soft", "duckDb": 0.0, "bargeIn": {"mode": "pause"},
    })
    sink = AudioSink(NullPlayback(), mode="stream")

    async def _n(*a, **k):
        return ""

    backend = LocalBackend(
        cfg, vad=_SilentVad(), tts=None, sink=sink,
        transcribe=_n, publish_text=_n, interrupt=_n,
    )
    backend._turn = VoiceState.SPEAKING
    assert backend._duck_armed()
