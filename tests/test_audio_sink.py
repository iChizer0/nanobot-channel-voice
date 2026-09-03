"""AudioSink write granularity and the producer-independent backlog ceiling."""

from __future__ import annotations

import asyncio

import pytest

from nanobot_channel_voice.audio.base import PlaybackSink, PlaybackStream
from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import (
    _GAIN_BLOCK_MS,
    MAX_BACKLOG_MS,
    UNPACED_BACKLOG_MS,
    AudioSink,
)
from nanobot_channel_voice.backend.base import OutputAudio

RATE = 16000


class _RecordingStream(PlaybackStream):
    """Records every write; ``on_write`` fires after the first one."""

    def __init__(self, on_write=None):
        self.sizes: list[int] = []
        self._on_write = on_write

    async def write(self, pcm: bytes) -> None:
        self.sizes.append(len(pcm))
        if self._on_write is not None and len(self.sizes) == 1:
            self._on_write()

    async def drain(self) -> None:
        pass

    async def kill(self) -> None:
        pass


class _RecordingSink(PlaybackSink):
    def __init__(self, on_write=None):
        self.stream: _RecordingStream | None = None
        self._on_write = on_write

    async def play_wav(self, wav_bytes: bytes) -> bool:
        return True

    async def abort(self) -> None:
        pass

    async def open_stream(self, rate: int) -> PlaybackStream:
        self.stream = _RecordingStream(self._on_write)
        return self.stream


def _pcm(ms: int) -> bytes:
    return b"\x01\x00" * (RATE * ms // 1000)


def _ms(nbytes: int) -> float:
    return nbytes / (2 * RATE) * 1000.0


def _play(ms: int, *, configure=None, on_write=None) -> list[int]:
    async def scenario():
        ps = _RecordingSink(on_write=(lambda: on_write(sink)) if on_write else None)
        sink = AudioSink(ps, mode="stream")
        if configure is not None:
            configure(sink)
        await sink.start()
        try:
            sink.enqueue(OutputAudio(epoch=sink.epoch, pcm=_pcm(ms), rate=RATE))
            await asyncio.wait_for(sink.wait_idle(), 5.0)
            return ps.stream.sizes
        finally:
            await sink.stop()

    return asyncio.run(scenario())


# ---- write granularity -------------------------------------------------------

def test_plain_playback_writes_lead_sized_blocks():
    # No envelope, no gate, no tap: 20 ms steps buy nothing and cost one executor
    # hop each, so the writer uses a quarter of the pacing lead.
    sizes = _play(480)
    assert _ms(sizes[0]) > _GAIN_BLOCK_MS * 2
    assert sum(sizes) == len(_pcm(480))
    assert len(sizes) <= 480 // _GAIN_BLOCK_MS // 3


def test_a_duck_floor_keeps_the_envelope_granularity():
    sizes = _play(240, configure=lambda s: s.configure_duck(0.25))
    assert {_ms(n) for n in sizes} == {float(_GAIN_BLOCK_MS)}


def test_pause_capable_keeps_the_envelope_granularity():
    sizes = _play(240, configure=lambda s: s.configure_pause(True))
    assert {_ms(n) for n in sizes} == {float(_GAIN_BLOCK_MS)}


def test_a_tap_registered_mid_piece_tightens_the_very_next_block():
    # The granularity is re-chosen per block, so a duck engaged (or a reference tap
    # attached) inside a long piece takes effect at the next boundary, not the next piece.
    class _Tap:
        def push_reference(self, pcm, rate, playout): pass
        def reference_dropped(self): pass

    sizes = _play(480, on_write=lambda sink: sink.set_reference_tap(_Tap()))
    assert _ms(sizes[0]) > _GAIN_BLOCK_MS * 2          # opened plain
    assert _ms(sizes[1]) == float(_GAIN_BLOCK_MS)      # tightened immediately after
    assert sum(sizes) == len(_pcm(480))


# ---- backlog ceiling ---------------------------------------------------------

def test_enqueue_drops_the_oldest_once_the_backlog_passes_the_cap():
    # The cloud backend never parks (pace_output_audio=False), so the queue is bounded
    # here instead: the oldest item goes, the producer is never blocked. The cap is a
    # memory valve far above the paced one: a realtime model streams a long reply ~3x
    # faster than it plays, and that backlog is legitimate.
    sink = AudioSink(NullPlayback(), mode="stream")  # worker not started: nothing drains
    item_ms = int(UNPACED_BACKLOG_MS // 3)
    for i in range(6):
        sink.enqueue(
            OutputAudio(epoch=sink.epoch, pcm=_pcm(item_ms) + b"\x00\x00" * i, rate=RATE)
        )
    assert sink.dropped_ms >= UNPACED_BACKLOG_MS / 3
    assert sink.backlog_ms() <= UNPACED_BACKLOG_MS + item_ms
    # Survivors are the NEWEST run, in order: the tail of the reply, not its head.
    kept = [sink._queue.get_nowait() for _ in range(sink._queue.qsize())]
    assert [len(a.pcm) for a in kept] == sorted(len(a.pcm) for a in kept)
    assert len(kept[0].pcm) > len(_pcm(item_ms))  # item 0 was dropped


def test_the_paced_producer_never_trips_the_ceiling():
    # A paced producer parks at MAX_BACKLOG_MS, an order of magnitude under the drop
    # valve, so the local path can never lose a chunk to it. Worker NOT started: the
    # backlog only grows, so the third half-cap item must block the producer instead.
    async def scenario():
        sink = AudioSink(NullPlayback(), mode="stream")
        half = OutputAudio(epoch=sink.epoch, pcm=_pcm(int(MAX_BACKLOG_MS // 2)), rate=RATE)
        for _ in range(3):  # the wait parks only PAST the cap: 3 halves get through
            await asyncio.wait_for(sink.wait_backlog_below(), 1.0)
            sink.enqueue(half)
        assert sink.backlog_ms() == 1.5 * MAX_BACKLOG_MS
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(sink.wait_backlog_below(), 0.05)
        sink.enqueue(half)  # a producer that raced the wait: still far under the valve
        assert sink.dropped_ms == 0.0 and sink._queue.qsize() == 4

    asyncio.run(scenario())
