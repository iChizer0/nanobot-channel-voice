"""transcribe_chunked: window-bounded piecewise decode for batch STT adapters."""

from __future__ import annotations

import asyncio

import numpy as np

from nanobot_channel_voice.stt.base import SttAdapter, transcribe_chunked

RATE = 16000


class _Recorder(SttAdapter):
    """Records every piece; replies from a script (then 'x')."""

    def __init__(self, texts: list[str] | None = None, limit_ms: int | None = None):
        self.max_decode_ms = limit_ms
        self.calls: list[bytes] = []
        self._texts = list(texts or [])

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        self.calls.append(pcm)
        return self._texts.pop(0) if self._texts else "x"


def run(coro):
    return asyncio.run(coro)


def pcm(ms: int, amp: int = 8000, silences: tuple[tuple[int, int], ...] = ()) -> bytes:
    """Constant-amplitude S16_LE mono with silent [start, end) ms bands."""
    samples = np.full(RATE * ms // 1000, amp, dtype=np.int16)
    for lo, hi in silences:
        samples[RATE * lo // 1000 : RATE * hi // 1000] = 0
    return samples.tobytes()


def test_unbounded_adapter_passes_through():
    adapter = _Recorder(limit_ms=None)
    audio = pcm(5000)
    assert run(transcribe_chunked(adapter, audio, RATE)) == "x"
    assert adapter.calls == [audio]


def test_within_window_passes_through():
    adapter = _Recorder(limit_ms=1000)
    audio = pcm(1000)  # exactly at the limit: no chunking
    run(transcribe_chunked(adapter, audio, RATE))
    assert adapter.calls == [audio]


def test_cuts_at_the_silent_band():
    adapter = _Recorder(texts=["a", "b"], limit_ms=1000)
    audio = pcm(1600, silences=((600, 800),))
    text = run(transcribe_chunked(adapter, audio, RATE))
    assert text == "a b"
    # The cut lands at the END of the silence (last quietest frame), not the window.
    assert [len(c) for c in adapter.calls] == [2 * RATE * 800 // 1000, 2 * RATE * 800 // 1000]
    assert b"".join(adapter.calls) == audio  # lossless split


def test_constant_energy_hard_cuts_at_the_window():
    adapter = _Recorder(limit_ms=100)
    audio = pcm(400)
    run(transcribe_chunked(adapter, audio, RATE))
    # No quiet gap anywhere: every piece fills the window exactly (never half of it).
    assert [len(c) for c in adapter.calls] == [2 * RATE * 100 // 1000] * 4
    assert b"".join(adapter.calls) == audio
    assert all(len(c) % 2 == 0 for c in adapter.calls)  # sample-aligned


def test_empty_piece_transcripts_are_skipped():
    adapter = _Recorder(texts=["a", "  ", "b"], limit_ms=100)
    text = run(transcribe_chunked(adapter, pcm(300), RATE))
    assert text == "a b"


def test_cjk_seams_join_bare_latin_seams_take_a_space():
    adapter = _Recorder(texts=["你好，", "世界。"], limit_ms=100)
    assert run(transcribe_chunked(adapter, pcm(200), RATE)) == "你好，世界。"
    adapter = _Recorder(texts=["hello", "你好", "world"], limit_ms=100)
    assert run(transcribe_chunked(adapter, pcm(300), RATE)) == "hello你好world"


def test_other_rates_cut_losslessly():
    rate = 8000
    adapter = _Recorder(limit_ms=100)
    audio = np.full(rate * 350 // 1000, 5000, dtype=np.int16).tobytes()
    run(transcribe_chunked(adapter, audio, rate))
    assert b"".join(adapter.calls) == audio
    assert all(len(c) <= 2 * rate * 100 // 1000 for c in adapter.calls)


def test_torn_trailing_byte_is_dropped_not_fatal():
    adapter = _Recorder(limit_ms=100)
    audio = pcm(250) + b"\x7f"  # e.g. an interrupted ffmpeg pipe mid-sample
    assert run(transcribe_chunked(adapter, audio, RATE)) == "x x x"
    assert b"".join(adapter.calls) == audio[:-1]


def test_adapter_window_declarations():
    from nanobot_channel_voice.stt.sensevoice import SenseVoiceOnDeviceStt
    from nanobot_channel_voice.stt.zipformer import ZipformerOnDeviceStt

    assert SttAdapter.max_decode_ms is None
    assert SenseVoiceOnDeviceStt.max_decode_ms == 30_000
    # A transducer is streamed internally on the batch path too (frames pop as they are
    # encoded), so no window: a cut would only reset the encoder caches mid-word.
    assert ZipformerOnDeviceStt.max_decode_ms is None
