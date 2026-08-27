"""Memory-safety bounds: sink backlog backpressure, TTS queue backstop, echo-filter
write-side eviction, compact metric rings, dense token tables, packed lexicons."""

from __future__ import annotations

import asyncio
import time

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import MAX_BACKLOG_MS, AudioSink
from nanobot_channel_voice.backend.base import OutputAudio
from nanobot_channel_voice.echo_reject import SelfEchoFilter
from nanobot_channel_voice.metrics import _MAX_SAMPLES, VoiceMetrics
from nanobot_channel_voice.stt.base import DenseTokenTable, read_token_table
from nanobot_channel_voice.tts.matcha import PackedLexicon


def _run(coro):
    return asyncio.run(coro)


# ---- sink backlog backpressure ---------------------------------------------

def test_wait_backlog_below_blocks_over_cap_and_flush_releases():
    async def _case():
        sink = AudioSink(NullPlayback(), mode="stream")
        # 40 s of 16 kHz PCM queued at the live epoch, worker not started: over cap.
        sink.enqueue(OutputAudio(epoch=sink.epoch, pcm=b"\x00" * (32000 * 40), rate=16000))
        waiter = asyncio.create_task(sink.wait_backlog_below(MAX_BACKLOG_MS))
        await asyncio.sleep(0.05)
        assert not waiter.done()
        await sink.flush()  # zeroes the backlog and wakes the waiter
        await asyncio.wait_for(waiter, timeout=1.0)

    _run(_case())


def test_wait_backlog_below_returns_as_playback_debits():
    async def _case():
        sink = AudioSink(NullPlayback(), mode="blob")
        await sink.start()
        try:
            from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes

            wav = pcm_to_wav_bytes(b"\x00" * (32000 * 20), 16000)  # 20 s each
            sink.enqueue(OutputAudio(epoch=sink.epoch, wav=wav))
            sink.enqueue(OutputAudio(epoch=sink.epoch, wav=wav))
            # Null blob playback is instant: the worker debits and the wait returns.
            await asyncio.wait_for(sink.wait_backlog_below(MAX_BACKLOG_MS), timeout=2.0)
        finally:
            await sink.stop()

    _run(_case())


def test_wait_backlog_below_is_immediate_under_cap():
    async def _case():
        sink = AudioSink(NullPlayback(), mode="stream")
        sink.enqueue(OutputAudio(epoch=sink.epoch, pcm=b"\x00" * 3200, rate=16000))
        await asyncio.wait_for(sink.wait_backlog_below(MAX_BACKLOG_MS), timeout=0.5)

    _run(_case())


# ---- TTS text queue backstop ------------------------------------------------

def test_tts_enqueue_overflow_drops_newest_without_raising():
    from test_local_backend import _build

    h = _build()
    backend = h.backend
    for i in range(backend._tts_queue.maxsize + 5):
        backend._tts_enqueue(f"chunk {i}")
    assert backend._tts_queue.qsize() == backend._tts_queue.maxsize
    # The oldest chunk survived (drop-newest keeps queued speech coherent).
    assert backend._tts_queue.get_nowait()[1] == "chunk 0"


# ---- echo filter write-side eviction ----------------------------------------

def test_note_spoken_evicts_expired_entries_without_a_reader():
    f = SelfEchoFilter(window_secs=0.01)
    f.note_spoken("the first announcement nobody answers")
    time.sleep(0.03)  # first entry's audible window lapses
    f.note_spoken("the second announcement nobody answers")
    assert len(f._spoken) == 1  # a reader-less room must not accumulate forever


# ---- metric sample ring ------------------------------------------------------

def test_metric_ring_caps_and_keeps_recent_samples():
    m = VoiceMetrics()
    for i in range(_MAX_SAMPLES + 500):
        m.observe("stt_ms", float(i))
    summary = m.snapshot()["latency_ms"]["stt_ms"]
    assert summary["n"] == _MAX_SAMPLES
    # The newest sample is present; the overwritten oldest is gone.
    assert summary["max"] == float(_MAX_SAMPLES + 499)
    assert summary["p50"] >= 500.0


# ---- dense token table -------------------------------------------------------

def test_read_token_table_dense_ids_become_a_list(tmp_path):
    p = tmp_path / "tokens.txt"
    p.write_text("<blk> 0\na 1\nb 2\n", encoding="utf-8")
    table = read_token_table(str(p))
    assert isinstance(table, DenseTokenTable)
    assert table[1] == "a"
    assert table.get(2) == "b"
    assert table.get(99) == ""  # same read API as the dict


def test_read_token_table_sparse_ids_stay_a_dict(tmp_path):
    p = tmp_path / "tokens.txt"
    p.write_text("<blk> 0\na 5\n", encoding="utf-8")
    table = read_token_table(str(p))
    assert isinstance(table, dict)
    assert table.get(5) == "a"
    assert table.get(1, "") == ""


# ---- backlog pacing is backend-scoped ---------------------------------------

def test_backend_pacing_declarations():
    from nanobot_channel_voice.backend.local import LocalBackend
    from nanobot_channel_voice.backend.openai_realtime import RealtimeBackend

    assert LocalBackend.pace_output_audio is True
    # The realtime rx loop is the control plane (barge-in, tools): never parked.
    assert RealtimeBackend.pace_output_audio is False


def test_backlog_gate_honors_the_backend_pacing_flag():
    from test_shell import _shell, _StubBackend

    async def _case(paced):
        old = getattr(_StubBackend, "pace_output_audio", None)
        _StubBackend.pace_output_audio = paced
        try:
            shell, _backend, sink = _shell()
        finally:
            if old is None:
                del _StubBackend.pace_output_audio
            else:
                _StubBackend.pace_output_audio = old
        gated = []

        async def spy(cap_ms=None):
            gated.append(cap_ms)

        shell._sink.wait_backlog_below = spy
        await shell._on_event(OutputAudio(epoch=sink.epoch, pcm=b"\x00" * 320, rate=16000))
        assert sink._queue.qsize() == 1  # the item is enqueued either way
        return bool(gated)

    assert _run(_case(True)) is True
    assert _run(_case(False)) is False


# ---- packed lexicon ----------------------------------------------------------

def test_packed_lexicon_rejects_an_entry_over_the_length_field():
    import pytest

    with pytest.raises(ValueError, match="2\\^16"):
        PackedLexicon({"pathological": list(range(1 << 16))})


def test_packed_lexicon_matches_the_dict_it_was_built_from():
    src = {"你": [10, 11], "你好": [10, 11, 12, 13], "hi": [7]}
    lex = PackedLexicon(src)
    assert len(lex) == 3
    assert set(lex) == set(src)
    for word, ids in src.items():
        assert word in lex
        assert lex[word] == ids
    assert "missing" not in lex
