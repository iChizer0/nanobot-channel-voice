"""The TTS worker's deadline (JIT) synthesis scheduler.

Playback paces the device; the scheduler paces SYNTHESIS: each call starts once
the buffered runway drains to its predicted cost (EMA x safety + margin) and is
sized down to what the runway can pay for when the deadline is already
infeasible. Driven with a fake TTS and a stubbed ``_runway_ms``: no audio device
or wall-clock pacing involved. The EMA is frozen (``_MPC_ALPHA = 0``) and seeded
per test, so the near-instant fake cannot drag the schedule mid-test.
"""

from __future__ import annotations

import asyncio

import pytest

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


def _build(tts: _FakeTts | None = None) -> tuple[LocalBackend, _FakeTts, AudioSink]:
    cfg = VoiceConfig.model_validate({})
    sink = AudioSink(NullPlayback(), mode="stream")
    tts = tts or _FakeTts()

    async def transcribe(pcm: bytes) -> str:
        return ""

    async def publish(text: str, token: str, notes: tuple[str, ...] = ()) -> None:
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


@pytest.fixture(autouse=True)
def _fast_poll_frozen_ema(monkeypatch):
    monkeypatch.setattr(local_mod, "_JIT_POLL_CAP_S", 0.01)
    monkeypatch.setattr(local_mod, "_MPC_ALPHA", 0.0)


def test_first_chunk_is_never_scheduled():
    backend, tts, sink = _build()
    backend._synth_mpc = 1000.0  # a schedule this slow would hold ~forever
    backend._runway_ms = lambda: 10_000.0

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


def test_jit_holds_then_releases_with_a_budget_sized_piece():
    backend, tts, sink = _build()
    backend._synth_mpc = 10.0  # ms/char: 100 chars -> 1 s predicted synth
    backlog = [0.0]
    backend._runway_ms = lambda: backlog[0]
    chunk = "word " * 19 + "word."  # 100 chars, single spaces

    async def scenario():
        backend._cur_turn = _Turn("t")
        worker = asyncio.create_task(backend._tts_worker())
        try:
            backend._tts_enqueue("first chunk.")
            await _until(lambda: len(tts.calls) == 1)  # first chunk never scheduled
            await backend._tts_queue.join()

            backlog[0] = 10_000.0
            backend._tts_enqueue(chunk)
            await asyncio.sleep(0.08)
            # need = 100*10/1000*2 + 0.4 = 2.4 s << 10 s of runway: held.
            assert tts.calls == ["first chunk."]

            backlog[0] = 2000.0  # < need: releases; budget = (2.0-0.4)/2 s = 80 chars
            await _until(lambda: len(tts.calls) == 2)
            assert 24 <= len(tts.calls[1]) <= 80
            backlog[0] = 0.0  # the remainder's own deadline (need ~0.8 s) releases too
            await asyncio.wait_for(backend._tts_queue.join(), 2.0)
            # Pieces reconstruct the chunk, in order, cut at word boundaries.
            assert " ".join(tts.calls[1:]) == chunk
            assert backend._metrics.counters.get("tts_piece_split", 0) >= 1
        finally:
            worker.cancel()

    asyncio.run(scenario())


def test_epoch_bump_mid_candidate_drops_the_remainder():
    holder: list[AudioSink] = []

    class _FlushOnSecondCall(_FakeTts):
        async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
            audio = await super().synthesize_pcm(text)
            if len(self.calls) == 2:
                await holder[0].flush()  # barge-in lands DURING piece 1's synthesis
            return audio

    backend, tts, sink = _build(_FlushOnSecondCall())
    holder.append(sink)
    backend._synth_mpc = 10.0
    backend._runway_ms = lambda: 0.0
    # Thin runway forces min-piece cuts: several pieces for this chunk.
    chunk = "word " * 23 + "word."  # 120 chars

    async def scenario():
        backend._cur_turn = _Turn("t")
        worker = asyncio.create_task(backend._tts_worker())
        try:
            backend._tts_enqueue("intro.")
            await _until(lambda: len(tts.calls) == 1)
            await backend._tts_queue.join()

            backend._tts_enqueue(chunk)
            await asyncio.wait_for(backend._tts_queue.join(), 2.0)
            await asyncio.sleep(0.05)
            # Piece 1 synthesized (the flush hit during it): pieces 2+ never ran.
            assert len(tts.calls) == 2
        finally:
            worker.cancel()

    asyncio.run(scenario())


def test_queue_join_blocks_until_the_carried_remainder_is_emitted():
    gate = asyncio.Event()

    class _GatedTts(_FakeTts):
        async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
            if len(self.calls) >= 2:  # intro + piece 1 flow; later pieces block
                await gate.wait()
            return await super().synthesize_pcm(text)

    backend, tts, sink = _build(_GatedTts())
    backend._synth_mpc = 10.0
    backend._runway_ms = lambda: 0.0
    chunk = "word " * 11 + "word."  # 60 chars -> three min-piece cuts

    async def scenario():
        backend._cur_turn = _Turn("t")
        worker = asyncio.create_task(backend._tts_worker())
        try:
            backend._tts_enqueue("intro.")
            await _until(lambda: len(tts.calls) == 1)
            await backend._tts_queue.join()

            backend._tts_enqueue(chunk)
            await _until(lambda: len(tts.calls) == 2)  # piece 1 emitted, piece 2 gated
            # The head item's task_done is held while the remainder pends: join must
            # NOT complete, or _settle would drain the stream under unspoken text.
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(backend._tts_queue.join(), 0.05)
            gate.set()
            await asyncio.wait_for(backend._tts_queue.join(), 2.0)
            assert " ".join(tts.calls[1:]) == chunk
        finally:
            worker.cancel()

    asyncio.run(scenario())


def test_unseeded_ema_synthesizes_the_whole_candidate_immediately():
    backend, tts, sink = _build()
    backend._synth_mpc = None
    backend._runway_ms = lambda: 10_000.0
    chunk = "word " * 19 + "word."  # 100 chars

    async def scenario():
        backend._cur_turn = _Turn("t")
        backend._cur_turn.tts_first_pending = False  # land straight on the JIT path
        worker = asyncio.create_task(backend._tts_worker())
        try:
            backend._tts_enqueue(chunk)
            await asyncio.wait_for(backend._tts_queue.join(), 0.4)
            assert tts.calls == [chunk]
        finally:
            worker.cancel()

    asyncio.run(scenario())


def test_barge_in_during_the_wait_drops_the_chunk():
    backend, tts, sink = _build()
    backend._synth_mpc = 10.0
    backlog = [0.0]
    backend._runway_ms = lambda: backlog[0]

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
            assert len(tts.calls) == 1  # held by the deadline

            await sink.flush()  # barge-in: epoch bump must release the wait...
            await asyncio.wait_for(backend._tts_queue.join(), 2.0)
            assert tts.calls == ["first chunk."]  # ...and the chunk dies unsynthesized
        finally:
            worker.cancel()

    asyncio.run(scenario())


def test_stale_epoch_coalesce_leaves_the_successor_queue_alone():
    """A dead-epoch head must not drain (and re-queue, reordering) the successor
    turn's chunks: the entry guard makes the drain a no-op."""
    backend, tts, sink = _build()
    backend._tts_queue.put_nowait((sink.epoch, "NEW ONE."))
    backend._tts_queue.put_nowait((sink.epoch, "NEW TWO."))
    pending = ["old tail."]
    backend._coalesce_into(sink.epoch - 1, pending)
    assert pending == ["old tail."]
    assert backend._tts_queue.get_nowait() == (sink.epoch, "NEW ONE.")
    assert backend._tts_queue.get_nowait() == (sink.epoch, "NEW TWO.")


def test_cut_index_spares_numbers_and_prefers_sentence_ends():
    text = "The value of pi is about 3.14159 and it continues forever onward."
    cut = local_mod._cut_index(text, 30, 10)
    assert not text[:cut].rstrip().endswith("3.")  # never inside a decimal
    assert text[:cut].endswith(" ")                # fell back to the space cut
    sent = "First point made. Second point follows here."
    assert sent[: local_mod._cut_index(sent, 30, 5)] == "First point made."
    cjk = "第一句話說完了。第二句话继续讲下去了。"
    assert cjk[: local_mod._cut_index(cjk, 12, 3)].endswith("。")


def test_cut_takes_a_sentence_just_over_the_budget_whole():
    # The systematic near-miss: the JIT releases when runway ~= need, so a whole-
    # sentence candidate's budget lands 1-4 chars short of its terminator.
    sent = "画面中间是打开的防盗门或大门，可以看到门框和门把手。"  # 26 chars
    for budget in (22, 24, 25):
        assert local_mod._cut_index(sent, budget, 24) == len(sent)
    en = "This is a long English sentence okay."
    assert local_mod._cut_index(en, len(en) - 3, 24) == len(en)


def test_cut_falls_to_a_clause_below_the_floor_before_cutting_blind():
    # Terminator beyond the reach, commas only below the floor: a half-floor clause
    # boundary beats a blind cut.
    text = "门外是一条走廊或者过道，光线比较明亮，远处能看到一些室外的绿植和很多高大的树木和花草。"
    assert text[: local_mod._cut_index(text, 24, 24)] == "门外是一条走廊或者过道，光线比较明亮，"


def test_cut_never_strands_a_tiny_tail():
    # No boundary anywhere: a remainder under _CUT_MIN_TAIL is swallowed whole
    # rather than stranded as an overhead-dominated job.
    text = "画面中间是打开的防盗门或大门可以看到门框和门把手"  # 24 chars, no punctuation
    assert local_mod._cut_index(text, 20, 20) == len(text)
    # beyond the reach the cap still applies
    assert local_mod._cut_index("画" * 60, 24, 24) == 24


def test_take_piece_pops_the_head_when_the_lookahead_takes_it_whole():
    backend, _, sink = _build()
    sent = "画面中间是打开的防盗门或大门，可以看到门框和门把手。"
    pending = [sent]
    assert backend._take_piece(pending, 24) == sent
    assert pending == []
    assert "tts_piece_split" not in backend._metrics.counters  # no seam happened
    assert backend._metrics.counters.get("tts_cut_extended") == 1


def test_blind_cut_is_counted():
    backend, _, sink = _build()
    pending = ["画" * 60]
    assert len(backend._take_piece(pending, 24)) == 24
    assert backend._metrics.counters.get("tts_cut_blind") == 1
    # A cap cut through digit-bound punctuation is the blindest seam there is.
    backend2, _, _ = _build()
    pending = ["误差约为0.0012345678901234567890123456789共三十位数字随后还有很多字继续下去"]
    assert len(backend2._take_piece(pending, 24)) == 24
    assert backend2._metrics.counters.get("tts_cut_blind") == 1


def test_cut_boundary_binds_only_into_latin_tokens():
    # "…" (and half-width . ! ?) before a hanzi IS a boundary: CJK never binds.
    text = "我看了一下这个方案感觉还行吧…不过还有几个细节需要再确认一下才能定"
    cut = local_mod._cut_index(text, 16, 24)
    assert text[:cut].endswith("…")
    # digits still bind: never cut inside "3.14"
    latin = "The value is about 3.14159 and then some more words follow here okay"
    cut = local_mod._cut_index(latin, 21, 10)  # window ends between '3.' and '14159'
    assert not latin[:cut].rstrip().endswith("3.")


def test_cut_rides_over_terminator_clusters_and_closers():
    text = "他说好的没问题你确定吗？！那我们就开始吧现在就出发"
    cut = local_mod._cut_index(text, 11, 8)
    assert text[:cut].endswith("？！")   # the cluster is one boundary
    quoted = "他跟我说了一句「我知道了。」然后就转身走了再也没有回来"
    cut = local_mod._cut_index(quoted, 12, 8)
    assert quoted[:cut].endswith("。」")  # the closer rides with its sentence


def test_cut_retreats_to_the_start_of_a_long_latin_token():
    # A number run crossing the window with its end beyond the lookahead: the cut
    # retreats to the token start instead of severing the number into two reads.
    text = "测量误差约为" + "0.001234567890123456789012345" + "左右基本可以忽略"
    cut = local_mod._cut_index(text, 24, 12)
    assert text[:cut] == "测量误差约为"


def test_pure_punct_piece_is_skipped_but_keeps_reply_state():
    backend, tts, sink = _build()
    backend._reply_tail = "需要我继续吗"

    async def scenario():
        backend._cur_turn = _Turn("t")
        backend._cur_turn.spoke_text = True
        assert await backend._synth_and_emit(sink.epoch, "？") is True

    asyncio.run(scenario())
    assert tts.calls == []
    assert backend._metrics.counters.get("tts_punct_piece_skipped") == 1
    # The reply's question mark still judges sentence-attention...
    assert backend._reply_tail == "需要我继续吗？"
    # ...and a deliberately silent reply must not trip the unvoiced-final fallback.
    assert backend._cur_turn.emitted_audio


def test_symbol_only_piece_still_synthesizes():
    backend, tts, sink = _build()

    async def scenario():
        backend._cur_turn = _Turn("t")
        await backend._synth_and_emit(sink.epoch, "$")  # espeak voices it as "dollar"

    asyncio.run(scenario())
    assert tts.calls == ["$"]


def test_sub_floor_piece_seeds_but_never_updates_the_cost_ema():
    class _SlowTts(_FakeTts):
        async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
            await asyncio.sleep(0.05)  # 50 ms for a 5-char piece: overhead-dominated
            return await super().synthesize_pcm(text)

    backend, tts, sink = _build(_SlowTts())
    backend._synth_mpc = None

    async def scenario():
        backend._cur_turn = _Turn("t")
        await backend._synth_and_emit(sink.epoch, "好的谢谢。")
        # An unseeded EMA runs the pipeline unscheduled: any seed beats none.
        assert backend._synth_mpc is not None
        backend._synth_mpc = 1.0
        await backend._synth_and_emit(sink.epoch, "好的谢谢。")
        assert backend._synth_mpc == 1.0  # 5 chars: overhead sample never poisons
        await backend._synth_and_emit(sink.epoch, "好的我马上把客厅的灯打开。")
        assert backend._synth_mpc > 1.0  # 13 chars >= floor // 2: a real speed sample

    asyncio.run(scenario())


class _PaddedTts:
    """Model padding: 100 ms of tone then 600 ms of silence, whatever the text says."""

    output_rate = 16000

    async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
        return b"\x00\x40" * 1600 + b"\x00\x00" * 9600


def _tail_ms(pcm: bytes, rate: int = 16000) -> float:
    import array

    samples = array.array("h")
    samples.frombytes(pcm)
    last = max(i for i, v in enumerate(samples) if abs(v) > 0.01 * 32767)
    return (len(samples) - 1 - last) / rate * 1000.0


def test_seam_silence_is_sized_by_what_the_piece_ends_with():
    backend, _, sink = _build(_PaddedTts())  # type: ignore[arg-type]
    seen: list[bytes] = []

    async def capture(event) -> None:
        seen.append(event.pcm)

    backend._on_event = capture

    async def scenario():
        backend._cur_turn = _Turn("t")
        for text in ("我已经保存好了。", "我已经保存好了，", "然后再"):
            await backend._synth_and_emit(sink.epoch, text)

    asyncio.run(scenario())
    sent, clause, cut = (_tail_ms(pcm) for pcm in seen)
    assert (sent, clause, cut) == (140.0, 280.0, 40.0)
    # A budget cut is not a boundary: it must not out-pause a sentence end.
    assert cut < sent < clause


def test_seam_tail_reads_through_closing_punctuation():
    assert local_mod._seam_tail_ms('He said "stop."') == local_mod._SEAM_SENT_MS
    assert local_mod._seam_tail_ms("「知道了」，") == local_mod._SEAM_CLAUSE_MS
    assert local_mod._seam_tail_ms("然后再") == local_mod._SEAM_OPEN_MS
    assert local_mod._seam_tail_ms("") == local_mod._SEAM_OPEN_MS


def test_runway_prefers_the_span_ledger():
    backend, tts, sink = _build()
    sink.backlog_ms = lambda: 7000.0  # type: ignore[method-assign]
    backend._spoken_spans = [("a.", 1000.0), ("b.", 2000.0)]
    backend._spans_gen = sink.stream_generation
    backend._spans_base_ms = 500.0
    sink.played_ms = lambda: 1500  # type: ignore[method-assign]
    assert backend._runway_ms() == 2000.0
    backend._spans_gen = sink.stream_generation - 1  # stale stream: fall back
    assert backend._runway_ms() == 7000.0


def test_short_chunks_coalesce_into_one_call():
    backend, tts, sink = _build()
    backend._synth_mpc = 10.0
    # 600 ms of runway: under need (16*10/1000*2 + 0.4 = 0.72 s) so no wait, and the
    # budget clamp floor (min_chars_first=24 >= want=16) covers both chunks whole.
    backend._runway_ms = lambda: 600.0

    async def scenario():
        backend._cur_turn = _Turn("t")
        worker = asyncio.create_task(backend._tts_worker())
        try:
            backend._tts_enqueue("intro.")
            await _until(lambda: len(tts.calls) == 1)
            await backend._tts_queue.join()

            backend._tts_enqueue("Okay.")
            backend._tts_enqueue("Sure thing.")
            await asyncio.wait_for(backend._tts_queue.join(), 2.0)
            assert tts.calls[1:] == ["Okay. Sure thing."]
            assert backend._metrics.counters.get("tts_coalesced") == 1
        finally:
            worker.cancel()

    asyncio.run(scenario())
