"""LocalBackend's barge-in verdict routing and the turn-staleness guards.

Every branch here pins a wrong behaviour observed once. Driven through
`_on_utterance` over hand-built `_PendingUtterance`s so no audio device, model
or event loop pump is needed.
"""

from __future__ import annotations

import asyncio

import pytest

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import OutputAudio, VoiceState
from nanobot_channel_voice.backend.local import LocalBackend, _PendingUtterance, _Turn
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.vad.base import Vad


class _SilentVad(Vad):
    """Never speech; records duck-synchronised floor scaling."""

    def __init__(self) -> None:
        self.floor_scales: list[float] = []

    def is_speech(self, frame: bytes) -> bool:
        return False

    def scale_floor(self, factor: float) -> None:
        self.floor_scales.append(factor)


class _Harness:
    """A LocalBackend wired to recording fakes, plus the calls it made."""

    def __init__(self, backend: LocalBackend) -> None:
        self.backend = backend
        self.published: list[tuple[str, str]] = []
        self.interrupts = 0
        self.transcript = ""
        self.on_transcribe = None


def _build(**cfg_over) -> _Harness:
    # aec="soft" turns open_mic on: half-duplex zeroes duckDb, so the duck
    # assertions would pass vacuously. minWords/ackPhrases are pinned, not defaulted.
    cfg = VoiceConfig.model_validate(
        {
            "aec": "soft",
            "duckDb": -12.0,
            "bargeIn": {"minWords": 2, "ackPhrases": ["ok", "right"], "heardMarker": True},
            **cfg_over,
        }
    )
    sink = AudioSink(NullPlayback(), mode="stream")
    vad = _SilentVad()

    async def transcribe(pcm: bytes) -> str:
        if harness.on_transcribe is not None:
            await harness.on_transcribe()
        return harness.transcript

    async def publish(text: str, token: str) -> None:
        harness.published.append((text, token))

    async def interrupt() -> None:
        harness.interrupts += 1

    backend = LocalBackend(
        cfg,
        vad=vad,
        tts=None,
        sink=sink,
        transcribe=transcribe,
        publish_text=publish,
        interrupt=interrupt,
    )

    harness = _Harness(backend)
    harness.sink = sink
    harness.vad = vad
    return harness


def _utt(
    *,
    preempted: bool = False,
    heard: str | None = None,
    onset_interrupting: bool = False,
    onset_at: float = 0.0,
) -> _PendingUtterance:
    return _PendingUtterance(
        pcm=b"\x00" * 3200,
        eager=None,
        closed_reason="silence",
        closed_at=0.0,
        preempted=preempted,
        heard=heard,
        onset_interrupting=onset_interrupting,
        onset_at=onset_at,
    )


def _run(coro):
    return asyncio.run(coro)


# ---- verdict routing --------------------------------------------------------

def test_self_echo_while_speaking_is_dropped_not_published():
    h = _build()
    h.backend._turn = VoiceState.SPEAKING
    h.backend._echo.note_spoken("the capital of france is paris")
    h.transcript = "the capital of france is paris"
    _run(h.backend._on_utterance(_utt()))
    assert h.published == []
    assert h.interrupts == 0


def test_interrupt_through_echo_when_enough_fresh_words():
    """A real soft-duplex barge-in is user+leak blended; containment calls it
    echo, so the fresh-word override is the only thing that lets it through.
    (Fresh CONTENT words: a fresh stop command instead consumes, see
    test_stop_commands.py.)"""
    h = _build()
    h.backend._turn = VoiceState.SPEAKING
    h.backend._echo.note_spoken("the capital of france is paris")
    h.transcript = "the capital of france is paris turn the lamp off"
    _run(h.backend._on_utterance(_utt()))
    assert h.interrupts == 1
    assert h.published and h.published[0][0].startswith("the capital")


def test_trailing_echo_at_idle_still_drops_without_interrupting():
    h = _build()
    h.backend._turn = VoiceState.IDLE
    h.backend._echo.note_spoken("all done let me know")
    h.transcript = "all done let me know"
    _run(h.backend._on_utterance(_utt()))
    assert h.published == []


def test_half_duplex_trailing_echo_is_dropped_too():
    """The half-duplex mic gate is a read-time approximation: capture lag or a
    mistimed hangover leaks the reply's audible tail past it. Gated on open_mic,
    the echo check let that leak publish as a user turn: the bot answering its
    own last sentence."""
    h = _build(aec="auto")
    h.backend._turn = VoiceState.IDLE
    h.backend._echo.note_spoken("all done let me know")
    h.transcript = "all done let me know"
    _run(h.backend._on_utterance(_utt()))
    assert h.published == []
    assert h.interrupts == 0


def test_half_duplex_fresh_speech_still_publishes():
    """The guard against over-dropping: recently-spoken bot words must not veto a
    genuinely different user utterance in half-duplex."""
    h = _build(aec="auto")
    h.backend._echo.note_spoken("the capital of france is paris")
    h.transcript = "please turn off the desk lamp"
    _run(h.backend._on_utterance(_utt()))
    assert [t for t, _ in h.published] == ["please turn off the desk lamp"]


def test_backchannel_while_speaking_keeps_the_reply():
    h = _build()
    h.backend._turn = VoiceState.SPEAKING
    h.transcript = "ok right"
    _run(h.backend._on_utterance(_utt()))
    assert h.published == []
    assert h.interrupts == 0
    assert h.backend._metrics.counters.get("barge_in_backchannel") == 1


def test_empty_transcript_from_capturing_settles_to_idle():
    h = _build()
    h.backend._turn = VoiceState.CAPTURING
    h.transcript = ""
    _run(h.backend._on_utterance(_utt()))
    assert h.backend._turn is VoiceState.IDLE


def test_preempted_but_empty_transcript_orphans_to_idle():
    """An early confirm already stopped the reply; the endpoint verdict says the
    trigger was not speech, so the session must not sit in a dead SPEAKING."""
    h = _build()
    h.backend._turn = VoiceState.SPEAKING
    h.transcript = ""
    _run(h.backend._on_utterance(_utt(preempted=True)))
    assert h.backend._turn is VoiceState.IDLE
    assert h.backend._metrics.counters.get("barge_in_early_orphan.empty") == 1


# ---- staleness guards -------------------------------------------------------

def test_reply_that_finished_during_stt_is_not_stop_ped():
    h = _build()
    h.backend._turn = VoiceState.SPEAKING
    h.transcript = "what time is it"

    async def drain_finished():
        h.backend._turn = VoiceState.IDLE  # the drain watcher reached IDLE

    h.on_transcribe = drain_finished
    _run(h.backend._on_utterance(_utt()))
    assert h.interrupts == 0  # no bogus /stop at a finished turn
    assert h.published


def test_turn_already_killed_is_not_interrupted_twice():
    """An early confirm (or a previous utterance still in STT) may already have
    killed the live turn; a second _do_interrupt re-flushes an empty sink and
    reports a heard-up-to marker against cleared spans."""
    h = _build()
    h.backend._turn = VoiceState.SPEAKING
    h.backend._cur_turn.abandon()  # what _do_interrupt does to the dead turn
    h.transcript = "actually never mind"
    _run(h.backend._on_utterance(_utt()))
    assert h.interrupts == 0
    assert h.published


def test_confirm_latches_ride_the_utterance_not_the_instance():
    """`preempted` is bound at close time: an instance-global latch could be
    consumed by a DIFFERENT queued utterance's intake and skip a needed /stop."""
    h = _build()
    h.backend._turn = VoiceState.SPEAKING
    h.backend._preempted = True  # set by a LATER utterance's early confirm
    h.transcript = ""
    _run(h.backend._on_utterance(_utt(preempted=False)))
    # This utterance was not the preempted one, so no orphan is recorded for it.
    assert "barge_in_early_orphan.empty" not in h.backend._metrics.counters


def test_publish_carries_the_new_turn_token():
    h = _build()
    h.backend._turn = VoiceState.IDLE
    h.transcript = "hello there"
    _run(h.backend._on_utterance(_utt()))
    text, token = h.published[0]
    assert text == "hello there"
    assert token and not h.backend.is_dead_turn(token)


def test_interrupt_kills_the_old_token_but_not_a_superseded_one():
    """Only a KILLED turn's final is muted: core may coalesce a queued follow-up
    into its still-running turn, and that combined reply (echoing the older
    token) must speak."""
    async def _case():
        h = _build()
        h.backend._turn = VoiceState.IDLE
        h.transcript = "first"
        await h.backend._on_utterance(_utt())
        _, first = h.published[0]
        # Superseded WITHOUT a kill (preempted skips the /stop): stays speakable.
        h.backend._turn = VoiceState.THINKING
        h.transcript = "and also"
        await h.backend._on_utterance(_utt(preempted=True))
        assert not h.backend.is_dead_turn(first)
        # A genuine interrupt kills the CURRENT token, before any await can race it.
        # (Content words: a bare stop command would consume via the stop rung instead.)
        _, second = h.published[1]
        h.backend._turn = VoiceState.THINKING
        h.transcript = "no not tokyo"
        await h.backend._on_utterance(_utt())
        assert h.backend.is_dead_turn(second)
        assert not h.backend.is_dead_turn(first)
        return h

    h = _run(_case())
    assert h.interrupts == 1  # only the genuine interrupt sent /stop


def test_dead_turn_verdict_still_flushes_late_audio():
    """A turn killed by an early confirm can still start audio afterwards (the
    timeout notice). The verdict must stop that audio (without a second /stop)
    or the notice talks over the new turn."""
    async def _case():
        h = _build()
        h.backend._turn = VoiceState.SPEAKING
        h.backend._cur_turn = _Turn("t-dead")
        h.backend._cur_turn.abandon()  # early confirm already killed it...
        h.backend._dead_tokens.append("t-dead")
        epoch = h.sink.epoch
        h.sink.enqueue(OutputAudio(epoch=epoch, pcm=b"\x00" * 3200, rate=16000))  # ...the notice
        h.transcript = "stop talking"
        await h.backend._on_utterance(_utt())
        assert h.sink.epoch > epoch  # flushed: the notice is dead
        assert h.interrupts == 0     # but no second /stop
        assert h.published           # and the new turn still published
        return h

    _run(_case())


# ---- duck lifecycle ---------------------------------------------------------

def test_preempted_turn_releases_a_duck_raised_during_stt():
    """Regression: _do_interrupt is skipped when preempted, so without an
    explicit clear the sink stays ducked and the NEXT reply plays attenuated."""
    h = _build()
    h.backend._turn = VoiceState.SPEAKING
    h.backend._engage_duck(suspect=True)  # candidate raised while STT ran
    assert h.sink._gain_target == pytest.approx(h.backend._duck_gain)
    h.transcript = "open the door"
    _run(h.backend._on_utterance(_utt(preempted=True)))
    assert h.sink._gain_target == 1.0
    assert h.backend._duck_onset is None
    assert h.vad.floor_scales[-1] == pytest.approx(1.0 / h.backend._duck_gain)


def test_false_barge_in_releases_the_duck_and_counts_it():
    h = _build()
    h.backend._turn = VoiceState.SPEAKING
    h.sink.configure_duck(0.25)
    h.backend._engage_duck(suspect=False)
    h.backend._echo.note_spoken("still working on that")
    h.transcript = "still working on that"
    _run(h.backend._on_utterance(_utt()))
    assert h.sink._gain_target == 1.0
    assert h.backend._metrics.counters.get("barge_in_false_resume.echo") == 1


# ---- heard-up-to mapping ----------------------------------------------------

def test_heard_text_cuts_inside_the_chunk_playback_stopped_in():
    h = _build()
    h.backend._spoken_spans = [("first sentence here", 1000.0), ("second one follows", 1000.0)]
    assert h.backend._heard_text(1000.0) == "first sentence here"
    partial = h.backend._heard_text(1500.0)
    assert partial.startswith("first sentence here second")
    assert partial.endswith("...")
    assert h.backend._heard_text(0.0) == ""


# ---- sink invariants the backend depends on ---------------------------------

def test_sink_drops_audio_produced_under_a_dead_epoch():
    """The headline invariant of the sink: the worker gates on the CARRIED
    epoch, so a chunk synthesized before a barge-in is never played."""
    played: list[bytes] = []

    class _Recording(NullPlayback):
        async def play_wav(self, wav_bytes: bytes) -> bool:
            played.append(wav_bytes)
            return True

    async def _case():
        sink = AudioSink(_Recording(), mode="blob")
        await sink.start()
        stale_epoch = sink.epoch
        await sink.flush()  # barge-in: epoch moves on
        sink.enqueue(OutputAudio(epoch=stale_epoch, wav=b"RIFF0000WAVE"))
        sink.enqueue(OutputAudio(epoch=sink.epoch, wav=b"RIFF0000WAVElive"))
        await sink.wait_idle()
        await asyncio.sleep(0)
        await sink.stop()

    _run(_case())
    assert played == [b"RIFF0000WAVElive"]


def test_stale_enqueue_does_not_inflate_the_backlog_forever():
    """Credit and debit must be gated identically: an item enqueued after a
    flush is dropped by the worker, so crediting it would leave a permanent
    floor in backlog_ms() and over-hold the self-echo filter."""
    sink = AudioSink(NullPlayback(), mode="blob")
    stale_epoch = sink.epoch
    asyncio.run(sink.flush())
    sink.enqueue(OutputAudio(epoch=stale_epoch, wav=b"RIFF0000WAVE"))
    assert sink.backlog_ms() == 0


# ---- capture-gap guard ------------------------------------------------------

def test_capture_gap_drops_the_open_utterance_and_its_speculation():
    """A mid-utterance capture restart must not let the utterance bridge the
    outage (the endpointer clock is frame-counted: no frames, no silence run),
    and its eager speculation must die with it, or the PRE-GAP transcript would
    be handed to the next utterance's close."""
    h = _build()
    b = h.backend
    ep = b._endpointer
    ep._in_speech = True
    ep._buf = bytearray(b"\x00" * 640)
    b._turn = VoiceState.CAPTURING
    b._eager_valid = True
    _run(b.on_capture_gap())
    assert not ep.in_speech
    assert b._eager_valid is False
    assert b._turn is VoiceState.IDLE  # never stranded presenting "capturing"
    assert b._metrics.counters.get("capture_gap_drop") == 1


def test_capture_gap_with_no_open_utterance_is_silent():
    h = _build()
    _run(h.backend.on_capture_gap())
    assert h.backend._metrics.counters.get("capture_gap_drop") is None


# ---- AEC warmup carve-out ---------------------------------------------------

class _StubAec:
    def __init__(self, ref_ms: float):
        self._ref = ref_ms

    def reference_ms(self) -> float:
        return self._ref


def _finished_task(text: str):
    async def _t():
        return text

    async def _make():
        task = asyncio.get_running_loop().create_task(_t())
        await task
        return task

    return asyncio.run(_make())


def _arm_eager(b, task) -> None:
    """The callback only trusts the CURRENT candidate's own decode."""
    b._eager_task = task
    b._eager_valid = True


def test_aec_warmup_holds_the_eager_confirm():
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    b._duck_onset = 1.0
    b._endpointer._in_speech = True
    b._aec = _StubAec(ref_ms=500.0)  # converging: too little reference processed
    task = _finished_task("brand new words entirely")
    _arm_eager(b, task)
    b._eager_confirm_cb(task)
    assert b._early_confirm is False
    assert b._metrics.counters.get("barge_in_warmup_hold") == 1


def test_after_enough_reference_audio_the_eager_confirm_fires():
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    b._duck_onset = 1.0
    b._endpointer._in_speech = True
    b._aec = _StubAec(ref_ms=5000.0)  # converged: plenty of reference processed
    task = _finished_task("brand new words entirely")
    _arm_eager(b, task)
    b._eager_confirm_cb(task)
    assert b._early_confirm is True
    assert b._metrics.counters.get("barge_in_eager_confirm") == 1


def test_stale_eager_task_cannot_judge_the_live_candidate():
    """A dropped/superseded candidate's decode must neither confirm nor acquit the
    utterance that is open NOW: its audio belongs to a different candidate."""
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    b._duck_onset = 1.0
    b._endpointer._in_speech = True
    b._aec = _StubAec(ref_ms=5000.0)
    task = _finished_task("brand new words entirely")
    _arm_eager(b, task)
    b._eager_valid = False  # what _drop_candidate / a skip leaves behind
    b._eager_confirm_cb(task)
    assert b._early_confirm is False
    assert b._early_release is None


# ---- metrics turn timeline (tool boundary + timeout) ------------------------

def test_tool_boundary_reanchors_post_tool_audio_as_continuation():
    async def _t():
        h = _build()
        b = h.backend
        h.transcript = "what's the weather"
        await b._on_utterance(_utt())  # publish -> THINKING, anchor armed
        sid = "s:1000000000000000000:0"
        await b.on_delta("One sec.", stream_id=sid)
        b._metrics.turn_first_audio()  # the pre-tool status line's chunk
        await b.on_stream_end(resuming=True, stream_id=sid)
        assert b._cur_turn.continuation_pending is True
        await b.on_delta("It is sunny out.", stream_id="s:1000000000000000000:1")
        assert b._cur_turn.continuation_pending is False
        b._metrics.turn_first_audio()  # first post-tool chunk
        lat = b._metrics.snapshot()["latency_ms"]
        assert lat["ttfa_ms"]["n"] == 1
        assert lat["continuation_ms"]["n"] == 1

    _run(_t())


def test_tool_only_turn_keeps_tool_time_out_of_ttfa():
    async def _t():
        h = _build()
        b = h.backend
        h.transcript = "look this up"
        await b._on_utterance(_utt())
        sid = "s:1000000000000000000:0"
        # The agent went straight to the tool call: no pre-tool delta, no audio.
        await b.on_stream_end(resuming=True, stream_id=sid)
        await b.on_delta("Found it.", stream_id="s:1000000000000000000:1")
        b._metrics.turn_first_audio()  # the turn's FIRST audio, after the tool ran
        lat = b._metrics.snapshot()["latency_ms"]
        assert "ttfa_ms" not in lat  # the tool wait must not read as model latency
        assert lat["continuation_ms"]["n"] == 1

    _run(_t())


def test_timeout_notice_audio_is_not_ttfa():
    async def _t():
        h = _build(agentTimeoutS=0.05)
        b = h.backend
        h.transcript = "hello there"
        await b._on_utterance(_utt())  # publish arms the timeout deadman
        for _ in range(80):  # let the watch fire (0.05 s stall budget)
            await asyncio.sleep(0.01)
            if b._metrics.counters.get("agent_turn_timeout"):
                break
        assert b._metrics.counters.get("agent_turn_timeout") == 1
        assert h.interrupts == 1  # the stuck run was /stop-ped
        b._metrics.turn_first_audio()  # the notice's first chunk
        snap = b._metrics.snapshot()
        assert "ttfa_ms" not in snap["latency_ms"]  # anchor was released
        assert snap["counters"].get("ttfa_unanchored") == 1

    _run(_t())
