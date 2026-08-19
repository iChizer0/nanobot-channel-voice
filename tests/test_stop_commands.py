"""The stop-command verdict class: a pure stop aimed at a live reply kills it
and is CONSUMED — nothing published, silence is the acknowledgment — while the
heard-up-to contract rides the NEXT publish as a pending note.

Driven through `_on_utterance` over hand-built `_PendingUtterance`s, same
harness shape as test_local_backend.py.
"""

from __future__ import annotations

import asyncio
import time

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import VoiceState
from nanobot_channel_voice.backend.local import LocalBackend, _heard_tail, _PendingUtterance
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.vad.base import Vad


class _SilentVad(Vad):
    def is_speech(self, frame: bytes) -> bool:
        return False

    def scale_floor(self, factor: float) -> None:
        pass


class _Harness:
    def __init__(self) -> None:
        self.published: list[tuple[str, str, tuple[str, ...]]] = []
        self.interrupts = 0
        self.transcript = ""
        self.on_transcribe = None


def _build(**cfg_over) -> _Harness:
    cfg = VoiceConfig.model_validate(
        {
            "aec": "soft",
            "duckDb": -12.0,
            "bargeIn": {
                "minWords": 2,
                "ackPhrases": ["ok", "right", "好的"],
                "stopPhrases": ["stop", "shut up", "wait", "停", "停停"],
                "heardMarker": True,
            },
            **cfg_over,
        }
    )
    harness = _Harness()

    async def transcribe(pcm: bytes) -> str:
        if harness.on_transcribe is not None:
            await harness.on_transcribe()
        return harness.transcript

    async def publish(text: str, token: str, notes: tuple[str, ...] = ()) -> None:
        harness.published.append((text, token, notes))

    async def interrupt() -> None:
        harness.interrupts += 1

    harness.backend = LocalBackend(
        cfg,
        vad=_SilentVad(),
        tts=None,
        sink=AudioSink(NullPlayback(), mode="stream"),
        transcribe=transcribe,
        publish_text=publish,
        interrupt=interrupt,
    )
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


def test_heard_tail_bounds_the_quote():
    # The cut point is the note's only new information — the full reply is already the
    # assistant turn above it. Short heard text passes through whole (no false ellipsis).
    assert _heard_tail("the first bit") == "the first bit"
    long = " ".join(f"w{i}" for i in range(30))
    tail = _heard_tail(long)
    assert tail.startswith("…") and tail.endswith("w29")
    assert len(tail.lstrip("…").split()) == 12
    # A zh reply is one unspaced run: character-bounded, same cut marker.
    zh = "明天上午晴转多云下午有阵雨气温二十到二十八度东南风三级预计夜间转小雨请记得带伞"
    zh_tail = _heard_tail(zh)
    assert zh_tail == "…" + zh[-20:]


def test_stop_while_speaking_is_consumed_silently():
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    h.transcript = "stop"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    assert h.published == []          # the whole point: no reply to "stop"
    assert h.interrupts == 1          # the live turn WAS killed over the bus
    assert b._turn is VoiceState.IDLE
    assert b._metrics.counters.get("barge_in_stop") == 1
    assert b._pending_note is not None
    assert "stopped your previous reply" in b._pending_note


def test_stop_note_rides_the_next_publish_once():
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    h.transcript = "stop"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    b._turn = VoiceState.IDLE
    h.transcript = "what about tomorrow"
    _run(b._on_utterance(_utt()))
    text, _, notes = h.published[0]
    assert text == "what about tomorrow"      # the user row stays pure speech
    assert any("stopped your previous reply" in n for n in notes)
    assert b._pending_note is None
    h.transcript = "and the day after"
    _run(b._on_utterance(_utt()))
    # The STOP note rides exactly one publish; this third utterance interrupts the
    # still-thinking second turn, so an interrupt marker legitimately rides instead.
    assert not any("stopped" in n for n in h.published[1][2])


def test_cold_stop_with_nothing_live_publishes():
    h = _build()
    h.backend._turn = VoiceState.IDLE
    h.transcript = "stop"
    _run(h.backend._on_utterance(_utt()))
    assert [t for t, _, _ in h.published] == ["stop"]
    assert h.interrupts == 0
    assert "barge_in_stop" not in h.backend._metrics.counters


def test_double_tap_grace_consumes_the_second_stop():
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    h.transcript = "stop"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    # Second bare stop lands after the kill already IDLEd the session.
    h.transcript = "stop"
    _run(b._on_utterance(_utt(onset_at=time.monotonic())))
    assert h.published == []
    assert b._metrics.counters.get("barge_in_stop") == 2
    assert h.interrupts == 1  # nothing live for the second one: no second /stop


def test_stop_plus_content_publishes_as_a_normal_interrupt():
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    h.transcript = "stop use tokyo instead"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    assert h.interrupts == 1
    assert h.published and h.published[0][0].startswith("stop use tokyo instead")
    assert "barge_in_stop" not in b._metrics.counters


def test_preempted_stop_consumes_without_a_second_kill():
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    b._cur_turn.abandon()  # the early confirm already killed the turn
    h.transcript = "stop stop"
    _run(b._on_utterance(_utt(preempted=True, heard="the first bit")))
    assert h.published == []
    assert h.interrupts == 0
    assert 'they heard up to: "the first bit"' in b._pending_note


def test_stop_through_echo_is_consumed():
    """Leak + a fresh stop word classifies as self-echo; one fresh stop must
    still kill AND consume — the old >=2-fresh-words bar dropped it entirely."""
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    b._echo.note_spoken("the weather in tokyo is sunny today")
    h.transcript = "the weather in tokyo is sunny stop"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    assert h.published == []
    assert h.interrupts == 1
    assert b._metrics.counters.get("barge_in_stop") == 1


def test_reply_that_drains_during_stt_is_still_consumed_by_the_onset_latch():
    """The user said stop AT a live reply; the reply finished while STT ran.
    Verdict-time state would launder the stop into a cold turn ("stop what?")."""
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    h.transcript = "stop"

    async def drain_finished():
        b._turn = VoiceState.IDLE

    h.on_transcribe = drain_finished
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    assert h.published == []
    assert h.interrupts == 0            # nothing live at the verdict: no bogus /stop
    assert b._pending_note is None      # nothing was cut off: no note
    assert b._metrics.counters.get("barge_in_stop") == 1


def test_multiword_stop_through_echo_consumes():
    """'shut up' blended with leak classifies as self-echo; the ordered fresh
    remainder restores contiguity, so multi-word stop phrases consume here too."""
    h = _build(bargeIn={
        "minWords": 2, "ackPhrases": ["ok"], "stopPhrases": ["stop", "shut up"],
        "heardMarker": True,
    })
    b = h.backend
    b._turn = VoiceState.SPEAKING
    b._echo.note_spoken("the weather in tokyo is sunny today")
    h.transcript = "the weather in tokyo is sunny shut up"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    assert h.published == []
    assert h.interrupts == 1
    assert b._metrics.counters.get("barge_in_stop") == 1


def test_content_kill_does_not_arm_the_stop_grace():
    """The grace exists for stop double-taps. A CONTENT barge-in's kill must not arm
    it: right after such a kill the agent may ask a question ('say cancel to abort'),
    and a bare 'cancel' answered at IDLE has to reach it."""
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    h.transcript = "use tokyo instead"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    assert h.interrupts == 1 and len(h.published) == 1
    b._turn = VoiceState.IDLE
    h.transcript = "stop"
    _run(b._on_utterance(_utt(onset_at=time.monotonic())))
    assert [t for t, _, _ in h.published][1] == "stop"  # forwarded, not consumed
    assert "barge_in_stop" not in b._metrics.counters


def test_stop_note_makes_no_heard_claim_when_accounting_is_off():
    h = _build(bargeIn={
        "minWords": 2, "ackPhrases": ["ok"], "stopPhrases": ["stop"],
        "heardMarker": False,
    })
    b = h.backend
    b._turn = VoiceState.SPEAKING
    h.transcript = "stop"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    note = b._pending_note
    assert note is not None and "do not resume" in note
    assert "heard" not in note  # no accounting -> no claim about what was heard


def test_cjk_stop_repetition_consumes_and_ack_repetition_keeps_the_reply():
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    h.transcript = "好的好的"  # fused backchannel: the old subset check killed the reply
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    assert h.published == []
    assert h.interrupts == 0
    assert b._metrics.counters.get("barge_in_backchannel") == 1
    b._turn = VoiceState.SPEAKING
    h.transcript = "停停停"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    assert h.published == []
    assert h.interrupts == 1
    assert b._metrics.counters.get("barge_in_stop") == 1
