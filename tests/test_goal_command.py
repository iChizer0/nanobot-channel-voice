"""Spoken entry to core's sustained-goal mode.

An ordinary turn ends the moment the model answers without calling a tool, so a
promise to keep trying IS the give-up. Core's ``/goal`` is the one mode that
re-prompts instead of ending — and speech can never produce a slash command, so
a configured phrase publishes the utterance behind it.

Driven through ``_on_utterance``, same harness shape as test_stop_commands.py.
"""

from __future__ import annotations

import asyncio

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import VoiceState
from nanobot_channel_voice.backend.local import LocalBackend, _PendingUtterance
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
    cfg = VoiceConfig.model_validate({"aec": "soft", **cfg_over})
    harness = _Harness()

    async def transcribe(pcm: bytes) -> str:
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


def _utt(*, onset_interrupting: bool = False) -> _PendingUtterance:
    return _PendingUtterance(
        pcm=b"\x00" * 3200,
        eager=None,
        closed_reason="silence",
        closed_at=0.0,
        preempted=False,
        heard=None,
        onset_interrupting=onset_interrupting,
        onset_at=0.0,
    )


def _run(coro):
    return asyncio.run(coro)


def test_a_commitment_phrase_publishes_the_whole_utterance_as_a_goal():
    h = _build()
    b = h.backend
    h.transcript = "find me a flight to Tokyo and keep working on it until it's booked"
    _run(b._on_utterance(_utt()))
    text, _, _ = h.published[0]
    # Verbatim behind the command: the sentence IS the objective, and core hands the
    # raw content to the model, which consolidates it via create_goal.
    assert text == "/goal " + h.transcript
    assert b._metrics.counters.get("goal_command") == 1


def test_an_ordinary_utterance_is_untouched():
    h = _build()
    h.transcript = "what's the weather tomorrow"
    _run(h.backend._on_utterance(_utt()))
    assert h.published[0][0] == "what's the weather tomorrow"
    assert "goal_command" not in h.backend._metrics.counters


def test_a_phrase_buried_in_a_fused_cjk_sentence_still_counts():
    # The reason phrase_within exists: PhraseMatcher.present decomposes a CJK token
    # entirely into lexicon singles, so a trigger inside real content never matches.
    h = _build()
    h.transcript = "持续处理这个问题直到解决"
    _run(h.backend._on_utterance(_utt()))
    assert h.published[0][0] == "/goal 持续处理这个问题直到解决"


def test_a_goal_kills_a_live_run_instead_of_injecting_into_it():
    h = _build()
    b = h.backend
    b._turn = VoiceState.THINKING
    b._cur_turn.continuation_pending = True  # mid-tool: an ordinary steer would inject
    h.transcript = "keep working on it until you get an answer"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    # Core dispatches a command inline rather than queueing it, so an injected /goal
    # would answer out of band while the old run kept going.
    assert h.interrupts == 1
    assert "midturn_injection" not in b._metrics.counters
    assert h.published[0][0].startswith("/goal ")


def test_a_goal_over_a_playing_answer_still_publishes_as_one():
    # The kill path and the steer path reach the publish tail differently; the
    # transform must survive both, and this is the one a ladder reorder would break.
    h = _build()
    b = h.backend
    b._turn = VoiceState.SPEAKING
    h.transcript = "actually keep working on it until you have a real answer"
    _run(b._on_utterance(_utt(onset_interrupting=True)))
    assert h.interrupts == 1
    assert h.published[0][0].startswith("/goal ")
    assert b._metrics.counters.get("goal_command") == 1


def test_an_empty_phrase_list_turns_the_feature_off():
    h = _build(goal={"phrases": []})
    h.transcript = "keep working on it until it's done"
    _run(h.backend._on_utterance(_utt()))
    assert h.published[0][0] == "keep working on it until it's done"
