"""Adaptive hangover (vad.hangoverMinMs): the learner's signal gating and the
backend's commit-on-publish wiring."""

from __future__ import annotations

import asyncio

import pytest

from nanobot_channel_voice.config import VadConfig, VoiceConfig
from nanobot_channel_voice.vad.adaptive import AdaptiveHangover
from nanobot_channel_voice.vad.base import Vad


def _learner(min_ms=300, max_ms=600) -> AdaptiveHangover:
    return AdaptiveHangover(min_ms, max_ms)


def test_starts_at_the_floor():
    assert _learner().value_ms() == 300


def test_awaiting_reply_resume_learns_the_cut_pause():
    a = _learner()
    a.note_close(close_mono=10.0, silence_ms=300.0)
    # Resume 0.2 s after the close, reply not yet audible: pause = 300 + 200 = 500.
    a.on_onset(awaiting_reply=True, speaking=False, audible_at=None, now=10.2)
    learn = a.take_pending()
    assert learn == pytest.approx(500.0)
    a.on_publish(learn)
    # EMA: 0.9*300 + 0.1*min(500*1.2, 600) = 330.
    assert a.value_ms() == 330


def test_barge_in_within_grace_of_audible_learns():
    a = _learner()
    a.note_close(close_mono=10.0, silence_ms=300.0)
    a.on_onset(awaiting_reply=False, speaking=True, audible_at=10.3, now=10.8)
    assert a.take_pending() is not None


def test_barge_in_well_into_the_reply_does_not_learn():
    a = _learner()
    a.note_close(close_mono=10.0, silence_ms=300.0)
    a.on_onset(awaiting_reply=False, speaking=True, audible_at=10.1, now=11.3)  # > 1 s grace
    assert a.take_pending() is None


def test_long_gap_is_a_new_thought_not_a_pause():
    a = _learner()
    a.note_close(close_mono=10.0, silence_ms=300.0)
    a.on_onset(awaiting_reply=True, speaking=False, audible_at=None, now=12.0)  # > 2*600
    assert a.take_pending() is None


def test_new_onset_supersedes_a_stale_candidate():
    """A candidate latched by a blip (never consumed by a close) must not be
    committed by a later, unrelated utterance."""
    a = _learner()
    a.note_close(close_mono=10.0, silence_ms=300.0)
    a.on_onset(awaiting_reply=True, speaking=False, audible_at=None, now=10.2)  # blip latches
    # 6 s later: a genuine interruption of the audible reply, NOT evidence.
    a.on_onset(awaiting_reply=False, speaking=True, audible_at=10.4, now=16.4)
    assert a.take_pending() is None


def test_dropped_anchor_measures_nothing():
    """A rejected close (echo/empty/ack) must not anchor the next resume gap."""
    a = _learner()
    a.note_close(close_mono=10.0, silence_ms=300.0)
    a.drop_anchor()
    a.on_onset(awaiting_reply=True, speaking=False, audible_at=None, now=10.2)
    assert a.take_pending() is None


def test_value_clamps_at_the_ceiling():
    a = _learner()
    for _ in range(200):
        a.on_publish(590.0)
    assert a.value_ms() == 600


def test_uncommitted_candidate_changes_nothing():
    a = _learner()
    a.note_close(close_mono=10.0, silence_ms=300.0)
    a.on_onset(awaiting_reply=True, speaking=False, audible_at=None, now=10.2)
    a.take_pending()  # bound to an utterance that is later REJECTED: never committed
    assert a.value_ms() == 300


def test_config_rejects_floor_at_or_above_ceiling():
    with pytest.raises(Exception, match="hangoverMinMs"):
        VadConfig.model_validate({"hangoverMs": 400, "hangoverMinMs": 400})


# ---- backend commit-on-publish ---------------------------------------------

class _SilentVad(Vad):
    def is_speech(self, frame: bytes) -> bool:
        return False


def test_publish_commits_and_retargets_the_endpointer():
    from nanobot_channel_voice.audio.null import NullPlayback
    from nanobot_channel_voice.backend.audio_sink import AudioSink
    from nanobot_channel_voice.backend.local import LocalBackend, _PendingUtterance

    cfg = VoiceConfig.model_validate({
        "vad": {"hangoverMs": 400, "hangoverMinMs": 100},
    })
    published = []

    async def transcribe(pcm: bytes) -> str:
        return "hello there"

    async def publish(text: str, token: str) -> None:
        published.append(text)

    async def interrupt() -> None:
        pass

    backend = LocalBackend(
        cfg, vad=_SilentVad(), tts=None,
        sink=AudioSink(NullPlayback(), mode="blob"),
        transcribe=transcribe, publish_text=publish, interrupt=interrupt,
    )
    assert backend._endpointer._hangover_frames == 100 // 20  # starts at the floor

    pending = _PendingUtterance(
        pcm=b"\x00" * 3200, eager=None, closed_reason="silence",
        closed_at=0.0, silence_ms=100, learn_ms=350.0,
    )
    asyncio.run(backend._on_utterance(pending))
    assert published
    # EMA from the 100 floor with sample min(350*1.2, 400) = 400 -> 130 ms -> 6 frames.
    assert backend._adaptive.value_ms() == 130
    assert backend._endpointer._hangover_frames == 130 // 20
    assert backend._metrics.counters.get("eou_hangover_learned") == 1


def test_turn_model_raises_the_adaptive_floor_above_consult():
    """hangoverMinMs below consultMs would silently disable the turn model: the
    floor is clamped to consultMs + one frame, with the ceiling untouched."""
    from nanobot_channel_voice.audio.null import NullPlayback
    from nanobot_channel_voice.backend.audio_sink import AudioSink
    from nanobot_channel_voice.backend.local import LocalBackend

    class _FakeAnalyzer:
        window_bytes = 16000 * 8 * 2
        last_probability = 0.5

        def assess(self, pcm: bytes) -> bool:
            return False

        def release(self) -> None:
            pass

    cfg = VoiceConfig.model_validate({
        "vad": {
            "hangoverMs": 600, "hangoverMinMs": 200,
            "turn": {"engine": "smartturn", "consultMs": 240},
        },
    })

    async def _n(*a, **k):
        return ""

    backend = LocalBackend(
        cfg, vad=_SilentVad(), tts=None,
        sink=AudioSink(NullPlayback(), mode="blob"),
        transcribe=_n, publish_text=_n, interrupt=_n,
        turn_analyzer=_FakeAnalyzer(),
    )
    # Floor clamped to 240 + 20 = 260 ms -> 13 frames > the 12-frame consult mark.
    assert backend._endpointer._hangover_frames == 260 // 20
    assert backend._endpointer._consult_frames == 240 // 20
    assert backend._adaptive._min == 260.0
