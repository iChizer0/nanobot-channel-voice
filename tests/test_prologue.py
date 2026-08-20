"""Prologue fillers: the per-wait escalation script, skip semantics, and prewarm."""

from __future__ import annotations

import asyncio

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import OutputAudio, VoiceState
from nanobot_channel_voice.backend.local import LocalBackend
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.vad.base import Vad


class _SilentVad(Vad):
    def is_speech(self, frame: bytes) -> bool:
        return False

    def scale_floor(self, factor: float) -> None:
        pass


def _pcm(text: str) -> bytes:
    """Deterministic, phrase-distinct S16 payload so emitted audio names its phrase."""
    return text.encode("utf-16-le") * 4


class _FakeTts:
    output_rate = 16000

    def __init__(self) -> None:
        self.probe_ok = True
        self.calls: list[str] = []  # cache MISSES only: _synth_filler caches

    async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
        self.calls.append(text)
        return _pcm(text)


def _build(**prologue):
    cfg = VoiceConfig.model_validate(
        {"playbackHangoverMs": 1, "prologue": {"enabled": True, **prologue}}
    )
    sink = AudioSink(NullPlayback(), mode="stream")
    tts = _FakeTts()

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
    return backend, tts, sink


async def _started(backend):
    events: list = []

    async def on_event(e) -> None:
        events.append(e)

    await backend.start(instructions=None, tools=[], on_event=on_event)
    return events


def _spoken(events) -> list[bytes]:
    return [e.pcm for e in events if isinstance(e, OutputAudio)]


def test_filler_script_runs_in_order_and_sticks_on_last():
    async def _t():
        b, tts, sink = _build(phrases=["one", "two"])
        events = await _started(b)
        b._turn = VoiceState.THINKING
        for step in (0, 1, 5):  # 5: any step past the end plays the LAST phrase
            assert await b._play_filler(sink.epoch, step) is True
        assert _spoken(events) == [_pcm("one"), _pcm("two"), _pcm("two")]
        assert tts.calls == ["one", "two"]  # the repeat came from the cache
        await b.close()

    asyncio.run(_t())


def test_filler_skips_while_user_is_speaking_without_consuming_the_script():
    async def _t():
        b, tts, sink = _build(phrases=["one"])
        events = await _started(b)
        b._turn = VoiceState.THINKING
        b._endpointer._in_speech = True
        assert await b._play_filler(sink.epoch, 0) is False  # the watch must not advance
        assert _spoken(events) == []
        b._endpointer._in_speech = False
        assert await b._play_filler(sink.epoch, 0) is True
        assert _spoken(events) == [_pcm("one")]
        await b.close()

    asyncio.run(_t())


def test_watch_restarts_the_script_per_wait():
    """A fresh wait opens with phrase 0 — a session-global rotation would open
    mid-script (the bug this pins)."""

    async def _t():
        b, tts, sink = _build(phrases=["one", "two"], afterMs=10, intervalMs=1000)
        events = await _started(b)
        for _ in range(2):  # two separate waits
            b._turn = VoiceState.THINKING
            b._arm_prologue()
            for _ in range(200):  # first filler after ~afterMs
                await asyncio.sleep(0.005)
                if _spoken(events):
                    break
            b._cur_turn.cancel_prologue()
            assert _spoken(events) == [_pcm("one")]
            events.clear()
            b._turn = VoiceState.IDLE
        await b.close()

    asyncio.run(_t())


def test_tool_boundary_rearm_opens_at_step_one():
    """The agent's own status line was the script's opener: the deferred canned
    filler must not de-escalate back to phrase 0."""

    async def _t():
        b, tts, sink = _build(phrases=["one", "two", "three"], intervalMs=1000)
        events = await _started(b)
        b._turn = VoiceState.THINKING
        b._arm_prologue(initial_ms=10, start_step=1)
        for _ in range(200):
            await asyncio.sleep(0.005)
            if _spoken(events):
                break
        b._cur_turn.cancel_prologue()
        assert _spoken(events) == [_pcm("two")]
        await b.close()

    asyncio.run(_t())


def test_first_delta_cancels_the_filler_watch():
    async def _t():
        b, tts, sink = _build(afterMs=20, intervalMs=1000)
        events = await _started(b)
        b._turn = VoiceState.THINKING
        b._arm_prologue()
        await b.on_delta("hi")  # the reply is arriving: no more filler
        await asyncio.sleep(0.06)
        assert _spoken(events) == []
        await b.close()

    asyncio.run(_t())


def test_first_filler_delay_adapts_to_typical_first_reply():
    b, _, _ = _build(afterMs=2000)
    # No history: the configured floor.
    assert b._prologue_delay_ms(None) == 2000.0
    # Fast model: the floor still rules.
    b._note_first_reply(400.0)
    assert b._prologue_delay_ms(None) == 2000.0
    # Slow model: stretch past typical latency so filler + answer never collide.
    b._first_reply_ema = None
    b._note_first_reply(4000.0)
    assert b._prologue_delay_ms(None) == 6000.0  # 1.5x EMA
    # EMA blends; a pathological turn is clamped at the 15 s sample cap.
    b._note_first_reply(120000.0)
    assert b._first_reply_ema == 0.3 * 15000.0 + 0.7 * 4000.0
    # Tool-boundary re-arms pass an explicit delay: no stretching.
    assert b._prologue_delay_ms(500) == 500.0


def test_speak_final_feeds_the_first_reply_ema():
    async def _t():
        b, _, _ = _build(afterMs=2000)
        await _started(b)
        b._cur_turn.await_first_token = True
        b._cur_turn.published_at = __import__("time").monotonic() - 3.0
        await b.speak_final("hello there")
        assert b._first_reply_ema is not None and 2800 <= b._first_reply_ema <= 3500
        await b.close()

    asyncio.run(_t())


def test_prewarm_fills_the_cache_and_respects_probe_ok_and_enabled():
    async def _t():
        b, tts, sink = _build(phrases=["one", "two"])
        await b.prewarm_canned()
        assert tts.calls == ["one", "two"]
        await b.prewarm_canned()  # cached: no re-synthesis
        assert tts.calls == ["one", "two"]

        b2, tts2, _ = _build(phrases=["one"])
        tts2.probe_ok = False  # a billing adapter must never prewarm
        await b2.prewarm_canned()
        assert tts2.calls == []

        b3, tts3, _ = _build(enabled=False)
        await b3.prewarm_canned()
        assert tts3.calls == []

        b4, tts4, _ = _build(phrases=["one"])
        b4._turn = VoiceState.THINKING  # a live turn owns the adapter: prewarm yields
        await b4.prewarm_canned()
        assert tts4.calls == []
        for backend in (b, b2, b3, b4):
            await backend.close()

    asyncio.run(_t())


def test_phrase_defaults_follow_the_tts_language():
    from nanobot_channel_voice.backend.local import (
        _PROLOGUE_BUILTINS,
        _PROLOGUE_FALLBACK,
        _prologue_phrases,
    )

    assert _prologue_phrases(None, "zh") == _PROLOGUE_BUILTINS["zh"]
    assert _prologue_phrases(None, None) == _PROLOGUE_FALLBACK
    assert _prologue_phrases(None, "fr") == _PROLOGUE_FALLBACK  # no builtin -> English
    assert _prologue_phrases(["custom"], "zh") == ["custom"]    # explicit always wins
    assert _prologue_phrases([], "zh") == []                    # explicit off


def test_backend_resolves_builtins_when_phrases_omitted():
    from nanobot_channel_voice.backend.local import _PROLOGUE_FALLBACK

    b, _tts, _sink = _build()  # _FakeTts declares no spoken_language -> English builtins
    assert b._prologue_phrases == _PROLOGUE_FALLBACK
