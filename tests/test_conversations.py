"""Scripted end-to-end conversations (the eval scenarios the harness exists for):
published bus turns, barge-in verdict outcomes, marker content, and the
false-candidate economics — through the REAL pipeline, not hand-built pendings."""

from __future__ import annotations

import asyncio

from eval_harness import EvalConversation

from nanobot_channel_voice.backend.base import VoiceState

_REPLY = "The weather in tokyo is sunny today. Tomorrow will be cloudy all day."


def _run(coro):
    return asyncio.run(coro)


def test_plain_turn_full_lifecycle():
    async def _case():
        async with EvalConversation() as c:
            await c.user_says("what is the weather in tokyo")
            assert c.texts() == ["what is the weather in tokyo"]
            await c.agent_replies("It is sunny today.")
            await c.wait_state(VoiceState.SPEAKING)
            await c.wait_state(VoiceState.IDLE)
            assert c.states[:2] == [VoiceState.CAPTURING, VoiceState.THINKING]
            assert c.states[-2:] == [VoiceState.SPEAKING, VoiceState.IDLE]
            assert c.interrupts == 0

    _run(_case())


def test_content_barge_in_kills_and_carries_the_heard_marker():
    async def _case():
        async with EvalConversation() as c:
            await c.user_says("what is the weather")
            await c.agent_replies(_REPLY)
            await c.wait_state(VoiceState.SPEAKING)
            await c.wait_played_ms(100)  # a few words audibly out
            await c.user_says("turn off the desk lamp")
            assert c.interrupts == 1                     # /stop went out
            assert len(c.texts()) == 2
            barge = c.texts()[1]
            assert barge.startswith("turn off the desk lamp")
            assert "[note: you were interrupted mid-reply" in barge
            assert "The weather" in barge                # heard-up-to names real words
            assert c.backend._turn is VoiceState.THINKING

    _run(_case())


def test_stop_mid_reply_goes_silent_and_notes_the_next_turn():
    async def _case():
        async with EvalConversation() as c:
            await c.user_says("tell me about tokyo")
            await c.agent_replies(_REPLY)
            await c.wait_state(VoiceState.SPEAKING)
            await c.wait_played_ms(100)
            epoch = c.sink.epoch
            await c.user_says("stop")
            assert c.texts() == ["tell me about tokyo"]  # NOTHING published for the stop
            assert c.interrupts == 1                     # but the live turn was killed
            assert c.sink.epoch > epoch                  # audio flushed
            assert c.backend._turn is VoiceState.IDLE
            assert c.counter("barge_in_stop") == 1
            assert c.counter("barge_in_duck") >= 1       # yielded before the verdict
            await c.user_says("what about osaka")
            note = c.texts()[1]
            assert note.startswith("what about osaka")
            assert "stopped your previous reply" in note
            assert 'heard only: "The weather' in note

    _run(_case())


def test_double_tap_stop_is_swallowed_by_the_grace():
    async def _case():
        async with EvalConversation() as c:
            await c.user_says("tell me about tokyo")
            await c.agent_replies(_REPLY)
            await c.wait_state(VoiceState.SPEAKING)
            await c.user_says("stop")
            await c.user_says("stop")                    # lands at IDLE, inside the grace
            assert c.texts() == ["tell me about tokyo"]
            assert c.counter("barge_in_stop") == 2
            assert c.interrupts == 1                     # nothing live the second time

    _run(_case())


def test_backchannel_leaves_the_reply_playing():
    async def _case():
        async with EvalConversation() as c:
            await c.user_says("tell me about tokyo")
            await c.agent_replies(_REPLY)
            await c.wait_state(VoiceState.SPEAKING)
            await c.user_says("ok right")
            assert c.counter("barge_in_backchannel") == 1
            assert c.interrupts == 0
            assert c.backend._turn is VoiceState.SPEAKING  # reply survived
            await c.wait_state(VoiceState.IDLE)            # and finishes naturally
            assert c.texts() == ["tell me about tokyo"]

    _run(_case())


def test_own_reply_words_after_drain_are_dropped_as_echo():
    async def _case():
        async with EvalConversation() as c:
            await c.user_says("tell me about tokyo")
            await c.agent_replies("It is sunny today.")
            await c.wait_state(VoiceState.SPEAKING)
            await c.wait_state(VoiceState.IDLE)
            await c.user_says("it is sunny today")       # the trailing leak shape
            assert c.texts() == ["tell me about tokyo"]
            assert c.interrupts == 0

    _run(_case())


def test_pause_probe_acquits_leak_and_the_reply_survives():
    async def _case():
        async with EvalConversation(bargeIn={
            "mode": "pause",
            "minWords": 2,
            "ackPhrases": ["ok", "right"],
            "stopPhrases": ["stop"],
            "heardMarker": True,
        }) as c:
            await c.user_says("tell me about tokyo")
            await c.agent_replies(_REPLY)
            await c.wait_state(VoiceState.SPEAKING)
            await c.wait_played_ms(60)
            await c.user_noise(speech_frames=5, silence_frames=13)  # buffered-tail shape
            assert c.counter("barge_in_false_resume.probe") == 1
            assert not c.sink.paused                     # released, not held to a verdict
            assert c.backend._turn is VoiceState.SPEAKING
            await c.wait_state(VoiceState.IDLE)          # the reply plays out in full
            assert c.texts() == ["tell me about tokyo"]  # the leak published nothing

    _run(_case())
