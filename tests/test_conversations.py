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
            assert c.texts()[1] == "turn off the desk lamp"  # pure speech, no note in-text
            [note] = c.notes()[1]                        # the marker rides BESIDE the text
            assert note.startswith("[voice event: you were interrupted mid-reply")
            assert "The weather" in note                 # heard-up-to names real words
            assert c.backend._turn is VoiceState.THINKING

    _run(_case())


def test_wait_phrase_and_markdown_probes():
    from nanobot_channel_voice.backend.local import (
        _MD_CARRY,
        _MD_PROBE,
        _opens_with_wait_phrase,
    )

    assert _opens_with_wait_phrase('"One moment." said the bot')
    assert _opens_with_wait_phrase("稍等，我看看天气")
    assert not _opens_with_wait_phrase("It is sunny today")
    assert not _opens_with_wait_phrase("Moments later it rained")  # prefix, not substring
    assert _opens_with_wait_phrase("\r\nOne moment. sunny")  # speak_final gets raw text
    assert _MD_PROBE.search("plain prose, no formatting at all") is None
    assert _MD_PROBE.search("a **bold** claim")
    assert _MD_PROBE.search("see [the docs](http://example.com)")
    assert _MD_PROBE.search("```py\nprint(1)\n```")
    assert _MD_PROBE.search("steps:\n- one\n- two")
    # A marker counts only after a real newline: these are mid-line delta fragments of
    # "nine - ten" and "3 * 4", not lists.
    assert _MD_PROBE.search("- ten, and that holds") is None
    assert _MD_PROBE.search("* 4 equals twelve") is None
    # Every separator the chunker's _line_start_at accepts, and _MD_CARRY wide enough that
    # no marker is lost wherever the delta seam falls inside the longest one.
    for sep in ("\n", "\r\n", "\r"):
        assert _MD_PROBE.search(f"steps:{sep}- one")
        longest = f"{sep}   ###### "
        assert all(
            _MD_PROBE.search(("x" * 20 + longest[:i])[-_MD_CARRY:] + longest[i:])
            for i in range(1, len(longest))
        )


def test_reply_contract_counters_markdown_and_wait_phrase():
    """The observability pair: turns whose reply carried markdown, and answer-bearing
    segments that opened on a wait-phrase. Wait-phrases are judged only where they are
    false — a status line before a tool call is allowed."""
    async def _case():
        async with EvalConversation() as c:
            await c.user_says("format something for me")
            await c.agent_replies("Here is **bold** and a [link](http://x). More **bold**.")
            await c.wait_state(VoiceState.IDLE)
            assert c.counter("reply_markdown") == 1  # per-turn latch, not per-hit

            # No tools at all: a wait-phrase opening the delivery is the classic
            # filler-plus-instant-answer failure.
            await c.user_says("quick question")
            await c.agent_replies("One moment. It is sunny today.")
            await c.wait_state(VoiceState.IDLE)
            assert c.counter("reply_wait_phrase") == 1

            # Tool chain: wait-phrases in the status slots (resuming segments) are
            # allowed; only the final segment is the delivery.
            b = c.backend
            await c.user_says("check the weather")
            await b.on_delta("Let me check the weather.")
            await b.on_stream_end(resuming=True)
            await b.on_delta("One moment.")               # status for the SECOND tool
            await b.on_stream_end(resuming=True)
            await b.on_delta("It is sunny in Tokyo today, all day long.")
            await b.on_stream_end(resuming=False)
            await c.wait_state(VoiceState.IDLE)
            assert c.counter("reply_wait_phrase") == 1    # unchanged: no false stall
            assert c.counter("agent_prologue") == 2       # both status lines spoke

            # Same chain, but the delivery itself opens on the stall.
            await c.user_says("and tomorrow")
            await b.on_delta("Let me check again.")
            await b.on_stream_end(resuming=True)
            await b.on_delta("One moment. Tomorrow will be cloudy all day long.")
            await b.on_stream_end(resuming=False)
            await c.wait_state(VoiceState.IDLE)
            assert c.counter("reply_wait_phrase") == 2
            assert c.counter("reply_markdown") == 1       # prose turns added nothing

            # Streamed the way a provider splits prose: the dash fragment lands at a
            # delta start, where re.M's ^ read it as a list.
            await c.user_says("what is nine minus one")
            for delta in ("It is", " eight", " - ", "give or take", " - all day."):
                await b.on_delta(delta)
            await b.on_stream_end(resuming=False)
            await c.wait_state(VoiceState.IDLE)
            assert c.counter("reply_markdown") == 1

            # A real marker split across the seam still counts (the carried tail).
            await c.user_says("list them")
            for delta in ("Steps:", "\n", "- one, then two, then three."):
                await b.on_delta(delta)
            await b.on_stream_end(resuming=False)
            await c.wait_state(VoiceState.IDLE)
            assert c.counter("reply_markdown") == 2

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
            assert c.texts()[1] == "what about osaka"
            [note] = c.notes()[1]
            assert "stopped your previous reply" in note
            assert 'heard up to: "The weather' in note

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
