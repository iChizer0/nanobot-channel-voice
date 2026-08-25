"""Scripted end-to-end conversations (the eval scenarios the harness exists for):
published bus turns, barge-in verdict outcomes, marker content, and the
false-candidate economics — through the REAL pipeline, not hand-built pendings."""

from __future__ import annotations

import asyncio
import time

from eval_harness import EvalConversation

from nanobot_channel_voice.backend.base import VoiceState

_REPLY = "The weather in tokyo is sunny today. Tomorrow will be cloudy all day."


def _run(coro):
    return asyncio.run(coro)


async def _until(pred, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() >= deadline:
            raise AssertionError("condition never held")
        await asyncio.sleep(0.005)


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


# ---- silence-proofing (P2): the user must never read silence as "no answer" --


def test_unvoiced_final_speaks_the_fallback_notice():
    """Every chunk of a final can die before the speaker (the speakability guard
    on an English core error over a monolingual voice, a degraded synth): the
    drain then speaks timeoutPhrase instead of settling silently."""
    from nanobot_channel_voice.config import VoiceConfig
    from nanobot_channel_voice.tts.base import TtsAdapter

    fallback = VoiceConfig().timeout_phrase

    class _MuteTts(TtsAdapter):
        output_rate = 16000

        def __init__(self) -> None:
            self.requests: list[str] = []

        async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
            return b""

        async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
            self.requests.append(text)
            # The fallback enqueues the WHOLE phrase (never chunked); "voiced" marks
            # the one ordinary reply this voice can speak.
            if text == fallback or "voiced" in text:
                return b"\x01\x00" * 1600
            return b""  # everything else is unspeakable for this voice

    async def _case():
        async with EvalConversation() as c:
            mute = _MuteTts()
            c.backend._tts = mute
            await c.user_says("do the thing")
            await c.agent_replies("Error calling LLM: timed out after 300s")
            await c.wait_state(VoiceState.IDLE)
            assert c.counter("reply_unvoiced_fallback") == 1
            assert fallback in mute.requests  # the notice was actually synthesized
            # A voiced reply must not trip it (and fallback_done never leaks over).
            await c.user_says("and again")
            await c.agent_replies("This is voiced content.")
            await c.wait_state(VoiceState.IDLE)
            assert c.counter("reply_unvoiced_fallback") == 1

    _run(_case())


# ---- mid-turn steering: a working run must survive being talked to ----------


def test_steer_over_a_status_line_keeps_the_run_working():
    """The reply segment still draining after a `resuming` end is a pre-tool status
    line, so talking over it is steering, not barge-in: the audio stops, the run does
    not, and the utterance rides into the live turn. Killing here is what strands a
    "that failed, let me try another way" chain in permanent silence."""
    async def _case():
        async with EvalConversation(earcons={"captured": True}) as c:
            b = c.backend
            await c.user_says("book me a table at eight")
            await b.on_delta(
                "The booking site rejected that request when I tried it a moment ago. "
                "I am going to try a completely different way of getting that table. "
                "Give me a little while longer and I will tell you how it goes."
            )
            await b.on_stream_end(resuming=True)          # more tools follow
            await c.wait_state(VoiceState.SPEAKING)
            await c.wait_played_ms(100)                   # a few words audibly out
            quiet_since = b._cur_turn.last_activity = time.monotonic() - 100.0
            dings = c.counter("earcon_captured")  # the opening turn dinged too
            await c.user_says("use the italian place on maple street")

            assert c.interrupts == 0                      # NO /stop: the chain lives
            assert c.counter("midturn_injection") == 1
            assert c.counter("midturn_hush") == 1         # but the audio did stop
            assert c.texts()[1] == "use the italian place on maple street"
            [note] = c.notes()[1]
            assert note.startswith("[voice event: the user spoke while you were working")
            assert "The booking site" in note              # heard-up-to names real words
            # Nothing configured to say anything, so the ding is the steer's receipt.
            assert c.counter("midturn_reassure") == 1
            await _until(lambda: c.counter("earcon_captured") == dings + 1)
            await c.wait_state(VoiceState.THINKING)       # mic reopened, tools running
            # User speech is not core liveness: a steer must not hold the deadman off
            # a wedged turn for as long as the user keeps asking.
            assert b._cur_turn.last_activity == quiet_since

            # The continuation still speaks, into the same turn — and a real barge-in
            # over THAT answer still kills, accounting for what the cut already aired.
            await b.on_delta(
                "Booked for eight at the italian place on maple street tonight. "
                "They are holding the table by the window until a quarter past. "
                "I will send the confirmation over to your phone as well."
            )
            await c.wait_state(VoiceState.SPEAKING)
            await c.wait_played_ms(100)
            await c.user_says("cancel it, we will walk there")
            assert c.interrupts == 1
            [note] = c.notes()[2]
            assert note.startswith("[voice event: you were interrupted mid-reply")
            assert "The booking site" in note      # the hushed words are still accounted

    _run(_case())


def test_steer_rung_stops_at_the_answer():
    """The widened rung is bounded by `continuation_pending`: once post-tool text
    starts arriving, what plays IS the answer and talking over it is a real barge-in
    again (kill + /stop + the interrupted-mid-reply marker)."""
    async def _case():
        async with EvalConversation() as c:
            b = c.backend
            await c.user_says("what is the weather")
            await b.on_delta("Let me check.")
            await b.on_stream_end(resuming=True)
            await b.on_delta(_REPLY)                      # the answer: pending cleared
            await c.wait_state(VoiceState.SPEAKING)
            await c.wait_played_ms(100)
            await c.user_says("turn off the desk lamp")

            assert c.interrupts == 1
            assert c.counter("midturn_injection") == 0
            [note] = c.notes()[1]
            assert note.startswith("[voice event: you were interrupted mid-reply")

    _run(_case())


def test_a_terminal_clears_the_steer_latch():
    """`continuation_pending` is a turn latch and the IDLE placeholder is shared, so every
    terminal must clear it: a final arriving as a plain send after a tool boundary (the
    tool_error path does exactly that) would otherwise leave the answer looking steerable,
    and barge-in over it would never /stop."""
    async def _case():
        async with EvalConversation() as c:
            b = c.backend
            await c.user_says("check the weather")
            await b.on_delta("Let me check.")
            await b.on_stream_end(resuming=True)
            await c.agent_replies(_REPLY)          # answer as a non-streamed final
            assert not b._cur_turn.continuation_pending
            await c.wait_state(VoiceState.SPEAKING)
            await c.wait_played_ms(100)
            await c.user_says("turn off the desk lamp")

            assert c.interrupts == 1
            assert c.counter("midturn_injection") == 0

    _run(_case())


def test_a_steer_gets_an_audible_receipt():
    """Core drains injections only at iteration boundaries, so the answer to a steer
    can be a whole tool call away. With a filler script configured, the steer is
    acknowledged out loud instead of leaving the earcon as its only sign."""
    async def _case():
        async with EvalConversation(
            prologue={"enabled": True, "afterMs": 60000}, earcons={"captured": True},
        ) as c:
            b = c.backend
            await c.user_says("check every room")
            await b.on_delta(
                "I am starting the sweep with the first room on the list right now. "
                "There are four more rooms to check after this one is finished. "
                "This will take me a little while, so please bear with me."
            )
            await b.on_stream_end(resuming=True)
            await c.wait_state(VoiceState.SPEAKING)
            await c.wait_played_ms(100)
            dings = c.counter("earcon_captured")
            await c.user_says("skip the garage entirely")

            assert c.counter("midturn_injection") == 1
            assert c.counter("midturn_reassure") == 1
            await c.wait_state(VoiceState.SPEAKING)       # the ack actually plays
            assert c.counter("earcon_captured") == dings  # words replace the ding
            assert c.interrupts == 0

    _run(_case())


def test_quiet_notice_speaks_while_the_core_is_busy():
    """The audible clock, not the core clock: a tool chain pushes last_activity with
    every progress event, so the old single-clock deadman could never speak during
    exactly the dead air the user hears. The notice fires; nothing is killed."""
    from nanobot_channel_voice.config import VoiceConfig

    class _RecordingTts:
        output_rate = 16000

        def __init__(self, inner) -> None:
            self.inner = inner
            self.requests: list[str] = []

        async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
            self.requests.append(text)
            return await self.inner.synthesize(text)

        async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
            self.requests.append(text)
            return await self.inner.synthesize_pcm(text)

    async def _case():
        async with EvalConversation(agentTimeoutS=10.0, stallNoticeS=0.1) as c:
            rec = _RecordingTts(c.backend._tts)
            c.backend._tts = rec
            stall = VoiceConfig().stall_phrase
            await c.user_says("audit the whole house")
            for _ in range(60):
                # What a busy tool chain does to the CORE clock, every 20 ms.
                c.backend.note_agent_activity()
                await asyncio.sleep(0.02)
                if stall in rec.requests:
                    break
            assert stall in rec.requests
            assert c.counter("agent_turn_quiet") >= 1
            assert c.counter("agent_turn_stall") == 0   # the core was never silent
            assert c.interrupts == 0                    # and is never killed for it

    _run(_case())


def test_stall_notice_speaks_and_the_run_survives():
    """First silent budget: the canned stall notice actually synthesizes and plays
    (the deadman unit tests run tts=None and skip it), the run is NOT /stop-ped,
    and a late reply still lands."""
    from nanobot_channel_voice.config import VoiceConfig

    class _RecordingTts:
        output_rate = 16000

        def __init__(self, inner) -> None:
            self.inner = inner
            self.requests: list[str] = []

        async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
            self.requests.append(text)
            return await self.inner.synthesize(text)

        async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
            self.requests.append(text)
            return await self.inner.synthesize_pcm(text)

    async def _case():
        async with EvalConversation(agentTimeoutS=0.15) as c:
            rec = _RecordingTts(c.backend._tts)
            c.backend._tts = rec
            stall = VoiceConfig().stall_phrase
            await c.user_says("do the slow thing")
            for _ in range(150):  # one silent budget, then the notice synthesizes
                await asyncio.sleep(0.01)
                if stall in rec.requests:
                    break
            assert c.counter("agent_turn_stall") == 1
            assert stall in rec.requests
            assert c.interrupts == 0  # the run survived its first stall
            await c.agent_replies("Done at last.")  # arrives inside the second budget
            await c.wait_state(VoiceState.IDLE)
            assert c.interrupts == 0
            assert not c.counter("agent_turn_timeout")

    _run(_case())


def test_streamed_placeholder_delivery_resets_the_audibility_ledger():
    """Unsolicited deliveries share the recycled turn object, and STREAMED ones
    (a cron fire with streaming on) never pass through speak_final's reset: a
    stale emitted_audio latch from an earlier audible delivery must not swallow
    the silence fallback for the next one."""
    from nanobot_channel_voice.config import VoiceConfig
    from nanobot_channel_voice.tts.base import TtsAdapter

    fallback = VoiceConfig().timeout_phrase

    class _MuteTts(TtsAdapter):
        output_rate = 16000

        async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
            return b""

        async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
            if text == fallback or "voiced" in text:
                return b"\x01\x00" * 1600
            return b""  # everything else is unspeakable for this voice

    async def _case():
        async with EvalConversation() as c:
            c.backend._tts = _MuteTts()
            # Delivery 1 streams and is audible: the ledger latches emitted_audio.
            await c.backend.on_delta(
                "This is voiced content.", stream_id="s:1000000000000000000:0"
            )
            await c.backend.on_stream_end(
                resuming=False, stream_id="s:1000000000000000000:0"
            )
            await c.wait_state(VoiceState.IDLE)
            assert c.counter("reply_unvoiced_fallback") == 0
            # Delivery 2 streams entirely unspeakable text: without the reset the
            # stale latch reads as "audio came out" and the user hears nothing.
            await c.backend.on_delta(
                "Unspeakable reminder text.", stream_id="s:1000000000000000001:0"
            )
            await c.backend.on_stream_end(
                resuming=False, stream_id="s:1000000000000000001:0"
            )
            await c.wait_state(VoiceState.IDLE)
            assert c.counter("reply_unvoiced_fallback") == 1

    _run(_case())


def test_dead_stream_residue_never_glues_onto_a_final():
    """core's delivery.fail sends the apology but never closes the stream: the
    chunker still holds the dead segment's partial, which must be discarded, not
    prepended to the apology."""

    class _RecordingTts:
        output_rate = 16000

        def __init__(self, inner) -> None:
            self.inner = inner
            self.requests: list[str] = []

        async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
            self.requests.append(text)
            return await self.inner.synthesize(text)

        async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
            self.requests.append(text)
            return await self.inner.synthesize_pcm(text)

    async def _case():
        async with EvalConversation() as c:
            rec = _RecordingTts(c.backend._tts)
            c.backend._tts = rec
            await c.user_says("question")
            # A short delta stays buffered under the first-chunk floor; the stream
            # then dies with no end marker (the core mid-stream exception gap).
            await c.backend.on_delta("Partial ", stream_id="s:1000000000000000000:0")
            await c.agent_replies("Sorry, I encountered an error.")
            await c.wait_state(VoiceState.IDLE)
            assert rec.requests  # the apology spoke
            assert not any("Partial" in t for t in rec.requests)

    _run(_case())
