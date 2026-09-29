"""Supervisor delegation: the collector's terminals, token identity on the bus glue, and
the handler's stop / replace / sweep / queueing paths."""

from __future__ import annotations

import asyncio
import json

import pytest

from nanobot_channel_voice.backend.base import AbandonedResult, DelegatedResult, ReceiptResult
from nanobot_channel_voice.channel import _DELEGATION_META, VoiceChannel, _ReplyCollector
from nanobot_channel_voice.metrics import VoiceMetrics


def run(coro):
    return asyncio.run(coro)


def test_streaming_terminal_joins_deltas():
    async def _case():
        c = _ReplyCollector(VoiceMetrics())
        c.add("Hello ")
        c.add("world")
        c.finish()
        assert await c.result() == "Hello world"

    run(_case())


def test_finish_falls_back_to_end_frame_content():
    async def _case():
        c = _ReplyCollector(VoiceMetrics())
        c.finish(fallback="only the end frame had text")
        assert await c.result() == "only the end frame had text"

    run(_case())


def test_first_terminal_wins():
    async def _case():
        c = _ReplyCollector(VoiceMetrics())
        c.set_final("first")
        c.finish(fallback="second")
        c.set_final("third")
        assert await c.result() == "first"

    run(_case())


def test_abandon_resolves_without_a_first_token_sample():
    async def _case():
        m = VoiceMetrics()
        c = _ReplyCollector(m)
        c.abandon("(interrupted)")
        assert await c.result() == "(interrupted)"
        # A delta landing in the tick before the slot clears is not an answer's timing.
        c.add("late delta")
        assert "delegation_first_token_ms" not in m.snapshot()["latency_ms"]

    run(_case())


def test_blank_retry_end_keeps_collecting_past_the_status_line():
    """Core's callback order for status line + tool call, blank continuation (it fires
    on_stream_end(resuming=False) and RETRIES), then the answer: the blank end must not
    resolve, or the status line is read aloud as the answer and the real one is lost."""
    async def _case():
        c = _ReplyCollector(VoiceMetrics())
        c.add("Let me check.")
        c.note_boundary()       # end(resuming=True): the tool runs
        c.finish()              # end(resuming=False) with nothing streamed: the blank retry
        assert not c._future.done()
        c.add("The answer is 42.")
        c.finish()
        assert await c.result() == "Let me check.\nThe answer is 42."

    run(_case())


def test_regular_final_joins_behind_what_streamed():
    """A last segment that streamed nothing (blank retries exhausted) ends in core's
    regular final send; what the earlier segments streamed still belongs to the reply."""
    async def _case():
        c = _ReplyCollector(VoiceMetrics())
        c.add("Checking.")
        c.note_boundary()
        c.finish()
        c.set_final("I could not produce a response.")
        assert await c.result() == "Checking.\nI could not produce a response."
        alone = _ReplyCollector(VoiceMetrics())
        alone.set_final("whole reply")  # streaming off: nothing streamed
        assert await alone.result() == "whole reply"

    run(_case())


def _supervisor_channel():
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice.config import VoiceConfig

    cfg = VoiceConfig.model_validate(
        {"backend": "openai", "realtime": {"toolMode": "supervisor", "apiKey": "k",
                                           "delegationTimeoutS": 5}}
    )
    channel = VoiceChannel(cfg, MessageBus(), tool_gateway=object())
    published: list[tuple[str, dict | None]] = []
    stops: list[bool] = []

    async def publish(text, metadata=None):
        published.append((text, metadata))

    async def stop():
        stops.append(True)

    channel._publish_user_text = publish  # type: ignore[method-assign]
    channel._publish_stop = stop  # type: ignore[method-assign]
    return channel, published, stops


def _args(request) -> str:
    return json.dumps({"request": request})


async def _answer(channel: VoiceChannel, text: str) -> None:
    token = channel._pending_delegation.token
    await channel.send_delta(channel.config.chat_id, text, {_DELEGATION_META: token},
                             stream_id="s:1:0")
    await channel.send_delta(channel.config.chat_id, "", {_DELEGATION_META: token},
                             stream_id="s:1:0", stream_end=True)


class _Announcer:
    """A cloud backend: records what the model is asked to voice."""

    def __init__(self) -> None:
        self.said: list[str] = []

    async def announce(self, text: str) -> None:
        self.said.append(text)


_CRON = {"_cron_trigger": {"job_id": "j1", "run_id": "r1"}}


async def _stream(channel, text, meta, *, sid="voice:local:1:0", end=False, resuming=False):
    await channel.send_delta(
        channel.config.chat_id, text, meta, stream_id=sid, stream_end=end, resuming=resuming,
    )


def test_stream_identity_is_the_token_on_every_delta():
    """Core echoes the inbound metadata onto every delta and end, so the delegation token
    is the identity: a /stop-ped predecessor's straggler carries the old token, and a cron
    turn's reply streaming meanwhile is voiced, never collected."""
    async def _case():
        channel, _, _ = _supervisor_channel()
        backend = channel._backend = _Announcer()
        current = channel._pending_delegation = _ReplyCollector(VoiceMetrics())
        stale = _ReplyCollector(VoiceMetrics())
        await _stream(channel, "old answer", {_DELEGATION_META: stale.token})
        await _stream(channel, "", {_DELEGATION_META: stale.token}, end=True)
        await _stream(channel, "reminder text", _CRON, sid="voice:local:2:0")
        await _stream(channel, "", _CRON, sid="voice:local:2:0", end=True)
        assert not current._future.done()
        await _stream(channel, "real ", {_DELEGATION_META: current.token})
        await _stream(channel, "answer", {_DELEGATION_META: current.token})
        await _stream(channel, "", {_DELEGATION_META: current.token}, end=True)
        assert await current.result() == "real answer"
        assert backend.said == ["reminder text"]

    run(_case())


def test_cloud_voices_what_the_agent_sends_on_its_own():
    """A message another channel sends here, or a heartbeat report, is voiced; the /stop
    ack, an already-streamed final, a finished delegation's straggler and another chat's
    delivery are not."""
    from nanobot.bus.events import OutboundMessage

    async def _case():
        channel, _, _ = _supervisor_channel()
        backend = channel._backend = _Announcer()

        def msg(text, chat=None, **meta):
            return OutboundMessage(channel="voice", chat_id=chat or channel.config.chat_id,
                                   content=text, metadata=meta)

        await channel.send(msg("Dinner is ready."))
        await channel.send(msg("Stopped 1 task(s).", _voice_cmd=True))
        await channel.send(msg("streamed already", _streamed=True))
        await channel.send(msg("late answer", **{_DELEGATION_META: "old-token"}))
        await channel.send(msg("elsewhere", chat="voice:other"))
        assert backend.said == ["Dinner is ready."]

    run(_case())


def test_a_provider_without_text_input_logs_an_agent_message_instead():
    """Qwen-Omni documents no user text item: sent anyway, the notice's response.create would
    answer nothing new, so the message is logged, not voiced."""
    from nanobot.bus.events import OutboundMessage

    from nanobot_channel_voice.audio.null import NullPlayback
    from nanobot_channel_voice.backend.audio_sink import AudioSink
    from nanobot_channel_voice.backend.openai_realtime import RealtimeBackend
    from nanobot_channel_voice.backend.profiles import PROFILES

    async def _case():
        channel, _, _ = _supervisor_channel()
        sink = AudioSink(NullPlayback(), mode="stream")
        announced: list[str] = []

        async def announce(text: str) -> None:
            announced.append(text)

        for key in ("qwen", "glm"):
            backend = RealtimeBackend(channel.config, sink=sink, profile=PROFILES[key])
            backend.announce = announce  # type: ignore[method-assign]
            channel._backend = backend
            await channel.send(OutboundMessage(
                channel="voice", chat_id=channel.config.chat_id, content=f"From {key}.",
            ))
        assert announced == ["From glm."]
        assert channel._metrics.snapshot()["counters"]["notice_unvoiced"] == 1

    run(_case())


def test_a_streamed_agent_turn_is_voiced_whole():
    async def _case():
        channel, _, _ = _supervisor_channel()
        backend = channel._backend = _Announcer()
        await _stream(channel, "Checking.", _CRON)
        await _stream(channel, "", _CRON, end=True, resuming=True)  # a tool boundary
        assert backend.said == []
        await _stream(channel, "Oven ", _CRON, sid="voice:local:1:1")
        await _stream(channel, "time.", _CRON, sid="voice:local:1:1")
        await _stream(channel, "", _CRON, sid="voice:local:1:1", end=True)
        assert backend.said == ["Checking.\nOven time."]
        assert "delegation_first_token_ms" not in channel._metrics.snapshot()["latency_ms"]

    run(_case())


def test_a_reply_cut_at_the_token_limit_is_collected_as_one_message():
    """Core ends a segment cut at the token limit with merge_next and streams the rest of
    the sentence as the next one: a delegation's reply and a notice read on unbroken."""
    async def _case():
        channel, _, _ = _supervisor_channel()
        backend = channel._backend = _Announcer()
        pending = channel._pending_delegation = _ReplyCollector(VoiceMetrics())
        chat, meta = channel.config.chat_id, {_DELEGATION_META: pending.token}
        await channel.send_delta(chat, "The capital of Fra", meta, stream_id="s:1:0")
        await channel.send_delta(chat, "", meta, stream_id="s:1:0", stream_end=True,
                                 resuming=True, merge_next=True)
        await channel.send_delta(chat, "nce is Paris.", meta, stream_id="s:1:1")
        await channel.send_delta(chat, "", meta, stream_id="s:1:1", stream_end=True)
        assert await pending.result() == "The capital of France is Paris."
        await _stream(channel, "Time to stre", _CRON)
        await channel.send_delta(chat, "", _CRON, stream_id="voice:local:1:0",
                                 stream_end=True, resuming=True, merge_next=True)
        assert backend.said == []
        await _stream(channel, "tch.", _CRON, sid="voice:local:1:1")
        await _stream(channel, "", _CRON, sid="voice:local:1:1", end=True)
        assert backend.said == ["Time to stretch."]

    run(_case())


def test_an_agent_turn_whose_last_segment_streamed_nothing_ends_in_its_final():
    from nanobot.bus.events import OutboundMessage

    async def _case():
        channel, _, _ = _supervisor_channel()
        backend = channel._backend = _Announcer()
        await _stream(channel, "Checking.", _CRON)
        await _stream(channel, "", _CRON, end=True, resuming=True)
        await _stream(channel, "", _CRON, sid="voice:local:1:1", end=True)  # a blank retry
        assert backend.said == []
        await channel.send(OutboundMessage(
            channel="voice", chat_id=channel.config.chat_id, content="Done.", metadata=_CRON,
        ))
        assert backend.said == ["Checking.\nDone."]

    run(_case())


def test_a_trigger_stamped_with_the_pending_token_is_not_the_delegations_reply():
    """A cron job snapshots the turn that created it, so a reminder the running delegation
    scheduled carries its token: the reminder is voiced, the delegation keeps waiting."""
    from nanobot.bus.events import OutboundMessage

    async def _case():
        channel, _, _ = _supervisor_channel()
        backend = channel._backend = _Announcer()
        pending = channel._pending_delegation = _ReplyCollector(VoiceMetrics())
        await channel.send(OutboundMessage(
            channel="voice", chat_id=channel.config.chat_id, content="Take a break.",
            metadata={_DELEGATION_META: pending.token, **_CRON},
        ))
        assert not pending._future.done()
        assert backend.said == ["Take a break."]

    run(_case())


def test_a_new_agent_stream_drops_one_that_never_ended_and_a_lone_end_frame_counts():
    async def _case():
        channel, _, _ = _supervisor_channel()
        backend = channel._backend = _Announcer()
        await _stream(channel, "cut off mid", _CRON, sid="voice:local:1:0")
        await _stream(channel, "", _CRON, sid="voice:local:1:0", end=True, resuming=True)
        await _stream(channel, "One frame.", _CRON, sid="voice:local:2:0", end=True)
        assert backend.said == ["One frame."]

    run(_case())


def _ask(channel: VoiceChannel, request: str, turn: str = "r1"):
    return asyncio.create_task(channel._delegate_to_nanobot("ask_nanobot", _args(request), turn))


def test_a_stop_stops_the_live_delegation_and_moots_the_queued_one():
    """Two ask_nanobot calls of one turn: the second queues on the lock. A consumed stop ends
    the work they serve, so the live one is /stop-ped and the queued one must not spend a
    whole nanobot turn on an answer nobody wants. A later call runs normally."""
    async def _case():
        channel, published, stops = _supervisor_channel()
        first = _ask(channel, "one")
        await asyncio.sleep(0)
        second = _ask(channel, "two")
        await asyncio.sleep(0.01)
        assert [t for t, _ in published] == ["one"]
        await channel._on_cloud_abandon()
        assert await first == "(stopped by the user)"
        assert await second == "(stopped by the user)"
        assert isinstance(await first, AbandonedResult)  # the backend resumes nothing
        assert [t for t, _ in published] == ["one"]  # "two" never ran
        assert stops == [True]
        assert channel._metrics.snapshot()["counters"]["delegation_stopped"] == 2
        third = _ask(channel, "three", "r2")
        await asyncio.sleep(0.01)
        await _answer(channel, "3")
        assert await third == "3"
        assert channel._pending_delegation is None

    run(_case())


def test_cancel_nanobot_stops_the_running_delegation_and_the_queued_one():
    """Where no transcript consumes a stop (Gemini, no input transcription), the model's
    cancel_nanobot does what a consumed stop does; its own answer resumes nothing."""
    async def _case():
        channel, published, stops = _supervisor_channel()
        first = _ask(channel, "one")
        await asyncio.sleep(0)
        second = _ask(channel, "two")
        await asyncio.sleep(0.01)
        out = await channel._supervisor_tool("cancel_nanobot", "{}", "r2")
        assert isinstance(out, ReceiptResult)
        assert await first == "(stopped by the user)"
        assert await second == "(stopped by the user)"
        assert [t for t, _ in published] == ["one"] and stops == [True]
        idle = await channel._supervisor_tool("cancel_nanobot", "{}", "r3")
        assert isinstance(idle, ReceiptResult) and stops == [True]  # nothing to stop
        third = asyncio.create_task(channel._supervisor_tool("ask_nanobot", _args("3"), "r4"))
        await asyncio.sleep(0.01)
        await _answer(channel, "three")
        assert await third == "three"
        assert isinstance(await third, DelegatedResult)  # the answer Gemini voices at once

    run(_case())


@pytest.mark.parametrize("cancel_first", [False, True])
def test_a_cancel_called_with_a_new_request_ends_only_the_older_work(cancel_first):
    """"Check London instead, forget Paris": one model turn may call cancel_nanobot and a new
    ask_nanobot, in either order. The cancel ends the older request, never its turn's own."""
    async def _case():
        channel, published, stops = _supervisor_channel()
        paris = _ask(channel, "Paris")
        await asyncio.sleep(0.01)
        calls = [("ask_nanobot", _args("London")), ("cancel_nanobot", "{}")]
        if cancel_first:
            calls.reverse()
        tasks = {name: asyncio.create_task(channel._supervisor_tool(name, args, "r2"))
                 for name, args in calls}
        await asyncio.sleep(0.01)
        assert isinstance(await paris, AbandonedResult)
        assert isinstance(await tasks["cancel_nanobot"], ReceiptResult)
        assert [t for t, _ in published] == ["Paris", "London"] and stops == [True]
        await _answer(channel, "Rain in London.")
        assert await tasks["ask_nanobot"] == "Rain in London."
        # the replaced request's answer stays abandoned through the handler
        assert not isinstance(await paris, DelegatedResult)

    run(_case())


def test_a_cancel_that_names_no_turn_stops_everything():
    """A call without a model turn cannot tell its own turn's request apart: the cancel then
    stops all pending work, as a consumed stop does."""
    async def _case():
        channel, _, stops = _supervisor_channel()
        first = _ask(channel, "one", "")
        await asyncio.sleep(0.01)
        await channel._supervisor_tool("cancel_nanobot", "{}", "")
        assert await first == "(stopped by the user)" and stops == [True]

    run(_case())


def test_a_later_turn_replaces_the_running_delegation_and_the_queued_one():
    """The user was heard again and the model asked anew: that request replaces everything
    asked before it, running (/stop-ped) or queued (never run)."""
    async def _case():
        channel, published, stops = _supervisor_channel()
        first = _ask(channel, "one")
        await asyncio.sleep(0)
        second = _ask(channel, "two")
        await asyncio.sleep(0.01)
        third = _ask(channel, "three", "r2")
        await asyncio.sleep(0.01)
        assert await first == "(replaced by a newer request)"
        assert await second == "(replaced by a newer request)"
        assert [t for t, _ in published] == ["one", "three"]
        assert stops == [True]
        await _answer(channel, "3")
        assert await third == "3"
        assert channel._metrics.snapshot()["counters"]["delegation_replaced"] == 2

    run(_case())


def test_calls_of_one_turn_queue_and_each_is_answered():
    async def _case():
        channel, published, stops = _supervisor_channel()
        first = _ask(channel, "one")
        await asyncio.sleep(0)
        second = _ask(channel, "two")
        await asyncio.sleep(0.01)
        await _answer(channel, "1")
        assert await first == "1"
        await asyncio.sleep(0.01)
        assert [t for t, _ in published] == ["one", "two"]
        await _answer(channel, "2")
        assert await second == "2"
        assert stops == []

    run(_case())


async def _answer_once_published(channel, published, count: int, text: str) -> None:
    while len(published) < count:
        await asyncio.sleep(0.001)
    await _answer(channel, text)


def test_an_answered_delegation_is_neither_replaced_nor_stopped():
    """The answer resolved but its handler has not returned yet: a stop or a later turn's
    call in that tick must not /stop the nanobot turn that just finished."""
    async def _case():
        channel, published, stops = _supervisor_channel()
        first = _ask(channel, "one")
        await asyncio.sleep(0.01)
        channel._pending_delegation.set_final("1")
        await channel._on_cloud_abandon()
        answering = asyncio.create_task(_answer_once_published(channel, published, 2, "3"))
        third = await channel._delegate_to_nanobot("ask_nanobot", _args("three"), "r2")
        assert await first == "1" and third == "3"
        assert stops == []
        await answering

    run(_case())


def test_a_swept_delegation_stops_its_turn():
    """The shell cancels tool tasks at teardown, or when the provider withdraws the call:
    nobody will hear the answer, so the nanobot turn is stopped like a timed-out one."""
    async def _case():
        channel, published, stops = _supervisor_channel()
        task = _ask(channel, "q")
        await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        assert stops == [True]
        assert channel._pending_delegation is None

    run(_case())


def test_non_string_arguments_are_taken_as_text():
    async def _case():
        channel, published, stops = _supervisor_channel()
        task = asyncio.create_task(channel._delegate_to_nanobot(
            "ask_nanobot", json.dumps({"request": {"city": "Oslo"}, "relevant_context": 7}),
            "r1",
        ))
        await asyncio.sleep(0.01)
        assert published[0][0] == "{'city': 'Oslo'}\n\n[context from the conversation: 7]"
        await _answer(channel, "ok")
        assert await task == "ok"

    run(_case())


def test_late_reply_from_stopped_delegation_cannot_resolve_next():
    """A /stop-ped delegation's late final (streaming OFF) must not resolve the NEXT one:
    only its exact token does. A stale one is dropped, and an unstamped delivery (a cron
    fire, another channel's send) is the agent's own message: voiced, never collected."""
    from nanobot.bus.events import OutboundMessage

    async def _case():
        channel, _, _ = _supervisor_channel()
        backend = channel._backend = _Announcer()
        stale = _ReplyCollector(VoiceMetrics())
        current = channel._pending_delegation = _ReplyCollector(VoiceMetrics())

        def reply(text: str, **meta) -> OutboundMessage:
            return OutboundMessage(
                channel="voice", chat_id=channel.config.chat_id, content=text, metadata=meta
            )

        await channel.send(reply("old answer", **{_DELEGATION_META: stale.token}))
        assert not current._future.done()
        await channel.send(reply("real answer", **{_DELEGATION_META: current.token}))
        assert await current.result() == "real answer"
        tokenless = channel._pending_delegation = _ReplyCollector(VoiceMetrics())
        await channel.send(reply("your 3pm reminder"))
        assert not tokenless._future.done()
        assert backend.said == ["your 3pm reminder"]

    run(_case())


def test_foreign_chat_delivery_is_neither_spoken_nor_collected():
    """One speaker, one chat: the message tool takes an arbitrary channel/chat, so a
    delivery addressed elsewhere must not be spoken, resolve a delegation, or touch the
    live turn's deadman/ledger."""
    from nanobot.bus.events import OutboundMessage
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice.config import VoiceConfig

    async def _case():
        channel = VoiceChannel(VoiceConfig(), MessageBus())
        spoken: list[str] = []
        deltas: list[str] = []
        touched: list[str] = []

        class _Local:
            def note_agent_activity(self): touched.append("deadman")
            def note_proactive(self): touched.append("proactive")
            def is_dead_turn(self, token): return False
            async def speak_final(self, text): spoken.append(text)
            async def announce(self, text): spoken.append(text)
            async def on_delta(self, delta, stream_id=None): deltas.append(delta)
            async def on_stream_end(self, *, resuming, stream_id=None): touched.append("end")

        channel._local = lambda: _Local()  # type: ignore[method-assign]
        foreign = "voice:somewhere-else"
        assert foreign != channel.config.chat_id
        await channel.send(OutboundMessage(
            channel="voice", chat_id=foreign, content="Your bank code is 4711.",
        ))
        await channel.send_delta(foreign, "secret ", None, stream_id="voice:x:1:0")
        await channel.send_delta(foreign, "", None, stream_id="voice:x:1:0", stream_end=True)
        assert (spoken, deltas, touched) == ([], [], [])

        # A delegation in flight must not collect it either.
        collector = _ReplyCollector(VoiceMetrics())
        channel._pending_delegation = collector
        await channel.send(OutboundMessage(
            channel="voice", chat_id=foreign, content="not our answer",
        ))
        assert not collector._future.done()

        # The session's own chat, stamped with the live token, still resolves.
        await channel.send(OutboundMessage(
            channel="voice", chat_id=channel.config.chat_id, content="ours",
            metadata={_DELEGATION_META: collector.token},
        ))
        assert await collector.result() == "ours"

        # ...and with no delegation pending it is spoken.
        channel._pending_delegation = None
        await channel.send(OutboundMessage(
            channel="voice", chat_id=channel.config.chat_id, content="said aloud",
        ))
        assert spoken == ["said aloud"]

    run(_case())


def test_first_token_recorded_once():
    async def _case():
        m = VoiceMetrics()
        c = _ReplyCollector(m)
        c.add("a")
        c.add("b")
        c.finish()
        await c.result()
        assert m.snapshot()["latency_ms"]["delegation_first_token_ms"]["n"] == 1

    run(_case())


def test_tool_boundary_does_not_latch_first_token():
    async def _t():
        m = VoiceMetrics()
        c = _ReplyCollector(m)
        # A tool-first delegation: the boundary arrives before any model token.
        c.note_boundary()
        assert "delegation_first_token_ms" not in m.snapshot()["latency_ms"]
        c.add("the answer")  # the REAL first token latches
        c.finish()
        lat = m.snapshot()["latency_ms"]
        assert lat["delegation_first_token_ms"]["n"] == 1
        assert (await c.result()).strip() == "the answer"

    asyncio.run(_t())


def test_supervisor_mode_needs_no_tool_gateway():
    """ask_nanobot runs a whole turn over the bus, so a core that passes no tool gateway
    (every official one) still delegates; only direct mode, which executes each tool
    through the gateway, falls back to persona-only, and says so."""
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice.config import VoiceConfig

    class _Log:
        def __init__(self):
            self.infos: list[str] = []
            self.warned: list[str] = []

        def info(self, msg, *a): self.infos.append(msg.format(*a))
        def warning(self, msg, *a): self.warned.append(msg.format(*a))

    def channel(realtime: dict) -> VoiceChannel:
        cfg = VoiceConfig.model_validate(
            {"backend": "openai", "realtime": {"apiKey": "k", **realtime}}
        )
        ch = VoiceChannel(cfg, MessageBus())
        assert ch._tool_gateway is None  # nothing in official core supplies one
        ch.logger = _Log()  # type: ignore[assignment]
        return ch

    async def _case():
        sup = channel({"toolMode": "supervisor"})
        tools, exec_tool = await sup._cloud_tools(True, "supervisor")
        assert [t.name for t in tools] == ["ask_nanobot", "cancel_nanobot"]
        assert exec_tool == sup._supervisor_tool  # a fresh bound method each access
        assert sup.logger.warned == []

        # Direct mode the user SET: persona-only, loudly, pointing at supervisor.
        direct = channel({"toolMode": "direct"})
        assert await direct._cloud_tools(True, "direct") == ([], None)
        [warning] = direct.logger.warned
        assert "persona-only" in warning and "toolMode='supervisor'" in warning

        # The DEFAULT toolMode was never asked for: every cloud start must not warn.
        quiet = channel({})
        assert await quiet._cloud_tools(True, "direct") == ([], None)
        assert quiet.logger.warned == [] and len(quiet.logger.infos) == 1

        # A model that cannot drive tools stays persona-only in either mode.
        assert await channel({})._cloud_tools(False, "supervisor") == ([], None)

    run(_case())


class _ReadGateway:
    """A tool gateway holding reads and writes; records what it executes."""

    _MY = {"type": "object", "required": ["action"], "properties": {
        "action": {"type": "string", "enum": ["check", "set"]},
        "key": {"type": "string"}, "value": {"description": "New value (for set)."}}}

    def __init__(self) -> None:
        self.ran: list[tuple] = []

    async def get_tool_definitions(self):
        names = ("read_file", "write_file", "exec", "list_dir", "find_files", "my",
                 "list_exec_sessions", "web_search")
        return [{"type": "function", "function": {
            "name": name, "description": f"core {name}",
            "parameters": self._MY if name == "my" else {"type": "object", "properties": {}},
        }} for name in names]

    async def execute_tool(self, name, args, *, channel, chat_id):
        self.ran.append((name, args, channel, chat_id))
        return f"{name} ran"

    async def get_agent_context(self, *, channel, chat_id, include_skills=True):
        return f"agent context (skills={include_skills})"


def test_supervisor_mode_reads_for_itself_through_the_gateway():
    """With a gateway, supervisor mode also holds the agent's read-only tools for checking
    on nanobot's work, run directly: no write, no shell, no web lookup, and ``my`` as its
    check alone. Everything else is still a delegation."""
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice.config import VoiceConfig

    async def _case():
        cfg = VoiceConfig.model_validate(
            {"backend": "openai", "realtime": {"toolMode": "supervisor", "apiKey": "k"}}
        )
        gw = _ReadGateway()
        ch = VoiceChannel(cfg, MessageBus(), tool_gateway=gw)
        tools, exec_tool = await ch._cloud_tools(True, "supervisor")
        assert [t.name for t in tools] == [
            "ask_nanobot", "cancel_nanobot", "read_file", "list_dir", "find_files", "my",
            "list_exec_sessions",
        ]
        my = tools[5]
        assert my.parameters["properties"]["action"]["enum"] == ["check"]
        assert "value" not in my.parameters["properties"]
        assert "set" not in my.parameters["properties"]["key"].get("description", "")
        assert my.parameters["required"] == ["action"] and "read-only" in my.description
        assert gw._MY["properties"]["action"]["enum"] == ["check", "set"]  # a copy narrowed

        assert await exec_tool("read_file", '{"path": "notes.md"}', "r1") == "read_file ran"
        assert gw.ran == [("read_file", '{"path": "notes.md"}', "voice", cfg.chat_id)]
        refused = await exec_tool("my", '{"action": "set", "key": "model", "value": "x"}', "r1")
        assert refused.startswith("Error:") and refused.is_error and len(gw.ran) == 1
        assert await exec_tool("my", '{"action": "check", "key": "subagents"}', "r1") == "my ran"
        assert gw.ran[-1][1] == '{"action": "check", "key": "subagents"}'
        assert await exec_tool("list_dir", '{"path": "."}', "r1") == "list_dir ran"

    run(_case())


@pytest.mark.parametrize("args, runs", [
    # core unwraps a lone "arguments" field, as an object or as JSON text
    ('{"arguments": {"action": "set", "key": "model_preset", "value": "x"}}', None),
    ('{"arguments": "{\\"action\\": \\"set\\", \\"key\\": \\"model\\"}"}', None),
    ('{"arguments": {"action": "check", "key": "subagents"}}',
     '{"action": "check", "key": "subagents"}'),
    # whatever else rides along, only a check is sent: a second wrapper, a stray value
    ('{"arguments": {"arguments": {"action": "set", "key": "model"}}}', '{"action": "check"}'),
    ('{"action": "check", "key": "model", "value": "x"}', '{"action": "check", "key": "model"}'),
    ('{"action": "inspect"}', '{"action": "check"}'),
    ("not json", '{"action": "check"}'),
])
def test_supervisor_my_is_rebuilt_as_a_check(args, runs):
    """``my`` is built, not filtered: core accepts a call wrapped in one ``arguments`` field,
    so reading the top level alone would let a set through where tools.my.allowSet is on."""
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice.config import VoiceConfig

    async def _case():
        cfg = VoiceConfig.model_validate(
            {"backend": "openai", "realtime": {"toolMode": "supervisor", "apiKey": "k"}}
        )
        gw = _ReadGateway()
        ch = VoiceChannel(cfg, MessageBus(), tool_gateway=gw)
        out = await ch._supervisor_tool("my", args, "r1")
        if runs is None:
            assert out.is_error and gw.ran == []
        else:
            assert [(name, sent) for name, sent, *_ in gw.ran] == [("my", runs)]

    run(_case())


@pytest.mark.parametrize("name", ["write_file", "exec", "apply_patch"])
def test_a_supervisor_never_runs_a_tool_it_does_not_hold(name):
    """A name outside the reads is a delegation, never a gateway call: a hallucinated write
    or shell carries no request, so it is asked again rather than run."""
    async def _case():
        channel, published, _ = _supervisor_channel()
        gw = channel._tool_gateway = _ReadGateway()
        out = await channel._supervisor_tool(name, '{"path": "x", "content": "y"}', "r1")
        assert out.startswith("I didn't catch") and gw.ran == [] and published == []

    run(_case())


def test_supervisor_instructions_name_the_reads_only_when_it_holds_them():
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice.channel import _INSPECT_RULE, _SUPERVISOR_RULES
    from nanobot_channel_voice.config import VoiceConfig
    from nanobot_channel_voice.history import SpokenHistory

    async def _case():
        cfg = VoiceConfig.model_validate(
            {"backend": "openai", "realtime": {"toolMode": "supervisor", "apiKey": "k"}}
        )
        ch = VoiceChannel(cfg, MessageBus(), tool_gateway=_ReadGateway())
        text = await ch._instructions_source(True, True, SpokenHistory(), inspects=True)(True)
        assert f"{_SUPERVISOR_RULES} {_INSPECT_RULE}" in text
        assert "skills=False" in text  # a delegating model reads no skill itself
        plain = await ch._instructions_source(True, True, SpokenHistory())(True)
        assert _SUPERVISOR_RULES in plain and _INSPECT_RULE not in plain

    run(_case())


def test_a_supervisor_whose_gateway_cannot_list_tools_still_delegates():
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice.config import VoiceConfig

    class _Broken(_ReadGateway):
        async def get_tool_definitions(self):
            raise RuntimeError("mcp connect failed")

    async def _case():
        cfg = VoiceConfig.model_validate(
            {"backend": "openai", "realtime": {"toolMode": "supervisor", "apiKey": "k"}}
        )
        ch = VoiceChannel(cfg, MessageBus(), tool_gateway=_Broken())
        tools, _ = await ch._cloud_tools(True, "supervisor")
        assert [t.name for t in tools] == ["ask_nanobot", "cancel_nanobot"]

    run(_case())


def test_direct_mode_leaves_out_the_context_bridge_tool():
    """voice_context carries local turns' runtime context and does nothing when called;
    a realtime model gets nanobot's context in its instructions instead."""
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice.config import VoiceConfig

    class _Gateway:
        async def get_tool_definitions(self):
            return [
                {"type": "function", "function": {"name": name, "parameters": {}}}
                for name in ("read_file", "voice_context", "web_search")
            ]

    async def _case():
        cfg = VoiceConfig.model_validate({"backend": "openai", "realtime": {"apiKey": "k"}})
        ch = VoiceChannel(cfg, MessageBus(), tool_gateway=_Gateway())
        tools, _ = await ch._cloud_tools(True, "direct")
        assert [t.name for t in tools] == ["read_file", "web_search"]

    run(_case())


def test_supervisor_mode_warns_under_a_unified_session(monkeypatch):
    """A delegated request is a bus turn too: in one shared session it can be folded into
    another channel's turn, whose reply the delegation never collects. Direct mode runs
    tools alone, no turns, so it stays quiet."""
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice import channel as channel_mod
    from nanobot_channel_voice.config import VoiceConfig

    monkeypatch.setattr(channel_mod, "unified_session", lambda: True)

    class _Gateway:
        async def get_tool_definitions(self):
            return []

    async def _case():
        cfg = VoiceConfig.model_validate(
            {"backend": "openai", "realtime": {"toolMode": "supervisor", "apiKey": "k"}}
        )
        channel = VoiceChannel(cfg, MessageBus(), tool_gateway=_Gateway())
        warned: list[str] = []

        class _Log:
            def info(self, msg, *a): pass
            def warning(self, msg, *a): warned.append(msg.format(*a))

        channel.logger = _Log()  # type: ignore[assignment]
        await channel._cloud_tools(True, "supervisor")
        assert len(warned) == 1 and "unifiedSession" in warned[0]
        warned.clear()
        await channel._cloud_tools(True, "direct")
        assert warned == []

    run(_case())
