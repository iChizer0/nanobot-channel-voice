"""Supervisor delegation: the collector's terminals, token identity on the bus glue, and
the handler's stop / replace / sweep / queueing paths."""

from __future__ import annotations

import asyncio
import json

from nanobot_channel_voice.backend.base import AbandonedResult
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
        assert isinstance(out, AbandonedResult)
        assert await first == "(stopped by the user)"
        assert await second == "(stopped by the user)"
        assert [t for t, _ in published] == ["one"] and stops == [True]
        idle = await channel._supervisor_tool("cancel_nanobot", "{}", "r3")
        assert isinstance(idle, AbandonedResult) and stops == [True]  # nothing to stop
        third = asyncio.create_task(channel._supervisor_tool("ask_nanobot", _args("3"), "r4"))
        await asyncio.sleep(0.01)
        await _answer(channel, "three")
        assert await third == "three"

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


def test_missing_tool_gateway_says_the_tool_mode_is_inert():
    """No shipped core passes a tool gateway to a plugin channel, so a configured
    toolMode silently produced a persona-only session: zero tools, no ask_nanobot, and
    not one log line saying why."""
    from nanobot.bus.queue import MessageBus

    from nanobot_channel_voice.config import VoiceConfig

    async def _case():
        cfg = VoiceConfig.model_validate(
            {"backend": "openai", "realtime": {"toolMode": "supervisor", "apiKey": "k"}}
        )
        channel = VoiceChannel(cfg, MessageBus())
        assert channel._tool_gateway is None  # nothing in core supplies one
        warned: list[str] = []

        infos: list[str] = []

        class _Log:
            def info(self, msg, *a): infos.append(msg.format(*a))
            def warning(self, msg, *a): warned.append(msg.format(*a))

        channel.logger = _Log()  # type: ignore[assignment]
        tools, exec_tool = await channel._cloud_tools(True, "supervisor")
        assert (tools, exec_tool) == ([], None)
        assert len(warned) == 1
        assert "toolMode='supervisor'" in warned[0]
        assert "persona-only" in warned[0]
        # The DEFAULT toolMode was never asked for: every cloud start must not warn.
        quiet = VoiceChannel(
            VoiceConfig.model_validate({"backend": "openai", "realtime": {"apiKey": "k"}}),
            MessageBus(),
        )
        quiet.logger = _Log()  # type: ignore[assignment]
        warned.clear()
        assert await quiet._cloud_tools(True, "direct") == ([], None)
        assert warned == [] and len(infos) == 1

        # With a gateway wired the mode works and stays quiet.
        channel._tool_gateway = object()
        warned.clear()
        tools, exec_tool = await channel._cloud_tools(True, "supervisor")
        assert [t.name for t in tools] == ["ask_nanobot", "cancel_nanobot"]
        assert exec_tool == channel._supervisor_tool  # a fresh bound method each access
        assert warned == []

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
