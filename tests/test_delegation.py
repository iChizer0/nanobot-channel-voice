"""Supervisor delegation: the collector's terminals, token identity on the bus glue, and
the handler's barge-in / sweep / queueing paths."""

from __future__ import annotations

import asyncio
import json

from nanobot_channel_voice.channel import _DELEGATION_META, VoiceChannel, _DelegationCollector
from nanobot_channel_voice.metrics import VoiceMetrics


def run(coro):
    return asyncio.run(coro)


def test_streaming_terminal_joins_deltas():
    async def _case():
        c = _DelegationCollector(VoiceMetrics())
        c.add("Hello ")
        c.add("world")
        c.finish()
        assert await c.result() == "Hello world"

    run(_case())


def test_finish_falls_back_to_end_frame_content():
    async def _case():
        c = _DelegationCollector(VoiceMetrics())
        c.finish(fallback="only the end frame had text")
        assert await c.result() == "only the end frame had text"

    run(_case())


def test_first_terminal_wins():
    async def _case():
        c = _DelegationCollector(VoiceMetrics())
        c.set_final("first")
        c.finish(fallback="second")
        c.set_final("third")
        assert await c.result() == "first"

    run(_case())


def test_abandon_resolves_without_a_first_token_sample():
    async def _case():
        m = VoiceMetrics()
        c = _DelegationCollector(m)
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
        c = _DelegationCollector(VoiceMetrics())
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
        c = _DelegationCollector(VoiceMetrics())
        c.add("Checking.")
        c.note_boundary()
        c.finish()
        c.set_final("I could not produce a response.")
        assert await c.result() == "Checking.\nI could not produce a response."
        alone = _DelegationCollector(VoiceMetrics())
        alone.set_final("whole reply")  # streaming off: nothing streamed
        assert await alone.result() == "whole reply"

    run(_case())


def test_stream_identity_is_the_token_on_every_delta():
    """Core echoes the inbound metadata onto every delta and end, so the delegation token
    is the identity: a /stop-ped predecessor's straggler carries the old token, a cron
    turn none, whatever their stream ids look like."""
    async def _case():
        m = VoiceMetrics()
        current = _DelegationCollector(m)
        stale = _DelegationCollector(m)

        class _Cfg:
            chat_id = "voice"

        class _Stub:
            config = _Cfg()
            _pending_delegation = current

        async def delta(text, meta, *, end=False, resuming=False):
            await VoiceChannel.send_delta(
                _Stub(), "voice", text, meta, stream_id="whatever:format",
                stream_end=end, resuming=resuming,
            )

        await delta("old answer", {_DELEGATION_META: stale.token})
        await delta("", {_DELEGATION_META: stale.token}, end=True)
        await delta("reminder text", None)
        assert not current._future.done()
        await delta("real ", {_DELEGATION_META: current.token})
        await delta("answer", {_DELEGATION_META: current.token})
        await delta("", {_DELEGATION_META: current.token}, end=True)
        assert await current.result() == "real answer"

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


def test_barge_in_stops_the_live_delegation_and_moots_the_queued_one():
    """Two ask_nanobot calls in one response: the second queues on the lock. A barge-in
    cancels the response that asked, so the queued call must not spend a whole nanobot
    turn on an answer the backend would drop as stale."""
    async def _case():
        channel, published, stops = _supervisor_channel()
        first = asyncio.create_task(channel._delegate_to_nanobot("ask_nanobot", _args("one")))
        await asyncio.sleep(0)
        second = asyncio.create_task(channel._delegate_to_nanobot("ask_nanobot", _args("two")))
        await asyncio.sleep(0.01)
        assert [t for t, _ in published] == ["one"]
        await channel._on_cloud_barge_in()
        assert await first == "(interrupted by the user)"
        assert await second == "(interrupted by the user)"
        assert [t for t, _ in published] == ["one"]  # "two" never ran
        assert stops == [True]
        assert channel._metrics.snapshot()["counters"]["delegation_interrupted"] == 2
        # A call made AFTER the barge-in runs normally.
        third = asyncio.create_task(channel._delegate_to_nanobot("ask_nanobot", _args("three")))
        await asyncio.sleep(0.01)
        await _answer(channel, "3")
        assert await third == "3"
        assert channel._pending_delegation is None

    run(_case())


def test_a_swept_delegation_stops_its_turn():
    """The shell cancels tool tasks at teardown: nobody will hear the answer, so the
    nanobot turn is stopped like a timed-out one instead of burning tokens."""
    async def _case():
        channel, published, stops = _supervisor_channel()
        task = asyncio.create_task(channel._delegate_to_nanobot("ask_nanobot", _args("q")))
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
        ))
        await asyncio.sleep(0.01)
        assert published[0][0] == "{'city': 'Oslo'}\n\n[context from the conversation: 7]"
        await _answer(channel, "ok")
        assert await task == "ok"

    run(_case())


def test_late_reply_from_stopped_delegation_cannot_resolve_next():
    """Regression: with bus streaming OFF, a /stop-ped delegation's turn can
    finish late and its bare final send used to resolve the NEXT delegation
    with the previous question's answer. The request carries a token the
    AgentLoop echoes onto its final, so only an exact match resolves: an
    unstamped delivery into this chat is another turn's, never our answer."""
    from nanobot.bus.events import OutboundMessage

    async def _case():
        m = VoiceMetrics()
        stale = _DelegationCollector(m)  # delegation A timed out; its turn is still running
        current = _DelegationCollector(m)  # delegation B, awaiting its answer

        class _Cfg:
            chat_id = "voice"

        class _Stub:
            config = _Cfg()
            _pending_delegation = current
            logger = __import__("loguru").logger

        def reply(text: str, **meta) -> OutboundMessage:
            return OutboundMessage(
                channel="voice", chat_id="voice", content=text, metadata=meta
            )

        # A's late final lands while B waits: must be swallowed.
        await VoiceChannel.send(_Stub(), reply("old answer", **{_DELEGATION_META: stale.token}))
        assert not current._future.done()
        await VoiceChannel.send(_Stub(), reply("real answer", **{_DELEGATION_META: current.token}))
        assert await current.result() == "real answer"

        # An unstamped delivery into the same chat (a cron fire, a message-tool send)
        # is somebody else's turn: it must not be read aloud as the delegated answer.
        tokenless = _DelegationCollector(m)
        _Stub._pending_delegation = tokenless
        await VoiceChannel.send(_Stub(), reply("your 3pm reminder"))
        assert not tokenless._future.done()

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
        collector = _DelegationCollector(VoiceMetrics())
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
        c = _DelegationCollector(m)
        c.add("a")
        c.add("b")
        c.finish()
        await c.result()
        assert m.snapshot()["latency_ms"]["delegation_first_token_ms"]["n"] == 1

    run(_case())


def test_tool_boundary_does_not_latch_first_token():
    async def _t():
        m = VoiceMetrics()
        c = _DelegationCollector(m)
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
        assert [t.name for t in tools] == ["ask_nanobot"]
        assert exec_tool == channel._delegate_to_nanobot  # a fresh bound method each access
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
