"""_DelegationCollector: terminal-signal races, tombstones, straggler watermark."""

from __future__ import annotations

import asyncio
import time

from nanobot_channel_voice.channel import _DelegationCollector
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


def test_abandon_resolves_and_marks_dead():
    async def _case():
        m = VoiceMetrics()
        c = _DelegationCollector(m)
        c.abandon("(interrupted)")
        assert c.dead is True
        assert await c.result() == "(interrupted)"
        # A dead collector must not record first-token timing for a turn that
        # never produced an answer.
        c.add("late delta")
        assert "delegation_first_token_ms" not in m.snapshot()["latency_ms"]

    run(_case())


def test_entomb_latches_without_resolving():
    async def _case():
        m = VoiceMetrics()
        c = _DelegationCollector(m)
        c.entomb()
        assert c.dead is True
        c.add("straggler")  # swallowed by the tombstone, no timing sample
        assert "delegation_first_token_ms" not in m.snapshot()["latency_ms"]
        c.finish()  # a late terminal resolves it (nobody is waiting)
        assert await c.result() == "straggler"

    run(_case())


def test_accepts_stream_watermark_rejects_prior_turns():
    async def _case():
        c = _DelegationCollector(VoiceMetrics())
        now_ns = time.time_ns()
        # Stream base embeds the turn's start time_ns; only turns that started
        # AFTER the collector existed can be answering this delegation.
        assert c.accepts_stream(f"voice:local:{now_ns - 10**9}:0") is False
        assert c.accepts_stream(f"voice:local:{now_ns + 10**9}:0") is True
        assert c.accepts_stream(f"voice:local:{now_ns + 10**9}") is True  # no segment part
        # Unrecognized formats keep the old accept-everything behavior.
        assert c.accepts_stream(None) is True
        assert c.accepts_stream("no-timestamps-here") is True
        assert c.accepts_stream("sess:123:0") is True  # digits too short for time_ns

    run(_case())


def test_cloud_barge_in_ignores_tombstone():
    """A dead tombstone left registered after a timeout must NOT make later
    barge-ins publish /stop or count delegation_interrupted (regression: the
    guard only checked `is None`, so one timeout turned every subsequent
    barge-in into a spurious bus /stop until the next delegation)."""
    from nanobot_channel_voice.channel import VoiceChannel

    async def _case():
        m = VoiceMetrics()
        c = _DelegationCollector(m)
        c.entomb()  # timed-out delegation left as tombstone

        stops: list[bool] = []

        class _Stub:
            _pending_delegation = c
            _metrics = m

            async def _publish_stop(self):
                stops.append(True)

        await VoiceChannel._on_cloud_barge_in(_Stub())
        assert stops == []
        assert m.snapshot()["counters"].get("delegation_interrupted", 0) == 0
        # A live collector still triggers the full path.
        live = _DelegationCollector(m)
        _Stub._pending_delegation = live
        await VoiceChannel._on_cloud_barge_in(_Stub())
        assert stops == [True]
        assert live.dead is True

    run(_case())


def test_late_reply_from_stopped_delegation_cannot_resolve_next():
    """Regression: with bus streaming OFF, a /stop-ped delegation's turn can
    finish late and its bare final send used to resolve the NEXT delegation
    with the previous question's answer. The request carries a token the
    AgentLoop echoes onto its final, so only an exact match resolves: an
    unstamped delivery into this chat is another turn's, never our answer."""
    from nanobot.bus.events import OutboundMessage

    from nanobot_channel_voice.channel import _DELEGATION_META, VoiceChannel

    async def _case():
        m = VoiceMetrics()
        stale = _DelegationCollector(m)
        stale.entomb()  # delegation A timed out; its turn is still running
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

        # A's late final lands after B replaced the tombstone: must be swallowed.
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

    from nanobot_channel_voice.channel import VoiceChannel
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
        from nanobot_channel_voice.channel import _DELEGATION_META

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

    from nanobot_channel_voice.channel import VoiceChannel
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
