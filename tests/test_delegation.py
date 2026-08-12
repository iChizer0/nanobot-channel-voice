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
    with the previous question's answer. The request now carries a token the
    AgentLoop echoes back; a mismatch is swallowed, absence keeps the old
    accept-everything behavior (same philosophy as accepts_stream)."""
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

        def reply(text: str, **meta) -> OutboundMessage:
            return OutboundMessage(
                channel="voice", chat_id="voice", content=text, metadata=meta
            )

        # A's late final lands after B replaced the tombstone: must be swallowed.
        await VoiceChannel.send(_Stub(), reply("old answer", **{_DELEGATION_META: stale.token}))
        assert not current._future.done()
        await VoiceChannel.send(_Stub(), reply("real answer", **{_DELEGATION_META: current.token}))
        assert await current.result() == "real answer"

        tokenless = _DelegationCollector(m)
        _Stub._pending_delegation = tokenless
        await VoiceChannel.send(_Stub(), reply("untagged answer"))
        assert await tokenless.result() == "untagged answer"

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
