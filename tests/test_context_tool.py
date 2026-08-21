"""The context bridge: voice runtime context delivered via core's tool-provider seam.

The load-bearing invariant: inbound metadata stays JSON-plain. RuntimeContextBlock
objects there crashed `cron add` from a voice turn (json.dumps of origin_metadata) and,
reloaded as dicts, crashed the job again at fire time — so the blocks now ride a tool's
runtime_context_provider, and these tests pin both the bridge and the cron round trip.
"""

from __future__ import annotations

import asyncio
import json
import tomllib
from pathlib import Path

import pytest
from nanobot.bus.queue import MessageBus
from nanobot.runtime_context import (
    RuntimeContextBlock,
    runtime_context_blocks_from_metadata,
)

from nanobot_channel_voice import context_tool
from nanobot_channel_voice.channel import VoiceChannel
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.context_tool import (
    VoiceContextBridge,
    VoiceContextTool,
    register_bridge,
    unregister_bridge,
)
from nanobot_channel_voice.streamid import TURN_META

_BLOCK = RuntimeContextBlock(source="voice", content="[Voice channel]\nx\n[/Voice channel]")


@pytest.fixture()
def clean_registry():
    """Isolate the module-global bridge registry per test."""
    saved = dict(context_tool._BRIDGES)
    context_tool._BRIDGES.clear()
    try:
        yield context_tool._BRIDGES
    finally:
        context_tool._BRIDGES.clear()
        context_tool._BRIDGES.update(saved)


class _Request:
    def __init__(self, channel="voice", chat_id="voice:local", metadata=None):
        self.channel = channel
        self.chat_id = chat_id
        self.metadata = metadata or {}


# ---- the provider -----------------------------------------------------------


def test_provider_serves_only_the_registered_session(clean_registry):
    bridge = register_bridge("voice", "voice:local", [_BLOCK])
    provider = VoiceContextTool().runtime_context_provider()
    blocks = asyncio.run(provider(_Request()))
    assert blocks[0] == _BLOCK
    assert blocks[-1].content.startswith("[time now: ")
    # Another channel/chat — a WebUI turn, a subagent's system request — gets nothing.
    assert asyncio.run(provider(_Request(channel="websocket"))) is None
    assert asyncio.run(provider(_Request(chat_id="other"))) is None
    unregister_bridge("voice", "voice:local", bridge)
    assert asyncio.run(provider(_Request())) is None


def test_provider_never_raises(clean_registry):
    # An exception here would abort the whole agent turn; a missing block must not.
    bridge = register_bridge("voice", "voice:local", [_BLOCK])
    bridge.resolve = None  # type: ignore[assignment] - simulate any internal fault
    provider = VoiceContextTool().runtime_context_provider()
    assert asyncio.run(provider(_Request())) is None
    assert asyncio.run(provider(object())) is None  # request without the attributes


def test_unregister_is_identity_checked(clean_registry):
    old = register_bridge("voice", "voice:local", [_BLOCK])
    fresh = register_bridge("voice", "voice:local", [_BLOCK])  # channel restart
    unregister_bridge("voice", "voice:local", old)  # the old instance's late stop()
    provider = VoiceContextTool().runtime_context_provider()
    assert asyncio.run(provider(_Request()))[0] == _BLOCK  # fresh bridge survives
    unregister_bridge("voice", "voice:local", fresh)


# ---- the bridge -------------------------------------------------------------


def test_notes_pop_once_and_the_map_stays_bounded():
    bridge = VoiceContextBridge([_BLOCK])
    bridge.stash_notes("t1", ("[voice event: a]",))
    bridge.stash_notes("t1", ())  # empty stash is a no-op, not an overwrite
    blocks = bridge.resolve({TURN_META: "t1"})
    assert blocks[-1].content.endswith("[voice event: a]")
    assert "\n" not in bridge.resolve({TURN_META: "t1"})[-1].content  # popped
    for i in range(context_tool._MAX_PENDING_NOTES + 5):
        bridge.stash_notes(f"n{i}", ("[voice event: x]",))
    assert len(bridge._notes) == context_tool._MAX_PENDING_NOTES
    assert "n0" not in bridge._notes  # oldest evicted first


def test_resolve_without_a_token_still_serves_contract_and_clock():
    # A cron fire echoes its CREATION metadata: dead token or no token at all.
    bridge = VoiceContextBridge([_BLOCK])
    for metadata in ({}, {TURN_META: "long-dead"}, {"_cron_trigger": {"job_id": "x"}}):
        blocks = bridge.resolve(metadata)
        assert blocks[0] == _BLOCK
        assert blocks[-1].content.startswith("[time now: ")


# ---- the cron round trip (the original bug, pinned) -------------------------


def test_voice_publish_metadata_survives_cron_capture(tmp_path):
    """End-to-end regression: a job created with a voice publish's metadata as
    origin_metadata must persist (gateway mode json.dumps), reload, and pass the
    fire-time normalize — the three steps that failed with blocks in metadata."""
    from nanobot.cron.service import CronService
    from nanobot.cron.types import CronSchedule

    channel = VoiceChannel(VoiceConfig(), MessageBus())
    channel._context_bridge = VoiceContextBridge([_BLOCK])
    seen: list[dict] = []

    async def _capture(**kwargs):
        seen.append(kwargs)

    channel._handle_message = _capture  # type: ignore[method-assign]
    asyncio.run(
        channel._publish_turn_text("remind me to stretch", "turn-1", ("[voice event: a]",))
    )
    [call] = seen
    captured = dict(call["metadata"])
    captured["_wants_stream"] = True  # BaseChannel stamps it on streaming channels

    svc = CronService(store_path=tmp_path / "jobs.json")
    svc._load_store()
    svc._running = True  # gateway mode: add_job persists via _save_store/json.dumps
    svc._arm_timer = lambda: None  # persistence is under test, not the event-loop timer
    job = svc.add_job(
        name="stretch",
        schedule=CronSchedule(kind="at", at_ms=32503680000000),
        message="remind me to stretch",
        session_key="voice:voice:local",
        origin_channel="voice",
        origin_chat_id="voice:local",
        origin_metadata=captured,
    )
    reloaded = CronService(store_path=tmp_path / "jobs.json")
    reloaded._load_store()
    [stored] = [j for j in reloaded.list_jobs() if j.id == job.id]
    om = stored.payload.origin_metadata
    assert om[TURN_META] == "turn-1"
    # Fire time: core reads context blocks off the echoed metadata; with no blocks
    # key there this returns [] instead of raising on reloaded dicts.
    assert runtime_context_blocks_from_metadata(om) == []


# ---- core loader integration ------------------------------------------------


def test_core_loader_registers_the_tool_and_forwards_the_provider(clean_registry):
    from nanobot.agent.tools.loader import ToolLoader
    from nanobot.agent.tools.registry import ToolRegistry

    loader = ToolLoader()
    loader._discovered = []  # built-ins aren't under test (their enabled() needs a ctx)
    loader._plugins = {"voice_context": VoiceContextTool}
    registry = ToolRegistry()
    created_before = context_tool._TOOL_CREATED
    context_tool._TOOL_CREATED = False
    try:
        assert loader.load(None, registry, scope="core") == ["voice_context"]
        assert context_tool.tool_created()  # the channel's health log keys off this
        [provider] = registry.get_runtime_context_providers()  # through the wrapper
        register_bridge("voice", "voice:local", [_BLOCK])
        blocks = asyncio.run(provider(_Request()))
        assert blocks[0] == _BLOCK
    finally:
        context_tool._TOOL_CREATED = created_before


def test_tool_is_a_schema_valid_no_op():
    tool = VoiceContextTool()
    schema = tool.to_schema()
    assert schema["function"]["name"] == "voice_context"
    assert schema["function"]["parameters"] == {
        "type": "object", "properties": {}, "required": [],
    }
    json.dumps(schema)
    out = asyncio.run(tool.execute())
    assert "no action" in out  # calling it must stay harmless


def test_entry_point_is_declared_in_pyproject():
    # Real discovery needs an installed dist; the wiring itself is pinned here.
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    assert data["project"]["entry-points"]["nanobot.tools"]["voice_context"] == (
        "nanobot_channel_voice.context_tool:VoiceContextTool"
    )
