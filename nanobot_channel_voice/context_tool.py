"""Runtime-context delivery for local-mode voice turns, via the tool-provider seam.

The blocks must NOT ride inbound metadata: tools snapshot request metadata verbatim
(cron ``json.dumps``es ``origin_metadata``), so ``RuntimeContextBlock`` objects there
crash ``cron add`` from a voice turn. Official core gives plugin channels no
``AgentLoop``, so ``register_runtime_context_provider`` is unreachable; a tool loaded via
the ``nanobot.tools`` entry point may instead supply a runtime-context provider, which
core resolves for every user turn and injected mid-turn message. So the channel registers
a bridge here and :class:`VoiceContextTool` (a no-op tool otherwise) serves the blocks.
"""

from __future__ import annotations

from collections import OrderedDict
from datetime import datetime
from typing import Any

from loguru import logger
from nanobot.agent.tools.base import Tool
from nanobot.runtime_context import RuntimeContextBlock

from nanobot_channel_voice.streamid import TURN_META

# English weekday names regardless of the process locale (%A localizes).
_WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


def _time_note() -> str:
    """The model's only clock: core injects no date or time anywhere (0.3.0), and
    without one the model invents a placeholder. Computed at resolve time, so a cron turn
    reads the fire-time clock, not its creation-time one."""
    now = datetime.now().astimezone()
    offset = now.strftime("%z")
    return (f"[time now: {now:%Y-%m-%d} ({_WEEKDAYS[now.weekday()]}) {now:%H:%M}, "
            f"UTC{offset[:3]}:{offset[3:]}]")


# Notes are popped at resolve; a turn killed before dispatch never resolves, so the map
# evicts oldest-first.
_MAX_PENDING_NOTES = 32


class VoiceContextBridge:
    """One live voice session's context source: the static contract blocks plus per-turn
    event notes keyed by the publish's turn token."""

    def __init__(self, blocks: list[RuntimeContextBlock]):
        self.blocks = blocks
        self._notes: OrderedDict[str, tuple[str, ...]] = OrderedDict()

    def stash_notes(self, token: str, notes: tuple[str, ...]) -> None:
        if not notes:
            return
        self._notes[token] = notes
        while len(self._notes) > _MAX_PENDING_NOTES:
            self._notes.popitem(last=False)

    def resolve(self, metadata: dict[str, Any]) -> list[RuntimeContextBlock]:
        """The blocks for one request: contract, then a per-turn block of the fresh time
        stamp plus any notes stashed under the request's turn token. A request without a
        live token (a cron fire) still gets the contract and the stamp."""
        token = metadata.get(TURN_META)
        notes = self._notes.pop(token, ()) if isinstance(token, str) else ()
        turn_block = RuntimeContextBlock(
            source="voice", content="\n".join((_time_note(), *notes))
        )
        return [*self.blocks, turn_block]


_BRIDGES: dict[tuple[str, str], VoiceContextBridge] = {}
_TOOL_CREATED = False


def register_bridge(
    channel: str, chat_id: str, blocks: list[RuntimeContextBlock]
) -> VoiceContextBridge:
    bridge = VoiceContextBridge(blocks)
    if _BRIDGES.get((channel, chat_id)) is not None:
        # Normal on a channel restart; anything else is two instances fighting.
        logger.info("voice context bridge replaced for {}:{}", channel, chat_id)
    _BRIDGES[(channel, chat_id)] = bridge
    return bridge


def unregister_bridge(channel: str, chat_id: str, bridge: VoiceContextBridge) -> None:
    """Remove *bridge* only if it is still the registered one: a restarted channel's
    fresh bridge must survive the old instance's late stop."""
    if _BRIDGES.get((channel, chat_id)) is bridge:
        del _BRIDGES[(channel, chat_id)]


def tool_created() -> bool:
    """Did core's tool loader instantiate :class:`VoiceContextTool`? False means the
    ``nanobot.tools`` entry point is invisible and no context reaches the model."""
    return _TOOL_CREATED


async def _provide(request: Any) -> list[RuntimeContextBlock] | None:
    """The provider core resolves per request. Never raises: an exception here aborts
    the whole turn."""
    try:
        key = (getattr(request, "channel", None), getattr(request, "chat_id", None))
        bridge = _BRIDGES.get(key)  # type: ignore[arg-type]
        if bridge is None:
            return None
        return bridge.resolve(getattr(request, "metadata", None) or {})
    except Exception:
        logger.exception("voice context provider failed; turn continues without the block")
        return None


class VoiceContextTool(Tool):
    """Entry-point carrier for the provider. Must stay registered (and so visible in the
    model's tool list) for the provider to resolve; ``enabled`` cannot gate on the channel
    because tools load once at loop startup while channels hot-enable later."""

    @property
    def name(self) -> str:
        return "voice_context"

    @property
    def description(self) -> str:
        return (
            "Internal bridge that supplies the voice channel's runtime context. "
            "It performs no action; there is never a reason to call it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    @classmethod
    def create(cls, ctx: Any) -> "VoiceContextTool":
        global _TOOL_CREATED
        _TOOL_CREATED = True
        return cls()

    async def execute(self, **kwargs: Any) -> str:
        # No session count: the tool is visible to EVERY channel's sessions.
        return "This tool only supplies runtime context and performs no action."

    def runtime_context_provider(self):
        return _provide
