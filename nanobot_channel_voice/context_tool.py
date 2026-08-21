"""Runtime-context delivery for local-mode voice turns, via the tool-provider seam.

The blocks must NOT ride inbound metadata: tools snapshot request metadata verbatim
(cron persists ``origin_metadata`` with plain ``json.dumps``), so ``RuntimeContextBlock``
objects there crash ``cron add`` from a voice turn — and, reloaded as dicts, crash the
job again at fire time. Official core hands plugin channels nothing but the bus (no
``AgentLoop``, so ``register_runtime_context_provider`` is unreachable), but tools loaded
through the ``nanobot.tools`` entry point may supply a runtime-context provider that core
resolves for every user turn and every injected mid-turn message. So the channel registers
a bridge here, and :class:`VoiceContextTool` (a no-op tool otherwise) serves the blocks —
including for a cron turn firing into the voice session, whose reply will be spoken.
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
    """The model's only clock: core injects no date or time anywhere (helpers'
    ``current_time_str`` is dead code as of 0.3.0), and without one the model answers
    a date/time question with an invented placeholder ("[Current Date and Time]").
    Computed at resolve time, so a cron turn firing hours after its creation reads
    the fire-time clock, never the creation-time one."""
    now = datetime.now().astimezone()
    offset = now.strftime("%z")
    return (f"[time now: {now:%Y-%m-%d} ({_WEEKDAYS[now.weekday()]}) {now:%H:%M}, "
            f"UTC{offset[:3]}:{offset[3:]}]")


# A stashed turn's notes are popped at resolve; turns killed before core dispatches
# them never resolve, so the map is bounded by evicting oldest-first well past any
# plausible in-flight count.
_MAX_PENDING_NOTES = 32


class VoiceContextBridge:
    """One live voice session's context source: the static contract blocks plus
    per-turn event notes keyed by the publish's turn token."""

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
        """The blocks for one request: contract, then a per-turn block of the fresh
        time stamp plus any notes stashed under the request's turn token. A request
        without a live token (a cron fire echoing its creation metadata) still gets
        the contract and the stamp — its reply is spoken like any other."""
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
    _BRIDGES[(channel, chat_id)] = bridge
    return bridge


def unregister_bridge(channel: str, chat_id: str, bridge: VoiceContextBridge) -> None:
    """Remove *bridge* if it is still the registered one (a restarted channel's fresh
    bridge must not be torn down by the old instance's late stop)."""
    if _BRIDGES.get((channel, chat_id)) is bridge:
        del _BRIDGES[(channel, chat_id)]


def tool_created() -> bool:
    """Did core's tool loader instantiate :class:`VoiceContextTool`? False means the
    ``nanobot.tools`` entry point is not visible and no context reaches the model."""
    return _TOOL_CREATED


async def _provide(request: Any) -> list[RuntimeContextBlock] | None:
    """The provider core resolves per request. Never raises: an exception here would
    abort the whole turn, and a turn without its context block beats no turn."""
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
    """Entry-point carrier for the provider. It must stay registered (and so appear in
    the model's tool list) for the provider to resolve; ``enabled`` cannot gate on the
    channel because tools load once at loop startup while channels hot-enable later."""

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
        return (
            f"Voice context bridge: {len(_BRIDGES)} active voice session(s). "
            "This tool only supplies runtime context and performs no action."
        )

    def runtime_context_provider(self):
        return _provide
