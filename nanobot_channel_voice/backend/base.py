"""VoiceBackend contract: the seam between the audio shell and a reasoning brain.

* ``on_event`` is dispatched ON THE SINGLE EVENT LOOP; a backend with worker threads or a
  WS rx task must marshal onto it (``call_soon_threadsafe`` / ``asyncio.Queue``) first.
  State is mutated only on the loop, lock-free.
* ``StateHint`` IS AUTHORITATIVE for the shell's ``VoiceState`` (mic-gating, drain guards,
  the "state x -> y" log); never infer it from observational transcript events.
* BARGE-IN OWNERSHIP IS BY EVENT MEMBERSHIP, NOT A FLAG: a backend owning barge-in (local)
  flushes/interrupts internally, emits no ``UserSpeechStarted`` and reflects transitions
  via ``StateHint``; emitting it (cloud) is the only thing that makes the shell flush,
  capture played-ms and call ``barge_in``.
* AUDIO CARRIES ITS OWN EPOCH: the sink gates on the epoch stamped into ``OutputAudio``,
  so the guard survives any scheduling before enqueue.

Event membership is OPTIONAL; the only hard ordering guarantees are on ``TurnDone`` and
``ToolCall``.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class VoiceState(str, Enum):
    IDLE = "idle"            # listening, mic open
    CAPTURING = "capturing"  # user speaking (utterance not yet accepted)
    THINKING = "thinking"    # user turn accepted, no output audio yet
    SPEAKING = "speaking"    # output flowing


# ---- Tool definition (cloud) ------------------------------------------------

@dataclass(frozen=True, slots=True)
class ToolDef:
    """PROVIDER-NEUTRAL tool declaration lifted out of nanobot's nested
    ``Tool.to_schema()``; each backend owns its own wire serialization."""

    name: str
    description: str
    parameters: dict[str, Any]

    @classmethod
    def from_nanobot_schema(cls, schema: dict[str, Any]) -> "ToolDef":
        fn = schema.get("function", schema)  # tolerate flat or nested
        return cls(
            name=fn["name"],
            description=fn.get("description", ""),
            parameters=fn.get("parameters", {"type": "object", "properties": {}}),
        )


# ---- Normalized events (OPTIONAL membership) --------------------------------

@dataclass(frozen=True, slots=True)
class OutputAudio:
    """Ready-to-play audio stamped with the sink epoch it was produced under. Exactly one
    payload, on the axis of SINK MODE not backend: ``wav`` a blob-mode TTS WAV played
    byte-for-byte, ``pcm`` stream-mode raw S16_LE mono @ ``rate`` for gapless output."""

    epoch: int
    wav: bytes | None = None
    pcm: bytes | None = None
    rate: int = 0


@dataclass(frozen=True, slots=True)
class StateHint:
    """Authoritative coarse-state transition; BOTH backends emit them."""

    state: VoiceState


@dataclass(frozen=True, slots=True)
class OutputTranscript:
    """CLOUD ONLY. Per-token assistant text as committed to speech. OBSERVATIONAL.
    ``local`` feeds its own ``SelfEchoFilter.note_spoken`` at TTS-emit time."""

    text: str


@dataclass(frozen=True, slots=True)
class InputTranscript:
    """Recognized USER speech. OBSERVATIONAL (logging only). ``cloud`` only, and only when
    ``realtime.inputTranscriptionModel`` is set; ``local`` logs its transcript internally."""

    text: str


@dataclass(frozen=True, slots=True)
class UserSpeechStarted:
    """CLOUD ONLY. Server-VAD onset; the shell's generic barge-in trigger."""


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """CLOUD ONLY, advisory: a real ``function_call`` item began. ``call_id`` is optional (a
    provider may announce an item before it has an id). ``local`` never emits it: its
    stream-end resume also fires on non-tool recoveries, so it is not a tool boundary."""

    name: str | None = None
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """CLOUD ONLY. The shell EXECUTEs the tool and returns the result via
    ``backend.submit_tool_result(call_id, output)``, ``call_id`` echoed verbatim. The
    backend's cancelled-response bookkeeping stops a late result reviving a dead turn."""

    call_id: str
    name: str
    arguments: str  # JSON string; passed straight to ToolRegistry.execute (it coerces)


@dataclass(frozen=True, slots=True)
class TurnDone:
    """The assistant turn is complete. Informational: the shell only logs it, each backend
    owns its drain-to-IDLE. Exactly once per completed turn: CLOUD suppresses it while any
    ``submit_tool_result`` obligation is outstanding (a tool turn spans >= 2 responses)."""


@dataclass(frozen=True, slots=True)
class Error:
    """A real fault. NOT emitted for empty STT/TTS (silent no-ops per the adapter
    contract). ``fatal=True`` => the shell stops the session."""

    message: str
    fatal: bool = False


VoiceEvent = (
    OutputAudio | StateHint | OutputTranscript | InputTranscript
    | UserSpeechStarted | ToolStarted | ToolCall | TurnDone | Error
)

OnEvent = Callable[[VoiceEvent], Awaitable[None]]


@runtime_checkable
class VoiceBackend(Protocol):
    # Capture rate is NOT here: the channel derives it from the provider profile /
    # device config when it builds the capture source, before any backend exists.

    # May the shell park this backend's OutputAudio emitter on the sink backlog cap?
    # True for a dedicated synthesis task (local); False when the emitter is the
    # control plane (the realtime rx loop also carries barge-in and tool events).
    pace_output_audio: bool

    async def start(
        self, *, instructions: str | None, tools: list[ToolDef], on_event: OnEvent
    ) -> None:
        """Begin a session and spawn the backend's tasks. ``local`` ignores ``instructions``
        / ``tools`` (nanobot owns persona + tools); ``cloud`` sends ``session.update``, then
        runs a single rx loop."""

    async def push_audio(self, pcm: bytes) -> None:
        """Feed one captured frame (S16_LE mono @ the shell's capture rate).
        ``local``: ``Endpointer.push``. ``cloud``: append when session-ready."""

    async def barge_in(self, played_ms: int) -> None:
        """CLOUD: stop the in-flight response after the shell flushed the sink, per
        ``profile.interrupt`` (truncate-vs-cancel semantics live on
        :data:`profiles.InterruptKind`). ``local``: no-op, never called."""

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        """CLOUD: ``conversation.item.create(function_call_output)`` now; then
        ``response.create`` at most once, after the triggering response is done AND all its
        tool results are in, unless it was cancelled or the dialect auto-continues.
        ``local``: no-op."""

    async def on_capture_gap(self) -> None:
        """The shell's capture stream broke and is restarting. ``local``: drop the open
        utterance — its endpointer clock is frame-counted, so left open it would bridge the
        outage and merge sentences. ``cloud``: no-op (server VAD owns segmentation)."""

    async def close(self) -> None:
        """Tear down tasks / connections. Idempotent. Sets a closing flag checked before any
        reconnect and before every ``on_event`` dispatch."""
