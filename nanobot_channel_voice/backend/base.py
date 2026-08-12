"""VoiceBackend contract: the seam between the audio shell and a reasoning brain.

* ``on_event`` is dispatched ON THE SINGLE EVENT LOOP; a backend using worker
  threads / a WS rx task must marshal onto it (``loop.call_soon_threadsafe`` /
  ``asyncio.Queue``) first. State is mutated only on the loop, lock-free.
* COARSE STATE IS AUTHORITATIVE VIA ``StateHint``: the shell's ``VoiceState``
  (mic-gating, drain guards, the "state x -> y" log) is driven by nothing else,
  and never inferred from observational transcript events.
* BARGE-IN OWNERSHIP IS BY EVENT MEMBERSHIP, NOT A FLAG. A backend owning its
  barge-in (local) drives ``sink.flush()`` / interrupt INTERNALLY, emits no
  ``UserSpeechStarted``, and reflects the transitions via ``StateHint``. One
  wanting the shell's generic barge-in (cloud) emits ``UserSpeechStarted``: the
  only event that makes the shell flush, capture played-ms and call ``barge_in``.
* AUDIO CARRIES ITS OWN EPOCH: the sink gates on the epoch stamped into
  ``OutputAudio``, so the guard survives any scheduling before enqueue.

Event membership is OPTIONAL: a backend emits only the subset it has; the only
hard ordering guarantees are on ``TurnDone`` and ``ToolCall``.
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
    """A PROVIDER-NEUTRAL tool declaration, lifted out of nanobot's nested
    ``Tool.to_schema()``. Each backend owns its wire serialization (OpenAI: flat
    ``{type:"function", name, ...}``; Gemini: a ``functionDeclarations`` entry)."""

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
    """Ready-to-play audio, stamped with the sink epoch it was produced under.
    Exactly one payload is set, on the axis of SINK MODE not backend: ``wav`` is a
    blob-mode TTS WAV played byte-for-byte; ``pcm`` is stream-mode raw S16_LE mono
    @ ``rate``, written to a persistent stream for gapless output."""

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
    """Recognized USER speech. OBSERVATIONAL (logging only). ``cloud`` only, and
    only when ``realtime.inputTranscriptionModel`` is set; ``local`` logs its
    transcript internally."""

    text: str


@dataclass(frozen=True, slots=True)
class UserSpeechStarted:
    """CLOUD ONLY. Server-VAD onset; the shell's generic barge-in trigger (flush
    the ``AudioSink``, then ``backend.barge_in(played_ms)``)."""


@dataclass(frozen=True, slots=True)
class ToolStarted:
    """CLOUD ONLY. A real ``function_call`` item began. Advisory (logging /
    preamble UX), though no ``TurnDone`` arrives until the turn's tool
    obligations resolve. ``call_id`` matches the later :class:`ToolCall` but is
    optional: a provider may announce an item before it has an id. ``local``
    never emits it: its internal stream-end resume also fires on non-tool
    recoveries, so it is not a tool boundary."""

    name: str | None = None
    call_id: str | None = None


@dataclass(frozen=True, slots=True)
class ToolCall:
    """CLOUD ONLY. The brain requests the shell EXECUTE a tool and return the
    result via ``backend.submit_tool_result(call_id, output)`` (``call_id`` echoed
    verbatim). ``epoch`` is the sink epoch at dispatch, for staleness METRICS
    only: the backend's cancelled-response bookkeeping is what keeps a result
    completing after a barge-in from resurrecting the turn."""

    call_id: str
    name: str
    arguments: str  # JSON string; passed straight to ToolRegistry.execute (it coerces)
    epoch: int = 0


@dataclass(frozen=True, slots=True)
class TurnDone:
    """The assistant turn is complete. Informational: the shell only logs it;
    each backend owns its own drain-to-IDLE. Exactly once per completed turn:
    CLOUD suppresses it while any ``submit_tool_result`` obligation is
    outstanding (a tool turn spans >= 2 provider responses)."""


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

    async def start(
        self, *, instructions: str | None, tools: list[ToolDef], on_event: OnEvent
    ) -> None:
        """Begin a session and spawn the backend's tasks. ``local`` ignores
        ``instructions`` / ``tools`` (nanobot owns persona + tools); ``cloud`` sends
        ``session.update``, then runs a single rx loop."""

    async def push_audio(self, pcm: bytes) -> None:
        """Feed one captured frame (S16_LE mono @ the shell's capture rate).
        ``local``: ``Endpointer.push``. ``cloud``: append when session-ready."""

    async def barge_in(self, played_ms: int) -> None:
        """CLOUD: stop the in-flight response after the shell flushed the sink,
        per ``profile.interrupt``. GA sends ``conversation.item.truncate`` to
        align model memory with what the user heard; server-VAD auto-cancel
        (``interrupt_response``, default on) handles the response itself, and a
        second ``response.cancel`` would race — sent only when
        ``realtime.interruptResponse`` is off. Beta dialects have neither
        truncate nor auto-cancel: explicit ``response.cancel`` only. ``local``:
        no-op (never called)."""

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        """CLOUD: ``conversation.item.create(function_call_output)`` now; then
        ``response.create`` at most once, after the triggering response is done
        AND all its tool results are in, unless it was cancelled or the dialect
        auto-continues. ``local``: no-op."""

    async def on_capture_gap(self) -> None:
        """The shell's capture stream broke and is being restarted. ``local``:
        drop the open utterance (its frame-counted endpointer clock stopped with
        the frames: left open it would bridge the outage and merge sentences).
        ``cloud``: no-op (server-side VAD owns segmentation)."""

    async def close(self) -> None:
        """Tear down tasks / connections. Idempotent. Sets a closing flag checked
        before any reconnect and before every ``on_event`` dispatch."""
