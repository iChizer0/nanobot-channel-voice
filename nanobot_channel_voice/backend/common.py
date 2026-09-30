"""One copy of the emit/turn pair (and small shared helpers) so the two backends
cannot drift."""

from __future__ import annotations

from .base import OnEvent, StateHint, VoiceState

# One tool result's share of a realtime session's context: the whole output enters it, and
# a spoken answer needs a fraction of it (nanobot's own text turns cut at 16 000).
TOOL_OUTPUT_CHARS = 8000


def clamp_tool_output(output: str, limit: int = TOOL_OUTPUT_CHARS) -> str:
    """An oversized tool output cut to ``limit`` chars (0 = unlimited): its head and its
    tail, where a command's errors and exit code are, around a note of the cut."""
    if limit <= 0 or len(output) <= limit:
        return output
    marker = (
        f"\n[... {len(output)} chars in all, the middle cut: a voice session keeps {limit} "
        "of a tool's output; narrow the call for the rest ...]\n"
    )
    if len(marker) >= limit:
        return marker.strip()[:limit]
    keep = limit - len(marker)
    head = keep // 2
    return output[:head] + marker + output[len(output) - (keep - head):]


def loggable_text(text: str, enabled: bool, cap: int = 80) -> str:
    """Transcript text for log lines, honoring ``voice.logTranscripts``; off (the default)
    a word count rides instead — user speech is personal data and gateway logs persist.
    Whitespace collapses to one space: an embedded ``\\n`` would split the log record."""
    words = text.split()
    return " ".join(words)[:cap] if enabled else f"<{len(words)} words>"


class TurnEventMixin:
    """The concrete backend defines ``_on_event`` (set at ``start()``), ``_closing`` and
    ``_turn``. ``_emit`` drops events once closing: a late worker callback (a TTS thread
    finishing after ``close()``) must not reach a stopped shell. ``_set_turn`` is the ONE
    place turn state changes."""

    _on_event: OnEvent | None
    _closing: bool
    _turn: VoiceState

    async def _emit(self, event) -> None:
        if self._on_event is not None and not self._closing:
            await self._on_event(event)

    async def _set_turn(self, state: VoiceState) -> None:
        if state is self._turn:
            return
        self._turn = state
        await self._emit(StateHint(state))
