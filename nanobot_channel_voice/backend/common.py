"""One copy of the emit/turn pair (and small shared helpers) so the two backends
cannot drift."""

from __future__ import annotations

from .base import OnEvent, StateHint, VoiceState


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
