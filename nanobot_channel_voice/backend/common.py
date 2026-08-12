"""One copy of the emit/turn pair so the two backends cannot drift."""

from __future__ import annotations

from .base import OnEvent, StateHint, VoiceState


class TurnEventMixin:
    """The concrete backend defines ``_on_event`` (set at ``start()``), ``_closing``
    and ``_turn``. ``_emit`` drops events once closing: a late worker callback (a
    TTS thread finishing after ``close()``, a final rx frame) must not reach a
    stopped shell. ``_set_turn`` is the ONE place turn state changes."""

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
