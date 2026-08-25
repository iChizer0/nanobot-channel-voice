"""Adaptive hangover: learn the user's own pause lengths.

Opt-in via ``vad.hangoverMinMs``: the hangover starts there and grows toward
``vad.hangoverMs`` (the ceiling) only on evidence that the endpointer cut a real pause
short — the user resumed while the reply was still THINKING, or barged in within a
short grace of it turning AUDIBLE. Pauses shorter than the current hangover never
endpoint, so the signal is one-sided: the value tracks cut pauses, not an average.
Samples commit only when the resuming utterance is ACCEPTED: rejected blips/echo must
teach nothing.
"""

from __future__ import annotations

import time

_ALPHA = 0.9          # EMA weight of the old value (LiveKit's default)
_MARGIN = 1.2         # cover the observed pause with headroom, not exactly
_AUDIBLE_GRACE_S = 1.0  # a barge-in this soon after first audio = "still talking"
_WINDOW_FACTOR = 2.0  # pauses beyond factor*ceiling are a new thought, not a pause


class AdaptiveHangover:
    """Feed it turn-lifecycle events; read ``value_ms()`` after each publish."""

    def __init__(self, min_ms: int, max_ms: int):
        self._min = float(min_ms)
        self._max = float(max_ms)
        self._value = float(min_ms)
        self._close_anchor: tuple[float, float] | None = None  # (mono, silence_ms)
        self._pending: float | None = None

    def value_ms(self) -> int:
        return round(self._value)

    def note_close(self, close_mono: float, silence_ms: float) -> None:
        """Anchor the next resume-gap measurement at close time: the fast resumes this
        learner exists to catch arrive before STT finishes."""
        self._close_anchor = (close_mono, silence_ms)

    def drop_anchor(self) -> None:
        """The anchored close was rejected: a gap measured against it would teach from
        audio that was not the user's turn."""
        self._close_anchor = None

    def on_onset(
        self,
        *,
        awaiting_reply: bool,
        speaking: bool,
        audible_at: float | None,
        now: float | None = None,
    ) -> None:
        """Latch a candidate pause when this onset suggests the previous close split the
        user's turn: reply not yet audible, or audible for under the grace. Every onset
        supersedes an unconsumed candidate; commit happens only at publish."""
        self._pending = None  # a candidate not consumed by its own close is dead
        if self._close_anchor is None:
            return
        now = time.monotonic() if now is None else now
        close_mono, silence_ms = self._close_anchor
        pause_ms = silence_ms + (now - close_mono) * 1000.0
        if pause_ms > self._max * _WINDOW_FACTOR:
            return  # too long ago: a new thought, not a cut pause
        if awaiting_reply or (
            speaking and audible_at is not None and (now - audible_at) <= _AUDIBLE_GRACE_S
        ):
            self._pending = pause_ms
        # else: an onset well into the audible reply is a real barge-in, not a cut pause.

    def take_pending(self) -> float | None:
        """Bind the latched candidate to the closing utterance, so an earlier queued
        utterance cannot consume it."""
        pending, self._pending = self._pending, None
        return pending

    def on_publish(self, learn_ms: float | None) -> None:
        """The utterance was ACCEPTED: commit its candidate, if it carried one."""
        if learn_ms is not None:
            sample = min(max(learn_ms * _MARGIN, self._min), self._max)
            self._value = min(max(_ALPHA * self._value + (1.0 - _ALPHA) * sample,
                                  self._min), self._max)
