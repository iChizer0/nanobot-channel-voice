"""Self-transcription rejection: drop STT that is the bot hearing its own TTS.

With the mic open while the bot speaks, capture contains the bot's own voice;
since the spoken text is known, a transcript whose words are mostly contained
in recently-spoken TTS is that echo, not a barge-in, and is dropped, while
genuinely different speech passes through to cancel-then-send. This makes
open-mic modes usable and hardens hardware AEC against residual echo.
Token-set containment, stdlib only; tuned for word-based languages.
"""

from __future__ import annotations

import time
from collections import deque

from nanobot_channel_voice.phrases import words_of

__all__ = ["SelfEchoFilter", "words_of"]


class SelfEchoFilter:
    def __init__(self, threshold: float = 0.6, window_secs: float = 12.0):
        self._threshold = threshold
        self._window_secs = window_secs
        self._spoken: deque[tuple[float, set[str]]] = deque()

    def note_spoken(self, text: str, hold_ms: float = 0.0) -> None:
        """Record TTS text about to play. ``hold_ms``, the estimated delay until
        it stops sounding (sink backlog + own duration), shifts the stamp so the
        eviction window runs from last-audible, not feed time: an LLM streams a
        30 s reply's text in seconds, and feed-time stamps would evict its tail
        mid-playback, letting the bot barge in on itself."""
        words = words_of(text)
        if words:
            self._spoken.append((time.monotonic() + hold_ms / 1000.0, words))

    def is_self_echo(self, transcript: str) -> bool:
        """True if *transcript* is mostly the bot's recently-spoken words (echo)."""
        self._evict()
        heard = words_of(transcript)
        if not heard or not self._spoken:
            return False
        spoken: set[str] = set().union(*(w for _, w in self._spoken))
        overlap = len(heard & spoken) / len(heard)
        return overlap >= self._threshold

    def fresh_words(self, transcript: str) -> set[str]:
        """Words in *transcript* that are NOT recently-spoken TTS. Callable from
        the frame worker thread while ``note_spoken`` runs on the loop: snapshot
        only, never mutates (skipping evict just makes the caller's min-words
        gate more conservative)."""
        heard = words_of(transcript)
        if not heard:
            return set()
        snapshot = list(self._spoken)
        spoken: set[str] = set().union(*(w for _, w in snapshot)) if snapshot else set()
        return heard - spoken

    def reset(self) -> None:
        self._spoken.clear()

    def _evict(self) -> None:
        cutoff = time.monotonic() - self._window_secs
        while self._spoken and self._spoken[0][0] < cutoff:
            self._spoken.popleft()
