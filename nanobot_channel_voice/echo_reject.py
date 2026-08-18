"""Self-transcription rejection: drop STT that is the bot hearing its own TTS.

With the mic open while the bot speaks, capture contains the bot's own voice;
since the spoken text is known, a transcript mostly contained in recently-spoken
TTS is that echo, not a barge-in, and is dropped, while genuinely different
speech passes through to cancel-then-send. This makes open-mic modes usable and
hardens hardware AEC against residual echo. Stdlib only.

The comparison alphabet (:func:`units_of`) is script-aware: spaced-script
tokens compare whole, CJK segments as character BIGRAMS — a fused zh run and
the TTS text it echoes then overlap whatever punctuation either side carries
(word-token containment could never see zh echo: the whole transcript is ONE
``\\w+`` token), while unigrams would be too loose (common hanzi recur in any
reply and would flag genuine speech).
"""

from __future__ import annotations

import time
from collections import deque

from nanobot_channel_voice.phrases import tokens_of, words_of

__all__ = ["SelfEchoFilter", "units_of", "words_of"]

_CJK_FLOOR = 0x2E80  # same script split as phrases/chunker


def units_of(text: str) -> set[str]:
    """The echo-comparison alphabet: lower-cased word tokens for spaced scripts,
    character bigrams per CJK segment (a lone CJK char stays a unigram)."""
    units: set[str] = set()
    for token in tokens_of(text):
        i, n = 0, len(token)
        while i < n:
            cjk = ord(token[i]) >= _CJK_FLOOR
            j = i + 1
            while j < n and (ord(token[j]) >= _CJK_FLOOR) == cjk:
                j += 1
            seg = token[i:j]
            if not cjk or len(seg) == 1:
                units.add(seg)
            else:
                units.update(seg[k : k + 2] for k in range(len(seg) - 1))
            i = j
    return units


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
        units = units_of(text)
        if units:
            self._spoken.append((time.monotonic() + hold_ms / 1000.0, units))

    def is_self_echo(self, transcript: str) -> bool:
        """True if *transcript* is mostly the bot's recently-spoken units (echo)."""
        self._evict()
        heard = units_of(transcript)
        if not heard or not self._spoken:
            return False
        spoken: set[str] = set().union(*(w for _, w in self._spoken))
        overlap = len(heard & spoken) / len(heard)
        return overlap >= self._threshold

    def fresh_words(self, transcript: str) -> set[str]:
        """Units in *transcript* that are NOT recently-spoken TTS. Callable from
        the frame worker thread while ``note_spoken`` runs on the loop: snapshot
        only, never mutates (skipping evict just makes the caller's min-words
        gate more conservative)."""
        heard = units_of(transcript)
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
