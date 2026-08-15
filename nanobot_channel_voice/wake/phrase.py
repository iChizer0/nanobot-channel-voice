"""Transcript tier of the wake gate: leading wake-phrase match + strip.

The wake phrase must LEAD the utterance (Alexa's contract): only hesitation
fillers may precede it, so "hey nanobot, weather" wakes while "I said hey
nanobot" stays content. Matching runs on the raw transcript with per-phrase
regexes whose tokens are joined by non-word runs, which makes them punctuation-
and case-insensitive for spaced scripts and a plain substring match inside
fused CJK runs ("小助手今天天气" matches phrase "小助手") — the same alphabet
rules as :mod:`..phrases`. Spaced-script tokens additionally end on a word
boundary ("nanobot" never matches inside "nanobots"); a CJK-final phrase keeps
matching its fused run.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from nanobot_channel_voice.phrases import FILLER_WORDS, _segments, tokens_of

# Tokens tolerated BEFORE the wake phrase: politeness fillers plus the
# hesitation noises STT engines actually emit at utterance starts. Anything
# else in front demotes the phrase to content. Unspaced CJK fillers arrive
# FUSED ("嗯那个"), so a prefix token may also be a greedy segmentation over
# this set (same rule phrases.py applies to command lexicons).
_LEAD_OK = frozenset(FILLER_WORDS) | frozenset({
    "um", "uh", "erm", "hmm", "hey", "so",
    "嗯", "那个", "诶", "请问",
    "あの", "えっと", "ねえ",
})

# Trailing separators consumed after a stripped phrase: whitespace and true
# clause punctuation only, so "hey nanobot, tell me..." publishes "tell me..."
# while sign/quote characters that bind to the following token survive
# ("hey nanobot -3 degrees" keeps its minus).
_SEP_RE = re.compile(r"^[\s,.!?;:…、，。．！？；：]+")

# Same script split the chunker uses: ideographs/kana/hangul upward are
# unspaced scripts where mid-run matches are legitimate.
_CJK_FLOOR = 0x2E80


def _lead_ok(prefix: str) -> bool:
    """Only filler material precedes the phrase; fused CJK runs are accepted
    when they segment entirely into ``_LEAD_OK`` entries."""
    return all(
        t in _LEAD_OK or _segments(t, _LEAD_OK) is not None
        for t in tokens_of(prefix)
    )


def _clean_end(text: str, end: int) -> bool:
    """The match ends at a word boundary for spaced scripts. A CJK-final match
    may continue into its fused run, and a spaced-script match may be followed
    by CJK (a new word by definition) — only same-script letter/digit
    continuation ("nanobot|s") rejects."""
    if end >= len(text) or ord(text[end - 1]) >= _CJK_FLOOR:
        return True
    nxt = text[end]
    return not (nxt.isalnum() or nxt == "_") or ord(nxt) >= _CJK_FLOOR


class WakePhrase:
    """Compiled wake-phrase list; ``strip`` is the one hot call. Falsy when no
    phrase survived tokenization (the gate then has no text tier)."""

    __slots__ = ("_patterns",)

    def __init__(self, phrases: Iterable[str]):
        self._patterns = [
            re.compile(
                r"[\W_]*".join(re.escape(t) for t in toks),
                re.IGNORECASE | re.UNICODE,
            )
            for p in phrases
            # casefold() folds what IGNORECASE's simple folding misses
            # (straße/STRASSE); identical variants dedupe via the set.
            for toks in {tuple(tokens_of(p)), tuple(tokens_of(p.casefold()))}
            if toks
        ]

    def __bool__(self) -> bool:
        return bool(self._patterns)

    def leads(self, text: str) -> bool:
        return self.strip(text)[0]

    def strip(self, text: str) -> tuple[bool, str]:
        """``(matched, remainder)``: whether a wake phrase leads *text* (only
        ``_LEAD_OK`` tokens may precede it), and the text after it with leading
        separators removed. ``(False, text)`` otherwise. ``search()`` per
        pattern suffices: any occurrence after a rejected one necessarily has
        non-filler content in front of it."""
        best = None
        for pat in self._patterns:
            m = pat.search(text)
            if (
                m is None
                or not _clean_end(text, m.end())
                or not _lead_ok(text[: m.start()])
            ):
                continue
            if best is None or m.start() < best.start() or (
                m.start() == best.start() and m.end() > best.end()
            ):
                best = m
        if best is None:
            return False, text
        return True, _SEP_RE.sub("", text[best.end():])
