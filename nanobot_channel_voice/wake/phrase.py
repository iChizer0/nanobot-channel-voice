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
from collections.abc import Callable, Iterable

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


def _lead_ok(prefix: str, extra: Callable[[str], bool] | None = None) -> bool:
    """Only filler material precedes the phrase; fused CJK runs are accepted
    when they segment entirely into ``_LEAD_OK`` entries. ``extra`` widens the
    acceptable set per token (the caller's leak test — see ``strip``)."""
    return all(
        t in _LEAD_OK
        or _segments(t, _LEAD_OK) is not None
        or (extra is not None and extra(t))
        for t in tokens_of(prefix)
    )


def _clean_start(text: str, start: int) -> bool:
    """Mirror of ``_clean_end`` for the left edge, needed where no ``_lead_ok``
    runs (``present``): a spaced-script match must not continue a same-script
    word leftward ("ro|bot" never matches phrase "bot"); a CJK-initial match
    may continue its fused run."""
    if start <= 0 or ord(text[start]) >= _CJK_FLOOR:
        return True
    prev = text[start - 1]
    return not (prev.isalnum() or prev == "_") or ord(prev) >= _CJK_FLOOR


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

    def leads(
        self, text: str, extra_lead: Callable[[str], bool] | None = None
    ) -> bool:
        return self.strip(text, extra_lead)[0]

    def present(self, text: str) -> bool:
        """A wake phrase occurs ANYWHERE in *text* (word-boundary rules of
        ``strip``, no leading demand): the mention test the wake echo veto
        runs against recently-spoken TTS."""
        return any(
            _clean_start(text, m.start()) and _clean_end(text, m.end())
            for pat in self._patterns
            for m in pat.finditer(text)
        )

    def strip(
        self, text: str, extra_lead: Callable[[str], bool] | None = None
    ) -> tuple[bool, str]:
        """``(matched, remainder)``: whether a wake phrase leads *text* (only
        ``_LEAD_OK`` tokens — widened per token by ``extra_lead`` — may precede
        it), and the text after it with leading separators removed. ``(False,
        text)`` otherwise. ``search()`` per pattern suffices: any occurrence
        after a rejected one necessarily has non-acceptable content in front of
        it."""
        best = None
        for pat in self._patterns:
            m = pat.search(text)
            if (
                m is None
                or not _clean_end(text, m.end())
                or not _lead_ok(text[: m.start()], extra_lead)
            ):
                continue
            if best is None or m.start() < best.start() or (
                m.start() == best.start() and m.end() > best.end()
            ):
                best = m
        if best is None:
            return False, text
        return True, _SEP_RE.sub("", text[best.end():])
