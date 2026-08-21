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


# Skeleton alphabet: vowels/glides drop (vowel confusion dominates STT errors
# on out-of-vocabulary names), doubles collapse, the first char survives.
_SOFT = frozenset("aeiouyhw")


def _skeleton(text: str) -> str:
    # Collapse ADJACENT duplicates first ("ll"), THEN drop the soft chars:
    # collapsing after the drop would fuse consonants that vowels separated
    # ("nano" must stay "nn").
    chars = [c for c in text.casefold() if "a" <= c <= "z"]
    if not chars:
        return ""
    dedup = [chars[0]]
    for c in chars[1:]:
        if c != dedup[-1]:
            dedup.append(c)
    return dedup[0] + "".join(c for c in dedup[1:] if c not in _SOFT)


def _edit_distance(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[-1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class FuzzyWake:
    """Head-of-utterance phonetic matcher for latin wake phrases the STT
    mangles ("hey nanobot" -> "he nine obt": consonant skeletons match within
    1 edit). STRIP-ONLY trust tier: callers consult it only for utterances
    that already pass on other evidence — a fuzzy match must never open the
    gate, so its worst false positive eats a name-like head from a turn that
    was already ours. CJK phrases opt out (a zh-capable STT renders zh names
    exactly; homophone drift is the alias layer's job), as do names whose
    skeleton is under 4 chars (too collision-prone)."""

    __slots__ = ("_keys",)

    def __init__(self, phrases: Iterable[str]):
        self._keys = []
        for p in phrases:
            toks = tokens_of(p)
            if not toks or any(ord(c) >= _CJK_FLOOR for t in toks for c in t):
                continue
            key = _skeleton("".join(toks))
            if len(key) >= 4:
                self._keys.append((p, key, len(toks)))

    def __bool__(self) -> bool:
        return bool(self._keys)

    def strip_head(self, text: str) -> tuple[str | None, str]:
        """``(phrase, remainder)`` when leading words of *text* skeleton-match
        a phrase — best distance wins, SMALLEST window on ties (a larger one
        could absorb a soft-only content word: "...bot you"); ``(None, text)``
        otherwise. Hesitation fillers may precede, like the exact tier ("um he
        nine obt..."); they are consumed with the match. Runs on the raw text
        so the remainder keeps its original spelling."""
        words = list(re.finditer(r"\w+", text, re.UNICODE))
        lead_max = 0
        for m in words[:3]:
            if m.group().casefold() in _LEAD_OK:
                lead_max += 1
            else:
                break
        best: tuple[int, int, str, int] | None = None  # (dist, k, phrase, end)
        for lead in range(lead_max + 1):
            for phrase, key, ptoks in self._keys:
                collected = ""
                for k, m in enumerate(words[lead: lead + ptoks + 2], 1):
                    tok = m.group()
                    if any(ord(c) >= _CJK_FLOOR for c in tok):
                        break  # a CJK head is not a mangled latin name
                    collected += tok
                    skel = _skeleton(collected)
                    d = _edit_distance(skel, key)
                    if d == 0 or (d == 1 and min(len(skel), len(key)) >= 5):
                        cand = (d, k, phrase, m.end())
                        if best is None or cand < best:
                            best = cand
        if best is None:
            return None, text
        return best[2], _SEP_RE.sub("", text[best[3]:])


class WakePhrase:
    """Compiled wake-phrase list; ``strip`` is the one hot call. Falsy when no
    phrase survived tokenization (the gate then has no text tier). An entry may
    be ``(display, spelling)``: the SPELLING matches (an STT's mis-render of
    the name), the DISPLAY is reported as ``matched`` — so an alias summons
    still routes the ack by the phrase the user actually called."""

    __slots__ = ("_patterns",)

    def __init__(self, phrases: Iterable[str | tuple[str, str]]):
        self._patterns = []
        for entry in phrases:
            display, spelling = (entry, entry) if isinstance(entry, str) else entry
            # casefold() folds what IGNORECASE's simple folding misses
            # (straße/STRASSE); identical variants dedupe via the set.
            for toks in {
                tuple(tokens_of(spelling)), tuple(tokens_of(spelling.casefold()))
            }:
                if toks:
                    self._patterns.append((
                        display,
                        re.compile(
                            r"[\W_]*".join(re.escape(t) for t in toks),
                            re.IGNORECASE | re.UNICODE,
                        ),
                    ))

    def __bool__(self) -> bool:
        return bool(self._patterns)

    def leads(
        self, text: str, extra_lead: Callable[[str], bool] | None = None
    ) -> bool:
        return self.strip(text, extra_lead)[0] is not None

    def present(self, text: str) -> bool:
        """A wake phrase occurs ANYWHERE in *text* (word-boundary rules of
        ``strip``, no leading demand): the mention test the wake echo veto
        runs against recently-spoken TTS."""
        return any(
            _clean_start(text, m.start()) and _clean_end(text, m.end())
            for _, pat in self._patterns
            for m in pat.finditer(text)
        )

    def strip(
        self, text: str, extra_lead: Callable[[str], bool] | None = None
    ) -> tuple[str | None, str]:
        """``(matched, remainder)``: the SOURCE phrase that leads *text* (only
        ``_LEAD_OK`` tokens — widened per token by ``extra_lead`` — may precede
        it; earliest/longest match wins) and the text after it, separators
        stripped; ``(None, text)`` otherwise. Truthy exactly on a match, so
        boolean callers read as before. ``search()`` per pattern suffices: any
        occurrence after a rejected one has non-acceptable content in front."""
        best, best_phrase = None, None
        for phrase, pat in self._patterns:
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
                best, best_phrase = m, phrase
        if best is None:
            return None, text
        return best_phrase, _SEP_RE.sub("", text[best.end():])
