"""Transcript tier of the wake gate: leading wake-phrase match + strip.

The phrase must LEAD the utterance: only hesitation fillers may precede it, so "hey
nanobot, weather" wakes while "I said hey nanobot" stays content. Per-phrase regexes
join tokens by non-word runs — punctuation- and case-insensitive for spaced scripts, a
substring match inside fused CJK runs ("小助手今天天气" matches "小助手"), the same
alphabet rules as :mod:`..phrases`. Spaced-script tokens also end on a word boundary
("nanobot" never matches inside "nanobots"); a CJK-final phrase keeps its fused run.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterable

from nanobot_channel_voice.phrases import FILLER_WORDS, _segments, tokens_of

# Tokens tolerated BEFORE the wake phrase; anything else in front demotes the phrase to
# content. Unspaced CJK fillers arrive FUSED ("嗯那个"), so a prefix token may also be a
# greedy segmentation over this set.
_LEAD_OK = frozenset(FILLER_WORDS) | frozenset({
    "um", "uh", "erm", "hmm", "hey", "so",
    "嗯", "那个", "诶", "请问",
    "あの", "えっと", "ねえ",
})

# Trailing separators consumed after a stripped phrase: whitespace and clause
# punctuation ONLY, so sign/quote characters binding to the next token survive
# ("hey nanobot -3 degrees" keeps its minus).
_SEP_RE = re.compile(r"^[\s,.!?;:…、，。．！？；：]+")

# Same split the chunker uses: from ideographs/kana/hangul up, mid-run matches are
# legitimate.
_CJK_FLOOR = 0x2E80


def _nfkc(s: str) -> str:
    return unicodedata.normalize("NFKC", s)


def _fold(text: str) -> tuple[str, list[int]]:
    """``(folded, source_index)``: the alphabet ``tokens_of`` uses (NFKC + lower), so a
    fullwidth STT rendering matches an ASCII phrase. Folded per SEGMENT, a segment being
    the shortest run whose NFKC is not the concatenation of its parts (halfwidth kana +
    voiced mark, NFD combining marks): callers slice the ORIGINAL text, and the map has one
    entry per folded char plus a terminator."""
    n = len(text)
    if text.isascii() or (
        unicodedata.is_normalized("NFKC", text) and "\u03a3" not in text
    ):
        # NFKC and lower() are then char-wise identities: no capital sigma (its lower is
        # context-bound), and no expanding char (İ) or the map would shift.
        folded = text.lower()
        if len(folded) == n:
            return folded, list(range(n + 1))
    out: list[str] = []
    source: list[int] = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and _nfkc(text[start:end + 1]) != _nfkc(text[start:end]) + _nfkc(text[end]):
            end += 1
        piece = _nfkc(text[start:end]).lower()
        out.append(piece)
        source += [start] * len(piece)
        start = end
    source.append(n)
    return "".join(out), source


def _after(source: list[int], end: int) -> int:
    """Original offset just past folded position ``end``; a match ending inside one
    source char's expansion (㍿ -> 株式会社) consumes the whole char."""
    while end < len(source) - 1 and source[end] == source[end - 1]:
        end += 1
    return source[end]


def _lead_ok(prefix: str, extra: Callable[[str], bool] | None = None) -> bool:
    """Only filler precedes the phrase; a fused CJK run passes if it segments entirely
    into ``_LEAD_OK``. ``extra`` widens the acceptable set per token."""
    return all(
        t in _LEAD_OK
        or _segments(t, _LEAD_OK) is not None
        or (extra is not None and extra(t))
        for t in tokens_of(prefix)
    )


def _clean_start(text: str, start: int) -> bool:
    """Left-edge mirror of ``_clean_end``, for callers with no ``_lead_ok`` check
    (``present``): "ro|bot" never matches phrase "bot"; a CJK-initial match may continue
    its fused run."""
    if start <= 0 or ord(text[start]) >= _CJK_FLOOR:
        return True
    prev = text[start - 1]
    return not (prev.isalnum() or prev == "_") or ord(prev) >= _CJK_FLOOR


def _clean_end(text: str, end: int) -> bool:
    """The match ends at a word boundary for spaced scripts. A CJK-final match may
    continue its fused run, and following CJK is a new word — only same-script
    letter/digit continuation ("nanobot|s") rejects."""
    if end >= len(text) or ord(text[end - 1]) >= _CJK_FLOOR:
        return True
    nxt = text[end]
    return not (nxt.isalnum() or nxt == "_") or ord(nxt) >= _CJK_FLOOR


# Skeleton alphabet: vowels/glides drop (vowel confusion dominates STT errors on
# out-of-vocabulary names), doubles collapse, the first char survives.
_SOFT = frozenset("aeiouyhw")

# Vocatives a phrase may open with ("hey nanobot"): in-vocabulary words the STT renders
# whole, clipped ("he") or as a homophone ("hay"), so the name group starts at word two.
_VOCATIVES = frozenset({"hey", "hi", "ok", "okay", "hello", "yo"})
# A voiced coda for its unvoiced pair is the commonest STT confusion on a name's last
# consonant ("nanobad", "nanobody" for "nanobot"): equal at the skeleton's end.
_CODA_FOLD = str.maketrans("dbgzv", "tpksf")


def _skeleton(text: str) -> str:
    # Collapse ADJACENT duplicates BEFORE dropping soft chars: the other order fuses
    # consonants that vowels separated ("nano" must stay "nn").
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
    """Head-of-utterance phonetic matcher for latin wake phrases the STT mangles ("hey
    nanobot" -> "he nine obt": consonant skeletons match within 1 INTERIOR edit — the
    measured renders keep the onset and coda consonants and never add one). STRIP-ONLY
    trust tier: a fuzzy match must NEVER open the gate, only trim a turn that already
    passed on other evidence. CJK phrases opt out (homophone drift is the alias layer's
    job), as do skeletons under 4 chars (too collision-prone)."""

    __slots__ = ("_keys",)

    def __init__(self, phrases: Iterable[str]):
        self._keys = []
        for p in phrases:
            toks = tokens_of(p)
            if not toks or any(ord(c) >= _CJK_FLOOR for t in toks for c in t):
                continue
            lead = toks[0] if len(toks) > 1 and toks[0] in _VOCATIVES else None
            # The name group keeps its own onset letter, so a vowel-initial render
            # ("enough bit" for "nanobot") is an edit, not a dropped soft char.
            key = (_skeleton(lead) if lead else "") + _skeleton("".join(toks[bool(lead):]))
            if len(key) >= 4:
                key = key[:-1] + key[-1:].translate(_CODA_FOLD)  # once; _distance folds skel
                self._keys.append((p, lead, key, len(toks)))

    def __bool__(self) -> bool:
        return bool(self._keys)

    @staticmethod
    def _distance(skel: str, key: str) -> int | None:
        """0/1 when *skel* renders *key* (one interior consonant swapped or dropped,
        never an extra one; the coda equal up to voicing), else None."""
        skel = skel[:-1] + skel[-1:].translate(_CODA_FOLD)
        d = _edit_distance(skel, key)
        if d == 0 or (
            d == 1 and 5 <= len(skel) <= len(key) and skel[0] == key[0] and skel[-1] == key[-1]
        ):
            return d
        return None

    def strip_head(self, text: str) -> tuple[str | None, str]:
        """``(phrase, remainder)`` when leading words of *text* skeleton-match a phrase
        — best distance wins, SMALLEST window on ties (a larger one could absorb a
        soft-only content word: "...bot you"); ``(None, text)`` otherwise. Hesitation
        fillers may precede and are consumed with the match. Runs on the raw text, so
        the remainder keeps its original spelling."""
        folded, source = _fold(text)
        words = list(re.finditer(r"\w+", folded, re.UNICODE))
        lead_max = 0
        for m in words[:3]:
            if m.group().casefold() in _LEAD_OK:
                lead_max += 1
            else:
                break
        best: tuple[int, int, str, int] | None = None  # (dist, k, phrase, end)
        for lead in range(lead_max + 1):
            for phrase, vocative, key, ptoks in self._keys:
                head = words[lead: lead + ptoks + 2]
                prefix = ""
                if vocative is not None:
                    # The same skeleton ("he", "hay", "hi" for hey), never another word:
                    # "the nanobot" and "his computer" are content.
                    first = head[0].group() if head else None
                    if first is None or _skeleton(first) != _skeleton(vocative):
                        continue
                    prefix, head = _skeleton(vocative), head[1:]
                collected = ""
                for k, m in enumerate(head, 1 if vocative is None else 2):
                    tok = m.group()
                    if any(ord(c) >= _CJK_FLOOR for c in tok):
                        break  # a CJK head is not a mangled latin name
                    collected += tok
                    d = self._distance(prefix + _skeleton(collected), key)
                    if d is not None:
                        cand = (d, k, phrase, m.end())
                        if best is None or cand < best:
                            best = cand
        if best is None:
            return None, text
        return best[2], _SEP_RE.sub("", text[_after(source, best[3]):])


class WakePhrase:
    """Compiled wake-phrase list; ``strip`` is the one hot call. Falsy when no phrase
    survived tokenization (the gate then has no text tier). An entry may be
    ``(display, spelling)``: the SPELLING matches (an STT mis-render), the DISPLAY is
    reported as ``matched``, so an alias summons still acks by the name called."""

    __slots__ = ("_patterns",)

    def __init__(self, phrases: Iterable[str | tuple[str, str]]):
        self._patterns = []
        for entry in phrases:
            display, spelling = (entry, entry) if isinstance(entry, str) else entry
            # casefold() folds what IGNORECASE misses (straße/STRASSE); identical
            # variants dedupe via the set.
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
        """A wake phrase occurs ANYWHERE in *text* (``strip``'s word-boundary rules, no
        leading demand): the mention test the wake echo veto runs against spoken TTS."""
        return self.count(text) > 0

    def count(self, text: str) -> int:
        """Mentions in *text* under ``present``'s rules (a reply can name the phrase
        twice; each is its own echo)."""
        folded, _ = _fold(text)
        return sum(
            1
            for _, pat in self._patterns
            for m in pat.finditer(folded)
            if _clean_start(folded, m.start()) and _clean_end(folded, m.end())
        )

    def strip(
        self, text: str, extra_lead: Callable[[str], bool] | None = None
    ) -> tuple[str | None, str]:
        """``(matched, remainder)``: the SOURCE phrase leading *text* (only ``_LEAD_OK``
        tokens, widened per token by ``extra_lead``, may precede it; earliest/longest
        wins) and the text after it, separators stripped; ``(None, text)`` otherwise.
        One ``search()`` per pattern suffices: any occurrence after a rejected one has
        non-acceptable content in front."""
        folded, source = _fold(text)
        best, best_phrase = None, None
        for phrase, pat in self._patterns:
            m = pat.search(folded)
            if (
                m is None
                or not _clean_end(folded, m.end())
                or not _lead_ok(folded[: m.start()], extra_lead)
            ):
                continue
            if best is None or m.start() < best.start() or (
                m.start() == best.start() and m.end() > best.end()
            ):
                best, best_phrase = m, phrase
        if best is None:
            return None, text
        return best_phrase, _SEP_RE.sub("", text[_after(source, best.end()):])
