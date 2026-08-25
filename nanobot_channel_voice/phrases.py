"""Lexicon matching for spoken command/backchannel classification.

The alphabet is NFKC-folded, lower-cased ``\\w+`` runs. Spaced scripts tokenize per word;
unspaced CJK comes out as fused runs ("好的好的", "止めてください"), so
single-token phrases also match as greedy longest-prefix segments inside a
run — the one extra rule CJK needs, a no-op for spaced scripts.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable

_WORD_RE = re.compile(r"\w+", re.UNICODE)

# Politeness/emphasis tokens allowed to accompany a command without making it
# "content" ("please stop", "no no stop", "止めてください"). Never sufficient on
# their own: pure() still demands a full command phrase.
FILLER_WORDS = frozenset({
    "please", "now", "just", "no",
    "请", "麻烦",
    "ください", "です", "お願い",
})


def tokens_of(text: str) -> list[str]:
    # NFKC: STT compatibility forms (fullwidth ＷｉＦｉ/７) meet canonical TTS text
    return _WORD_RE.findall(unicodedata.normalize("NFKC", text).lower())


def words_of(text: str) -> set[str]:
    """The comparison alphabet used across the confirm stage: lower-cased word tokens."""
    return set(tokens_of(text))


class PhraseLexicon:
    """A normalized phrase list: ``phrases`` as token tuples, ``singles`` the
    one-token phrases (the only ones that can match inside a fused CJK run),
    ``words`` the flat alphabet."""

    __slots__ = ("phrases", "singles", "words")

    def __init__(self, phrases: Iterable[str]):
        tokenized = [tuple(tokens_of(p)) for p in phrases]
        self.phrases: list[tuple[str, ...]] = [p for p in tokenized if p]
        self.singles = frozenset(p[0] for p in self.phrases if len(p) == 1)
        self.words = frozenset(w for p in self.phrases for w in p)


def _segments(token: str, singles: frozenset[str]) -> list[str] | None:
    """Greedy longest-prefix decomposition of a fused run into single-token
    phrases; None when any position fails to match."""
    out: list[str] = []
    i = 0
    while i < len(token):
        for j in range(len(token), i, -1):
            if token[i:j] in singles:
                out.append(token[i:j])
                i = j
                break
        else:
            return None
    return out


def _contiguous(phrase: tuple[str, ...], tokens: list[str]) -> bool:
    n = len(phrase)
    return any(tuple(tokens[i:i + n]) == phrase for i in range(len(tokens) - n + 1))


class PhraseMatcher:
    """A command lexicon fused with its companion vocabularies, unions built ONCE:
    per-call union construction would land on the frame-hop poll path."""

    __slots__ = ("_command", "_singles", "_words")

    def __init__(
        self,
        command: PhraseLexicon,
        *companions: PhraseLexicon,
        extra: frozenset[str] = frozenset(),
    ):
        self._command = command
        words = set(command.words) | extra
        singles = set(command.singles) | extra
        for lex in companions:
            words |= lex.words
            singles |= lex.singles
        self._words = frozenset(words)
        self._singles = frozenset(singles)

    def covers(self, tokens: Iterable[str]) -> bool:
        """Every token is known vocabulary (word, extra, or a decomposable fused
        run). Empty input is NOT covered."""
        tokens = list(tokens)
        return bool(tokens) and all(
            t in self._words or _segments(t, self._singles) is not None for t in tokens
        )

    def present(self, tokens: list[str]) -> bool:
        """A full command phrase occurs: contiguous for multi-word phrases, direct
        or as a segment of a fused run for single-token ones. Tokens must be in
        utterance order for multi-word phrases to count."""
        cmd = self._command
        if any(_contiguous(p, tokens) for p in cmd.phrases if len(p) > 1):
            return True
        for t in tokens:
            if t in cmd.singles:
                return True
            if t not in self._words:
                segs = _segments(t, self._singles)
                if segs is not None and any(s in cmd.singles for s in segs):
                    return True
        return False

    def pure(self, tokens: list[str]) -> bool:
        """The utterance is ENTIRELY command/companion/filler material AND contains
        at least one full command phrase. The entirety rule is what keeps a lexicon
        safe here: "stop the music" has non-lexicon words, so it is content, not a
        command."""
        if not tokens:
            return False
        cmd = self._command
        hit = any(_contiguous(p, tokens) for p in cmd.phrases if len(p) > 1)
        for t in tokens:
            if t in cmd.singles:
                hit = True
            elif t in self._words:
                continue
            else:
                segs = _segments(t, self._singles)
                if segs is None:
                    return False
                hit = hit or any(s in cmd.singles for s in segs)
        return hit


def covered(
    tokens: Iterable[str], *lexicons: PhraseLexicon, extra: frozenset[str] = frozenset()
) -> bool:
    """One-shot :meth:`PhraseMatcher.covers` (hot paths hold a matcher instead)."""
    first, rest = lexicons[0], lexicons[1:]
    return PhraseMatcher(first, *rest, extra=extra).covers(tokens)


def pure_command(
    tokens: list[str],
    command: PhraseLexicon,
    *companions: PhraseLexicon,
    extra: frozenset[str] = FILLER_WORDS,
) -> bool:
    """One-shot :meth:`PhraseMatcher.pure` (hot paths hold a matcher instead)."""
    return PhraseMatcher(command, *companions, extra=extra).pure(tokens)


def phrase_within(text: str, lexicon: PhraseLexicon) -> bool:
    """A full phrase occurs anywhere in an utterance that is otherwise free content.

    ``PhraseMatcher.present`` cannot serve here: its fused-run rule decomposes a CJK
    token ENTIRELY into lexicon singles, so a trigger buried in a real zh/ja sentence
    never matches. An all-unspaced phrase is compared against the fused transcript,
    where its own spacing is meaningless too ("持续 处理" must still find 持续处理…);
    anything with a spaced-script token keeps its boundaries, so "stop" is never
    "stopwatch".
    """
    tokens = tokens_of(text)
    spaced = " " + " ".join(tokens) + " "
    fused = "".join(tokens)
    for phrase in lexicon.phrases:
        if all(not token.isascii() for token in phrase):
            if "".join(phrase) in fused:
                return True
        elif f" {' '.join(phrase)} " in spaced:
            return True
    return False
