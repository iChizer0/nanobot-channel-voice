"""Self-transcription rejection: drop STT that is the bot hearing its own TTS.

A transcript mostly contained in recently-spoken TTS is echo, not a barge-in, and is
dropped; genuinely different speech passes through to cancel-then-send. Stdlib only.

The comparison alphabet (:func:`units_of`) is script-aware: spaced-script tokens compare
whole, CJK segments as character BIGRAMS — word-token containment could never see zh
echo (a fused run is ONE ``\\w+`` token) and unigrams are too loose (common hanzi recur
in any reply).

STT renders audio differently than the TTS text wrote it, so units are bridged three
times: spoken text grows number-reading VARIANTS, its CJK bigrams also run across
punctuation and the seam between two spoken pieces (a transcript punctuated elsewhere, or
not at all, runs them together), and a heard Latin unit matches as a substring of the
spoken text's space-stripped Latin stream ("Wi-Fi"/"playsomemusic"). No bridge makes a
``protect`` unit (the stop lexicon) ours: "stop" heard during spoken "unstoppable" must
stay fresh evidence. Fresh evidence also forgives one misheard CJK character
(:meth:`SelfEchoFilter.fresh_words`), never a stop word's.
"""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Iterable, Iterator
from typing import NamedTuple

from nanobot_channel_voice.phrases import tokens_of, words_of

# The same functions the TTS side speaks digits through: variants match what became
# audible by construction; its optional deps load lazily, so the import stays top-level.
from nanobot_channel_voice.tts.text_frontend import (
    en_digit_words,
    verbalize_numbers_en,
    verbalize_numbers_zh,
    zh_digit_words,
    zh_numeral_value,
)

__all__ = ["SelfEchoFilter", "units_of", "words_of"]

_CJK_FLOOR = 0x2E80  # same script split as wake/phrase.py, backend/local.py, tts/router.py

_DIGIT_RUN = re.compile(r"\d+")
_ZH_NUM_RUN = re.compile(r"[零一二三四五六七八九十百千万亿两]+")


def _script_segs(token: str) -> Iterator[tuple[bool, str]]:
    """Maximal same-script ``(is_cjk, segment)`` spans of one token."""
    i, n = 0, len(token)
    while i < n:
        cjk = ord(token[i]) >= _CJK_FLOOR
        j = i + 1
        while j < n and (ord(token[j]) >= _CJK_FLOOR) == cjk:
            j += 1
        yield cjk, token[i:j]
        i = j


def units_of(text: str) -> set[str]:
    """The echo-comparison alphabet: lower-cased word tokens for spaced scripts,
    character bigrams per CJK segment (a lone CJK char stays a unigram)."""
    units: set[str] = set()
    for token in tokens_of(text):
        for cjk, seg in _script_segs(token):
            if not cjk or len(seg) == 1:
                units.add(seg)
            else:
                units.update(seg[k : k + 2] for k in range(len(seg) - 1))
    return units


def _cjk_stream(text: str) -> str:
    """CJK characters in order, joined across punctuation and spacing; a non-CJK word
    between two still parts them (a space)."""
    return "".join(
        seg if cjk else " " for token in tokens_of(text) for cjk, seg in _script_segs(token)
    )


def _pairs(stream: str) -> set[str]:
    return {p for k in range(len(stream) - 1) if " " not in (p := stream[k : k + 2])}


def _latin_stream(text: str) -> str:
    """Space/hyphen-stripped non-CJK material, in order: the string an STT respacing of
    the same audio must still be a substring of."""
    return "".join(
        seg for token in tokens_of(text) for cjk, seg in _script_segs(token) if not cjk
    )


def _number_variants(text: str) -> list[str]:
    """Alternate renderings of *text*'s numbers, as whole texts so CJK bigrams form
    across the number boundary (七点/点四). Bigrams are local: one variant per reading
    covers all, no cross-product needed."""
    variants: list[str] = []
    if _DIGIT_RUN.search(text):
        zh = verbalize_numbers_zh(text)
        variants.append(zh)
        liang = re.sub("二(?=[百千万])", "两", zh)  # colloquial cloud-TTS reading
        if liang != zh:
            variants.append(liang)
        variants.append(_DIGIT_RUN.sub(lambda m: zh_digit_words(m.group()), text))
        variants.append(verbalize_numbers_en(text))
        variants.append(_DIGIT_RUN.sub(lambda m: en_digit_words(m.group()), text))
    if _ZH_NUM_RUN.search(text):
        variants.append(_ZH_NUM_RUN.sub(
            lambda m: str(v) if (v := zh_numeral_value(m.group())) is not None
            else m.group(), text,
        ))
    return [v for v in variants if v != text]


class _Note(NamedTuple):
    until: float  # last audible: eviction runs from here
    units: set[str]
    latin: str  # space-stripped Latin stream: the respacing bridge
    cjk: str  # CJK characters as spoken; consecutive notes run on
    alt: str  # the same per number reading, space-parted
    tails: str  # each reading's last CJK character: the seam with the next note
    text: str


def _contexts(most_left: int, most_right: int) -> list[tuple[int, int]]:
    """How many characters to match either side of a misheard one: three in all, fewer
    only in a run too short for three."""
    if most_left + most_right < 3:
        return [(1, 1)]
    return [
        (left, right)
        for left, right in ((2, 1), (1, 2))
        if left <= most_left and right <= most_right
    ]


def _one_off(said: str, left: str, right: str) -> bool:
    """*said* holds *left* then *right*, adjacent or one character apart."""
    if left + right in said:
        return True
    i = said.find(left)
    while i >= 0:
        j = i + len(left)
        if j < len(said) and said[j] != " " and said.startswith(right, j + 1):
            return True
        i = said.find(left, i + 1)
    return False


class SelfEchoFilter:
    def __init__(
        self,
        threshold: float = 0.6,
        window_secs: float = 12.0,
        protect: Iterable[str] = (),
    ):
        self._threshold = threshold
        self._window_secs = window_secs
        self._protect = frozenset(protect)
        self._spoken: deque[_Note] = deque()

    def note_spoken(self, text: str, hold_ms: float = 0.0) -> None:
        """Record TTS text about to play. ``hold_ms`` (delay until it stops sounding:
        sink backlog + own duration) shifts the stamp so eviction runs from
        last-audible, not feed time — text streams far faster than it plays, and
        feed-time stamps let the bot barge in on its own tail."""
        # Write-side eviction: the read side only runs when someone speaks back.
        self._evict()
        units = units_of(text)
        if not units:
            return
        variants = _number_variants(text)
        streams = [_cjk_stream(r) for r in (text, *variants)]
        # What the text does not spell out never makes a stop word ours: a number read
        # out, and bigrams across a clause mark or the seam with the piece before (pieces
        # come per clause), which a transcript punctuated elsewhere runs together.
        tails = self._spoken[-1].tails if self._spoken else ""
        guessed: set[str] = set()
        for variant in variants:
            guessed |= units_of(variant)
        for stream in streams:
            guessed |= _pairs(stream)
            if stream[:1].strip():
                guessed.update(t + stream[0] for t in tails)
        units |= guessed - self._protect
        self._spoken.append(_Note(
            time.monotonic() + hold_ms / 1000.0, units, _latin_stream(text), streams[0],
            " ".join(streams[1:]), "".join(s[-1] for s in streams if s[-1:].strip()), text,
        ))

    def is_self_echo(self, transcript: str) -> bool:
        """True if *transcript* is mostly the bot's recently-spoken units (echo)."""
        self._evict()
        heard = units_of(transcript)
        if not heard or not self._spoken:
            return False
        spoken, streams = self._spoken_view(self._spoken)
        covered = sum(1 for u in heard if u in spoken or self._absorbed(u, streams))
        return covered / len(heard) >= self._threshold

    def fresh_words(self, transcript: str) -> set[str]:
        """Units in *transcript* that are NOT recently-spoken TTS. A CJK bigram we never
        spoke is still ours when one misheard character explains it (STT on our own
        leak): the reply holds its neighbourhood with a character swapped, added or
        dropped. A stop word is never explained away. Callable from the frame worker
        thread while ``note_spoken`` runs on the loop: snapshot only, never mutates
        (skipping evict only makes the caller's min-words gate stricter)."""
        tokens = tokens_of(transcript)
        if not tokens:
            return set()
        for _ in range(3):  # loop-side evict can mutate under the snapshot: retry
            try:
                entries = list(self._spoken)
                break
            except RuntimeError:
                entries = None
        if entries is None:
            return set()  # "nothing fresh" = this poll is skipped, later ones re-run
        spoken, streams = self._spoken_view(entries)
        said: str | None = None  # the CJK we spoke, joined for the first bigram to explain
        fresh: set[str] = set()
        for token in tokens:
            for cjk, seg in _script_segs(token):
                if not cjk or len(seg) == 1:
                    if seg not in spoken and (
                        seg in self._protect or not self._absorbed(seg, streams)
                    ):
                        fresh.add(seg)
                    continue
                for k in range(len(seg) - 1):
                    pair = seg[k : k + 2]
                    if pair in spoken:
                        continue
                    if pair not in self._protect:
                        if said is None:
                            said = "".join(n.cjk for n in entries) + " " + " ".join(
                                n.alt for n in entries
                            )
                        if self._misheard(seg, k, said):
                            continue
                    fresh.add(pair)
        return fresh

    def fresh_readings(self, transcript: str, fresh: set[str]) -> list[list[str]]:
        """*fresh* in utterance order, as the stop lexicon may read it: the tokens holding a
        fresh unit whole ("やめてください"), and, where it differs, each CJK run cut to its
        stretches outside our own words (a character is ours in a bigram neither fresh nor
        a stop word's) that hold a fresh unit, so a stop said into our words reads alone."""
        whole = [t for t in tokens_of(transcript) if units_of(t) & fresh]
        cut: list[str] = []
        for token in whole:
            segs = list(_script_segs(token))
            if not any(cjk for cjk, _ in segs):
                cut.append(token)
                continue
            for cjk, seg in segs:
                if not cjk or len(seg) == 1:
                    if seg in fresh:
                        cut.append(seg)
                    continue
                ours = [False] * len(seg)
                hot = [False] * len(seg)
                for k in range(len(seg) - 1):
                    pair = seg[k : k + 2]
                    if pair in fresh:
                        hot[k] = hot[k + 1] = True
                    elif pair not in self._protect:
                        ours[k] = ours[k + 1] = True
                start = None
                for i in range(len(seg) + 1):
                    if i < len(seg) and not ours[i]:
                        start = i if start is None else start
                        continue
                    if start is not None and any(hot[start:i]):
                        cut.append(seg[start:i])
                    start = None
        return [whole] if cut == whole else [whole, cut]

    def recent_text(self, max_age_s: float | None = None) -> str:
        """The unexpired spoken texts, oldest first, joined: the ordered view the wake
        echo veto searches. ``max_age_s`` narrows it to what stopped sounding that
        recently (still playing = age 0). Loop-side only (evicts; ``fresh_words`` is the
        thread-safe one)."""
        self._evict()
        cutoff = -float("inf") if max_age_s is None else time.monotonic() - max_age_s
        return " ".join(n.text for n in self._spoken if n.until >= cutoff)

    def cut(self, unheard: Iterable[str]) -> None:
        """Playback stopped NOW: ``unheard`` (the texts that never sounded, in spoken order,
        so the newest notes) go — a user repeating them is fresh; what remains ends its
        hold now, so eviction runs from the cut, not from a playout that never came."""
        for text in reversed(list(unheard)):
            if self._spoken and self._spoken[-1].text == text:
                self._spoken.pop()
            elif units_of(text):
                break  # noted, yet not the tail: the ledger moved on (unit-less = never noted)
        now = time.monotonic()
        self._spoken = deque(n._replace(until=min(n.until, now)) for n in self._spoken)

    def reset(self) -> None:
        self._spoken.clear()

    @staticmethod
    def _spoken_view(entries: Iterable[_Note]) -> tuple[set[str], list[str]]:
        units: set[str] = set()
        streams: list[str] = []
        for note in entries:
            units |= note.units
            if note.latin:
                streams.append(note.latin)
        return units, streams

    def _misheard(self, seg: str, k: int, said: str) -> bool:
        """Bigram *k* of CJK run *seg* is ours but for one character: either of its two
        swapped or added, or one dropped between them. A stop word's character is never
        the misheard one."""
        n, stop = len(seg), self._protect
        for i in (k, k + 1):
            if 0 < i < n - 1 and not any(
                u in stop for u in (seg[i], seg[i - 1 : i + 1], seg[i : i + 2])
            ) and any(
                _one_off(said, seg[i - left : i], seg[i + 1 : i + 1 + right])
                for left, right in _contexts(i, n - 1 - i)
            ):
                return True
        return not (seg[k] in stop or seg[k + 1] in stop) and any(
            _one_off(said, seg[k + 1 - left : k + 1], seg[k + 1 : k + 1 + right])
            for left, right in _contexts(k + 1, n - 1 - k)
        )

    @staticmethod
    def _absorbed(unit: str, streams: list[str]) -> bool:
        # Latin-respacing bridge; >=2 chars so stray single letters stay fresh.
        return (
            len(unit) >= 2
            and ord(unit[0]) < _CJK_FLOOR
            and any(unit in s for s in streams)
        )

    def _evict(self) -> None:
        cutoff = time.monotonic() - self._window_secs
        while self._spoken and self._spoken[0].until < cutoff:
            self._spoken.popleft()
