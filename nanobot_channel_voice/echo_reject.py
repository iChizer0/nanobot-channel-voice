"""Self-transcription rejection: drop STT that is the bot hearing its own TTS.

A transcript mostly contained in recently-spoken TTS is echo, not a barge-in, and is
dropped; genuinely different speech passes through to cancel-then-send. Stdlib only.

The comparison alphabet (:func:`units_of`) is script-aware: spaced-script tokens compare
whole, CJK segments as character BIGRAMS — word-token containment could never see zh
echo (a fused run is ONE ``\\w+`` token) and unigrams are too loose (common hanzi recur
in any reply).

STT renders audio differently than the TTS text wrote it, so units are bridged twice:
spoken text grows number-reading VARIANTS, and a heard Latin unit matches as a substring
of the spoken text's space-stripped Latin stream ("Wi-Fi"/"playsomemusic"). ``protect``
units (the stop lexicon) are exempt from the latter: "stop" heard during spoken
"unstoppable" must stay fresh evidence.
"""

from __future__ import annotations

import re
import time
from collections import deque
from collections.abc import Iterable, Iterator

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
        # (last-audible deadline, units, latin stream, raw text) per note.
        self._spoken: deque[tuple[float, set[str], str, str]] = deque()

    def note_spoken(self, text: str, hold_ms: float = 0.0) -> None:
        """Record TTS text about to play. ``hold_ms`` (delay until it stops sounding:
        sink backlog + own duration) shifts the stamp so eviction runs from
        last-audible, not feed time — text streams far faster than it plays, and
        feed-time stamps let the bot barge in on its own tail."""
        units = units_of(text)
        if not units:
            return
        for variant in _number_variants(text):
            units |= units_of(variant)
        self._spoken.append(
            (time.monotonic() + hold_ms / 1000.0, units, _latin_stream(text), text)
        )

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
        """Units in *transcript* that are NOT recently-spoken TTS. Callable from the frame
        worker thread while ``note_spoken`` runs on the loop: snapshot only, never
        mutates (skipping evict only makes the caller's min-words gate stricter)."""
        heard = units_of(transcript)
        if not heard:
            return set()
        spoken, streams = self._spoken_view(list(self._spoken))
        return {
            u for u in heard
            if u not in spoken
            and (u in self._protect or not self._absorbed(u, streams))
        }

    def recent_text(self, max_age_s: float | None = None) -> str:
        """The unexpired spoken texts, oldest first, joined: the ordered view the wake
        echo veto searches. ``max_age_s`` narrows it to what stopped sounding that
        recently (still playing = age 0). Loop-side only (evicts; ``fresh_words`` is the
        thread-safe one)."""
        self._evict()
        cutoff = -float("inf") if max_age_s is None else time.monotonic() - max_age_s
        return " ".join(t for deadline, _, _, t in self._spoken if deadline >= cutoff)

    def reset(self) -> None:
        self._spoken.clear()

    @staticmethod
    def _spoken_view(
        entries: Iterable[tuple[float, set[str], str, str]],
    ) -> tuple[set[str], list[str]]:
        units: set[str] = set()
        streams: list[str] = []
        for _, u, stream, _text in entries:
            units |= u
            if stream:
                streams.append(stream)
        return units, streams

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
        while self._spoken and self._spoken[0][0] < cutoff:
            self._spoken.popleft()
