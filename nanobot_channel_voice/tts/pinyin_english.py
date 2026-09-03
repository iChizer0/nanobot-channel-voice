"""English-into-pinyin fallback for lexicon (zh) Matcha models.

The zh token space can only say Mandarin: acronyms spell as the Mandarin letter
readings, words transliterate espeak IPA to the nearest pinyin syllables. A
loanword-accent approximation by design — full English sentences belong on an
English engine. Every emitted syllable is validated against the live token set, so
an unmappable fragment degrades to spelling (or silence) rather than a bad id.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from nanobot_channel_voice.tts.espeak import BATCH_SEP

# Mandarin letter readings as (syllable, preferred tone) sequences; resolution tries
# the tone ladder then the toneless twin, and a letter with any unresolvable syllable
# is dropped at build.
_LETTERS = {
    "a": [("ei", 1)], "b": [("bi", 4)], "c": [("xi", 1)], "d": [("di", 4)],
    "e": [("yi", 1)], "f": [("ai", 4), ("fu", 2)], "g": [("ji", 4)],
    "h": [("ai", 1), ("chi", 3)], "i": [("ai", 4)], "j": [("jie", 2)],
    "k": [("kai", 1)], "l": [("ai", 4), ("le", 4)], "m": [("ai", 4), ("mu", 3)],
    "n": [("en", 1)], "o": [("ou", 1)], "p": [("pi", 4)], "q": [("kou", 4)],
    "r": [("a", 1), ("er", 3)], "s": [("ai", 4), ("si", 1)], "t": [("ti", 4)],
    "u": [("you", 1)], "v": [("wei", 1)], "w": [("da", 2), ("bu", 4), ("liu", 4)],
    "x": [("ai", 4), ("ke", 4), ("si", 1)], "y": [("wai", 1)], "z": [("zei", 2)],
}

# espeak IPA nuclei -> pinyin finals (diphthongs before monophthongs at parse).
_VOWELS = {
    "aɪ": "ai", "eɪ": "ei", "ɔɪ": "ai", "aʊ": "ao", "oʊ": "ou", "əʊ": "ou",
    "ɪə": "i", "eə": "ai", "ʊə": "u",
    "i": "i", "ɪ": "i", "e": "ei", "ɛ": "ai", "æ": "a", "a": "a", "ɑ": "a",
    "ʌ": "a", "ɒ": "ao", "ɔ": "ao", "o": "ou", "u": "u", "ʊ": "u",
    "ə": "e", "ɜ": "e", "ɚ": "er", "ɐ": "a",
}
# IPA onsets -> pinyin initials.
_INITIALS = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "ɡ": "g", "g": "g",
    "m": "m", "n": "n", "f": "f", "v": "w", "s": "s", "z": "z",
    "θ": "s", "ð": "z", "ʃ": "sh", "ʒ": "r", "h": "h",
    "tʃ": "ch", "dʒ": "zh", "ts": "c", "w": "w", "j": "y", "ɹ": "r",
    "r": "r", "l": "l",
}
# A consonant with no vowel takes a filler-vowel syllable: pinyin has no clusters
# ("street" -> si te rui te).
_ALONE = {
    "p": ("pu", 1), "b": ("bu", 4), "t": ("te", 4), "d": ("de", 2),
    "k": ("ke", 4), "ɡ": ("ge", 2), "g": ("ge", 2), "m": ("mu", 3),
    "n": ("en", 1), "ŋ": ("en", 1), "f": ("fu", 2), "v": ("wu", 1),
    "s": ("si", 1), "z": ("zi", 1), "θ": ("si", 1), "ð": ("zi", 1),
    "ʃ": ("shi", 1), "ʒ": ("ri", 4), "h": ("he", 1), "tʃ": ("chi", 1),
    "dʒ": ("ji", 2), "ts": ("ci", 1), "w": ("wu", 1), "j": ("yi", 1),
    "ɹ": ("er", 2), "r": ("er", 2), "l": ("le", 4),
}
# Bare-nucleus syllables need the y/w onset spelling pinyin uses.
_ALONE_VOWEL = {"i": "yi", "u": "wu"}

_STRIP = set("ˈˌːˑ̩ʼ'‿")  # stress/length/syllabic marks carry no segment
_FLAG_RE = re.compile(r"\([a-z0-9-]+\)")  # espeak language-switch flags
_MULTI = sorted(
    (k for k in {*_VOWELS, *_INITIALS, *_ALONE} if len(k) > 1),
    key=len, reverse=True,
)
_TONE_LADDER = ("1", "2", "4", "3", "5", "")
_CACHE_CAP = 4096


def _parse(ipa: str) -> list[str]:
    ipa = _FLAG_RE.sub("", ipa)
    out: list[str] = []
    i = 0
    while i < len(ipa):
        if ipa[i] in _STRIP or ipa[i].isspace():
            i += 1
            continue
        for sym in _MULTI:
            if ipa.startswith(sym, i):
                out.append(sym)
                i += len(sym)
                break
        else:
            out.append(ipa[i])
            i += 1
    return out


class EnglishToPinyin:
    """Word -> zh token ids; falsy when the token set resolves no letter at all."""

    def __init__(
        self, token2id: dict[str, int], phonemize: Callable[[str], str] | None = None
    ):
        self._token2id = token2id
        self._phonemize = phonemize
        self._letters: dict[str, list[int]] = {}
        for letter, seq in _LETTERS.items():
            ids = [self._resolve(s, t) for s, t in seq]
            if None not in ids:
                self._letters[letter] = ids  # type: ignore[assignment]
        self._cache: dict[str, list[int]] = {}

    def __bool__(self) -> bool:
        return bool(self._letters)

    def _resolve(self, syl: str, tone: int) -> int | None:
        for t in (str(tone), *_TONE_LADDER):
            tid = self._token2id.get(syl + t)
            if tid is not None:
                return tid
        return None

    def word_ids(self, word: str) -> list[int]:
        """Token ids for one Latin word run; [] = nothing mappable."""
        ids = self._cache.get(word)
        if ids is None:
            ids = self._put(word, self._ids(word))
        return ids

    def _put(self, word: str, ids: list[int]) -> list[int]:
        if len(self._cache) >= _CACHE_CAP:
            self._cache.clear()
        self._cache[word] = ids
        return ids

    def _spell(self, word: str) -> list[int]:
        return [i for ch in word.lower() for i in self._letters.get(ch, ())]

    def prime(self, words: list[str]) -> None:
        """Batch-phonemize uncached words in ONE espeak call: subprocess espeak spawns per
        call, so an English clause would otherwise cost a process per word."""
        if self._phonemize is None:
            return
        fresh = list(dict.fromkeys(
            w for w in words if w not in self._cache and not w.isupper()
        ))
        if len(fresh) < 2:
            return
        try:
            lines = self._phonemize(BATCH_SEP.join(fresh)).splitlines()
        except Exception:  # noqa: BLE001 - the per-word path reports instead
            return
        if len(lines) != len(fresh):
            return  # espeak re-claused the batch: per-word calls stay correct
        for word, ipa in zip(fresh, lines):
            self._put(word, self._from_ipa(ipa) or self._spell(word))

    def _ids(self, word: str) -> list[int]:
        # All-caps reads as an acronym (USB, CPU); anything unmappable spells out.
        if not word.isupper() and self._phonemize is not None:
            ids = self._transliterate(word)
            if ids:
                return ids
        return self._spell(word)

    def _transliterate(self, word: str) -> list[int]:
        try:
            ipa = self._phonemize(word)  # type: ignore[misc]
        except Exception:  # noqa: BLE001 - espeak hiccup degrades to spelling
            return []
        return self._from_ipa(ipa)

    def _from_ipa(self, ipa: str) -> list[int]:
        """Pure IPA -> zh token ids, so ``prime`` can map a batched line."""
        phones = _parse(ipa)
        out: list[int] = []
        i = 0
        while i < len(phones):
            p = phones[i]
            if p in _VOWELS:
                final = _VOWELS[p]
                tid = self._resolve(_ALONE_VOWEL.get(final, final), 1)
                if tid is not None:
                    out.append(tid)
                i += 1
                continue
            onset = _INITIALS.get(p)
            nxt = phones[i + 1] if i + 1 < len(phones) else None
            if onset is not None and nxt in _VOWELS:
                candidates = []
                after = i + 2
                if after < len(phones) and phones[after] in ("n", "ŋ") and (
                    after + 1 >= len(phones) or phones[after + 1] not in _VOWELS
                ):
                    coda = "n" if phones[after] == "n" else "ng"
                    candidates.append((onset + _VOWELS[nxt] + coda, after + 1))
                candidates.append((onset + _VOWELS[nxt], after))
                for syl, resume in candidates:
                    tid = self._resolve(syl, 1)
                    if tid is not None:
                        out.append(tid)
                        i = resume
                        break
                else:
                    alone = _ALONE.get(p)
                    if alone is not None:
                        tid = self._resolve(*alone)
                        if tid is not None:
                            out.append(tid)
                    i += 1  # the vowel re-enters as a bare nucleus
                continue
            alone = _ALONE.get(p)
            if alone is not None:
                tid = self._resolve(*alone)
                if tid is not None:
                    out.append(tid)
            i += 1  # unknown IPA symbols are skipped
        return out
