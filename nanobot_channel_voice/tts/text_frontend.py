"""Pluggable TTS text frontends: per-language normalization run before the char
tokenizer of a char-input TTS model (e.g. MMS-TTS / VITS). Language deps import lazily,
so a missing one raises an error the registry turns into the system-TTS fallback.

* ``none``:     identity; Latin scripts need no romanization.
* ``uroman``:   romanize ANY script to Latin (the documented MMS path for non-Latin
                 languages; ``[uroman]`` extra). Context-free: good for kana,
                 imperfect for kanji.
* ``japanese``: ``pyopenjtalk`` reads kanji -> katakana, then ``uroman`` romanizes
                 the reading (``[japanese]`` extra).
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable


@runtime_checkable
class TextFrontend(Protocol):
    def normalize(self, text: str) -> str:
        """Transform text into the form the model's char tokenizer expects."""
        ...


class NoneFrontend:
    def normalize(self, text: str) -> str:
        return text


class UromanFrontend:
    def __init__(self) -> None:
        try:
            from uroman import Uroman
        except ImportError as e:
            raise RuntimeError(
                "text_frontend='uroman' needs the [uroman] extra: "
                "pip install 'nanobot-channel-voice[uroman]'"
            ) from e
        self._uroman = Uroman()

    def normalize(self, text: str) -> str:
        return self._uroman.romanize_string(text)


class JapaneseFrontend:
    def __init__(self) -> None:
        try:
            import pyopenjtalk
        except ImportError as e:
            raise RuntimeError(
                "text_frontend='japanese' needs the [japanese] extra: "
                "pip install 'nanobot-channel-voice[japanese]'"
            ) from e
        self._pyopenjtalk = pyopenjtalk
        self._roman = UromanFrontend()  # kana -> romaji; also asserts uroman present

    def normalize(self, text: str) -> str:
        kana = self._pyopenjtalk.g2p(text, kana=True)  # kanji -> katakana reading
        return self._roman.normalize(kana)


# ---- English number verbalization ------------------------------------------
#
# MMS char vocabs carry few or no digits (mms-tts-eng has "0"-"6", no 7, 8, 9!), so
# unverbalized numbers are silently mangled: "at 7:45" tokenizes to "4 5", corrupting
# exactly the highest-value content. English-only; other languages need their own
# verbalizer and rely on the shell's speakability-guard warning instead.

_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = ("", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety")
_SCALES = ((10**9, "billion"), (10**6, "million"), (10**3, "thousand"), (100, "hundred"))
_ORDINAL_IRREGULAR = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}


def _int_words(n: int) -> str:
    if n >= 10**12:  # phone-number/id territory: read the digits out
        return _digit_words(str(n))
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, rest = divmod(n, 10)
        return _TENS[tens] + (f" {_ONES[rest]}" if rest else "")
    for scale, name in _SCALES:
        if n >= scale:
            head, rest = divmod(n, scale)
            return f"{_int_words(head)} {name}" + (f" {_int_words(rest)}" if rest else "")
    raise AssertionError("unreachable")


def _digit_words(digits: str) -> str:
    return " ".join(_ONES[int(d)] for d in digits)


def _ordinal_words(n: int) -> str:
    words = _int_words(n)
    head, _, last = words.rpartition(" ")
    if last in _ORDINAL_IRREGULAR:
        last = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return f"{head} {last}".strip()


def _time_words(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    if minute == 0:
        return f"{_int_words(hour)} o'clock"
    if minute < 10:
        return f"{_int_words(hour)} oh {_int_words(minute)}"
    return f"{_int_words(hour)} {_int_words(minute)}"


_RE_TIME = re.compile(r"\b(\d{1,2}):([0-5]\d)\b")
_RE_GROUPED = re.compile(r"\b\d{1,3}(?:,\d{3})+\b")
_RE_DECIMAL = re.compile(r"\b(\d+)\.(\d+)\b")
_RE_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
_RE_PERCENT = re.compile(r"\b(\d+)\s*%")
_RE_INT = re.compile(r"\d+")


def verbalize_numbers_en(text: str) -> str:
    """Expand digits into English words a char-vocab TTS can actually speak."""
    text = _RE_TIME.sub(_time_words, text)
    text = _RE_GROUPED.sub(lambda m: _int_words(int(m.group().replace(",", ""))), text)
    text = _RE_DECIMAL.sub(
        lambda m: f"{_int_words(int(m.group(1)))} point {_digit_words(m.group(2))}", text
    )
    text = _RE_ORDINAL.sub(lambda m: _ordinal_words(int(m.group(1))), text)
    text = _RE_PERCENT.sub(lambda m: f"{_int_words(int(m.group(1)))} percent", text)
    return _RE_INT.sub(lambda m: _int_words(int(m.group())), text)


_FRONTENDS = {
    "none": NoneFrontend,
    "uroman": UromanFrontend,
    "japanese": JapaneseFrontend,
}


def make_text_frontend(name: str) -> TextFrontend:
    """Build a text frontend by name. Unknown name -> ValueError; missing optional
    dependency -> RuntimeError (the registry then falls back to system TTS).
    ``none`` never fails."""
    try:
        factory = _FRONTENDS[name]
    except KeyError:
        raise ValueError(
            f"unknown TTS text_frontend '{name}' (expected: {', '.join(_FRONTENDS)})"
        ) from None
    return factory()


__all__ = [
    "TextFrontend",
    "NoneFrontend",
    "UromanFrontend",
    "JapaneseFrontend",
    "make_text_frontend",
    "verbalize_numbers_en",
]
