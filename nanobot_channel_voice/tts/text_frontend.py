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


# ---- English number verbalization -------------------------------------------
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


# ---- Chinese number verbalization -------------------------------------------
#
# Same failure as English: the matcha zh lexicon voices 零..九 but has no "0".."9"
# entries, so unverbalized digits are dropped as OOV: exactly the highest-value content.

_ZH_DIGITS = "零一二三四五六七八九"
_ZH_SMALL_UNITS = ("", "十", "百", "千")
_ZH_GROUP_UNITS = ("", "万", "亿", "万亿")


def _zh_four(n: int) -> str:
    """0 < n < 10000 with the in-group 零 collapse (1005 -> 一千零五)."""
    out = ""
    zero_pending = False
    for i in (3, 2, 1, 0):
        d = n // 10**i % 10
        if d == 0:
            zero_pending = bool(out)
        else:
            if zero_pending:
                out += "零"
                zero_pending = False
            out += _ZH_DIGITS[d] + _ZH_SMALL_UNITS[i]
    return out


def _zh_int(n: int) -> str:
    if n == 0:
        return "零"
    if n >= 10**13:  # id/phone territory: read the digits out
        return _zh_digit_words(str(n))
    groups = []
    while n:
        groups.append(n % 10000)
        n //= 10000
    out = ""
    for i in reversed(range(len(groups))):
        g = groups[i]
        if g == 0:
            continue
        if out and g < 1000:  # a skipped group or leading zeros: 100000001 -> 一亿零一
            out += "零"
        out += _zh_four(g) + _ZH_GROUP_UNITS[i]
    if out.startswith("一十"):  # 12 -> 十二, but 112 keeps 一百一十二
        out = out[1:]
    return out


def _zh_digit_words(digits: str) -> str:
    return "".join(_ZH_DIGITS[int(d)] for d in digits)


def _zh_number(number: str) -> str:
    """"3" -> 三, "3.5" -> 三点五."""
    head, _, frac = number.partition(".")
    return _zh_int(int(head)) + (f"点{_zh_digit_words(frac)}" if frac else "")


def _zh_time_words(m: re.Match) -> str:
    hour, minute = int(m.group(1)), int(m.group(2))
    if minute == 0:
        return f"{_zh_int(hour)}点"
    pad = "零" if minute < 10 else ""
    return f"{_zh_int(hour)}点{pad}{_zh_int(minute)}分"


# \b never fires beside CJK (both \w): anchor on digit lookarounds instead. Percent
# is decimal-aware and runs BEFORE decimal or "3.5%" loses its integer part.
_RE_TIME_ZH = re.compile(r"(?<!\d)(\d{1,2})[:：]([0-5]\d)(?!\d)")
_RE_GROUPED_ZH = re.compile(r"(?<!\d)\d{1,3}(?:,\d{3})+(?!\d)")
_RE_PERCENT_ZH = re.compile(r"(?<!\d)(\d+(?:\.\d+)?)\s*[%％]")
_RE_DECIMAL_ZH = re.compile(r"(?<!\d)(\d+)\.(\d+)(?!\d)")
_RE_INT_ZH = re.compile(r"\d+")


def verbalize_numbers_zh(text: str) -> str:
    """Expand digits into Chinese words the matcha zh lexicon can actually speak."""
    text = _RE_TIME_ZH.sub(_zh_time_words, text)
    text = _RE_GROUPED_ZH.sub(lambda m: m.group().replace(",", ""), text)
    text = _RE_PERCENT_ZH.sub(lambda m: f"百分之{_zh_number(m.group(1))}", text)
    text = _RE_DECIMAL_ZH.sub(
        lambda m: f"{_zh_int(int(m.group(1)))}点{_zh_digit_words(m.group(2))}", text
    )
    return _RE_INT_ZH.sub(lambda m: _zh_int(int(m.group())), text)


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
    "verbalize_numbers_zh",
]
