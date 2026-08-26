"""Pluggable TTS text frontends: per-language normalization run before the char
tokenizer of a char-input TTS model (e.g. MMS-TTS / VITS). Language deps import lazily,
so a missing one raises an error the registry turns into the system-TTS fallback.

* ``none``:     identity; Latin scripts need no romanization.
* ``uroman``:   romanize ANY script to Latin (the documented MMS path; ``[uroman]``
                 extra). Context-free: good for kana, imperfect for kanji.
* ``japanese``: ``pyopenjtalk`` reads kanji -> katakana, then ``uroman`` romanizes
                 the reading (``[japanese]`` extra).
"""

from __future__ import annotations

import re
from collections.abc import Callable
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


# ---- rendering helpers ------------------------------------------------------

_RE_FUSABLE = re.compile(r"[0-9A-Za-z]")


def _pad(prev: str, nxt: str) -> str:
    """A space wherever a reading would fuse with what it sits against: "gate B12"
    must not read "Btwelve"."""
    return " " if _RE_FUSABLE.match(prev) and _RE_FUSABLE.match(nxt) else ""


def _sub_padded(pattern: re.Pattern[str], text: str, render: Callable[[str], str]) -> str:
    def _one(m: re.Match[str]) -> str:
        rep = render(m.group())
        return (_pad(text[m.start() - 1] if m.start() else "", rep[:1]) + rep
                + _pad(rep[-1:], text[m.end():m.end() + 1]))

    return pattern.sub(_one, text)


# ---- English number verbalization -------------------------------------------
# MMS char vocabs carry few or no digits (mms-tts-eng has "0"-"6", no 7, 8, 9!), so
# unverbalized numbers are silently mangled: "at 7:45" tokenizes to "4 5". English
# only; other languages need their own verbalizer.

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
        return en_digit_words(str(n))
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


def en_digit_words(digits: str) -> str:
    return " ".join(_ONES[int(d)] for d in digits)


def _en_run_words(run: str) -> str:
    """A cardinal never carries a leading zero: reading "007" as one deletes it."""
    return en_digit_words(run) if len(run) > 1 and run[0] == "0" else _int_words(int(run))


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


# Multi-dot runs are addresses and versions. The decimal rules match only the FIRST
# dot, so without this "192.168.1.1" keeps a literal "." the engine cannot voice.
_RE_DOTTED = re.compile(r"(?<!\d)\d+(?:\.\d+){2,}(?!\d)")
_RE_TIME = re.compile(r"\b(\d{1,2}):([0-5]\d)\b")
_GROUPED = r"\d{1,3}(?:,\d{3})+"  # the thousands-separator grammar, shared by every amount pattern
_RE_GROUPED = re.compile(rf"\b{_GROUPED}\b")
# Lookaround anchors, not \b: a glued unit ("3.5kg") is still a decimal, and \b
# between digit and letter never fires.
_RE_DECIMAL = re.compile(rf"(?<!\d)({_GROUPED}|\d+)\.(\d+)(?!\d)")
_RE_ORDINAL = re.compile(r"\b(\d+)(st|nd|rd|th)\b", re.IGNORECASE)
# Amount-aware like the zh twin, and run BEFORE decimal/grouped: those passes eat
# the digits and strand a "%" no char vocab can voice.
_RE_PERCENT = re.compile(rf"(?<!\d)(?:{_GROUPED}|\d+)(?:\.\d+)?\s*%")
_RE_INT = re.compile(r"\d+")


def _en_amount_words(amount: str) -> str:
    head, _, frac = amount.partition(".")
    words = _int_words(int(head.replace(",", "")))
    return f"{words} point {en_digit_words(frac)}" if frac else words


def _en_percent(text: str) -> str:
    return _en_amount_words(text.rstrip("%").strip()) + " percent"


def _dotted_words(
    number: str, point: str, digits: Callable[[str], str], run_words: Callable[[str], str]
) -> str:
    """Dotted parts joined by the spoken dot: a dotted quad is an address (digit-wise),
    anything else a version (parts are numbers)."""
    parts = number.split(".")
    ip = len(parts) == 4 and all(len(p) <= 3 and int(p) <= 255 for p in parts)
    return point.join((digits if ip else run_words)(p) for p in parts)


def verbalize_numbers_en(text: str) -> str:
    """Expand digits into English words a char-vocab TTS can actually speak."""
    if not _RE_INT.search(text):
        return text
    text = _RE_ISO_T.sub(" ", text)
    text = _sub_dates(text, _en_date_words)
    text = _sub_month_dates(text)
    text = _RE_EN_DECADE.sub(_en_decade, text)
    text = _fractions(_ranges(text, "en"), "en")
    text = _sub_padded(_RE_CURRENCY, text, _en_currency)
    text = _RE_DEGREES.sub(_en_degrees, text)
    text = _sub_padded(
        _RE_DOTTED, text,
        lambda run: _dotted_words(run, " point ", en_digit_words, _en_run_words),
    )
    text = _RE_TIME.sub(_time_words, text)
    text = _sub_padded(_RE_PERCENT, text, _en_percent)
    # Decimal before grouped, or "1,234.56" loses its integer part and strands a ".".
    text = _sub_padded(_RE_DECIMAL, text, _en_amount_words)
    text = _RE_GROUPED.sub(lambda m: _int_words(int(m.group().replace(",", ""))), text)
    text = _RE_ORDINAL.sub(lambda m: _ordinal_words(int(m.group(1))), text)
    text = _read_sequences(text, "en", en_digit_words, _en_year_words)
    return _sub_padded(_RE_INT, text, _en_run_words)


# ---- Chinese number verbalization -------------------------------------------
# Same failure as English: the matcha zh lexicon voices 零..九 but has no "0".."9"
# entries, so unverbalized digits are dropped as OOV.

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
        return zh_digit_words(str(n))
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


def zh_digit_words(digits: str) -> str:
    return "".join(_ZH_DIGITS[int(d)] for d in digits)


def _zh_run_words(run: str) -> str:
    """Leading-zero identifier, as ``_en_run_words``."""
    return zh_digit_words(run) if len(run) > 1 and run[0] == "0" else _zh_int(int(run))


def _zh_number(number: str) -> str:
    """"3" -> 三, "3.5" -> 三点五."""
    head, _, frac = number.partition(".")
    return _zh_int(int(head)) + (f"点{zh_digit_words(frac)}" if frac else "")


def _zh_time_words(m: re.Match) -> str:
    hour, minute, sec = int(m.group(1)), int(m.group(2)), m.group(3)
    pad = "零" if minute < 10 else ""
    if sec is not None:
        # Clock wording for a duration too: it stays intelligible, and the alternative
        # left the ":" to reach the model as a mid-number pause.
        return f"{_zh_int(hour)}点{pad}{_zh_int(minute)}分{_zh_int(int(sec))}秒"
    if minute == 0:
        return f"{_zh_int(hour)}点"
    return f"{_zh_int(hour)}点{pad}{_zh_int(minute)}分"


# \b never fires beside CJK (both \w): anchor on digit lookarounds instead. Percent
# is decimal-aware and runs BEFORE decimal or "3.5%" loses its integer part.
_RE_TIME_ZH = re.compile(r"(?<!\d)(\d{1,2})[:：]([0-5]\d)(?:[:：]([0-5]\d))?(?!\d)")
_RE_GROUPED_ZH = re.compile(rf"(?<!\d){_GROUPED}(?!\d)")
_RE_PERCENT_ZH = re.compile(rf"(?<!\d)((?:{_GROUPED}|\d+)(?:\.\d+)?)\s*[%％]")
_RE_DECIMAL_ZH = re.compile(rf"(?<!\d)({_GROUPED}|\d+)\.(\d+)(?!\d)")
# A year reads digit-wise in Mandarin (2026年 = 二零二六年, never 两千零二十六年), bounded
# to 1000-2999 (the residue is a round duration, "共2000年"). A range's head carries no
# 年 of its own (2020至2024年), so the lookahead reaches through the connective.
_RE_YEAR_ZH = re.compile(r"(?<!\d)[12]\d{3}(?=年|[至到~～][12]\d{3}年)")
# Generation labels read digit-wise: 90后/85后 are 九零后/八五后, never 九十后.
# 后面/后边/后方/后头 mark position ("排在10后面"), not a cohort.
_RE_ZH_GEN = re.compile(r"(?<!\d)\d[05](?=后(?![面边方头]))")
# "08月" is a date field: drop the pad before the leading-zero sequence rule claims it
# (零八月). Only 月/日 are safe — 号 marks an identifier (订单08号) as often as a day,
# and 分/秒 keep their 零 (三点零五分).
_RE_ZH_PADDED_DATE = re.compile(r"(?<!\d)0([1-9])(?=[月日])")
_RE_INT_ZH = re.compile(r"\d+")


_ZH_VALUE = {ch: i for i, ch in enumerate(_ZH_DIGITS)} | {"两": 2}


def zh_numeral_value(run: str) -> int | None:
    """Value of a zh numeral run (四十五 -> 45, 两百 -> 200); with no unit character it
    reads positionally (一二三 -> 123). None when a character is not numeral material."""
    if not run or any(ch not in _ZH_VALUE and ch not in "十百千万亿" for ch in run):
        return None
    if all(ch in _ZH_VALUE for ch in run):
        return int("".join(str(_ZH_VALUE[ch]) for ch in run))
    total = section = num = 0
    for ch in run:
        if ch in _ZH_VALUE:
            num = _ZH_VALUE[ch]
        elif ch == "万":  # closes its own section only (一亿零五万 keeps the 亿)
            total += ((section + num) or 1) * 10000
            section = num = 0
        elif ch == "亿":  # 万亿 composes, so 亿 scales everything accumulated
            total = ((total + section + num) or 1) * 10**8
            section = num = 0
        else:
            section += (num or 1) * {"十": 10, "百": 100, "千": 1000}[ch]
            num = 0
    return total + section + num


def verbalize_numbers_zh(text: str) -> str:
    """Expand digits into Chinese words the matcha zh lexicon can actually speak."""
    if not _RE_INT_ZH.search(text):
        return text
    text = _RE_ISO_T.sub(" ", text)
    text = _sub_dates(text, _zh_date)
    text = _RE_ZH_PADDED_DATE.sub(r"\1", text)
    text = _fractions(_ranges(text, "zh"), "zh")
    text = _sub_padded(_RE_CURRENCY, text, _zh_currency)
    text = _RE_DEGREES.sub(_zh_degrees, text)
    text = _sub_padded(
        _RE_DOTTED, text,
        lambda run: _dotted_words(run, "点", zh_digit_words, _zh_run_words),
    )
    text = _RE_TIME_ZH.sub(_zh_time_words, text)
    text = _RE_PERCENT_ZH.sub(
        lambda m: f"百分之{_zh_number(m.group(1).replace(',', ''))}", text
    )
    text = _RE_DECIMAL_ZH.sub(
        lambda m: f"{_zh_int(int(m.group(1).replace(',', '')))}点{zh_digit_words(m.group(2))}",
        text,
    )
    # To WORDS, after the amount passes and before the sequence pass: dropping the
    # separators alone leaves a 7-digit run that _SEQ_MIN_BARE claims as an id.
    text = _RE_GROUPED_ZH.sub(lambda m: _zh_int(int(m.group().replace(",", ""))), text)
    # Sequences first: _RE_YEAR_ZH would eat the tail of "2020-2024年" and strand the head.
    text = _read_sequences(text, "zh", zh_digit_words)
    text = _RE_YEAR_ZH.sub(lambda m: zh_digit_words(m.group()), text)
    text = _RE_ZH_GEN.sub(lambda m: zh_digit_words(m.group()), text)
    return _sub_padded(_RE_INT_ZH, text, _zh_run_words)


# ---- dates, currency and degrees ---------------------------------------------
# Dates are ISO order only: DD-MM and MM-DD are ambiguous with each other and with
# id fragments.

_HYPHENS = r"\-‑–"  # escaped: it is interpolated into character classes
_MONTHS_EN = ("January", "February", "March", "April", "May", "June", "July",
              "August", "September", "October", "November", "December")
_MONTH_DAYS = (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)  # Feb 29 allowed blind
# Same separator both times; a trailing 日/号 means the writer already suffixed the date,
# which the renderers would double. "." is a separator too (2026.8.19), so a year-based
# version (2024.1.2) reads as a date; ".digit" after the day bails out.
_RE_ISO_DATE = re.compile(
    rf"(?<![\d{_HYPHENS}/.])[12]\d{{3}}([{_HYPHENS}/.])(?:0?[1-9]|1[0-2])\1"
    rf"(?:0?[1-9]|[12]\d|3[01])(?![\d{_HYPHENS}/日号]|\.\d)"
)
# The ISO timestamp's T reads as a letter and glues the time shut ("19T21:33").
_RE_ISO_T = re.compile(r"(?<=\d)T(?=\d{1,2}:\d{2})")
# symbol -> (en singular, en plural, zh); the regex class derives from the keys so they
# cannot drift. ¥ reads as CNY.
_CURRENCIES = {
    "$": ("dollar", "dollars", "美元"), "€": ("euro", "euros", "欧元"),
    "£": ("pound", "pounds", "英镑"), "¥": ("yuan", "yuan", "元"),
    "￥": ("yuan", "yuan", "元"),
}
# A scale word belongs in front of the relocated unit ($5 million -> 5 million dollars).
# Shapes the pass cannot own — glued suffix ($100k), range or time tail ($20-30, $12:30),
# 多/几万, explicit unit (¥199元) — fall through. The digit in the guard stops
# backtracking onto a prefix ($1,234k vs $1).
_RE_CURRENCY = re.compile(
    rf"[{''.join(_CURRENCIES)}]\s?(?:{_GROUPED}|\d+)(?:\.\d+)?"
    rf"(?:\s?(?:thousand|million|billion|trillion)\b|[万亿]+)?"
    rf"(?!\d|[A-Za-z]|[{_HYPHENS}:：,，]\d|[多几][万亿]"
    rf"|\s?(?:元|(?:dollars?|euros?|pounds?|yuan|yen|thousand|million|billion|trillion)\b))",
    re.IGNORECASE,
)
# The minus survives only here: elsewhere a hyphen marks a range or id as often as a
# sign, and the sign is OOV in every lexicon path. The lookbehinds keep "20-30°C" a
# range and "wind-3°C" a compound.
_RE_DEGREES = re.compile(
    r"(?<!\d)(?:(?<![A-Za-z])([-−])\s?)?(\d+(?:\.\d+)?)\s*(?:°\s?[CF](?![A-Za-z])|℃|℉)",
    re.IGNORECASE,
)
# The two-codepoint spellings _RE_DEGREES accepts, foldable to the single codepoints
# (non-alpha, so a script classifier keeps them neutral).
_RE_DEGREE_MARK = re.compile(r"(?<=\d)(\s?)°\s?([CF])(?![A-Za-z])", re.IGNORECASE)
_RE_ONE = re.compile(r"1(?:\.0+)?")  # the amounts that read singular


def _date_parts(date: str) -> tuple[str, int, int]:
    year, month, day = date.split(date[4])  # the separator always follows a 4-digit year
    return year, int(month), int(day)


def _sub_dates(text: str, render: Callable[[str], str]) -> str:
    """``_sub_padded`` over ISO dates, plus the calendar check the regex cannot do: an
    impossible day (2026-02-30) is an identifier and keeps its sequence reading."""

    def _one(m: re.Match[str]) -> str:
        _, month, day = _date_parts(m.group())
        if day > _MONTH_DAYS[month - 1]:
            return m.group()
        rep = render(m.group())
        return (_pad(text[m.start() - 1] if m.start() else "", rep[:1]) + rep
                + _pad(rep[-1:], text[m.end():m.end() + 1]))

    return _RE_ISO_DATE.sub(_one, text)


def _zh_date(date: str) -> str:
    year, month, day = _date_parts(date)
    return f"{zh_digit_words(year)}年{_zh_int(month)}月{_zh_int(day)}日"


def _en_date_words(date: str) -> str:
    year, month, day = _date_parts(date)
    return f"{_MONTHS_EN[month - 1]} {_ordinal_words(day)}, {_en_year_words(year)}"


# Written-out dates — the shape "what's the date" answers take. Case-sensitive:
# "may 15 minutes" and "march 5 miles" are verbs.
_MONTH_ALT = "|".join(_MONTHS_EN)
_RE_EN_MONTH_DAY = re.compile(
    rf"\b({_MONTH_ALT})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+([12]\d{{3}})\b)?"
    rf"(?!\d|[:：]\d)"  # "May 12:30" is a month and a clock, not a day
)
_RE_EN_MONTH_YEAR = re.compile(rf"\b({_MONTH_ALT}),?\s+([12]\d{{3}})\b(?![\d:：])")
# A bare 4-digit decade is unambiguous; a 2-digit one needs a determiner or "30s" could
# be thirty seconds. The possessive ("1990's hits") is a spoken homophone of the plural.
_RE_EN_DECADE = re.compile(
    r"\b(?:([12]\d{2}0)|(?:the|his|her|their|your|my|our|early|late|mid)"
    r"[ \-]['’]?([2-9]0))['’]?s\b"
)


def _en_decade(m: re.Match[str]) -> str:
    run = m.group(1) or m.group(2)
    words = _en_year_words(run) if len(run) == 4 else _TENS[int(run) // 10]
    plural = words[:-1] + "ies" if words.endswith("y") else words + "s"
    return plural if m.group(1) else m.group()[: m.start(2) - m.start()] + plural


def _sub_month_dates(text: str) -> str:
    """"August 19(, 2026)" and "August(,) 2026": ordinal day, spoken year. A trailing
    unit ("May 5 minutes") or an impossible day keeps the quantity reading."""

    def _day(m: re.Match[str]) -> str:
        day = int(m.group(2))
        if not 1 <= day <= _MONTH_DAYS[_MONTHS_EN.index(m.group(1))]:
            return m.group()
        if m.group(3) is None and _RE_EN_UNIT.match(text[m.end():]):
            return m.group()
        year = f", {_en_year_words(m.group(3))}" if m.group(3) else ""
        return f"{m.group(1)} {_ordinal_words(day)}{year}"

    def _year(m: re.Match[str]) -> str:
        if _RE_EN_UNIT.match(text[m.end():]):
            return m.group()
        return f"{m.group(1)} {_en_year_words(m.group(2))}"

    text = _RE_EN_MONTH_DAY.sub(_day, text)
    return _RE_EN_MONTH_YEAR.sub(_year, text)


# Trailing fraction zeros are price formatting, not speech: $5.00 is five dollars.
_RE_FRAC_TAIL = re.compile(r"(\.\d*[1-9])0+(?!\d)")  # 3.50 -> 3.5
_RE_FRAC_ZEROS = re.compile(r"\.0+(?!\d)")           # 5.00 -> 5


def _trim_frac_zeros(amount: str) -> str:
    return _RE_FRAC_ZEROS.sub("", _RE_FRAC_TAIL.sub(r"\1", amount))


def _en_currency(text: str) -> str:
    amount = _trim_frac_zeros(text[1:].lstrip())
    singular, plural, _ = _CURRENCIES[text[0]]
    return f"{amount} {singular if _RE_ONE.fullmatch(amount) else plural}"


def _zh_currency(text: str) -> str:
    return _trim_frac_zeros(text[1:].lstrip()) + _CURRENCIES[text[0]][2]


def _en_degrees(m: re.Match[str]) -> str:
    unit = "degree" if _RE_ONE.fullmatch(m.group(2)) else "degrees"
    scale = "Fahrenheit" if m.group()[-1] in "Ff℉" else "Celsius"
    return f"{'minus ' if m.group(1) else ''}{m.group(2)} {unit} {scale}"


def _zh_degrees(m: re.Match[str]) -> str:
    scale = "华氏度" if m.group()[-1] in "Ff℉" else "摄氏度"
    return ("零下" if m.group(1) else "") + m.group(2) + scale


def fold_degree_marks(text: str) -> str:
    """``25°C``/``25° F`` -> ``25℃``/``25℉``, the single-codepoint twins: one degree
    grammar for callers that must not split the scale letter from its number."""
    return _RE_DEGREE_MARK.sub(
        lambda m: m.group(1) + ("℉" if m.group(2) in "Ff" else "℃"), text
    )


# ---- sequence detection ------------------------------------------------------
# A digit run reads as a SEQUENCE, not a quantity, when the surface form carries evidence
# a cardinal cannot produce or a trigger word names it. A miss is the cardinal reading.

_SEQ_MIN_BARE = 7  # conversational text does not state a 7-digit quantity ungrouped
_SEQ_MIN_GLUE = 3  # shorter and letter-glued is a model name: COVID-19, gate B12
_CTX_LEFT = 40     # trigger window, wide enough for four English words
_CTX_RIGHT = 16    # unit window, wide enough for "kilometres"

_RE_RUN = re.compile(r"\d+")
# Group spans cover their separators, so "555-1234" loses the hyphen instead of voicing it.
_RE_HYPHEN_GROUP = re.compile(rf"(?<![\d{_HYPHENS}])\d+(?:[{_HYPHENS}]\d+)+(?![\d{_HYPHENS}])")
_RE_INTL_PHONE = re.compile(rf"\+\d{{1,3}}(?:[{_HYPHENS} \u00a0]\d{{2,4}}){{2,}}")
_RE_YEAR_RANGE = re.compile(rf"^[12]\d{{3}}[{_HYPHENS}][12]\d{{3}}$")
# Only cues that mean a year far more often than not: "a value of 1234" is not one.
_RE_EN_YEAR_CUE = re.compile(r"\b(?:in|since|the year)\s+$", re.I)
_RE_GLUE_L = re.compile(r"[A-Za-z]$")
_RE_GLUE_R = re.compile(r"^[A-Za-z]")
_RE_WORD = re.compile(r"[A-Za-z]+")

_EN_TRIGGERS = frozenset("""
phone telephone mobile cell call calling dial fax hotline extension ext
zip zipcode zipcodes postcode postcodes pin otp passcode password digits
code confirmation account routing iban swift card
order invoice receipt tracking reference ref ticket case claim policy
flight room apartment apt door
serial sku isbn vin imei mac badge id identifier number num
licence license plate port status error iso
member membership employee student patient record file batch lot
""".split())
_EN_TRIGGER_PHRASES = ("area code", "postal code", "zip code", "verification code",
                       "order number", "phone number", "account number", "case number")
# A unit or counter after the run pins it to a quantity; checked before every positive
# rule, so "1000000 residents" survives the length rule.
_RE_EN_UNIT = re.compile(r"""^\s*(?:percent|dollars?|euros?|pounds?|cents?|yen|yuan|
    items?|records?|files?|people|persons?|users?|times?|copies|pages?|
    seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|weeks?|months?|years?|degrees?|
    kilomet(?:er|re)s?|km|met(?:er|re)s?|m|miles?|feet|ft|inches|
    kilograms?|kg|grams?|g|lbs?|lit(?:er|re)s?|l|ml|pixels?|bytes?|[kmgt]b|
    residents?|employees?|options?|calories)\b""", re.I | re.X)

_ZH_TRIGGERS = ("打 拨 拨打 电话 手机 座机 号码 传真 热线 分机 区号 邮编 邮政编码 验证码 "
                "确认码 密码 口令 动态码 账号 账户 卡号 银行卡 尾号 后四位 订单号 订单 单号 "
                "快递 运单 工号 学号 编号 序号 序列号 编码 身份证 护照 房间 房号 门牌 座位 "
                "车位 航班 车次 车牌 牌照 端口 状态码 错误码 型号 批号 档案 病历 会员").split()
_ZH_UNITS = ("元 块 角 美元 欧元 个 只 条 张 台 部 件 份 位 人 名 家 次 遍 页 章 岁 年 月 日 "
             "天 小时 分钟 秒 周 度 米 公里 千米 厘米 毫米 里 克 千克 公斤 吨 升 毫升 斤 两 "
             "倍 成 折 分 平方米 亿 万 种 款 层 楼 步").split()
_RE_ZH_TRIGGER = re.compile("(?:" + "|".join(_ZH_TRIGGERS) + r")[^0-9]{0,6}$")
# Longest-first, or 公里 loses its first character to 里. The Latin branch mirrors
# _RE_EN_UNIT's: zh writes "512MB" too, and without it _is_sequence's glue rule reads those
# digit-wise. Two letters minimum, so 1080p/4K/5G stay identifiers.
_RE_ZH_UNIT = re.compile(
    r"^\s*(?:(?:" + "|".join(sorted(_ZH_UNITS, key=len, reverse=True)) + r")"
    r"|(?:[kmg]?hz|[kmgt]b|fps|dpi|bpm|km|cm|mm|kg|mg|ml|ms|kw)(?![a-z]))",
    re.I,
)

# Neither separator is silent: espeak names them ("five dash ten", "one slash two") and the
# zh lexicon drops them, fusing "5-10分钟" into 五十分钟. "\s*%" is ONE optional unit — a bare
# trailing "\s*" would eat the space before the unit ("5 to 10minutes").
_RE_RANGE = re.compile(
    rf"(?<![\d{_HYPHENS}.])(\d+(?:\.\d+)?(?:\s*%)?)\s*[{_HYPHENS}~～]\s*"
    rf"(\d+(?:\.\d+)?(?:\s*%)?)(?![\d.])"
)
# Denominators a reader says as a fraction; 24/7, 16/9 and a bare M/D date keep their
# literal reading rather than inventing "twenty fifths".
_FRACTION_DENOMS = frozenset((2, 3, 4, 5, 6, 8, 10, 16))
_RE_FRACTION = re.compile(r"(?<![\d./])(\d{1,2})/(\d{1,2})(?![\d/])")


def _ranges(text: str, lang: str) -> str:
    """Give a quantity range its connective, unit-anchored exactly as the sequence pass
    is: with nothing behind it a hyphen run is as often an id."""
    unit = _RE_ZH_UNIT if lang == "zh" else _RE_EN_UNIT
    to = "到" if lang == "zh" else " to "

    def connect(m: re.Match[str]) -> str:
        lo, hi = m.group(1), m.group(2)
        if not (lo.endswith("%") or unit.match(text[m.end():m.end() + _CTX_RIGHT])):
            return m.group()
        if lo.isdigit() and len(lo) == 4 and hi.isdigit() and len(hi) <= 2:
            return m.group()  # "2026-8月": a date shape the date pass did not claim
        return f"{lo}{to}{hi}"

    return _RE_RANGE.sub(connect, text)


def _fractions(text: str, lang: str) -> str:
    """Common fractions to words; a proper fraction only, so "16/9" stays an aspect ratio."""
    def read(m: re.Match[str]) -> str:
        num, den = int(m.group(1)), int(m.group(2))
        if den not in _FRACTION_DENOMS or not 0 < num < den:
            return m.group()
        if lang == "zh":
            return f"{_zh_int(den)}分之{_zh_int(num)}"
        name = {2: "half", 4: "quarter"}.get(den) or _ordinal_words(den)
        return f"{_int_words(num)} {name}{'s' if num > 1 else ''}"

    return _RE_FRACTION.sub(read, text)


def _en_triggered(left: str) -> bool:
    """Whole-token match over the last four words: as a substring, "phone" would make
    every iPhone a phone number."""
    tail = [t.lower() for t in _RE_WORD.findall(left)[-4:]]
    return bool(_EN_TRIGGERS.intersection(tail)) or any(
        phrase in " ".join(tail) for phrase in _EN_TRIGGER_PHRASES
    )


def _is_sequence(run: str, left: str, right: str, lang: str) -> bool:
    """Positive evidence only; the caller owns the unit guard."""
    if len(run) > 1 and run[0] == "0":  # also the espeak path's only leading-zero rule
        return True
    if len(run) >= _SEQ_MIN_GLUE and (_RE_GLUE_L.search(left) or _RE_GLUE_R.match(right)):
        return True
    if lang == "zh" and _RE_ZH_TRIGGER.search(left):
        return True
    if _en_triggered(left):  # triggers are additive: a zh sentence still says ISO or PIN
        return True
    return len(run) >= _SEQ_MIN_BARE


def _en_year_words(run: str) -> str:
    """1999 -> nineteen ninety nine, 2008 -> two thousand eight, 1900 -> nineteen hundred."""
    n = int(run)
    if n % 100 == 0:
        return f"{_int_words(n // 100)} hundred" if n % 1000 else _int_words(n)
    hi, lo = divmod(n, 100)
    if hi % 10 == 0 and lo < 10:
        return _int_words(n)
    return f"{_int_words(hi)} {'oh ' if lo < 10 else ''}{_int_words(lo)}"


def _en_year_split(run: str) -> str:
    """Engine-native year: espeak reads "19 99" as nineteen ninety nine. An unsplittable
    tail keeps the engine's own cardinal."""
    return f"{run[:2]} {run[2:]}" if int(run) % 100 >= 10 else run


def _sequence_spans(text: str, lang: str) -> list[tuple[int, int, str, list[str]]]:
    """(start, end, kind, parts) per group read as a sequence, non-overlapping and in text
    order; ``kind`` is "digits", "year" or "year_range"."""
    unit = _RE_ZH_UNIT if lang == "zh" else _RE_EN_UNIT
    spans: list[tuple[int, int, str, list[str]]] = []
    for g in sorted((*_RE_INTL_PHONE.finditer(text), *_RE_HYPHEN_GROUP.finditer(text)),
                    key=lambda g: (g.start(), -g.end())):
        if spans and g.start() < spans[-1][1]:  # "+1-415-555-2671" matches both patterns
            continue
        parts = _RE_RUN.findall(g.group())
        if _RE_YEAR_RANGE.match(g.group()):
            spans.append((*g.span(), "year_range", parts))  # "2020-2024年" keeps its 年
        elif unit.match(text[g.end():g.end() + _CTX_RIGHT]):
            continue  # "5-10 minutes" is a quantity range: the same guard bare runs get
        elif (g.group()[0] == "+" or len(parts) > 2 or len({len(p) for p in parts}) > 1
                or sum(map(len, parts)) >= _SEQ_MIN_BARE):
            spans.append((*g.span(), "digits", parts))
    claimed = [(s, e) for s, e, _, _ in spans]
    for m in _RE_RUN.finditer(text):
        s, e = m.span()
        if any(a <= s < b for a, b in claimed):
            continue
        left, right = text[max(0, s - _CTX_LEFT):s], text[e:e + _CTX_RIGHT]
        if unit.match(right):
            continue
        # Sequence evidence outranks the year cue: "the account ending in 1234" is an account.
        if _is_sequence(m.group(), left, right, lang):
            spans.append((s, e, "digits", [m.group()]))
        elif (lang == "en" and _RE_EN_YEAR_CUE.search(left)
                and len(m.group()) == 4 and 1000 <= int(m.group()) <= 2999):
            spans.append((s, e, "year", [m.group()]))
    return sorted(spans)


def _read_sequences(
    text: str, lang: str, digits: Callable[[str], str],
    year: Callable[[str], str] | None = None,
) -> str:
    """Replace each sequence span with its reading; unclaimed runs fall through to the
    cardinal pass. ``year`` defaults to ``digits``: Chinese years are read digit-wise."""
    year = year or digits
    join, to = ("", "到") if lang == "zh" else (", ", " to " if lang == "en" else ", ")
    out: list[str] = []
    last = 0
    for s, e, kind, parts in _sequence_spans(text, lang):
        if kind == "digits":
            rep = join.join(digits(part) for part in parts)
        elif kind == "year":
            rep = year(parts[0])
        else:
            rep = to.join(year(part) for part in parts)
        out += [text[last:s], _pad(text[s - 1] if s else "", rep[:1]),
                rep, _pad(rep[-1:], text[e:e + 1])]
        last = e
    out.append(text[last:])
    return "".join(out)


def space_digit_sequences(text: str, language: str | None = "en") -> str:
    """Sequences re-spaced into single digits, for an engine that owns its own number
    grammar (espeak names spaced digits in every voice). English also renders dates,
    currency and degrees to words — dates fully, so the sequence pass cannot re-shred
    them; other languages keep the language-neutral spacing."""
    if not _RE_INT.search(text):
        return text
    text = _RE_ISO_T.sub(" ", text)
    en = bool(language and language.startswith("en"))
    if en:
        text = _sub_dates(text, _en_date_words)
        text = _sub_month_dates(text)
        text = _RE_EN_DECADE.sub(_en_decade, text)
        text = _fractions(_ranges(text, "en"), "en")
        text = _sub_padded(_RE_CURRENCY, text, _en_currency)
        text = _RE_DEGREES.sub(_en_degrees, text)
    return _read_sequences(
        text, "en" if en else "", lambda run: " ".join(run), _en_year_split if en else None
    )


_FRONTENDS = {
    "none": NoneFrontend,
    "uroman": UromanFrontend,
    "japanese": JapaneseFrontend,
}


def make_text_frontend(name: str) -> TextFrontend:
    """Build a text frontend by name. Unknown name -> ValueError; missing optional dep ->
    RuntimeError (the registry then falls back to system TTS). ``none`` never fails."""
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
    "space_digit_sequences",
    "verbalize_numbers_zh",
    "en_digit_words",
    "zh_digit_words",
    "zh_numeral_value",
]
