"""Turn streamed reply text into speakable chunks.

Fed the agent's ``_stream_delta`` incrementally, it emits complete sentences as
soon as they form so TTS can start before the full answer arrives. Long
sentences split at clause punctuation past ``min_chars``; runaways force-split
at ``max_chars``. Markdown is stripped so TTS does not read ``asterisk``.
"""

from __future__ import annotations

import re

# ASCII terminators need a following separator to count (so "3.14" and
# mid-stream tokens are not split); CJK terminators stand alone.
_ASCII_TERM = ".!?…"
_CJK_TERM = "。！？"
_AFTER_TERM = "\"')]}»”’"
_SECONDARY = ",;:，、；："

# Line-start anchored (<=3 spaces of indent), the same rule as the streaming fence
# drop: two ``` runs mid-sentence are prose, and an unanchored regex would eat the
# words between.
_RE_FENCE = re.compile(r"(?m)^[ \t]{0,3}```[\s\S]*?```")
_RE_INLINE_CODE = re.compile(r"`([^`]*)`")
_RE_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_RE_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_RE_EMPHASIS = re.compile(r"([*_~]{1,3})(\S(?:.*?\S)?)\1")
_RE_HEADER = re.compile(r"(?m)^\s{0,3}#{1,6}\s*")
_RE_BULLET = re.compile(r"(?m)^\s{0,3}[-*+]\s+")
_RE_QUOTE = re.compile(r"(?m)^\s{0,3}>\s?")
_RE_WS = re.compile(r"[ \t]+")

# Curly quotes -> ASCII BEFORE the whitelist below, which would drop them.
_SMART_PUNCT = str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"'})
# Speakable whitelist: Unicode word chars (CJK/Cyrillic survive), whitespace and
# punctuation a TTS front-end can voice or pause on. Everything else (emoji,
# arrows, box drawing, missed markdown) becomes a space (never glue neighbors)
# and collapses in _RE_WS; espeak and MMS VITS read raw codepoint names aloud.
_RE_UNSPEAKABLE = re.compile(
    r"[^\w\s.,;:!?'\"()\-/%&+=@°$€£¥₹₽¢…。，！？、；：「」『』（）《》〈〉・·—–]"
)


def sanitize(text: str) -> str:
    """Best-effort markdown -> plain speakable text (+ speakable-charset pass)."""
    text = _RE_FENCE.sub(" ", text)
    text = _RE_INLINE_CODE.sub(r"\1", text)
    text = _RE_IMAGE.sub(" ", text)
    text = _RE_LINK.sub(r"\1", text)
    text = _RE_EMPHASIS.sub(r"\2", text)
    text = _RE_HEADER.sub("", text)
    text = _RE_BULLET.sub("", text)
    text = _RE_QUOTE.sub("", text)
    text = text.replace("|", " ")
    text = text.translate(_SMART_PUNCT)
    text = _RE_UNSPEAKABLE.sub(" ", text)
    text = _RE_WS.sub(" ", text)
    return text


def _primary_cut(buf: str) -> int:
    """Index of the first sentence boundary, or -1."""
    for i, ch in enumerate(buf):
        if ch == "\n" or ch in _CJK_TERM:
            return i
        if ch in _ASCII_TERM:
            nxt = buf[i + 1] if i + 1 < len(buf) else ""
            if nxt and (nxt.isspace() or nxt in _AFTER_TERM):
                return i
    return -1


def _secondary_cut(buf: str, min_chars: int) -> int:
    """Index of the first clause boundary at/after ``min_chars``, or -1. A
    separator BETWEEN digits (1,902,567 / 7:45) is number punctuation, not a
    clause: cutting there mangles the reading and can strand a digits-only
    chunk on the wrong bilingual engine. Digit+separator at the buffer END is
    the same case mid-arrival, so it waits for the next delta to disambiguate
    (mirrors the primary cut's trailing-terminator hold)."""
    for i in range(min_chars - 1, len(buf)):
        if buf[i] not in _SECONDARY:
            continue
        if i > 0 and buf[i - 1].isdigit() and (i + 1 == len(buf) or buf[i + 1].isdigit()):
            continue
        return i
    return -1


def _soft_cut(buf: str, limit: int) -> int:
    """Index to break a runaway at: last space before ``limit``, else the cap."""
    window = buf[:limit]
    space = window.rfind(" ")
    return space if space > 0 else limit - 1


class SentenceChunker:
    def __init__(self, min_chars: int = 60, max_chars: int = 240, min_chars_first: int | None = None):
        self._min = max(1, min_chars)
        self._max = max(self._min, max_chars)
        # The FIRST chunk of a turn may cut at an earlier clause boundary (nothing
        # plays until it exists, so its floor is the TTFA knob); later chunks keep
        # min_chars for prosody. Clamped to min_chars.
        self._first_min = min(self._min, max(1, min_chars_first)) if min_chars_first else self._min
        self._spoke = False  # a chunk was emitted since the last flush()
        self._buf = ""
        # Fenced code is dropped INCREMENTALLY, before cutting: the "\n" primary cut
        # splits a streamed fence long before sanitize()'s whole-fence regex matches.
        self._raw = ""  # not yet classified
        self._in_fence = False  # fence parity carried across deltas
        self._prev_char = ""  # last classified char; "" = start of message

    def set_first_floor(self, chars: int) -> None:
        """Adjust the first-chunk clause floor (device-speed calibration): slow
        TTS needs a LARGER first chunk so its playback covers the next chunk's
        synthesis (no mid-reply gap); fast TTS can cut early for TTFA."""
        self._first_min = min(self._min, max(1, chars))

    def feed(self, delta: str) -> list[str]:
        if delta:
            self._raw += delta
            self._buf += self._drain_raw()
        chunks: list[str] = []
        while True:
            cut = self._next_cut()
            if cut is None:
                break
            piece, self._buf = self._buf[: cut + 1], self._buf[cut + 1 :]
            text = sanitize(piece).strip()
            if text:
                chunks.append(text)
                self._spoke = True
        return chunks

    def flush(self) -> str | None:
        """Return any buffered remainder (call at stream end)."""
        self._spoke = False  # next turn's chunk 1 gets the first-chunk floor again
        self._buf += self._drain_raw()
        # Held-back trailing backticks are plain text at stream end; an unclosed
        # fence means the rest of the message was code; drop it.
        if not self._in_fence:
            self._buf += self._raw
        self._raw = ""
        self._in_fence = False
        self._prev_char = ""
        text = sanitize(self._buf).strip()
        self._buf = ""
        return text or None

    def _line_start_at(self, text: str, i: int) -> bool:
        """Is position ``i`` at a line start, allowing the up-to-3 spaces of indent
        CommonMark permits for a fence opener (list items!) and CR for CRLF?
        Falls back to ``_prev_char`` when the lookback runs off the delta."""
        j, spaces = i, 0
        while j > 0 and text[j - 1] == " " and spaces < 3:
            j -= 1
            spaces += 1
        if j > 0:
            return text[j - 1] in ("\n", "\r")
        return self._prev_char in ("", "\n", "\r")

    def _update_prev(self, text: str) -> None:
        """Record the classification context at *text*'s end, collapsing a trailing
        newline+indent run to "\\n" so an opener split across deltas still sees it."""
        if text:
            self._prev_char = "\n" if self._line_start_at(text, len(text)) else text[-1]

    def _drain_raw(self) -> str:
        """Move classified text out of ``_raw``, dropping fenced-code content.
        Consumes up to the last complete ```` ``` ```` marker; a trailing run of
        1-2 backticks is held back, since the next delta may complete it. Info
        string ("```python") and body are inside the fence and dropped with it;
        a closed fence collapses to one space so words are never glued together.
        """
        out: list[str] = []
        raw = self._raw
        search_from = 0
        while True:
            i = raw.find("```", search_from)
            if i == -1:
                break
            if self._in_fence:
                # Closing marker accepted anywhere: demanding a line start would
                # mute the rest of the reply on a sloppy close.
                out.append(" ")
                self._in_fence = False
                self._prev_char = "`"
                raw = raw[i + 3 :]
                search_from = 0
                continue
            if not self._line_start_at(raw, i):
                # Backticks mid-sentence ("wrap it in ``` fences") are prose, not an
                # opener: one stray marker must not mute everything after it.
                search_from = i + 3
                continue
            out.append(raw[:i])
            self._in_fence = True
            self._prev_char = "`"
            raw = raw[i + 3 :]
            search_from = 0
        held = 0
        while held < 2 and held < len(raw) and raw[len(raw) - 1 - held] == "`":
            held += 1
        if held:  # trim raw to end at what PRECEDES the held-back backticks
            self._raw = raw[len(raw) - held :]
            raw = raw[: len(raw) - held]
        else:
            self._raw = ""
        if not self._in_fence:
            out.append(raw)
        self._update_prev(raw)
        return "".join(out)

    def _next_cut(self) -> int | None:
        buf = self._buf
        if not buf:
            return None
        cut = _primary_cut(buf)
        if cut != -1 and cut < self._max:
            return cut
        if len(buf) >= self._max:
            # Hard cap, taken even when a sentence end lies PAST it (a long
            # sentence fed at once): downstream synthesis has real input
            # budgets (MMS-TTS), so no chunk may exceed max_chars.
            return _soft_cut(buf, self._max)
        # No primary boundary reachable: fall back to a clause boundary at the floor.
        floor = self._min if self._spoke else self._first_min
        if len(buf) >= floor:
            cut = _secondary_cut(buf, floor)
        return cut if cut != -1 else None
