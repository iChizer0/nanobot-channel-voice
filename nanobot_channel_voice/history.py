"""What a cloud session said, kept on the device for the session that replaces it.

A realtime provider forgets the conversation when its socket closes (a park, a turn cap,
a dropped network), unless it resumes one (xAI, Gemini). The shell feeds this log from
the transcripts it already receives; a fresh session's instructions carry it back.
"""

from __future__ import annotations

import time
import unicodedata
from dataclasses import dataclass, field

# A replay is context, not a transcript: the recent exchange, bounded, and never one from
# long ago (xAI keeps its own resumable history 30 minutes).
_MAX_TURNS = 12
_MAX_CHARS = 3000
_MAX_REPLY_CHARS = 500
_TTL_S = 30 * 60.0
# A live transcript piece (Gemini streams them while the user speaks) continues the line
# it follows only this soon after it; later, or after a reply, it is a new utterance.
_PIECE_GAP_S = 2.0


@dataclass
class _Turn:
    role: str  # "user" | "assistant"
    t: float
    text: str = ""  # a reply's, once it closed
    onset: bool = False  # user only: opened by an onset, the slot a late transcript fills
    updated: float = 0.0  # user only: when its last transcript piece landed
    # an open reply: (reply audio ms received when the text arrived, text)
    pieces: list[tuple[float, str]] = field(default_factory=list)
    cut: bool = False  # interrupted: only what was heard is kept


def _unspaced(char: str) -> bool:
    """Han, kana and their punctuation run together; Hangul is spaced like Latin."""
    return "\u2e80" <= char <= "\u9fff" or "\uf900" <= char <= "\ufaff" or "\uff00" <= char <= "\uffef"


def _join(left: str, right: str) -> str:
    """Two transcript pieces, spaced as the script spaces words."""
    if (
        not left or not right or left[-1].isspace() or right[0].isspace()
        or unicodedata.category(right[0]).startswith("P")
        or _unspaced(left[-1]) or _unspaced(right[0])
    ):
        return left + right
    return f"{left} {right}"


class SpokenHistory:
    """Turns in the order they were spoken. Each onset opens a user slot, which a late
    transcript finds by counting back the onsets since; an assistant reply keeps only
    what was heard of it."""

    def __init__(self) -> None:
        self._turns: list[_Turn] = []
        self._reply: _Turn | None = None
        self._reply_audio_ms = 0.0

    # ---- feeding (the shell, on the loop) ----------------------------------

    def user_onset(self, backlog_ms: float) -> None:
        """The user started speaking: the reply they may have cut off keeps what played
        (the audio it received minus what the sink had not yet played)."""
        self._close_reply(heard_ms=self._reply_audio_ms - backlog_ms)
        self._append(_Turn("user", time.monotonic(), onset=True))

    def user_text(self, text: str, later_onsets: int | None = None) -> None:
        """A transcript: late, for the utterance ``later_onsets`` onsets back (OpenAI
        dialects), or live, a piece of what is being said (Gemini, ``None``)."""
        if not text.strip():
            return
        now = time.monotonic()
        if later_onsets is not None:
            slots = [turn for turn in self._turns if turn.onset]
            if later_onsets < len(slots):
                slot = slots[-1 - later_onsets]
                slot.text, slot.updated = _join(slot.text, text), now
                return
        # The line this piece continues: the last one, if still being spoken (a final piece
        # may land after the reply began), or an onset slot nothing followed yet.
        line = next((turn for turn in reversed(self._turns) if turn.role == "user"), None)
        if line is None or not (
            (line.text and now - line.updated <= _PIECE_GAP_S)
            or (not line.text and line is self._turns[-1])
        ):
            line = self._append(_Turn("user", now))
        line.text, line.updated = _join(line.text, text), now

    def reply_audio(self, ms: float) -> None:
        self._open_reply()
        self._reply_audio_ms += ms

    def reply_text(self, text: str) -> None:
        if text:
            self._open_reply().pieces.append((self._reply_audio_ms, text))

    def reply_done(self) -> None:
        self._close_reply(heard_ms=None)

    def discard_open_reply(self) -> None:
        """What surfaced while a barge-in flushed was the cut-off reply's tail, not a new
        one: nothing answers speech that has only just begun."""
        if self._reply is not None:
            self._turns = [turn for turn in self._turns if turn is not self._reply]
            self._reply = None

    # ---- reading -----------------------------------------------------------

    def render(self) -> str:
        """The replay, oldest first, or "" when there is nothing recent to carry."""
        now = time.monotonic()
        lines: list[str] = []
        for turn in self._turns:
            text = self._text(turn) if now - turn.t <= _TTL_S else ""
            if text and turn.role == "user":
                lines.append(f"User: {text}")
            elif text:
                lines.append(f"You{' (cut off)' if turn.cut else ''}: {text}")
        lines = lines[-_MAX_TURNS:]  # after the empties: an untranscribed onset is no turn
        while lines and sum(len(line) + 1 for line in lines) > _MAX_CHARS:
            lines.pop(0)
        return "\n".join(lines)

    # ---- internals ---------------------------------------------------------

    def _open_reply(self) -> _Turn:
        if self._reply is None:
            self._reply = self._append(_Turn("assistant", time.monotonic()))
            self._reply_audio_ms = 0.0
        return self._reply

    def _close_reply(self, *, heard_ms: float | None) -> None:
        reply, self._reply = self._reply, None
        if reply is None:
            return
        pieces = reply.pieces
        if heard_ms is not None and self._reply_audio_ms - heard_ms > 1.0:
            # A piece's audio follows its stamp: heard only once playback passed it.
            pieces = [piece for piece in pieces if piece[0] < heard_ms]
            reply.cut = len(pieces) < len(reply.pieces)
        reply.text, reply.pieces = "".join(text for _, text in pieces), []

    def _text(self, turn: _Turn) -> str:
        text = turn.text if turn is not self._reply else "".join(p for _, p in turn.pieces)
        # One line per turn: a newline inside a transcript must not read as a new speaker.
        text = " ".join(text.split())
        if turn.role == "assistant" and len(text) > _MAX_REPLY_CHARS:
            text = text[:_MAX_REPLY_CHARS].rstrip() + "…"
        return text

    def _append(self, turn: _Turn) -> _Turn:
        self._turns.append(turn)
        # Room for the empty slots of untranscribed onsets; render bounds the replay.
        del self._turns[: max(0, len(self._turns) - 4 * _MAX_TURNS)]
        return turn


__all__ = ["SpokenHistory"]
