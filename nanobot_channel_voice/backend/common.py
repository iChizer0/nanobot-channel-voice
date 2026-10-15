"""One copy of the emit/turn pair (and small shared helpers) so the two backends
cannot drift."""

from __future__ import annotations

import threading

from loguru import logger

from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.metrics import VoiceMetrics
from nanobot_channel_voice.wake.base import WakeDetector

from .base import OnEvent, StateHint, VoiceState

# One tool result's share of a realtime session's context: the whole output enters it, and
# a spoken answer needs a fraction of it (nanobot's own text turns cut at 16 000).
TOOL_OUTPUT_CHARS = 8000


def clamp_tool_output(output: str, limit: int = TOOL_OUTPUT_CHARS) -> str:
    """An oversized tool output cut to ``limit`` chars (0 = unlimited): its head and its
    tail, where a command's errors and exit code are, around a note of the cut."""
    if limit <= 0 or len(output) <= limit:
        return output
    marker = (
        f"\n[... {len(output)} chars in all, the middle cut: a voice session keeps {limit} "
        "of a tool's output; narrow the call for the rest ...]\n"
    )
    if len(marker) >= limit:
        return marker.strip()[:limit]
    keep = limit - len(marker)
    head = keep // 2
    return output[:head] + marker + output[len(output) - (keep - head):]


def loggable_text(text: str, enabled: bool, cap: int = 80) -> str:
    """Transcript text for log lines, honoring ``voice.logTranscripts``; off (the default)
    a word count rides instead — user speech is personal data and gateway logs persist.
    Whitespace collapses to one space: an embedded ``\\n`` would split the log record."""
    words = text.split()
    return " ".join(words)[:cap] if enabled else f"<{len(words)} words>"


class TurnEventMixin:
    """The concrete backend defines ``_on_event`` (set at ``start()``), ``_closing`` and
    ``_turn``. ``_emit`` drops events once closing: a late worker callback (a TTS thread
    finishing after ``close()``) must not reach a stopped shell. ``_set_turn`` is the ONE
    place turn state changes."""

    _on_event: OnEvent | None
    _closing: bool
    _turn: VoiceState

    async def _emit(self, event) -> None:
        if self._on_event is not None and not self._closing:
            await self._on_event(event)

    async def _set_turn(self, state: VoiceState) -> None:
        if state is self._turn:
            return
        self._turn = state
        await self._emit(StateHint(state))


class WakeScores:
    """A wake detector's readout: its peak and longest run at the threshold since last taken,
    at an utterance's onset (what a reply's tail or a dropped utterance scored) and at its
    close (the session's peaks in the metrics); logged under debug.wakeScores."""

    def __init__(
        self, detector: WakeDetector | None, lock: threading.Lock, metrics: VoiceMetrics,
        config: VoiceConfig,
    ) -> None:
        self._detector = detector
        self._lock = lock
        self._metrics = metrics
        self._enabled = config.debug.wake_scores
        self._knobs = (config.wake.openwakeword.threshold, config.wake.openwakeword.patience)
        self._log = logger.bind(component="voice")

    def _take(self) -> tuple[float, int] | None:
        if self._detector is None:
            return None
        with self._lock:  # the hop is idle between frames; insurance
            return self._detector.take_stats()

    def onset(self) -> None:
        self._note("between utterances", metric=False)

    def close(self) -> None:
        self._note("over the utterance", metric=True)

    def _note(self, what: str, *, metric: bool) -> None:
        stats = self._take()
        if stats is None:
            return
        peak, run = stats
        if metric:  # in percent: the ring's one-decimal quantiles keep the resolution
            self._metrics.observe("wake_peak_pct", peak * 100.0)
        if self._enabled:
            self._log.info(
                "wake scores {}: peak {:.2f}, longest run {} chunk(s) at/above {:.2f} "
                "(patience {})",
                what, peak, run, *self._knobs,
            )
