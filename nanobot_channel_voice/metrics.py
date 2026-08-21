"""VoiceMetrics: the measurement seam for tool-call success and turn latency.

One collector shared by the backend / shell / channel; in-process, no locks:
callers stay on the event loop except the threaded hop path's counter bumps,
whose worst-case race loses an increment (never corrupts); hot paths a
``monotonic()`` read and a deque append.

Methodology:

* **Anchor at end-of-speech**, not at an API call: the cloud anchor is
  ``input_audio_buffer.speech_stopped``, local uses ``turn_anchor(offset_ms=...)``
  back-dated past the hangover. With no anchor, record NOTHING and count
  ``ttfa_unanchored``: an un-anchored number looks comparable and isn't. The
  counter also absorbs audio after a DELIBERATE release (``turn_end`` on agent
  timeout / session loss): recovery speech is real audio outside any measured turn.
* **Keep raw samples; never average or add percentiles** (the p99 of a pipeline
  is not the sum of its stages'), so quantiles are computed at read time.
* **Refuse percentiles the sample size cannot support**: ``p90`` needs n >= 10,
  ``p99`` n >= 1000; below that the field is ``None``. ``n`` is ALWAYS reported.
* **Enqueue-side, not ear-side**: "first audio" is when a frame reaches the
  ``AudioSink`` (true voice-to-voice latency needs a mic in the room), and the
  snapshot's ``_enqueue_side`` marker says so.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

# Ring bound; 4096 is well over what any percentile we will claim needs.
_MAX_SAMPLES = 4096
# Bounds the leak from a call that neither completes nor gets discarded.
_MAX_INFLIGHT = 256

# Minimum n before a quantile is reported (p99 at n=50 describes half a sample).
_MIN_N_P90 = 10
_MIN_N_P99 = 1000


def _now_ms() -> float:
    return time.monotonic() * 1000.0


class _Samples:
    """Raw latency samples for one metric, with nearest-rank quantiles."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf: deque[float] = deque(maxlen=_MAX_SAMPLES)

    def add(self, ms: float) -> None:
        self._buf.append(ms)

    def __len__(self) -> int:
        return len(self._buf)

    @staticmethod
    def _quantile(ordered: list[float], q: float) -> float:
        # Nearest-rank, no interpolation: every reported value actually occurred.
        n = len(ordered)
        rank = max(1, min(n, int(-(-n * q // 1))))  # ceil(n*q), clamped
        return round(ordered[rank - 1], 1)

    def summary(self) -> dict:
        n = len(self._buf)
        if n == 0:
            return {"n": 0, "p50": None, "p90": None, "p99": None, "max": None}
        ordered = sorted(self._buf)
        return {
            "n": n,
            "p50": self._quantile(ordered, 0.50),
            "p90": self._quantile(ordered, 0.90) if n >= _MIN_N_P90 else None,
            "p99": self._quantile(ordered, 0.99) if n >= _MIN_N_P99 else None,
            "max": round(ordered[-1], 1),
        }


@dataclass(slots=True)
class _CallRecord:
    """One tool call's timeline, joined by ``call_id``, the only identity that
    crosses the backend -> shell boundary."""

    name: str
    seen_at: float                      # function_call item observed (backend)
    dispatched_at: float | None = None  # ToolCall emitted (backend)
    spawned_at: float | None = None     # task created (shell)
    epoch: int = 0                      # sink epoch at dispatch (staleness check)


@dataclass(slots=True)
class VoiceMetrics:
    """One session's metrics: backends record the turn timeline and call
    sightings, the shell tool execution, the channel supervisor delegation
    segments."""

    counters: dict[str, int] = field(default_factory=dict)
    _latency: dict[str, _Samples] = field(default_factory=dict)
    _calls: dict[str, _CallRecord] = field(default_factory=dict)
    # End-of-speech anchor; None between turns and without an end-of-speech signal,
    # reset per turn so a stale anchor cannot leak into the next TTFA.
    _anchor: float | None = None
    _ttfa_done: bool = False
    # A tool turn is >= 2 responses; audio after the continuation is measured from
    # it, not as TTFA; folding them lets a slow tool look like a slow model.
    _in_continuation: bool = False

    # ---- primitives ---------------------------------------------------------

    def count(self, key: str, n: int = 1) -> None:
        self.counters[key] = self.counters.get(key, 0) + n

    def observe(self, key: str, ms: float) -> None:
        if ms < 0:
            return  # a negative segment means the clocks were misordered; drop it
        bucket = self._latency.get(key)  # not setdefault: it builds a deque per call
        if bucket is None:
            bucket = self._latency[key] = _Samples()
        bucket.add(ms)

    # ---- turn timeline ------------------------------------------------------

    def turn_anchor(self, offset_ms: float = 0.0) -> None:
        """End of user speech: the clock every turn metric is measured from.
        ``offset_ms`` back-dates it for callers that learn of end-of-speech late
        (the local endpointer confirms ``hangoverMs`` + any STT after the last
        speech frame), keeping local numbers comparable with the cloud's
        ``speech_stopped``."""
        self._anchor = _now_ms() - max(0.0, offset_ms)
        self._ttfa_done = False
        self._in_continuation = False

    def turn_thinking(self) -> None:
        """``response.created``: the provider accepted the turn."""
        if self._anchor is not None and not self._in_continuation:
            self.observe("think_ms", _now_ms() - self._anchor)

    def turn_continuation(self) -> None:
        """All tool results are in and the turn resumes. Re-anchors so the audio
        that follows is measured as continuation latency, not as TTFA."""
        self._anchor = _now_ms()
        self._ttfa_done = False
        self._in_continuation = True

    def turn_first_audio(self) -> None:
        """First output frame enqueued. TTFA: the headline number. Latched, or
        every frame in the turn would record a sample against the one anchor."""
        if self._ttfa_done:
            return
        self._ttfa_done = True
        if self._anchor is None:
            self.count("ttfa_unanchored")
            return
        key = "continuation_ms" if self._in_continuation else "ttfa_ms"
        self.observe(key, _now_ms() - self._anchor)

    def turn_end(self) -> None:
        self._anchor = None
        self._ttfa_done = False
        self._in_continuation = False

    # ---- tool call timeline -------------------------------------------------

    def call_seen(self, call_id: str, name: str) -> None:
        # Idempotent: `tool_calls` is the denominator of every success rate, and a
        # re-announced item would double-count it and reset seen_at.
        if call_id in self._calls:
            return
        if len(self._calls) >= _MAX_INFLIGHT:
            # One eviction suffices; prefer a record that never reached the shell,
            # since evicting a spawned one forfeits its latency sample and staleness verdict.
            victim = next(
                (cid for cid, r in self._calls.items() if r.spawned_at is None),
                next(iter(self._calls)),
            )
            del self._calls[victim]
            self.count("inflight_overflow")
        self._calls[call_id] = _CallRecord(name=name, seen_at=_now_ms())
        self.count("tool_calls")

    def call_dispatched(self, call_id: str, epoch: int) -> None:
        rec = self._calls.get(call_id)
        if rec is None:
            return
        rec.dispatched_at = _now_ms()
        rec.epoch = epoch
        self.observe("tool_dispatch_ms", rec.dispatched_at - rec.seen_at)

    def call_spawned(self, call_id: str) -> None:
        rec = self._calls.get(call_id)
        if rec is not None:
            rec.spawned_at = _now_ms()

    def call_finished(self, call_id: str, *, outcome: str, mode: str) -> None:
        """Close a call out. ``outcome`` is the success classification (see
        ``VoiceShell._on_tool_call``); ``mode`` is ``direct`` | ``supervisor``, and
        execution latency is bucketed BY MODE so the two stay comparable."""
        self.count(f"tool_{outcome}")
        rec = self._calls.pop(call_id, None)
        if rec is None or rec.spawned_at is None:
            return
        self.observe(f"tool_exec_ms.{mode}", _now_ms() - rec.spawned_at)

    def call_stale(self, call_id: str, sink_epoch: int) -> bool:
        """True if the call was dispatched under a superseded sink epoch: the user
        barged in while it ran, so its result is for a dead turn. Counted, never
        acted on; the backend's stale guard owns correctness."""
        rec = self._calls.get(call_id)
        if rec is None or rec.epoch == sink_epoch:
            return False
        self.count("tool_stale")
        return True

    def calls_dropped(self, call_ids: set[str], reason: str) -> int:
        """Void obligations that will never be answered (session loss, teardown).
        Only still-open calls count, so the caller should log the RETURNED count:
        the provider's pending set can include calls the shell already finished."""
        dropped = [cid for cid in call_ids if self._calls.pop(cid, None) is not None]
        if dropped:
            self.count(f"tool_dropped.{reason}", len(dropped))
        return len(dropped)

    def calls_abandoned(self, call_ids: set[str]) -> None:
        """A cancelled response's obligations. Only those never DISPATCHED to the
        shell are dropped: one that reached the shell is still answered
        (``submit_tool_result`` is sent), it just won't resume the dead turn."""
        for cid in call_ids:
            rec = self._calls.get(cid)
            if rec is not None and rec.dispatched_at is None:
                del self._calls[cid]
                self.count("tool_dropped.cancelled")

    # ---- readout ------------------------------------------------------------

    @property
    def has_data(self) -> bool:
        """Anything worth reporting? Latency counts too: a pure-conversation session
        has no counters, and keying on those alone would drop its TTFA."""
        return bool(self.counters or self._latency)

    def snapshot(self) -> dict:
        return {
            "_enqueue_side": True,
            "counters": dict(sorted(self.counters.items())),
            "latency_ms": {k: self._latency[k].summary() for k in sorted(self._latency)},
            "inflight": len(self._calls),
        }

    def summary_line(self) -> str:
        """One-line human summary for the session-end log."""
        c = self.counters
        calls = c.get("tool_calls", 0)
        ok = c.get("tool_ok", 0)
        parts = [f"tool_calls={calls}", f"ok={ok}"]
        for key in ("tool_error", "tool_exception", "tool_cancelled", "tool_stale", "tool_no_seam"):
            if c.get(key):
                parts.append(f"{key.removeprefix('tool_')}={c[key]}")
        for key in sorted(k for k in c if k.startswith("tool_dropped.")):
            parts.append(f"{key.removeprefix('tool_')}={c[key]}")
        for key in ("ttfa_ms", "think_ms"):
            s = self._latency.get(key)
            if s is not None and len(s):
                parts.append(f"{key.removesuffix('_ms')}_p50={s.summary()['p50']}ms(n={len(s)})")
        return " ".join(parts)
