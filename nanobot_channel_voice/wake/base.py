"""Acoustic wake-word detector interface (the ``wake.engine`` tier).

A detector consumes the same post-AEC capture frames the VAD sees and raises a debounced
hit on the wake phrase. It runs inside the frame hop and must NEVER raise per-frame: a
broken model reports non-hits; construction is where incompatibility fails.
"""

from __future__ import annotations


class WakeDetector:
    heavy = True  # neural inference per chunk: run off the event loop

    # Most recent classifier score (0..1); None before the first decision.
    last_score: float | None = None

    # After push() returns True: bytes between where the detector places the phrase END
    # (a run of decisions: its first) and the end of the frame just pushed; the backend
    # subtracts it from its stream position. 0 = at the frame boundary.
    last_hit_back_bytes: int = 0

    def push(self, frame: bytes) -> bool:
        """Consume one capture frame (S16LE mono); True on a NEW debounced hit."""
        raise NotImplementedError

    def reset(self) -> None:
        """Drop streaming state (capture gap: the audio is discontinuous)."""

    def take_stats(self) -> tuple[float, int] | None:
        """The highest score and the longest run at the threshold since the last call, then
        both start over; None from a detector that keeps no such count."""
        return None

    def release(self) -> None:
        """Release model resources (ORT session / NPU context). Idempotent."""

