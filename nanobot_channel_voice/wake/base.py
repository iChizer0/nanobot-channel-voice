"""Acoustic wake-word detector interface (the ``wake.engine`` acoustic tier).

A detector consumes the same post-AEC capture frames the VAD sees and raises a
debounced hit when the configured wake phrase is heard. It runs inside the
frame hop (heavy-class, off the event loop) and must never raise per-frame:
a broken model reports non-hits, construction is where incompatibility fails.
"""

from __future__ import annotations


class WakeDetector:
    heavy = True  # neural inference per chunk: run off the event loop

    # Most recent classifier score (0..1), for logging/triage; None before the
    # first decision.
    last_score: float | None = None

    # After push() returns True: bytes between the hit's decision point and the
    # end of the frame just pushed — the backend subtracts it from its stream
    # position to locate the phrase END. 0 (default) = hit at the frame boundary.
    last_hit_back_bytes: int = 0

    def push(self, frame: bytes) -> bool:
        """Consume one capture frame (S16LE mono); True on a NEW debounced hit."""
        raise NotImplementedError

    def reset(self) -> None:
        """Drop streaming state (capture gap: the audio is discontinuous)."""

    def release(self) -> None:
        """Release model resources (ORT session / NPU context). Idempotent."""
