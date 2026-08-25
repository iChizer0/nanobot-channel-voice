"""Zero-dependency RMS-energy VAD with an adaptive noise floor."""

from __future__ import annotations

import threading

# Hot path: per frame for the whole session; numpy-accelerated with the [ondevice] extra.
from nanobot_channel_voice.audio.pcm import pcm_rms as _rms_normalized
from nanobot_channel_voice.vad.base import Vad


class EnergyVad(Vad):
    """With ``fixed_threshold == 0`` it tracks the background level and flags a frame a
    few times louder than the floor: quiet close-mic setups only. A positive
    ``fixed_threshold`` (normalized 0..1) makes it static."""

    _MIN_THRESHOLD = 0.01   # absolute floor so true silence never trips speech
    _SPEECH_MULT = 3.0      # a frame must be this many x the noise floor
    _FLOOR_ATTACK = 0.05    # EMA rate toward the frame level on non-speech frames
    _FLOOR_CREEP = 0.001    # tiny adaptation during speech so the floor can't lock on

    def __init__(self, fixed_threshold: float = 0.0):
        self._fixed = fixed_threshold
        self._noise: float | None = None
        # scale_floor arrives from the LOOP while is_speech may run on the hop thread:
        # unsynchronized, the read-modify-write loses the duck step it exists for.
        self._lock = threading.Lock()

    def scale_floor(self, factor: float) -> None:
        with self._lock:
            if self._fixed <= 0 and self._noise is not None and factor > 0:
                self._noise *= factor

    def is_speech(self, frame: bytes) -> bool:
        rms = _rms_normalized(frame)  # pure math: outside the lock
        if self._fixed > 0:
            return rms >= self._fixed
        with self._lock:
            if self._noise is None:
                # Seed low so a cold start mid-speech still trips speech; a quiet
                # first frame seeds itself and tracks up.
                self._noise = min(rms, self._MIN_THRESHOLD / self._SPEECH_MULT)
            threshold = max(self._MIN_THRESHOLD, self._noise * self._SPEECH_MULT)
            speech = rms >= threshold
            rate = self._FLOOR_CREEP if speech else self._FLOOR_ATTACK
            self._noise = (1 - rate) * self._noise + rate * rms
        return speech
