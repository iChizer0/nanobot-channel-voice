"""WebRTC VAD backend (optional ``webrtcvad`` dependency)."""

from __future__ import annotations

from nanobot_channel_voice.audio.base import frame_bytes
from nanobot_channel_voice.vad.base import Vad


class WebRtcVad(Vad):
    """Requires 8/16/32/48 kHz and 10/20/30 ms frames; anything else raises at construction."""

    def __init__(self, sample_rate: int, frame_ms: int, aggressiveness: int):
        import webrtcvad

        # Validate HERE, not per frame: is_speech swallows exceptions, so an invalid
        # rate/frame combination would silently return "no speech" forever (a
        # permanently deaf session). Raising lets make_vad fall back to energy.
        if sample_rate not in (8000, 16000, 32000, 48000):
            raise ValueError(f"webrtcvad supports 8/16/32/48 kHz, not {sample_rate} Hz")
        if frame_ms not in (10, 20, 30):
            raise ValueError(f"webrtcvad supports 10/20/30 ms frames, not {frame_ms} ms")
        self._vad = webrtcvad.Vad(aggressiveness)
        self._sample_rate = sample_rate
        self._frame_bytes = frame_bytes(sample_rate, frame_ms)

    def is_speech(self, frame: bytes) -> bool:
        if len(frame) != self._frame_bytes:
            return False
        try:
            return self._vad.is_speech(frame, self._sample_rate)
        except Exception:  # noqa: BLE001
            return False
