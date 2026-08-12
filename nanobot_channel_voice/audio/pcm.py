"""Shared S16_LE-mono arithmetic. Durations are best-effort metadata everywhere they
are used, so both duration helpers return 0.0 rather than raise on malformed input.
"""

from __future__ import annotations

import array
import io
import math
import wave

try:  # optional acceleration: numpy ships with the [ondevice] extra
    import numpy as _np
except ImportError:  # pragma: no cover - the pure-python path is tested directly
    _np = None  # type: ignore[assignment]


def pcm_ms(nbytes: int, rate: int) -> float:
    """Playback duration in ms of ``nbytes`` of raw S16_LE mono at ``rate``."""
    return nbytes / (2 * rate) * 1000.0 if rate > 0 else 0.0


def pcm_rms(pcm: bytes) -> float:
    """Root-mean-square amplitude of S16_LE PCM, normalized to 0..1."""
    if len(pcm) < 2:
        return 0.0
    if _np is not None:
        a = _np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2").astype(_np.float64)
        return float(_np.sqrt((a * a).mean())) / 32768.0 if a.size else 0.0
    samples = array.array("h")
    samples.frombytes(pcm if len(pcm) % 2 == 0 else pcm[:-1])
    if not samples:
        return 0.0
    acc = 0
    for v in samples:
        acc += v * v
    return math.sqrt(acc / len(samples)) / 32768.0


def wav_duration_ms(blob: bytes) -> float:
    """Playback duration of a WAV blob in ms (0.0 for empty/unparseable)."""
    if not blob:
        return 0.0
    try:
        with wave.open(io.BytesIO(blob), "rb") as w:
            rate = w.getframerate()
            if not rate:
                return 0.0
            frames = w.getnframes()
            frame_b = w.getsampwidth() * w.getnchannels()
            if frame_b > 0:
                # The header can LIE: a non-seekable writer (espeak-ng --stdout)
                # stamps a placeholder data-chunk size (0x7ffff000 ~= 13.5 h). Never
                # claim more audio than the blob physically holds; 44 is the canonical
                # PCM header, and an honest header's frame count wins the min anyway.
                frames = min(frames, max(0, len(blob) - 44) // frame_b)
            return frames / rate * 1000.0
    except Exception:  # noqa: BLE001 - duration is best-effort metadata
        return 0.0


def pcm_to_wav_bytes(
    pcm: bytes,
    sample_rate: int,
    *,
    channels: int = 1,
    sampwidth: int = 2,
) -> bytes:
    """Wrap raw PCM in a WAV container. ``channels``/``sampwidth`` let a re-wrap carry
    the SOURCE geometry through: a stereo blob re-emitted as mono plays at half speed."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sampwidth)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm)
    return buf.getvalue()
