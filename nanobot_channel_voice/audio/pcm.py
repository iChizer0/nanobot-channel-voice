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


def pcm_peak(pcm: bytes) -> float:
    """Peak absolute amplitude of S16_LE PCM, normalized to 0..1. Separates a clip that is
    quiet everywhere from one that is silence around a blip."""
    if len(pcm) < 2:
        return 0.0
    if _np is not None:
        a = _np.frombuffer(pcm[: len(pcm) // 2 * 2], dtype="<i2")
        return float(_np.abs(a.astype(_np.int32)).max()) / 32768.0 if a.size else 0.0
    samples = array.array("h")
    samples.frombytes(pcm if len(pcm) % 2 == 0 else pcm[:-1])
    return max((abs(v) for v in samples), default=0) / 32768.0


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


def quietest_split(pcm: bytes, rate: int, *, back_ms: float = 240.0, win_ms: float = 10.0) -> int:
    """Byte offset (even) of the END of the quietest ``win_ms`` window inside
    the trailing ``back_ms`` of ``pcm``; ties go to the LATEST window;
    ``len(pcm)`` when no full window fits."""
    n = len(pcm) & ~1
    win_b = max(2, int(rate * win_ms / 1000.0) * 2)
    if n < win_b or rate <= 0:
        return len(pcm)
    lo = max(0, n - (int(rate * back_ms / 1000.0) * 2)) // win_b * win_b
    lo = min(lo, (n - win_b) // win_b * win_b)  # >=1 full window on the grid
    if _np is not None:
        a = _np.frombuffer(pcm[lo : lo + (n - lo) // win_b * win_b], dtype="<i2")
        power = (a.astype(_np.float64) ** 2).reshape(-1, win_b // 2).sum(axis=1)
        idx = power.size - 1 - int(_np.argmin(power[::-1]))  # ties -> latest
        return lo + (idx + 1) * win_b
    best_end, best_power = n, None
    for off in range(lo, n - win_b + 1, win_b):
        samples = array.array("h")
        samples.frombytes(pcm[off : off + win_b])
        power = 0
        for v in samples:
            power += v * v
        if best_power is None or power <= best_power:
            best_power, best_end = power, off + win_b
    return best_end


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
                # The header can LIE: a non-seekable writer (espeak-ng --stdout) stamps
                # a placeholder data size (0x7ffff000 ~= 13.5 h). 44 = canonical PCM
                # header; an honest header's frame count wins the min anyway.
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


def wav_pcm(blob: bytes) -> tuple[bytes, int]:
    """S16 PCM + rate from a WAV blob, downmixed to mono; ``(b"", 0)`` on
    anything unreadable (calibration-grade tolerance, not a decoder)."""
    try:
        with wave.open(io.BytesIO(blob), "rb") as w:
            if w.getsampwidth() != 2:
                return b"", 0
            rate, channels = w.getframerate(), w.getnchannels()
            pcm = w.readframes(w.getnframes())
    except Exception:  # noqa: BLE001 - malformed input reads as no audio
        return b"", 0
    if channels > 1:
        usable = len(pcm) // (2 * channels) * (2 * channels)
        if _np is not None:
            frames = _np.frombuffer(pcm[:usable], dtype="<i2").reshape(-1, channels)
            pcm = (frames.sum(axis=1, dtype=_np.int32) // channels).astype("<i2").tobytes()
        else:
            samples = array.array("h", pcm[:usable])
            pcm = array.array(
                "h",
                (
                    sum(samples[i : i + channels]) // channels
                    for i in range(0, len(samples), channels)
                ),
            ).tobytes()
    return pcm, rate


def _antialias_taps(src_rate: int, dst_rate: int) -> list[float]:
    """Hann-windowed sinc lowpass at ``0.45 * dst_rate``, ODD length so the group delay
    is a whole sample and the filtered stream stays time-aligned."""
    # Floor 31: at ratios near 1 the transition band is narrowest, and 9-17 taps left
    # only -10 dB just above the new Nyquist; 31+ gives -20 dB or better.
    n = min(63, max(31, int(8 * src_rate / dst_rate))) | 1
    fc = 0.45 * dst_rate / src_rate
    mid = (n - 1) // 2
    taps = []
    for k in range(n):
        t = k - mid
        sinc = 2 * fc if t == 0 else math.sin(2 * math.pi * fc * t) / (math.pi * t)
        taps.append(sinc * (0.5 - 0.5 * math.cos(2 * math.pi * k / (n - 1))))
    total = sum(taps)
    return [v / total for v in taps]


def resample_pcm(pcm: bytes, src_rate: int, dst_rate: int) -> bytes:
    """Linear-interpolation resample of S16_LE mono.

    Downsampling is lowpass-filtered first: bare interpolation folds every above-Nyquist
    component straight into the speech band (20 kHz lands on 2 kHz at 44.1k -> 22.05k).
    """
    if src_rate == dst_rate or src_rate <= 0 or dst_rate <= 0 or len(pcm) < 4:
        return pcm
    n_in = (len(pcm) & ~1) // 2
    n_out = int(n_in * dst_rate / src_rate)
    if n_out < 1:
        return b""
    taps = _antialias_taps(src_rate, dst_rate) if dst_rate < src_rate else None
    step = src_rate / dst_rate
    if _np is not None:
        values = _np.frombuffer(pcm[: n_in * 2], dtype="<i2").astype(_np.float32)
        if taps is not None:
            # Edge padding, not zeros: a zero-padded convolution fades both ends.
            padded = _np.pad(values, len(taps) // 2, mode="edge")
            values = _np.convolve(padded, _np.asarray(taps, dtype=_np.float32), mode="valid")
        x = _np.arange(n_out, dtype=_np.float64) * step
        i = _np.minimum(x.astype(_np.int64), n_in - 2)
        # Clamped with the taps: past the last pair an unclamped weight EXTRAPOLATES,
        # which wraps int16 on a loud final sample.
        frac = _np.minimum(x - i, 1.0).astype(_np.float32)
        out = values[i] + (values[i + 1] - values[i]) * frac
        return _np.clip(out, -32768.0, 32767.0).astype(_np.int16).tobytes()
    # No numpy = the cloud-only install, where this resamples short cues: the O(N*taps)
    # pure-Python filter would cost ~50 ms per second of audio on the loop, so skip it.
    values = array.array("h", pcm[: n_in * 2])
    out = array.array("h", bytes(2 * n_out))
    for j in range(n_out):
        x = j * step
        i = min(int(x), n_in - 2)
        frac = min(1.0, x - i)  # see the numpy path
        raw = values[i] + (values[i + 1] - values[i]) * frac
        out[j] = max(-32768, min(32767, int(raw)))
    return out.tobytes()


def fade_tail_pcm(pcm: bytes, rate: int, *, ms: float = 10.0) -> bytes:
    """Linear fade-out over the last ``ms`` of S16_LE mono: a length-capped
    cut would otherwise end mid-waveform in an audible click."""
    n = len(pcm) & ~1
    k = min(n // 2, int(rate * ms / 1000.0))
    if k <= 0:
        return pcm
    samples = array.array("h", pcm[:n])
    base = len(samples) - k
    for j in range(k):
        samples[base + j] = int(samples[base + j] * (k - 1 - j) / k)
    return samples.tobytes()


def ding_pcm(rate: int, *, peak: float = 0.18) -> bytes:
    """The "captured" receipt cue (~230 ms, S16 mono): a STRUCK rising fifth (A5 -> E6),
    measured least speech-like of the audition set; ``peak`` ~-15 dBFS sits under
    speech."""
    return _struck_pcm(rate, ((880.0, 0.0, 140.0, 45.0), (1318.5, 60.0, 170.0, 60.0)), peak)


def dong_pcm(rate: int, *, peak: float = 0.18) -> bytes:
    """The attention-close cue: ``ding_pcm``'s pair reversed (E6 -> A5) so the contour
    falls and the long ring lands on the LOW note."""
    return _struck_pcm(rate, ((1318.5, 0.0, 140.0, 45.0), (880.0, 60.0, 170.0, 60.0)), peak)


def _struck_pcm(rate: int, notes: tuple, peak: float) -> bytes:
    mix = [0.0] * max(int(rate * (s + d) / 1000) for _, s, d, _ in notes)
    attack = max(1, int(rate * 0.003))
    for freq, start_ms, dur_ms, tau_ms in notes:
        off = int(rate * start_ms / 1000)
        tau = rate * tau_ms / 1000.0
        w = 2 * math.pi * freq / rate
        for i in range(int(rate * dur_ms / 1000)):
            env = (0.5 - 0.5 * math.cos(math.pi * min(i, attack) / attack)) * math.exp(
                -max(0, i - attack) / tau
            )
            mix[off + i] += env * (
                math.sin(w * i)
                + 0.30 * math.sin(2 * w * i)
                + 0.12 * math.sin(3 * w * i)
            )
    scale = peak * 32767.0 / (max(abs(v) for v in mix) or 1.0)
    return array.array("h", (int(v * scale) for v in mix)).tobytes()
