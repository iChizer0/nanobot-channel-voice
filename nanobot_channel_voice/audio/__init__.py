"""Audio backend selection."""

from __future__ import annotations

import shutil

from loguru import logger

from nanobot_channel_voice.audio.base import CaptureSource, PlaybackSink
from nanobot_channel_voice.audio.null import NullCapture, NullPlayback
from nanobot_channel_voice.config import AudioConfig
from nanobot_channel_voice.engines import describe_build_error

__all__ = ["CaptureSource", "PlaybackSink", "make_audio"]


def _make_pyalsa(cfg: AudioConfig) -> tuple[CaptureSource, PlaybackSink] | None:
    try:
        import alsaaudio  # noqa: F401

        from nanobot_channel_voice.audio.pyalsa import PyAlsaCapture, PyAlsaPlayback
    except Exception as exc:  # noqa: BLE001 - missing/broken optional dep
        logger.warning(
            "voice: pyalsaaudio unavailable ({}); falling back", describe_build_error(exc)
        )
        return None
    return (
        PyAlsaCapture(cfg.capture_device, cfg.sample_rate, cfg.frame_ms),
        PyAlsaPlayback(cfg.playback_device),
    )


def _make_subprocess(cfg: AudioConfig) -> tuple[CaptureSource, PlaybackSink] | None:
    if not (shutil.which(cfg.arecord_path) and shutil.which(cfg.aplay_path)):
        return None
    from nanobot_channel_voice.audio.alsa import AlsaCapture, AlsaPlayback

    return (
        AlsaCapture(cfg.capture_device, cfg.sample_rate, cfg.frame_ms, cfg.arecord_path),
        AlsaPlayback(cfg.playback_device, cfg.aplay_path),
    )


def make_audio(cfg: AudioConfig) -> tuple[CaptureSource, PlaybackSink]:
    """Build the (capture, playback) pair, degrading to null when unavailable."""
    if cfg.backend == "pyalsa":
        backend = _make_pyalsa(cfg) or _make_subprocess(cfg)
        if backend is not None:
            return backend
        logger.warning("voice: pyalsa and arecord/aplay both unavailable; using null audio backend")
    elif cfg.backend == "alsa":
        backend = _make_subprocess(cfg)
        if backend is not None:
            return backend
        logger.warning("voice: arecord/aplay not found; using null audio backend")
    return NullCapture(cfg.sample_rate, cfg.frame_ms), NullPlayback()
