"""TTS adapter interface + WAV helpers.

:meth:`TtsAdapter.synthesize` returns a complete, self-describing WAV blob, played
byte-for-byte by blob mode; adapters with a native raw-PCM path also declare
``output_rate`` and implement :meth:`TtsAdapter.synthesize_pcm`, which the local
pipeline drives through the gapless stream-mode sink instead. Synthesis is per
speakable chunk (a sentence/clause from the chunker) for low first-audio latency.
"""

from __future__ import annotations

import abc

from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes


class TtsAdapter(abc.ABC):
    output_rate: int | None = None  # None => WAV-only, no synthesize_pcm()

    # False when a synthesis is billed (cloud): the startup calibration probe skips
    # such adapters. Per-INSTANCE: one class serves cloud and local compat servers.
    probe_ok: bool = True

    # The ONE ISO 639-1 code this adapter can actually voice, or None for unrestricted/
    # unknown. Every on-device engine is fixed to a single language at load, so text in
    # any other language comes out silent or as noise; the channel projects this into
    # the agent's context so replies are written in a language the speaker can say.
    # Per-INSTANCE: the same class serves several languages.
    spoken_language: str | None = None

    @abc.abstractmethod
    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        """Return a complete WAV blob for *text*, or ``b""`` on failure."""

    async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
        """Raw S16_LE mono PCM at ``output_rate`` (only when it is set)."""
        raise NotImplementedError("raw-PCM synthesis not supported by this adapter")

    def release(self) -> None:
        """Give back accelerator resources; idempotent, no-op by default. Needed because
        refcount-GC does NOT free an RKNN context, so an in-process channel restart
        would exhaust the NPU."""

    async def warmup(self) -> None:
        """Prime caches/arenas so the first real chunk pays no cold-start. No-op by
        default; nothing at startup may be billed, so cloud-backed adapters keep it a
        no-op and clear ``probe_ok``."""


def is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


_SPLIT_PUNCT = set(",.!?;:") | set("、。，．！？；：")


def split_for_budget(text: str, max_len: int) -> list[str]:
    """Split ``text`` into pieces of at most ``max_len`` chars for a fixed input
    budget: prefer a space, then clause punctuation (CJK included, space-free text
    must not be hard-cut mid-run), then a hard cut."""
    pieces: list[str] = []
    text = text.strip()
    while len(text) > max_len:
        window = text[: max_len + 1]
        cut = window.rfind(" ")
        if cut <= 0:
            cut = max(
                (i + 1 for i in range(max_len) if window[i] in _SPLIT_PUNCT),
                default=max_len,
            )
        pieces.append(text[:cut].strip())
        text = text[cut:].strip()
    if text:
        pieces.append(text)
    return [p for p in pieces if p]


def floats_to_pcm(samples) -> bytes:
    """Clip a float32 waveform to [-1, 1] and quantise to raw S16_LE mono PCM."""
    import numpy as np  # lazy: this module must import without numpy

    arr = np.clip(np.asarray(samples, dtype=np.float32).reshape(-1), -1.0, 1.0)
    return (arr * 32767.0).astype("<i2").tobytes()


def floats_to_wav(samples, rate: int) -> bytes:
    """:func:`floats_to_pcm` wrapped as WAV; empty input returns ``b""``."""
    pcm = floats_to_pcm(samples)
    return pcm_to_wav_bytes(pcm, rate) if pcm else b""
