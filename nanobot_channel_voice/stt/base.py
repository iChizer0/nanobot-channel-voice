"""STT adapter interface + WAV helpers.

Default path: captured PCM goes to a temp WAV (helpers here) for nanobot's
``BaseChannel.transcribe_audio`` (cloud, or a local OpenAI-compatible Whisper server);
``make_stt`` returns ``None``. On-device path: an :class:`SttAdapter` over RKNN/ONNX
models, audio never leaving the device, streaming engines decoding during capture via
:class:`SttStream`.
"""

from __future__ import annotations

import abc
import os
import tempfile
from contextlib import suppress
from pathlib import Path

from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes


class SttStream(abc.ABC):
    """One utterance's decode stream, returned by :meth:`SttAdapter.stream_start`.

    CALLER-owned and per-utterance: after :meth:`finish` the handle is spent, and
    abandoning one (rejected blip, barge-in) is just dropping the reference. A FRESH
    handle per utterance means a ``finish()`` still chewing in a worker thread can never
    touch the next utterance's state; no locks. Methods are sync and cheap-per-call by
    contract: the local backend calls ``accept`` one frame at a time inside the same
    off-loop hop that runs a heavy VAD, and runs ``finish`` in its own thread.
    """

    def partial(self) -> str:
        """Transcript-so-far (no tail flush) for the early-confirm gate; the default
        (no partials) disables early confirmation."""
        return ""

    @abc.abstractmethod
    def accept(self, pcm: bytes) -> None:
        """Feed one S16_LE mono frame; decode any ready chunks."""

    @abc.abstractmethod
    def finish(self) -> str:
        """Flush the tail and return the utterance transcript (handle is spent)."""


class SttAdapter(abc.ABC):
    # True => stream_start() is implemented and the local backend feeds frames DURING
    # capture, finishing the SttStream at the endpoint instead of calling transcribe().
    streaming: bool = False

    @abc.abstractmethod
    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe S16_LE mono PCM, returning text (``""`` on empty/failure)."""

    async def warmup(self) -> None:
        """Prime caches/arenas once, in the background, at session start. No-op by
        default; cloud-backed adapters must keep it so (no billed warmup calls)."""

    def stream_start(self) -> SttStream:
        raise NotImplementedError("streaming STT not supported by this adapter")

    def release(self) -> None:
        """Give back accelerator resources; idempotent. No-op unless the adapter holds
        an :class:`OnDeviceModel`: refcount-GC does NOT free an RKNN context, so an
        in-process channel restart would exhaust the NPU."""


def pcm_to_float_mono(pcm: bytes, src_rate: int, dst_rate: int):
    """Decode S16_LE mono PCM to float32 in [-1, 1], resampled to ``dst_rate``.

    Downsampling goes through the frequency domain (truncating the spectrum at the new
    Nyquist == a brick-wall anti-alias filter): linear interpolation folds all 8-24 kHz
    energy (fricatives, hiss) into the speech band on a 48k -> 16k capture, measurably
    degrading recognition. Upsampling stays linear: no aliasing risk, cheaper.
    """
    import numpy as np  # lazy: keeps this module import-safe without numpy

    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    if src_rate == dst_rate or not len(audio):
        return audio
    n = int(round(len(audio) * dst_rate / src_rate))
    if n <= 0:
        return audio[:0]
    if dst_rate < src_rate:
        spec = np.fft.rfft(audio)
        spec = spec[: n // 2 + 1]
        # irfft(.., n) divides by n while the forward rfft was unnormalized, so
        # the shorter round trip scales amplitude by len(audio)/n; undo it.
        return (np.fft.irfft(spec, n) * (n / len(audio))).astype(np.float32)
    x = np.linspace(0.0, len(audio), n, endpoint=False)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


def read_token_table(path: str) -> dict[int, str]:
    """Parse a sherpa ``<token> <id>`` tokens file into ``{id: token}``."""
    tokens: dict[int, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            # rstrip \r too: a CRLF-converted file fails isdigit() on EVERY line.
            token, _, idx = line.rstrip("\r\n").rpartition(" ")
            if token and idx.lstrip("-").isdigit():
                tokens[int(idx)] = token
    if not tokens:
        # An empty table decodes every utterance to "": a mute STT that looks healthy.
        raise ValueError(f"no tokens parsed from {path} (wrong format?)")
    return tokens


def write_temp_wav(pcm: bytes, sample_rate: int) -> str:
    """Write S16_LE mono PCM to a fresh 0600 temp WAV; the CALLER deletes it."""
    fd, path = tempfile.mkstemp(prefix="voice-", suffix=".wav")
    os.close(fd)
    try:
        Path(path).write_bytes(pcm_to_wav_bytes(pcm, sample_rate))
    except Exception:
        with suppress(OSError):
            os.unlink(path)
        raise
    return path
