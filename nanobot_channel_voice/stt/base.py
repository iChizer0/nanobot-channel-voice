"""STT adapter interface + WAV helpers + window-bounded long-form decode.

Default path: ``make_stt`` returns None and captured PCM goes to a temp WAV for
nanobot's ``BaseChannel.transcribe_audio``. On-device path: an :class:`SttAdapter` over
RKNN/ONNX, streaming engines decoding during capture via :class:`SttStream`; audio that
may outrun an adapter's decode window goes through :func:`transcribe_chunked`.
"""

from __future__ import annotations

import abc
import os
import tempfile
from contextlib import suppress
from math import gcd
from pathlib import Path

from loguru import logger

from nanobot_channel_voice.audio.pcm import pcm_ms, pcm_to_wav_bytes


class SttStream(abc.ABC):
    """One utterance's decode stream, returned by :meth:`SttAdapter.stream_start`.

    CALLER-owned and per-utterance: after :meth:`finish` the handle is spent, abandoning
    one is just dropping the reference. A FRESH handle per utterance means a ``finish()``
    still running in a worker thread can never touch the next utterance's state; no
    locks. Methods must be sync and cheap-per-call: ``accept`` runs one frame at a time
    inside the same off-loop hop as a heavy VAD.
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

    # For the voice context's transcript-accuracy line: "attention" (AR — can
    # hallucinate fluent text never spoken) vs "ctc"/"transducer" (frame-synchronous,
    # mis-substitutions only). "" = undeclared: claim no invention.
    decoder_family: str = ""

    # Longest audio one transcribe() call decodes faithfully (None = unbounded).
    # Callers with possibly-longer input route it through transcribe_chunked().
    max_decode_ms: int | None = None

    @abc.abstractmethod
    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        """Transcribe S16_LE mono PCM, returning text (``""`` on empty/failure)."""

    async def warmup(self) -> None:
        """Prime caches/arenas once, in the background, at session start. Cloud-backed
        adapters must keep the no-op default (no billed warmup calls)."""

    def stream_start(self) -> SttStream:
        raise NotImplementedError("streaming STT not supported by this adapter")

    def release(self) -> None:
        """Give back accelerator resources; idempotent. Refcount-GC does NOT free an
        RKNN context, so an in-process channel restart would exhaust the NPU."""


# Chunked decode: trailing span searched for a cut before each window boundary, and
# the RMS granularity of that search.
_CHUNK_SEARCH_MS = 3000
_CHUNK_RMS_MS = 20
# Scripts written without ASCII spaces. Hangul is deliberately absent: Korean spaces.
_NO_SPACE_SCRIPTS = (
    (0x3000, 0x303F), (0x3040, 0x30FF), (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF), (0xF900, 0xFAFF), (0xFF01, 0xFF60),
)


def _joins_bare(ch: str) -> bool:
    cp = ord(ch)
    return any(lo <= cp <= hi for lo, hi in _NO_SPACE_SCRIPTS)


def _join_pieces(texts: list[str]) -> str:
    """Join pieces: a space at latin seams, bare adjacency when either side is a
    no-space script (zh/ja)."""
    out = ""
    for text in texts:
        text = text.strip()
        if not text:
            continue
        if out and not (_joins_bare(out[-1]) or _joins_bare(text[0])):
            out += " "
        out += text
    return out


def _quietest_cut(samples, lo: int, hi: int, frame: int) -> int:
    """Cut index inside ``[lo, hi)``: the END of the quietest RMS frame, LAST minimum
    winning ties so pieces still fill most of the window."""
    import numpy as np

    span = samples[lo:hi]
    n = len(span) // frame
    if n <= 1:
        return hi
    # Frames align to the END of the span, so the constant-energy cut is exactly ``hi``.
    tail = len(span) - n * frame
    r = np.square(span[tail:].reshape(n, frame).astype(np.float32)).sum(axis=1)
    idx = n - 1 - int(np.argmin(r[::-1]))
    return lo + tail + (idx + 1) * frame


async def transcribe_chunked(adapter: SttAdapter, pcm: bytes, sample_rate: int) -> str:
    """Transcribe S16_LE mono PCM of any length against ``adapter.max_decode_ms``.

    Within-window input passes straight through. Longer input is cut into window-sized
    pieces at the quietest instant near each boundary and decoded SEQUENTIALLY (callers
    rely on one decode in flight per adapter). Without this, fixed-window models drop
    everything past the window and attention models pay O(T^2) activations.
    """
    limit = adapter.max_decode_ms
    if limit is None or pcm_ms(len(pcm), sample_rate) <= limit:
        return await adapter.transcribe(pcm, sample_rate)
    import numpy as np  # lazy: keeps this module import-safe without numpy

    pcm = pcm[: len(pcm) // 2 * 2]  # a torn trailing byte must not fail the decode
    samples = np.frombuffer(pcm, dtype="<i2")
    window = max(1, sample_rate * limit // 1000)
    search = max(1, sample_rate * _CHUNK_SEARCH_MS // 1000)
    frame = max(1, sample_rate * _CHUNK_RMS_MS // 1000)
    pieces: list[bytes] = []
    start = 0
    while len(samples) - start > window:
        hi = start + window
        # Floor at half a window: a huge search span must not shrink pieces below it.
        cut = _quietest_cut(samples, max(hi - search, start + window // 2), hi, frame)
        pieces.append(pcm[start * 2 : cut * 2])
        start = cut
    pieces.append(pcm[start * 2 :])
    logger.bind(component="stt").info(
        "utterance ({:.1f}s) exceeds the {:.1f}s decode window; decoding in {} pieces",
        pcm_ms(len(pcm), sample_rate) / 1000, limit / 1000, len(pieces),
    )
    return _join_pieces([await adapter.transcribe(piece, sample_rate) for piece in pieces])


# Downsample block geometry. A whole-signal rfft promotes float32 to complex128 and
# peaks at ~20x the PCM, which a multi-minute upload cannot afford; each block is
# resampled with margins that are then discarded, so every kept span had full context.
_RESAMPLE_BLOCK_S = 10.0
_RESAMPLE_MARGIN_S = 0.5


def _fft_resample(np, audio, n_out: int):
    """Spectrum truncated at the new Nyquist = brick-wall anti-alias."""
    spec = np.fft.rfft(audio)[: n_out // 2 + 1]
    # irfft(.., n) divides by n while the forward rfft was unnormalized, so the
    # shorter round trip scales amplitude by len(audio)/n; undo it.
    return np.fft.irfft(spec, n_out) * (n_out / len(audio))


def _downsample(np, audio, src_rate: int, dst_rate: int, n: int):
    ratio = gcd(src_rate, dst_rate)
    down, up = src_rate // ratio, dst_rate // ratio
    core = down * max(1, int(_RESAMPLE_BLOCK_S * src_rate) // down)
    margin = down * max(1, int(_RESAMPLE_MARGIN_S * src_rate) // down)
    total = len(audio)
    if total <= core + 2 * margin:
        return _fft_resample(np, audio, n).astype(np.float32)
    out = np.empty(n, dtype=np.float32)
    start = 0
    while start < total:
        # Block boundaries stay on multiples of ``down``, so every kept span maps to an
        # exact output range and the blocks tile the output without a seam.
        last = start + core >= total
        lo, hi = max(0, start - margin), total if last else min(total, start + core + margin)
        if not last:
            # A clipped margin must keep hi - lo on the resample grid, or the block's
            # ratio drifts and its content lands up to a period off at the seam.
            hi = lo + (hi - lo) // down * down
        at = start // down * up
        head = (start - lo) // down * up
        body = (n - at) if last else core // down * up
        if body <= 0:
            break
        tail = 0 if last else (hi - start - core) // down * up
        block = _fft_resample(np, audio[lo:hi], head + body + tail)
        out[at : at + body] = block[head : head + body]
        start += core
    return out


def pcm_to_float_mono(pcm: bytes, src_rate: int, dst_rate: int):
    """Decode S16_LE mono PCM to float32 in [-1, 1], resampled to ``dst_rate``.

    Downsampling MUST go through the frequency domain: linear interpolation folds
    8-24 kHz energy into the speech band on a 48k -> 16k capture and measurably degrades
    recognition. Upsampling stays linear: no aliasing risk, cheaper.
    """
    import numpy as np  # lazy: keeps this module import-safe without numpy

    # One buffer, not astype()-then-divide: the intermediate is the size of the input.
    audio = np.multiply(np.frombuffer(pcm, dtype="<i2"), 1.0 / 32768.0, dtype=np.float32)
    if src_rate == dst_rate or not len(audio):
        return audio
    n = int(round(len(audio) * dst_rate / src_rate))
    if n <= 0:
        return audio[:0]
    if dst_rate < src_rate:
        return _downsample(np, audio, src_rate, dst_rate, n)
    x = np.linspace(0.0, len(audio), n, endpoint=False)
    return np.interp(x, np.arange(len(audio)), audio).astype(np.float32)


class DenseTokenTable(list):
    """``{id: token}`` for the common dense case (ids 0..n-1), as a list: a 25k-entry
    dict costs ~2 MB more than the strings it holds. Same read API as the dict."""

    def get(self, idx: int, default: str = "") -> str:
        return self[idx] if 0 <= idx < len(self) else default


def read_token_table(path: str) -> dict[int, str] | DenseTokenTable:
    """Parse a sherpa-style ``<token> <id>`` tokens file into ``{id: token}``
    (a :class:`DenseTokenTable` when the ids are exactly 0..n-1)."""
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
    if len(tokens) == max(tokens) + 1 and min(tokens) == 0:
        return DenseTokenTable(tokens[i] for i in range(len(tokens)))
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
