"""Audio backend interfaces: the only view the interaction loop has of the concrete
backend (arecord/aplay subprocess, in-process pyalsaaudio, or null). All PCM is 16-bit
little-endian mono at the configured sample rate; blob playback additionally receives
a complete, self-describing WAV.
"""

from __future__ import annotations

import abc


def frame_bytes(sample_rate: int, frame_ms: int) -> int:
    """Bytes in one mono S16_LE frame of ``frame_ms`` at ``sample_rate``."""
    return (sample_rate * frame_ms // 1000) * 2


class CaptureSource(abc.ABC):
    @abc.abstractmethod
    async def start(self) -> None: ...

    @abc.abstractmethod
    async def read_frame(self) -> bytes:
        """Return exactly one frame of S16_LE mono PCM, or ``b""`` at end of stream."""

    async def flush(self) -> int:
        """Discard capture already buffered behind :meth:`read_frame` (pipe/queue
        backlog) without blocking on new audio, returning the bytes dropped. The mic
        gate applies at READ time but the device records continuously, so frames
        captured while the bot was audible would replay into the VAD as fresh speech;
        the shell calls this at the gate-reopen edge to realign the read position with
        the wall clock. Default: paced sources hold no backlog."""
        return 0

    @abc.abstractmethod
    async def stop(self) -> None: ...


class PlaybackStream(abc.ABC):
    """One open raw-PCM playback stream (S16_LE mono, fixed rate).

    The CALLER owns it, so whoever holds the handle can always ``kill()`` it
    (including while another task is inside ``drain()``) and barge-in never races hidden
    backend state. ``drain()`` and ``kill()`` are terminal, idempotent, and may run
    concurrently with each other or an in-flight ``write()``, which then discards.
    """

    @abc.abstractmethod
    async def write(self, pcm: bytes) -> None: ...

    @property
    def dead(self) -> bool:
        """The device died UNDER the stream (not a deliberate drain/kill) and writes are
        being discarded, so the sink can reopen instead of feeding a corpse for the rest
        of the turn. False for backends that cannot tell."""
        return False

    @abc.abstractmethod
    async def drain(self) -> None:
        """End the stream and block until buffered audio finishes playing."""

    @abc.abstractmethod
    async def kill(self) -> None:
        """Stop playback NOW (barge-in), discarding buffered audio."""


class PlaybackSink(abc.ABC):
    """A speaker sink that plays complete WAV blobs one at a time. :meth:`open_stream`
    is the optional gapless raw-PCM path used by the stream-mode ``AudioSink``; the
    default raises so a backend without it fails loudly rather than mis-playing."""

    @abc.abstractmethod
    async def play_wav(self, wav_bytes: bytes) -> bool:
        """Play a complete WAV blob, returning when done: ``True`` if it finished
        naturally, ``False`` if aborted (barge-in) or failed."""

    @abc.abstractmethod
    async def abort(self) -> None:
        """Stop in-flight BLOB playback immediately (barge-in). Streams are killed
        through their own handles by whoever owns them."""

    async def stop(self) -> None:
        """Release resources; override only when teardown must differ from barge-in."""
        await self.abort()

    async def open_stream(self, rate: int) -> PlaybackStream:
        """Open a raw-PCM stream at ``rate`` Hz. The caller owns the handle and must
        ``drain()`` or ``kill()`` it."""
        raise NotImplementedError("streaming playback not supported by this backend")
