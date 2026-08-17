"""In-process ALSA capture/playback via ``pyalsaaudio`` (libasound bindings).

Alternative to the ``arecord``/``aplay`` subprocess backend (``audio.backend =
"pyalsa"``; the ``[pyalsa]`` extra, which needs ``libasound2-dev`` to build). Opens
named PCMs (including ``dsnoop``/``dmix`` ``plug`` devices) and uses ``snd_pcm_drop``
for instant barge-in. Blocking libasound calls never run on the event loop: capture
lives in a reader thread, playback/streaming on the default executor.
"""

from __future__ import annotations

import asyncio
import io
import threading
import wave
from contextlib import suppress

from loguru import logger

from nanobot_channel_voice.aio import put_drop_oldest
from nanobot_channel_voice.audio.base import (
    CaptureSource,
    PlaybackSink,
    PlaybackStream,
    frame_bytes,
)

# WAV sample width (bytes) -> ALSA format name, resolved by getattr at use time.
_WIDTH_TO_FORMAT = {1: "PCM_FORMAT_U8", 2: "PCM_FORMAT_S16_LE", 4: "PCM_FORMAT_S32_LE"}


class PyAlsaCapture(CaptureSource):
    def __init__(self, device: str, sample_rate: int, frame_ms: int):
        self._device = device
        self._rate = sample_rate
        self._periodsize = max(1, sample_rate * frame_ms // 1000)
        self._frame_bytes = frame_bytes(sample_rate, frame_ms)
        self._pcm = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._queue: asyncio.Queue[bytes] | None = None
        self._log = logger.bind(component="pyalsa-capture")

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue(maxsize=50)  # ~1 s at 20 ms frames; drop-oldest bounds latency
        # On the executor (dsnoop setup can block), degrading to EOF like the arecord
        # backend: a bad/busy device must surface later as b"", not raise out of
        # start() and crash channel startup.
        try:
            self._pcm = await self._loop.run_in_executor(None, self._open_pcm)
        except Exception as exc:  # noqa: BLE001
            self._log.warning("pyalsa capture open failed ({}); no audio input", exc)
            self._pcm = None
            self._running = False
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader, name="pyalsa-capture", daemon=True)
        self._thread.start()
        self._log.info("pyalsa capture started (device={})", self._device)

    def _open_pcm(self):
        import alsaaudio

        return alsaaudio.PCM(
            type=alsaaudio.PCM_CAPTURE,
            mode=alsaaudio.PCM_NORMAL,
            device=self._device,
            rate=self._rate,
            channels=1,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=self._periodsize,
        )

    def _reader(self) -> None:
        buf = bytearray()
        target = self._frame_bytes
        while self._running:
            try:
                length, data = self._pcm.read()
            except Exception as exc:  # noqa: BLE001
                if self._running:
                    self._log.warning("capture read failed: {}", exc)
                # Stop so read_frame() sees `not _running` and returns EOF instead of
                # blocking forever on an empty queue (deaf-mic hang).
                self._running = False
                self._push(b"")
                return
            if length < 0 or not data:
                continue
            # Re-chunk to EXACT frame_bytes: ALSA may return a period differing from
            # the request, but the endpointer counts frames assuming a fixed frame_ms.
            buf += data
            while len(buf) >= target:
                self._push(bytes(buf[:target]))
                del buf[:target]

    def _push(self, data: bytes) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._enqueue, data)

    def _enqueue(self, data: bytes) -> None:
        if self._queue is not None:
            put_drop_oldest(self._queue, data)

    async def read_frame(self) -> bytes:
        q = self._queue
        if q is None:
            return b""
        if not self._running and q.empty():
            return b""
        return await q.get()

    async def flush(self) -> int:
        """Drop the hand-off queue's backlog (the reader thread stays at the device's
        live edge itself, so the queue IS this backend's entire stale window)."""
        q = self._queue
        if q is None:
            return 0
        dropped = 0
        while True:
            try:
                dropped += len(q.get_nowait())
            except asyncio.QueueEmpty:
                if dropped:
                    self._log.debug(
                        "flushed {} ms of stale capture backlog",
                        int(dropped / (2 * self._rate) * 1000),
                    )
                return dropped

    async def stop(self) -> None:
        self._running = False
        # EOF sentinel: the reader's NORMAL exit pushes nothing, so a caller already
        # parked in `await q.get()` would hang forever.
        self._push(b"")
        thread = self._thread
        joined = True
        if thread is not None:
            with suppress(Exception):  # bounded: a live device yields every frame_ms
                await asyncio.get_running_loop().run_in_executor(None, thread.join, 2.0)
            joined = not thread.is_alive()
        self._thread = None
        # Only once the reader has exited: closing a snd_pcm_t while another thread is
        # blocked in pcm.read() on it is undefined behaviour (can segfault). A wedged
        # reader instead leaks the handle to the daemon thread / process exit.
        if self._pcm is not None and joined:
            with suppress(Exception):
                self._pcm.close()
        self._pcm = None


class PyAlsaPlayback(PlaybackSink):
    """Blob WAVs *and* gapless raw-PCM streams, both on the executor.

    ``play_wav`` opens a PCM at the WAV's own rate/format (a ``plug:`` device
    resamples to hardware, mirroring ``aplay``). ``abort()`` covers the blob PCM only
    (streams are killed through their handles).
    """

    def __init__(self, device: str):
        self._device = device
        self._abort = False
        self._current = None   # blob-mode PCM, owned by _play_blocking
        self._lock = threading.Lock()
        self._log = logger.bind(component="pyalsa-playback")

    # ---- blob mode ----------------------------------------------------------

    async def play_wav(self, wav_bytes: bytes) -> bool:
        if not wav_bytes:
            return True
        self._abort = False
        return await asyncio.get_running_loop().run_in_executor(None, self._play_blocking, wav_bytes)

    def _play_blocking(self, wav_bytes: bytes) -> bool:
        import alsaaudio

        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav:
                rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
                pcm_data = wav.readframes(wav.getnframes())
        except Exception as exc:  # noqa: BLE001
            self._log.warning("unreadable WAV: {}", exc)
            return False

        fmt_name = _WIDTH_TO_FORMAT.get(width)
        if fmt_name is None:
            self._log.warning("unsupported sample width: {} bytes", width)
            return False

        periodsize = max(64, rate // 50)  # ~20 ms
        fbytes = channels * width
        try:
            pcm = alsaaudio.PCM(
                type=alsaaudio.PCM_PLAYBACK,
                mode=alsaaudio.PCM_NORMAL,
                device=self._device,
                rate=rate,
                channels=channels,
                format=getattr(alsaaudio, fmt_name),
                periodsize=periodsize,
            )
        except Exception as exc:  # noqa: BLE001
            self._log.warning("pyalsa open failed: {}", exc)
            return False

        with self._lock:
            self._current = pcm
        chunk = periodsize * fbytes
        try:
            for offset in range(0, len(pcm_data), chunk):
                if self._abort:
                    return False
                pcm.write(pcm_data[offset : offset + chunk])
            return not self._abort
        except Exception as exc:  # noqa: BLE001
            if not self._abort:
                self._log.warning("pyalsa write failed: {}", exc)
            return False
        finally:
            with self._lock:  # serialized with abort()'s drop
                self._current = None
                with suppress(Exception):
                    pcm.close()

    # ---- stream mode (gapless raw PCM) --------------------------------------

    async def open_stream(self, rate: int) -> PlaybackStream:
        pcm = await asyncio.get_running_loop().run_in_executor(
            None, self._open_stream_blocking, rate
        )
        return _PyAlsaPlaybackStream(pcm)

    def _open_stream_blocking(self, rate: int):
        import alsaaudio

        return alsaaudio.PCM(
            type=alsaaudio.PCM_PLAYBACK,
            mode=alsaaudio.PCM_NORMAL,
            device=self._device,
            rate=rate,
            channels=1,
            format=alsaaudio.PCM_FORMAT_S16_LE,
            periodsize=max(64, rate // 50),
        )

    # ---- lifecycle ----------------------------------------------------------

    async def abort(self) -> None:
        self._abort = True
        # Off the loop (drop is a libasound call) and INSIDE the lock the writer's
        # finally closes under: libasound's thread-safety layer does not cover
        # snd_pcm_close, so a lock-free drop could hit a freed handle (native crash).
        # The writer never holds the lock around pcm.write, so this cannot block.
        await asyncio.get_running_loop().run_in_executor(None, self._abort_blocking)

    def _abort_blocking(self) -> None:
        with self._lock:
            if self._current is not None:
                with suppress(Exception):
                    self._current.drop()


class _PyAlsaPlaybackStream(PlaybackStream):
    """One persistent stream-mode PCM, all blocking calls on the executor.

    Writes are serialized by the AudioSink worker, but ``kill()`` (barge-in) may land
    at ANY moment and overlap ``drain()``. So ``pcm.drop()`` runs OUTSIDE the io-lock —
    the documented way to unblock an in-flight ``pcm.write`` instantly — while
    ``pcm.close()`` always runs INSIDE it, never freeing the C object under a writer.
    Because libasound deliberately does not protect ``snd_pcm_close``, drop and close
    serialize on their own mutex (a lock-free drop overlapping close is a use-after-free
    of the ``snd_pcm_t``); drop never blocks nor takes the io-lock, so that mutex is
    cheap and cycle-free.
    """

    def __init__(self, pcm):
        self._pcm = pcm
        self._io_lock = threading.Lock()  # write/drain/close, not drop
        self._drop_lock = threading.Lock()  # drop vs close
        self._closed = False  # written under _drop_lock, right before close()
        # Separate intent flag: kill() cannot set _dead before its drop (that skips
        # _close_pcm and leaks the snd_pcm_t), and _dead sits behind the io-lock the
        # unblocked writer still holds.
        self._killing = False
        self._dead = False
        self._death_logged = False

    @property
    def dead(self) -> bool:
        return self._death_logged  # a deliberate drain/kill never latches it

    async def write(self, pcm: bytes) -> None:
        # _death_logged short-circuits too: after a real death every ~20 ms block
        # would otherwise cost an executor hop plus a failing libasound write.
        if self._dead or self._death_logged or not pcm:
            return
        await asyncio.get_running_loop().run_in_executor(None, self._write_blocking, pcm)

    def _write_blocking(self, data: bytes) -> None:
        with self._io_lock:
            if self._dead:
                return
            try:
                self._pcm.write(data)
            except Exception as exc:  # noqa: BLE001
                # A barge-in raises here too (kill()'s drop deliberately lands while we
                # hold the io-lock), so _killing/_closed mean "not a device failure".
                if not self._death_logged and not self._killing and not self._closed:
                    self._death_logged = True
                    logger.bind(component="pyalsa").warning(
                        "playback PCM write failed ({}); stream audio is being lost", exc
                    )

    async def drain(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self._drain_blocking)

    def _drain_blocking(self) -> None:
        with self._io_lock:
            if self._dead:
                return
            self._dead = True
            with suppress(Exception):
                self._pcm.drain()
            self._close_pcm()

    def _close_pcm(self) -> None:
        """Callers hold the io-lock; the drop-mutex keeps a concurrent kill()'s drop
        off the freed handle."""
        with self._drop_lock:
            self._closed = True
            with suppress(Exception):
                self._pcm.close()

    async def kill(self) -> None:
        await asyncio.get_running_loop().run_in_executor(None, self._kill_blocking)

    def _kill_blocking(self) -> None:
        # drop() first, outside the io-lock: unblocks a writer stuck in pcm.write and
        # cuts audio NOW.
        with self._drop_lock:
            self._killing = True  # latch intent BEFORE the drop the writer will raise on
            if not self._closed:
                with suppress(Exception):
                    self._pcm.drop()
        with self._io_lock:
            if self._dead:
                return
            self._dead = True
            self._close_pcm()
