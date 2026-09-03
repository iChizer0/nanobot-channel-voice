"""ALSA capture/playback via the ``arecord``/``aplay`` subprocesses: named ALSA PCMs
(shared ``dsnoop``/``dmix`` ``plug`` devices work by name), no C-extension build,
barge-in is an instant ``kill()``.
"""

from __future__ import annotations

import asyncio
from collections import deque
from contextlib import suppress

from loguru import logger

from nanobot_channel_voice.audio.base import (
    CaptureSource,
    PlaybackSink,
    PlaybackStream,
    frame_bytes,
)

_PIPE = asyncio.subprocess.PIPE
_DEVNULL = asyncio.subprocess.DEVNULL

# Ring depth for a capture device, in ms. alsa-utils otherwise defaults to a 500 ms
# buffer and a 125 ms period, so frames reach the endpointer in 125 ms bursts.
_CAPTURE_RING_MS = 100


async def _terminate(proc: asyncio.subprocess.Process | None) -> None:
    """Kill *proc* and reap it, but never block the loop indefinitely."""
    if proc is None or proc.returncode is not None:
        return
    with suppress(ProcessLookupError):
        proc.kill()  # SIGKILL: barge-in must cut instantly; arecord/aplay flush nothing
    with suppress(Exception):
        await asyncio.wait_for(proc.wait(), timeout=2.0)


class _StderrTail:
    """Continuously drain a child's stderr, keeping only the last few chunks.

    alsa-utils prints xrun complaints unconditionally (``-q`` does not silence them).
    Unread, the 64 KB pipe buffer fills and the child blocks in ``fprintf``, silently
    starving audio with no EOF for anything upstream to notice."""

    def __init__(self, proc: asyncio.subprocess.Process):
        self._chunks: deque[bytes] = deque(maxlen=4)  # last <=16 KB
        self._task: asyncio.Task | None = (
            asyncio.ensure_future(self._pump(proc.stderr)) if proc.stderr is not None else None
        )

    async def _pump(self, stream: asyncio.StreamReader) -> None:
        with suppress(Exception):
            while chunk := await stream.read(4096):
                self._chunks.append(chunk)

    async def text(self) -> str:
        """The retained tail; call once the child has exited, so the pump hits EOF."""
        if self._task is not None:
            with suppress(Exception):
                await asyncio.wait_for(asyncio.shield(self._task), timeout=1.0)
        return b"".join(self._chunks).decode("utf-8", "replace").strip()


class AlsaCapture(CaptureSource):
    """Continuous mic capture: one long-lived ``arecord`` streaming raw PCM."""

    def __init__(
        self,
        device: str,
        sample_rate: int,
        frame_ms: int,
        arecord_path: str = "arecord",
    ):
        self._device = device
        self._sample_rate = sample_rate
        self._frame_bytes = frame_bytes(sample_rate, frame_ms)
        # flush(): a backlogged frame returns ~instantly, a LIVE one takes a full
        # frame_ms; half a period separates them and bounds the whole flush.
        self._flush_step_s = frame_ms / 2000.0
        self._flush_cap = sample_rate * 2 * 10  # never spin past ~10 s of backlog
        # One period IS one frame, so capture granularity matches the endpointer's.
        self._period_frames = max(1, sample_rate * frame_ms // 1000)
        self._periods = max(4, -(-_CAPTURE_RING_MS // frame_ms))
        self._bin = arecord_path
        self._proc: asyncio.subprocess.Process | None = None
        self._stderr: _StderrTail | None = None
        self._eof_logged = False
        self._log = logger.bind(component="alsa-capture")

    async def start(self) -> None:
        cmd = [
            self._bin, "-q",
            "-D", self._device,
            "-f", "S16_LE",
            "-c", "1",
            "-r", str(self._sample_rate),
            "-t", "raw",
            "--period-size", str(self._period_frames),
            "--buffer-size", str(self._period_frames * self._periods),
        ]
        self._eof_logged = False
        self._proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=_PIPE, stderr=_PIPE,
        )
        self._stderr = _StderrTail(self._proc)
        self._log.info("arecord started (device={})", self._device)

    async def read_frame(self) -> bytes:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return b""
        try:
            return await proc.stdout.readexactly(self._frame_bytes)
        except asyncio.IncompleteReadError:
            # arecord exited. Latch one line per start() cycle BEFORE the stderr await,
            # so a concurrent reader cannot slip through; stop() nulls _proc, so a
            # deliberate stop stays silent.
            if self._running_died() and not self._eof_logged:
                self._eof_logged = True
                err = await self._stderr.text() if self._stderr else ""
                self._log.warning("arecord ended (rc={}): {}", proc.returncode, err or "eof")
            return b""  # empty frames drive the shell's eof_streak -> restart

    def _running_died(self) -> bool:
        return self._proc is not None and self._proc.returncode is not None

    async def flush(self) -> int:
        """Drop buffered whole frames until the stream runs at the live edge.

        ``readexactly`` under a timeout: it extracts only once ALL n bytes are present,
        so a timeout mid-wait leaves the buffer intact and S16 alignment cannot shear."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return 0
        dropped = 0
        try:
            while dropped < self._flush_cap:
                await asyncio.wait_for(
                    proc.stdout.readexactly(self._frame_bytes), timeout=self._flush_step_s
                )
                dropped += self._frame_bytes
        except (TimeoutError, asyncio.IncompleteReadError):
            pass  # live edge reached, or arecord ended (read_frame diagnoses that)
        if dropped:
            self._log.debug(
                "flushed {} ms of stale capture backlog",
                int(dropped / (2 * self._sample_rate) * 1000),
            )
        return dropped

    async def stop(self) -> None:
        proc, self._proc = self._proc, None
        await _terminate(proc)


class AlsaPlayback(PlaybackSink):
    """Plays one WAV blob at a time by piping it to a fresh ``aplay``."""

    def __init__(self, device: str, aplay_path: str = "aplay"):
        self._device = device
        self._bin = aplay_path
        self._proc: asyncio.subprocess.Process | None = None  # blob-mode aplay
        self._aborted = False
        self._log = logger.bind(component="alsa-playback")

    async def play_wav(self, wav_bytes: bytes) -> bool:
        if not wav_bytes:
            return True
        cmd = [self._bin, "-q", "-D", self._device, "-"]
        # Clear BEFORE the spawn await, re-check AFTER publishing _proc: an abort() in
        # the fork/exec window would otherwise be clobbered and the blob play through.
        self._aborted = False
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, stdin=_PIPE, stdout=_DEVNULL, stderr=_PIPE,
            )
        except OSError as exc:
            self._log.warning("aplay launch failed: {}", exc)
            return False

        self._proc = proc
        if self._aborted:
            self._proc = None
            await _terminate(proc)
            return False
        stderr = b""
        try:
            # An abort() mid-write returns here promptly, as BrokenPipeError.
            _, stderr = await proc.communicate(wav_bytes)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except asyncio.CancelledError:
            # Kill NOW: once the finally clears _proc no later abort() can see this
            # child, and an orphaned aplay keeps playing its buffered audio.
            await _terminate(proc)
            raise
        except Exception as exc:  # noqa: BLE001 - never crash the loop
            self._log.warning("aplay error: {}", exc)
        finally:
            self._proc = None

        if self._aborted:
            return False
        if proc.returncode not in (0, None):
            msg = stderr.decode("utf-8", "replace").strip() if stderr else ""
            self._log.warning("aplay rc={} {}", proc.returncode, msg)
            return False
        return True

    async def abort(self) -> None:
        self._aborted = True
        proc, self._proc = self._proc, None
        await _terminate(proc)  # blob only; streams die via their handles

    # ---- streaming (raw PCM to one persistent aplay per handle) -------------

    async def open_stream(self, rate: int) -> PlaybackStream:
        cmd = [
            self._bin, "-q", "-D", self._device,
            "-f", "S16_LE", "-c", "1", "-r", str(rate), "-t", "raw", "-",
        ]
        # A spawn failure raises to the caller (the AudioSink worker logs it).
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdin=_PIPE, stdout=_DEVNULL, stderr=_PIPE,
        )
        self._log.info("aplay stream started (device={}, rate={})", self._device, rate)
        return _AlsaPlaybackStream(proc)


class _AlsaPlaybackStream(PlaybackStream):
    """One persistent ``aplay``: the handle IS the process, killable at any point,
    including mid-``drain()`` — that is what stops the buffered tail on barge-in."""

    def __init__(self, proc: asyncio.subprocess.Process):
        self._proc = proc
        self._stderr = _StderrTail(proc)
        self._killed = False
        self._eof = False  # drain() sent EOF: late writes are expected no-ops
        self._death_logged = False
        self._log = logger.bind(component="alsa")

    @property
    def dead(self) -> bool:
        return self._proc.returncode is not None and not self._killed and not self._eof

    async def _log_death(self) -> None:
        """Surface an aplay that died underneath us: device-open failures land AFTER
        spawn and would otherwise show only as eaten writes. Once per handle; a
        deliberate kill() is not a death."""
        if self._death_logged or self._killed:
            return
        self._death_logged = True
        err = await self._stderr.text()
        self._log.warning(
            "aplay stream died (rc={}): {}", self._proc.returncode, err[:200] or "no stderr",
        )

    async def write(self, pcm: bytes) -> None:
        proc = self._proc
        if self._killed or self._eof or proc.stdin is None or not pcm:
            return
        if proc.returncode is not None:  # exited under us (not our kill)
            await self._log_death()
            return
        try:
            proc.stdin.write(pcm)
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            await self._log_death()  # killed mid-write is normal; _log_death filters it

    async def drain(self) -> None:
        # Idempotent: the sink's reaper re-drains a handle whose first drain was
        # cancelled, and a second write_eof() would raise and needlessly terminate.
        proc = self._proc
        try:
            if not self._eof and proc.stdin is not None:
                self._eof = True
                proc.stdin.write_eof()  # EOF -> aplay plays out its buffer, then exits
            await asyncio.wait_for(proc.wait(), timeout=30.0)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except TimeoutError:
            # Wedged device (e.g. a stuck dmix); _killed so rc=-9 is not a "death".
            self._killed = True
            self._log.warning("aplay did not exit within 30s after EOF; killing")
            await _terminate(proc)
        except Exception as exc:  # noqa: BLE001 - never hang/crash the loop
            self._killed = True
            self._log.warning("aplay drain failed ({}); killing", exc)
            await _terminate(proc)
        if proc.returncode not in (0, None):  # exited nonzero on its own after EOF
            await self._log_death()

    async def kill(self) -> None:
        # Safe at any moment: _terminate no-ops on an already-exited process, and a
        # concurrent drain()'s wait() simply returns once we kill.
        self._killed = True
        await _terminate(self._proc)
