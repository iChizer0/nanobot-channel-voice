"""_PyAlsaPlaybackStream drop/close serialization (the libasound UAF race).

libasound's thread-safety layer does not protect snd_pcm_close, so a barge-in
kill()'s lock-free drop() overlapping a drain()'s close() was a use-after-free
of the snd_pcm_t: a native segfault, which no amount of suppress(Exception)
catches. The fake PCM below detects exactly that overlap.
"""

from __future__ import annotations

import asyncio
import threading
import time
from contextlib import contextmanager

from loguru import logger

from nanobot_channel_voice.audio.pyalsa import _PyAlsaPlaybackStream


class _InstrumentedPcm:
    def __init__(self):
        self.closed = False
        self.violations: list[str] = []

    def write(self, data):
        time.sleep(0.001)

    def drain(self):
        time.sleep(0.003)

    def drop(self):
        if self.closed:
            self.violations.append("drop entered after close")
        time.sleep(0.003)  # widen the window: close landing here is the UAF
        if self.closed:
            self.violations.append("close ran during drop")

    def close(self):
        self.closed = True


class _StickyPcm(_InstrumentedPcm):
    """A write that parks until drop() lands, then fails the way libasound does:
    dropping a PCM under a blocked writer returns -EBADFD, not a short write."""

    def __init__(self):
        super().__init__()
        self.unblock = threading.Event()

    def write(self, data):
        self.unblock.wait(timeout=5.0)
        raise OSError(77, "File descriptor in bad state")

    def drop(self):
        super().drop()
        self.unblock.set()  # a real drop makes the blocked write return


@contextmanager
def _warnings():
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)


def test_kill_drop_never_overlaps_drain_close():
    async def _round():
        pcm = _InstrumentedPcm()
        stream = _PyAlsaPlaybackStream(pcm)
        await stream.write(b"\x00" * 64)
        # The documented overlap: drain() may overlap a late kill().
        await asyncio.gather(stream.drain(), stream.kill())
        assert pcm.closed  # somebody closed it exactly once, cleanly
        return pcm.violations

    async def _case():
        for _ in range(50):
            violations = await _round()
            assert violations == []

    asyncio.run(_case())


def test_kill_unblocks_a_stuck_writer_without_diagnosing_a_death():
    """drop() must stay OUT of the io-lock: it is what unblocks pcm.write. The
    -EBADFD that write then raises is OUR doing, so `dead` must stay clear (the
    contract reserves it for a device dying UNDER the stream) and no death may
    be logged: it would warn of lost stream audio on every barge-in that
    catches a write in flight."""

    async def _case():
        pcm = _StickyPcm()
        stream = _PyAlsaPlaybackStream(pcm)
        with _warnings() as messages:
            writer = asyncio.ensure_future(stream.write(b"\x00" * 64))
            await asyncio.sleep(0.05)  # writer is now parked inside pcm.write
            await asyncio.wait_for(stream.kill(), timeout=1.0)  # must not deadlock
            await asyncio.wait_for(writer, timeout=1.0)
        assert pcm.violations == []
        assert pcm.closed
        assert stream.dead is False
        assert [m for m in messages if "write failed" in m] == []

    asyncio.run(_case())


def test_write_failure_without_a_kill_is_a_death():
    """The contrast case: without it the test above passes on a `dead` that can
    never latch at all."""

    class _DyingPcm(_InstrumentedPcm):
        def write(self, data):
            raise OSError(19, "No such device")

    async def _case():
        pcm = _DyingPcm()
        stream = _PyAlsaPlaybackStream(pcm)
        with _warnings() as messages:
            await stream.write(b"\x00" * 64)
        assert stream.dead is True
        assert any("write failed" in m for m in messages)

    asyncio.run(_case())


def test_capture_flush_drains_the_handoff_queue():
    """PyAlsaCapture's stale window is its hand-off queue (the reader thread stays
    at the device's live edge); flush must empty it non-blocking and count bytes."""
    from nanobot_channel_voice.audio.pyalsa import PyAlsaCapture

    async def _case():
        cap = PyAlsaCapture("null", 16000, 20)
        assert await cap.flush() == 0  # never started: nothing to drain, no hang
        cap._queue = asyncio.Queue()
        for _ in range(5):
            cap._queue.put_nowait(b"\x00" * 640)
        assert await cap.flush() == 5 * 640
        assert cap._queue.empty()
        assert await cap.flush() == 0

    asyncio.run(_case())
