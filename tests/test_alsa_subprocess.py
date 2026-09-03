"""ALSA subprocess plumbing: a chatty child's stderr must never wedge audio."""

from __future__ import annotations

import asyncio
import stat
from contextlib import contextmanager

from loguru import logger

from nanobot_channel_voice.audio.alsa import AlsaCapture


@contextmanager
def _warnings():
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        yield messages
    finally:
        logger.remove(sink)

# ~500 KB of xrun spam: well past the 64 KB OS pipe plus asyncio's ~64 KB
# StreamReader buffer. alsa-utils prints these unconditionally (-q does not
# silence xrun()); before the stderr drain the child blocked inside fprintf
# before ever producing the audio, and read_frame() hung forever with no EOF.
_STUB = """#!/bin/sh
i=0
while [ $i -lt 12000 ]; do
  echo 'overrun!!! (at least 100.000 ms long)' >&2
  i=$((i+1))
done
dd if=/dev/zero bs=640 count=50 2>/dev/null
"""


def test_capture_survives_stderr_spam(tmp_path):
    stub = tmp_path / "fake_arecord"
    stub.write_text(_STUB)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    async def _case():
        cap = AlsaCapture("null", 16000, 20, arecord_path=str(stub))
        await cap.start()
        try:
            frames = [await cap.read_frame() for _ in range(50)]
            assert all(len(f) == 640 for f in frames)
        finally:
            await cap.stop()

    asyncio.run(asyncio.wait_for(_case(), timeout=15.0))


# A large burst of already-captured audio (the pipe/StreamReader backlog a lagging
# consumer faces), a pause marking the live edge, then one distinctive live frame.
_BACKLOGGED = r"""#!/bin/sh
dd if=/dev/zero bs=640 count=100 2>/dev/null
sleep 1
dd if=/dev/zero bs=640 count=1 2>/dev/null | LC_ALL=C tr '\0' '\377'
"""


def test_flush_discards_backlog_and_keeps_frame_alignment(tmp_path):
    """flush() must eat the buffered backlog in WHOLE frames and stop at the live
    edge: a partial-frame discard would shear the S16 sample alignment and turn
    every later frame into static."""
    stub = tmp_path / "backlogged_arecord"
    stub.write_text(_BACKLOGGED)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    async def _case():
        cap = AlsaCapture("null", 16000, 20, arecord_path=str(stub))
        await cap.start()
        try:
            assert len(await cap.read_frame()) == 640  # one consumed normally
            flushed = await cap.flush()
            assert flushed % 640 == 0  # whole frames only
            assert flushed >= 640 * 50  # the bulk of the 99-frame backlog is gone
            # The live-edge frame still arrives intact behind any un-flushed tail:
            # proof the stream never sheared.
            frame = await cap.read_frame()
            while frame and frame != b"\xff" * 640:
                assert len(frame) == 640
                frame = await cap.read_frame()
            assert frame == b"\xff" * 640
        finally:
            await cap.stop()

    asyncio.run(asyncio.wait_for(_case(), timeout=15.0))


# One frame, then a device-open failure. Every read after the child exits raises
# IncompleteReadError, so an unlatched diagnosis would repeat forever (the shell
# polls read_frame() continuously while it waits out the eof_streak).
_DYING = """#!/bin/sh
dd if=/dev/zero bs=640 count=1 2>/dev/null
echo 'audio open error: Device or resource busy' >&2
exit 1
"""


def test_dead_arecord_is_diagnosed_once(tmp_path):
    stub = tmp_path / "dying_arecord"
    stub.write_text(_DYING)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    async def _case():
        cap = AlsaCapture("null", 16000, 20, arecord_path=str(stub))
        with _warnings() as messages:
            await cap.start()
            try:
                assert len(await cap.read_frame()) == 640
                assert await cap.read_frame() == b""
                while cap._proc.returncode is None:  # reaped (outer wait_for bounds this)
                    await asyncio.sleep(0.01)
                assert [await cap.read_frame() for _ in range(5)] == [b""] * 5
            finally:
                await cap.stop()
        assert len([m for m in messages if "arecord ended" in m]) == 1

    asyncio.run(asyncio.wait_for(_case(), timeout=15.0))


def test_arecord_argv_pins_the_capture_ring(tmp_path):
    """Without --period-size/--buffer-size, alsa-utils defaults to a 500 ms buffer and a
    125 ms period, so frames reach the endpointer in 125 ms bursts."""
    stub = tmp_path / "argv_arecord"
    out = tmp_path / "argv.txt"
    stub.write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > {out}\nsleep 5\n')
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)

    async def _case():
        cap = AlsaCapture("null", 16000, 20, arecord_path=str(stub))
        await cap.start()
        try:
            for _ in range(50):
                await asyncio.sleep(0.02)
                if out.exists() and out.read_text():
                    break
        finally:
            await cap.stop()
        return out.read_text().split()

    argv = asyncio.run(asyncio.wait_for(_case(), timeout=15.0))
    assert "--period-size" in argv and argv[argv.index("--period-size") + 1] == "320"
    # 5 periods of 20 ms: one frame of granularity, 100 ms of overrun headroom.
    assert argv[argv.index("--buffer-size") + 1] == "1600"


def test_capture_ring_scales_with_a_short_frame(tmp_path):
    cap = AlsaCapture("null", 16000, 10)
    assert (cap._period_frames, cap._periods) == (160, 10)  # still ~100 ms

