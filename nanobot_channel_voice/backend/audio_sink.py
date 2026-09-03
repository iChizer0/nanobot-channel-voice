"""Shared audio output sink.

One worker plays queued audio in order, gating on each item's CARRIED interrupt epoch
so a barge-in drops already-queued audio from the cancelled turn.

Modes: ``blob`` plays TTS WAVs byte-for-byte via ``PlaybackSink.play_wav``; ``stream``
writes raw PCM to a persistent device stream (gapless + played-ms accounting).
Synthesis lives in the backends; the duck envelope is applied here, per written block,
stream mode only.
"""

from __future__ import annotations

import array
import asyncio
import time
from contextlib import suppress
from typing import Literal

from loguru import logger

from nanobot_channel_voice.audio.base import PlaybackSink, PlaybackStream
from nanobot_channel_voice.audio.pcm import pcm_ms, pcm_to_wav_bytes, wav_duration_ms

from .base import OutputAudio

# Duck envelope (stream mode): attack fast so the bot yields at speech onset, release
# slow so the level doesn't pump on VAD flicker.
_DUCK_ATTACK_MS = 30.0
_DUCK_RELEASE_MS = 250.0
# Write-ahead of the wall clock: bounds both gain-change latency and barge-in discard.
_STREAM_LEAD_MS = 240.0
# Tighter lead when a duck floor (or pause) is configured: the lead IS the duck's
# audible latency (written audio keeps its baked gain); trades underrun headroom for
# reaction time.
_DUCK_STREAM_LEAD_MS = 120.0
_GAIN_BLOCK_MS = 20  # envelope granularity (one gain step per block)
# A ~250 ms cue is audibly late behind an open this slow; below it, nobody notices.
_SLOW_OPEN_MS = 80.0
# Producer-side backlog ceiling: playback is real time while blob-mode synthesis is
# not, so an uncapped queue holds a whole reply's audio. Far above the JIT runway.
MAX_BACKLOG_MS = 30_000.0
# Memory safety valve for producers that never park (a realtime model streams a reply
# ~3x faster than it plays, so a long reply legitimately backlogs well past the paced
# cap); sized so only a runaway source ever trips it.
UNPACED_BACKLOG_MS = 300_000.0


try:  # numpy ships with the [ondevice] extra; pure-python loops are the fallback
    import numpy as _np
except ImportError:  # pragma: no cover - environment-dependent
    _np = None


def trim_lead_silence(pcm: bytes, rate: int, *, cap_ms: float, threshold: float = 0.01) -> bytes:
    """Cap the leading silence of raw S16_LE PCM at ``cap_ms``.

    VITS-family TTS emits up to ~250 ms before the first phoneme. Trimming only the
    excess keeps a preroll for soft onsets and deliberate pauses. ``threshold`` is the
    onset peak as a fraction of full scale; an all-silent scan returns unchanged.
    """
    n = len(pcm) & ~1
    if n == 0 or rate <= 0:
        return pcm
    thresh = threshold * 32767.0
    if _np is not None:
        x = _np.frombuffer(pcm[:n], dtype=_np.int16)
        idx = _np.flatnonzero(_np.abs(x.astype(_np.int32)) > thresh)
        first = int(idx[0]) if idx.size else None
    else:
        samples = array.array("h")
        samples.frombytes(pcm[:n])
        first = None
        for i in range(min(len(samples), rate * 2)):  # bound the pure-python scan
            if abs(samples[i]) > thresh:
                first = i
                break
    if first is None:
        return pcm
    cap_samples = int(rate * cap_ms / 1000.0)
    if first <= cap_samples:
        return pcm
    return pcm[(first - cap_samples) * 2:]


def trim_tail_silence(pcm: bytes, rate: int, *, cap_ms: float, threshold: float = 0.01) -> bytes:
    """Cap the TRAILING silence of raw S16_LE PCM at ``cap_ms``.

    The tail is the same whatever the text ends with (measured 580-790 ms on matcha), so it
    is padding and never a rendered pause: it holds a canned clip's turn SPEAKING with the
    half-duplex mic gated, and makes a reply chunk's seam. All-silent: unchanged.
    """
    n = len(pcm) & ~1
    if n == 0 or rate <= 0:
        return pcm
    thresh = threshold * 32767.0
    if _np is not None:
        x = _np.frombuffer(pcm[:n], dtype=_np.int16)
        idx = _np.flatnonzero(_np.abs(x.astype(_np.int32)) > thresh)
        last = int(idx[-1]) if idx.size else None
    else:
        samples = array.array("h")
        samples.frombytes(pcm[:n])
        last = None
        for i in range(len(samples) - 1, max(-1, len(samples) - rate * 2 - 1), -1):
            if abs(samples[i]) > thresh:
                last = i
                break
    if last is None:
        return pcm
    end = last + 1 + int(rate * cap_ms / 1000.0)
    if end * 2 >= n:
        return pcm
    return pcm[: end * 2]


def scale_pcm(pcm: bytes, gain: float) -> bytes:
    """Scale raw S16_LE PCM by linear ``gain``; boosts saturate at full scale."""
    if gain == 1.0 or len(pcm) < 2:
        return pcm
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if _np is not None:
        # Per 20 ms block on the loop; the python loop is ~100x costlier on RK-class SoCs.
        return (
            _np.clip(_np.frombuffer(pcm, dtype=_np.int16) * gain, -32768.0, 32767.0)
            .astype(_np.int16)
            .tobytes()
        )
    samples = array.array("h")
    samples.frombytes(pcm)
    for i in range(len(samples)):
        samples[i] = max(-32768, min(32767, int(samples[i] * gain)))
    return samples.tobytes()


class AudioSink:
    def __init__(self, sink: PlaybackSink, *, mode: Literal["blob", "stream"] = "blob"):
        self._sink = sink
        self._mode = mode
        self._queue: asyncio.Queue[OutputAudio] = asyncio.Queue()
        self._epoch = 0
        self._worker: asyncio.Task | None = None
        self._idle = asyncio.Event()
        self._idle.set()
        # One device stream per turn; the handle stays in this slot until truly
        # finished (drain included) so flush() can ALWAYS kill it.
        self._stream: PlaybackStream | None = None
        self._stream_open_t = 0.0
        self._warned_slow_open = False
        self._bytes = 0
        # Bumped per FRESH stream: played_ms() restarts at 0, voiding earlier offsets.
        self._generation = 0
        self._rate = 0
        self._queued_ms = 0.0  # duration of queued-but-unwritten items (backlog_ms)
        self._dropped_ms = 0.0  # audio the overflow guard discarded this session
        self._overflow_warned = False
        # Set whenever _queued_ms shrinks (item played, flush): wakes wait_backlog_below.
        self._space = asyncio.Event()
        self._space.set()
        # Drain (EOF) begun: unwritable, still killable; flush() kills BOTH.
        self._draining: PlaybackStream | None = None
        self._reapers: set[asyncio.Task] = set()  # detached natural-drain finishers
        # Streams the reapers still hold: _draining is one slot a later drain
        # overwrites, so flush()/stop() sweep this set too.
        self._parked: set[PlaybackStream] = set()
        # Duck floor, set once by the owner; 1.0 = feature off.
        self._duck_floor = 1.0
        self._gain = 1.0
        self._gain_target = 1.0
        # Pause gate (bargeIn.mode="pause"): set = flowing. Clearing stalls the writer
        # between blocks; only the ~lead ms already at the device rings out.
        self._pause_gate = asyncio.Event()
        self._pause_gate.set()
        self._paused_at: float | None = None
        self._pause_capable = False  # tightens the pacing lead, like a duck floor
        self._ref_tap = None
        self._log = logger.bind(component="sink")

    def set_reference_tap(self, tap) -> None:
        """Register an AEC front-end (``push_reference``/``reference_dropped``).
        Stream mode only; fed each block as written — post-envelope, i.e. what the
        speaker plays — with its playout time."""
        self._ref_tap = tap

    @property
    def stream_mode(self) -> bool:
        return self._mode == "stream"

    async def prewarm(self, rate: int) -> None:
        """Play ~40 ms of silence through the real device path once, so device open is
        off the first reply's TTFA and a wrong playbackDevice fails loudly at startup.
        Own short-lived handle, worker ``_stream`` slot untouched — the caller must gate
        on an idle turn. Never raises."""
        silence = b"\x00" * (int(rate / 1000 * 40) * 2)
        try:
            if self._mode == "stream":
                stream = await self._sink.open_stream(rate)
                try:
                    await stream.write(silence)
                    await stream.drain()
                except BaseException:
                    await stream.kill()
                    raise
            else:
                await self._sink.play_wav(pcm_to_wav_bytes(silence, rate))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - warmup must never take the channel down
            logger.warning(
                "voice: playback prewarm failed ({}); the first reply will retry "
                "the device open — check audio.playbackDevice if this repeats", exc,
            )

    def configure_duck(self, floor: float) -> None:
        """Set the duck depth as a linear gain floor (e.g. -12 dB -> 0.25)."""
        self._duck_floor = min(1.0, max(0.0, floor))

    def duck(self, active: bool) -> None:
        """Duck (or release) live playback toward the configured floor.

        Reversible stage 1 of duck-then-confirm: the caller flushes or releases once the
        transcript verdict is in. No-op in blob mode / with no floor.
        """
        self._gain_target = self._duck_floor if active else 1.0

    def configure_pause(self, capable: bool) -> None:
        """Declare pause-mode barge-in, once by the owner: tightens the pacing lead so a
        pause silences within ~one lead."""
        self._pause_capable = capable

    def pause(self, active: bool) -> None:
        """Pause (or resume) live stream playback. Nothing is discarded: the writer
        stalls between blocks and continues where it stopped. No-op in blob mode."""
        if self._mode != "stream":
            return
        if active:
            if self._pause_gate.is_set():
                self._pause_gate.clear()
                self._paused_at = time.monotonic()
        else:
            if self._paused_at is not None and self._stream is not None:
                # Splice the paused span out of the stream clock, or every
                # elapsed-based read counts the silence as playout.
                self._stream_open_t += time.monotonic() - self._paused_at
            self._pause_gate.set()
            self._paused_at = None

    def restore_playback(self) -> None:
        """End any candidate attenuation (full gain, gate open) so no exit path can
        strand playback quiet or stalled."""
        self.duck(False)
        self.pause(False)

    @property
    def paused(self) -> bool:
        return not self._pause_gate.is_set()

    def _clock_now(self) -> float:
        """Playout clock: wall time, FROZEN at the pause edge while paused, else every
        read-side consumer thinks the paused audio kept draining. The writer's
        starvation re-anchor rebases it on resume."""
        return self._paused_at if self._paused_at is not None else time.monotonic()

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def stream_generation(self) -> int:
        """Identity of the stream ``played_ms()`` currently measures."""
        return self._generation

    @property
    def next_generation(self) -> int:
        """Identity audio enqueued NOW will play on: the stream opens lazily in the
        worker, so a producer reading ``stream_generation`` before its first write
        anchors to a stream that is already gone."""
        live = (
            self._stream is not None
            and self._stream is not self._draining
            and self._stream not in self._parked
            and not self._stream.dead  # _stream_write reopens under a dead handle
        )
        return self._generation if live else self._generation + 1

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def played_ms(self) -> int:
        """Stream mode: ms ACTUALLY heard on the current stream, ``min(wall-clock since
        open, buffered)``. Authoritative for ``conversation.item.truncate``: the cloud
        streams far faster than real time, so bytes-emitted over-counts. Blob: 0."""
        if self._stream is None or self._rate <= 0:
            return 0
        elapsed = (self._clock_now() - self._stream_open_t) * 1000.0
        buffered = self._bytes / (2 * self._rate) * 1000.0
        return int(min(elapsed, buffered))

    def _lead_ms(self) -> float:
        if self._duck_floor < 1.0 or self._pause_capable:
            return _DUCK_STREAM_LEAD_MS
        return _STREAM_LEAD_MS

    def lead_ms(self) -> float:
        """Write-ahead the device may still play after a pause/kill; consumers deriving
        playout physics ask here instead of copying the constants."""
        return self._lead_ms()

    def backlog_ms(self) -> int:
        """Estimated ms of accepted-but-not-yet-audible audio: queued items plus the
        written-but-unplayed lead. Deliberately over-counts the in-flight item by up to
        one lead — its consumer is the echo hold, where more only guards longer.
        """
        total = self._queued_ms
        if self._mode == "stream" and self._stream is not None and self._rate > 0:
            elapsed = (self._clock_now() - self._stream_open_t) * 1000.0
            buffered = self._bytes / (2 * self._rate) * 1000.0
            total += max(0.0, buffered - elapsed)
        return int(total)

    @staticmethod
    def _item_ms(item: OutputAudio) -> float:
        if item.pcm is not None and item.rate:
            return pcm_ms(len(item.pcm), item.rate)
        if item.wav is not None:
            # Blobs count too: the echo filter's hold must cover the whole blob.
            return wav_duration_ms(item.wav)
        return 0.0

    def starved(self) -> bool:
        """Is playback running (or about to run) dry? Stream: the wall clock consumed
        everything buffered; blob: the queue is idle. Meaningful only when checked as a
        NON-first chunk is enqueued (synthesis lost the race, the user heard a gap)."""
        if self._mode == "stream":
            if not self._pause_gate.is_set():
                return False  # deliberately silent, not starved
            if self._stream is None:
                # Accepted-but-unwritten audio has not reached the device: not starvation.
                return self._queued_ms <= 0.0
            if self._rate <= 0:
                return True
            elapsed = (time.monotonic() - self._stream_open_t) * 1000.0
            return elapsed >= self._bytes / (2 * self._rate) * 1000.0
        return not self.busy

    def starved_ms(self) -> float:
        """How long the stream has been audibly dry; 0 when healthy, paused, between
        streams, or blob. Read BEFORE the next write: ``_stream_write``'s starvation
        re-anchor rebases the clock and erases the evidence."""
        if self._mode != "stream" or self._stream is None or self._rate <= 0:
            return 0.0
        if not self._pause_gate.is_set():
            return 0.0
        elapsed = (self._clock_now() - self._stream_open_t) * 1000.0
        return max(0.0, elapsed - self._bytes / (2 * self._rate) * 1000.0)

    async def start(self) -> None:
        if self._worker is None:
            self._worker = asyncio.create_task(self._run())

    async def stop(self) -> None:
        # A cancelled worker never debits again: release any parked producer now.
        self._space.set()
        if self._worker is not None:
            self._worker.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await self._worker
            self._worker = None
        await self._kill_streams()
        if self._reapers:
            await asyncio.gather(*list(self._reapers), return_exceptions=True)
        await self._sink.stop()

    async def _kill_streams(self) -> None:
        """Stop every handle this sink can still reach: live, draining, parked."""
        stream, self._stream = self._stream, None
        draining, self._draining = self._draining, None
        parked, self._parked = self._parked, set()
        for s in {id(x): x for x in (draining, stream, *parked) if x is not None}.values():
            with suppress(Exception):
                await s.kill()

    def enqueue(self, audio: OutputAudio) -> None:
        """Queue ready-to-play audio. Sync, non-blocking; the item keeps its own
        epoch and the worker drops it if a flush intervened."""
        self._idle.clear()
        self._drop_overflow()  # judged BEFORE this item: a paced producer already waited
        if audio.epoch == self._epoch:
            # Credit only at the live epoch: a flush zeroes the counter (see _run).
            self._queued_ms += self._item_ms(audio)
        self._queue.put_nowait(audio)

    def _drop_overflow(self) -> None:
        """Bound the queue for producers that never park (see pace_output_audio): past the
        cap the OLDEST queued item goes, trading one skip for unbounded growth."""
        dropped = 0.0
        while self._queued_ms > UNPACED_BACKLOG_MS and not self._queue.empty():
            item = self._queue.get_nowait()
            self._queue.task_done()
            if item.epoch == self._epoch:
                item_ms = self._item_ms(item)
                self._queued_ms = max(0.0, self._queued_ms - item_ms)
                dropped += item_ms
        if not dropped:
            return
        self._dropped_ms += dropped
        self._space.set()
        if not self._overflow_warned:
            self._overflow_warned = True
            self._log.warning(
                "playback backlog over {:.0f} s; dropping the oldest queued audio "
                "({:.0f} ms so far) — the source is producing faster than real time",
                UNPACED_BACKLOG_MS / 1000.0, self._dropped_ms,
            )

    @property
    def dropped_ms(self) -> float:
        """Audio the overflow guard discarded, cumulative."""
        return self._dropped_ms

    async def wait_backlog_below(self, cap_ms: float = MAX_BACKLOG_MS) -> None:
        """Producer-side backpressure: block until the queued backlog is under
        ``cap_ms``. Cannot wedge — the worker debits as items play (real time) and
        ``flush()`` zeroes the backlog outright, both waking waiters."""
        while self._queued_ms > cap_ms:
            self._space.clear()
            if self._queued_ms <= cap_ms:  # a debit raced the clear
                break
            await self._space.wait()

    async def flush(self) -> int:
        """Barge-in: invalidate queued/in-flight audio and stop playback now.
        Returns ms actually heard on the current stream (for the cloud's
        ``conversation.item.truncate``); blob mode returns 0."""
        self._epoch += 1
        drained = 0
        while not self._queue.empty():
            with suppress(asyncio.QueueEmpty):
                self._queue.get_nowait()
                self._queue.task_done()
                drained += 1
        self._queued_ms = 0.0
        self._space.set()
        played = self.played_ms() if self._mode == "stream" else 0
        # Kill through OUR handles: valid mid-drain and for reaper-parked tails too.
        await self._kill_streams()
        # A flush ends any candidate interrupt: next turn at full level, unpaused (a
        # writer stalled on the gate wakes into the epoch check).
        self._gain = self._gain_target = 1.0
        self._pause_gate.set()
        self._paused_at = None
        if self._ref_tap is not None:
            self._ref_tap.reference_dropped()  # the killed tail never sounds
        await self._sink.abort()  # blob playback (no-op in stream mode)
        self._idle.set()
        if drained:
            self._log.debug("flushed {} queued item(s) at epoch {}", drained, self._epoch)
        return played

    async def wait_idle(self) -> None:
        """Block until the queue is empty (blob: playback done; stream: all PCM written)."""
        await self._idle.wait()

    async def drain_stream(self) -> None:
        """Stream mode: after the queue drains, end the stream and block until the
        buffered audio finishes playing (no-op in blob mode / when closed). The
        handle stays in ``self._stream`` DURING the drain so a concurrent
        ``flush()`` can kill the tail; detached after, if a flush didn't first."""
        # A paused drain WAITS for the verdict rather than forcing the gate: forcing
        # would play the final chunk at FULL level into the open mic. Verdicts always
        # resolve (release/kill/orphan), so this cannot wedge.
        await self._pause_gate.wait()
        await self.wait_idle()
        stream = self._stream
        if self._mode != "stream" or stream is None:
            return
        self._draining = stream  # EOF imminent: new writes must open fresh
        try:
            await stream.drain()
        except asyncio.CancelledError:
            # EOF already reached the device, so the tail plays out; killing here would
            # chop a status line's end on every fast tool round-trip. Park instead.
            self._spawn_reaper(stream)
            raise
        if self._draining is stream:
            self._draining = None
        if self._stream is stream:  # a concurrent flush may have taken it
            self._stream = None

    def _note_open_cost(self, open_ms: float) -> None:
        """A drain EOFs the stream, so every segment reopens the device; the shorter the
        segment the more of it this is (a ~250 ms earcon most of all)."""
        if open_ms < _SLOW_OPEN_MS or self._warned_slow_open:
            return
        self._warned_slow_open = True
        self._log.warning(
            "playback device open took {:.0f} ms and EVERY segment pays it: expect a late "
            "earcon and a late first word. Check audio.playbackDevice — a plughw:/dmix path "
            "rebuilds its conversion chain on each open.",
            open_ms,
        )

    def _spawn_reaper(self, stream: PlaybackStream) -> None:
        self._parked.add(stream)

        async def _reap() -> None:
            try:
                with suppress(Exception):
                    await stream.drain()  # idempotent: EOF resend no-ops, then reap
            finally:
                self._parked.discard(stream)
                if self._stream is stream:
                    self._stream = None
                if self._draining is stream:
                    self._draining = None

        task = asyncio.create_task(_reap())
        self._reapers.add(task)
        task.add_done_callback(self._reapers.discard)

    async def _run(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                if item.epoch != self._epoch:  # pre-play guard on the CARRIED epoch
                    continue
                if item.wav is not None:
                    await self._sink.play_wav(item.wav)
                elif item.pcm is not None:
                    await self._stream_write(item.pcm, item.rate, item.epoch)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never let one item kill the worker
                self._log.warning("playback error: {}", exc)
            finally:
                if item.epoch == self._epoch:
                    # Debit only what enqueue credited, else a dead item thins the
                    # NEXT turn's echo hold.
                    self._queued_ms = max(0.0, self._queued_ms - self._item_ms(item))
                self._space.set()
                self._queue.task_done()
                if self._queue.empty():
                    self._idle.set()

    def _block_bytes(self, rate: int) -> int:
        """Write granularity. Only a gain envelope, a pause gate or an AEC tap needs
        _GAIN_BLOCK_MS steps; plain playback pays one executor hop per block for nothing,
        so it writes a quarter of the lead instead (the device ring keeps 3/4 of it)."""
        shaped = (
            self._duck_floor < 1.0 or self._pause_capable or self._ref_tap is not None
        )
        ms = _GAIN_BLOCK_MS if shaped else self._lead_ms() / 4.0
        return max(2, int(rate * ms / 1000.0) * 2)

    async def _stream_write(self, pcm: bytes, rate: int, epoch: int) -> None:
        stream = self._stream
        if stream is not None and (stream is self._draining or stream in self._parked):
            # EOF'd (or parked in a reaper): no more writes. Open fresh; flush kills both.
            stream = None
        if stream is not None and stream.dead:
            # Device died UNDER the handle (e.g. exclusive hw: refused a second open):
            # without a reopen every later write this turn is silently discarded.
            with suppress(Exception):
                await stream.kill()  # reap; already dead, so nothing audible stops
            if self._stream is stream:
                self._stream = None
            stream = None
        if stream is not None and rate != self._rate:  # rate change (unexpected): reopen
            # Handle stays in _stream during the drain: a flush must reach the tail.
            self._draining = stream
            try:
                await stream.drain()
            except asyncio.CancelledError:
                self._spawn_reaper(stream)  # same parking as drain_stream
                raise
            finally:
                if self._draining is stream:
                    self._draining = None
                if self._stream is stream:
                    self._stream = None
            stream = None
        if stream is None:
            open_t0 = time.monotonic()
            stream = await self._sink.open_stream(rate)
            self._note_open_cost((time.monotonic() - open_t0) * 1000.0)
            if epoch != self._epoch:
                # flush()'s sweep cannot reach an unpublished handle; leaked, it would
                # make an exclusive hw: device refuse the next turn's open.
                await stream.kill()
                return
            self._stream = stream
            self._rate = rate
            self._stream_open_t = time.monotonic()
            if self._paused_at is not None:
                # Opened while paused: re-anchor, or pause(False)'s splice adds a span
                # that predates the stream and pushes its clock past now.
                self._paused_at = self._stream_open_t
            self._bytes = 0
            self._gain = self._gain_target  # a fresh stream never inherits a mid-ramp duck
            self._generation += 1  # played_ms() restarts: tell span consumers
        # Blocks paced to _lead_ms() ahead of the wall clock, envelope per block: pacing is
        # what makes a gain change audible promptly. A flush mid-write kills the handle and
        # write no-ops (PlaybackStream contract).
        lead_cap_s = self._lead_ms() / 1000.0
        off = 0
        while off < len(pcm):
            if not self._pause_gate.is_set():
                # Stall between blocks; pause(False) or flush() re-opens the gate, so
                # this cannot wedge.
                await self._pause_gate.wait()
            now = time.monotonic()
            lead_s = self._bytes / (2 * rate) - (now - self._stream_open_t)
            if lead_s < 0.0:
                # Starved: re-anchor to the RESUMED playback, else pacing buffers
                # gap+lead ahead (late duck) and AEC stamps run early.
                self._stream_open_t = now - self._bytes / (2 * rate)
                lead_s = 0.0
            # Re-chosen per block: a tap or duck floor registered mid-piece must tighten
            # the granularity from the very next block. Paced so the lead AFTER the write
            # stays within the cap: the device ring is sized for the cap, not cap + block.
            block_b = self._block_bytes(rate)
            block_s = min(block_b, len(pcm) - off) / (2 * rate)
            if lead_s > lead_cap_s - block_s:
                await asyncio.sleep(lead_s - (lead_cap_s - block_s))
            if epoch != self._epoch or self._stream is not stream:
                return  # flushed while pacing: the rest of this chunk is dead
            block = pcm[off:off + block_b]
            if self._gain != self._gain_target:
                span = 1.0 - self._duck_floor
                ramp = _DUCK_ATTACK_MS if self._gain_target < self._gain else _DUCK_RELEASE_MS
                step = span * (_GAIN_BLOCK_MS / ramp) if ramp > 0 else span
                if self._gain_target < self._gain:
                    self._gain = max(self._gain_target, self._gain - step)
                else:
                    self._gain = min(self._gain_target, self._gain + step)
            if self._gain < 0.9995:
                block = scale_pcm(block, self._gain)
            if self._ref_tap is not None:
                # Playout = stream position on the wall clock, floored at now.
                pos_s = self._bytes / (2 * rate)
                playout = max(time.monotonic(), self._stream_open_t + pos_s)
                self._ref_tap.push_reference(block, rate, playout)
            await stream.write(block)
            self._bytes += len(block)
            off += len(block)
