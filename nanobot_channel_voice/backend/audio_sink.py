"""Shared audio output sink.

A single worker plays queued audio in order. Each item carries the interrupt
*epoch* it was produced under (:class:`.base.OutputAudio`) and the worker gates
on that CARRIED epoch, so a barge-in that bumps the epoch drops audio produced
for the cancelled turn even if it was already queued.

Two modes: ``blob`` plays opaque TTS WAVs byte-for-byte via
``PlaybackSink.play_wav``; ``stream`` (cloud + raw-PCM local TTS) writes raw PCM
to a persistent device stream, for gapless output + played-ms accounting.

Synthesis lives in the backends. The duck ENVELOPE is applied here, per written
block, stream mode only; the backend decides only when to engage it.
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

# Duck envelope (stream mode). Sidechain practice: attack fast so the bot yields
# at speech onset, release slow so the level doesn't pump when VAD flickers.
_DUCK_ATTACK_MS = 30.0
_DUCK_RELEASE_MS = 250.0
# Max audio buffered ahead of the wall clock: writes are paced to this lead, so a
# gain change is audible within ~this bound and a barge-in discards at most this.
_STREAM_LEAD_MS = 240.0
# Tighter lead when a duck floor is configured: the lead IS the duck's audible
# latency (already-written audio plays at its baked gain), so trade underrun
# headroom (a starve re-anchors gracefully) for reaction time.
_DUCK_STREAM_LEAD_MS = 120.0
_GAIN_BLOCK_MS = 20  # envelope granularity (one gain step per block)


try:  # numpy ships with the [ondevice] extra; pure-python loops are the fallback
    import numpy as _np
except ImportError:  # pragma: no cover - environment-dependent
    _np = None


def trim_lead_silence(pcm: bytes, rate: int, *, cap_ms: float, threshold: float = 0.01) -> bytes:
    """Cap the leading silence of raw S16_LE PCM at ``cap_ms``.

    VITS-family TTS (MMS, Kokoro-class) emits up to ~250 ms of silence before the
    first phoneme: pure time-to-first-audio on a turn's FIRST chunk, stretched
    inter-sentence pauses later. Trimming only the excess over ``cap_ms`` keeps a
    preroll for soft onsets and preserves deliberate pauses. ``threshold`` is the
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

    The tail twin of ``trim_lead_silence``, for CANNED clips (acks, fillers):
    playback holds the turn in SPEAKING — and the half-duplex mic gated — until
    the padding drains too (measured 580-790 ms of model tail on matcha ack
    phrases). Replies never take this: their chunk boundaries carry deliberate
    pauses. An all-silent scan returns unchanged.
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
        # Stream mode: one device stream per turn. The handle stays in this slot
        # until truly finished (drain included) so flush() can ALWAYS kill it.
        self._stream: PlaybackStream | None = None
        self._stream_open_t = 0.0
        self._bytes = 0
        # Bumped per FRESH stream: played_ms() restarts at 0 there, voiding any
        # offset captured against an earlier one (span accounting keys to this).
        self._generation = 0
        self._rate = 0
        self._queued_ms = 0.0  # duration of queued-but-unwritten items (backlog_ms)
        # Drain (EOF) begun: unwritable, still killable; flush() kills BOTH.
        self._draining: PlaybackStream | None = None
        self._reapers: set[asyncio.Task] = set()  # detached natural-drain finishers
        # Streams the reapers still hold: `_draining` is a single slot a later
        # drain overwrites, so flush()/stop() sweep this set as well.
        self._parked: set[PlaybackStream] = set()
        # Duck floor, set once by the owner; 1.0 = feature off.
        self._duck_floor = 1.0
        self._gain = 1.0
        self._gain_target = 1.0
        # Pause gate (bargeIn.mode="pause"): set = flowing. Clearing it stalls the
        # stream writer between blocks; only the ~lead ms already at the device
        # ring out. Read-side clocks freeze at the pause edge (_clock_now).
        self._pause_gate = asyncio.Event()
        self._pause_gate.set()
        self._paused_at: float | None = None
        self._pause_capable = False  # tightens the pacing lead, like a duck floor
        self._ref_tap = None
        self._log = logger.bind(component="sink")

    def set_reference_tap(self, tap) -> None:
        """Register an AEC front-end (``push_reference``/``reference_dropped``).
        Stream mode only (a tap on a blob-mode sink gets nothing): fed each block
        as written — post-envelope, i.e. what the speaker plays — with its
        playout time."""
        self._ref_tap = tap

    @property
    def stream_mode(self) -> bool:
        return self._mode == "stream"

    async def prewarm(self, rate: int) -> None:
        """Play ~40 ms of silence through the real device path once, at warmup,
        so device open (dmix spin-up, aplay page-in, PCM negotiation) is off the
        first reply's TTFA and a wrong playbackDevice fails loudly at startup.
        Own short-lived handle, worker ``_stream`` slot untouched — the backend
        gates the call on an idle turn. Never raises."""
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

        Stage 1 of duck-then-confirm, hence reversible: the caller confirms
        (flush) or releases once the transcript verdict is in. The stream writer
        advances the envelope block by block. No-op in blob mode / with no floor.
        """
        self._gain_target = self._duck_floor if active else 1.0

    def configure_pause(self, capable: bool) -> None:
        """Declare pause-mode barge-in, once by the owner: tightens the pacing
        lead so a pause silences within ~one lead, like a configured duck floor."""
        self._pause_capable = capable

    def pause(self, active: bool) -> None:
        """Pause (or resume) live stream playback: the reversible stage of
        pause-then-confirm. Nothing is discarded: the writer stalls between
        blocks and continues exactly where it stopped. No-op in blob mode."""
        if self._mode != "stream":
            return
        if active:
            if self._pause_gate.is_set():
                self._pause_gate.clear()
                self._paused_at = time.monotonic()
        else:
            if self._paused_at is not None and self._stream is not None:
                # Splice the paused span out of the stream clock, or every
                # elapsed-based read (played_ms, backlog_ms, starved_ms) counts
                # the silence as playout until the next write re-anchors.
                self._stream_open_t += time.monotonic() - self._paused_at
            self._pause_gate.set()
            self._paused_at = None

    def restore_playback(self) -> None:
        """Teardown convenience: end any candidate attenuation (gain target back
        to full and the pause gate open) so no exit path can strand playback
        quiet or stalled."""
        self.duck(False)
        self.pause(False)

    @property
    def paused(self) -> bool:
        return not self._pause_gate.is_set()

    def _clock_now(self) -> float:
        """The playout clock: wall time, FROZEN at the pause edge while paused:
        else every read-side consumer (echo hold via backlog_ms, played_ms spans)
        thinks the paused audio kept draining. The writer's starvation re-anchor
        rebases the clock on resume, so post-resume reads are consistent."""
        return self._paused_at if self._paused_at is not None else time.monotonic()

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def stream_generation(self) -> int:
        """Identity of the stream ``played_ms()`` currently measures."""
        return self._generation

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def played_ms(self) -> int:
        """Stream mode: ms the user has ACTUALLY heard on the current stream:
        ``min(wall-clock since open, buffered duration)``, floored. Authoritative
        for ``conversation.item.truncate``: the cloud streams far faster than real
        time, so bytes-emitted would wildly over-count. Blob mode returns 0."""
        if self._stream is None or self._rate <= 0:
            return 0
        elapsed = (self._clock_now() - self._stream_open_t) * 1000.0
        buffered = self._bytes / (2 * self._rate) * 1000.0
        return int(min(elapsed, buffered))

    def _lead_ms(self) -> float:
        """Pacing lead for stream writes; tightened when a duck floor or pause
        capability is configured (both need gain/silence changes audible fast)."""
        if self._duck_floor < 1.0 or self._pause_capable:
            return _DUCK_STREAM_LEAD_MS
        return _STREAM_LEAD_MS

    def lead_ms(self) -> float:
        """The write-ahead the device may still play after a pause/kill: the sink owns
        pacing policy, so consumers deriving playout physics (the leak-death probe
        window) ask here instead of copying the constants."""
        return self._lead_ms()

    def backlog_ms(self) -> int:
        """Estimated ms of accepted-but-not-yet-audible audio: queued items plus the
        written-but-unplayed lead. Over-counts the in-flight item by up to one lead
        on purpose: its consumer is the echo hold, where more only guards longer.
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
        """Is playback running (or about to run) dry? Stream: the wall clock has
        consumed everything buffered. Blob: the queue is idle. Meaningful when
        checked as a NON-first chunk is enqueued: synthesis lost the race and the
        user heard a gap."""
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
        if audio.epoch == self._epoch:
            # Credit only what _run's finally debits: a flush between the producer's
            # epoch read and this call zeroes the counter, stranding the credit.
            self._queued_ms += self._item_ms(audio)
        self._queue.put_nowait(audio)

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
        played = self.played_ms() if self._mode == "stream" else 0
        # Kill through OUR handles: valid at any point in a stream's life,
        # including mid-drain and reaper-parked tails. Nothing survives a barge-in.
        await self._kill_streams()
        # A flush ends any candidate interrupt: the next turn plays at full level,
        # unpaused (a writer stalled on the gate wakes into the epoch check).
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
        # A paused drain WAITS for the candidate's verdict instead of forcing the
        # gate open: forcing it would play the turn's final chunk at FULL level
        # into the open mic mid-confirm. A release resumes the tail on its own; a
        # kill's flush opens the gate with a dead epoch. Verdicts always resolve
        # (release/kill/orphan; failure paths restore playback), so no wedge.
        await self._pause_gate.wait()
        await self.wait_idle()
        stream = self._stream
        if self._mode != "stream" or stream is None:
            return
        self._draining = stream  # EOF imminent: new writes must open fresh
        try:
            await stream.drain()
        except asyncio.CancelledError:
            # EOF is already with the device (the alsa handle sends it before its
            # first await), so the tail still plays out; killing here would chop a
            # status line's end on every fast tool round-trip. Park instead: writes
            # open fresh via _draining, barge-in still kills it, the reaper reaps.
            self._spawn_reaper(stream)
            raise
        if self._draining is stream:
            self._draining = None
        if self._stream is stream:  # a concurrent flush may have taken it
            self._stream = None

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
                    # Debit only what enqueue credited: a flush zeroes the counter,
                    # so debiting a dead item would thin the NEXT turn's echo hold.
                    self._queued_ms = max(0.0, self._queued_ms - self._item_ms(item))
                self._queue.task_done()
                if self._queue.empty():
                    self._idle.set()

    async def _stream_write(self, pcm: bytes, rate: int, epoch: int) -> None:
        stream = self._stream
        if stream is not None and (stream is self._draining or stream in self._parked):
            # A drain EOF'd this handle (maybe a cancelled one, still parked in a
            # reaper): no more writes. Open fresh; the tail rings on, flush kills both.
            stream = None
        if stream is not None and stream.dead:
            # The device died UNDER the handle (e.g. an exclusive hw: device
            # rejected a second open after a cancelled drain, then the first
            # exited): without a reopen, every later write this turn is discarded.
            with suppress(Exception):
                await stream.kill()  # reap; already dead, so nothing audible stops
            if self._stream is stream:
                self._stream = None
            stream = None
        if stream is not None and rate != self._rate:  # rate change (unexpected): reopen
            # Keep the handle in _stream during the drain so a flush kills the tail.
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
            stream = await self._sink.open_stream(rate)
            if epoch != self._epoch:
                # flush() landed while opening: its sweep cannot reach an
                # unpublished handle, so kill it here — leaked, it would make an
                # exclusive hw: device refuse the next turn's open.
                await stream.kill()
                return
            self._stream = stream
            self._rate = rate
            self._stream_open_t = time.monotonic()
            self._bytes = 0
            self._generation += 1  # played_ms() restarts: tell span consumers
        # ~20 ms blocks, paced to _lead_ms() ahead of the wall clock, envelope per
        # block: pacing is what makes a gain change audible promptly. A flush
        # mid-write kills the handle and write no-ops (PlaybackStream contract).
        lead_cap_s = self._lead_ms() / 1000.0
        block_b = max(2, (rate * _GAIN_BLOCK_MS // 1000) * 2)
        for off in range(0, len(pcm), block_b):
            if not self._pause_gate.is_set():
                # Paused (pause-then-confirm): stall between blocks. A release
                # (pause(False)) or flush() re-opens the gate, so this cannot
                # wedge; the starvation re-anchor below rebases the clock on resume.
                await self._pause_gate.wait()
            now = time.monotonic()
            lead_s = self._bytes / (2 * rate) - (now - self._stream_open_t)
            if lead_s < 0.0:
                # Starved: re-anchor the stream clock to the RESUMED playback or
                # everything downstream drifts by the gap: pacing would buffer
                # gap+lead ahead (late duck) and AEC stamps would run early.
                self._stream_open_t = now - self._bytes / (2 * rate)
                lead_s = 0.0
            if lead_s > lead_cap_s:
                await asyncio.sleep(lead_s - lead_cap_s)
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
                # Playout = stream position on the wall clock, floored at now (a
                # starved stream plays the block the moment it lands).
                pos_s = self._bytes / (2 * rate)
                playout = max(time.monotonic(), self._stream_open_t + pos_s)
                self._ref_tap.push_reference(block, rate, playout)
            await stream.write(block)
            self._bytes += len(block)
