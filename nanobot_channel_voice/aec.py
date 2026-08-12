"""Software acoustic echo cancellation (``aec="webrtc"``, ``[aec]`` extra).

Wraps WebRTC's AEC3 (``livekit.rtc`` AudioProcessingModule) as a capture
front-end: our own TTS is subtracted from the mic before VAD/STT see it, so
full-duplex barge-in works without hardware AEC.

Reference alignment: the stream-mode sink paces writes against the wall clock
and stamps each reference block with the time its audio leaves the speaker; the
canceller withholds the block until then, leaving AEC3's delay estimator only
the residual device latency (ALSA buffer + DAC, tens of ms, inside its tracked
range) instead of our full pacing lead.

Threading: ``push_reference`` runs on the event loop, ``process`` on the
session's single capture path; the handoff is a deque of immutable tuples
(GIL-atomic append/popleft) and the APM is touched only by ``process``: single
consumer, no locks.
"""

from __future__ import annotations

import time
from collections import deque

from loguru import logger


class EchoCanceller:
    """One APM instance: 10 ms framing, reference playout-time gating."""

    def __init__(self, capture_rate: int, *, device_delay_ms: int = 50):
        from livekit import rtc  # [aec] extra

        self._rtc = rtc
        self._apm = rtc.AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=False,  # otherwise leave the pipeline's signal untouched
            auto_gain_control=False,
            high_pass_filter=False,
        )
        if capture_rate % 100:
            # AEC3/APM frames are EXACTLY rate/100 samples; 22050 (Piper) can't
            # make one and would raise per frame inside the capture path.
            raise ValueError(f"AEC needs a sample rate divisible by 100, got {capture_rate}")
        self._rate = capture_rate
        self._frame_b = (capture_rate // 100) * 2  # 10 ms of S16_LE mono
        self._pending: deque[tuple[float, bytes, int]] = deque()  # playout_at, pcm, rate
        self._ref_carry = b""  # sub-10 ms remainder of the last reference push
        self._carry_rate = 0
        self._warned_rate = 0
        self._fails = 0
        self._bypassed = False
        self._ref_ms = 0.0
        # Post-pacing device latency (ALSA/dmix buffer + DAC; audio.playoutDelayMs):
        # a hint only, AEC3 tracks the true delay itself.
        self._device_delay_ms = device_delay_ms
        self._log = logger.bind(component="aec")
        self._log.info("webrtc AEC3 front-end up (capture {} Hz)", capture_rate)

    def reference_ms(self) -> float:
        """Cumulative reference audio accepted: how much playback AEC3 has had a
        chance to learn from. A convergence proxy: wall time is wrong for it (the
        clock runs during silence, when the filter learns nothing)."""
        return self._ref_ms

    # ---- reference (our own playback), event-loop side --------------------

    def push_reference(self, pcm: bytes, rate: int, playout_at: float) -> None:
        """Queue playback audio stamped with the wall-clock time it becomes
        audible; the sink derives ``playout_at`` from the stream's byte position
        vs. its open time."""
        if self._bypassed:
            return  # process() no longer drains _pending; don't grow it forever
        if rate % 100:
            # Same rate/100 framing constraint as capture; degrade to no-reference
            # instead of raising inside process_reverse_stream per frame.
            if self._warned_rate != rate:
                self._warned_rate = rate
                self._log.warning("AEC reference rate {} not divisible by 100; ignoring reference", rate)
            return
        if self._carry_rate != rate:
            self._ref_carry, self._carry_rate = b"", rate
        # The carry sounds BEFORE this block: back-date the stamp by its duration,
        # else every frame is withheld past its true playout and AEC3 re-converges per push.
        base = playout_at - len(self._ref_carry) / (2.0 * rate)
        self._ref_ms += len(pcm) / (2.0 * rate) * 1000.0
        data = self._ref_carry + pcm
        frame_b = (rate // 100) * 2  # 10 ms at the reference's own rate
        step_s = 0.010
        offset = 0.0
        n_full = len(data) // frame_b
        for i in range(n_full):
            self._pending.append(
                (base + offset, data[i * frame_b:(i + 1) * frame_b], rate)
            )
            offset += step_s
        self._ref_carry = data[n_full * frame_b:]

    def reference_dropped(self) -> None:
        """The stream was killed (barge-in): its unplayed tail never sounds."""
        self._pending.clear()
        self._ref_carry = b""

    # ---- capture, single-consumer side ------------------------------------

    def process(self, pcm: bytes) -> bytes:
        """Run one capture frame through the canceller; returns cleaned PCM.
        Feeds every reference block whose playout time has arrived (silence is
        implicit: with nothing due, AEC3 sees no render activity). Only whole
        10 ms APM frames are processed; ``audio.frameMs`` (10/20/30) at a rate
        divisible by 100 never leaves a remainder."""
        if self._bypassed:
            return pcm
        rtc = self._rtc
        now = time.monotonic()
        try:
            while True:
                try:
                    # peek+pop races reference_dropped()'s barge-in clear(); deque ops
                    # are GIL-atomic, so the lost race surfaces as IndexError: nothing due.
                    if not self._pending or self._pending[0][0] > now:
                        break
                    _, ref, rate = self._pending.popleft()
                except IndexError:
                    break
                self._apm.process_reverse_stream(
                    rtc.AudioFrame(data=ref, sample_rate=rate,
                                   num_channels=1, samples_per_channel=len(ref) // 2)
                )
            out = bytearray()
            n_full = len(pcm) // self._frame_b
            for i in range(n_full):
                chunk = pcm[i * self._frame_b:(i + 1) * self._frame_b]
                frame = rtc.AudioFrame(data=chunk, sample_rate=self._rate,
                                       num_channels=1, samples_per_channel=len(chunk) // 2)
                # APM contract: the delay hint applies to the NEXT process_stream,
                # so it is set per frame, not hoisted out of the loop.
                self._apm.set_stream_delay_ms(self._device_delay_ms)
                self._apm.process_stream(frame)
                out += frame.data.tobytes()
        except Exception as exc:  # noqa: BLE001 - AEC must never deafen capture
            # Upstream of the endpointer: an escaping raise costs the session every
            # frame from here on. Drop the now-meaningless reference timeline.
            self._pending.clear()
            self._ref_carry = b""
            self._fails += 1
            if self._fails >= 3:
                self._bypassed = True
                self._log.error("AEC disabled for this session after 3 failures: {}", exc)
            else:
                self._log.warning("AEC frame failed ({}); passing capture through", exc)
            return pcm
        self._fails = 0
        return bytes(out)


def make_echo_canceller(capture_rate: int, *, device_delay_ms: int = 50) -> EchoCanceller | None:
    """Build the AEC front-end, or ``None`` with a warning; callers degrade
    (local falls back to soft-duplex, cloud open-mic refuses to start)."""
    try:
        return EchoCanceller(capture_rate, device_delay_ms=device_delay_ms)
    except ImportError:
        logger.bind(component="aec").warning(
            "aec='webrtc' needs the [aec] extra (pip install "
            "'nanobot-channel-voice[aec]'); falling back to soft-duplex"
        )
        return None
    except Exception as exc:  # noqa: BLE001 - never let AEC break voice startup
        logger.bind(component="aec").warning(
            "AEC init failed ({}); falling back to soft-duplex", exc
        )
        return None
