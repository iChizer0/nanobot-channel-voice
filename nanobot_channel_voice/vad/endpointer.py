"""Utterance endpointing over a per-frame VAD decision stream."""

from __future__ import annotations

from collections import deque

from nanobot_channel_voice.vad.base import Vad


class Endpointer:
    """Feed frames one at a time with :meth:`push`; it returns the complete utterance
    PCM on the frame that ends one, otherwise None. A short ring of pre-trigger frames
    is prepended so the onset is not clipped."""

    def __init__(
        self,
        vad: Vad,
        *,
        frame_ms: int,
        start_frames: int,
        hangover_ms: int,
        min_utterance_ms: int,
        max_utterance_ms: int,
        preroll_ms: int = 0,
        eager_ms: int = 0,
        consult_ms: int = 0,
        consult_cap_bytes: int = 0,
    ):
        self._vad = vad
        self._frame_ms = frame_ms
        self._start_frames = max(1, start_frames)
        self._hangover_frames = max(1, hangover_ms // frame_ms)
        # Eager mark: at a silence run of eager_ms (< hangoverMs), snapshot the
        # utterance so far for SPECULATIVE STT — exactly valid whenever the endpoint
        # then confirms by silence. 0 (or >= hangover) disables.
        self._eager_frames = max(0, eager_ms // frame_ms)
        self._eager: bytes | None = None
        # Consult mark: like eager, but the snapshot goes to an end-of-turn model; a
        # COMPLETE verdict closes via close_now() without waiting out the hangover.
        # 0 (or >= hangover) disables.
        self._consult_frames = max(0, consult_ms // frame_ms)
        # Snapshot only the tail the model scores: the copy is on the frame hot path.
        self._consult_cap = max(0, consult_cap_bytes)
        self._consult: bytes | None = None
        self._consult_active: int | None = None  # _active at snapshot; close_now staleness guard
        # Monotonic across utterances (NOT reset): a verdict about one pause can never
        # authorize closing a different one.
        self._consult_gen = 0
        # Why the last utterance closed ("silence" | "max" | "eou"), read right after
        # push() returns one. A max close invalidates the eager snapshot and has no
        # hangover lag to back-date.
        self.closed_reason = "silence"
        # Did the FINAL silence run reach the eager mark? If close_now() fired first,
        # a still-valid eager task belongs to an EARLIER pause.
        self.eager_covered = True
        # Trailing silence the last close consumed (frame-quantized), for metrics
        # back-dating.
        self.closed_silence_ms = 0
        # Close-time snapshots for the per-utterance summary (probs are None on engines
        # without one): live counters die in reset(), so these are the only copy left.
        self.closed_active_ms = 0
        self.closed_prob_peak: float | None = None
        self.closed_prob_mean: float | None = None
        self._min_frames = min_utterance_ms // frame_ms
        self._max_frames = max(1, max_utterance_ms // frame_ms)
        # Onset-confirmation frames PLUS pre-roll from before the VAD flagged speech, so
        # decision lag doesn't clip the first word. Pre-roll is prepended to the
        # utterance but never counted toward min/max length.
        preroll_frames = max(0, preroll_ms // frame_ms)
        self._pretrigger: deque[bytes] = deque(maxlen=self._start_frames + preroll_frames)
        # Continuation hysteresis: the caller lowers the onset bar while a follow-up is
        # still answerable. None = normal bar. Written from the loop, read per-frame on
        # the hop thread: GIL-atomic int/None, no lock needed.
        self.start_frames_override: int | None = None
        # Audio-dump support: a min-filter reject parks its audio in last_rejected
        # instead of vanishing; the caller consumes the slot.
        self.keep_rejected = False
        self.last_rejected: bytes | None = None
        # Bytes pushed, session-cumulative — NEVER reset, so a wake hit latched in these
        # coordinates stays comparable across utterances. open_pos = where the open
        # utterance's buffer starts; closed_open_pos = the same, frozen at close.
        self._pos = 0
        self.open_pos = 0
        self.closed_open_pos = 0
        self.reset()

    def reset(self) -> None:
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._buf = bytearray()
        self._frames = 0
        self._active = 0  # speech-FLAGGED frames only
        self._prob_peak: float | None = None
        self._prob_sum = 0.0
        self._prob_n = 0
        self._eager = None
        self._eager_active: int | None = None
        self._consult = None
        self._consult_active = None
        self._pretrigger.clear()
        self.last_rejected = None
        self._vad.reset()

    def set_hangover_ms(self, ms: int) -> None:
        """Retarget the hangover (adaptive endpointing). GIL-atomic int swap, safe
        against the hop thread; shrinking it below the eager/consult marks just stops
        those tiers firing."""
        self._hangover_frames = max(1, ms // self._frame_ms)

    @property
    def pos(self) -> int:
        """Bytes pushed so far (the wake-trim coordinate system)."""
        return self._pos

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def silence_run_ms(self) -> int:
        """Trailing silence inside the OPEN utterance (0 whenever a speech frame
        lands); meaningful only while ``in_speech``."""
        return self._silence_run * self._frame_ms

    @property
    def active_ms(self) -> int:
        """Speech-flagged audio accumulated by the open utterance (pauses excluded)."""
        return self._active * self._frame_ms

    @property
    def last_speech_ms(self) -> int:
        """Offset of the most recent speech flag from the confirming run's first frame.
        Frame-domain, so capture bursts cannot skew the pause-probe's leak attribution
        the way a wall clock would."""
        return (self._frames - self._silence_run) * self._frame_ms

    def eager_still_current(self) -> bool:
        """No speech since the eager snapshot, so that decode still describes the whole
        utterance (mirror of the consult tier's ``_consult_active`` pin)."""
        return self._eager_active is not None and self._active == self._eager_active

    @property
    def speech_run(self) -> int:
        """Consecutive speech-flagged frames of the CURRENT onset candidate; 0 once one
        dies. Meaningful only while ``in_speech`` is False — it freezes at onset."""
        return self._speech_run

    def open_pcm(self) -> bytes | None:
        """Snapshot of the OPEN utterance (pre-roll included), None when idle."""
        return bytes(self._buf) if self._in_speech else None

    def take_eager(self) -> bytes | None:
        """Consume the eager-mark snapshot, if any since the last call. The caller owns
        the speculation: replace it when a newer one appears, discard it if the
        utterance is rejected or closes by max-length."""
        pcm, self._eager = self._eager, None
        return pcm

    def take_consult(self) -> tuple[int, bytes] | None:
        """Consume the consult-mark snapshot, if any since the last call: audio for an
        end-of-turn model, with the generation to pass back to :meth:`close_now`."""
        pcm, self._consult = self._consult, None
        if pcm is None:
            return None
        return self._consult_gen, pcm

    def close_now(self, gen: int) -> bytes | None:
        """End the utterance on a COMPLETE verdict about consult snapshot ``gen``,
        without waiting out the hangover. None (and no state change) when the verdict is
        STALE: superseded snapshot, speech resumed, already closed, or still under
        minUtteranceMs (the hangover close treats that blip normally)."""
        if (
            not self._in_speech
            or gen != self._consult_gen
            or self._consult_active is None
            or self._active != self._consult_active
            or self._silence_run < self._consult_frames
            or self._active < self._min_frames
        ):
            return None
        self._snap_close("eou")
        self.eager_covered = (
            not self._eager_frames or self._silence_run >= self._eager_frames
        )
        utterance = bytes(self._buf)
        self.reset()
        return utterance

    def _snap_close(self, reason: str) -> None:
        self.closed_reason = reason
        self.closed_open_pos = self.open_pos
        self.closed_silence_ms = self._silence_run * self._frame_ms
        self.closed_active_ms = self._active * self._frame_ms
        self.closed_prob_peak = self._prob_peak
        self.closed_prob_mean = (
            self._prob_sum / self._prob_n if self._prob_n else None
        )

    def _note_prob(self) -> None:
        p = self._vad.last_prob
        if p is not None:
            self._prob_n += 1
            self._prob_sum += p
            if self._prob_peak is None or p > self._prob_peak:
                self._prob_peak = p

    def push(self, frame: bytes) -> bytes | None:
        if not frame:
            return None
        self._pos += len(frame)
        speech = self._vad.is_speech(frame)

        if not self._in_speech:
            self._pretrigger.append(frame)
            if speech:
                self._speech_run += 1
                self._note_prob()  # the confirm run's flags belong to the utterance
                start = self.start_frames_override or self._start_frames
                if self._speech_run >= start:
                    self._in_speech = True
                    self._buf = bytearray(b"".join(self._pretrigger))  # pre-roll + confirm frames
                    self.open_pos = self._pos - len(self._buf)  # buffer start, stream coords
                    self._frames = self._speech_run
                    self._active = self._speech_run
                    self._silence_run = 0
            else:
                self._speech_run = 0
                # A dead candidate's probabilities describe nothing that survives.
                self._prob_peak, self._prob_sum, self._prob_n = None, 0.0, 0
            return None

        self._buf += frame
        self._frames += 1
        if speech:
            self._active += 1
            self._note_prob()
        self._silence_run = 0 if speech else self._silence_run + 1

        # Once per silence run, and every FINAL run passes it before the hangover, so
        # the newest snapshot is the one the endpoint confirms. The hangover compare is
        # what enforces "eagerMs >= hangoverMs disables eager".
        if (
            self._eager_frames
            and self._silence_run == self._eager_frames
            and self._eager_frames < self._hangover_frames
        ):
            self._eager = bytes(self._buf)
            self._eager_active = self._active  # eager_still_current staleness pin

        # Same once-per-final-run property as the eager mark. _consult_active pins the
        # snapshot to THIS pause: resumed speech bumps _active, so a late verdict can
        # never close over audio the model did not score.
        if (
            self._consult_frames
            and self._silence_run == self._consult_frames
            and self._consult_frames < self._hangover_frames
        ):
            if self._consult_cap and len(self._buf) > self._consult_cap:
                self._consult = bytes(memoryview(self._buf)[-self._consult_cap:])
            else:
                self._consult = bytes(self._buf)
            self._consult_active = self._active
            self._consult_gen += 1

        if self._silence_run >= self._hangover_frames or self._frames >= self._max_frames:
            self._snap_close(
                "silence" if self._silence_run >= self._hangover_frames else "max"
            )
            self.eager_covered = True  # hangover > eager mark; max-length closes discard eager
            utterance = bytes(self._buf)
            # Only speech-flagged frames satisfy minUtteranceMs: pauses never count, so
            # a blip padded with silence cannot pass.
            long_enough = self._active >= self._min_frames
            self.reset()
            if not long_enough and self.keep_rejected:
                self.last_rejected = utterance  # after reset(), which clears the slot
            return utterance if long_enough else None
        return None
