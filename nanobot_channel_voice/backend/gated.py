"""GatedUplink: local ears in front of a cloud brain (``realtime.uplink`` = vad | wake).

A decorator :class:`VoiceBackend` wrapping a :class:`ManualTurnBackend`. The shell sees one
backend; the inner adapter sees only what the on-device detectors admit: per utterance,
the endpointer's pre-roll + speech + hangover (``"vad"``), and only inside an attention
window the acoustic wake word opened (``"wake"``). Everything else never leaves the box —
which on every vendor but OpenAI is what the listening bill is made of
(``DESIGN-cloud-uplink-gate.md``).

Frame path (one hop per capture frame, off the loop when a detector is neural):
AEC -> wake detector -> Endpointer (+ Smart Turn early close). The gate does NOT own
``VoiceState``: it mirrors the inner backend's ``StateHint`` for its own decisions and the
adapter keeps emitting them from ``begin_activity`` / ``end_activity`` and its wire events.
"""

from __future__ import annotations

import asyncio
import math
import threading
import time

from loguru import logger

from nanobot_channel_voice.aio import cancel_and_wait, cancel_task
from nanobot_channel_voice.audio.pcm import resample_pcm
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.metrics import VoiceMetrics
from nanobot_channel_voice.vad import Endpointer, resolve_preroll_ms
from nanobot_channel_voice.vad.base import Vad
from nanobot_channel_voice.wake.base import WakeDetector
from nanobot_channel_voice.wake.phrase import WakePhrase

from .audio_sink import AudioSink
from .base import (
    ManualTurnBackend,
    OnEvent,
    OutputTranscript,
    StateHint,
    ToolDef,
    VoiceState,
)

# Own reply said the phrase: veto acoustic hits for this long past the transcript delta,
# plus whatever the sink still had queued then (the words play that much later).
_WAKE_ECHO_S = 3.0
# Transcript deltas are token sized on the OpenAI dialects, so the phrase is looked for
# in the delta plus this much of the reply before it (longer than any spoken phrase).
_WAKE_ECHO_TAIL = 64


def _reply_is_question(text: str) -> bool:
    tail = text.rstrip().rstrip("\"'”’)」』")
    return tail.endswith(("?", "？"))


class GatedUplink:
    def __init__(
        self,
        inner: ManualTurnBackend,
        *,
        config: VoiceConfig,
        sink: AudioSink,
        vad: Vad,
        turn_analyzer=None,
        wake_detector: WakeDetector | None = None,
        aec=None,
        capture_rate: int,
        uplink_rate: int,
        open_mic: bool,
        metrics: VoiceMetrics | None = None,
    ):
        for name in ("begin_activity", "end_activity", "park"):
            if not hasattr(inner, name):
                raise RuntimeError(
                    f"backend {type(inner).__name__} cannot run under a gated uplink "
                    f"(no {name}); set realtime.uplink='server'"
                )
        self._inner = inner
        self.pace_output_audio = getattr(inner, "pace_output_audio", False)
        self._metrics = metrics if metrics is not None else VoiceMetrics()
        self._log = logger.bind(component="voice")
        rt = config.realtime
        self._mode = rt.uplink
        self._idle_park_s = rt.idle_park_s
        self._sink = sink
        self._aec = aec
        self._vad = vad
        self._turn = turn_analyzer
        self._wake = wake_detector if self._mode == "wake" else None
        self._frame_ms = config.audio.frame_ms
        self._capture_rate = capture_rate
        self._uplink_rate = uplink_rate
        self._open_mic = open_mic
        if self._mode == "wake" and self._wake is None:
            raise RuntimeError(
                'realtime.uplink="wake" needs the acoustic wake detector, which did not '
                "build (see the warning above); fix wake.openwakeword or use uplink='vad'"
            )
        hangover = config.vad.hangover_ms
        if self._turn is not None:
            # The turn model consults inside the hangover; a hangover at/under the
            # consult mark would close by silence first, every pause.
            hangover = max(hangover, config.vad.turn.consult_ms + self._frame_ms)
        self._ep = Endpointer(
            vad,
            frame_ms=self._frame_ms,
            start_frames=config.vad.start_frames,
            hangover_ms=hangover,
            min_utterance_ms=config.vad.min_utterance_ms,
            max_utterance_ms=config.vad.max_utterance_ms,
            preroll_ms=resolve_preroll_ms(config.vad, self._frame_ms),
            consult_ms=config.vad.turn.consult_ms if self._turn is not None else 0,
            consult_cap_bytes=getattr(self._turn, "window_bytes", 0),
        )
        # Neural detectors run per frame off the loop, one hop at a time (the shell awaits
        # each push, and on_capture_gap runs between pushes on the same task): the lock is
        # cheap insurance for the detectors' streaming state, never contended today.
        self._threaded = bool(getattr(vad, "heavy", False)) or self._wake is not None
        self._hop_lock = threading.Lock()

        # Attention. "vad": always open. "wake": opened by a hit, extended while a turn
        # is in flight, re-armed for wake.windowS at IDLE ("conversation") or spent by
        # the next commit unless the reply asked a question ("sentence").
        wake = config.wake
        self._wake_mode = wake.mode if self._mode == "wake" else "off"
        self._attention = wake.attention
        self._window_s = wake.window_s
        self._window_until = 0.0 if self._mode == "wake" else math.inf
        self._spent = False
        # A bare phrase heard while the agent works: strict admits the next onset (the
        # command after the phrase) until this, or until a reply speaks or the turn ends.
        self._claim_until = 0.0
        self._phrase = WakePhrase(list(wake.phrases) + list(wake.aliases))
        self._phrase_echo_until = 0.0
        self._reply_tail = ""  # the reply's last _WAKE_ECHO_TAIL chars + its latest delta
        if self._mode == "wake" and wake.ack.enabled:
            self._log.info(
                "voice: wake.ack is spoken by the local TTS, which a cloud session has "
                "none of: a summon gets no audible receipt"
            )

        self._state = VoiceState.IDLE
        self._active = False          # begin_activity sent, end_activity not yet
        self._hit_pos: int | None = None  # phrase END in endpointer stream coordinates
        # Speech-flagged ms of the open utterance when a hit adopted it: what closes with
        # less than minUtteranceMs of speech past it was the phrase alone.
        self._hit_active_ms: int | None = None
        self._min_utterance_ms = config.vad.min_utterance_ms
        self._eou_gen: int | None = None  # COMPLETE verdict awaiting the next frame
        self._consult_task: asyncio.Task | None = None
        self._park_task: asyncio.Task | None = None
        self._ducked = False
        # Suspicion duck (open mic): attenuate the reply while a pre-onset run grows.
        self._duck_frames = config.barge_in.duck_start_frames
        duck_db = config.duck_db if open_mic else 0.0
        sink.configure_duck(10.0 ** (duck_db / 20.0) if duck_db < 0 else 1.0)
        self._on_event: OnEvent | None = None
        self._closing = False

    # ---- VoiceBackend -------------------------------------------------------

    async def start(
        self, *, instructions: str | None, tools: list[ToolDef], on_event: OnEvent
    ) -> None:
        self._on_event = on_event
        await self._inner.start(
            instructions=instructions, tools=tools, on_event=self._on_inner_event
        )
        self._log.info(
            "voice: gated uplink ({}): only {} leaves the device",
            self._mode,
            "speech inside the wake attention window" if self._mode == "wake"
            else "endpointed speech",
        )
        self._schedule_park()

    async def push_audio(self, pcm: bytes) -> None:
        if self._closing or not pcm:
            return
        self._metrics.count("capture_ms", self._frame_ms)
        prev_speech = self._ep.in_speech
        if self._threaded:
            pcm, hit, utterance = await asyncio.to_thread(self._hop, pcm)
        else:
            pcm, hit, utterance = self._hop(pcm)
        now = time.monotonic()
        # A hit may adopt the open utterance wholesale (this frame included): then neither
        # the onset nor the per-frame upload below may send it again.
        adopted = await self._on_wake_hit(now) if hit else False
        if self._ep.in_speech and not prev_speech:
            if not self._active:
                await self._on_onset(now)
        elif self._ep.in_speech and self._active and not adopted:
            await self._upload(pcm)
        if prev_speech and not self._ep.in_speech:
            await self._on_close(utterance)
        self._duck_step()

    async def push_gated_audio(self, pcm: bytes) -> None:
        """Half-duplex wake tap (shell routes frames dropped while the bot speaks): the
        wake word is the only barge-in there. No AEC, no endpointer — the mic-reopen
        flush discards this audio, so the command belongs after the reply stops."""
        if self._closing or not pcm:
            return
        if self._active or self._ep.in_speech:
            # The mic gated under an open utterance (a reply started while the user
            # spoke): frame-counted, it would never close, then commit garbage at the
            # reopen. Same rule as a capture gap.
            self._metrics.count("gate_activity_gated")
            await self._abort_open()
        if self._wake is None:
            return
        if await asyncio.to_thread(self._gated_hop, pcm):
            await self._on_wake_hit(time.monotonic())

    async def barge_in(self, played_ms: int) -> None:
        await self._inner.barge_in(played_ms)

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        await self._inner.submit_tool_result(call_id, output)

    async def on_capture_gap(self) -> None:
        await self._abort_open()

    async def _abort_open(self) -> None:
        """Discontinuous audio: nothing open survives it, on either side."""
        with self._hop_lock:
            self._ep.reset()
            if self._wake is not None:
                self._wake.reset()
        self._eou_gen = None
        cancel_task(self._consult_task)
        if self._active:
            self._active = False
            await self._inner.end_activity(commit=False)

    async def close(self) -> None:
        self._closing = True
        for task in (self._consult_task, self._park_task):
            await cancel_and_wait(task)
        self._consult_task = self._park_task = None
        await self._inner.close()
        for engine in (self._vad, self._wake, self._turn):
            release = getattr(engine, "release", None)
            if release is not None:
                try:
                    release()
                except Exception as exc:  # noqa: BLE001 - teardown must finish
                    self._log.debug("release failed: {}", exc)

    # ---- the hop (sync, off-loop when neural) --------------------------------

    def _hop(self, pcm: bytes) -> tuple[bytes, bool, bytes | None]:
        with self._hop_lock:
            if self._aec is not None:
                pcm = self._aec.process(pcm)
            hit = False
            if self._wake is not None and self._wake.push(pcm):
                hit = True
                # Phrase end in stream coordinates (this frame not pushed yet), so a
                # same-breath command uploads from AFTER the phrase.
                self._hit_pos = (
                    self._ep.pos + len(pcm)
                    - getattr(self._wake, "last_hit_back_bytes", 0)
                )
            utterance = self._ep.push(pcm)
            if utterance is None and self._eou_gen is not None:
                gen, self._eou_gen = self._eou_gen, None
                utterance = self._ep.close_now(gen)
                self._metrics.count("eou_close_early" if utterance else "eou_close_stale")
            return pcm, hit, utterance

    def _gated_hop(self, pcm: bytes) -> bool:
        with self._hop_lock:
            return bool(self._wake.push(pcm))

    # ---- gate decisions (loop) ----------------------------------------------

    def _window_open(self, now: float) -> bool:
        return now < self._window_until

    def _live(self) -> bool:
        return self._state in (VoiceState.THINKING, VoiceState.SPEAKING)

    async def _on_wake_hit(self, now: float) -> bool:
        """True when the hit adopted the open utterance (buffer uploaded, this frame in)."""
        if now < self._phrase_echo_until:
            self._metrics.count("wake_echo_suppressed")
            self._log.info("wake hit suppressed (own reply speaks the phrase)")
            return False
        self._metrics.count("wake_hit")
        score = getattr(self._wake, "last_score", None)
        self._log.info("wake hit{}", f" (score={score:.2f})" if score is not None else "")
        if self._active:
            return False  # the name mid-upload changes nothing (the window stays inf)
        # max: never shortens an engaged turn's window (inf until IDLE).
        self._window_until = max(self._window_until, now + self._window_s)
        self._spent = False
        if self._ep.in_speech:
            # Same breath ("hey nanobot, what's the weather"): adopt the open utterance
            # from the phrase end; the endpointer keeps its clock, we keep uploading.
            buf = self._ep.open_pcm() or b""
            offset = 0
            if self._hit_pos is not None:
                offset = min(len(buf), max(0, self._hit_pos - self._ep.open_pos)) & ~1
            await self._open_activity(buf[offset:])
            if self._active:
                self._hit_active_ms = self._ep.active_ms
            return self._active
        if self._live():
            # Claimed first: the kill can settle THINKING (a tool still owes its answer).
            self._claim_until = now + self._window_s
            if self._state is VoiceState.SPEAKING:
                # A bare summon over the audible reply: kill it and listen (the local
                # _wake_kill). While the agent works the query survives, as locally.
                await self._kill_reply()
        return False

    async def _on_onset(self, now: float) -> None:
        if not self._window_open(now):
            self._metrics.count("gate_dropped_onsets")  # a later hit may still adopt it
            return
        if self._live() and self._wake_mode == "strict" and now >= self._claim_until:
            # Public-room posture: only the phrase interrupts; a hit later in this same
            # utterance adopts it (see _on_wake_hit).
            self._metrics.count("gate_dropped_onsets")
            return
        await self._open_activity(self._ep.open_pcm() or b"")

    async def _open_activity(self, pcm: bytes) -> None:
        self._cancel_park()
        self._hit_active_ms = None  # an adoption sets it after; a lost session left it
        self._claim_until = 0.0  # spent on this utterance
        try:
            await self._inner.begin_activity()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - parked and could not resume
            self._metrics.count("gate_activity_failed")
            self._log.warning("uplink could not open an activity ({}); utterance dropped", exc)
            self._schedule_park()  # no turn will settle IDLE to re-arm it
            return
        self._active = True
        self._window_until = math.inf  # engaged: the turn owns attention until IDLE
        self._release_duck()  # the shell's flush restored gain; keep our bookkeeping true
        self._metrics.count("uplink_utterances")
        if pcm:
            await self._upload(pcm)

    async def _upload(self, pcm: bytes) -> None:
        self._metrics.count("uplink_ms", len(pcm) * 1000 // (2 * self._capture_rate))
        if self._uplink_rate != self._capture_rate:
            pcm = resample_pcm(pcm, self._capture_rate, self._uplink_rate)
        await self._inner.push_audio(pcm)
        consult = self._ep.take_consult()
        if consult is not None and self._turn is not None:
            gen, snapshot = consult
            cancel_task(self._consult_task)
            self._consult_task = asyncio.create_task(self._run_consult(gen, snapshot))

    async def _on_close(self, utterance: bytes | None) -> None:
        cancel_task(self._consult_task)
        self._eou_gen = None
        if not self._active:
            return  # never uploaded (window shut / strict-dropped)
        self._active = False
        hit_active_ms, self._hit_active_ms = self._hit_active_ms, None
        if utterance is None:
            # Min-length reject: the blip already went up; take it back.
            self._metrics.count("gate_blip_aborted")
            await self._inner.end_activity(commit=False)
            return
        if (
            hit_active_ms is not None
            and self._ep.closed_active_ms - hit_active_ms < self._min_utterance_ms
        ):
            # The phrase alone (the detector fires before the utterance closes, so a bare
            # summon adopts too): nothing to answer, the window stays open for the
            # command — the local path publishes nothing either.
            self._metrics.count("gate_bare_summon")
            self._log.debug("bare summon: nothing after the phrase; window open")
            self._claim_until = time.monotonic() + self._window_s  # see _on_wake_hit
            await self._inner.end_activity(commit=False)
            return
        if self._attention == "sentence":
            self._spent = True
        await self._inner.end_activity(commit=True)

    async def _kill_reply(self) -> None:
        """Cancel the live reply and settle with the window open (THINKING while a tool
        still owes its answer): begin (the shell flushes + the adapter cancels) then an
        uncommitted end (nothing to answer)."""
        self._cancel_park()
        try:
            await self._inner.begin_activity()
            await self._inner.end_activity(commit=False)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._log.warning("wake barge-in failed ({})", exc)

    async def _run_consult(self, gen: int, pcm: bytes) -> None:
        try:
            complete = await asyncio.to_thread(self._turn.assess, pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken model must not kill capture
            self._log.warning("eou consult failed: {}", exc)
            return
        self._metrics.count("eou_complete" if complete else "eou_incomplete")
        if complete:
            self._eou_gen = gen  # consumed by the next hop's close_now

    def _duck_step(self) -> None:
        """Open mic: attenuate the reply while a pre-onset speech run grows (a false run
        costs a dip, a real one ends in begin_activity and the flush)."""
        if not self._open_mic or self._ep.in_speech:
            return
        suspect = (
            self._state is VoiceState.SPEAKING
            and self._ep.speech_run >= self._duck_frames
            and self._window_open(time.monotonic())
            and self._wake_mode != "strict"
        )
        if suspect and not self._ducked:
            self._ducked = True
            self._sink.duck(True)
        elif not suspect and self._ducked and self._ep.speech_run == 0:
            self._release_duck()

    def _release_duck(self) -> None:
        if self._ducked:
            self._ducked = False
            self._sink.duck(False)

    # ---- inner events -------------------------------------------------------

    async def _on_inner_event(self, event) -> None:
        if isinstance(event, StateHint):
            prev, self._state = self._state, event.state
            if event.state in (VoiceState.SPEAKING, VoiceState.IDLE):
                self._claim_until = 0.0
            if (
                event.state in (VoiceState.THINKING, VoiceState.SPEAKING)
                and prev in (VoiceState.IDLE, VoiceState.CAPTURING)
            ):
                self._reply_tail = ""  # a reply begins (some protocols skip THINKING)
            elif event.state is VoiceState.IDLE and prev is not VoiceState.IDLE:
                if self._active:
                    # Only a lost session settles IDLE under an open activity (our own
                    # aborts clear the flag first): the utterance dies with it.
                    self._active = False
                    self._metrics.count("gate_activity_lost")
                self._on_idle(time.monotonic())
        elif isinstance(event, OutputTranscript):
            before = self._reply_tail[-_WAKE_ECHO_TAIL:]
            self._reply_tail = before + event.text
            # Stamped by a delta that COMPLETES a mention (one more than the tail already
            # held), not by every delta after it — and again by a second mention.
            if (
                self._wake is not None
                and self._phrase.count(self._reply_tail) > self._phrase.count(before)
            ):
                self._phrase_echo_until = (
                    time.monotonic() + _WAKE_ECHO_S + self._sink.backlog_ms() / 1000.0
                )
        if self._on_event is not None and not self._closing:
            await self._on_event(event)

    def _on_idle(self, now: float) -> None:
        self._release_duck()
        if self._mode == "wake":
            spent = self._spent and self._attention == "sentence"
            self._spent = False
            if spent and not _reply_is_question(self._reply_tail):
                self._window_until = now  # the summoned sentence was answered
            else:
                self._window_until = now + self._window_s
        self._schedule_park()

    def _schedule_park(self) -> None:
        self._cancel_park()
        if self._idle_park_s <= 0 or self._closing:
            return
        self._park_task = asyncio.create_task(self._park_after_idle())

    def _cancel_park(self) -> None:
        cancel_task(self._park_task)

    async def _park_after_idle(self) -> None:
        await asyncio.sleep(self._idle_park_s)
        if self._state is VoiceState.IDLE and not self._active and not self._closing:
            await self._inner.park()


__all__ = ["GatedUplink"]
