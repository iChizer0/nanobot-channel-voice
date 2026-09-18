"""RealtimeTransport: the plumbing every WebSocket speech-to-speech adapter shares.

One rx task owns connect + receive + bounded reconnect (and park/resume); a sender task
drains a bounded drop-oldest audio queue so a slow socket never stalls capture; control
frames go through one bounded ``_send``; a PROGRESS deadman guards the turn; the drain
task owns the SPEAKING -> IDLE settle. Subclasses own the wire: the hello frame, the
audio frame shape, event dispatch, per-session turn state and the deadman's recovery.

Under the gated uplink (``ManualTurnBackend``) the speech-onset/offset transitions the
server's VAD events used to drive come from ``begin_activity`` / ``end_activity``; the
state half lives here (``_on_speech_started`` / ``_on_speech_stopped``), the wire half in
the subclass hooks.
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import suppress

from loguru import logger

from nanobot_channel_voice.aio import (
    Throttle,
    cancel_and_wait,
    cancel_task,
    put_drop_oldest,
    wait_for_stall,
)
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.metrics import VoiceMetrics

from .audio_sink import AudioSink
from .base import Error, OnEvent, ToolDef, UserSpeechStarted, VoiceState
from .common import TurnEventMixin

_SEND_Q_MAX = 64  # ~1.3s of 20ms frames; drop-oldest past this
# Control frames go out from the rx loop (_handle_event), where websockets' unbounded
# drain() past its write high-water mark would stall barge-in and every later server
# event. The budget bounds OUR wait only: the frame is committed either way (see _send).
_SEND_TIMEOUT_S = 2.0
_BACKOFF = (0.5, 1.0, 2.0)
# Un-park budget: connect + hello round trip. Past it the gate drops the utterance
# rather than blocking capture behind a dead network.
_RESUME_TIMEOUT_S = 5.0
# Healthy session: resets the backoff budget, so an endpoint that recycles long sessions
# (Qwen turn caps, Gemini's ~10-min socket) never reaches "reconnect exhausted".
_HEALTHY_SESSION_S = 30.0


def _load_connect():
    """websockets >=13 is pinned: the ``websockets.asyncio`` client shipped in 13.0."""
    try:
        from websockets.asyncio.client import connect
    except ImportError as e:
        raise RuntimeError(
            "the realtime backends need the [realtime] extra: pip install "
            "'nanobot-channel-voice[realtime]'"
        ) from e
    return connect


class _SetupError(RuntimeError):
    """Our own connect could not be prepared (a key, a URL, a tool schema): config, not
    network — never walked up the reconnect ladder."""


def _rejection(exc: BaseException) -> tuple[str, bool] | None:
    """``(why, is_auth)`` when the provider REFUSED us rather than dropped us, else None:
    an HTTP 4xx on the handshake (``.response.status_code`` = modern websockets,
    ``.status_code`` = legacy; 429 is load, not refusal) or a 1007/1008 close after it
    (Gemini: "API key not valid", a rejected setup). Retried, a refusal only burns the ladder."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if status in (401, 403):
        return f"HTTP {status}", True
    if isinstance(status, int) and 400 <= status < 500 and status != 429:
        return f"HTTP {status}", False
    close = getattr(exc, "rcvd", None)
    code = getattr(close, "code", None)
    if code in (1007, 1008):
        reason = str(getattr(close, "reason", "") or exc).strip()
        return f"close {code}: {reason}", "api key" in reason.lower()
    return None


class RealtimeTransport(TurnEventMixin):
    # The rx loop emits audio AND carries barge-in/tool/done events: parked on the sink
    # backlog it would defer open-mic barge-in by the whole buffered reply (and starve
    # the WS keepalive). The queue stays reply-bounded, epoch-dropped.
    pace_output_audio = False

    def __init__(
        self,
        config: VoiceConfig,
        *,
        sink: AudioSink,
        metrics: VoiceMetrics | None = None,
        aec=None,
    ):
        # Shared with the shell/channel: one call's segments land in one collector.
        self._metrics = metrics if metrics is not None else VoiceMetrics()
        self._rt = config.realtime
        self._sink = sink
        # Software AEC3 front-end (barge_in="aec" w/o hardware AEC); sink feeds the ref.
        self._aec = aec
        # Gated uplink: the gate owns turn boundaries (begin/end_activity), no server VAD
        # runs, and every barge-in is a client-side cancel.
        self._manual = config.realtime.uplink != "server"
        self._parked = False
        self._on_event: OnEvent | None = None
        self._instructions: str | None = None
        self._tools: list[ToolDef] = []

        self._ws = None
        self._ready = asyncio.Event()
        self._closing = False
        self._rx_task: asyncio.Task | None = None
        # Parked rx tasks still closing their socket: swept at close().
        self._orphan_rx: set[asyncio.Task] = set()
        self._sender_task: asyncio.Task | None = None
        self._drain_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._send_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_SEND_Q_MAX)
        self._warn_throttle = Throttle()
        self._connected_at: float | None = None  # this socket's open, until its hello lands
        self._ever_ready = False
        self._auth_fails = 0
        self._progress_t = 0.0  # feeds the turn deadman
        self._turn = VoiceState.IDLE
        # Barge-in latency clock (monotonic ms): set at onset, eaten by the next
        # barge-in. Session-scoped (reset by the subclass), or a reconnect inherits it.
        self._speech_started_at: float | None = None
        # Set at onset, read before the CAPTURING transition overwrites the turn state.
        self._onset_interrupting = False
        # The only span in which uplink frames may feed the deadman (see the sender loop).
        self._user_speaking = False
        self._log = logger.bind(component="voice")

    # ---- subclass hooks -----------------------------------------------------

    def _connect_args(self) -> tuple[str, dict[str, str]]:
        """``(url, headers)`` for the socket."""
        raise NotImplementedError

    def _hello_payload(self) -> dict:
        """The first frame after connect (session.update / setup)."""
        raise NotImplementedError

    def _audio_frame(self, pcm: bytes) -> dict:
        """The wire frame carrying one captured chunk."""
        raise NotImplementedError

    async def _handle_event(self, evt: dict) -> None:
        raise NotImplementedError

    def _reset_turn_state(self, *, reason: str = "init") -> None:
        """Forget everything about the session's turns (a new socket rejects old ids)."""

    def _on_drained(self) -> None:
        """Playback fully drained: per-item bookkeeping the subclass keeps dies here."""

    async def _watchdog_recover(self) -> str | None:
        """The deadman fired: stop whatever the server may still be generating. Returns
        the fault to report, or None when silence was a legitimate outcome."""
        return "realtime turn timed out"

    async def _activity_begin_wire(self) -> None:
        """Manual turns: the vendor's start marker, if any."""

    async def _activity_end_wire(self, *, commit: bool) -> None:
        """Manual turns: hand the audio to the model (``commit``) or discard it."""

    # ---- VoiceBackend -------------------------------------------------------

    @property
    def metrics(self) -> VoiceMetrics:
        return self._metrics

    async def start(
        self, *, instructions: str | None, tools: list[ToolDef], on_event: OnEvent
    ) -> None:
        self._on_event = on_event
        self._instructions = instructions
        self._tools = tools or []
        self._closing = False
        self._rx_task = asyncio.create_task(self._rx_loop())
        self._sender_task = asyncio.create_task(self._sender_loop())

    async def push_audio(self, pcm: bytes) -> None:
        # AEC before the ready-gate: the filter needs a continuous capture timeline (and
        # this drains due reference blocks). Loop-side: ~0.05 ms per 10 ms frame pair.
        if self._aec is not None:
            pcm = self._aec.process(pcm)
        # Session-ready barrier: drop until the format/VAD config is applied. Under the
        # server uplink nothing else notices a hello that is never acknowledged.
        if self._closing or not self._ready.is_set():
            if (
                self._connected_at is not None
                and time.monotonic() - self._connected_at > _RESUME_TIMEOUT_S
                and self._warn_throttle.ready()
            ):
                self._log.warning(
                    "realtime session not ready {:.0f}s after connect (no session.updated/"
                    "setupComplete); mic frames are being dropped", _RESUME_TIMEOUT_S,
                )
            return
        if put_drop_oldest(self._send_q, pcm) is not None:
            self._warn_backpressure()  # dropped a frame to stay near real time

    async def on_capture_gap(self) -> None:
        """No-op: the provider's server VAD sees the uplink go quiet on its own (under the
        gated uplink the gate ends the activity itself)."""

    async def close(self) -> None:
        self._closing = True
        self._ready.clear()
        # cancel_and_wait re-raises the CALLER's cancellation; the sweep stays complete
        # because VoiceShell.stop shields _teardown, so nothing cancels close() from above.
        for task in (
            self._drain_task, self._watchdog_task, self._sender_task, self._rx_task,
            *self._orphan_rx,
        ):
            await cancel_and_wait(task)
        self._drain_task = self._watchdog_task = self._sender_task = self._rx_task = None
        ws, self._ws = self._ws, None
        if ws is not None:
            with suppress(Exception):
                await ws.close()

    # ---- ManualTurnBackend (gated uplink) -----------------------------------

    async def begin_activity(self) -> None:
        if self._closing:
            return
        if self._parked:
            await self._resume()
        else:
            # A connect in flight (first hello, or the ladder after a drop): the same
            # budget as an un-park, else the frames would go into the void and the
            # deadman would report a fault for a turn the server never saw.
            await self._await_ready("reconnect")
        await self._on_speech_started()
        await self._activity_begin_wire()

    async def end_activity(self, *, commit: bool = True) -> None:
        if self._closing:
            return
        # The frames ride the sender task; a control frame would overtake them.
        await self._flush_uplink()
        if commit:
            self._on_speech_stopped()
            await self._activity_end_wire(commit=True)
            return
        self._user_speaking = False
        await self._activity_end_wire(commit=False)
        if self._turn is VoiceState.CAPTURING:
            # Nothing was asked: no response will come, so the deadman armed at the onset
            # has nothing to guard. Any other state is a reply in flight, whose deadman
            # (and state) stays.
            self._cancel_watchdog()
            await self._set_turn(VoiceState.IDLE)

    async def park(self) -> None:
        if self._parked or self._closing:
            return
        self._parked = True
        task, self._rx_task = self._rx_task, None
        if task is not None:
            cancel_task(task)  # before the bookkeeping: no event may land after the reset
            self._orphan_rx.add(task)
            task.add_done_callback(self._orphan_rx.discard)
        await self._on_session_lost()
        self._metrics.count("park")
        self._log.info("realtime session parked (idle); reconnects on the next utterance")
        # The close handshake (websockets' close_timeout) holds nothing up: an onset
        # cancels this wait and resumes on a fresh socket while the old one finishes.
        if task is not None:
            await asyncio.wait({task})

    async def _resume(self) -> None:
        if not self._parked:
            return  # begin_activity's guard; a resume of a live loop would double it
        t0 = time.monotonic()
        self._parked = False
        self._rx_task = asyncio.create_task(self._rx_loop())
        await self._await_ready("resume")
        self._metrics.observe("resume_latency_ms", (time.monotonic() - t0) * 1000.0)

    async def _await_ready(self, what: str) -> None:
        """Bounded wait for the session hello; past the budget, or once the rx loop gave
        up (ladder exhausted), the gate drops the utterance rather than blocking capture
        behind a dead network."""
        if self._ready.is_set():
            return
        ready = asyncio.ensure_future(self._ready.wait())
        rx = self._rx_task
        try:
            await asyncio.wait(
                {ready, rx} if rx is not None else {ready}, timeout=_RESUME_TIMEOUT_S,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            ready.cancel()
        if self._ready.is_set():
            return
        why = (
            "reconnect gave up" if rx is not None and rx.done()
            else f"no session within {_RESUME_TIMEOUT_S:.0f}s"
        )
        raise RuntimeError(f"realtime session did not {what}: {why}")

    async def _flush_uplink(self) -> None:
        """Wait for every queued mic frame to reach the socket, bounded: a dead sender
        (closing) or a congested uplink must not wedge the turn."""
        with suppress(TimeoutError):
            await asyncio.wait_for(self._send_q.join(), _SEND_TIMEOUT_S)

    # ---- speech edges (server VAD events or the gate) ------------------------

    async def _on_speech_started(self) -> None:
        self._cancel_drain()
        self._arm_watchdog()  # recover if the server never turns this into a response
        self._user_speaking = True
        self._speech_started_at = time.monotonic() * 1000.0
        # Read BEFORE the CAPTURING transition overwrites it.
        self._onset_interrupting = self._turn in (
            VoiceState.THINKING, VoiceState.SPEAKING,
        )
        await self._set_turn(VoiceState.CAPTURING)
        await self._emit(UserSpeechStarted())

    def _on_speech_stopped(self) -> None:
        # MEASUREMENT ONLY: the anchor turn latency is measured from (end of user speech).
        self._metrics.turn_anchor()
        self._user_speaking = False  # uplink frames stop feeding the deadman here
        self._progress_t = time.monotonic()  # end of speech IS turn progress

    def _record_barge_in(self, mechanism: str) -> None:
        """Every onset routes here; only an interrupting one is a sample (the stamp
        clears regardless)."""
        stamp, self._speech_started_at = self._speech_started_at, None
        if stamp is None or not self._onset_interrupting:
            return
        self._metrics.observe(
            f"barge_in_ms.{mechanism}", time.monotonic() * 1000.0 - stamp,
        )

    # ---- connection / io ----------------------------------------------------

    async def _rx_loop(self) -> None:
        attempt = 0
        me = asyncio.current_task()
        while not self._closing and not self._parked:
            started = time.monotonic()
            try:
                await self._connect_and_run()
                # A CLEAN server close lands here, not in `except` (Qwen's per-session turn
                # cap, Gemini's goAway); still a disconnect: same teardown + backoff.
            except asyncio.CancelledError:
                raise
            except _SetupError as exc:
                await self._emit(Error(message=str(exc), fatal=True))
                break
            except Exception as exc:  # noqa: BLE001 - reconnect on any transport failure
                if self._closing or self._rx_task is not me:
                    break  # closing, or a resumed session overtook this loop (orphan)
                # A refusal is not a blip: fatal at once if the session NEVER worked (a
                # config error), else one ladder retry (proxy blip, key rotation).
                rejected = _rejection(exc)
                if rejected is not None:
                    why, is_auth = rejected
                    self._auth_fails += 1
                    if not self._ever_ready or self._auth_fails >= 2:
                        hint = (
                            "check the realtime.apiKey for this provider" if is_auth
                            else "check realtime.model/baseUrl and the tool schemas"
                        )
                        await self._emit(Error(
                            message=f"realtime {'auth ' if is_auth else ''}rejected "
                                    f"({why}): {hint}",
                            fatal=True,
                        ))
                        break
                await self._emit(Error(message=f"realtime disconnected: {exc}", fatal=False))
            if self._closing or self._parked or self._rx_task is not me:
                break
            await self._on_session_lost()
            # Only back-to-back FAST failures walk the ladder to the fatal rung.
            if time.monotonic() - started >= _HEALTHY_SESSION_S:
                attempt = 0
            if attempt >= len(_BACKOFF):
                if self._manual:
                    # Gated uplink: no socket while idle is its normal state, so give the
                    # network up and let the next utterance run the ladder again — a
                    # fatal here would tear the channel down for a WiFi blip at summon.
                    self._parked = True
                    self._metrics.count("reconnect_parked")
                    await self._emit(Error(
                        message="realtime reconnect exhausted; parked until the next "
                                "utterance",
                        fatal=False,
                    ))
                    break
                await self._emit(Error(message="realtime reconnect exhausted", fatal=True))
                break
            await asyncio.sleep(_BACKOFF[attempt])
            attempt += 1

    async def _on_session_lost(self) -> None:
        """Teardown shared by every way a session can end, clean or not."""
        self._ready.clear()
        self._connected_at = None  # the ladder's own log covers the gap to the next hello
        # A surviving watchdog would fire, with real side effects, into the next session.
        self._cancel_watchdog()
        self._cancel_drain()
        self._reset_turn_state(reason="session_lost")
        # Session-scoped: a stamp or latch carried across a reconnect would measure the
        # new session's first onset against a turn that no longer exists.
        self._speech_started_at = None
        self._onset_interrupting = False
        self._user_speaking = False
        # A surviving anchor would measure new audio against a turn that no longer exists.
        self._metrics.turn_end()
        # The half-duplex mic gate keys on SPEAKING: dropping while SPEAKING wedges forever
        # (mic gated -> no audio out -> no speech_started to move off SPEAKING).
        await self._set_turn(VoiceState.IDLE)

    async def _connect_and_run(self) -> None:
        connect = _load_connect()
        try:
            url, headers = self._connect_args()
            hello = self._hello_payload()
        except Exception as exc:
            raise _SetupError(f"realtime connect could not be prepared: {exc}") from exc
        async with connect(url, additional_headers=headers) as ws:
            try:
                self._ws = ws
                self._connected_at = time.monotonic()
                await self._send(hello)
                async for raw in ws:
                    if self._closing:
                        break
                    try:
                        evt = json.loads(raw)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(evt, dict):
                        continue  # valid JSON scalar/array: not a protocol event
                    try:
                        await self._handle_event(evt)
                    except asyncio.CancelledError:
                        raise
                    except Exception:  # noqa: BLE001
                        # A handler bug must not kill a HEALTHY connection:
                        # a reconnect loses the server-side conversation state.
                        self._log.exception("event handler failed for {}", self._event_name(evt))
            finally:
                # _ws must not outlive the socket: submit_tool_result must see None and
                # drop the frame, not raise and skip its bookkeeping. Own socket only: a
                # park cancelled mid-handshake lets a resumed session overtake this one.
                if self._ws is ws:
                    self._ws = None

    @staticmethod
    def _event_name(evt: dict) -> str:
        return str(evt.get("type") or next(iter(evt), "?"))

    async def _sender_loop(self) -> None:
        while not self._closing:
            try:
                pcm = await self._send_q.get()
            except asyncio.CancelledError:
                raise
            try:
                ws = self._ws
                if ws is None or self._closing or not self._ready.is_set():
                    continue
                # Unbounded on purpose: a congested uplink must block HERE so _send_q's
                # drop-oldest bounds mic staleness instead of the transport buffer growing.
                await ws.send(json.dumps(self._audio_frame(pcm)))
                if self._user_speaking and self._turn is VoiceState.CAPTURING:
                    # A monologue longer than turn_timeout_s emits no server events, so the
                    # watchdog armed at onset would fire mid-sentence. AUDIBLY speaking
                    # only: post-offset silence (still CAPTURING) or idle frames in
                    # THINKING would mask a server that never answers.
                    self._progress_t = time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - transient; frames may drop
                self._log.debug("append failed: {}", exc)
            finally:
                self._send_q.task_done()

    async def _send(self, obj: dict) -> None:
        """One control frame, waited on for at most _SEND_TIMEOUT_S. websockets hands the
        whole frame to the transport synchronously before its only await (drain), so a
        timeout means "committed, uplink congested", never "lost": callers' bookkeeping
        runs. No lock: frames are written whole, so concurrent sends cannot interleave."""
        ws = self._ws
        if ws is None:
            return
        try:
            await asyncio.wait_for(ws.send(json.dumps(obj)), _SEND_TIMEOUT_S)
        except TimeoutError:
            self._warn_backpressure()

    def _warn_backpressure(self) -> None:
        if not self._warn_throttle.ready():
            return
        self._log.warning(
            "realtime uplink is congested (dropping mic frames); check network/bandwidth "
            "to the provider."
        )

    # ---- drain + watchdog ---------------------------------------------------

    def _start_drain(self) -> None:
        self._cancel_drain()
        self._drain_task = asyncio.create_task(self._drain())

    def _cancel_drain(self) -> None:
        if self._drain_task is not None and not self._drain_task.done():
            self._drain_task.cancel()

    async def _drain(self) -> None:
        try:
            await self._sink.drain_stream()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # A device failure must not skip the IDLE transition: the watchdog died at
            # turn end and gated-mic SPEAKING mutes the mic — nothing else recovers.
            self._log.warning("drain failed ({}); forcing IDLE", exc)
        self._on_drained()
        if self._turn is VoiceState.CAPTURING:
            # A completion landing after the next onset: the onset owns the state, and
            # the deadman that completion cancelled guards the answer it is still owed.
            self._arm_watchdog()
            return
        with suppress(Exception):  # a raising dispatcher must not strand SPEAKING
            await self._set_turn(VoiceState.IDLE)

    def _arm_watchdog(self) -> None:
        self._cancel_watchdog()
        self._progress_t = time.monotonic()
        self._watchdog_task = asyncio.create_task(self._watchdog())

    def _cancel_watchdog(self) -> None:
        if self._watchdog_task is not None and not self._watchdog_task.done():
            self._watchdog_task.cancel()

    async def _watchdog(self) -> None:
        try:
            # DEADMAN, not a whole-turn cap: deltas push _progress_t forward, so a long
            # streaming reply never trips it; only turn_timeout_s of true silence does.
            await wait_for_stall(lambda: self._progress_t, self._rt.turn_timeout_s)
            fault = await self._watchdog_recover()
            self._metrics.turn_end()
            if fault is not None:
                self._log.warning("realtime turn watchdog fired (no progress); recovering")
                await self._emit(Error(message=fault, fatal=False))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            # The deadman is the last recovery: dying here strands a gated mic in
            # SPEAKING, and the task exception would surface only at GC.
            self._log.warning("realtime turn watchdog failed ({}); forcing IDLE", exc)
        with suppress(Exception):  # the same dispatcher that just raised
            await self._set_turn(VoiceState.IDLE)


__all__ = ["RealtimeTransport", "_load_connect"]
