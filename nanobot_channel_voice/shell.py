"""VoiceShell: the shared audio shell around a swappable :class:`VoiceBackend`.

Owns the *edge*: the capture pump (mic frames -> ``backend.push_audio``), the
shared :class:`AudioSink` fed from ``OutputAudio``, the coarse ``VoiceState``
mirror (driven only by ``StateHint``; the backend is the single source of truth
for turn state), mic gating and lifecycle. Turn logic, barge-in, the echo
filter and drain live in the backend. For cloud the shell also routes
``ToolCall`` -> nanobot and performs the generic barge-in (flush on
``UserSpeechStarted``, played-ms to ``backend.barge_in``); local emits neither
event, so those paths stay dormant.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress
from typing import Any

from loguru import logger

from nanobot_channel_voice.aio import cancel_and_wait
from nanobot_channel_voice.audio.base import CaptureSource
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.metrics import VoiceMetrics
from nanobot_channel_voice.telemetry import VoiceTracer

from .backend.audio_sink import AudioSink
from .backend.base import (
    Error,
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    StateHint,
    ToolCall,
    ToolDef,
    ToolStarted,
    TurnDone,
    UserSpeechStarted,
    VoiceBackend,
    VoiceState,
)
from .backend.common import loggable_text

# (name, arguments_json) -> tool result (str or JSON-encodable). Cloud tool seam.
ExecToolFn = Callable[[str, str], Awaitable[Any]]
# Notified after a cloud barge-in (sink flushed, backend told), so the channel can
# abandon an in-flight ask_nanobot delegation the user just talked over.
BargeInFn = Callable[[], Awaitable[None]]
# Notified AFTER the shell tore itself down on a fatal backend error; without it the
# channel's start() blocks forever on a stop event nobody sets, presenting a healthy
# channel around a dead shell.
FatalFn = Callable[[], Awaitable[None]]

__all__ = ["VoiceShell", "VoiceState"]  # VoiceState re-exported beside its mirror


class VoiceShell:
    def __init__(
        self,
        config: VoiceConfig,
        *,
        capture: CaptureSource,
        sink: AudioSink,
        backend: VoiceBackend,
        open_mic: bool,
        exec_tool: ExecToolFn | None = None,
        on_barge_in: BargeInFn | None = None,
        on_fatal: FatalFn | None = None,
        tool_mode: str = "direct",
        metrics: VoiceMetrics | None = None,
        tracer: VoiceTracer | None = None,
    ):
        # Label only; execution latency is bucketed by it (direct vs supervisor).
        self._tool_mode = tool_mode
        self._metrics = metrics if metrics is not None else VoiceMetrics()
        # Resolved once: a disabled/absent tracer yields no-op spans, so no branching.
        self._tracer = tracer if tracer is not None else VoiceTracer(config.telemetry)
        self._capture = capture
        self._sink = sink
        self._backend = backend
        # Mic policy comes from the channel: local = config.open_mic (full|soft|webrtc);
        # cloud = realtime.bargeIn ("aec" => open, "gated" => gated while SPEAKING).
        self._open_mic = open_mic
        self._exec_tool = exec_tool
        self._on_barge_in = on_barge_in
        self._on_fatal = on_fatal
        self._log_transcripts = config.log_transcripts
        # "half" == gated while SPEAKING; the rest name the mechanism keeping it open.
        self._duplex_name = (
            (
                "full" if config.full_duplex
                else "webrtc" if config.aec == "webrtc"
                else "soft" if config.soft_duplex
                else "aec"  # cloud: hardware AEC asserted via realtime.aecAvailable
            )
            if open_mic
            else "half"
        )
        self._state = VoiceState.IDLE
        self._running = False
        self._stopped = False  # latched at teardown start: gates late event dispatch
        self._stop_task: asyncio.Task | None = None
        self._capture_task: asyncio.Task | None = None
        self._fatal_task: asyncio.Task | None = None
        # Referenced so they aren't GC'd mid-flight and can be cancelled on stop.
        self._tool_tasks: set[asyncio.Task] = set()
        self._log = logger.bind(component="voice")

    # ---- lifecycle -------------------------------------------------------

    async def start(
        self, *, instructions: str | None = None, tools: list[ToolDef] | None = None
    ) -> None:
        self._running = True
        await self._capture.start()
        await self._sink.start()
        await self._backend.start(
            instructions=instructions, tools=tools or [], on_event=self._on_event
        )
        self._capture_task = asyncio.create_task(self._capture_loop())
        self._log.info(
            "voice session up (duplex={}, backend={})",
            self._duplex_name,
            type(self._backend).__name__,
        )

    async def stop(self) -> None:
        """Idempotent AND joinable: every caller awaits the same teardown task,
        shielded so a caller cancelled mid-stop (the manager can cancel
        ``channel.stop()``) aborts its own wait, not the teardown."""
        self._running = False
        if self._stop_task is None:
            self._stopped = True
            self._stop_task = asyncio.create_task(self._teardown())
        await asyncio.shield(self._stop_task)

    async def _teardown(self) -> None:
        await cancel_and_wait(self._capture_task)
        self._capture_task = None
        for task in list(self._tool_tasks):  # results are moot once tearing down
            await cancel_and_wait(task)
        self._tool_tasks.clear()
        # Tasks down before devices, each step guarded: _stopped is latched above, so
        # a raise would skip the rest forever (later stop()s return at the latch),
        # leaving devices running behind a stopped channel.
        for step in (self._backend.close, self._sink.stop, self._capture.stop):
            try:
                await step()
            except Exception as exc:  # noqa: BLE001
                self._log.warning("shell teardown step {} failed: {}",
                                  getattr(step, "__qualname__", step), exc)
        # The session's only readout, emitted where the sample set is final.
        if self._metrics.has_data:
            self._log.info("voice session metrics: {}", self._metrics.summary_line())

    @property
    def state(self) -> VoiceState:
        return self._state

    # ---- capture pump ----------------------------------------------------

    async def _capture_loop(self) -> None:
        eof_streak = 0
        restarts = 0
        was_gated = False
        while self._running:
            try:
                frame = await self._capture.read_frame()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log.warning("capture read error: {}", exc)
                frame = b""
            if not frame:
                eof_streak += 1
                if eof_streak >= 5:
                    # Device gone (USB mic unplugged, arecord died). Bounded restart;
                    # a mic that comes back re-arms the budget on the first good frame.
                    if restarts >= 3:
                        # Permanent deafness is fatal, not degraded: otherwise the
                        # session plays on while presenting healthy.
                        self._log.error("capture ended permanently; tearing the session down")
                        self._spawn_fatal_stop()
                        return
                    restarts += 1
                    self._log.warning("capture ended; restarting capture ({}/3)", restarts)
                    # An utterance open at mic death must not bridge the outage
                    # (the endpointer's clock is frame-counted: no frames, no
                    # silence run). A failing hook must not stop the restart, but
                    # it must be SEEN: silence here is utterances merging quietly.
                    try:
                        await self._backend.on_capture_gap()
                    except asyncio.CancelledError:
                        raise
                    except Exception as exc:  # noqa: BLE001
                        self._log.warning("capture-gap hook failed: {}", exc)
                    with suppress(Exception):
                        await self._capture.stop()
                    await asyncio.sleep(2.0)
                    with suppress(Exception):
                        await self._capture.start()
                    eof_streak = 0
                    continue
                await asyncio.sleep(0.2)
                continue
            eof_streak = 0
            restarts = 0

            if self._mic_gated():
                was_gated = True
                continue
            if was_gated:
                # Gate-reopen edge: the gate is applied at READ time, but under lag
                # frames captured while it was closed (the bot audible) can still be
                # buffered: released now, they would reach the VAD as fresh speech
                # and endpoint as unintelligible echo blobs (stt_empty every turn).
                # Realign with the wall clock: drop the in-hand frame (it predates or
                # spans the edge) and the source's backlog with it.
                was_gated = False
                try:
                    if await self._capture.flush():
                        self._metrics.count("capture_gate_flush")
                    # The pipe is empty now; a local backend's capture-debt accounting
                    # must not keep describing the backlog this just discarded.
                    note = getattr(self._backend, "note_capture_flush", None)
                    if note is not None:
                        note()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - flush is hardening, not load-bearing
                    self._log.debug("capture flush failed: {}", exc)
                continue
            try:
                await self._backend.push_audio(frame)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - transient cloud WS send failure
                self._log.warning("push_audio error: {}", exc)

    def _mic_gated(self) -> bool:
        # Half-duplex mutes while the bot speaks so it never hears itself; full/soft
        # duplex and the cloud AEC path keep the mic open for barge-in.
        if self._open_mic:
            return False
        return self._state is VoiceState.SPEAKING

    # ---- backend event dispatch (always on the loop) ---------------------

    async def _on_event(self, event) -> None:
        if self._stopped:
            # The backend's rx loop can still deliver events between the tool-task
            # cancel sweep and backend.close(); acting on them would spawn work (a
            # ToolCall -> a full delegation) that nothing will ever cancel.
            return
        if isinstance(event, StateHint):
            self._apply_state(event.state)
        elif isinstance(event, OutputAudio):
            self._sink.enqueue(event)
        elif isinstance(event, UserSpeechStarted):
            await self._cloud_barge_in()
        elif isinstance(event, ToolCall):
            self._spawn_tool_task(event)
        elif isinstance(event, OutputTranscript):
            pass  # observational; nothing here consumes assistant text
        elif isinstance(event, InputTranscript):
            self._log.debug("user: {}", loggable_text(event.text, self._log_transcripts))
        elif isinstance(event, ToolStarted):
            self._log.info("tool_started: {}", event.name or "?")
        elif isinstance(event, TurnDone):
            self._log.debug("turn_done")
        elif isinstance(event, Error):
            self._log.warning("backend error: {}", event.message)
            if event.fatal:
                self._spawn_fatal_stop()  # defer: never tear down from inside dispatch

    def _spawn_fatal_stop(self) -> None:
        # Strong reference: asyncio only weak-refs pending tasks.
        if self._fatal_task is not None and not self._fatal_task.done():
            return
        self._fatal_task = asyncio.create_task(self._fatal_stop())

    async def _fatal_stop(self) -> None:
        """Tear the shell down on a fatal backend error, then tell the owner:
        the callback runs even if teardown raises."""
        try:
            await self.stop()
        finally:
            if self._on_fatal is not None:
                await self._on_fatal()

    def _apply_state(self, state: VoiceState) -> None:
        """The ONLY place VoiceState changes: mirrors a StateHint from the backend."""
        if state != self._state:
            self._log.info("state {} -> {}", self._state.value, state.value)
            self._state = state

    async def _cloud_barge_in(self) -> None:
        # Cloud only. State is already CAPTURING (the backend emits StateHint before
        # UserSpeechStarted); flush, then hand played-ms on for item.truncate.
        played_ms = await self._sink.flush()
        try:
            await self._backend.barge_in(played_ms)
        finally:
            # Abandoning the delegation must not depend on the wire step: a dead
            # socket would otherwise keep paying for a turn nobody will hear.
            if self._on_barge_in is not None:
                await self._on_barge_in()

    def _spawn_tool_task(self, ev: ToolCall) -> None:
        """Run a tool call OFF the event-dispatch path: ``_on_event`` is awaited on
        the backend's receive loop, so awaiting a slow tool here would stall EVERY
        other event: audio deltas, and crucially a barge-in ``UserSpeechStarted``
        (a supervisor delegation takes ~1-2 s at minimum). Safe because the backend's
        tool state machine keys on ``call_id`` and tolerates any result order."""
        self._metrics.call_spawned(ev.call_id)
        task = asyncio.create_task(self._on_tool_call(ev))
        self._tool_tasks.add(task)
        task.add_done_callback(self._tool_tasks.discard)

    async def _on_tool_call(self, ev: ToolCall) -> None:
        # Cloud only. Route through nanobot's ToolRegistry (security parity); degrade to
        # an error string the model can recover from when the seam is absent. ToolRegistry
        # NEVER raises for a tool failure: ToolResult.error(...) is a str subclass, so
        # `is_error` is the only signal; relying on exceptions would report ~0% failures.
        outcome = "ok"
        with self._tracer.tool_span(ev.name, ev.call_id, ev.arguments) as span:
            if self._exec_tool is None:
                outcome = "no_seam"
                output = "Error: tool execution is unavailable in this voice session."
            else:
                try:
                    result = await self._exec_tool(ev.name, ev.arguments)
                    if getattr(result, "is_error", False):
                        outcome = "error"
                    output = result if isinstance(result, str) else json.dumps(result)
                except asyncio.CancelledError:
                    # Teardown: drop the call rather than submit a bogus result; the
                    # unanswered obligation is counted as its own outcome.
                    self._metrics.call_finished(
                        ev.call_id, outcome="cancelled", mode=self._tool_mode
                    )
                    self._tracer.tool_outcome(
                        span, outcome="cancelled", mode=self._tool_mode,
                        stale=False, result=None,
                    )
                    raise
                except Exception as exc:  # noqa: BLE001 - report the failure to the model
                    outcome = "exception"
                    output = json.dumps({"error": str(exc)})
                    # Recorded here, not by the span CM: we swallow the exception, so
                    # the tracer would otherwise see a clean exit.
                    span.record_exception(exc)
            # Did the user barge in while this ran? Counted only for visibility: the
            # backend's own stale guard keeps the result from reviving a dead turn.
            stale = self._metrics.call_stale(ev.call_id, self._sink.epoch)
            self._metrics.call_finished(ev.call_id, outcome=outcome, mode=self._tool_mode)
            self._tracer.tool_outcome(
                span, outcome=outcome, mode=self._tool_mode, stale=stale, result=output,
            )
        if outcome != "ok":
            self._log.info("tool {} -> {}", ev.name, outcome)
        try:
            await self._backend.submit_tool_result(ev.call_id, output)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - socket may be gone mid-teardown
            self._log.debug("submit_tool_result failed: {}", exc)
