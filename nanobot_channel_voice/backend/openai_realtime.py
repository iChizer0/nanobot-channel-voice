"""RealtimeBackend: e2e speech-to-speech over the OpenAI Realtime API **dialect**.

A raw WebSocket client behind :class:`VoiceBackend`, driving every supplier that speaks
the protocol (GA: OpenAI/xAI/Azure; beta: Qwen-Omni/GLM/StepFun) off a pure-data
:class:`~.profiles.RealtimeProfile`. The provider does ASR + reasoning + TTS; the plugin
owns mic capture, playback and tool routing.

One rx task owns connect + receive + bounded reconnect; a sender task drains a bounded
drop-oldest audio queue so a slow socket never stalls capture; every frame goes through
one lock-guarded ``_send``. A tool turn spans >= 2 responses: each ``function_call``
registers an obligation, the continuation fires exactly once (triggering response done,
all outputs submitted, not cancelled), and ``TurnDone`` comes only from the turn's final
``response.done``. ``_handle_event``'s only send is that continuation, so it is testable
against canned server frames.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import json
import time
from contextlib import suppress

from loguru import logger

from nanobot_channel_voice.aio import Throttle, put_drop_oldest, wait_for_stall
from nanobot_channel_voice.config import VoiceConfig, resolve_openai_key
from nanobot_channel_voice.metrics import VoiceMetrics
from nanobot_channel_voice.phrases import (
    FILLER_WORDS,
    PhraseLexicon,
    PhraseMatcher,
    tokens_of,
)

from .audio_sink import AudioSink
from .base import (
    Error,
    InputTranscript,
    OnEvent,
    OutputAudio,
    OutputTranscript,
    ToolCall,
    ToolDef,
    ToolStarted,
    TurnDone,
    UserSpeechStarted,
    VoiceState,
)
from .common import TurnEventMixin, loggable_text
from .profiles import RealtimeProfile

_SEND_Q_MAX = 64  # ~1.3s of 20ms frames; drop-oldest past this

# Grace (mirrors local's _KILL_GRACE_S): a bare stop right after a consumed one is a
# double-tap, not a new turn. Suppress covers a stop transcript landing before the server
# creates the response answering it; new user speech ends the window.
_STOP_GRACE_S = 3.0
_STOP_SUPPRESS_S = 2.0
_BACKOFF = (0.5, 1.0, 2.0)
# Healthy session: resets the backoff budget, so an endpoint that recycles long sessions
# (Qwen turn caps) never reaches "reconnect exhausted".
_HEALTHY_SESSION_S = 30.0


def _status_detail(resp: dict) -> str:
    """Best-effort detail from ``response.status_details`` (GA: ``.error.message`` /
    ``.reason``; beta unvalidated). Must not raise: ``response.done`` handling would abort
    with the watchdog already cancelled, skipping the turn's end (TurnDone, drain)."""
    details = resp.get("status_details")
    if isinstance(details, str):
        return details
    if not isinstance(details, dict):
        return ""
    err = details.get("error")
    if isinstance(err, dict) and isinstance(err.get("message"), str):
        return err["message"]
    if isinstance(err, str) and err:
        return err
    reason = details.get("reason")
    return reason if isinstance(reason, str) else ""


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` **in place**."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def _normalize_schema(schema: dict) -> dict:
    """Lossily flatten a JSON Schema for providers rejecting list ``type`` (nullable
    unions) and ``anyOf``/``allOf``/``oneOf`` — opt-in via
    ``RealtimeProfile.flatten_tool_schema``. Union -> first non-null member, combinator ->
    first ``type``-bearing branch: the goal is acceptance, not a round-trip."""
    if not isinstance(schema, dict):
        return schema
    result: dict = {}
    for key, value in schema.items():
        if key == "type" and isinstance(value, list):
            non_null = [t for t in value if t != "null"]
            if non_null:
                result[key] = non_null[0]
            else:
                result[key] = "string"  # degenerate all-"null" union; arbitrary fallback
        elif key == "properties" and isinstance(value, dict):
            result[key] = {k: _normalize_schema(v) for k, v in value.items()}
        elif key == "items" and isinstance(value, dict):
            result[key] = _normalize_schema(value)
        elif key in ("anyOf", "allOf", "oneOf") and isinstance(value, list):
            simple = next((alt for alt in value if isinstance(alt, dict) and "type" in alt), None)
            if simple:
                result.update(_normalize_schema(simple))
        else:
            result[key] = value
    return result


def _tool_to_wire(tool: ToolDef, *, flatten: bool = False) -> dict:
    """The flat OpenAI-function wire shape, identical in both dialects."""
    params = _normalize_schema(tool.parameters) if flatten else tool.parameters
    return {
        "type": "function",
        "name": tool.name,
        "description": tool.description,
        "parameters": params,
    }


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


class RealtimeBackend(TurnEventMixin):
    def __init__(
        self,
        config: VoiceConfig,
        *,
        sink: AudioSink,
        profile: RealtimeProfile,
        metrics: VoiceMetrics | None = None,
        aec=None,
    ):
        # Shared with the shell/channel: one call's segments land in one collector.
        self._metrics = metrics if metrics is not None else VoiceMetrics()
        self._rt = config.realtime
        self._profile = profile
        self._model = config.realtime.model or profile.default_model
        self._voice = config.realtime.voice or profile.default_voice_for(self._model)
        # supports_tools is the CHANNEL's business; the backend needs only these two.
        caps = profile.capabilities_for(self._model)
        self._needs_response_create_after_tools = bool(caps["needs_response_create_after_tools"])
        self._max_tool_output_chars = int(caps.get("max_tool_output_chars", 0) or 0)
        self._sink = sink
        # Software AEC3 front-end (barge_in="aec" w/o hardware AEC); sink feeds the ref.
        self._aec = aec
        self._on_event: OnEvent | None = None
        self._instructions: str | None = None
        self._tools: list[ToolDef] = []

        self._ws = None
        self._ready = asyncio.Event()
        self._closing = False
        self._rx_task: asyncio.Task | None = None
        self._sender_task: asyncio.Task | None = None
        self._drain_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None
        self._send_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=_SEND_Q_MAX)
        self._send_lock = asyncio.Lock()
        self._warn_throttle = Throttle()
        self._ever_ready = False
        self._auth_fails = 0
        # _progress_t feeds the turn deadman; _last_error details a bare failed response.done.
        self._progress_t = 0.0
        self._last_error: str | None = None

        self._turn = VoiceState.IDLE
        # Needs input transcription; without it the persona rule is the only (soft) cover.
        self._log_transcripts = config.log_transcripts
        self._stop_match = (
            PhraseMatcher(
                PhraseLexicon(config.barge_in.stop_phrases),
                PhraseLexicon(config.barge_in.ack_phrases),
                extra=FILLER_WORDS,
            )
            if config.barge_in.stop_phrases and self._rt.input_transcription_model
            else None
        )
        # The stop latch + clocks live in _reset_turn_state, per LATEST onset.
        self._log = logger.bind(component="voice")  # before _reset_turn_state: it logs
        self._reset_turn_state()

    # ---- turn/session bookkeeping -------------------------------------------

    def _reset_turn_state(self, *, reason: str = "init") -> None:
        # getattr: also runs from __init__, before the maps exist.
        pending: set[str] = set()
        for cids in getattr(self, "_tools_pending", {}).values():
            pending |= cids
        # The collector decides the real count: an answered call can linger in the map.
        dropped = self._metrics.calls_dropped(pending, reason)
        if dropped:
            self._log.warning("dropping {} unanswered tool obligation(s) on {}", dropped, reason)

        self._active_response_id: str | None = None
        self._audio_item_id: str | None = None
        self._item_base_played = 0
        # Never carry a dead session's fault into the next one's failure detail.
        self._last_error = None
        # call_ids THIS session announced: a result finishing after a reconnect must drop
        # (the new session rejects unknown call_ids). Cancelled responses KEEP entries:
        # still answered, they just resume nothing.
        self._session_calls: set[str] = set()
        # Insertion-ordered: the size bound must evict the OLDEST rid, not a
        # hash-arbitrary one whose late deltas still stream.
        self._cancelled_responses: dict[str, None] = {}
        self._response_done: set[str] = set()
        self._response_had_tools: dict[str, bool] = {}
        self._tools_pending: dict[str, set[str]] = {}
        self._call_to_response: dict[str, str] = {}
        self._fn_names: dict[str, str] = {}
        self._fn_args: dict[str, str] = {}
        # Session-scoped: a suppress window or latch carried across a reconnect (well
        # inside the 2 s window) would cancel the NEW session's first response at birth.
        self._onset_interrupting = False
        self._last_stop_consume = float("-inf")
        self._stop_suppress_until = 0.0
        # Barge-in latency clock (monotonic ms): set at server-VAD onset, eaten by the
        # next barge-in; session-scoped, or a reconnect inherits a stale onset.
        self._speech_started_at: float | None = None
        # The only span in which uplink frames may feed the deadman (see the sender loop).
        self._user_speaking = False

    @property
    def metrics(self) -> VoiceMetrics:
        return self._metrics

    def _api_key(self) -> str:
        key = resolve_openai_key(self._rt.api_key)
        if not key:
            raise RuntimeError(
                f"no API key for realtime provider '{self._profile.key}' "
                "(set channels.voice.realtime.apiKey or OPENAI_API_KEY)"
            )
        return key

    # ---- VoiceBackend contract ----------------------------------------------

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
        # Session-ready barrier: drop until the format/VAD config is applied.
        if self._closing or not self._ready.is_set():
            return
        if put_drop_oldest(self._send_q, pcm) is not None:
            self._warn_backpressure()  # dropped a frame to stay near real time

    async def barge_in(self, played_ms: int) -> None:
        try:
            if self._profile.interrupt == "cancel":
                # Beta: WE must stop the response or the model keeps generating audio.
                await self._cancel_active()
                return
            # GA: truncate only — a second response.cancel would race the server's
            # auto-cancel, which interruptResponse=off disables.
            if not self._rt.interrupt_response:
                await self._cancel_active()
            else:
                self._mark_cancelled()
            item = self._audio_item_id
            if item is None:
                return  # nothing audible to truncate (already marked cancelled above)
            # One truncate per item: a second barge-in before the next output_item.added
            # would re-truncate at audio_end_ms=0 (a flushed sink restarts played_ms),
            # wiping the model's memory of audio the user DID hear.
            self._audio_item_id = None
            audio_end = max(0, played_ms - self._item_base_played)
            try:
                # A dead socket must not escape: the shell's post-barge-in callback runs.
                await self._send({
                    "type": "conversation.item.truncate",
                    "item_id": item, "content_index": 0, "audio_end_ms": audio_end,
                })
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log.debug("truncate failed: {}", exc)
        finally:
            # On the way OUT: the latency includes the sink flush + send. Per mechanism.
            self._record_barge_in()

    def _record_barge_in(self) -> None:
        # Every onset routes here; only an interrupting one is a sample (stamp clears
        # regardless).
        stamp, self._speech_started_at = self._speech_started_at, None
        if stamp is None or not self._onset_interrupting:
            return
        self._metrics.observe(
            f"barge_in_ms.{self._profile.interrupt}",
            time.monotonic() * 1000.0 - stamp,
        )

    def _note_cancelled(self, rid: str) -> None:
        """Record a dead response so late deltas drop; bounded (it lives a whole session)."""
        self._cancelled_responses[rid] = None
        while len(self._cancelled_responses) > 256:  # rids are only checked while recent
            del self._cancelled_responses[next(iter(self._cancelled_responses))]

    def _mark_cancelled(self) -> None:
        if self._active_response_id:
            self._note_cancelled(self._active_response_id)

    async def _cancel_active(self) -> None:
        """Send ``response.cancel`` for the live response, at most once."""
        rid = self._active_response_id
        if not rid or rid in self._cancelled_responses:
            return
        with suppress(Exception):
            await self._send({"type": "response.cancel"})
        self._note_cancelled(rid)

    async def _consume_stop(self, text: str) -> None:
        """A pure stop command: the response answering it dies unspoken (cancelled now if
        live, at birth via the suppress window if not) and queued audio is flushed.
        Deliberately NO conversation.item.truncate: the tracked item may still be the
        PREVIOUS reply's, and truncating a fully-heard item corrupts the model's memory."""
        self._metrics.count("barge_in_stop")
        self._log.info(
            "stop command consumed (cloud): '{}'",
            loggable_text(text, self._log_transcripts, 40),
        )
        self._last_stop_consume = time.monotonic()
        self._cancel_drain()
        rid = self._active_response_id
        if rid is not None and rid not in self._cancelled_responses:
            await self._cancel_active()
        else:
            # Nothing live to cancel: the response answering THIS stop is still unborn.
            self._stop_suppress_until = time.monotonic() + _STOP_SUPPRESS_S
        await self._sink.flush()
        # The item dies with the flush: played_ms() restarts at 0, so a later barge-in
        # against the old base would truncate a partially-heard item at 0. Cleared, the
        # next barge-in sends nothing and the model over-remembers instead.
        self._audio_item_id = None
        self._item_base_played = 0
        if self._turn is not VoiceState.IDLE:
            await self._set_turn(VoiceState.IDLE)

    def _clamp_tool_output(self, output: str) -> str:
        """Clamp an oversized tool output to protect the realtime context window; the
        marker keeps the model aware the tool ran."""
        limit = self._max_tool_output_chars
        if limit <= 0 or len(output) <= limit:
            return output
        marker = (
            f"[output truncated: {len(output)} chars exceeds the voice session's "
            f"{limit}-char tool-result limit; use offset/limit or ask for a specific range]"
        )
        if len(marker) >= limit:
            return marker[:limit]
        keep = max(0, limit - len(marker) - 1)  # leave one newline separator
        return output[:keep] + "\n" + marker

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        # Always satisfy the pending call; only trigger a new response if the turn lives.
        if call_id not in self._session_calls:
            # The issuing session is gone (reconnect); this one would reject the call_id.
            self._log.debug("dropping tool result for unknown call_id {} (session lost)", call_id)
            return
        output = self._clamp_tool_output(output)
        payload = {
            "type": "conversation.item.create",
            "item": {"type": "function_call_output", "call_id": call_id, "output": output},
        }
        # call_id/size only: serializing the payload would double every result's JSON cost.
        self._log.debug("submit_tool_result call_id={} ({} chars)", call_id, len(output))
        await self._send(payload)
        # Answered exactly once: drop the bookkeeping, incl. the duplicate-submit guard.
        self._session_calls.discard(call_id)
        rid = self._call_to_response.pop(call_id, None)
        self._fn_names.pop(call_id, None)
        self._fn_args.pop(call_id, None)
        self._log.debug("submit_tool_result call_id={} -> rid={}", call_id, rid)
        if rid is None:
            return
        pending = self._tools_pending.get(rid)
        if pending is not None:
            pending.discard(call_id)
            self._log.debug("tools pending for rid={}: {}", rid, pending)
        await self._maybe_respond(rid)

    async def on_capture_gap(self) -> None:
        """No-op: the provider's server VAD sees the uplink go quiet on its own."""

    async def close(self) -> None:
        self._closing = True
        self._ready.clear()
        for task in (self._drain_task, self._watchdog_task, self._sender_task, self._rx_task):
            if task is not None:
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
        self._drain_task = self._watchdog_task = self._sender_task = self._rx_task = None
        ws, self._ws = self._ws, None
        if ws is not None:
            with suppress(Exception):
                await ws.close()

    # ---- connection / io ----------------------------------------------------

    async def _rx_loop(self) -> None:
        attempt = 0
        while not self._closing:
            started = time.monotonic()
            try:
                await self._connect_and_run()
                # A CLEAN server close lands here, not in `except` (Qwen's per-session turn
                # cap); still a disconnect: same teardown + backoff.
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any transport failure
                if self._closing:
                    break
                # An auth-rejected HANDSHAKE is credentials, not a blip: fatal at once if
                # the key NEVER worked, else one ladder retry (proxy blip, key rotation).
                # .response.status_code = modern websockets InvalidStatus; .status_code =
                # legacy InvalidStatusCode.
                status = getattr(getattr(exc, "response", None), "status_code", None)
                if status is None:
                    status = getattr(exc, "status_code", None)
                if status in (401, 403):
                    self._auth_fails += 1
                    if not self._ever_ready or self._auth_fails >= 2:
                        await self._emit(Error(
                            message=f"realtime auth rejected (HTTP {status}): check "
                                    "realtime.apiKey / OPENAI_API_KEY for this provider",
                            fatal=True,
                        ))
                        break
                await self._emit(Error(message=f"realtime disconnected: {exc}", fatal=False))
            if self._closing:
                break
            await self._on_session_lost()
            # Only back-to-back FAST failures walk the ladder to the fatal rung.
            if time.monotonic() - started >= _HEALTHY_SESSION_S:
                attempt = 0
            if attempt >= len(_BACKOFF):
                await self._emit(Error(message="realtime reconnect exhausted", fatal=True))
                break
            await asyncio.sleep(_BACKOFF[attempt])
            attempt += 1

    async def _on_session_lost(self) -> None:
        """Teardown shared by every way a session can end, clean or not."""
        self._ready.clear()
        # A surviving watchdog would fire, with real side effects, into the next session.
        self._cancel_watchdog()
        self._cancel_drain()
        self._reset_turn_state(reason="session_lost")
        # A surviving anchor would measure new audio against a turn that no longer exists.
        self._metrics.turn_end()
        # The half-duplex mic gate keys on SPEAKING: dropping while SPEAKING wedges forever
        # (mic gated -> no audio out -> no speech_started to move off SPEAKING).
        await self._set_turn(VoiceState.IDLE)

    async def _connect_and_run(self) -> None:
        connect = _load_connect()
        url = self._profile.connect_url(self._rt.base_url, self._model)
        headers = self._profile.auth_headers(self._api_key())
        async with connect(url, additional_headers=headers) as ws:
            try:
                self._ws = ws
                await self._send(self._session_update_payload())
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
                        self._log.exception("event handler failed for {}", evt.get("type"))
            finally:
                # _ws must not outlive the socket: submit_tool_result must see None and
                # drop the frame, not raise and skip its bookkeeping.
                self._ws = None

    async def _sender_loop(self) -> None:
        while not self._closing:
            try:
                pcm = await self._send_q.get()
            except asyncio.CancelledError:
                raise
            try:
                if self._closing or not self._ready.is_set():
                    continue
                await self._send({
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode("ascii"),
                })
                if self._user_speaking and self._turn is VoiceState.CAPTURING:
                    # A monologue longer than turn_timeout_s emits no server events, so the
                    # watchdog armed at speech_started would fire mid-sentence. AUDIBLY
                    # speaking only: post-speech_stopped silence (still CAPTURING) or idle
                    # frames in THINKING would mask a server that never answers.
                    self._progress_t = time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - transient; frames may drop
                self._log.debug("append failed: {}", exc)
            finally:
                self._send_q.task_done()

    async def _send(self, obj: dict) -> None:
        ws = self._ws
        if ws is None:
            return
        async with self._send_lock:
            await ws.send(json.dumps(obj))

    def _session_update_payload(self) -> dict:
        """Per-dialect; the receive path (``_handle_event``) is dialect-agnostic."""
        if self._profile.dialect == "beta":
            return self._beta_session_payload()
        return self._ga_session_payload()

    def _ga_session_payload(self) -> dict:
        # GA (OpenAI/xAI/Azure OpenAI): nested session.audio.{input,output}.format.
        audio_in: dict = {"format": {"type": "audio/pcm", "rate": self._profile.input_rate}}
        if self._rt.server_vad:
            audio_in["turn_detection"] = {
                "type": "server_vad", "interrupt_response": self._rt.interrupt_response,
            }
        else:
            audio_in["turn_detection"] = None
        if self._rt.input_transcription_model:
            audio_in["transcription"] = {"model": self._rt.input_transcription_model}
        audio_out: dict = {"format": {"type": "audio/pcm", "rate": self._profile.output_rate}}
        session: dict = {
            "type": "realtime",
            "instructions": self._instructions or "",
            "output_modalities": ["audio"],
            "audio": {"input": audio_in, "output": audio_out},
            "tools": [_tool_to_wire(t, flatten=self._profile.flatten_tool_schema)
                      for t in self._tools],
        }
        # xAI nests the audio formats GA-style but reads `voice` from the session root.
        if self._profile.voice_in_session_root:
            session["voice"] = self._voice
        else:
            audio_out["voice"] = self._voice
        return {"type": "session.update", "session": session}

    def _beta_session_payload(self) -> dict:
        # beta (Qwen-Omni/GLM/StepFun): flat input_audio_format + top-level turn_detection,
        # no conversation.item.truncate. The format STRING is per-vendor; see profiles.py.
        session: dict = {
            "modalities": ["text", "audio"],
            "instructions": self._instructions or "",
            "voice": self._voice,
            "input_audio_format": self._profile.input_format,
            "output_audio_format": self._profile.output_format,
            "turn_detection": {"type": "server_vad"} if self._rt.server_vad else None,
        }
        if self._rt.input_transcription_model:
            session["input_audio_transcription"] = {"model": self._rt.input_transcription_model}
        if self._tools:
            session["tools"] = [_tool_to_wire(t, flatten=self._profile.flatten_tool_schema)
                                for t in self._tools]
        if self._profile.session_extras:
            # Deep-copy: session_extras lives on the PROFILES singleton; _deep_merge aliases.
            _deep_merge(session, copy.deepcopy(self._profile.session_extras))
        payload = {"type": "session.update", "session": session}
        self._log.debug("session.update payload: {}", json.dumps(payload, ensure_ascii=False))
        return payload

    def _warn_backpressure(self) -> None:
        if not self._warn_throttle.ready():
            return
        self._log.warning(
            "realtime uplink is congested (dropping mic frames); check network/bandwidth "
            "to the Realtime API."
        )

    # ---- event mapping ------------------------------------------------------

    async def _handle_event(self, evt: dict) -> None:
        t = evt.get("type", "")
        if t in ("session.created", "session.updated"):
            self._ready.set()
            self._ever_ready = True
            self._auth_fails = 0
        elif t == "input_audio_buffer.speech_started":
            self._cancel_drain()
            self._arm_watchdog()  # recover if the server never turns this into a response
            self._user_speaking = True
            self._speech_started_at = time.monotonic() * 1000.0
            # Read BEFORE the CAPTURING transition overwrites it. New speech also ends the
            # suppression: whatever follows answers the NEW utterance, not a consumed stop.
            self._onset_interrupting = self._turn in (
                VoiceState.THINKING, VoiceState.SPEAKING,
            )
            self._stop_suppress_until = 0.0
            await self._set_turn(VoiceState.CAPTURING)
            await self._emit(UserSpeechStarted())
        elif t == "input_audio_buffer.speech_stopped":
            # MEASUREMENT ONLY: the anchor turn latency is measured from (end of user
            # speech); absent without server_vad, and then it goes unrecorded.
            self._metrics.turn_anchor()
            self._user_speaking = False  # uplink frames stop feeding the deadman here
            self._progress_t = time.monotonic()  # end of speech IS turn progress
        elif t == "response.created":
            rid = (evt.get("response") or {}).get("id")
            if time.monotonic() < self._stop_suppress_until:
                # Kill the consumed stop's response at birth; silence is the acknowledgment.
                self._stop_suppress_until = 0.0
                self._active_response_id = rid
                await self._cancel_active()
                return
            self._cancel_drain()
            self._active_response_id = rid
            if rid:
                self._response_had_tools.setdefault(rid, False)
            self._last_error = None  # only errors seen inside THIS response may detail it
            self._metrics.turn_thinking()
            await self._set_turn(VoiceState.THINKING)
            self._arm_watchdog()
        elif t == "response.output_item.added":
            await self._on_item_added(evt)
        elif t in ("response.output_audio_transcript.delta", "response.audio_transcript.delta"):
            self._adopt_response(evt)
            if self._is_live(evt):
                self._progress_t = time.monotonic()
                await self._emit(OutputTranscript(evt.get("delta", "")))
        elif t in ("response.output_audio.delta", "response.audio.delta"):
            self._adopt_response(evt)
            await self._on_audio_delta(evt)
        elif t == "response.function_call_arguments.delta":
            self._adopt_response(evt)
            cid = evt.get("call_id")
            # Liveness-gated: a cancelled response's args accumulate scratch nothing cleans
            # (the cancel can race ahead of output_item.added, so the cid maps to no rid).
            if cid is not None and self._is_live(evt):
                self._fn_args[cid] = self._fn_args.get(cid, "") + evt.get("delta", "")
        elif t == "response.function_call_arguments.done":
            await self._on_fn_done(evt)
        elif t == "response.done":
            await self._on_response_done(evt)
        elif t == "conversation.item.input_audio_transcription.completed":
            text = evt.get("transcript", "")
            if text:
                await self._emit(InputTranscript(text))
                if (
                    self._stop_match is not None
                    and (
                        self._onset_interrupting
                        or self._active_response_id is not None
                        or time.monotonic() - self._last_stop_consume <= _STOP_GRACE_S
                    )
                    and self._stop_match.pure(tokens_of(text))
                ):
                    await self._consume_stop(text)
        elif t == "error":
            await self._on_error(evt)

    def _adopt_response(self, evt: dict) -> None:
        """Create-on-first-sight: a dialect may emit ``response.*`` before (or without)
        ``response.created``, and ``_is_live`` fails closed with no active id. Never adopts
        over a live id, nor over a cancelled/completed one (both sit in
        ``_cancelled_responses``, which outlives ``_cleanup_response``)."""
        rid = evt.get("response_id")
        if (
            rid
            and self._active_response_id is None
            and rid not in self._cancelled_responses
        ):
            self._cancel_drain()
            self._active_response_id = rid
            self._response_had_tools.setdefault(rid, False)
            self._arm_watchdog()
            self._log.debug("adopted response {} (no response.created seen)", rid)

    def _is_live(self, evt: dict) -> bool:
        """True if the event belongs to the active, non-cancelled response."""
        rid = evt.get("response_id") or self._active_response_id
        if rid is None:
            # Fail CLOSED: the active id clears at turn end, so `None == None` would let a
            # rid-less straggler play audio or re-run a tool for a dead turn.
            return False
        return rid == self._active_response_id and rid not in self._cancelled_responses

    async def _on_item_added(self, evt: dict) -> None:
        self._adopt_response(evt)
        if not self._is_live(evt):
            # A just-cancelled response's in-flight item must not overwrite the truncate
            # target, re-register discarded obligations, or emit a spurious ToolStarted.
            return
        item = evt.get("item") or {}
        rid = evt.get("response_id") or self._active_response_id
        if item.get("type") == "message":
            self._audio_item_id = item.get("id")
            # Truncate baseline: where this item's audio STARTS. played + backlog, not
            # played alone, or an item added over a buffered tail over-counts audio_end_ms
            # past the item's real length (GA rejects that); backlog over-counts, so
            # audio_end under-counts — the safe way. Valid only within one sink stream
            # generation: a republish restarts played_ms() at 0.
            self._item_base_played = self._sink.played_ms() + self._sink.backlog_ms()
        elif item.get("type") == "function_call":
            cid = item.get("call_id")
            name = item.get("name", "")
            if rid:
                self._response_had_tools[rid] = True
                self._tools_pending.setdefault(rid, set())
                if cid:
                    self._tools_pending[rid].add(cid)
                    self._call_to_response[cid] = rid
            if cid:
                if cid not in self._session_calls:  # fn_done may have registered first
                    self._session_calls.add(cid)
                    self._metrics.call_seen(cid, name)
                self._fn_names[cid] = name
                self._fn_args.setdefault(cid, "")
            await self._emit(ToolStarted(name or None, call_id=cid))

    async def _on_audio_delta(self, evt: dict) -> None:
        if not self._is_live(evt):
            return  # stale-response guard (cancelled/inactive) -> drop
        b64 = evt.get("delta")
        if not b64:
            return
        try:
            pcm = base64.b64decode(b64)
        except (ValueError, TypeError):
            return
        if self._turn is not VoiceState.SPEAKING:
            await self._set_turn(VoiceState.SPEAKING)
        self._progress_t = time.monotonic()  # feed the deadman: the turn is alive
        # TTFA, latched to the turn's first frame at ENQUEUE: device playout is excluded.
        self._metrics.turn_first_audio()
        await self._emit(OutputAudio(epoch=self._sink.epoch, pcm=pcm, rate=self._profile.output_rate))

    async def _on_fn_done(self, evt: dict) -> None:
        self._adopt_response(evt)
        if not self._is_live(evt):
            self._log.debug("function_call_arguments.done ignored (not live): {}", evt)
            if (cid := evt.get("call_id")) is not None:
                self._fn_names.pop(cid, None)  # dead turn: drop its scratch now
                self._fn_args.pop(cid, None)
            return
        cid = evt.get("call_id")
        if cid is None:
            return
        self._progress_t = time.monotonic()  # tool activity is turn progress too
        # PRIMARY obligation registration: a dialect may skip output_item.added entirely,
        # leaving the call invisible to the continuation logic. Double-add is a no-op.
        rid = evt.get("response_id") or self._active_response_id
        if rid:
            self._response_had_tools[rid] = True
            self._tools_pending.setdefault(rid, set()).add(cid)
            self._call_to_response.setdefault(cid, rid)
        if cid not in self._session_calls:
            self._session_calls.add(cid)
            self._metrics.call_seen(cid, evt.get("name") or "")
        name = evt.get("name") or self._fn_names.get(cid, "")
        args = evt.get("arguments")
        if args is None:
            args = self._fn_args.get(cid, "")
        self._log.debug("tool call ready: {}({})", name, args)
        self._metrics.call_dispatched(cid, self._sink.epoch)
        await self._emit(ToolCall(call_id=cid, name=name, arguments=args))

    async def _on_response_done(self, evt: dict) -> None:
        resp = evt.get("response") or {}
        rid = resp.get("id")
        status = resp.get("status")
        self._log.debug("response.done rid={} status={}", rid, status)
        self._cancel_watchdog()
        if rid and rid in self._cancelled_responses:
            # Already declared dead (barge-in/watchdog): falling through to completed would
            # fire a second TurnDone + turn_end (wiping the new utterance's anchor) and
            # drain CAPTURING -> IDLE mid-speech.
            self._discard_response_tools(rid)
            if rid == self._active_response_id:
                self._active_response_id = None
            # The deadman's ONLY re-arm point on the GA path: without it one interruption
            # disarms the watchdog for the session. IDLE = the stop already settled the turn.
            if self._turn is not VoiceState.IDLE:
                self._arm_watchdog()
            return
        if status == "cancelled":
            if rid:
                self._note_cancelled(rid)
                self._discard_response_tools(rid)
                if rid == self._active_response_id:
                    self._active_response_id = None
            # SERVER-initiated cancel (turn_detected): the paired speech_started may arrive
            # AFTER this event or never, wedging a gated-mic session in SPEAKING forever (a
            # later one just re-arms again). CAPTURING counts too: the cancel can land while
            # the user is still speaking.
            if self._turn is not VoiceState.IDLE:
                self._arm_watchdog()
            return
        if status in ("failed", "incomplete"):
            # Announced calls whose arguments.done never arrives would wedge _maybe_respond
            # forever: drop them like a cancellation, but still end the turn cleanly.
            if rid:
                self._note_cancelled(rid)
                self._discard_response_tools(rid)
                if rid == self._active_response_id:
                    self._active_response_id = None
            # Providers may put the reason in a top-level `error` event just BEFORE the
            # failed done, leaving status_details bare.
            detail = _status_detail(resp) or self._last_error
            self._last_error = None
            if status == "failed":
                await self._emit(Error(message=f"realtime response failed: {detail or 'no detail'}",
                                       fatal=False))
            else:  # incomplete: a routine cap (max tokens / content filter), not an Error
                self._log.warning("realtime response incomplete: {}", detail or "no detail")
            self._metrics.turn_end()
            await self._emit(TurnDone())
            self._start_drain()
            return
        # completed
        if rid:
            self._response_done.add(rid)  # _maybe_respond's "trigger finished" gate
        if rid and self._response_had_tools.get(rid):
            # Turn still LIVE, so the response stays "active": a barge-in during the tool
            # run must be able to cancel it, or the abandoned results re-trigger
            # response.create over the user's new speech.
            await self._maybe_respond(rid)
        else:
            if rid == self._active_response_id:
                self._active_response_id = None  # finished turn: nothing left to cancel
            if rid:
                # Outlives _cleanup_response: a late event for this finished response must
                # not be adopted as a new live one.
                self._note_cancelled(rid)
                self._cleanup_response(rid)
            self._metrics.turn_end()  # release the anchor; the next turn re-arms it
            await self._emit(TurnDone())  # informational; drain owns the -> IDLE
            self._start_drain()

    async def _maybe_respond(self, rid: str) -> None:
        pending = self._tools_pending.get(rid)
        self._log.debug("_maybe_respond rid={} pending={} done={} cancelled={}",
                        rid, pending, rid in self._response_done, rid in self._cancelled_responses)
        if pending:
            return  # more tool outputs outstanding
        if rid not in self._response_done:
            return  # triggering response not finished yet
        if rid in self._cancelled_responses:
            self._cleanup_response(rid)  # barged out: outputs submitted, no new response
            return
        # Re-anchor before the frame: what follows is continuation latency, not TTFA.
        self._metrics.turn_continuation()
        if self._needs_response_create_after_tools:
            self._log.debug("firing response.create for rid={}", rid)
            await self._send({"type": "response.create"})
        else:
            # Auto-continuing dialects resume on the function_call_output alone.
            self._log.debug("auto-continuing dialect; no response.create for rid={}", rid)
        # response.done cancelled the watchdog: the create -> response.created gap is the
        # one window with no deadman, and a continuation never started wedges the session
        # (mic gated while SPEAKING, so nothing can recover it).
        self._arm_watchdog()
        self._cleanup_response(rid)

    def _cleanup_response(self, rid: str) -> None:
        self._tools_pending.pop(rid, None)
        self._response_done.discard(rid)
        self._response_had_tools.pop(rid, None)

    def _discard_response_tools(self, rid: str) -> None:
        """Drop a cancelled response's tool bookkeeping. A call already dispatched to the
        shell still gets answered (``_session_calls`` keeps it); it just resumes nothing,
        its rid mapping being gone."""
        orphans = self._tools_pending.pop(rid, set())
        self._metrics.calls_abandoned(orphans)
        for cid in orphans:
            self._call_to_response.pop(cid, None)
            self._fn_names.pop(cid, None)
            self._fn_args.pop(cid, None)
        self._response_had_tools.pop(rid, None)
        self._response_done.discard(rid)

    async def _on_error(self, evt: dict) -> None:
        err = evt.get("error") or {}
        code = err.get("code", "")
        msg = err.get("message", "unknown realtime error")
        # Benign: cancelling with nothing active, truncate races, etc.
        benign = code in ("response_cancel_not_active", "input_audio_buffer_commit_empty")
        fatal = code in ("invalid_api_key", "insufficient_quota", "model_not_found")
        if benign:
            self._log.debug("realtime error (benign): {}", msg)
            return
        # Kept so a following response.done(failed) with bare status_details can use it.
        self._last_error = f"{code}: {msg}" if code else msg
        await self._emit(Error(message=f"realtime error: {msg}", fatal=fatal))

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
            # response.done and gated-mic SPEAKING mutes the mic — nothing else recovers.
            self._log.warning("drain failed ({}); forcing IDLE", exc)
        self._audio_item_id = None
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
            self._log.warning("realtime turn watchdog fired (no progress); recovering")
            rid, self._active_response_id = self._active_response_id, None
            if rid:
                # Recover, not just report: stragglers drop via _is_live, a late tool result
                # cannot re-trigger the response, and the server stops generating.
                self._note_cancelled(rid)
                self._discard_response_tools(rid)
                with suppress(Exception):
                    await self._send({"type": "response.cancel"})
            self._metrics.turn_end()
            await self._emit(Error(message="realtime turn timed out", fatal=False))
            await self._set_turn(VoiceState.IDLE)
        except asyncio.CancelledError:
            raise
