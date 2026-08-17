"""RealtimeBackend: e2e speech-to-speech over the OpenAI Realtime API **dialect**.

A raw WebSocket client behind the :class:`VoiceBackend` contract. The same code drives
every supplier speaking OpenAI's Realtime protocol (OpenAI, xAI, Azure GA sub-dialect;
Alibaba Qwen-Omni, Zhipu GLM, StepFun beta) by reading a pure-data
:class:`~.profiles.RealtimeProfile` for the bits that differ; see
``DESIGN-realtime-providers.md``. The provider does ASR + reasoning + TTS in one session;
the plugin owns local mic capture + speaker playback and routes tool calls to nanobot.

One rx task owns connect + receive + bounded reconnect; a separate sender task drains a
bounded drop-oldest audio queue so a slow socket never stalls capture; every frame goes
through one lock-guarded ``_send``. A tool turn spans >= 2 responses: each
``function_call`` registers an obligation, the continuation fires exactly once — after
the triggering response is done, all its outputs are submitted, and only if it was not
cancelled — and ``TurnDone`` comes only from the turn's final ``response.done``.
``_handle_event`` emits normalized events and mutates state; the only frame it can send
is the tool continuation, so it is unit-testable against canned server frames.
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

# Stop-command consume (transcript-gated; the cloud half of rd DESIGN-stop-commands.md).
# Grace mirrors local's _KILL_GRACE_S: a second bare stop right after a consumed one is a
# double-tap, not a new turn. The suppress window covers transcripts that land BEFORE the
# server creates the response answering them — that response is cancelled at creation;
# new user speech ends the window (whatever follows answers the NEW utterance).
_STOP_GRACE_S = 3.0
_STOP_SUPPRESS_S = 2.0
_BACKOFF = (0.5, 1.0, 2.0)
# A session this long counts as healthy and resets the backoff budget: separates "the
# endpoint recycles long sessions" (Qwen turn caps) from "it rejects us", which alone
# should reach "reconnect exhausted".
_HEALTHY_SESSION_S = 30.0


def _status_detail(resp: dict) -> str:
    """Best-effort human detail from ``response.status_details``. Shape-defensive: GA puts
    failures under ``.error.message`` and caps under ``.reason``, the beta dialects are
    unvalidated, and a raise here would abort ``response.done`` handling with the watchdog
    already cancelled, skipping the turn's end (TurnDone, drain)."""
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
    """Lossily flatten a JSON Schema for providers that accept neither a list ``type``
    (nullable unions) nor the ``anyOf``/``allOf``/``oneOf`` combinators — opt-in via
    ``RealtimeProfile.flatten_tool_schema``, i.e. only Qwen-Omni-Realtime. A union narrows
    to its first non-null member and a combinator to its first ``type``-bearing branch:
    the goal is acceptance, not a round-trip."""
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
        # Shared with the shell/channel so one call's segments (seen -> dispatched ->
        # executed -> continuation) land in one place.
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
        # Stop-command consume: active only with input transcription on — without
        # transcripts the matcher is None and the persona rule is the only (soft) cover.
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
        # The stop targeting latch and its clocks live in _reset_turn_state: the local
        # onset latch's cloud twin, per LATEST onset (the transcript event carries no
        # item->onset mapping to be more precise with).
        self._log = logger.bind(component="voice")  # before _reset_turn_state: it logs
        self._reset_turn_state()

    # ---- turn/session bookkeeping -------------------------------------------

    def _reset_turn_state(self, *, reason: str = "init") -> None:
        # Pending call_ids are obligations the model will never see answered — invisible
        # unless logged here. getattr: also runs from __init__, before the maps exist.
        pending: set[str] = set()
        for cids in getattr(self, "_tools_pending", {}).values():
            pending |= cids
        # The collector decides the real count: a call the shell already answered is no
        # longer open, though it can linger in _tools_pending.
        dropped = self._metrics.calls_dropped(pending, reason)
        if dropped:
            self._log.warning("dropping {} unanswered tool obligation(s) on {}", dropped, reason)

        self._active_response_id: str | None = None
        self._audio_item_id: str | None = None
        self._item_base_played = 0
        # Never carry a dead session's fault into the next one's failure detail.
        self._last_error = None
        # Every call_id THIS session announced: a result finishing after a reconnect must
        # be dropped, since the new session rejects unknown call_ids. Cancelled responses
        # KEEP their entries: still answered, they just resume nothing.
        self._session_calls: set[str] = set()
        # Insertion-ordered so the size bound evicts the OLDEST rid; set.pop() is
        # hash-arbitrary and could evict the one rid whose late deltas still stream.
        self._cancelled_responses: dict[str, None] = {}
        self._response_done: set[str] = set()
        self._response_had_tools: dict[str, bool] = {}
        self._tools_pending: dict[str, set[str]] = {}
        self._call_to_response: dict[str, str] = {}
        self._fn_names: dict[str, str] = {}
        self._fn_args: dict[str, str] = {}
        # Stop-consume state is session-scoped: a suppress window or onset latch carried
        # across a reconnect would cancel the NEW session's first response at birth (a
        # dropped socket can reconnect well inside the 2 s window).
        self._onset_interrupting = False
        self._last_stop_consume = float("-inf")
        self._stop_suppress_until = 0.0
        # Barge-in latency clock (monotonic ms): set at server-VAD onset, consumed by
        # the next barge-in. Session-scoped, or a reconnect inherits a stale onset.
        self._speech_started_at: float | None = None
        # speech_started..speech_stopped: the only span in which uplink frames may
        # feed the turn deadman (see the sender loop). Session-scoped.
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
        # AEC before the ready-gate: the adaptive filter wants a continuous capture timeline
        # (and this drains due reference blocks even while frames are dropped). Loop-side is
        # fine: AEC3 measures ~0.05 ms per 10 ms frame pair.
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
                # Beta: WE must stop the response (see InterruptKind) or the model keeps
                # generating (and billing) audio nobody will hear.
                await self._cancel_active()
                return
            # GA (see InterruptKind): truncate only — a second response.cancel would race
            # the server's auto-cancel, unless interruptResponse is off (no auto-cancel).
            if not self._rt.interrupt_response:
                await self._cancel_active()
            else:
                self._mark_cancelled()
            item = self._audio_item_id
            if item is None:
                return  # nothing audible to truncate (already marked cancelled above)
            # One truncate per item: a second barge-in before the next output_item.added
            # would re-truncate the same item at audio_end_ms=0 (the sink was flushed, so
            # played_ms restarts), wiping the model's memory of audio the user DID hear.
            self._audio_item_id = None
            audio_end = max(0, played_ms - self._item_base_played)
            try:
                # A dead socket must not escape barge_in: the shell's post-barge-in
                # callback (abandon the in-flight delegation) must still run.
                await self._send({
                    "type": "conversation.item.truncate",
                    "item_id": item, "content_index": 0, "audio_end_ms": audio_end,
                })
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                self._log.debug("truncate failed: {}", exc)
        finally:
            # On the way OUT so the measured latency includes the shell's sink flush and the
            # frame send; bucketed by mechanism, since truncate and cancel differ structurally.
            self._record_barge_in()

    def _record_barge_in(self) -> None:
        # The shell routes EVERY onset here; only one that interrupted a live turn
        # (_onset_interrupting) is a barge-in sample. The stamp clears either way.
        stamp, self._speech_started_at = self._speech_started_at, None
        if stamp is None or not self._onset_interrupting:
            return
        self._metrics.observe(
            f"barge_in_ms.{self._profile.interrupt}",
            time.monotonic() * 1000.0 - stamp,
        )

    def _note_cancelled(self, rid: str) -> None:
        """Record a dead response so late deltas drop (the cloud analog of the local
        ``_rejected_base``); bounded, since the set lives for the whole session."""
        self._cancelled_responses[rid] = None
        while len(self._cancelled_responses) > 256:  # rids are only checked while recent
            del self._cancelled_responses[next(iter(self._cancelled_responses))]

    def _mark_cancelled(self) -> None:
        if self._active_response_id:
            self._note_cancelled(self._active_response_id)

    async def _cancel_active(self) -> None:
        """Send ``response.cancel`` for the live response, at most once. Shared by the beta
        barge-in path and the GA path with ``realtime.interruptResponse`` off."""
        rid = self._active_response_id
        if not rid or rid in self._cancelled_responses:
            return
        with suppress(Exception):
            await self._send({"type": "response.cancel"})
        self._note_cancelled(rid)

    async def _consume_stop(self, text: str) -> None:
        """A pure stop command (transcript-gated): the response answering it dies unspoken
        — cancelled now if live, at birth via the suppress window if not — and queued ack
        audio is flushed. Deliberately NO conversation.item.truncate: the tracked audio
        item may still be the PREVIOUS reply's (the ack often has no item yet), and
        truncating a fully-heard item corrupts the model's memory of it."""
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
            # Nothing live to cancel (or only a corpse awaiting its done): the response
            # answering THIS stop is still unborn — arm the at-birth suppression.
            self._stop_suppress_until = time.monotonic() + _STOP_SUPPRESS_S
        await self._sink.flush()
        # The tracked item dies with the flush: played_ms() restarts at 0, so a LATER
        # barge-in computing audio_end against the old base would truncate the
        # partially-heard pre-stop item at 0 — wiping the model's memory of audio the
        # user DID hear, the exact corruption skipping the truncate above avoids.
        # Cleared, the next barge-in finds no item and sends nothing: the model
        # over-remembers (the docstring's chosen direction) instead. An item announced
        # after this point cannot re-arm the slot: the stop's cancel/suppress marks its
        # response dead and _on_item_added drops non-live items.
        self._audio_item_id = None
        self._item_base_played = 0
        if self._turn is not VoiceState.IDLE:
            await self._set_turn(VoiceState.IDLE)

    def _clamp_tool_output(self, output: str) -> str:
        """Clamp an oversized tool output to protect the realtime context window; the
        marker keeps the model aware the tool ran, so it can ask for a smaller slice."""
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
        # call_id/size only: serializing the (unbounded on GA) payload for a usually
        # disabled debug line would double every tool result's JSON cost.
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
                # A CLEAN server close lands here, not in `except` (e.g. Qwen's per-session
                # turn cap); still a disconnect, so it takes the same teardown + backoff.
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - reconnect on any transport failure
                if self._closing:
                    break
                # A rejected HANDSHAKE with an auth status is a credential problem, not a
                # transport blip: fatal at once when the key has NEVER worked, but after a
                # healthy session one transient 401/403 (proxy blip, key rotation) gets one
                # ladder retry first. (.response.status_code = modern websockets
                # InvalidStatus; .status_code = legacy InvalidStatusCode.)
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
            # Only back-to-back FAST failures (bad key, wrong URL) walk the ladder to the
            # fatal rung; a session that stayed up proves the endpoint healthy.
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
        # The dead session's timers must die with it: a surviving watchdog would fire
        # (with real side effects now) minutes into the RECONNECTED session.
        self._cancel_watchdog()
        self._cancel_drain()
        self._reset_turn_state(reason="session_lost")
        # An anchor surviving the outage would measure the reconnected session's
        # first audio against a turn that no longer exists.
        self._metrics.turn_end()
        # Reset the coarse state too: the shell's half-duplex mic gate keys on SPEAKING, so
        # dropping while SPEAKING would wedge forever (mic gated -> no audio to the server
        # -> no speech_started to move off SPEAKING).
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
                        # A handler bug must not tear down a HEALTHY connection: a
                        # reconnect loses the server-side conversation state.
                        self._log.exception("event handler failed for {}", evt.get("type"))
            finally:
                # _ws must not outlive the socket: submit_tool_result has to see None and
                # drop the frame, not raise ConnectionClosed and skip its bookkeeping.
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
                    # A monologue longer than turn_timeout_s emits no server events until
                    # speech_stopped, so the watchdog armed at speech_started would fire
                    # mid-sentence. Only while the user is AUDIBLY speaking: the silence
                    # frames that keep flowing after speech_stopped (the state stays
                    # CAPTURING until response.created) must not feed the deadman, or a
                    # server that acks the speech and never answers wedges CAPTURING
                    # forever — the exact hang the speech_started arming exists to catch.
                    # Idle mic frames feeding it in THINKING would likewise mask a hang.
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
            # Deep-copy: session_extras lives on the PROFILES singleton, and _deep_merge
            # would alias its nested dicts into this payload.
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
            # Stop targeting latch, read BEFORE the CAPTURING transition overwrites it.
            # New speech also ends any pending created-response suppression: whatever
            # response follows answers the NEW utterance, not a consumed stop.
            self._onset_interrupting = self._turn in (
                VoiceState.THINKING, VoiceState.SPEAKING,
            )
            self._stop_suppress_until = 0.0
            await self._set_turn(VoiceState.CAPTURING)
            await self._emit(UserSpeechStarted())
        elif t == "input_audio_buffer.speech_stopped":
            # MEASUREMENT ONLY, no state change or event: the anchor every turn-latency
            # number is measured from (end of user speech); absent without server_vad,
            # and then turn latency simply goes unrecorded.
            self._metrics.turn_anchor()
            self._user_speaking = False  # uplink frames stop feeding the deadman here
            self._progress_t = time.monotonic()  # end of speech IS turn progress
        elif t == "response.created":
            rid = (evt.get("response") or {}).get("id")
            if time.monotonic() < self._stop_suppress_until:
                # The response answering a consumed stop: kill it at birth, before any
                # audio exists, and stay out of THINKING — silence is the acknowledgment.
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
            # Liveness-gated like every other delta: a cancelled response's argument stream
            # otherwise accumulates scratch nothing cleans — the cancel can race ahead of
            # output_item.added, so the cid never maps to a rid the discard path can reach.
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
        """Create-on-first-sight: a dialect may emit its first ``response.*`` event before
        (or without) ``response.created`` — the s2s reference server emits it lazily — and
        ``_is_live`` fails closed with no active id. Never adopts over a live, cancelled,
        or finished id."""
        rid = evt.get("response_id")
        if (
            rid
            and self._active_response_id is None
            and rid not in self._cancelled_responses
            and rid not in self._response_done
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
            # Fail CLOSED: the active id is cleared at turn end, so `None == None` would
            # let a rid-less straggler play audio or re-run a tool for a dead turn.
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
            # Truncate baseline: where this item's audio STARTS on the stream. played +
            # backlog, not played alone, else a continuation item added while the previous
            # tail is still buffered over-counts audio_end_ms past the item's real length
            # (GA rejects that); backlog_ms() over-counts by design, so audio_end
            # under-counts — the safe direction for the model's memory. Valid only against
            # played_ms() from the SAME sink stream generation: a republished stream
            # restarts played_ms() at 0 and voids the base.
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
        # TTFA, latched to the turn's first frame; recorded at ENQUEUE, so device playout
        # is excluded (VoiceMetrics marks snapshots `_enqueue_side`).
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
        # PRIMARY obligation registration: output_item.added usually got here first, but a
        # dialect may skip it entirely (the s2s reference server emits ONLY this event for
        # a tool call, its response.done carrying no output[] to mine), which would leave
        # the call invisible to the continuation logic. Set-based; the double-add is a no-op.
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
            # WE already declared this response dead (barge-in / watchdog), so the server's
            # status is moot. Falling through to the completed branch would fire a second
            # TurnDone + turn_end (wiping the interrupting utterance's fresh anchor) and
            # force CAPTURING -> IDLE mid-speech via the drain.
            self._discard_response_tools(rid)
            if rid == self._active_response_id:
                self._active_response_id = None
            return
        if status == "cancelled":
            if rid:
                self._note_cancelled(rid)
                self._discard_response_tools(rid)
                if rid == self._active_response_id:
                    self._active_response_id = None
            # SERVER-initiated cancel (turn_detected): the paired speech_started usually
            # follows and re-arms everything, but it may arrive AFTER this event (the s2s
            # reference server emits done-then-started) or never, wedging a gated-mic
            # session in SPEAKING forever. A later speech_started just re-arms again.
            if self._turn in (VoiceState.THINKING, VoiceState.SPEAKING):
                self._arm_watchdog()
            return
        if status in ("failed", "incomplete"):
            # A failed response can end with announced function calls whose arguments.done
            # never arrives, and those obligations would wedge _maybe_respond forever. Drop
            # them like a cancellation, but still end the turn cleanly.
            if rid:
                self._note_cancelled(rid)
                self._discard_response_tools(rid)
                if rid == self._active_response_id:
                    self._active_response_id = None
            # Providers may put the reason in a top-level `error` event just BEFORE the
            # failed done, leaving status_details bare (the s2s reference server does).
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
            self._response_done.add(rid)
        if rid and self._response_had_tools.get(rid):
            # Tool wait: the turn is still LIVE, so the response stays "active"; a barge-in
            # during the (possibly long) tool run must be able to mark it cancelled, or the
            # abandoned results re-trigger response.create over the user's new speech. The
            # continuation's response.created replaces the id.
            await self._maybe_respond(rid)
        else:
            if rid == self._active_response_id:
                self._active_response_id = None  # finished turn: nothing left to cancel
            if rid:
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
        # Re-anchor before the frame goes out: however the dialect resumes, the audio that
        # follows is continuation latency, not TTFA.
        self._metrics.turn_continuation()
        if self._needs_response_create_after_tools:
            self._log.debug("firing response.create for rid={}", rid)
            await self._send({"type": "response.create"})
        else:
            # Auto-continuing dialects resume on the function_call_output alone.
            self._log.debug("auto-continuing dialect; no response.create for rid={}", rid)
        # response.done cancelled the watchdog, so the create -> response.created gap is the
        # one window with no deadman: a continuation the provider never starts wedges the
        # session (mic gated while SPEAKING, so nothing can recover it).
        self._arm_watchdog()
        self._cleanup_response(rid)

    def _cleanup_response(self, rid: str) -> None:
        self._tools_pending.pop(rid, None)
        self._response_done.discard(rid)
        self._response_had_tools.pop(rid, None)

    def _discard_response_tools(self, rid: str) -> None:
        """Drop a cancelled response's tool bookkeeping so it can't leak. A call whose
        ``ToolCall`` already reached the shell still gets answered (``_session_calls``
        keeps it); it just triggers no new response, its rid mapping being gone."""
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
            # A device-layer failure (USB audio unplugged mid-playout) must not skip the
            # IDLE transition: the watchdog died at response.done and gated-mic SPEAKING
            # mutes the mic, so no speech_started could ever recover it.
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
            # DEADMAN, not a whole-turn cap: audio deltas push _progress_t forward, so a
            # long reply still streaming never trips it, while one that emits nothing at
            # all recovers after turn_timeout_s of true silence.
            await wait_for_stall(lambda: self._progress_t, self._rt.turn_timeout_s)
            self._log.warning("realtime turn watchdog fired (no progress); recovering")
            rid, self._active_response_id = self._active_response_id, None
            if rid:
                # Recover, not just report: stragglers drop via _is_live, a late tool
                # result cannot re-trigger the response, and the server stops generating.
                self._note_cancelled(rid)
                self._discard_response_tools(rid)
                with suppress(Exception):
                    await self._send({"type": "response.cancel"})
            self._metrics.turn_end()
            await self._emit(Error(message="realtime turn timed out", fatal=False))
            await self._set_turn(VoiceState.IDLE)
        except asyncio.CancelledError:
            raise
