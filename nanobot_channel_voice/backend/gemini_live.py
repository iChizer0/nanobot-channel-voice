"""GeminiLiveBackend: e2e speech-to-speech over the Gemini Live API (``BidiGenerateContent``).

Sibling of :mod:`.openai_realtime` on the shared :class:`~.transport.RealtimeTransport`;
this module is the Gemini wire (verified against the Live API reference, 2026-09-16):

* ``setup`` -> ``setupComplete``; audio up as ``realtimeInput.audio`` (16 kHz PCM); audio
  down as ``serverContent.modelTurn.parts[].inlineData`` (24 kHz PCM).
* No playback-aligned truncate: on ``interrupted`` the server keeps what it already sent,
  so ``barge_in`` is a no-op past the shell's flush.
* Tools auto-continue: ``toolCall.functionCalls`` -> ``toolResponse.functionResponses``,
  no ``response.create``. Every declaration is ``NON_BLOCKING`` (mandatory on the
  extended-thinking model; the base model then keeps talking while the tool runs) and
  the response carries ``scheduling`` INSIDE ``response``.
* ``turnComplete`` is NOT idle on the extended-thinking model: ``interactionStatus``
  (``IN_PROGRESS`` | ``IDLE``, on ``serverContent`` or the message root) is. A turn holds
  THINKING through an IN_PROGRESS completion (or while a tool the shell runs is
  outstanding); IDLE, or an unlabeled completion with nothing pending, drains.
* Proactive audio means a non-answer is a legitimate outcome: the deadman settles to
  IDLE without an Error when nothing was ever generated.
* ``sessionResumptionUpdate.newHandle`` rides every reconnect and un-park; ``goAway``
  is a clean close the ladder reconnects through.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time

from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.metrics import VoiceMetrics

from .audio_sink import AudioSink
from .base import (
    InputTranscript,
    OutputAudio,
    OutputTranscript,
    ToolCall,
    ToolDef,
    ToolStarted,
    TurnDone,
    VoiceState,
)
from .common import loggable_text
from .openai_realtime import _normalize_schema
from .transport import RealtimeTransport

DEFAULT_BASE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
DEFAULT_MODEL = "gemini-3.8-live"
DEFAULT_VOICE = "Kore"
INPUT_RATE = 16000
OUTPUT_RATE = 24000
# A resumption handle is valid for 2 h after the session's termination (Live API
# session management); ours lapses a minute earlier, since a rejected setup would walk
# the reconnect ladder — the overnight-parked device must start fresh, not die.
_RESUME_HANDLE_S = 2 * 3600.0 - 60.0
# JSON Schema keywords the function-declaration schema rejects (OpenAPI subset).
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {"additionalProperties", "$schema", "$defs", "$ref", "title", "examples", "default"}
)


def resolve_gemini_key(explicit: str | None) -> str | None:
    """``realtime.apiKey`` or the Google SDKs' own variables — never OPENAI_API_KEY."""
    return explicit or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def _gemini_schema(schema: dict) -> dict:
    """Function-declaration parameters: unions/combinators flattened (as for Qwen), then
    the keywords the OpenAPI subset rejects dropped at every level."""
    def strip(node):
        if isinstance(node, dict):
            return {k: strip(v) for k, v in node.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
        if isinstance(node, list):
            return [strip(v) for v in node]
        return node

    return strip(_normalize_schema(schema))


def _tool_to_wire(tool: ToolDef) -> dict:
    return {
        "name": tool.name,
        "description": tool.description,
        "parameters": _gemini_schema(tool.parameters),
        "behavior": "NON_BLOCKING",
    }


def _pcm_rate(mime: str | None, default: int) -> int:
    """``audio/pcm;rate=24000`` -> 24000."""
    for param in (mime or "").split(";")[1:]:
        key, _, value = param.strip().partition("=")
        if key == "rate" and value.isdigit():
            return int(value)
    return default


def _status_of(msg: dict) -> str | None:
    """``interactionStatus`` wherever this message carries it (root or serverContent)."""
    status = msg.get("interactionStatus")
    if status is None:
        status = (msg.get("serverContent") or {}).get("interactionStatus")
    return status if isinstance(status, str) else None


class GeminiLiveBackend(RealtimeTransport):
    def __init__(
        self,
        config: VoiceConfig,
        *,
        sink: AudioSink,
        metrics: VoiceMetrics | None = None,
        aec=None,
    ):
        super().__init__(config, sink=sink, metrics=metrics, aec=aec)
        self._model = config.realtime.model or DEFAULT_MODEL
        self._voice = config.realtime.voice or DEFAULT_VOICE
        # The delegated answer IS the reply the user waits for; a direct tool's result
        # can wait for the filler to finish.
        self._scheduling = (
            "INTERRUPT" if config.realtime.tool_mode == "supervisor" else "WHEN_IDLE"
        )
        self._log_transcripts = config.log_transcripts
        self._resume_handle: str | None = None
        # The handle's clock: its receipt, then the session's termination (what the
        # validity is documented from).
        self._handle_since = 0.0
        self._reset_turn_state()

    # ---- turn/session bookkeeping -------------------------------------------

    def _reset_turn_state(self, *, reason: str = "init") -> None:
        if reason == "session_lost":
            self._handle_since = time.monotonic()
        pending: set[str] = set(getattr(self, "_pending_calls", ()))
        dropped = self._metrics.calls_dropped(pending, reason)
        if dropped:
            self._log.warning("dropping {} unanswered tool obligation(s) on {}", dropped, reason)
        self._pending_calls: set[str] = set()   # announced, unanswered (this session)
        self._generating = False   # audio/text seen since the last turn boundary
        self._in_progress = False  # extended thinking: background work continues
        # The server cut the model off (its VAD, or our activityStart): that turn's
        # completion is not a turn end for the shell — the onset owns the state.
        self._interrupted = False
        # Manual turns: audio already on the wire when our activityStart lands is stale
        # (the shell flushed); dropped until the server's ``interrupted`` echo or the
        # activity's end, whichever first.
        self._dead_audio = False
        # An uncommitted activity end (blip / bare summon): the model may still answer
        # the empty activity. That answer dies unheard; the NEXT committed activity's
        # plays (WS ordering: an answer to activity N precedes N+1's, and N+1's start
        # interrupts it if still streaming).
        self._suppress_turn = False

    def _api_key(self) -> str:
        key = resolve_gemini_key(self._rt.api_key)
        if not key:
            raise RuntimeError(
                "no API key for realtime provider 'gemini' (set channels.voice.realtime."
                "apiKey or GEMINI_API_KEY)"
            )
        return key

    # ---- VoiceBackend -------------------------------------------------------

    async def barge_in(self, played_ms: int) -> None:
        # No truncate on this protocol: the server keeps what it already sent; the shell
        # already flushed the sink. Measure, and make sure the dead turn's tail drops.
        self._generating = False
        self._record_barge_in("interrupt")

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        if call_id not in self._pending_calls:
            # Cancelled by the server, or issued by a session that is gone.
            self._log.debug("dropping tool result for call {} (not pending)", call_id)
            return
        self._pending_calls.discard(call_id)
        try:
            result = json.loads(output)  # a JSON tool result rides as structure
        except (ValueError, TypeError):
            result = output
        if result is None:
            result = output  # "null" is still an answer
        await self._send({
            "toolResponse": {
                "functionResponses": [{
                    "id": call_id,
                    "response": {"result": result, "scheduling": self._scheduling},
                }],
            },
        })
        if self._pending_calls:
            return  # siblings still running: their budget, not the deadman's
        # The continuation is the model's; what follows is continuation latency.
        self._metrics.turn_continuation()
        self._arm_watchdog()

    # ---- ManualTurnBackend wire (gated uplink) ------------------------------

    async def _activity_begin_wire(self) -> None:
        if self._generating:
            self._dead_audio = True  # in flight until the server's interrupted echo
        await self._send({"realtimeInput": {"activityStart": {}}})

    async def _activity_end_wire(self, *, commit: bool) -> None:
        # No discard on this protocol: close the activity either way, and let an answer
        # to an uncommitted one die unheard.
        self._dead_audio = False
        self._suppress_turn = not commit
        await self._send({"realtimeInput": {"activityEnd": {}}})

    # ---- wire ---------------------------------------------------------------

    def _connect_args(self) -> tuple[str, dict[str, str]]:
        base = self._rt.base_url or DEFAULT_BASE_URL
        sep = "&" if "?" in base else "?"
        return f"{base}{sep}key={self._api_key()}", {}

    def _hello_payload(self) -> dict:
        if (
            self._resume_handle
            and self._handle_since
            and time.monotonic() - self._handle_since > _RESUME_HANDLE_S
        ):
            self._log.info("gemini: resumption handle expired; starting a fresh session")
            self._resume_handle = None
        model = self._model if self._model.startswith("models/") else f"models/{self._model}"
        generation: dict = {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": self._voice}},
            },
        }
        if self._rt.thinking_level:
            generation["thinkingConfig"] = {"thinkingLevel": self._rt.thinking_level.upper()}
        setup: dict = {
            "model": model,
            "generationConfig": generation,
            # Input transcription is one switch here (no model); it also feeds the stop
            # rule's transcript. Output transcription feeds the gate's echo veto.
            "outputAudioTranscription": {},
            # Resumption + compression: the socket lives ~10 min and an audio-only
            # session 15 min without them; both are how a long conversation survives.
            "sessionResumption": (
                {"handle": self._resume_handle} if self._resume_handle else {}
            ),
            "contextWindowCompression": {"slidingWindow": {}},
        }
        if self._instructions:
            setup["systemInstruction"] = {"parts": [{"text": self._instructions}]}
        if self._tools:
            setup["tools"] = [{"functionDeclarations": [_tool_to_wire(t) for t in self._tools]}]
        if self._rt.input_transcription_model:
            setup["inputAudioTranscription"] = {}
        if self._rt.proactive_audio:
            setup["proactivity"] = {"proactiveAudio": True}
        if self._manual:
            setup["realtimeInputConfig"] = {
                "automaticActivityDetection": {"disabled": True},
                "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
                "turnCoverage": "TURN_INCLUDES_ONLY_ACTIVITY",
            }
        payload = {"setup": setup}
        self._log.opt(lazy=True).debug(
            "setup payload: {}", lambda: json.dumps(payload, ensure_ascii=False)
        )
        return payload

    def _audio_frame(self, pcm: bytes) -> dict:
        return {
            "realtimeInput": {
                "audio": {
                    "mimeType": f"audio/pcm;rate={INPUT_RATE}",
                    "data": base64.b64encode(pcm).decode("ascii"),
                },
            },
        }

    # ---- event mapping ------------------------------------------------------

    async def _handle_event(self, msg: dict) -> None:
        if "setupComplete" in msg:
            self._ready.set()
            self._ever_ready = True
            self._auth_fails = 0
        elif "serverContent" in msg:
            await self._on_server_content(msg["serverContent"] or {}, _status_of(msg))
        elif "toolCall" in msg:
            await self._on_tool_call(msg["toolCall"] or {}, _status_of(msg))
        elif "toolCallCancellation" in msg:
            self._on_tool_cancel(msg["toolCallCancellation"] or {})
        elif "sessionResumptionUpdate" in msg:
            upd = msg["sessionResumptionUpdate"] or {}
            if upd.get("resumable") and upd.get("newHandle"):
                self._resume_handle = upd["newHandle"]
                self._handle_since = time.monotonic()
        elif "goAway" in msg:
            # The server closes shortly; the ladder reconnects with the handle.
            self._log.info(
                "gemini: goAway ({} left); reconnecting with the resumption handle",
                (msg["goAway"] or {}).get("timeLeft", "?"),
            )
        elif "usageMetadata" in msg:
            self._log.debug("gemini usage: {}", msg["usageMetadata"])
        elif "error" in msg:
            err = msg["error"] or {}
            self._log.warning("gemini error: {}", err.get("message") or err)

    async def _on_server_content(self, sc: dict, status: str | None) -> None:
        if sc.get("interrupted"):
            await self._on_interrupted()
        if status == "IN_PROGRESS":
            self._in_progress = True
        transcript = (sc.get("outputTranscription") or {}).get("text")
        if transcript and not self._suppress_turn:
            self._progress_t = time.monotonic()
            await self._emit(OutputTranscript(transcript))
        heard = (sc.get("inputTranscription") or {}).get("text")
        if heard:
            self._log.debug("user: {}", loggable_text(heard, self._log_transcripts))
            await self._emit(InputTranscript(heard))
        for part in (sc.get("modelTurn") or {}).get("parts") or []:
            blob = part.get("inlineData") or {}
            if blob.get("data"):
                await self._on_audio(blob)
        if sc.get("turnComplete"):
            await self._on_turn_complete(status)

    async def _on_audio(self, blob: dict) -> None:
        if self._suppress_turn or self._dead_audio:
            return  # answering an activity nobody committed / cut off already
        try:
            pcm = base64.b64decode(blob["data"])
        except (ValueError, TypeError):
            return
        if not self._generating:
            self._generating = True
            self._metrics.turn_thinking()
        if self._turn is not VoiceState.SPEAKING:
            await self._set_turn(VoiceState.SPEAKING)
        self._progress_t = time.monotonic()  # feed the deadman: the turn is alive
        self._metrics.turn_first_audio()  # latched to the turn's first frame
        await self._emit(OutputAudio(
            epoch=self._sink.epoch, pcm=pcm,
            rate=_pcm_rate(blob.get("mimeType"), OUTPUT_RATE),
        ))

    async def _on_turn_complete(self, status: str | None) -> None:
        self._generating = False
        if self._interrupted:
            # The cut-off turn's end: the shell already flushed and the onset owns the
            # state; a drain here would flip CAPTURING -> IDLE mid-speech and TurnDone
            # would count a turn nobody heard out.
            self._interrupted = False
            return
        if self._suppress_turn:
            # The unheard answer to an uncommitted activity ended; the flag lives on
            # until the next committed activity (nothing else is owed an answer).
            self._cancel_watchdog()
            if self._turn is not VoiceState.IDLE:
                await self._set_turn(VoiceState.IDLE)
            return
        if status == "IN_PROGRESS" or self._pending_calls:
            # The filler is spoken, the work goes on (extended thinking's background
            # reasoning, or a NON_BLOCKING tool the shell is still running). Hold
            # THINKING - the gate's attention and the park timer key on IDLE - and let
            # playback drain. An unlabeled completion with nothing pending is DONE: a
            # missed label then costs a benign extra turn, never a stuck THINKING.
            if self._pending_calls:
                self._cancel_watchdog()  # the shell's tool task has its own budget
            else:
                self._arm_watchdog()  # background reasoning: settled silently if quiet
            self._start_hold_thinking()
            return
        self._in_progress = False
        self._cancel_watchdog()
        self._metrics.turn_end()
        if not self._pending_calls:
            await self._emit(TurnDone())
        self._start_drain()

    def _start_hold_thinking(self) -> None:
        """Drain the spoken filler, then settle to THINKING instead of IDLE."""
        self._cancel_drain()
        self._drain_task = asyncio.create_task(self._drain_to_thinking())

    async def _drain_to_thinking(self) -> None:
        try:
            await self._sink.drain_stream()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - never strand SPEAKING
            self._log.warning("drain failed ({}); holding THINKING", exc)
        if self._turn is VoiceState.SPEAKING:  # not a CAPTURING the next onset owns
            await self._set_turn(VoiceState.THINKING)

    async def _on_interrupted(self) -> None:
        # Server-side VAD (or our activityStart) cut the model off. WS ordering: what
        # follows on the wire is new generation, never the dead turn's tail.
        self._interrupted = self._generating
        self._generating = False
        self._in_progress = False
        self._dead_audio = False
        self._cancel_drain()
        if self._manual:
            # begin_activity already emitted the onset; this is the server's echo of it.
            return
        await self._on_speech_started()
        # No offset event exists on this protocol: left "speaking", idle frames would
        # feed the deadman forever and an unanswered interruption never settles.
        self._user_speaking = False

    async def _on_tool_call(self, call: dict, status: str | None) -> None:
        if status == "IN_PROGRESS":
            self._in_progress = True
        for fc in call.get("functionCalls") or []:
            cid = fc.get("id")
            name = fc.get("name") or ""
            if not cid:
                continue
            self._progress_t = time.monotonic()
            self._pending_calls.add(cid)
            self._metrics.call_seen(cid, name)
            await self._emit(ToolStarted(name or None, call_id=cid))
            args = fc.get("args")
            arguments = json.dumps(args if args is not None else {}, ensure_ascii=False)
            self._log.debug("tool call ready: {}({})", name, arguments)
            self._metrics.call_dispatched(cid, self._sink.epoch)
            await self._emit(ToolCall(call_id=cid, name=name, arguments=arguments))
        if self._pending_calls:
            self._cancel_watchdog()  # the shell's tool task has its own budget

    def _on_tool_cancel(self, cancel: dict) -> None:
        # The shell's tool task runs on; its result then finds no pending call and drops.
        ids = {i for i in cancel.get("ids") or [] if isinstance(i, str)}
        self._metrics.calls_abandoned(ids & self._pending_calls)
        self._pending_calls -= ids

    # ---- transport hooks ----------------------------------------------------

    async def _watchdog_recover(self) -> str | None:
        if self._generating:
            # Audio stalled mid-stream: a real fault.
            self._generating = False
            self._in_progress = False
            return "realtime turn timed out"
        # Quiet: proactive audio declined to answer (or a bare summon), or background
        # reasoning is simply taking longer than the deadman. Not a fault - settle to
        # IDLE and listen; a late answer still plays from there.
        self._metrics.count("turn_background_settled" if self._in_progress else "turn_unanswered")
        return None


__all__ = ["GeminiLiveBackend", "resolve_gemini_key", "INPUT_RATE", "OUTPUT_RATE"]
