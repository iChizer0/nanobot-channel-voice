"""The voice channel: thin glue mapping nanobot's channel contract onto a ``VoiceShell``
around a swappable ``VoiceBackend`` (``local`` or a realtime provider). Capture -> STT
publishes via ``_handle_message``, so allow-list and session routing work as for any
channel; barge-in publishes the priority ``/stop``, then the new utterance
(cancel-then-send). ``send_delta`` text is spoken chunk-by-chunk, ``send`` only genuine
final messages (see :func:`_speakable`)."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import deque
from contextlib import suppress
from typing import Any

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.runtime_context import RuntimeContextBlock

from nanobot_channel_voice.aio import cancel_and_wait
from nanobot_channel_voice.audio import make_audio
from nanobot_channel_voice.audio.pcm import pcm_ms, wav_duration_ms
from nanobot_channel_voice.backend import gemini_live
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import NOTICE_MARK, AbandonedResult, ToolDef, VoiceState
from nanobot_channel_voice.backend.common import loggable_text
from nanobot_channel_voice.backend.gated import GatedUplink
from nanobot_channel_voice.backend.gemini_live import GeminiLiveBackend, resolve_gemini_key
from nanobot_channel_voice.backend.local import LocalBackend
from nanobot_channel_voice.backend.openai_realtime import RealtimeBackend, _load_connect
from nanobot_channel_voice.backend.profiles import backend_kind, resolve_profile
from nanobot_channel_voice.config import (
    VoiceConfig,
    consume_import_json,
    resolve_openai_key,
    transcription_gap,
    unified_session,
)
from nanobot_channel_voice.context_tool import (
    VoiceContextBridge,
    register_bridge,
    tool_created,
    unregister_bridge,
)
from nanobot_channel_voice.metrics import VoiceMetrics
from nanobot_channel_voice.shell import VoiceShell
from nanobot_channel_voice.streamid import TURN_META, base_of, unique_token
from nanobot_channel_voice.stt import SttAdapter, make_stt, transcribe_chunked, write_temp_wav
from nanobot_channel_voice.telemetry import VoiceTracer
from nanobot_channel_voice.tts import TtsAdapter, make_tts
from nanobot_channel_voice.tts.base import CALIBRATION_TEXT, startup_text
from nanobot_channel_voice.vad import EnergyVad, make_turn_analyzer, make_vad
from nanobot_channel_voice.wake import make_wake_detector

_DEFAULT_PERSONA = (
    "You are a helpful, concise voice assistant. Keep replies short and conversational."
)

# Direct mode: the filler is what the user hears across a tool round-trip, else the line
# goes dead. Appended only when tools are declared.
_DIRECT_RULES = (
    "Before a tool call that will keep the user waiting, say a brief neutral filler "
    "in the user's language, such as \"One moment.\" or \"Let me check.\" (never "
    "implying success or failure), then call it with no further speech. Skip the "
    "filler when you expect the answer immediately. The reply that delivers the "
    "answer is pure answer: never open it with wait phrases or progress narration."
)

# Silence-is-the-ack, model-side half: enforcement is backend._consume_stop's
# transcript-gated response.cancel, which needs input transcription and can lose the race
# to a fast ack. Appended in EVERY mode; the second sentence is the only cover for a
# stopped tool's answer where no transcript abandons the work.
_STOP_RULE = (
    "If the user only tells you to stop, be quiet, or wait, do not answer — "
    "produce no speech at all. A request they stopped stays stopped: do not read out "
    "its result when it arrives."
)

# How the model voices what the agent sends on its own (backend announce). Every mode.
_NOTICE_RULE = (
    f"A user message that starts with {NOTICE_MARK} was not said by the user: it is a "
    "message for them (a reminder, a report, a message from another channel). Say it "
    "to them as written, then stop."
)

# Supervisor mode (Responder-Thinker): the realtime model owns the conversational surface
# and delegates reasoning/tool work to nanobot; the filler masks the round-trip.
_SUPERVISOR_RULES = (
    "Handle greetings, small talk, and clarifying questions yourself. For ANYTHING "
    "that needs a fact you don't already know, an action, a lookup, or multi-step "
    "work, you MUST delegate: FIRST say a brief neutral filler in the user's "
    "language, such as \"One moment.\" or \"Let me check.\" (never implying "
    "success or failure), THEN call "
    "the ask_nanobot tool with the user's request. When it returns, read the answer "
    "aloud naturally and concisely as if it were your own, never mention the tool "
    "or that you delegated."
)

# Supervisor's only declared tool: persona + this schema is the whole realtime context,
# MCP/skills/memory stay in nanobot.
_SUPERVISOR_TOOL = ToolDef(
    name="ask_nanobot",
    description=(
        "Delegate the user's request to the nanobot agent, which can reason over "
        "multiple steps, use tools, and access memory and files. Call this whenever "
        "the user wants an action taken or a fact you do not already know. Always "
        "speak a brief neutral filler to the user BEFORE calling this. A new call "
        "replaces one still running: include anything from that request the user "
        "still wants."
    ),
    parameters={
        "type": "object",
        "properties": {
            "request": {
                "type": "string",
                "description": "The user's request, phrased in full as a standalone "
                "instruction the agent can act on without the spoken history.",
            },
            "relevant_context": {
                "type": "string",
                "description": "Context from the spoken conversation the agent needs "
                "but wouldn't otherwise have (names, prior answers, preferences). "
                "Omit if none.",
            },
        },
        "required": ["request"],
    },
)

# The ask_nanobot result for work the user stopped, or a newer request replaced: satisfies
# the function call, and resumes nothing (see AbandonedResult).
_DELEGATION_STOPPED = AbandonedResult("(stopped by the user)")
_DELEGATION_REPLACED = AbandonedResult("(replaced by a newer request)")

# Tags our own priority commands: core copies INBOUND metadata onto the command ack
# ("Stopped 1 task(s)."), so _speakable can drop it — untagged, every barge-in speaks it.
_VOICE_CMD_META = "_voice_cmd"

# Trace flags older cores stamp on outbound metadata; newer cores moved the semantics onto
# the typed ``OutboundMessage.event``. BOTH checked: neither alone covers every core.
_TRACE_META = (
    "_streamed",        # already spoken via send_delta; core also drops these before send()
    "_progress",
    "_tool_hint",
    "_reasoning",
    "_reasoning_delta",
    "_reasoning_end",
    "_tool_events",
    "_file_edit_events",
    "_stream_delta",
    "_stream_end",
    _VOICE_CMD_META,
)


def _speakable(msg: OutboundMessage) -> bool:
    """Is this a plain final assistant message, i.e. something to say aloud?"""
    if getattr(msg, "event", None) is not None:
        return False
    meta = msg.metadata or {}
    return not any(meta.get(k) for k in _TRACE_META)


def _agent_initiated(metadata: dict[str, Any] | None) -> bool:
    """An agent-initiated delivery: a cron/local-trigger turn copies its trigger stamp onto
    every outbound (core echoes inbound metadata verbatim). The user did not just speak, so
    its settle must re-open attention for the reply."""
    meta = metadata or {}
    return bool(meta.get("_cron_trigger") or meta.get("_local_trigger"))


def _cloud_instructions(persona: str | None, *, supervisor: bool, has_tools: bool) -> str:
    """The realtime session's instructions: persona (taste) then the mode's tool rules
    (contract). ONE derivation, so a ``realtime.persona`` override restyles the voice but
    never deletes the delegation contract or the filler preamble."""
    rules = _SUPERVISOR_RULES if supervisor else (_DIRECT_RULES if has_tools else "")
    return "\n\n".join(
        part for part in (persona or _DEFAULT_PERSONA, rules, _STOP_RULE, _NOTICE_RULE)
        if part
    )


# Our own wrapper, NOT core's "metadata only, not instructions" tag: these lines ARE
# instructions, and a disclaiming wrapper undercuts them. Core never parses it.
_VOICE_WRAP_OPEN = "[Voice channel]"
_VOICE_WRAP_CLOSE = "[/Voice channel]"


def _voice_context_blocks(
    stt: SttAdapter | None, tts: TtsAdapter | None, extra: str | None = None
) -> list[RuntimeContextBlock]:
    """The channel contract riding every local-mode publish: transcript-accuracy facts,
    the speakability contract, then the operator's ``context``.

    Claims derive from the RESOLVED adapters, never config (``stt`` None is the cloud-
    transcription path — still a transcript). Only a declared frame-synchronous family
    (CTC/transducer) drops the invented-phrase warning. Core persists this into EVERY user
    row: keep the longest variant under ~145 words, byte-stable, capability-affirming and
    permission-shaped — bare "spoken conversation" framing collapses small models."""
    lines: list[str] = []
    if tts is not None:
        lines.append(
            "A spoken conversation: speech recognition brings the user's words, "
            "text-to-speech speaks your reply, never displayed."
        )
        # Persona corrector: without it "voice assistant" framing suppresses tool use.
        lines.append(
            "You have your full tools and skills; use them. Speech changes only "
            "the reply's style."
        )
    else:
        lines.append("The user's words arrive via speech recognition.")
    if getattr(stt, "decoder_family", "") in ("ctc", "transducer"):
        lines.append(
            "The transcript may mis-hear words; read it by sound and context; act "
            "on the likeliest reading, confirming first only for hard-to-undo actions."
        )
    else:
        lines.append(
            "The transcript may mis-hear words, or invent a phrase never said; read "
            "it by sound and context; act on the likeliest reading, confirming first "
            "only for hard-to-undo actions."
        )
    if tts is not None:
        lines.append(
            "Write plain prose for the ear: no markdown, code, URLs, or emoji."
        )
        langs = getattr(tts, "spoken_languages", None)  # bilingual router
        lang = getattr(tts, "spoken_language", None)
        if langs:
            named = " and ".join(f"'{code}'" for code in langs)
            lines.append(
                f"The voice pronounces ISO 639-1 {named}; reply in whichever the "
                "user speaks; mixing is fine; other scripts are dropped or voiced "
                "as noise."
            )
        elif lang:
            lines.append(
                f"The voice pronounces only ISO 639-1 '{lang}'; reply in '{lang}' "
                "only — other scripts are dropped or voiced as noise."
            )
        lines.append(
            # The backend detects this status line (agent_prologue) to defer the filler.
            "Thinking aloud briefly is fine. Before a slow tool call, say one short "
            "sentence about what you are doing; keep the answer for after the "
            "results, with no wait-phrases (\"One moment\")."
        )
        # A plain answer ENDS the turn in core, so "I will keep trying" is itself a
        # give-up: the one prompt-side counterweight (goal.phrases is the enforced one).
        lines.append(
            "If a step fails, try another way, and always say how it ended."
        )
    if extra and extra.strip():
        lines.append(extra.strip())
    content = "\n".join((_VOICE_WRAP_OPEN, *lines, _VOICE_WRAP_CLOSE))
    return [RuntimeContextBlock(source="voice", content=content)]


# Stamped on a delegated ask_nanobot request; core echoes inbound metadata onto every delta,
# end and final of the turn it opens, so an exact match is the delegation's identity (no
# stamp: a cron fire or message-tool send; a stale one: a /stop-ped predecessor's straggler).
_DELEGATION_META = "_voice_delegation"


class _ReplyCollector:
    """Collects one nanobot turn's reply off the bus: a supervisor delegation's, or (cloud)
    one the agent started itself. Streaming ON: deltas accumulate and an end that closes a
    segment WITH content resolves (the turn's final never reaches a channel: core drops
    it). Streaming OFF, or a last segment that streamed nothing: the regular ``send``
    resolves, joined behind what streamed."""

    def __init__(self, metrics: VoiceMetrics, *, timed: bool = True) -> None:
        self._future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._parts: list[str] = []
        self._segment = False  # content since the last boundary
        self._metrics = metrics
        self._started_at = time.monotonic()
        self._first_token = not timed  # only a delegation's first token is a latency
        self.token = unique_token()
        self.foreign = 0  # deliveries into our chat that were not ours, while we waited

    def _mark_first_token(self) -> None:
        if self._first_token:
            return
        self._first_token = True
        self._metrics.observe(
            "delegation_first_token_ms", (time.monotonic() - self._started_at) * 1000.0
        )

    def add(self, delta: str) -> None:
        if delta:
            self._mark_first_token()
            self._parts.append(delta)
            self._segment = True

    def note_boundary(self) -> None:
        """A tool-boundary segment break: separates the reply's parts WITHOUT latching the
        first-token clock — the separator is channel-fabricated, and latching would
        under-report TTFT for exactly the tool-first delegations supervisor mode is for."""
        self._parts.append("\n")
        self._segment = False

    def finish(self, fallback: str = "") -> None:
        """A non-resuming end is terminal only when its segment carried content: core fires
        one mid-turn on its blank-response retry too (resolving there drops the real answer)."""
        self.add(fallback)
        if self._segment:
            self._resolve(self._text())

    def set_final(self, text: str) -> None:
        """The regular final: the whole reply, or what a last segment that streamed nothing
        (blank retries) ended on; either way it is also first token."""
        self._mark_first_token()
        self._resolve("\n".join(part for part in (self._text(), text.strip()) if part))

    def abandon(self, text: str) -> None:
        """Release with no answer (stopped or replaced): a delta landing in the tick before
        the slot clears must not be timed as one."""
        self._first_token = True
        self._resolve(text)

    @property
    def resolved(self) -> bool:
        return self._future.done()

    @property
    def reply(self) -> str:
        return self._future.result() if self._future.done() else ""

    def _text(self) -> str:
        return "".join(self._parts).strip()

    def _resolve(self, text: str) -> None:
        if not self._future.done():
            self._future.set_result(text)

    async def result(self) -> str:
        return await self._future


def _delegation_reply(pending: _ReplyCollector | None, metadata: dict[str, Any]) -> bool:
    """Whether a delivery is the pending delegation's reply; anything else into the chat
    meanwhile counts as foreign. A trigger turn echoes the stamp of the turn that created
    it, so a trigger-stamped delivery is never the reply."""
    if pending is None:
        return False
    if metadata.get(_DELEGATION_META) == pending.token and not _agent_initiated(metadata):
        return True
    pending.foreign += 1
    return False


def _straggler(metadata: dict[str, Any]) -> bool:
    """A delegation's traffic after it was answered, stopped or replaced (a trigger turn
    carries its creator's stamp, see _delegation_reply)."""
    return metadata.get(_DELEGATION_META) is not None and not _agent_initiated(metadata)


def _collect(collector: _ReplyCollector, delta: str, *, stream_end: bool, resuming: bool) -> None:
    """One streamed piece of a reply. A resuming end is only a tool boundary: resolving
    there would truncate to the pre-tool status line."""
    if stream_end:
        if resuming:
            collector.note_boundary()
        else:
            collector.finish(fallback=delta or "")
    else:
        collector.add(delta or "")


# One model load at a time, process-wide: a restart waits for the load it cancelled
# (two model stacks resident at once exhaust the NPU / RAM).
_LOAD_LOCK = threading.Lock()


class _Load:
    __slots__ = ("cancelled",)

    def __init__(self) -> None:
        self.cancelled = False


def _run_load(load: _Load, fn, args: tuple):
    with _LOAD_LOCK:
        result = fn(*args)
        if load.cancelled:  # start() gave up: nothing will own this
            _release(result)
            return None
        return result


def _release(result) -> None:
    for item in result if isinstance(result, tuple) else (result,):
        release = getattr(item, "release", None)
        if callable(release):
            with suppress(Exception):
                release()


def _release_late(fut: asyncio.Future) -> None:
    if not fut.cancelled() and fut.exception() is None:
        _release(fut.result())


class VoiceChannel(BaseChannel):
    name = "voice"
    display_name = "Voice"
    # Defaults only (the ChannelManager overwrites all three). Progress/tool-event traffic
    # is WANTED: it feeds the deadman's liveness tap in send(), never spoken.
    send_progress = True
    send_tool_hints = True
    show_reasoning = False
    # With the gateway injected, cloud tool calls route through nanobot's ToolRegistry.
    wants_tool_gateway = True

    def __init__(self, config: Any, bus: MessageBus, *, tool_gateway: Any = None):
        if isinstance(config, dict):
            config = VoiceConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: VoiceConfig = config
        self._tool_gateway = tool_gateway  # an AgentLoop, or None (persona-only)
        self._shell: VoiceShell | None = None
        self._backend: LocalBackend | RealtimeBackend | GeminiLiveBackend | GatedUplink | None = None
        self._stop_event: asyncio.Event | None = None
        self._stt: SttAdapter | None = None
        self._stt_server = None             # stt.serve: local /v1/audio/transcriptions
        self._tts_adapter = None            # local mode only; kept for warmup
        # Local mode only (cloud speaks under its own persona). Delivered via the context
        # bridge, NEVER inbound metadata: non-JSON metadata corrupts every tool that
        # snapshots it (cron's origin_metadata).
        self._voice_context: list[RuntimeContextBlock] = []
        self._context_bridge: VoiceContextBridge | None = None
        self._warmup_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None  # debug.metricsIntervalS reporter
        # Supervisor mode only: the in-flight ask_nanobot delegation the bus glue collects.
        # One slot — a bus reply can't be correlated to a concurrent delegation, so the lock
        # serializes them.
        self._pending_delegation: _ReplyCollector | None = None
        # Cloud: a streamed reply of a turn the agent started itself, and its stream base.
        self._notice: _ReplyCollector | None = None
        self._notice_base: str | None = None
        self._delegation_lock = asyncio.Lock()
        self._asked_turn: str | None = None  # the model turn of the newest delegation
        self._cloud_stops = 0  # every consumed stop that ended tool work (_on_cloud_abandon)
        # Local mode: killed-turn tokens whose core re-run was /stop-ped (once each).
        self._stopped_reruns: deque[str] = deque(maxlen=16)
        # One per session, shared with backend and shell: segments join on call_id.
        self._metrics = VoiceMetrics()
        self._tracer = VoiceTracer(self.config.telemetry)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return VoiceConfig().model_dump(by_alias=True)

    # ---- lifecycle ----------------------------------------------------------

    def start_error_message(self, error: Exception) -> str | None:
        """The WebUI's "Failed" box: core shows this text, else "Check gateway logs".
        start() words its own refusals for the operator (a missing key or extra, weights,
        a device, a port) as RuntimeError/OSError; anything else is a bug, and the
        generic pointer to the logs is the honest message for it."""
        if isinstance(error, (RuntimeError, OSError)):
            return " ".join(str(error).split()) or None
        return None

    async def start(self) -> None:
        self._running = True
        self._stop_event = asyncio.Event()
        if self.config.import_json:
            # A WebUI paste pending in config.json: expand it into real section keys and
            # delete the blob. self.config already carries the merged values, so a failed
            # rewrite only defers the blob; it can never yield a wrong runtime config.
            try:
                imported = await asyncio.to_thread(consume_import_json)
            except Exception as exc:  # noqa: BLE001 - config file may have changed underneath
                self.logger.warning(
                    "voice config import: importJson not expanded into config.json ({}); "
                    "it will be retried next start", exc,
                )
            else:
                if imported:
                    self.logger.info(
                        "voice config import: merged {} top-level keys into channels.voice "
                        "and removed importJson from config.json", imported,
                    )
        shell: VoiceShell | None = None
        server = None
        started = False
        try:
            kind = backend_kind(self.config.backend)
            if kind in ("openai_dialect", "gemini"):
                self._stt = None  # the provider does ASR; never load on-device models
                shell, instructions, tools = await self._build_cloud(kind)
            elif kind == "local":
                # Off the loop: a dozen ORT/RKNN session loads inline would freeze the
                # gateway (every other channel, cron, the WebUI) for the whole build.
                # Nothing here binds to a loop — Queue/Event bind lazily on first use.
                self._stt = await self._load(make_stt, self.config.stt)
                self._warn_if_transcription_unconfigured()
                await self._warn_if_unified_session()
                shell, backend, tts, blocks = await self._load(self._build_local)
                instructions, tools = "", []
                if self._running:  # a cancelled start() must not register a dead bridge
                    self._backend, self._tts_adapter, self._voice_context = backend, tts, blocks
                    self._context_bridge = register_bridge(self.name, self.config.chat_id, blocks)
                    self._announce_context()
            else:
                # Refuse loudly: falling through to local would run the wrong brain.
                raise RuntimeError(f"voice backend kind '{kind}' is not implemented")
            if not self._running:
                return  # stop() raced the build
            self.logger.info(
                "voice channel starting (backend={}, capture={}, playback={})",
                self.config.backend,
                self.config.audio.capture_device, self.config.audio.playback_device,
            )
            await shell.start(instructions=instructions, tools=tools)
            if not self._running:
                return  # stop() landed mid-start, before _shell was published
            self._shell = shell
            # A serve endpoint that cannot bind refuses loudly (WebUI dictation would be
            # silently broken) and takes the shell down.
            server = await self._start_stt_server()
            if not self._running:
                return  # stop() raced the bind: the handle was not published yet
            started = True
        finally:
            if not started:
                # Raised or raced: a never-started backend must not stay registered (speak_final
                # would queue into a worker that never runs), and a half-started shell holds
                # devices (shell.start's first await already spawned arecord).
                self._running = False
                self._shell = None
                self._stt_server = None
                self._backend = None
                self._drop_bridge()
                if server is not None:
                    with suppress(Exception):
                        await server.stop()
                if shell is not None:
                    with suppress(Exception):
                        await shell.stop()
                self._release_stt()
        # Off the critical path: the first turn then pays no cold start (ORT/RKNN/TRT).
        self._warmup_task = asyncio.create_task(self._warmup())
        if self.config.debug.metrics_interval_s:
            self._metrics_task = asyncio.create_task(
                self._metrics_reporter(self.config.debug.metrics_interval_s)
            )
        await self._stop_event.wait()
        self.logger.info("voice channel stopped")

    def _build_local(self) -> tuple[VoiceShell, LocalBackend, TtsAdapter | None, list]:
        """Pure: the thread may outlive a cancelled start(), so it touches neither this
        instance nor the global context bridge; start() publishes on the loop."""
        # The adapter's window, not config: a whisper export may override chunkLength.
        window = None if self._stt is None else self._stt.max_decode_ms
        if window is not None and self.config.vad.max_utterance_ms > window:
            # Harmless (decoded in pieces), but name the seam once; the whisper defaults
            # land here: 30 s cap vs a 20 s export.
            self.logger.info(
                "stt decode window ({:.0f}s) is under vad.maxUtteranceMs ({}); longer "
                "utterances are decoded in window-sized pieces cut at the quietest gap",
                window / 1000, self.config.vad.max_utterance_ms,
            )
        capture, sink_dev = make_audio(self.config.audio)
        vad = make_vad(self.config.vad, self.config.audio.sample_rate, self.config.audio.frame_ms)
        turn_analyzer = make_turn_analyzer(
            self.config.vad, self.config.audio.sample_rate, self.config.audio.frame_ms
        )
        wake_detector = make_wake_detector(
            self.config.wake, self.config.audio.sample_rate, self.config.audio.frame_ms
        )
        tts = make_tts(self.config.tts)
        if tts is not None:
            # The engine that actually LOADED: a failed build degrades to the system voice
            # behind one warning, and sounds like mispronunciation rather than a swap.
            langs = getattr(tts, "spoken_languages", None) or (
                getattr(tts, "spoken_language", None),
            )
            self.logger.info(
                "voice tts resolved: {} (configured '{}', {} Hz, {})",
                type(tts).__name__, self.config.tts.provider,
                getattr(tts, "output_rate", None) or "wav",
                "+".join(lang for lang in langs if lang) or "language unknown",
            )
        blocks = _voice_context_blocks(self._stt, tts, self.config.context)
        # Raw-PCM TTS streams gaplessly through one persistent player (no per-chunk aplay
        # spawn); WAV-only adapters keep blob mode.
        pcm_capable = tts is not None and getattr(tts, "output_rate", None) is not None
        audio_sink = AudioSink(sink_dev, mode="stream" if pcm_capable else "blob")
        # Software AEC ([aec] extra), wired twice: the sink feeds it our playback as the
        # reference, the backend runs capture through it. NOT building it is the degrade to
        # soft-duplex — a starved canceller cancels nothing while its warmup hold suppresses
        # early-confirm forever.
        aec_stage = None
        if self.config.aec == "webrtc":
            # AEC3 frames 10 ms blocks: rate % 100 != 0 (matcha's 22050) drops every
            # reference block -> the starved canceller above
            tts_rate = getattr(tts, "output_rate", None) if pcm_capable else None
            if tts_rate is None:
                self.logger.warning(
                    "aec='webrtc' needs a raw-PCM TTS for its reference signal, but "
                    "tts.provider='{}' plays WAV blobs (its sink never feeds the "
                    "reference tap); use an on-device raw-PCM engine or set "
                    "tts.audioFormat='pcm', falling back to soft-duplex",
                    self.config.tts.provider,
                )
            elif tts_rate % 100:
                self.logger.warning(
                    "aec='webrtc' cannot frame the {} Hz reference from "
                    "tts.provider='{}' (rate must be divisible by 100); "
                    "falling back to soft-duplex",
                    tts_rate, self.config.tts.provider,
                )
            else:
                from nanobot_channel_voice.aec import make_echo_canceller

                aec_stage = make_echo_canceller(
                    self.config.audio.sample_rate,
                    device_delay_ms=self.config.audio.playout_delay_ms,
                )
                if aec_stage is not None:
                    audio_sink.set_reference_tap(aec_stage)
        # A streaming adapter decodes DURING speech, so eager speculation is pointless;
        # batch on-device adapters get it. Never the nanobot delegate: it may be a billed
        # cloud API, where a resumed speaker wastes one call per pause.
        streaming = self._stt is not None and getattr(self._stt, "streaming", False)
        backend = LocalBackend(
            self.config,
            vad=vad,
            tts=tts,
            sink=audio_sink,
            transcribe=self._transcribe_pcm,
            publish_text=self._publish_turn_text,
            interrupt=self._publish_stop,
            metrics=self._metrics,
            eager_ms=self.config.stt.eager_ms if (self._stt is not None and not streaming) else 0,
            stt_stream=self._stt if streaming else None,
            aec=aec_stage,
            turn_analyzer=turn_analyzer,
            wake_detector=wake_detector,
        )
        shell = VoiceShell(
            self.config,
            capture=capture,
            sink=audio_sink,
            backend=backend,
            open_mic=self.config.open_mic,
            on_fatal=self.stop,  # a dead shell must release the channel too
            # Shared, else the shell builds its own and re-logs the telemetry banner.
            metrics=self._metrics,
            tracer=self._tracer,
        )
        return shell, backend, tts, blocks

    def _announce_context(self) -> None:
        if tool_created():
            # `context` is unbounded and the per-turn cost invisible: say it once.
            self.logger.info(
                "voice context: {} words ride every published utterance (voice_context tool)",
                sum(len(block.content.split()) for block in self._voice_context),
            )
        else:
            self.logger.warning(
                "voice_context bridge tool is not registered with the agent loop "
                "(nanobot.tools entry point not visible?): NO voice context reaches "
                "the model; reinstall the plugin so the gateway sees its entry points"
            )

    async def _build_cloud(self, kind: str) -> tuple[VoiceShell, str, list]:
        rt = self.config.realtime
        # Fail fast on STATIC config errors before any device is claimed: left to the
        # backend they raise in the rx task, where the reconnect ladder reads them as
        # transport blips.
        _load_connect()  # missing [realtime] extra: a transport-shaped error otherwise
        if kind == "gemini":
            profile = None
            input_rate = gemini_live.INPUT_RATE
            if not resolve_gemini_key(rt.api_key):
                raise RuntimeError(
                    "no API key for realtime provider 'gemini' "
                    "(set channels.voice.realtime.apiKey or GEMINI_API_KEY)"
                )
            supported = True
        else:
            profile = resolve_profile(self.config.backend)  # openai/xai/azure/qwen/glm/stepfun
            profile.base_url(rt.base_url)  # raises for a provider with no default (Azure)
            input_rate = profile.input_rate
            if not resolve_openai_key(rt.api_key):
                raise RuntimeError(
                    f"no API key for realtime provider '{profile.key}' "
                    "(set channels.voice.realtime.apiKey or OPENAI_API_KEY)"
                )
            # Capability is PER MODEL: a newer generation can enable tools.
            model = rt.model or profile.default_model
            supported = bool(profile.capabilities_for(model)["supports_tools"])
        gated = rt.uplink != "server"
        # Capture at the PROVIDER's input rate (24 kHz OpenAI/xAI/Azure, 16 kHz Qwen/GLM/
        # Gemini); ALSA `plug` resamples the device. Gated: the on-device detectors are
        # all 16 kHz models, so capture runs there and the gate upsamples the frames it
        # admits. Playback opens at the OUTPUT rate: asymmetric rates are fine.
        capture_rate = 16000 if gated else input_rate
        # The mic stays open for server-VAD barge-in only with echo cancellation: asserted
        # hardware/OS AEC (aecAvailable=true or aec="hardware", one physical fact) or AEC3
        # (aec="webrtc"). Else the shell's SPEAKING gate: barge-in resumes after playback.
        aec_stage = None
        if rt.barge_in == "aec":
            hw_aec = rt.aec_available or self.config.full_duplex
            if not hw_aec and self.config.aec == "webrtc":
                from nanobot_channel_voice.aec import make_echo_canceller

                # Cloud playback is always stream-mode, so the playout-timed tap works.
                aec_stage = make_echo_canceller(
                    capture_rate,
                    device_delay_ms=self.config.audio.playout_delay_ms,
                )
            if not hw_aec and aec_stage is None:
                raise RuntimeError(
                    "cloud open-mic needs echo cancellation: set channels.voice."
                    "realtime.aecAvailable=true (or aec='hardware') for hardware/OS AEC, "
                    "aec='webrtc' for the software canceller ([aec] extra), or "
                    "realtime.bargeIn='gated'."
                )
        open_mic = rt.barge_in == "aec"  # aec => open; gated => shell gates while SPEAKING
        # Nothing here claims a device yet: capture/playback open in shell.start().
        audio_cfg = self.config.audio.model_copy(update={"sample_rate": capture_rate})
        capture, sink_dev = make_audio(audio_cfg)
        audio_sink = AudioSink(sink_dev, mode="stream")
        if aec_stage is not None:
            audio_sink.set_reference_tap(aec_stage)
        # Gated: the gate runs AEC first (its VAD needs the cancelled signal).
        inner_aec = None if gated else aec_stage
        if profile is None:
            inner = GeminiLiveBackend(
                self.config, sink=audio_sink, metrics=self._metrics, aec=inner_aec,
            )
        else:
            inner = RealtimeBackend(
                self.config, sink=audio_sink, profile=profile, metrics=self._metrics,
                aec=inner_aec,
            )
        if not gated:
            self._backend = inner
        else:
            # Last, so nothing built after them can leak them: the gate owns them from
            # here (released in its close). The config validator refused the static
            # fallbacks, but a model that fails to LOAD degrades the same way at runtime
            # (energy VAD / no wake detector), and a gate built on those bills on noise
            # or never uploads.
            self._backend = await self._build_gate(
                inner, audio_sink, aec_stage, capture_rate=capture_rate,
                uplink_rate=input_rate, open_mic=open_mic,
            )
        tools, exec_tool = await self._cloud_tools(supported, rt.tool_mode)
        shell = VoiceShell(
            self.config,
            capture=capture,
            sink=audio_sink,
            backend=self._backend,
            open_mic=open_mic,
            exec_tool=exec_tool,
            on_abandon=self._on_cloud_abandon,  # a consumed stop ends a delegation
            on_fatal=self.stop,
            tool_mode=rt.tool_mode,
            metrics=self._metrics,
            tracer=self._tracer,
        )
        # Supervisor rules only when the delegated tool is wired; direct rules only when
        # there are tools whose round-trip needs masking.
        supervisor = rt.tool_mode == "supervisor" and exec_tool is not None
        instructions = _cloud_instructions(
            rt.persona, supervisor=supervisor, has_tools=bool(tools)
        )
        return shell, instructions, tools

    async def _build_gate(
        self, inner, audio_sink: AudioSink, aec_stage, *,
        capture_rate: int, uplink_rate: int, open_mic: bool,
    ) -> GatedUplink:
        rt = self.config.realtime
        frame_ms = self.config.audio.frame_ms
        engines: list = []
        try:
            vad = await self._load(make_vad, self.config.vad, capture_rate, frame_ms)
            engines.append(vad)
            if isinstance(vad, EnergyVad):
                raise RuntimeError(
                    f"realtime.uplink='{rt.uplink}' needs the {self.config.vad.engine} "
                    "VAD, which did not load (see the warning above)"
                )
            turn_analyzer = await self._load(
                make_turn_analyzer, self.config.vad, capture_rate, frame_ms
            )
            engines.append(turn_analyzer)
            # uplink="vad" never consults the detector (the gate drops it unreleased).
            wake_detector = None
            if rt.uplink == "wake":
                wake_detector = await self._load(
                    make_wake_detector, self.config.wake, capture_rate, frame_ms
                )
                engines.append(wake_detector)
                if wake_detector is None:
                    raise RuntimeError(
                        "realtime.uplink='wake' needs the acoustic wake detector, which "
                        "did not load (see the warning above); fix wake.openwakeword or "
                        "use uplink='vad'"
                    )
            return GatedUplink(
                inner, config=self.config, sink=audio_sink, vad=vad,
                turn_analyzer=turn_analyzer, wake_detector=wake_detector, aec=aec_stage,
                capture_rate=capture_rate, uplink_rate=uplink_rate,
                open_mic=open_mic, metrics=self._metrics,
            )
        except BaseException:
            for engine in engines:
                if engine is not None:
                    with suppress(Exception):
                        engine.release()
            raise

    async def _cloud_tools(self, supported: bool, tool_mode: str):
        """(tool_defs, exec_tool) for the realtime model, or ([], None) persona-only.

        ``supported`` is the profile's tool capability: a provider whose function-call flow
        isn't the OpenAI exchange (Qwen) stays persona-only even with the gateway wired.
        ``"direct"`` declares nanobot's N tools, each call a guarded ``execute_tool`` slice
        the realtime model sequences; ``"supervisor"`` declares ONE (``ask_nanobot``)
        delegating the whole request, so multi-step planning leaves the weak model."""
        gw = self._tool_gateway
        if gw is None:
            # No core passes one today. A toolMode the user SET is inert: say so, or a
            # supervisor session looks like a plain chatbot that forgot how to delegate.
            level = (
                "warning" if "tool_mode" in self.config.realtime.model_fields_set else "info"
            )
            getattr(self.logger, level)(
                "voice: realtime.toolMode='{}' has no effect — this nanobot build passes no "
                "tool gateway to plugin channels (VoiceChannel.wants_tool_gateway is "
                "unread), so the session is persona-only: no nanobot tools, no ask_nanobot "
                "delegation. Use backend='local' for the full agent.",
                tool_mode,
            )
            return [], None
        if not supported:
            self.logger.info(
                "voice: provider '{}' does not support the tool-call seam; persona-only",
                self.config.backend,
            )
            return [], None

        if tool_mode == "supervisor":
            self.logger.info(
                "voice: supervisor tool mode, realtime model delegates reasoning to nanobot"
            )
            await self._warn_if_unified_session()
            # Not execute_tool: a delegated request is a whole turn, driven over the bus.
            return [_SUPERVISOR_TOOL], self._delegate_to_nanobot

        # Direct mode. The gateway derives the session key from channel/chat_id as the bus
        # does, so cloud tools share the voice session's working dir / memory.
        tools = [ToolDef.from_nanobot_schema(s) for s in await gw.get_tool_definitions()]

        async def exec_tool(name: str, args: str, turn: str):
            return await gw.execute_tool(
                name, args, channel=self.name, chat_id=self.config.chat_id,
            )

        return tools, exec_tool

    async def _delegate_to_nanobot(self, name: str, args: str, turn: str) -> str:
        """``ask_nanobot`` handler (supervisor mode): run a full nanobot turn over the bus
        and return its final text for the realtime model to speak.

        An ordinary inbound message, so it runs a complete turn under the normal guards in
        the voice session; the reply is collected from ``send``/``send_delta`` into the one
        ``_pending_delegation`` slot, serialized by ``_delegation_lock`` because the shell
        runs tool calls concurrently off its rx loop."""
        try:
            params = json.loads(args) if args else {}
        except (ValueError, TypeError):
            params = None
        if isinstance(params, dict):
            request = str(params.get("request") or "").strip()
            context = str(params.get("relevant_context") or "").strip()
        else:
            # args wasn't a JSON object: take it whole rather than drop a real call.
            request = (args or "").strip()
            context = ""
        if not request:
            return "I didn't catch what you needed. Could you say that again?"
        text = f"{request}\n\n[context from the conversation: {context}]" if context else request

        # Queue wait is its own component: a delegation can wait on the lock as long as it
        # then takes to run, and folding them would blame the AgentLoop.
        queued_at = time.monotonic()
        stops = self._cloud_stops
        if turn != self._asked_turn:
            # A later model turn (the user was heard again): this request replaces every
            # one asked before, running or queued. Calls of one turn queue instead.
            self._asked_turn = turn
            pending = self._pending_delegation
            if pending is not None and not pending.resolved:
                self._metrics.count("delegation_replaced")
                pending.abandon(_DELEGATION_REPLACED)
                await self._publish_stop()
        timeout_s = self.config.realtime.delegation_timeout_s
        async with self._delegation_lock:
            if stops != self._cloud_stops:
                # A stop ended the pending work while this call queued behind it.
                self._metrics.count("delegation_stopped")
                return _DELEGATION_STOPPED
            if turn != self._asked_turn:
                self._metrics.count("delegation_replaced")
                return _DELEGATION_REPLACED
            self._metrics.observe(
                "delegation_wait_ms", (time.monotonic() - queued_at) * 1000.0
            )
            collector = _ReplyCollector(self._metrics)
            self._pending_delegation = collector
            try:
                await self._publish_user_text(
                    text, metadata={_DELEGATION_META: collector.token}
                )
                return await asyncio.wait_for(collector.result(), timeout=timeout_s)
            except TimeoutError:
                self._metrics.count("delegation_timeout")
                # foreign > 0 and no answer: a core that stopped echoing inbound metadata.
                self.logger.warning(
                    "delegation timed out after {}s ({} deliveries into this chat carried "
                    "another turn's stamp meanwhile); answering with a retry prompt",
                    timeout_s, collector.foreign,
                )
                await self._publish_stop()  # still RUNNING: nobody will hear its answer
                return "I couldn't finish that in time. Please try again."
            except asyncio.CancelledError:
                await self._publish_stop()  # the shell swept the task (teardown): as above
                raise
            finally:
                if self._pending_delegation is collector:
                    self._pending_delegation = None

    async def _on_cloud_abandon(self) -> None:
        """A consumed stop ended the pending tool work: a delegation in flight is /stop-ped
        and answered as stopped, and one queued behind it gives up."""
        self._cloud_stops += 1
        collector = self._pending_delegation
        if collector is None or collector.resolved:
            return
        self._metrics.count("delegation_stopped")
        collector.abandon(_DELEGATION_STOPPED)
        await self._publish_stop()

    async def _start_stt_server(self):
        """``stt.serve``: expose the loaded on-device STT as a local OpenAI-compatible
        endpoint (WebUI dictation, voice notes).

        SINGLETON by construction: the server borrows ``self._stt`` (memory-limited targets
        can't fit two copies), built here only under a cloud backend, which never loads STT
        itself. Returns the server so the caller's reference survives a concurrent
        ``stop()`` clearing the attribute."""
        cfg = self.config.stt.serve
        if not cfg.enabled:
            return None
        if self._stt is None:
            # cloud backend + serve: build ONCE here, off the loop like the local load.
            self._stt = await self._load(make_stt, self.config.stt)
        if self._stt is None:
            # Config validation rejects provider='nanobot', so None means the engine
            # degraded (make_stt logged why); a silently absent endpoint breaks dictation.
            raise RuntimeError(
                "stt.serve is enabled but no on-device STT adapter could be built "
                "(see the preceding voice log line for what is missing)"
            )
        from nanobot_channel_voice.stt.serve import SttHttpServer

        server = SttHttpServer(self._stt, cfg)
        await server.start()
        self._stt_server = server  # published only once actually bound
        return server

    async def _warmup(self) -> None:
        """Warm each on-device adapter once, then (``perf.calibrate``) measure the WARM
        steady state; failures are logged and ignored (optimizations, never gates). Capture
        is already live, so hop-cost accounting is held across each saturating burst and
        ONLY those — ``_calibrate``'s wait-for-IDLE is live conversation, kept accounted."""
        local = self._local()
        if local is not None:
            local.hold_hop_accounting(True)
        try:
            for target in (self._stt, self._tts_adapter):
                if target is None:
                    continue
                try:
                    await target.warmup()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001
                    self.logger.debug(
                        "voice warmup failed for {}: {}", type(target).__name__, exc
                    )
            if local is not None:
                await local.prewarm_playback()  # device open off the first reply's TTFA
                await local.prewarm_canned()  # gated internally on probe_ok + IDLE
                await local.learn_wake_aliases()  # gated on on-device STT + probe_ok
        finally:
            if local is not None:
                local.hold_hop_accounting(False)
        if self.config.perf.calibrate:
            await self._calibrate()

    async def _calibrate(self) -> None:
        """Measure warm STT/TTS on THIS device and hand the numbers to the local backend.
        After warmup only, keeping cold-start noise out of the measurements."""
        local = self._local()
        if local is None:
            return  # cloud paces itself; nothing to derive
        # The probes share the live adapters, which keep STRICTLY one decode in flight, so
        # a real utterance would be both slowed and measured wrong. Best-effort wait.
        for _ in range(30):
            shell = self._shell
            if shell is None or shell.state is VoiceState.IDLE:
                break
            await asyncio.sleep(1.0)
        stt_ms: float | None = None
        tts_rtf: float | None = None
        tts_ms_per_char: float | None = None
        local.hold_hop_accounting(True)  # probes only: the IDLE wait above stays accounted
        try:
            # A streaming adapter decodes during capture and never calls transcribe(), so
            # timing it would describe a path this pipeline does not take.
            if self._stt is not None and not getattr(self._stt, "streaming", False):
                rate = self.config.audio.sample_rate
                t0 = time.monotonic()
                # Fixed-cost probe: 1 s of silence at the capture rate. For a fixed-window
                # model (whisper) this IS the decode floor; length-proportional engines
                # underestimate long utterances.
                await self._stt.transcribe(b"\x00" * (2 * rate), rate)
                stt_ms = (time.monotonic() - t0) * 1000.0
            tts = self._tts_adapter
            # probe_ok is False on cloud adapters: no startup billing, and a cloud RTF
            # measures the network, not the box.
            if tts is not None and getattr(tts, "probe_ok", True):
                # The engine's own language: an English probe through a zh/ja lexicon
                # trips the empty-synth guard below.
                text = startup_text(
                    CALIBRATION_TEXT, getattr(tts, "spoken_language", None)
                )
                rate = getattr(tts, "output_rate", None)
                t0 = time.monotonic()
                if rate:
                    audio_s = pcm_ms(len(await tts.synthesize_pcm(text)), rate) / 1000.0
                else:
                    audio_s = wav_duration_ms(await tts.synthesize(text)) / 1000.0
                if audio_s > 0.2:  # a failed/empty synth must not calibrate anything
                    synth_s = time.monotonic() - t0
                    tts_rtf = synth_s / audio_s
                    tts_ms_per_char = synth_s * 1000.0 / max(1, len(text))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self.logger.debug("voice calibration failed: {}", exc)
            return
        finally:
            local.hold_hop_accounting(False)
        local.apply_calibration(
            stt_cost_ms=stt_ms,
            tts_rtf=tts_rtf,
            tts_ms_per_char=tts_ms_per_char,
            # Pydantic tracks fields the user SET; an explicit minCharsFirst always wins.
            chunk_floor_pinned="min_chars_first"
            in getattr(self.config.chunker, "model_fields_set", set()),
        )

    async def _metrics_reporter(self, interval_s: float) -> None:
        """``debug.metricsIntervalS``: the live snapshot, one JSON line per interval."""
        while True:
            await asyncio.sleep(interval_s)
            if self._metrics.has_data:
                self.logger.info(
                    "voice metrics: {}",
                    json.dumps(
                        self._metrics.snapshot(), ensure_ascii=False,
                        separators=(",", ":"),
                    ),
                )

    def _warn_if_transcription_unconfigured(self) -> None:
        """``stt.provider="nanobot"`` with nothing behind it is a deaf channel that still
        reports healthy: every utterance decodes to ``""`` and is dropped as silence."""
        if self._stt is not None or self.config.stt.provider != "nanobot":
            return  # on-device: loaded, or its own build warning already named the gap
        gap = transcription_gap()
        if gap is not None:
            self.logger.warning(
                "voice: stt.provider='nanobot' delegates every utterance to nanobot's "
                "transcription, but {} — the channel will start and hear NOTHING. Configure "
                "it, or set stt.provider to an on-device engine.",
                gap,
            )

    async def _warn_if_unified_session(self) -> None:
        """Voice owns the turns it publishes: a barge-in stops them, and it waits for their
        reply. Core folds a message that lands mid-turn into the running turn, which in one
        shared session crosses channels. Off the loop: the check parses the whole config."""
        if await asyncio.to_thread(unified_session):
            self.logger.warning(
                "voice: agents.defaults.unifiedSession runs every channel in one session, so "
                "turns cross channels: a message sent elsewhere while voice is mid-turn joins "
                "that turn and is answered aloud, speech during another channel's turn is "
                "answered there while voice waits for a reply that never comes, and a spoken "
                "stop or a barge-in cancels whichever turn is running. Turn unifiedSession off "
                "to keep each channel's turns its own."
            )

    def _drop_bridge(self) -> None:
        """Stop serving context for this channel instance. Identity-checked: a late
        teardown must not remove a restarted channel's fresh bridge."""
        if self._context_bridge is not None:
            unregister_bridge(self.name, self.config.chat_id, self._context_bridge)
            self._context_bridge = None

    async def stop(self) -> None:
        self._running = False
        self._drop_bridge()
        await cancel_and_wait(self._metrics_task)
        self._metrics_task = None
        await cancel_and_wait(self._warmup_task)
        self._warmup_task = None
        if self._stt_server is not None:
            # Before the shell: no new serve-side decode may outlive the teardown below.
            with suppress(Exception):
                await self._stt_server.stop()
            self._stt_server = None
        if self._shell is not None:
            await self._shell.stop()
            self._shell = None
            self._backend = None
        # Last, so no decode can be running against it. The backend freed TTS/VAD.
        self._release_stt()
        if self._stop_event is not None:
            self._stop_event.set()

    @staticmethod
    async def _load(fn, *args):
        """A model load off the loop. Core cancels the start task and a thread cannot be
        interrupted: a cancelled load frees its result (in the thread, or via the callback
        when it landed before the flag) instead of leaking a session past the restart."""
        load = _Load()
        fut = asyncio.get_running_loop().run_in_executor(None, _run_load, load, fn, args)
        try:
            return await asyncio.shield(fut)
        except asyncio.CancelledError:
            load.cancelled = True
            fut.add_done_callback(_release_late)
            raise

    def _release_stt(self) -> None:
        if self._stt is not None:
            with suppress(Exception):
                self._stt.release()
            self._stt = None

    # ---- input helpers ------------------------------------------------------

    async def _transcribe_pcm(self, pcm: bytes) -> str:
        if self._stt is not None:
            # Chunked: vad.maxUtteranceMs may exceed the adapter's decode window.
            return await transcribe_chunked(self._stt, pcm, self.config.audio.sample_rate)
        # No on-device STT: hand a WAV to nanobot's transcription layer. Off the loop —
        # a ~1 MB mkstemp+wave write and its unlink can stall an SD-card SBC for tens of ms,
        # and capture, VAD and sink pacing share it.
        path = await asyncio.to_thread(write_temp_wav, pcm, self.config.audio.sample_rate)
        try:
            return await self.transcribe_audio(path)
        finally:
            with suppress(OSError):
                await asyncio.to_thread(os.unlink, path)

    async def _publish_turn_text(
        self, text: str, turn_token: str, notes: tuple[str, ...] = ()
    ) -> None:
        """Publish a captured utterance tagged with the turn it opens: core echoes inbound
        metadata onto that turn's final send, so ``send`` can tell the live turn's reply
        from a barged-out one's straggler. Notes ride the context bridge keyed by the token,
        keeping metadata JSON-plain (tools snapshot it, cron persists the snapshot). ``text``
        is the transcript verbatim, except a goal verdict (``LocalBackend._is_goal``)."""
        if self._context_bridge is not None:
            self._context_bridge.stash_notes(turn_token, notes)
        await self._publish_user_text(text, metadata={TURN_META: turn_token})

    async def _publish_user_text(
        self, text: str, metadata: dict[str, Any] | None = None
    ) -> None:
        await self._handle_message(
            sender_id=self.config.sender_id,
            chat_id=self.config.chat_id,
            content=text,
            metadata=metadata or None,
            is_dm=False,
        )

    async def _publish_stop(self) -> None:
        # Priority command: cancels this session's in-flight turn before the new utterance
        # is published. Bypasses _handle_message, so it is never gated or streamed.
        await self.bus.publish_inbound(
            InboundMessage(
                channel=self.name,
                sender_id=self.config.sender_id,
                chat_id=self.config.chat_id,
                content="/stop",
                metadata={_VOICE_CMD_META: True},  # the ack inherits it; _speakable drops
            )
        )

    # ---- output -------------------------------------------------------------

    def _local(self) -> LocalBackend | None:
        """The backend when it's the local pipeline (bus glue speaks the reply), else None:
        cloud's text bus is not the reasoning path, so ``send``/``send_delta`` are inert —
        EXCEPT supervisor mode, which collects a delegated reply before this check."""
        return self._backend if isinstance(self._backend, LocalBackend) else None

    async def send(self, msg: OutboundMessage) -> None:
        # A pending delegation's non-streaming terminal; _stream_end in send_delta is the
        # streaming one, first wins. Only OUR session chat qualifies: a delivery routed
        # elsewhere (cron) must not resolve it.
        meta = msg.metadata or {}
        if msg.chat_id != self.config.chat_id:
            # One speaker, one chat: a delivery addressed elsewhere (the message tool takes
            # an arbitrary channel/chat) must neither be spoken nor touch this turn's state.
            return
        if _delegation_reply(self._pending_delegation, meta):
            if _speakable(msg):
                text = (msg.content or "").strip()
                if text:
                    self._pending_delegation.set_final(text)
            return
        local = self._local()
        if local is None:
            if self.config.backend != "local":
                await self._cloud_send(msg, meta)
            return
        # ANY traffic for our chat proves the core is alive on this session: feed the
        # deadman BEFORE filtering, so it measures a silent core, not a long tool run.
        local.note_agent_activity()
        if not _speakable(msg):
            return
        turn = meta.get(TURN_META)
        if turn is not None and local.is_dead_turn(turn) and not _agent_initiated(meta):
            # A killed turn's late final; a superseded-but-live turn still speaks. Trigger-
            # stamped sends are exempt: a cron job snapshots its CREATION turn's token and
            # every fire echoes it, so the gate would eat the reminder itself.
            return
        text = (msg.content or "").strip()
        if not text:
            return
        if _agent_initiated(meta):
            local.note_proactive()
        await local.speak_final(text)

    async def _cloud_send(self, msg: OutboundMessage, meta: dict[str, Any]) -> None:
        """Cloud: the model voices what the agent sent on its own (a cron or trigger turn's
        reply, a message another channel sent here, a heartbeat report). A delegation's
        straggler and non-reply traffic stay silent."""
        if not _speakable(msg) or _straggler(meta):
            return
        text = (msg.content or "").strip()
        if self._notice is not None and _agent_initiated(meta):
            # The collected turn ends here: its last segment streamed nothing.
            self._notice.set_final(text)
            await self._take_notice()
        elif text:
            await self._announce(text)

    async def _cloud_delta(
        self, delta: str, meta: dict[str, Any], stream_id: str | None, *,
        stream_end: bool, resuming: bool,
    ) -> None:
        if _straggler(meta):
            return
        base = base_of(stream_id)
        if self._notice is None or base != self._notice_base:
            if stream_end and not delta:
                return  # nothing collected to end (an aborted stream's close)
            # A new turn; one that never ended with content is dropped.
            self._notice = _ReplyCollector(self._metrics, timed=False)
            self._notice_base = base
        _collect(self._notice, delta, stream_end=stream_end, resuming=resuming)
        if self._notice.resolved:
            await self._take_notice()

    async def _take_notice(self) -> None:
        notice, self._notice = self._notice, None
        if notice is not None and notice.reply:
            await self._announce(notice.reply)

    async def _announce(self, text: str) -> None:
        announce = getattr(self._backend, "announce", None)
        shown = loggable_text(text, self.config.log_transcripts)
        if announce is None:
            self.logger.warning("voice: no session to voice an agent message: '{}'", shown)
            return
        self.logger.info("voice: the model voices an agent message: '{}'", shown)
        await announce(text)

    async def send_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
        *,
        stream_id: str | None = None,
        stream_end: bool = False,
        resuming: bool = False,
        merge_next: bool = False,
    ) -> None:
        # The manager passes stream framing as kwargs unconditionally, so declaring these
        # parameters is load-bearing: an override without them fails every delta.
        if chat_id != self.config.chat_id:
            return  # addressed elsewhere; see send()
        if stream_end and merge_next:
            # A reply cut at the token limit: the next segment continues this sentence, often
            # mid-word, so this end is no boundary (the manager passes merge_next only to a
            # send_delta that declares it).
            stream_end = resuming = False
        if _delegation_reply(self._pending_delegation, metadata or {}):
            _collect(self._pending_delegation, delta, stream_end=stream_end, resuming=resuming)
            return
        local = self._local()
        if local is None:
            if self.config.backend != "local":
                await self._cloud_delta(
                    delta, metadata or {}, stream_id, stream_end=stream_end, resuming=resuming,
                )
            return
        turn = (metadata or {}).get(TURN_META)
        if turn is not None and not _agent_initiated(metadata) and local.is_stale_stream(turn):
            # Core re-runs a stopped turn's pending injections as their own turn (the cancelled
            # run re-publishes its queue): keep it silent, and /stop it once while it still runs.
            if (not stream_end or resuming) and turn not in self._stopped_reruns:
                self._stopped_reruns.append(turn)
                await self._publish_stop()
            return
        if _agent_initiated(metadata):
            # A cron/trigger turn streaming into this chat: mark BEFORE the delta plays,
            # so the settle re-opens attention.
            local.note_proactive()
        if stream_end:
            await local.on_stream_end(resuming=resuming, stream_id=stream_id)
        else:
            await local.on_delta(delta, stream_id=stream_id)
