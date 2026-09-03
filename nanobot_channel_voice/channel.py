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
import time
from contextlib import suppress
from typing import Any

from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.runtime_context import RuntimeContextBlock

from nanobot_channel_voice.aio import cancel_and_wait
from nanobot_channel_voice.audio import make_audio
from nanobot_channel_voice.audio.pcm import pcm_ms, wav_duration_ms
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import ToolDef, VoiceState
from nanobot_channel_voice.backend.local import LocalBackend
from nanobot_channel_voice.backend.openai_realtime import RealtimeBackend, _load_connect
from nanobot_channel_voice.backend.profiles import backend_kind, resolve_profile
from nanobot_channel_voice.config import (
    VoiceConfig,
    consume_import_json,
    resolve_openai_key,
    transcription_gap,
)
from nanobot_channel_voice.context_tool import (
    VoiceContextBridge,
    register_bridge,
    tool_created,
    unregister_bridge,
)
from nanobot_channel_voice.metrics import VoiceMetrics
from nanobot_channel_voice.shell import VoiceShell
from nanobot_channel_voice.streamid import TURN_META, started_ns, unique_token
from nanobot_channel_voice.stt import SttAdapter, make_stt, transcribe_chunked, write_temp_wav
from nanobot_channel_voice.telemetry import VoiceTracer
from nanobot_channel_voice.tts import TtsAdapter, make_tts
from nanobot_channel_voice.tts.base import CALIBRATION_TEXT, startup_text
from nanobot_channel_voice.vad import make_turn_analyzer, make_vad
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
# to a fast ack. Appended in EVERY mode.
_STOP_RULE = (
    "If the user only tells you to stop, be quiet, or wait, do not answer — "
    "produce no speech at all."
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
        "speak a brief neutral filler to the user BEFORE calling this."
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

# The ask_nanobot result on a mid-delegation barge-in: only satisfies the function call,
# the backend's stale-response guard drops it.
_DELEGATION_INTERRUPTED = "(interrupted by the user)"

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
        part for part in (persona or _DEFAULT_PERSONA, rules, _STOP_RULE) if part
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


# Stamped on a delegated ask_nanobot request; the AgentLoop echoes inbound metadata onto
# the turn's FINAL send (a /stop-ped delegation can finish minutes later with a bare final
# send). The token must match: an unstamped delivery into this chat is someone else's
# turn (a cron fire, a message-tool send), never the delegation's answer.
_DELEGATION_META = "_voice_delegation"


class _DelegationCollector:
    """Collects one delegated nanobot turn's reply off the bus (supervisor mode); the first
    terminal resolves the future. Streaming ON: deltas accumulate, ``_stream_end`` resolves,
    the turn's final never reaches a channel (core drops it). OFF: one final ``send``
    resolves."""

    def __init__(self, metrics: VoiceMetrics) -> None:
        self._future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._parts: list[str] = []
        self._metrics = metrics
        self._started_at = time.monotonic()
        self._first_token = False
        # Tombstone: a /stop-ped turn's reply may still be in flight, so a dead collector
        # stays registered to swallow it, not resolve the NEXT delegation.
        self.dead = False
        # Watermark against a PREVIOUS turn's stragglers: stream ids embed the turn's start
        # time_ns (streamid) and one answering THIS delegation starts after the collector.
        # An unrecognized id format accepts everything.
        self._created_ns = time.time_ns()
        self.token = unique_token()  # non-streaming identity, see _DELEGATION_META

    def accepts_stream(self, stream_id: str | None) -> bool:
        ns = started_ns(stream_id)
        return ns is None or ns >= self._created_ns

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

    def note_boundary(self) -> None:
        """A tool-boundary segment break: separates the reply's parts WITHOUT latching the
        first-token clock — the separator is channel-fabricated, and latching would
        under-report TTFT for exactly the tool-first delegations supervisor mode is for."""
        self._parts.append("\n")

    def finish(self, fallback: str = "") -> None:
        text = "".join(self._parts).strip() or (fallback or "").strip()
        if not text:
            # Not a terminal: core fires on_stream_end(resuming=False) mid-turn on its
            # blank-response retry, so keep collecting (the real end or delegationTimeoutS
            # decides) and do not mark a first token, which would suppress the real one.
            return
        self._mark_first_token()
        self._resolve(text)

    def set_final(self, text: str) -> None:
        """The non-streaming terminal: one whole reply, so it is also first token."""
        self._mark_first_token()
        self._resolve(text)

    def entomb(self) -> None:
        """Mark dead AND latch first-token: a tombstone's late reply must neither resolve
        the next delegation nor time a delegation that failed."""
        self.dead = True
        self._first_token = True

    def abandon(self, text: str) -> None:
        """Release the delegation, latching first-token: no answer was produced, and the
        cancelled turn's late ``send``/``send_delta`` (possible until
        ``_pending_delegation`` clears) must not be timed as one."""
        self._first_token = True
        self.dead = True
        self._resolve(text)

    def _resolve(self, text: str) -> None:
        if not self._future.done():
            self._future.set_result(text)

    async def result(self) -> str:
        return await self._future


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
        self._backend: LocalBackend | RealtimeBackend | None = None
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
        self._pending_delegation: _DelegationCollector | None = None
        self._delegation_lock = asyncio.Lock()
        # One per session, shared with backend and shell: segments join on call_id.
        self._metrics = VoiceMetrics()
        self._tracer = VoiceTracer(self.config.telemetry)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return VoiceConfig().model_dump(by_alias=True)

    # ---- lifecycle ----------------------------------------------------------

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
        try:
            kind = backend_kind(self.config.backend)
            if kind == "openai_dialect":
                self._stt = None  # the provider does ASR; never load on-device models
                shell, instructions, tools = await self._build_cloud()
            elif kind == "local":
                # Off the loop: a dozen ORT/RKNN session loads inline would freeze the
                # gateway (every other channel, cron, the WebUI) for the whole build.
                # Nothing here binds to a loop — Queue/Event bind lazily on first use.
                self._stt = await asyncio.to_thread(make_stt, self.config.stt)
                self._warn_if_transcription_unconfigured()
                shell, instructions, tools = await asyncio.to_thread(self._build_local)
            else:
                # Refuse loudly: falling through to local would run the wrong brain.
                raise RuntimeError(f"voice backend kind '{kind}' is not implemented")
            if not self._running:
                self._backend = None
                self._drop_bridge()
                return  # stop() raced the build; nothing was started yet
            self.logger.info(
                "voice channel starting (backend={}, capture={}, playback={})",
                self.config.backend,
                self.config.audio.capture_device, self.config.audio.playback_device,
            )
            await shell.start(instructions=instructions, tools=tools)
        except BaseException:
            # Must not report healthy, leave a never-started backend registered (speak_final
            # queues into a worker that never runs), or leave a half-started shell holding
            # devices (shell.start's first await already spawned arecord).
            self._running = False
            self._backend = None
            self._drop_bridge()
            if shell is not None:
                with suppress(Exception):
                    await shell.stop()
            raise
        if not self._running:
            # stop() landed mid-start, before _shell was published: tear it down here.
            await shell.stop()
            self._backend = None
            self._drop_bridge()
            return
        self._shell = shell
        try:
            server = await self._start_stt_server()
        except BaseException:
            # A serve endpoint that cannot bind refuses loudly (WebUI dictation would be
            # silently broken) and takes the shell down.
            self._running = False
            self._shell = None
            self._backend = None
            self._drop_bridge()
            with suppress(Exception):
                await shell.stop()
            raise
        if not self._running:
            # stop() raced the endpoint coming up: the handle is published only after the
            # bind, so stop() saw None. THIS frame owns the listener.
            if server is not None:
                with suppress(Exception):
                    await server.stop()
            self._stt_server = None
            self._shell = None
            self._backend = None
            self._drop_bridge()
            await shell.stop()  # idempotent
            return
        # Off the critical path: the first turn then pays no cold start (ORT/RKNN/TRT).
        self._warmup_task = asyncio.create_task(self._warmup())
        if self.config.debug.metrics_interval_s:
            self._metrics_task = asyncio.create_task(
                self._metrics_reporter(self.config.debug.metrics_interval_s)
            )
        await self._stop_event.wait()
        self.logger.info("voice channel stopped")

    def _build_local(self) -> tuple[VoiceShell, str, list]:
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
        self._tts_adapter = tts
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
        self._voice_context = _voice_context_blocks(self._stt, tts, self.config.context)
        self._context_bridge = register_bridge(
            self.name, self.config.chat_id, self._voice_context
        )
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
        self._backend = LocalBackend(
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
            backend=self._backend,
            open_mic=self.config.open_mic,
            on_fatal=self.stop,  # a dead shell must release the channel too
            # Shared, else the shell builds its own and re-logs the telemetry banner.
            metrics=self._metrics,
            tracer=self._tracer,
        )
        return shell, "", []

    async def _build_cloud(self) -> tuple[VoiceShell, str, list]:
        rt = self.config.realtime
        profile = resolve_profile(self.config.backend)  # openai/xai/azure/qwen/glm/stepfun
        # Fail fast on STATIC config errors before any device is claimed: left to the
        # backend they raise in the rx task, where the reconnect ladder reads them as
        # transport blips.
        profile.base_url(rt.base_url)  # raises for a provider with no default (Azure)
        _load_connect()  # missing [realtime] extra: a transport-shaped error otherwise
        if not resolve_openai_key(rt.api_key):
            raise RuntimeError(
                f"no API key for realtime provider '{profile.key}' "
                "(set channels.voice.realtime.apiKey or OPENAI_API_KEY)"
            )
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
                    profile.input_rate,
                    device_delay_ms=self.config.audio.playout_delay_ms,
                )
            if not hw_aec and aec_stage is None:
                raise RuntimeError(
                    "cloud open-mic needs echo cancellation: set channels.voice."
                    "realtime.aecAvailable=true (or aec='hardware') for hardware/OS AEC, "
                    "aec='webrtc' for the software canceller ([aec] extra), or "
                    "realtime.bargeIn='gated'."
                )
        if not rt.server_vad:
            # No client-side commit / response.create path exists: audio would stream up
            # forever and the session would never answer.
            raise RuntimeError(
                "realtime.serverVad=false is not supported: the channel relies on "
                "server-side turn detection to commit audio and create responses."
            )
        open_mic = rt.barge_in == "aec"  # aec => open; gated => shell gates while SPEAKING
        # Capture at the PROVIDER's input rate (24 kHz OpenAI/xAI/Azure, 16 kHz Qwen/GLM);
        # ALSA `plug` resamples the device. Playback opens at the OUTPUT rate: asymmetric
        # rates are fine.
        audio_cfg = self.config.audio.model_copy(update={"sample_rate": profile.input_rate})
        capture, sink_dev = make_audio(audio_cfg)
        audio_sink = AudioSink(sink_dev, mode="stream")
        if aec_stage is not None:
            audio_sink.set_reference_tap(aec_stage)
        self._backend = RealtimeBackend(
            self.config, sink=audio_sink, profile=profile, metrics=self._metrics,
            aec=aec_stage,
        )
        # Capability is PER MODEL: a newer generation can enable tools.
        model = self.config.realtime.model or profile.default_model
        supported = bool(profile.capabilities_for(model)["supports_tools"])
        tools, exec_tool = await self._cloud_tools(supported, rt.tool_mode)
        shell = VoiceShell(
            self.config,
            capture=capture,
            sink=audio_sink,
            backend=self._backend,
            open_mic=open_mic,
            exec_tool=exec_tool,
            on_barge_in=self._on_cloud_barge_in,  # abandon a delegation talked over
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
            # Not execute_tool: a delegated request is a whole turn, driven over the bus.
            return [_SUPERVISOR_TOOL], self._delegate_to_nanobot

        # Direct mode. The gateway derives the session key from channel/chat_id as the bus
        # does, so cloud tools share the voice session's working dir / memory.
        tools = [ToolDef.from_nanobot_schema(s) for s in await gw.get_tool_definitions()]

        async def exec_tool(name: str, args: str):
            return await gw.execute_tool(
                name, args, channel=self.name, chat_id=self.config.chat_id,
            )

        return tools, exec_tool

    async def _delegate_to_nanobot(self, name: str, args: str) -> str:
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
            request = (params.get("request") or "").strip()
            context = (params.get("relevant_context") or "").strip()
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
        # Its own budget: an ask_nanobot turn is a full AgentLoop run (tool-heavy ones
        # exceed 30 s), whereas turn_timeout_s is the REALTIME WIRE watchdog.
        timeout_s = self.config.realtime.delegation_timeout_s or self.config.realtime.turn_timeout_s
        async with self._delegation_lock:
            self._metrics.observe(
                "delegation_wait_ms", (time.monotonic() - queued_at) * 1000.0
            )
            collector = _DelegationCollector(self._metrics)
            self._pending_delegation = collector
            try:
                await self._publish_user_text(
                    text, metadata={_DELEGATION_META: collector.token}
                )
                return await asyncio.wait_for(collector.result(), timeout=timeout_s)
            except TimeoutError:
                self._metrics.count("delegation_timeout")
                self.logger.warning(
                    "delegation timed out after {}s; answering with a retry prompt",
                    timeout_s,
                )
                # Still RUNNING: stop it, or it burns tokens on an answer nobody hears.
                collector.entomb()
                await self._publish_stop()
                return "I couldn't finish that in time. Please try again."
            finally:
                # A dead collector stays as a tombstone; the next delegation replaces it.
                if self._pending_delegation is collector and not collector.dead:
                    self._pending_delegation = None

    async def _on_cloud_barge_in(self) -> None:
        """Supervisor mode: the user talked over an in-flight ask_nanobot delegation — stop
        the moot nanobot turn (``/stop``) and release the delegation. No-op otherwise."""
        collector = self._pending_delegation
        if collector is None or collector.dead:
            return  # nothing delegating (a tombstone is not a live delegation)
        self._metrics.count("delegation_interrupted")
        await self._publish_stop()
        collector.abandon(_DELEGATION_INTERRUPTED)

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
            self._stt = make_stt(self.config.stt)  # cloud backend + serve: build ONCE here
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
        if self._stt is not None:
            # Last, so no decode can be running against it. The backend freed TTS/VAD.
            with suppress(Exception):
                self._stt.release()
            self._stt = None
        if self._stop_event is not None:
            self._stop_event.set()

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
        if self._pending_delegation is not None:
            if meta.get(_DELEGATION_META) != self._pending_delegation.token:
                # Another turn's reply (a straggler, a cron fire), not our answer. Logged:
                # a core that stopped echoing the stamp would show up as a silent hang.
                self.logger.debug("voice: unstamped delivery ignored while delegating")
                return
            if _speakable(msg):
                text = (msg.content or "").strip()
                if text:
                    self._pending_delegation.set_final(text)
            return
        local = self._local()
        if local is None:
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

    async def send_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
        *,
        stream_id: str | None = None,
        stream_end: bool = False,
        resuming: bool = False,
    ) -> None:
        # The manager passes stream framing as kwargs unconditionally, so declaring these
        # parameters is load-bearing: an override without them fails every delta. Supervisor
        # mode accumulates the delegated reply here; a resuming end is only a tool boundary,
        # so resolving there would truncate to the pre-tool status line.
        if chat_id != self.config.chat_id:
            return  # addressed elsewhere; see send()
        if self._pending_delegation is not None:
            if not self._pending_delegation.accepts_stream(stream_id):
                return  # a stopped PREVIOUS turn's queued straggler, not our answer
            if stream_end:
                if resuming:
                    self._pending_delegation.note_boundary()  # segment break, keep collecting
                else:
                    self._pending_delegation.finish(fallback=delta or "")
            else:
                self._pending_delegation.add(delta or "")
            return
        local = self._local()
        if local is None:
            return
        if _agent_initiated(metadata):
            # A cron/trigger turn streaming into this chat: mark BEFORE the delta plays,
            # so the settle re-opens attention.
            local.note_proactive()
        if stream_end:
            await local.on_stream_end(resuming=resuming, stream_id=stream_id)
        else:
            await local.on_delta(delta, stream_id=stream_id)
