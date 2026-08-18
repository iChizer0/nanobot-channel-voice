"""The voice channel: thin glue mapping nanobot's channel contract onto a ``VoiceShell``
around a swappable ``VoiceBackend`` (``local`` or a realtime provider). Capture -> STT
publishes via ``_handle_message``, so allow-list and session routing work as for any
channel; barge-in publishes the priority ``/stop``, then the new utterance
(cancel-then-send); ``send_delta`` text is spoken chunk-by-chunk, while ``send`` speaks
only genuine final messages (see :func:`_speakable`)."""

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
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_INPUT_META,
    RuntimeContextBlock,
    wrap_runtime_context_lines,
)

from nanobot_channel_voice.aio import cancel_and_wait
from nanobot_channel_voice.audio import make_audio
from nanobot_channel_voice.audio.pcm import pcm_ms, wav_duration_ms
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import ToolDef, VoiceState
from nanobot_channel_voice.backend.local import TURN_META, LocalBackend
from nanobot_channel_voice.backend.openai_realtime import RealtimeBackend
from nanobot_channel_voice.backend.profiles import backend_kind, resolve_profile
from nanobot_channel_voice.config import (
    VoiceConfig,
    consume_import_json,
    resolve_openai_key,
)
from nanobot_channel_voice.metrics import VoiceMetrics
from nanobot_channel_voice.shell import VoiceShell
from nanobot_channel_voice.streamid import started_ns, unique_token
from nanobot_channel_voice.stt import SttAdapter, make_stt, transcribe_chunked, write_temp_wav
from nanobot_channel_voice.telemetry import VoiceTracer
from nanobot_channel_voice.tts import TtsAdapter, make_tts
from nanobot_channel_voice.tts.base import CALIBRATION_TEXT, startup_text
from nanobot_channel_voice.vad import make_turn_analyzer, make_vad
from nanobot_channel_voice.wake import make_wake_detector

_DEFAULT_PERSONA = (
    "You are a helpful, concise voice assistant. Keep replies short and conversational."
)

# Direct tool mode: the filler preamble is what the user hears across a tool round-trip;
# without it the line goes dead. Appended only when tools are actually declared.
_DIRECT_RULES = (
    "Before a tool call that will keep the user waiting, say a brief neutral filler "
    "in the user's language, such as \"One moment.\" or \"Let me check.\" (never "
    "implying success or failure), then call it with no further speech. Skip the "
    "filler when you expect the answer immediately. The reply that delivers the "
    "answer is pure answer: never open it with wait phrases or progress narration."
)

# Silence-is-the-ack, model-side half: the enforcement is the client's transcript-gated
# response.cancel (backend._consume_stop), but that needs input transcription enabled and
# can lose a race to a very fast ack — this line keeps the un-cancellable head short and
# covers providers whose transcription events never arrive. Appended in EVERY mode.
_STOP_RULE = (
    "If the user only tells you to stop, be quiet, or wait, do not answer — "
    "produce no speech at all."
)

# Supervisor tool mode (Responder-Thinker): the realtime model owns the conversational
# surface, delegating reasoning/tool work to nanobot; the filler masks the round-trip.
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

# Supervisor mode's only declared tool: persona + this schema is the whole realtime
# context, while MCP/skills/memory stay in nanobot.
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

# The ask_nanobot result when the user barges in mid-delegation; only satisfies the
# function call, since the backend's stale-response guard drops it (barge-in already
# cancelled the response).
_DELEGATION_INTERRUPTED = "(interrupted by the user)"

# Tags our own priority commands. Core copies INBOUND metadata onto the command ack
# ("Stopped 1 task(s)."), so this is how _speakable drops it: untagged, every confirmed
# barge-in speaks the ack (local) or resolves a delegation with it (supervisor).
_VOICE_CMD_META = "_voice_cmd"

# Trace flags older cores stamp on outbound metadata; newer cores moved the same semantics
# onto the typed ``OutboundMessage.event``. BOTH are checked: neither alone covers every
# core the plugin runs against.
_TRACE_META = (
    "_streamed",        # already spoken via send_delta
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


def _cloud_instructions(persona: str | None, *, supervisor: bool, has_tools: bool) -> str:
    """The realtime session's instructions: persona (taste) then the mode's tool rules
    (contract). ONE derivation, so a ``realtime.persona`` override can restyle the voice
    but never delete the delegation contract or the filler preamble."""
    rules = _SUPERVISOR_RULES if supervisor else (_DIRECT_RULES if has_tools else "")
    return "\n\n".join(
        part for part in (persona or _DEFAULT_PERSONA, rules, _STOP_RULE) if part
    )


def _voice_context_blocks(
    tts: TtsAdapter | None, extra: str | None = None
) -> list[RuntimeContextBlock]:
    """The local backend's turn context: what the agent must know to be SPEAKABLE, the
    pre-tool narration nudge, plus the operator's voice-scoped ``context`` lines.

    Core's format-hint ladder has no ``voice`` branch, so without this the agent writes
    for a screen and answers in whatever language the user spoke; markdown the chunker
    strips after the fact, but every on-device engine is fixed to ONE language and
    non-Latin text survives ``sanitize`` intact. Derived from the RESOLVED adapter, not
    config, so a degrade-to-system fallback cannot leave claims true of an engine that
    never loaded; ``extra`` stands alone, so a text-out session (tts.enabled=false) still
    carries it. Both absent => no block."""
    lines: list[str] = []
    if tts is not None:
        lines += [
            "This message was spoken aloud, and your reply is read back through a "
            "text-to-speech engine; it is never displayed.",
            "Write plain conversational prose. Markdown, headings, tables, code blocks, "
            "URLs and emoji are stripped before speaking, so they are heard as nothing "
            "or as mangled words.",
            # The backend detects this spoken status line (agent_prologue) and defers
            # the canned filler behind it.
            "Before tool work that will keep the user waiting — a search, a web "
            "request, multi-step work — say one short sentence about what you are "
            "doing (never implying success or failure) and put NOTHING else in "
            "that message: the tool call follows it. Skip the sentence when you "
            "expect to answer right away.",
            "The message that delivers the answer is pure answer: begin with it. "
            "Never include wait phrases or progress narration (\"One moment\", "
            "\"I'm checking...\") there — it plays after the work is done, so "
            "those words are false and only delay the answer.",
        ]
        langs = getattr(tts, "spoken_languages", None)  # bilingual router
        lang = getattr(tts, "spoken_language", None)
        if langs:
            named = " and ".join(f"'{code}'" for code in langs)
            lines.append(
                f"The speech engine can only pronounce ISO 639-1 languages {named}. "
                "Reply in whichever of these the user speaks (mixing them is fine), "
                "and avoid quoting words in other scripts: they are dropped or "
                "voiced as noise."
            )
        elif lang:
            lines.append(
                f"The speech engine can only pronounce ISO 639-1 language '{lang}'. "
                f"Reply in '{lang}' regardless of the language spoken to you, and "
                "avoid quoting words in other scripts: they are dropped or voiced "
                "as noise."
            )
    if extra and extra.strip():
        lines.append(extra.strip())
    if not lines:
        return []
    return [RuntimeContextBlock(source="voice", content=wrap_runtime_context_lines(lines))]


# Stamped on a delegated ask_nanobot request; the AgentLoop echoes inbound metadata onto
# the turn's FINAL send, so the reply carries the token back, the non-streaming analogue
# of accepts_stream, since a /stop-ped delegation can finish minutes later with a bare
# final send. Mismatch => straggler; absent => accept, as for an unrecognized stream id.
_DELEGATION_META = "_voice_delegation"


class _DelegationCollector:
    """Collects one delegated nanobot turn's reply off the bus (supervisor mode); the
    first terminal the streaming contract produces resolves the future. Streaming ON:
    deltas accumulate and ``_stream_end`` resolves, the final ``_streamed`` send being
    swallowed upstream, never reaching the channel (see nanobot ``bus/outbound_events``).
    Streaming OFF: a single final ``send`` resolves."""

    def __init__(self, metrics: VoiceMetrics) -> None:
        self._future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._parts: list[str] = []
        self._metrics = metrics
        # A supervisor turn is a full AgentLoop run, so its first token is the TTFT analogue.
        self._started_at = time.monotonic()
        self._first_token = False
        # Tombstone: a /stop-ped turn's reply may still be in flight, so a dead collector
        # stays registered to swallow it instead of resolving the NEXT delegation.
        self.dead = False
        # Watermark against a PREVIOUS turn's stragglers: core stream ids embed the turn's
        # start time_ns (see streamid) and any turn answering THIS delegation starts after
        # the collector, so an older base is a stopped turn's queued deltas racing the
        # tombstone swap. An unrecognized id format accepts everything.
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
        """A tool-boundary segment break: separate the reply's parts WITHOUT latching
        the first-token clock — the separator is channel-fabricated, not a model token,
        and latching would under-report TTFT for exactly the tool-first delegations
        supervisor mode exists for."""
        self._parts.append("\n")

    def finish(self, fallback: str = "") -> None:
        text = "".join(self._parts).strip() or (fallback or "").strip()
        if not text:
            # Not a terminal: core also fires on_stream_end(resuming=False) mid-turn on its
            # blank-response retry path, so keep collecting: the real end or
            # delegationTimeoutS decides. Marking a first token would suppress the real one.
            return
        self._mark_first_token()
        self._resolve(text)

    def set_final(self, text: str) -> None:
        """The non-streaming terminal: one whole reply, so it is also first token."""
        self._mark_first_token()
        self._resolve(text)

    def entomb(self) -> None:
        """Mark dead AND latch the first-token flag: a tombstone's late reply must
        neither resolve the next delegation nor time a delegation that failed."""
        self.dead = True
        self._first_token = True

    def abandon(self, text: str) -> None:
        """Release the delegation, latching the first-token flag: the user barged in, so
        no answer was produced, and the cancelled turn's late ``send``/``send_delta``
        (possible until ``_pending_delegation`` is cleared) must not be timed as one."""
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
    # Defaults only: the ChannelManager overwrites all three; `send()` gates traces.
    send_progress = False
    send_tool_hints = False
    show_reasoning = False
    # Core tool-gateway seam: when the ChannelManager injects the gateway, the cloud
    # backend routes the model's tool calls through nanobot's guarded ToolRegistry.
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
        # Local mode only: speakability context stamped on every published utterance; empty
        # for cloud, where the provider both reasons and speaks under its own persona.
        self._voice_context: list[RuntimeContextBlock] = []
        self._warmup_task: asyncio.Task | None = None
        self._metrics_task: asyncio.Task | None = None  # debug.metricsIntervalS reporter
        # Supervisor mode only: the in-flight ask_nanobot delegation whose reply the bus
        # glue collects. One slot, since a bus reply can't be correlated to a specific
        # concurrent delegation, so the lock serializes them.
        self._pending_delegation: _DelegationCollector | None = None
        self._delegation_lock = asyncio.Lock()
        # One per session, shared with backend and shell, so a tool call's segments
        # (seen -> dispatched -> executed -> continuation) join on call_id in one place.
        self._metrics = VoiceMetrics()
        # Optional OTel mirror, resolved once so a disabled exporter costs nothing per call.
        self._tracer = VoiceTracer(self.config.telemetry)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return VoiceConfig().model_dump(by_alias=True)

    # ---- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        self._running = True
        self._stop_event = asyncio.Event()
        if self.config.import_json:
            # A WebUI paste is pending in config.json: expand it into the real section
            # keys and delete the blob. self.config already carries the merged values
            # (the schema folded the paste at parse time), so a failed rewrite only
            # means the blob is consumed on a later start, never a wrong runtime config.
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
                self._stt = make_stt(self.config.stt)
                shell, instructions, tools = self._build_local()
            else:
                # A declared-but-unimplemented kind (a future "gemini") must refuse
                # loudly; falling through to local would silently run the wrong brain.
                raise RuntimeError(f"voice backend kind '{kind}' is not implemented")
            if not self._running:
                self._backend = None
                return  # stop() raced the build; nothing was started yet
            self.logger.info(
                "voice channel starting (backend={}, capture={}, playback={})",
                self.config.backend,
                self.config.audio.capture_device, self.config.audio.playback_device,
            )
            await shell.start(instructions=instructions, tools=tools)
        except BaseException:
            # Must not report healthy, leave a never-started backend registered
            # (speak_final queues into a worker that never runs), or leave a half-started
            # shell holding devices (shell.start's first await already spawned arecord).
            self._running = False
            self._backend = None
            if shell is not None:
                with suppress(Exception):
                    await shell.stop()
            raise
        if not self._running:
            # stop() landed mid-start and found nothing to stop (self._shell is published
            # only below); tear the shell down here.
            await shell.stop()
            self._backend = None
            return
        self._shell = shell
        try:
            server = await self._start_stt_server()
        except BaseException:
            # A configured serve endpoint that cannot bind (port in use, bad host) refuses
            # loudly rather than leave WebUI dictation broken, and takes the shell down.
            self._running = False
            self._shell = None
            self._backend = None
            with suppress(Exception):
                await shell.stop()
            raise
        if not self._running:
            # stop() raced the endpoint coming up: the handle is published only after the
            # bind, so stop() saw None and skipped it; THIS frame owns the listener, so
            # stop it via the local and clear the just-published handle.
            if server is not None:
                with suppress(Exception):
                    await server.stop()
            self._stt_server = None
            self._shell = None
            self._backend = None
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
            # The whisper defaults land here (30 s cap vs a 20 s export); harmless —
            # long captures decode in pieces — but say once where the seams come from.
            self.logger.info(
                "stt decode window ({:.0f}s) is under vad.maxUtteranceMs ({}); longer "
                "utterances are decoded in window-sized pieces cut at the quietest gap",
                window / 1000, self.config.vad.max_utterance_ms,
            )
        if self.config.stt.provider == "zipformer" and self.config.audio.sample_rate != 16000:
            # The streaming path feeds raw capture frames to the model unresampled (only
            # the batch path resamples), so another rate yields time-stretched garbage.
            raise RuntimeError(
                "stt.provider='zipformer' requires audio.sampleRate=16000 "
                f"(configured: {self.config.audio.sample_rate})"
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
        self._voice_context = _voice_context_blocks(tts, self.config.context)
        # A raw-PCM TTS (MMS, openai with audioFormat=pcm) streams gaplessly through one
        # persistent player, no per-chunk aplay spawn; WAV-only adapters keep blob mode.
        pcm_capable = tts is not None and getattr(tts, "output_rate", None) is not None
        audio_sink = AudioSink(sink_dev, mode="stream" if pcm_capable else "blob")
        # Software AEC ([aec] extra), wired twice: the sink feeds it our playback as the
        # reference, the backend runs capture through it. A missing extra or a WAV-blob TTS
        # (its sink never feeds the reference tap) warns and degrades to soft-duplex; NOT
        # building the canceller is the degrade, since open_mic already holds for "webrtc"
        # without it, and a reference-starved canceller cancels nothing while its warmup
        # hold suppresses the early-confirm shortcuts forever (reference_ms() never advances).
        aec_stage = None
        if self.config.aec == "webrtc":
            # AEC3 frames 10 ms blocks: a rate % 100 != 0 (matcha's 22050) would drop
            # every reference block -> the starved canceller described above
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
        # batch on-device adapters get it. The nanobot delegate gets neither: it may be a
        # billed cloud API, where a resumed speaker wastes one call per pause.
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
            # Sharing these stops the shell building its own and re-logging the telemetry banner.
            metrics=self._metrics,
            tracer=self._tracer,
        )
        return shell, "", []

    async def _build_cloud(self) -> tuple[VoiceShell, str, list]:
        rt = self.config.realtime
        profile = resolve_profile(self.config.backend)  # openai/xai/azure/qwen/glm/stepfun
        # Fail fast on STATIC config errors, before any device is claimed: left to the
        # backend they raise in the rx task, where the reconnect ladder mistakes them for
        # transport blips (three retries, then "reconnect exhausted").
        profile.base_url(rt.base_url)  # raises for a provider with no default (Azure)
        if not resolve_openai_key(rt.api_key):
            raise RuntimeError(
                f"no API key for realtime provider '{profile.key}' "
                "(set channels.voice.realtime.apiKey or OPENAI_API_KEY)"
            )
        # The mic stays open for server-VAD barge-in only with echo cancellation: asserted
        # hardware/OS AEC (aecAvailable=true or the top-level aec="hardware", same physical
        # fact) or the software AEC3 canceller (aec="webrtc"). Else the gated fallback,
        # the shell's SPEAKING mic-gate, where barge-in resumes after playback.
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
            # No client-side input_audio_buffer.commit / response.create path exists, so
            # audio would stream up forever and the session would never answer.
            raise RuntimeError(
                "realtime.serverVad=false is not supported: the channel relies on "
                "server-side turn detection to commit audio and create responses."
            )
        open_mic = rt.barge_in == "aec"  # aec => open; gated => shell gates while SPEAKING
        # Capture at the PROVIDER's input rate (24 kHz OpenAI/xAI/Azure, 16 kHz Qwen/GLM);
        # the ALSA `plug` layer resamples the device. Stream-mode playback opens its
        # persistent stream at the provider's output rate, so asymmetric rates are fine.
        audio_cfg = self.config.audio.model_copy(update={"sample_rate": profile.input_rate})
        capture, sink_dev = make_audio(audio_cfg)
        audio_sink = AudioSink(sink_dev, mode="stream")
        if aec_stage is not None:
            audio_sink.set_reference_tap(aec_stage)
        self._backend = RealtimeBackend(
            self.config, sink=audio_sink, profile=profile, metrics=self._metrics,
            aec=aec_stage,
        )
        # Capability is PER MODEL, so a newer generation can enable tools while older
        # ones stay persona-only.
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
        # Supervisor rules only when the delegated tool is actually wired; direct rules
        # only when there are tools whose round-trip needs masking.
        supervisor = rt.tool_mode == "supervisor" and exec_tool is not None
        instructions = _cloud_instructions(
            rt.persona, supervisor=supervisor, has_tools=bool(tools)
        )
        return shell, instructions, tools

    async def _cloud_tools(self, supported: bool, tool_mode: str):
        """(tool_defs, exec_tool) for the realtime model, or ([], None) persona-only.

        ``supported`` is the profile's tool capability: a provider whose function-call
        flow isn't the OpenAI exchange (e.g. Qwen) stays persona-only even with the
        gateway wired. ``tool_mode`` picks the surface: ``"direct"`` declares nanobot's N
        tools, each call a guarded ``execute_tool`` slice with the realtime model planning
        the sequence; ``"supervisor"`` declares ONE tool (``ask_nanobot``) delegating the
        whole request to nanobot's AgentLoop over the bus, so multi-step planning leaves
        the weak model. See ``REPORT-realtime-reasoning-latency.md`` section 6.1."""
        gw = self._tool_gateway
        if gw is None or not supported:
            if gw is not None and not supported:
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

        # Direct mode. The gateway derives the session key from channel/chat_id exactly as
        # the bus does, so cloud tools share the voice session's working dir / memory.
        tools = [ToolDef.from_nanobot_schema(s) for s in await gw.get_tool_definitions()]

        async def exec_tool(name: str, args: str):
            return await gw.execute_tool(
                name, args, channel=self.name, chat_id=self.config.chat_id,
            )

        return tools, exec_tool

    async def _delegate_to_nanobot(self, name: str, args: str) -> str:
        """``ask_nanobot`` handler (supervisor mode): run a full nanobot turn over the bus
        and return its final text for the realtime model to speak.

        The request goes out as an ordinary inbound message, so it runs a complete turn
        (planning, multi-tool, memory, MCP, skills) under the normal guards, sharing the
        voice session; the reply is collected from ``send``/``send_delta`` into the one
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

        # Queuing is its own latency component: a delegation can wait on the lock as long
        # as it later takes to run, and folding the two would blame the AgentLoop.
        queued_at = time.monotonic()
        # Its own budget: an ask_nanobot turn is a full AgentLoop run (tool-heavy ones
        # routinely exceed 30 s), whereas turn_timeout_s is the REALTIME WIRE watchdog.
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
        """Supervisor mode: the user talked over an in-flight ask_nanobot delegation, so
        cancel the moot nanobot turn (``/stop``, as local barge-in does) and release the
        delegation; no-op in direct mode or with nothing delegating."""
        collector = self._pending_delegation
        if collector is None or collector.dead:
            return  # nothing delegating (a tombstone is not a live delegation)
        self._metrics.count("delegation_interrupted")
        await self._publish_stop()
        collector.abandon(_DELEGATION_INTERRUPTED)

    async def _start_stt_server(self):
        """``stt.serve``: expose the loaded on-device STT to nanobot core as a local
        OpenAI-compatible endpoint (WebUI dictation, channel voice notes).

        SINGLETON by construction: the server borrows ``self._stt`` (memory-limited
        targets can't fit two copies), building one only under a cloud backend, which
        never loads STT itself. Returns the server so the caller's reference survives a
        concurrent ``stop()`` clearing the attribute."""
        cfg = self.config.stt.serve
        if not cfg.enabled:
            return None
        if self._stt is None:
            self._stt = make_stt(self.config.stt)  # cloud backend + serve: build ONCE here
        if self._stt is None:
            # Config validation rejects provider='nanobot', so None means the engine
            # degraded (missing model files/deps; make_stt logged why), and an enabled
            # endpoint that silently isn't there breaks WebUI dictation invisibly.
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
        steady state to derive this device's pacing knobs. Failures are logged and ignored
        (both are optimizations, never gates). Capture is already live, so hop-cost
        accounting is held across each saturating burst — and ONLY the bursts:
        ``_calibrate``'s wait-for-IDLE is live conversation and must stay accounted."""
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
                await local.prewarm_fillers()  # gated internally on probe_ok + IDLE
        finally:
            if local is not None:
                local.hold_hop_accounting(False)
        if self.config.perf.calibrate:
            await self._calibrate()

    async def _calibrate(self) -> None:
        """Measure warm STT/TTS on THIS device and hand the numbers to the local backend
        (see DESIGN-local-latency-and-engines.md Part E). Runs only after warmup, keeping
        cold-start noise out of the measurements."""
        local = self._local()
        if local is None:
            return  # cloud paces itself; nothing to derive
        # The probes share the live adapters, which keep STRICTLY one decode in flight, so
        # a real utterance would both be slowed and measured wrong. Best-effort wait.
        for _ in range(30):
            shell = self._shell
            if shell is None or shell.state is VoiceState.IDLE:
                break
            await asyncio.sleep(1.0)
        stt_ms: float | None = None
        tts_rtf: float | None = None
        local.hold_hop_accounting(True)  # probes only: the IDLE wait above stays accounted
        try:
            if self._stt is not None:
                t0 = time.monotonic()
                # Fixed-cost probe: 1 s of silence (16 kHz * 2 B = 32000 B). For a
                # fixed-window model (whisper) this IS the per-utterance decode floor;
                # length-proportional engines underestimate long utterances.
                await self._stt.transcribe(b"\x00" * 32000, 16000)
                stt_ms = (time.monotonic() - t0) * 1000.0
            tts = self._tts_adapter
            # probe_ok is False for cloud-backed adapters: nothing at startup may be
            # billed, and a cloud RTF measures the network, not the box.
            if tts is not None and getattr(tts, "probe_ok", True):
                # The engine's own language: an English probe through a zh/ja
                # lexicon would trip the empty-synth guard below (see tts.base).
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
                    tts_rtf = (time.monotonic() - t0) / audio_s
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
            # Pydantic tracks fields the user SET; an explicit minCharsFirst always wins.
            chunk_floor_pinned="min_chars_first"
            in getattr(self.config.chunker, "model_fields_set", set()),
        )

    async def _metrics_reporter(self, interval_s: float) -> None:
        """``debug.metricsIntervalS``: the live snapshot as one JSON line per interval
        (no transcript content), so distributions are readable during a repro."""
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

    async def stop(self) -> None:
        self._running = False
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
            # Chunked: vad.maxUtteranceMs may exceed the adapter's decode window (the
            # defaults do for whisper: 30 s cap vs a 20 s export).
            return await transcribe_chunked(self._stt, pcm, self.config.audio.sample_rate)
        # No on-device STT: hand a WAV to nanobot's transcription layer (cloud, or a local
        # Whisper server). Off the loop: a ~1 MB mkstemp+wave write and its unlink can
        # stall an SD-card SBC for tens of ms, and capture, VAD and sink pacing share it.
        path = await asyncio.to_thread(write_temp_wav, pcm, self.config.audio.sample_rate)
        try:
            return await self.transcribe_audio(path)
        finally:
            with suppress(OSError):
                await asyncio.to_thread(os.unlink, path)

    async def _publish_turn_text(self, text: str, turn_token: str) -> None:
        """Publish a captured utterance tagged with the turn it opens: core echoes inbound
        metadata onto that turn's final send, so ``send`` can tell the live turn's reply
        from a barged-out one's straggler."""
        await self._publish_user_text(text, metadata={TURN_META: turn_token})

    async def _publish_user_text(self, text: str, metadata: dict[str, Any] | None = None) -> None:
        meta = dict(metadata or {})
        if self._voice_context:
            # Per turn, not once per session: the block rides the user message, so
            # compaction can drop an older copy, and there is no signal here to notice.
            meta[RUNTIME_CONTEXT_INPUT_META] = self._voice_context
        await self._handle_message(
            sender_id=self.config.sender_id,
            chat_id=self.config.chat_id,
            content=text,
            metadata=meta or None,
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
        """The backend when it's the local pipeline (bus glue speaks the reply), else None.

        For cloud the text bus is not the reasoning path, so ``send``/``send_delta`` are
        inert, EXCEPT supervisor mode, which collects a delegated reply before this check.
        """
        return self._backend if isinstance(self._backend, LocalBackend) else None

    async def send(self, msg: OutboundMessage) -> None:
        # A pending delegation's non-streaming terminal (a genuine final send, not a
        # trace); _stream_end in send_delta is the streaming one, first wins. Only OUR
        # session chat qualifies: a delivery routed elsewhere (cron) must not resolve it.
        meta = msg.metadata or {}
        if self._pending_delegation is not None and msg.chat_id == self.config.chat_id:
            echoed = meta.get(_DELEGATION_META)
            if echoed is not None and echoed != self._pending_delegation.token:
                return  # a stopped PREVIOUS delegation's late reply, not our answer
            if _speakable(msg):
                text = (msg.content or "").strip()
                if text:
                    self._pending_delegation.set_final(text)
            return
        local = self._local()
        if local is None or not _speakable(msg):
            return
        turn = meta.get(TURN_META)
        if turn is not None and local.is_dead_turn(turn):
            return  # a killed turn's late final; a superseded-but-live turn still speaks
        text = (msg.content or "").strip()
        if text:
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
        # The manager passes stream framing (stream_id/stream_end/resuming) as kwargs
        # unconditionally, so declaring these parameters is load-bearing: an override
        # without them fails delivery on every delta. Supervisor mode accumulates the
        # delegated reply here; a resuming end is only a tool boundary, and resolving there
        # would truncate to the pre-tool status line.
        if self._pending_delegation is not None and chat_id == self.config.chat_id:
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
        if stream_end:
            await local.on_stream_end(resuming=resuming, stream_id=stream_id)
        else:
            await local.on_delta(delta, stream_id=stream_id)
