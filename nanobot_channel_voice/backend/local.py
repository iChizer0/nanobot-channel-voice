"""LocalBackend: the on-box VAD/STT/TTS pipeline + nanobot over the text bus.

Owns the authoritative turn lifecycle (CAPTURING -> THINKING -> SPEAKING -> drain -> IDLE),
barge-in, the self-echo filter, the stale-turn guards and the TTS stage; the shell mirrors state
via ``StateHint`` events. Audio leaves as ``OutputAudio`` events (the shell enqueues them on the
shared sink); the direct sink reference here is for control and pacing (flush, duck/pause,
epoch, backlog/drain), never for enqueueing audio. STT/TTS thread inside their adapters, so
every callback here stays on the event loop."""

from __future__ import annotations

import asyncio
import io
import math
import threading
import time
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from nanobot_channel_voice.aio import (
    Throttle,
    cancel_and_wait,
    cancel_task,
    put_drop_oldest,
    wait_for_stall,
)
from nanobot_channel_voice.audio.pcm import pcm_ms, pcm_rms, pcm_to_wav_bytes, wav_duration_ms
from nanobot_channel_voice.chunker import SentenceChunker
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.dump import AudioDumper, default_dump_root
from nanobot_channel_voice.echo_reject import SelfEchoFilter
from nanobot_channel_voice.metrics import VoiceMetrics
from nanobot_channel_voice.phrases import (
    FILLER_WORDS,
    PhraseLexicon,
    PhraseMatcher,
    tokens_of,
)
from nanobot_channel_voice.streamid import base_of, started_ns, unique_token
from nanobot_channel_voice.tts.base import TtsAdapter, is_wav
from nanobot_channel_voice.vad import Endpointer, Vad, flag_lag_ms, resolve_preroll_ms
from nanobot_channel_voice.vad.adaptive import AdaptiveHangover

from .audio_sink import AudioSink, scale_pcm, trim_lead_silence
from .base import OnEvent, OutputAudio, ToolDef, VoiceState
from .common import TurnEventMixin, loggable_text

TranscribeFn = Callable[[bytes], Awaitable[str]]
PublishTextFn = Callable[[str, str], Awaitable[None]]
InterruptFn = Callable[[], Awaitable[None]]

# Inbound-metadata key carrying the publishing turn's token; core echoes it onto that turn's
# FINAL send, so the channel can tell a live reply from a barged-out turn's straggler — the
# non-streamed analogue of the stream-id watermark (a final ``send`` carries no stream id).
TURN_META = "_voice_turn"

# Synthesis-ahead cap for the TTS worker (stream mode): the sink paces the DEVICE (240/120 ms
# lead) but nothing paces SYNTHESIS, so a fast adapter renders a whole reply that a barge-in
# then flushes — wasted NPU/CPU cycles and cloud billing. 4 s dwarfs any sane
# tts_first_chunk_ms, so the gate can never starve playback into an audible gap.
_SYNTH_BACKLOG_MS = 4000.0
_BACKLOG_POLL_S = 0.25

# AEC3 needs a few seconds of double-talk-free REFERENCE AUDIO to converge; until then its
# residual can transcribe as "fresh words", so the early-confirm shortcuts hold back — the
# endpoint verdict still decides, so genuine barge-in is delayed to the endpoint, not lost.
# Measured in accepted reference audio, not wall time: silence teaches the filter nothing.
_AEC_WARMUP_REF_MS = 3000.0

# Capture debt = wall time the frame hop has overrun its budget and not yet paid back by
# draining the pipe faster than real time; response latency grows by exactly this much.
# Warn well below the ~2 s ALSA pipe. A pump-idle arrival gap (mic gate, capture restart)
# means the pipe was flushed, so debt resets instead of reading as lag.
_CAPTURE_DEBT_WARN_MS = 500.0
_PUMP_GAP_RESET_MS = 1000.0
# The pipeline is finite (arecord's 64 KB kernel pipe ~= 2 s at 16 kHz S16LE; the pyalsa
# queue drops-oldest at ~1 s), so real lag saturates at its depth and any further deficit
# is DROPPED audio, not more latency. An uncapped integral reports fiction on a device
# chronically a hair over budget ("~17615 ms behind" after a few minutes).
_CAPTURE_DEBT_CAP_MS = 2000.0

# A pure stop command targets a live reply — or one stopped moments ago ("stop... STOP!"
# double-taps land after the first kill already IDLEd the session). Anchored ONLY to kills
# a stop consume performed: a content barge-in's kill must not arm it, or "cancel" spoken
# as the ANSWER to a question the agent asks right after being interrupted gets swallowed;
# and a consume that killed nothing must not extend the chain, or consecutive cold
# "wait"s while IDLE are swallowed indefinitely.
_KILL_GRACE_S = 3.0

# After a pause-probe/early release the resumed playback re-leaks into the open mic
# immediately; without a short engage holdoff the candidate loop flaps at ~1 Hz. A genuine
# onset during the holdoff waits at most this long: engagement is state-driven (any armed,
# unclaimed open utterance engages), so expiry re-engages mid-utterance.
_ENGAGE_HOLDOFF_S = 0.5

# Early-RELEASE (the acquittal twin of the min-words early confirm), DUCK MODE ONLY: this
# many consecutive streaming-partial polls with zero fresh words, no sooner than this far
# into the candidate, restore the level before the endpoint verdict. The floor absorbs
# decoder partial latency, and a slower-than-floor first partial costs only level pumping
# in duck mode; pause mode takes no transcript acquittal at all — a wrong one there DROPS
# real speech, and the pause-probe owns that mode.
_RELEASE_POLLS = 2
_EARLY_RELEASE_MS = 600.0

# Pause-probe: sustained-silence floor before the leak attribution is read. Above VAD
# flicker gaps inside real speech (plosives/inter-word, <~150 ms); the attribution itself
# is the leak-death anchor, not this floor.
_PROBE_SILENCE_MS = 200.0

# High false-candidate rate = the operator-visible symptom of weak/missing AEC: name the
# way out in the log instead of leaving a stuttering session to metrics archaeology. Only
# leak-shaped acquittals count — backchannels/blips/pre-onset suspicion deaths are normal
# conversation, not echo evidence.
_FALSE_WARN_WINDOW_S = 60.0
_FALSE_WARN_N = 10
_LEAK_REASONS = frozenset({"probe", "partial", "eager", "echo", "empty"})


def _swallow_result(task: asyncio.Task) -> None:
    """Retrieve an abandoned speculative decode's outcome, so it never logs 'exception was never
    retrieved'."""
    if not task.cancelled():
        task.exception()


def _interrupt_marker(heard: str | None) -> str | None:
    """The heard-up-to note riding an interrupting utterance's publish. ``heard`` is the
    heard text ("" = cut before anything sounded), None = marker disabled/blob mode."""
    if heard is None:
        return None
    return (
        f'[note: you were interrupted mid-reply; the user heard only: "{heard}"]'
        if heard
        else "[note: you were interrupted before your reply was heard]"
    )


def _stop_note(stop_text: str, heard: str | None) -> str:
    """The pending note a CONSUMED stop leaves for the NEXT publish: a consumed command
    elicits no reply, so the heard-up-to contract rides the turn that follows. The trailing
    clause stops the next reply opening with "as I was saying...". ``heard`` None means
    accounting was unavailable (blob mode / heardMarker off) — make NO claim then; only
    "" may claim the cut landed before anything sounded."""
    if heard:
        return (
            f'[note: the user stopped your previous reply with "{stop_text}"; '
            f'they heard only: "{heard}"; do not resume it unless asked]'
        )
    if heard == "":
        return (
            f'[note: the user stopped your previous reply with "{stop_text}" '
            "before hearing it; do not resume it unless asked]"
        )
    return (
        f'[note: the user stopped your previous reply with "{stop_text}"; '
        "do not resume it unless asked]"
    )


@dataclass(slots=True)
class _PendingUtterance:
    """One endpointed utterance, snapshotted AT CLOSE TIME so processing can be deferred: the
    eager/stream-finish task belongs to THIS utterance, and ``closed_at`` back-dates the metrics
    anchor past any queue wait. ``eager_always_valid``: a STREAMING adapter's finish task saw
    exactly this utterance's audio, so its transcript is valid however the utterance closed
    (unlike eager speculation, which a max-length close invalidates)."""

    pcm: bytes
    eager: asyncio.Task | None
    closed_reason: str  # Endpointer.closed_reason: "silence" | "max" | "eou"
    closed_at: float
    silence_ms: int = 0  # trailing silence the close consumed (Endpointer.closed_silence_ms)
    raw: bytes | None = None  # pre-AEC span of pcm, for the audio dump (None = no AEC/dump)
    learn_ms: float | None = None  # adaptive-hangover candidate bound to THIS utterance
    eager_always_valid: bool = False
    # Diagnostics snapshotted at close (endpointer counters die in its reset):
    # speech-flagged ms, VAD probability stats, the segment id shared by the
    # summary log line and the dump filename, and the verdict metadata the
    # summary builds for the dump manifest.
    active_ms: int = 0
    prob_peak: float | None = None
    prob_mean: float | None = None
    seg_id: int = 0
    meta: dict | None = None
    # Early-confirm state, bound AT CLOSE: as instance-global latches a DIFFERENT queued
    # utterance's intake could consume them (slow STT, two in flight), skipping a needed /stop.
    preempted: bool = False
    heard: str | None = None
    # Turn state AT VAD ONSET (plus its wall time): whether a stop command targets a live
    # reply is decided by when the user STARTED speaking, not by what survived until the
    # verdict — a reply draining during the stop's own STT window must not launder the stop
    # into a cold turn.
    onset_interrupting: bool = False
    onset_at: float = 0.0


class _Turn:
    """Everything owned by ONE published turn: stage latches, the per-segment spoke flag, the
    filler/tool-boundary watchers, the turn's stream base. Created FRESH at every publish (so
    no reset can be forgotten) and abandoned as a unit on barge-in. ``idle()`` is the
    before-first-publish placeholder (same shape, latches dead), so readers never need a None
    guard."""

    __slots__ = (
        "published_at", "chunk_await", "tts_first_pending", "await_first_token",
        "segment_spoke", "prologue_task", "midturn_task", "timeout_task", "base",
        "last_activity", "dead", "token", "audible_at", "continuation_pending",
    )

    def __init__(self, token: str = ""):
        self.published_at = time.monotonic()
        self.token = token  # echoed back on the turn's final send (see TURN_META)
        self.dead = False   # abandoned by a barge-in; nothing may interrupt it twice
        # One-shot TTFA stage timers, consumed by the turn's first speakable chunk, first
        # synthesis and first delta respectively.
        self.chunk_await = True
        self.tts_first_pending = True
        self.await_first_token = True
        self.segment_spoke = False  # current stream segment produced audible chunks
        self.prologue_task: asyncio.Task | None = None
        self.midturn_task: asyncio.Task | None = None
        self.timeout_task: asyncio.Task | None = None  # stalled-agent deadman
        self.base: str | None = None  # learned from the first delta's stream id
        self.last_activity = self.published_at  # any delta/segment end pushes this
        self.audible_at: float | None = None  # first audio frame emitted (adaptive hangover)
        # A resuming stream end passed: the next delta (the earliest observable resume
        # edge) re-anchors the metrics clock so tool time never lands in ttfa_ms.
        self.continuation_pending = False

    @classmethod
    def idle(cls) -> _Turn:
        turn = cls()
        turn.published_at = 0.0
        turn.chunk_await = turn.tts_first_pending = turn.await_first_token = False
        return turn

    def cancel_prologue(self) -> None:
        task, self.prologue_task = self.prologue_task, None
        cancel_task(task)

    def cancel_midturn(self) -> None:
        task, self.midturn_task = self.midturn_task, None
        cancel_task(task)

    def cancel_timeout(self) -> None:
        task, self.timeout_task = self.timeout_task, None
        cancel_task(task)

    def abandon(self) -> None:
        """Barge-in: this turn is dead; nothing it owns may fire or count."""
        self.dead = True
        self.cancel_prologue()
        self.cancel_midturn()
        self.cancel_timeout()
        self.segment_spoke = False
        self.chunk_await = self.tts_first_pending = self.await_first_token = False
        self.continuation_pending = False


def _scale_wav(wav_bytes: bytes, gain: float) -> bytes:
    """Attenuate a 16-bit WAV blob by linear *gain* (0..1), for ducking."""
    if gain >= 1.0 or not is_wav(wav_bytes):
        return wav_bytes
    try:
        with wave.open(io.BytesIO(wav_bytes), "rb") as w:
            if w.getsampwidth() != 2:
                return wav_bytes  # scale_pcm assumes S16
            rate, channels = w.getframerate(), w.getnchannels()
            pcm = w.readframes(w.getnframes())
        # Keep the source geometry: a stereo blob re-wrapped as mono would play at half speed.
        return pcm_to_wav_bytes(scale_pcm(pcm, gain), rate, channels=channels)
    except Exception:  # noqa: BLE001 - never let ducking break playback
        return wav_bytes


class LocalBackend(TurnEventMixin):
    def __init__(
        self,
        config: VoiceConfig,
        *,
        vad: Vad,
        tts: TtsAdapter | None,
        sink: AudioSink,
        transcribe: TranscribeFn,
        publish_text: PublishTextFn,
        interrupt: InterruptFn,
        metrics: VoiceMetrics | None = None,
        eager_ms: int = 0,
        stt_stream=None,  # a streaming=True SttAdapter, or None (batch STT)
        aec=None,  # an EchoCanceller front-end (aec="webrtc"), or None
        turn_analyzer=None,  # a SmartTurnAnalyzer, or None (silence-only endpointing)
    ):
        self._cfg = config
        self._tts = tts
        self._sink = sink
        self._transcribe = transcribe
        self._publish_text = publish_text
        self._interrupt = interrupt
        # Shared with the channel: local TTFA lands in the same collector as the cloud path's.
        self._metrics = metrics if metrics is not None else VoiceMetrics()

        # Software AEC: every capture frame passes through it before the VAD/STT (same off-loop
        # hop), the sink feeding it our playback as the reference.
        self._aec = aec
        self._full_duplex = config.full_duplex or aec is not None
        # Mic open while speaking -> the echo filter stops the bot barging in on itself (with
        # AEC, on the residual); user speech still interrupts.
        self._open_mic = config.open_mic
        self._echo = SelfEchoFilter(config.echo_reject_threshold)
        self._vad = vad  # kept for duck floor scaling and release() at close
        # Continuation hysteresis: half the onset bar while a reply is pending (THINKING), so
        # quick "...and also--" follow-ups confirm faster.
        self._cont_start_frames = max(1, config.vad.start_frames // 2)
        self._vad_heavy = getattr(vad, "heavy", False)  # neural VAD -> run per-frame off the loop
        # The adaptive-hangover floor (see vad.adaptive) may never undercut the consult tier, or
        # the silence close preempts the turn model on every pause and it silently never runs.
        hangover_floor = config.vad.hangover_min_ms or config.vad.hangover_ms
        if turn_analyzer is not None:
            tier_floor = config.vad.turn.consult_ms + config.audio.frame_ms
            if hangover_floor < tier_floor:
                logger.warning(
                    "voice: vad.hangoverMinMs ({}) undercuts vad.turn.consultMs ({}); "
                    "raising the adaptive floor to {} ms so the turn model can fire",
                    hangover_floor, config.vad.turn.consult_ms, tier_floor,
                )
                hangover_floor = tier_floor
        self._endpointer = Endpointer(
            vad,
            frame_ms=config.audio.frame_ms,
            start_frames=config.vad.start_frames,
            min_utterance_ms=config.vad.min_utterance_ms,
            max_utterance_ms=config.vad.max_utterance_ms,
            hangover_ms=hangover_floor,
            preroll_ms=resolve_preroll_ms(config.vad, config.audio.frame_ms),
            eager_ms=eager_ms,  # 0 for cloud-delegated STT (never speculate against billed calls)
            consult_ms=config.vad.turn.consult_ms if turn_analyzer is not None else 0,
            consult_cap_bytes=getattr(turn_analyzer, "window_bytes", 0),
        )
        self._adaptive = (
            AdaptiveHangover(hangover_floor, config.vad.hangover_ms)
            if config.vad.hangover_min_ms is not None else None
        )
        self._eager_ms = eager_ms
        # End-of-turn model (vad.turn): consulted once per pause off-loop; a COMPLETE verdict
        # parks its gen in _eou_close_gen for the next frame to consume (GIL-atomic write, the
        # same cross-thread pattern as _early_confirm).
        self._turn_analyzer = turn_analyzer
        self._consult_task: asyncio.Task | None = None
        self._eou_close_gen: int | None = None
        self._consult_fail_throttle = Throttle()
        # Streaming STT decodes DURING speech; only the tail flush is left at the endpoint. At
        # onset the ring below is replayed into the fresh stream so it hears the same audio the
        # utterance keeps.
        self._stt_stream = stt_stream
        # The LIVE utterance's caller-owned handle (see stt.base.SttStream): fresh at each
        # onset, taken (for its finish thread) at the endpoint, dropped for a rejected blip.
        # Writes are serialized by the one-frame-at-a-time push await plus the hop lock.
        self._stt_live = None
        frame_ms = config.audio.frame_ms
        ring = (resolve_preroll_ms(config.vad, frame_ms) // frame_ms) + config.vad.start_frames
        self._recent: deque[bytes] = deque(maxlen=max(1, ring))
        # Eager (speculative) STT of the utterance-so-far, started at the eager mark, consumed
        # (or discarded) at the endpoint. STRICTLY one decode in flight: it cannot be aborted,
        # and with a slow ASR (whisper RTF ~0.6 on device) stacking starves the one that
        # matters.
        self._eager_task: asyncio.Task | None = None
        self._eager_valid = False  # the in-flight task belongs to the CURRENT silence run
        self._worker_decoding = False  # the utterance worker's final decode is in flight
        # The pump enqueues, the worker serializes: an INLINE multi-second STT (RTF 0.6 x a 5 s
        # utterance = 3 s) would starve the arecord pipe (~2 s of buffer at 16 kHz) into
        # overruns and leave the VAD deaf to barge-ins.
        self._utt_queue: asyncio.Queue[_PendingUtterance] = asyncio.Queue(maxsize=4)
        self._utt_task: asyncio.Task | None = None
        # Capture-segment id, shared by the summary log line and the dump filename:
        # the dumper's own write order can trail capture order.
        self._seg_seq = 0
        self._chunker = SentenceChunker(
            config.chunker.min_chars, config.chunker.max_chars, config.chunker.min_chars_first
        )
        # Open-mic ducking floor: stream mode ducks toward it DYNAMICALLY while a candidate
        # interrupt is live; blob mode can't change gain mid-chunk, so it pre-bakes the gain.
        duck_db = config.duck_db if self._open_mic else 0.0
        self._duck_gain = 10.0 ** (duck_db / 20.0) if duck_db < 0 else 1.0
        sink.configure_duck(self._duck_gain)
        # Pause-then-confirm (bargeIn.mode="pause"): the confirm window stops the stream instead
        # of attenuating it — the leak vanishes; a false verdict resumes where playback stopped.
        self._duck_pause = (
            config.barge_in.mode == "pause" and self._open_mic and sink.stream_mode
        )
        if self._duck_pause:
            sink.configure_pause(True)
        self._duck_onset: float | None = None  # anchor of the live candidate, if any
        # Engaged on SUSPICION (a pre-onset run of bargeIn.duckStartFrames), cleared when the
        # onset confirms: only a suspicion duck may be released by its run dying pre-onset, a
        # confirmed one stays until the verdict.
        self._duck_suspect = False
        self._duck_frames = config.barge_in.duck_start_frames
        # An early confirm already interrupted; _on_utterance must not repeat it.
        self._preempted = False
        self._early_confirm = False  # set by the frame hop or eager callback, consumed on the loop
        self._partial_countdown = 0
        # Frames per streaming-partial poll: ~100 ms of audio at any frame size.
        self._partial_every = max(1, 100 // max(1, config.audio.frame_ms))
        # Serializes the frame hop against loop-side resets.
        self._hop_lock = threading.Lock()
        # Does per-frame work thread at all? The LIGHT path (energy/webrtc VAD, batch STT, no
        # AEC) pushes the endpointer on the loop, so its resets must run inline too, or the lock
        # protects only one side of the race.
        self._threaded_hop = (
            self._vad_heavy or self._stt_stream is not None or self._aec is not None
        )
        # Backchannel ignore-list and stop-command lexicon (see _on_utterance's ladder).
        # Matchers precompute the merged vocabularies once: the per-call unions were
        # churn on the frame-hop poll path.
        self._ack_lex = PhraseLexicon(config.barge_in.ack_phrases)
        self._stop_lex = PhraseLexicon(config.barge_in.stop_phrases)
        self._ack_words = self._ack_lex.words
        self._ack_match = PhraseMatcher(self._ack_lex)
        self._stop_match = PhraseMatcher(
            self._stop_lex, self._ack_lex, extra=FILLER_WORDS
        )
        self._min_fresh_words = config.barge_in.min_words
        # Stop-command targeting: turn state latched at VAD onset (see _PendingUtterance),
        # and the wall time of the last stop-consume kill for the double-tap grace.
        self._onset_interrupting = False
        self._onset_at = 0.0
        self._last_kill = float("-inf")
        # A consumed stop's heard-up-to note, appended to the NEXT publish.
        self._pending_note: str | None = None
        # Early-RELEASE flag (reason string), the acquittal twin of _early_confirm: set by
        # the frame hop or the eager callback, consumed on the loop (duck mode only).
        self._early_release: str | None = None
        self._empty_polls = 0
        # An early release acquitted the still-open utterance: suppress state-driven
        # re-engagement until it closes, or acquit/engage flaps for its whole length.
        self._acquitted_open = False
        # Pause-probe leak-death window, derived (never a knob): after a pause engages,
        # leak can keep flagging for exactly the sink's write-ahead + device playout +
        # the VAD's decision lag. Speech whose LAST flag falls inside that window is
        # attributable to the buffered tail; speech flagged beyond it is a person.
        self._leak_death_ms = max(
            200.0,
            sink.lead_ms()
            + config.audio.playout_delay_ms
            + flag_lag_ms(config.vad, config.audio.frame_ms),
        )
        # Suspicion engage lands duckStartFrames-1 frames after the run's first flag.
        self._engage_skew_ms = (
            (config.barge_in.duck_start_frames - 1) * config.audio.frame_ms
        )
        self._probe_holdoff_until = 0.0
        # False-candidate rate window for the operator warning (P0c).
        self._false_times: deque[float] = deque()
        self._false_warn = Throttle(_FALSE_WARN_WINDOW_S)
        # Heard-up-to accounting (stream mode): (chunk text, duration ms) of the CURRENT segment
        # in playback order. A confirmed barge-in maps the sink's played_ms through these into
        # "what the user actually heard" and appends it as a bracketed note: the stand-in for
        # history truncation (core keeps the full text; a channel can't edit history).
        self._spoken_spans: list[tuple[str, float]] = []
        # played_ms offset of the segment's first chunk: the stream may predate the segment (a
        # cancelled filler's stream gets reused), so spans map from played-base, not stream
        # open. _spans_gen pins the stream it was measured on: a fresh stream restarts
        # played_ms() at 0 and voids the base.
        self._spans_base_ms = 0.0
        self._spans_gen = -1
        # Text of this turn's PRIOR segments that played out (folded in by _settle): without it
        # a barge-in during a tool wait reports "nothing heard" after a fully-heard status line.
        self._heard_prefix = ""
        self._early_heard: str | None = None  # heard text at an EARLY confirm, consumed at close

        # Epoch tagged at enqueue, so a chunk queued before a barge-in is dropped BEFORE
        # synthesis (the worker re-checks after it, too).
        self._tts_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        self._tts_task: asyncio.Task | None = None
        self._drain_task: asyncio.Task | None = None

        self._turn = VoiceState.IDLE
        self._on_event: OnEvent | None = None
        self._closing = False
        self._cur_turn = _Turn.idle()
        # Tokens of KILLED turns (barge-in / timeout), for send()'s stale-reply gate. A merely
        # superseded turn is not here: core may coalesce a queued follow-up into its
        # still-running turn, and that combined reply must speak. Bounded: with agentTimeoutS
        # set (the default), no straggler outlives a few turns.
        self._dead_tokens: deque[str] = deque(maxlen=8)
        # Raw-PCM output follows the SINK's mode (the channel derived it from the TTS adapter),
        # so we can never emit pcm into a blob sink or vice versa.
        self._pcm_out = sink.stream_mode
        # Phrase -> audio; session-scoped. Filled by prewarm_fillers() at channel
        # warmup (probe_ok engines only) and lazily on first use otherwise.
        self._fillers: dict[str, bytes] = {}
        # Stream identity is "<turn-base>:<segment>", the base stable across a turn. The live
        # base rides the _Turn; the barged-out base stays here so a DEAD turn's late deltas keep
        # dropping after the turn object is gone.
        self._rejected_base: str | None = None
        # Watermark over the base's embedded start time (time_ns), covering what _rejected_base
        # cannot: a barge-in DURING THINKING never learned the cancelled turn's base, so its
        # late deltas would garble the new turn.
        self._reject_started_before_ns = 0
        # Frame-hop accounting: compute (inside the hop lock) vs overhead (executor
        # dispatch, lock wait, loop resume) EMAs attribute a slow hop to the engine or to
        # contention; capture debt is the resulting real backlog, and gates the warning.
        self._hop_compute_ema = 0.0
        self._hop_overhead_ema = 0.0
        self._capture_debt_ms = 0.0
        self._debt_episode = False  # currently in a warned lag episode (metric edge)
        self._last_push_end = 0.0
        self._probe_hold = False  # warmup/calibration probes: drop their hop samples
        self._warn_throttle = Throttle()
        self._log = logger.bind(component="voice")
        # Audio dump (debug.dumpAudio): every endpointed segment leaves as a
        # verdict-named WAV for by-ear false-barge-in triage. A setup failure costs
        # the diagnostics, never the session.
        self._dumper: AudioDumper | None = None
        if config.debug.dump_audio:
            try:
                root = (
                    Path(config.debug.dump_dir).expanduser()
                    if config.debug.dump_dir
                    else default_dump_root()
                )
                self._dumper = AudioDumper(
                    root,
                    config.audio.sample_rate,
                    config.debug.dump_max_mb * 1024 * 1024,
                )
            except Exception as exc:  # noqa: BLE001
                self._log.warning("audio dump disabled ({})", exc)
        self._endpointer.keep_rejected = self._dumper is not None
        # Rolling pre-AEC mirror of capture for the .raw.wav twins, sized to one
        # whole segment (pre-roll + confirm + max utterance). A segment is always
        # the stream's trailing bytes, so its twin is a tail slice; without AEC the
        # segment already IS the raw audio and no ring is kept.
        self._dump_raw: deque[bytes] | None = None
        if self._dumper is not None and aec is not None:
            self._dump_raw = deque(
                maxlen=config.vad.start_frames
                + resolve_preroll_ms(config.vad, frame_ms) // frame_ms
                + config.vad.max_utterance_ms // frame_ms
                + 2
            )

    def hold_hop_accounting(self, active: bool) -> None:
        """Warmup/calibration probes deliberately saturate the device while capture is
        already live; their hop samples would indict the steady state, so drop them.
        Release also discards debt from around the burst: its backlog drains in
        milliseconds once the box idles, so carrying it would only seed a false warning."""
        self._probe_hold = active
        if not active:
            self._capture_debt_ms = 0.0

    def note_capture_flush(self) -> None:
        """The shell flushed the capture source (mic-gate reopen): the pipe's backlog
        was physically discarded, and the debt that described it goes with it."""
        self._capture_debt_ms = 0.0

    def apply_calibration(
        self,
        *,
        stt_cost_ms: float | None,
        tts_rtf: float | None,
        chunk_floor_pinned: bool,
    ) -> None:
        """Feed warm steady-state measurements into the PACING knobs (perf.calibrate).

        Only pacing is ever derived: knobs whose worst case is latency or smoothness.
        Correctness knobs (hangover, thresholds, duplex, echo) stay put; auto-tuning them makes
        behavior unreproducible. An explicitly configured value wins (``chunk_floor_pinned``)."""
        parts: list[str] = []
        if tts_rtf is not None and tts_rtf > 0:
            parts.append(f"tts_rtf={tts_rtf:.2f}")
            if not chunk_floor_pinned:
                cfg = self._cfg.chunker
                # Chunk 1's PLAYBACK must cover chunk 2's SYNTHESIS (chars scale with duration):
                # chars1 >= safety * rtf * minChars. Fast TTS keeps the small TTFA-driven floor;
                # slow TTS grows it so the reply never gaps right after the first chunk.
                floor = math.ceil(1.2 * tts_rtf * cfg.min_chars)
                eff = max(cfg.min_chars_first, min(cfg.min_chars, floor))
                if eff != cfg.min_chars_first:
                    self._chunker.set_first_floor(eff)
                    parts.append(f"minCharsFirst {cfg.min_chars_first}->{eff}")
        if stt_cost_ms is not None:
            parts.append(f"stt~{stt_cost_ms:.0f}ms")
            window = max(0, self._cfg.vad.hangover_ms - self._eager_ms)
            if self._eager_ms and stt_cost_ms > window + 250:
                # The fix at this speed is a faster engine, not more overlap (see
                # DESIGN-local-latency-and-engines.md section A.5).
                parts.append(
                    f"note: eager overlap hides only ~{window}ms of that decode"
                )
        if parts:
            self._log.info("perf calibration: {}", "; ".join(parts))

    def is_dead_turn(self, token: str) -> bool:
        """Was this token's turn killed? Its late final must stay silent (see
        :data:`TURN_META`). Registered synchronously with ``abandon()``, so there is no
        window in which a just-killed turn still passes."""
        return token in self._dead_tokens

    def _is_rejected(self, base: str | None) -> bool:
        """Does this stream belong to a barged-out turn?

        Exact match first, else the watermark: the base embeds the turn's start ``time_ns`` (see
        :mod:`..streamid`), so a turn that STARTED before the last barge-in was already
        /stop-ped even if it never produced a delta while live. A base without the timestamp
        (upstream format change) skips the watermark rather than over-rejecting."""
        if base is None:
            return False
        if base == self._rejected_base:
            return True
        ns = started_ns(base)
        return ns is not None and ns < self._reject_started_before_ns

    # ---- VoiceBackend contract ------------------------------------------

    async def start(
        self, *, instructions: str | None, tools: list[ToolDef], on_event: OnEvent
    ) -> None:
        # instructions/tools are ignored: nanobot owns persona + tools (text bus).
        self._on_event = on_event
        self._closing = False
        if self._tts_task is None:
            self._tts_task = asyncio.create_task(self._tts_worker())
        if self._utt_task is None:
            self._utt_task = asyncio.create_task(self._utterance_worker())

    async def push_audio(self, pcm: bytes) -> None:
        prev_speech = self._endpointer.in_speech
        # Written on the loop BEFORE the hop; the await is the happens-before edge.
        self._endpointer.start_frames_override = (
            self._cont_start_frames if self._turn is VoiceState.THINKING else None
        )
        t0 = time.monotonic()
        # Off-loop so heavy VAD/streaming STT can't stall delta/TTS handling or the sink. Frames
        # are awaited one at a time, so the endpointer/stream is never entered concurrently.
        if self._threaded_hop:
            utterance, compute_ms = await asyncio.to_thread(
                self._push_frame_sync, pcm, prev_speech
            )
        else:
            utterance = self._push_with_model_close(pcm)  # light path: loop-side, no lock
            compute_ms = (time.monotonic() - t0) * 1000.0
        self._track_hop_cost(t0, compute_ms)
        if self._turn_analyzer is not None:
            snap = self._endpointer.take_consult()
            if snap is not None:
                if self._consult_task is None or self._consult_task.done():
                    self._consult_task = asyncio.create_task(self._run_consult(*snap))
                else:
                    # A previous pause's inference is still chewing: skip rather than
                    # stack (the gen guard would reject its verdict anyway).
                    self._metrics.count("eou_consult_skipped")
        if self._endpointer.in_speech and not prev_speech:
            # Stop-command targeting is decided by the state NOW, at onset (see
            # _PendingUtterance.onset_interrupting).
            self._onset_at = time.monotonic()
            self._onset_interrupting = self._turn in (
                VoiceState.THINKING, VoiceState.SPEAKING,
            )
            if self._adaptive is not None:
                # CAPTURING here means a PREVIOUS utterance is still in its STT/queue window (a
                # fresh onset sees IDLE): exactly the fast-resume the learner exists to catch.
                self._adaptive.on_onset(
                    awaiting_reply=self._turn in (VoiceState.THINKING, VoiceState.CAPTURING),
                    speaking=self._turn is VoiceState.SPEAKING,
                    audible_at=self._cur_turn.audible_at,
                )
            if self._turn is VoiceState.IDLE:
                await self._set_turn(VoiceState.CAPTURING)
                self._log.info("vad_start")
        elif not self._endpointer.in_speech:
            if (
                self._duck_suspect
                and self._duck_onset is not None
                and self._endpointer.speech_run == 0
            ):
                # The suspicion run died before onset: no candidate remains. (A CONFIRMED duck
                # reaches here too, but with suspect cleared.)
                self._duck_suspect = False
                self._release_duck("suspect")
            elif (
                self._duck_onset is None
                and self._endpointer.speech_run >= self._duck_frames
                and self._duck_armed()
            ):
                # Duck on SUSPICION, a few frames before the onset confirms: a false dip costs
                # one attack/release cycle, waiting out vad.startFrames talks over the user.
                self._engage_duck(suspect=True)
        if not self._endpointer.in_speech:
            self._acquitted_open = False  # the acquittal latch dies with its utterance
        elif (
            self._duck_onset is None
            and not self._acquitted_open
            and self._duck_armed()
        ):
            # Stage 1 of the two-stage barge-in: yield NOW, reversibly; the verdict in
            # _on_utterance confirms (kill + /stop) or releases. State-driven, not
            # edge-driven: an onset whose edge fell inside the post-acquittal holdoff
            # still engages the first frame the holdoff expires.
            self._engage_duck(suspect=False)
        if self._early_confirm:
            # The min-words gate hit mid-utterance: stop audio + /stop now; _on_utterance still
            # publishes the full transcript at the close. The utterance must still be OPEN (or
            # closing on THIS frame): a confirm landing after a past close would ride the WRONG
            # future utterance.
            self._early_confirm = False
            if (self._endpointer.in_speech or utterance is not None) and self._turn in (
                VoiceState.SPEAKING,
                VoiceState.THINKING,
            ):
                self._preempted = True
                self._metrics.count("barge_in_early_confirm")
                self._early_heard = await self._do_interrupt()
            # else: the reply finished (drain won the race); the utterance still publishes.
        release, self._early_release = self._early_release, None
        if release is not None and not self._duck_pause and self._candidate_contested():
            # Transcript-based acquittal before the endpoint verdict — duck mode only: a
            # wrong one costs level pumping, and the utterance rides to its verdict. Pause
            # mode gets NO transcript acquittal (decoder latency would drop real speech);
            # the probe below owns that mode.
            self._release_duck(release)
            self._acquitted_open = True  # don't re-engage over the acquitted utterance
        if self._duck_pause and self._candidate_contested():
            # Pause-probe: the pause silences leak but not a person, so a candidate whose
            # LAST speech flag fits the post-engage leak-death window is our own tail.
            # Frame-domain on both sides (a wall clock mis-attributes under capture lag);
            # the skew covers engage landing duckStartFrames-1 frames into the run — a
            # later mid-utterance engage only over-estimates, i.e. probes less.
            if (
                self._endpointer.silence_run_ms >= _PROBE_SILENCE_MS
                and self._endpointer.last_speech_ms - self._engage_skew_ms
                <= self._leak_death_ms
            ):
                await self._drop_candidate("probe")
        if self._stt_stream is not None:
            # Streaming: the transcript materializes by finishing THIS utterance's own handle;
            # taking it OUT of the slot isolates the finish thread from the next utterance.
            if utterance is not None:
                stream, self._stt_live = self._stt_live, None
                if stream is not None:
                    task = asyncio.create_task(asyncio.to_thread(stream.finish))
                    task.add_done_callback(_swallow_result)
                else:  # defensive: no live handle (shouldn't happen) -> batch decode
                    task = None
                self._queue_utterance(
                    self._make_pending(utterance, task, always_valid=task is not None)
                )
            elif prev_speech and not self._endpointer.in_speech:
                self._dump_blip()
                self._release_duck("blip")  # rejected by the min filter mid-candidate
                await self._orphan_if_confirmed("blip")
            return
        eager_pcm = self._endpointer.take_eager()
        if eager_pcm is not None:
            if (
                self._eager_task is not None and not self._eager_task.done()
            ) or self._worker_decoding:
                # A decode is already chewing (a prior speculation, or the worker's final decode
                # of the PREVIOUS utterance) and cannot be aborted: don't stack, see _eager_task.
                self._eager_valid = False
                self._metrics.count("stt_eager_skipped")
            else:
                task = asyncio.create_task(self._transcribe(eager_pcm))
                task.add_done_callback(_swallow_result)
                task.add_done_callback(self._eager_confirm_cb)
                self._eager_task = task
                self._eager_valid = True
                self._metrics.count("stt_eager_start")
        elif prev_speech and not self._endpointer.in_speech and utterance is None:
            self._eager_valid = False  # blip rejected by the min filter; speculation is moot
            self._dump_blip()
            self._release_duck("blip")
            await self._orphan_if_confirmed("blip")
        if utterance is not None:
            eager = self._eager_task if self._eager_valid else None
            if eager is not None and not self._endpointer.eager_covered:
                # A model close beat this run's eager re-mark: the task belongs to an earlier
                # pause and would truncate the utterance to its first half; batch-decode instead.
                self._metrics.count("stt_eager_stale")
                self._eager_valid = False
                eager = None
            if eager is not None:
                # Handed off: the task belongs to THIS utterance's final silence run. (An
                # INVALIDATED task keeps its slot so the guard above sees it until it finishes.)
                self._eager_task = None
                self._eager_valid = False
            self._queue_utterance(self._make_pending(utterance, eager))

    def _next_seg(self) -> int:
        self._seg_seq += 1
        return self._seg_seq

    def _make_pending(
        self, pcm: bytes, eager: asyncio.Task | None, *, always_valid: bool = False
    ) -> _PendingUtterance:
        """Snapshot one closed utterance. Runs on the frame that closed it, before any
        await, so the endpointer close fields and confirm latches are exactly its own."""
        ep = self._endpointer
        preempted, heard = self._take_confirm_latches()
        return _PendingUtterance(
            pcm=pcm,
            eager=eager,
            closed_reason=ep.closed_reason,
            closed_at=time.monotonic(),
            silence_ms=ep.closed_silence_ms,
            learn_ms=self._adaptive.take_pending() if self._adaptive else None,
            eager_always_valid=always_valid,
            preempted=preempted,
            heard=heard,
            onset_interrupting=self._onset_interrupting,
            onset_at=self._onset_at,
            raw=self._raw_tail(len(pcm)),
            active_ms=ep.closed_active_ms,
            prob_peak=ep.closed_prob_peak,
            prob_mean=ep.closed_prob_mean,
            seg_id=self._next_seg(),
        )

    def _raw_tail(self, nbytes: int) -> bytes | None:
        """The last ``nbytes`` of pre-AEC capture: a segment's raw twin. Valid because
        a segment is contiguous trailing audio and callers run between frame pushes
        (or under the hop lock), so the ring cannot advance mid-slice."""
        if self._dump_raw is None or nbytes <= 0:
            return None
        frames: list[bytes] = []
        total = 0
        for frame in reversed(self._dump_raw):
            frames.append(frame)
            total += len(frame)
            if total >= nbytes:
                break
        data = b"".join(reversed(frames))
        return data[-nbytes:] if len(data) > nbytes else data

    def _dump_blip(self) -> None:
        """A min-filter reject may still have engaged the duck: keep it audible.
        No-op unless the endpointer parked one this frame (probe/gap drops reset
        the slot first)."""
        if self._dumper is None:
            return
        pcm, self._endpointer.last_rejected = self._endpointer.last_rejected, None
        if pcm:
            self._dumper.submit(
                "blip", pcm, self._raw_tail(len(pcm)),
                seq=self._next_seg(), meta={"wall": round(time.time(), 3)},
            )

    def _candidate_contested(self) -> bool:
        """A duck/pause candidate is live and no confirm has claimed it. ``in_speech``
        implies this frame closed nothing (push resets before returning an utterance);
        ``_early_confirm`` can re-arm during this frame's awaits (eager callback), so
        callers re-evaluate after awaiting."""
        return (
            self._duck_onset is not None
            and not self._early_confirm
            and not self._preempted
            and self._endpointer.in_speech
        )

    def _take_confirm_latches(self) -> tuple[bool, str | None]:
        """Consume the early-confirm latches for the utterance closing RIGHT NOW: the confirm
        can only have fired during its own capture, so binding at close time is exact."""
        preempted, self._preempted = self._preempted, False
        heard, self._early_heard = self._early_heard, None
        return preempted, heard

    async def _orphan_if_confirmed(self, reason: str) -> None:
        """A candidate closed WITHOUT producing an utterance (min-filter blip) after an early
        confirm killed the reply: no _on_utterance runs for it, so settle the dead turn here."""
        preempted, _ = self._take_confirm_latches()
        if preempted:
            await self._orphaned_confirm(reason)

    def _push_with_model_close(self, pcm: bytes) -> bytes | None:
        """One frame through the endpointer, honoring a pending COMPLETE verdict: when
        the frame itself does not close the utterance, a verdict raised since the last
        frame force-closes it now (validated against the pause the model scored).
        Runs on the hop thread or the loop; the verdict slot is a GIL-atomic write
        from the loop, consumed exactly once here."""
        utterance = self._endpointer.push(pcm)
        if utterance is None:
            gen = self._eou_close_gen
            if gen is not None:
                self._eou_close_gen = None
                utterance = self._endpointer.close_now(gen)
                if utterance is None:
                    self._metrics.count("eou_close_stale")
                else:
                    self._metrics.count("eou_close_early")
        return utterance

    async def _run_consult(self, gen: int, pcm: bytes) -> None:
        """Score one pause with the end-of-turn model, off-loop. COMPLETE raises the
        close flag for the frame path; INCOMPLETE simply lets the hangover run."""
        t0 = time.monotonic()
        try:
            complete = await asyncio.to_thread(self._turn_analyzer.assess, pcm)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a broken model must not kill capture
            if self._consult_fail_throttle.ready():
                self._log.warning("eou consult failed: {}", exc)
            return
        ms = (time.monotonic() - t0) * 1000.0
        self._metrics.observe("eou_consult_ms", ms)
        self._metrics.count("eou_complete" if complete else "eou_incomplete")
        if complete:
            self._eou_close_gen = gen
        self._log.debug(
            "eou_consult: p={:.2f} -> {} ({} ms)",
            self._turn_analyzer.last_probability,
            "complete" if complete else "incomplete", int(ms),
        )

    def _push_frame_sync(self, pcm: bytes, prev_speech: bool) -> tuple[bytes | None, float]:
        """One frame through the endpointer AND (when streaming) the STT stream, in a single
        off-loop hop. At onset a FRESH handle is started and the pre-roll ring replayed into it,
        mirroring the endpointer's own pre-trigger; a rejected blip's handle is just dropped.

        Returns ``(utterance, compute_ms)``: the clock starts after the hop lock, so the
        caller can split engine cost from dispatch/contention overhead."""
        with self._hop_lock:
            t0 = time.monotonic()
            if self._aec is not None:
                raw = pcm
                # Subtract our own playback BEFORE the endpointer/STT hear the frame, so echo
                # never becomes "speech".
                pcm = self._aec.process(pcm)
                if self._dump_raw is not None:
                    # process() floors to whole 10 ms blocks: mirror only what the
                    # pipeline heard, or later raw twins slice misaligned.
                    self._dump_raw.append(raw if len(raw) == len(pcm) else raw[: len(pcm)])
            utterance = self._push_with_model_close(pcm)
            if utterance is not None:
                # Closed: the ring's pre-onset context belongs to THIS utterance, not to a fast
                # re-onset that follows.
                self._recent.clear()
            stt = self._stt_stream
            if stt is not None:
                if self._endpointer.in_speech:
                    if not prev_speech:
                        self._stt_live = stt.stream_start()
                        for frame in self._recent:
                            self._stt_live.accept(frame)
                    if self._stt_live is not None:
                        self._stt_live.accept(pcm)
                        if (
                            self._duck_onset is not None
                            and not self._preempted
                            and not self._early_confirm
                            and utterance is None
                        ):
                            # Early confirm from streaming partials (_judge_fresh); zero
                            # fresh words across consecutive polls release early instead
                            # (consumed duck-mode-only, see the loop-side block).
                            self._partial_countdown -= 1
                            if self._partial_countdown <= 0:
                                self._partial_countdown = self._partial_every
                                ptext = self._stt_live.partial()
                                fresh = self._echo.fresh_words(ptext) - self._ack_words
                                if self._judge_fresh(ptext, fresh) is None:
                                    if fresh:
                                        self._empty_polls = 0
                                    else:
                                        self._empty_polls += 1
                                        onset = self._duck_onset
                                        if (
                                            self._empty_polls >= _RELEASE_POLLS
                                            and onset is not None
                                            and (time.monotonic() - onset) * 1000.0
                                            >= _EARLY_RELEASE_MS
                                        ):
                                            self._early_release = "partial"
                elif prev_speech and utterance is None:
                    self._stt_live = None  # blip rejected by the min filter: abandon its stream
                elif not prev_speech:
                    self._recent.append(pcm)  # idle: keep pre-onset context warm
            return utterance, (time.monotonic() - t0) * 1000.0

    def _queue_utterance(self, pending: _PendingUtterance) -> None:
        if self._adaptive is not None:
            # Anchor the resume-gap clock at CLOSE time: a fast resume lands long
            # before this utterance clears STT (dropped again if it gets rejected).
            self._adaptive.note_close(pending.closed_at, float(pending.silence_ms))
        # Never block the capture pump. Drop-oldest: a full queue means STT is hopelessly
        # behind, and the newest speech is the one the user remembers.
        dropped = put_drop_oldest(self._utt_queue, pending)
        if dropped is not None:
            # The id makes the hole in the utt #N sequence explainable from the log.
            self._log.warning(
                "utterance queue full; dropping oldest (utt #{}, {} ms)",
                dropped.seg_id, int(pcm_ms(len(dropped.pcm), self._cfg.audio.sample_rate)),
            )

    async def _utterance_worker(self) -> None:
        while True:
            pending = await self._utt_queue.get()
            verdict = "error"
            try:
                verdict = await self._on_utterance(pending)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad utterance must not kill the worker
                # The id is the "error"-verdict dump's only log-side correlation.
                self._log.warning("utterance #{} processing failed: {}", pending.seg_id, exc)
            finally:
                self._utt_queue.task_done()
            if self._dumper is not None:
                # Post-verdict, so the filename carries the ladder's outcome.
                self._dumper.submit(
                    verdict, pending.pcm, pending.raw,
                    seq=pending.seg_id, meta=pending.meta,
                )

    async def barge_in(self, played_ms: int) -> None:
        return  # cloud-only

    async def submit_tool_result(self, call_id: str, output: str) -> None:
        return  # cloud-only (nanobot runs tools internally over the bus)

    async def _reset_endpointer(self) -> None:
        """Reset VAD/endpointer streaming state without racing the frame hop.

        A bare loop-side ``reset()`` could interleave mid-``push`` on the hop thread (torn
        counters, undefined neural-VAD cache state). The hop lock serializes them, taken
        off-loop so an in-flight frame never stalls the loop. Skipped mid-utterance: wiping a
        barge-in being captured would lose its first words."""
        def _reset() -> None:
            with self._hop_lock:
                if not self._endpointer.in_speech:
                    self._endpointer.reset()
                    self._recent.clear()

        if self._threaded_hop:
            await asyncio.to_thread(_reset)
        else:
            _reset()  # light path: already serialized with the loop-side pushes

    async def on_capture_gap(self) -> None:
        """The capture stream broke (device restart). The endpointer's clock is frame-counted:
        with no frames flowing, an open utterance silently bridges the outage and merges two
        sentences into one, so drop it, with its STT stream handle and any speculative duck."""
        snap: bytes | None = None
        raw: bytes | None = None

        def _reset() -> None:
            nonlocal snap, raw
            with self._hop_lock:
                if self._endpointer.in_speech:
                    self._log.warning("capture gap mid-utterance; dropping the open utterance")
                    self._metrics.count("capture_gap_drop")
                    if self._dumper is not None:
                        snap = self._endpointer.open_pcm()
                        raw = self._raw_tail(len(snap)) if snap else None
                self._stt_live = None
                self._endpointer.reset()
                self._recent.clear()
                if self._dump_raw is not None:
                    # Pre-gap audio no longer abuts the stream; never splice it
                    # into a later segment's raw twin.
                    self._dump_raw.clear()

        if self._threaded_hop:
            await asyncio.to_thread(_reset)
        else:
            _reset()
        if snap is not None and self._dumper is not None:
            self._dumper.submit(
                "gap", snap, raw,
                seq=self._next_seg(), meta={"wall": round(time.time(), 3)},
            )
        # The dropped utterance's speculation dies with it: a still-valid eager
        # task would hand the PRE-GAP transcript to the next utterance's close.
        self._eager_valid = False
        # The restart re-opens the device: whatever backlog the debt described is gone.
        self._capture_debt_ms = 0.0
        if self._duck_onset is not None:
            self._release_duck("gap")
        await self._orphan_if_confirmed("gap")
        if self._turn is VoiceState.CAPTURING:
            # Nothing will publish the dropped utterance; without this the session
            # presents "capturing" forever and the next onset's IDLE edge is lost.
            await self._set_turn(VoiceState.IDLE)

    async def close(self) -> None:
        self._closing = True
        self._eager_valid = False
        for task in (self._eager_task, self._utt_task, self._cur_turn.prologue_task,
                     self._cur_turn.midturn_task, self._cur_turn.timeout_task,
                     self._tts_task, self._drain_task, self._consult_task):
            await cancel_and_wait(task)
        self._consult_task = None
        self._eager_task = self._utt_task = self._cur_turn.prologue_task = None
        self._cur_turn.midturn_task = self._cur_turn.timeout_task = None
        self._tts_task = self._drain_task = None
        # Pooled adapter resources (e.g. an httpx client); optional per adapter.
        aclose = getattr(self._tts, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                await aclose()
        # An RKNN context is NOT freed by refcount-GC, so an in-process channel restart would
        # load a second copy and exhaust the NPU cores.
        for engine in (self._tts, self._vad, self._turn_analyzer):
            if engine is not None:
                with suppress(Exception):
                    engine.release()
        if self._dumper is not None:
            await asyncio.to_thread(self._dumper.close)

    # ---- input: capture -> STT -> publish (bus) --------------------------

    async def _on_utterance(self, pending: _PendingUtterance) -> str:
        """Run the verdict ladder over one closed utterance. Returns the verdict token
        (``empty``/``echo``/``stop``/``ack``/``interrupt``/``publish``) - the audio
        dump names the segment's file with it."""
        pcm = pending.pcm
        interrupting = self._turn in (VoiceState.THINKING, VoiceState.SPEAKING)
        preempted, heard = pending.preempted, pending.heard
        t0 = time.monotonic()
        self._worker_decoding = True  # the next utterance's eager must not stack on this
        try:
            text, stt_mode = await self._finish_stt(pending)
        finally:
            self._worker_decoding = False
        stt_ms = int((time.monotonic() - t0) * 1000)
        if interrupting and self._turn not in (VoiceState.THINKING, VoiceState.SPEAKING):
            # The reply finished on its own while STT ran (drain reached IDLE): the stale
            # snapshot would fire a bogus /stop at a finished turn.
            interrupting = False

        def _summary(verdict: str) -> str:
            """One line per judged utterance: id (matches the dump filename), verdict,
            durations, close shape, STT cost/path, VAD confidence, head loudness (1 s
            cap: the numpy-less fallback loops on the event loop; quiet = our own
            playback tail, loud = real speech the model failed on). Also stamps
            ``pending.meta`` for the dump manifest."""
            dur_ms = int(pcm_ms(len(pcm), self._cfg.audio.sample_rate))
            rms = pcm_rms(pcm[: self._cfg.audio.sample_rate * 2])
            vad = (
                f" vad={pending.prob_mean:.2f}~{pending.prob_peak:.2f}"
                if pending.prob_mean is not None else ""
            )
            self._log.info(
                "utt #{}: {} dur={}ms active={}ms close={}+{}ms stt={}ms/{}{}"
                " rms={:.3f} interrupt={} text='{}'",
                pending.seg_id, verdict, dur_ms, pending.active_ms,
                pending.closed_reason, pending.silence_ms, stt_ms, stt_mode, vad, rms,
                interrupting or preempted,
                loggable_text(text, self._cfg.log_transcripts, 80),
            )
            # Verdict/duration/rms stay out: the dump writer stamps those on every
            # record itself, blips and probe drops included.
            pending.meta = {
                # Capture-side close time (epoch s): submit-time stamps trail the verdict.
                "wall": round(time.time() - (time.monotonic() - pending.closed_at), 3),
                "active_ms": pending.active_ms,
                "close": pending.closed_reason,
                "silence_ms": pending.silence_ms,
                "stt_ms": stt_ms,
                "stt": stt_mode,
                "interrupt": bool(interrupting or preempted),
            }
            if pending.prob_mean is not None:
                pending.meta["prob_mean"] = round(pending.prob_mean, 3)
                pending.meta["prob_peak"] = round(pending.prob_peak, 3)
            if self._cfg.log_transcripts:  # same privacy gate as the log line
                pending.meta["text"] = text
            return verdict

        if not text:
            if self._adaptive is not None:
                self._adaptive.drop_anchor()  # not the user's turn: nothing to resume from
            self._release_duck("empty")
            if preempted:
                await self._orphaned_confirm("empty")
            elif self._turn is VoiceState.CAPTURING:
                await self._set_turn(VoiceState.IDLE)  # no drain, no hangover
            return _summary("empty")

        # A transcript that is mostly the bot's own words is self-echo: drop it WHATEVER the
        # turn state or duplex mode. The trailing echo closes its VAD hangover (~600 ms) after
        # the drain reached IDLE (~250 ms), so gating on "interrupting" would publish the bot's
        # own last sentence as a user turn; the echo window is play-time stamped, so an
        # out-of-turn hit IS our own voice. Half-duplex is NOT exempt: its mic gate is a
        # read-time approximation that a mistimed hangover or capture lag leaks the reply's
        # audible, transcribable tail past.
        if self._echo.is_self_echo(text):
            # A genuine soft-duplex interruption is a MIXTURE (user + leak) that containment
            # classifies as echo. Words that are neither our TTS nor backchannels evidence a
            # person talking through it, so only an interrupt-shaped turn overrides — and a
            # single fresh STOP word is enough (the kill switch must survive the leak).
            fresh = self._echo.fresh_words(text) - self._ack_words
            fresh_seq = self._fresh_seq(text, fresh)
            if (interrupting or preempted) and self._stop_match.pure(fresh_seq):
                # The fresh remainder is pure command material: a stop said THROUGH the
                # leak. Ordered, so multi-word phrases ("shut up") match here too.
                self._log.info(
                    "stop through echo: '{}'",
                    loggable_text(text, self._cfg.log_transcripts, 60),
                )
                await self._consume_stop(
                    " ".join(fresh_seq), heard,
                    interrupting=interrupting, preempted=preempted,
                )
                return _summary("stop")
            if (interrupting or preempted) and (
                len(fresh) >= self._min_fresh_words
                or self._stop_match.present(fresh_seq)
            ):
                self._metrics.count("barge_in_through_echo")
                self._log.info(
                    "interrupt through echo ({} fresh words): '{}'",
                    len(fresh), loggable_text(text, self._cfg.log_transcripts, 60),
                )
            else:
                if self._adaptive is not None:
                    self._adaptive.drop_anchor()  # the bot's own words anchor nothing
                self._release_duck("echo")
                if preempted:
                    await self._orphaned_confirm("echo")
                elif self._turn is VoiceState.CAPTURING:
                    # The trailing echo's vad_start put us in CAPTURING; without this settle the
                    # session sits there until the user speaks.
                    await self._set_turn(VoiceState.IDLE)
                return _summary("echo")

        # Stop command aimed at a live reply: kill it and CONSUME the utterance — publishing
        # "stop" would make the agent answer it ("okay, stopping"), the exact reply the user
        # just asked not to hear. Targeting is decided at ONSET (a reply draining during this
        # utterance's STT window still counts), plus a short kill-anchored grace for
        # double-taps. A cold stop with nothing live falls through and publishes: with
        # context the agent can answer it sensibly, and an answer to a question the agent
        # just asked ("say cancel to abort") must reach it.
        if (
            pending.onset_interrupting
            or preempted
            or interrupting
            or pending.onset_at - self._last_kill <= _KILL_GRACE_S
        ) and self._is_stop(text):
            await self._consume_stop(
                text, heard, interrupting=interrupting, preempted=preempted
            )
            return _summary("stop")

        # Backchannel ("okay", "uh-huh") while the bot works/speaks: keep the reply. A wrong
        # call costs only ~a second of duck, hence a lexicon, not a classifier.
        if (interrupting or preempted) and self._is_ack(text):
            self._metrics.count("barge_in_backchannel")
            if self._adaptive is not None:
                self._adaptive.drop_anchor()  # a backchannel is not a turn to resume
            self._release_duck("ack")
            if preempted:
                await self._orphaned_confirm("ack")
            return _summary("ack")

        # Anchor at the TRUE end of speech, only for ACCEPTED utterances (a rejected echo/empty
        # must not corrupt a live turn's clock): back-date past STT time and queue wait to the
        # endpoint close, plus silence_ms — the frame-quantized trailing silence the close
        # consumed (the full hangover, or less when the turn model closed early).
        offset = (time.monotonic() - pending.closed_at) * 1000.0 + float(pending.silence_ms)
        self._metrics.turn_anchor(offset_ms=offset)
        self._metrics.observe("stt_ms", float(stt_ms))
        if self._adaptive is not None:
            # Commit false-endpoint evidence, if any; the resume anchor was set at close time.
            self._adaptive.on_publish(pending.learn_ms)
            hangover = self._adaptive.value_ms()
            self._endpointer.set_hangover_ms(hangover)
            self._metrics.observe("hangover_ms", float(hangover))
            if pending.learn_ms is not None:
                self._metrics.count("eou_hangover_learned")
                self._log.debug(
                    "adaptive hangover: cut pause ~{} ms -> hangover {} ms",
                    int(pending.learn_ms), hangover,
                )

        killed, heard = await self._kill_live_reply(
            interrupting=interrupting, preempted=preempted, heard=heard
        )
        # After a preempted turn the sink can still be ducked from a candidate raised during the
        # STT window, with nothing else to clear it until the NEXT turn drains.
        self._clear_duck()
        self._chunker.flush()          # discard any partial chunk from a prior turn
        self._echo.reset()             # forget the old reply's words (AFTER is_self_echo)
        await self._set_turn(VoiceState.THINKING)
        self._cur_turn = _Turn(unique_token())
        self._heard_prefix = ""        # heard accounting restarts with the new turn
        parts = [text]
        marker = _interrupt_marker(heard)
        if marker:
            parts.append(marker)
        if self._pending_note is not None:
            # A consumed stop's note describes the PREVIOUS (killed) reply; the marker above
            # describes the one killed by THIS utterance. Both may ride one publish.
            parts.append(self._pending_note)
            self._pending_note = None
        await self._publish_text("\n\n".join(parts), self._cur_turn.token)
        self._arm_prologue()
        self._arm_timeout()
        return _summary("interrupt" if killed else "publish")

    async def _do_interrupt(self) -> str | None:
        """Cancel-then-send barge-in: invalidate the dead turn, stop audio, /stop.

        Invalidating FIRST means deltas the cancelled turn emits before /stop lands are dropped
        instead of bleeding into the new turn; the watermark additionally rejects a turn whose
        base we never saw. Returns the heard-up-to TEXT for the interrupting utterance
        ("" = cut before anything sounded; None = accounting unavailable/disabled)."""
        self._rejected_base = self._cur_turn.base
        self._reject_started_before_ns = time.time_ns()
        if self._cur_turn.token:
            self._dead_tokens.append(self._cur_turn.token)
        self._cur_turn.abandon()
        # Left running, the dead turn's drain watcher would wake on a transient queue-empty
        # moment and drain the SUCCESSOR turn's live stream (wiping its heard-up-to spans).
        cancel_task(self._drain_task)
        self._log.info("barge_in (state was {})", self._turn.value)
        played = await self._sink.flush()  # stop bot audio now (epoch++; resets the duck)
        if self._duck_onset is not None:
            # Onset -> silence (cloud analog: barge_in_ms.{truncate,cancel}).
            self._metrics.observe(
                "barge_in_ms.local", (time.monotonic() - self._duck_onset) * 1000.0
            )
        # Candidate resolved; the flush restored the gain. The VAD floor stays scaled down:
        # playback is gone, so the lowered floor is the better estimate and re-adapts on its own.
        self._duck_onset = None
        self._duck_suspect = False
        heard = None
        if self._pcm_out and self._cfg.barge_in.heard_marker:
            # played is stream-relative: subtract the segment's base (a reused filler stream's
            # audio must not count as reply) and prepend the turn's completed segments. A base
            # measured on a REPLACED stream is void: that stream's played_ms restarted at 0.
            base = (
                self._spans_base_ms
                if self._sink.stream_generation == self._spans_gen
                else 0.0
            )
            heard = self._heard_text(max(0.0, float(played) - base))
            heard = f"{self._heard_prefix} {heard}".strip()
        self._spoken_spans.clear()
        self._heard_prefix = ""
        await self._interrupt()
        return heard

    def _heard_text(self, played_ms: float) -> str:
        """Map the sink's played-ms into the chunk texts the user actually heard:
        chunk-granular, with a word-proportional cut inside the chunk playback stopped in: the
        estimate LiveKit/Pipecat use absent word timestamps."""
        out: list[str] = []
        acc = 0.0
        for chunk_text, dur in self._spoken_spans:
            if acc + dur <= played_ms + 50.0:  # fully heard (scheduling slack)
                out.append(chunk_text)
                acc += dur
                continue
            frac = (played_ms - acc) / dur if dur > 0 else 0.0
            if frac > 0.15:  # a word or two into the chunk: keep the heard part
                cut_words = chunk_text.split()
                keep = max(1, int(len(cut_words) * frac))
                out.append(" ".join(cut_words[:keep]) + "...")
            break
        return " ".join(out).strip()

    def _in_aec_warmup(self) -> bool:
        """AEC3 still converging (too little reference audio accepted): hold the
        early-confirm shortcuts (the endpoint verdict still decides)."""
        if self._aec is None:
            return False
        ref_ms = getattr(self._aec, "reference_ms", None)
        return ref_ms is not None and ref_ms() < _AEC_WARMUP_REF_MS

    def _duck_armed(self) -> bool:
        return (
            self._open_mic
            and self._turn is VoiceState.SPEAKING
            and self._sink.stream_mode
            and (self._duck_gain < 1.0 or self._duck_pause)
            # Post-acquittal holdoff: resumed playback re-leaks immediately; without this
            # the candidate loop flaps at ~1 Hz under weak/absent AEC.
            and time.monotonic() >= self._probe_holdoff_until
        )

    def _engage_duck(self, *, suspect: bool) -> None:
        """Duck now, reversibly. ``suspect`` marks a pre-onset engage; a confirmed onset passes
        False, which also graduates a running suspicion duck without restarting its clock or
        re-counting it."""
        self._duck_suspect = suspect
        if self._duck_onset is not None:
            return
        self._duck_onset = time.monotonic()
        self._partial_countdown = 0
        self._empty_polls = 0
        self._early_release = None  # a stale acquittal must not kill a fresh candidate
        if self._duck_pause:
            # The leak stops entirely, so the VAD floor is left alone: it re-adapts on its own,
            # and the user's speech (which just out-competed the elevated floor) keeps flagging.
            self._sink.pause(True)
        else:
            self._sink.duck(True)
            # The leak the mic hears is about to drop by duck_gain: step the adaptive VAD floor
            # with it (no-op for non-adaptive VADs) so a real interjection isn't under-detected
            # while the floor re-converges. Undone on clear/release, never on a confirmed kill.
            self._vad.scale_floor(self._duck_gain)
        self._metrics.count("barge_in_duck")

    def _duck_if_capturing(self) -> None:
        """Engage the duck when the turn ENTERS SPEAKING with the user already mid-utterance
        (started during THINKING, or between segments): there is no in_speech edge then, so the
        reply would otherwise play at full volume until the transcript verdict."""
        if (
            self._duck_armed()
            and self._endpointer.in_speech
            and not self._acquitted_open
        ):
            self._engage_duck(suspect=False)

    def _eager_confirm_cb(self, task) -> None:
        """Batch analog of the streaming min-words confirm: the eager decode lands mid-hangover,
        so enough fresh words while still ducked confirm the interrupt without waiting out the
        endpoint + final decode. Runs on the loop; the next capture frame consumes the flag,
        re-validating that the utterance is still open before latching."""
        if task.cancelled() or task.exception() is not None:
            return
        if task is not self._eager_task or not self._eager_valid:
            # A dropped/superseded candidate's decode (or a handed-off finish) must not
            # judge the live one: its audio is a different utterance's.
            return
        if (
            self._closing
            or self._duck_onset is None
            or self._preempted
            or self._early_confirm
            or not self._endpointer.in_speech  # closed: the endpoint verdict owns it
        ):
            return
        text = task.result() or ""
        fresh = self._echo.fresh_words(text) - self._ack_words
        if self._judge_fresh(text, fresh) == "confirm":
            self._metrics.count("barge_in_eager_confirm")
        elif (
            not fresh
            and not self._duck_pause
            and self._endpointer.eager_still_current()
        ):
            # The snapshot still covers everything flagged (no speech resumed since) and
            # decoded to NOTHING fresh: acquit the duck ~a hangover before the verdict.
            # Pause mode never acquits on transcripts (the probe owns it).
            self._early_release = "eager"

    def _clear_duck(self) -> None:
        """Drop any live candidate and restore the level + VAD floor. No verdict."""
        if self._duck_onset is None:
            return
        self._duck_onset = None
        self._duck_suspect = False
        if self._duck_pause:
            self._sink.pause(False)
        else:
            self._sink.duck(False)
            if self._duck_gain > 0:
                self._vad.scale_floor(1.0 / self._duck_gain)  # leak returns to full level

    def _release_duck(self, reason: str) -> None:
        """A candidate interrupt turned out false: restore the level, keep the reply."""
        if self._duck_onset is None:
            return
        self._clear_duck()
        self._metrics.count(f"barge_in_false_resume.{reason}")
        self._log.info("false barge-in ({}); duck released", reason)
        if reason not in _LEAK_REASONS:
            return  # backchannels/blips are conversation, not echo evidence
        now = time.monotonic()
        self._false_times.append(now)
        while self._false_times and now - self._false_times[0] > _FALSE_WARN_WINDOW_S:
            self._false_times.popleft()
        if len(self._false_times) >= _FALSE_WARN_N and self._false_warn.ready():
            self._log.warning(
                "voice: {} false barge-in candidates in the last minute — likely our own "
                'playback leaking into the mic. If this device\'s AEC is weak, consider '
                'aec="webrtc" (layers AEC3 on top), bargeIn.mode="duck", or raising '
                "bargeIn.duckStartFrames",
                len(self._false_times),
            )

    async def _drop_candidate(self, reason: str) -> None:
        """Pause-mode acquittal while the utterance is still OPEN: resumed playback would
        pour leak into it until a max-length close, so the candidate audio (leak-tail sized
        by the probe's attribution) is dropped whole — stream handle, speculation, endpointer
        state and the pre-onset ring, whose context belonged to the dropped candidate (the
        same rule as the close/capture-gap drops) — then the pause releases and
        re-engagement holds off briefly."""
        snap: bytes | None = None
        raw: bytes | None = None

        def _reset() -> None:
            nonlocal snap, raw
            with self._hop_lock:
                if self._dumper is not None:
                    # Last moment this audio exists: snapshot the leak evidence.
                    snap = self._endpointer.open_pcm()
                    raw = self._raw_tail(len(snap)) if snap else None
                self._stt_live = None
                self._endpointer.reset()
                self._recent.clear()

        if self._threaded_hop:
            await asyncio.to_thread(_reset)
        else:
            _reset()
        if snap is not None and self._dumper is not None:
            self._dumper.submit(
                reason, snap, raw,
                seq=self._next_seg(), meta={"wall": round(time.time(), 3)},
            )
        self._eager_valid = False
        self._probe_holdoff_until = time.monotonic() + _ENGAGE_HOLDOFF_S
        self._release_duck(reason)

    async def _orphaned_confirm(self, reason: str) -> None:
        """An early confirm killed the reply, but the endpoint verdict says the trigger wasn't
        real user speech, nothing to publish. Settle to IDLE so the session doesn't sit in a
        dead SPEAKING (audio flushed, mic gated in half-duplex) until the user happens to speak
        again."""
        self._metrics.count(f"barge_in_early_orphan.{reason}")
        self._log.warning("early confirm orphaned ({}); reply already stopped", reason)
        if self._turn is not VoiceState.IDLE:
            # THINKING included: nothing can produce deltas past the watermark, so a standing
            # THINKING shows a dead "thinking" forever.
            await self._set_turn(VoiceState.IDLE)

    def _is_ack(self, text: str) -> bool:
        return self._ack_match.covers(tokens_of(text))

    def _is_stop(self, text: str) -> bool:
        """Pure stop command: entirely stop/ack/filler material with a full stop phrase
        present (see PhraseMatcher.pure). Mixed content is NOT a stop."""
        return self._stop_match.pure(tokens_of(text))

    @staticmethod
    def _fresh_seq(text: str, fresh: set[str]) -> list[str]:
        """The fresh words in UTTERANCE order: multi-word stop phrases need contiguity,
        which the fresh SET destroyed."""
        return [t for t in tokens_of(text) if t in fresh]

    def _judge_fresh(self, text: str, fresh: set[str]) -> str | None:
        """The shared confirm arm of both early-verdict sites (streaming partials, eager
        decode): "confirm" when the fresh evidence clears the bar — min words, or a full
        stop phrase in the ordered fresh remainder (ONE suffices: the most urgent intent
        must not be the slowest kill) — "hold" under the AEC warmup, None otherwise."""
        stop_early = False
        if len(fresh) < self._min_fresh_words:
            if not fresh or not self._stop_match.present(self._fresh_seq(text, fresh)):
                return None
            stop_early = True
        if self._in_aec_warmup():
            self._metrics.count("barge_in_warmup_hold")
            return "hold"
        self._early_confirm = True
        if stop_early:
            self._metrics.count("barge_in_stop_early")
        return "confirm"

    async def _kill_live_reply(
        self, *, interrupting: bool, preempted: bool, heard: str | None
    ) -> tuple[bool, str | None]:
        """Kill the live reply exactly once — the publish tail and the stop consume both
        route here. Returns (killed, heard)."""
        if not interrupting or preempted:
            return preempted, heard
        if not self._cur_turn.dead:
            heard = await self._do_interrupt()
        else:
            # Already killed (early confirm, or a previous utterance's verdict): never
            # /stop twice, and no heard-up-to against cleared spans. But audio started
            # AFTER the kill (the timeout notice) is still playing: stop it, and its
            # drain watcher with it, or the notice talks over what follows.
            cancel_task(self._drain_task)
            await self._sink.flush()
        return True, heard

    async def _consume_stop(
        self, stop_text: str, heard: str | None, *, interrupting: bool, preempted: bool
    ) -> None:
        """A pure stop command: kill whatever is live and publish NOTHING — silence is the
        acknowledgment. The heard-up-to contract survives as a pending note on the next
        publish; a stop that killed nothing (the reply finished on its own) leaves no note
        and never arms the double-tap grace."""
        killed, heard = await self._kill_live_reply(
            interrupting=interrupting, preempted=preempted, heard=heard
        )
        self._clear_duck()
        self._chunker.flush()
        # The echo window deliberately stays armed (unlike the publish path, which resets
        # it for a NEW turn): leak captured during this stop's own STT window must still
        # classify as self-echo, or the bot publishes its own tail as a cold user turn
        # right after being told to shut up. The window ages out on its own.
        if self._adaptive is not None:
            self._adaptive.drop_anchor()  # a command is not a turn to learn pauses from
        if killed:
            self._last_kill = time.monotonic()  # arms/extends the double-tap grace
            self._pending_note = _stop_note(stop_text, heard)
        self._metrics.count("barge_in_stop")
        if self._turn is not VoiceState.IDLE:
            await self._set_turn(VoiceState.IDLE)

    async def _finish_stt(self, pending: _PendingUtterance) -> tuple[str, str]:
        """The utterance's transcript, plus which path produced it (``stream`` — the
        streaming handle's tail flush, ``eager`` — the speculation, ``batch`` — a fresh
        decode): the mode names what the recorded ``stt_ms`` residual wait actually paid for.

        A silence-closed utterance is the speculation plus trailing silence, so its transcript
        is exactly valid and only the residual wait is paid. A max-length close means speech
        continued past the snapshot: the stale speculation is DRAINED (it cannot be aborted) and
        discarded, so the fresh decode never contends with it."""
        task = pending.eager
        if task is not None:
            if pending.closed_reason != "max" or pending.eager_always_valid:
                try:
                    text = await task
                    if pending.eager_always_valid:
                        self._metrics.count("stt_stream_finish")
                        return text, "stream"
                    self._metrics.count("stt_eager_hit")
                    return text, "eager"
                except asyncio.CancelledError:
                    raise
                except Exception as exc:  # noqa: BLE001 - fall back to a fresh decode
                    self._log.debug("eager STT failed ({}); decoding fresh", exc)
            else:
                self._metrics.count("stt_eager_stale")
                with suppress(Exception):
                    await task
        stale = self._eager_task
        if stale is not None and not stale.done():
            # An INVALIDATED speculation can still be chewing (its slot is kept so the stacking
            # guard sees it); the fresh decode below would otherwise run concurrently, breaking
            # the one-decode invariant in exactly the slow-ASR regime it protects. No new eager
            # can appear here: _worker_decoding wraps this call.
            self._metrics.count("stt_eager_drained")
            with suppress(Exception):
                await stale
        return await self._transcribe(pending.pcm), "batch"

    # ---- output: streamed reply -> chunker -> TTS -> sink ----------------

    async def on_delta(self, delta: str, stream_id: str | None = None) -> None:
        """A streamed assistant text chunk (``_stream_delta``)."""
        if not delta:
            return
        base = base_of(stream_id)
        if self._is_rejected(base):
            return
        if base is not None:
            self._cur_turn.base = base
        self._cancel_prologue()  # the reply is arriving; no more filler
        self._cancel_midturn()   # a new segment began; the old boundary watch is stale
        self._cur_turn.last_activity = time.monotonic()
        if self._cur_turn.await_first_token and self._cur_turn.published_at:
            self._cur_turn.await_first_token = False
            first_ms = (time.monotonic() - self._cur_turn.published_at) * 1000.0
            self._metrics.observe("agent_first_token_ms", first_ms)
            self._log.info("first_token ({} ms after publish)", int(first_ms))
        if self._cur_turn.continuation_pending:
            # First post-tool token: re-anchor so the resumed segment's audio lands in
            # continuation_ms, never in ttfa_ms (a slow tool must not read as a slow model).
            self._cur_turn.continuation_pending = False
            self._metrics.turn_continuation()
        if self._turn is not VoiceState.SPEAKING:
            await self._set_turn(VoiceState.SPEAKING)
            self._duck_if_capturing()
        for chunk in self._chunker.feed(delta):
            self._tts_enqueue(chunk)  # echo filter is fed at EMIT time (see _tts_worker)
            self._cur_turn.segment_spoke = True

    async def on_stream_end(self, *, resuming: bool, stream_id: str | None = None) -> None:
        """A ``_stream_end`` marker. ``resuming`` means tool calls follow.

        The segment is COMPLETE model output either way, so the chunker is always flushed: a
        short pre-tool status line ("One moment.") is under the first-chunk floor and would
        otherwise sit buffered (silent) through the whole tool wait. On ``resuming``,
        _midturn_watch reopens the wait."""
        base = base_of(stream_id)
        if self._is_rejected(base):
            return
        self._cur_turn.last_activity = time.monotonic()
        tail = self._chunker.flush()
        if tail:
            self._tts_enqueue(tail)
            self._cur_turn.segment_spoke = True
        if resuming:
            spoke, self._cur_turn.segment_spoke = self._cur_turn.segment_spoke, False
            if spoke:
                # The agent masked the tool wait with its own spoken status line.
                self._metrics.count("agent_prologue")
            self._cur_turn.continuation_pending = True
            self._arm_midturn(spoke)
            return
        self._cur_turn.segment_spoke = False
        self._schedule_drain()

    async def speak_final(self, text: str) -> None:
        """A non-streamed final assistant message (streaming disabled / fallback)."""
        self._cancel_prologue()
        self._cancel_midturn()
        await self._set_turn(VoiceState.SPEAKING)
        self._duck_if_capturing()
        for chunk in self._chunker.feed(text):
            self._tts_enqueue(chunk)
        tail = self._chunker.flush()
        if tail:
            self._tts_enqueue(tail)
        self._schedule_drain()

    # ---- TTS stage + drain ----------------------------------------------

    def _tts_enqueue(self, text: str) -> None:
        if not text:
            return
        if self._cur_turn.chunk_await and self._cur_turn.published_at:
            self._cur_turn.chunk_await = False
            self._metrics.observe(
                "chunker_wait_ms", (time.monotonic() - self._cur_turn.published_at) * 1000.0
            )
        self._tts_queue.put_nowait((self._sink.epoch, text))

    async def _tts_worker(self) -> None:
        while True:
            epoch, text = await self._tts_queue.get()
            try:
                if self._tts is None or epoch != self._sink.epoch:
                    continue  # tts off, or barged in before synthesis
                if self._pcm_out and not self._cur_turn.tts_first_pending:
                    # Backlog gate: hold synthesis while plenty of audio is queued/unplayed. A
                    # verdict resolves a paused sink and a barge-in flush bumps the epoch, so
                    # this always releases.
                    if self._sink.backlog_ms() > _SYNTH_BACKLOG_MS:
                        self._metrics.count("tts_backlog_gated")
                        while (
                            epoch == self._sink.epoch
                            and self._sink.backlog_ms() > _SYNTH_BACKLOG_MS
                        ):
                            await asyncio.sleep(_BACKLOG_POLL_S)
                        if epoch != self._sink.epoch:
                            continue  # barged in while gated
                    # TTS is BEHIND: coalesce same-epoch chunks into one call, fewer calls and
                    # prosody seams on high-RTF adapters. Never for the first chunk (it would
                    # fight the first-chunk floor). No await between the epoch check and this
                    # drain, so every drained item shares the head's (current) epoch.
                    while (
                        len(text) < self._cfg.chunker.max_chars
                        and not self._tts_queue.empty()
                    ):
                        nxt_epoch, nxt = self._tts_queue.get_nowait()
                        self._tts_queue.task_done()
                        if nxt_epoch != epoch:  # defensive; unreachable today
                            self._tts_queue.put_nowait((nxt_epoch, nxt))
                            break
                        # >= U+2E80 is CJK and up: those scripts take no space.
                        sep = "" if (text[-1].isspace() or ord(text[-1]) >= 0x2E80) else " "
                        text = f"{text}{sep}{nxt}"
                        self._metrics.count("tts_coalesced")
                t0 = time.monotonic()
                if self._pcm_out:
                    audio = await self._tts.synthesize_pcm(text)
                else:
                    audio = await self._tts.synthesize(text)
                synth_ms = (time.monotonic() - t0) * 1000.0
                # Every chunk: steady-state TTS drift is otherwise invisible until it gaps.
                self._metrics.observe("tts_synth_ms", synth_ms)
                if epoch != self._sink.epoch:  # barged in during synthesis
                    continue
                if not audio:
                    continue
                was_first = self._cur_turn.tts_first_pending
                if was_first:
                    self._cur_turn.tts_first_pending = False
                    self._cur_turn.audible_at = time.monotonic()
                    self._metrics.observe("tts_first_chunk_ms", synth_ms)
                elif self._sink.starved():
                    # Synthesis lost the race against playback (audible mid-reply gap): the
                    # chunk floors are too small for this TTS speed.
                    self._metrics.count("tts_gap")
                if self._pcm_out:
                    # A turn's first chunk keeps only 20 ms of lead silence (the rest is pure
                    # TTFA); later chunks keep 120 ms, so inter-sentence pauses stay natural.
                    audio = trim_lead_silence(
                        audio, self._tts.output_rate, cap_ms=20.0 if was_first else 120.0
                    )
                if self._duck_gain < 1.0 and not self._pcm_out:
                    # Blob fallback: no mid-chunk gain control, so bake the static duck in.
                    audio = await asyncio.to_thread(_scale_wav, audio, self._duck_gain)
                self._metrics.turn_first_audio()  # latched: TTFA on the turn's first frame
                dur_ms = self._audio_ms(audio)
                self._log.debug(
                    "tts: {} chars -> {:.0f} ms audio in {:.0f} ms (rtf {:.2f})",
                    len(text), dur_ms, synth_ms,
                    synth_ms / dur_ms if dur_ms > 0 else 0.0,
                )
                if self._pcm_out:
                    # Heard-up-to span, appended in sink FIFO order.
                    gen = self._sink.stream_generation
                    if self._spoken_spans and gen != self._spans_gen:
                        # The spans' stream was EOF'd and replaced since they were anchored (a
                        # cancelled tool-boundary settle: its fold never ran, and the tail rang
                        # out in full). Fold them as heard: against a stale base the mapping
                        # garbles.
                        spoken = " ".join(t for t, _ in self._spoken_spans).strip()
                        self._heard_prefix = f"{self._heard_prefix} {spoken}".strip()
                        self._spoken_spans.clear()
                    if not self._spoken_spans:
                        # Segment start: anchor at the CURRENT played position; the stream may
                        # already have played a filler's audio.
                        self._spans_base_ms = float(self._sink.played_ms())
                        self._spans_gen = gen
                    self._spoken_spans.append((text, dur_ms))
                # Fed HERE, not at chunker feed: the eviction window runs from when the words
                # stop being AUDIBLE, hence backlog + this chunk's playtime. Earlier, and a long
                # reply's tail ages out mid-playback and reads back as user speech.
                self._echo.note_spoken(text, hold_ms=self._sink.backlog_ms() + dur_ms)
                await self._emit(self._audio_event(epoch, audio))
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never let one chunk kill the worker
                self._log.warning("tts error: {}", exc)
            finally:
                self._tts_queue.task_done()

    def _schedule_drain(self) -> None:
        # Reply complete: no filler, tool boundary, or stall watch can still apply to this turn.
        self._cancel_prologue()
        self._cancel_midturn()
        self._cur_turn.cancel_timeout()
        cancel_task(self._drain_task)
        self._drain_task = asyncio.create_task(self._drain_watch(self._sink.epoch))

    def _audio_event(self, epoch: int, audio: bytes) -> OutputAudio:
        if self._pcm_out:
            return OutputAudio(epoch=epoch, pcm=audio, rate=self._tts.output_rate)
        return OutputAudio(epoch=epoch, wav=audio)

    def _audio_ms(self, audio: bytes) -> float:
        """Playback duration of one synthesized chunk (pcm by rate, wav by header)."""
        if self._pcm_out:
            return pcm_ms(len(audio), self._tts.output_rate)
        return wav_duration_ms(audio)

    # ---- prologue (filler while the agent works) -------------------------

    def _arm_prologue(self, initial_ms: int | None = None, start_step: int = 0) -> None:
        """Arm the filler timer for the turn just published (cancels any prior).

        ``initial_ms`` overrides the first delay and ``start_step`` opens the script
        mid-way: the tool-boundary re-arm passes ``intervalMs`` + step 1 when the agent
        just spoke its own status line — that line WAS the script's opener, so the
        canned filler neither piles on top of it nor de-escalates back to phrase 0."""
        self._cancel_prologue()
        if (
            self._closing
            or self._tts is None
            or not self._cfg.prologue.enabled
            or not self._cfg.prologue.phrases
        ):
            return
        self._cur_turn.prologue_task = asyncio.create_task(
            self._prologue_watch(self._sink.epoch, initial_ms, start_step)
        )

    def _cancel_prologue(self) -> None:
        self._cur_turn.cancel_prologue()

    def _arm_midturn(self, spoke: bool) -> None:
        """Arm the tool-boundary watcher after a ``resuming`` stream end."""
        self._cancel_midturn()
        if self._closing:
            return
        self._cur_turn.midturn_task = asyncio.create_task(
            self._midturn_watch(self._sink.epoch, spoke)
        )

    def _cancel_midturn(self) -> None:
        self._cur_turn.cancel_midturn()

    def _arm_timeout(self) -> None:
        """Arm the stalled-agent deadman for the turn just published.

        A voice channel must never end a turn in dead air: with no activity at all (no delta, no
        segment end) for ``agentTimeoutS``, speak a notice, /stop the stuck run and settle
        rather than sit in THINKING forever. Activity pushes ``_Turn.last_activity``, so long
        tool runs survive as long as the agent streams its pre-tool status line. With core
        streaming DISABLED nothing pushes it, so agentTimeoutS is a hard cap on any turn:
        the accepted cost of recovering wedged ones (raise it if long tool runs matter)."""
        self._cur_turn.cancel_timeout()
        if self._closing or self._cfg.agent_timeout_s is None:
            return
        self._cur_turn.timeout_task = asyncio.create_task(self._timeout_watch(self._cur_turn))

    async def _timeout_watch(self, turn: _Turn) -> None:
        try:
            budget = float(self._cfg.agent_timeout_s)
            await wait_for_stall(lambda: turn.last_activity, budget)
            if self._closing or self._cur_turn is not turn or self._turn is not VoiceState.THINKING:
                return  # the turn moved on (or is speaking/draining), not stalled
            self._log.warning(
                "agent turn stalled ({}s with no activity); speaking timeout notice",
                int(budget),
            )
            self._metrics.count("agent_turn_timeout")
            # Detach OURSELVES first: _do_interrupt abandons the _Turn, which cancels its
            # tasks, including this one, unless the slot is empty.
            turn.timeout_task = None
            await self._do_interrupt()  # watermark + /stop the stuck run
            if self._closing or self._cur_turn is not turn:
                return  # a verdict published a successor during the interrupt's awaits
            # After the successor guard, or this would wipe a successor's fresh anchor;
            # released so the recovery notice is not a ~timeout-sized TTFA sample.
            self._metrics.turn_end()
            await self.speak_final(self._cfg.timeout_phrase)  # then drain -> IDLE
        except asyncio.CancelledError:
            raise
        except Exception:
            self._log.exception("timeout watch failed")

    async def _settle(self, epoch: int) -> bool:
        """Wait until the segment enqueued at *epoch* is fully synthesized and audibly played
        out, then report whether that epoch still owns the pipeline. Shared by every watcher
        that follows playback with a transition.

        The epoch is re-checked BEFORE each side effect: a watcher that survived into a
        successor turn (its epoch died while it waited) must do nothing, or it drains the
        successor's live stream and wipes its heard-up-to spans."""
        await self._tts_queue.join()         # all text synthesized + emitted
        if self._closing or epoch != self._sink.epoch:
            return False
        await self._sink.drain_stream()      # all audio audibly played out
        if self._closing or epoch != self._sink.epoch:
            return False
        if self._spoken_spans:  # played out in full -> fold into the heard-prefix
            spoken = " ".join(t for t, _ in self._spoken_spans).strip()
            self._heard_prefix = f"{self._heard_prefix} {spoken}".strip()
            self._spoken_spans.clear()
        return True

    async def _midturn_watch(self, epoch: int, spoke: bool) -> None:
        """After a segment's audio drains at a tool boundary, reopen the wait.

        Returns the turn to THINKING (lifting the half-duplex mic gate, so the user can barge in
        during a long tool run) and re-arms the canned prologue. Cancelled by the next segment's
        first delta, the final drain, barge-in and stop; the epoch guard covers a barge-in that
        already flushed the sink."""
        try:
            if not await self._settle(epoch):
                return
            if self._turn is VoiceState.SPEAKING:
                if not self._full_duplex:
                    # Tail guard: reopening the mic on the segment's own tail would read as a
                    # barge-in and /stop the live tool turn.
                    await asyncio.sleep(self._cfg.playback_hangover_ms / 1000)
                    if self._closing or epoch != self._sink.epoch:
                        return
                await self._set_turn(VoiceState.THINKING)  # tools running; mic back open
            self._arm_prologue(
                initial_ms=self._cfg.prologue.interval_ms if spoke else None,
                start_step=1 if spoke else 0,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dead watcher must not wedge the mic
            self._log.warning("midturn watch failed ({}); reopening the wait", exc)
            if epoch == self._sink.epoch and self._turn is VoiceState.SPEAKING:
                with suppress(Exception):
                    await self._set_turn(VoiceState.THINKING)

    async def _prologue_watch(
        self, epoch: int, initial_ms: int | None = None, start_step: int = 0
    ) -> None:
        """After ``afterMs`` of THINKING with no reply, speak a neutral filler, then keep the
        wait alive every ``intervalMs``. Cancelled by the first delta / speak_final / barge-in /
        drain; the epoch guard covers the rest.

        Phrases are an escalation script: consumed in order per wait, the last one
        repeating. A SKIPPED filler (user mid-utterance, state moved) does not advance
        the script: nothing was said."""
        try:
            cfg = self._cfg.prologue
            await asyncio.sleep((cfg.after_ms if initial_ms is None else initial_ms) / 1000)
            step = start_step
            while (
                not self._closing
                and self._turn is VoiceState.THINKING
                and epoch == self._sink.epoch
            ):
                if await self._play_filler(epoch, step):
                    step += 1
                await asyncio.sleep(cfg.interval_ms / 1000)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - filler is best-effort, never fatal
            # A raise after the flip to SPEAKING gates a half-duplex mic forever.
            self._log.warning("prologue filler failed ({}); reopening the wait", exc)
            if epoch == self._sink.epoch and self._turn is VoiceState.SPEAKING:
                with suppress(Exception):
                    await self._set_turn(VoiceState.THINKING)

    async def _synth_filler(self, text: str) -> bytes:
        """Synthesize-and-cache one filler phrase (~150 ms at MMS RTF 0.15 for a short
        one). A transient failure is never cached as permanent silence."""
        audio = self._fillers.get(text)
        if audio is None:
            audio = await (
                self._tts.synthesize_pcm(text) if self._pcm_out else self._tts.synthesize(text)
            )
            if audio:
                self._fillers[text] = audio
        return audio

    async def prewarm_fillers(self) -> None:
        """Pre-synthesize the prologue phrases (channel warmup) so filler #1 never pays
        synthesis inside the wait it masks. ``probe_ok`` gates it like the calibrate
        probe: a cloud TTS must never bill at startup (its phrases stay lazy)."""
        if (
            self._tts is None
            or not self._cfg.prologue.enabled
            or not getattr(self._tts, "probe_ok", True)
        ):
            return
        # An escalation script is short; a pathological list must not burn startup
        # synth (the tail stays lazy). Capped, and abandoned the moment a real turn
        # starts: adapters tolerate overlapped synthesis (RKNN serializes, ORT runs
        # concurrently) but the contention would inflate the first reply's TTFA.
        for text in self._cfg.prologue.phrases[:8]:
            if self._closing or self._turn is not VoiceState.IDLE:
                return
            try:
                audio = await self._synth_filler(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - warmup is an optimization, never a gate
                self._log.debug("filler prewarm failed for '{}': {}", text, exc)
                return
            if not audio:
                return  # adapters degrade to b"" — a broken one fails every phrase

    async def _play_filler(self, epoch: int, step: int) -> bool:
        """Speak escalation-script phrase ``step`` (clamped to the last). Returns whether
        the phrase was emitted, so a skip does not advance the script."""
        phrases = self._cfg.prologue.phrases
        if not phrases:
            return False
        text = phrases[min(step, len(phrases) - 1)]
        audio = await self._synth_filler(text)
        if not audio or self._turn is not VoiceState.THINKING or epoch != self._sink.epoch:
            return False
        if self._endpointer.in_speech:
            # A filler now would flip to SPEAKING and gate the half-duplex mic, punching a hole
            # in the very speech the wait yields to.
            return False
        self._metrics.count("prologue_filler")
        # Our own filler must not read back as user speech: half-duplex gates the mic via
        # SPEAKING; soft-duplex needs the echo filter to know the words.
        self._echo.note_spoken(text, hold_ms=self._sink.backlog_ms() + self._audio_ms(audio))
        if self._duck_gain < 1.0 and not self._pcm_out:
            audio = await asyncio.to_thread(_scale_wav, audio, self._duck_gain)
        await self._set_turn(VoiceState.SPEAKING)
        await self._emit(self._audio_event(epoch, audio))
        # Emitted: every path from here returns True.
        if not await self._settle(epoch):  # audibly complete before reopening the mic
            return True
        if self._turn is VoiceState.SPEAKING:
            if not self._full_duplex:
                # Tail guard: let device latency/reverb settle, or the filler re-triggers the VAD.
                await asyncio.sleep(self._cfg.playback_hangover_ms / 1000)
                if self._closing or epoch != self._sink.epoch:
                    return True
            await self._set_turn(VoiceState.THINKING)  # still waiting; mic back open
        return True

    async def _drain_watch(self, epoch: int) -> None:
        """Return to IDLE once the reply finishes playing (and a hangover).

        Gates on ``tts_queue.join()`` FIRST (inside ``_settle``) so synthesis (the slow upstream
        step) completes before we wait on the sink; otherwise the sink looks idle between two
        synthesized chunks and drains early."""
        try:
            if not await self._settle(epoch):
                return  # barge-in started a new turn; it owns the state now
            # Turn over: never carry a duck/pause (or a candidate whose target is gone) into
            # the next one. The VAD floor stays scaled, as in _do_interrupt.
            self._sink.restore_playback()
            self._duck_onset = None
            self._duck_suspect = False
            if not self._full_duplex:
                await asyncio.sleep(self._cfg.playback_hangover_ms / 1000)
                if self._closing or epoch != self._sink.epoch:
                    return
            await self._reset_endpointer()  # skipped mid-utterance; hop-lock serialized
            if self._turn in (VoiceState.SPEAKING, VoiceState.THINKING):
                await self._set_turn(VoiceState.IDLE)
            self._metrics.turn_end()  # release the anchor; the next turn re-arms it
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a dead watcher must not wedge the mic
            # Stranding the turn in SPEAKING is a permanently deaf half-duplex session.
            self._log.warning("drain watch failed ({}); forcing IDLE", exc)
            self._sink.restore_playback()  # never strand the duck floor or the pause gate
            self._duck_onset = None
            if epoch == self._sink.epoch and self._turn in (
                VoiceState.SPEAKING, VoiceState.THINKING
            ):
                with suppress(Exception):
                    await self._set_turn(VoiceState.IDLE)
                self._metrics.turn_end()

    def _track_hop_cost(self, t0: float, compute_ms: float) -> None:
        """Per-frame hop accounting; warn (throttled) only when capture actually lags.

        Slow frames grow response latency by the accumulated capture debt (the pipe buffers
        in order) — up to the pipeline's depth, past which the source drops audio and the
        debt is pinned at the cap rather than integrating fiction. The compute/overhead
        split names the culprit: compute near the budget is the engine stack itself; overhead
        dominating is dispatch/contention (bulk STT/TTS stealing cores), which no VAD or NPU
        change fixes."""
        now = time.monotonic()
        gap_ms = (t0 - self._last_push_end) * 1000.0
        self._last_push_end = now
        if self._probe_hold:
            return  # probes saturate the box on purpose; their samples indict nothing
        total_ms = (now - t0) * 1000.0
        budget = float(self._cfg.audio.frame_ms)
        overhead_ms = max(0.0, total_ms - compute_ms)
        self._hop_compute_ema = 0.9 * self._hop_compute_ema + 0.1 * compute_ms
        self._hop_overhead_ema = 0.9 * self._hop_overhead_ema + 0.1 * overhead_ms
        self._metrics.observe("hop_compute_ms", compute_ms)
        self._metrics.observe("hop_overhead_ms", overhead_ms)
        if gap_ms > _PUMP_GAP_RESET_MS:
            self._capture_debt_ms = 0.0
        self._capture_debt_ms = min(
            _CAPTURE_DEBT_CAP_MS, max(0.0, self._capture_debt_ms + total_ms - budget)
        )
        if self._capture_debt_ms == 0.0:
            self._debt_episode = False  # fully drained: the next crossing is a new episode
        if self._capture_debt_ms < _CAPTURE_DEBT_WARN_MS:
            return
        if not self._debt_episode:
            self._debt_episode = True
            self._metrics.count("capture_behind")  # once per lag episode, never throttled
        if not self._warn_throttle.ready():
            return
        if self._capture_debt_ms >= _CAPTURE_DEBT_CAP_MS:
            state = "capture pipeline is saturated; the source is dropping audio"
        else:
            state = f"capture is ~{self._capture_debt_ms:.0f} ms behind real time"
        if not self._threaded_hop:
            # Light path: compute and total share one clock, so the split cannot
            # attribute; name both suspects.
            hint = "the VAD/STT stack or concurrent inference is over the frame budget"
        elif self._hop_compute_ema > 0.8 * budget:
            hint = "the VAD/AEC/streaming-STT stack is too slow for this device; use lighter engines"
        else:
            hint = "concurrent bulk inference is starving the frame path, not the VAD itself"
        self._log.warning(
            "{} (frame hop ~{:.1f} ms compute + ~{:.1f} ms scheduling/contention vs the "
            "{:.0f} ms budget): {}",
            state, self._hop_compute_ema, self._hop_overhead_ema, budget, hint,
        )
