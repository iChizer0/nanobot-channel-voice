"""LocalBackend: the on-box VAD/STT/TTS pipeline + nanobot over the text bus.

Owns the turn lifecycle (CAPTURING -> THINKING -> SPEAKING -> drain -> IDLE), barge-in, the
self-echo filter, stale-turn guards and the TTS stage; the shell mirrors state via ``StateHint``.
Audio leaves as ``OutputAudio`` events — the direct sink reference is for control only (flush,
duck/pause, epoch, backlog/drain), never for enqueueing. STT/TTS thread inside their adapters,
so every callback here stays on the event loop."""

from __future__ import annotations

import asyncio
import io
import math
import re
import threading
import time
import wave
from collections import deque
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from importlib.metadata import version as _dist_version
from pathlib import Path

from loguru import logger

from nanobot_channel_voice.aio import (
    Throttle,
    cancel_and_wait,
    cancel_task,
    put_drop_oldest,
    wait_until,
)
from nanobot_channel_voice.audio.pcm import (
    ding_pcm,
    dong_pcm,
    fade_tail_pcm,
    pcm_ms,
    pcm_rms,
    pcm_to_wav_bytes,
    quietest_split,
    resample_pcm,
    wav_duration_ms,
    wav_pcm,
)
from nanobot_channel_voice.chunker import SentenceChunker
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.dump import AudioDumper, default_dump_root
from nanobot_channel_voice.echo_reject import SelfEchoFilter, units_of
from nanobot_channel_voice.metrics import VoiceMetrics
from nanobot_channel_voice.phrases import (
    FILLER_WORDS,
    PhraseLexicon,
    PhraseMatcher,
    phrase_within,
    tokens_of,
)
from nanobot_channel_voice.streamid import base_of, started_ns, unique_token
from nanobot_channel_voice.tts.base import TtsAdapter, is_wav
from nanobot_channel_voice.vad import Endpointer, Vad, flag_lag_ms, resolve_preroll_ms
from nanobot_channel_voice.vad.adaptive import AdaptiveHangover
from nanobot_channel_voice.wake.phrase import FuzzyWake, WakePhrase

from .audio_sink import AudioSink, scale_pcm, trim_lead_silence, trim_tail_silence
from .base import OnEvent, OutputAudio, ToolDef, VoiceState
from .common import TurnEventMixin, loggable_text

TranscribeFn = Callable[[bytes], Awaitable[str]]
# (text, turn_token, notes); notes ride the context bridge keyed by the token, keeping the
# persisted user row pure speech (see channel._publish_turn_text).
PublishTextFn = Callable[[str, str, tuple[str, ...]], Awaitable[None]]
InterruptFn = Callable[[], Awaitable[None]]

# JIT synthesis (stream mode): a call starts when the unplayed runway drains to its predicted
# cost (per-char EMA x SAFETY + MARGIN) and shrinks to fit; the ahead cap bounds a barge-in flush.
_SYNTH_AHEAD_CAP_S = 4.0
_SYNTH_LEAD_SAFETY = 2.0
_SYNTH_LEAD_MARGIN_S = 0.4
_JIT_POLL_CAP_S = 0.25  # wait granularity: pause freeze / barge-in / fresh text re-check
# Cost EMA: slow samples lift it instantly (undershoot gaps audibly, overshoot only
# synthesizes early); clamped against outliers, floored against near-instant adapters.
_MPC_ALPHA = 0.3
_MPC_MIN = 1e-3
_GAP_COUNT_MIN_MS = 20.0  # dry spells under this are inaudible

# AEC3 convergence budget, in ACCEPTED REFERENCE AUDIO not wall time: until then the residual
# transcribes as "fresh words", so early-confirm holds back (the endpoint verdict still decides).
_AEC_WARMUP_REF_MS = 3000.0

# Capture debt = unpaid frame-hop overrun; response latency grows by exactly this much. A
# pump-idle arrival gap (mic gate, capture restart) means the pipe was flushed: debt resets.
_CAPTURE_DEBT_WARN_MS = 500.0
_PUMP_GAP_RESET_MS = 1000.0
# The pipe is finite (arecord 64 KB ~= 2 s at 16 kHz S16LE), so lag saturates at its depth and
# further deficit is DROPPED audio; uncapped, the integral reports fiction ("~17615 ms behind").
_CAPTURE_DEBT_CAP_MS = 2000.0

# Double-tap grace for a stop landing after the first kill already IDLEd the session. Armed ONLY
# by kills a stop consume performed (a content barge-in's kill would swallow "cancel" answering
# the agent's next question) and never by a consume that killed nothing.
_KILL_GRACE_S = 3.0

# Post-acquittal engage holdoff: resumed playback re-leaks at once and the candidate loop would
# flap at ~1 Hz. Engagement is state-driven, so expiry re-engages mid-utterance.
_ENGAGE_HOLDOFF_S = 0.5

# Early-RELEASE, DUCK MODE ONLY: this many consecutive zero-fresh-word partial polls, no sooner
# than this far into the candidate (the floor absorbs decoder latency), restore the level before
# the verdict. Pause mode takes no transcript acquittal — a wrong one there DROPS real speech.
_RELEASE_POLLS = 2
_EARLY_RELEASE_MS = 600.0

# Pause-probe silence floor before the leak attribution is read; above VAD flicker (<~150 ms).
_PROBE_SILENCE_MS = 200.0

# High false-candidate rate = the operator-visible symptom of weak/missing AEC. Only
# leak-shaped acquittals count; backchannels/blips are conversation, not echo evidence.
_FALSE_WARN_WINDOW_S = 60.0
_FALSE_WARN_N = 10
_LEAK_REASONS = frozenset({"probe", "partial", "eager", "echo", "empty"})

# A wake hit binds to the utterance whose VAD onset it follows within this window (detection
# trails the phrase's END, the onset precedes its START). An unconsumed hit goes stale.
_WAKE_ATTACH_S = 2.5

# Fast-path ack pacing: the quiet bar outlasts inter-word gaps (~100-250 ms) yet beats the verdict
# ack's hangover + STT floor (~1 s); quiet past the window is a same-breath COMMAND's hangover.
_FAST_ACK_QUIET_S = 0.35
_FAST_ACK_MIN_QUIET_MS = 240
_FAST_ACK_POLL_S = 0.07
_FAST_ACK_WINDOW_S = 0.8

# At-close ack shape gate: a bare summon (phrase + lead fillers) stays under this; wake+command
# runs longer and must wait for its reply instead of a spurious ack.
_CLOSE_ACK_MAX_ACTIVE_MS = 1300

# Past this a cue is a jingle: it gates the half-duplex mic and delays what queues behind it.
_EARCON_MAX_MS = 600
_EARCON_MAX_FILE_B = 2_000_000  # refuse absurd files unread: ~10 s of 48 k stereo already


def _swallow_result(task: asyncio.Task) -> None:
    """Retrieve an abandoned decode's outcome: no "exception was never retrieved"."""
    if not task.cancelled():
        task.exception()


# prologue.phrases=None: built-ins keyed to the TTS language; an unspeakable phrase is silence.
_PROLOGUE_BUILTINS = {
    "zh": ["稍等。", "还在处理中。", "还需要一点时间。"],
    "ja": ["少々お待ちください。", "まだ作業中です。", "もう少し時間がかかりそうです。"],
    "ko": ["잠시만요.", "아직 처리 중입니다.", "시간이 조금 더 걸릴 것 같습니다."],
    "de": ["Einen Moment.", "Ich arbeite noch daran.", "Es dauert noch einen Moment."],
}
_PROLOGUE_FALLBACK = ["One moment.", "Still working on it.", "This is taking a little longer."]

# First-filler delay = this multiple of typical first-reply latency; at TYPICAL latency the
# filler collides with the answer and delays it behind its own audio.
_PROLOGUE_TTFT_FACTOR = 1.5


def _prologue_phrases(configured: list[str] | None, language: str | None) -> list[str]:
    if configured is not None:
        return configured  # explicit always wins; [] = script off
    return _PROLOGUE_BUILTINS.get(language or "", _PROLOGUE_FALLBACK)


# wake.ack.phrases=None: built-ins keyed like the prologue's. Statement forms only (rising
# interjections die on flat prosody); never a wake phrase (it would echo-veto our own hits).
_WAKE_ACK_BUILTINS = {
    "zh": ["在呢。", "我在。"],
    "ja": ["はい。"],
    "ko": ["네."],
    "de": ["Ja?"],
}
_WAKE_ACK_FALLBACK = ["Yes?", "I'm here."]


def _wake_ack_phrases(configured: list[str] | None, language: str | None) -> list[str]:
    if configured is not None:
        return configured
    return _WAKE_ACK_BUILTINS.get(language or "", _WAKE_ACK_FALLBACK)


# Ack language routing by the called name's script. Kana beats han (ja mixes both), han alone
# reads zh; latin is en/de-ambiguous and defers to the TTS.
_SCRIPT_ROWS = {"han": "zh", "kana": "ja", "hangul": "ko"}


def _script_class(text: str) -> str | None:
    seen_han = seen_latin = False
    for ch in text:
        cp = ord(ch)
        if 0x3040 <= cp <= 0x30FF:
            return "kana"
        if 0xAC00 <= cp <= 0xD7AF or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            return "hangul"
        if 0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF:
            seen_han = True
        elif cp < 0x2E80 and ch.isalpha():
            seen_latin = True
    if seen_han:
        return "han"
    return "latin" if seen_latin else None


def _uniform_script(phrases: list[str]) -> str | None:
    classes = {_script_class(p) for p in phrases} - {None}
    return classes.pop() if len(classes) == 1 else None


# Contract-regression rates, never enforcement. Line-start markers anchor on a real separator,
# never re.M's ^ (a delta begins mid-line, where ^ read "9 - ten" as a list); the separator set
# matches the chunker's _line_start_at. _WAIT_PHRASES is stall-claims only.
_MD_PROBE = re.compile(r"```|[\r\n][ \t]{0,3}(?:#{1,6}\s|[-*+]\s)|\[[^\]]+\]\([^)]+\)|\*\*")
_MD_CARRY = 10  # len("\n   ###### ") - 1: no line-start marker is lost at a delta seam
_WAIT_PHRASES = tuple(dict.fromkeys(
    [
        phrase.rstrip(".。").lower()
        for phrases in (*_PROLOGUE_BUILTINS.values(), _PROLOGUE_FALLBACK)
        for phrase in phrases
    ]
    + ["one moment", "just a moment", "hold on", "请稍等"]
))


def _opens_with_wait_phrase(text: str) -> bool:
    return text.lstrip(" \t\r\n\"'“”‘’(（[【").lower().startswith(_WAIT_PHRASES)


_NOTE_CJK_FLOOR = 0x2E80  # same split as echo_reject.py, wake/phrase.py, tts/router.py


def _heard_tail(heard: str, *, max_words: int = 12, max_cjk_chars: int = 20) -> str:
    """Bound a heard-up-to quote to its tail (the cut point is the only new information):
    words for Latin, characters for an unspaced CJK tail."""
    heard = " ".join(heard.split())
    if any(ord(ch) >= _NOTE_CJK_FLOOR for ch in heard[-6:]):
        cut, tail = len(heard) > max_cjk_chars, heard[-max_cjk_chars:]
    else:
        parts = heard.split(" ")
        cut, tail = len(parts) > max_words, " ".join(parts[-max_words:])
    if len(tail) > 80:  # a mid-text unspaced run escapes both counts: hard cap, chars
        cut, tail = True, tail[-80:]
    return f"…{tail}" if cut else tail


def _interrupt_marker(heard: str | None) -> str | None:
    """The heard-up-to event note riding an interrupting utterance's publish. ``heard`` is
    the heard text ("" = cut before anything sounded), None = marker disabled/blob mode."""
    if heard is None:
        return None
    return (
        f'[voice event: you were interrupted mid-reply; the user heard up to: "{_heard_tail(heard)}"]'
        if heard
        else "[voice event: you were interrupted before your reply was heard]"
    )


def _steer_marker(heard: str | None) -> str | None:
    """The heard-up-to note for a mid-turn steer. NOT ``_interrupt_marker``: nothing was
    cancelled, and a model told it was interrupted apologizes or starts the answer over."""
    if heard is None:
        return None
    return (
        "[voice event: the user spoke while you were working; they heard up to: "
        f'"{_heard_tail(heard)}"]'
        if heard
        else "[voice event: the user spoke while you were working]"
    )


def _wake_note(heard: str | None) -> str:
    """The pending note a wake-word kill leaves for the NEXT publish (the wake
    itself publishes nothing) — same contract as a consumed stop's note."""
    if heard:
        return (
            "[voice event: the user cut your reply short with the wake word; "
            f'they heard up to: "{_heard_tail(heard)}"; do not resume it unless asked]'
        )
    if heard == "":
        return (
            "[voice event: the user cut your reply short with the wake word "
            "before hearing it; do not resume it unless asked]"
        )
    return (
        "[voice event: the user cut your reply short with the wake word; "
        "do not resume it unless asked]"
    )


def _stop_note(stop_text: str, heard: str | None) -> str:
    """The pending note a CONSUMED stop leaves for the NEXT publish (a consumed command elicits
    no reply); the "do not resume" clause blocks an "as I was saying..." reopen. ``heard`` None
    = accounting unavailable: make NO claim; only "" may claim the cut landed before sound."""
    if heard:
        return (
            f'[voice event: the user stopped your previous reply with "{stop_text}"; '
            f'they heard up to: "{_heard_tail(heard)}"; do not resume it unless asked]'
        )
    if heard == "":
        return (
            f'[voice event: the user stopped your previous reply with "{stop_text}" '
            "before hearing it; do not resume it unless asked]"
        )
    return (
        f'[voice event: the user stopped your previous reply with "{stop_text}"; '
        "do not resume it unless asked]"
    )


class _AbandonCalibrationError(Exception):
    """A real turn started mid-calibration: stop, keep what was learned."""


@dataclass(slots=True)
class _PendingUtterance:
    """One endpointed utterance, snapshotted AT CLOSE TIME so processing can be deferred;
    ``closed_at`` back-dates the metrics anchor past the queue wait. ``eager_always_valid``:
    a streaming finish task saw exactly this audio, so it survives any close reason (eager
    speculation does not survive a max-length close)."""

    pcm: bytes
    eager: asyncio.Task | None
    closed_reason: str  # Endpointer.closed_reason: "silence" | "max" | "eou"
    closed_at: float
    silence_ms: int = 0  # trailing silence the close consumed (Endpointer.closed_silence_ms)
    raw: bytes | None = None  # pre-AEC span of pcm, for the audio dump (None = no AEC/dump)
    learn_ms: float | None = None  # adaptive-hangover candidate bound to THIS utterance
    eager_always_valid: bool = False
    # Diagnostics snapshotted at close (endpointer counters die in its reset); seg_id is shared
    # by the summary log line and the dump filename, meta feeds the dump manifest.
    active_ms: int = 0
    prob_peak: float | None = None
    prob_mean: float | None = None
    seg_id: int = 0
    meta: dict | None = None
    # Early-confirm state bound AT CLOSE: as globals, another queued utterance could consume
    # them (slow STT, two in flight) and skip a needed /stop.
    preempted: bool = False
    heard: str | None = None
    # Turn state AT VAD ONSET (plus its wall time): stop targeting is decided by when the user
    # STARTED speaking, or a reply draining during the stop's own STT window launders it cold.
    onset_interrupting: bool = False
    onset_at: float = 0.0
    # Wake-gate evidence bound at close: a tier hit claimed this utterance, whether its onset
    # fell while the bot was AUDIBLY speaking (strict gates those always, and THINKING onsets
    # once the window is shut), and whether the window was open AT ONSET — judged then, never
    # at the verdict, or a later wake retroactively blesses queued speech.
    wake_hit: bool = False
    # Leading wake-phrase bytes of ``pcm``, hidden from STT so it cannot smear them in. 0 = none.
    trim_bytes: int = 0
    # Trim baked into the eager snapshot; that speculation is valid only when it == trim_bytes.
    eager_trim: int = 0
    onset_speaking: bool = False
    onset_window_open: bool = False


class _Turn:
    """Everything owned by ONE published turn: stage latches, watchers, the stream base.
    Created FRESH at every publish (no reset can be forgotten), abandoned as a unit on
    barge-in. ``idle()`` is the placeholder before the first publish, so readers need no
    None guard."""

    __slots__ = (
        "published_at", "chunk_await", "tts_first_pending", "await_first_token",
        "segment_spoke", "prologue_task", "midturn_task", "timeout_task", "base",
        "last_activity", "last_audible", "dead", "tokens", "audible_at",
        "continuation_pending",
        "md_counted", "md_carry", "segment_first",
        "spoke_text", "emitted_audio", "fallback_done", "answered", "proactive",
    )

    def __init__(self, token: str = ""):
        self.published_at = time.monotonic()
        # Every token published under this turn (publish + each injection): core echoes inbound
        # metadata onto the final, so a straggler may carry ANY of them and all must die
        # together (see streamid.TURN_META).
        self.tokens = [token] if token else []
        self.dead = False   # abandoned by a barge-in; nothing may interrupt it twice
        # One-shot TTFA stage timers: first speakable chunk, first synthesis, first delta.
        self.chunk_await = True
        self.tts_first_pending = True
        self.await_first_token = True
        self.segment_spoke = False  # current stream segment produced audible chunks
        self.prologue_task: asyncio.Task | None = None
        self.midturn_task: asyncio.Task | None = None
        self.timeout_task: asyncio.Task | None = None  # stalled-agent deadman
        self.base: str | None = None  # learned from the first delta's stream id
        self.last_activity = self.published_at  # any delta/segment end pushes this
        # Words out (see _note_spoken); separate from last_activity, which a working tool chain
        # pushes forever and so can never measure dead air.
        self.last_audible = self.published_at
        self.audible_at: float | None = None  # first audio frame emitted (adaptive hangover)
        # A resuming stream end passed: the next delta (earliest observable resume edge)
        # re-anchors the metrics clock so tool time never lands in ttfa_ms.
        self.continuation_pending = False
        self.md_counted = False  # reply_markdown counted once per turn
        self.md_carry = "\n"     # delta-seam tail (see _note_reply_markdown)
        # First chunk of the CURRENT segment; judged for reply_wait_phrase only if the segment
        # is the answer-bearing one (the same opener BEFORE a tool call is a status line).
        self.segment_first: str | None = None
        # Audibility ledger for the unvoiced-final fallback; speak_final resets the trio (the
        # IDLE placeholder is shared across unsolicited deliveries).
        self.spoke_text = False
        self.emitted_audio = False
        self.fallback_done = False
        # An ANSWER segment produced text; a pre-tool status line does not count.
        self.answered = False
        # Agent-initiated delivery: its settle re-opens sentence attention (no re-wake needed).
        self.proactive = False

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
        self.segment_first = None


def _dump_session_header(config: VoiceConfig, frame_ms: int) -> dict:
    """First manifest record: the tuning-relevant config in effect, so records from different
    threshold experiments stay attributable. Curated — active VAD engine only, no model paths;
    ``neg_threshold`` stays as configured (null = engine-default hysteresis)."""
    vad = config.vad
    detector: dict = {"engine": vad.engine}
    if vad.engine == "energy":
        detector["threshold"] = vad.energy_threshold  # 0 = adaptive noise floor
    elif vad.engine == "webrtc":
        detector["aggressiveness"] = vad.aggressiveness
    elif vad.engine == "firered":
        detector.update(
            threshold=vad.firered.threshold,
            smooth_frames=vad.firered.smooth_frames,
            min_volume=vad.firered.min_volume,
        )
    elif vad.engine == "silero":
        detector.update(
            threshold=vad.silero.threshold,
            neg_threshold=vad.silero.neg_threshold,
            min_volume=vad.silero.min_volume,
        )
    header = {
        "wall": round(time.time(), 3),
        "sample_rate": config.audio.sample_rate,
        "frame_ms": frame_ms,
        "vad": detector,
        "start_frames": vad.start_frames,
        "preroll_ms": resolve_preroll_ms(vad, frame_ms),
        "hangover_ms": vad.hangover_ms,
        "min_utterance_ms": vad.min_utterance_ms,
        "max_utterance_ms": vad.max_utterance_ms,
    }
    if vad.hangover_min_ms is not None:
        header["hangover_min_ms"] = vad.hangover_min_ms
    if vad.turn.engine != "none":
        header["turn"] = {
            "engine": vad.turn.engine,
            "threshold": vad.turn.threshold,
            "consult_ms": vad.turn.consult_ms,
        }
    if config.wake.mode != "off":
        header["wake"] = {
            "mode": config.wake.mode,
            "engine": config.wake.engine,
            "window_s": config.wake.window_s,
            "attention": config.wake.attention,
        }
        if config.wake.engine == "openwakeword":
            header["wake"]["threshold"] = config.wake.openwakeword.threshold
    with suppress(Exception):  # uninstalled source tree: only the version is lost
        header["version"] = _dist_version("nanobot-channel-voice")
    return header


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


# How far past the budget a cut may reach for a word boundary: ~one long word, ~100 ms.
_CUT_LOOKAHEAD = 16
_CUT_SENT = set(".!?…。！？")
_CUT_CLAUSE = set(",;:，、；：")


def _cut_index(text: str, budget: int, floor: int) -> int:
    """Cut for an over-budget chunk: last sentence end in [floor, budget], else clause
    punctuation, else space, else the next word boundary just past it — TTS front-ends
    terminate fragments, so a bad seam gets sentence-final prosody. Ranges INCLUDE ``floor``
    (exclusive ones are empty at ``budget == floor``, and the slice lands mid-word); the
    look-ahead overshoots by at most a word, which the 2x synthesis lead absorbs."""
    window = min(len(text), budget)
    for punct in (_CUT_SENT, _CUT_CLAUSE):
        for i in range(window - 1, floor - 2, -1):
            # ASCII punct binds to a following alnum ("3.14", "1,234"); CJK stands alone.
            if text[i] in punct and (
                ord(text[i]) >= 0x2E80 or i + 1 >= len(text) or not text[i + 1].isalnum()
            ):
                return i + 1
    space = text.rfind(" ", max(0, floor - 1), window)
    if space > 0:
        return space + 1
    # Only a spaced script can be cut mid-word; CJK slices cleanly at any character,
    # and one CJK "word" of look-ahead would be a second of unbudgeted speech.
    if (
        window < len(text)
        and text[window - 1].isalnum()
        and text[window].isalnum()
        and ord(text[window]) < 0x2E80
    ):
        nxt = text.find(" ", window)
        end = len(text) if nxt < 0 else nxt  # no space left: a short tail goes whole
        if end <= window + _CUT_LOOKAHEAD:
            return min(end + 1, len(text))
    return window


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
        wake_detector=None,  # a WakeDetector (acoustic tier), or None (text tier only)
    ):
        self._cfg = config
        self._tts = tts
        self._prologue_phrases = _prologue_phrases(
            config.prologue.phrases, getattr(tts, "spoken_language", None)
        )
        self._wake_ack_list = _wake_ack_phrases(
            config.wake.ack.phrases, getattr(tts, "spoken_language", None)
        )
        self._ack_step = 0  # round-robin cursor
        self._ack_task: asyncio.Task | None = None
        self._fast_ack_task: asyncio.Task | None = None
        # One-shot stamp the verdict rungs consume: a fast-acked summon never acks twice.
        self._fast_acked_at = float("-inf")
        self._first_reply_ema: float | None = None  # ms; feeds _prologue_delay_ms
        self._sink = sink
        self._transcribe = transcribe
        self._publish_text = publish_text
        self._interrupt = interrupt
        # Shared with the channel: local TTFA lands in the same collector as the cloud path's.
        self._metrics = metrics if metrics is not None else VoiceMetrics()

        # Software AEC: capture frames pass it before the VAD/STT (same hop), sink-fed reference.
        self._aec = aec
        self._full_duplex = config.full_duplex or aec is not None
        # Mic open while speaking -> the echo filter stops the bot barging in on itself (with
        # AEC, on the residual); user speech still interrupts.
        self._open_mic = config.open_mic
        # Protected stop words survive Latin-respacing absorption ("stop" inside spoken
        # "unstoppable"): the kill switch stays fresh evidence.
        self._echo = SelfEchoFilter(
            config.echo_reject_threshold,
            protect=units_of(" ".join(config.barge_in.stop_phrases)),
        )
        self._vad = vad  # kept for duck floor scaling and release() at close
        # Continuation hysteresis: half the onset bar while THINKING, so follow-ups confirm faster.
        self._cont_start_frames = max(1, config.vad.start_frames // 2)
        self._vad_heavy = getattr(vad, "heavy", False)  # neural VAD -> run per-frame off the loop
        # The adaptive-hangover floor (see vad.adaptive) may never undercut the consult tier, or
        # the silence close preempts the turn model on every pause and it silently never runs.
        hangover_floor = config.vad.hangover_min_ms or config.vad.hangover_ms
        adaptive = config.vad.hangover_min_ms is not None
        if turn_analyzer is not None:
            tier_floor = config.vad.turn.consult_ms + config.audio.frame_ms
            if hangover_floor < tier_floor:
                logger.warning(
                    "voice: {} ({}) undercuts vad.turn.consultMs ({}); raising {} to "
                    "{} ms so the turn model can fire",
                    "vad.hangoverMinMs" if adaptive else "vad.hangoverMs", hangover_floor,
                    config.vad.turn.consult_ms,
                    "the adaptive floor" if adaptive else "the endpointing hangover",
                    tier_floor,
                )
                hangover_floor = tier_floor
        if eager_ms and eager_ms >= hangover_floor:
            # The eager mark is taken inside the hangover: at/past the FLOOR it never fires.
            clamped = hangover_floor - config.audio.frame_ms
            logger.warning(
                "voice: stt.eagerMs ({}) is not inside the {} ms endpointing hangover "
                "floor; {}",
                eager_ms, hangover_floor,
                f"clamping to {clamped} ms" if clamped > 0 else "eager STT disabled",
            )
            eager_ms = max(0, clamped)
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
        self._hangover_floor = hangover_floor  # what the eager overlap is measured against
        # End-of-turn model (vad.turn): consulted once per pause off-loop; a COMPLETE verdict
        # parks its gen in _eou_close_gen for the next frame (GIL-atomic write, as _early_confirm).
        self._turn_analyzer = turn_analyzer
        self._consult_task: asyncio.Task | None = None
        self._eou_close_gen: int | None = None
        self._consult_fail_throttle = Throttle()
        # Streaming STT decodes DURING speech; only the tail flush is left at the endpoint. At
        # onset the ring below is replayed in, so the stream hears what the utterance keeps.
        self._stt_stream = stt_stream
        # The LIVE utterance's caller-owned handle (see stt.base.SttStream): fresh at each onset,
        # taken at the endpoint (for its finish thread), dropped for a rejected blip. Writes are
        # serialized by the one-frame-at-a-time push await plus the hop lock.
        self._stt_live = None
        frame_ms = config.audio.frame_ms
        ring = (resolve_preroll_ms(config.vad, frame_ms) // frame_ms) + config.vad.start_frames
        self._recent: deque[bytes] = deque(maxlen=max(1, ring))
        # Eager (speculative) STT of the utterance-so-far, consumed or discarded at the endpoint.
        # STRICTLY one decode in flight: it cannot be aborted, and with a slow ASR (whisper RTF
        # ~0.6 on device) stacking starves the one that matters.
        self._eager_task: asyncio.Task | None = None
        self._eager_trim = 0  # wake trim baked into the CURRENT slot's snapshot
        self._eager_valid = False  # the in-flight task belongs to the CURRENT silence run
        self._worker_decoding = False  # the utterance worker's final decode is in flight
        # The pump enqueues, the worker serializes: an INLINE multi-second STT (RTF 0.6 x 5 s =
        # 3 s) would overrun the ~2 s arecord pipe and leave the VAD deaf to barge-ins.
        self._utt_queue: asyncio.Queue[_PendingUtterance] = asyncio.Queue(maxsize=4)
        self._utt_task: asyncio.Task | None = None
        # Capture-segment id shared by the summary line and the dump filename (writes reorder).
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
        self._wake_confirm = False   # the pending early confirm is wake-driven (wake_kill_ms)
        self._partial_countdown = 0
        # Frames per streaming-partial poll: ~100 ms of audio at any frame size.
        self._partial_every = max(1, 100 // max(1, config.audio.frame_ms))
        # Serializes the frame hop against loop-side resets.
        self._hop_lock = threading.Lock()
        # Does per-frame work thread at all? The LIGHT path (energy/webrtc VAD, batch STT, no
        # AEC, no acoustic wake) pushes the endpointer on the loop, so its resets must run inline
        # too or the lock protects only one side of the race. An acoustic wake detector forces
        # the threaded hop: heavy-class inference, and fed only inside _push_frame_sync.
        self._threaded_hop = (
            self._vad_heavy
            or self._stt_stream is not None
            or self._aec is not None
            or wake_detector is not None
        )
        # Backchannel ignore-list and stop-command lexicon (see _on_utterance's ladder); the
        # matchers precompute merged vocabularies once — per-call unions churned the hop path.
        self._ack_lex = PhraseLexicon(config.barge_in.ack_phrases)
        self._stop_lex = PhraseLexicon(config.barge_in.stop_phrases)
        # In the echo filter's UNIT alphabet: backchannels subtract from fresh evidence, any script.
        self._ack_words = units_of(" ".join(config.barge_in.ack_phrases))
        self._ack_match = PhraseMatcher(self._ack_lex)
        self._stop_match = PhraseMatcher(
            self._stop_lex, self._ack_lex, extra=FILLER_WORDS
        )
        # Deliberately NOT echo-protected like stop phrases: a goal is costly to start by accident.
        goal_phrases = config.goal.phrases
        self._goal_lex = PhraseLexicon(goal_phrases) if goal_phrases else None
        self._min_fresh_words = config.barge_in.min_words
        # Wake-word gate: "gate" requires the phrase to START a conversation from cold (window
        # shut); "strict" additionally to barge into a live reply (while SPEAKING, non-wake
        # speech neither ducks nor confirms — hit-or-ignore) and to steer a shut-window THINKING
        # turn. Two tiers feed one claim: the acoustic detector (hop) and the transcript prefix.
        wake_phrase = (
            WakePhrase(self._wake_entries()) if config.wake.mode != "off" else None
        )
        self._wake_phrase = wake_phrase if wake_phrase else None  # falsy = nothing matchable
        # Strip-only phonetic tier for mangled latin names; see FuzzyWake.
        fuzzy = FuzzyWake(config.wake.phrases) if config.wake.mode != "off" else None
        self._wake_fuzzy = fuzzy if fuzzy else None
        self._wake_detector = wake_detector
        wake_mode = config.wake.mode
        if wake_mode != "off" and self._wake_phrase is None and wake_detector is None:
            # Zero working tiers would gate every utterance FOREVER: a healthy-looking but
            # permanently deaf channel. Only degenerate (unmatchable) phrase lists land here.
            logger.warning(
                "voice: wake.mode='{}' has no working tier (no matchable phrases, "
                "no acoustic detector); wake gating disabled",
                wake_mode,
            )
            wake_mode = "off"
        if (
            wake_mode == "strict"
            and self._stt_stream is None
            and wake_detector is None
        ):
            # Legal but slow: the phrase is only seen at the eager/endpoint decode, so a wake
            # kill lands ~1 s+ later with the reply at full volume throughout.
            logger.warning(
                "voice: wake.mode='strict' with a batch STT and no acoustic "
                "engine confirms barge-in only at the eager/endpoint decode; "
                'configure wake.engine="openwakeword" or a streaming STT '
                "(zipformer) for prompt wake kills"
            )
        self._wake_mode = wake_mode
        # For the acoustic-hit echo check: our own reply saying the phrase must not wake us.
        self._wake_phrases_text = tuple(config.wake.phrases)
        # Ack routing fallback for acoustic-only summons (no matched text).
        self._wake_phrases_script = _uniform_script(config.wake.phrases)
        self._wake_window_s = config.wake.window_s
        self._wake_attention = config.wake.attention
        self._wake_until = 0.0  # attention deadline (monotonic); cold onsets past it are gated
        # Last SYNTHESIZED reply segment, for sentence-attention's question
        # check (an interrupted reply's unheard tail may misjudge — accepted).
        self._reply_tail = ""
        # Last hit and its loop-side consumption watermark; hop-thread writes are GIL-atomic.
        self._wake_hit_at = float("-inf")
        self._wake_seen_at = float("-inf")
        self._wake_claimed = False  # the hit is bound to the OPEN utterance; dies at close/reset
        # Acoustic hit position in the endpointer's stream coordinate (bytes), None before the
        # first hit. Monotonic, so a stale value lies BEFORE any later buffer and trims nothing.
        self._wake_hit_pos: int | None = None
        # Streaming-STT restart handshake: armed loop-side once the hit survives the vetoes,
        # consumed by the next hop (a fresh handle drops phrase audio the live stream ate).
        self._stt_restart = False
        self._stt_restarted = False  # the CURRENT utterance's stream excludes the phrase
        # Canned speech (filler=THINKING, ack=IDLE) plays as SPEAKING for the mic gate but is
        # NOT a reply: onsets resolve against the state beneath it, and killing it is a flush,
        # never a /stop. The nonce keys the flag to its playback, so a cancelled predecessor's
        # cleanup cannot wipe the successor's.
        self._canned_base: VoiceState | None = None
        self._canned_nonce: object | None = None
        # Stop-command targeting: turn state latched at VAD onset (see _PendingUtterance),
        # and the wall time of the last stop-consume kill for the double-tap grace.
        self._onset_interrupting = False
        self._onset_speaking = False
        self._onset_window_open = False
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
        # Pause-probe leak-death window, derived (never a knob): after a pause engages, leak
        # keeps flagging for the sink's write-ahead + device playout + VAD decision lag. Speech
        # whose LAST flag falls inside that window is the buffered tail; beyond it, a person.
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
        # False-candidate rate window for the operator warning.
        self._false_times: deque[float] = deque()
        self._false_warn = Throttle(_FALSE_WARN_WINDOW_S)
        # Heard-up-to accounting (stream mode): (chunk text, duration ms) of the CURRENT segment
        # in playback order. A barge-in maps the sink's played_ms through these into a bracketed
        # note — the stand-in for history truncation, which a channel cannot do.
        self._spoken_spans: list[tuple[str, float]] = []
        # played_ms offset of the segment's first chunk: the stream may predate the segment (a
        # cancelled filler's stream is reused), so spans map from played-base, not stream open.
        # _spans_gen pins the stream measured on — a fresh stream restarts played_ms() at 0.
        self._spans_base_ms = 0.0
        self._spans_gen = -1
        # Text of this turn's PRIOR segments that played out (folded in by _settle): without it
        # a barge-in during a tool wait reports "nothing heard" after a fully-heard status line.
        self._heard_prefix = ""
        self._early_heard: str | None = None  # heard text at an EARLY confirm, consumed at close

        # Epoch tagged at enqueue: a chunk queued before a barge-in dies before synthesis.
        self._tts_queue: asyncio.Queue[tuple[int, str]] = asyncio.Queue()
        self._tts_task: asyncio.Task | None = None
        self._drain_task: asyncio.Task | None = None
        # Per-char synth cost (ms) EMA for the JIT schedule; the worker's first chunk seeds it.
        self._synth_mpc: float | None = None

        self._turn = VoiceState.IDLE
        self._on_event: OnEvent | None = None
        self._closing = False
        self._cur_turn = _Turn.idle()
        # Tokens of KILLED turns (barge-in / timeout), for send()'s stale-reply gate. A merely
        # superseded turn is NOT here: core may coalesce a queued follow-up into its running
        # turn, and that combined reply must speak.
        self._dead_tokens: deque[str] = deque(maxlen=8)
        # Raw-PCM output follows the SINK's mode (the channel derived it from the TTS adapter),
        # so we can never emit pcm into a blob sink or vice versa.
        self._pcm_out = sink.stream_mode
        # Phrase -> audio, session-scoped: prewarm_canned() fills it (probe_ok only), else lazily.
        self._fillers: dict[str, bytes] = {}
        # Notification cues (earcons.*), built at init. None = off, or an unbuildable cue.
        self._earcon_audio: bytes | None = None
        self._earcon_task: asyncio.Task | None = None
        self._attention_audio: bytes | None = None
        self._attention_task: asyncio.Task | None = None
        self._attention_cued = True  # one close cue per episode; no episode yet
        want_attention = config.earcons.attention
        if want_attention and self._wake_mode == "off":
            logger.warning(
                "voice: earcons.attention marks the wake window closing; "
                "wake gating is off, cue disabled"
            )
            want_attention = False
        if (config.earcons.captured or want_attention) and (
            self._pcm_out and not getattr(tts, "output_rate", None)
        ):
            logger.warning("voice: earcons need a TTS stream rate on a pcm sink; cues disabled")
        else:
            if config.earcons.captured:
                self._earcon_audio = self._build_earcon(config.earcons.path, ding_pcm)
            if want_attention:
                self._attention_audio = self._build_earcon(
                    config.earcons.attention_path, dong_pcm
                )
        # Stream identity is "<turn-base>:<segment>", the base stable across a turn. The live
        # base rides the _Turn; the barged-out base stays here so a DEAD turn's late deltas keep
        # dropping after the turn object is gone.
        self._rejected_base: str | None = None
        # Watermark over the base's embedded start time (time_ns), covering what _rejected_base
        # cannot: a barge-in DURING THINKING never learned the cancelled turn's base, so its
        # late deltas would garble the new turn.
        self._reject_started_before_ns = 0
        # Frame-hop accounting: compute (inside the hop lock) vs overhead (dispatch, lock wait,
        # loop resume) EMAs attribute a slow hop to the engine or to contention; capture debt is
        # the resulting real backlog, and gates the warning.
        self._hop_compute_ema = 0.0
        self._hop_overhead_ema = 0.0
        self._capture_debt_ms = 0.0
        self._debt_episode = False  # currently in a warned lag episode (metric edge)
        self._last_push_end = 0.0
        self._probe_hold = False  # warmup/calibration probes: drop their hop samples
        self._warn_throttle = Throttle()
        self._log = logger.bind(component="voice")
        # Audio dump (debug.dumpAudio): every endpointed segment leaves as a verdict-named WAV.
        # A setup failure costs the diagnostics, never the session.
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
                    header=_dump_session_header(config, frame_ms),
                )
            except Exception as exc:  # noqa: BLE001
                self._log.warning("audio dump disabled ({})", exc)
        self._endpointer.keep_rejected = self._dumper is not None
        # Rolling pre-AEC mirror for the .raw.wav twins, sized to one whole segment (pre-roll +
        # confirm + max utterance). A segment is always the stream's trailing bytes, so its twin
        # is a tail slice; without AEC the segment already IS the raw audio.
        self._dump_raw: deque[bytes] | None = None
        if self._dumper is not None and aec is not None:
            self._dump_raw = deque(
                maxlen=config.vad.start_frames
                + resolve_preroll_ms(config.vad, frame_ms) // frame_ms
                + config.vad.max_utterance_ms // frame_ms
                + 2
            )

    def hold_hop_accounting(self, active: bool) -> None:
        """Drop hop samples while a warmup/calibration probe saturates the device: they would
        indict the steady state. Release also discards the burst's debt — it drains in ms once
        the box idles, so carrying it would only seed a false warning."""
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
        tts_ms_per_char: float | None = None,
        chunk_floor_pinned: bool,
    ) -> None:
        """Feed warm steady-state measurements into the PACING knobs (perf.calibrate). Only
        pacing is derived — correctness knobs (hangover, thresholds, duplex, echo) stay put, or
        behavior stops being reproducible. An explicit value wins (``chunk_floor_pinned``)."""
        parts: list[str] = []
        if tts_ms_per_char is not None and tts_ms_per_char > 0 and self._synth_mpc is None:
            # Seed the JIT cost model: unseeded, the whole first reply synthesizes as
            # unscheduled whole-candidates (see _jit_pipeline) — the call-sizing race the
            # scheduler exists to prevent. Real observations then take over via the EMA.
            self._synth_mpc = max(tts_ms_per_char, _MPC_MIN)
            parts.append(f"jit_mpc={self._synth_mpc:.1f}ms/ch")
        if tts_rtf is not None and tts_rtf > 0:
            parts.append(f"tts_rtf={tts_rtf:.2f}")
            if not chunk_floor_pinned:
                cfg = self._cfg.chunker
                # Chunk 1's PLAYBACK must cover chunk 2's SYNTHESIS: chars1 >= safety * rtf *
                # minChars. Slow TTS grows the floor so the reply never gaps after chunk 1.
                floor = math.ceil(1.2 * tts_rtf * cfg.min_chars)
                eff = max(cfg.min_chars_first, min(cfg.min_chars, floor))
                if eff != cfg.min_chars_first:
                    self._chunker.set_first_floor(eff)
                    parts.append(f"minCharsFirst {cfg.min_chars_first}->{eff}")
        if stt_cost_ms is not None:
            parts.append(f"stt~{stt_cost_ms:.0f}ms")
            window = max(0, self._hangover_floor - self._eager_ms)
            if self._eager_ms and stt_cost_ms > window + 250:
                # The fix at this speed is a faster engine, not more overlap.
                parts.append(
                    f"note: eager overlap hides only ~{window}ms of that decode"
                )
        if parts:
            self._log.info("perf calibration: {}", "; ".join(parts))

    def is_dead_turn(self, token: str) -> bool:
        """Was this token's turn killed? Its late final must stay silent (see
        :data:`~nanobot_channel_voice.streamid.TURN_META`). Registered synchronously with
        ``abandon()``: no window lets a just-killed turn pass."""
        return token in self._dead_tokens

    def _is_rejected(self, base: str | None) -> bool:
        """Does this stream belong to a barged-out turn? Exact match first, else the watermark:
        the base embeds the turn's start ``time_ns`` (see :mod:`..streamid`), so a turn that
        STARTED before the last barge-in was already /stop-ped even without a live delta. A base
        without the timestamp skips the watermark."""
        if base is None:
            return False
        if base == self._rejected_base:
            return True
        ns = started_ns(base)
        return ns is not None and ns < self._reject_started_before_ns

    # ---- VoiceBackend contract ----------------------------------------------

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
        if self._wake_mode != "off" and self._wake_hit_at > self._wake_seen_at:
            # A hop-side wake hit (acoustic, or a strict-mode text match). The loop is the
            # single confirm authority: the hop only latches, so the echo/warmup suppressions
            # below veto before anything irreversible happens.
            self._wake_seen_at = self._wake_hit_at
            live = self._turn in (VoiceState.SPEAKING, VoiceState.THINKING)
            base = self._turn
            if base is VoiceState.SPEAKING and self._canned_base is not None:
                base = self._canned_base  # canned audio is not a reply
            if self._wake_hit_echoed():
                # Our own TTS said the phrase and weak/absent AEC heard it back: no window,
                # no kill, no claim blessing the trailing echo.
                self._wake_claimed = False
                self._metrics.count("wake_echo_suppressed")
                self._log.info("wake hit suppressed (own reply speaks the phrase)")
            else:
                self._touch_wake()
                score = getattr(self._wake_detector, "last_score", None)
                self._log.info(
                    "wake hit{}", f" (score={score:.2f})" if score is not None else ""
                )
                if (
                    self._stt_stream is not None
                    and self._endpointer.in_speech
                    and self._wake_hit_pos is not None
                    and self._wake_hit_pos > self._endpointer.open_pos
                ):
                    # The live stream ate the phrase: the next hop swaps in a fresh handle.
                    self._stt_restart = True
                if live and self._in_aec_warmup():
                    # The residual can false-trigger while AEC3 converges: hold the kill; the
                    # claim still rides to the endpoint verdict (delayed, not lost).
                    self._metrics.count("wake_warmup_hold")
                elif base is VoiceState.SPEAKING and self._endpointer.in_speech:
                    # Over a live reply mid-utterance: confirm without the min-words bar.
                    self._early_confirm = True
                    self._wake_confirm = True
                elif base is VoiceState.SPEAKING:
                    # No utterance to bind the confirm to (the bare phrase, or a residual the
                    # VAD missed): kill directly and settle, and the follow-up publishes cold
                    # inside the window the hit just opened.
                    await self._wake_kill()
                elif base is VoiceState.THINKING:
                    # Summoned while the agent works: the query must survive. An open utterance
                    # rides its claim to the verdict; the bare phrase answers now.
                    if not self._endpointer.in_speech:
                        self._reassure()
                else:
                    # Cold summon: probe for the fast ack (the verdict ack pays hangover + STT).
                    self._arm_fast_ack()
        if self._turn_analyzer is not None:
            snap = self._endpointer.take_consult()
            if snap is not None:
                if self._consult_task is None or self._consult_task.done():
                    self._consult_task = asyncio.create_task(self._run_consult(*snap))
                else:
                    # A previous pause's inference is still chewing: skip rather than stack.
                    self._metrics.count("eou_consult_skipped")
        if self._endpointer.in_speech and not prev_speech:
            if (
                self._wake_claimed
                and time.monotonic() - self._wake_hit_at > _WAKE_ATTACH_S
            ):
                # A hit nothing followed must not bless a later, unrelated utterance.
                self._wake_claimed = False
            # Stop targeting is decided by the state NOW, at onset (see _PendingUtterance).
            self._onset_at = time.monotonic()
            base_turn = self._turn
            if base_turn is VoiceState.SPEAKING and self._canned_base is not None:
                # Canned audio is not a reply: the onset joins the state beneath it.
                base_turn = self._canned_base
            self._onset_interrupting = base_turn in (
                VoiceState.THINKING, VoiceState.SPEAKING,
            )
            self._onset_speaking = base_turn is VoiceState.SPEAKING
            # Window openness is judged AT ONSET, never at the verdict: a later wake must not
            # retroactively bless bystander speech already in flight.
            self._onset_window_open = self._onset_at < self._wake_until
            if self._adaptive is not None:
                # CAPTURING here means a PREVIOUS utterance is still in its STT/queue window (a
                # fresh onset sees IDLE): exactly the fast-resume the learner exists to catch.
                self._adaptive.on_onset(
                    awaiting_reply=base_turn in (VoiceState.THINKING, VoiceState.CAPTURING),
                    speaking=base_turn is VoiceState.SPEAKING,
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
            # A confirm already claimed this utterance: a duck now only pollutes metrics.
            and not self._early_confirm
            and not self._preempted
            and self._duck_armed()
        ):
            # Stage 1 of the two-stage barge-in: yield NOW, reversibly; _on_utterance's verdict
            # confirms (kill + /stop) or releases. State-driven, not edge-driven: an onset whose
            # edge fell inside the post-acquittal holdoff still engages once it expires.
            self._engage_duck(suspect=False)
        if self._early_confirm:
            # The min-words gate hit mid-utterance: stop audio + /stop now; _on_utterance still
            # publishes the full transcript at close. The utterance must still be OPEN (or
            # closing on THIS frame), or the confirm rides the WRONG future utterance.
            self._early_confirm = False
            wake_confirm, self._wake_confirm = self._wake_confirm, False
            if (
                (self._endpointer.in_speech or utterance is not None)
                and self._turn in (VoiceState.SPEAKING, VoiceState.THINKING)
                # Canned audio is not the reply: an ack has no turn to /stop, and a
                # THINKING-base clip's steer must reach the inject rung, never a kill.
                and self._canned_base is None
            ):
                if self._cur_turn.continuation_pending and not wake_confirm:
                    # Mid-tool: cut the status line, leave the verdict to the inject rung.
                    # A wake hit still kills — being named IS a demand for the floor.
                    await self._hush_midturn()
                else:
                    self._preempted = True
                    self._metrics.count("barge_in_early_confirm")
                    self._early_heard = await self._do_interrupt()
                    if wake_confirm:
                        self._observe_wake_kill()
            # else: the reply finished (drain won the race); the utterance still publishes.
        release, self._early_release = self._early_release, None
        if release is not None and not self._duck_pause and self._candidate_contested():
            # Transcript acquittal before the endpoint verdict — duck mode only: a wrong one
            # costs level pumping. Pause mode gets none (decoder latency would drop real
            # speech); the probe below owns that mode.
            self._release_duck(release)
            self._acquitted_open = True  # don't re-engage over the acquitted utterance
        if self._duck_pause and self._candidate_contested():
            # Pause-probe: the pause silences leak but not a person, so a candidate whose LAST
            # speech flag fits the leak-death window is our own tail. Frame-domain on both sides
            # (a wall clock mis-attributes under capture lag); the skew covers engage landing
            # duckStartFrames-1 frames into the run.
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
                restarted, self._stt_restarted = self._stt_restarted, False
                if stream is not None:
                    task = asyncio.create_task(asyncio.to_thread(stream.finish))
                    task.add_done_callback(_swallow_result)
                else:  # defensive: no live handle (shouldn't happen) -> batch decode
                    task = None
                pending = self._make_pending(utterance, task, always_valid=task is not None)
                if pending.trim_bytes and not restarted:
                    # The hit raced the close before the restart hop, so the handle ate the
                    # phrase: demote to a fresh batch decode (-1 never matches).
                    pending.eager_always_valid = False
                    pending.eager_trim = -1
                self._queue_utterance(pending)
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
                # Pre-trim a wake claim's phrase audio: the speculation stays valid at close.
                trim = self._open_trim(eager_pcm)
                eager_pcm = eager_pcm[trim:]
                if eager_pcm:
                    task = asyncio.create_task(self._transcribe(eager_pcm))
                    task.add_done_callback(_swallow_result)
                    task.add_done_callback(self._eager_confirm_cb)
                    self._eager_task = task
                    self._eager_valid = True
                    self._eager_trim = trim
                    self._metrics.count("stt_eager_start")
                else:
                    self._eager_valid = False  # pure phrase: nothing to speculate on
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
        wake_hit = self._take_wake_claim()
        trim = 0
        if wake_hit and self._wake_hit_pos is not None:
            # A hit inside this span marks the phrase end; stale positions trim nothing (monotonic).
            start = ep.closed_open_pos
            if start < self._wake_hit_pos <= start + len(pcm):
                trim = self._snap_trim(pcm, self._wake_hit_pos - start)
                self._metrics.count("wake_trim")
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
            wake_hit=wake_hit,
            trim_bytes=trim,
            eager_trim=self._eager_trim,
            onset_speaking=self._onset_speaking,
            onset_window_open=self._onset_window_open,
            raw=self._raw_tail(len(pcm)),
            active_ms=ep.closed_active_ms,
            prob_peak=ep.closed_prob_peak,
            prob_mean=ep.closed_prob_mean,
            seg_id=self._next_seg(),
        )

    def _raw_tail(self, nbytes: int) -> bytes | None:
        """The last ``nbytes`` of pre-AEC capture: a segment's raw twin. Valid only because
        callers run between frame pushes (or under the hop lock), so the ring cannot advance
        mid-slice."""
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
        """Dump a min-filter reject; no-op unless the endpointer parked one this frame
        (probe/gap drops clear the slot first). A reject IS a close and no push has run since,
        so the ``closed_*`` snapshot is this blip's own."""
        if self._dumper is None:
            return
        ep = self._endpointer
        pcm, ep.last_rejected = ep.last_rejected, None
        if not pcm:
            return
        meta = {
            "wall": round(time.time(), 3),
            "active_ms": ep.closed_active_ms,
            "close": ep.closed_reason,
            "silence_ms": ep.closed_silence_ms,
        }
        if ep.closed_prob_mean is not None:
            meta["prob_mean"] = round(ep.closed_prob_mean, 3)
            meta["prob_peak"] = round(ep.closed_prob_peak, 3)
        self._dumper.submit(
            "blip", pcm, self._raw_tail(len(pcm)), seq=self._next_seg(), meta=meta,
        )

    def _candidate_contested(self) -> bool:
        """A duck/pause candidate is live and unclaimed. ``_early_confirm`` can re-arm during
        this frame's awaits (eager callback), so callers must re-evaluate after awaiting."""
        return (
            self._duck_onset is not None
            and not self._early_confirm
            and not self._preempted
            and self._endpointer.in_speech
        )

    def _take_wake_claim(self) -> bool:
        """Consume the wake claim for the utterance closing RIGHT NOW: a hit can only have
        fired during its own capture, so binding at close time is exact."""
        claimed, self._wake_claimed = self._wake_claimed, False
        return claimed

    def _open_trim(self, pcm: bytes) -> int:
        """Wake-phrase bytes at the head of the OPEN utterance's prefix ``pcm`` (0 = no live
        claim, or the hit lies outside it). Runs on the same bytes ``_make_pending`` will see,
        so eager and close compute the SAME trim (the _finish_stt validity invariant)."""
        if not self._wake_claimed or self._wake_hit_pos is None:
            return 0
        start = self._endpointer.open_pos
        if start < self._wake_hit_pos <= start + len(pcm):
            return self._snap_trim(pcm, self._wake_hit_pos - start)
        return 0

    def _snap_trim(self, pcm: bytes, hit: int) -> int:
        """Snap a wake trim back from the hit to the quietest recent dip: the hit trails the
        true phrase end by up to the decision cadence. Never later than the hit."""
        return quietest_split(pcm[:hit], self._cfg.audio.sample_rate)

    async def _wake_kill(self) -> None:
        """Kill the live reply on wake evidence and settle to IDLE; the follow-up
        (if any) publishes cold inside the window the hit just opened."""
        killed, heard = await self._kill_live_reply(
            interrupting=True, preempted=False, heard=None
        )
        if killed:
            self._last_kill = time.monotonic()
            self._pending_note = _wake_note(heard)
        self._clear_duck()
        self._metrics.count("wake_kill")
        self._observe_wake_kill()
        if self._turn is not VoiceState.IDLE:
            await self._set_turn(VoiceState.IDLE)
        self._arm_wake_ack()

    def _observe_wake_kill(self) -> None:
        """Hit-latch -> audio-stopped latency (``wake_kill_ms``). Recency-guarded:
        text-tier verdict matches carry no hit stamp."""
        dt = time.monotonic() - self._wake_hit_at
        if dt < _WAKE_ATTACH_S:
            self._metrics.observe("wake_kill_ms", dt * 1000.0)

    def _tts_speaks(self, lang: str) -> bool:
        """Whether the session's TTS can voice *lang* (None declaration =
        unrestricted, e.g. cloud adapters)."""
        langs = getattr(self._tts, "spoken_languages", None)
        if langs:
            return lang in langs
        single = getattr(self._tts, "spoken_language", None)
        return single is None or single == lang

    def _ack_pool(self, matched: str | None) -> list[str]:
        """Ack phrases for THIS summon: same-script entries win when the called name's script is
        known; built-ins cross to the matched script's row only if the TTS can voice it (a
        zh-only engine keeps the zh ack for an English summon)."""
        hint = _script_class(matched) if matched else self._wake_phrases_script
        return self._ack_pool_of(hint)

    def _ack_pool_of(self, hint: str | None) -> list[str]:
        pool = self._wake_ack_list
        if hint is None:
            return pool
        same = [p for p in pool if _script_class(p) == hint]
        if same:
            return same
        if self._cfg.wake.ack.phrases is None:
            row = _SCRIPT_ROWS.get(hint)
            if row is not None and self._tts_speaks(row):
                return _WAKE_ACK_BUILTINS[row]
            if hint == "latin" and self._tts_speaks("en"):
                return _WAKE_ACK_FALLBACK
        return pool

    def _ack_reachable_texts(self) -> list[str]:
        """Every ack a summon can pick: the resolved list plus each phrase script's crossover
        pool, so prewarm covers the first cross-script summon."""
        texts = list(self._wake_ack_list[:8])
        cfg = self._cfg.wake
        hints = {_script_class(p) for p in cfg.phrases + cfg.aliases}
        hints.discard(None)
        for hint in sorted(hints):
            for text in self._ack_pool_of(hint)[:2]:
                if text not in texts:
                    texts.append(text)
        return texts

    def _arm_fast_ack(self) -> None:
        """Probe for the fast-path ack after a cold acoustic hit. Open-mic only: the SPEAKING
        flip would gate a half-duplex mic mid-capture (that mode acks at the verdict)."""
        if (
            self._closing
            or self._tts is None
            or not self._cfg.wake.ack.enabled
            or not self._open_mic
            or not self._wake_ack_list
        ):
            return
        cancel_task(self._fast_ack_task)
        self._fast_ack_task = asyncio.create_task(
            self._fast_ack_probe(self._sink.epoch, self._wake_hit_at)
        )

    def _arm_close_ack(self, pending: _PendingUtterance) -> None:
        """Bare-summon-shaped close carrying a wake claim: speak the ack DURING the STT wait.
        Nothing is captured by then, so both duplex modes take it. The stamp precedes the task,
        so a verdict racing the playback reads continue-not-restart."""
        if (
            self._closing
            or self._tts is None
            or not self._cfg.wake.ack.enabled
            or not self._wake_ack_list
            or pending.onset_interrupting  # a summon over a reply acks at its kill
            or pending.active_ms > _CLOSE_ACK_MAX_ACTIVE_MS
            or time.monotonic() - self._fast_acked_at < _WAKE_ATTACH_S  # probe spoke
            # Open mic: a close whose trailing silence covered the probe's grace belongs to
            # the probe — quiet there means same-breath command.
            or (
                self._open_mic
                and pending.silence_ms >= _FAST_ACK_QUIET_S * 1000
            )
        ):
            return
        self._fast_acked_at = time.monotonic()
        cancel_task(self._fast_ack_task)
        self._fast_ack_task = asyncio.create_task(
            self._wake_ack(self._sink.epoch, None, fast=True)
        )

    async def _fast_ack_probe(self, epoch: int, hit_at: float) -> None:
        """Ack a bare summon before the endpoint verdict: a cold hit whose claim survives the
        quiet window is very likely the bare phrase. Polls, because a single instant coin-flips
        on where the hit landed. ``_fast_acked_at`` tells the verdict not to ack twice."""
        try:
            await asyncio.sleep(_FAST_ACK_QUIET_S)
            while True:
                if (
                    self._closing
                    or epoch != self._sink.epoch
                    or self._turn not in (VoiceState.IDLE, VoiceState.CAPTURING)
                    or not self._wake_claimed
                    or self._wake_hit_at != hit_at
                    or time.monotonic() - hit_at >= _FAST_ACK_WINDOW_S
                ):
                    return
                if self._fast_ack_quiet():
                    self._fast_acked_at = time.monotonic()
                    await self._wake_ack(epoch, None, fast=True)
                    return
                await asyncio.sleep(_FAST_ACK_POLL_S)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a probe must never wedge the loop
            self._log.warning("fast wake ack failed ({})", exc)

    def _take_fast_ack(self) -> bool:
        """One-shot, recency-bounded: did the fast path already answer the
        summon whose verdict is running now?"""
        recent = time.monotonic() - self._fast_acked_at < _WAKE_ATTACH_S
        self._fast_acked_at = float("-inf")
        return recent

    def _fast_ack_quiet(self) -> bool:
        """The user stopped after the phrase: ``speech_run`` freezes at onset, so the
        in-utterance signal is ``silence_run_ms``."""
        ep = self._endpointer
        if ep.in_speech:
            return ep.silence_run_ms >= _FAST_ACK_MIN_QUIET_MS
        return ep.speech_run == 0

    def _arm_wake_ack(self, matched: str | None = None) -> None:
        """Speak the ack for the bare-wake settle that just ran; task-shaped, as the settle
        path must not wait out synthesis or playback. ``matched`` = the phrase the transcript
        tier saw (language routing), None = acoustic."""
        if (
            self._closing
            or self._tts is None
            or not self._cfg.wake.ack.enabled
            or not self._wake_ack_list
        ):
            return
        cancel_task(self._ack_task)
        self._ack_task = asyncio.create_task(self._wake_ack(self._sink.epoch, matched))

    def _reassure(self, matched: str | None = None, *, metric: str = "wake_reassure") -> bool:
        """Answer a re-summon or a steer during THINKING without touching the query. Prologue
        script first (an IDLE-style ack would invite a fresh command), else the wake ack; rides
        the prologue task slot. False = neither is configured, so the caller owes a receipt."""
        self._metrics.count(metric)
        if self._cfg.prologue.enabled and self._prologue_phrases:
            self._arm_prologue(initial_ms=0)
            return True
        if (
            self._closing
            or self._tts is None
            or not self._cfg.wake.ack.enabled
            or not self._wake_ack_list
        ):
            return False
        self._cancel_prologue()
        self._cur_turn.prologue_task = asyncio.create_task(
            self._wake_ack(self._sink.epoch, matched, base=VoiceState.THINKING)
        )
        return True

    async def _wake_ack(
        self,
        epoch: int,
        matched: str | None = None,
        *,
        fast: bool = False,
        base: VoiceState = VoiceState.IDLE,
    ) -> None:
        """One ack playback: base -> SPEAKING -> settle -> base. Canned audio: any turn outcome
        flushes it via _kill_live_reply's canned branch, never a /stop or a heard-up-to note.
        ``fast`` tolerates the still-open summon; ``base=THINKING`` is the reassure clip."""
        nonce = object()
        try:
            phrases = self._ack_pool(matched)
            text = phrases[self._ack_step % len(phrases)]
            audio = await self._synth_filler(text)
            state_ok = (
                self._turn in (VoiceState.IDLE, VoiceState.CAPTURING)
                if fast else self._turn is base
            )
            quiet = (
                self._fast_ack_quiet() if fast else not self._endpointer.in_speech
            )
            if (
                not audio
                or self._closing
                or epoch != self._sink.epoch
                or not state_ok
                or not quiet  # never talk over the follow-up command
            ):
                if fast:
                    # Nothing was spoken: the verdict rung must ack after all.
                    self._fast_acked_at = float("-inf")
                return
            self._ack_step += 1
            if base is VoiceState.IDLE:  # reassures counted at their arm site
                self._metrics.count("wake_ack")
                if fast:
                    self._metrics.count("wake_ack_fast")  # committed, not merely probed
                dt = time.monotonic() - self._wake_hit_at
                if dt < _WAKE_ATTACH_S:
                    # Recency-guarded like wake_kill_ms: text-tier matches carry no stamp.
                    self._metrics.observe("wake_ack_ms", dt * 1000.0)
            self._log.info(
                "wake {} ('{}')",
                "ack" if base is VoiceState.IDLE else "reassure", text,
            )
            self._note_spoken(text, self._sink.backlog_ms() + self._audio_ms(audio))
            self._canned_base, self._canned_nonce = base, nonce
            await self._canned_playback(epoch, audio, base)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - an ack must never wedge the session
            self._log.warning("wake ack failed ({})", exc)
            if self._turn is VoiceState.SPEAKING and epoch == self._sink.epoch:
                with suppress(Exception):
                    await self._set_turn(base)
        finally:
            if self._canned_nonce is nonce:
                self._canned_base = None

    async def push_gated_audio(self, pcm: bytes) -> None:
        """Half-duplex wake tap: the shell routes the frames it drops while the bot speaks here,
        keeping the acoustic tier hot and making the wake word the ONLY barge-in channel there
        (no duck/confirm machinery runs). The gate-reopen flush keeps the phrase out of STT."""
        if (
            self._closing
            or not pcm
            or self._wake_detector is None
            or self._wake_mode == "off"
        ):
            return

        def _push() -> bool:
            with self._hop_lock:
                return self._wake_detector.push(pcm)

        if not await asyncio.to_thread(_push):
            return
        # Consumed here, not by push_audio's block (no frame will carry it there).
        self._wake_hit_at = self._wake_seen_at = time.monotonic()
        self._wake_claimed = True
        # The phrase never entered the endpointer stream: nothing to trim.
        self._wake_hit_pos = self._endpointer.pos
        self._metrics.count("wake_hit")
        if self._wake_hit_echoed():
            self._wake_claimed = False
            self._metrics.count("wake_echo_suppressed")
            self._log.info("wake hit suppressed (own reply speaks the phrase)")
            return
        self._touch_wake()
        score = getattr(self._wake_detector, "last_score", None)
        self._log.info(
            "wake hit over gated mic{}",
            f" (score={score:.2f})" if score is not None else "",
        )
        if self._canned_base is VoiceState.THINKING:
            # Summon over the FILLER: the query must survive. Flush the clip and stay THINKING
            # — the cut is the earcon, and another clip would re-gate the reopened mic.
            self._cancel_prologue()
            await self._sink.flush()
            if self._turn is VoiceState.SPEAKING:
                await self._set_turn(VoiceState.THINKING)
            return
        await self._wake_kill()

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
        """One frame through the endpointer, honoring a pending COMPLETE verdict: a verdict
        raised since the last frame force-closes the utterance now (validated against the pause
        the model scored). The verdict slot is a GIL-atomic write, consumed exactly once here."""
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
        off-loop hop. At onset a FRESH handle starts and the pre-roll ring is replayed into it;
        a rejected blip's handle is dropped. Returns ``(utterance, compute_ms)`` — the clock
        starts after the hop lock, so the caller can split engine cost from overhead."""
        with self._hop_lock:
            t0 = time.monotonic()
            if self._aec is not None:
                raw = pcm
                # Subtract our own playback BEFORE the endpointer/STT hear the frame, so echo
                # never becomes "speech".
                pcm = self._aec.process(pcm)
                if self._dump_raw is not None:
                    # process() floors to whole 10 ms blocks: mirror only what the pipeline heard.
                    self._dump_raw.append(raw if len(raw) == len(pcm) else raw[: len(pcm)])
            if self._wake_detector is not None and self._wake_detector.push(pcm):
                # Post-AEC: a hit while the bot speaks is judged on the residual.
                self._wake_hit_at = time.monotonic()
                self._wake_claimed = True
                # Phrase end in stream coordinates (this frame not eaten yet), for the close trim.
                self._wake_hit_pos = (
                    self._endpointer.pos + len(pcm)
                    - getattr(self._wake_detector, "last_hit_back_bytes", 0)
                )
                self._metrics.count("wake_hit")
            utterance = self._push_with_model_close(pcm)
            if utterance is not None:
                # Closed: the ring's pre-onset context belongs to THIS utterance, not to a fast
                # re-onset that follows.
                self._recent.clear()
            stt = self._stt_stream
            restart, self._stt_restart = self._stt_restart, False  # one-shot
            if stt is not None:
                if self._endpointer.in_speech:
                    if not prev_speech:
                        self._stt_live = stt.stream_start()
                        self._stt_restarted = False
                        for frame in self._recent:
                            self._stt_live.accept(frame)
                    elif restart and self._stt_live is not None:
                        # Confirmed wake claim: everything the handle ate so far is phrase
                        # audio. A fresh handle (no pre-roll replay) keeps it out.
                        self._stt_live = stt.stream_start()
                        self._stt_restarted = True
                    if self._stt_live is not None:
                        self._stt_live.accept(pcm)
                        poll_confirm = (
                            self._duck_onset is not None
                            and not self._preempted
                            and not self._early_confirm
                            and utterance is None
                        )
                        # Strict mode's text-tier unlock: with no claim nothing can engage,
                        # so partials are scanned for the wake prefix, not fresh words.
                        poll_wake = (
                            self._wake_mode == "strict"
                            and not self._wake_claimed
                            and self._wake_phrase is not None
                            and not self._preempted
                            and utterance is None
                            and self._turn is VoiceState.SPEAKING
                        )
                        # Cold text-tier fast path (ack on): a phrase-leading partial latches
                        # the claim — the only pre-verdict evidence a no-acoustic config has.
                        poll_wake_cold = (
                            self._wake_mode != "off"
                            and self._cfg.wake.ack.enabled
                            and not self._wake_claimed
                            and self._wake_phrase is not None
                            and utterance is None
                            and self._turn in (VoiceState.IDLE, VoiceState.CAPTURING)
                        )
                        if poll_confirm or poll_wake or poll_wake_cold:
                            # Early confirm from partials (_judge_fresh); consecutive
                            # zero-fresh polls early-release instead (duck mode only).
                            self._partial_countdown -= 1
                            if self._partial_countdown <= 0:
                                self._partial_countdown = self._partial_every
                                ptext = self._stt_live.partial()
                                if (
                                    (poll_wake or poll_wake_cold)
                                    and self._wake_strip_leaky(ptext)[0]
                                ):
                                    # Latch only; the loop-side consumption vets.
                                    self._wake_hit_at = time.monotonic()
                                    self._wake_claimed = True
                                elif poll_confirm:
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
                    self._stt_restarted = False
                elif not prev_speech:
                    self._recent.append(pcm)  # idle: keep pre-onset context warm
            return utterance, (time.monotonic() - t0) * 1000.0

    def _queue_utterance(self, pending: _PendingUtterance) -> None:
        if self._adaptive is not None:
            # Anchor the resume-gap clock at CLOSE: a fast resume beats this utterance's STT.
            self._adaptive.note_close(pending.closed_at, float(pending.silence_ms))
        if pending.wake_hit:
            self._arm_close_ack(pending)
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
                # Nothing else settles this utterance: CAPTURING would stand until the user speaks.
                if self._adaptive is not None:
                    self._adaptive.drop_anchor()
                with suppress(Exception):
                    if pending.preempted:
                        await self._orphaned_confirm("error")
                    elif self._turn is VoiceState.CAPTURING:
                        await self._set_turn(VoiceState.IDLE)
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
        """Reset VAD/endpointer streaming state without racing the frame hop: a bare loop-side
        ``reset()`` could interleave mid-``push`` (torn counters, undefined neural-VAD cache).
        The hop lock is taken off-loop. Skipped mid-utterance: it would lose a barge-in's words."""
        def _reset() -> None:
            with self._hop_lock:
                if not self._endpointer.in_speech:
                    self._endpointer.reset()
                    self._recent.clear()
                    # The wake claim deliberately survives: a hit over the reply's tail must
                    # still bless the follow-up (onset staleness bounds it to _WAKE_ATTACH_S).

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
                self._wake_claimed = False
                if self._wake_detector is not None:
                    # Discontinuous stream: context spliced across the gap scores garbage.
                    self._wake_detector.reset()
                if self._dump_raw is not None:
                    # Pre-gap audio no longer abuts the stream; never splice into a later twin.
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
        # The dropped utterance's speculation dies with it, or a still-valid eager task hands
        # the PRE-GAP transcript to the next utterance's close.
        self._eager_valid = False
        # The restart re-opens the device: whatever backlog the debt described is gone.
        self._capture_debt_ms = 0.0
        if self._duck_onset is not None:
            self._release_duck("gap")
        await self._orphan_if_confirmed("gap")
        if self._turn is VoiceState.CAPTURING:
            # Nothing will publish the dropped utterance: without this, CAPTURING stands forever.
            await self._set_turn(VoiceState.IDLE)

    async def close(self) -> None:
        self._closing = True
        self._eager_valid = False
        for task in (self._eager_task, self._utt_task, self._cur_turn.prologue_task,
                     self._cur_turn.midturn_task, self._cur_turn.timeout_task,
                     self._tts_task, self._drain_task, self._consult_task,
                     self._ack_task, self._fast_ack_task, self._earcon_task,
                     self._attention_task):
            await cancel_and_wait(task)
        self._consult_task = None
        self._eager_task = self._utt_task = self._cur_turn.prologue_task = None
        self._cur_turn.midturn_task = self._cur_turn.timeout_task = None
        self._tts_task = self._drain_task = self._ack_task = None
        self._fast_ack_task = self._earcon_task = self._attention_task = None
        # Pooled adapter resources (e.g. an httpx client); optional per adapter.
        aclose = getattr(self._tts, "aclose", None)
        if aclose is not None:
            with suppress(Exception):
                await aclose()
        # An RKNN context is NOT freed by refcount-GC, so an in-process channel restart would
        # load a second copy and exhaust the NPU cores.
        for engine in (self._tts, self._vad, self._turn_analyzer, self._wake_detector):
            if engine is not None:
                with suppress(Exception):
                    engine.release()
        if self._dumper is not None:
            await asyncio.to_thread(self._dumper.close)

    # ---- input: capture -> STT -> publish (bus) -----------------------------

    async def _on_utterance(self, pending: _PendingUtterance) -> str:
        """Run the verdict ladder over one closed utterance. Returns the verdict token
        (``empty``/``echo``/``wake``/``gated``/``stop``/``ack``/``inject``/``goal``/
        ``interrupt``/``publish``) - the audio dump names the segment's file with it."""
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
        # Recomputed, not merely downgraded: the pre-STT snapshot exists so a turn that
        # FINISHED during the window is not /stop-ped, but one that STARTED in it (a cron
        # delivery) is just as live, and its drain watcher would settle whatever we publish.
        interrupting = self._turn in (VoiceState.THINKING, VoiceState.SPEAKING)

        def _summary(verdict: str) -> str:
            """One line per judged utterance (the id matches the dump filename), plus the
            ``pending.meta`` stamp for the dump manifest. Head loudness is 1 s-capped: the
            numpy-less fallback loops on the event loop."""
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
            # Verdict/duration/rms stay out: the dump writer stamps those on every record.
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
            if pending.trim_bytes:
                pending.meta["wake_trim_ms"] = int(
                    pcm_ms(pending.trim_bytes, self._cfg.audio.sample_rate)
                )
            if self._cfg.log_transcripts:  # same privacy gate as the log line
                pending.meta["text"] = text
            # One-shot per utterance, not per rung: a PUBLISHING claimed utterance never
            # reaches _take_fast_ack, and the stale latch would mute the next summon's ack.
            self._fast_acked_at = float("-inf")
            return verdict

        if not text:
            if pending.wake_hit:
                # Acoustic wake with nothing left after the trim: attention, not content.
                self._metrics.count("wake_only")
                if self._adaptive is not None:
                    self._adaptive.drop_anchor()
                self._touch_wake()
                acked_fast = self._take_fast_ack()
                if not preempted and (
                    self._canned_base is VoiceState.IDLE
                    or (
                        acked_fast
                        and self._turn in (VoiceState.IDLE, VoiceState.CAPTURING)
                    )
                ):
                    # An ack already speaks for this summon: let it finish — it owns the
                    # state, and flush-and-restart is an audible stutter.
                    self._clear_duck()
                    if (
                        self._canned_base is not VoiceState.IDLE
                        and self._turn is VoiceState.CAPTURING
                    ):
                        await self._set_turn(VoiceState.IDLE)
                    return _summary("wake")
                if (
                    not preempted
                    and pending.onset_interrupting
                    and not pending.onset_speaking
                ):
                    # Summoned during THINKING ("are you there?"): never kill the query. A
                    # reply that arrived meanwhile IS the answer, a playing filler already
                    # speaks, otherwise reassure. No kill -> no note, no settle.
                    self._clear_duck()
                    if self._turn is VoiceState.THINKING:
                        self._reassure()
                    elif self._turn in (VoiceState.IDLE, VoiceState.CAPTURING):
                        self._arm_wake_ack()
                    return _summary("wake")
                killed, k_heard = await self._kill_live_reply(
                    interrupting=interrupting, preempted=preempted, heard=heard
                )
                if killed:
                    self._last_kill = time.monotonic()
                    self._pending_note = _wake_note(k_heard)
                    if not preempted:  # a preempted kill was observed at confirm
                        self._observe_wake_kill()
                self._clear_duck()
                if self._turn is not VoiceState.IDLE:
                    await self._set_turn(VoiceState.IDLE)
                self._arm_wake_ack()
                return _summary("wake")
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
        # own last sentence as a user turn. Half-duplex is NOT exempt: its mic gate is a
        # read-time approximation that capture lag leaks the reply's transcribable tail past.
        if self._echo.is_self_echo(text):
            # A genuine soft-duplex interruption is a MIXTURE (user + leak) that containment
            # classifies as echo. Only an interrupt-shaped turn overrides, on words that are
            # neither our TTS nor backchannels — one fresh STOP word suffices (the kill switch
            # must survive the leak).
            fresh = self._echo.fresh_words(text) - self._ack_words
            fresh_seq = self._fresh_seq(text, fresh)
            # Strict wake gating applies to the echo-rung overrides too (same predicate as
            # _wake_verdict): otherwise a bystander's "stop" through the leak kills the reply,
            # or a drain-tail mixture steers a shut-window THINKING turn. Wake evidence — a
            # claim, or a leading phrase in the transcript — unblocks.
            wake_blocked = (
                self._wake_mode == "strict"
                and (pending.onset_speaking or not pending.onset_window_open)
                and not pending.wake_hit
                and not (
                    self._wake_phrase is not None
                    and self._wake_strip_leaky(text)[0]
                )
            )
            if (
                (interrupting or preempted)
                and not wake_blocked
                and self._stop_match.pure(fresh_seq)
            ):
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
            if (interrupting or preempted) and not wake_blocked and (
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

        # Wake gate: a cold utterance without the phrase — or, in strict mode, an unwoken
        # attempt over a live reply — is dropped WHOLE (content spoken at a gated device must
        # not reach the agent). Above the stop rung on purpose: a bystander's "stop" is
        # bystander speech, while the engaged user's arrives in-window or wake-prefixed.
        if self._wake_mode != "off":
            gate, text, wake_name = self._wake_verdict(pending, text)
            if gate == "gated":
                self._metrics.count("wake_gated")
                if self._adaptive is not None:
                    self._adaptive.drop_anchor()  # not a turn: nothing to resume from
                self._release_duck("gated")
                if preempted:
                    await self._orphaned_confirm("gated")
                elif self._turn is VoiceState.CAPTURING:
                    await self._set_turn(VoiceState.IDLE)
                return _summary("gated")
            if gate == "wake":
                # The bare wake phrase: attention, not content. Kill a live reply, open the
                # window, publish nothing. The kill arms the same grace and pending note a
                # consumed stop leaves: a follow-up bare "stop" must stay consumable, and the
                # next turn's agent must know the reply was cut.
                self._metrics.count("wake_only")
                if self._adaptive is not None:
                    self._adaptive.drop_anchor()
                acked_fast = self._take_fast_ack()
                if not preempted and (
                    self._canned_base is VoiceState.IDLE
                    or (
                        acked_fast
                        and self._turn in (VoiceState.IDLE, VoiceState.CAPTURING)
                    )
                ):
                    # An ack already speaks for this summon: let it finish (see above).
                    self._clear_duck()
                    if (
                        self._canned_base is not VoiceState.IDLE
                        and self._turn is VoiceState.CAPTURING
                    ):
                        await self._set_turn(VoiceState.IDLE)
                    return _summary("wake")
                if (
                    not preempted
                    and pending.onset_interrupting
                    and not pending.onset_speaking
                ):
                    # Summoned during THINKING: reassure, never kill (see above).
                    self._clear_duck()
                    if self._turn is VoiceState.THINKING:
                        self._reassure(wake_name)
                    elif self._turn in (VoiceState.IDLE, VoiceState.CAPTURING):
                        self._arm_wake_ack(wake_name)
                    return _summary("wake")
                killed, heard = await self._kill_live_reply(
                    interrupting=interrupting, preempted=preempted, heard=heard
                )
                if killed:
                    self._last_kill = time.monotonic()
                    self._pending_note = _wake_note(heard)
                    if not preempted:  # a preempted kill was observed at confirm
                        self._observe_wake_kill()
                self._clear_duck()
                if self._turn is not VoiceState.IDLE:
                    await self._set_turn(VoiceState.IDLE)
                self._arm_wake_ack(wake_name)
                return _summary("wake")

        # Stop command aimed at a live reply: kill it and CONSUME the utterance — publishing
        # "stop" would make the agent answer it. Targeting is decided at ONSET (a reply draining
        # during this utterance's STT window still counts), plus a short kill-anchored grace for
        # double-taps. A cold stop falls through and publishes: it may be the answer to a
        # question the agent just asked ("say cancel to abort").
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

        # Steering, not barge-in: a run still WORKING takes the utterance as a mid-turn
        # injection instead of a kill. Decided before the metrics anchor (an injection extends
        # the RUNNING turn), and the verdict joins the state beneath a canned clip.
        verdict_state = self._turn
        if verdict_state is VoiceState.SPEAKING and self._canned_base is not None:
            verdict_state = self._canned_base
        # A goal command can never be injected: core dispatches commands inline, so an injected
        # "/goal ..." would answer out of band while the old run kept going. Kill first.
        goal = self._is_goal(text)
        inject = (
            interrupting
            and not preempted
            and not goal
            and not self._cur_turn.dead
            and verdict_state in (VoiceState.THINKING, VoiceState.SPEAKING)
            and (
                # Inside a tool, so what plays is a status line: cut the audio, never the run.
                # Cleared by the first post-tool delta (talking over the ANSWER is a barge-in).
                self._cur_turn.continuation_pending
                # Or nothing audible to talk over in the first place.
                or (verdict_state is VoiceState.THINKING and not pending.onset_speaking)
            )
        )

        # Anchor at the TRUE end of speech, only for ACCEPTED utterances (a rejected echo/empty
        # must not corrupt a live turn's clock): back-date past STT time and queue wait to the
        # close, plus silence_ms — the trailing silence the close consumed.
        offset = (time.monotonic() - pending.closed_at) * 1000.0 + float(pending.silence_ms)
        if not inject:
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

        if inject:
            # No /stop, no fresh _Turn: core injects into the live run's pending queue. The
            # fresh token matters only if the run ends first — the message then opens a normal
            # turn. Attention/wake bookkeeping is untouched (same turn), and last_activity is
            # NOT pushed: it measures the CORE, and a user who keeps asking would hold the
            # deadman off a wedged turn for as long as they ask.
            canned = self._canned_base is not None
            if self._turn is VoiceState.SPEAKING:
                await self._hush_midturn()
            self._clear_duck()
            self._metrics.count("midturn_injection")
            notes: list[str] = []
            if self._pending_note is not None:
                notes.append(self._pending_note)
                self._pending_note = None
            inject_token = unique_token()
            # Fresh token (the context bridge keys notes by it), but owned by the RUNNING
            # turn: a later barge-in must kill this publish too.
            self._cur_turn.tokens.append(inject_token)
            await self._publish_text(text, inject_token, tuple(notes))
            # Core drains injections only at iteration boundaries, so the answer can be a
            # whole tool call away: the steer needs a receipt now.
            if canned:
                # The cut clip is the receipt; its script resumes mid-way, not on the same phrase.
                self._arm_prologue(initial_ms=self._cfg.prologue.interval_ms, start_step=1)
            elif not self._reassure(metric="midturn_reassure"):
                # Nothing to say: the ding is the receipt. Last, as a prologue arm sweeps earcons.
                self._arm_earcon()
            return _summary("inject")

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
        self._reply_tail = ""          # the old reply's question must not re-open attention
        notes: list[str] = []
        marker = _interrupt_marker(heard)
        if marker:
            notes.append(marker)
        if self._pending_note is not None:
            # A consumed stop's note describes the PREVIOUS (killed) reply; the marker above
            # describes the one killed by THIS utterance. Both may ride one publish.
            notes.append(self._pending_note)
            self._pending_note = None
        # An accepted turn is a fresh attention episode; a shut window cues at the settle.
        self._attention_cued = False
        if self._wake_attention == "sentence":
            # One wake, one sentence: the publish SPENDS the attention (the settle may re-open
            # it — see _set_turn). The cue watcher sleeps on the old deadline, so this one
            # backward move must kick it; the settle re-arms.
            self._wake_until = 0.0
            cancel_task(self._attention_task)
        else:
            self._touch_wake()  # an accepted turn keeps the conversation's attention
        if goal:
            self._metrics.count("goal_command")
        await self._publish_text(
            f"/goal {text}" if goal else text, self._cur_turn.tokens[0], tuple(notes)
        )
        self._arm_prologue()
        self._arm_earcon()  # after _arm_prologue: its cancel sweep covers earcons
        self._arm_timeout()
        # "goal" over "interrupt"/"publish": interrupt= already flags a live turn, and a
        # mis-fired trigger is what needs finding in the dump (named by this verdict).
        return _summary("goal" if goal else "interrupt" if killed else "publish")

    async def _do_interrupt(self) -> str | None:
        """Cancel-then-send barge-in: invalidate the dead turn, stop audio, /stop. Invalidating
        FIRST drops deltas the cancelled turn emits before /stop lands. Returns the heard-up-to
        TEXT ("" = cut before anything sounded, None = accounting unavailable/disabled)."""
        self._rejected_base = self._cur_turn.base
        self._reject_started_before_ns = time.time_ns()
        self._dead_tokens.extend(self._cur_turn.tokens)
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
        heard = self._take_heard(played)
        await self._interrupt()
        return heard

    def _take_heard(self, played: float) -> str | None:
        """Fold the sink's played-ms into heard TEXT and close the span ledger: every path
        that cuts audio mid-sentence owes the agent the same account."""
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
        return heard

    async def _hush_midturn(self) -> None:
        """``_do_interrupt``'s audio half without its turn-ending half: the user takes the floor,
        the run keeps working. What plays between a ``resuming`` end and the next delta is a
        status line, not the answer — /stop-ping there kills a "let me try another way" chain.
        The boundary watch dies with the epoch, so the state flip happens here."""
        self._cancel_prologue()
        self._cancel_midturn()
        played = await self._sink.flush()  # epoch++; restores level and pause gate
        self._duck_onset = None  # VAD floor stays scaled, as in _do_interrupt
        self._duck_suspect = False
        heard = self._take_heard(played)
        marker = _steer_marker(heard)
        if marker is not None and self._pending_note is None:
            self._pending_note = marker  # rides the steer's own publish
        if heard:
            # The turn lives on, so its ledger does: a later kill must account for these words.
            self._heard_prefix = heard
        if self._turn is VoiceState.SPEAKING:
            await self._set_turn(VoiceState.THINKING)  # tools still running; mic reopens
        self._metrics.count("midturn_hush")
        self._log.info("midturn steer: audio cut, run kept")

    def _heard_text(self, played_ms: float) -> str:
        """Map the sink's played-ms into the chunk texts the user actually heard: chunk-granular,
        with a word-proportional cut inside the chunk playback stopped in."""
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
                if len(cut_words) > 1:
                    keep = max(1, int(len(cut_words) * frac))
                    out.append(" ".join(cut_words[:keep]) + "...")
                else:
                    # An unspaced CJK chunk is ONE "word": proportion it by character,
                    # or every partial hearing reports the whole sentence as heard.
                    out.append(chunk_text[: max(1, int(len(chunk_text) * frac))] + "...")
            break
        return " ".join(out).strip()

    async def _set_turn(self, state: VoiceState) -> None:
        # A settle re-opens the attention window; a rejected-utterance settle
        # (CAPTURING -> IDLE) and canned tails deliberately do not, or a bystander
        # (or the close cue itself) would hold the gate open. "sentence" mode
        # re-opens only for a reply ending in a question.
        settled = state is VoiceState.IDLE and self._turn in (
            VoiceState.SPEAKING, VoiceState.THINKING,
        )
        if settled and self._canned_base is None:
            if (
                self._wake_attention != "sentence"
                or self._reply_asked_question()
                or self._cur_turn.proactive  # the machine spoke first: invite a reply
            ):
                self._touch_wake()
            self._cur_turn.proactive = False  # consumed; the placeholder is shared
        await super()._set_turn(state)
        if settled:
            # AFTER the flip: a watcher spawned here must see IDLE, or it exits as if live.
            self._arm_attention_cue()

    def _reply_asked_question(self) -> bool:
        tail = self._reply_tail.rstrip().rstrip("\"'”’」』)]）】…")
        return tail.endswith(("?", "？"))

    def _touch_wake(self) -> None:
        """(Re)open the attention window: cold starts inside it need no wake phrase. No-op when
        gating is off or windowS is 0 (every cold start gated)."""
        if self._wake_mode != "off" and self._wake_window_s > 0:
            self._wake_until = time.monotonic() + self._wake_window_s
            self._attention_cued = False  # a fresh episode re-arms the close cue
            self._arm_attention_cue()

    def _wake_hit_echoed(self) -> bool:
        """Recently-spoken TTS literally SAYS a wake phrase: a hit now is very likely the bot
        hearing itself. Ordered-mention test, so a reply using the phrase's words APART does not
        lock the wake word out; unmatchable phrases fall back to unit containment."""
        if self._wake_phrase is not None:
            return self._wake_phrase.present(self._echo.recent_text())
        return any(
            p and not self._echo.fresh_words(p) for p in self._wake_phrases_text
        )

    def _wake_strip_leaky(self, text: str) -> tuple[str | None, str]:
        """``WakePhrase.strip`` that also tolerates OUR OWN leaked words before the phrase. Only
        tokens with zero fresh units are acceptable lead, so a bystander's real words still
        demote. Safe on the hop: fresh_words snapshots, strip is pure."""
        fresh = self._echo.fresh_words(text)
        return self._wake_phrase.strip(
            text, extra_lead=lambda tok: not (units_of(tok) & fresh)
        )

    def _wake_verdict(
        self, pending: _PendingUtterance, text: str
    ) -> tuple[str, str, str | None]:
        """Judge one utterance against the wake gate. Returns ``(decision, text, matched)``:
        ``"pass"`` (leading phrase stripped), ``"wake"`` (bare phrase, attention only) or
        ``"gated"``; ``matched`` = the phrase that fired (None = acoustic), routing the ack."""
        matched, stripped = (
            self._wake_strip_leaky(text)
            if self._wake_phrase is not None else (None, text)
        )
        if matched is not None:
            self._metrics.count("wake_text")
            # An unsure user repeats the name ("小娜小娜"): iterate the single-pass strip here,
            # the one consumer that publishes the remainder. A clitic-bound remainder is content.
            while stripped:
                again, rest = self._wake_strip_leaky(stripped)
                if again is None or (rest and rest[0] in "'’-"):
                    break
                stripped = rest
        woke = pending.wake_hit or matched is not None
        window_open = pending.onset_window_open
        required = (
            # Strict gates audible barge-in always, and THINKING continuations once the window
            # is shut (no bystander steering a long tool run). In-window THINKING follow-ups,
            # and all of gate mode's, stay free: the user's own continuation.
            (self._wake_mode == "strict" and (pending.onset_speaking or not window_open))
            or (not pending.onset_interrupting and not window_open)
        )
        if matched is None and self._wake_fuzzy is not None and (
            pending.wake_hit or not required
        ):
            # The STT mangled the name: the fuzzy tier is STRIP-ONLY (see FuzzyWake) — consulted
            # only when the utterance passes ANYWAY, so it scrubs the mangle and makes bare
            # mangles summons, but never opens the gate or unlocks strict.
            matched, stripped = self._wake_fuzzy.strip_head(text)
            if matched is not None:
                self._metrics.count("wake_fuzzy_strip")
                woke = True
        if required and not woke:
            return "gated", text, matched
        if woke:
            self._touch_wake()
        if matched is not None and not tokens_of(stripped):
            return "wake", "", matched
        return "pass", stripped if matched is not None else text, matched

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
            # Strict: unclaimed speech over a live reply never ducks, so the reply plays
            # through crowds and the duck/acquit machinery stays cold until a hit claims the
            # utterance. Canned audio (filler/ack) is not a reply and yields to anyone.
            and (
                self._wake_mode != "strict"
                or self._wake_claimed
                or self._canned_base is not None
            )
            # The fast ack plays inside the summon's trailing hangover: stale in-speech is not
            # the user talking; the first fresh speech frame zeroes silence_run and re-arms.
            and not (
                self._canned_base is VoiceState.IDLE
                and self._endpointer.in_speech
                and self._endpointer.silence_run_ms > 0
            )
            # Post-acquittal holdoff: resumed playback re-leaks and the loop flaps at ~1 Hz.
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
        so enough fresh words while still ducked confirm without the final decode. Runs on the
        loop; the next frame consumes the flag, re-validating that the utterance is still open."""
        if task.cancelled() or task.exception() is not None:
            return
        if task is not self._eager_task or not self._eager_valid:
            # A superseded candidate's decode must not judge the live one: different audio.
            return
        if (
            self._closing
            or self._preempted
            or self._early_confirm
            or not self._endpointer.in_speech  # closed: the endpoint verdict owns it
        ):
            return
        if (
            self._wake_mode == "strict"
            and not self._wake_claimed
            and self._wake_phrase is not None
            and self._turn is VoiceState.SPEAKING
            and self._canned_base is None  # canned audio takes the normal verdict
        ):
            # Batch analog of the strict partial scan: an unclaimed candidate takes no
            # fresh-words verdict, only the wake-prefix unlock. Latch only; the loop vets.
            if self._wake_strip_leaky(task.result() or "")[0]:
                self._wake_hit_at = time.monotonic()
                self._wake_claimed = True
            return
        if (
            self._wake_mode != "off"
            and self._cfg.wake.ack.enabled
            and not self._wake_claimed
            and self._wake_phrase is not None
            and self._turn in (VoiceState.IDLE, VoiceState.CAPTURING)
        ):
            # Batch analog of the cold partial scan: a speculation that is the BARE phrase
            # latches at the pause, so the fast ack speaks before the final decode.
            matched, rest = self._wake_strip_leaky(task.result() or "")
            if matched is not None and not tokens_of(rest):
                self._wake_hit_at = time.monotonic()
                self._wake_claimed = True
            return
        if self._duck_onset is None:
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
        """Pause-mode acquittal while the utterance is still OPEN: resumed playback would pour
        leak into it until a max-length close, so the candidate audio is dropped whole — stream
        handle, speculation, endpointer state, pre-onset ring — then the pause releases."""
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
                # The wake claim deliberately survives (as in _reset_endpointer): an
                # AEC-warmup-held hit must still bless the follow-up; onset staleness bounds it.

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
        """An early confirm killed the reply but the endpoint verdict found no user speech:
        settle to IDLE, or the session sits in a dead SPEAKING (audio flushed, mic gated in
        half-duplex) until the user happens to speak again."""
        self._metrics.count(f"barge_in_early_orphan.{reason}")
        self._log.warning("early confirm orphaned ({}); reply already stopped", reason)
        if self._turn is not VoiceState.IDLE:
            # THINKING included: nothing produces deltas past the watermark, so it stays dead.
            await self._set_turn(VoiceState.IDLE)

    def _is_ack(self, text: str) -> bool:
        return self._ack_match.covers(tokens_of(text))

    def _is_stop(self, text: str) -> bool:
        """Pure stop command: entirely stop/ack/filler material with a full stop phrase
        present (see PhraseMatcher.pure). Mixed content is NOT a stop."""
        return self._stop_match.pure(tokens_of(text))

    def _is_goal(self, text: str) -> bool:
        """A spoken commitment ("keep working on it until it's done") rides core's /goal command,
        whose sustained-goal mode is the only one where a plain answer does not end the turn.
        Unlike a stop, the phrase sits INSIDE real content, so the whole utterance is the goal."""
        return self._goal_lex is not None and phrase_within(text, self._goal_lex)

    @staticmethod
    def _fresh_seq(text: str, fresh: set[str]) -> list[str]:
        """The fresh words in UTTERANCE order: multi-word stop phrases need contiguity, which
        the fresh SET destroyed. ``fresh`` is the echo filter's UNIT alphabet (CJK bigrams),
        tokens are the lexicon's — a token is fresh when any of its units is."""
        return [t for t in tokens_of(text) if units_of(t) & fresh]

    def _note_spoken(self, text: str, hold_ms: float) -> None:
        """Words just went out: they must not read back as user speech, and they are
        what ``stallNoticeS`` measures (a wordless earcon is a receipt, not an update)."""
        self._echo.note_spoken(text, hold_ms=hold_ms)
        self._cur_turn.last_audible = time.monotonic()

    def _judge_fresh(self, text: str, fresh: set[str]) -> str | None:
        """The shared confirm arm of both early-verdict sites (streaming partials, eager
        decode): "confirm" when the fresh evidence clears the bar — min words, or ONE full stop
        phrase in the ordered fresh remainder — "hold" under the AEC warmup, None otherwise."""
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
        if self._canned_base is VoiceState.IDLE and not preempted:
            # Only a wake ack is live: flush it; killed=False keeps the wake/stop notes honest.
            # A preempted verdict killed a real reply, so its accounting wins and the ack plays.
            self._canned_base = None
            cancel_task(self._drain_task)
            await self._sink.flush()
            return False, heard
        if not interrupting or preempted:
            return preempted, heard
        if not self._cur_turn.dead:
            heard = await self._do_interrupt()
        else:
            # Already killed (early confirm, or a prior verdict): never /stop twice, and no
            # heard-up-to against cleared spans. But audio started AFTER the kill (the timeout
            # notice) still plays: stop it and its drain watcher, or it talks over what follows.
            cancel_task(self._drain_task)
            await self._sink.flush()
        return True, heard

    async def _consume_stop(
        self, stop_text: str, heard: str | None, *, interrupting: bool, preempted: bool
    ) -> None:
        """A pure stop command: kill whatever is live and publish NOTHING — silence is the
        acknowledgment. The heard-up-to contract survives as a pending note on the next publish;
        a stop that killed nothing leaves no note and never arms the double-tap grace."""
        killed, heard = await self._kill_live_reply(
            interrupting=interrupting, preempted=preempted, heard=heard
        )
        self._clear_duck()
        self._chunker.flush()
        # The echo window deliberately stays armed (unlike the publish path, which resets it
        # for a NEW turn): leak captured during this stop's own STT window must still classify
        # as self-echo. It ages out on its own.
        if self._adaptive is not None:
            self._adaptive.drop_anchor()  # a command is not a turn to learn pauses from
        if killed:
            self._last_kill = time.monotonic()  # arms/extends the double-tap grace
            self._pending_note = _stop_note(stop_text, heard)
        self._metrics.count("barge_in_stop")
        if self._turn is not VoiceState.IDLE:
            await self._set_turn(VoiceState.IDLE)

    async def _finish_stt(self, pending: _PendingUtterance) -> tuple[str, str]:
        """The utterance's transcript plus the path that produced it (``stream`` = the streaming
        handle's tail flush, ``eager`` = the speculation, ``batch`` = a fresh decode), naming what
        ``stt_ms`` paid for. A max-length close means speech continued past the speculation: it
        is DRAINED (it cannot be aborted) and discarded, so the fresh decode never contends."""
        task = pending.eager
        if (
            task is not None
            and not pending.eager_always_valid
            and pending.trim_bytes != pending.eager_trim
        ):
            # The decode saw differently-trimmed audio (a hit after the eager mark, or a stream
            # handle the restart missed): drain it (one-decode invariant) and decode fresh.
            self._metrics.count("stt_eager_stale")
            with suppress(Exception):
                await task
            task = None
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
            # guard sees it); the fresh decode below would otherwise break the one-decode
            # invariant in the slow-ASR regime it protects. No new eager can appear here:
            # _worker_decoding wraps this call.
            self._metrics.count("stt_eager_drained")
            with suppress(Exception):
                await stale
        pcm = pending.pcm[pending.trim_bytes:]
        if not pcm:  # everything was wake phrase; the empty rung takes it
            return "", "batch"
        return await self._transcribe(pcm), "batch"

    # ---- output: streamed reply -> chunker -> TTS -> sink -------------------

    def note_agent_activity(self) -> None:
        """Non-delta bus traffic (progress, tool events, any send) proves the core is alive:
        push the deadman so it measures a silent core, never a long-running tool."""
        self._cur_turn.last_activity = time.monotonic()

    def note_proactive(self) -> None:
        """This delivery is agent-initiated (cron/trigger metadata on its sends): its settle
        re-opens sentence attention, so "snooze it" needs no re-wake after a reminder speaks.
        Sticky until the settle consumes it (see _set_turn)."""
        self._cur_turn.proactive = True

    async def on_delta(self, delta: str, stream_id: str | None = None) -> None:
        """A streamed assistant text chunk (``_stream_delta``)."""
        if not delta:
            return
        base = base_of(stream_id)
        if self._is_rejected(base):
            return
        if base is not None:
            self._cur_turn.base = base
        if self._turn in (VoiceState.IDLE, VoiceState.CAPTURING):
            # No published turn is live, so this stream IS an unsolicited delivery (cron fire
            # with streaming on) riding the recycled turn object: restart its audibility ledger,
            # as speak_final does, or a stale emitted_audio latch swallows the silence fallback.
            turn = self._cur_turn
            turn.spoke_text = turn.emitted_audio = turn.fallback_done = False
            turn.answered = False
            self._heard_prefix = ""  # as speak_final does: not this delivery's words
        self._cancel_prologue()  # the reply is arriving; no more filler
        self._cancel_midturn()   # a new segment began; the old boundary watch is stale
        self._cur_turn.last_activity = time.monotonic()
        if self._cur_turn.await_first_token and self._cur_turn.published_at:
            self._cur_turn.await_first_token = False
            first_ms = (time.monotonic() - self._cur_turn.published_at) * 1000.0
            self._metrics.observe("agent_first_token_ms", first_ms)
            self._note_first_reply(first_ms)
            self._log.info("first_token ({} ms after publish)", int(first_ms))
        if self._cur_turn.continuation_pending:
            # First post-tool token: re-anchor so the resumed segment's audio lands in
            # continuation_ms, never in ttfa_ms (a slow tool must not read as a slow model).
            self._cur_turn.continuation_pending = False
            self._metrics.turn_continuation()
        self._note_reply_markdown(delta)
        if self._turn is not VoiceState.SPEAKING:
            await self._set_turn(VoiceState.SPEAKING)
            self._duck_if_capturing()
        for chunk in self._chunker.feed(delta):
            if self._cur_turn.segment_first is None:
                self._cur_turn.segment_first = chunk
            self._tts_enqueue(chunk)  # echo filter is fed at EMIT time (see _tts_worker)
            self._cur_turn.segment_spoke = True

    async def on_stream_end(self, *, resuming: bool, stream_id: str | None = None) -> None:
        """A ``_stream_end`` marker; ``resuming`` means tool calls follow. The segment is
        COMPLETE model output either way, so the chunker is always flushed — a short pre-tool
        status line is under the first-chunk floor and would sit silent through the tool wait."""
        base = base_of(stream_id)
        if self._is_rejected(base):
            return
        self._cur_turn.last_activity = time.monotonic()
        tail = self._chunker.flush()
        if tail:
            if self._cur_turn.segment_first is None:
                self._cur_turn.segment_first = tail
            self._tts_enqueue(tail)
            self._cur_turn.segment_spoke = True
        # Judged only now: the same opener before a tool call is a status line, and only a
        # non-resuming end marks this segment as the one that delivers the answer.
        first, self._cur_turn.segment_first = self._cur_turn.segment_first, None
        if not resuming and first and _opens_with_wait_phrase(first):
            self._metrics.count("reply_wait_phrase")
        spoke, self._cur_turn.segment_spoke = self._cur_turn.segment_spoke, False
        if resuming:
            if spoke:
                # The agent masked the tool wait with its own spoken status line.
                self._metrics.count("agent_prologue")
            self._cur_turn.continuation_pending = True
            self._arm_midturn(spoke)
            return
        self._cur_turn.continuation_pending = False
        if spoke:
            self._cur_turn.answered = True
        elif self._turn is VoiceState.THINKING:
            # Core fires a non-resuming end on its blank-response RETRY path too, so an empty
            # terminal does not prove the turn is over (and nothing plays, so nothing drains).
            # Hold: let the final or the deadman decide, never disarm under a live run.
            return
        self._schedule_drain()

    def _note_reply_markdown(self, text: str) -> None:
        """Turn-level latch: rate of turns whose reply carried markdown at all. The carry
        joins a marker split across deltas; it starts as a newline because a reply opens one."""
        turn = self._cur_turn
        if turn.md_counted:
            return
        probe = turn.md_carry + text
        if _MD_PROBE.search(probe):
            turn.md_counted = True
            self._metrics.count("reply_markdown")
        turn.md_carry = probe[-_MD_CARRY:]

    async def speak_final(self, text: str) -> None:
        """A non-streamed final assistant message (streaming disabled / fallback /
        an unsolicited delivery — cron reply, message-tool send)."""
        self._cancel_prologue()
        self._cancel_midturn()
        # A final while a segment is open: the stream died without its end marker (core's
        # delivery.fail never closes it), or this send is out-of-band mid-turn. Discard the
        # buffered partial — losing a fragment's words beats gluing them onto this text.
        self._chunker.flush()
        # This send is its own delivery, and the IDLE placeholder is shared across
        # unsolicited ones: a stale ledger swallows the fallback, a stale steer latch
        # reads a finished delivery as still working.
        turn = self._cur_turn
        turn.spoke_text = turn.emitted_audio = turn.fallback_done = False
        turn.continuation_pending = False
        turn.answered = True  # a whole message stands as the turn's spoken reply
        if self._turn in (VoiceState.IDLE, VoiceState.CAPTURING):
            # A fresh delivery, not a continuation: the previous turn's heard-up-to is not ours.
            self._heard_prefix = ""
        self._note_reply_markdown(text)
        if _opens_with_wait_phrase(text):  # the whole message IS the delivery
            self._metrics.count("reply_wait_phrase")
        if self._cur_turn.await_first_token and self._cur_turn.published_at:
            # Non-streaming: the whole generation IS the wait the filler masks.
            self._cur_turn.await_first_token = False
            self._note_first_reply(
                (time.monotonic() - self._cur_turn.published_at) * 1000.0
            )
        await self._set_turn(VoiceState.SPEAKING)
        self._duck_if_capturing()
        for chunk in self._chunker.feed(text):
            self._tts_enqueue(chunk)
        tail = self._chunker.flush()
        if tail:
            self._tts_enqueue(tail)
        self._schedule_drain()

    async def speak_unanswered(self, text: str) -> None:
        """Core's substitute for a final the model never produced (see the channel's
        ``_streamed_final``), spoken only when this turn said nothing of its own: without it a
        blank reply after a failed tool reads as a dead device."""
        if self._cur_turn.answered:
            return  # the turn answered; that stamp was honest
        self._metrics.count("reply_unanswered_final")
        self._log.warning("turn produced no answer of its own; speaking core's notice")
        await self.speak_final(text)

    # ---- TTS stage + drain --------------------------------------------------

    def _tts_enqueue(self, text: str) -> None:
        if not text:
            return
        self._cur_turn.spoke_text = True  # text went in; the fallback checks audio came out
        if self._cur_turn.chunk_await and self._cur_turn.published_at:
            self._cur_turn.chunk_await = False
            self._metrics.observe(
                "chunker_wait_ms", (time.monotonic() - self._cur_turn.published_at) * 1000.0
            )
        self._tts_queue.put_nowait((self._sink.epoch, text))

    async def _tts_worker(self) -> None:
        while True:
            # Pre-block snapshot: streamed enqueues set segment_spoke, so post-get it is
            # already on; the prior value separates a text stall from idle/tool waits.
            spoke = self._cur_turn.segment_spoke
            wait_epoch = self._sink.epoch
            wait_t0 = time.monotonic()
            epoch, text = await self._tts_queue.get()
            try:
                if self._tts is None or epoch != self._sink.epoch:
                    continue  # tts off, or barged in before synthesis
                wait_ms = (time.monotonic() - wait_t0) * 1000.0
                if (
                    spoke
                    and epoch == wait_epoch
                    and self._pcm_out
                    and wait_ms >= _GAP_COUNT_MIN_MS
                    and self._sink.starved_ms() > 0.0
                ):
                    self._metrics.observe("tts_text_wait_ms", wait_ms)
                if self._pcm_out and not self._cur_turn.tts_first_pending:
                    await self._jit_pipeline(epoch, text)
                else:
                    # Turn's first chunk (pure TTFA) and blob mode: as-is, unscheduled.
                    await self._synth_and_emit(epoch, text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - never let one chunk kill the worker
                self._log.warning("tts error: {}", exc)
            finally:
                self._tts_queue.task_done()

    def _coalesce_into(self, epoch: int, pending: list[str]) -> None:
        """Drain same-epoch queued chunks into the candidate up to max_chars (checked before
        append). Drained items are task_done'd here, so queue.join() accounting stays exact.
        No-op once the epoch died: draining then re-queues a successor's chunks out of order."""
        if epoch != self._sink.epoch:
            return
        while (
            sum(map(len, pending)) < self._cfg.chunker.max_chars
            and not self._tts_queue.empty()
        ):
            nxt_epoch, nxt = self._tts_queue.get_nowait()
            self._tts_queue.task_done()
            if nxt_epoch != epoch:  # defensive; the entry guard makes this unreachable
                self._tts_queue.put_nowait((nxt_epoch, nxt))
                break
            pending.append(nxt)
            self._metrics.count("tts_coalesced")

    def _take_piece(self, pending: list[str], budget: int) -> str:
        """Whole chunks greedily up to *budget*, CJK-aware separator between them; the head is
        cut inside only when it alone exceeds the budget (those seams cost prosody: counted),
        never below the first-chunk floor."""
        head = pending[0]
        if len(head) > budget:
            cut = _cut_index(head, budget, self._cfg.chunker.min_chars_first)
            piece, rest = head[:cut].strip(), head[cut:].strip()
            if rest:
                pending[0] = rest
                self._metrics.count("tts_piece_split")
                return piece
            pending.pop(0)  # the cut only shed whitespace: the whole head went
            return piece
        parts = [pending.pop(0)]
        size = len(parts[0])
        while pending and size + len(pending[0]) <= budget:
            nxt = pending.pop(0)
            last = parts[-1]
            # >= U+2E80 is CJK and up: those scripts take no space.
            parts.append("" if (last[-1].isspace() or ord(last[-1]) >= 0x2E80) else " ")
            parts.append(nxt)
            size += len(nxt)
        return "".join(parts)

    def _runway_ms(self) -> float:
        """Unplayed audio ahead of the listener, CONTINUOUS: backlog_ms over-counts the in-flight
        sink item until its write returns (a step signal the deadline math cannot use), so prefer
        the span ledger against played_ms. Falls back to backlog_ms between segments (no spans)."""
        if self._spoken_spans and self._spans_gen == self._sink.stream_generation:
            rem = (
                self._spans_base_ms
                + sum(dur for _, dur in self._spoken_spans)
                - self._sink.played_ms()
            )
            return max(0.0, rem)
        return float(self._sink.backlog_ms())

    async def _jit_pipeline(self, epoch: int, head: str) -> None:
        """Deadline-scheduled synthesis of one queue item plus coalesced successors. The
        remainder is carried HERE, never re-queued (that would reorder it behind fresh arrivals);
        the head's task_done stays with the caller until every piece is emitted, so _settle's
        queue.join() covers it. A verdict resolves a paused sink, a barge-in bumps the epoch."""
        pending = [head]
        while pending:
            if epoch != self._sink.epoch:
                return  # barged in: the remainder dies with the turn
            self._coalesce_into(epoch, pending)
            mpc = self._synth_mpc
            if mpc is None:
                # Unseeded: no basis to schedule, so take the whole candidate.
                budget = sum(map(len, pending))
            else:
                mpc = max(mpc, _MPC_MIN)
                floor = self._cfg.chunker.min_chars_first
                # The ahead cap folded into WANT keeps release size == cut size.
                cap_chars = int(
                    (_SYNTH_AHEAD_CAP_S - _SYNTH_LEAD_MARGIN_S)
                    / _SYNTH_LEAD_SAFETY * 1000.0 / mpc
                )
                wait_t0 = time.monotonic()
                while True:
                    want = min(
                        sum(map(len, pending)),
                        self._cfg.chunker.max_chars,
                        max(cap_chars, floor),
                    )
                    need_s = want * mpc / 1000.0 * _SYNTH_LEAD_SAFETY + _SYNTH_LEAD_MARGIN_S
                    surplus = self._runway_ms() / 1000.0 - need_s
                    if surplus <= 0.0 or epoch != self._sink.epoch:
                        break
                    # Floored sleep (no churn at the threshold), capped for prompt re-checks.
                    await asyncio.sleep(min(max(surplus, 0.02), _JIT_POLL_CAP_S))
                    self._coalesce_into(epoch, pending)
                if epoch != self._sink.epoch:
                    return
                waited_ms = (time.monotonic() - wait_t0) * 1000.0
                if waited_ms >= 1.0:
                    self._metrics.observe("tts_jit_wait_ms", waited_ms)
                runway_s = self._runway_ms() / 1000.0
                budget_s = max(runway_s - _SYNTH_LEAD_MARGIN_S, 0.0) / _SYNTH_LEAD_SAFETY
                # Floor first, then candidate size: short text is never inflated.
                budget = min(want, max(floor, int(budget_s * 1000.0 / mpc)))
            piece = self._take_piece(pending, budget)
            if not await self._synth_and_emit(epoch, piece):
                return

    async def _synth_and_emit(self, epoch: int, text: str) -> bool:
        """Shared synthesis tail of both worker paths (cost EMA, trim, spans, echo,
        metrics). False only when the epoch died during synthesis (the caller drops
        its remainder); empty audio returns True — the rest may still speak."""
        t0 = time.monotonic()
        if self._pcm_out:
            audio = await self._tts.synthesize_pcm(text)
        else:
            audio = await self._tts.synthesize(text)
        synth_ms = (time.monotonic() - t0) * 1000.0
        # Every chunk: steady-state TTS drift is otherwise invisible until it gaps.
        self._metrics.observe("tts_synth_ms", synth_ms)
        if audio and text:
            # EMA before the epoch check: a barged-in call is still a valid speed sample.
            obs = synth_ms / len(text)
            ema = self._synth_mpc
            if ema is None:
                self._synth_mpc = obs
            else:
                obs = min(max(obs, ema / 4.0), ema * 4.0)
                self._synth_mpc = max(obs, (1.0 - _MPC_ALPHA) * ema + _MPC_ALPHA * obs)
        if epoch != self._sink.epoch:  # barged in during synthesis
            return False
        if not audio:
            return True
        was_first = self._cur_turn.tts_first_pending
        if was_first:
            self._cur_turn.tts_first_pending = False
            self._cur_turn.audible_at = time.monotonic()
            self._metrics.observe("tts_first_chunk_ms", synth_ms)
        elif self._pcm_out:
            # Synthesis lost the race; only a RUNNING stream gone dry counts, not a drained one.
            dry_ms = self._sink.starved_ms()
            if dry_ms >= _GAP_COUNT_MIN_MS:
                self._metrics.count("tts_gap")
                self._metrics.observe("tts_gap_ms", dry_ms)
        elif self._sink.starved():
            self._metrics.count("tts_gap")  # blob mode: the queue ran idle
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
            gen = self._sink.next_generation
            if self._spoken_spans and gen != self._spans_gen:
                # The spans' stream was EOF'd and replaced since they were anchored (a cancelled
                # tool-boundary settle whose fold never ran, tail rung out in full): fold them
                # as heard, since against a stale base the mapping garbles.
                spoken = " ".join(t for t, _ in self._spoken_spans).strip()
                self._heard_prefix = f"{self._heard_prefix} {spoken}".strip()
                self._spoken_spans.clear()
            if not self._spoken_spans:
                # Segment start: anchor at the CURRENT played position (the stream may already
                # have played a filler); a stream this audio will not play on reports someone
                # else's position.
                self._spans_base_ms = (
                    float(self._sink.played_ms())
                    if gen == self._sink.stream_generation else 0.0
                )
                self._spans_gen = gen
            self._spoken_spans.append((text, dur_ms))
        # Fed HERE, not at chunker feed: the eviction window runs from when the words
        # stop being AUDIBLE, hence backlog + this chunk's playtime. Earlier, and a long
        # reply's tail ages out mid-playback and reads back as user speech.
        self._note_spoken(text, self._sink.backlog_ms() + dur_ms)
        self._reply_tail = text  # the LAST segment judges sentence-attention's "?"
        self._cur_turn.emitted_audio = True  # the unvoiced-final fallback stands down
        await self._emit(self._audio_event(epoch, audio))
        return True

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

    # ---- prologue (filler while the agent works) ----------------------------

    def _arm_prologue(self, initial_ms: int | None = None, start_step: int = 0) -> None:
        """Arm the filler timer for the turn just published (cancels any prior). ``initial_ms``
        overrides the first delay and ``start_step`` opens the script mid-way: the tool-boundary
        re-arm passes ``intervalMs`` + step 1 when the agent's own status line WAS the opener."""
        self._cancel_prologue()
        if (
            self._closing
            or self._tts is None
            or not self._cfg.prologue.enabled
            or not self._prologue_phrases
        ):
            return
        self._cur_turn.prologue_task = asyncio.create_task(
            self._prologue_watch(self._sink.epoch, initial_ms, start_step)
        )

    def _cancel_prologue(self) -> None:
        # One sweep for "no more canned audio this wait": every caller must kill a pending
        # earcon too, or its _settle joins the reply's queue and fights the drain watcher.
        self._cur_turn.cancel_prologue()
        cancel_task(self._earcon_task)
        self._earcon_task = None

    def _build_earcon(self, path: str | None, synth) -> bytes | None:
        """One cue, shaped once at init. A custom WAV wins; an unusable file degrades loudly to
        ``synth``'s built-in. Edge-trim runs BEFORE the length cap (a padded export must not
        spend the budget on silence while the cut eats the sound); a real cut fades."""
        rate = self._tts.output_rate if self._pcm_out else 16000
        pcm = b""
        if path:
            try:
                size = Path(path).stat().st_size
                if size > _EARCON_MAX_FILE_B:
                    raise ValueError(f"{size / 1e6:.1f} MB; a cue asset should be tiny")
                src, src_rate = wav_pcm(Path(path).read_bytes())
                if not src:
                    raise ValueError("not a readable S16 WAV")
                if self._pcm_out:
                    src = resample_pcm(src, src_rate, rate)
                else:
                    rate = src_rate  # blob playback follows the header: no resample
                src = trim_lead_silence(src, rate, cap_ms=20.0)
                src = trim_tail_silence(src, rate, cap_ms=120.0)
                cap = int(rate * _EARCON_MAX_MS / 1000) * 2
                if len(src) > cap:
                    logger.warning(
                        "voice: earcon '{}' is {:.0f} ms; truncating to {} ms "
                        "(a cue must stay short)",
                        path, pcm_ms(len(src), rate), _EARCON_MAX_MS,
                    )
                    src = fade_tail_pcm(src[:cap], rate)
                pcm = src
            except Exception as exc:  # noqa: BLE001 - degrade loudly, never mute
                logger.warning(
                    "voice: earcon file '{}' unusable ({}); using the built-in",
                    path, exc,
                )
                rate = self._tts.output_rate if self._pcm_out else 16000
                pcm = b""
        if not pcm:
            pcm = synth(rate)
        if self._cfg.earcons.gain_db:
            pcm = scale_pcm(pcm, 10.0 ** (self._cfg.earcons.gain_db / 20.0))
        audio = pcm if self._pcm_out else pcm_to_wav_bytes(pcm, rate)
        return self._prep_canned(audio)

    def _arm_earcon(self) -> None:
        if self._earcon_audio is None or self._closing:
            return
        cancel_task(self._earcon_task)
        self._earcon_task = asyncio.create_task(self._play_earcon(self._sink.epoch))

    async def _play_earcon(self, epoch: int) -> None:
        """The "captured" receipt cue: canned THINKING audio like a filler, but wordless
        (nothing transcribable, so no echo note) and ~¼ s. Skipped once the user resumed
        speaking."""
        try:
            if (
                self._closing
                or epoch != self._sink.epoch
                or self._turn is not VoiceState.THINKING
                or self._endpointer.in_speech  # never ding over the user
            ):
                return
            self._metrics.count("earcon_captured")
            nonce = object()
            self._canned_base, self._canned_nonce = VoiceState.THINKING, nonce
            try:
                await self._canned_playback(epoch, self._earcon_audio, VoiceState.THINKING)
            finally:
                if self._canned_nonce is nonce:
                    self._canned_base = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a cue must never wedge the turn
            self._log.warning("earcon failed ({})", exc)
            if self._turn is VoiceState.SPEAKING and epoch == self._sink.epoch:
                with suppress(Exception):
                    await self._set_turn(VoiceState.THINKING)

    def _arm_attention_cue(self) -> None:
        """Ensure the close-cue watcher runs for the current attention episode. A live watcher
        re-reads the deadline itself, so it is never replaced — except by its own tail, where
        the episode was reopened mid-cue and the successor takes over."""
        if self._attention_audio is None or self._attention_cued or self._closing:
            return
        task = self._attention_task
        if task is not None and not task.done() and task is not asyncio.current_task():
            return
        self._attention_task = asyncio.create_task(self._attention_watch())

    async def _attention_watch(self) -> None:
        """Play the attention-close cue at the first quiet IDLE moment past the window deadline.
        Deadline moves re-sleep; a live turn exits (its settle re-arms); CAPTURING and canned
        tails are polled out, since a rejected utterance's settle never re-arms."""
        try:
            while True:
                if self._closing or self._attention_cued:
                    return
                wait = self._wake_until - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                    continue
                if self._turn in (VoiceState.SPEAKING, VoiceState.THINKING):
                    if self._canned_base is None:
                        return
                    await asyncio.sleep(0.25)
                    continue
                if self._turn is not VoiceState.IDLE or self._endpointer.in_speech:
                    await asyncio.sleep(0.25)
                    continue
                break
            self._attention_cued = True
            await self._play_attention_cue()
            if not self._attention_cued and not self._closing:
                # A summon flushed the cue and reopened the window (no-op if the settle re-armed).
                self._arm_attention_cue()
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a cue must never wedge the session
            self._log.warning("attention cue failed ({})", exc)

    async def _play_attention_cue(self) -> None:
        """The close cue: canned IDLE-base playback like a wake ack, but wordless. A summon
        mid-cue flushes it through the canned kill branch and owns the state."""
        epoch = self._sink.epoch
        self._metrics.count("earcon_attention")
        self._log.info("attention window closed")
        nonce = object()
        self._canned_base, self._canned_nonce = VoiceState.IDLE, nonce
        try:
            await self._canned_playback(epoch, self._attention_audio, VoiceState.IDLE)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a cue must never wedge the session
            self._log.warning("attention cue failed ({})", exc)
            if self._turn is VoiceState.SPEAKING and epoch == self._sink.epoch:
                with suppress(Exception):
                    await self._set_turn(VoiceState.IDLE)
        finally:
            if self._canned_nonce is nonce:
                self._canned_base = None

    def _note_first_reply(self, ms: float) -> None:
        """Feed the filler-delay EMA. Sample clamped: one pathological turn must
        not push the filler horizon out for the rest of the session."""
        sample = min(ms, 15000.0)
        ema = self._first_reply_ema
        self._first_reply_ema = sample if ema is None else 0.3 * sample + 0.7 * ema

    def _prologue_delay_ms(self, initial_ms: int | None) -> float:
        """First-filler delay: ``afterMs`` is the floor, stretched past the session's typical
        first-reply latency so fillers mark ANOMALOUS waits, not ordinary generation."""
        if initial_ms is not None:
            return float(initial_ms)
        base = float(self._cfg.prologue.after_ms)
        ema = self._first_reply_ema
        return base if ema is None else max(base, _PROLOGUE_TTFT_FACTOR * ema)

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
        """Arm the stalled-agent deadman for the turn just published. Two clocks, because dead
        air and a dead core are different failures: ``last_audible`` (words out) drives the
        NOTICE at ``stallNoticeS`` and never touches the run; ``last_activity`` (deltas, segment
        ends, the send() tap) drives the KILL at ``agentTimeoutS`` — one silent budget warns, a
        SECOND /stops. With streaming and progress both off, 2x agentTimeoutS caps any turn."""
        self._cur_turn.cancel_timeout()
        if self._closing or self._cfg.agent_timeout_s is None:
            return
        self._cur_turn.timeout_task = asyncio.create_task(self._timeout_watch(self._cur_turn))

    async def _timeout_watch(self, turn: _Turn) -> None:
        try:
            budget = float(self._cfg.agent_timeout_s)
            quiet = self._cfg.stall_notice_s  # None: notices ride the core clock alone
            notified = False
            stamp = turn.last_activity

            def _due() -> float:
                core = turn.last_activity + budget
                return min(turn.last_audible + quiet, core) if quiet else core

            while True:
                await wait_until(_due)
                if self._closing or self._cur_turn is not turn:
                    return
                if self._turn is VoiceState.IDLE:
                    return  # settled without us; nothing left to recover
                if self._turn is not VoiceState.THINKING:
                    # Audio in flight (reply tail, canned filler) or the user mid-utterance:
                    # not a silent wedge. Re-arm rather than exit, or a filler playing at the
                    # deadline strips the turn's only recovery.
                    stamp = turn.last_activity = turn.last_audible = time.monotonic()
                    continue
                dead_core = time.monotonic() - turn.last_activity >= budget
                if notified and turn.last_activity != stamp:
                    # The core came back after the notice, then went silent for a fresh
                    # budget: a NEW escalation — the kill always follows its own warning.
                    notified = False
                if dead_core and notified:
                    break
                if dead_core:
                    notified = True
                    self._metrics.count("agent_turn_stall")
                    self._log.warning(
                        "agent turn stalled ({}s with no activity); speaking stall "
                        "notice, killing after another budget", int(budget),
                    )
                    stamp = turn.last_activity = time.monotonic()  # a full budget to the kill
                else:
                    # Working, just not saying anything: speak and back off, never kill.
                    self._metrics.count("agent_turn_quiet")
                    self._log.info(
                        "agent turn audibly silent for {}s while the core works",
                        int(quiet),
                    )
                    quiet = min(quiet * 2.0, budget)
                # Rides the prologue slot: the reply's first delta cancels it like any filler,
                # and its _settle never fights a live drain.
                self._cancel_prologue()
                self._cur_turn.prologue_task = asyncio.create_task(
                    self._stall_notice(self._sink.epoch)
                )
                # A SKIPPED notice must not spin the loop; a played one stamps this itself.
                turn.last_audible = time.monotonic()
            self._log.warning(
                "agent turn silent for another {}s; giving up and /stop-ping it",
                int(budget),
            )
            self._metrics.count("agent_turn_timeout")
            # Detach OURSELVES first: _do_interrupt's abandon() cancels this very task.
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

    async def _stall_notice(self, epoch: int) -> None:
        """Speak stallPhrase over the silent wait (canned THINKING clip, wake-ack
        bracket). Best-effort: any guard failing skips the speech — the escalation
        clock keeps running either way, so the kill still lands one budget later."""
        nonce = object()
        try:
            if self._closing or self._tts is None:
                return
            audio = await self._synth_filler(self._cfg.stall_phrase)
            if (
                not audio
                or epoch != self._sink.epoch
                or self._turn is not VoiceState.THINKING
                or self._endpointer.in_speech  # never talk over the user
            ):
                return
            self._note_spoken(
                self._cfg.stall_phrase, self._sink.backlog_ms() + self._audio_ms(audio)
            )
            self._canned_base, self._canned_nonce = VoiceState.THINKING, nonce
            await self._canned_playback(epoch, audio, VoiceState.THINKING)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - a notice must never wedge the session
            self._log.warning("stall notice failed ({})", exc)
            if self._turn is VoiceState.SPEAKING and epoch == self._sink.epoch:
                with suppress(Exception):
                    await self._set_turn(VoiceState.THINKING)
        finally:
            if self._canned_nonce is nonce:
                self._canned_base = None

    async def _canned_playback(self, epoch: int, audio: bytes, base: VoiceState) -> None:
        """The canned-audio dance every ack/filler/cue shares: SPEAKING -> emit -> settle ->
        half-duplex tail guard (device latency/reverb must not re-trigger the VAD) -> ``base``.
        Callers own the _canned_base/_canned_nonce bracket; a flush mid-dance takes the state."""
        await self._set_turn(VoiceState.SPEAKING)
        await self._emit(self._audio_event(epoch, audio))
        if not await self._settle(epoch):
            return
        if self._turn is VoiceState.SPEAKING:
            if not self._full_duplex:
                await asyncio.sleep(self._cfg.playback_hangover_ms / 1000)
                if self._closing or epoch != self._sink.epoch:
                    return
            await self._set_turn(base)

    async def _settle(self, epoch: int) -> bool:
        """Wait until the segment enqueued at *epoch* is fully synthesized and audibly played out,
        then report whether that epoch still owns the pipeline. Re-checked BEFORE each side
        effect: a watcher surviving into a successor turn would drain its stream and wipe spans."""
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
        """After a segment's audio drains at a tool boundary, reopen the wait: back to THINKING
        (lifting the half-duplex mic gate, so the user can barge in during a long tool run) and
        re-arm the prologue. Cancelled by the next delta, the final drain, barge-in and stop."""
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
        wait alive every ``intervalMs``; cancelled by the first delta / speak_final / barge-in /
        drain. Phrases escalate in order, the last repeating; a SKIPPED filler does not advance."""
        try:
            cfg = self._cfg.prologue
            await asyncio.sleep(self._prologue_delay_ms(initial_ms) / 1000)
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

    def _prep_canned(self, audio: bytes) -> bytes:
        """One-time cache-fill shaping for a canned clip: cap the edge silence (model padding
        delays the voice and holds the half-duplex mic) and bake the blob-mode static duck."""
        if self._pcm_out:
            rate = getattr(self._tts, "output_rate", None) or 0
            audio = trim_lead_silence(audio, rate, cap_ms=20.0)
            return trim_tail_silence(audio, rate, cap_ms=120.0)
        pcm, rate = wav_pcm(audio)
        if not pcm:
            return audio  # unparseable blob: play as-is
        pcm = trim_lead_silence(pcm, rate, cap_ms=20.0)
        pcm = trim_tail_silence(pcm, rate, cap_ms=120.0)
        if self._duck_gain < 1.0:
            pcm = scale_pcm(pcm, self._duck_gain)
        return pcm_to_wav_bytes(pcm, rate)

    async def _synth_filler(self, text: str) -> bytes:
        """Synthesize-and-cache one canned phrase, stored PLAYABLE (edge-trimmed, duck baked).
        A transient failure is never cached as permanent silence."""
        audio = self._fillers.get(text)
        if audio is None:
            audio = await (
                self._tts.synthesize_pcm(text) if self._pcm_out else self._tts.synthesize(text)
            )
            if audio:
                audio = await asyncio.to_thread(self._prep_canned, audio)
            if audio:
                self._fillers[text] = audio
        return audio

    async def _phrase_pcm(self, text: str, rate: int) -> bytes:
        """One calibration clip: the session TTS's own audio at capture rate."""
        if getattr(self._tts, "output_rate", None):
            pcm, src = await self._tts.synthesize_pcm(text), self._tts.output_rate
        else:
            pcm, src = wav_pcm(await self._tts.synthesize(text))
        return resample_pcm(pcm, src, rate) if pcm else b""

    def _wake_entries(self, learned: list[tuple[str, str]] = ()) -> list:
        """WakePhrase entries: phrases, then config aliases (attributed to the single phrase
        when unambiguous, so their ack routes the called name), then calibration pairs."""
        cfg = self._cfg.wake
        src = cfg.phrases[0] if len(cfg.phrases) == 1 else None
        entries: list = list(cfg.phrases)
        entries += [(src, a) if src else a for a in cfg.aliases]
        entries += list(learned)
        return entries

    def _admit_alias(self, phrase: str, rendered: str, floor: tuple) -> str | None:
        """Vet one calibration render before it becomes a wake alias: an alias
        WAKES, so a garbage decode here is a standing false-trigger."""
        rendered = rendered.strip()
        toks = tuple(tokens_of(rendered))
        if not toks:
            return None
        if floor and (
            toks == floor
            # A hallucinating STT's floor varies around a stem: prefix cousins are the same
            # floor. Multi-token only — a 1-token floor prefix would ban too much.
            or (len(floor) >= 2 and toks[: len(floor)] == floor)
            or (len(toks) >= 2 and floor[: len(toks)] == toks)
        ):
            return None
        if all(ord(c) < 0x2E80 for c in rendered) and (
            len(toks) == 1 and len(toks[0]) < 5
        ):
            return None  # a short latin word ("you") must never become a wake
        if self._wake_phrase.leads(rendered):
            return None  # this STT renders the phrase fine
        if len(toks) > len(tokens_of(phrase)) + 3 or len(rendered) > 2 * len(phrase) + 8:
            return None  # garbage-length decode
        if self._is_stop(rendered):
            return None  # a stop-shaped render must never wake
        units = units_of(rendered)
        if units and units <= self._ack_words:
            return None  # pure backchannel material
        if any(WakePhrase([rendered]).present(a) for a in self._wake_ack_list):
            return None  # every ack would echo-veto the summons (validator's rule)
        return rendered

    async def learn_wake_aliases(self) -> None:
        """Warmup calibration: the session TTS speaks each wake phrase, the session STT decodes
        it, and a mis-render registers as an alias (both clip shapes per phrase — renders are
        context-dependent). On-device STT only (a delegate may bill), abandoned once a turn runs."""
        cfg = self._cfg.wake
        if (
            cfg.mode == "off"
            or not cfg.learn_aliases
            or self._wake_phrase is None
            or self._tts is None
            or not getattr(self._tts, "probe_ok", True)
            or self._cfg.stt.provider == "nanobot"
        ):
            return
        rate = self._cfg.audio.sample_rate
        learned: list[tuple[str, str]] = []  # (source phrase, its render)
        try:
            floor = tuple(tokens_of(await self._transcribe(b"\x00" * (2 * rate))))
            for phrase in cfg.phrases[:4]:
                cjk = any(ord(c) >= 0x2E80 for c in phrase)
                for variant in (phrase, phrase + ("。" if cjk else ".")):
                    if self._closing or self._turn is not VoiceState.IDLE:
                        raise _AbandonCalibrationError
                    pcm = await self._phrase_pcm(variant, rate)
                    if not pcm:
                        continue  # unspeakable by this TTS: prewarm warns
                    rendered = await self._transcribe(
                        b"\x00" * rate + pcm + b"\x00" * rate
                    )
                    alias = self._admit_alias(phrase, rendered, floor)
                    if alias and all(alias != a for _, a in learned):
                        learned.append((phrase, alias))
        except _AbandonCalibrationError:
            pass
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - calibration is an optimization
            self._log.warning("wake alias calibration failed ({})", exc)
        if learned:
            # Atomic swap: the hop only reads it; pairs route an alias's ack by the called name.
            self._wake_phrase = WakePhrase(self._wake_entries(learned))
            for _ in learned:
                self._metrics.count("wake_alias_learned")
            self._log.info(
                "wake: this STT renders the phrase as {} — treating those as the "
                "phrase too (pin with wake.aliases, disable with "
                "wake.learnAliases=false)",
                [a for _, a in learned],
            )

    async def prewarm_playback(self) -> None:
        """One silent play through the real device at warmup (see AudioSink.prewarm): dmix
        spin-up / page-in / PCM negotiation move off the first reply's TTFA, and a wrong
        playbackDevice fails loudly. Skipped with tts off, or when a turn is live (hw: PCMs
        are exclusive)."""
        if self._closing or self._tts is None:
            return
        if self._turn is not VoiceState.IDLE:
            return
        await self._sink.prewarm(getattr(self._tts, "output_rate", None) or 16000)

    async def prewarm_canned(self) -> None:
        """Pre-synthesize the canned phrases (prologue fillers, wake acks) at channel warmup so
        the first never pays synthesis inside the moment it masks. ``probe_ok`` gates it like
        the calibrate probe: a cloud TTS must never bill at startup (its phrases stay lazy)."""
        if self._tts is None or not getattr(self._tts, "probe_ok", True):
            return
        # Capped (a pathological list must not burn startup synth; the tail stays lazy) and
        # abandoned the moment a real turn starts: adapters tolerate overlapped synthesis, but
        # the contention would inflate the first reply's TTFA.
        phrases: list[str] = []
        if self._cfg.prologue.enabled:
            phrases += self._prologue_phrases[:8]
        if self._cfg.wake.ack.enabled:
            phrases += self._ack_reachable_texts()
        for text in phrases:
            if self._closing or self._turn is not VoiceState.IDLE:
                return
            try:
                audio = await self._synth_filler(text)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - warmup is an optimization, never a gate
                self._log.debug("canned prewarm failed for '{}': {}", text, exc)
                return
            if not audio:
                self._log.warning(
                    "voice: canned phrase '{}' synthesized to silence (phrase not "
                    "speakable by this TTS engine?) — it will never play", text,
                )
                return

    async def _play_filler(self, epoch: int, step: int) -> bool:
        """Speak escalation-script phrase ``step`` (clamped to the last). Returns whether
        the phrase was emitted, so a skip does not advance the script."""
        phrases = self._prologue_phrases
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
        self._note_spoken(text, self._sink.backlog_ms() + self._audio_ms(audio))
        nonce = object()
        self._canned_base, self._canned_nonce = VoiceState.THINKING, nonce
        try:
            await self._canned_playback(epoch, audio, VoiceState.THINKING)
            return True
        finally:
            if self._canned_nonce is nonce:
                self._canned_base = None

    async def _drain_watch(self, epoch: int) -> None:
        """Return to IDLE once the reply finishes playing (plus a hangover). Gates on
        ``tts_queue.join()`` FIRST (inside ``_settle``) so synthesis completes before we wait on
        the sink; otherwise the sink looks idle between two chunks and drains early."""
        try:
            if not await self._settle(epoch):
                return  # barge-in started a new turn; it owns the state now
            turn = self._cur_turn
            if (
                turn.spoke_text
                and not turn.emitted_audio
                and not turn.fallback_done
                and self._tts is not None
            ):
                # Every chunk died before the speaker (an English core error over a monolingual
                # voice, a degraded synth): the silence must not read as "no answer", and
                # timeoutPhrase doubles as the audible failure notice — localize them together.
                # Turn-scoped on purpose: one audible chunk anywhere stands it down.
                turn.fallback_done = True
                self._metrics.count("reply_unvoiced_fallback")
                self._log.warning(
                    "reply text produced no audible audio (unspeakable for this "
                    "voice?); speaking the fallback notice"
                )
                # Released first: the notice's emit must not read as the turn's first audio.
                self._metrics.turn_end()
                self._tts_enqueue(self._cfg.timeout_phrase)
                if not await self._settle(epoch):
                    return
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
        """Per-frame hop accounting; warn (throttled) only when capture actually lags. Slow
        frames grow response latency by the accumulated capture debt, up to the pipeline's depth,
        past which the source drops audio and the debt pins at the cap. The compute/overhead
        split names the culprit: compute near the budget is the engine stack, overhead is
        dispatch/contention (bulk STT/TTS stealing cores), which no VAD change fixes."""
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
            # Light path: compute and total share one clock, so name both suspects.
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
