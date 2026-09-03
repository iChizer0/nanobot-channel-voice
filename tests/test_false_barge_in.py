"""False-candidate economics: the pause-probe release (pausing removed the
cause; leak dies, a person talks through), the early-release acquittal twin,
the post-acquittal engage holdoff, and the derived probe windows."""

from __future__ import annotations

import asyncio
import time

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import VoiceState
from nanobot_channel_voice.backend.local import LocalBackend
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.vad.base import Vad
from nanobot_channel_voice.vad.endpointer import Endpointer

_FRAME = b"\x01\x00" * 320  # 20 ms @ 16 kHz


class _ScriptedVad(Vad):
    """is_speech pops from a script; an exhausted script reads silence."""

    def __init__(self, script: list[bool] | None = None) -> None:
        self.script = list(script or [])

    def is_speech(self, frame: bytes) -> bool:
        return self.script.pop(0) if self.script else False

    def scale_floor(self, factor: float) -> None:
        pass


class _Harness:
    def __init__(self) -> None:
        self.published: list[tuple[str, str]] = []
        self.interrupts = 0
        self.transcript = ""


def _build(vad: Vad, *, mode: str = "pause", stt_stream=None, **cfg_over) -> _Harness:
    cfg = VoiceConfig.model_validate(
        {"aec": "soft", "duckDb": -12.0, "bargeIn": {"mode": mode}, **cfg_over}
    )
    harness = _Harness()

    async def transcribe(pcm: bytes) -> str:
        return harness.transcript

    async def publish(text: str, token: str, notes: tuple[str, ...] = ()) -> None:
        harness.published.append((text, token))

    async def interrupt() -> None:
        harness.interrupts += 1

    harness.backend = LocalBackend(
        cfg,
        vad=vad,
        tts=None,
        sink=AudioSink(NullPlayback(), mode="stream"),
        transcribe=transcribe,
        publish_text=publish,
        interrupt=interrupt,
        stt_stream=stt_stream,
    )
    harness.sink = harness.backend._sink
    return harness


def _run(coro):
    return asyncio.run(coro)


# ---- endpointer instrumentation ---------------------------------------------

def test_endpointer_exposes_silence_and_active_runs():
    ep = Endpointer(
        _ScriptedVad([True] * 6 + [False] * 3),
        frame_ms=20, start_frames=5, hangover_ms=600,
        min_utterance_ms=200, max_utterance_ms=15000,
    )
    for _ in range(9):
        assert ep.push(_FRAME) is None
    assert ep.in_speech
    assert ep.active_ms == 120     # 6 speech frames
    assert ep.silence_run_ms == 60  # 3 trailing silence frames


# ---- the pause-probe --------------------------------------------------------

def test_probe_drops_a_leak_candidate_and_releases_the_pause():
    """5 speech frames (the buffered leak tail) then sustained silence: the
    pause removed the cause, so the candidate is dropped LONG before the
    hangover + STT verdict — and nothing is published for it."""
    async def _case():
        # leak_death_ms = max(200, 120 + 50 + 80) = 250 -> 13 silence frames.
        vad = _ScriptedVad([True] * 5 + [False] * 40)
        h = _build(vad)
        b = h.backend
        b._turn = VoiceState.SPEAKING
        for _ in range(5):
            await b.push_audio(_FRAME)
        assert h.sink.paused              # suspicion at 2 frames engaged the pause
        assert b._endpointer.in_speech    # onset latched at 5
        for _ in range(13):
            await b.push_audio(_FRAME)
        assert not h.sink.paused                       # released...
        assert not b._endpointer.in_speech             # ...and the utterance dropped
        assert b._metrics.counters.get("barge_in_false_resume.probe") == 1
        assert b._turn is VoiceState.SPEAKING          # the reply lives on
        assert h.published == [] and h.interrupts == 0
        return h

    _run(_case())


def test_probe_holdoff_blocks_immediate_reengagement():
    async def _case():
        vad = _ScriptedVad([True] * 5 + [False] * 13 + [True] * 6)
        h = _build(vad)
        b = h.backend
        b._turn = VoiceState.SPEAKING
        for _ in range(18):
            await b.push_audio(_FRAME)
        assert b._metrics.counters.get("barge_in_false_resume.probe") == 1
        for _ in range(6):  # resumed playback re-leaks immediately
            await b.push_audio(_FRAME)
        assert not h.sink.paused          # holdoff: the loop must not flap
        assert b._duck_onset is None
        return h

    _run(_case())


def test_probe_exempts_speech_that_outlived_the_leak_window():
    """The last speech flag landing beyond engage+leak_death means the pause did NOT
    silence the source: a person. The candidate rides to the normal verdict."""
    async def _case():
        vad = _ScriptedVad([True] * 26 + [False] * 14)  # last flag ~480 ms post-engage
        h = _build(vad)
        b = h.backend
        b._turn = VoiceState.SPEAKING
        for _ in range(26 + 14):
            await b.push_audio(_FRAME)
        assert h.sink.paused              # still held for the verdict
        assert b._endpointer.in_speech
        assert "barge_in_false_resume.probe" not in b._metrics.counters
        return h

    _run(_case())


def test_probe_never_drops_a_short_real_barge_in():
    """A crisp one-word interruption (~360 ms) then silence: short, but its speech
    outlived the leak window, so it must reach the verdict and interrupt — the
    pre-anchor probe condition (active-length cap) destroyed exactly this."""
    async def _case():
        vad = _ScriptedVad([True] * 18 + [False] * 40)
        h = _build(vad)
        b = h.backend
        b._turn = VoiceState.SPEAKING
        h.transcript = "louder please"
        for _ in range(18 + 31):  # through the 600 ms hangover close
            await b.push_audio(_FRAME)
        assert "barge_in_false_resume.probe" not in b._metrics.counters
        pending = b._utt_queue.get_nowait()  # worker isn't running in the harness
        await b._on_utterance(pending)
        assert h.interrupts == 1
        assert [t for t, _ in h.published][0].startswith("louder please")
        return h

    _run(_case())


def test_engage_recovers_when_the_holdoff_expires_mid_utterance():
    """An onset whose edge fell inside the holdoff still gets its pause the first
    frame the holdoff expires: engagement is state-driven, not edge-driven."""
    async def _case():
        vad = _ScriptedVad([True] * 5 + [False] * 13 + [True] * 11)
        h = _build(vad)
        b = h.backend
        b._turn = VoiceState.SPEAKING
        for _ in range(18):
            await b.push_audio(_FRAME)
        assert not h.sink.paused          # probe acquitted the leak candidate
        for _ in range(10):               # real onset lands inside the holdoff
            await b.push_audio(_FRAME)
        assert b._endpointer.in_speech and b._duck_onset is None
        b._probe_holdoff_until = time.monotonic() - 1.0
        await b.push_audio(_FRAME)        # first frame past the holdoff
        assert h.sink.paused
        return h

    _run(_case())


def test_probe_window_derives_from_playout_delay():
    h = _build(_ScriptedVad(), audio={"playoutDelayMs": 300})
    assert h.backend._leak_death_ms == 500.0  # 120 lead + 300 device + 80 flag lag
    h2 = _build(_ScriptedVad())
    assert h2.backend._leak_death_ms == 250.0


# ---- early release (the acquittal twin) -------------------------------------

def _finished_task(text: str):
    async def _t():
        return text

    async def _make():
        task = asyncio.get_running_loop().create_task(_t())
        await task
        return task

    return asyncio.run(_make())


async def _noop_text(text):
    return text


def _prime_eager(b, text: str, *, eager_active: int | None = 0):
    """A candidate mid-utterance with ITS OWN eager decode completed: the callback
    trusts only the current task (identity + validity) and an unstale snapshot."""
    b._endpointer._in_speech = True
    b._endpointer._buf = bytearray(_FRAME)
    b._endpointer._eager_active = eager_active
    task = asyncio.get_running_loop().create_task(_noop_text(text))
    b._eager_task = task
    b._eager_valid = True
    return task


def test_empty_eager_releases_a_duck_candidate():
    async def _case():
        h = _build(_ScriptedVad([False]), mode="duck")
        b = h.backend
        b._turn = VoiceState.SPEAKING
        b._engage_duck(suspect=False)
        task = _prime_eager(b, "")
        await task
        b._eager_confirm_cb(task)
        assert b._early_release == "eager"
        await b.push_audio(_FRAME)  # the frame path consumes the flag
        assert b._metrics.counters.get("barge_in_false_resume.eager") == 1
        assert h.sink._gain_target == 1.0
        return h

    _run(_case())


def test_resumed_speech_voids_an_empty_eager_acquittal():
    """The user coughed (eager decodes to ''), then started a real sentence while the
    decode ran: the snapshot no longer describes the utterance, so it must not acquit."""
    async def _case():
        h = _build(_ScriptedVad([False]), mode="duck")
        b = h.backend
        b._turn = VoiceState.SPEAKING
        b._engage_duck(suspect=False)
        task = _prime_eager(b, "", eager_active=1)
        b._endpointer._active = 4  # speech resumed since the snapshot
        await task
        b._eager_confirm_cb(task)
        assert b._early_release is None
        return h

    _run(_case())


def test_pause_mode_ignores_transcript_acquittals():
    """In pause mode a wrong transcript-based release would DROP real speech (decoder
    latency is unbounded); acquittal there belongs to the probe alone."""
    async def _case():
        h = _build(_ScriptedVad([True]))  # pause mode
        b = h.backend
        b._turn = VoiceState.SPEAKING
        b._engage_duck(suspect=False)
        b._endpointer._in_speech = True
        b._endpointer._buf = bytearray(_FRAME)
        b._early_release = "partial"  # a transcript acquittal arriving in pause mode
        await b.push_audio(_FRAME)
        assert h.sink.paused              # not released...
        assert b._endpointer.in_speech    # ...and nothing dropped
        return h

    _run(_case())


def test_false_rate_warning_counts_only_leak_shaped_reasons():
    h = _build(_ScriptedVad(), mode="duck")
    b = h.backend
    b._turn = VoiceState.SPEAKING
    for _ in range(3):
        b._engage_duck(suspect=False)
        b._release_duck("ack")  # backchannels are conversation, not echo evidence
    assert len(b._false_times) == 0
    for _ in range(2):
        b._engage_duck(suspect=False)
        b._release_duck("echo")
    assert len(b._false_times) == 2


class _FakeSttHandle:
    def accept(self, pcm: bytes) -> None:
        pass

    def partial(self) -> str:
        return ""

    def finish(self) -> str:
        return ""


class _FakeSttStream:
    streaming = True

    def stream_start(self) -> _FakeSttHandle:
        return _FakeSttHandle()


def test_empty_partial_polls_release_a_stale_candidate():
    """Two consecutive empty-fresh partial polls past the elapsed floor acquit
    the candidate without waiting for the endpoint + final decode."""
    async def _case():
        vad = _ScriptedVad([True] * 12)
        h = _build(vad, mode="duck", stt_stream=_FakeSttStream())
        b = h.backend
        b._turn = VoiceState.SPEAKING
        b._engage_duck(suspect=False)
        b._duck_onset = time.monotonic() - 1.0  # past the 600 ms release floor
        b._endpointer._in_speech = True
        b._endpointer._buf = bytearray(_FRAME)
        b._stt_live = _FakeSttHandle()
        for _ in range(7):  # polls land every ~100 ms of frames: two by frame 6
            await b.push_audio(_FRAME)
        assert b._metrics.counters.get("barge_in_false_resume.partial") == 1
        assert h.sink._gain_target == 1.0
        return h

    _run(_case())


# ---- min-filter blips -------------------------------------------------------

def _blip(b, vad, tag: int = 1):
    """Onset-confirming but under-length speech, then the closing hangover."""
    frame = bytes((tag, 0)) * 320

    async def _go():
        vad.script = [True] * 7  # >= startFrames (5), < minUtteranceMs (10 frames)
        for _ in range(7):
            await b.push_audio(frame)
        assert b._turn is VoiceState.CAPTURING
        for _ in range(32):
            await b.push_audio(bytes((tag + 1, 0)) * 320)

    return _go()


def test_a_rejected_blip_settles_capturing_back_to_idle():
    """The onset flips IDLE -> CAPTURING before the min filter judges; no _on_utterance
    runs for a blip, so without an explicit settle CAPTURING stands indefinitely."""
    async def _case():
        vad = _ScriptedVad()
        h = _build(vad, mode="duck")
        for _ in range(3):  # a noisy room is a run of them
            await _blip(h.backend, vad)
            assert h.backend._turn is VoiceState.IDLE
        return h

    _run(_case())


def test_a_blip_inside_a_queued_utterances_window_leaves_capturing_alone():
    """CAPTURING at onset can mean a PREVIOUS utterance is still decoding; settling it
    would flip the state under that turn and teach the adaptive hangover a bogus pause."""
    async def _case():
        vad = _ScriptedVad()
        h = _build(vad, mode="duck")
        h.backend._worker_decoding = True  # the worker's final decode is in flight
        await _blip(h.backend, vad)
        assert h.backend._turn is VoiceState.CAPTURING
        h.backend._worker_decoding = False
        await _blip(h.backend, vad)
        assert h.backend._turn is VoiceState.IDLE
        return h

    _run(_case())


def test_a_rejected_blip_settles_capturing_on_the_streaming_path():
    async def _case():
        vad = _ScriptedVad()
        h = _build(vad, mode="duck", stt_stream=_FakeSttStream())
        await _blip(h.backend, vad)
        assert h.backend._turn is VoiceState.IDLE
        return h

    _run(_case())


class _RecordingHandle:
    def __init__(self, log: list[int]) -> None:
        self._log = log

    def accept(self, pcm: bytes) -> None:
        self._log.append(pcm[0])

    def partial(self) -> str:
        return ""

    def finish(self) -> str:
        return ""


class _RecordingSttStream:
    streaming = True

    def __init__(self) -> None:
        self.handles: list[list[int]] = []

    def stream_start(self) -> _RecordingHandle:
        log: list[int] = []
        self.handles.append(log)
        return _RecordingHandle(log)


def test_a_rejected_blip_drops_the_pre_onset_ring():
    """The endpointer clears its own pre-trigger on a reject; the STT replay ring must go
    with it, or the rejected audio (and what preceded it) leads the NEXT utterance."""
    async def _case():
        vad = _ScriptedVad()
        stt = _RecordingSttStream()
        h = _build(vad, mode="duck", stt_stream=stt)
        b = h.backend
        for _ in range(25):  # idle: fills the ring with tag 1
            await b.push_audio(bytes((1, 0)) * 320)
        await _blip(b, vad, tag=2)  # tag 2 speech, tag 3 hangover
        assert len(stt.handles) == 1
        for _ in range(4):  # the real utterance's own pre-roll
            await b.push_audio(bytes((4, 0)) * 320)
        vad.script = [True] * 15
        for _ in range(15):
            await b.push_audio(bytes((5, 0)) * 320)
        assert len(stt.handles) == 2
        assert not ({1, 2} & set(stt.handles[1]))
        return h

    _run(_case())
