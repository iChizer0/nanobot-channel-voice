"""Conversation-level wake-gate behavior over the eval harness: cold-start
gating and stripping, the attention window, wake-only utterances, and strict
mode's barge-in contract (unwoken speech never ducks or kills a live reply; a
wake claim both engages and confirms)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest
from eval_harness import _FRAME, EvalConversation

from nanobot_channel_voice.backend.base import VoiceState
from nanobot_channel_voice.backend.local import LocalBackend
from nanobot_channel_voice.echo_reject import SelfEchoFilter
from nanobot_channel_voice.wake.base import WakeDetector

_REPLY = "Once upon a time there was a very long story that keeps going"


class _ScriptDetector(WakeDetector):
    """Acoustic tier stand-in: fires once when armed, counts frames."""

    def __init__(self):
        self.frames = 0
        self.fire = False
        self.released = False

    def push(self, frame: bytes) -> bool:
        self.frames += 1
        fire, self.fire = self.fire, False
        if fire:
            self.last_score = 0.97
        return fire

    def release(self) -> None:
        self.released = True


def _run(coro):
    return asyncio.run(coro)


def _wake(mode: str, **over) -> dict:
    block = {"mode": mode, "phrases": ["hey nanobot"]}
    block.update(over)
    return {"wake": block}


async def _speaking_reply(conv: EvalConversation, ask: str = "hey nanobot tell me a story"):
    await conv.user_says(ask)
    await conv.agent_replies(_REPLY)
    await conv.wait_state(VoiceState.SPEAKING)
    await conv.wait_played_ms(30)


# ---- gate mode: cold starts -------------------------------------------------

def test_cold_utterance_without_phrase_is_gated():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("what time is it")
            assert conv.texts() == []
            assert conv.counter("wake_gated") == 1
            assert conv.backend._turn is VoiceState.IDLE

    _run(_case())


def test_phrase_wakes_and_is_stripped_from_the_publish():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("hey nanobot what time is it")
            assert conv.texts() == ["what time is it"]
            assert conv.counter("wake_text") == 1

    _run(_case())


def test_attention_window_frees_the_follow_up():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("hey nanobot what time is it")
            await conv.agent_replies("It is noon.")
            await conv.wait_state(VoiceState.IDLE)  # settle: the next turn is COLD
            await conv.user_says("and in tokyo")
            assert conv.texts() == ["what time is it", "and in tokyo"]
            assert conv.counter("wake_gated") == 0

    _run(_case())


def test_window_zero_regates_every_cold_start():
    async def _case():
        async with EvalConversation(**_wake("gate", windowS=0)) as conv:
            await conv.user_says("hey nanobot what time is it")
            await conv.agent_replies("It is noon.")
            await conv.wait_state(VoiceState.IDLE)  # settle: the next turn is COLD
            await conv.user_says("and in tokyo")
            assert conv.texts() == ["what time is it"]
            assert conv.counter("wake_gated") == 1

    _run(_case())


def test_bare_phrase_opens_attention_without_publishing():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("hey nanobot")
            assert conv.texts() == []
            assert conv.counter("wake_only") == 1
            await conv.user_says("turn on the lights")
            assert conv.texts() == ["turn on the lights"]

    _run(_case())


def test_gate_mode_barge_in_stays_free():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await _speaking_reply(conv)
            await conv.user_says("actually use tokyo instead")
            assert conv.interrupts == 1
            assert conv.texts()[-1].startswith("actually use tokyo instead")

    _run(_case())


# ---- strict mode: barge-in requires a wake claim ----------------------------

def test_strict_unwoken_speech_neither_ducks_nor_kills():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await _speaking_reply(conv)
            await conv.user_says("loud bystander chatter here")
            assert conv.interrupts == 0
            assert conv.counter("barge_in_duck") == 0
            assert conv.counter("wake_gated") == 1
            assert conv.texts() == ["tell me a story"]

    _run(_case())


def test_strict_acoustic_claim_early_confirms_mid_utterance():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await _speaking_reply(conv)
            # Hand-rolled user_says: the hit latch lands MID-utterance so the
            # loop consumption takes the early-confirm path, not the verdict
            # kill.
            conv._stt.append("turn off the lights")
            conv.vad.flag = True
            for _ in range(6):
                await conv.backend.push_audio(_FRAME)
            conv.backend._wake_hit_at = time.monotonic()  # hop latch writes
            conv.backend._wake_claimed = True
            for _ in range(14):
                await conv.backend.push_audio(_FRAME)
            conv.vad.flag = False
            for _ in range(conv.backend._cfg.vad.hangover_ms // 20 + 2):
                await conv.backend.push_audio(_FRAME)
            await conv.backend._utt_queue.join()
            assert conv.interrupts == 1
            assert conv.counter("barge_in_early_confirm") == 1
            assert conv.texts()[-1].startswith("turn off the lights")

    _run(_case())


def test_strict_bystander_stop_through_echo_is_gated():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await conv.user_says("hey nanobot tell me a story")
            await conv.agent_replies("The weather in tokyo is sunny today and tomorrow")
            await conv.wait_state(VoiceState.SPEAKING)
            await conv.wait_played_ms(30)
            # Echo-shaped mixture: mostly the bot's own words plus a fresh
            # "stop" — the stop-through-echo override must NOT bypass strict.
            await conv.user_says("the weather in tokyo is sunny stop")
            assert conv.interrupts == 0
            assert conv.counter("barge_in_stop") == 0
            assert conv.texts() == ["tell me a story"]

    _run(_case())


def test_wake_hit_with_no_open_utterance_kills_directly():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await _speaking_reply(conv)
            # The residual was too quiet for the VAD but not the wake model:
            # no utterance ever opens, the reply must still die.
            conv.backend._wake_hit_at = time.monotonic()
            conv.backend._wake_claimed = True
            await conv.user_noise(speech_frames=0, silence_frames=5)
            assert conv.interrupts == 1
            assert conv.counter("wake_kill") == 1
            assert conv.backend._turn is VoiceState.IDLE
            assert conv.backend._pending_note is not None

    _run(_case())


def test_acoustic_detector_is_fed_through_the_hop_and_claims():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(wake_detector=det, **_wake("strict")) as conv:
            await conv.user_says("hey nanobot tell me a story")
            assert det.frames > 0  # the hop actually feeds the detector
            await conv.agent_replies(_REPLY)
            await conv.wait_state(VoiceState.SPEAKING)
            await conv.wait_played_ms(30)
            det.fire = True  # next frame: acoustic hit -> claim -> kill
            await conv.user_says("turn off the lights")
            assert conv.counter("wake_hit") == 1
            assert conv.interrupts == 1
            assert conv.texts()[-1].startswith("turn off the lights")
        assert det.released

    _run(_case())


def test_own_reply_speaking_the_phrase_is_suppressed():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("hey nanobot introduce yourself")
            await conv.agent_replies("You can always say hey nanobot to wake me")
            await conv.wait_state(VoiceState.SPEAKING)
            await conv.wait_played_ms(30)
            # The detector "hears" the reply say the phrase (leak, weak AEC).
            conv.backend._wake_hit_at = time.monotonic()
            conv.backend._wake_claimed = True
            await conv.user_noise(speech_frames=0, silence_frames=3)
            assert conv.counter("wake_echo_suppressed") == 1
            assert conv.interrupts == 0          # the reply survives
            assert conv.backend._wake_claimed is False  # no blessing left behind

    _run(_case())


def test_the_veto_lets_go_once_the_phrase_is_out_of_earshot():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            said = "You can always say hey nanobot to wake me"
            await conv.user_says("hey nanobot introduce yourself")
            await conv.agent_replies(said)
            await conv.wait_state(VoiceState.SPEAKING)
            await conv.wait_played_ms(30)
            conv.backend._echo.reset()               # re-note it as long since quiet
            conv.backend._note_spoken(said, -5000.0)
            conv.backend._wake_hit_at = time.monotonic()
            conv.backend._wake_claimed = True
            await conv.user_noise(speech_frames=0, silence_frames=3)
            assert conv.counter("wake_echo_suppressed") == 0
            assert conv.interrupts == 1              # the summon lands

    _run(_case())


def test_the_echo_horizon_spans_the_open_utterance_the_text_tier_reads():
    cfg = SimpleNamespace(audio=SimpleNamespace(sample_rate=16000))
    ep = SimpleNamespace(in_speech=False, pos=16000 * 2 * 8, open_pos=0)  # 8 s of capture
    stub = SimpleNamespace(_endpointer=ep, _cfg=cfg)
    assert LocalBackend._wake_echo_horizon(stub) == pytest.approx(3.0)  # earshot floor
    ep.in_speech = True
    assert LocalBackend._wake_echo_horizon(stub) == pytest.approx(8.0)


def test_an_unmatchable_phrase_list_does_not_veto_every_acoustic_hit():
    stub = SimpleNamespace(
        _echo=SelfEchoFilter(), _wake_phrase=None, _wake_echo_horizon=lambda: 3.0,
    )
    stub._wake_phrases_text = ("!!!",)  # tokenizes to nothing: never evidence of echo
    assert LocalBackend._wake_hit_echoed(stub) is False
    stub._wake_phrases_text = ("hey nanobot",)
    assert LocalBackend._wake_hit_echoed(stub) is False
    stub._echo.note_spoken("say hey nanobot to wake me")
    assert LocalBackend._wake_hit_echoed(stub) is True


def test_wake_only_kill_arms_the_stop_grace_and_note():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await _speaking_reply(conv, ask="hey nanobot tell me a story")
            await conv.user_says("hey nanobot")  # bare phrase mid-reply: kill
            assert conv.interrupts == 1
            assert conv.counter("wake_only") == 1
            assert conv.backend._pending_note is not None
            await conv.user_says("stop")  # double-tap inside the grace: consumed
            assert conv.counter("barge_in_stop") == 1
            assert conv.texts() == ["tell me a story"]

    _run(_case())


def test_strict_wake_prefixed_stop_kills_and_is_consumed():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await _speaking_reply(conv)
            await conv.user_says("hey nanobot stop")
            assert conv.interrupts == 1
            assert conv.counter("barge_in_stop") == 1
            assert conv.texts() == ["tell me a story"]  # the stop published nothing
            await conv.wait_state(VoiceState.IDLE)

    _run(_case())


def test_strict_thinking_continuation_stays_free():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await conv.user_says("hey nanobot tell me a story")
            # No reply yet: the session sits in THINKING; a follow-up is the
            # user's own continuation and must not be gated.
            await conv.wait_state(VoiceState.THINKING)
            await conv.user_says("make it short please")
            assert conv.texts()[-1].startswith("make it short please")
            assert conv.counter("wake_gated") == 0

    _run(_case())


# ---- off mode stays untouched -----------------------------------------------

def test_mode_off_neither_gates_nor_strips():
    async def _case():
        async with EvalConversation() as conv:
            await conv.user_says("hey nanobot what time is it")
            assert conv.texts() == ["hey nanobot what time is it"]

    _run(_case())


# ---- acoustic-tier audio hygiene: the wake sound never reaches STT ----------


def _spy_backend(conv):
    """(seen, pendings): record batch-STT inputs and queued utterances."""
    b = conv.backend
    seen, pendings = [], []
    orig_t = b._transcribe

    async def spy(pcm):
        seen.append(pcm)
        return await orig_t(pcm)

    b._transcribe = spy
    orig_q = b._queue_utterance

    def q(p):
        pendings.append(p)
        orig_q(p)

    b._queue_utterance = q
    return seen, pendings


async def _push_frames(conv, n, *, fire_at=None, det=None):
    for i in range(n):
        if fire_at is not None and i == fire_at:
            det.fire = True
        await conv.backend.push_audio(_FRAME)


async def _close_utterance(conv):
    conv.vad.flag = False
    hangover = conv.backend._cfg.vad.hangover_ms // 20
    for _ in range(hangover + 2):
        await conv.backend.push_audio(_FRAME)
    await conv.backend._utt_queue.join()


def test_acoustic_hit_trims_the_phrase_from_batch_stt():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(wake_detector=det, **_wake("gate")) as conv:
            seen, pendings = _spy_backend(conv)
            conv._stt.append("what time is it")
            conv.vad.flag = True
            await _push_frames(conv, 20, fire_at=8, det=det)
            await _close_utterance(conv)
            (p,) = pendings
            assert p.wake_hit and 0 < p.trim_bytes < len(p.pcm)
            assert len(seen[0]) == len(p.pcm) - p.trim_bytes  # STT saw only the tail
            assert conv.texts() == ["what time is it"]
            assert conv.counter("wake_trim") == 1

    _run(_case())


def test_bare_acoustic_wake_is_attention_only_after_the_trim():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(wake_detector=det, **_wake("gate")) as conv:
            conv._stt.append("")  # a language-limited STT yields nothing for the tail
            conv.vad.flag = True
            await _push_frames(conv, 20, fire_at=10, det=det)
            await _close_utterance(conv)
            assert conv.texts() == []
            assert conv.counter("wake_only") == 1
            await conv.user_says("what time is it")  # the window is open: publishes cold
            assert conv.texts() == ["what time is it"]

    _run(_case())


def test_half_duplex_wake_hit_is_the_only_barge_in():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, aec="auto", **_wake("gate")
        ) as conv:
            assert not conv.backend._open_mic
            await _speaking_reply(conv)
            det.fire = True
            await conv.backend.push_gated_audio(_FRAME)  # the shell's gated-mic tap
            assert conv.counter("wake_kill") == 1
            await conv.wait_state(VoiceState.IDLE)
            await conv.user_says("what time is it")
            # The wake kill's heard-up-to note rides the follow-up publish, beside the text.
            assert conv.texts()[-1] == "what time is it"
            [note] = conv.notes()[-1]
            assert note.startswith("[voice event: the user cut your reply short with the wake word")

    _run(_case())


class _StreamHandle:
    def __init__(self):
        self.fed = 0

    def accept(self, frame: bytes) -> None:
        self.fed += len(frame)

    def partial(self) -> str:
        return ""

    def finish(self) -> str:
        return "stream text"


class _StreamStt:
    streaming = True

    def __init__(self):
        self.handles: list[_StreamHandle] = []

    def stream_start(self) -> _StreamHandle:
        handle = _StreamHandle()
        self.handles.append(handle)
        return handle


def test_streaming_stt_restarts_at_the_claim_and_drops_the_phrase():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(wake_detector=det, **_wake("gate")) as conv:
            stt = _StreamStt()
            conv.backend._stt_stream = stt  # detector => threaded hop, streaming path live
            conv.vad.flag = True
            await _push_frames(conv, 8, fire_at=6, det=det)  # onset at 4, hit at 6
            await _push_frames(conv, 6)                       # the command tail
            await _close_utterance(conv)
            assert len(stt.handles) == 2                      # fresh handle at the claim
            assert conv.texts() == ["stream text"]            # finish of the CLEAN handle
            assert conv.counter("wake_trim") == 1

    _run(_case())


def test_streaming_close_racing_the_restart_demotes_to_batch():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(wake_detector=det, **_wake("gate")) as conv:
            stt = _StreamStt()
            conv.backend._stt_stream = stt
            conv.vad.flag = True
            await _push_frames(conv, 10)
            conv.vad.flag = False
            hangover = conv.backend._cfg.vad.hangover_ms // 20
            for i in range(hangover):
                if i == hangover - 1:
                    det.fire = True  # hit lands ON the closing hop: no restart possible
                await conv.backend.push_audio(_FRAME)
            await conv.backend._utt_queue.join()
            assert len(stt.handles) == 1                # the restart never ran
            # The handle ate the phrase; the whole span trims away -> attention only.
            assert conv.texts() == []
            assert conv.counter("wake_only") == 1
            assert conv.counter("stt_eager_stale") == 1  # the tainted finish was drained

    _run(_case())


def test_eager_speculation_is_pretrimmed_and_survives_the_wake_close():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(wake_detector=det, **_wake("gate")) as conv:
            b = conv.backend
            b._endpointer._eager_frames = 5  # re-enable eager (the harness builds with 0)
            seen, pendings = _spy_backend(conv)
            conv._stt.append("what time is it")
            conv.vad.flag = True
            await _push_frames(conv, 20, fire_at=8, det=det)
            await _close_utterance(conv)
            (p,) = pendings
            assert p.trim_bytes > 0 and p.eager_trim == p.trim_bytes
            assert len(seen) == 1                       # ONE decode: the speculation held
            assert len(seen[0]) < len(p.pcm)            # and it saw pre-trimmed audio
            assert conv.texts() == ["what time is it"]
            assert conv.counter("stt_eager_hit") == 1   # accepted, not drained

    _run(_case())


# ---- 2026-08-19 review refinements ------------------------------------------


def test_name_mention_reply_does_not_suppress_the_wake():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("hey nanobot introduce yourself")
            await conv.agent_replies("They asked what nanobot can do yesterday")
            await conv.wait_state(VoiceState.SPEAKING)
            await conv.wait_played_ms(30)
            # The reply says the phrase's words APART (and "they" contains
            # "hey"): a mention, not the phrase — the veto must not lock the
            # wake word out.
            conv.backend._wake_hit_at = time.monotonic()
            conv.backend._wake_claimed = True
            await conv.user_noise(speech_frames=0, silence_frames=3)
            assert conv.counter("wake_echo_suppressed") == 0
            assert conv.counter("wake_kill") == 1
            assert conv.interrupts == 1

    _run(_case())


def test_strict_wake_through_leak_prefix_unlocks():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await conv.user_says("hey nanobot tell me a story")
            await conv.agent_replies("The weather in tokyo is sunny today and tomorrow")
            await conv.wait_state(VoiceState.SPEAKING)
            await conv.wait_played_ms(30)
            # Mixture transcript: OUR leaked words precede the phrase (below
            # the echo threshold, so the verdict site judges it). The leak
            # prefix must not demote the wake.
            await conv.user_says("the weather is sunny hey nanobot stop")
            assert conv.interrupts == 1
            assert conv.counter("barge_in_stop") == 1
            assert conv.texts() == ["tell me a story"]

    _run(_case())


def test_strict_wake_through_echoed_mixture_unlocks():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await conv.user_says("hey nanobot tell me a story")
            await conv.agent_replies("The weather in tokyo is sunny today and tomorrow")
            await conv.wait_state(VoiceState.SPEAKING)
            await conv.wait_played_ms(30)
            # Echo-CLASSIFIED mixture (6/9 covered): the echo rung's
            # wake_blocked gate must accept the leak-prefixed phrase too.
            await conv.user_says("the weather in tokyo is sunny hey nanobot stop")
            assert conv.counter("barge_in_through_echo") == 1
            assert conv.counter("barge_in_stop") == 1
            assert conv.interrupts == 1

    _run(_case())


def test_wake_kill_is_measured_and_engages_no_duck():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await _speaking_reply(conv)
            conv._stt.append("turn off the lights")
            conv.vad.flag = True
            for _ in range(6):
                await conv.backend.push_audio(_FRAME)
            conv.backend._wake_hit_at = time.monotonic()  # hop latch writes
            conv.backend._wake_claimed = True
            for _ in range(14):
                await conv.backend.push_audio(_FRAME)
            await _close_utterance(conv)
            assert conv.interrupts == 1
            assert conv.texts()[-1].startswith("turn off the lights")
            # The confirm claimed the utterance: no duck engages in the kill
            # cycle NOR over the dead reply's remainder, and the hit->silence
            # latency landed in the new metric.
            assert conv.counter("barge_in_duck") == 0
            assert "wake_kill_ms" in conv.backend._metrics._latency

    _run(_case())


def test_probe_drop_preserves_a_held_wake_claim():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            # An AEC-warmup-held claim: the pause-probe acquitting the LEAK
            # must not eat the wake evidence (onset staleness bounds it).
            conv.backend._wake_claimed = True
            conv.backend._wake_hit_at = time.monotonic()
            await conv.backend._drop_candidate("probe")
            assert conv.backend._wake_claimed is True

    _run(_case())


def test_strict_batch_no_acoustic_warns_at_startup():
    from loguru import logger as loguru_logger

    async def _case():
        messages: list[str] = []
        sink = loguru_logger.add(lambda m: messages.append(str(m)), level="WARNING")
        try:
            async with EvalConversation(**_wake("strict")):
                pass  # batch STT, no detector: the slow-confirm config
            assert sum("eager/endpoint decode" in m for m in messages) == 1
            async with EvalConversation(wake_detector=_ScriptDetector(), **_wake("strict")):
                pass  # an acoustic tier silences it
            assert sum("eager/endpoint decode" in m for m in messages) == 1
        finally:
            loguru_logger.remove(sink)

    _run(_case())


def test_wake_trim_snaps_to_the_quiet_gap_before_the_hit():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(wake_detector=det, **_wake("gate")) as conv:
            _, pendings = _spy_backend(conv)
            conv._stt.append("what time is it")
            conv.vad.flag = True
            await _push_frames(conv, 8)
            await conv.backend.push_audio(b"\x00\x00" * 320)  # phrase/command gap
            det.fire = True
            await _push_frames(conv, 6)  # hit trails into the command's audio
            await _close_utterance(conv)
            (p,) = pendings
            # The cut lands at the gap's END, not at the later hit-chunk end:
            # silence before it, the command's first sample after it.
            assert p.trim_bytes > 0
            assert p.pcm[p.trim_bytes - 2 : p.trim_bytes] == b"\x00\x00"
            assert p.pcm[p.trim_bytes : p.trim_bytes + 2] == b"\x01\x00"
            assert conv.counter("wake_trim") == 1
            assert conv.texts() == ["what time is it"]

    _run(_case())


# ---- canned audio (prologue filler / wake ack) is not a reply ----------------

def test_strict_stop_during_filler_is_consumed_not_gated():
    """The engaged user's stop lands while a prologue filler plays (THINKING
    flipped to SPEAKING for the canned audio): strict must resolve the onset
    against the state beneath the filler, not gate the very user it serves."""

    async def _case():
        async with EvalConversation(
            **_wake("strict"),
            prologue={"enabled": True, "afterMs": 0, "phrases": ["x" * 400]},
        ) as conv:
            await conv.user_says("hey nanobot what's the weather")
            assert conv.texts() == ["what's the weather"]
            await conv.wait_state(VoiceState.SPEAKING)  # the filler, no reply yet
            await conv.user_says("wait")
            assert conv.counter("wake_gated") == 0
            assert conv.counter("barge_in_stop") == 1
            assert conv.interrupts == 1  # the in-flight agent turn was stopped
            assert conv.texts() == ["what's the weather"]  # the stop was consumed

    _run(_case())


def test_strict_steer_during_filler_passes_the_gate_and_injects():
    async def _case():
        async with EvalConversation(
            **_wake("strict"),
            prologue={"enabled": True, "afterMs": 0, "phrases": ["x" * 400]},
        ) as conv:
            await conv.user_says("hey nanobot what's the weather")
            await conv.wait_state(VoiceState.SPEAKING)
            await conv.user_says("actually make that in celsius please")
            assert conv.counter("wake_gated") == 0
            # The filler is canned audio over a THINKING run: the verdict joins the
            # state beneath it, so the steer INJECTS into the live turn — never a
            # kill of the very run the filler masks.
            assert conv.interrupts == 0
            assert conv.texts()[-1] == "actually make that in celsius please"
            assert not any("interrupted" in n for n in conv.notes()[-1])

    _run(_case())


# ---- wake acknowledgment (wake.ack) ------------------------------------------

async def _until(cond, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while not cond():
        if time.monotonic() >= deadline:
            raise AssertionError("condition never held")
        await asyncio.sleep(0.005)


def test_bare_wake_summons_the_ack_then_listens():
    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            await conv.user_says("hey nanobot")
            assert conv.texts() == []
            assert conv.counter("wake_only") == 1
            await _until(lambda: conv.counter("wake_ack") == 1)
            await _until(lambda: VoiceState.SPEAKING in conv.states)  # it played
            await conv.wait_state(VoiceState.IDLE)  # and the mic reopened
            await conv.user_says("turn on the lights")  # window open, ack invisible
            assert conv.texts() == ["turn on the lights"]
            assert conv.notes() == [()]

    _run(_case())


def test_wake_plus_command_does_not_ack():
    async def _case():
        async with EvalConversation(
            **_wake("gate", ack={"enabled": True})
        ) as conv:
            await conv.user_says("hey nanobot what time is it")
            assert conv.texts() == ["what time is it"]
            await asyncio.sleep(0.05)
            assert conv.counter("wake_ack") == 0

    _run(_case())


def test_command_over_the_playing_ack_publishes_clean():
    """Speech that lands on the ack's tail: strict must not gate it, the ack is
    flushed (not /stopped), and no interrupted/heard note is manufactured."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            **_wake("strict", ack={"enabled": True, "phrases": ["y" * 400]}),
        ) as conv:
            await conv.user_says("hey nanobot")
            await conv.wait_state(VoiceState.SPEAKING)  # the ack, ~2.4 s of audio
            await conv.user_says("turn on the lights please")
            assert conv.texts() == ["turn on the lights please"]
            assert conv.notes()[-1] == ()
            assert conv.interrupts == 0
            assert conv.counter("wake_gated") == 0
            assert conv.counter("barge_in_duck") == 1  # the ack yielded under speech

    _run(_case())


def test_wake_during_reply_kills_acks_and_still_notes():
    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1, **_wake("gate", ack={"enabled": True})
        ) as conv:
            await _speaking_reply(conv)
            await conv.user_says("hey nanobot")
            assert conv.interrupts == 1  # the reply died
            assert conv.counter("wake_only") == 1
            await _until(lambda: conv.counter("wake_ack") == 1)
            await conv.wait_state(VoiceState.IDLE)
            await conv.user_says("next question")
            assert conv.texts()[-1] == "next question"
            assert any(
                "cut your reply short with the wake word" in n
                for n in conv.notes()[-1]
            )

    _run(_case())


def test_resummon_during_the_playing_ack_lets_it_finish():
    """A repeat summon while the ack speaks does not flush-and-restart it (an
    audible stutter for a phrase this short): the playing ack IS the answer."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["z" * 400]}),
        ) as conv:
            await conv.user_says("hey nanobot")
            await conv.wait_state(VoiceState.SPEAKING)
            await conv.user_says("hey nanobot")  # re-summon over the playing ack
            assert conv.counter("wake_only") == 2
            assert conv.counter("wake_ack") == 1  # the first ack just continues
            assert conv.interrupts == 0
            await conv.user_says("do the thing")  # a real turn flushes the tail
            assert conv.texts() == ["do the thing"]
            assert conv.notes() == [()]  # canned audio never leaves a note

    _run(_case())


def test_fast_ack_plays_unducked_through_the_summons_hangover():
    """The fast ack plays inside the summon utterance's trailing hangover: that
    stale in-speech must not duck (or, in pause mode, self-pause) it — while
    FRESH speech resuming still ducks it at once."""

    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i" * 400]}),
        ) as conv:
            conv._stt.append("")
            conv.vad.flag = True
            await _push_frames(conv, 10, fire_at=8, det=det)
            conv.vad.flag = False
            await _push_frames(conv, 13)
            await _until(lambda: conv.counter("wake_ack_fast") == 1, timeout=2.0)
            await conv.wait_state(VoiceState.SPEAKING)
            await _push_frames(conv, 5)  # capture keeps flowing under the ack
            assert conv.counter("barge_in_duck") == 0
            conv.vad.flag = True  # the user resumes: NOW the ack yields
            await _push_frames(conv, 3)
            assert conv.counter("barge_in_duck") == 1
            conv.vad.flag = False
            await _close_utterance(conv)

    _run(_case())


def test_ack_skips_while_the_user_is_already_speaking():
    async def _case():
        async with EvalConversation(**_wake("gate", ack={"enabled": True})) as conv:
            conv.backend._endpointer._in_speech = True
            await conv.backend._wake_ack(conv.sink.epoch)
            assert conv.counter("wake_ack") == 0
            assert conv.backend._turn is VoiceState.IDLE
            conv.backend._endpointer._in_speech = False

    _run(_case())


def test_ack_prewarm_and_language_defaults():
    from nanobot_channel_voice.backend.local import (
        _WAKE_ACK_BUILTINS,
        _WAKE_ACK_FALLBACK,
        _wake_ack_phrases,
    )

    assert _wake_ack_phrases(None, "zh") == _WAKE_ACK_BUILTINS["zh"]
    assert _wake_ack_phrases(None, None) == _WAKE_ACK_FALLBACK
    assert _wake_ack_phrases(None, "fr") == _WAKE_ACK_FALLBACK
    assert _wake_ack_phrases(["custom"], "zh") == ["custom"]

    async def _case():
        async with EvalConversation(**_wake("gate", ack={"enabled": True})) as conv:
            await conv.backend.prewarm_canned()
            assert "Yes?" in conv.backend._fillers  # synthesized into the shared cache

    _run(_case())


def test_ack_language_follows_the_tts_not_the_called_name():
    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            **_wake("gate", phrases=["hey nanobot", "小娜"], ack={"enabled": True}),
        ) as conv:
            spoken = set(conv.backend._wake_ack_list)
            for phrase, n in (("小娜", 1), ("hey nanobot", 2)):
                await conv.user_says(phrase)
                await _until(lambda n=n: conv.counter("wake_ack") == n)
                # Either summon acks in the engine's OWN language; a zh name never pulls
                # a zh ack out of an engine resolved to another one.
                assert set(conv.backend._fillers) & spoken
                assert "在呢。" not in conv.backend._fillers
                await conv.wait_state(VoiceState.IDLE)

    _run(_case())


def test_cold_acoustic_summon_fast_acks_before_the_endpoint():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            conv._stt.append("")
            conv.vad.flag = True
            await _push_frames(conv, 10, fire_at=8, det=det)
            conv.vad.flag = False
            await _push_frames(conv, 13)  # ~260 ms trailing silence, hangover still open
            await _until(lambda: conv.counter("wake_ack_fast") == 1, timeout=2.0)
            assert conv.counter("wake_ack") == 1  # spoke without waiting the verdict out
            await _close_utterance(conv)          # now the endpoint verdict runs
            assert conv.counter("wake_only") == 1
            assert conv.counter("wake_ack") == 1  # the rung did NOT double-ack
            await conv.wait_state(VoiceState.IDLE)
            await conv.user_says("what time is it")
            assert conv.texts() == ["what time is it"]
            assert conv.notes() == [()]

    _run(_case())


def test_same_breath_command_suppresses_the_fast_ack():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, **_wake("gate", ack={"enabled": True}),
        ) as conv:
            conv._stt.append("what time is it")
            conv.vad.flag = True
            await _push_frames(conv, 10, fire_at=4, det=det)
            await asyncio.sleep(0.45)   # the probe window elapses mid-speech
            await _push_frames(conv, 6)
            assert conv.counter("wake_ack_fast") == 0
            await _close_utterance(conv)
            assert conv.texts() == ["what time is it"]
            await asyncio.sleep(0.05)
            assert conv.counter("wake_ack") == 0  # wake+command never acks

    _run(_case())


def test_half_duplex_cold_summon_acks_at_the_close_not_mid_capture():
    """No SPEAKING flip while the summon utterance is OPEN (it would gate the
    mic mid-capture) — but at the close that argument is void, so the ack
    speaks during the STT wait instead of after the verdict."""

    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, aec="auto", playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            assert not conv.backend._open_mic
            conv._stt.append("")
            conv.vad.flag = True
            await _push_frames(conv, 10, fire_at=8, det=det)
            conv.vad.flag = False
            await _push_frames(conv, 13)
            await asyncio.sleep(0.45)
            # Utterance still open: no probe runs in half-duplex.
            assert conv.counter("wake_ack_fast") == 0
            await _close_utterance(conv)
            await _until(lambda: conv.counter("wake_ack_fast") == 1)
            assert conv.counter("wake_only") == 1
            assert conv.counter("wake_ack") == 1  # the rung did NOT double-ack

    _run(_case())


def test_fast_ack_is_not_self_paused_in_pause_mode():
    """Pause-mode twin of the unducked test: pre-fix, the stale hangover would
    pause the sink the moment the fast ack started — the ack then sat frozen
    until the verdict's duck clear, defeating the fast path entirely."""

    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, playbackHangoverMs=1,
            bargeIn={"mode": "pause"},
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            conv._stt.append("")
            conv.vad.flag = True
            await _push_frames(conv, 10, fire_at=8, det=det)
            conv.vad.flag = False
            await _push_frames(conv, 13)
            await _until(lambda: conv.counter("wake_ack_fast") == 1, timeout=2.0)
            await _push_frames(conv, 5)  # frames flow while the ack plays
            assert conv.counter("barge_in_duck") == 0  # no self-pause
            await _close_utterance(conv)
            await conv.wait_state(VoiceState.IDLE)  # the ack drained on its own

    _run(_case())


# ---- repeat summons ----------------------------------------------------------


def test_double_call_is_a_bare_summon():
    """The universal double-call habit ("小娜小娜"): pre-fix, the second phrase
    leaked to the agent as content."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            await conv.user_says("hey nanobot hey nanobot")
            assert conv.texts() == []
            assert conv.counter("wake_only") == 1
            await _until(lambda: conv.counter("wake_ack") == 1)

    _run(_case())


def test_repeated_phrase_strips_to_the_command():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("hey nanobot hey nanobot what time is it")
            assert conv.texts() == ["what time is it"]

    _run(_case())


def test_repeated_zh_phrase_strips_fused():
    async def _case():
        async with EvalConversation(**_wake("gate", phrases=["小娜"])) as conv:
            await conv.user_says("小娜小娜今天天气")
            assert conv.texts() == ["今天天气"]
            await conv.user_says("小娜，小娜")  # separated repeat, window open
            assert conv.texts() == ["今天天气"]
            assert conv.counter("wake_only") == 1

    _run(_case())


def test_clitic_bound_repeat_stays_content():
    async def _case():
        async with EvalConversation(**_wake("gate", phrases=["nanobot"])) as conv:
            await conv.user_says("nanobot nanobot's cool")
            assert conv.texts() == ["nanobot's cool"]

    _run(_case())


def test_resummon_during_thinking_keeps_the_query():
    """The "are you there?" check while the agent works: pre-fix it /stopped
    the in-flight turn and manufactured a false cut-short note."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            await conv.user_says("hey nanobot what's the weather")
            assert conv.texts() == ["what's the weather"]
            await conv.user_says("hey nanobot")  # anxious re-summon, THINKING
            assert conv.interrupts == 0
            assert conv.counter("wake_only") == 1
            assert conv.counter("wake_reassure") == 1
            await _until(lambda: VoiceState.SPEAKING in conv.states)  # the clip
            await conv.wait_state(VoiceState.THINKING)  # back beneath it
            await conv.agent_replies("here is the weather")
            await conv.wait_state(VoiceState.IDLE)
            await conv.user_says("thanks")
            assert conv.texts()[-1] == "thanks"
            assert conv.notes()[-1] == ()  # no false "cut your reply" note

    _run(_case())


def test_resummon_during_thinking_prefers_the_prologue_filler():
    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            prologue={"enabled": True, "afterMs": 60000, "phrases": ["on it"]},
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            await conv.user_says("hey nanobot what's the weather")
            await conv.user_says("hey nanobot")
            assert conv.interrupts == 0
            await _until(lambda: conv.counter("prologue_filler") == 1)
            await asyncio.sleep(0.05)
            assert conv.counter("wake_ack") == 0  # the filler IS the reassure

    _run(_case())


def test_resummon_during_thinking_without_canned_audio_is_silent_but_safe():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("hey nanobot what's the weather")
            await conv.user_says("hey nanobot")
            assert conv.interrupts == 0
            assert conv.counter("wake_reassure") == 1
            assert VoiceState.SPEAKING not in conv.states  # nothing to say
            await conv.agent_replies("here is the weather")  # query survived
            await conv.wait_state(VoiceState.IDLE)

    _run(_case())


def test_acoustic_resummon_during_thinking_keeps_the_query():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            await conv.user_says("hey nanobot what's the weather")
            conv._stt.append("")
            conv.vad.flag = True
            await _push_frames(conv, 10, fire_at=4, det=det)
            await _close_utterance(conv)
            assert conv.interrupts == 0
            assert conv.counter("wake_only") == 1
            assert conv.counter("wake_reassure") == 1
            await conv.agent_replies("here is the weather")
            await conv.wait_state(VoiceState.IDLE)

    _run(_case())


def test_resummon_during_playing_filler_lets_it_finish():
    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            prologue={"enabled": True, "afterMs": 0, "phrases": ["y" * 400]},
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            await conv.user_says("hey nanobot what's the weather")
            await conv.wait_state(VoiceState.SPEAKING)  # the filler, ~2.4 s
            await conv.user_says("hey nanobot")  # over the playing filler
            assert conv.interrupts == 0
            assert conv.counter("wake_only") == 1
            assert conv.counter("wake_reassure") == 0  # the filler speaks
            assert conv.counter("prologue_filler") == 1

    _run(_case())


def test_half_duplex_tap_over_filler_keeps_the_query():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, aec="auto", playbackHangoverMs=1,
            prologue={"enabled": True, "afterMs": 0, "phrases": ["y" * 400]},
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            await conv.user_says("hey nanobot what's the weather")
            await conv.wait_state(VoiceState.SPEAKING)  # the filler gates the mic
            det.fire = True
            await conv.backend.push_gated_audio(_FRAME)
            assert conv.backend._turn is VoiceState.THINKING  # flushed, not killed
            assert conv.interrupts == 0
            assert conv.counter("wake_kill") == 0
            await conv.agent_replies("here is the weather")
            await conv.wait_state(VoiceState.IDLE)

    _run(_case())


def test_wake_during_reply_over_own_speech_still_kills():
    """The floor-taking contract is untouched: a summon over an AUDIBLE reply
    still kills it (test_wake_during_reply_kills_acks_and_still_notes covers
    the verdict path; this pins the base-resolution refactor)."""

    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            await _speaking_reply(conv)
            det.fire = True
            await conv.backend.push_audio(_FRAME)  # hit, no utterance open
            await _until(lambda: conv.interrupts == 1)
            assert conv.counter("wake_kill") == 1

    _run(_case())


# ---- fast-ack robustness -----------------------------------------------------


def test_fast_ack_fires_at_late_quiet():
    """Quiet that misses the single probe instant but lands moments later:
    pre-fix the one-shot probe fell all the way back to the verdict ack."""

    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            conv._stt.append("")
            conv.vad.flag = True
            await _push_frames(conv, 10, fire_at=8, det=det)
            conv.vad.flag = False
            await _push_frames(conv, 8)   # 160 ms: under the 240 ms quiet bar
            await asyncio.sleep(0.4)      # the old single-shot instant passes
            assert conv.counter("wake_ack_fast") == 0
            await _push_frames(conv, 5)   # 260 ms total: quiet NOW
            await _until(lambda: conv.counter("wake_ack_fast") == 1, timeout=2.0)
            await _close_utterance(conv)
            assert conv.counter("wake_ack") == 1

    _run(_case())


def test_close_ack_catches_the_short_hangover_summon():
    """A hangover shorter than the probe's grace kills the claim before any
    poll: the at-close hook is the catch."""

    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, playbackHangoverMs=1,
            vad={"hangoverMs": 100},
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            conv._stt.append("")
            conv.vad.flag = True
            await _push_frames(conv, 10, fire_at=8, det=det)
            await _close_utterance(conv)
            await _until(lambda: conv.counter("wake_ack_fast") == 1)
            assert conv.counter("wake_only") == 1
            assert conv.counter("wake_ack") == 1

    _run(_case())


def test_close_ack_skips_the_long_command():
    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, **_wake("gate", ack={"enabled": True}),
        ) as conv:
            conv._stt.append("what time is it")
            conv.vad.flag = True
            await _push_frames(conv, 75, fire_at=8, det=det)  # 1.5 s of speech
            await _close_utterance(conv)
            assert conv.texts() == ["what time is it"]
            await asyncio.sleep(0.05)
            assert conv.counter("wake_ack") == 0

    _run(_case())


# ---- text-tier fast path -----------------------------------------------------


class _ScriptedPartialHandle:
    def __init__(self, owner):
        self._owner = owner

    def accept(self, frame: bytes) -> None:
        pass

    def partial(self) -> str:
        return self._owner.partial_text

    def finish(self) -> str:
        return self._owner.finish_text


class _ScriptedPartialStt:
    streaming = True

    def __init__(self):
        self.partial_text = ""
        self.finish_text = "hey nanobot"

    def stream_start(self) -> _ScriptedPartialHandle:
        return _ScriptedPartialHandle(self)


def test_cold_partial_wake_latches_the_text_tier_fast_ack():
    """No acoustic tier: the phrase leading a streaming partial is the only
    pre-verdict wake evidence — with the ack on it must arm the fast path."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            stt = _ScriptedPartialStt()
            conv.backend._stt_stream = stt
            conv.backend._threaded_hop = True  # streaming implies it in real builds
            conv.vad.flag = True
            await _push_frames(conv, 4)
            stt.partial_text = "hey nanobot"
            await _push_frames(conv, 8)  # next ~100 ms partial poll latches
            assert conv.backend._wake_claimed
            conv.vad.flag = False
            await _push_frames(conv, 13)
            await _until(lambda: conv.counter("wake_ack_fast") == 1, timeout=2.0)
            await _close_utterance(conv)
            assert conv.counter("wake_only") == 1
            assert conv.counter("wake_ack") == 1

    _run(_case())


async def _done_task(text: str) -> asyncio.Task:
    async def _ret() -> str:
        return text

    task = asyncio.get_running_loop().create_task(_ret())
    await task
    return task


def test_eager_bare_phrase_latches_the_text_tier_fast_ack():
    """Batch STT twin: a speculation decoding to the BARE phrase latches at the
    pause, so the ack speaks before the final decode."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            conv._stt.append("hey nanobot")  # the close's own decode
            conv.vad.flag = True
            await _push_frames(conv, 10)
            task = await _done_task("hey nanobot")
            conv.backend._eager_task = task
            conv.backend._eager_valid = True
            conv.backend._eager_confirm_cb(task)
            assert conv.backend._wake_claimed
            conv.vad.flag = False
            await _push_frames(conv, 13)
            await _until(lambda: conv.counter("wake_ack_fast") == 1, timeout=2.0)
            await _close_utterance(conv)
            assert conv.counter("wake_only") == 1
            assert conv.counter("wake_ack") == 1

    _run(_case())


def test_eager_command_never_latches_the_cold_claim():
    async def _case():
        async with EvalConversation(
            **_wake("gate", ack={"enabled": True}),
        ) as conv:
            conv._stt.append("what time is it")
            conv.vad.flag = True
            await _push_frames(conv, 10)
            task = await _done_task("what time is it")
            conv.backend._eager_task = task
            conv.backend._eager_valid = True
            conv.backend._eager_confirm_cb(task)
            assert not conv.backend._wake_claimed  # not bare: no latch
            await _close_utterance(conv)
            assert conv.texts() == []  # still gated (no wake evidence)
            assert conv.counter("wake_gated") == 1

    _run(_case())


# ---- cold echo veto ----------------------------------------------------------


def test_cold_hit_off_the_own_reply_tail_is_vetoed():
    """The bot just SAID the phrase and the reply drained to IDLE: a detector
    hit off the tail/reverb must not claim, open the window, or fast-ack."""

    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(
            wake_detector=det, playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["i am here"]}),
        ) as conv:
            await conv.user_says("hey nanobot tell me how to call you")
            await conv.agent_replies("just say hey nanobot to wake me")
            await conv.wait_state(VoiceState.IDLE)
            wake_until = conv.backend._wake_until
            det.fire = True
            await conv.backend.push_audio(_FRAME)  # cold hit, echo window hot
            assert conv.counter("wake_echo_suppressed") == 1
            assert not conv.backend._wake_claimed
            assert conv.backend._wake_until == wake_until  # window untouched
            await asyncio.sleep(0.45)
            assert conv.counter("wake_ack") == 0

    _run(_case())


# ---- STT mis-render robustness (aliases, calibration, fuzzy strip) -----------


def test_alias_config_wakes_and_strips():
    """Measured: the zh-en zipformer renders "hey nanobot" as 嘿难道爸 — an
    alias is a first-class spelling of the phrase."""

    async def _case():
        async with EvalConversation(
            **_wake("gate", aliases=["嘿难道爸"])
        ) as conv:
            await conv.user_says("嘿难道爸")
            assert conv.texts() == []
            assert conv.counter("wake_only") == 1
            await conv.user_says("嘿难道爸今天天气")
            assert conv.texts() == ["今天天气"]

    _run(_case())


def test_fuzzy_strip_scrubs_the_mangled_name_inside_the_window():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("hey nanobot hello")
            await conv.agent_replies("hi there")
            await conv.wait_state(VoiceState.IDLE)
            await conv.user_says("he nine obt turn on the lights")
            assert conv.texts()[-1] == "turn on the lights"
            assert conv.counter("wake_fuzzy_strip") == 1

    _run(_case())


def test_fuzzy_bare_mangle_is_a_summons():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("hey nanobot hello")
            await conv.agent_replies("hi there")
            await conv.wait_state(VoiceState.IDLE)
            await conv.user_says("he nine ought")  # mangled bare re-summon
            assert conv.texts() == ["hello"]  # nothing new published
            assert conv.counter("wake_only") == 1

    _run(_case())


def test_fuzzy_never_opens_the_gate():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:
            await conv.user_says("he nine obt turn on the lights")  # cold
            assert conv.texts() == []
            assert conv.counter("wake_gated") == 1
            assert conv.counter("wake_fuzzy_strip") == 0

    _run(_case())


def test_fuzzy_never_unlocks_strict():
    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await _speaking_reply(conv)
            await conv.user_says("he nine obt stop it now")
            assert conv.interrupts == 0
            assert conv.counter("wake_gated") == 1

    _run(_case())


def test_fuzzy_strip_under_an_acoustic_claim():
    """The trim missed (late hit) and the mangled name reached STT: the claim
    licenses the fuzzy scrub."""

    async def _case():
        det = _ScriptDetector()
        async with EvalConversation(wake_detector=det, **_wake("gate")) as conv:
            conv._stt.append("he nine obt what time is it")
            conv.vad.flag = True
            await _push_frames(conv, 20, fire_at=4, det=det)
            await _close_utterance(conv)
            assert conv.texts() == ["what time is it"]
            assert conv.counter("wake_fuzzy_strip") == 1

    _run(_case())


def test_calibration_learns_the_stt_render():
    async def _case():
        async with EvalConversation(
            stt={"provider": "zipformer"}, **_wake("gate")
        ) as conv:
            # calibration decodes: silence floor, then the two clip shapes
            conv._stt.extend(["", "he nine obt", "he nine obt"])
            await conv.backend.learn_wake_aliases()
            assert conv.counter("wake_alias_learned") == 1
            # the learned alias is a full spelling: it WAKES from cold
            await conv.user_says("he nine obt what time is it")
            assert conv.texts() == ["what time is it"]

    _run(_case())


def test_calibration_rejects_floor_and_stop_renders():
    async def _case():
        async with EvalConversation(
            stt={"provider": "zipformer"}, **_wake("gate")
        ) as conv:
            # a decode equal to the silence floor is hallucination, not a render
            conv._stt.extend(["thank you", "thank you", "thank you"])
            await conv.backend.learn_wake_aliases()
            assert conv.counter("wake_alias_learned") == 0
            # a stop-shaped render must never become a wake alias
            conv._stt.extend(["", "stop", "stop"])
            await conv.backend.learn_wake_aliases()
            assert conv.counter("wake_alias_learned") == 0
            # a floor VARIANT (prefix cousin) is still hallucination
            conv._stt.extend(
                ["thank you", "thank you for watching", "thank you for watching"]
            )
            await conv.backend.learn_wake_aliases()
            assert conv.counter("wake_alias_learned") == 0
            # a short common latin word must never become a wake
            conv._stt.extend(["", "you", "you"])
            await conv.backend.learn_wake_aliases()
            assert conv.counter("wake_alias_learned") == 0

    _run(_case())


def test_calibration_skips_the_delegated_transcriber():
    async def _case():
        async with EvalConversation(**_wake("gate")) as conv:  # provider "nanobot"
            conv._stt.extend(["", "he nine obt", "he nine obt"])
            await conv.backend.learn_wake_aliases()
            assert conv.counter("wake_alias_learned") == 0  # may bill: never probed
            assert list(conv._stt)  # nothing was consumed

    _run(_case())


# ---- sentence attention (wake.attention="sentence") --------------------------


def test_sentence_attention_spends_the_window_on_publish():
    async def _case():
        async with EvalConversation(
            **_wake("gate", attention="sentence")
        ) as conv:
            await conv.user_says("hey nanobot what time is it")
            await conv.agent_replies("It is noon.")
            await conv.wait_state(VoiceState.IDLE)
            await conv.user_says("and in tokyo")  # cold: the summon was spent
            assert conv.texts() == ["what time is it"]
            assert conv.counter("wake_gated") == 1
            await conv.user_says("hey nanobot and in tokyo")
            assert conv.texts()[-1] == "and in tokyo"

    _run(_case())


def test_sentence_attention_bare_summon_grants_one_sentence():
    async def _case():
        async with EvalConversation(
            **_wake("gate", attention="sentence")
        ) as conv:
            await conv.user_says("hey nanobot")
            await conv.user_says("turn on the lights")  # the granted sentence
            assert conv.texts() == ["turn on the lights"]
            await conv.agent_replies("Done.")
            await conv.wait_state(VoiceState.IDLE)
            await conv.user_says("and the fan")  # needs a fresh summon
            assert conv.texts() == ["turn on the lights"]
            assert conv.counter("wake_gated") == 1

    _run(_case())


def test_sentence_attention_reopens_for_the_agents_question():
    async def _case():
        async with EvalConversation(
            **_wake("gate", attention="sentence")
        ) as conv:
            await conv.user_says("hey nanobot book a flight")
            await conv.agent_replies("Which city do you mean?")
            await conv.wait_state(VoiceState.IDLE)
            await conv.user_says("beijing")  # the answer must pass cold
            assert conv.texts()[-1] == "beijing"
            await conv.agent_replies("Booked for Beijing.")
            await conv.wait_state(VoiceState.IDLE)
            await conv.user_says("thanks a lot")  # statement reply: re-gated
            assert conv.texts()[-1] == "beijing"
            assert conv.counter("wake_gated") == 1

    _run(_case())


def test_learned_alias_summon_acks_in_the_called_language():
    """The user said the ENGLISH name; the STT wrote 嘿难道爸. The ack must
    follow the called name (canonical phrase), not the mis-render's script."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1, stt={"provider": "zipformer"},
            **_wake("gate", ack={"enabled": True}),
        ) as conv:
            conv._stt.extend(["", "嘿难道爸", "嘿难道爸"])
            await conv.backend.learn_wake_aliases()
            assert conv.counter("wake_alias_learned") == 1
            await conv.user_says("嘿难道爸")
            assert conv.counter("wake_only") == 1
            await _until(lambda: conv.counter("wake_ack") == 1)
            assert not any("在" in t for t in conv.backend._fillers)

    _run(_case())


# ---- ack pre-compute / cache coverage ----------------------------------------


def test_prewarm_covers_every_reachable_ack():
    """The first ack must not pay synthesis inside the very moment it masks, so every
    phrase a summon can route to has to be hot before the session opens."""

    async def _case():
        async with EvalConversation(
            **_wake("gate", phrases=["hey nanobot", "小娜"], ack={"enabled": True}),
        ) as conv:
            await conv.backend.prewarm_canned()
            assert set(conv.backend._ack_reachable_texts()) <= set(conv.backend._fillers)

    _run(_case())


def test_a_skipped_ack_is_counted_and_the_fast_stamp_resets():
    """Every other summon response logs; an ack eaten by a guard must not vanish without
    a trace, and a fast bail must hand the ack back to the verdict rung."""

    async def _case():
        async with EvalConversation(**_wake("gate", ack={"enabled": True})) as conv:
            b = conv.backend
            b._fast_acked_at = time.monotonic()
            await b._wake_ack(b._sink.epoch - 1, None, fast=True)  # stale epoch
            assert conv.counter("wake_ack_skipped") == 1
            assert conv.counter("wake_ack") == 0
            assert b._fast_acked_at == float("-inf")

    _run(_case())


def test_prewarm_outlives_an_unspeakable_phrase():
    """One silent phrase must not abandon the rest of the warmup: the acks behind it
    still have to be hot for the first summon."""

    async def _case():
        async with EvalConversation(
            **_wake("gate", ack={"enabled": True, "phrases": ["mute one", "loud two"]}),
        ) as conv:
            b = conv.backend
            real = b._tts.synthesize_pcm

            async def picky(text, *, voice=None):
                return b"" if text == "mute one" else await real(text, voice=voice)

            b._tts.synthesize_pcm = picky
            await b.prewarm_canned()
            assert "loud two" in b._fillers
            assert "mute one" not in b._fillers

    _run(_case())


def test_an_inaudible_clip_is_never_cached_and_retries():
    """A clip that plays as nothing is a failed synthesis, not a phrase: caching it would
    make one bad moment permanent while every later reply synthesizes fine."""

    async def _case():
        async with EvalConversation(
            **_wake("gate", ack={"enabled": True, "phrases": ["hello there"]}),
        ) as conv:
            b = conv.backend
            real = b._tts.synthesize_pcm
            hush = [True]

            async def maybe_hush(text, *, voice=None):
                pcm = await real(text, voice=voice)
                return bytes(len(pcm)) if hush[0] else pcm  # audible-length pure silence

            b._tts.synthesize_pcm = maybe_hush
            assert await b._synth_filler("hello there") == b""
            assert "hello there" not in b._fillers      # not remembered as a phrase
            hush[0] = False
            assert await b._synth_filler("hello there")  # the retry gets real audio
            assert "hello there" in b._fillers

    _run(_case())


def test_the_ack_rotates_past_a_phrase_this_engine_cannot_voice():
    """A pool whose FIRST entry synthesizes silent must not answer every summon with
    nothing: the cursor steps past it and the next summon speaks the one that works."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1,
            **_wake("gate", ack={"enabled": True, "phrases": ["mute one", "loud two"]}),
        ) as conv:
            b = conv.backend
            real = b._tts.synthesize_pcm

            async def picky(text, *, voice=None):
                return b"" if text == "mute one" else await real(text, voice=voice)

            b._tts.synthesize_pcm = picky
            await b._wake_ack(b._sink.epoch)          # picks 'mute one': nothing to play
            assert conv.counter("wake_ack") == 0
            await b._wake_ack(b._sink.epoch)          # rotated on: 'loud two' speaks
            assert conv.counter("wake_ack") == 1
            assert "loud two" in b._fillers

    _run(_case())


def test_synth_filler_bakes_the_blob_duck_once():
    """Blob mode + duckDb: trims and the static attenuation live IN the cache,
    so a cache-hit ack pays a dict lookup, not per-play thread hops."""

    async def _case():
        from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes, wav_pcm
        from nanobot_channel_voice.backend.audio_sink import (
            scale_pcm,
            trim_lead_silence,
            trim_tail_silence,
        )

        async with EvalConversation(**_wake("gate", ack={"enabled": True})) as conv:
            b = conv.backend
            b._pcm_out = False  # blob mode: no mid-chunk gain control
            raw = await b._tts.synthesize("hello there")
            pcm, rate = wav_pcm(raw)
            expect = pcm_to_wav_bytes(
                scale_pcm(
                    trim_tail_silence(
                        trim_lead_silence(pcm, rate, cap_ms=20.0), rate, cap_ms=120.0
                    ),
                    b._duck_gain,
                ),
                rate,
            )
            baked = await b._synth_filler("hello there")
            assert baked == expect
            assert b._fillers["hello there"] == baked  # cached PLAYABLE

    _run(_case())


def test_canned_cache_trims_the_edge_silence():
    """Model padding around a canned clip delays the audible ack and holds the
    half-duplex mic gated after it (measured ~150 ms lead / 580-790 ms tail on
    matcha): the cache stores the clip edge-trimmed to 20/120 ms caps."""

    async def _case():
        async with EvalConversation(**_wake("gate", ack={"enabled": True})) as conv:
            b = conv.backend

            def silence(ms: int) -> bytes:
                return b"\x00\x00" * (16 * ms)

            class _PaddedTts:
                output_rate = 16000
                spoken_language = None

                async def synthesize_pcm(self, text, *, voice=None):
                    return silence(500) + b"\x00\x40" * (16 * 200) + silence(700)

            b._tts = _PaddedTts()
            audio = await b._synth_filler("hi")
            assert len(audio) == 2 * 16 * (20 + 200 + 120)

    _run(_case())


# ---- earcons: the "captured" receipt cue -------------------------------------


def test_captured_earcon_dings_at_publish():
    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1, earcons={"captured": True},
        ) as conv:
            assert conv.backend._earcon_audio  # synthesized at init, no TTS call
            await conv.user_says("hello there")
            await _until(lambda: conv.counter("earcon_captured") == 1)
            await _until(lambda: VoiceState.SPEAKING in conv.states)  # it played
            await conv.wait_state(VoiceState.THINKING)  # and yielded back
            await conv.agent_replies("hi")
            await conv.wait_state(VoiceState.IDLE)

    _run(_case())


def test_earcon_stays_silent_for_gated_and_stop():
    """Only ACCEPTED turns ding: a gated bystander must not learn the device
    is live, and a consumed stop's acknowledgment IS the silence."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1, earcons={"captured": True}, **_wake("gate"),
        ) as conv:
            await conv.user_says("what time is it")  # cold: gated
            await asyncio.sleep(0.05)
            assert conv.counter("earcon_captured") == 0
            await _speaking_reply(conv)  # publish (dings) + reply underway
            assert conv.counter("earcon_captured") == 1
            await conv.user_says("stop")  # consumed: silence is the ack
            await asyncio.sleep(0.05)
            assert conv.counter("earcon_captured") == 1

    _run(_case())


def test_earcon_skips_when_the_user_already_resumed():
    async def _case():
        async with EvalConversation(earcons={"captured": True}) as conv:
            conv.vad.flag = True
            for _ in range(6):
                await conv.backend.push_audio(_FRAME)  # mid-utterance
            await conv.backend._play_earcon(conv.backend._sink.epoch)
            assert conv.counter("earcon_captured") == 0

    _run(_case())


def test_custom_earcon_file_loads_resamples_and_truncates(tmp_path):
    async def _case():
        from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes

        short = tmp_path / "cue.wav"
        short.write_bytes(pcm_to_wav_bytes(b"\x00\x40" * int(22050 * 0.3), 22050))
        long_ = tmp_path / "long.wav"
        long_.write_bytes(pcm_to_wav_bytes(b"\x00\x40" * (22050 * 2), 22050))

        async with EvalConversation(
            earcons={"captured": True, "path": str(short)},
        ) as conv:
            audio = conv.backend._earcon_audio  # resampled 22.05k -> 16k
            assert audio and abs(len(audio) - int(0.3 * 16000) * 2) <= 200
        async with EvalConversation(
            earcons={"captured": True, "path": str(long_)},
        ) as conv:
            # a receipt cue must stay short: truncated to the 600 ms cap
            assert len(conv.backend._earcon_audio) == 16000 * 2 * 600 // 1000
        async with EvalConversation(
            earcons={"captured": True, "path": str(tmp_path / "missing.wav")},
        ) as broken, EvalConversation(earcons={"captured": True}) as ref:
            # unreadable file degrades loudly to the built-in
            assert broken.backend._earcon_audio == ref.backend._earcon_audio

    _run(_case())


def test_earcon_gain_applies_at_build():
    async def _case():
        import array as _array

        async with EvalConversation(
            earcons={"captured": True, "gainDb": -12.0},
        ) as quiet_conv, EvalConversation(earcons={"captured": True}) as ref:
            quiet = max(abs(x) for x in _array.array("h", quiet_conv.backend._earcon_audio))
            loud = max(abs(x) for x in _array.array("h", ref.backend._earcon_audio))
            assert 0.22 <= quiet / loud <= 0.28  # -12 dB ~ 0.251

    _run(_case())


def test_custom_earcon_pad_trims_before_the_length_cap(tmp_path):
    """A padded export (silence + tone + silence) must not spend the 600 ms
    budget on the pads while the cut eats the tone: edge-trim runs first."""

    async def _case():
        from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes

        rate = 16000
        padded = tmp_path / "padded.wav"
        padded.write_bytes(pcm_to_wav_bytes(
            b"\x00\x00" * int(rate * 0.5)      # 500 ms lead pad
            + b"\x00\x40" * int(rate * 0.3)    # 300 ms tone
            + b"\x00\x00" * int(rate * 0.4),   # 400 ms tail pad
            rate,
        ))
        async with EvalConversation(
            earcons={"captured": True, "path": str(padded)},
        ) as conv:
            audio = conv.backend._earcon_audio
            # kept: 20 ms lead cap + the FULL tone + 120 ms tail cap
            assert len(audio) == 2 * int(rate * (0.02 + 0.3 + 0.12))

    _run(_case())


def test_earcon_truncation_fades_the_cut(tmp_path):
    """A >600 ms sound is cut mid-waveform: the cut must fade, not click."""

    async def _case():
        import array as _array

        from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes

        long_ = tmp_path / "long.wav"
        long_.write_bytes(pcm_to_wav_bytes(b"\x00\x40" * (16000 * 2), 16000))
        async with EvalConversation(
            earcons={"captured": True, "path": str(long_)},
        ) as conv:
            samples = _array.array("h", conv.backend._earcon_audio)
            assert len(samples) == 16000 * 600 // 1000
            assert samples[-160] > 8000  # 10 ms before the cut: still loud
            assert abs(samples[-1]) < 200  # the cut itself lands at ~zero

    _run(_case())


def test_blob_mode_earcon_keeps_the_source_rate(tmp_path):
    """Blob playback follows the WAV header, so a custom cue skips the lossy
    linear resample and plays at its own rate."""

    async def _case():
        from nanobot_channel_voice.audio.pcm import ding_pcm, pcm_to_wav_bytes, wav_pcm

        cue = tmp_path / "cue.wav"
        cue.write_bytes(pcm_to_wav_bytes(b"\x00\x40" * 4410, 44100))
        async with EvalConversation(
            earcons={"captured": True, "path": str(cue)},
        ) as conv:
            b = conv.backend
            b._pcm_out = False
            _, rate = wav_pcm(b._build_earcon(str(cue), ding_pcm))
            assert rate == 44100

    _run(_case())


def test_oversized_earcon_file_degrades_to_the_builtin(tmp_path):
    async def _case():
        from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes

        big = tmp_path / "big.wav"
        big.write_bytes(pcm_to_wav_bytes(b"\x00\x40" * (16000 * 70), 16000))  # ~2.2 MB
        async with EvalConversation(
            earcons={"captured": True, "path": str(big)},
        ) as conv, EvalConversation(earcons={"captured": True}) as ref:
            assert conv.backend._earcon_audio == ref.backend._earcon_audio

    _run(_case())


# ---- strict mode: THINKING continuations ------------------------------------


def test_strict_thinking_in_window_steers_freely():
    """Conversation attention: the publish touches the window, so a follow-up
    while the agent works steers without the name — by mid-turn injection now
    (published into the LIVE run, never a /stop)."""

    async def _case():
        async with EvalConversation(**_wake("strict")) as conv:
            await conv.user_says("hey nanobot tell me a story")
            assert conv.backend._turn is VoiceState.THINKING
            await conv.user_says("actually make it short")
            assert conv.texts()[-1] == "actually make it short"
            assert conv.interrupts == 0  # injected; the story run survives

    _run(_case())


def test_strict_thinking_cold_window_gates_bystanders():
    """Sentence attention: the publish SPENDS the window, so unwoken speech
    during a long tool run must not steer (kill) the pending query."""

    async def _case():
        async with EvalConversation(**_wake("strict", attention="sentence")) as conv:
            await conv.user_says("hey nanobot tell me a story")
            assert conv.backend._turn is VoiceState.THINKING
            await conv.user_says("bystander chatter meanwhile")
            assert conv.texts() == ["tell me a story"]
            assert conv.counter("wake_gated") == 1
            assert conv.interrupts == 0
            assert conv.backend._turn is VoiceState.THINKING  # the query survives

    _run(_case())


def test_strict_thinking_cold_window_wake_steers():
    async def _case():
        async with EvalConversation(**_wake("strict", attention="sentence")) as conv:
            await conv.user_says("hey nanobot tell me a story")
            await conv.user_says("hey nanobot make it a joke instead")
            assert conv.texts()[-1] == "make it a joke instead"
            assert conv.interrupts == 0  # injected into the live run, not killed

    _run(_case())


def test_gate_thinking_cold_window_stays_free():
    """Gate mode trusts the room: sentence-spent THINKING amendments pass bare."""

    async def _case():
        async with EvalConversation(**_wake("gate", attention="sentence")) as conv:
            await conv.user_says("hey nanobot tell me a story")
            await conv.user_says("actually make it short")
            assert conv.texts()[-1] == "actually make it short"

    _run(_case())


# ---- earcons: the attention-close cue ---------------------------------------


def test_attention_cue_plays_when_the_window_lapses():
    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1, earcons={"attention": True},
            **_wake("gate", windowS=0.3),
        ) as conv:
            assert conv.backend._attention_audio  # built at init, no TTS call
            await conv.user_says("hey nanobot what time is it")
            await conv.agent_replies("It is noon.")
            await conv.wait_state(VoiceState.IDLE)
            assert conv.counter("earcon_attention") == 0  # window still open
            await _until(lambda: conv.counter("earcon_attention") == 1)
            await conv.wait_state(VoiceState.IDLE)  # the cue settles back
            # The cue's own tail must not re-open the window: cold speech is
            # now gated, and the episode's cue fired exactly once.
            await conv.user_says("and in tokyo")
            assert conv.texts() == ["what time is it"]
            assert conv.counter("wake_gated") == 1
            await asyncio.sleep(0.45)
            assert conv.counter("earcon_attention") == 1

    _run(_case())


def test_attention_cue_fires_at_settle_when_the_window_is_spent():
    """Sentence attention: a non-question reply settles with the window spent,
    so the cue plays right there (the End-of-Request position)."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1, earcons={"attention": True},
            **_wake("gate", attention="sentence"),
        ) as conv:
            await conv.user_says("hey nanobot what time is it")
            await conv.agent_replies("It is noon.")
            await _until(lambda: conv.counter("earcon_attention") == 1)
            await conv.wait_state(VoiceState.IDLE)

    _run(_case())


def test_attention_cue_waits_out_a_question_window():
    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1, earcons={"attention": True},
            **_wake("gate", attention="sentence", windowS=0.3),
        ) as conv:
            await conv.user_says("hey nanobot book a table")
            await conv.agent_replies("For how many people?")
            await conv.wait_state(VoiceState.IDLE)
            assert conv.counter("earcon_attention") == 0  # the "?" re-opened it
            await _until(lambda: conv.counter("earcon_attention") == 1)

    _run(_case())


def test_attention_cue_needs_the_gate():
    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1, earcons={"attention": True},
        ) as conv:  # wake.mode="off": disabled loudly at init
            assert conv.backend._attention_audio is None
            await conv.user_says("hello there")
            await conv.agent_replies("hi")
            await conv.wait_state(VoiceState.IDLE)
            await asyncio.sleep(0.05)
            assert conv.counter("earcon_attention") == 0

    _run(_case())


def test_attention_cue_fires_at_settle_with_window_zero():
    """windowS=0 keeps the gate always cold: every conversation ends with the
    cue at its settle (the publish resets the episode; the touch no-ops)."""

    async def _case():
        async with EvalConversation(
            playbackHangoverMs=1, earcons={"attention": True},
            **_wake("gate", windowS=0),
        ) as conv:
            await conv.user_says("hey nanobot what time is it")
            await conv.agent_replies("It is noon.")
            await _until(lambda: conv.counter("earcon_attention") == 1)
            await conv.wait_state(VoiceState.IDLE)

    _run(_case())


# ---- agent-initiated deliveries (cron/trigger) re-open attention -------------


def test_proactive_delivery_reopens_sentence_attention():
    """A reminder speaking is the MACHINE opening the conversation: after its
    settle the user answers ("snooze it") without a re-wake, statement or not.
    An ordinary statement reply keeps spending the window as before."""

    async def _case():
        import time as _time

        async with EvalConversation(**_wake("strict", attention="sentence")) as conv:
            conv.backend.note_proactive()  # the channel saw _cron_trigger metadata
            await conv.agent_replies("Time for your meeting.")  # no question mark
            await conv.wait_state(VoiceState.IDLE)
            assert conv.backend._wake_until > _time.monotonic()  # window open
            # The flag was consumed at the settle: a normal statement reply spends
            # the window again (sentence semantics unchanged for user-initiated turns).
            await conv.user_says("hey nanobot tell me a story")
            await conv.agent_replies("Once upon a time, the end.")
            await conv.wait_state(VoiceState.IDLE)
            assert conv.backend._wake_until <= _time.monotonic()

    _run(_case())


def test_half_duplex_without_an_acoustic_tier_warns_that_wake_cannot_interrupt():
    """audio.aec="auto" (the default) is half-duplex: the shell mutes the mic while the bot
    speaks and the gated tap feeds the acoustic detector alone, so the text tier is deaf to
    a barge-in. Silently, until this warning."""
    from loguru import logger as loguru_logger

    seen: list[str] = []

    async def _case(**over):
        handle = loguru_logger.add(lambda m: seen.append(str(m)), level="WARNING")
        try:
            async with EvalConversation(**over):
                pass
        finally:
            loguru_logger.remove(handle)

    _run(_case(aec="auto", **_wake("gate")))
    assert any("cannot interrupt a reply" in m for m in seen)
    seen.clear()
    _run(_case(aec="soft", **_wake("gate")))  # open mic: the phrase is heard live
    assert not any("cannot interrupt a reply" in m for m in seen)
    seen.clear()
    _run(_case(aec="auto", **_wake("off")))  # no wake word to lose
    assert not any("cannot interrupt a reply" in m for m in seen)
