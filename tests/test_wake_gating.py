"""Conversation-level wake-gate behavior over the eval harness: cold-start
gating and stripping, the attention window, wake-only utterances, and strict
mode's barge-in contract (unwoken speech never ducks or kills a live reply; a
wake claim both engages and confirms)."""

from __future__ import annotations

import asyncio
import time

from eval_harness import _FRAME, EvalConversation

from nanobot_channel_voice.backend.base import VoiceState
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
