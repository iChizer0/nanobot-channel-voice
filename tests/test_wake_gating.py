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


# ---- off mode stays untouched ----------------------------------------------

def test_mode_off_neither_gates_nor_strips():
    async def _case():
        async with EvalConversation() as conv:
            await conv.user_says("hey nanobot what time is it")
            assert conv.texts() == ["hey nanobot what time is it"]

    _run(_case())
