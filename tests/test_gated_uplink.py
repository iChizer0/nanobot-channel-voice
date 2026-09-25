"""GatedUplink: what leaves the device, and when, per DESIGN-cloud-uplink-gate.md §8.

A scripted VAD and wake detector drive the gate over synthetic frames; a fake inner
backend records the call sequence and plays the adapter's state hints back.
"""

from __future__ import annotations

import asyncio
import math

import pytest

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import (
    OutputTranscript,
    StateHint,
    UserSpeechStarted,
    VoiceState,
)
from nanobot_channel_voice.backend.gated import GatedUplink
from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.vad.base import Vad
from nanobot_channel_voice.wake.base import WakeDetector

RATE = 16000
FRAME_MS = 20
FRAME = RATE * 2 * FRAME_MS // 1000  # 640 bytes


def frame(i: int) -> bytes:
    return bytes([i % 256]) * FRAME


class ScriptVad(Vad):
    heavy = False

    def __init__(self, flags: list[bool]):
        self.flags = list(flags)

    def is_speech(self, frame: bytes) -> bool:
        return self.flags.pop(0) if self.flags else False


class ScriptWake(WakeDetector):
    heavy = False

    def __init__(self, hits: set[int], back_bytes: int = 0):
        self.hits = hits
        self.n = 0
        self.last_hit_back_bytes = back_bytes

    def push(self, frame: bytes) -> bool:
        self.n += 1
        return self.n in self.hits


class FakeInner:
    pace_output_audio = False
    voices_notices = True

    def __init__(self):
        self.calls: list = []
        self.on_event = None
        self.fail_begin = False
        self.live = False  # begin emits UserSpeechStarted when a reply is live
        self.waiting = False  # a tool call owes its answer: a discard settles THINKING

    async def start(self, *, instructions, tools, on_event):
        self.on_event = on_event

    async def push_audio(self, pcm):
        self.calls.append(("push", pcm))

    async def begin_activity(self):
        if self.fail_begin:
            raise RuntimeError("no socket")
        self.calls.append(("begin",))
        await self.on_event(StateHint(VoiceState.CAPTURING))
        if self.live:
            await self.on_event(UserSpeechStarted())

    async def end_activity(self, *, commit=True):
        self.calls.append(("end", commit))
        if not commit:
            await self.on_event(
                StateHint(VoiceState.THINKING if self.waiting else VoiceState.IDLE)
            )

    async def park(self):
        self.calls.append(("park",))

    async def barge_in(self, played_ms):
        self.calls.append(("barge_in", played_ms))

    async def submit_tool_result(self, call_id, output):
        self.calls.append(("result", call_id))

    async def announce(self, text):
        self.calls.append(("announce", text))

    async def close(self):
        self.calls.append(("close",))

    async def emit(self, event):
        await self.on_event(event)

    def pushed(self) -> bytes:
        return b"".join(pcm for kind, pcm in ((c[0], c[1]) for c in self.calls if c[0] == "push"))

    def kinds(self) -> list[str]:
        return [c[0] for c in self.calls]


# hangoverMs floors at 100 = 5 frames: an utterance closes on the 5th silence frame.
VAD_CFG = {
    "start_frames": 2, "preroll_ms": 0, "hangover_ms": 100, "min_utterance_ms": 40,
}


def _recorder():
    async def on_event(e):
        pass

    return on_event


def build(mode="vad", *, vad, detector=None, uplink_rate=RATE, open_mic=False, **cfg):
    config = VoiceConfig(realtime={"uplink": mode, "idleParkS": 0}, vad=VAD_CFG, **cfg)
    inner = FakeInner()
    sink = AudioSink(NullPlayback(), mode="stream")
    gate = GatedUplink(
        inner, config=config, sink=sink, vad=vad, wake_detector=detector,
        capture_rate=RATE, uplink_rate=uplink_rate, open_mic=open_mic,
    )
    shell_events: list = []

    async def on_event(e):
        shell_events.append(e)

    return gate, inner, shell_events, on_event


async def feed(gate, n: int, start: int = 0) -> list[bytes]:
    frames = [frame(i) for i in range(start, start + n)]
    for f in frames:
        await gate.push_audio(f)
    return frames


def test_vad_mode_uploads_one_endpointed_utterance():
    async def _run():
        # 3 silence, 5 speech, 6 silence (the 5th closes)
        vad = ScriptVad([False] * 3 + [True] * 5 + [False] * 6)
        gate, inner, _, on_event = build(vad=vad)
        await gate.start(instructions=None, tools=[], on_event=on_event)
        frames = await feed(gate, 14)
        kinds = inner.kinds()
        assert kinds[0] == "begin" and kinds[-1] == "end"
        assert inner.calls[-1] == ("end", True)
        # Onset on frame 4 uploads the ring (the pre-roll floor holds 6 frames, so all
        # of 0..4), then each frame through the 4th hangover frame; the closing frame
        # (12) and the trailing silence (13) never leave.
        joined = inner.pushed()
        assert joined == b"".join(frames[:12])
        counters = gate._metrics.snapshot()["counters"]
        assert counters["uplink_utterances"] == 1
        assert counters["capture_ms"] == 14 * FRAME_MS
        assert counters["uplink_ms"] == len(joined) * 1000 // (2 * RATE)
        await gate.close()

    asyncio.run(_run())


def test_blip_is_taken_back_uncommitted():
    async def _run():
        # Onset needs 2 speech frames (both count as active), so a blip is a close with
        # active < min: min 60 ms = 3 frames.
        config_min = {**VAD_CFG, "min_utterance_ms": 60}
        vad = ScriptVad([True, True] + [False] * 5)
        config = VoiceConfig(realtime={"uplink": "vad", "idleParkS": 0}, vad=config_min)
        inner = FakeInner()
        gate = GatedUplink(
            inner, config=config, sink=AudioSink(NullPlayback(), mode="stream"), vad=vad,
            capture_rate=RATE, uplink_rate=RATE, open_mic=False,
        )
        await gate.start(instructions=None, tools=[], on_event=_recorder())
        await feed(gate, 7)
        assert inner.calls[0] == ("begin",)
        assert inner.calls[-1] == ("end", False)
        assert gate._metrics.snapshot()["counters"]["gate_blip_aborted"] == 1
        await gate.close()

    asyncio.run(_run())


def test_wake_mode_uploads_nothing_until_the_phrase():
    async def _run():
        vad = ScriptVad([True] * 6 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake(set()),
            wake={"mode": "gate", "phrases": ["hey nanobot"]},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await feed(gate, 11)
        assert inner.kinds() == []
        assert gate._metrics.snapshot()["counters"]["gate_dropped_onsets"] == 1
        await gate.close()

    asyncio.run(_run())


def test_same_breath_command_uploads_from_the_phrase_end():
    async def _run():
        # Speech from frame 0; the hit lands on frame 4 with the phrase ending one
        # frame back (= the start of frame 4) => the upload starts at frame 4.
        vad = ScriptVad([True] * 8 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake({5}, back_bytes=FRAME),
            wake={"mode": "gate", "phrases": ["hey nanobot"]},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        frames = await feed(gate, 13)
        assert inner.calls[0] == ("begin",)
        first = inner.calls[1][1]
        assert first == frames[4]  # adopted buffer: phrase end -> hit frame
        assert inner.pushed() == b"".join(frames[4:12])  # then frame by frame to the close
        assert inner.calls[-1] == ("end", True)
        assert gate._metrics.snapshot()["counters"]["wake_hit"] == 1
        await gate.close()

    asyncio.run(_run())


def test_bare_summon_adopted_mid_utterance_is_discarded_and_opens_the_window():
    """The detector fires before the phrase's own utterance closes (measured on
    openWakeWord), so a bare summon adopts like a same-breath command; with no speech
    past the phrase it is taken back uncommitted, and the window stays open."""

    async def _run():
        # Speech frames 0-4, the hit on the last of them, then silence to the close.
        vad = ScriptVad([True] * 5 + [False] * 5 + [True] * 6 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake({5}, back_bytes=FRAME),
            wake={"mode": "gate", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await feed(gate, 10)
        # Adopted hit frame + 4 hangover frames went up (the closing frame never does).
        assert inner.kinds() == ["begin"] + ["push"] * 5 + ["end"]
        assert inner.calls[-1] == ("end", False)
        counters = gate._metrics.snapshot()["counters"]
        assert counters["gate_bare_summon"] == 1 and counters["wake_hit"] == 1
        inner.calls.clear()
        await feed(gate, 11, start=10)  # the command after the beat, inside the window
        assert inner.calls[0] == ("begin",) and inner.calls[-1] == ("end", True)
        await gate.close()

    asyncio.run(_run())


def test_bare_summon_over_a_reply_kills_it_and_opens_the_window():
    async def _run():
        vad = ScriptVad([False] * 3 + [True] * 4 + [False] * 5)
        gate, inner, shell_events, on_event = build(
            "wake", vad=vad, detector=ScriptWake({2}),
            wake={"mode": "gate", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.SPEAKING))
        inner.live = True
        await feed(gate, 2)  # hit on frame 2, no speech
        assert inner.calls == [("begin",), ("end", False)]
        assert any(isinstance(e, UserSpeechStarted) for e in shell_events)
        inner.live = False
        inner.calls.clear()
        await feed(gate, 10, start=2)  # the command, inside the window
        assert inner.calls[0] == ("begin",) and inner.calls[-1] == ("end", True)
        await gate.close()

    asyncio.run(_run())


def test_a_bare_summon_while_the_agent_works_keeps_the_query_and_admits_the_command():
    """Over THINKING (a delegation's wait) the bare phrase cancels nothing, as locally; in
    strict mode it opens the window that lets the command after it through."""
    async def _run():
        vad = ScriptVad([False] * 3 + [True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake({2}),
            wake={"mode": "strict", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.THINKING))
        await feed(gate, 2)  # hit on frame 2, no speech
        assert inner.calls == []
        await feed(gate, 10, start=2)  # the command after the beat
        assert inner.calls[0] == ("begin",) and inner.calls[-1] == ("end", True)
        await gate.close()

    asyncio.run(_run())


class _Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def monotonic(self) -> float:
        return self.t


@pytest.mark.parametrize("attention, follow_up_at, admitted", [
    ("conversation", 1020.0, True),   # the hit's window shut at 1015; the commit re-opened it
    ("conversation", 1030.0, False),  # ... for windowS only
    ("sentence", 1012.0, False),      # the commit spent the hit's window
])
def test_strict_follow_ups_over_a_working_turn_follow_the_window_as_locally(
    monkeypatch, attention, follow_up_at, admitted,
):
    """Local strict needs the phrase to steer a working turn only once the attention
    window is shut; the gate keeps the same window, not a narrower one."""
    from nanobot_channel_voice.backend import gated

    clock = _Clock()
    monkeypatch.setattr(gated, "time", clock)

    async def _run():
        # A same-breath summoned command (hit on its second frame), then a follow-up.
        vad = ScriptVad([True] * 6 + [False] * 5 + [True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake({2}),
            wake={"mode": "strict", "phrases": ["hey nanobot"], "windowS": 15,
                  "attention": attention},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await feed(gate, 6)  # the phrase at 1000, the command runs on
        clock.t = 1010.0
        await feed(gate, 5, start=6)  # committed at 1010
        assert inner.calls[-1] == ("end", True)
        await inner.emit(StateHint(VoiceState.THINKING))  # the agent works
        inner.calls.clear()
        clock.t = follow_up_at
        await feed(gate, 9, start=11)  # a follow-up without the phrase
        assert (inner.kinds()[:1] == ["begin"]) is admitted
        await gate.close()

    asyncio.run(_run())


def test_a_bare_summon_during_a_tool_wait_admits_the_command_after_it():
    """The adopted phrase alone is taken back and the adapter settles THINKING again (a
    call still owes its answer): strict still lets the command after the beat through."""
    async def _run():
        vad = ScriptVad([True] * 5 + [False] * 5 + [True] * 6 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake({5}, back_bytes=FRAME),
            wake={"mode": "strict", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.THINKING))
        inner.waiting = True
        await feed(gate, 10)
        assert inner.calls[-1] == ("end", False)
        assert gate._state is VoiceState.THINKING
        inner.calls.clear()
        await feed(gate, 11, start=10)
        assert inner.calls[0] == ("begin",) and inner.calls[-1] == ("end", True)
        await gate.close()

    asyncio.run(_run())


def test_strict_needs_the_phrase_over_a_reply_even_inside_the_window():
    """The window frees steering a working turn only: over an audible reply strict needs
    the phrase in the utterance itself, as locally."""
    async def _run():
        vad = ScriptVad([False] * 3 + [True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake({2}),
            wake={"mode": "strict", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.THINKING))
        await feed(gate, 2)  # a bare summon: the window opens
        await inner.emit(StateHint(VoiceState.SPEAKING))  # the answer is audible
        await feed(gate, 10, start=2)
        assert inner.kinds() == []
        await gate.close()

    asyncio.run(_run())


def test_a_bare_summon_while_the_agent_works_keeps_the_engaged_window():
    """The turn owns attention until IDLE: a phrase said during a long wait must not shrink
    the window to windowS, or a later onset in the same wait would be dropped."""
    async def _run():
        vad = ScriptVad([True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake({11}),
            wake={"mode": "gate", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.THINKING))
        await inner.emit(StateHint(VoiceState.IDLE))  # window open (conversation)
        await feed(gate, 9)  # an utterance, committed
        assert inner.calls[-1] == ("end", True)
        await inner.emit(StateHint(VoiceState.THINKING))
        assert gate._window_until == math.inf
        await feed(gate, 2, start=9)  # frame 11: a bare hit during the wait
        assert gate._metrics.snapshot()["counters"]["wake_hit"] == 1
        assert gate._window_until == math.inf
        await gate.close()

    asyncio.run(_run())


def test_a_notice_reaches_the_inner_and_holds_the_park():
    async def _run():
        config = VoiceConfig(realtime={"uplink": "vad", "idleParkS": 60}, vad=VAD_CFG)
        inner = FakeInner()
        gate = GatedUplink(
            inner, config=config, sink=AudioSink(NullPlayback(), mode="stream"),
            vad=ScriptVad([]), capture_rate=RATE, uplink_rate=RATE, open_mic=False,
        )
        await gate.start(instructions=None, tools=[], on_event=_recorder())
        inner.voices_notices = False  # the channel asks before it announces
        assert gate.voices_notices is False
        inner.voices_notices = True
        assert gate._park_task is not None and not gate._park_task.done()
        await gate.announce("Dinner is ready.")
        await asyncio.sleep(0)
        assert inner.calls == [("announce", "Dinner is ready.")]
        assert gate._park_task.done()  # the settle after its reply re-arms it
        await gate.close()

    asyncio.run(_run())


def test_half_duplex_wake_tap_barges_in():
    async def _run():
        gate, inner, _, on_event = build(
            "wake", vad=ScriptVad([]), detector=ScriptWake({1}),
            wake={"mode": "gate", "phrases": ["hey nanobot"]},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.SPEAKING))
        await gate.push_gated_audio(frame(0))
        assert inner.calls == [("begin",), ("end", False)]
        await gate.close()

    asyncio.run(_run())


def test_strict_mode_drops_an_unsummoned_onset_over_a_reply():
    async def _run():
        vad = ScriptVad([True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake(set()),
            wake={"mode": "strict", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        # Engaged then idle: the window is open (conversation attention).
        await inner.emit(StateHint(VoiceState.THINKING))
        await inner.emit(StateHint(VoiceState.IDLE))
        await inner.emit(StateHint(VoiceState.SPEAKING))
        await feed(gate, 9)
        assert inner.kinds() == []
        assert gate._metrics.snapshot()["counters"]["gate_dropped_onsets"] == 1
        await gate.close()

    asyncio.run(_run())


def test_gate_mode_onset_inside_the_window_interrupts():
    async def _run():
        vad = ScriptVad([True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake(set()),
            wake={"mode": "gate", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.THINKING))
        await inner.emit(StateHint(VoiceState.IDLE))
        await inner.emit(StateHint(VoiceState.SPEAKING))
        await feed(gate, 9)
        assert inner.kinds()[0] == "begin" and inner.calls[-1] == ("end", True)
        await gate.close()

    asyncio.run(_run())


def test_window_shuts_after_window_s_and_at_zero():
    async def _run():
        vad = ScriptVad([True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake(set()),
            wake={"mode": "gate", "phrases": ["hey nanobot"], "windowS": 0},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.THINKING))
        await inner.emit(StateHint(VoiceState.IDLE))
        await feed(gate, 9)
        assert inner.kinds() == []  # windowS=0: every cold start needs the phrase
        await gate.close()

    asyncio.run(_run())


def test_sentence_attention_is_spent_unless_the_reply_asks():
    async def _run():
        def make(reply: str):
            vad = ScriptVad([True] * 4 + [False] * 5 + [True] * 4 + [False] * 5)
            gate, inner, _, on_event = build(
                "wake", vad=vad, detector=ScriptWake({1}),
                wake={"mode": "gate", "phrases": ["hey nanobot"], "windowS": 15,
                      "attention": "sentence"},
            )
            return gate, inner, on_event, reply

        for reply, expect_second in (("It is sunny.", False), ("Which city?", True)):
            gate, inner, on_event, reply = make(reply)
            await gate.start(instructions=None, tools=[], on_event=on_event)
            await feed(gate, 9)  # hit + first sentence
            assert inner.calls[-1] == ("end", True)
            await inner.emit(StateHint(VoiceState.THINKING))
            await inner.emit(OutputTranscript(reply))
            await inner.emit(StateHint(VoiceState.SPEAKING))
            await inner.emit(StateHint(VoiceState.IDLE))
            inner.calls.clear()
            await feed(gate, 9, start=9)  # a follow-up without the phrase
            assert (inner.kinds() != []) is expect_second, reply
            await gate.close()

    asyncio.run(_run())


def test_own_reply_saying_the_phrase_vetoes_the_hit():
    """OpenAI-family transcript deltas are token sized: the phrase never sits inside
    one, so the veto must read across delta boundaries — and stamp once per mention,
    not again on every delta that follows it."""

    async def _run():
        gate, inner, _, on_event = build(
            "wake", vad=ScriptVad([False] * 3), detector=ScriptWake({2}),
            wake={"mode": "gate", "phrases": ["hey nanobot"]},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.SPEAKING))
        for delta in ("Just say hey", " nano", "bot and"):
            await inner.emit(OutputTranscript(delta))
        stamp = gate._phrase_echo_until
        assert stamp > 0.0
        await inner.emit(OutputTranscript(" I'm here."))
        assert gate._phrase_echo_until == stamp
        await feed(gate, 3)
        assert inner.kinds() == []
        assert gate._metrics.snapshot()["counters"]["wake_echo_suppressed"] == 1
        # A SECOND mention inside the tail window is its own echo: stamped again, or
        # the phrase spoken ten seconds later summons the bot on its own reply.
        await asyncio.sleep(0.01)
        await inner.emit(OutputTranscript(" Any time, say hey nanobot again."))
        assert gate._phrase_echo_until > stamp
        await gate.close()

    asyncio.run(_run())


def test_failed_begin_drops_the_utterance_without_pushing():
    async def _run():
        vad = ScriptVad([True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(vad=vad)
        inner.fail_begin = True
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await feed(gate, 9)
        assert inner.kinds() == []
        assert gate._metrics.snapshot()["counters"]["gate_activity_failed"] == 1
        await gate.close()

    asyncio.run(_run())


def test_idle_parks_the_inner_and_an_onset_cancels_the_timer():
    async def _run():
        vad = ScriptVad([True] * 2)
        config = VoiceConfig(realtime={"uplink": "vad", "idleParkS": 0.05}, vad=VAD_CFG)
        inner = FakeInner()
        gate = GatedUplink(
            inner, config=config, sink=AudioSink(NullPlayback(), mode="stream"), vad=vad,
            capture_rate=RATE, uplink_rate=RATE, open_mic=False,
        )
        await gate.start(instructions=None, tools=[], on_event=_recorder())
        await asyncio.sleep(0.1)
        assert inner.calls == [("park",)]
        inner.calls.clear()
        await inner.emit(StateHint(VoiceState.THINKING))
        await inner.emit(StateHint(VoiceState.IDLE))  # re-arms the timer
        await feed(gate, 2)  # onset cancels it
        await asyncio.sleep(0.1)
        assert "park" not in inner.kinds()
        await feed(gate, 5, start=2)  # silence closes the utterance
        assert inner.calls[-1] == ("end", True)
        await inner.emit(StateHint(VoiceState.THINKING))
        await inner.emit(StateHint(VoiceState.IDLE))
        await asyncio.sleep(0.1)
        assert inner.calls[-1] == ("park",)
        await gate.close()

    asyncio.run(_run())


def test_uplink_is_resampled_to_the_provider_rate():
    async def _run():
        vad = ScriptVad([True] * 3 + [False] * 5)
        gate, inner, _, on_event = build(vad=vad, uplink_rate=24000)
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await feed(gate, 8)
        # ring (0, 1) + 2 + four hangover frames = 7 frames, each 1.5x longer at 24 kHz
        assert len(inner.pushed()) == 7 * FRAME * 3 // 2
        counters = gate._metrics.snapshot()["counters"]
        assert counters["uplink_ms"] == 7 * FRAME_MS  # measured at the capture rate
        await gate.close()

    asyncio.run(_run())


def test_capture_gap_aborts_an_open_activity():
    async def _run():
        vad = ScriptVad([True] * 4)
        gate, inner, _, on_event = build(vad=vad)
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await feed(gate, 4)
        assert inner.calls[0] == ("begin",)
        await gate.on_capture_gap()
        assert inner.calls[-1] == ("end", False)
        await gate.close()

    asyncio.run(_run())


def test_shell_events_pass_through_unchanged():
    async def _run():
        gate, inner, shell_events, on_event = build(vad=ScriptVad([]))
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.THINKING))
        await inner.emit(OutputTranscript("hi"))
        assert [type(e).__name__ for e in shell_events] == ["StateHint", "OutputTranscript"]
        await gate.barge_in(120)
        await gate.submit_tool_result("c1", "{}")
        assert inner.calls[-2:] == [("barge_in", 120), ("result", "c1")]
        await gate.close()
        assert inner.calls[-1] == ("close",)

    asyncio.run(_run())


def test_mic_gating_under_an_open_utterance_aborts_it():
    """Half-duplex: a reply started while the user spoke; the frame-counted endpointer
    would never close under a gated mic, then commit garbage at the reopen."""

    async def _run():
        vad = ScriptVad([True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(vad=vad)
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await feed(gate, 3)  # onset: activity open
        assert inner.calls[0] == ("begin",)
        await inner.emit(StateHint(VoiceState.SPEAKING))
        await gate.push_gated_audio(frame(3))  # the shell now routes frames here
        assert inner.calls[-1] == ("end", False)
        assert not gate._ep.in_speech
        assert gate._metrics.snapshot()["counters"]["gate_activity_gated"] == 1
        await gate.push_gated_audio(frame(4))  # idempotent: nothing open now
        assert inner.calls[-1] == ("end", False) and inner.kinds().count("end") == 1
        await gate.close()

    asyncio.run(_run())


class _ResetCountingWake(ScriptWake):
    def __init__(self) -> None:
        super().__init__(set())
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1


def test_the_mute_edge_keeps_the_wake_detector_hearing():
    """Half-duplex: the tap feeds the detector on without a gap, so dropping the open
    utterance at the mute keeps its context (a phrase said as the reply starts); a capture
    gap is discontinuous audio and still resets it."""

    async def _run():
        wake = _ResetCountingWake()
        gate, inner, _, on_event = build(
            "wake", vad=ScriptVad([True] * 4 + [False] * 5), detector=wake,
            wake={"mode": "gate", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.THINKING))
        await inner.emit(StateHint(VoiceState.IDLE))  # window open (conversation)
        await feed(gate, 3)  # onset: activity open
        await inner.emit(StateHint(VoiceState.SPEAKING))
        await gate.push_gated_audio(frame(3))  # the mute drops the open utterance
        assert inner.calls[-1] == ("end", False)
        assert wake.resets == 0
        await gate.on_capture_gap()
        assert wake.resets == 1
        await gate.close()

    asyncio.run(_run())


def test_adoption_mark_does_not_outlive_a_lost_session():
    """A session lost under an adopted (same-breath) activity must not leave its
    speech mark behind: the next plain onset would be judged against it."""

    async def _run():
        vad = ScriptVad([True] * 6 + [False] * 5 + [True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake({5}, back_bytes=FRAME),
            wake={"mode": "gate", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await feed(gate, 5)  # adopted at the hit, still open
        assert gate._active and gate._hit_active_ms is not None
        await inner.emit(StateHint(VoiceState.IDLE))  # the session died under it
        await feed(gate, 15, start=5)  # it closes unsent; then a short plain command
        assert inner.calls[-1] == ("end", True)  # judged on its own speech, committed
        assert "gate_bare_summon" not in gate._metrics.snapshot()["counters"]
        await gate.close()

    asyncio.run(_run())


def test_session_lost_under_an_open_activity_drops_it():
    async def _run():
        vad = ScriptVad([True] * 4 + [False] * 5)
        gate, inner, _, on_event = build(vad=vad)
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await feed(gate, 3)
        assert inner.calls[0] == ("begin",)
        await inner.emit(StateHint(VoiceState.IDLE))  # the adapter's _on_session_lost
        inner.calls.clear()
        await feed(gate, 6, start=3)  # the utterance closes: nothing to commit
        assert inner.kinds() == []
        assert gate._metrics.snapshot()["counters"]["gate_activity_lost"] == 1
        await gate.close()

    asyncio.run(_run())


def test_reply_tail_resets_when_a_reply_starts_without_thinking():
    async def _run():
        gate, inner, _, on_event = build(vad=ScriptVad([]))
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.SPEAKING))
        await inner.emit(OutputTranscript("one"))
        await inner.emit(StateHint(VoiceState.IDLE))
        await inner.emit(StateHint(VoiceState.SPEAKING))  # Gemini server VAD: no THINKING
        await inner.emit(OutputTranscript("two"))
        assert gate._reply_tail == "two"
        await gate.close()

    asyncio.run(_run())


def test_failed_begin_re_arms_the_park_timer():
    """The onset cancelled the idle timer; a begin that fails (resume budget expired)
    must not leave the socket up until the next successful turn."""

    async def _run():
        config = VoiceConfig(realtime={"uplink": "vad", "idleParkS": 0.05}, vad=VAD_CFG)
        inner = FakeInner()
        vad = ScriptVad([True] * 3 + [False] * 6)
        gate = GatedUplink(
            inner, config=config, sink=AudioSink(NullPlayback(), mode="stream"), vad=vad,
            capture_rate=RATE, uplink_rate=RATE, open_mic=False,
        )
        await gate.start(instructions=None, tools=[], on_event=_recorder())
        await asyncio.sleep(0.1)
        assert inner.calls == [("park",)]
        inner.calls.clear()
        inner.fail_begin = True
        await feed(gate, 9)  # onset (fails) ... close
        assert gate._metrics.snapshot()["counters"]["gate_activity_failed"] == 1
        await asyncio.sleep(0.1)
        assert inner.calls == [("park",)]
        await gate.close()

    asyncio.run(_run())


def test_a_hit_under_an_open_activity_keeps_the_engaged_window():
    """"hey nanobot ... hey nanobot" mid-upload: the turn owns attention until IDLE, so
    a reply longer than windowS still takes barge-in onsets."""

    async def _run():
        vad = ScriptVad([True] * 8 + [False] * 6)
        gate, inner, _, on_event = build(
            "wake", vad=vad, detector=ScriptWake({5}),
            wake={"mode": "gate", "phrases": ["hey nanobot"], "windowS": 15},
        )
        await gate.start(instructions=None, tools=[], on_event=on_event)
        await inner.emit(StateHint(VoiceState.THINKING))
        await inner.emit(StateHint(VoiceState.IDLE))  # window open (conversation)
        await feed(gate, 3)  # onset inside the window: engaged
        assert gate._active and gate._window_until == math.inf
        await feed(gate, 3, start=3)  # frame 5 carries a hit while _active
        assert gate._active and gate._window_until == math.inf
        assert gate._metrics.snapshot()["counters"]["wake_hit"] == 1
        await gate.close()

    asyncio.run(_run())
