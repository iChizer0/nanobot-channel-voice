"""RealtimeBackend's pure surfaces: helpers + `_handle_event` on canned frames.

No network anywhere: `_handle_event` mutates state and emits normalized events,
and `_send` no-ops with no websocket.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from nanobot_channel_voice.aio import cancel_and_wait
from nanobot_channel_voice.audio.base import PlaybackStream
from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend import openai_realtime as rt
from nanobot_channel_voice.backend import transport
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import (
    NOTICE_BACKLOG,
    AbandonedResult,
    Error,
    ImageResult,
    InputTranscript,
    OutputAudio,
    ReceiptResult,
    StateHint,
    ToolCall,
    ToolDef,
    ToolImage,
    ToolsAbandoned,
    ToolStarted,
    TurnDone,
    UserSpeechStarted,
    VoiceState,
)
from nanobot_channel_voice.backend.profiles import PROFILES
from nanobot_channel_voice.config import VoiceConfig

# ---- pure helper functions --------------------------------------------------


def test_status_detail_is_defensive_about_shape():
    assert rt._status_detail({"status_details": "plain string"}) == "plain string"
    assert rt._status_detail({"status_details": {"error": {"message": "boom"}}}) == "boom"
    assert rt._status_detail({"status_details": {"error": "bare"}}) == "bare"
    assert rt._status_detail({"status_details": {"reason": "max_output_tokens"}}) == "max_output_tokens"
    assert rt._status_detail({"status_details": 42}) == ""
    assert rt._status_detail({}) == ""


def test_normalize_schema_reduces_unions_and_combinators():
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": ["string", "null"]},
            "b": {"anyOf": [{"const": 1}, {"type": "number"}]},
            "c": {"type": "array", "items": {"type": ["integer", "null"]}},
            "d": {"type": ["null"]},
        },
    }
    out = rt._normalize_schema(schema)
    assert out["properties"]["a"]["type"] == "string"
    assert out["properties"]["b"]["type"] == "number"  # first type-bearing branch
    assert out["properties"]["c"]["items"]["type"] == "integer"
    assert out["properties"]["d"]["type"] == "string"  # all-null fallback


def test_tool_to_wire_flattens_only_on_request():
    tool = ToolDef(name="t", description="d",
                   parameters={"type": "object",
                               "properties": {"x": {"type": ["string", "null"]}}})
    full = rt._tool_to_wire(tool)
    assert full["parameters"]["properties"]["x"]["type"] == ["string", "null"]
    flat = rt._tool_to_wire(tool, flatten=True)
    assert flat["parameters"]["properties"]["x"]["type"] == "string"
    assert flat["type"] == "function" and flat["name"] == "t"


def test_deep_merge_recurses_dicts_and_overwrites_leaves():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    rt._deep_merge(base, {"a": {"c": 9}, "e": 4})
    assert base == {"a": {"b": 1, "c": 9}, "d": 3, "e": 4}


def test_tooldef_from_nanobot_schema_tolerates_flat_and_nested():
    nested = {"type": "function", "function": {"name": "n", "description": "d",
                                               "parameters": {"type": "object"}}}
    flat = {"name": "n2"}
    assert ToolDef.from_nanobot_schema(nested).name == "n"
    t = ToolDef.from_nanobot_schema(flat)
    assert t.name == "n2" and t.parameters == {"type": "object", "properties": {}}


def test_clamp_tool_output_marker_semantics():
    from nanobot_channel_voice.backend.common import TOOL_OUTPUT_CHARS, clamp_tool_output

    assert clamp_tool_output("x" * 100_000, 0) == "x" * 100_000  # 0 = unlimited
    assert clamp_tool_output("short", 200) == "short"
    clamped = clamp_tool_output("y" * 500, 200)
    assert len(clamped) == 200 and "500 chars in all, the middle cut" in clamped
    assert len(clamp_tool_output("z" * 500, 10)) == 10  # marker itself longer than the cap
    # A failing command's verdict is at its end: the cut keeps the head AND the tail.
    run = "collected 90 items\n" + "test_x PASSED\n" * 800 + "STDERR:\nboom\n\nExit code: 1"
    cut = clamp_tool_output(run)
    assert len(cut) == TOOL_OUTPUT_CHARS
    assert cut.startswith("collected 90 items") and cut.endswith("boom\n\nExit code: 1")


# ---- _handle_event over canned server frames --------------------------------


def make_backend() -> tuple[rt.RealtimeBackend, list]:
    backend = rt.RealtimeBackend(
        VoiceConfig(), sink=AudioSink(NullPlayback(), mode="stream"),
        profile=PROFILES["openai"],
    )
    events: list = []

    async def on_event(e):
        events.append(e)

    backend._on_event = on_event
    return backend, events


def drive(frames: list[dict], *, after=None):
    async def _run():
        backend, events = make_backend()
        for f in frames:
            await backend._handle_event(f)
        if after is not None:
            await after(backend)
        await backend.close()
        return backend, events

    return asyncio.run(_run())


def b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def make_sending_backend(sink: AudioSink) -> tuple[rt.RealtimeBackend, list[dict]]:
    backend = rt.RealtimeBackend(VoiceConfig(), sink=sink, profile=PROFILES["openai"])
    sent: list[dict] = []

    async def record(payload):
        sent.append(payload)

    async def on_event(e):
        pass

    backend._send = record
    backend._on_event = on_event
    return backend, sent


async def publish_stream(sink: AudioSink, ms: int = 1000, rate: int = 24000) -> None:
    """Give the sink a live stream, so played_ms()/stream_generation are real."""
    await sink.start()
    sink.enqueue(OutputAudio(epoch=sink.epoch, pcm=b"\x00" * (rate * 2 * ms // 1000),
                             rate=rate))
    await sink.wait_idle()


def test_at_rest_only_idle_with_nothing_queued_or_owed():
    backend, _events = make_backend()
    assert backend.at_rest
    backend._notices.append("a reminder waits")
    assert not backend.at_rest  # a rebuild would drop it
    backend._notices.clear()
    backend._user_speaking = True
    assert not backend.at_rest
    backend._user_speaking = False
    backend._turn = VoiceState.THINKING
    assert not backend.at_rest


def hints(events) -> list[VoiceState]:
    return [e.state for e in events if isinstance(e, StateHint)]


def test_plain_turn_full_lifecycle():
    seen = {}

    async def wait_drain(backend):
        seen["ready"] = backend._ready.is_set()  # close() clears it later
        await backend._drain_task  # completed turn drains to IDLE

    backend, events = drive([
        {"type": "session.created"},
        {"type": "session.updated"},
        {"type": "input_audio_buffer.speech_started"},
        {"type": "input_audio_buffer.speech_stopped"},
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.output_audio.delta", "response_id": "r1", "delta": b64(b"\x01\x02")},
        {"type": "response.done", "response": {"id": "r1", "status": "completed"}},
    ], after=wait_drain)
    assert seen["ready"] is True
    assert any(isinstance(e, UserSpeechStarted) for e in events)
    audio = [e for e in events if isinstance(e, OutputAudio)]
    assert len(audio) == 1 and audio[0].pcm == b"\x01\x02" and audio[0].rate == 24000
    assert sum(isinstance(e, TurnDone) for e in events) == 1
    assert hints(events) == [
        VoiceState.CAPTURING, VoiceState.THINKING, VoiceState.SPEAKING, VoiceState.IDLE,
    ]


def test_double_barge_in_truncates_the_item_once():
    """A second barge-in before the next output_item.added must NOT re-truncate the
    same item: a truncate at audio_end_ms=0 wipes the model's memory of audio the
    user actually heard."""

    async def _run():
        backend, _ = make_backend()
        sent: list[dict] = []

        async def record(payload):
            sent.append(payload)

        backend._send = record
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        await backend._handle_event({
            "type": "response.output_item.added",
            "response_id": "r1",
            "item": {"type": "message", "id": "item-1"},
        })
        await publish_stream(backend._sink)  # the item's audio opens the stream it stamped
        await backend.barge_in(1500)
        await backend.barge_in(0)  # sink already flushed: played_ms restarted
        truncates = [p for p in sent if p["type"] == "conversation.item.truncate"]
        assert len(truncates) == 1
        assert truncates[0]["item_id"] == "item-1"
        assert truncates[0]["audio_end_ms"] == 1500
        await backend.close()

    asyncio.run(_run())


def test_truncate_base_restarts_with_a_fresh_stream():
    """Turn N's drain is still playing out when turn N+1 starts, so response.created
    parks the old handle: played_ms() still reads it, but the new item's audio opens a
    FRESH stream whose clock starts at 0. Basing on the parked stream truncated every
    such turn at audio_end_ms=0, wiping audio the user actually heard."""

    class _SlowDrain(PlaybackStream):
        def __init__(self):
            self.gate = asyncio.Event()

        async def write(self, pcm: bytes) -> None:
            await asyncio.sleep(0)

        async def drain(self) -> None:
            await self.gate.wait()  # a real device plays its tail out here

        async def kill(self) -> None:
            self.gate.set()

    class _SlowDrainPlayback(NullPlayback):
        def __init__(self):
            self.opened = 0

        async def open_stream(self, rate: int) -> PlaybackStream:
            self.opened += 1
            return _SlowDrain()

    async def _run():
        playback = _SlowDrainPlayback()
        sink = AudioSink(playback, mode="stream")
        backend, sent = make_sending_backend(sink)

        # Turn 1 plays 1 s, completes, and its drain parks inside stream.drain().
        await backend._handle_event(_created("r1"))
        await publish_stream(sink)
        await backend._handle_event(
            {"type": "response.done", "response": {"id": "r1", "status": "completed"}})
        await asyncio.sleep(0.05)

        # Turn 2: the cancelled drain parks turn 1's handle, still the one played_ms reads.
        await backend._handle_event(_created("r2"))
        await asyncio.sleep(0.05)
        stale = sink.played_ms()
        assert stale > 0  # the parked stream's clock, not turn 2's
        await backend._handle_event({
            "type": "response.output_item.added",
            "response_id": "r2",
            "item": {"type": "message", "id": "item-2"},
        })
        # Only the sink backlog, never the parked stream's elapsed clock.
        assert backend._item_base_played < stale

        sink.enqueue(OutputAudio(epoch=sink.epoch, pcm=b"\x00" * 48000, rate=24000))
        await sink.wait_idle()
        assert playback.opened == 2

        await backend.barge_in(await sink.flush())
        truncates = [p for p in sent if p["type"] == "conversation.item.truncate"]
        assert len(truncates) == 1 and truncates[0]["audio_end_ms"] > 0
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_stream_reopen_under_the_item_skips_the_truncate():
    """The device dying mid-item makes the sink reopen, restarting played_ms() at 0
    under a base measured on the old stream: the truncate must be skipped, not sent at
    0. Over-remembering is the safe direction."""

    class _Dies(PlaybackStream):
        def __init__(self):
            self.is_dead = False

        @property
        def dead(self) -> bool:
            return self.is_dead

        async def write(self, pcm: bytes) -> None:
            await asyncio.sleep(0)

        async def drain(self) -> None:
            pass

        async def kill(self) -> None:
            pass

    class _DyingPlayback(NullPlayback):
        def __init__(self):
            self.streams: list[_Dies] = []

        async def open_stream(self, rate: int) -> PlaybackStream:
            self.streams.append(_Dies())
            return self.streams[-1]

    async def _run():
        playback = _DyingPlayback()
        sink = AudioSink(playback, mode="stream")
        backend, sent = make_sending_backend(sink)

        await backend._handle_event(_created("r1"))
        await publish_stream(sink)
        await backend._handle_event({
            "type": "response.output_item.added",
            "response_id": "r1",
            "item": {"type": "message", "id": "item-1"},
        })
        gen = sink.stream_generation
        assert backend._item_gen == gen and backend._item_base_played > 0
        _audio_to(sink, backend)
        chunk = {"type": "response.output_audio.delta", "response_id": "r1",
                 "delta": b64(b"\x00" * 48000)}
        await backend._handle_event(chunk)  # the item's first audio, on this stream
        await sink.wait_idle()

        playback.streams[-1].is_dead = True  # device gone; the sink reopens on the next write
        await backend._handle_event(chunk)  # mid-item: only the first audio re-takes the base
        await sink.wait_idle()
        assert sink.stream_generation == gen + 1

        await backend.barge_in(await sink.flush())
        assert not [p for p in sent if p["type"] == "conversation.item.truncate"]
        assert backend._metrics.counters.get("truncate_skipped_stale_stream") == 1
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


class _GatedDrain(PlaybackStream):
    """A device playing its tail out after EOF until released."""

    def __init__(self):
        self.done = asyncio.Event()

    async def write(self, pcm: bytes) -> None:
        await asyncio.sleep(0)

    async def drain(self) -> None:
        await self.done.wait()

    async def kill(self) -> None:
        self.done.set()


class _GatedDrainPlayback(NullPlayback):
    def __init__(self):
        self.streams: list[_GatedDrain] = []

    async def open_stream(self, rate: int) -> PlaybackStream:
        self.streams.append(_GatedDrain())
        return self.streams[-1]


def _audio_to(sink: AudioSink, backend) -> None:
    async def on_event(e):  # the shell's part
        if isinstance(e, OutputAudio):
            sink.enqueue(e)

    backend._on_event = on_event


def test_an_item_added_over_an_ending_stream_is_based_on_its_own():
    """A cue's stream still plays out when the reply's item is added: the item plays on a
    fresh stream from 0, so that tail is no part of its base."""

    async def _run():
        playback = _GatedDrainPlayback()
        sink = AudioSink(playback, mode="stream")
        backend, sent = make_sending_backend(sink)
        _audio_to(sink, backend)
        await publish_stream(sink, ms=230)  # the receipt
        ending = asyncio.create_task(sink.drain_stream())
        await asyncio.sleep(0.05)
        await backend._handle_event(_created("r1"))
        await backend._handle_event({"type": "response.output_item.added", "response_id": "r1",
                                     "item": {"type": "message", "id": "item-1"}})
        assert backend._item_base_played == 0
        await backend._handle_event({"type": "response.output_audio.delta", "response_id": "r1",
                                     "delta": b64(b"\x00" * 48000)})
        await sink.wait_idle()
        await backend.barge_in(700)
        truncate = [p for p in sent if p["type"] == "conversation.item.truncate"]
        assert truncate and truncate[0]["audio_end_ms"] == 700
        playback.streams[0].done.set()
        await ending
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_stream_ended_before_the_items_audio_is_measured_again():
    """The stream the item was added over ends before its first audio (a cue ended it):
    the audio plays on a fresh stream, so the base is re-taken there, not left stale."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        backend, sent = make_sending_backend(sink)
        _audio_to(sink, backend)
        await publish_stream(sink, ms=500)  # the last reply's filler, the stream held open
        await backend._handle_event(_created("r2"))
        await backend._handle_event({"type": "response.output_item.added", "response_id": "r2",
                                     "item": {"type": "message", "id": "item-2"}})
        await sink.drain_stream()
        await backend._handle_event({"type": "response.output_audio.delta", "response_id": "r2",
                                     "delta": b64(b"\x00" * 48000)})
        await sink.wait_idle()
        await backend.barge_in(700)
        truncate = [p for p in sent if p["type"] == "conversation.item.truncate"]
        assert truncate and truncate[0]["audio_end_ms"] == 700
        assert not backend._metrics.counters.get("truncate_skipped_stale_stream")
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def _played_out_then_onset(
    *, next_item: bool = False, ended: str | None = "drain",
) -> list[dict]:
    """r1's item plays out whole on a stream a hold kept open (no _on_drained); a cue's drain,
    or a reaper, then ends it (``ended``; None leaves it open), and the user starts talking.
    ``next_item`` adds r2's item over that stream first, with none of its own audio."""

    async def _run():
        playback = _GatedDrainPlayback()
        sink = AudioSink(playback, mode="stream")
        backend, sent = make_sending_backend(sink)

        async def on_event(e):  # the shell's part: play, and flush at the onset
            if isinstance(e, OutputAudio):
                sink.enqueue(e)
            elif isinstance(e, UserSpeechStarted):
                await backend.barge_in(await sink.flush())

        backend._on_event = on_event
        await sink.start()
        await backend._handle_event(_created("r1"))
        await backend._handle_event({"type": "response.output_item.added", "response_id": "r1",
                                     "item": {"type": "message", "id": "item-1"}})
        await backend._handle_event({"type": "response.output_audio.delta", "response_id": "r1",
                                     "delta": b64(b"\x00" * 38400)})
        await sink.wait_idle()
        if next_item:
            await backend._handle_event({"type": "response.output_item.added",
                                         "response_id": "r1",
                                         "item": {"type": "message", "id": "item-2"}})
        if ended is not None:
            ending = asyncio.create_task(sink.drain_stream())
            await asyncio.sleep(0)
            if ended == "reaper":
                await cancel_and_wait(ending)  # parked: the reaper plays the tail out
            playback.streams[0].done.set()
            await asyncio.sleep(0.01)
            assert not sink.stream_open
        await backend._handle_event({"type": "input_audio_buffer.speech_started"})
        await backend.close()
        await sink.stop()
        return [p for p in sent if p["type"] == "conversation.item.truncate"]

    return asyncio.run(_run())


@pytest.mark.parametrize("ended", ["drain", "reaper"])
def test_an_item_that_played_out_whole_is_not_truncated(ended):
    """Its stream ended by playing out, so played_ms() reads 0: a truncate would wipe audio
    the user heard in full. Talked over while its stream still plays, it is cut there."""
    assert _played_out_then_onset(ended=ended) == []
    [cut] = _played_out_then_onset(ended=None)
    assert cut["item_id"] == "item-1" and cut["audio_end_ms"] > 0


def test_an_item_added_over_a_played_out_stream_is_still_truncated_at_zero():
    """None of its own audio played: the user heard none of it."""
    assert _played_out_then_onset(next_item=True) == [{
        "type": "conversation.item.truncate", "item_id": "item-2", "content_index": 0,
        "audio_end_ms": 0,
    }]


def test_a_congested_uplink_cannot_stall_a_control_frame(monkeypatch):
    """websockets' drain() waits forever past its write high-water mark, and barge_in
    sends from the rx loop: unbounded, one stuck audio append froze barge-in and every
    later server event behind the send lock."""
    monkeypatch.setattr(transport, "_SEND_TIMEOUT_S", 0.1)

    class _CongestedWs:
        def __init__(self):
            self.unblock = asyncio.Event()
            self.sent = 0

        async def send(self, data):
            self.sent += 1
            if self.sent == 1:
                await self.unblock.wait()  # TCP backpressure

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        backend, _ = make_sending_backend(sink)
        del backend._send  # the real _send: this test is about the wire path
        ws = _CongestedWs()
        backend._ws = ws
        backend._ready.set()
        backend._sender_task = asyncio.create_task(backend._sender_loop())

        await publish_stream(sink)  # a live stream, so barge_in reaches the wire
        await backend._handle_event(_created("r1"))
        await backend._handle_event({
            "type": "response.output_item.added",
            "response_id": "r1",
            "item": {"type": "message", "id": "item-1"},
        })

        await backend.push_audio(b"\x00" * 640)
        await asyncio.sleep(0.01)  # the sender is now parked inside ws.send
        assert ws.sent == 1

        await asyncio.wait_for(backend.barge_in(1500), timeout=1.0)  # must not hang
        assert ws.sent == 2  # the truncate goes out as soon as the append is abandoned

        ws.unblock.set()
        backend._closing = True
        await cancel_and_wait(backend._sender_task)
        backend._sender_task = None
        backend._ws = None
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_slow_send_is_committed_not_lost_so_tool_bookkeeping_runs(monkeypatch):
    """websockets writes the frame before its first await, so a send that outlives the
    budget still reaches the server: treating it as lost skipped the call's bookkeeping
    and _maybe_respond, and the turn died with the result delivered."""
    monkeypatch.setattr(transport, "_SEND_TIMEOUT_S", 0.05)

    class _StuckWs:
        def __init__(self):
            self.payloads: list[str] = []
            self.unblock = asyncio.Event()

        async def send(self, data):
            self.payloads.append(data)  # committed to the transport ...
            await self.unblock.wait()   # ... but drain() never returns

    async def _run():
        backend, _ = make_sending_backend(AudioSink(NullPlayback(), mode="stream"))
        del backend._send
        ws = _StuckWs()
        backend._ws = ws
        backend._ready.set()
        backend._active_response_id = "r1"
        backend._session_calls.add("c1")
        backend._call_to_response["c1"] = "r1"
        backend._tools_pending["r1"] = {"c1"}
        await asyncio.wait_for(backend.submit_tool_result("c1", "ok"), 2.0)
        assert "c1" not in backend._session_calls
        assert not backend._tools_pending.get("r1")
        assert any('"function_call_output"' in p for p in ws.payloads)
        ws.unblock.set()
        backend._ws = None
        await backend.close()

    asyncio.run(_run())


def test_response_cancel_names_the_response_on_the_ga_dialect():
    """A cancel delayed by a congested uplink lands on whatever is active THEN; unnamed,
    it would kill the successor response."""
    ga = rt.RealtimeBackend(
        VoiceConfig(), sink=AudioSink(NullPlayback(), mode="stream"), profile=PROFILES["openai"],
    )
    assert ga._cancel_frame("r1") == {"type": "response.cancel", "response_id": "r1"}
    beta = rt.RealtimeBackend(
        VoiceConfig(), sink=AudioSink(NullPlayback(), mode="stream"), profile=PROFILES["qwen"],
    )
    assert beta._cancel_frame("r1") == {"type": "response.cancel"}


def test_truncate_debits_backlog_the_sink_dropped():
    """The base counts queued backlog; audio the overflow valve later dropped never played,
    so the item started that much earlier than the base says."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        backend, sent = make_sending_backend(sink)
        await publish_stream(sink)
        await backend._handle_event(_created("r1"))
        await backend._handle_event({
            "type": "response.output_item.added",
            "response_id": "r1",
            "item": {"type": "message", "id": "item-1"},
        })
        base = backend._item_base_played
        sink._dropped_ms += 500.0  # the valve fired after the item was added
        await backend.barge_in(1500)
        truncate = [p for p in sent if p["type"] == "conversation.item.truncate"]
        assert truncate and truncate[0]["audio_end_ms"] == max(0, int(1500 - (base - 500)))
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_watchdog_settles_to_idle_even_when_recovery_raises():
    """The deadman is the last recovery: dying inside it strands a gated mic in
    SPEAKING, and the task exception would surface only at GC."""

    async def _run():
        cfg = VoiceConfig.model_validate({"realtime": {"turnTimeoutS": 0.01}})
        backend = rt.RealtimeBackend(
            cfg, sink=AudioSink(NullPlayback(), mode="stream"), profile=PROFILES["openai"],
        )
        hints_seen: list = []

        async def on_event(e):
            hints_seen.append(e)
            raise RuntimeError("dispatch blew up")  # the StateHint dispatch dies too

        backend._on_event = on_event
        backend._turn = VoiceState.SPEAKING
        backend._arm_watchdog()
        for _ in range(200):
            await asyncio.sleep(0.01)
            if backend._watchdog_task.done():
                break
        assert backend._watchdog_task.done()
        assert backend._watchdog_task.exception() is None  # not left for the GC to report
        assert backend._turn is VoiceState.IDLE
        assert [type(e).__name__ for e in hints_seen] == ["Error", "StateHint"]
        await backend.close()

    asyncio.run(_run())


def test_close_lets_the_callers_cancellation_through():
    """nanobot cancels channel.stop() from above; swallowing that CancelledError would
    let teardown run on as if nothing happened."""

    async def _run():
        backend, _ = make_sending_backend(AudioSink(NullPlayback(), mode="stream"))
        started = asyncio.Event()

        async def _park():
            started.set()
            await asyncio.Event().wait()

        backend._rx_task = asyncio.create_task(_park())
        await started.wait()

        closer = asyncio.create_task(backend.close())
        await asyncio.sleep(0)
        closer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closer
        assert backend._rx_task is None or backend._rx_task.cancelled()

    asyncio.run(_run())


def test_cancelled_response_drops_late_deltas_and_turndone():
    _, events = drive([
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.done", "response": {"id": "r1", "status": "cancelled"}},
        {"type": "response.output_audio.delta", "response_id": "r1", "delta": b64(b"\x01")},
    ])
    assert not any(isinstance(e, OutputAudio) for e in events)
    assert not any(isinstance(e, TurnDone) for e in events)


def test_ridless_straggler_fails_closed_after_turn_end():
    async def straggler(backend):
        # rid cleared at turn end: a delta naming NO rid must not play.
        await backend._handle_event(
            {"type": "response.output_audio.delta", "delta": b64(b"\x99")})

    _, events = drive([
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.done", "response": {"id": "r1", "status": "completed"}},
    ], after=straggler)
    assert not any(isinstance(e, OutputAudio) for e in events)


def test_response_adopted_when_created_never_arrived():
    _, events = drive([
        {"type": "session.created"},
        # First sight of this response is its audio (lazy response.created).
        {"type": "response.output_audio.delta", "response_id": "rX", "delta": b64(b"\x05")},
    ])
    audio = [e for e in events if isinstance(e, OutputAudio)]
    assert len(audio) == 1 and audio[0].pcm == b"\x05"


def test_beta_delta_name_is_also_matched():
    _, events = drive([
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.audio.delta", "response_id": "r1", "delta": b64(b"\x07")},
    ])
    assert any(isinstance(e, OutputAudio) for e in events)


def test_tool_call_flow_suppresses_turndone_until_results():
    async def submit(backend):
        await backend.submit_tool_result("c1", "ok")

    backend, events = drive([
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.output_item.added", "response_id": "r1",
         "item": {"type": "function_call", "call_id": "c1", "name": "read"}},
        {"type": "response.function_call_arguments.delta", "call_id": "c1", "delta": '{"p":'},
        {"type": "response.function_call_arguments.delta", "call_id": "c1", "delta": ' 1}'},
        {"type": "response.function_call_arguments.done", "call_id": "c1",
         "response_id": "r1", "name": "read"},
        {"type": "response.done", "response": {"id": "r1", "status": "completed"}},
    ], after=submit)
    started = [e for e in events if isinstance(e, ToolStarted)]
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert started and started[0].call_id == "c1"
    assert len(calls) == 1 and calls[0].arguments == '{"p": 1}'
    # A tool turn is >= 2 responses: no TurnDone on the triggering response.
    assert not any(isinstance(e, TurnDone) for e in events)
    assert not backend._tools_pending  # continuation bookkeeping cleaned up


def test_fn_done_alone_registers_the_obligation():
    # A dialect may emit ONLY arguments.done (no output_item.added).
    _, events = drive([
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "response.function_call_arguments.done", "call_id": "c9",
         "response_id": "r1", "name": "t", "arguments": "{}"},
    ])
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert len(calls) == 1 and calls[0].call_id == "c9" and calls[0].arguments == "{}"


def test_failed_response_still_ends_the_turn():
    _, events = drive([
        {"type": "response.created", "response": {"id": "r1"}},
        {"type": "error", "error": {"code": "server_error", "message": "boom"}},
        {"type": "response.done", "response": {"id": "r1", "status": "failed",
                                               "status_details": {}}},
    ])
    errors = [e for e in events if isinstance(e, Error)]
    # The pre-failure `error` frame detail is attached to the failed done.
    assert any("boom" in e.message for e in errors)
    assert sum(isinstance(e, TurnDone) for e in events) == 1


def test_benign_errors_are_swallowed_and_fatal_codes_flagged():
    _, events = drive([
        {"type": "error", "error": {"code": "response_cancel_not_active", "message": "x"}},
        {"type": "error", "error": {"code": "invalid_api_key", "message": "bad key"}},
    ])
    errors = [e for e in events if isinstance(e, Error)]
    assert len(errors) == 1 and errors[0].fatal is True


def test_submit_tool_result_for_unknown_call_is_dropped():
    """A call_id the CURRENT session never issued (lost across a reconnect) must
    not reach the wire: the new session rejects the unknown call_id with an error."""

    async def _run():
        backend, _ = make_backend()
        sent: list[dict] = []

        async def record(payload):
            sent.append(payload)

        backend._send = record  # _send no-ops without a websocket: record to see the guard
        await backend.submit_tool_result("ghost", "late result")
        assert sent == []
        await backend.close()

    asyncio.run(_run())


@pytest.mark.parametrize("key", ["openai", "xai", "qwen", "glm", "stepfun"])
def test_session_update_payload_shapes(key):
    async def _run():
        backend = rt.RealtimeBackend(
            VoiceConfig(), sink=AudioSink(NullPlayback(), mode="stream"),
            profile=PROFILES[key],
        )
        payload = backend._session_update_payload()
        await backend.close()
        return payload

    payload = asyncio.run(_run())
    assert payload["type"] == "session.update"
    session = payload["session"]
    if PROFILES[key].dialect == "ga":
        assert session["audio"]["input"]["format"]["rate"] == PROFILES[key].input_rate
    else:
        assert session["input_audio_format"] == PROFILES[key].input_format
        assert session["output_audio_format"] == PROFILES[key].output_format
    if key == "glm":
        assert session["beta_fields"] == {"chat_mode": "audio"}  # session_extras merged


def _session_of(key: str, **realtime) -> dict:
    async def _run():
        backend = rt.RealtimeBackend(
            VoiceConfig.model_validate({"realtime": realtime}),
            sink=AudioSink(NullPlayback(), mode="stream"),
            profile=PROFILES[key],
        )
        payload = backend._session_update_payload()
        await backend.close()
        return payload["session"]

    return asyncio.run(_run())


def test_interrupt_response_rides_the_wire_only_when_off():
    """``true`` is the server default and xAI documents no such field: the default
    config sends nothing undocumented to any GA vendor."""
    assert _session_of("openai")["audio"]["input"]["turn_detection"] == {"type": "server_vad"}
    td = _session_of("xai", interruptResponse=False)["audio"]["input"]["turn_detection"]
    assert td == {"type": "server_vad", "interrupt_response": False}


def test_xai_session_extensions_are_profile_gated():
    xai = _session_of("xai")
    assert xai["reasoning"] == {"effort": "none"}  # the plugin default: answer, don't deliberate
    assert xai["resumption"] == {"enabled": True}
    assert xai["audio"]["input"]["format"]["rate"] == 16000
    assert _session_of("xai", reasoningEffort="high")["reasoning"] == {"effort": "high"}
    openai = _session_of("openai")
    assert "reasoning" not in openai and "resumption" not in openai


def test_xai_conversation_id_rides_every_reconnect():
    """The id from ``conversation.created`` goes into the connect URL of every later
    socket (reconnect, un-park) and survives the per-session state reset; a profile
    without resumption ignores the event."""

    async def _run():
        def make(key):
            return rt.RealtimeBackend(
                VoiceConfig.model_validate({"realtime": {"apiKey": "k"}}),
                sink=AudioSink(NullPlayback(), mode="stream"), profile=PROFILES[key],
            )

        created = {"type": "conversation.created", "conversation": {"id": "conv/1"}}
        xai = make("xai")
        fresh, _ = xai._connect_args()
        assert "conversation_id" not in fresh
        await xai._handle_event(created)
        xai._reset_turn_state(reason="session_lost")  # what a reconnect or park does
        resumed, _ = xai._connect_args()
        assert resumed == fresh + "&conversation_id=conv%2F1"
        # Older than the vendor's cache (minus the margin): forgotten, never sent stale.
        assert xai._resumes_conversation()  # the provider holds the history: no replay
        xai._conversation_t -= PROFILES["xai"].resumption_ttl_s
        assert not xai._resumes_conversation()
        assert xai._connect_args()[0] == fresh and xai._conversation_id is None
        await xai.close()

        plain = make("openai")
        await plain._handle_event(created)
        assert "conversation_id" not in plain._connect_args()[0]
        await plain.close()

    asyncio.run(_run())


# ---- stop-command consume (transcript-gated) --------------------------------


def make_stop_backend(cfg: VoiceConfig | None = None):
    backend = rt.RealtimeBackend(
        cfg or VoiceConfig.model_validate(
            {"realtime": {"inputTranscriptionModel": "whisper-1"}}
        ),
        sink=AudioSink(NullPlayback(), mode="stream"),
        profile=PROFILES["openai"],
    )
    sent: list[dict] = []

    async def _send(frame):
        sent.append(frame)

    backend._send = _send
    events: list = []

    async def on_event(e):
        events.append(e)

    backend._on_event = on_event
    return backend, sent, events


def _created(rid: str) -> dict:
    return {"type": "response.created", "response": {"id": rid}}


def _stop_t(text: str = "stop") -> dict:
    return {"type": "conversation.item.input_audio_transcription.completed",
            "transcript": text}


def test_stop_transcript_cancels_the_live_ack_response():
    async def _case():
        b, sent, events = make_stop_backend()
        b._turn = VoiceState.SPEAKING  # "stop" said over a reply
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event(_created("r1"))  # the ack response is already live
        epoch = b._sink.epoch
        await b._handle_event(_stop_t())
        assert {"type": "response.cancel", "response_id": "r1"} in sent
        assert "r1" in b._cancelled_responses
        assert b._sink.epoch > epoch  # queued ack audio flushed
        assert b._turn is VoiceState.IDLE
        assert b._metrics.counters.get("barge_in_stop") == 1
        assert any(isinstance(e, InputTranscript) for e in events)  # still logged
        await b.close()

    asyncio.run(_case())


def test_stop_transcript_before_the_response_suppresses_it_at_birth():
    async def _case():
        b, sent, _ = make_stop_backend()
        b._turn = VoiceState.SPEAKING  # reply audible when the user spoke
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        assert b._onset_interrupting
        await b._handle_event(_stop_t())  # no response yet: window armed
        assert b._turn is VoiceState.IDLE
        await b._handle_event(_created("r2"))  # the ack arrives late...
        assert {"type": "response.cancel", "response_id": "r2"} in sent  # ...dies at birth
        assert "r2" in b._cancelled_responses
        assert b._turn is VoiceState.IDLE  # never THINKING
        await b.close()

    asyncio.run(_case())


def test_mixed_transcript_is_not_consumed():
    async def _case():
        b, sent, _ = make_stop_backend()
        b._turn = VoiceState.SPEAKING
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event(_stop_t("stop using celsius"))
        assert sent == []
        assert "barge_in_stop" not in b._metrics.counters
        await b.close()

    asyncio.run(_case())


def test_cold_stop_is_forwarded_not_consumed():
    async def _case():
        b, sent, _ = make_stop_backend()
        await b._handle_event({"type": "input_audio_buffer.speech_started"})  # IDLE onset
        await b._handle_event(_stop_t())
        assert sent == []  # nothing live, no grace: the model may answer contextually
        await b._handle_event(_created("r3"))
        assert b._turn is VoiceState.THINKING  # the response lives
        await b.close()

    asyncio.run(_case())


def test_a_cold_stop_is_forwarded_even_when_its_reply_is_born_first():
    """The usual order: the reply to "stop" is born before its transcript lands. That reply
    is the stop's own, not one it aims at: forwarded, it may answer "say stop to cancel"."""
    async def _case():
        b, sent, _ = make_stop_backend()
        await b._handle_event({"type": "input_audio_buffer.speech_started"})  # IDLE onset
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event({"type": "input_audio_buffer.committed", "item_id": "i-stop"})
        await b._handle_event(_created("r-stop"))
        await b._handle_event({**_stop_t(), "item_id": "i-stop"})
        assert sent == []
        assert "barge_in_stop" not in b._metrics.counters
        assert b._turn is VoiceState.THINKING  # the reply lives
        await b.close()

    asyncio.run(_case())


def test_a_cold_stop_is_forwarded_when_its_reply_is_adopted():
    """A dialect that skips response.created: the reply adopted at its first event is still
    the stop's own."""
    async def _case():
        b, sent, _ = make_stop_backend()
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event({"type": "input_audio_buffer.committed", "item_id": "i-stop"})
        await b._handle_event({"type": "response.audio_transcript.delta",
                               "response_id": "r-stop", "delta": "Okay"})
        assert b._active_response_id == "r-stop"
        await b._handle_event({**_stop_t(), "item_id": "i-stop"})
        assert sent == []
        await b.close()

    asyncio.run(_case())


def test_a_stop_said_before_the_reply_it_stops_is_born_still_consumes_it():
    """A question, then "stop" before its reply started: the reply born during the stop
    is the one the user meant, so the stop consumes it."""
    async def _case():
        b, sent, _ = make_stop_backend()
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event({"type": "input_audio_buffer.committed", "item_id": "i-ask"})
        await b._handle_event({"type": "input_audio_buffer.speech_started"})  # still CAPTURING
        await b._handle_event(_created("r-ask"))
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event({"type": "input_audio_buffer.committed", "item_id": "i-stop"})
        await b._handle_event({**_stop_t(), "item_id": "i-stop"})
        assert {"type": "response.cancel", "response_id": "r-ask"} in sent
        await b.close()

    asyncio.run(_case())


def test_grace_consumes_the_double_tap():
    async def _case():
        b, sent, _ = make_stop_backend()
        b._turn = VoiceState.SPEAKING
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event(_created("r1"))
        await b._handle_event(_stop_t())  # consumed: r1 cancelled
        await b._handle_event(
            {"type": "response.done", "response": {"id": "r1", "status": "cancelled"}}
        )
        await b._handle_event({"type": "input_audio_buffer.speech_started"})  # cold onset
        await b._handle_event(_stop_t())  # double-tap, inside the grace
        assert b._metrics.counters.get("barge_in_stop") == 2
        await b._handle_event(_created("r2"))  # its response dies at birth
        assert "r2" in b._cancelled_responses
        await b.close()

    asyncio.run(_case())


def test_new_speech_clears_the_suppress_window():
    async def _case():
        b, sent, _ = make_stop_backend()
        b._turn = VoiceState.SPEAKING
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event(_stop_t())  # no active: window armed
        await b._handle_event({"type": "input_audio_buffer.speech_started"})  # new intent
        await b._handle_event(_created("r9"))
        assert sent == []  # not suppressed
        assert b._turn is VoiceState.THINKING
        await b.close()

    asyncio.run(_case())


def _barge_in_shell(b, events: list) -> None:
    """``on_event`` does what VoiceShell._cloud_barge_in does: flush, then ``barge_in``."""

    async def on_event(e):
        events.append(e)
        if isinstance(e, UserSpeechStarted):
            await b.barge_in(await b._sink.flush())

    b._on_event = on_event


def test_a_late_stop_transcript_leaves_the_next_utterance_alone():
    """The stop's transcript lands after the user started the next utterance, whose onset
    owns the state: the stop must not settle it IDLE, nor arm the suppress window that
    would kill its answer at birth."""
    async def _case():
        b, sent, events = make_stop_backend()
        _barge_in_shell(b, events)
        await b._handle_event(_created("r0"))  # a reply is live
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event({"type": "input_audio_buffer.committed", "item_id": "i-stop"})
        await b._handle_event(
            {"type": "response.done", "response": {"id": "r0", "status": "cancelled"}}
        )
        await b._handle_event(_created("r-stop"))  # the server answers "stop"
        await b._handle_event({"type": "input_audio_buffer.speech_started"})  # "what time"
        await b._handle_event(
            {"type": "response.done", "response": {"id": "r-stop", "status": "cancelled"}}
        )
        await b._handle_event({**_stop_t(), "item_id": "i-stop"})
        assert b._turn is VoiceState.CAPTURING
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event({"type": "input_audio_buffer.committed", "item_id": "i-next"})
        await b._handle_event(_created("r-answer"))
        assert "r-answer" not in b._cancelled_responses
        assert b._turn is VoiceState.THINKING
        await b.close()

    asyncio.run(_case())


def test_a_stop_for_the_latest_utterance_is_still_consumed():
    async def _case():
        b, sent, events = make_stop_backend()
        _barge_in_shell(b, events)
        await b._handle_event(_created("r0"))
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event({"type": "input_audio_buffer.committed", "item_id": "i-stop"})
        await b._handle_event(_created("r-stop"))
        await b._handle_event({**_stop_t(), "item_id": "i-stop"})
        assert {"type": "response.cancel", "response_id": "r-stop"} in sent
        assert b._turn is VoiceState.IDLE
        await b.close()

    asyncio.run(_case())


async def _dispatched_wait(b) -> None:
    await b._handle_event(_created("r1"))
    await b._handle_event({"type": "response.function_call_arguments.done", "response_id": "r1",
                           "call_id": "c1", "name": "ask_nanobot", "arguments": "{}"})
    await b._handle_event({"type": "response.done",
                           "response": {"id": "r1", "status": "completed"}})


def test_a_consumed_stop_abandons_the_pending_tool_work():
    """"stop" said into a delegation's wait ends the work, not only the reply: the channel
    is told (ToolsAbandoned), and the answer, whenever it lands, resumes nothing."""
    async def _case():
        b, sent, events = make_stop_backend()
        await _dispatched_wait(b)
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event({"type": "input_audio_buffer.committed", "item_id": "i-stop"})
        await b._handle_event(_created("r-stop"))
        await b._handle_event({**_stop_t(), "item_id": "i-stop"})
        assert sum(isinstance(e, ToolsAbandoned) for e in events) == 1
        assert b._turn is VoiceState.IDLE
        sent.clear()
        await b.submit_tool_result("c1", "the answer")
        assert [p["type"] for p in sent] == ["conversation.item.create"]
        await b.close()

    asyncio.run(_case())


def test_the_models_cancel_ends_the_wait_it_leaves():
    """No transcript consumed the stop: the model called cancel_nanobot, and both answers
    come back abandoned, resuming nothing. The wait they held ends with the last one,
    whichever of the two lands last."""
    async def _case(receipt_first: bool):
        b, sent, _ = make_stop_backend(VoiceConfig())
        await _dispatched_wait(b)
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event(_created("r2"))
        await b._handle_event({"type": "response.function_call_arguments.done", "response_id": "r2",
                               "call_id": "c2", "name": "cancel_nanobot", "arguments": "{}"})
        await b._handle_event({"type": "response.done",
                               "response": {"id": "r2", "status": "completed"}})
        await b._drain_task
        assert b._turn is VoiceState.THINKING
        sent.clear()
        answers = [("c1", AbandonedResult("(stopped by the user)")),
                   ("c2", ReceiptResult("(stopped)"))]
        if receipt_first:
            answers.reverse()
        await b.submit_tool_result(*answers[0])
        await asyncio.sleep(0.01)
        assert b._turn is VoiceState.THINKING  # the other answer is still owed
        await b.submit_tool_result(*answers[1])
        await b._drain_task
        assert b._turn is VoiceState.IDLE
        assert [p["type"] for p in sent] == ["conversation.item.create"] * 2
        await b.close()

    asyncio.run(_case(receipt_first=False))
    asyncio.run(_case(receipt_first=True))


def test_an_abandoned_answer_leaves_a_live_or_unborn_reply_its_state():
    """A replaced request's answer lands while the newer one's reply is live, or its
    continuation is asked for but not born: that reply owns the state, not a settle."""
    async def _case():
        b, sent, _ = make_stop_backend(VoiceConfig())
        await _dispatched_wait(b)  # r1: the old request's call c1
        await b._handle_event(_created("r2"))  # the user asked anew
        await b.submit_tool_result("c1", AbandonedResult("(replaced by a newer request)"))
        await asyncio.sleep(0.01)
        assert b._turn is VoiceState.THINKING
        await b._handle_event({"type": "response.function_call_arguments.done", "response_id": "r2",
                               "call_id": "c2", "name": "ask_nanobot", "arguments": "{}"})
        await b._handle_event({"type": "response.done",
                               "response": {"id": "r2", "status": "completed"}})
        await b._handle_event(_created("r3"))  # and anew again
        await b._handle_event({"type": "response.function_call_arguments.done", "response_id": "r3",
                               "call_id": "c3", "name": "ask_nanobot", "arguments": "{}"})
        await b._handle_event({"type": "response.done",
                               "response": {"id": "r3", "status": "completed"}})
        await b.submit_tool_result("c3", "the answer")  # its continuation is asked for
        assert b._continuation_unborn
        await b.submit_tool_result("c2", AbandonedResult("(replaced by a newer request)"))
        await asyncio.sleep(0.01)
        assert b._turn is VoiceState.THINKING
        await b.close()

    asyncio.run(_case())


@pytest.mark.parametrize("cancel_answered_first", [True, False])
def test_a_request_asked_with_the_cancel_still_continues_its_turn(cancel_answered_first):
    """"Check London instead, forget Paris": one response calls ask_nanobot and
    cancel_nanobot. The cancel's answer is abandoned, London's is not: the response resumes
    once both are in, in either order."""
    async def _case():
        b, sent, _ = make_stop_backend(VoiceConfig())
        await _dispatched_wait(b)  # r1: Paris
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event(_created("r2"))
        for cid, name in (("c2", "ask_nanobot"), ("c3", "cancel_nanobot")):
            await b._handle_event({"type": "response.function_call_arguments.done",
                                   "response_id": "r2", "call_id": cid, "name": name,
                                   "arguments": "{}"})
        await b._handle_event({"type": "response.done",
                               "response": {"id": "r2", "status": "completed"}})
        sent.clear()
        await b.submit_tool_result("c1", AbandonedResult("(replaced by a newer request)"))
        answers = [("c3", ReceiptResult("(stopped)")), ("c2", "Rain in London.")]
        if not cancel_answered_first:
            answers.reverse()
        for cid, output in answers:
            await b.submit_tool_result(cid, output)
        assert [p["type"] for p in sent] == ["conversation.item.create"] * 3 + ["response.create"]
        await b.close()

    asyncio.run(_case())


def test_a_stop_after_one_answer_leaves_the_turn_quiet():
    """Two calls of one response: one answered, then the other stopped. The stop ends the
    response's turn, answer included, so nothing is asked for."""
    async def _case():
        b, sent, _ = make_stop_backend(VoiceConfig())
        await b._handle_event(_created("r1"))
        for cid in ("c1", "c2"):
            await b._handle_event({"type": "response.function_call_arguments.done",
                                   "response_id": "r1", "call_id": cid, "name": "ask_nanobot",
                                   "arguments": "{}"})
        await b._handle_event({"type": "response.done",
                               "response": {"id": "r1", "status": "completed"}})
        sent.clear()
        await b.submit_tool_result("c1", "Paris is sunny.")
        await b.submit_tool_result("c2", AbandonedResult("(stopped by the user)"))
        assert [p["type"] for p in sent] == ["conversation.item.create"] * 2
        await b.close()

    asyncio.run(_case())


def test_an_abandoned_answer_before_the_users_reply_keeps_its_latency_sample():
    """It lands after the user stopped speaking and before their reply exists: nothing to
    settle, and the reply's TTFA is still measured from the end of their speech."""
    async def _case():
        b, _, _ = make_stop_backend(VoiceConfig())
        await _dispatched_wait(b)
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b.submit_tool_result("c1", AbandonedResult("(stopped by the user)"))
        assert b._turn is VoiceState.CAPTURING
        await b._handle_event(_created("r2"))
        await b._handle_event({"type": "response.output_audio.delta", "response_id": "r2",
                               "delta": base64.b64encode(b"\x00\x00").decode()})
        assert b._metrics.snapshot()["latency_ms"]["ttfa_ms"]["n"] == 1
        await b.close()

    asyncio.run(_case())


def test_a_refused_create_takes_its_kill_at_birth_with_it():
    """An onset marks the unborn continuation to die at birth; when the server refuses that
    create instead, the mark must not kill the next response born, the user's own."""
    async def _case():
        b, sent, events = make_stop_backend()
        _barge_in_shell(b, events)
        await _dispatched_wait(b)
        await b.submit_tool_result("c1", "ok")
        assert b._continuation_unborn
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        assert b._kill_at_birth
        await b._handle_event({"type": "error", "error": {
            "code": "conversation_already_has_active_response", "message": "busy"}})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event(_created("r2"))
        assert "r2" not in b._cancelled_responses
        await b.close()

    asyncio.run(_case())


def test_without_transcription_model_the_matcher_is_inert():
    async def _case():
        b, sent, _ = make_stop_backend(VoiceConfig())
        assert b._stop_match is None
        await b._handle_event(_created("r1"))
        await b._handle_event(_stop_t())
        assert sent == []
        assert "r1" not in b._cancelled_responses
        await b.close()

    asyncio.run(_case())


def test_cloud_instructions_always_carry_the_stop_rule():
    from nanobot_channel_voice.channel import _STOP_RULE, _cloud_instructions

    for sup, tools in ((False, False), (False, True), (True, True)):
        assert _STOP_RULE in _cloud_instructions(None, supervisor=sup, has_tools=tools)
    assert _STOP_RULE in _cloud_instructions("custom persona", supervisor=False,
                                             has_tools=False)


def test_barge_in_latency_records_only_real_interrupts():
    async def _run():
        backend, _ = make_backend()
        sent: list[dict] = []

        async def record(payload):
            sent.append(payload)

        backend._send = record
        # Onset over a quiet session: nothing live, so nothing to measure.
        await backend._handle_event({"type": "input_audio_buffer.speech_started"})
        await backend.barge_in(0)
        assert "barge_in_ms.truncate" not in backend._metrics.snapshot()["latency_ms"]
        # Onset over a live response: one sample.
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        await backend._handle_event({"type": "input_audio_buffer.speech_started"})
        await backend.barge_in(0)
        lat = backend._metrics.snapshot()["latency_ms"]
        assert lat["barge_in_ms.truncate"]["n"] == 1
        await backend.close()

    asyncio.run(_run())


def test_session_lost_releases_the_metrics_anchor():
    async def _run():
        backend, _ = make_backend()
        await backend._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await backend._on_session_lost()
        backend._metrics.turn_first_audio()  # a reconnect-quirk response's first audio
        snap = backend._metrics.snapshot()
        assert "ttfa_ms" not in snap["latency_ms"]  # the outage must not be a sample
        assert snap["counters"].get("ttfa_unanchored") == 1
        await backend.close()

    asyncio.run(_run())


def test_consumed_stop_voids_the_truncate_target():
    """The next utterance's barge-in must NOT truncate the partially-heard pre-stop
    item at audio_end_ms=0 (the stop's flush restarted played_ms, voiding the base)."""

    async def _run():
        backend, _ = make_backend()
        sent: list[dict] = []

        async def record(payload):
            sent.append(payload)

        backend._send = record
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        await backend._handle_event({
            "type": "response.output_item.added",
            "response_id": "r1",
            "item": {"type": "message", "id": "item-1"},
        })
        backend._item_base_played = 3000  # user heard ~3 s before saying "stop"
        await backend._consume_stop("stop")
        assert backend._audio_item_id is None
        # Next utterance: onset barge-in over the (now idle) session.
        await backend._handle_event({"type": "input_audio_buffer.speech_started"})
        await backend.barge_in(0)
        assert not any(p["type"] == "conversation.item.truncate" for p in sent)
        await backend.close()

    asyncio.run(_run())


def test_uplink_deadman_feed_stops_at_speech_stopped():
    """Silence frames after speech_stopped must not feed the turn deadman: a server
    that acks the speech and never creates a response has to be recoverable."""

    async def _run():
        backend, _ = make_backend()
        assert backend._user_speaking is False
        await backend._handle_event({"type": "input_audio_buffer.speech_started"})
        assert backend._user_speaking is True  # monologue frames may refresh
        await backend._handle_event({"type": "input_audio_buffer.speech_stopped"})
        assert backend._user_speaking is False  # the sender gate goes cold here
        # The watchdog armed at speech_started is still pending: with the feed cut,
        # wait_for_stall now sees a frozen _progress_t and can fire after
        # turn_timeout_s (not exercised in real time here).
        assert backend._watchdog_task is not None and not backend._watchdog_task.done()
        await backend.close()

    asyncio.run(_run())


def test_watchdog_recovers_a_capturing_wedge():
    """speech_stopped acked, response never created: with the uplink feed gated off,
    the watchdog armed at speech_started fires and settles the session to IDLE."""

    async def _run():
        cfg = VoiceConfig.model_validate({"realtime": {"turnTimeoutS": 0.05}})
        backend = rt.RealtimeBackend(
            cfg, sink=AudioSink(NullPlayback(), mode="stream"), profile=PROFILES["openai"],
        )
        events: list = []

        async def on_event(e):
            events.append(e)

        backend._on_event = on_event
        await backend._handle_event({"type": "input_audio_buffer.speech_started"})
        await backend._handle_event({"type": "input_audio_buffer.speech_stopped"})
        for _ in range(80):
            await asyncio.sleep(0.01)
            if any(isinstance(e, Error) for e in events):
                break
        timeouts = [e for e in events if isinstance(e, Error)]
        assert timeouts and not timeouts[0].fatal and "timed out" in timeouts[0].message
        assert hints(events)[-1] is VoiceState.IDLE
        await backend.close()

    asyncio.run(_run())


def test_barge_in_on_an_unborn_continuation_kills_it_at_birth():
    """Between the post-tool response.create and its response.created the active id still
    names the finished trigger: an onset in that window must kill the continuation once
    its id is known, or it plays over the user."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        backend, sent = make_sending_backend(sink)
        events: list = []

        async def on_event(e):  # what VoiceShell._cloud_barge_in does
            events.append(e)
            if isinstance(e, UserSpeechStarted):
                await backend.barge_in(await sink.flush())

        backend._on_event = on_event
        await backend._handle_event({"type": "session.created"})
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        await backend._handle_event({"type": "response.function_call_arguments.done",
                                     "response_id": "r1", "call_id": "c1", "name": "t",
                                     "arguments": "{}"})
        await backend._handle_event({"type": "response.done",
                                     "response": {"id": "r1", "status": "completed"}})
        await backend.submit_tool_result("c1", "ok")
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        await backend._handle_event({"type": "input_audio_buffer.speech_started"})
        assert not any(p["type"] == "response.cancel" for p in sent)  # r1 is done
        await backend._handle_event({"type": "response.created", "response": {"id": "r2"}})
        assert sent[-1] == {"type": "response.cancel", "response_id": "r2"}
        await backend._handle_event({"type": "response.output_audio.delta",
                                     "response_id": "r2", "delta": b64(b"\x01")})
        assert not any(isinstance(e, OutputAudio) for e in events)
        assert hints(events)[-1] is VoiceState.CAPTURING
        # The veto during the tool run itself is unchanged: the trigger is marked
        # (server VAD auto-cancels; nothing to send) and its result resumes nothing.
        await backend._handle_event({"type": "response.done",
                                     "response": {"id": "r2", "status": "cancelled"}})
        await backend._handle_event({"type": "response.created", "response": {"id": "r3"}})
        await backend._handle_event({"type": "response.function_call_arguments.done",
                                     "response_id": "r3", "call_id": "c2", "name": "t",
                                     "arguments": "{}"})
        await backend._handle_event({"type": "response.done",
                                     "response": {"id": "r3", "status": "completed"}})
        await backend._handle_event({"type": "input_audio_buffer.speech_started"})
        assert "r3" in backend._cancelled_responses and backend._kill_at_birth is False
        await backend.submit_tool_result("c2", "late")
        assert [p["type"] for p in sent[-2:]] == ["response.cancel", "conversation.item.create"]
        assert backend._continuation_unborn is False  # no continuation asked for
        await backend.close()

    asyncio.run(_run())


def test_rejected_hello_is_fatal_until_session_updated():
    """A rejected session.update yields only an error event, and the session would run
    on the server's defaults (no tools, its own audio format). The hello is the one
    frame on the wire before session.updated, so an error there is its rejection."""

    async def _run():
        backend, events = make_backend()
        backend._hello_payload()  # as _connect_and_run does right after connect
        await backend._handle_event({"type": "session.created"})
        await backend._handle_event({"type": "error", "error": {
            "type": "invalid_request_error", "code": "invalid_value",
            "message": "Invalid value: 'audio/pcm'"}})
        errors = [e for e in events if isinstance(e, Error)]
        assert len(errors) == 1 and errors[0].fatal
        assert "session.update rejected" in errors[0].message
        assert "Invalid value" in errors[0].message
        await backend._handle_event({"type": "session.updated"})
        await backend._handle_event({"type": "error", "error": {
            "code": "server_error", "message": "later"}})
        assert [e.fatal for e in events if isinstance(e, Error)] == [True, False]
        # After a working session, a pre-ready error on a reconnect is a blip: the
        # ladder owns it, never a fatal stop.
        backend._ready.clear()
        backend._hello_payload()
        await backend._handle_event({"type": "error", "error": {
            "code": "server_error", "message": "overloaded"}})
        assert [e.fatal for e in events if isinstance(e, Error)] == [True, False, False]
        assert "session.update rejected" in events[-1].message
        await backend.close()

    asyncio.run(_run())


def test_ready_opens_on_session_updated_not_session_created():
    """session.created precedes the hello's acceptance (it describes the server's
    defaults); frames must wait for our session.update to be applied."""

    async def _run():
        backend, _ = make_backend()
        await backend._handle_event({"type": "session.created"})
        assert not backend._ready.is_set()
        await backend._handle_event({"type": "session.updated"})
        assert backend._ready.is_set()
        await backend.close()

    asyncio.run(_run())


def test_hello_rejection_is_attributed_by_event_id():
    """The hello carries an event_id; an error naming ANOTHER client event before
    session.updated is not the hello's rejection (non-fatal), while an unattributed
    one, or one naming the hello, is."""

    async def _run():
        backend, events = make_backend()
        hello = backend._hello_payload()
        assert hello["event_id"].startswith("hello-")
        await backend._handle_event({"type": "error", "error": {
            "code": "invalid_value", "message": "other", "event_id": "evt-other"}})
        await backend._handle_event({"type": "error", "error": {
            "code": "invalid_value", "message": "ours", "event_id": hello["event_id"]}})
        assert [e.fatal for e in events if isinstance(e, Error)] == [False, True]
        await backend.close()

    asyncio.run(_run())


def test_a_tool_run_after_the_filler_holds_thinking_not_speaking():
    """A tool-bearing response completes with its call outstanding: the filler drains and
    the state settles to THINKING, as on Gemini. Left SPEAKING, a half-duplex mic stayed
    gated for the whole wait, so a delegation could be neither stopped nor steered."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent = make_sending_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await ev({"type": "response.created", "response": {"id": "r1"}})
        await ev({"type": "response.audio.delta", "response_id": "r1",
                  "delta": b64(b"\x00" * 3200)})
        assert backend._turn is VoiceState.SPEAKING
        await ev({"type": "response.output_item.added", "response_id": "r1",
                  "item": {"type": "function_call", "call_id": "c1", "name": "ask_nanobot"}})
        await ev({"type": "response.function_call_arguments.done", "response_id": "r1",
                  "call_id": "c1", "name": "ask_nanobot", "arguments": "{}"})
        await ev({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
        await backend._drain_task
        assert backend._turn is VoiceState.THINKING
        assert backend._active_response_id is None  # the call, not the response, is live
        await backend.submit_tool_result("c1", "answer")
        assert [p["type"] for p in sent[-2:]] == ["conversation.item.create", "response.create"]
        assert backend._turn is VoiceState.THINKING
        await ev({"type": "response.created", "response": {"id": "r2"}})
        await ev({"type": "response.audio.delta", "response_id": "r2",
                  "delta": b64(b"\x00" * 3200)})
        assert backend._turn is VoiceState.SPEAKING
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        await backend._drain_task
        assert backend._turn is VoiceState.IDLE
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


_FILLER = b"\x00" * (24000 * 2 * 300 // 1000)  # 300 ms at the profile's 24 kHz


async def _fast_tool_turn(backend, ev, *, result_before_done: bool = True) -> None:
    """A filler, its call answered at once (a direct-mode read), the continuation born
    while the filler still sounds."""
    await ev({"type": "session.updated"})
    await ev({"type": "response.created", "response": {"id": "r1"}})
    await ev({"type": "response.audio.delta", "response_id": "r1", "delta": b64(_FILLER)})
    await ev({"type": "response.function_call_arguments.done", "response_id": "r1",
              "call_id": "c1", "name": "read_file", "arguments": "{}"})
    if result_before_done:
        await backend.submit_tool_result("c1", "the file")
    await ev({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
    if not result_before_done:
        await backend.submit_tool_result("c1", "the file")
    await ev({"type": "response.created", "response": {"id": "r2"}})


def _hints(events) -> list[VoiceState]:
    return [e.state for e in events if isinstance(e, StateHint)]


def _playing_backend(sink: AudioSink):
    """The shell's part: output audio goes to the sink."""
    backend, sent, events = _tool_wait_backend(sink)

    async def on_event(e):
        events.append(e)
        if isinstance(e, OutputAudio):
            sink.enqueue(e)

    backend._on_event = on_event
    return backend, sent, events


@pytest.mark.parametrize("result_before_done", [True, False])
def test_a_continuation_born_under_its_filler_stays_speaking_until_it_played(
    result_before_done,
):
    """THINKING at its birth would open a half-duplex mic onto the rest of the filler,
    whose echo then cuts it off and cancels the answer (a fast tool: before or after the
    filler's response.done). The turn goes THINKING once the filler has played."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, _, events = _playing_backend(sink)
        await _fast_tool_turn(backend, backend._handle_event,
                              result_before_done=result_before_done)
        assert sink.backlog_ms() > 0
        assert backend._turn is VoiceState.SPEAKING
        await asyncio.wait_for(sink.wait_played(), 2.0)
        await asyncio.sleep(0.05)
        # The sink's clock says it played; the device buffer still sounds it for a moment.
        assert backend._turn is VoiceState.SPEAKING
        await asyncio.wait_for(backend._drain_task, 2.0)
        assert sink.backlog_ms() == 0
        assert _hints(events) == [VoiceState.THINKING, VoiceState.SPEAKING, VoiceState.THINKING]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_continuation_heard_before_its_filler_ends_keeps_the_turn_speaking():
    """Its audio lands on the same stream right behind the filler, with no THINKING
    between; and it owns the turn from there: a gap in it before its response.done is not
    the end of the filler's tail."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, _, events = _playing_backend(sink)
        ev = backend._handle_event
        await _fast_tool_turn(backend, ev)
        await asyncio.wait_for(sink.wait_idle(), 2.0)  # all written, its last lead sounding
        await asyncio.sleep(0.01)
        assert sink.backlog_ms() > 0
        generation = sink.stream_generation
        hold = backend._drain_task
        await ev({"type": "response.audio.delta", "response_id": "r2", "delta": b64(_FILLER)})
        await asyncio.wait_for(sink.wait_idle(), 2.0)
        assert sink.stream_generation == generation  # no device reopen at the seam
        await asyncio.wait_for(sink.wait_played(), 2.0)  # all played, the response still live
        await asyncio.wait({hold}, timeout=0.5)  # a hold that outlived the seam settles here
        assert backend._turn is VoiceState.SPEAKING
        assert _hints(events) == [VoiceState.THINKING, VoiceState.SPEAKING]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_continuation_announced_under_its_filler_is_still_truncated_when_cut():
    """Its message item lands while the filler plays; the filler then ends before its
    audio. The stream stayed open, so that item's truncate base holds: a barge-in into the
    answer still tells the model how much of it was heard."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _playing_backend(sink)
        ev = backend._handle_event
        await _fast_tool_turn(backend, ev)
        await ev({"type": "response.output_item.added", "response_id": "r2",
                  "item": {"type": "message", "id": "item-r2"}})
        await asyncio.wait_for(backend._drain_task, 2.0)
        assert backend._turn is VoiceState.THINKING
        await ev({"type": "response.audio.delta", "response_id": "r2", "item_id": "item-r2",
                  "delta": b64(_FILLER)})
        await asyncio.sleep(0.1)
        await backend.barge_in(await sink.flush())
        truncate = [p for p in sent if p["type"] == "conversation.item.truncate"]
        assert [p["item_id"] for p in truncate] == ["item-r2"]
        assert 0 < truncate[0]["audio_end_ms"] <= 300
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


async def _read_image_call(backend, output) -> list[dict]:
    """A direct-mode read_file call, answered after its response ended, with ``output``."""
    sent: list[dict] = []

    async def record(payload):
        sent.append(payload)

    backend._send = record
    ev = backend._handle_event
    await ev({"type": "session.updated"})
    await ev({"type": "response.created", "response": {"id": "r1"}})
    await ev({"type": "response.function_call_arguments.done", "response_id": "r1",
              "call_id": "c1", "name": "read_file", "arguments": "{}"})
    await ev({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
    await backend.submit_tool_result("c1", output)
    await backend.close()
    return sent


def test_a_tools_image_follows_its_output_as_a_labelled_user_message():
    """A call's output is text only: the picture a read returned rides a user message right
    behind it, before the continuation is asked for, labelled as the tool's."""
    backend, _ = make_backend()
    assert backend.image_types == frozenset({"image/png", "image/jpeg"})
    image = ToolImage("image/png", b"\x89PNG")
    sent = asyncio.run(_read_image_call(backend, ImageResult("(Image file: a.png)", (image,))))
    assert [p["type"] for p in sent] == ["conversation.item.create"] * 2 + ["response.create"]
    assert sent[0]["item"] == {
        "type": "function_call_output", "call_id": "c1", "output": "(Image file: a.png)",
    }
    message = sent[1]["item"]
    assert message["type"] == "message" and message["role"] == "user"
    assert message["content"][0]["type"] == "input_text"
    assert message["content"][1:] == [
        {"type": "input_image", "image_url": "data:image/png;base64,iVBORw=="},
    ]


def test_a_session_lost_during_the_output_send_gets_no_image():
    """The reconnect finished while the output went out: the image message would land in a
    session that never had the call."""
    backend, _ = make_backend()
    image = ToolImage("image/png", b"\x89PNG")

    async def _run():
        sent: list[dict] = []

        async def send(payload):
            sent.append(payload)
            if payload.get("item", {}).get("type") == "function_call_output":
                backend._reset_turn_state(reason="session_lost")

        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await ev({"type": "response.created", "response": {"id": "r1"}})
        await ev({"type": "response.function_call_arguments.done", "response_id": "r1",
                  "call_id": "c1", "name": "read_file", "arguments": "{}"})
        backend._send = send
        await backend.submit_tool_result("c1", ImageResult("(Image file: a.png)", (image,)))
        await backend.close()
        return sent

    assert [p["item"]["type"] for p in asyncio.run(_run())] == ["function_call_output"]


def test_which_profiles_show_a_tools_image():
    """GA takes it as a user message; Qwen only as frames of a turn the client commits, so
    not under server VAD."""
    sink = AudioSink(NullPlayback(), mode="stream")

    def takes(config):
        return {
            key for key, profile in PROFILES.items()
            if rt.RealtimeBackend(config, sink=sink, profile=profile).image_types
        }

    assert takes(VoiceConfig()) == {"openai", "azure"}
    assert takes(VoiceConfig(realtime={"uplink": "vad"})) == {"openai", "azure", "qwen"}


async def _tool_wait(ev, rid: str = "r1", cid: str = "c1") -> None:
    """A filler, then a dispatched ask_nanobot call, then the response's completion."""
    await ev({"type": "response.created", "response": {"id": rid}})
    await ev({"type": "response.output_item.added", "response_id": rid,
              "item": {"type": "message", "id": f"item-{rid}"}})
    await ev({"type": "response.audio.delta", "response_id": rid, "item_id": f"item-{rid}",
              "delta": b64(b"\x00" * 3200)})
    await ev({"type": "response.function_call_arguments.done", "response_id": rid,
              "call_id": cid, "name": "ask_nanobot", "arguments": "{}"})
    await ev({"type": "response.done", "response": {"id": rid, "status": "completed"}})


async def _aside(ev, rid: str) -> None:
    """The user speaks and the server answers it (server VAD), audibly."""
    await ev({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})
    await ev({"type": "input_audio_buffer.speech_stopped"})
    await ev({"type": "input_audio_buffer.committed", "item_id": f"u-{rid}"})
    await ev({"type": "response.created", "response": {"id": rid}})
    await ev({"type": "response.audio.delta", "response_id": rid, "delta": b64(b"\x00" * 320)})


def _tool_wait_backend(sink: AudioSink):
    backend, sent = make_sending_backend(sink)
    events: list = []

    async def on_event(e):
        events.append(e)

    backend._on_event = on_event
    return backend, sent, events


def test_talk_during_the_tool_wait_keeps_the_call_for_the_users_turn():
    """Speech into the THINKING wait: no truncate for the drained filler (the server refuses
    an end past its length), and the call stays owed. Its answer, landing mid-speech, never
    cuts in: the user's own response is created after it and sees it, so nothing re-asks."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, events = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await _tool_wait(ev)
        await backend._drain_task
        assert backend._turn is VoiceState.THINKING
        sent.clear()
        await ev({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})
        assert backend._turn is VoiceState.CAPTURING
        await backend.barge_in(100)
        assert not any(p["type"] == "conversation.item.truncate" for p in sent)
        await backend.submit_tool_result("c1", "the answer")
        assert [p["type"] for p in sent] == ["conversation.item.create"]
        await ev({"type": "input_audio_buffer.speech_stopped"})
        await ev({"type": "response.created", "response": {"id": "r2"}})
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        await backend._drain_task
        assert [p["type"] for p in sent] == ["conversation.item.create"]
        assert backend._turn is VoiceState.IDLE
        assert sum(isinstance(e, TurnDone) for e in events) == 1
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_an_answer_landing_under_the_reply_to_an_aside_is_asked_for_after_it():
    """The user's aside is answered while the delegation runs, and the answer lands under
    that reply: it never cuts in, the turn holds THINKING past the aside's end (no
    TurnDone, no IDLE), and a re-ask speaks it."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, events = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await _tool_wait(ev)
        await _aside(ev, "r2")
        sent.clear()
        await backend.submit_tool_result("c1", "the answer")
        assert [p["type"] for p in sent] == ["conversation.item.create"]
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        await backend._drain_task
        assert backend._turn is VoiceState.THINKING
        assert not any(isinstance(e, TurnDone) for e in events)
        await ev({"type": "response.created", "response": {"id": "r3"}})
        await ev({"type": "response.done", "response": {"id": "r3", "status": "completed"}})
        await backend._drain_task
        assert backend._turn is VoiceState.IDLE
        assert sum(isinstance(e, TurnDone) for e in events) == 1
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_the_reply_to_an_aside_ends_in_the_wait_not_the_turn():
    """The aside's reply ends before the delegation does: the turn goes on in THINKING,
    and the answer landing afterwards asks for its own continuation."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, events = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await _tool_wait(ev)
        await _aside(ev, "r2")
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        await backend._drain_task
        assert backend._turn is VoiceState.THINKING
        assert not any(isinstance(e, TurnDone) for e in events)
        sent.clear()
        await backend.submit_tool_result("c1", "the answer")
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_filler_cut_mid_stream_keeps_its_dispatched_call():
    """Talking over the filler cancels its response, not the work: the dispatched call keeps
    its obligation (its answer continues the turn once the user is heard out), while a
    call only announced, whose arguments never finished, is dropped."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, events = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await ev({"type": "response.created", "response": {"id": "r1"}})
        await ev({"type": "response.audio.delta", "response_id": "r1",
                  "delta": b64(b"\x00" * 3200)})
        await ev({"type": "response.function_call_arguments.done", "response_id": "r1",
                  "call_id": "c1", "name": "ask_nanobot", "arguments": "{}"})
        await ev({"type": "response.output_item.added", "response_id": "r1",
                  "item": {"type": "function_call", "call_id": "c2", "name": "ask_nanobot"}})
        await ev({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})
        await backend.barge_in(50)
        await ev({"type": "response.done", "response": {"id": "r1", "status": "cancelled"}})
        assert backend._tools_pending == {"r1": {"c1"}}
        assert [e.turn for e in events if isinstance(e, ToolCall)] == ["r1"]
        await ev({"type": "input_audio_buffer.speech_stopped"})
        await ev({"type": "response.created", "response": {"id": "r2"}})
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        await backend._drain_task
        assert backend._turn is VoiceState.THINKING
        sent.clear()
        await backend.submit_tool_result("c1", "the answer")
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_an_abandoned_result_answers_the_call_and_resumes_nothing():
    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await _tool_wait(ev)
        sent.clear()
        await backend.submit_tool_result("c1", AbandonedResult("(replaced by a newer request)"))
        assert [p["type"] for p in sent] == ["conversation.item.create"]
        assert backend._tools_pending == {} and not backend._answer_owed()
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_re_ask_talked_over_before_its_birth_dies_there():
    """The deferred answer's re-ask is a continuation like any other: an onset before its
    response.created kills it at birth instead of letting it play over the user."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, events = _tool_wait_backend(sink)
        _barge_in_shell(backend, events)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await _tool_wait(ev)
        await _aside(ev, "r2")
        await backend.submit_tool_result("c1", "the answer")
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        assert sent[-1] == {"type": "response.create"}
        await ev({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})
        await ev({"type": "response.created", "response": {"id": "r3"}})
        assert sent[-1] == {"type": "response.cancel", "response_id": "r3"}
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_an_auto_continuing_dialect_is_never_re_asked():
    """Where the server resumes on the function_call_output itself, a deferred answer is
    not asked for again: that would be a second continuation."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        backend._needs_response_create_after_tools = False
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await _tool_wait(ev)
        await _aside(ev, "r2")
        sent.clear()
        await backend.submit_tool_result("c1", "the answer")
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        assert [p["type"] for p in sent] == ["conversation.item.create"]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_failed_response_keeps_the_call_it_dispatched():
    """The response that asked failed after dispatching: the call still owes its answer, so
    the turn waits in THINKING (no TurnDone) and the answer continues it."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, events = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await ev({"type": "response.created", "response": {"id": "r1"}})
        await ev({"type": "response.function_call_arguments.done", "response_id": "r1",
                  "call_id": "c1", "name": "ask_nanobot", "arguments": "{}"})
        await ev({"type": "response.done", "response": {"id": "r1", "status": "failed"}})
        await backend._drain_task
        assert backend._turn is VoiceState.THINKING
        assert not any(isinstance(e, TurnDone) for e in events)
        sent.clear()
        await backend.submit_tool_result("c1", "the answer")
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_the_deadman_gives_up_a_stalled_response_not_its_call():
    """The response that asked stalls with no done: the deadman cancels the response, and
    the dispatched call's answer still continues the turn from THINKING."""

    async def _run():
        cfg = VoiceConfig.model_validate({"realtime": {"turnTimeoutS": 0.05}})
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend = rt.RealtimeBackend(cfg, sink=sink, profile=PROFILES["openai"])
        sent: list[dict] = []
        events: list = []

        async def record(payload):
            sent.append(payload)

        async def on_event(e):
            events.append(e)

        backend._send = record
        backend._on_event = on_event
        ev = backend._handle_event
        await ev({"type": "response.created", "response": {"id": "r1"}})
        await ev({"type": "response.function_call_arguments.done", "response_id": "r1",
                  "call_id": "c1", "name": "ask_nanobot", "arguments": "{}"})
        for _ in range(80):
            await asyncio.sleep(0.01)
            if backend._watchdog_task.done():
                break
        assert hints(events)[-1] is VoiceState.THINKING
        sent.clear()
        await backend.submit_tool_result("c1", "the answer")
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


async def _settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


def _notice_frames(sent: list[dict]) -> list[str]:
    return [p["item"]["content"][0]["text"] for p in sent
            if p["type"] == "conversation.item.create" and p["item"]["type"] == "message"]


def test_a_notice_is_voiced_once_the_session_is_quiet():
    """Held while a reply plays and while the user speaks; voiced as its own response at
    the next quiet settle, a marked user text item plus a create, one per settle."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await ev({"type": "response.created", "response": {"id": "r1"}})
        await backend.announce("Reminder: check the oven.")
        await backend.announce("Dinner is ready.")
        await _settle()
        assert _notice_frames(sent) == []
        await ev({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})
        await ev({"type": "response.done", "response": {"id": "r1", "status": "cancelled"}})
        await _settle()
        assert _notice_frames(sent) == []  # the user holds the floor
        await ev({"type": "input_audio_buffer.speech_stopped"})
        await ev({"type": "response.created", "response": {"id": "r2"}})
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        await backend._drain_task
        await _settle()
        assert _notice_frames(sent) == ["[notice] Reminder: check the oven."]
        assert sent[-1] == {"type": "response.create"} and backend._continuation_unborn
        await ev({"type": "response.created", "response": {"id": "r3"}})
        await ev({"type": "response.done", "response": {"id": "r3", "status": "completed"}})
        await backend._drain_task
        await _settle()
        assert _notice_frames(sent)[-1] == "[notice] Dinner is ready."
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_notice_whose_request_fails_still_ends_its_turn():
    """A create refused with an error (a rate limit) is ended by the deadman alone: THINKING
    from the send, the notice still ends in a settle, which re-arms a gated uplink's park."""

    async def _run():
        cfg = VoiceConfig.model_validate({"realtime": {"turnTimeoutS": 0.05}})
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend = rt.RealtimeBackend(cfg, sink=sink, profile=PROFILES["openai"])
        sent: list[dict] = []
        events: list = []

        async def record(payload):
            sent.append(payload)

        async def on_event(e):
            events.append(e)

        backend._send = record
        backend._on_event = on_event
        await backend._handle_event({"type": "session.updated"})
        await backend.announce("Dinner is ready.")
        await backend._notice_task
        assert _notice_frames(sent) == ["[notice] Dinner is ready."]
        await backend._handle_event({"type": "error", "error": {
            "code": "rate_limit_exceeded", "message": "slow down"}})
        for _ in range(80):
            await asyncio.sleep(0.01)
            if backend._watchdog_task.done():
                break
        assert hints(events) == [VoiceState.THINKING, VoiceState.IDLE]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_notice_waits_for_a_reply_the_user_is_talking_under():
    """Server VAD: a reply born while the user speaks settles IDLE under their speech when it
    ends; the notice still waits for them to finish."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await ev({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})
        await ev({"type": "response.created", "response": {"id": "r1"}})
        await ev({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
        await backend._drain_task
        await backend.announce("Dinner is ready.")
        await _settle()
        assert _notice_frames(sent) == []
        await ev({"type": "input_audio_buffer.speech_stopped"})
        await ev({"type": "response.created", "response": {"id": "r2"}})
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        await backend._drain_task
        await _settle()
        assert _notice_frames(sent) == ["[notice] Dinner is ready."]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_notice_waits_for_the_reply_to_an_aside_during_a_tool_wait():
    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await _tool_wait(ev)
        await backend._drain_task
        await ev({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})
        await ev({"type": "input_audio_buffer.speech_stopped"})
        await ev({"type": "response.created", "response": {"id": "r2"}})  # not audible yet
        assert backend._turn is VoiceState.THINKING and backend._waiting_on_tools()
        await backend.announce("Dinner is ready.")
        await _settle()
        assert _notice_frames(sent) == []
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        await backend._drain_task
        await _settle()
        assert _notice_frames(sent) == ["[notice] Dinner is ready."]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_second_notice_waits_for_the_first_ones_reply():
    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await backend.announce("Reminder: check the oven.")
        await _settle()
        await backend.announce("Dinner is ready.")
        await _settle()
        assert _notice_frames(sent) == ["[notice] Reminder: check the oven."]
        await ev({"type": "response.created", "response": {"id": "r1"}})
        await ev({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
        await backend._drain_task
        await _settle()
        assert _notice_frames(sent)[-1] == "[notice] Dinner is ready."
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_second_notice_in_a_tool_wait_waits_for_the_first_ones_reply():
    """A tool wait is THINKING whether a notice went out or not, so only its unborn reply
    keeps a second notice from asking for a second response before the first is born."""
    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await _tool_wait(ev)
        await backend._drain_task
        await backend.announce("Reminder: check the oven.")
        await _settle()
        await backend.announce("Dinner is ready.")
        await _settle()
        assert _notice_frames(sent) == ["[notice] Reminder: check the oven."]
        await ev({"type": "response.created", "response": {"id": "r2"}})
        await ev({"type": "response.done", "response": {"id": "r2", "status": "completed"}})
        await backend._drain_task
        await _settle()
        assert _notice_frames(sent)[-1] == "[notice] Dinner is ready."
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_notice_goes_out_during_a_tool_wait():
    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await _tool_wait(ev)
        await backend._drain_task
        await backend.announce("Dinner is ready.")
        await _settle()
        assert _notice_frames(sent) == ["[notice] Dinner is ready."]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_notice_waits_for_a_session_and_outlives_a_lost_one():
    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        await backend.announce("Dinner is ready.")
        await _settle()
        assert _notice_frames(sent) == []  # no session yet
        await backend._on_session_lost()
        await _settle()
        await backend._handle_event({"type": "session.updated"})
        await _settle()
        assert _notice_frames(sent) == ["[notice] Dinner is ready."]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_notices_waiting_out_an_outage_keep_only_the_newest():
    """They outlive a lost session, so an outage must not bank every message for a
    read-out at reconnect: past the cap the oldest goes, counted."""
    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        for i in range(NOTICE_BACKLOG + 2):
            await backend.announce(f"Message {i}.")
        await _settle()
        assert list(backend._notices) == [f"Message {i}." for i in range(2, NOTICE_BACKLOG + 2)]
        assert backend._metrics.counters.get("notice_dropped") == 2
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_notice_that_fails_to_go_out_yields_to_a_refilled_queue():
    """A failed send puts the notice back first, as the oldest; but the bus kept delivering
    while the send hung, and past the cap the oldest goes: this one, counted, not the
    newest arrival."""
    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        backend._ready.set()

        async def hung_send(text: str) -> None:
            for i in range(NOTICE_BACKLOG):
                await backend.announce(f"Late {i}.")
            raise RuntimeError("socket closed")

        backend._voice_notice = hung_send
        await backend.announce("First.")
        await _settle()
        assert list(backend._notices) == [f"Late {i}." for i in range(NOTICE_BACKLOG)]
        assert backend._metrics.counters.get("notice_dropped") == 1
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_notice_talked_over_while_it_is_sent_dies_at_birth():
    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, events = _tool_wait_backend(sink)
        _barge_in_shell(backend, events)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        record = backend._send

        async def racing_send(payload):
            await record(payload)
            if payload["type"] == "conversation.item.create":
                await ev({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})

        backend._send = racing_send
        await backend.announce("Dinner is ready.")
        await _settle()
        await ev({"type": "response.created", "response": {"id": "r1"}})
        assert sent[-1] == {"type": "response.cancel", "response_id": "r1"}
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_notice_whose_send_fails_waits_for_the_next_session():
    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, _ = _tool_wait_backend(sink)
        ev = backend._handle_event
        record = backend._send
        broken = [True]

        async def failing_send(payload):
            if broken[0]:
                raise ConnectionError("socket closed")
            await record(payload)

        backend._send = failing_send
        await ev({"type": "session.updated"})
        await backend.announce("Dinner is ready.")
        await _settle()
        assert list(backend._notices) == ["Dinner is ready."]
        broken[0] = False
        await backend._on_session_lost()
        await ev({"type": "session.updated"})
        await _settle()
        assert _notice_frames(sent) == ["[notice] Dinner is ready."]
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_create_is_unborn_before_its_frame_goes_out():
    """The user's onset can be handled while response.create is still being sent: the
    continuation must already count as unborn then, or it is born over the user."""

    async def _run():
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend, sent, events = _tool_wait_backend(sink)
        _barge_in_shell(backend, events)
        ev = backend._handle_event
        await ev({"type": "session.updated"})
        await ev({"type": "response.created", "response": {"id": "r1"}})
        await ev({"type": "response.function_call_arguments.done", "response_id": "r1",
                  "call_id": "c1", "name": "ask_nanobot", "arguments": "{}"})
        await ev({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
        record = backend._send

        async def racing_send(payload):
            await record(payload)
            if payload == {"type": "response.create"}:
                await ev({"type": "input_audio_buffer.speech_started", "audio_start_ms": 0})

        backend._send = racing_send
        await backend.submit_tool_result("c1", "the answer")
        await ev({"type": "response.created", "response": {"id": "r2"}})
        assert sent[-1] == {"type": "response.cancel", "response_id": "r2"}
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_the_deadman_during_a_tool_wait_settles_thinking():
    """The watchdog armed at an aside's onset fires with the delegation still running: the
    wait goes on in THINKING (IDLE would let a gated uplink park and lose the answer)."""

    async def _run():
        cfg = VoiceConfig.model_validate({"realtime": {"turnTimeoutS": 0.05}})
        sink = AudioSink(NullPlayback(), mode="stream")
        await sink.start()
        backend = rt.RealtimeBackend(cfg, sink=sink, profile=PROFILES["openai"])
        events: list = []

        async def on_event(e):
            events.append(e)

        backend._on_event = on_event
        ev = backend._handle_event
        await ev({"type": "response.created", "response": {"id": "r1"}})
        await ev({"type": "response.function_call_arguments.done", "response_id": "r1",
                  "call_id": "c1", "name": "ask_nanobot", "arguments": "{}"})
        await ev({"type": "response.done", "response": {"id": "r1", "status": "completed"}})
        await ev({"type": "input_audio_buffer.speech_started"})
        await ev({"type": "input_audio_buffer.speech_stopped"})
        for _ in range(80):
            await asyncio.sleep(0.01)
            if backend._watchdog_task is not None and backend._watchdog_task.done():
                break
        assert hints(events)[-1] is VoiceState.THINKING
        await backend.close()
        await sink.stop()

    asyncio.run(_run())


def test_a_late_transcript_says_how_many_onsets_came_after_it():
    """The spoken history places a transcript by its utterance's onset: one that lands
    after the next onset began counts it."""
    async def _case():
        b, _, events = make_stop_backend()
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event({"type": "input_audio_buffer.committed", "item_id": "u1"})
        await b._handle_event({"type": "input_audio_buffer.speech_started"})
        await b._handle_event({"type": "input_audio_buffer.speech_stopped"})
        await b._handle_event({"type": "input_audio_buffer.committed", "item_id": "u2"})
        for item, text in (("u1", "first"), ("u2", "second")):
            await b._handle_event({
                "type": "conversation.item.input_audio_transcription.completed",
                "item_id": item, "transcript": text,
            })
        got = [(e.text, e.later_onsets) for e in events if isinstance(e, InputTranscript)]
        assert got == [("first", 1), ("second", 0)]
        await b.close()

    asyncio.run(_case())
