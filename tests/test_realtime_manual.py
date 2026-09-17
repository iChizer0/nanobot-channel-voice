"""RealtimeBackend under the gated uplink: manual turns, park/resume. No network."""

from __future__ import annotations

import asyncio
import base64

import pytest

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend import openai_realtime as rt
from nanobot_channel_voice.backend import transport
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import (
    Error,
    ManualTurnBackend,
    StateHint,
    UserSpeechStarted,
    VoiceState,
)
from nanobot_channel_voice.backend.profiles import PROFILES
from nanobot_channel_voice.config import VoiceConfig


def _manual_config(**realtime) -> VoiceConfig:
    # backend stays "local" so the engine validator (silero/openwakeword) is not in play:
    # the adapter reads only realtime.uplink.
    return VoiceConfig(realtime={"uplink": "vad", **realtime})


def make_backend(profile="openai", config: VoiceConfig | None = None):
    backend = rt.RealtimeBackend(
        config or _manual_config(),
        sink=AudioSink(NullPlayback(), mode="stream"),
        profile=PROFILES[profile],
    )
    sent: list[dict] = []
    events: list = []

    async def record(payload):
        sent.append(payload)

    async def on_event(e):
        events.append(e)

    backend._send = record
    backend._on_event = on_event
    return backend, sent, events


def b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def run(coro):
    return asyncio.run(coro)


def test_adapter_satisfies_the_manual_turn_protocol():
    backend, _, _ = make_backend()
    assert isinstance(backend, ManualTurnBackend)


def test_manual_ga_session_has_no_server_vad():
    backend, _, _ = make_backend()
    session = backend._session_update_payload()["session"]
    assert session["audio"]["input"]["turn_detection"] is None
    server, _, _ = make_backend(config=VoiceConfig())
    td = server._session_update_payload()["session"]["audio"]["input"]["turn_detection"]
    assert td["type"] == "server_vad"


def test_manual_beta_session_uses_the_profile_client_mode():
    glm, _, _ = make_backend("glm")
    assert glm._session_update_payload()["session"]["turn_detection"] == {"type": "client_vad"}
    qwen, _, _ = make_backend("qwen")
    assert qwen._session_update_payload()["session"]["turn_detection"] is None
    qwen_server, _, _ = make_backend("qwen", config=VoiceConfig())
    assert qwen_server._session_update_payload()["session"]["turn_detection"] == {
        "type": "server_vad"
    }


def test_begin_then_commit_drives_the_speech_transitions_and_the_wire():
    async def _run():
        backend, sent, events = make_backend()
        backend._ready.set()
        await backend.begin_activity()
        assert [e.state for e in events if isinstance(e, StateHint)] == [VoiceState.CAPTURING]
        assert any(isinstance(e, UserSpeechStarted) for e in events)
        assert backend._user_speaking is True
        assert backend._watchdog_task is not None and not backend._watchdog_task.done()
        await backend.end_activity()
        assert [p["type"] for p in sent] == ["input_audio_buffer.commit", "response.create"]
        assert backend._user_speaking is False
        await backend.close()

    run(_run())


def test_uncommitted_end_clears_the_buffer_and_settles_idle():
    async def _run():
        backend, sent, events = make_backend()
        backend._ready.set()
        await backend.begin_activity()
        await backend.end_activity(commit=False)
        assert [p["type"] for p in sent] == ["input_audio_buffer.clear"]
        assert [e.state for e in events if isinstance(e, StateHint)] == [
            VoiceState.CAPTURING, VoiceState.IDLE,
        ]
        task = backend._watchdog_task
        assert task is None or task.done() or task.cancelling()
        await backend.close()

    run(_run())


def test_commit_waits_for_queued_frames_first():
    """Audio rides the sender task; the commit must not overtake it."""

    async def _run():
        backend, sent, _ = make_backend()
        backend._ready.set()
        order: list[str] = []

        async def drain_one():
            pcm = await backend._send_q.get()
            order.append(f"audio:{pcm!r}")
            backend._send_q.task_done()

        async def record(payload):
            order.append(payload["type"])
            sent.append(payload)

        backend._send = record
        await backend.push_audio(b"\x01\x02")
        drainer = asyncio.create_task(drain_one())
        await backend.end_activity()
        await drainer
        assert order == ["audio:b'\\x01\\x02'", "input_audio_buffer.commit", "response.create"]
        await backend.close()

    run(_run())


def test_manual_barge_in_always_cancels_client_side():
    """No server VAD => no server auto-cancel, whatever interruptResponse says."""

    async def _run():
        backend, sent, _ = make_backend()
        backend._active_response_id = "r1"
        await backend.barge_in(0)
        assert [p["type"] for p in sent] == ["response.cancel"]
        assert "r1" in backend._cancelled_responses
        await backend.close()

    run(_run())


def test_server_speech_events_are_ignored_under_manual_turns():
    async def _run():
        backend, _, events = make_backend()
        await backend._handle_event({"type": "input_audio_buffer.speech_started"})
        await backend._handle_event({"type": "input_audio_buffer.speech_stopped"})
        assert events == []
        await backend.close()

    run(_run())


def _stub_connection(backend, *, ready: bool = True):
    """Replace the socket loop: readies the session (or not) and idles until cancelled."""
    connects = []

    async def fake_connect_and_run():
        connects.append(1)
        if ready:
            backend._ready.set()
            backend._ever_ready = True
        await asyncio.Event().wait()

    backend._connect_and_run = fake_connect_and_run
    return connects


def test_park_closes_the_loop_and_begin_resumes_it():
    async def _run():
        backend, sent, events = make_backend()
        connects = _stub_connection(backend)
        await backend.start(instructions="", tools=[], on_event=backend._on_event)
        await asyncio.sleep(0)
        assert backend._ready.is_set() and connects == [1]
        await backend.park()
        assert backend._parked and backend._rx_task is None and not backend._ready.is_set()
        await backend.park()  # idempotent
        await backend.begin_activity()
        assert not backend._parked and backend._ready.is_set() and connects == [1, 1]
        assert backend.metrics.snapshot()["counters"].get("park") == 1
        assert [e.state for e in events if isinstance(e, StateHint)][-1] == VoiceState.CAPTURING
        await backend.close()

    run(_run())


def test_resume_timeout_raises_so_the_gate_drops_the_utterance(monkeypatch):
    monkeypatch.setattr(transport, "_RESUME_TIMEOUT_S", 0.05)

    async def _run():
        backend, _, _ = make_backend()
        _stub_connection(backend, ready=False)
        backend._parked = True
        with pytest.raises(RuntimeError, match="resume"):
            await backend.begin_activity()
        await backend.close()

    run(_run())


def _failing_connection(backend):
    """Every connect attempt fails at once (network down)."""
    attempts = []

    async def fake_connect_and_run():
        attempts.append(1)
        raise OSError("connect refused")

    backend._connect_and_run = fake_connect_and_run
    return attempts


def test_reconnect_exhausted_under_a_gate_parks_instead_of_dying(monkeypatch):
    """A WiFi blip at summon time must not tear the channel down: the gated uplink
    gives the network up and the next utterance runs the ladder again."""
    monkeypatch.setattr(transport, "_BACKOFF", (0.0, 0.0))
    monkeypatch.setattr(transport, "_RESUME_TIMEOUT_S", 0.2)

    async def _run():
        backend, _, events = make_backend()
        attempts = _failing_connection(backend)
        backend._parked = True
        with pytest.raises(RuntimeError, match="resume"):
            await backend.begin_activity()
        await asyncio.sleep(0.05)
        assert backend._parked and len(attempts) == 3
        errors = [e for e in events if isinstance(e, Error)]
        assert errors and not any(e.fatal for e in errors)
        assert "parked" in errors[-1].message
        assert backend.metrics.snapshot()["counters"].get("reconnect_parked") == 1
        # The network is back: the next summon resumes through a fresh ladder.
        connects = _stub_connection(backend)
        await backend.begin_activity()
        assert not backend._parked and backend._ready.is_set() and connects == [1]
        await backend.close()

    run(_run())


def test_reconnect_exhausted_under_server_vad_stays_fatal(monkeypatch):
    """No gate drives reconnects there: exhaustion is the end of the session."""
    monkeypatch.setattr(transport, "_BACKOFF", (0.0, 0.0))

    async def _run():
        backend, _, events = make_backend(config=VoiceConfig())
        _failing_connection(backend)
        await backend.start(instructions="", tools=[], on_event=backend._on_event)
        await asyncio.sleep(0.05)
        assert any(isinstance(e, Error) and e.fatal for e in events)
        assert not backend._parked
        await backend.close()

    run(_run())


def test_begin_activity_waits_out_a_connect_in_flight(monkeypatch):
    """An utterance during the first hello (or a reconnect) waits the un-park budget
    for the session instead of sending into the void; past it the gate drops it."""
    monkeypatch.setattr(transport, "_RESUME_TIMEOUT_S", 0.2)

    async def _run():
        backend, sent, events = make_backend()
        ready_at = asyncio.Event()

        async def slow_connect_and_run():
            await ready_at.wait()
            backend._ready.set()
            await asyncio.Event().wait()

        backend._connect_and_run = slow_connect_and_run
        await backend.start(instructions="", tools=[], on_event=backend._on_event)
        opening = asyncio.create_task(backend.begin_activity())
        await asyncio.sleep(0.05)
        assert not opening.done()  # waiting, not CAPTURING
        ready_at.set()
        await opening
        assert backend._turn is VoiceState.CAPTURING
        await backend.close()

        late, _, _ = make_backend()
        _stub_connection(late, ready=False)
        await late.start(instructions="", tools=[], on_event=late._on_event)
        with pytest.raises(RuntimeError, match="reconnect"):
            await late.begin_activity()
        assert late._turn is VoiceState.IDLE
        await late.close()

    run(_run())


def test_refused_commit_is_re_asked_when_the_active_response_ends():
    """Onset between commit and response.created: the server refuses the second
    response.create; the audio is in the conversation, so re-ask after the first ends."""

    async def _run():
        backend, sent, _ = make_backend()
        backend._ready.set()
        await backend.end_activity()  # commit + create #1
        await backend._handle_event({"type": "error", "error": {
            "code": "conversation_already_has_active_response", "message": "busy"}})
        assert backend._retry_create is True
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        await backend._handle_event({"type": "response.done",
                                     "response": {"id": "r1", "status": "completed"}})
        assert [p["type"] for p in sent] == [
            "input_audio_buffer.commit", "response.create", "response.create",
        ]
        assert backend._retry_create is False
        await backend.close()

    run(_run())


def test_refused_commit_is_not_re_asked_after_a_tool_continuation_or_discard():
    async def _run():
        backend, sent, _ = make_backend()
        backend._ready.set()
        await backend._handle_event({"type": "error", "error": {
            "code": "conversation_already_has_active_response", "message": "busy"}})
        # A tool turn's own continuation IS a response.create: not a second one.
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        await backend._handle_event({"type": "response.function_call_arguments.done",
                                     "response_id": "r1", "call_id": "c1", "name": "t",
                                     "arguments": "{}"})
        await backend._handle_event({"type": "response.done",
                                     "response": {"id": "r1", "status": "completed"}})
        assert sent == [] and backend._retry_create is True
        await backend.end_activity(commit=False)  # the activity was discarded after all
        assert backend._retry_create is False
        await backend.close()

    run(_run())


def test_park_and_resume_interleaved_leave_one_live_loop():
    async def _run():
        backend, _, _ = make_backend()
        connects = _stub_connection(backend)
        await backend.start(instructions="", tools=[], on_event=backend._on_event)
        await asyncio.sleep(0)
        parking = asyncio.create_task(backend.park())
        await asyncio.sleep(0)  # park is inside its teardown awaits
        await backend.begin_activity()  # an onset lands mid-park: must wait, then resume
        await parking
        assert not backend._parked and backend._ready.is_set()
        assert backend._rx_task is not None and not backend._rx_task.done()
        assert connects == [1, 1]
        await backend.close()

    run(_run())


def test_drain_never_flips_capturing_to_idle():
    """A completion landing after the next onset: the onset owns the state."""

    async def _run():
        backend, _, events = make_backend()
        backend._ready.set()
        await backend.begin_activity()
        backend._cancel_watchdog()  # as response.done does
        backend._start_drain()
        await backend._drain_task
        assert [e.state for e in events if isinstance(e, StateHint)] == [VoiceState.CAPTURING]
        assert backend._watchdog_task is not None and not backend._watchdog_task.done()
        await backend.close()

    run(_run())


def test_discarding_an_activity_under_a_live_reply_keeps_its_deadman():
    """The gate aborts an open utterance when the mic gates for a reply: the reply's
    deadman (the only recovery from a stalled stream) must survive."""

    async def _run():
        backend, sent, events = make_backend()
        backend._ready.set()
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        await backend._handle_event({"type": "response.output_audio.delta",
                                     "response_id": "r1", "delta": b64(b"\x01")})
        assert backend._turn is VoiceState.SPEAKING
        await backend.end_activity(commit=False)
        assert backend._turn is VoiceState.SPEAKING
        assert backend._watchdog_task is not None and not backend._watchdog_task.done()
        assert sent[-1]["type"] == "input_audio_buffer.clear"
        await backend.close()

    run(_run())
