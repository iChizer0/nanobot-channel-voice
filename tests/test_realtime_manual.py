"""RealtimeBackend under the gated uplink: manual turns, park/resume. No network."""

from __future__ import annotations

import asyncio
import base64
from types import SimpleNamespace

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


def test_a_park_cancelled_in_its_close_handshake_resumes_on_a_fresh_session():
    """The gate's onset cancels the idle timer's park() inside cancel_and_wait(rx) (the
    socket's close handshake) and calls begin_activity at once: the resume must not
    inherit the parked session's ready flag, and nothing goes out before the new hello."""

    async def _run():
        backend, sent, events = make_backend()
        connects: list[int] = []
        release = asyncio.Event()
        seen: dict = {}

        async def connect_and_run():
            n = len(connects) + 1
            connects.append(n)
            if n == 2:
                seen.update(ready=backend._ready.is_set(), turn=backend._turn)
                await release.wait()  # the reconnect is still in flight
            backend._ws = object()
            backend._ready.set()
            backend._ever_ready = True
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                backend._ws = None  # the real finally
                await asyncio.sleep(0.2)  # websockets' close handshake
                raise

        marks: list = []

        async def begin_wire():
            marks.append((len(connects), backend._ready.is_set()))

        backend._connect_and_run = connect_and_run
        backend._activity_begin_wire = begin_wire
        await backend.start(instructions="", tools=[], on_event=backend._on_event)
        await asyncio.sleep(0)
        old = backend._rx_task
        parking = asyncio.create_task(backend.park())
        await asyncio.sleep(0.05)  # inside the handshake wait
        assert backend._parked and not parking.done()
        parking.cancel()  # gate._cancel_park, then begin_activity on the same task

        async def release_later():
            await asyncio.sleep(0.05)
            release.set()

        releaser = asyncio.create_task(release_later())
        await backend.begin_activity()
        await releaser
        assert seen == {"ready": False, "turn": VoiceState.IDLE}
        assert marks == [(2, True)]
        assert backend._turn is VoiceState.CAPTURING and not backend._parked
        assert connects == [1, 2]
        assert backend._rx_task is not old and not backend._rx_task.done()
        await asyncio.sleep(0.25)  # the old socket finished closing on its own
        assert old.done() and not backend._rx_task.done()
        assert backend._ws is not None
        await backend.close()

    run(_run())


def test_reconnect_wait_ends_when_the_ladder_gives_up(monkeypatch):
    """An utterance during a reconnect in flight: once the ladder parks the backend,
    no session is coming, so the gate gets its answer then, not at the budget."""
    monkeypatch.setattr(transport, "_BACKOFF", (0.02, 0.02))
    monkeypatch.setattr(transport, "_RESUME_TIMEOUT_S", 2.0)

    async def _run():
        backend, _, _ = make_backend()
        attempts = _failing_connection(backend)
        await backend.start(instructions="", tools=[], on_event=backend._on_event)
        t0 = asyncio.get_running_loop().time()
        with pytest.raises(RuntimeError, match="reconnect"):
            await backend.begin_activity()
        assert asyncio.get_running_loop().time() - t0 < 1.0
        assert backend._parked and len(attempts) == 3
        assert backend._turn is VoiceState.IDLE
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


def make_shell_backend(profile="openai", config: VoiceConfig | None = None):
    """``on_event`` does what VoiceShell._cloud_barge_in does: flush, then ``barge_in``."""
    backend, sent, events = make_backend(profile, config)

    async def on_event(e):
        events.append(e)
        if isinstance(e, UserSpeechStarted):
            await backend.barge_in(await backend._sink.flush())

    backend._on_event = on_event
    return backend, sent, events


def test_manual_onset_on_an_unborn_continuation_kills_it_at_birth():
    """The gate's onset lands between the post-tool response.create and its
    response.created: naming the finished trigger cancels nothing (the server answers
    response_cancel_not_active) and the continuation then plays over the user."""

    async def _run():
        backend, sent, events = make_shell_backend()
        backend._ready.set()
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        await backend._handle_event({"type": "response.function_call_arguments.done",
                                     "response_id": "r1", "call_id": "c1", "name": "t",
                                     "arguments": "{}"})
        await backend._handle_event({"type": "response.done",
                                     "response": {"id": "r1", "status": "completed"}})
        await backend.submit_tool_result("c1", "ok")
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        await backend.begin_activity()
        assert not any(p["type"] == "response.cancel" for p in sent)
        await backend._handle_event({"type": "response.created", "response": {"id": "r2"}})
        assert sent[-1] == {"type": "response.cancel", "response_id": "r2"}
        await backend._handle_event({"type": "response.output_audio.delta",
                                     "response_id": "r2", "delta": b64(b"\x01")})
        assert backend._turn is VoiceState.CAPTURING
        assert not any(isinstance(e, StateHint) and e.state is VoiceState.SPEAKING
                       for e in events)
        await backend.close()

    run(_run())


def test_a_reply_born_under_the_next_activity_dies_at_birth():
    """The user resumed before the committed utterance's response.created: their onset had
    no id to cancel, so the reply born during their speech is the talked-over one. It dies
    at birth, and their own commit answers both."""

    async def _run():
        backend, sent, events = make_shell_backend()
        backend._ready.set()
        await backend.begin_activity()
        await backend.end_activity()  # Q1: commit + create
        await backend.begin_activity()  # Q2, before Q1's reply is born
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        assert sent[-1] == {"type": "response.cancel", "response_id": "r1"}
        await backend._handle_event({"type": "response.output_audio.delta",
                                     "response_id": "r1", "delta": b64(b"\x01")})
        assert backend._turn is VoiceState.CAPTURING
        await backend.end_activity()
        await backend._handle_event({"type": "response.created", "response": {"id": "r2"}})
        assert "r2" not in backend._cancelled_responses
        assert backend._turn is VoiceState.THINKING
        await backend.close()

    run(_run())


async def _dispatched_wait(backend) -> None:
    await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
    await backend._handle_event({"type": "response.function_call_arguments.done",
                                 "response_id": "r1", "call_id": "c1", "name": "ask_nanobot",
                                 "arguments": "{}"})
    await backend._handle_event({"type": "response.done",
                                 "response": {"id": "r1", "status": "completed"}})


def test_a_blip_during_the_tool_wait_returns_to_thinking():
    """A discarded activity during a delegation's wait settles back to THINKING, not IDLE:
    the gate parks an IDLE socket, and the answer would land on a closed session."""

    async def _run():
        backend, sent, _ = make_shell_backend()
        backend._ready.set()
        await _dispatched_wait(backend)
        await backend.begin_activity()
        await backend.end_activity(commit=False)
        assert backend._turn is VoiceState.THINKING
        sent.clear()
        await backend.submit_tool_result("c1", "the answer")
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        await backend.close()

    run(_run())


def test_an_answer_landing_during_a_blip_is_asked_for_when_it_is_discarded():
    """The answer never cuts into an open activity; when that activity turns out to be a
    blip, nothing else would ask for the answer, so the discard does."""

    async def _run():
        backend, sent, _ = make_shell_backend()
        backend._ready.set()
        await _dispatched_wait(backend)
        await backend.begin_activity()
        sent.clear()
        await backend.submit_tool_result("c1", "the answer")
        assert [p["type"] for p in sent] == ["conversation.item.create"]
        await backend.end_activity(commit=False)
        assert [p["type"] for p in sent] == [
            "conversation.item.create", "input_audio_buffer.clear", "response.create",
        ]
        await backend.close()

    run(_run())


def test_a_notice_resumes_a_parked_session():
    async def _run():
        backend, sent, _ = make_backend()
        resumed: list[bool] = []

        async def resume():
            resumed.append(True)
            backend._parked = False
            backend._ready.set()

        backend._parked = True
        backend._resume = resume  # type: ignore[method-assign]
        await backend.announce("Dinner is ready.")
        for _ in range(5):
            await asyncio.sleep(0)
        assert resumed == [True]
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        await backend.close()

    run(_run())


def test_a_notice_answers_a_re_ask_left_owed():
    """A refused commit leaves a re-ask owed; a notice's create answers it too, so the
    notice's reply ending must not ask again."""

    async def _run():
        backend, sent, _ = make_backend()
        backend._ready.set()
        await backend._handle_event({"type": "error", "error": {
            "code": "conversation_already_has_active_response", "message": "busy"}})
        await backend.announce("Dinner is ready.")
        for _ in range(5):
            await asyncio.sleep(0)
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        await backend._handle_event({"type": "response.done",
                                     "response": {"id": "r1", "status": "completed"}})
        assert [p["type"] for p in sent] == ["conversation.item.create", "response.create"]
        await backend.close()

    run(_run())


def test_refused_commit_is_not_re_asked_while_the_user_speaks():
    """The deferred response.create re-issued on the cancelled done of the response the
    user just barged in on answers A over B, and B's own commit is refused again: it
    waits for B's commit, which carries A's audio too."""

    async def _run():
        backend, sent, _ = make_shell_backend()
        backend._ready.set()
        await backend._handle_event({"type": "response.created", "response": {"id": "r0"}})
        await backend.begin_activity()  # A barges in on r0
        await backend.end_activity()  # commit + create, refused: r0 is still active
        await backend._handle_event({"type": "error", "error": {
            "code": "conversation_already_has_active_response", "message": "busy"}})
        sent.clear()
        await backend.begin_activity()  # B starts before r0's done lands
        await backend._handle_event({"type": "response.done",
                                     "response": {"id": "r0", "status": "cancelled"}})
        assert sent == [] and backend._retry_create is True
        await backend.end_activity()
        assert [p["type"] for p in sent] == ["input_audio_buffer.commit", "response.create"]
        assert backend._retry_create is False  # B's create answers both
        await backend._handle_event({"type": "response.created", "response": {"id": "r1"}})
        await backend._handle_event({"type": "response.done",
                                     "response": {"id": "r1", "status": "completed"}})
        assert [p["type"] for p in sent] == ["input_audio_buffer.commit", "response.create"]
        await backend.close()

    run(_run())


def test_a_discarded_blip_still_answers_the_deferred_commit():
    """The twin of the test above where B is a blip: its discard carries no create, and
    r0's done already passed while B spoke, so nobody else would re-ask — A's committed
    audio must get its response.create here."""

    async def _run():
        backend, sent, _ = make_shell_backend()
        backend._ready.set()
        await backend._handle_event({"type": "response.created", "response": {"id": "r0"}})
        await backend.begin_activity()
        await backend.end_activity()
        await backend._handle_event({"type": "error", "error": {
            "code": "conversation_already_has_active_response", "message": "busy"}})
        sent.clear()
        await backend.begin_activity()
        await backend._handle_event({"type": "response.done",
                                     "response": {"id": "r0", "status": "cancelled"}})
        assert sent == [] and backend._retry_create is True
        await backend.end_activity(commit=False)  # B: min-length reject
        assert [p["type"] for p in sent] == ["input_audio_buffer.clear", "response.create"]
        assert backend._retry_create is False
        await backend.close()

    run(_run())


def test_an_orphaned_rx_task_raising_from_its_close_never_re_enters_the_ladder():
    """A park cancelled mid-handshake orphans the old rx task; a close raising anything but
    CancelledError must not run the ladder (it would reset the resumed session and open a
    third socket beside it)."""

    async def _run():
        backend, sent, events = make_backend()
        backend._send = transport.RealtimeTransport._send.__get__(backend)
        connects: list[int] = []

        async def connect_and_run():
            n = len(connects) + 1
            connects.append(n)
            ws = SimpleNamespace(name=f"ws{n}", send=_async_noop)
            try:
                backend._ws = ws
                await backend._handle_event({"type": "session.updated"})
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                if backend._ws is ws:
                    backend._ws = None
                await asyncio.sleep(0.05)
                raise ConnectionError(f"{ws.name}: close handshake failed") from None

        backend._connect_and_run = connect_and_run
        await backend.start(instructions="", tools=[], on_event=backend._on_event)
        await asyncio.sleep(0.01)
        parking = asyncio.create_task(backend.park())
        await asyncio.sleep(0.01)
        parking.cancel()
        await backend.begin_activity()
        assert connects == [1, 2] and backend._turn is VoiceState.CAPTURING
        await asyncio.sleep(0.2)  # the orphan's close raised meanwhile
        assert connects == [1, 2]
        assert backend._turn is VoiceState.CAPTURING and backend._ready.is_set()
        assert backend._ws.name == "ws2"
        assert not backend._orphan_rx
        assert not [e for e in events if isinstance(e, Error)]
        await backend.close()

    run(_run())


async def _async_noop(*_args, **_kwargs):
    return None


def test_a_provider_refusal_on_a_never_ready_session_is_fatal_at_once(monkeypatch):
    """A wrong model (HTTP 404 at the handshake) or a rejected setup (close 1008) is
    deterministic: walking the ladder, and under a gate re-walking it per utterance
    while parked, only burns connects. Never ready = a config error = fatal."""
    monkeypatch.setattr(transport, "_BACKOFF", (0.0, 0.0))

    async def _run():
        backend, _, events = make_backend()
        attempts: list[int] = []

        async def refused():
            attempts.append(1)
            raise OSError(SimpleNamespace(status_code=404))

        backend._connect_and_run = refused
        refused_exc = SimpleNamespace(response=SimpleNamespace(status_code=404))

        async def connect_and_run():
            attempts.append(1)
            raise type("InvalidStatus", (Exception,), {"response": refused_exc.response})()

        backend._connect_and_run = connect_and_run
        await backend.start(instructions="", tools=[], on_event=backend._on_event)
        for _ in range(50):
            if backend._rx_task.done():
                break
            await asyncio.sleep(0.01)
        assert len(attempts) == 1
        [error] = [e for e in events if isinstance(e, Error)]
        assert error.fatal and "rejected (HTTP 404)" in error.message
        assert "realtime.model" in error.message
        await backend.close()

    run(_run())


def test_a_connect_that_cannot_be_prepared_is_fatal_not_laddered(monkeypatch):
    """A key, a URL or a tool schema that raises while the hello is built is our own
    config, deterministic on every attempt: fatal at once, never a 'disconnected'."""
    monkeypatch.setattr(transport, "_BACKOFF", (0.0, 0.0))
    monkeypatch.setenv("OPENAI_API_KEY", "k")

    async def _run():
        backend, _, events = make_backend()
        attempts: list[int] = []

        def broken_hello():
            attempts.append(1)
            raise AttributeError("'NoneType' object has no attribute 'get'")

        backend._hello_payload = broken_hello
        monkeypatch.setattr(transport, "_load_connect", lambda: _never_connect)
        await backend.start(instructions="", tools=[], on_event=backend._on_event)
        for _ in range(50):
            if backend._rx_task.done():
                break
            await asyncio.sleep(0.01)
        assert len(attempts) == 1
        [error] = [e for e in events if isinstance(e, Error)]
        assert error.fatal and "could not be prepared" in error.message
        await backend.close()

    run(_run())


def _never_connect(*_args, **_kwargs):
    raise AssertionError("the hello must be built before the socket opens")
