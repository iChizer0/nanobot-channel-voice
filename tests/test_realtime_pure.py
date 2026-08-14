"""RealtimeBackend's pure surfaces: helpers + `_handle_event` on canned frames.

No network anywhere: `_handle_event` mutates state and emits normalized events,
and `_send` no-ops with no websocket.
"""

from __future__ import annotations

import asyncio
import base64

import pytest

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend import openai_realtime as rt
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import (
    Error,
    InputTranscript,
    OutputAudio,
    StateHint,
    ToolCall,
    ToolDef,
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
    b = rt.RealtimeBackend.__new__(rt.RealtimeBackend)
    b._max_tool_output_chars = 0
    assert b._clamp_tool_output("x" * 100_000) == "x" * 100_000  # 0 = unlimited
    b._max_tool_output_chars = 200
    assert b._clamp_tool_output("short") == "short"
    clamped = b._clamp_tool_output("y" * 500)
    assert len(clamped) <= 200 and "truncated" in clamped
    b._max_tool_output_chars = 10  # marker itself longer than the cap
    assert len(b._clamp_tool_output("z" * 500)) == 10


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


def hints(events) -> list[VoiceState]:
    return [e.state for e in events if isinstance(e, StateHint)]


def test_plain_turn_full_lifecycle():
    seen = {}

    async def wait_drain(backend):
        seen["ready"] = backend._ready.is_set()  # close() clears it later
        await backend._drain_task  # completed turn drains to IDLE

    backend, events = drive([
        {"type": "session.created"},
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
        await backend.barge_in(1500)
        await backend.barge_in(0)  # sink already flushed: played_ms restarted
        truncates = [p for p in sent if p["type"] == "conversation.item.truncate"]
        assert len(truncates) == 1
        assert truncates[0]["item_id"] == "item-1"
        assert truncates[0]["audio_end_ms"] == 1500
        await backend.close()

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


@pytest.mark.parametrize("key", ["openai", "qwen", "glm", "stepfun"])
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


# ---- stop-command consume (transcript-gated; DESIGN-stop-commands.md P1) ----


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
        await b._handle_event(_created("r1"))  # the ack response is already live
        epoch = b._sink.epoch
        await b._handle_event(_stop_t())
        assert {"type": "response.cancel"} in sent
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
        assert {"type": "response.cancel"} in sent  # ...and dies at birth
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
