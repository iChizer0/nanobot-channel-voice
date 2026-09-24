"""GeminiLiveBackend on canned server messages. No network: `_send` records frames."""

from __future__ import annotations

import asyncio
import base64

from nanobot_channel_voice.audio.null import NullPlayback
from nanobot_channel_voice.backend import gemini_live as gl
from nanobot_channel_voice.backend.audio_sink import AudioSink
from nanobot_channel_voice.backend.base import (
    Error,
    InputTranscript,
    ManualTurnBackend,
    OutputAudio,
    OutputTranscript,
    StateHint,
    ToolCall,
    ToolDef,
    ToolStarted,
    TurnDone,
    UserSpeechStarted,
    VoiceState,
)
from nanobot_channel_voice.config import VoiceConfig


def b64(pcm: bytes) -> str:
    return base64.b64encode(pcm).decode("ascii")


def make_backend(config: VoiceConfig | None = None):
    backend = gl.GeminiLiveBackend(
        config or VoiceConfig(backend="gemini"),
        sink=AudioSink(NullPlayback(), mode="stream"),
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


def hints(events) -> list[VoiceState]:
    return [e.state for e in events if isinstance(e, StateHint)]


def audio_msg(pcm: bytes, **extra) -> dict:
    return {"serverContent": {
        "modelTurn": {"parts": [{"inlineData": {
            "mimeType": "audio/pcm;rate=24000", "data": b64(pcm)}}]},
        **extra,
    }}


def drive(frames: list[dict], *, config=None, after=None):
    async def _run():
        backend, sent, events = make_backend(config)
        for f in frames:
            await backend._handle_event(f)
        if after is not None:
            await after(backend)
        await backend.close()
        return backend, sent, events

    return asyncio.run(_run())


# ---- pure helpers -----------------------------------------------------------


def test_schema_strips_openapi_rejects_and_flattens_unions():
    out = gl._gemini_schema({
        "type": "object", "title": "Args", "additionalProperties": False,
        "properties": {"a": {"type": ["string", "null"], "default": "x"}},
        "$defs": {},
    })
    assert out == {"type": "object", "properties": {"a": {"type": "string"}}}


def test_tool_wire_is_non_blocking():
    wire = gl._tool_to_wire(ToolDef(name="t", description="d", parameters={"type": "object"}))
    assert wire["behavior"] == "NON_BLOCKING" and wire["name"] == "t"


def test_parameterless_tool_declares_no_parameters():
    """nanobot's list_exec_sessions ships ``properties: {}``; Gemini rejects an empty
    OBJECT ("should be non-empty for OBJECT type") and a rejected setup closes the
    socket, so a parameterless declaration leaves ``parameters`` unset."""
    bare = gl._tool_to_wire(ToolDef(name="t", description="d", parameters={
        "type": "object", "properties": {}, "additionalProperties": False,
    }))
    assert "parameters" not in bare
    typed = gl._tool_to_wire(ToolDef(name="t", description="d", parameters={
        "type": "object", "properties": {"a": {"type": "string"}},
    }))
    assert typed["parameters"] == {"type": "object", "properties": {"a": {"type": "string"}}}


def test_schema_types_every_property():
    """A typeless property (nanobot's ``my.value``), a ``$ref`` one (MCP) and an OBJECT
    with no properties are all setup rejections; each is declared as a string."""
    out = gl._gemini_schema({
        "type": "object",
        "properties": {
            "value": {"description": "any"},
            "ref": {"$ref": "#/$defs/A"},
            "opts": {"type": "object", "properties": {}, "required": [],
                     "description": "free-form"},
            "tags": {"type": "array", "items": {"$ref": "#/$defs/B"}},
            "nested": {"type": "object", "properties": {"n": {"description": "x"}}},
        },
        "required": ["value"],
        "$defs": {"A": {"type": "object"}, "B": {"type": "string"}},
    })
    assert out["properties"]["value"] == {"type": "string", "description": "any"}
    assert out["properties"]["ref"] == {"type": "string"}
    assert out["properties"]["opts"] == {"type": "string", "description": "free-form"}
    assert out["properties"]["tags"] == {"type": "array", "items": {"type": "string"}}
    assert out["properties"]["nested"]["properties"]["n"] == {"type": "string", "description": "x"}
    assert out["required"] == ["value"] and "$defs" not in out


def test_pcm_rate_parses_mime_or_defaults():
    assert gl._pcm_rate("audio/pcm;rate=24000", 16000) == 24000
    assert gl._pcm_rate("audio/pcm", 16000) == 16000
    assert gl._pcm_rate(None, 16000) == 16000


def test_status_is_read_from_root_or_server_content():
    assert gl._status_of({"interactionStatus": "IN_PROGRESS"}) == "IN_PROGRESS"
    assert gl._status_of({"serverContent": {"interactionStatus": "IDLE"}}) == "IDLE"
    assert gl._status_of({"serverContent": {}}) is None


def test_key_resolution_never_falls_back_to_openai(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    assert gl.resolve_gemini_key(None) is None
    monkeypatch.setenv("GOOGLE_API_KEY", "g-key")
    assert gl.resolve_gemini_key(None) == "g-key"
    assert gl.resolve_gemini_key("explicit") == "explicit"


# ---- setup payload ----------------------------------------------------------


def test_setup_payload_shape_server_vad():
    backend, _, _ = make_backend()
    backend._instructions = "be brief"
    backend._tools = [ToolDef(name="t", description="d", parameters={"type": "object"})]
    setup = backend._hello_payload()["setup"]
    assert setup["model"] == "models/gemini-3.8-live"
    assert setup["generationConfig"]["responseModalities"] == ["AUDIO"]
    assert setup["generationConfig"]["speechConfig"]["voiceConfig"]["prebuiltVoiceConfig"][
        "voiceName"] == "Kore"
    assert "thinkingConfig" not in setup["generationConfig"]
    assert setup["systemInstruction"] == {"parts": [{"text": "be brief"}]}
    assert setup["tools"][0]["functionDeclarations"][0]["behavior"] == "NON_BLOCKING"
    assert setup["sessionResumption"] == {} and "slidingWindow" in setup["contextWindowCompression"]
    assert "realtimeInputConfig" not in setup and "proactivity" not in setup
    assert "inputAudioTranscription" not in setup and setup["outputAudioTranscription"] == {}


def test_setup_payload_extended_thinking_manual_turns():
    cfg = VoiceConfig(backend="gemini", vad={"engine": "silero"}, realtime={
        "uplink": "vad", "model": "gemini-3.8-live-extended-thinking",
        "thinkingLevel": "high", "proactiveAudio": True, "inputTranscriptionModel": "on",
    })
    backend, _, _ = make_backend(cfg)
    backend._resume_handle = "h1"
    setup = backend._hello_payload()["setup"]
    assert setup["model"] == "models/gemini-3.8-live-extended-thinking"
    assert setup["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "HIGH"}
    assert setup["proactivity"] == {"proactiveAudio": True}
    assert setup["inputAudioTranscription"] == {}
    assert setup["sessionResumption"] == {"handle": "h1"}
    assert setup["realtimeInputConfig"] == {
        "automaticActivityDetection": {"disabled": True},
        "activityHandling": "START_OF_ACTIVITY_INTERRUPTS",
        "turnCoverage": "TURN_INCLUDES_ONLY_ACTIVITY",
    }


def test_thinking_level_goes_only_to_the_models_that_take_one():
    """Live-verified refusals: an extended-thinking model rejects a setup without a level,
    the base model one with it."""
    think = VoiceConfig(backend="gemini", realtime={"model": "gemini-3.8-live-extended-thinking"})
    setup = make_backend(think)[0]._hello_payload()["setup"]
    assert setup["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "LOW"}
    base = VoiceConfig(backend="gemini", realtime={"thinkingLevel": "high"})
    generation = make_backend(base)[0]._hello_payload()["setup"]["generationConfig"]
    assert "thinkingConfig" not in generation


def test_proactive_audio_connects_where_it_exists(monkeypatch):
    """Only the v1alpha surface knows `proactivity`; an explicit baseUrl still wins."""
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    backend = make_backend(VoiceConfig(backend="gemini", realtime={"proactiveAudio": True}))[0]
    assert backend._connect_args()[0] == gl.PROACTIVE_BASE_URL + "?key=g-key"
    assert ".v1alpha." in gl.PROACTIVE_BASE_URL
    own = VoiceConfig(backend="gemini", realtime={"proactiveAudio": True, "baseUrl": "wss://x/ws"})
    assert make_backend(own)[0]._connect_args()[0] == "wss://x/ws?key=g-key"


def test_connect_url_carries_the_key_and_no_headers(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    backend, _, _ = make_backend()
    url, headers = backend._connect_args()
    assert url == gl.DEFAULT_BASE_URL + "?key=g-key" and headers == {}
    assert backend._audio_frame(b"\x01\x02") == {"realtimeInput": {"audio": {
        "mimeType": "audio/pcm;rate=16000", "data": b64(b"\x01\x02")}}}


def test_adapter_satisfies_the_manual_turn_protocol():
    backend, _, _ = make_backend()
    assert isinstance(backend, ManualTurnBackend)


# ---- turns -------------------------------------------------------------------


def test_plain_turn_audio_then_turn_complete_drains_to_idle():
    seen = {}

    async def wait_drain(backend):
        seen["ready"] = backend._ready.is_set()
        await backend._drain_task

    _, _, events = drive([
        {"setupComplete": {}},
        audio_msg(b"\x01\x02"),
        {"serverContent": {"outputTranscription": {"text": "hi"}}},
        {"serverContent": {"turnComplete": True}},
    ], after=wait_drain)
    assert seen["ready"] is True
    audio = [e for e in events if isinstance(e, OutputAudio)]
    assert len(audio) == 1 and audio[0].pcm == b"\x01\x02" and audio[0].rate == 24000
    assert [e.text for e in events if isinstance(e, OutputTranscript)] == ["hi"]
    assert sum(isinstance(e, TurnDone) for e in events) == 1
    assert hints(events) == [VoiceState.SPEAKING, VoiceState.IDLE]


def test_in_progress_completion_holds_thinking_until_idle():
    async def after(backend):
        await backend._drain_task  # the filler drained -> THINKING, not IDLE
        assert backend._turn is VoiceState.THINKING
        await backend._handle_event(audio_msg(b"\x03", interactionStatus="IDLE"))
        await backend._handle_event({"serverContent": {"turnComplete": True,
                                                       "interactionStatus": "IDLE"}})
        await backend._drain_task

    _, _, events = drive([
        {"setupComplete": {}},
        audio_msg(b"\x01", interactionStatus="IN_PROGRESS"),
        {"serverContent": {"turnComplete": True, "interactionStatus": "IN_PROGRESS"}},
    ], after=after)
    assert hints(events) == [
        VoiceState.SPEAKING, VoiceState.THINKING, VoiceState.SPEAKING, VoiceState.IDLE,
    ]
    assert sum(isinstance(e, TurnDone) for e in events) == 1  # only the IDLE completion


def test_tool_call_round_trip_is_non_blocking_and_auto_continues():
    async def after(backend):
        await backend._drain_task  # completion with a pending call holds THINKING
        assert backend._turn is VoiceState.THINKING
        await backend.submit_tool_result("c1", '{"temp": 21}')
        await backend.submit_tool_result("c1", "{}")  # answered once
        await backend._handle_event(audio_msg(b"\x02"))
        await backend._handle_event({"serverContent": {"turnComplete": True}})
        await backend._drain_task

    backend, sent, events = drive([
        {"setupComplete": {}},
        audio_msg(b"\x01"),
        {"toolCall": {"functionCalls": [{"id": "c1", "name": "weather",
                                         "args": {"city": "Oslo"}}]},
         "interactionStatus": "IN_PROGRESS"},
        {"serverContent": {"turnComplete": True}},
    ], after=after)
    calls = [e for e in events if isinstance(e, ToolCall)]
    assert len(calls) == 1 and calls[0].call_id == "c1" and calls[0].name == "weather"
    assert calls[0].arguments == '{"city": "Oslo"}'
    assert any(isinstance(e, ToolStarted) for e in events)
    assert sent == [{"toolResponse": {"functionResponses": [{
        "id": "c1", "name": "weather",  # FunctionResponse.name is REQUIRED on the wire
        "response": {"result": {"temp": 21}, "scheduling": "WHEN_IDLE"},
    }]}}]
    assert sum(isinstance(e, TurnDone) for e in events) == 1  # the final turn only
    assert hints(events)[-1] is VoiceState.IDLE


def test_supervisor_results_interrupt():
    cfg = VoiceConfig(backend="gemini", realtime={"toolMode": "supervisor"})

    async def after(backend):
        await backend.submit_tool_result("c1", "plain text answer")

    _, sent, _ = drive([
        {"toolCall": {"functionCalls": [{"id": "c1", "name": "ask_nanobot", "args": {}}]}},
    ], config=cfg, after=after)
    resp = sent[0]["toolResponse"]["functionResponses"][0]["response"]
    assert resp == {"result": "plain text answer", "scheduling": "INTERRUPT"}


def test_cancelled_tool_result_is_dropped():
    async def after(backend):
        await backend.submit_tool_result("c1", "{}")

    _, sent, _ = drive([
        {"toolCall": {"functionCalls": [{"id": "c1", "name": "t", "args": {}}]}},
        {"toolCallCancellation": {"ids": ["c1"]}},
    ], after=after)
    assert sent == []


def test_server_vad_interrupted_is_the_barge_in_onset():
    _, _, events = drive([
        {"setupComplete": {}},
        audio_msg(b"\x01"),
        {"serverContent": {"interrupted": True}},
    ])
    assert any(isinstance(e, UserSpeechStarted) for e in events)
    assert hints(events) == [VoiceState.SPEAKING, VoiceState.CAPTURING]


def test_manual_interrupted_is_only_the_servers_echo():
    cfg = VoiceConfig(backend="gemini", vad={"engine": "silero"}, realtime={"uplink": "vad"})

    async def after(backend):
        backend._ready.set()
        await backend.begin_activity()
        await backend._handle_event({"serverContent": {"interrupted": True}})
        await backend.end_activity()

    _, sent, events = drive([{"setupComplete": {}}, audio_msg(b"\x01")], config=cfg, after=after)
    assert sum(isinstance(e, UserSpeechStarted) for e in events) == 1
    assert sent == [
        {"realtimeInput": {"activityStart": {}}},
        {"realtimeInput": {"activityEnd": {}}},
    ]


def test_uncommitted_activity_end_suppresses_the_answer_until_the_next_commit():
    cfg = VoiceConfig(backend="gemini", vad={"engine": "silero"}, realtime={"uplink": "vad"})

    async def after(backend):
        backend._ready.set()
        await backend.begin_activity()
        await backend.end_activity(commit=False)
        await backend._handle_event(audio_msg(b"\x01"))  # the model answered anyway
        await backend._handle_event({"serverContent": {"turnComplete": True}})
        await backend._handle_event(audio_msg(b"\x09"))  # still nobody asked
        await backend.begin_activity()
        await backend.end_activity()  # a real question: its answer plays
        await backend._handle_event(audio_msg(b"\x02"))

    _, sent, events = drive([{"setupComplete": {}}], config=cfg, after=after)
    assert [e.pcm for e in events if isinstance(e, OutputAudio)] == [b"\x02"]
    assert [next(iter(p["realtimeInput"])) for p in sent] == [
        "activityStart", "activityEnd", "activityStart", "activityEnd",
    ]


def test_unanswered_blip_does_not_eat_the_next_reply():
    """Proactive audio may never answer the empty activity: the flag must not outlive
    the next committed one."""
    cfg = VoiceConfig(backend="gemini", vad={"engine": "silero"}, realtime={"uplink": "vad"})

    async def after(backend):
        backend._ready.set()
        await backend.begin_activity()
        await backend.end_activity(commit=False)
        await backend.begin_activity()
        await backend.end_activity()
        await backend._handle_event(audio_msg(b"\x02"))

    _, _, events = drive([{"setupComplete": {}}], config=cfg, after=after)
    assert [e.pcm for e in events if isinstance(e, OutputAudio)] == [b"\x02"]


def test_manual_barge_in_drops_in_flight_audio_until_the_echo():
    cfg = VoiceConfig(backend="gemini", vad={"engine": "silero"}, realtime={"uplink": "vad"})

    async def after(backend):
        backend._ready.set()
        await backend.begin_activity()  # over a live reply: activityStart interrupts
        await backend._handle_event(audio_msg(b"\x08"))  # already on the wire: stale
        await backend._handle_event({"serverContent": {"interrupted": True,
                                                       "turnComplete": True}})
        await backend.end_activity()
        await backend._handle_event(audio_msg(b"\x02"))

    _, _, events = drive([{"setupComplete": {}}, audio_msg(b"\x01")], config=cfg, after=after)
    assert [e.pcm for e in events if isinstance(e, OutputAudio)] == [b"\x01", b"\x02"]
    # The cut-off turn's completion is not a turn end: no TurnDone, no drain to IDLE.
    assert not any(isinstance(e, TurnDone) for e in events)
    assert hints(events) == [VoiceState.SPEAKING, VoiceState.CAPTURING, VoiceState.SPEAKING]


def make_shell_backend(config: VoiceConfig):
    """``on_event`` does what VoiceShell._cloud_barge_in does: flush, then ``barge_in``."""
    backend, sent, events = make_backend(config)

    async def on_event(e):
        events.append(e)
        if isinstance(e, UserSpeechStarted):
            await backend.barge_in(await backend._sink.flush())

    backend._on_event = on_event
    return backend, sent, events


def test_manual_barge_in_via_the_shell_drops_the_dead_turn():
    """begin_activity emits the onset BEFORE activityStart goes out; the shell's
    barge_in in between must not clear what arms the dead-audio guards."""
    cfg = VoiceConfig(backend="gemini", vad={"engine": "silero"}, realtime={"uplink": "vad"})

    async def _run():
        backend, _, events = make_shell_backend(cfg)
        await backend._handle_event({"setupComplete": {}})
        await backend._handle_event(audio_msg(b"\x01"))
        await backend.begin_activity()  # over a live reply: activityStart interrupts
        assert backend._dead_audio is True
        await backend._handle_event(audio_msg(b"\x08"))  # already on the wire: stale
        await backend._handle_event({"serverContent": {"interrupted": True,
                                                       "turnComplete": True}})
        await backend.end_activity()
        await backend._handle_event(audio_msg(b"\x02"))
        await backend.close()
        return events

    events = asyncio.run(_run())
    assert [e.pcm for e in events if isinstance(e, OutputAudio)] == [b"\x01", b"\x02"]
    assert not any(isinstance(e, TurnDone) for e in events)
    assert hints(events) == [VoiceState.SPEAKING, VoiceState.CAPTURING, VoiceState.SPEAKING]


def test_manual_barge_in_echo_after_the_commit_is_not_a_turn_end():
    """A short utterance commits before the server's ``interrupted`` echo lands: the
    dead turn's completion must not count as a turn (TurnDone, anchor wiped, deadman
    disarmed) for the answer the commit is still owed."""
    cfg = VoiceConfig(backend="gemini", vad={"engine": "silero"}, realtime={"uplink": "vad"})

    async def _run():
        backend, _, events = make_shell_backend(cfg)
        await backend._handle_event({"setupComplete": {}})
        await backend._handle_event(audio_msg(b"\x01"))
        await backend.begin_activity()
        await backend.end_activity()
        assert backend.metrics._anchor is not None
        await backend._handle_event({"serverContent": {"interrupted": True}})
        await backend._handle_event({"serverContent": {"turnComplete": True}})
        await asyncio.sleep(0)
        assert backend.metrics._anchor is not None
        assert backend._turn is VoiceState.CAPTURING
        assert backend._watchdog_task is not None and not backend._watchdog_task.done()
        await backend.close()
        return events

    events = asyncio.run(_run())
    assert not any(isinstance(e, TurnDone) for e in events)
    assert hints(events) == [VoiceState.SPEAKING, VoiceState.CAPTURING]


def test_manual_unspoken_completion_settles_idle():
    """Manual turns get no THINKING at commit: a completion with nothing spoken
    (proactive audio declined) must settle now, not sit in CAPTURING for the deadman."""
    cfg = VoiceConfig(backend="gemini", vad={"engine": "silero"}, realtime={"uplink": "vad"})

    async def _run():
        backend, _, events = make_shell_backend(cfg)
        await backend._handle_event({"setupComplete": {}})
        await backend.begin_activity()
        await backend.end_activity()
        await backend._handle_event({"serverContent": {"turnComplete": True}})
        await asyncio.sleep(0)
        assert backend._turn is VoiceState.IDLE
        task = backend._watchdog_task
        assert task is None or task.done() or task.cancelling()
        # A completion landing under the NEXT open activity is that onset's to settle.
        await backend.begin_activity()
        await backend._handle_event({"serverContent": {"turnComplete": True}})
        await asyncio.sleep(0)
        assert backend._turn is VoiceState.CAPTURING
        await backend.close()
        return events

    events = asyncio.run(_run())
    assert hints(events) == [VoiceState.CAPTURING, VoiceState.IDLE, VoiceState.CAPTURING]
    assert sum(isinstance(e, TurnDone) for e in events) == 2


def test_blip_answer_completion_under_the_next_activity_keeps_the_onset():
    """The unheard answer to a blip completes while the user's real utterance is open:
    the onset owns the state and the deadman (the gate reads IDLE under an open
    activity as a lost session)."""
    cfg = VoiceConfig(backend="gemini", vad={"engine": "silero"}, realtime={"uplink": "vad"})

    async def _run():
        backend, _, events = make_shell_backend(cfg)
        await backend._handle_event({"setupComplete": {}})
        await backend.begin_activity()
        await backend.end_activity(commit=False)
        await backend.begin_activity()  # the real question
        await backend._handle_event(audio_msg(b"\x01"))  # the blip's answer, unheard
        await backend._handle_event({"serverContent": {"turnComplete": True}})
        await asyncio.sleep(0)
        assert backend._turn is VoiceState.CAPTURING
        assert backend._watchdog_task is not None and not backend._watchdog_task.done()
        await backend.end_activity()
        await backend._handle_event(audio_msg(b"\x02"))
        await backend.close()
        return events

    events = asyncio.run(_run())
    assert [e.pcm for e in events if isinstance(e, OutputAudio)] == [b"\x02"]
    assert hints(events) == [
        VoiceState.CAPTURING, VoiceState.IDLE, VoiceState.CAPTURING, VoiceState.SPEAKING,
    ]


def test_server_vad_interrupted_turn_completion_does_not_end_the_turn():
    _, _, events = drive([
        {"setupComplete": {}},
        audio_msg(b"\x01"),
        {"serverContent": {"interrupted": True}},
        {"serverContent": {"turnComplete": True}},  # the dead turn's end
    ])
    assert not any(isinstance(e, TurnDone) for e in events)
    assert hints(events) == [VoiceState.SPEAKING, VoiceState.CAPTURING]


def test_deadman_waits_for_sibling_tool_calls():
    async def after(backend):
        await backend.submit_tool_result("c1", "{}")
        assert backend._watchdog_task is None or backend._watchdog_task.done()
        await backend.submit_tool_result("c2", "{}")
        assert backend._watchdog_task is not None and not backend._watchdog_task.done()

    drive([
        {"toolCall": {"functionCalls": [{"id": "c1", "name": "a", "args": {}},
                                        {"id": "c2", "name": "b", "args": {}}]}},
    ], after=after)


def test_input_transcript_and_resumption_handle():
    async def after(backend):
        assert backend._resume_handle == "h2"
        assert backend._hello_payload()["setup"]["sessionResumption"] == {"handle": "h2"}

    _, _, events = drive([
        {"sessionResumptionUpdate": {"newHandle": "h1", "resumable": True}},
        {"sessionResumptionUpdate": {"newHandle": "h2", "resumable": True}},
        {"sessionResumptionUpdate": {"newHandle": "h3", "resumable": False}},
        {"serverContent": {"inputTranscription": {"text": "what time is it"}}},
        {"goAway": {"timeLeft": "10s"}},
    ], after=after)
    assert [e.text for e in events if isinstance(e, InputTranscript)] == ["what time is it"]


def test_quiet_deadman_settles_without_error(monkeypatch):
    """Proactive audio declined to answer: IDLE, no fault."""
    cfg = VoiceConfig(backend="gemini", vad={"engine": "silero"},
                      realtime={"uplink": "vad", "turnTimeoutS": 0.05})

    async def after(backend):
        backend._ready.set()
        await backend.begin_activity()
        await backend.end_activity()
        await asyncio.sleep(0.15)

    backend, _, events = drive([{"setupComplete": {}}], config=cfg, after=after)
    assert not any(isinstance(e, Error) for e in events)
    assert hints(events)[-1] is VoiceState.IDLE
    assert backend.metrics.snapshot()["counters"]["turn_unanswered"] == 1


def test_stalled_audio_deadman_is_a_fault():
    cfg = VoiceConfig(backend="gemini", realtime={"turnTimeoutS": 0.05})

    async def after(backend):
        backend._arm_watchdog()
        await asyncio.sleep(0.15)

    _, _, events = drive([{"setupComplete": {}}, audio_msg(b"\x01")], config=cfg, after=after)
    assert any(isinstance(e, Error) and not e.fatal for e in events)
    assert hints(events)[-1] is VoiceState.IDLE


def test_resumption_handle_lapses_after_the_vendor_validity():
    """Two hours after the session's termination the handle is dead server-side; an
    overnight-parked device must set up fresh rather than reconnect with it."""
    b, _, _ = make_backend()
    asyncio.run(b._handle_event({"sessionResumptionUpdate": {"resumable": True, "newHandle": "h1"}}))
    b._reset_turn_state(reason="session_lost")  # the socket closed (park, drop, goAway)
    assert b._hello_payload()["setup"]["sessionResumption"] == {"handle": "h1"}
    b._handle_since -= gl._RESUME_HANDLE_S + 1
    assert b._hello_payload()["setup"]["sessionResumption"] == {}
    assert b._resume_handle is None


def test_rejection_recognizes_auth_and_refusal_shapes():
    """A refusal is deterministic (retrying only burns the ladder): auth on 401/403 or an
    API-key close reason, any other 4xx/1007/1008 a config refusal; 429 and 1006 are
    load and blips."""
    from types import SimpleNamespace

    from nanobot_channel_voice.backend.transport import _rejection

    handshake = SimpleNamespace(response=SimpleNamespace(status_code=401))
    assert _rejection(handshake) == ("HTTP 401", True)
    assert _rejection(SimpleNamespace(status_code=404)) == ("HTTP 404", False)
    assert _rejection(SimpleNamespace(status_code=429)) is None
    closed = SimpleNamespace(rcvd=SimpleNamespace(code=1007, reason="API key not valid."))
    assert _rejection(closed) == ("close 1007: API key not valid.", True)
    setup = SimpleNamespace(rcvd=SimpleNamespace(code=1008, reason="invalid setup"))
    assert _rejection(setup) == ("close 1008: invalid setup", False)
    blip = SimpleNamespace(rcvd=SimpleNamespace(code=1006, reason=""))
    assert _rejection(blip) is None
    assert _rejection(RuntimeError("boom")) is None


def test_server_vad_unanswered_interruption_settles_silently():
    cfg = VoiceConfig(backend="gemini", realtime={"turnTimeoutS": 0.05})

    async def after(backend):
        # Idle frames keep flowing after the interruption; they must not feed the deadman.
        assert backend._user_speaking is False
        await asyncio.sleep(0.15)

    backend, _, events = drive([
        {"setupComplete": {}}, audio_msg(b"\x01"), {"serverContent": {"interrupted": True}},
    ], config=cfg, after=after)
    assert not any(isinstance(e, Error) for e in events)
    assert hints(events) == [VoiceState.SPEAKING, VoiceState.CAPTURING, VoiceState.IDLE]
    assert backend.metrics.snapshot()["counters"]["turn_unanswered"] == 1


def test_schema_infers_object_for_an_implicit_object_property():
    """A property that declares properties but no type is an object, not a string."""
    wire = gl._gemini_schema({
        "type": "object",
        "properties": {"where": {"properties": {"city": {"type": "string"}}}},
    })
    assert wire["properties"]["where"]["type"] == "object"
    assert wire["properties"]["where"]["properties"]["city"] == {"type": "string"}


def test_a_call_the_user_talked_over_answers_silently():
    """A barge-in lands while a call is pending: its late result joins the context SILENT
    (INTERRUPT would talk over the new utterance, WHEN_IDLE speak a stale answer after it)
    and arms no deadman. A call announced after the onset keeps its schedule."""
    cfg = VoiceConfig(backend="gemini", realtime={"toolMode": "supervisor"})

    async def after(backend):
        await backend.barge_in(0)
        await backend.submit_tool_result("c1", "(interrupted by the user)")
        assert backend._watchdog_task is None or backend._watchdog_task.done()
        await backend._handle_event(
            {"toolCall": {"functionCalls": [{"id": "c2", "name": "ask_nanobot", "args": {}}]}}
        )
        await backend.submit_tool_result("c2", "fresh answer")

    _, sent, _ = drive([
        {"toolCall": {"functionCalls": [{"id": "c1", "name": "ask_nanobot", "args": {}}]}},
    ], config=cfg, after=after)
    schedules = [
        m["toolResponse"]["functionResponses"][0]["response"]["scheduling"]
        for m in sent if "toolResponse" in m
    ]
    assert schedules == ["SILENT", "INTERRUPT"]
