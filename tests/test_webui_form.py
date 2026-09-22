"""The validator's ``form`` spec: the voice panel's dynamic fields, derived from the
pydantic schema so kinds/choices/nullability cannot drift, and shaped by the section's
own state (backend, engines, wake mode)."""

from __future__ import annotations

import re
from itertools import product

from nanobot_channel_voice.config import VoiceConfig
from nanobot_channel_voice.webui_form import Store, build_form


def _fields(section: dict) -> dict[str, dict]:
    form = build_form(VoiceConfig.model_validate(section))
    return {f["key"]: f for s in form["sections"] for f in s["fields"]}


def _section_ids(section: dict) -> list[str]:
    return [s["id"] for s in build_form(VoiceConfig.model_validate(section))["sections"]]


def _sections(section: dict) -> dict[str, list[dict]]:
    return {s["id"]: s["fields"] for s in build_form(VoiceConfig.model_validate(section))["sections"]}


def test_kinds_choices_and_nullability_come_from_the_schema():
    """Every row carries the resolved value, so the spec names no default; what it does
    say is whether an emptied input may be null (the panel writes it so) rather than
    withdrawn."""
    fields = _fields({})
    backend = fields["backend"]
    assert backend["kind"] == "enum" and "default" not in backend and "optional" not in backend
    values = [c["value"] for c in backend["choices"]]
    assert values[:2] == ["local", "openai"] and "gemini" in values
    assert fields["vad.hangoverMs"] == {
        "key": "vad.hangoverMs", "kind": "int", "label": "End of speech", "unit": "ms", "advanced": True,
        "help": "Silence after speech that ends the turn.", "value": 600,
    }
    assert fields["tts.enabled"]["kind"] == "bool"
    assert fields["allowFrom"]["kind"] == "list" and fields["allowFrom"]["value"] == ["*"]
    assert fields["tts.apiBase"]["kind"] == "string" and "value" not in fields["tts.apiBase"]
    assert fields["agentTimeoutS"]["optional"] is True and fields["agentTimeoutS"]["value"] == 300.0
    assert "value" not in _fields({"agentTimeoutS": None})["agentTimeoutS"]
    assert fields["device"]["optional"] is True and _fields({"backend": "openai"})["realtime.model"]["optional"] is True
    assert fields["tts.apiKey"]["optional"] is True  # the panel keeps a secret's empty as withdrawn
    assert "optional" not in fields["vad.hangoverMs"] and "optional" not in fields["allowFrom"]
    # a number the schema lets go negative: the panel's numeric keypads have no minus key
    assert fields["earcons.gainDb"]["signed"] is True
    assert not any("signed" in f for f in (fields["vad.hangoverMs"], fields["agentTimeoutS"], fields["backend"]))


def test_secrets_are_reported_configured_never_echoed():
    fields = _fields({"tts": {"apiKey": "sk-secret"}})
    assert fields["tts.apiKey"]["kind"] == "secret"
    assert fields["tts.apiKey"]["configured"] is True and "value" not in fields["tts.apiKey"]
    assert _fields({})["tts.apiKey"]["configured"] is False


def test_sections_follow_the_backend():
    """Every section is always there; one that another section's switch turns off is
    advanced with a note naming that switch, so it stays configurable under Advanced."""
    def in_use(section):  # what the panel shows with Advanced closed
        return [
            s["id"] for s in build_form(VoiceConfig.model_validate(section))["sections"]
            if not s.get("advanced") and any(not f.get("advanced") for f in s["fields"])
        ]

    def notes(section):
        return {s["id"]: s["note"] for s in build_form(VoiceConfig.model_validate(section))["sections"] if s.get("advanced")}

    assert _section_ids({}) == ["general", "audio", "stt", "tts", "interruptions", "waiting", "vad", "wake", "cues", "access"]
    # the normal view is what a first setup picks: the backend, the engines and their
    # models, the filler, the wake mode, the cues
    assert in_use({}) == ["general", "stt", "tts", "waiting", "wake", "cues"]
    assert notes({}) == {}
    # Interruptions carry their own switch (the mic during a reply) and every row of
    # theirs is advanced, the audio-path details, so the section is Advanced-only in any
    # state and never a not-in-use section while replies are spoken
    assert [f["key"] for f in _sections({})["interruptions"]] == ["aec"]
    assert all(f["advanced"] for f in _sections({"aec": "soft", "tts": {"provider": "matcha"}})["interruptions"])
    # Audio is the hardware facts alone, Listening the detectors and the silence that ends
    # a turn, every row advanced, so the panel shows both under Advanced only; the rows
    # keep the resolved values in force, the identity line and the pipeline row name the
    # detectors that run
    assert all(f["advanced"] for f in _sections({})["audio"])
    assert [f["key"] for f in _sections({})["audio"]] == ["audio.captureDevice", "audio.playbackDevice", "audio.backend"]
    listening = _sections({"vad": {"engine": "silero", "turn": {"engine": "smartturn"}}})["vad"]
    assert all(f["advanced"] for f in listening)
    assert [f["key"] for f in listening] == [
        "vad.engine", "vad.silero.weights", "vad.silero.threshold", "vad.hangoverMs", "vad.turn.engine", "vad.turn.weights",
    ]
    # nothing spoken, nothing to wait through or interrupt
    assert notes({"tts": {"enabled": False}}) == {"interruptions": "Not in use until Speak replies is on.", "waiting": "Not in use until Speak replies is on."}
    # a cloud provider: the served STT is advanced outright, the detectors until the gate runs them
    cloud = {"backend": "openai"}
    assert _section_ids(cloud) == ["general", "provider", "audio", "stt", "vad", "wake", "access"]
    assert in_use(cloud) == ["general", "provider"]
    assert notes(cloud) == {
        "stt": "The provider transcribes for itself. Serve transcription runs an engine on this device for other clients.",
        "vad": "Not in use until Send audio is On speech or After wake word.",
        "wake": "Not in use until Send audio is After wake word.",
    }
    gated = {"backend": "openai", "realtime": {"uplink": "vad"}, "vad": {"engine": "silero"}}
    assert in_use(gated) == ["general", "provider"]  # Listening is Advanced-only, in use or not
    assert set(notes(gated)) == {"stt", "wake"}
    woken = {**gated, "realtime": {"uplink": "wake"}, "wake": {"mode": "gate", "engine": "openwakeword", "phrases": ["hey"]}}
    assert in_use(woken) == ["general", "provider", "wake"]
    fields = _fields(gated)
    assert "vad.silero.weights" in fields
    # the gate endpoints with the same three: its detector, the silence that ends the
    # utterance it uploads (GatedUplink reads vad.hangoverMs), and the turn model
    assert [k for k in fields if k.startswith("vad.")] == [
        "vad.engine", "vad.silero.weights", "vad.silero.threshold", "vad.hangoverMs", "vad.turn.engine",
    ]
    assert "realtime.baseUrl" not in fields
    assert "realtime.baseUrl" in _fields({"backend": "azure"})
    # the canceller an open mic needs sits right under its switch, in cloud terms; a gated
    # mic has no use for it
    assert "aec" not in fields and all(f["advanced"] for f in _sections(cloud)["audio"])
    open_mic = _sections({**cloud, "realtime": {"bargeIn": "aec"}})["provider"]
    keys = [f["key"] for f in open_mic]
    assert keys[keys.index("realtime.bargeIn") + 1] == "aec"
    aec = next(f for f in open_mic if f["key"] == "aec")
    assert aec["help"] == (
        "Open mic barge-in needs the reply's echo cancelled. WebRTC does it in software and needs "
        "the `aec` extra, Hardware trusts the device."
    )
    # only the cancellers that work there; the saved Auto then selects nothing and the
    # Resolved setup row says what to pick
    assert [c["value"] for c in aec["choices"]] == ["webrtc", "hardware"] and aec["value"] == "auto"
    # the served STT under a cloud provider: the engine, its model and address once serving
    assert [k for k in _fields(cloud) if k.startswith("stt.")] == ["stt.serve.enabled"]
    served = _fields({**cloud, "stt": {"provider": "whisper", "serve": {"enabled": True}}})
    assert [k for k in served if k.startswith("stt.")] == [
        "stt.serve.enabled", "stt.provider", "stt.whisper.weights", "stt.whisper.language",
        "stt.serve.host", "stt.serve.port", "stt.serve.apiKey",
    ]
    assert "audio.backend" in _fields(cloud)


def test_engine_choice_reveals_its_weights(monkeypatch, tmp_path):
    monkeypatch.setenv("NANOBOT_VOICE_MODELS_DIR", str(tmp_path / "store"))  # no cached index
    fields = _fields({"stt": {"provider": "sensevoice"}, "tts": {"provider": "matcha"}})
    assert fields["stt.sensevoice.weights"]["kind"] == "weights"
    assert fields["stt.sensevoice.weights"]["help"].startswith("No model index is cached")
    assert "stt.sensevoice.language" in fields
    assert "tts.matcha.weights" in fields and "tts.model" not in fields
    assert "stt.nanobot.weights" not in _fields({})
    off = _fields({"tts": {"enabled": False}})
    assert "tts.provider" not in off
    wake = _fields({"wake": {"mode": "gate", "phrases": ["hey"], "engine": "openwakeword"}})
    assert wake["wake.phrases"]["value"] == ["hey"] and "wake.openwakeword.weights" in wake
    # phrases are typed BEFORE a mode can be picked: a mode without them is refused, and a
    # refused section keeps the previous form
    assert [k for k in _fields({}) if k.startswith("wake.")] == ["wake.mode"]  # Off: the mode alone
    # with a mode on, the Model row leads (the tiers: Transcript, heads, Custom, Local
    # files), then the files, Phrases, and the threshold once a head is the detector
    gate = list(_fields({"wake": {"mode": "gate", "phrases": ["hey"]}}))
    assert gate[gate.index("wake.mode"):gate.index("earcons.captured")] == [
        "wake.mode", "wake.openwakeword.weights", "wake.phrases",
        "wake.attention", "wake.windowS", "wake.aliases", "wake.ack.enabled",
    ]
    assert "wake.ack.phrases" in _fields({"wake": {"mode": "gate", "phrases": ["hey"], "ack": {"enabled": True}}})
    # a head of your own (Custom): the engine without a key shows the head's path input,
    # a picked head hides it, a path set beside a key keeps it in view (the path wins)
    oww = {"mode": "gate", "phrases": ["hey"], "engine": "openwakeword"}
    keys = list(_fields({"wake": oww}))
    assert keys[keys.index("wake.openwakeword.weights") + 1:keys.index("wake.openwakeword.threshold")] == ["wake.openwakeword.modelPath", "wake.phrases"]
    assert keys[keys.index("wake.openwakeword.threshold") + 1] == "wake.attention"
    assert wake["wake.openwakeword.weights"]["custom"] == "files" and wake["wake.openwakeword.weights"]["customOpen"] is True
    assert "custom" not in wake["vad.engine"]
    assert wake["wake.openwakeword.weights"]["help"].startswith("Transcript matches the phrase in the transcription.")
    picked = _fields({"wake": {**oww, "openwakeword": {"weights": "wake/openwakeword/hey/onnx"}}})
    assert "wake.openwakeword.modelPath" not in picked and "wake.openwakeword.threshold" in picked
    assert picked["wake.openwakeword.weights"]["customOpen"] is False
    override = _fields({"wake": {**oww, "openwakeword": {"weights": "wake/openwakeword/hey/onnx", "modelPath": "mine.onnx"}}})
    assert override["wake.openwakeword.modelPath"]["value"] == "mine.onnx"
    assert "wake.openwakeword.melPath" not in override  # the feature models are the store's
    assert "custom" not in _fields({"stt": {"provider": "whisper"}})["stt.whisper.weights"]  # a store key only
    # the turn block's engine is its own choice: its model row follows that choice
    assert "vad.turn.weights" not in _fields({})
    keys = list(_fields({"vad": {"turn": {"engine": "smartturn"}}}))
    assert keys[keys.index("vad.turn.engine") + 1] == "vad.turn.weights"
    assert keys.index("vad.hangoverMs") < keys.index("vad.turn.engine")


def test_unlisted_paths_get_a_humanized_label():
    from nanobot_channel_voice.webui_form import _field

    dumped = VoiceConfig().model_dump(by_alias=True)
    assert _field(dumped, "vad.hangoverMinMs")["label"] == "Hangover min ms"


def _every_field() -> dict[str, dict]:
    """Fields across the backends, engines and gates the form can branch on."""
    seen: dict[str, dict] = {}
    for backend, uplink in product(["local", "openai", "azure"], ["server", "wake"]):
        for stt, tts in product(["nanobot", "whisper", "sensevoice"], ["openai", "matcha", "system"]):
            seen.update(_fields({
                "backend": backend,
                "realtime": {"uplink": uplink},
                "stt": {"provider": stt}, "tts": {"provider": tts},
                "vad": {"engine": "silero", "turn": {"engine": "smartturn"}},
                "wake": {"mode": "gate", "phrases": ["hey"], "engine": "openwakeword"},
            }))
    return seen


def test_options_read_as_words_not_config_values():
    """Labels are the panel's text; a value falling through as-is is a table gap. Every
    label is capitalized, except names their owners spell otherwise."""
    for key, field in _every_field().items():
        for choice in field.get("choices", []):
            label = choice["label"]
            assert label != choice["value"], (key, choice)
            assert label[:1].isupper() or label in ("xAI", "openWakeWord"), (key, choice)
    assert _fields({})["stt.provider"]["choices"][0] == {"value": "nanobot", "label": "Internal"}
    assert {c["label"] for c in _fields({})["aec"]["choices"]} == {"Auto", "Open mic", "WebRTC", "Hardware"}


def test_help_names_labels_in_prose_and_literals_in_backticks():
    """The panel sets backtick spans as code; an odd count would leave one dangling, and a
    raw config value in prose would contradict the label the option shows. Help reads as
    one or two short sentences, never a semicolon list."""
    for key, field in _every_field().items():
        help_text = field.get("help", "")
        assert help_text.count("`") % 2 == 0, (key, help_text)
        assert ";" not in help_text, (key, help_text)
        assert help_text == "" or help_text.endswith("."), (key, help_text)
        # sentences start capitalized (a lowercase brand name goes mid-sentence)
        for sentence in filter(None, (part.strip() for part in re.split(r"(?<=\.)\s+", help_text))):
            assert not sentence[0].isalpha() or sentence[0].isupper(), (key, sentence)
        prose = re.sub(r"`[^`]*`", "", help_text)
        assert "[" not in prose and "_" not in prose, (key, help_text)
        for choice in field.get("choices", []):
            # "nanobot" in prose is the product, not the Internal option; a value that is
            # a word of its own label ("wake" in After wake word) reads as the label
            value = choice["value"]
            if value and value != "nanobot" and not re.search(rf"\b{re.escape(value)}\b", choice["label"].lower()):
                assert not re.search(rf"\b{re.escape(value)}\b", prose), (key, choice)


def test_field_labels_do_not_echo_their_section():
    for backend in ("local", "openai"):
        for section in build_form(VoiceConfig.model_validate({"backend": backend}))["sections"]:
            for field in section["fields"]:
                assert field["label"] != section["label"], (section["id"], field["key"])


def test_the_served_endpoint_says_what_to_point_at_it():
    """The API base is the Host and Port rows added up, which nothing else in the panel
    says; and core reads a provider without a key as unconfigured, whatever this one is."""
    served = _fields({"stt": {"provider": "whisper", "serve": {"enabled": True}}})
    assert served["stt.serve.port"]["help"] == "The API base is `http://127.0.0.1:8035/v1`."
    # beyond loopback the schema demands a key, so this is the shape that reaches the form
    open_to_lan = _fields({"stt": {"provider": "whisper", "serve": {
        "enabled": True, "host": "0.0.0.0", "port": 9000, "apiKey": "k",
    }}})
    assert open_to_lan["stt.serve.port"]["help"] == (
        "The API base is `http://<this machine>:9000/v1` for a client on the network."
    )
    assert served["stt.serve.apiKey"]["help"].startswith(
        "Callers send it as a bearer token. nanobot's own transcription needs one either way"
    )


def test_behaviour_rows_follow_their_switches():
    """The rows a listener tunes: a served endpoint's address once served, a filler's
    timing once enabled, a custom cue's clip once the cue is on, the pace of the voice
    in use; the cloud provider gets its persona, the park delay under a gate, xAI its
    reasoning knob; the wake ack is local (a cloud summon has no voice for it)."""
    base = _fields({})
    assert "stt.serve.host" not in base and "prologue.afterMs" not in base and "earcons.path" not in base
    served = _fields({"stt": {"provider": "whisper", "serve": {"enabled": True}}})
    assert [k for k in served if k.startswith("stt.serve.")] == ["stt.serve.enabled", "stt.serve.host", "stt.serve.port", "stt.serve.apiKey"]
    assert served["stt.serve.apiKey"]["kind"] == "secret"
    filler = _fields({"prologue": {"enabled": True}})
    assert [k for k in filler if k.startswith("prologue.")] == ["prologue.enabled", "prologue.afterMs", "prologue.intervalMs", "prologue.phrases"]
    assert filler["prologue.afterMs"]["unit"] == "ms" and filler["prologue.afterMs"]["label"] == "First filler after"
    cues = _fields({"earcons": {"captured": True, "attention": True}})
    assert [k for k in cues if k.startswith("earcons.")] == ["earcons.captured", "earcons.attention", "earcons.gainDb", "earcons.path", "earcons.attentionPath"]
    assert "tts.matcha.speed" in _fields({"tts": {"provider": "matcha"}}) and "tts.mms.speakingRate" in _fields({"tts": {"provider": "mms"}})
    assert "tts.openai.speed" not in _fields({"tts": {"provider": "openai"}})
    assert base["logTranscripts"]["kind"] == "bool"
    # interruptions: the switch alone under Auto; Pause and the heard marker need a
    # streaming voice, the duck level shows while ducking is what happens; every row
    # advanced
    assert [k for k in _fields({}) if k.startswith("bargeIn.") or k in ("aec", "duckDb")] == ["aec"]
    def rows(section):
        return [k for k in _fields({"aec": "soft", **section}) if k.startswith("bargeIn.") or k == "duckDb"]
    assert rows({"tts": {"provider": "matcha"}}) == ["bargeIn.mode", "duckDb", "bargeIn.minWords", "bargeIn.heardMarker"]
    assert rows({"tts": {"provider": "matcha"}, "bargeIn": {"mode": "pause"}}) == ["bargeIn.mode", "bargeIn.minWords", "bargeIn.heardMarker"]
    assert rows({"tts": {"provider": "openai"}}) == ["duckDb", "bargeIn.minWords"]  # a WAV voice ducks
    assert rows({"tts": {"provider": "openai", "audioFormat": "pcm"}})[0] == "bargeIn.mode"
    assert rows({"tts": {"provider": "system"}}) == ["duckDb", "bargeIn.minWords"]
    soft = _fields({"aec": "soft", "tts": {"provider": "matcha"}})
    assert soft["bargeIn.mode"]["choices"] == [{"value": "duck", "label": "Duck"}, {"value": "pause", "label": "Pause"}]
    assert soft["bargeIn.mode"]["advanced"] is True
    assert soft["duckDb"] == {"key": "duckDb", "kind": "float", "label": "Duck level", "unit": "dB", "advanced": True,
                              "signed": True, "value": -12.0,
                              "help": "Reply volume while you speak. 0 keeps it, -12 is about a quarter as loud."}
    assert soft["bargeIn.minWords"]["label"] == "Interrupt after" and soft["bargeIn.minWords"]["unit"] == "words"
    assert "tts.audioFormat" in base and "tts.audioFormat" not in _fields({"tts": {"provider": "matcha"}})  # the default voice is OpenAI
    # waiting: the agent timeout arms the watch, so the stall notice and both phrases hang
    # off it (an empty timeout speaks nothing, whatever the notice says); the stall phrase
    # is the timeout's own warning, so it stays without a notice
    def waits(section):
        return [k for k in _fields(section) if k in ("stallNoticeS", "agentTimeoutS", "stallPhrase", "timeoutPhrase")]
    assert waits({}) == ["agentTimeoutS", "stallNoticeS", "stallPhrase", "timeoutPhrase"]
    assert waits({"stallNoticeS": None}) == ["agentTimeoutS", "stallNoticeS", "stallPhrase", "timeoutPhrase"]
    assert waits({"agentTimeoutS": None}) == ["agentTimeoutS"]
    assert base["stallNoticeS"]["unit"] == "s" and base["stallNoticeS"]["label"] == "Stall notice after"
    assert "vad.silero.threshold" in _fields({"vad": {"engine": "silero"}}) and "vad.energy.threshold" not in base
    cloud = _fields({"backend": "openai"})
    assert "realtime.persona" in cloud and "realtime.idleParkS" not in cloud and "realtime.reasoningEffort" not in cloud
    assert "realtime.idleParkS" in _fields({"backend": "openai", "realtime": {"uplink": "vad"}, "vad": {"engine": "silero"}})
    assert "realtime.reasoningEffort" in _fields({"backend": "xai"})
    cloud_wake = _fields({"backend": "openai", "realtime": {"uplink": "wake"}, "vad": {"engine": "silero"},
                          "wake": {"mode": "gate", "phrases": ["hey"], "engine": "openwakeword"}})
    assert "wake.attention" in cloud_wake and "wake.ack.enabled" not in cloud_wake and "wake.aliases" not in cloud_wake
    # no STT in a cloud session: no Transcript tier on its Model row
    assert [c["value"] for c in cloud_wake["wake.openwakeword.weights"]["choices"]] == []
    assert cloud_wake["wake.openwakeword.weights"]["help"] == "The index lists no head, Custom takes one of your own."


def test_the_uncommon_rows_are_advanced():
    """Behind the panel's Advanced toggle, in place: the hardware facts, the audio-path
    details, the listening internals, the tuning numbers, the niche switches with their
    rows, the phrase lists and the access list. The rows a first setup needs (backend,
    the speech engines and their models, wake mode and phrases, the switches, a cloud
    provider's key, persona and uplink) are not, and every advanced key is a row some
    section shows."""
    from nanobot_channel_voice.webui_form import _ADVANCED

    shown = {}
    for section in (
        {"aec": "soft", "stt": {"provider": "whisper", "serve": {"enabled": True}}, "tts": {"provider": "matcha"},
         "prologue": {"enabled": True}, "vad": {"engine": "silero", "turn": {"engine": "smartturn"}},
         "wake": {"mode": "gate", "engine": "openwakeword", "phrases": ["hey"], "ack": {"enabled": True}},
         "earcons": {"captured": True, "attention": True}},
        {"backend": "xai", "realtime": {"uplink": "vad"}, "vad": {"engine": "firered"}},
        {"aec": "soft", "tts": {"provider": "openai", "audioFormat": "pcm"}},
        {"tts": {"provider": "mms"}},
        {"tts": {"provider": "supertonic"}},
    ):
        shown.update(_fields(section))
    assert _ADVANCED <= set(shown)
    advanced = {k for k, f in shown.items() if f.get("advanced")}
    assert advanced == _ADVANCED
    for key in ("backend", "stt.provider", "stt.whisper.weights",
                "tts.enabled", "tts.provider", "tts.matcha.weights", "prologue.enabled",
                "wake.mode", "wake.openwakeword.weights", "wake.phrases",
                "wake.ack.enabled", "earcons.captured", "realtime.apiKey", "realtime.persona", "realtime.uplink"):
        assert key not in advanced, key


def _heads(*stems: str) -> Store:
    return Store(
        index={f"wake/openwakeword/{stem}/onnx": {"files": {}} for stem in stems},
        installed=frozenset(),
        platforms=("onnx",),
    )


def _wake_row(section: dict, store: Store) -> dict:
    form = build_form(VoiceConfig.model_validate(section), store)
    rows = {f["key"]: f for s in form["sections"] for f in s["fields"]}
    return rows["wake.openwakeword.weights"]


def test_a_head_pick_fills_phrases_without_dropping_the_typed_ones():
    """A head hears one phrase and Phrases must carry it, so the pick brings it. The rest
    of the list was typed on the row: only the phrase of the head being replaced goes.
    Every head names the list, the one in force included, since the panel applies a pick
    to the config it has at the click rather than the one this form was built from."""
    store = _heads("hey-nanobot", "alexa")
    oww = {"mode": "gate", "engine": "openwakeword"}

    def offered(section):
        return {
            c["value"]: (c.get("sets") or {}).get("wake.phrases")
            for c in _wake_row(section, store)["choices"]
        }

    fresh = offered({"wake": {**oww, "phrases": ["computer"]}})
    assert fresh["wake/openwakeword/hey-nanobot/onnx"] == ["computer", "hey nanobot"]
    assert fresh["wake/openwakeword/alexa/onnx"] == ["computer", "alexa"]
    picked = offered({
        "wake": {**oww, "phrases": ["computer", "alexa"],
                 "openwakeword": {"weights": "wake/openwakeword/alexa/onnx"}},
    })
    assert picked["wake/openwakeword/hey-nanobot/onnx"] == ["computer", "hey nanobot"]
    assert picked["wake/openwakeword/alexa/onnx"] == ["computer", "alexa"]  # what it already hears


def test_the_shared_backbone_is_not_a_head():
    """The backbone is a store key of its own (the index derives it), so it can be typed
    into Custom: it names no phrase, and the section still needs its head file."""
    store = Store(
        index={"wake/openwakeword/backbone/onnx": {"files": {}}},
        installed=frozenset(),
        platforms=("onnx",),
    )
    section = {"wake": {"mode": "gate", "phrases": ["hey"], "engine": "openwakeword",
                        "openwakeword": {"weights": "wake/openwakeword/backbone/onnx"}}}
    form = build_form(VoiceConfig.model_validate(section), store)
    rows = {f["key"]: f for s in form["sections"] for f in s["fields"]}
    assert "wake.openwakeword.modelPath" in rows  # the head row a picked head hides
    assert "selected head hears" not in rows["wake.phrases"].get("help", "")
    assert [c["value"] for c in rows["wake.openwakeword.weights"]["choices"] if c["value"]] == []
