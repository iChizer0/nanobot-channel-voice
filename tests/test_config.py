"""Config models: aliasing, parse-time rejection, forbid semantics, duplex derivations."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from nanobot_channel_voice.config import (
    FireRedVadConfig,
    MmsTtsConfig,
    OnDeviceRuntime,
    SenseVoiceSttConfig,
    SileroVadConfig,
    SttConfig,
    SupertonicTtsConfig,
    TtsConfig,
    VadConfig,
    VoiceConfig,
    WhisperSttConfig,
    ZipformerSttConfig,
)


def test_camel_and_snake_both_accepted():
    a = VoiceConfig.model_validate({"audio": {"captureDevice": "plug:dsnoop"}})
    b = VoiceConfig.model_validate({"audio": {"capture_device": "plug:dsnoop"}})
    assert a.audio.capture_device == b.audio.capture_device == "plug:dsnoop"


def test_camel_twin_folds_instead_of_failing():
    """Core's WebUI writers spell keys camelCase literally, next to a hand-written
    snake_case twin; with plain forbid the whole section used to fail. Equal twins
    fold silently; a genuine edit (the WebUI writes camelCase) wins over the
    shadowed hand spelling."""
    same = VoiceConfig.model_validate(
        {"audio": {"capture_device": "plug:dsnoop", "captureDevice": "plug:dsnoop"}}
    )
    assert same.audio.capture_device == "plug:dsnoop"
    differs = VoiceConfig.model_validate(
        {"audio": {"capture_device": "hand-edit", "captureDevice": "webui-edit"}}
    )
    assert differs.audio.capture_device == "webui-edit"
    # nested blocks fold at their own level too
    nested = VoiceConfig.model_validate(
        {"stt": {"serve": {"api_key": "old", "apiKey": "new"}, "provider": "whisper"}}
    )
    assert nested.stt.serve.api_key == "new"


def test_materialized_default_twin_never_clobbers_hand_data():
    """Core's enable/disable toggle merges the manifest DEFAULTS in as camelCase
    siblings of hand-written snake_case values ('' for unset strings/secrets,
    the literal default otherwise). That filler must lose to the user's data."""
    cfg = VoiceConfig.model_validate(
        {
            "log_transcripts": True,
            "logTranscripts": False,  # materialized manifest default
            "allow_from": ["console"],
            "allowFrom": ["*"],  # materialized manifest default
            "sender_id": "console",
            "audio": {"capture_device": "plug:dsnoop", "captureDevice": "default"},
            "realtime": {"api_key": "sk-hand", "apiKey": ""},  # '' = unset-secret filler
        }
    )
    assert cfg.log_transcripts is True
    assert cfg.allow_from == ["console"]
    assert cfg.audio.capture_device == "plug:dsnoop"
    assert cfg.realtime.api_key == "sk-hand"


def test_twin_fold_does_not_mutate_the_callers_mapping():
    # Core holds (and may re-save) the section dict it passes in; folding must
    # never edit it in place.
    section = {"audio": {"capture_device": "a", "captureDevice": "b"}}
    VoiceConfig.model_validate(section)
    assert section["audio"] == {"capture_device": "a", "captureDevice": "b"}


def test_typos_still_forbidden_after_twin_fold():
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"audio": {"captureDevic": "x"}})
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"capture_device": "x"})  # right key, wrong level


def test_runtime_knobs_live_on_engine_blocks_only():
    # Every on-device ENGINE block inherits the accelerator knobs; the parents
    # do not carry them at all (the old flat copies were silently per-engine).
    engine_blocks = (
        WhisperSttConfig, SenseVoiceSttConfig, ZipformerSttConfig,
        MmsTtsConfig, SupertonicTtsConfig, FireRedVadConfig, SileroVadConfig,
    )
    for block in engine_blocks:
        assert issubclass(block, OnDeviceRuntime)
        assert block.model_validate({"coreMask": "0_1"}).core_mask == "0_1"
    for parent in (VadConfig, SttConfig, TtsConfig):
        assert not issubclass(parent, OnDeviceRuntime)
        with pytest.raises(ValidationError):
            parent.model_validate({"coreMask": "0_1"})  # forbid: not a parent key


def test_stt_serve_requires_an_ondevice_engine():
    """serve.enabled with provider='nanobot' would route core's transcription
    back into core's transcription: circular; reject at parse time."""
    with pytest.raises(ValidationError, match="circular"):
        VoiceConfig.model_validate({"stt": {"provider": "nanobot", "serve": {"enabled": True}}})
    VoiceConfig.model_validate({"stt": {"provider": "whisper", "serve": {"enabled": True}}})
    VoiceConfig.model_validate({"stt": {"serve": {"enabled": False}}})  # default provider ok when off


def test_transcripts_stay_out_of_logs_by_default():
    from nanobot_channel_voice.backend.common import loggable_text

    assert VoiceConfig().log_transcripts is False
    assert loggable_text("open the pod bay doors", False) == "<5 words>"
    assert loggable_text("open the pod bay doors", True, 8) == "open the"


def test_loggable_text_collapses_newlines_into_one_line():
    from nanobot_channel_voice.backend.common import loggable_text

    # STT/model text can carry newlines; a log record must stay one line.
    assert loggable_text("line one\nline two\n\n\tline three", True) == (
        "line one line two line three"
    )
    assert loggable_text("a\nb c", False) == "<3 words>"


def test_debug_metrics_interval_parses_and_rejects_zero():
    assert VoiceConfig().debug.metrics_interval_s is None
    cfg = VoiceConfig.model_validate({"debug": {"metricsIntervalS": 30}})
    assert cfg.debug.metrics_interval_s == 30.0
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"debug": {"metricsIntervalS": 0}})


def test_stt_serve_beyond_loopback_requires_a_key():
    """0.0.0.0 without auth hands the decoder (and the mic-adjacent surface)
    to the whole network; reject at parse time."""
    with pytest.raises(ValidationError, match="apiKey"):
        VoiceConfig.model_validate(
            {"stt": {"provider": "whisper", "serve": {"enabled": True, "host": "0.0.0.0"}}}
        )
    VoiceConfig.model_validate(
        {"stt": {"provider": "whisper",
                 "serve": {"enabled": True, "host": "0.0.0.0", "apiKey": "k"}}}
    )
    VoiceConfig.model_validate(  # loopback stays keyless-friendly
        {"stt": {"provider": "whisper", "serve": {"enabled": True}}}
    )


def test_vad_engine_rate_mismatch_is_rejected_at_parse_time():
    """Regression: firered@44100 parsed fine, then make_vad demoted the raise
    to a warning and silently handed the user the energy fallback."""
    with pytest.raises(ValidationError, match="cannot run at"):
        VoiceConfig.model_validate(
            {"vad": {"engine": "firered"}, "audio": {"sampleRate": 44100}}
        )
    with pytest.raises(ValidationError, match="cannot run at"):
        VoiceConfig.model_validate(
            {"vad": {"engine": "webrtc"}, "audio": {"sampleRate": 44100}}
        )
    with pytest.raises(ValidationError, match="cannot run at"):
        VoiceConfig.model_validate(
            {"vad": {"engine": "silero"}, "audio": {"sampleRate": 48000}}
        )
    # The energy engine runs anywhere; the neural engines at their rates.
    VoiceConfig.model_validate({"vad": {"engine": "energy"}, "audio": {"sampleRate": 44100}})
    VoiceConfig.model_validate({"vad": {"engine": "firered"}, "audio": {"sampleRate": 16000}})
    VoiceConfig.model_validate({"vad": {"engine": "webrtc"}, "audio": {"sampleRate": 48000}})
    VoiceConfig.model_validate({"vad": {"engine": "silero"}, "audio": {"sampleRate": 16000}})
    VoiceConfig.model_validate({"vad": {"engine": "silero"}, "audio": {"sampleRate": 8000}})
    # A cloud backend never captures at audio.sampleRate: the gate runs its detectors at 16 kHz.
    VoiceConfig.model_validate({
        "backend": "gemini", "vad": {"engine": "silero"}, "audio": {"sampleRate": 48000},
        "realtime": {"uplink": "vad"},
    })


def test_silero_hysteresis_pair_is_ordered_at_parse_time():
    with pytest.raises(ValidationError, match="negThreshold"):
        SileroVadConfig.model_validate({"threshold": 0.5, "negThreshold": 0.5})
    assert SileroVadConfig.model_validate(
        {"threshold": 0.5, "negThreshold": 0.35}
    ).neg_threshold == 0.35
    assert SileroVadConfig().neg_threshold is None  # derived at build: threshold - 0.15


def test_mp3_audio_format_is_rejected_at_parse_time():
    # mp3 parsed fine before and produced a permanently mute channel; now the
    # error surfaces at startup where the manager logs it.
    with pytest.raises(ValidationError):
        TtsConfig.model_validate({"audioFormat": "mp3"})


def test_unknown_keys_are_rejected_loudly():
    # extra="forbid": a typo'd key is a startup error, never a silent no-op
    # (pre-release, so no deployed configs argue for a softer mode).
    with pytest.raises(ValidationError):
        VadConfig.model_validate({"hangoverMS": 800})
    assert VadConfig.model_validate({"hangoverMs": 700}).hangover_ms == 700


def test_core_progress_overrides_are_accepted_in_either_spelling():
    """Core's ChannelManager reads sendProgress/sendToolHints/showReasoning off the raw
    section (either spelling) as per-channel overrides; forbid must not reject them, and
    an unset one stays None so the export never pins core's global default."""
    for key in ("sendProgress", "send_progress", "sendToolHints", "showReasoning"):
        VoiceConfig.model_validate({key: False})
    assert VoiceConfig.model_validate({"sendProgress": True}).send_progress is True
    unset = VoiceConfig()
    assert (unset.send_progress, unset.send_tool_hints, unset.show_reasoning) == (None,) * 3
    assert "sendProgress" not in unset.model_dump(by_alias=True, exclude_unset=True)


def test_every_path_field_expands_the_user_directory():
    """The loaders open ``*Path``/``*Dir`` values verbatim, so ``~`` is expanded at parse
    time — on the shared base, so every path-holding field (walked here, not hand-listed)
    is covered, including engine blocks and the bilingual matcha secondary."""
    import os

    from nanobot_channel_voice.config import (
        DebugConfig,
        EarconsConfig,
        MatchaTtsConfig,
        OpenWakeWordConfig,
        _VoiceBase,
    )

    def walk(model, seen=frozenset()):
        if model in seen:
            return
        seen = seen | {model}
        yield model
        for info in model.model_fields.values():
            for nested in (info.annotation, *getattr(info.annotation, "__args__", ())):
                if isinstance(nested, type) and hasattr(nested, "model_fields"):
                    yield from walk(nested, seen)

    path_fields = {
        model: [n for n in model.model_fields if n.rsplit("_", 1)[-1] in ("path", "dir")]
        for model in walk(VoiceConfig)
    }
    path_fields = {m: names for m, names in path_fields.items() if names}
    assert sum(map(len, path_fields.values())) >= 40  # the walker sees the whole tree
    for model, expected in (
        (SileroVadConfig, ["model_path"]),
        (OpenWakeWordConfig, ["mel_path", "mel_filters_path", "embedding_path", "model_path"]),
        (EarconsConfig, ["path", "attention_path"]),
        (DebugConfig, ["dump_dir"]),
        (MatchaTtsConfig, ["acoustic_model_path", "lexicon_overrides_path", "espeak_data_dir"]),
    ):
        assert set(expected) <= set(path_fields[model]), model
    extras = {EarconsConfig: {"captured": True, "attention": True}}
    for model, names in path_fields.items():
        assert issubclass(model, _VoiceBase), model  # the seam every one inherits
        values = {n: f"~/{n}" for n in names} | extras.get(model, {})
        if model is MatchaTtsConfig:  # acoustic XOR encoder/decoder: check it alone
            assert model(acoustic_model_path="~/a").acoustic_model_path == os.path.expanduser("~/a")
            values.pop("acoustic_model_path")
        inst = model.model_validate(values)
        for n in values:
            if n in names:
                assert getattr(inst, n) == os.path.expanduser(f"~/{n}"), (model, n)
    # end to end, through the top-level section, and only leading ~ (a bare name stays)
    cfg = VoiceConfig.model_validate({
        "vad": {"silero": {"modelPath": "~/m/silero.onnx"}},
        "tts": {"matcha": {"espeakPath": "espeak-ng", "secondary": {"tokensPath": "~/t.txt"}}},
    })
    assert cfg.vad.silero.model_path == os.path.expanduser("~/m/silero.onnx")
    assert cfg.tts.matcha.secondary.tokens_path == os.path.expanduser("~/t.txt")
    assert cfg.tts.matcha.espeak_path == "espeak-ng"


def test_gemini_fields_parse():
    cfg = VoiceConfig.model_validate({
        "backend": "gemini",
        "realtime": {"thinkingLevel": "low", "proactiveAudio": True},
    })
    assert cfg.backend == "gemini" and cfg.realtime.thinking_level == "low"
    assert cfg.realtime.proactive_audio is True
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"realtime": {"thinkingLevel": "max"}})


def test_xai_reasoning_effort_parses():
    cfg = VoiceConfig.model_validate({"backend": "xai", "realtime": {"reasoningEffort": "high"}})
    assert cfg.realtime.reasoning_effort == "high"
    assert VoiceConfig().realtime.reasoning_effort == "none"
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"realtime": {"reasoningEffort": "medium"}})


def test_gated_uplink_refuses_the_static_fallback_engines():
    base = {"backend": "openai", "realtime": {"uplink": "vad"}}
    with pytest.raises(ValidationError, match="neural VAD"):
        VoiceConfig.model_validate(base)  # energy VAD: opens on any noise
    assert VoiceConfig.model_validate({**base, "vad": {"engine": "silero"}}).realtime.uplink == "vad"
    wake = {"backend": "openai", "realtime": {"uplink": "wake"}, "vad": {"engine": "silero"}}
    with pytest.raises(ValidationError, match="wake.mode"):
        VoiceConfig.model_validate(wake)
    with pytest.raises(ValidationError, match="openwakeword"):
        VoiceConfig.model_validate({**wake, "wake": {"mode": "gate", "phrases": ["hi bot"]}})
    ok = VoiceConfig.model_validate({
        **wake, "wake": {"mode": "gate", "phrases": ["hi bot"], "engine": "openwakeword"},
    })
    assert ok.realtime.uplink == "wake" and ok.realtime.idle_park_s == 60.0
    # The local backend ignores realtime.*: no engine demand there.
    assert VoiceConfig.model_validate({"realtime": {"uplink": "wake"}}).backend == "local"


def test_server_vad_field_is_gone():
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"realtime": {"serverVad": False}})


def test_backend_accepts_exactly_the_dialect_family():
    for name in ("local", "openai", "xai", "azure", "qwen", "glm", "stepfun", "gemini"):
        assert VoiceConfig.model_validate({"backend": name}).backend == name
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"backend": "openai_realtime"})  # alias removed


@pytest.mark.parametrize(
    ("aec", "full", "soft", "open_mic"),
    [
        ("hardware", True, False, True),
        ("soft", False, True, True),
        ("webrtc", False, False, True),
        ("auto", False, False, False),
    ],
)
def test_duplex_derivations_are_single_sourced(aec, full, soft, open_mic):
    cfg = VoiceConfig.model_validate({"aec": aec})
    assert cfg.full_duplex is full
    assert cfg.soft_duplex is soft
    assert cfg.open_mic is open_mic


def test_aec_is_a_pure_enum():
    """The JSON-bool spellings are gone: every duplex mode is a string, so the
    WebUI enum covers the whole choice space (a bool would render unselectable)."""
    for value in (True, False):
        with pytest.raises(ValidationError):
            VoiceConfig.model_validate({"aec": value})


def test_playout_delay_is_an_audio_device_knob():
    from nanobot_channel_voice.config import AudioConfig

    assert AudioConfig().playout_delay_ms == 50
    assert AudioConfig.model_validate({"playoutDelayMs": 120}).playout_delay_ms == 120
    with pytest.raises(ValidationError):
        AudioConfig.model_validate({"playoutDelayMs": -1})


def test_resolve_openai_key_prefers_explicit_over_env(monkeypatch):
    from nanobot_channel_voice.config import resolve_openai_key

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert resolve_openai_key("explicit") == "explicit"
    assert resolve_openai_key(None) == "env-key"
    monkeypatch.delenv("OPENAI_API_KEY")
    assert resolve_openai_key(None) is None


def test_range_validation_still_applies():
    with pytest.raises(ValidationError):
        VadConfig.model_validate({"hangoverMs": 50})  # ge=100
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"duckDb": 3.0})   # le=0


# ---- importJson: the WebUI paste-to-import transport ------------------------


def test_import_json_paste_wins_over_the_section():
    import json

    paste = json.dumps({"vad": {"hangover_ms": 800}, "duckDb": -6})
    cfg = VoiceConfig.model_validate(
        {"vad": {"hangoverMs": 500, "engine": "webrtc"}, "importJson": paste}
    )
    assert cfg.vad.hangover_ms == 800   # paste replaces the twin in the other spelling
    assert cfg.vad.engine == "webrtc"   # untouched siblings survive the merge
    assert cfg.duck_db == -6
    assert cfg.import_json == paste     # retained verbatim so start() knows to consume


def test_import_json_accepts_wrapped_documents():
    """People paste whole config files; the channels/voice wrappers are unambiguous
    (no VoiceConfig field carries either name) and get unwrapped."""
    import json

    whole = json.dumps({"channels": {"voice": {"tts": {"provider": "system"}}}})
    assert VoiceConfig.model_validate({"importJson": whole}).tts.provider == "system"
    inner = json.dumps({"voice": {"tts": {"provider": "system"}}})
    assert VoiceConfig.model_validate({"importJson": inner}).tts.provider == "system"


def test_import_json_is_linted_by_the_schema():
    import json

    with pytest.raises(ValidationError, match="not valid JSON"):
        VoiceConfig.model_validate({"importJson": "{oops"})
    with pytest.raises(ValidationError, match="JSON object"):
        VoiceConfig.model_validate({"importJson": "[1, 2]"})
    # a paste that parses but violates the schema fails like any other config
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"importJson": json.dumps({"vad": {"engine": "nope"}})})


def test_import_json_accepts_the_persisted_object(tmp_path):
    """The WebUI's json field writes the paste into config.json as an OBJECT (its
    validate path still sends the typed string): both must merge, wrap-unwrap, and
    consume the same way, and the caller's object must not be mutated."""
    import json

    from nanobot_channel_voice.config import consume_import_json, parse_import_blob

    paste = {"channels": {"voice": {"vad": {"hangover_ms": 800}, "enabled": False}}}
    before = json.dumps(paste)
    assert parse_import_blob(paste) == {"vad": {"hangover_ms": 800}}
    assert json.dumps(paste) == before  # unwrapping/popping worked on a copy
    cfg = VoiceConfig.model_validate({"enabled": True, "importJson": paste})
    assert cfg.vad.hangover_ms == 800 and cfg.enabled is True
    assert cfg.import_json == paste  # retained so start() knows to consume
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"channels": {"voice": {"enabled": True, "importJson": paste}}}),
        encoding="utf-8",
    )
    assert consume_import_json(path) == 1
    voice = json.loads(path.read_text(encoding="utf-8"))["channels"]["voice"]
    assert voice == {"enabled": True, "vad": {"hangoverMs": 800}}
    with pytest.raises(ValidationError, match="JSON object"):
        VoiceConfig.model_validate({"importJson": 7})


def test_import_json_empty_filler_is_ignored():
    """An emptied box (or a hand-written empty value) must neither merge nor mark an
    import as pending."""
    assert VoiceConfig.model_validate({"importJson": ""}).import_json is None
    assert VoiceConfig.model_validate({"importJson": {}}).import_json is None
    assert VoiceConfig.model_validate({"import_json": None}).import_json is None


def test_consume_import_json_expands_and_deletes(tmp_path):
    import json

    from nanobot_channel_voice.config import consume_import_json

    path = tmp_path / "config.json"
    paste = {"vad": {"hangover_ms": 800}, "tts": {"provider": "system"}}
    path.write_text(
        json.dumps(
            {
                "providers": {"openai": {"apiKey": "k"}},
                "channels": {
                    "voice": {
                        "enabled": True,
                        "vad": {"hangoverMs": 500},
                        "importJson": json.dumps(paste),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    assert consume_import_json(path) == 2
    saved = json.loads(path.read_text(encoding="utf-8"))
    voice = saved["channels"]["voice"]
    assert "importJson" not in voice
    assert voice["vad"] == {"hangoverMs": 800}  # canonical camelCase, twin replaced
    assert voice["tts"] == {"provider": "system"}
    assert voice["enabled"] is True
    assert saved["providers"] == {"openai": {"apiKey": "k"}}  # rest of the file untouched
    assert consume_import_json(path) == 0  # idempotent: nothing pending anymore


def test_import_paste_never_carries_enabled(tmp_path):
    """`enabled` belongs to the WebUI toggle: a paste from a disabled section must neither
    flip the channel off nor be written back by start(). allowFrom stays importable."""
    import json

    from nanobot_channel_voice.config import consume_import_json, parse_import_blob

    paste = json.dumps({
        "enabled": False, "allowFrom": ["u1"], "senderId": "u1", "vad": {"hangoverMs": 800},
    })
    assert "enabled" not in parse_import_blob(paste)
    cfg = VoiceConfig.model_validate({"enabled": True, "importJson": paste})
    assert cfg.enabled is True
    assert cfg.allow_from == ["u1"]
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"channels": {"voice": {"enabled": True, "importJson": paste}}}),
        encoding="utf-8",
    )
    consume_import_json(path)
    voice = json.loads(path.read_text(encoding="utf-8"))["channels"]["voice"]
    assert voice["enabled"] is True
    assert voice["allowFrom"] == ["u1"] and voice["vad"] == {"hangoverMs": 800}


def test_reset_directive_drops_the_sections_own_keys(tmp_path, monkeypatch):
    """`{"$reset": true}` is the panel's Reset: the section's own keys go, the rest of
    the paste stacks on `enabled` alone, and the defaults show through whatever the paste
    does not set, the schema's or the operator's layer (never copied in, so a baseline
    changed later still reaches every key the section leaves alone)."""
    import json

    from nanobot_channel_voice.config import DEFAULTS_ENV, consume_import_json, merge_import

    monkeypatch.delenv(DEFAULTS_ENV, raising=False)
    saved = {"enabled": True, "device": "rv1126b", "stt": {"provider": "whisper"}, "vad": {"hangoverMs": 500}}
    assert merge_import(saved, {"$reset": True}) == {"enabled": True}
    assert merge_import(saved, {"$reset": True, "vad": {"hangover_ms": 800}}) == {"enabled": True, "vad": {"hangoverMs": 800}}
    assert merge_import(saved, {"$reset": False, "duckDb": -6}) == {**saved, "duckDb": -6}  # not a reset
    cfg = VoiceConfig.model_validate({**saved, "importJson": json.dumps({"$reset": True})})
    assert cfg.stt.provider == "nanobot" and cfg.device is None and cfg.enabled is True

    defaults = tmp_path / "voice-defaults.json"
    defaults.write_text(json.dumps({"channels": {"voice": {"enabled": False, "device": "rk3588", "tts": {"provider": "system"}}}}))
    monkeypatch.setenv(DEFAULTS_ENV, str(defaults))
    reset = merge_import(saved, {"$reset": True, "stt": {"provider": "sensevoice"}})
    assert reset == {"enabled": True, "stt": {"provider": "sensevoice"}}  # the layer is read, not written
    cfg = VoiceConfig.model_validate({**saved, "importJson": {"$reset": True, "stt": {"provider": "sensevoice"}}})
    assert (cfg.device, cfg.tts.provider, cfg.stt.provider, cfg.enabled) == ("rk3588", "system", "sensevoice", True)
    monkeypatch.setenv(DEFAULTS_ENV, json.dumps({"device": "rk3588"}))  # the JSON itself
    assert VoiceConfig.model_validate({**saved, "importJson": {"$reset": True}}).device == "rk3588"

    # start() writes the reset section, not a merge over the old keys and not the layer
    monkeypatch.setenv(DEFAULTS_ENV, str(defaults))
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"channels": {"voice": {**saved, "importJson": {"$reset": True, "aec": "soft"}}}}), encoding="utf-8")
    assert consume_import_json(path) == 2
    voice = json.loads(path.read_text(encoding="utf-8"))["channels"]["voice"]
    assert voice == {"enabled": True, "aec": "soft"}


def test_operator_defaults_lie_beneath_the_section(tmp_path, monkeypatch):
    """$NANOBOT_VOICE_DEFAULTS is a layer under channels.voice, read wherever a section
    is resolved: a fresh section runs the baseline, a key the section sets wins in either
    spelling (a null too, being a value of its own), a partial block is filled key by key,
    what the baseline leaves unset comes from the schema, and the section is left as
    written. Unreadable defaults refuse every section, naming the variable."""
    import json

    from nanobot_channel_voice.config import DEFAULTS_ENV, layer_defaults

    baseline = {
        "device": "rk3588",
        "audio": {"captureDevice": "plughw:1,0", "playbackDevice": "plughw:1,0"},
        "stt": {"provider": "sensevoice", "sensevoice": {"weights": "stt/sensevoice/small/rknn.rk3588"}},
        "vad": {"engine": "silero", "silero": {"weights": "vad/silero/v6/rknn.rk3588"}},
        "enabled": True,  # core's key, never part of the layer
        "$reset": True,  # nor the directive
    }
    monkeypatch.setenv(DEFAULTS_ENV, json.dumps(baseline))
    fresh = VoiceConfig.model_validate({"enabled": False, "importJson": ""})  # as onboarding writes it
    assert (fresh.enabled, fresh.device, fresh.audio.capture_device) == (False, "rk3588", "plughw:1,0")
    assert (fresh.stt.provider, fresh.stt.sensevoice.weights, fresh.vad.engine) == ("sensevoice", "stt/sensevoice/small/rknn.rk3588", "silero")
    assert (fresh.tts.provider, fresh.vad.hangover_ms, fresh.audio.backend) == ("openai", 600, "alsa")  # the schema's
    section = {"device": None, "audio": {"capture_device": "hw:2,0"}, "vad": {"hangoverMs": 800}, "stt": {"provider": "whisper"}}
    over = VoiceConfig.model_validate(section)
    assert (over.device, over.audio.capture_device, over.audio.playback_device) == (None, "hw:2,0", "plughw:1,0")
    assert (over.vad.engine, over.vad.hangover_ms, over.vad.silero.weights) == ("silero", 800, "vad/silero/v6/rknn.rk3588")
    assert (over.stt.provider, over.stt.sensevoice.weights) == ("whisper", "stt/sensevoice/small/rknn.rk3588")
    layered = layer_defaults(section)
    assert layered["audio"] == {"capture_device": "hw:2,0", "playbackDevice": "plughw:1,0"}  # as written, filled
    assert "enabled" not in layered and "$reset" not in layered and layered["device"] is None
    assert section == {"device": None, "audio": {"capture_device": "hw:2,0"}, "vad": {"hangoverMs": 800}, "stt": {"provider": "whisper"}}
    monkeypatch.setenv(DEFAULTS_ENV, str(tmp_path / "gone.json"))
    with pytest.raises(ValidationError, match="NANOBOT_VOICE_DEFAULTS: cannot read"):
        VoiceConfig.model_validate({})
    with pytest.raises(ValueError, match="NANOBOT_VOICE_DEFAULTS: cannot read"):
        layer_defaults({})
    # a relative path resolves against each process's cwd, so the refusal names the file
    # the gateway actually looked for, not the spelling the operator's shell used
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(DEFAULTS_ENV, "gone.json")
    with pytest.raises(ValueError, match=f"cannot read {tmp_path.resolve()}/gone.json"):
        layer_defaults({})
    monkeypatch.setenv(DEFAULTS_ENV, "{nope")
    with pytest.raises(ValidationError, match="NANOBOT_VOICE_DEFAULTS: importJson is not valid JSON"):
        VoiceConfig.model_validate({})
    monkeypatch.delenv(DEFAULTS_ENV)
    assert layer_defaults(section) is section  # no layer, nothing touched


def test_one_rule_decides_what_a_pending_paste_is(tmp_path, monkeypatch):
    """The schema, the CLI's sync and the form's lenient shaping all resolve a section
    through one primitive, so they cannot disagree about which blob is pending: the twin
    spellings, the fillers core materializes, and a blob that is unusable whatever it
    looks like."""
    import json

    from nanobot_channel_voice.config import DEFAULTS_ENV, split_paste
    from nanobot_channel_voice.sync import voice_section
    from nanobot_channel_voice.webui_form import lenient_config

    monkeypatch.delenv(DEFAULTS_ENV, raising=False)
    paste = json.dumps({"aec": "soft"})
    for filler in (None, "", [], {}):
        assert split_paste({"enabled": True, "importJson": filler}) == ({"enabled": True}, None)
    assert split_paste({"import_json": paste, "aec": "auto"}) == ({"aec": "auto"}, paste)
    # both spellings present: the camel one is the paste, and neither key survives
    assert split_paste({"importJson": paste, "import_json": "{}"})[1] == paste

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"channels": {"voice": {"import_json": paste, "device": "rk3588"}}}))
    assert voice_section(path) == {"device": "rk3588", "aec": "soft"}
    # whitespace is not a paste anyone can use: every consumer refuses it, none skips it
    path.write_text(json.dumps({"channels": {"voice": {"importJson": "   "}}}))
    with pytest.raises(Exception, match="importJson is not valid JSON"):
        voice_section(path)
    with pytest.raises(ValidationError, match="importJson is not valid JSON"):
        VoiceConfig.model_validate({"importJson": "   "})
    assert lenient_config({"importJson": "   ", "aec": "soft"}).aec == "soft"  # shapes anyway


def test_consume_import_json_leaves_files_without_a_pending_paste(tmp_path):
    import json

    from nanobot_channel_voice.config import consume_import_json

    assert consume_import_json(tmp_path / "missing.json") == 0
    path = tmp_path / "config.json"
    original = {"channels": {"voice": {"enabled": True, "importJson": ""}}}
    path.write_text(json.dumps(original), encoding="utf-8")
    assert consume_import_json(path) == 0
    # the '' filler stays put, like every other materialized secret default
    assert json.loads(path.read_text(encoding="utf-8")) == original


def test_wake_ack_validators():
    from nanobot_channel_voice.config import VoiceConfig

    base = {"mode": "gate", "phrases": ["hey nanobot"]}
    ok = VoiceConfig.model_validate({"wake": {**base, "ack": {"enabled": True}}})
    assert ok.wake.ack.enabled and ok.wake.ack.phrases is None
    # Off by default, and inert config parses without wake at all.
    assert VoiceConfig().wake.ack.enabled is False

    with pytest.raises(ValidationError, match="wake.mode"):
        VoiceConfig.model_validate({"wake": {"mode": "off", "ack": {"enabled": True}}})
    with pytest.raises(ValidationError, match="windowS"):
        VoiceConfig.model_validate(
            {"wake": {**base, "windowS": 0, "ack": {"enabled": True}}}
        )
    with pytest.raises(ValidationError, match="nothing to say"):
        VoiceConfig.model_validate(
            {"wake": {**base, "ack": {"enabled": True, "phrases": []}}}
        )
    # An ack that speaks a wake phrase would echo-veto the bot's own hits.
    with pytest.raises(ValidationError, match="contains a wake phrase"):
        VoiceConfig.model_validate(
            {"wake": {**base, "ack": {"enabled": True, "phrases": ["just say hey nanobot"]}}}
        )
    with pytest.raises(ValidationError, match="contains a wake phrase"):
        VoiceConfig.model_validate(
            {"wake": {"mode": "gate", "phrases": ["小助手"],
                      "ack": {"enabled": True, "phrases": ["我在，小助手在听"]}}}
        )
    # Boundary-aware, not substring panic: "hey" inside "they" is no mention.
    VoiceConfig.model_validate(
        {"wake": {**base, "ack": {"enabled": True, "phrases": ["they say I am here"]}}}
    )


def test_wake_alias_and_attention_validators():
    base = {"mode": "gate", "phrases": ["hey nanobot"]}
    with pytest.raises(ValidationError, match="non-empty"):
        VoiceConfig.model_validate({"wake": {**base, "aliases": [" "]}})
    with pytest.raises(ValidationError, match="windowS"):
        VoiceConfig.model_validate(
            {"wake": {**base, "attention": "sentence", "windowS": 0}}
        )
    # an ack phrase containing an ALIAS causes the same echo-veto lockout
    with pytest.raises(ValidationError, match="contains a wake phrase"):
        VoiceConfig.model_validate({"wake": {
            **base,
            "aliases": ["嘿难道爸"],
            "ack": {"enabled": True, "phrases": ["好的嘿难道爸"]},
        }})
    # aliases alone are legal and inert richness
    cfg = VoiceConfig.model_validate({"wake": {**base, "aliases": ["he nine obt"]}})
    assert cfg.wake.aliases == ["he nine obt"]
    assert cfg.wake.attention == "conversation"


def test_earcons_path_validators():
    # a path with its cue not enabled is inert config: reject it loudly at load
    with pytest.raises(ValidationError, match="earcons.captured is not enabled"):
        VoiceConfig.model_validate({"earcons": {"path": "/x/cue.wav"}})
    with pytest.raises(ValidationError, match="earcons.attention is not enabled"):
        VoiceConfig.model_validate({"earcons": {"attentionPath": "/x/close.wav"}})
    with pytest.raises(ValidationError, match="file path"):
        VoiceConfig.model_validate({"earcons": {"captured": True, "path": "  "}})
    cfg = VoiceConfig.model_validate({"earcons": {
        "captured": True, "path": "/x/cue.wav",
        "attention": True, "attentionPath": "/x/close.wav",
    }})
    assert cfg.earcons.path == "/x/cue.wav"
    assert cfg.earcons.attention_path == "/x/close.wav"


def test_notice_phrases_must_not_contain_stop_phrases():
    # A notice that SPEAKS a stop phrase taints it as self-echo (exactly-spoken
    # words are never fresh): the invited command would be swallowed. Word-bounded
    # for Latin so "unstoppable" stays legal; substring for CJK.
    with pytest.raises(ValidationError, match="stop phrase"):
        VoiceConfig(stallPhrase="Still working. Say stop if you want me to give up.")
    with pytest.raises(ValidationError, match="stop phrase"):
        VoiceConfig(timeoutPhrase="出错了，说停止可以取消。")
    VoiceConfig(stallPhrase="An unstoppable effort continues.")  # word-bounded
    VoiceConfig()  # the shipped defaults are self-consistent


def test_every_16k_only_engine_is_rate_checked_at_parse_time():
    """Same trap as the neural VADs, three more engines wide: smartturn, openWakeWord
    and zipformer all demand 16 kHz and all degraded at runtime behind one log line
    (silence-only endpointing, transcript-only wake, a hard start() raise)."""
    oww = {"melPath": "/m.onnx", "embeddingPath": "/e.onnx", "modelPath": "/h.onnx"}
    for section in (
        {"vad": {"turn": {"engine": "smartturn"}}},
        {"wake": {"mode": "gate", "phrases": ["hi"], "engine": "openwakeword",
                  "openwakeword": oww}},
        {"stt": {"provider": "zipformer"}},
    ):
        with pytest.raises(ValidationError, match="cannot run at"):
            VoiceConfig.model_validate({**section, "audio": {"sampleRate": 48000}})
        VoiceConfig.model_validate({**section, "audio": {"sampleRate": 16000}})
    # wake.engine is only a claim while the gate is on: mode="off" ignores it.
    VoiceConfig.model_validate(
        {"wake": {"engine": "openwakeword"}, "audio": {"sampleRate": 48000}}
    )


def test_empty_allow_from_is_rejected():
    """[] denies every speaker and voice has no pairing flow to recover through, so
    core's materialized list filler would leave a healthy-looking, deaf channel."""
    with pytest.raises(ValidationError, match="denies every speaker"):
        VoiceConfig.model_validate({"allowFrom": []})
    assert VoiceConfig().allow_from == ["*"]
    assert VoiceConfig.model_validate({"allowFrom": ["local"]}).allow_from == ["local"]


def test_allow_from_must_admit_the_mics_sender():
    """The mic publishes every utterance as senderId: a list naming other ids (copied from
    a chat channel) makes core drop each one, and the turn sits THINKING to the deadman."""
    with pytest.raises(ValidationError, match="nor senderId 'local'"):
        VoiceConfig.model_validate({"allowFrom": ["telegram-4711"]})
    cfg = VoiceConfig.model_validate({"allowFrom": ["kitchen"], "senderId": "kitchen"})
    assert cfg.allow_from == ["kitchen"]


def test_transcription_gap_reports_an_unusable_delegate():
    """stt.provider='nanobot' hands every utterance to core, whose failure path returns
    '' — indistinguishable from silence. The gap string is what start() and the WebUI
    check speak; an unrecognized core shape yields None rather than a false alarm."""
    import nanobot_channel_voice.config as voice_config

    class _Eff:
        enabled, configured, provider = True, False, "groq"

    monkey = {"resolve_transcription_config": lambda _cfg: _Eff(),
              "load_config": lambda: object()}
    import sys
    import types

    fake_tr = types.ModuleType("nanobot.audio.transcription")
    fake_tr.resolve_transcription_config = monkey["resolve_transcription_config"]
    fake_loader = types.ModuleType("nanobot.config.loader")
    fake_loader.load_config = monkey["load_config"]
    real = (sys.modules.get("nanobot.audio.transcription"),
            sys.modules.get("nanobot.config.loader"))
    sys.modules["nanobot.audio.transcription"] = fake_tr
    sys.modules["nanobot.config.loader"] = fake_loader
    try:
        assert "no API key" in (voice_config.transcription_gap() or "")
        _Eff.configured = True
        assert voice_config.transcription_gap() is None
        _Eff.enabled = False
        assert "disabled" in (voice_config.transcription_gap() or "")
        fake_tr.resolve_transcription_config = None  # a core without the seam
        assert voice_config.transcription_gap() is None
    finally:
        for name, mod in zip(
            ("nanobot.audio.transcription", "nanobot.config.loader"), real, strict=True
        ):
            if mod is not None:
                sys.modules[name] = mod
            else:
                del sys.modules[name]


def test_unified_session_reads_core_and_tolerates_a_core_without_it(tmp_path, monkeypatch):
    """``agents.defaults.unifiedSession`` exactly as core's own loader reads it, core's
    default without a file; a loader that fails reads as off, never as a false alarm."""
    import nanobot.config.loader as loader

    import nanobot_channel_voice.config as voice_config

    path = tmp_path / "config.json"
    monkeypatch.setattr(loader, "get_config_path", lambda: path)
    assert voice_config.unified_session() is False  # no config file: core's default
    path.write_text('{"agents": {"defaults": {"unifiedSession": true}}}')
    assert voice_config.unified_session() is True
    path.write_text('{"agents": {"defaults": {"unifiedSession": false}}}')
    assert voice_config.unified_session() is False

    def unreadable():
        raise RuntimeError("config.json is not valid JSON")

    monkeypatch.setattr(loader, "load_config", unreadable)
    assert voice_config.unified_session() is False


def test_the_model_index_is_https_or_a_file_named_absolutely(monkeypatch):
    """``index`` names where the on-device models come from, the built-in index unless
    set, empty for none; each entry an https URL or a file on this machine, named
    absolutely (the gateway and a shell resolve a relative path differently) and never
    plain http, the index pinning every model's hash. A board's defaults can carry it,
    and the readers that need the index alone check it as the schema does."""
    import json

    from nanobot_channel_voice import weights as w
    from nanobot_channel_voice.config import resolve_section, section_index

    assert VoiceConfig().index == list(w.DEFAULT_INDEX_SOURCES)
    for fine in (
        ["https://hf-mirror.com/o/r/resolve/main/weights-index.json"],
        ["file:///mnt/usb/weights-index.json", "/srv/local.json", "~/idx.json"],
        ["http://127.0.0.1:8000/i.json"],
        [],
    ):
        assert VoiceConfig.model_validate({"index": fine}).index == fine
    for refused, why in (
        (["http://mirror.lan/i.json"], "index entry 'http://mirror.lan/i.json': an index must be https or a file"),
        (["weights-index.json"], "is a relative path, name the file absolutely"),
        ([""], "index has an empty entry"),
        (["ftp://a.example/i.json"], "'ftp' is not an index scheme"),
    ):
        with pytest.raises(ValidationError, match=why):
            VoiceConfig.model_validate({"index": refused})
    mirror = ["https://hf-mirror.com/o/r/resolve/main/weights-index.json"]
    monkeypatch.setenv("NANOBOT_VOICE_DEFAULTS", json.dumps({"index": mirror}))
    assert VoiceConfig().index == mirror
    assert section_index(resolve_section({})[0]) == mirror
    assert section_index(resolve_section({"index": []})[0]) == []  # the section's own word stands
    monkeypatch.delenv("NANOBOT_VOICE_DEFAULTS")
    assert section_index({}) == list(w.DEFAULT_INDEX_SOURCES)
    with pytest.raises(ValueError, match="index must be a list"):
        section_index({"index": "https://a.example/i.json"})
    with pytest.raises(ValueError, match="must be https or a file"):
        section_index({"index": ["http://mirror.lan/i.json"]})
