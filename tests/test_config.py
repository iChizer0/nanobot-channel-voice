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


def test_gemini_is_not_a_valid_backend():
    with pytest.raises(ValidationError):
        VoiceConfig.model_validate({"backend": "gemini"})


def test_backend_accepts_exactly_the_dialect_family():
    for name in ("local", "openai", "xai", "azure", "qwen", "glm", "stepfun"):
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


def test_import_json_empty_filler_is_ignored():
    """Core's enable toggle materializes secret defaults as ''; that filler (and an
    emptied box) must neither merge nor mark an import as pending."""
    assert VoiceConfig.model_validate({"importJson": ""}).import_json is None
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
