"""The nanobot >= 0.3.0 channel-package shim (shim/nanobot/channels/voice).

v0.3.0 dropped the ``nanobot.channels`` entry-point group; channels are
discovered as subpackages of ``nanobot.channels`` with a dependency-free
``manifest.py``, which is this plugin's ONLY registration path (hence the
``nanobot-ai>=0.3.0`` floor: a missing registry here is a broken install, not
a skip). The shim is validated by FILE PATH, so these run whether or not the
wheel is installed.
"""

from __future__ import annotations

import importlib.util
import pathlib

from nanobot.channels import plugin as plugin_mod

_SHIM = pathlib.Path(__file__).resolve().parents[1] / "shim" / "nanobot" / "channels" / "voice"


def _load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_manifest_declares_a_valid_channel_package():
    plugin = _load("voice_shim_manifest", _SHIM / "manifest.py").PLUGIN
    assert isinstance(plugin, plugin_mod.ChannelPlugin)  # __post_init__ already validated it
    assert plugin.name == "voice"
    # load_channel_package enforces that the runtime target stays inside the
    # channel package; drifting from this breaks discovery on >= 0.3.0.
    assert plugin.runtime.startswith("nanobot.channels.voice.")


def test_runtime_shim_reexports_the_channel():
    mod = _load("voice_shim_runtime", _SHIM / "runtime.py")
    from nanobot_channel_voice.channel import VoiceChannel

    assert mod.VoiceChannel is VoiceChannel
    assert VoiceChannel.name == "voice"  # load_channel_class checks this match


def test_setup_spec_is_import_only():
    """The WebUI surface is ONE write-only paste box; the full rationale lives on
    the manifest's SETUP_SPEC comment, which this pin protects."""
    plugin = _load("voice_shim_manifest", _SHIM / "manifest.py").PLUGIN
    spec = plugin.setup
    assert spec is not None
    assert list(spec.fields) == ["importJson"]
    # importJson is secret-kind although it's not a credential: the paste may CONTAIN
    # credentials, and secret is the one kind core never echoes back to a browser.
    assert spec.secrets == frozenset({"importJson"})
    # No required fields: the validator is authoritative, and a bare section
    # must not read as needs_setup (tier-0 needs nothing).
    assert spec.required == ()
    public = spec.to_public_dict("voice")
    assert [f["key"] for f in public["fields"]] == ["channels.voice.importJson"]


def test_enable_toggle_materialization_stays_allow_everyone():
    """The deny-everyone hazard the SETUP_SPEC comment describes: an undeclared
    allowFrom must stay unmaterialized by core's toggle, leaving the schema
    default ["*"] to govern is_allowed."""
    from nanobot.channels.contracts import channel_default_config

    from nanobot_channel_voice.config import VoiceConfig

    plugin = _load("voice_shim_manifest", _SHIM / "manifest.py").PLUGIN
    materialized = channel_default_config(plugin)
    # The toggle writes only what the spec declares (+ enabled); the paste box's
    # '' filler is stripped again by the config layer at parse time.
    assert set(materialized) == {"enabled", "importJson"}
    cfg = VoiceConfig.model_validate({**materialized, "enabled": True})
    assert cfg.allow_from == ["*"]
    assert cfg.import_json is None


def test_setup_validator_reports_plugin_schema_errors():
    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    good = manifest._validate({"enabled": True, "backend": "local"}, ctx)
    assert all(c.get("status") != "fail" for c in good.get("checks", []))
    bad = manifest._validate({"enabled": True, "backend": "gemini"}, ctx)
    fails = [c for c in bad.get("checks", []) if c.get("status") == "fail"]
    assert fails and "backend" in fails[0].get("message", "")


def _check_ids(payload):
    return {c["id"]: c for c in payload["checks"]}


def test_setup_validator_nudges_config_file_keys(monkeypatch):
    """Every key is config-file surface (the form is one paste box), so the
    validator carries the credential guidance as non-blocking 'skipped' notes
    naming the file keys."""
    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # local backend + default cloud TTS + no key anywhere -> nudge
    out = manifest._validate({"enabled": True}, ctx)
    assert _check_ids(out)["tts_key"]["status"] == "skipped"
    assert "tts.apiKey" in _check_ids(out)["tts_key"]["message"]
    # any of a key / an apiBase / the env silences it; none of them block enabling
    for values in (
        {"tts": {"apiKey": "sk-x"}},
        {"tts": {"apiBase": "http://localhost:8880/v1"}},
        {"tts": {"enabled": False}},
        {"tts": {"provider": "system"}},
    ):
        assert "tts_key" not in _check_ids(manifest._validate(values, ctx))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert "tts_key" not in _check_ids(manifest._validate({"enabled": True}, ctx))

    # cloud backend without a key -> realtime nudge; a real key silences it, and the
    # env-export alternative is offered only where an OpenAI key would work
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = manifest._validate({"backend": "xai"}, ctx)
    assert _check_ids(out)["realtime_key"]["status"] == "skipped"
    assert "OPENAI_API_KEY" not in _check_ids(out)["realtime_key"]["message"]
    assert "OPENAI_API_KEY" in _check_ids(
        manifest._validate({"backend": "openai"}, ctx)
    )["realtime_key"]["message"]
    assert "tts_key" not in _check_ids(out)  # tts is local-backend-only guidance
    assert "realtime_key" not in _check_ids(
        manifest._validate({"backend": "xai", "realtime": {"apiKey": "k"}}, ctx)
    )
    assert out["can_enable"] is True  # notes never gate Check-and-enable

    # the env fallback is NOT silence for a non-OpenAI backend: start() would send
    # the OpenAI key to a provider that rejects it, and the one row must say so
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    out = manifest._validate({"backend": "xai"}, ctx)
    message = _check_ids(out)["realtime_key"]["message"]
    assert "OPENAI_API_KEY" in message and "reject" in message
    assert "realtime_key" not in _check_ids(manifest._validate({"backend": "openai"}, ctx))
    assert "realtime_key" not in _check_ids(
        manifest._validate({"backend": "xai", "realtime": {"apiKey": "k"}}, ctx)
    )


def test_setup_validator_is_backend_aware(monkeypatch):
    """The form is one paste box whatever the backend, so the validator rows are
    the only surface that can say which keys the chosen backend ignores and what
    the section resolved to (backend, engines, devices)."""
    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")  # silence the key nudges

    # local: the schema row names the backend, the pipeline check the resolved
    # engine trio + the file home, the devices row the PCMs + the export command
    out = manifest._validate({"enabled": True}, ctx)
    assert "backend='local'" in _check_ids(out)["schema"]["message"]
    pipeline = _check_ids(out)["pipeline"]
    assert pipeline["status"] == "pass"
    for expected in ("vad.engine='energy'", "stt.provider='nanobot'", "config.json"):
        assert expected in pipeline["message"]
    assert "realtime_unused" not in _check_ids(out)
    assert "local_unused" not in _check_ids(out)
    devices = _check_ids(manifest._validate({"audio": {"captureDevice": "plug:mic"}}, ctx))
    assert "plug:mic" in devices["manual_review"]["message"]
    assert "nanobot-voice config" in devices["manual_review"]["message"]

    # local + ANY non-default realtime.* value -> flagged as ignored (a misplaced
    # edit); default-compared, so non-credential knobs are covered too
    out = manifest._validate({"realtime": {"model": "gpt-realtime"}}, ctx)
    assert _check_ids(out)["realtime_unused"]["status"] == "skipped"
    out = manifest._validate({"realtime": {"toolMode": "supervisor"}}, ctx)
    assert _check_ids(out)["realtime_unused"]["status"] == "skipped"

    # local + an engine that cannot build -> the pipeline check degrades to warn
    out = manifest._validate({"vad": {"engine": "firered"}}, ctx)
    pipeline = _check_ids(out)["pipeline"]
    assert pipeline["status"] == "warn"
    assert "fall back" in pipeline["message"]
    assert "vad.firered.modelPath" in pipeline["message"]
    assert out["can_enable"] is True  # warn stays non-blocking

    # cloud: no pipeline chatter, but any configured local-only block is flagged
    # unused, the row NAMING the touched blocks
    out = manifest._validate({"backend": "openai", "stt": {"provider": "whisper"}}, ctx)
    assert "backend='openai'" in _check_ids(out)["schema"]["message"]
    assert "pipeline" not in _check_ids(out)
    assert _check_ids(out)["local_unused"]["status"] == "skipped"
    assert "stt" in _check_ids(out)["local_unused"]["message"]
    out = manifest._validate({"backend": "openai", "prologue": {"enabled": True}}, ctx)
    assert "prologue" in _check_ids(out)["local_unused"]["message"]
    assert "local_unused" not in _check_ids(manifest._validate({"backend": "openai"}, ctx))

    # azure is the one profile with no default endpoint
    out = manifest._validate({"backend": "azure"}, ctx)
    assert "realtime.baseUrl" in _check_ids(out)["realtime_endpoint"]["message"]
    assert "realtime_endpoint" not in _check_ids(
        manifest._validate(
            {"backend": "azure", "realtime": {"baseUrl": "wss://r.openai.azure.com/x"}}, ctx
        )
    )


def test_setup_validator_lints_the_import_paste(monkeypatch):
    """The importJson box is write-only (secret-kind), so the Check button is the
    only pre-restart feedback: a good paste gets its own row saying what will
    happen to it, a bad paste fails the schema check with the parse error."""
    import json

    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    paste = json.dumps({"vad": {"hangoverMs": 800}})
    out = manifest._validate({"enabled": True, "importJson": paste}, ctx)
    assert _check_ids(out)["import"]["status"] == "pass"
    assert "expanded" in _check_ids(out)["import"]["message"]
    assert "import" not in _check_ids(manifest._validate({"enabled": True}, ctx))

    bad = manifest._validate({"importJson": "{not json"}, ctx)
    schema = _check_ids(bad)["schema"]
    assert schema["status"] == "fail"
    assert "not valid JSON" in schema["message"]
    # a paste that parses but breaks the schema is linted by the full plugin schema
    worse = manifest._validate({"importJson": json.dumps({"vad": {"engine": "nope"}})}, ctx)
    assert _check_ids(worse)["schema"]["status"] == "fail"


def test_setup_validator_stays_within_the_rendered_check_budget(monkeypatch):
    """Core's WebUI renders only the first 6 checks; the worst case of either
    branch must fit or later checks silently vanish."""
    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    local_worst = manifest._validate(
        {"vad": {"engine": "firered"}, "realtime": {"model": "m"}, "importJson": "{}"}, ctx
    )
    cloud_worst = manifest._validate(
        {"backend": "azure", "stt": {"provider": "whisper"}, "importJson": "{}"}, ctx
    )
    assert len(local_worst["checks"]) <= 6
    assert len(cloud_worst["checks"]) <= 6
    # the env-fed non-OpenAI key variant REPLACES the no-key row, never adds one
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    cloud_env_worst = manifest._validate(
        {"backend": "azure", "stt": {"provider": "whisper"}, "importJson": "{}"}, ctx
    )
    assert len(cloud_env_worst["checks"]) <= 6
