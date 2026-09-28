"""The nanobot channel-package shim (shim/nanobot/channels/voice).

Channels are discovered as subpackages of ``nanobot.channels`` with a
dependency-free ``manifest.py``, which is this plugin's ONLY registration path
(a missing registry here is a broken install, not a skip). The manifest declares
the ``json`` field kind and a validator whose rows the WebUI shows, which set the
``nanobot-ai>=0.3.5`` floor. The shim is validated by FILE PATH, so these run
whether or not the wheel is installed.
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
    # load_channel_package enforces that the runtime AND connector targets stay inside
    # the channel package; drifting from this breaks discovery on >= 0.3.0.
    assert plugin.runtime.startswith("nanobot.channels.voice.")
    assert plugin.connector.startswith("nanobot.channels.voice.")


def test_runtime_shim_reexports_the_channel():
    mod = _load("voice_shim_runtime", _SHIM / "runtime.py")
    from nanobot_channel_voice.channel import VoiceChannel

    assert mod.VoiceChannel is VoiceChannel
    assert VoiceChannel.name == "voice"  # load_channel_class checks this match


def test_connect_shim_reexports_the_sync_connector():
    mod = _load("voice_shim_connect", _SHIM / "connect.py")
    from nanobot_channel_voice.webui_sync import VoiceSyncStore

    assert mod.VoiceSyncStore is VoiceSyncStore
    assert callable(getattr(VoiceSyncStore(), "handle"))  # load_connector's one check


def test_manifest_logo_is_an_inline_image_where_core_can_show_one():
    """The Settings tile: a wheel-installed channel gets no compiled icon, so the manifest
    carries one inline. Only cores whose ChannelPlugin has ``logo_url`` receive it; the
    official 0.3.5 must still construct the plugin (and show initials)."""
    import base64
    import dataclasses

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    svg = manifest._LOGO_SVG
    assert svg.startswith("<svg ") and "<script" not in svg
    supported = any(f.name == "logo_url" for f in dataclasses.fields(plugin_mod.ChannelPlugin))
    if not supported:
        assert manifest._LOGO == {} and not hasattr(manifest.PLUGIN, "logo_url")
        return
    prefix = "data:image/svg+xml;base64,"
    assert manifest.PLUGIN.logo_url.startswith(prefix)
    assert base64.b64decode(manifest.PLUGIN.logo_url[len(prefix):]).decode() == svg


def test_manifest_names_the_panel_only_where_core_shipped_it():
    """The voice panel lives in core's tree (nanobot/channels/voice/webui/) and activates
    on the exact `webui` path the manifest returns; a path this core does not ship would
    make discovery refuse the manifest for a missing entry."""
    from importlib.resources import files

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    webui = manifest.PLUGIN.webui
    assert webui in (None, "webui/index.tsx")
    if webui is not None:  # named: it must resolve next to this shim, or discovery refuses
        assert files("nanobot.channels").joinpath("voice", *webui.split("/")).is_file()


def test_setup_spec_is_import_only():
    """The WebUI surface is ONE paste box, echoed while pending; the full rationale lives
    on the manifest's SETUP_SPEC comment, which this pin protects."""
    plugin = _load("voice_shim_manifest", _SHIM / "manifest.py").PLUGIN
    spec = plugin.setup
    assert spec is not None
    assert list(spec.fields) == ["importJson"]
    # json kind: the WebUI's textarea, checked for a JSON object at save time, persisted
    # as an object and shown while pending — the echo trade-off the SETUP_SPEC comment
    # takes knowingly, so no secret.
    assert spec.fields["importJson"].kind == "json"
    assert spec.secrets == frozenset()
    # The box renders up front (payload `required`) yet nothing is required: no
    # requirement means neither core's validator nor the >= 0.3.5 browser gate can block
    # enabling a bare section, and start() deleting a consumed paste never strands a
    # re-enable. Bare-enable is pinned below via can_enable.
    assert spec.required == ()
    public = spec.to_public_dict("voice")
    assert [f["key"] for f in public["fields"]] == ["channels.voice.importJson"]
    assert public["fields"][0]["required"] is True
    assert public.get("requirements", []) == []
    # No Check button: the generic pane then shows only the validator's `message` (see
    # the failure test), and the voice panel reads the checks itself.
    assert "verifies_connection" not in public


def test_enable_toggle_materialization_stays_allow_everyone():
    """The deny-everyone hazard the SETUP_SPEC comment describes: an undeclared
    allowFrom must stay unmaterialized by core's toggle, leaving the schema
    default ["*"] to govern is_allowed."""
    from nanobot.channels.contracts import channel_default_config

    from nanobot_channel_voice.config import VoiceConfig

    plugin = _load("voice_shim_manifest", _SHIM / "manifest.py").PLUGIN
    materialized = channel_default_config(plugin)
    # The toggle writes only what the spec declares (+ enabled); a json field has no
    # filler default, so the section stays bare.
    assert set(materialized) == {"enabled"}
    cfg = VoiceConfig.model_validate({**materialized, "enabled": True})
    assert cfg.allow_from == ["*"]
    assert cfg.import_json is None


def test_setup_validator_reports_plugin_schema_errors():
    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    good = manifest._validate({"enabled": True, "backend": "local"}, ctx)
    assert all(c.get("status") != "fail" for c in good.get("checks", []))
    bad = manifest._validate({"enabled": True, "backend": "nova"}, ctx)
    fails = [c for c in bad.get("checks", []) if c.get("status") == "fail"]
    # pydantic's reason under the key it names, without the type/input/URL trailer
    assert fails and fails[0]["message"].startswith("backend: Input should be 'local', 'openai'")
    assert "input_value" not in fails[0]["message"] and "For further information" not in fails[0]["message"]
    gated = manifest._validate({"backend": "openai", "realtime": {"uplink": "vad"}}, ctx)
    assert gated["message"].startswith('realtime.uplink="vad" needs a neural VAD')
    empty = manifest._validate({"allowFrom": []}, ctx)
    assert empty["message"].startswith("allowFrom is empty")  # the key in the reason is not repeated
    # core's generic pane surfaces only `message` on a refused enable: it carries the detail
    assert bad["message"] == fails[0]["message"] and not bad["can_enable"]
    assert good["form"]["sections"][0]["fields"][0]["key"] == "backend"

    # A refused section still shapes a form, leniently: what validates of it shapes the
    # fields, a refused part falls back to its default, so the edit that broke it (or the
    # key an older config carries) is fixed in place instead of freezing the panel.
    def shape(values):
        payload = manifest._validate(values, ctx)
        assert not payload["can_enable"]
        return [s["id"] for s in payload["form"]["sections"]], [f["key"] for s in payload["form"]["sections"] for f in s["fields"]]

    assert shape({"backend": "nova"})[0][1] == "audio"  # the refused field: its default shapes
    ids, keys = shape({"enabled": True, "importJson": {"wake": {"mode": "gate"}}})
    # a gate without phrases: the block keeps its mode, so the Model row that fills the
    # phrases is there, and nothing that needs the engine
    assert keys[keys.index("wake.mode"):keys.index("wake.attention")] == ["wake.mode", "wake.openwakeword.weights", "wake.phrases"]
    payload = manifest._validate({"enabled": True, "importJson": {"wake": {"mode": "gate", "engine": "openwakeword", "openwakeword": {"threshold": 7}}}}, ctx)
    fields = {f["key"]: f for s in payload["form"]["sections"] for f in s["fields"]}
    assert fields["wake.openwakeword.threshold"]["value"] == 0.5  # the refused leaf at its default, the rest kept
    ids, _ = shape({"enabled": True, "legacyKey": 1, "importJson": {"backend": "openai"}})
    assert ids == ["general", "provider", "audio", "stt", "vad", "wake", "access"]  # an old config's stray key, backend switched
    ids, _ = shape({"stt": {"whisper": {"bogus": 1}}, "importJson": {"backend": "gemini", "realtime": {"uplink": "vad"}}})
    assert ids == ["general", "provider", "audio", "stt", "vad", "wake", "access"]  # cross-field refusal: Listening is there to fix it
    assert "provider" in shape({"backend": "openai", "importJson": "{not json"})[0]  # an unusable paste: the saved shape


def _check_ids(payload):
    return {c["id"]: c for c in payload["checks"]}


def test_setup_validator_nudges_by_form_label(monkeypatch):
    """The credential guidance rides non-blocking 'skipped' rows under the form, so a
    row names the section and the row to fill in, never a config key."""
    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # local backend + default cloud TTS + no key anywhere -> nudge
    out = manifest._validate({"enabled": True}, ctx)
    assert _check_ids(out)["tts_key"]["status"] == "skipped"
    assert _check_ids(out)["tts_key"]["message"].startswith("OpenAI text-to-speech has no API key. Fill it in under Text-to-speech")
    assert "tts.apiKey" not in _check_ids(out)["tts_key"]["message"]
    # any of a key / an apiBase / the env silences it; none of them block enabling
    for values in (
        {"tts": {"apiKey": "sk-x"}},
        {"tts": {"apiBase": "http://localhost:8880/v1"}},
        {"tts": {"enabled": False}},
        {"tts": {"provider": "system"}},
    ):
        assert "tts_key" not in _check_ids(manifest._validate(values, ctx))
    # the other OpenAI dialect runs the same builder, which raises keyless too
    compat = _check_ids(manifest._validate({"tts": {"provider": "openai_compat"}}, ctx))
    assert compat["tts_key"]["message"].startswith("OpenAI-compatible text-to-speech has no API key")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    assert "tts_key" not in _check_ids(manifest._validate({"enabled": True}, ctx))

    # cloud backend without a key -> realtime nudge; a real key silences it, and the
    # env-export alternative is offered only where an OpenAI key would work
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = manifest._validate({"backend": "xai"}, ctx)
    assert _check_ids(out)["realtime_key"]["status"] == "skipped"
    assert "OPENAI_API_KEY" not in _check_ids(out)["realtime_key"]["message"]
    assert _check_ids(manifest._validate({"backend": "openai"}, ctx))["realtime_key"]["message"] == (
        "OpenAI needs an API key. Fill it in under Provider or export OPENAI_API_KEY in the gateway environment."
    )
    assert _check_ids(out)["realtime_key"]["message"] == "xAI needs an API key. Fill it in under Provider."
    assert "tts_key" not in _check_ids(out)  # tts is local-backend-only guidance
    assert "realtime_key" not in _check_ids(
        manifest._validate({"backend": "xai", "realtime": {"apiKey": "k"}}, ctx)
    )
    assert out["can_enable"] is True  # notes never gate the toggle

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

    # an open mic on the cloud path needs a real canceller (start() refuses otherwise):
    # the row names the two rows that fix it, both advanced, and any of the start-time
    # outs silences it (the WebRTC one with its binding, pinned installed here)
    monkeypatch.setattr(manifest, "find_spec", lambda _name: object())
    out = manifest._validate({"backend": "openai", "realtime": {"bargeIn": "aec"}}, ctx)
    assert _check_ids(out)["realtime_aec"]["message"] == (
        "Open mic barge-in needs echo cancellation. Under Provider in Advanced, set Echo "
        "cancellation to WebRTC or Hardware, or Barge-in to Gated."
    )
    for values in (
        {"backend": "openai", "realtime": {"bargeIn": "aec"}, "aec": "webrtc"},
        {"backend": "openai", "realtime": {"bargeIn": "aec"}, "aec": "hardware"},
        {"backend": "openai", "realtime": {"bargeIn": "aec", "aecAvailable": True}},
        {"backend": "openai", "realtime": {"bargeIn": "gated"}, "aec": "soft"},
        {"backend": "local", "aec": "soft"},
    ):
        assert "realtime_aec" not in _check_ids(manifest._validate(values, ctx)), values


def test_setup_validator_is_backend_aware(monkeypatch, tmp_path):
    """The rows under the form are the Resolved setup: what the section resolved to
    (backend, engines, devices), and which saved settings the chosen backend ignores."""
    from nanobot.channels.contracts import ChannelValidationContext

    import nanobot_channel_voice.config as voice_config
    import nanobot_channel_voice.tts.system as system_tts

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")  # silence the key nudges
    # The pipeline row reads the LIVE install's transcription config and PATH; pin both so
    # the rows below describe the section under test, not this machine.
    monkeypatch.setattr(voice_config, "transcription_gap", lambda: None)
    monkeypatch.setattr(system_tts, "system_voice", lambda *_a: ("espeak", "/usr/bin/espeak-ng"))

    # local: no schema pass row (the identity line names the backend), the pipeline
    # check the resolved engine trio, the devices row the PCMs; options read by label,
    # never config value
    out = manifest._validate({"enabled": True}, ctx)
    assert "schema" not in _check_ids(out)
    assert out["can_enable"] is True  # a note never gates the toggle
    pipeline = _check_ids(out)["pipeline"]
    assert pipeline["status"] == "pass"
    assert pipeline["message"] == "Energy, Internal, OpenAI."
    assert "realtime_unused" not in _check_ids(out)
    assert "local_unused" not in _check_ids(out)
    # NOT core's reserved `manual_review` id: the WebUI hides that one, and this row is
    # what makes the summary line's "not verified" honest
    devices = _check_ids(manifest._validate({"audio": {"captureDevice": "plug:mic"}}, ctx))
    assert devices["audio_devices"]["message"] == (
        "The microphone 'plug:mic' and speaker 'default' are opened when the channel starts. Change them under Audio in Advanced."
    )
    assert devices["audio_devices"]["status"] == "skipped"
    assert out["status"] == "configured"  # the WebUI's "not verified", never "connected"
    # the identity line is the resolved backend and engines ("Local · Energy, Internal,
    # OpenAI" as core joins it): the pass rows show no message
    assert out["identity"] == {"name": "Local", "workspace": "Energy, Internal, OpenAI"}
    assert manifest._validate({"tts": {"enabled": False}}, ctx)["identity"]["workspace"] == (
        "Energy, Internal, no TTS"
    )
    # the panel's pending patch rides importJson and says so itself: no row for it
    assert "import" not in _check_ids(manifest._validate({"importJson": {"vad": {"hangoverMs": 800}}}, ctx))

    # local + ANY non-default realtime.* value -> flagged as ignored (a misplaced
    # edit); default-compared, so non-credential knobs are covered too
    out = manifest._validate({"realtime": {"model": "gpt-realtime"}}, ctx)
    assert _check_ids(out)["realtime_unused"]["status"] == "skipped"
    out = manifest._validate({"realtime": {"toolMode": "supervisor"}}, ctx)
    assert _check_ids(out)["realtime_unused"]["status"] == "skipped"

    # local + an engine without a model -> the pipeline check degrades to warn, naming
    # the section whose Model row fixes it, where the panel shows it, and what stands in
    # until then
    out = manifest._validate({"vad": {"engine": "firered"}}, ctx)
    pipeline = _check_ids(out)["pipeline"]
    assert pipeline["status"] == "warn"
    assert pipeline["message"] == "FireRed has no model, pick one under Listening in Advanced. Until then Energy listens."
    assert out["can_enable"] is True  # warn stays non-blocking
    both = {"vad": {"engine": "firered", "turn": {"engine": "smartturn"}}, "tts": {"provider": "matcha"}}
    assert _check_ids(manifest._validate(both, ctx))["pipeline"]["message"].endswith(
        "Until then Energy listens, turns end on silence alone and System speaks."
    )
    # serving that same engine has no stand-in: _start_stt_server raises instead
    served = {"stt": {"provider": "whisper", "serve": {"enabled": True}}}
    pipeline = _check_ids(manifest._validate(served, ctx))["pipeline"]
    assert pipeline["status"] == "skipped"
    assert pipeline["message"] == (
        "Whisper has no model, pick one under Speech-to-text. Serve transcription is on, so "
        "the channel does not start without Whisper."
    )

    # stt.provider='nanobot' with nothing behind it decodes every utterance to "",
    # which the pipeline cannot tell from silence: the check must say so, not pass
    monkeypatch.setattr(voice_config, "transcription_gap", lambda: "there is no API key")
    pipeline = _check_ids(manifest._validate({"enabled": True}, ctx))["pipeline"]
    assert pipeline["status"] == "warn"
    assert pipeline["message"] == (
        "Internal delegates to nanobot's transcription, but there is no API key, so every "
        "utterance would be heard as silence."
    )
    # an on-device engine that loads takes core transcription off the path; one that falls
    # back puts Internal on it, which hears nothing either
    paths = ("encoderPath", "decoderPath", "vocabPath", "melFiltersPath")
    loads = {"stt": {"provider": "whisper", "whisper": {p: f"/m/{p}" for p in paths}}}
    assert "silence" not in _check_ids(manifest._validate(loads, ctx))["pipeline"]["message"]
    assert _check_ids(manifest._validate({"stt": {"provider": "sensevoice"}}, ctx))["pipeline"]["message"] == (
        "SenseVoice has no model, pick one under Speech-to-text. Until then Internal transcribes. "
        "Internal delegates to nanobot's transcription, but there is no API key, so every "
        "utterance would be heard as silence."
    )
    monkeypatch.setattr(voice_config, "transcription_gap", lambda: None)

    # unfetched weights keys warn as one sentence, with Apply as the remedy when the
    # cached index lists every one of them (the Models section says what Apply moves); with
    # no index cached yet (a first sync that failed), once the panel has loaded one
    monkeypatch.setenv("NANOBOT_VOICE_MODELS_DIR", str(tmp_path / "empty-store"))
    section = {"vad": {"engine": "firered", "firered": {"weights": "vad/firered/onnx"}}}
    message = _check_ids(manifest._validate(section, ctx))["pipeline"]["message"]
    assert message == (
        "The FireRed model is not fetched yet, Apply downloads it once the model index has "
        "loaded. Until then Energy listens."
    )
    import json as _json

    from nanobot_channel_voice import weights as w

    entry = {"files": {"model.onnx": {"url": "https://x/m.onnx", "sha256": "0" * 64, "size": 1}}}
    cache = tmp_path / "empty-store" / w.INDEX_CACHE
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(_json.dumps({"models": {"vad/firered/onnx": entry, "tts/matcha/en/onnx": entry}}))
    section["tts"] = {"provider": "matcha", "matcha": {"weights": "tts/matcha/en/onnx"}}
    message = _check_ids(manifest._validate(section, ctx))["pipeline"]["message"]
    assert message == (
        "The FireRed and Matcha models are not fetched yet, Apply downloads them. "
        "Until then Energy listens and System speaks."
    )
    # a notice in the index holds Apply until it is accepted under Models: the remedy
    # names the model whose notice it is, one or several, never a download the button
    # refuses
    noticed = {**entry, "license": "CC-BY-NC-SA-4.0", "accept": "non-commercial use only"}
    cache.write_text(_json.dumps({"models": {"vad/firered/onnx": entry, "tts/matcha/en/onnx": noticed}}))
    message = _check_ids(manifest._validate(section, ctx))["pipeline"]["message"]
    assert message.startswith(
        "The FireRed and Matcha models are not fetched yet, Apply downloads them once the "
        "Matcha notice under Models is accepted. Until then"
    )
    cache.write_text(_json.dumps({"models": {"vad/firered/onnx": noticed, "tts/matcha/en/onnx": noticed}}))
    assert "once the FireRed and Matcha notices under Models are accepted." in _check_ids(
        manifest._validate(section, ctx)
    )["pipeline"]["message"]
    one = {"tts": {"provider": "matcha", "matcha": {"weights": "tts/matcha/en/onnx"}}}
    assert _check_ids(manifest._validate(one, ctx))["pipeline"]["message"] == (
        "The Matcha model is not fetched yet, Apply downloads it once its notice under Models "
        "is accepted. Until then System speaks."
    )
    section["tts"]["matcha"]["weights"] = "tts/matcha/mine/onnx"  # one of them unlisted: no Apply
    assert _check_ids(manifest._validate(section, ctx))["pipeline"]["message"].startswith(
        "The FireRed and Matcha models are not fetched yet. Until then"
    )

    # cloud: no pipeline chatter, but any configured local-only block is flagged
    # unused, the row NAMING the touched blocks
    out = manifest._validate({"backend": "openai", "stt": {"provider": "whisper"}}, ctx)
    assert "schema" not in _check_ids(out)
    assert out["identity"] == {"name": "OpenAI", "workspace": "gpt-realtime"}  # the profile default
    assert manifest._validate({"backend": "xai", "realtime": {"model": "grok-x"}}, ctx)[
        "identity"
    ] == {"name": "xAI", "workspace": "grok-x"}
    assert manifest._validate({"backend": "gemini"}, ctx)["identity"]["workspace"].startswith(
        "gemini-"
    )
    assert "pipeline" not in _check_ids(out)
    assert _check_ids(out)["local_unused"]["status"] == "skipped"
    assert "stt" in _check_ids(out)["local_unused"]["message"]
    out = manifest._validate({"backend": "openai", "prologue": {"enabled": True}}, ctx)
    assert "prologue" in _check_ids(out)["local_unused"]["message"]
    assert "local_unused" not in _check_ids(manifest._validate({"backend": "openai"}, ctx))

    # azure is the one profile with no default endpoint
    out = manifest._validate({"backend": "azure"}, ctx)
    assert _check_ids(out)["realtime_endpoint"]["message"] == (
        "Azure OpenAI has no default endpoint. Fill in the Endpoint under Provider with your resource URL."
    )
    assert "realtime_endpoint" not in _check_ids(
        manifest._validate(
            {"backend": "azure", "realtime": {"baseUrl": "wss://r.openai.azure.com/x"}}, ctx
        )
    )


def test_setup_validator_unused_row_follows_what_the_cloud_path_reads(monkeypatch):
    """The gated uplink runs vad.* and, under uplink="wake", wake.*; stt.serve loads stt.*;
    the gate reads bargeIn.duckStartFrames. The row must not call a required block "not
    used"."""
    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    cloud = {"backend": "openai"}
    vad = {"engine": "silero", "silero": {"modelPath": "/x.onnx"}}
    wake = {"mode": "gate", "phrases": ["hey"], "engine": "openwakeword"}

    def unused(section):
        row = _check_ids(manifest._validate(section, ctx)).get("local_unused")
        return row["message"].split("(")[1].split(")")[0] if row else ""

    assert unused({**cloud, "vad": vad}) == "vad"  # server uplink: local-only indeed
    assert unused({**cloud, "realtime": {"uplink": "vad"}, "vad": vad}) == ""
    # uplink="vad" drops the wake detector, so a configured wake.* IS dead there
    assert unused({**cloud, "realtime": {"uplink": "vad"}, "vad": vad, "wake": wake}) == "wake"
    assert unused({**cloud, "realtime": {"uplink": "wake"}, "vad": vad, "wake": wake}) == ""
    # the rest of the local pipeline stays flagged under a gate
    gated = {**cloud, "realtime": {"uplink": "vad"}, "vad": vad}
    assert unused({**gated, "prologue": {"enabled": True}}) == "prologue"
    assert unused({**gated, "bargeIn": {"duckStartFrames": 3}}) == ""
    assert unused({**cloud, "bargeIn": {"duckStartFrames": 3}}) == "bargeIn"
    assert unused({**cloud, "bargeIn": {"stopPhrases": ["halt"]}}) == ""  # cloud-read
    # stt.serve borrows the on-device STT under a cloud backend
    serve = {"provider": "whisper", "serve": {"enabled": True}}
    assert unused({**cloud, "stt": serve}) == ""
    assert unused({**cloud, "stt": {"provider": "whisper"}}) == "stt"
    # the chip every board names reaches each engine block, but it configures none of them
    assert unused({**cloud, "device": "rv1126b"}) == ""
    assert unused({**cloud, "device": "rv1126b", "vad": vad}) == "vad"


def test_setup_validator_gate_row_mirrors_the_cloud_start(monkeypatch, tmp_path):
    """A gated uplink's on-device engines get the pipeline row's treatment on the cloud
    path, where channel._build_gate refuses to start on a detector that did not load
    (nothing stands in for it: a fallback would upload on any noise, or never) and Smart
    Turn alone degrades. The row is named after the gate picked, and there is none while
    everything loads."""
    import json

    from nanobot.channels.contracts import ChannelValidationContext

    from nanobot_channel_voice import weights as w

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setenv("NANOBOT_VOICE_MODELS_DIR", str(tmp_path / "store"))
    cloud = {"backend": "openai"}
    silero = {"engine": "silero", "silero": {"modelPath": "/v.onnx"}}
    head = {"modelPath": "/h.onnx", "embeddingPath": "/e.onnx", "melPath": "/m.onnx"}
    wake = {"mode": "gate", "engine": "openwakeword", "phrases": ["hey"]}

    def gate(section):
        return _check_ids(manifest._validate(section, ctx)).get("gate")

    assert gate({**cloud, "vad": silero}) is None  # Continuous runs nothing on the device
    # ...unless transcription is served: _start_stt_server raises on an engine that failed
    served = gate({**cloud, "stt": {"provider": "whisper", "serve": {"enabled": True}}})
    assert (served["label"], served["status"]) == ("Serve transcription", "skipped")
    assert served["message"] == (
        "Whisper has no model, pick one under Speech-to-text. The channel does not start without it."
    )
    assert gate({**cloud, "stt": {"provider": "whisper"}}) is None  # not served, not loaded
    assert gate({**cloud, "realtime": {"uplink": "vad"}, "vad": silero}) is None
    assert gate({**cloud, "realtime": {"uplink": "wake"}, "vad": silero, "wake": {**wake, "openwakeword": head}}) is None
    row = gate({**cloud, "realtime": {"uplink": "vad"}, "vad": {"engine": "silero"}})
    assert (row["label"], row["status"]) == ("On speech", "skipped")
    assert row["message"] == (
        "Silero has no model, pick one under Listening in Advanced. The channel does not start without it."
    )
    row = gate({**cloud, "realtime": {"uplink": "wake"}, "vad": silero, "wake": wake})
    assert (row["label"], row["status"]) == ("After wake word", "skipped")
    assert row["message"] == "openWakeWord has no model, pick one under Wake word. The channel does not start without it."
    assert gate({**cloud, "realtime": {"uplink": "wake"}, "vad": {"engine": "firered"}, "wake": wake})["message"] == (
        "FireRed has no model, pick one under Listening in Advanced. openWakeWord has no model, "
        "pick one under Wake word. The channel does not start without them."
    )
    # a listed model Apply can fetch, and Smart Turn's degradation on top
    cache = tmp_path / "store" / w.INDEX_CACHE
    cache.parent.mkdir(parents=True)
    entry = {"files": {"m.onnx": {"url": "https://x/m", "sha256": "0" * 64, "size": 1}}}
    cache.write_text(json.dumps({"models": {"vad/silero/v6/onnx": entry}}))
    listed = {"engine": "silero", "silero": {"weights": "vad/silero/v6/onnx"}}
    assert gate({**cloud, "realtime": {"uplink": "vad"}, "vad": listed})["message"] == (
        "The Silero model is not fetched yet, Apply downloads it. The channel does not start without it."
    )
    row = gate({**cloud, "realtime": {"uplink": "vad"}, "vad": {**silero, "turn": {"engine": "smartturn"}}})
    assert (row["label"], row["status"]) == ("On speech", "warn")
    assert row["message"] == (
        "Smart Turn has no model, pick one under Listening in Advanced. Until then turns end on silence alone."
    )
    # the gate runs the same analyzer, so it says the same about a window that never
    # consults it - and not on top of a model that would not load anyway
    loaded = {"engine": "smartturn", "modelPath": "/t.onnx", "consultMs": 240}
    row = gate({**cloud, "realtime": {"uplink": "vad"}, "vad": {**silero, "hangoverMs": 240, "turn": loaded}})
    assert (row["label"], row["status"]) == ("On speech", "warn")
    assert row["message"] == (
        "Smart Turn is never consulted: `vad.turn.consultMs` (240 ms) is at or past End of "
        "speech (240 ms), so turns end on silence alone."
    )
    unloadable = gate({**cloud, "realtime": {"uplink": "vad"}, "vad": {**silero, "hangoverMs": 240, "turn": {"engine": "smartturn", "consultMs": 240}}})
    assert "never consulted" not in unloadable["message"]
    # continuous uplink: the analyzer is not the gate's, so the row stays away
    assert gate({**cloud, "vad": {**silero, "hangoverMs": 240, "turn": loaded}}) is None


def test_setup_validator_resolves_the_realtime_key_per_backend(monkeypatch):
    """backend='gemini' reads GEMINI_API_KEY/GOOGLE_API_KEY and never OPENAI_API_KEY
    (channel.py's start-time check), so its row must neither claim an OpenAI fallback
    nor nudge when a Google key is exported."""
    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    for var in ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    row = _check_ids(manifest._validate({"backend": "gemini"}, ctx))["realtime_key"]
    assert "GEMINI_API_KEY" in row["message"] and "GOOGLE_API_KEY" in row["message"]
    assert "OPENAI_API_KEY" not in row["message"]
    # an OpenAI key is no fallback for gemini: still the plain nudge, no "reject" warning
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    row = _check_ids(manifest._validate({"backend": "gemini"}, ctx))["realtime_key"]
    assert "OPENAI_API_KEY" not in row["message"] and "reject" not in row["message"]
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.setenv(var, "g-key")
        assert "realtime_key" not in _check_ids(manifest._validate({"backend": "gemini"}, ctx))
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert "realtime_key" not in _check_ids(
        manifest._validate({"backend": "gemini", "realtime": {"apiKey": "k"}}, ctx)
    )


def test_setup_validator_lints_the_pending_patch(monkeypatch):
    """A bad pending patch (the Advanced box, or a hand-written importJson) fails the
    schema check with the parse error; a good one is linted by the full plugin schema and
    gets no row of its own: the panel says a pending patch is pending."""
    import json

    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

    paste = {"vad": {"hangoverMs": 800}}
    for form in (json.dumps(paste), paste):
        out = manifest._validate({"enabled": True, "importJson": form}, ctx)
        assert "import" not in _check_ids(out)
        assert out["can_enable"] is True

    bad = manifest._validate({"importJson": "{not json"}, ctx)
    schema = _check_ids(bad)["schema"]
    assert schema["status"] == "fail"
    assert "not valid JSON" in schema["message"]
    worse = manifest._validate({"importJson": json.dumps({"vad": {"engine": "nope"}})}, ctx)
    assert _check_ids(worse)["schema"]["status"] == "fail"

    # the panel's Reset: the rows and the form read the defaults under the saved section,
    # with the section's own edits stacked after the directive
    monkeypatch.delenv("NANOBOT_VOICE_DEFAULTS", raising=False)
    saved = {"enabled": True, "device": "rv1126b", "stt": {"provider": "whisper"}, "tts": {"provider": "system"}}
    out = manifest._validate({**saved, "importJson": {"$reset": True}}, ctx)
    assert out["identity"] == {"name": "Local", "workspace": "Energy, Internal, OpenAI"}
    fields = {f["key"]: f for s in out["form"]["sections"] for f in s["fields"]}
    assert fields["stt.provider"]["value"] == "nanobot" and "value" not in fields["device"]
    out = manifest._validate({**saved, "importJson": {"$reset": True, "tts": {"provider": "system"}}}, ctx)
    assert out["identity"]["workspace"] == "Energy, Internal, System"
    # the operator's defaults lie beneath the section: the pane resolves a fresh section
    # to the baseline, the section's own keys over it, and Reset returns to the baseline
    baseline = {"device": "rk3588", "stt": {"provider": "sensevoice"}, "vad": {"engine": "silero", "silero": {"modelPath": "/v.onnx"}}}
    monkeypatch.setenv("NANOBOT_VOICE_DEFAULTS", json.dumps(baseline))
    out = manifest._validate({"enabled": False, "importJson": ""}, ctx)
    assert out["identity"] == {"name": "Local", "workspace": "Silero, SenseVoice, OpenAI"}
    fields = {f["key"]: f for s in out["form"]["sections"] for f in s["fields"]}
    assert (fields["device"]["value"], fields["stt.provider"]["value"], fields["vad.engine"]["value"]) == ("rk3588", "sensevoice", "silero")
    assert fields["tts.provider"]["value"] == "openai"  # the schema's, the baseline being partial
    out = manifest._validate(saved, ctx)
    assert out["identity"]["workspace"] == "Silero, Whisper, System"
    out = manifest._validate({**saved, "importJson": {"$reset": True}}, ctx)
    assert out["identity"]["workspace"] == "Silero, SenseVoice, OpenAI"
    fields = {f["key"]: f for s in out["form"]["sections"] for f in s["fields"]}
    assert fields["device"]["value"] == "rk3588"
    # unreadable operator defaults refuse every section: the schema row names the
    # variable, the form keeps the section's own shape (a reset's is the schema's)
    monkeypatch.setenv("NANOBOT_VOICE_DEFAULTS", "/nonexistent/voice-defaults.json")
    out = manifest._validate(saved, ctx)
    assert _check_ids(out)["schema"]["status"] == "fail"
    assert _check_ids(out)["schema"]["message"].startswith("NANOBOT_VOICE_DEFAULTS: cannot read")
    fields = {f["key"]: f for s in out["form"]["sections"] for f in s["fields"]}
    assert fields["stt.provider"]["value"] == "whisper"
    out = manifest._validate({**saved, "importJson": {"$reset": True}}, ctx)
    assert _check_ids(out)["schema"]["message"].startswith("NANOBOT_VOICE_DEFAULTS: cannot read")
    fields = {f["key"]: f for s in out["form"]["sections"] for f in s["fields"]}
    assert fields["stt.provider"]["value"] == "nanobot"


def test_setup_validator_stays_within_the_rendered_check_budget(monkeypatch):
    """Core's WebUI renders only the first 6 checks; the worst case of either
    branch must fit or later checks silently vanish."""
    from nanobot.channels.contracts import ChannelValidationContext

    from nanobot_channel_voice import running
    from nanobot_channel_voice.config import VoiceConfig

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # every local row at once: a fallback, a live channel that started without an engine that
    # loads now, a cloud setting Local ignores, the key nudge, and the devices row
    paths = ("encoderPath", "decoderPath", "vocabPath", "melFiltersPath")
    local = {
        "vad": {"engine": "firered"}, "realtime": {"model": "m"}, "importJson": "{}",
        "stt": {"provider": "whisper", "whisper": {p: f"/m/{p}" for p in paths}},
    }
    started = running.Started(VoiceConfig.model_validate(local), {"stt": None})
    monkeypatch.setattr(running, "_current", (object(), started))
    local_worst = manifest._validate(local, ctx)
    assert [c["id"] for c in local_worst["checks"]] == [
        "pipeline", "running", "realtime_unused", "tts_key", "audio_devices",
    ]
    # every cloud row at once: no key, no endpoint, an open mic without a canceller, a
    # wake gate without its head, a local-only block, and the devices row
    cloud = {
        "backend": "azure", "stt": {"provider": "whisper"}, "importJson": "{}",
        "realtime": {"bargeIn": "aec", "uplink": "wake"},
        "vad": {"engine": "silero", "silero": {"modelPath": "/v.onnx"}},
        "wake": {"mode": "gate", "engine": "openwakeword", "phrases": ["hey"]},
    }
    cloud_worst = manifest._validate(cloud, ctx)
    assert len(local_worst["checks"]) <= 6
    assert [c["id"] for c in cloud_worst["checks"]] == [
        "realtime_key", "realtime_aec", "realtime_endpoint", "gate", "local_unused", "audio_devices",
    ]
    # the env-fed non-OpenAI key variant REPLACES the no-key row, never adds one
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    cloud_env_worst = manifest._validate(cloud, ctx)
    assert len(cloud_env_worst["checks"]) <= 6


def test_cloud_rows_name_the_modules_start_would_fail_on(monkeypatch):
    """_build_cloud loads the websockets transport before it claims a device, and the
    software canceller its livekit binding: neither is a schema error, so the rows name
    the extra instead of letting start() raise."""
    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    absent = {"websockets", "livekit"}
    monkeypatch.setattr(manifest, "find_spec", lambda name: None if name in absent else object())

    out = manifest._validate({"backend": "openai"}, ctx)
    assert out["can_enable"] is True  # a missing extra is the operator's to install
    assert _check_ids(out)["realtime_module"]["message"].endswith(
        "pip install 'nanobot-channel-voice[realtime]'."
    )
    # the software canceller counts as one only with its binding, and says which it needs
    aec = {"backend": "openai", "realtime": {"bargeIn": "aec"}, "aec": "webrtc"}
    assert _check_ids(manifest._validate(aec, ctx))["realtime_aec"]["message"].endswith(
        "The WebRTC canceller needs its binding: pip install 'nanobot-channel-voice[aec]'."
    )
    absent = {"websockets"}  # binding installed: the canceller is real, the row goes
    assert "realtime_aec" not in _check_ids(manifest._validate(aec, ctx))
    absent = set()
    assert "realtime_module" not in _check_ids(manifest._validate({"backend": "openai"}, ctx))
    # local sections load neither
    assert not {"realtime_module", "realtime_aec"} & set(_check_ids(manifest._validate({"enabled": True}, ctx)))


def test_pipeline_row_names_a_turn_model_that_is_never_consulted():
    """make_turn_analyzer's other fallback, which no preflight can see: a consult window
    at or past the hangover loads the model and never asks it."""
    from nanobot.channels.contracts import ChannelValidationContext

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    turn = {"engine": "smartturn", "modelPath": "/t.onnx", "consultMs": 240}
    idle = {"enabled": True, "vad": {"hangoverMs": 240, "turn": turn}}
    message = _check_ids(manifest._validate(idle, ctx))["pipeline"]["message"]
    assert "Smart Turn is never consulted: `vad.turn.consultMs` (240 ms) is at or past " \
        "End of speech (240 ms), so turns end on silence alone." in message
    consulted = {"enabled": True, "vad": {"hangoverMs": 600, "turn": turn}}
    assert "never consulted" not in _check_ids(manifest._validate(consulted, ctx))["pipeline"]["message"]
    # not said of a model that would not load: it is not consulted for the louder reason
    unloadable = {"enabled": True, "vad": {"hangoverMs": 240, "turn": {"engine": "smartturn", "consultMs": 240}}}
    assert "never consulted" not in _check_ids(manifest._validate(unloadable, ctx))["pipeline"]["message"]


def _install(tmp_path, key, *names):
    """A fetched key in the test's store, each file linked from a source of its own."""
    from nanobot_channel_voice import weights as w

    src = tmp_path / "src" / key.replace("/", "_")
    src.mkdir(parents=True)
    for name in names:
        (src / name).write_bytes(b"x")
    w.fetch(key, {"files": {name: {"url": (src / name).as_uri()} for name in names}})


_WHISPER = "stt/whisper/base/onnx"
_WHISPER_FILES = ("encoder.onnx", "decoder.onnx", "vocab.json", "mel_filters.npz")


def test_pipeline_row_names_a_system_voice_that_cannot_speak(monkeypatch):
    """Without espeak-ng the System voice returns nothing, picked or standing in: a reply
    nobody hears, which no row showed."""
    from nanobot.channels.contracts import ChannelValidationContext

    import nanobot_channel_voice.config as voice_config
    import nanobot_channel_voice.tts.system as system_tts

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.setattr(voice_config, "transcription_gap", lambda: None)
    monkeypatch.setattr(system_tts, "system_voice", lambda *_a: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    silent = "System needs espeak-ng, which is not installed, so replies would be silent."
    for section in (
        {"tts": {"provider": "system"}},
        {"tts": {"provider": "matcha"}},  # no model: System stands in
        {"enabled": True},  # OpenAI without a key: System stands in
    ):
        pipeline = _check_ids(manifest._validate(section, ctx))["pipeline"]
        assert pipeline["status"] == "warn" and pipeline["message"].endswith(silent), section
    for section in ({"tts": {"provider": "system", "enabled": False}}, {"tts": {"apiKey": "sk-x"}}):
        assert _check_ids(manifest._validate(section, ctx))["pipeline"]["status"] == "pass", section
    monkeypatch.setattr(system_tts, "system_voice", lambda *_a: ("espeak", "/usr/bin/espeak-ng"))
    assert _check_ids(manifest._validate({"tts": {"provider": "system"}}, ctx))["pipeline"]["status"] == "pass"


def test_running_row_tells_what_the_live_channel_started_without(monkeypatch, tmp_path):
    """The pipeline row weighs the config against the store; the channel runs what loaded
    at its start. A model fetched since, or one that failed to load, gets a row of its own
    and the panel's restart; one the pipeline row already names as missing gets neither."""
    from nanobot.channels.contracts import ChannelValidationContext

    import nanobot_channel_voice.config as voice_config
    import nanobot_channel_voice.tts.system as system_tts
    from nanobot_channel_voice import running
    from nanobot_channel_voice import weights as w
    from nanobot_channel_voice.config import VoiceConfig
    from nanobot_channel_voice.engines import Fallback

    manifest = _load("voice_shim_manifest", _SHIM / "manifest.py")
    ctx = ChannelValidationContext()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    monkeypatch.setattr(voice_config, "transcription_gap", lambda: None)
    monkeypatch.setattr(system_tts, "system_voice", lambda *_a: ("espeak", "/usr/bin/espeak-ng"))
    restart = "Apply and restart" if manifest._WEBUI else "Turning the channel off and on"
    section = {"stt": {"provider": "whisper", "whisper": {"weights": _WHISPER}}}

    def started(fell_back, config=section):
        record = running.Started(VoiceConfig.model_validate(config), fell_back)
        monkeypatch.setattr(running, "_current", (object(), record))

    started({"stt": Fallback(key=_WHISPER)})
    out = manifest._validate(section, ctx)  # still missing: the pipeline row speaks for both
    assert "running" not in _check_ids(out) and "restart" not in out
    _install(tmp_path, _WHISPER, *_WHISPER_FILES)  # fetched since, by the CLI
    out = manifest._validate(section, ctx)
    assert _check_ids(out)["pipeline"]["status"] == "pass"
    live = _check_ids(out)["running"]
    assert (live["label"], live["status"]) == ("Running channel", "warn")
    assert live["message"] == (
        "It started without Whisper, so Internal transcribes instead. The Whisper model is "
        f"in place now. {restart} loads it."
    )
    assert out["restart"] is True
    monkeypatch.setattr(voice_config, "transcription_gap", lambda: "its provider has no API key")
    assert _check_ids(manifest._validate(section, ctx))["running"]["message"].endswith(
        "Internal delegates to nanobot's transcription, but its provider has no API key, so "
        "every utterance is heard as silence."
    )
    monkeypatch.setattr(voice_config, "transcription_gap", lambda: None)

    started({"stt": None})  # its files were there and it did not load: a restart retries
    assert _check_ids(manifest._validate(section, ctx))["running"]["message"].endswith(
        f"Whisper did not load, the gateway log says why. {restart} tries again."
    )

    # a restart runs the section as it stands, not as the channel started: another model of
    # the same engine that is in place is what it loads
    other = "stt/whisper/small/onnx"
    started({"stt": Fallback(key=other)}, config={"stt": {"provider": "whisper", "whisper": {"weights": other}}})
    out = manifest._validate(section, ctx)
    assert _check_ids(out)["running"]["message"].endswith(f"The Whisper model is in place now. {restart} loads it.")
    assert out["restart"] is True

    # a pending edit picked another engine: this row keeps the live one, nothing to restart
    w.prune(_WHISPER)
    started({"stt": Fallback(key=_WHISPER)})
    out = manifest._validate({"stt": {"provider": "sensevoice"}}, ctx)
    assert _check_ids(out)["running"]["message"] == "It started without Whisper, so Internal transcribes instead."
    assert "restart" not in out

    # the keyless OpenAI voice has its own row; a key in the section makes the restart worth it
    monkeypatch.delenv("OPENAI_API_KEY")
    started({"tts": None}, config={})
    out = manifest._validate({}, ctx)
    assert "running" not in _check_ids(out) and "tts_key" in _check_ids(out)
    out = manifest._validate({"tts": {"apiKey": "sk-x"}}, ctx)
    assert _check_ids(out)["running"]["message"] == (
        "It started without OpenAI, so System speaks instead. OpenAI did not load, the gateway "
        f"log says why. {restart} tries again."
    )
    assert out["restart"] is True


def test_a_late_withdraw_leaves_the_restarted_channels_record():
    from nanobot_channel_voice import running

    old, new = object(), object()
    running.publish(old, running.Started(None, {}))
    record = running.Started(None, {"vad": None})
    running.publish(new, record)
    running.withdraw(old)  # the stopped instance's teardown, after the restart
    assert running.current() is record
    running.withdraw(new)
    assert running.current() is None
