"""``nanobot-voice config``: the export counterpart of the WebUI importJson box.

It prints the EFFECTIVE section (schema-validated: twins folded, a pending import
merged) in canonical camelCase, secrets dropped by default so the output is safe to
paste into another install's Import Json box or a bug report.
"""

from __future__ import annotations

import json

from nanobot_channel_voice.cli import main


def _write(tmp_path, section):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"channels": {"voice": section}}), encoding="utf-8")
    return str(path)


def test_config_prints_canonical_paste_ready_json(tmp_path, capsys):
    path = _write(
        tmp_path,
        {"vad": {"hangover_ms": 800, "hangoverMs": 800}, "tts": {"provider": "system"}},
    )
    assert main(["config", "--config", path]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["vad"] == {"hangoverMs": 800}  # twins folded, canonical camelCase
    assert out["tts"] == {"provider": "system"}
    assert "importJson" not in out


def test_config_omits_secrets_unless_asked(tmp_path, capsys):
    path = _write(tmp_path, {"backend": "openai", "realtime": {"apiKey": "sk-live"}})
    assert main(["config", "--config", path]) == 0
    captured = capsys.readouterr()
    assert "apiKey" not in json.loads(captured.out).get("realtime", {})
    assert "omitted" in captured.err  # the drop is announced, never silent
    assert main(["config", "--config", path, "--secrets"]) == 0
    assert json.loads(capsys.readouterr().out)["realtime"]["apiKey"] == "sk-live"


def test_config_shows_the_effective_state_of_a_pending_import(tmp_path, capsys):
    """A saved-but-not-yet-consumed WebUI paste is part of the effective config; the
    transport field itself never exports."""
    path = _write(tmp_path, {"duck_db": -12, "importJson": json.dumps({"duckDb": -6})})
    assert main(["config", "--config", path]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["duckDb"] == -6
    assert "importJson" not in out


def test_config_full_includes_defaults(tmp_path, capsys):
    path = _write(tmp_path, {})
    assert main(["config", "--config", path]) == 0
    assert json.loads(capsys.readouterr().out) == {}  # nothing configured -> nothing
    assert main(["config", "--config", path, "--full"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["vad"]["hangoverMs"] == 600  # every effective default materialized


def test_config_reports_an_invalid_section(tmp_path, capsys):
    path = _write(tmp_path, {"vad": {"engine": "nope"}})
    assert main(["config", "--config", path]) == 2
    assert "does not validate" in capsys.readouterr().err


def test_every_credential_shaped_field_has_the_api_key_leaf():
    """``_scrub_secrets`` drops by the literal leaf ``apiKey``: a credential field
    under any OTHER leaf must fail here instead of exporting in cleartext."""
    from nanobot_channel_voice.config import VoiceConfig

    def walk(model, prefix=""):
        for name, info in model.model_fields.items():
            ann = info.annotation
            for nested in (ann, *getattr(ann, "__args__", ())):
                if isinstance(nested, type) and hasattr(nested, "model_fields"):
                    yield from walk(nested, f"{prefix}{name}.")
            yield f"{prefix}{name}"

    credential_words = {"key", "token", "secret", "password", "credential", "bearer", "auth"}
    suspicious = [
        path
        for path in walk(VoiceConfig)
        # whole segments, so tokens_path (a vocab FILE) does not trip the net
        if credential_words & set(path.rsplit(".", 1)[-1].split("_"))
    ]
    assert suspicious, "the walker went blind: it must at least see the apiKey fields"
    assert all(path.rsplit(".", 1)[-1] == "api_key" for path in suspicious), suspicious
