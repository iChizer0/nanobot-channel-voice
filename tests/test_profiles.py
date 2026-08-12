"""RealtimeProfile table: dispatch, URLs, auth, per-model overrides."""

from __future__ import annotations

import pytest

from nanobot_channel_voice.backend.profiles import (
    PROFILES,
    backend_kind,
    normalize_backend,
    resolve_profile,
)


def test_normalize_backend_lowercases_and_strips():
    assert normalize_backend(" Qwen ") == "qwen"
    assert normalize_backend("") == ""


def test_backend_kind_classification():
    assert backend_kind("local") == "local"
    assert backend_kind("openai") == "openai_dialect"
    for key in ("xai", "azure", "qwen", "glm", "stepfun"):
        assert backend_kind(key) == "openai_dialect"
    assert backend_kind("anything-else") == "local"


def test_resolve_profile_rejects_non_dialect_names():
    with pytest.raises(RuntimeError, match="not an OpenAI-dialect"):
        resolve_profile("local")


def test_connect_url_appends_model_once():
    p = PROFILES["openai"]
    assert p.connect_url(None, "gpt-realtime").endswith("/realtime?model=gpt-realtime")
    # A baseUrl that already pins ?model= wins; never send two.
    pinned = "wss://x.example/v1/realtime?model=frozen"
    assert p.connect_url(pinned, "other") == pinned
    q = p.connect_url("wss://x.example/rt?api-version=1", "m")
    assert q == "wss://x.example/rt?api-version=1&model=m"


def test_azure_requires_base_url_with_actionable_error():
    with pytest.raises(RuntimeError, match="realtime.baseUrl"):
        PROFILES["azure"].base_url(None)
    assert PROFILES["azure"].base_url("wss://res.example") == "wss://res.example"


def test_auth_header_shapes():
    assert PROFILES["openai"].auth_headers("K") == {"Authorization": "Bearer K"}
    assert PROFILES["azure"].auth_headers("K") == {"api-key": "K"}


def test_qwen_capabilities_gate_tools_per_model_generation():
    p = PROFILES["qwen"]
    base = p.capabilities_for(p.default_model)
    assert base["supports_tools"] is False  # qwen3 generation is persona-only
    new = p.capabilities_for("qwen3.5-omni-flash-realtime")
    assert new["supports_tools"] is True
    assert new["max_tool_output_chars"] == 8000


def test_longest_prefix_wins_for_voice_overrides():
    p = PROFILES["qwen"]
    assert p.default_voice_for("qwen3-omni-flash-realtime") == "Chelsie"
    assert p.default_voice_for("qwen3.5-omni-flash-realtime") == "Tina"
    assert p.default_voice_for(None) == "Chelsie"


def test_beta_profiles_carry_vendor_format_strings():
    # These vocabularies are mutually incompatible: the table must keep them apart.
    assert PROFILES["qwen"].input_format == "pcm" and PROFILES["qwen"].output_format == "pcm"
    assert PROFILES["glm"].input_format == "pcm16" and PROFILES["glm"].output_format == "pcm"
    assert PROFILES["stepfun"].input_format == "pcm16"


def test_dialect_interrupt_pairing():
    for key, p in PROFILES.items():
        if p.dialect == "ga":
            assert p.interrupt == "truncate", key
        else:
            assert p.interrupt == "cancel", key
