"""What the agent is told: the realtime persona/rules split, and the local backend's
speakability context block."""

from __future__ import annotations

import asyncio

from nanobot.bus.queue import MessageBus
from nanobot.runtime_context import (
    RUNTIME_CONTEXT_END,
    RUNTIME_CONTEXT_INPUT_META,
    RUNTIME_CONTEXT_TAG,
    normalize_runtime_context_blocks,
)

from nanobot_channel_voice.backend.local import TURN_META
from nanobot_channel_voice.channel import (
    _DEFAULT_PERSONA,
    _DIRECT_RULES,
    _STOP_RULE,
    _SUPERVISOR_RULES,
    VoiceChannel,
    _cloud_instructions,
    _voice_context_blocks,
)
from nanobot_channel_voice.config import VoiceConfig


class _FakeTts:
    def __init__(self, language: str | None = None):
        self.spoken_language = language


# ---- realtime instructions ------------------------------------------------


def test_default_persona_carries_the_mode_rules():
    direct = _cloud_instructions(None, supervisor=False, has_tools=True)
    assert direct == f"{_DEFAULT_PERSONA}\n\n{_DIRECT_RULES}\n\n{_STOP_RULE}"
    supervisor = _cloud_instructions(None, supervisor=True, has_tools=True)
    assert supervisor == f"{_DEFAULT_PERSONA}\n\n{_SUPERVISOR_RULES}\n\n{_STOP_RULE}"


def test_persona_only_session_gets_no_tool_rules():
    # No tools declared => no round-trip to mask, so the filler preamble is dead text.
    # The stop rule is mode-independent: silence-is-the-ack holds even tool-less.
    assert _cloud_instructions(None, supervisor=False, has_tools=False) == (
        f"{_DEFAULT_PERSONA}\n\n{_STOP_RULE}"
    )


def test_persona_override_replaces_style_but_never_the_contract():
    mine = "You are a laconic ship's computer."
    out = _cloud_instructions(mine, supervisor=True, has_tools=True)
    assert out.startswith(mine)
    assert _DEFAULT_PERSONA not in out
    assert _SUPERVISOR_RULES in out
    assert "ask_nanobot" in out


def test_direct_rules_survive_a_persona_override():
    out = _cloud_instructions("Speak like a pirate.", supervisor=False, has_tools=True)
    assert _DIRECT_RULES in out
    assert _SUPERVISOR_RULES not in out


# ---- local speakability context -------------------------------------------


def test_no_tts_means_no_block():
    assert _voice_context_blocks(None) == []


def test_block_is_wrapped_and_names_the_engine_language():
    [block] = _voice_context_blocks(_FakeTts("en"))
    assert block.source == "voice"
    assert block.content.startswith(RUNTIME_CONTEXT_TAG)
    assert block.content.endswith(RUNTIME_CONTEXT_END)
    assert "text-to-speech" in block.content
    assert "Markdown" in block.content
    assert "'en'" in block.content
    # Core accepts it verbatim off inbound metadata.
    assert normalize_runtime_context_blocks([block]) == [block]


def test_unrestricted_engine_claims_no_language():
    # openai/openai_compat (and a custom-vocab MMS) speak an unknown set: the format
    # half still applies, but inventing a language would be worse than staying quiet.
    [block] = _voice_context_blocks(_FakeTts(None))
    assert "ISO 639-1" not in block.content
    assert "Markdown" in block.content


def test_operator_context_rides_after_the_derived_lines():
    [block] = _voice_context_blocks(_FakeTts("en"), "  Address the user as Captain.  ")
    body = block.content
    assert body.rstrip().endswith(f"Address the user as Captain.\n{RUNTIME_CONTEXT_END}")
    assert body.index("Markdown") < body.index("Captain")  # facts first, operator last


def test_operator_context_survives_a_disabled_tts():
    # tts.enabled=false is a listen-only session: the speakability lines would be false,
    # the operator's own guidance is not.
    [block] = _voice_context_blocks(None, "Answer in one sentence.")
    assert "text-to-speech" not in block.content
    assert "Answer in one sentence." in block.content
    assert _voice_context_blocks(None, "   ") == []  # whitespace is not guidance


# ---- what make_tts settles ------------------------------------------------


def test_tts_language_fills_an_engine_that_cannot_know_its_own():
    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.tts import make_tts

    # openai_compat drives arbitrary local servers (piper-http, Kokoro) whose language
    # the plugin cannot introspect; tts.language is the operator's declaration.
    cfg = TtsConfig.model_validate({"provider": "openai_compat", "language": "de"})
    assert make_tts(cfg).spoken_language == "de"
    # Undeclared stays unknown: the block then claims no language.
    assert make_tts(TtsConfig.model_validate({"provider": "openai_compat"})).spoken_language is None


def test_engine_declared_language_beats_a_conflicting_config():
    from loguru import logger as loguru_logger

    from nanobot_channel_voice import tts as tts_registry
    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.engines import EngineSpec
    from nanobot_channel_voice.tts import make_tts

    class _SelfDeclared(tts_registry.TtsAdapter):
        spoken_language = "en"  # e.g. built-in-vocab MMS, or supertonic.language

        async def synthesize(self, text, *, voice=None):
            return b""

    original = tts_registry.ENGINES["openai"]
    tts_registry.ENGINES["openai"] = EngineSpec(build=lambda cfg: _SelfDeclared())
    messages: list[str] = []
    sink = loguru_logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        adapter = make_tts(TtsConfig.model_validate({"provider": "openai", "language": "de"}))
    finally:
        loguru_logger.remove(sink)
        tts_registry.ENGINES["openai"] = original
    assert adapter.spoken_language == "en"  # the engine is the ground truth
    assert any("conflicts" in m and "'de'" in m for m in messages)


# ---- what the adapters declare --------------------------------------------


def test_system_adapter_declares_its_espeak_voice():
    from nanobot_channel_voice.tts.system import SystemTtsAdapter

    assert SystemTtsAdapter(language="de").spoken_language == "de"
    assert SystemTtsAdapter().spoken_language is None  # the host default voice


def test_mms_names_only_the_built_in_english_vocab():
    import pytest

    np = pytest.importorskip("numpy")
    assert np is not None
    from nanobot_channel_voice.tts.mms import _ENG_VOCAB, MmsTtsAdapter

    def build(vocab):  # __init__ only stores; no model is touched
        return MmsTtsAdapter(
            encoder=None, decoder=None, vocab=vocab, frontend=None,  # type: ignore[arg-type]
            max_length=200, speaking_rate=1.0,
        )

    assert build(_ENG_VOCAB).spoken_language == "en"
    assert build(dict(_ENG_VOCAB)).spoken_language is None  # a loaded vocab.json


# ---- the publish seam -----------------------------------------------------


def _capturing_channel() -> tuple[VoiceChannel, list[dict]]:
    channel = VoiceChannel(VoiceConfig(), MessageBus())
    seen: list[dict] = []

    async def _capture(**kwargs):
        seen.append(kwargs)

    channel._handle_message = _capture  # type: ignore[method-assign]
    return channel, seen


def test_published_utterance_carries_the_block_alongside_its_turn_token():
    channel, seen = _capturing_channel()
    channel._voice_context = _voice_context_blocks(_FakeTts("en"))
    asyncio.run(channel._publish_turn_text("what time is it", "turn-1"))
    [call] = seen
    assert call["content"] == "what time is it"
    # The block must not displace the turn token: send() correlates the reply on it.
    assert call["metadata"][TURN_META] == "turn-1"
    assert call["metadata"][RUNTIME_CONTEXT_INPUT_META] == channel._voice_context


def test_cloud_publishes_without_a_block():
    # _voice_context is built in _build_local only, so a delegated supervisor request
    # (provider-side TTS, provider-owned persona) stays clean.
    channel, seen = _capturing_channel()
    asyncio.run(channel._publish_user_text("delegated request"))
    [call] = seen
    assert call["metadata"] is None
