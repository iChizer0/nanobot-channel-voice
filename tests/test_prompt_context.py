"""What the agent is told: the realtime persona/rules split, and the local backend's
speakability context block."""

from __future__ import annotations

import asyncio
import re

from nanobot.bus.queue import MessageBus
from nanobot.runtime_context import normalize_runtime_context_blocks

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
from nanobot_channel_voice.context_tool import VoiceContextBridge
from nanobot_channel_voice.streamid import TURN_META


class _FakeTts:
    def __init__(self, language: str | None = None, languages: tuple[str, ...] = ()):
        self.spoken_language = language
        if languages:  # only a router / a bilingual model declares the plural
            self.spoken_languages = languages


# ---- realtime instructions --------------------------------------------------


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


# ---- local speakability context ---------------------------------------------


class _FakeStt:
    def __init__(self, decoder_family: str = ""):
        self.decoder_family = decoder_family


def test_block_is_wrapped_and_names_the_engine_language():
    [block] = _voice_context_blocks(None, _FakeTts("en"))
    assert block.source == "voice"
    # The contract wrapper is OURS, not core's "metadata only, not instructions" tag:
    # these lines are instructions, and a wrapper disclaiming them undercuts them.
    assert block.content.startswith("[Voice channel]")
    assert block.content.endswith("[/Voice channel]")
    assert "text-to-speech" in block.content
    assert "markdown" in block.content
    assert "'en'" in block.content
    # Core accepts it verbatim off inbound metadata.
    assert normalize_runtime_context_blocks([block]) == [block]


def test_block_stays_inside_its_token_budget():
    # Every word rides every turn's prefill, so the ceiling is pinned at the LONGEST
    # variant: an unverified decoder (invented-phrase clause) plus a bilingual voice
    # (6 words over the mono line). A failure means re-bloat — trim, don't raise it.
    [block] = _voice_context_blocks(_FakeStt(), _FakeTts("zh", ("zh", "en")))
    assert len(block.content.split()) <= 145
    [mono] = _voice_context_blocks(_FakeStt("ctc"), _FakeTts("en"))
    assert len(mono.content.split()) < len(block.content.split())


def test_context_permits_thinking_and_nudges_pre_tool_narration():
    [block] = _voice_context_blocks(None, _FakeTts("en"))
    assert "Thinking aloud briefly is fine." in block.content  # the model's scratchpad
    assert "one short sentence" in block.content               # pre-tool status line
    assert "after the results" in block.content                # no premature answer
    assert "wait-phrases" in block.content                     # none in the delivery
    # The old suppressors must stay gone: they banned the intermediate tokens small
    # non-reasoning models think with.
    assert "NOTHING else" not in block.content
    assert "pure answer" not in block.content


def test_capability_affirmation_rides_spoken_sessions():
    # "Spoken conversation" framing alone collapses small models into a can't-do-
    # anything voice-assistant persona; the affirmation sits right next to it. A
    # listen-only session (tts off) keeps the input facts without the reply framing.
    [block] = _voice_context_blocks(None, _FakeTts("en"))
    assert "full tools and skills" in block.content
    assert block.content.index("spoken conversation") < block.content.index("full tools")
    [listen_only] = _voice_context_blocks(_FakeStt("ctc"), None)
    assert "tools and skills" not in listen_only.content


def test_confirmation_is_scoped_to_hard_to_undo_actions():
    # Blanket confirm-before-acting made the model ask instead of act on benign
    # requests — the observed voice-mode tool collapse. Both decoder variants scope it.
    for stt in (None, _FakeStt("ctc")):
        [block] = _voice_context_blocks(stt, _FakeTts("en"))
        assert "hard-to-undo" in block.content
        assert "act on the likeliest reading" in block.content
        assert "before acting on them" not in block.content


def test_every_stt_gets_the_mishear_line():
    # ALL ASR substitutes similar sounds — and stt=None (the cloud-transcription path)
    # is still a transcript, so the line rides even without an on-device adapter.
    for stt in (None, _FakeStt("ctc"), _FakeStt("transducer"), _FakeStt("attention")):
        [block] = _voice_context_blocks(stt, _FakeTts("en"))
        assert "may mis-hear words" in block.content
        assert "read it by sound and context" in block.content


def test_only_a_frame_synchronous_decoder_claims_no_invention():
    # Frame-synchronous decoders cannot confabulate, so they alone drop the clause.
    # Everything else keeps it: whisper-class AR decoders hallucinate on noise, and
    # remote/undeclared are unverified (core's transcription defaults are all whisper-*).
    for family in ("ctc", "transducer"):
        [block] = _voice_context_blocks(_FakeStt(family), _FakeTts("en"))
        assert "never said" not in block.content
    for stt in (_FakeStt("attention"), _FakeStt(), None):
        [block] = _voice_context_blocks(stt, _FakeTts("en"))
        assert "never said" in block.content


def test_bilingual_voice_invites_code_switching_instead_of_one_language():
    # A router (or the single zh-en model) says both, so pinning one would leave half
    # the voice unused.
    [block] = _voice_context_blocks(_FakeStt("ctc"), _FakeTts("zh", ("zh", "en")))
    assert "'zh' and 'en'" in block.content
    assert "mixing is fine" in block.content
    assert "reply in 'zh' only" not in block.content


def test_unrestricted_engine_claims_no_language():
    # openai/openai_compat (and a custom-vocab MMS) speak an unknown set: the format
    # half still applies, but inventing a language would be worse than staying quiet.
    [block] = _voice_context_blocks(None, _FakeTts(None))
    assert "ISO 639-1" not in block.content
    assert "markdown" in block.content


def test_operator_context_rides_after_the_derived_lines():
    [block] = _voice_context_blocks(None, _FakeTts("en"), "  Address the user as Captain.  ")
    body = block.content
    assert body.rstrip().endswith("Address the user as Captain.\n[/Voice channel]")
    assert body.index("markdown") < body.index("Captain")  # facts first, operator last


def test_disabled_tts_keeps_the_input_lines_and_drops_speakability():
    # tts.enabled=false is a listen-only session: the speakability lines would be false,
    # but the input is still a speech-recognition transcript, so its facts stay.
    [block] = _voice_context_blocks(_FakeStt("ctc"), None, "Answer in one sentence.")
    assert "text-to-speech" not in block.content
    assert "speech recognition" in block.content
    assert "may mis-hear words" in block.content
    assert "Answer in one sentence." in block.content
    [bare] = _voice_context_blocks(None, None, "   ")  # whitespace is not guidance
    assert "Answer" not in bare.content and "may mis-hear words" in bare.content


# ---- what make_tts settles --------------------------------------------------


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


# ---- what the adapters declare ----------------------------------------------


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


# ---- the publish seam -------------------------------------------------------


def _capturing_channel() -> tuple[VoiceChannel, list[dict]]:
    channel = VoiceChannel(VoiceConfig(), MessageBus())
    seen: list[dict] = []

    async def _capture(**kwargs):
        seen.append(kwargs)

    channel._handle_message = _capture  # type: ignore[method-assign]
    return channel, seen


def test_published_metadata_is_json_plain_and_carries_only_the_token():
    # Tools snapshot request metadata verbatim and cron PERSISTS that snapshot with
    # plain json.dumps: one non-JSON value there breaks `cron add` from a voice turn.
    # The blocks ride the context bridge instead (see test_context_tool).
    import json

    channel, seen = _capturing_channel()
    channel._voice_context = _voice_context_blocks(None, _FakeTts("en"))
    channel._context_bridge = VoiceContextBridge(channel._voice_context)
    asyncio.run(channel._publish_turn_text("what time is it", "turn-1"))
    [call] = seen
    assert call["content"] == "what time is it"
    # The turn token is the only key: send() correlates the reply on it.
    assert call["metadata"] == {TURN_META: "turn-1"}
    json.dumps(call["metadata"])  # the cron-capture invariant, pinned forever


def test_bridge_resolve_serves_the_block_and_the_clock():
    channel, seen = _capturing_channel()
    channel._voice_context = _voice_context_blocks(None, _FakeTts("en"))
    channel._context_bridge = VoiceContextBridge(channel._voice_context)
    asyncio.run(channel._publish_turn_text("what time is it", "turn-1"))
    [call] = seen
    blocks = channel._context_bridge.resolve(call["metadata"])
    assert blocks[:-1] == channel._voice_context
    assert len(channel._voice_context) == 1  # resolve built a fresh list
    # Every resolved turn is stamped with the model's only clock: core injects no date
    # or time anywhere, and without one the model invents a placeholder when asked.
    assert re.fullmatch(
        r"\[time now: \d{4}-\d{2}-\d{2} \((?:Mon|Tues|Wednes|Thurs|Fri|Satur|Sun)day\) "
        r"\d{2}:\d{2}, UTC[+-]\d{2}:\d{2}\]",
        blocks[-1].content,
    )
    # Core accepts the resolved blocks verbatim off the provider seam.
    assert normalize_runtime_context_blocks(blocks) == blocks


def test_cloud_publishes_without_a_block():
    # The bridge is registered in _build_local only, so a delegated supervisor request
    # (provider-side TTS, provider-owned persona) stays clean.
    channel, seen = _capturing_channel()
    assert channel._context_bridge is None
    asyncio.run(channel._publish_user_text("delegated request"))
    [call] = seen
    assert call["metadata"] is None


def test_event_notes_ride_the_bridge_never_the_user_row():
    # Notes are model-only: they never join the user content (the persisted row stays
    # pure speech) and never ride metadata (tools snapshot it). Stashed under the turn
    # token, they come back once — on that turn's resolve — after the time stamp.
    channel, seen = _capturing_channel()
    channel._voice_context = _voice_context_blocks(None, _FakeTts("en"))
    channel._context_bridge = VoiceContextBridge(channel._voice_context)
    note = '[voice event: you were interrupted mid-reply; the user heard up to: "…sunny"]'
    asyncio.run(channel._publish_turn_text("and tomorrow", "turn-2", (note,)))
    [call] = seen
    assert call["content"] == "and tomorrow"
    assert call["metadata"] == {TURN_META: "turn-2"}
    blocks = channel._context_bridge.resolve(call["metadata"])
    assert blocks[:-1] == channel._voice_context
    assert blocks[-1].source == "voice"
    # The time stamp shares the per-turn block; the event note rides right after it.
    stamp, event = blocks[-1].content.split("\n")
    assert stamp.startswith("[time now: ")
    assert event == note
    # Popped: a later resolve of the same token (a cron fire echoing stale creation
    # metadata) gets the stamp only, never a replayed note.
    again = channel._context_bridge.resolve(call["metadata"])
    assert "\n" not in again[-1].content and again[-1].content.startswith("[time now: ")


def test_stacked_notes_share_one_block():
    channel, seen = _capturing_channel()
    channel._context_bridge = VoiceContextBridge([])  # no contract configured
    asyncio.run(
        channel._publish_turn_text(
            "next question", "turn-3", ("[voice event: a]", "[voice event: b]")
        )
    )
    [call] = seen
    [block] = channel._context_bridge.resolve(call["metadata"])
    stamp, *events = block.content.split("\n")
    assert stamp.startswith("[time now: ")
    assert events == ["[voice event: a]", "[voice event: b]"]
