"""Bilingual script-routed TTS: run segmentation, engine dispatch, the matcha
build wiring (vocoder sharing, failure cleanup), and the context-line surface."""

from __future__ import annotations

import asyncio

import pytest

from nanobot_channel_voice.config import MatchaTtsConfig, TtsConfig
from nanobot_channel_voice.tts.base import TtsAdapter
from nanobot_channel_voice.tts.router import ScriptRoutedTts, script_runs

# ---- run segmentation -------------------------------------------------------


def test_script_runs_split_by_script_with_neutral_riders():
    assert script_runs("你好，请打开WiFi设置。") == [
        (True, "你好，请打开"), (False, "WiFi"), (True, "设置。"),
    ]
    assert script_runs("Hello 世界 42.") == [(False, "Hello "), (True, "世界 42.")]
    assert script_runs("3个") == [(True, "3个")]          # leading neutral rides forward
    assert script_runs("42!") == [(False, "42!")]          # nothing scripted -> primary
    assert script_runs("你好。OK") == [(True, "你好。"), (False, "OK")]  # 。 is neutral


# ---- the router adapter -----------------------------------------------------


class _Eng(TtsAdapter):
    output_rate = 22050

    def __init__(self, lang: str, tag: str):
        self.spoken_language = lang
        self._tag = tag
        self.calls: list[str] = []
        self.warmed = False
        self.released = False

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        return b""

    async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
        self.calls.append(text)
        return f"[{self._tag}:{text}]".encode()

    async def warmup(self) -> None:
        self.warmed = True

    def release(self) -> None:
        self.released = True


def test_router_dispatches_runs_and_joins_with_a_gap():
    zh, en = _Eng("zh", "zh"), _Eng("en", "en")
    r = ScriptRoutedTts(zh, en)
    assert r.spoken_languages == ("zh", "en") and r.output_rate == 22050
    out = asyncio.run(r.synthesize_pcm("你好，请打开WiFi设置。"))
    gap = b"\x00\x00" * int(0.06 * 22050)
    assert out == (
        "[zh:你好，请打开]".encode() + gap + b"[en:WiFi]" + gap + "[zh:设置。]".encode()
    )
    # Single-run text passes through gapless, whatever the language order.
    flipped = ScriptRoutedTts(_Eng("en", "en"), _Eng("zh", "zh"))
    assert asyncio.run(flipped.synthesize_pcm("你好。")) == "[zh:你好。]".encode()
    assert asyncio.run(flipped.synthesize_pcm("42!")) == b"[en:42!]"  # primary takes neutral

    asyncio.run(r.warmup())
    r.release()
    assert zh.warmed and en.warmed and zh.released and en.released


def test_router_rejects_incoherent_pairs():
    with pytest.raises(ValueError, match="one CJK-language engine"):
        ScriptRoutedTts(_Eng("en", "a"), _Eng("de", "b"))
    slow = _Eng("zh", "zh")
    slow.output_rate = 16000
    with pytest.raises(ValueError, match="output rates differ"):
        ScriptRoutedTts(slow, _Eng("en", "en"))


# ---- config + build wiring --------------------------------------------------


def test_secondary_config_parses_and_never_nests():
    cfg = MatchaTtsConfig.model_validate({
        "acousticModelPath": "zh.onnx", "vocoderPath": "v.onnx",
        "tokensPath": "t.txt", "lexiconPath": "l.txt",
        "secondary": {"acousticModelPath": "en.onnx", "vocoderPath": "v.onnx",
                      "tokensPath": "t2.txt"},
    })
    assert cfg.secondary is not None and cfg.secondary.acoustic_model_path == "en.onnx"
    with pytest.raises(Exception, match="do not nest"):
        MatchaTtsConfig.model_validate({
            "acousticModelPath": "a.onnx",
            "secondary": {"acousticModelPath": "b.onnx",
                          "secondary": {"acousticModelPath": "c.onnx"}},
        })


def _matcha_cfg(**secondary) -> TtsConfig:
    return TtsConfig.model_validate({
        "provider": "matcha",
        "matcha": {
            "acousticModelPath": "zh.onnx", "vocoderPath": "v.onnx",
            "tokensPath": "t.txt", "lexiconPath": "l.txt",
            "secondary": {"acousticModelPath": "en.onnx", "tokensPath": "t2.txt",
                          **secondary},
        },
    })


def test_build_shares_the_vocoder_only_on_equal_paths(monkeypatch):
    from nanobot_channel_voice.tts import _build_matcha, matcha

    shares = []

    class FakeAdapter:
        @classmethod
        def from_config(cls, cfg, *, vocoder_share=None):
            shares.append(vocoder_share)
            eng = _Eng("zh" if cfg.lexicon_path else "en", cfg.acoustic_model_path)
            eng._vocoder = object()
            return eng

    monkeypatch.setattr(matcha, "MatchaTtsAdapter", FakeAdapter)
    router = _build_matcha(_matcha_cfg(vocoderPath="v.onnx"))
    assert isinstance(router, ScriptRoutedTts)
    assert shares[1] is not None                     # same path: primary's spec shared
    shares.clear()
    _build_matcha(_matcha_cfg(vocoderPath="other.onnx"))
    assert shares[1] is None                         # different vocoder: own session


def test_build_releases_the_primary_when_the_secondary_fails(monkeypatch):
    from nanobot_channel_voice.tts import _build_matcha, matcha

    built = []

    class FakeAdapter:
        @classmethod
        def from_config(cls, cfg, *, vocoder_share=None):
            if not cfg.lexicon_path:
                raise ValueError("secondary boom")
            eng = _Eng("zh", "zh")
            eng._vocoder = None
            built.append(eng)
            return eng

    monkeypatch.setattr(matcha, "MatchaTtsAdapter", FakeAdapter)
    with pytest.raises(ValueError, match="secondary boom"):
        _build_matcha(_matcha_cfg())
    assert built[0].released


def test_context_line_names_both_languages():
    from nanobot_channel_voice.channel import _voice_context_blocks

    (block,) = _voice_context_blocks(ScriptRoutedTts(_Eng("zh", "a"), _Eng("en", "b")))
    assert "'zh' and 'en'" in block.content
    assert "mixing them is fine" in block.content
    (single,) = _voice_context_blocks(_Eng("zh", "a"))
    assert "only pronounce ISO 639-1 language 'zh'" in single.content


def test_router_requires_declared_languages():
    with pytest.raises(ValueError, match="declare spoken_language"):
        ScriptRoutedTts(_Eng(None, "a"), _Eng("zh", "b"))  # type: ignore[arg-type]


def test_build_releases_both_when_the_pair_is_incoherent(monkeypatch):
    from nanobot_channel_voice.tts import _build_matcha, matcha

    built = []

    class FakeAdapter:
        @classmethod
        def from_config(cls, cfg, *, vocoder_share=None):
            eng = _Eng("zh" if cfg.lexicon_path else None, "x")  # type: ignore[arg-type]
            eng._vocoder = None
            built.append(eng)
            return eng

    monkeypatch.setattr(matcha, "MatchaTtsAdapter", FakeAdapter)
    with pytest.raises(ValueError, match="declare spoken_language"):
        _build_matcha(_matcha_cfg())
    assert [e.released for e in built] == [True, True]  # RKNN contexts must not leak
