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
    assert script_runs("42!") == [(None, "42!")]           # nothing scripted -> primary


def test_degree_fold_keeps_temperatures_whole_across_the_split():
    # synthesize_pcm folds °C/°F to their single non-alpha codepoints before
    # classifying: splitting "今天25°C" before the C would strand it on the Latin
    # engine and hide the unit from the zh degrees pass. One grammar (_RE_DEGREE_MARK)
    # owns the fold, so the spaced form works and "°Chill"-style words stay words.
    from nanobot_channel_voice.tts.text_frontend import fold_degree_marks

    assert script_runs(fold_degree_marks("今天25°C很热")) == [(True, "今天25℃很热")]
    assert script_runs(fold_degree_marks("今天25° C很热")) == [(True, "今天25℃很热")]
    assert fold_degree_marks("今天25°Chill很热") == "今天25°Chill很热"
    assert script_runs(fold_degree_marks("It is 25°F, 挺热")) == [
        (False, "It is 25℉, "), (True, "挺热"),
    ]
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


def test_router_dispatches_runs_with_continuation_hints():
    zh, en = _Eng("zh", "zh"), _Eng("en", "en")
    r = ScriptRoutedTts(zh, en)
    assert r.spoken_languages == ("zh", "en") and r.output_rate == 22050
    # Non-final fragments carry a continuation comma (clause contour + voiced
    # pause AT THE ENGINE), so the join is plain concatenation, no silent gap.
    out = asyncio.run(r.synthesize_pcm("你好，请打开WiFi设置。"))
    assert out == "[zh:你好，请打开，][en:WiFi,][zh:设置。]".encode()
    # A fragment already ending in pause punctuation is not double-hinted.
    assert asyncio.run(r.synthesize_pcm("你好。OK")) == "[zh:你好。][en:OK]".encode()
    # Single-run text passes through untouched, whatever the language order.
    flipped = ScriptRoutedTts(_Eng("en", "en"), _Eng("zh", "zh"))
    assert asyncio.run(flipped.synthesize_pcm("你好。")) == "[zh:你好。]".encode()
    assert asyncio.run(flipped.synthesize_pcm("42!")) == b"[en:42!]"  # primary takes neutral
    # ... whichever slot the primary is: a digits-only chunk must speak the
    # session's main language, not whichever engine happens to be Latin.
    assert asyncio.run(r.synthesize_pcm("338!")) == b"[zh:338!]"

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

    (block,) = _voice_context_blocks(None, ScriptRoutedTts(_Eng("zh", "a"), _Eng("en", "b")))
    assert "'zh' and 'en'" in block.content
    assert "mixing is fine" in block.content
    (single,) = _voice_context_blocks(None, _Eng("zh", "a"))
    assert "only pronounces ISO 639-1 language 'zh'" in single.content


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
