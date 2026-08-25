"""The on-device speakability guard: score, warn once, skip the unvoiceable piece.

Exercised through a stub engine (no models, no numpy math beyond the shell's own) plus
the two real ``_can_speak`` implementations.
"""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from loguru import logger as loguru_logger  # noqa: E402

from nanobot_channel_voice.tts.ondevice_base import (  # noqa: E402
    _MIN_SPEAKABLE,
    OnDeviceTtsAdapter,
)


class _AsciiEngine(OnDeviceTtsAdapter):
    """Speaks ASCII letters only; one sample per character, so waveform length says
    exactly which pieces were synthesized."""

    output_rate = 16000
    _label = "stub"
    _join_gap_s = 0.0
    spoken_language = "en"

    def __init__(self, budget: int = 1000):
        super().__init__()
        self._budget = budget
        self.spoken: list[str] = []

    def _piece_budget(self) -> int:
        return self._budget

    def _can_speak(self, ch: str) -> bool:
        return ch.isascii() and ch.isalpha()

    def _synthesize_piece(self, text: str) -> np.ndarray:
        self.spoken.append(text)
        return np.ones(len(text), dtype=np.float32)


class _PaddedEngine(_AsciiEngine):
    """One loud sample per character, wrapped in the model's silent utterance padding."""

    _join_gap_s = 0.1

    def _synthesize_piece(self, text: str) -> np.ndarray:
        self.spoken.append(text)
        pad = np.zeros(int(0.6 * self.output_rate), dtype=np.float32)
        return np.concatenate([pad, np.ones(len(text), dtype=np.float32), pad])


def _warnings(fn) -> list[str]:
    messages: list[str] = []
    sink = loguru_logger.add(lambda m: messages.append(str(m)), level="WARNING")
    try:
        fn()
    finally:
        loguru_logger.remove(sink)
    return messages


# ---- scoring ----------------------------------------------------------------


def test_punctuation_and_spaces_never_count_against_a_piece():
    # MMS vocabs carry no punctuation at all; scoring it would flag plain English.
    ratio, bad = _AsciiEngine()._speakability("Hello, world — it's 100% fine!")
    assert bad == {"1", "0"}                    # only the digits this stub cannot say
    assert ratio == pytest.approx(17 / 20)      # 20 content chars, "100" unvoiceable
    assert ratio > _MIN_SPEAKABLE               # ...and the piece still speaks


def test_textless_input_scores_speakable():
    # Nothing to get wrong: no content chars means no claim either way.
    assert _AsciiEngine()._speakability("... —— !?") == (1.0, set())


def test_foreign_script_scores_zero():
    ratio, bad = _AsciiEngine()._speakability("你好世界")
    assert ratio == 0.0
    assert bad == set("你好世界")
    assert ratio < _MIN_SPEAKABLE


# ---- the guard in the synthesis loop ----------------------------------------


def test_unvoiceable_piece_is_skipped_and_the_rest_still_speaks():
    engine = _AsciiEngine(budget=20)
    # Two budget pieces: one English, one Chinese.
    out = engine._synthesize_floats("hello there friend " + "你好世界再见吧朋友们啊")
    assert engine.spoken == ["hello there friend"]
    assert out.size == len("hello there friend")


def test_a_fully_unvoiceable_chunk_is_silent_not_noise():
    engine = _AsciiEngine()
    assert engine._synthesize_floats("你好世界").size == 0
    assert engine.spoken == []  # the model is never asked to voice it


def test_a_few_stray_chars_still_speak():
    # Above the threshold the piece is a sentence with gaps, not another language.
    engine = _AsciiEngine()
    assert engine._synthesize_floats("naive resume of Zoe").size > 0
    assert engine.spoken


def test_warning_names_the_characters_once_per_adapter():
    engine = _AsciiEngine()
    msgs = _warnings(lambda: [
        engine._synthesize_floats("你好世界"),
        engine._synthesize_floats("你好世界"),  # same chars: already reported
    ])
    named = [m for m in msgs if "cannot voice" in m]
    assert len(named) == 1
    assert "你" in named[0]
    assert "en" in named[0]  # the configured language, so the fix is obvious
    # A fresh adapter reports again: the state is per instance, not per process.
    assert [m for m in _warnings(lambda: _AsciiEngine()._synthesize_floats("你好"))
            if "cannot voice" in m]


def test_skipped_pieces_are_reported_separately_from_failures():
    engine = _AsciiEngine()
    msgs = _warnings(lambda: engine._synthesize_floats("你好世界"))
    assert any("mostly unvoiceable" in m and "silent" in m for m in msgs)


# ---- the real engines' verdicts ---------------------------------------------


def test_mms_can_speak_follows_its_vocab():
    from nanobot_channel_voice.tts.mms import _ENG_VOCAB, MmsTtsAdapter

    adapter = MmsTtsAdapter(
        encoder=None, decoder=None, vocab=_ENG_VOCAB, frontend=None,  # type: ignore[arg-type]
        max_length=200, speaking_rate=1.0,
    )
    assert adapter._can_speak("A")          # lowercased before lookup, as the tokenizer does
    assert adapter._can_speak("3")
    assert not adapter._can_speak("7")      # mms-tts-eng really has no 7/8/9
    assert not adapter._can_speak("好")


def test_supertonic_can_speak_follows_its_indexer():
    from nanobot_channel_voice.tts.supertonic import SupertonicTtsAdapter

    indexer = np.full(65536, -1, dtype=np.int32)
    for ch in "abc":
        indexer[ord(ch)] = ord(ch)
    adapter = SupertonicTtsAdapter.__new__(SupertonicTtsAdapter)  # no models needed
    adapter._indexer = indexer

    assert adapter._can_speak("a")
    assert not adapter._can_speak("z")      # in range, but unmapped => gathered as noise
    assert not adapter._can_speak("好")
    assert not adapter._can_speak("😀")     # non-BMP decomposes to nothing => silence


def test_an_interior_budget_seam_drops_the_model_padding():
    engine = _PaddedEngine(budget=4)
    wav = engine._synthesize_floats("aaaa bbbb")
    assert engine.spoken == ["aaaa", "bbbb"]
    loud = np.flatnonzero(wav > 0.5)
    interior = int(loud[-1] - loud[0] + 1 - loud.size)
    # 10 ms kept per side plus the engine's own 100 ms join gap -- not 2x600 ms of padding.
    assert 115 <= interior / engine.output_rate * 1000 <= 125
    # Outer edges are left alone: the backend sizes those against the NEXT chunk.
    assert loud[0] >= int(0.5 * engine.output_rate)
