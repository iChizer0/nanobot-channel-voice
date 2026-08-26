"""The on-device speakability guard: score, warn once, skip the unvoiceable piece.

Exercised through a stub engine (no models, no numpy math beyond the shell's own) plus
the two real ``_can_speak`` implementations.
"""

from __future__ import annotations

import asyncio

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


class _QuietEngine(_AsciiEngine):
    """Utterance-length output at the wrong LEVEL: what a mis-set mel scale/bias gives.
    Long enough to be judged — the level check ignores sub-utterance fragments."""

    peak = 0.03

    def _synthesize_piece(self, text: str) -> np.ndarray:
        self.spoken.append(text)
        return np.full(int(0.5 * self.output_rate), self.peak, dtype=np.float32)


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


def test_a_mis_scaled_engine_is_named_once_per_language():
    """A level error crashes nothing and empties nothing — every downstream audibility,
    duck and barge-in judgement just reads a signal that is not there."""
    quiet = _QuietEngine()
    said = _warnings(lambda: asyncio.run(quiet.warmup()))
    peaks = [m for m in said if "peaks at" in m]
    assert len(peaks) == 1 and "0.030" in peaks[0] and "stub" in peaks[0]
    assert not [m for m in _warnings(lambda: asyncio.run(quiet.warmup())) if "peaks at" in m]

    loud = _AsciiEngine()                        # peak 1.0: nothing to say
    assert not [m for m in _warnings(lambda: asyncio.run(loud.warmup())) if "peaks at" in m]

    # A whole utterance of ZEROS is the case that matters most and nothing else reports it:
    # the bytes are non-empty, so every length and duration check downstream is satisfied.
    dead = _QuietEngine()
    dead.peak = 0.0
    assert [m for m in _warnings(lambda: asyncio.run(dead.warmup())) if "peaks at 0.000" in m]


def test_a_short_fragment_never_decides_the_level():
    """A single soft word is not evidence of a mis-scaled model: the verdict waits for a
    real utterance, and the engine stays unjudged until one arrives."""

    class _Fragment(_QuietEngine):
        def _synthesize_piece(self, text: str) -> np.ndarray:
            return np.full(int(0.05 * self.output_rate), self.peak, dtype=np.float32)

    tiny = _Fragment()
    assert not [m for m in _warnings(lambda: asyncio.run(tiny.warmup())) if "peaks at" in m]
    assert not tiny._level_checked                 # unjudged, not judged-and-cleared
    assert [m for m in _warnings(lambda: asyncio.run(_QuietEngine().warmup()))
            if "peaks at" in m]


def test_a_short_only_collapse_is_named_apart_from_a_dead_leg():
    """Measured on a converted matcha split: the warmup phrase (3 tokens) came out at 0.09
    while whole replies were fine. Both read as one mis-scaled peak, and the fixes differ —
    so the verdict re-measures the same leg on a long sentence before naming a cause."""

    class _ShortOnly(_QuietEngine):
        def _synthesize_piece(self, text: str) -> np.ndarray:
            self.spoken.append(text)
            loud = 0.9 if len(text) > 12 else 0.03
            return np.full(int(0.5 * self.output_rate), loud, dtype=np.float32)

    said = [m for m in _warnings(lambda: asyncio.run(_ShortOnly().warmup())) if "peaks at" in m]
    assert len(said) == 1 and "SHORT input only" in said[0]
    assert "0.030" in said[0] and "0.900" in said[0]     # both measurements, not one
    assert "melScale" not in said[0]                     # that fix is for a dead leg

    dead = [m for m in _warnings(lambda: asyncio.run(_QuietEngine().warmup())) if "peaks at" in m]
    assert "every length" in dead[0] and "melScale" in dead[0]


def test_each_declared_language_is_levelled_separately():
    """A bilingual routes scripts through different sub-frontends, so one leg can be dead
    while the other is fine — and a single one-shot check would miss whichever ran second."""

    class _HalfDead(_QuietEngine):
        spoken_languages = ("zh", "en")

        def _can_speak(self, ch: str) -> bool:
            return True                          # voiceable: this is the LEVEL path

        def _synthesize_piece(self, text: str) -> np.ndarray:
            loud = any(ord(c) < 0x2E80 for c in text)
            return np.full(int(0.5 * self.output_rate), 0.9 if loud else 0.0, np.float32)

    tts = _HalfDead()
    said = _warnings(lambda: asyncio.run(tts.warmup()))
    peaks = [m for m in said if "peaks at" in m]
    assert len(peaks) == 1 and "(zh)" in peaks[0]   # named, and only the dead leg
    assert tts._level_checked == {"zh", "en"}       # en was judged too, and passed
