"""Script-routed bilingual TTS (``tts.matcha.secondary``): two single-language
engines behind one adapter, text split into script runs — CJK runs to the
CJK-language engine, everything else to the Latin one, neutral characters
(digits, punctuation, space) riding with the run they follow. Runs synthesize
SEQUENTIALLY (on-device engines keep one decode in flight), and both engines
must share ``output_rate``. Loudness is deliberately NOT equalized: measured
cross-engine mismatch ~1.5 dB is below within-engine variance.
"""

from __future__ import annotations

from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes
from nanobot_channel_voice.tts.base import TtsAdapter
from nanobot_channel_voice.tts.ondevice_base import edge_trim_pcm
from nanobot_channel_voice.tts.text_frontend import fold_degree_marks

_CJK_LANGS = frozenset({"zh", "ja", "ko"})
_PAUSE_PUNCT = set("，,、；;。.!?！？…：:")


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    # CJK punctuation (0x3000-303F) and fullwidth forms (FF01-FF5E) stay neutral:
    # both engines fold them, and a lone 。 must not force an engine switch.
    if 0x3000 <= cp <= 0x303F or 0xFF01 <= cp <= 0xFF5E:
        return False
    return cp >= 0x2E80


def script_runs(text: str) -> list[tuple[bool | None, str]]:
    """(is_cjk, run) spans covering ``text``; text with no scripted character at all
    yields one ``(None, text)`` run, which PRIMARY takes — a digits-only chunk speaks
    in the session's main language, not whichever engine happens to be Latin."""
    runs: list[tuple[bool | None, str]] = []
    cls: bool | None = None
    start = 0
    for i, ch in enumerate(text):
        if not ch.isalpha() and not _is_cjk(ch):
            continue  # neutral: rides with the current run
        c = _is_cjk(ch)
        if cls is None:
            cls = c
        elif c != cls:
            runs.append((cls, text[start:i]))
            cls, start = c, i
    runs.append((cls, text[start:]))
    return runs


def _hint(run: str, cjk: bool) -> str:
    """Continuation comma for a NON-FINAL fragment: measured to buy a voiced pause and
    a non-terminal contour, so seams are model-voiced, never synthetic silence."""
    stripped = run.rstrip()
    if not stripped or stripped[-1] in _PAUSE_PUNCT:
        return run
    return stripped + ("，" if cjk else ",")


class ScriptRoutedTts(TtsAdapter):
    """``primary``/``secondary`` in config order; routing is by LANGUAGE, so either
    slot may hold the CJK engine. ``spoken_language`` stays None (no single-language
    constraint); ``spoken_languages`` carries both for the channel's context line."""

    def __init__(self, primary: TtsAdapter, secondary: TtsAdapter):
        p_lang, s_lang = primary.spoken_language, secondary.spoken_language
        if primary.output_rate != secondary.output_rate:
            raise ValueError(
                f"matcha secondary: output rates differ "
                f"({primary.output_rate} vs {secondary.output_rate})"
            )
        if not p_lang or not s_lang:
            raise ValueError(
                "matcha secondary: both engines must declare spoken_language "
                f"(got {p_lang!r} and {s_lang!r})"
            )
        if (p_lang in _CJK_LANGS) == (s_lang in _CJK_LANGS):
            raise ValueError(
                f"matcha secondary: need one CJK-language engine and one Latin "
                f"(got {p_lang!r} and {s_lang!r})"
            )
        self._primary = primary
        self._cjk = primary if p_lang in _CJK_LANGS else secondary
        self._latin = secondary if self._cjk is primary else primary
        self._engines = (primary, secondary)
        self.output_rate = primary.output_rate
        self.spoken_languages = (p_lang, s_lang)

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        pcm = await self.synthesize_pcm(text, voice=voice)
        return pcm_to_wav_bytes(pcm, self.output_rate) if pcm else b""

    async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
        # ℃/℉ are non-alpha: folded, a temperature never splits its scale letter
        # onto the Latin engine ("今天25°C").
        runs = script_runs(fold_degree_marks(text))
        if len(runs) == 1:
            cjk, run = runs[0]
            engine = self._primary if cjk is None else self._engine(cjk)
            return await engine.synthesize_pcm(run)
        parts = []
        for i, (cjk, run) in enumerate(runs):
            if i < len(runs) - 1:
                run = _hint(run, cjk)
            pcm = await self._engine(cjk).synthesize_pcm(run)
            if pcm:
                parts.append(pcm)
        # A script switch is not an utterance boundary: the LEAD padding of every part
        # after the first goes (~0.3 s on zh). Tails stay: they hold the model-voiced
        # pause the _hint comma bought. Both engines share output_rate.
        return b"".join(
            edge_trim_pcm(pcm, self.output_rate, lead=i > 0) for i, pcm in enumerate(parts)
        )

    def _engine(self, cjk: bool) -> TtsAdapter:
        return self._cjk if cjk else self._latin

    async def warmup(self) -> None:
        for engine in self._engines:
            await engine.warmup()

    def release(self) -> None:
        for engine in self._engines:
            engine.release()
