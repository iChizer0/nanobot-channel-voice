"""Script-routed bilingual TTS (``tts.matcha.secondary``): two single-language
engines behind one adapter, text split into script runs — CJK runs to the
CJK-language engine, everything else to the Latin one, neutral characters
(digits, punctuation, space) riding with the run they follow. Runs synthesize
SEQUENTIALLY (on-device engines keep one decode in flight) and concatenate with
a short gap; each engine's own frontend/normalization applies to its runs.
Engine-agnostic in principle, but both must share ``output_rate``.
"""

from __future__ import annotations

from nanobot_channel_voice.audio.pcm import pcm_to_wav_bytes
from nanobot_channel_voice.tts.base import TtsAdapter

_CJK_LANGS = frozenset({"zh", "ja", "ko"})
_GAP_S = 0.06  # code-switch seam; the engines' edge fades land in it


def _is_cjk(ch: str) -> bool:
    cp = ord(ch)
    # CJK punctuation (0x3000-303F) and fullwidth forms (FF01-FF5E) stay neutral:
    # both engines fold them, and a lone 。 must not force an engine switch.
    if 0x3000 <= cp <= 0x303F or 0xFF01 <= cp <= 0xFF5E:
        return False
    return cp >= 0x2E80


def script_runs(text: str) -> list[tuple[bool, str]]:
    """(is_cjk, run) spans covering ``text``; a text with no scripted character
    at all yields one ``(False, text)`` run (the caller's primary handles it)."""
    runs: list[tuple[bool, str]] = []
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
    runs.append((cls if cls is not None else False, text[start:]))
    return runs


class ScriptRoutedTts(TtsAdapter):
    """``primary``/``secondary`` in config order; routing is by LANGUAGE, so
    either slot may hold the CJK engine. ``spoken_language`` stays None (no
    single-language constraint); ``spoken_languages`` carries both for the
    channel's context line."""

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
        self._gap = b"\x00\x00" * int(_GAP_S * self.output_rate)

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        pcm = await self.synthesize_pcm(text, voice=voice)
        return pcm_to_wav_bytes(pcm, self.output_rate) if pcm else b""

    async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
        runs = script_runs(text)
        if len(runs) == 1:
            return await self._engine(runs[0][0]).synthesize_pcm(runs[0][1])
        parts = []
        for cjk, run in runs:
            pcm = await self._engine(cjk).synthesize_pcm(run)
            if pcm:
                parts.append(pcm)
        return self._gap.join(parts)

    def _engine(self, cjk: bool) -> TtsAdapter:
        return self._cjk if cjk else self._latin

    async def warmup(self) -> None:
        for engine in self._engines:
            await engine.warmup()

    def release(self) -> None:
        for engine in self._engines:
            engine.release()
