"""Shared shell for the on-device numpy TTS engines (mms, supertonic, matcha).

Owns everything EXCEPT the model math: strip/empty guards, the ``to_thread`` hop off
the event loop, the degrade-to-empty error policy ("never let one chunk kill the
player"), warmup, the speakability guard, and the split-for-budget -> synthesize ->
join-with-gap loop. An engine supplies :meth:`_synthesize_piece` plus the knobs below.
Imported only by the engine modules, which the registry imports lazily.
"""

from __future__ import annotations

import asyncio

import numpy as np
from loguru import logger

from nanobot_channel_voice.tts.base import (
    TtsAdapter,
    floats_to_pcm,
    floats_to_wav,
    split_for_budget,
)

# Below this fraction of voiceable content chars a piece is a language the engine does
# not speak, not "text with a few gaps", and synthesizing is worse than skipping:
# Supertonic resolves unknown ids from the end of the embedding table (noise), MMS drops
# them (word salad), both indistinguishable from a broken speaker, while a skip + WARNING
# names the misconfiguration. Above it, the drops are stray accents or symbols.
_MIN_SPEAKABLE = 0.5


class OnDeviceTtsAdapter(TtsAdapter):
    # Subclasses set (class- or instance-level): output_rate = sample rate of
    # _synthesize_piece output; _label = engine name in the failure log line;
    # _join_gap_s = silence between budget-split pieces; _log = a bound logger.
    _label: str = "on-device"
    _join_gap_s: float = 0.1
    _log = logger

    def __init__(self) -> None:
        # Warn once per character per adapter: a wrong-language reply repeats the same
        # chars every chunk.
        self._warned_unspeakable: set[str] = set()

    # ---- knobs the engine supplies ------------------------------------------

    def _piece_budget(self) -> int:
        """Max characters per synthesized piece (model input budget)."""
        raise NotImplementedError

    def _normalize(self, text: str) -> str:
        """Text front-end hook, run once before budget splitting."""
        return text

    def _can_speak(self, ch: str) -> bool:
        """Can this engine voice ``ch``? Default True: an engine that cannot answer
        opts out of the guard rather than blocking synthesis it might manage."""
        return True

    def _synthesize_piece(self, text: str) -> np.ndarray:
        """One budget-sized piece -> float32 waveform at ``output_rate``."""
        raise NotImplementedError

    # ---- the speakability guard ---------------------------------------------

    def _speakability(self, text: str) -> tuple[float, set[str]]:
        """(voiceable fraction, unvoiceable chars) over the CONTENT (alnum) characters.

        Every char vocab here drops punctuation by design, so scoring it would flag
        ordinary English as unspeakable; no content at all (digits-only after
        verbalization, punctuation, empty) scores 1.0: nothing to get wrong."""
        content = [c for c in text if c.isalnum()]
        if not content:
            return 1.0, set()
        bad = {c for c in content if not self._can_speak(c)}
        if not bad:
            return 1.0, set()
        return sum(c not in bad for c in content) / len(content), bad

    def _warn_unspeakable(self, chars: set[str]) -> None:
        fresh = chars - self._warned_unspeakable
        if not fresh:
            return
        self._warned_unspeakable.update(fresh)
        self._log.warning(
            "{} TTS cannot voice {!r} (language {}): those characters are NOT spoken. "
            "Configure a TTS for the language the agent replies in, or constrain the "
            "agent to the configured one.",
            self._label, "".join(sorted(fresh)), self.spoken_language or "unset",
        )

    # ---- the shared shell ---------------------------------------------------

    async def synthesize(self, text: str, *, voice: str | None = None) -> bytes:
        text = text.strip()
        if not text:
            return b""
        # Inference is blocking (NPU/CPU bound); keep it off the event loop.
        return await asyncio.to_thread(self._synthesize_sync, text)

    async def synthesize_pcm(self, text: str, *, voice: str | None = None) -> bytes:
        text = text.strip()
        if not text:
            return b""
        return await asyncio.to_thread(self._synthesize_pcm_sync, text)

    async def warmup(self) -> None:
        await self.synthesize("Okay.")

    def _synthesize_sync(self, text: str) -> bytes:
        try:
            samples = self._synthesize_floats(text)
            return floats_to_wav(samples, self.output_rate) if samples.size else b""
        except Exception as exc:  # noqa: BLE001 - never let one chunk kill the player
            self._log.warning("on-device {} TTS failed: {}", self._label, exc)
            return b""

    def _synthesize_pcm_sync(self, text: str) -> bytes:
        try:
            samples = self._synthesize_floats(text)
            return floats_to_pcm(samples) if samples.size else b""
        except Exception as exc:  # noqa: BLE001 - never let one chunk kill the player
            self._log.warning("on-device {} TTS failed: {}", self._label, exc)
            return b""

    def _join_gap(self) -> np.ndarray:
        return np.zeros(int(self._join_gap_s * self.output_rate), dtype=np.float32)

    def _halve_and_retry(self, text: str) -> np.ndarray:
        """Halve at the space (or midpoint) nearest the middle and synthesize both:
        for fixed-window overflows. Callers own the unsplittable single-char case."""
        text = text.strip()
        mid = (len(text) + 1) // 2
        left = text.rfind(" ", 1, mid)
        right = text.find(" ", mid, len(text) - 1)
        cands = [c for c in (left, right) if c > 0]
        cut = (min(cands, key=lambda c: abs(c - mid)) + 1) if cands else mid
        parts = [
            p for p in (
                self._synthesize_piece(text[:cut].strip()),
                self._synthesize_piece(text[cut:].strip()),
            ) if p.size
        ]
        if len(parts) == 2:
            parts.insert(1, self._join_gap())
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    def _synthesize_floats(self, text: str) -> np.ndarray:
        text = self._normalize(text)
        budget = self._piece_budget()
        pieces = split_for_budget(text, budget)
        if len(pieces) > 1:
            self._log.debug("split into {} pieces for the {}-char budget", len(pieces), budget)
        gap = self._join_gap()
        waves: list[np.ndarray] = []
        failed: list[Exception] = []
        skipped = 0
        for piece in pieces:
            ratio, unvoiceable = self._speakability(piece)
            if unvoiceable:
                self._warn_unspeakable(unvoiceable)
            if ratio < _MIN_SPEAKABLE:
                # Per piece, not per reply: an English answer quoting one foreign
                # sentence keeps the English and drops only the quote.
                skipped += 1
                continue
            try:
                # Salvage is top-level only: an engine may recurse inside
                # _synthesize_piece, so a raise in a sub-half drops the whole piece.
                samples = self._synthesize_piece(piece)
            except Exception as exc:  # noqa: BLE001 - keep the pieces that worked
                failed.append(exc)
                continue
            if samples.size:
                if waves:
                    waves.append(gap)
                waves.append(samples)
        if skipped:
            self._log.warning(
                "on-device {} TTS: {}/{} pieces are mostly unvoiceable and were NOT "
                "spoken; {}",
                self._label, skipped, len(pieces),
                "speaking the rest" if waves else "this chunk is silent",
            )
        if failed:
            self._log.warning(
                "on-device {} TTS: {}/{} pieces failed ({}); {}",
                self._label, len(failed), len(pieces), failed[-1],
                "speaking the rest" if waves else "nothing speakable left",
            )
        if not waves:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(waves)
