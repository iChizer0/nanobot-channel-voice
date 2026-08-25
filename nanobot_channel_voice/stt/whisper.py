"""On-device Whisper ASR (RKNN/ONNX), ported from the Rockchip rknn_model_zoo
``examples/whisper`` demo: a log-mel front-end feeds a fixed-length encoder, then a
sliding 12-token window decoder greedily emits the transcript. Preprocessing is numpy
only (no torch on the NPU board); ``numpy`` is imported at module load, so
:func:`make_stt` imports this module lazily.
"""

from __future__ import annotations

import asyncio
from contextlib import ExitStack

import numpy as np
from loguru import logger
from numpy.lib.stride_tricks import sliding_window_view

from nanobot_channel_voice.config import WhisperSttConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.stt.base import SttAdapter, pcm_to_float_mono
from nanobot_channel_voice.stt.whisper_tokenizer import (
    byte_level_decode,
    detect_language,
    language_token,
    language_tokens,
    read_vocab,
    resolve_languages,
    suppressed_token_ids,
)

# Audio front-end; must match the exported model (chunkLength is configurable).
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80
MELS_FILTERS_SIZE = N_FFT // 2 + 1

# Structural tokens, fixed for the tiny/base/medium family; the language token itself
# is data (whisper_tokenizer.language_token).
SOT = 50258            # <|startoftranscript|>
EOT = 50257            # <|endoftext|>
TRANSCRIBE = 50359     # <|transcribe|>
NOTIMESTAMPS = 50363   # <|notimestamps|>
TIMESTAMP_BEGIN = 50364
MAX_TOKENS = 12        # decoder context window (the exported decoder is fixed at this)
MAX_DECODE_STEPS = 448  # cap: a model that never emits EOT must not hang the daemon


def load_mel_filters(path: str) -> np.ndarray:
    """Load the (80, 201) mel filterbank from the flat text file (ref ``mel_filters``)."""
    return np.loadtxt(path, dtype=np.float32).reshape((N_MELS, MELS_FILTERS_SIZE))


def log_mel_spectrogram(audio: np.ndarray, mel_filters: np.ndarray) -> np.ndarray:
    """numpy port of the reference ``log_mel_spectrogram`` (== torch.stft, center=True)."""
    # Periodic Hann, identical to torch.hann_window(N_FFT).
    window = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    padded = np.pad(audio, N_FFT // 2, mode="reflect")             # center=True
    frames = sliding_window_view(padded, N_FFT)[::HOP_LENGTH]      # (n_frames, N_FFT)
    spec = np.fft.rfft(frames * window, n=N_FFT, axis=1)          # (n_frames, 201)
    # |z|**2 directly: np.abs would pay a sqrt per bin that this squares away.
    s = spec[:-1]
    mag = np.square(s.real)
    mag += np.square(s.imag)
    # Pin float32 whatever rfft returned (complex128 under numpy<2): halves the
    # (80,201)@(201,n) memory traffic.
    magnitudes = mag.T.astype(np.float32)                          # (201, n_frames-1)
    mel_spec = mel_filters @ magnitudes                            # (80, n_frames-1)
    log_spec = np.log10(np.clip(mel_spec, 1e-10, None))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)


class WhisperOnDeviceStt(SttAdapter):
    decoder_family = "attention"

    def __init__(
        self,
        *,
        encoder: OnDeviceModel,
        decoder: OnDeviceModel,
        vocab: dict[str, str],
        mel_filters: np.ndarray,
        lang_token: int,
        chunk_length: int,
        max_frames: int | None = None,
        candidates: dict[int, str] | None = None,
        min_confidence: float = 0.0,
        suppressed: tuple[int, ...] = (),
    ):
        self._encoder = encoder
        self._decoder = decoder
        self._vocab = vocab
        self._mel_filters = mel_filters
        self._lang_token = lang_token       # PREFERRED language token (en=50259, ja=50266, ...)
        # DECODABLE languages: per-utterance ID candidates; empty => fixed to lang_token.
        self._candidates = candidates or {}
        self._min_confidence = min_confidence
        # Out-of-language ids: O(1) membership array (hot path) + index array (masked).
        self._suppressed_ids = np.asarray(suppressed, dtype=np.int64) if suppressed else None
        self._blocked: np.ndarray | None = None
        if self._suppressed_ids is not None:
            self._blocked = np.zeros(int(self._suppressed_ids.max()) + 1, dtype=bool)
            self._blocked[self._suppressed_ids] = True
        # Encoder time dim; from_config prefers the export's own when it declares one.
        self._max_frames = max_frames or chunk_length * 100
        self.max_decode_ms = self._max_frames * 10  # mel hop = 10 ms at 16 kHz
        self._log = logger.bind(component="stt-whisper")

    @classmethod
    def from_config(cls, cfg: WhisperSttConfig) -> WhisperOnDeviceStt:
        lang, codes = resolve_languages(cfg.language, cfg.languages)
        requested = (cfg.language or "en").lower()
        if requested != lang and "language" in cfg.model_fields_set:
            # Fires for a single-entry languages list too (codes collapses to ()), but
            # only when the user actually SET a language.
            detail = (
                f"stt.whisper.languages {list(codes)}"
                if codes else "a single-entry stt.whisper.languages"
            )
            logger.warning(
                "voice: stt.whisper.language '{}' is not honored ({} prefers '{}'). "
                "The decodable set is a guarantee, so it is not widened to include it.",
                requested, detail, lang,
            )
        # Unknown codes raise before any model loads, so make_stt falls back cleanly.
        candidates = language_tokens(codes)
        lang_token = language_token(lang)
        # Small assets first, so a bad path fails before loading models.
        vocab = read_vocab(cfg.vocab_path)        # type: ignore[arg-type]
        suppressed = suppressed_token_ids(vocab, codes)
        mel_filters = load_mel_filters(cfg.mel_filters_path)  # type: ignore[arg-type]
        model_kw = dict(
            core_mask=cfg.core_mask, target=cfg.target, device_id=cfg.device_id,
            providers=cfg.execution_providers, provider_options=cfg.provider_options,
        )
        with ExitStack() as models:  # any failure below releases every loaded model
            encoder = models.enter_context(
                OnDeviceModel(cfg.encoder_path, **model_kw)  # type: ignore[arg-type]
            )
            decoder = models.enter_context(
                OnDeviceModel(cfg.decoder_path, **model_kw)  # type: ignore[arg-type]
            )
            # Prefer the export's own window over chunkLength when it declares one
            # (ONNX only): a mismatch makes every encoder run raise.
            max_frames = cfg.chunk_length * 100
            shape = encoder.input_shape("x")
            if shape is not None and len(shape) == 3:
                if shape[1] != N_MELS:
                    raise ValueError(
                        f"whisper encoder wants {shape[1]} mel bins; this front-end "
                        f"only produces {N_MELS}"
                    )
                if shape[2] != max_frames:
                    logger.warning(
                        "voice: stt.whisper.chunkLength={}s does not match the encoder "
                        "export ({:.0f}s window); using the model's",
                        cfg.chunk_length, shape[2] / 100,
                    )
                max_frames = shape[2]
            adapter = cls(
                encoder=encoder,
                decoder=decoder,
                vocab=vocab,
                mel_filters=mel_filters,
                lang_token=lang_token,
                chunk_length=cfg.chunk_length,
                max_frames=max_frames,
                candidates=candidates,
                min_confidence=cfg.language_min_confidence,
                suppressed=suppressed,
            )
            adapter._validate()  # inside the stack: a bad export releases both models
            models.pop_all()  # success: the adapter owns the models now
            return adapter

    def _validate(self) -> None:
        """Prove the encoder/decoder I/O contract HERE, so an incompatible export makes
        make_stt fall back instead of failing inside every utterance's swallowed try.
        Works on RKNN, which cannot introspect."""
        x_mel = np.zeros((1, N_MELS, self._max_frames), dtype=np.float32)
        try:
            out_encoder = self._encoder.run([("x", x_mel)])[0]
            self._decoder.run([
                ("tokens", np.asarray([self._window(self._lang_token)], dtype=np.int64)),
                ("audio", out_encoder),
            ])
        except Exception as exc:  # noqa: BLE001 - re-raise as a clear construction error
            raise RuntimeError(
                f"whisper export rejected the expected inputs (encoder 'x'"
                f"[1,{N_MELS},{self._max_frames}], decoder 'tokens'[1,{MAX_TOKENS}] "
                f"+ 'audio'): {exc}"
            ) from exc

    def release(self) -> None:
        for model in (self._encoder, self._decoder):
            model.release()

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if not pcm:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, pcm, sample_rate)

    async def warmup(self) -> None:
        """One dummy decode so the first real utterance pays no cold-start."""
        await self.transcribe(b"\x00" * SAMPLE_RATE, SAMPLE_RATE)  # 0.5 s of silence

    def _transcribe_sync(self, pcm: bytes, sample_rate: int) -> str:
        try:
            audio = pcm_to_float_mono(pcm, sample_rate, SAMPLE_RATE)
            if audio.shape[0] < N_FFT:
                return ""  # shorter than one analysis window; reflect-pad would raise
            mel = log_mel_spectrogram(audio, self._mel_filters)
            if mel.shape[1] > self._max_frames:
                self._log.warning(
                    "utterance ({:.1f}s) exceeds the {:.0f}s encoder window; tail dropped "
                    "-- long audio belongs on transcribe_chunked",
                    mel.shape[1] * HOP_LENGTH / SAMPLE_RATE,
                    self._max_frames * HOP_LENGTH / SAMPLE_RATE,
                )
            # Pad/trim the mel to the fixed encoder length. The reference zero-pads the
            # AUDIO, which lands at max(mel_max - 2.0, -1.5) after log10/floor/normalize;
            # plain 0.0 padding is the port mismatch behind hallucinations.
            fill = max(float(mel.max()) - 2.0, -1.5) if mel.size else 0.0
            x_mel = np.full((N_MELS, self._max_frames), fill, dtype=np.float32)
            real = min(mel.shape[1], self._max_frames)
            x_mel[:, :real] = mel[:, :real]
            x_mel = x_mel[None, ...]  # (1, 80, max_frames)

            out_encoder = self._encoder.run([("x", x_mel)])[0]
            return self._decode(out_encoder).strip()
        except Exception as exc:  # noqa: BLE001 - never let STT crash the capture loop
            self._log.warning("on-device STT failed: {}", exc)
            return ""

    def _pick_token(self, row) -> tuple[int, bool]:
        """Greedy pick constrained to the DECODABLE languages, plus whether it was
        re-picked. Per-decode state stays in the caller's locals: decodes on one adapter
        legitimately overlap. Deliberately optimistic — masking up front would copy ~52k
        floats every step, so only the rare blocked argmax pays a masked re-argmax."""
        token = int(row.argmax())
        if self._blocked is None or token >= self._blocked.shape[0] or not self._blocked[token]:
            return token, False
        masked = np.asarray(row, dtype=np.float32).copy()
        masked[self._suppressed_ids] = -np.inf
        return int(masked.argmax()), True

    def _window(self, lang_token: int) -> list[int]:
        """The 12-token decoder window: the real 4-token prompt, then filler. Indices
        0-3 are never evicted (``pop_id`` floors at 4), pinning SOT at position 0 —
        what makes language detection free."""
        return [SOT, lang_token, TRANSCRIBE, NOTIMESTAMPS] * (MAX_TOKENS // 4)

    def _decode(self, out_encoder) -> str:
        """Greedy sliding-window decode (reference ``run_decoder``). With languages
        enabled the first step also does language ID: ``out[0, 0]`` is the SOT row
        :func:`detect_language` reads; only a disagreement costs a redone step."""
        lang_token = self._lang_token
        tokens = self._window(lang_token)
        tokens_str = ""
        recent: list[int] = []  # last emitted TEXT tokens, for the repetition bail
        pop_id = MAX_TOKENS
        steps = 0
        suppressed_hits = 0
        hit_cap = False
        detect_pending = bool(self._candidates)

        while True:
            if steps >= MAX_DECODE_STEPS:
                hit_cap = True  # only exit that truncates; EOT and the bail are clean
                break
            steps += 1
            out = self._decoder.run(
                [("tokens", np.asarray([tokens], dtype=np.int64)), ("audio", out_encoder)]
            )[0]

            if detect_pending:
                detect_pending = False  # latched: at most one detection per utterance
                picked = detect_language(
                    out[0, 0], self._candidates, min_confidence=self._min_confidence
                )
                if picked is not None and picked != lang_token:
                    # The pass just run used the wrong prompt: its row is unusable.
                    self._log.debug(
                        "language detected: {} (was {})",
                        self._candidates[picked], self._candidates.get(lang_token, "?"),
                    )
                    lang_token = picked
                    tokens = self._window(lang_token)
                    continue

            next_token, suppressed = self._pick_token(out[0, -1])
            suppressed_hits += suppressed
            next_token_str = self._vocab.get(str(next_token), "")
            tokens.append(next_token)

            if next_token == EOT:
                break
            if next_token >= TIMESTAMP_BEGIN:
                # Timestamps stay out of the transcript but must still ROLL the window:
                # the reference grows it past the fixed 12, which the export rejects.
                if pop_id > 4:
                    pop_id -= 1
                tokens.pop(pop_id)
                continue
            if pop_id > 4:
                pop_id -= 1
            tokens.pop(pop_id)
            if next_token < EOT:
                # Non-timestamp specials roll the window but never reach the transcript:
                # the flat vocab carries their literal strings, so a greedy <|nospeech|>
                # on noise would reach the agent as user speech.
                tokens_str += next_token_str
                # Repetition trap (music, hum): eight IDENTICAL consecutive text tokens
                # never occur in real speech; bail and shed the looping tail.
                recent.append(next_token)
                if len(recent) > 8:
                    del recent[0]
                if len(recent) == 8 and len(set(recent)) == 1:
                    self._log.warning("greedy decode stuck repeating; bailing early")
                    if next_token_str:  # [:-0] would wipe the whole transcript
                        tokens_str = tokens_str[: -8 * len(next_token_str)]
                    break

        if hit_cap:
            self._log.warning("decode hit the {}-step cap; truncating", MAX_DECODE_STEPS)
        if suppressed_hits:
            # A steady stream means the decodable set is too narrow for what is said.
            self._log.debug(
                "{}/{} step(s) fell outside the decodable languages and were re-picked",
                suppressed_hits, steps,
            )

        return byte_level_decode(tokens_str)
