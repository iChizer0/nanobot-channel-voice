"""On-device MMS-TTS (VITS) over RKNN/ONNX, ported from the Rockchip rknn_model_zoo
``examples/mms_tts`` demo: the encoder predicts per-token durations + prior
distribution, a host-side "middle process" (the only math between the two models,
reimplemented in numpy so the board needs no torch) expands them into the alignment
``attn`` and output mask, and the decoder vocodes the waveform. Imported lazily by
:func:`make_tts`, so the plugin imports without numpy.
"""

from __future__ import annotations

import json
from contextlib import ExitStack

import numpy as np
from loguru import logger

from nanobot_channel_voice.config import MmsTtsConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.tts.ondevice_base import OnDeviceTtsAdapter
from nanobot_channel_voice.tts.text_frontend import (
    TextFrontend,
    make_text_frontend,
    verbalize_numbers_en,
)

SAMPLE_RATE = 16000  # mms-tts output rate (ref sf.write samplerate=16000)
UPSAMPLE = 256       # waveform samples per output frame (prod of the decoder upsample rates)
_JOIN_GAP_S = 0.06   # short pause between budget-split pieces (masks the concat seam)


# facebook/mms-tts-eng character vocab (ref ``vocab``): the built-in default. Each MMS
# language is a separate model + vocab; others supply their own vocab.json by path.
_ENG_VOCAB = {
    " ": 19, "'": 1, "-": 14, "0": 23, "1": 15, "2": 28, "3": 11, "4": 27, "5": 35, "6": 36, "_": 30,
    "a": 26, "b": 24, "c": 12, "d": 5, "e": 7, "f": 20, "g": 37, "h": 6, "i": 18, "j": 16, "k": 0,
    "l": 21, "m": 17, "n": 29, "o": 22, "p": 13, "q": 34, "r": 25, "s": 8, "t": 33, "u": 4, "v": 32,
    "w": 9, "x": 31, "y": 3, "z": 2, "–": 10,  # en-dash U+2013 (ref vocab), not hyphen-minus
}


def load_mms_vocab(path: str) -> dict[str, int]:
    """Load an MMS ``vocab.json`` (``{char: id}``) for a non-English language."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {ch: int(i) for ch, i in raw.items()}


def preprocess_input(
    text: str, max_length: int, vocab: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Char-tokenise ``text`` and pad/trim to ``max_length`` (ref ``preprocess_input``):
    each kept char becomes a ``0`` (blank) followed by its id, a trailing ``0`` closes
    the sequence; returns batched ``(input_ids, attention_mask)``, empty when no char
    maps. Lowercasing is the whole Latin-script frontend. Unmapped chars are dropped
    silently (reference behaviour); the shell's speakability guard reports them once per
    character, seeing the same vocab through :meth:`MmsTtsAdapter._can_speak`."""
    input_id: list[int] = []
    for ch in text.lower():
        if ch not in vocab:
            continue
        input_id.append(0)
        input_id.append(int(vocab[ch]))
    if not input_id:
        return np.empty((1, 0), dtype=np.int64), np.empty((1, 0), dtype=np.int64)
    input_id.append(0)
    attention_mask = [1] * len(input_id)

    pad_len = max_length - len(input_id)
    if pad_len <= 0:
        input_id = input_id[:max_length]
        attention_mask = attention_mask[:max_length]
    else:
        input_id = input_id + [0] * pad_len
        attention_mask = attention_mask + [0] * pad_len

    return (
        np.array(input_id, dtype=np.int64)[None, ...],
        np.array(attention_mask, dtype=np.int64)[None, ...],
    )


def middle_process(
    log_duration, input_padding_mask, max_length: int, speaking_rate: float
) -> tuple[np.ndarray, np.ndarray, int]:
    """numpy port of the reference ``middle_process``: ``attn`` + output mask."""
    log_duration = np.asarray(log_duration, dtype=np.float32)
    input_padding_mask = np.asarray(input_padding_mask, dtype=np.float32)

    length_scale = 1.0 / speaking_rate
    duration = np.ceil(np.exp(log_duration) * input_padding_mask * length_scale)
    predicted_lengths = np.maximum(duration.sum(axis=(1, 2)), 1).astype(np.int64)  # (batch,)
    predicted_lengths_max_real = int(predicted_lengths.max())
    predicted_lengths_max = max_length * 2

    out_idx = np.arange(predicted_lengths_max)
    output_padding_mask = (out_idx[None, :] < predicted_lengths[:, None])[:, None, :].astype(np.float32)

    # attn_mask: (batch, 1, out_length, in_length)
    attn_mask = input_padding_mask[:, :, None, :] * output_padding_mask[:, :, :, None]
    batch_size, _, output_length, input_length = attn_mask.shape

    cum_duration = np.cumsum(duration, axis=-1).reshape(batch_size * input_length, 1)
    valid_indices = (np.arange(output_length)[None, :] < cum_duration).astype(np.float32)
    valid_indices = valid_indices.reshape(batch_size, input_length, output_length)
    # first difference along the input axis (== reference F.pad([..,1,0,..]) then subtract)
    padded = np.pad(valid_indices, ((0, 0), (1, 0), (0, 0)))[:, :-1, :]
    padded_indices = valid_indices - padded
    attn = np.transpose(padded_indices[:, None, :, :], (0, 1, 3, 2)) * attn_mask

    return attn.astype(np.float32), output_padding_mask.astype(np.float32), predicted_lengths_max_real


class MmsTtsAdapter(OnDeviceTtsAdapter):
    output_rate = SAMPLE_RATE
    _label = "MMS"
    _join_gap_s = _JOIN_GAP_S

    def __init__(
        self,
        *,
        encoder: OnDeviceModel,
        decoder: OnDeviceModel,
        vocab: dict[str, int],
        frontend: TextFrontend,
        max_length: int,
        speaking_rate: float,
    ):
        super().__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._vocab = vocab
        self._frontend = frontend
        # The built-in vocab IS mms-tts-eng: expand numbers into English words, since
        # it has no "7"/"8"/"9" and "7:45" would otherwise tokenize to "4 5".
        self._verbalize_en = vocab is _ENG_VOCAB
        # A supplied vocab.json carries no language label, so only the built-in one can be
        # named; a custom vocab claims nothing rather than claiming English wrongly.
        self.spoken_language = "en" if self._verbalize_en else None
        self._max_length = max_length
        self._speaking_rate = speaking_rate
        self._log = logger.bind(component="tts-mms")

    @classmethod
    def from_config(cls, cfg: MmsTtsConfig) -> MmsTtsAdapter:
        # Vocab and frontend first: a missing file or G2P dep must fail before any
        # model is loaded, so the registry can fall back to system TTS.
        vocab = load_mms_vocab(cfg.vocab_path) if cfg.vocab_path else _ENG_VOCAB
        frontend = make_text_frontend(cfg.text_frontend)
        model_kw = dict(
            core_mask=cfg.core_mask, target=cfg.target, device_id=cfg.device_id,
            providers=cfg.execution_providers, provider_options=cfg.provider_options,
        )
        with ExitStack() as models:
            encoder = models.enter_context(
                OnDeviceModel(cfg.encoder_path, **model_kw)  # type: ignore[arg-type]
            )
            decoder = models.enter_context(
                OnDeviceModel(cfg.decoder_path, **model_kw)  # type: ignore[arg-type]
            )
            adapter = cls(
                encoder=encoder,
                decoder=decoder,
                vocab=vocab,
                frontend=frontend,
                max_length=cfg.max_length,
                speaking_rate=cfg.speaking_rate,
            )
            models.pop_all()  # success: the adapter owns the models now
            return adapter

    def release(self) -> None:
        for model in (self._encoder, self._decoder):
            model.release()

    def _normalize(self, text: str) -> str:
        text = self._frontend.normalize(text)
        return verbalize_numbers_en(text) if self._verbalize_en else text

    def _can_speak(self, ch: str) -> bool:
        # Lowercased, exactly as preprocess_input tokenizes: the vocab IS the answer.
        return ch.lower() in self._vocab

    def _piece_budget(self) -> int:
        return max(1, (self._max_length - 1) // 2)  # 2 ids per char + 1 closer

    def _synthesize_piece(self, text: str) -> np.ndarray:
        input_ids, attention_mask = preprocess_input(text, self._max_length, self._vocab)
        if input_ids.shape[1] == 0:
            return np.zeros(0, dtype=np.float32)

        log_duration, input_padding_mask, prior_means, prior_log_variances = self._encoder.run(
            [("input_ids", input_ids), ("attention_mask", attention_mask)]
        )
        attn, output_padding_mask, real_len = middle_process(
            log_duration, input_padding_mask, self._max_length, self._speaking_rate
        )
        window = 2 * self._max_length
        if real_len > window:
            # Predicted audio outruns the decoder's FIXED output window and would be
            # clipped mid-word; the char budget bounds input, not duration, and duration
            # is ~linear in text length, so splitting into halves fixes it.
            stripped = text.strip()
            if len(stripped) > 1:
                mid = (len(stripped) + 1) // 2
                left = stripped.rfind(" ", 1, mid)
                right = stripped.find(" ", mid, len(stripped) - 1)
                cands = [c for c in (left, right) if c > 0]
                cut = (min(cands, key=lambda c: abs(c - mid)) + 1) if cands else mid
                left = self._synthesize_piece(stripped[:cut].strip())
                right = self._synthesize_piece(stripped[cut:].strip())
                parts = [p for p in (left, right) if p.size]
                if len(parts) == 2:
                    gap = np.zeros(int(_JOIN_GAP_S * SAMPLE_RATE), dtype=np.float32)
                    parts.insert(1, gap)
                return (
                    np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
                )
            self._log.warning("MMS: single unsplittable piece exceeds the decoder window")
            real_len = window
        waveform = self._decoder.run(
            [
                ("attn", attn),
                ("output_padding_mask", output_padding_mask),
                ("prior_means", np.asarray(prior_means, dtype=np.float32)),
                ("prior_log_variances", np.asarray(prior_log_variances, dtype=np.float32)),
            ]
        )[0]

        return np.asarray(waveform[0][: real_len * UPSAMPLE], dtype=np.float32)
