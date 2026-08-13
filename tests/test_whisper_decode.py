"""Whisper greedy decode loop over a scripted decoder (no ONNX needed)."""

from __future__ import annotations

import numpy as np

from nanobot_channel_voice.stt import whisper as w

_VOCAB_SIZE = 51865


class _ScriptedDecoder:
    """Emits one scripted token per step via an argmax-friendly logits row."""

    def __init__(self, script: list[int]):
        self._script = list(script)

    def run(self, inputs):
        out = np.zeros((1, w.MAX_TOKENS, _VOCAB_SIZE), dtype=np.float32)
        out[0, -1, self._script.pop(0)] = 1.0
        return [out]


def make_adapter(script: list[int], vocab: dict[str, str]) -> w.WhisperOnDeviceStt:
    return w.WhisperOnDeviceStt(
        encoder=None,  # _decode never touches the encoder
        decoder=_ScriptedDecoder(script),
        vocab=vocab,
        mel_filters=np.zeros((1, 1), dtype=np.float32),
        lang_token=50259,
        chunk_length=20,
    )


def test_decode_window_follows_the_export():
    vocab = {"50257": "<|endoftext|>"}
    assert make_adapter([50257], vocab).max_decode_ms == 20_000  # chunkLength fallback
    adapter = w.WhisperOnDeviceStt(
        encoder=None,
        decoder=_ScriptedDecoder([50257]),
        vocab=vocab,
        mel_filters=np.zeros((1, 1), dtype=np.float32),
        lang_token=50259,
        chunk_length=20,
        max_frames=3000,
    )
    assert adapter.max_decode_ms == 30_000  # the export's own window wins


def test_specials_and_timestamps_never_reach_transcript():
    """Regression: non-timestamp specials (50258-50363) passed the old
    EOT/timestamp-only filter, so a greedy <|nospeech|> pick on noise
    published the literal token string to the agent as user speech."""
    vocab = {
        "100": "hi",
        "50257": "<|endoftext|>",
        "50259": "<|en|>",
        "50362": "<|nospeech|>",
        "50364": "<|0.00|>",
    }
    adapter = make_adapter([50362, 100, 50364, 50259, 50257], vocab)
    assert adapter._decode(None) == "hi"


def test_pure_nospeech_decode_is_empty():
    vocab = {"50362": "<|nospeech|>", "50257": "<|endoftext|>"}
    adapter = make_adapter([50362, 50257], vocab)
    assert adapter._decode(None) == ""


def test_repetition_trap_bails_early_and_drops_the_loop():
    """Regression: a greedy loop stuck on one token burned all 448 decoder
    passes and returned ~440 junk tokens as user speech."""
    vocab = {"100": "hi", "200": " la", "50257": "<|endoftext|>"}
    # Pure loop: bails within the scripted budget and returns nothing.
    adapter = make_adapter([100] * 20, vocab)
    assert adapter._decode(None) == ""
    # Real text followed by a loop: keeps the text, sheds the whole loop
    # (the bail fires the moment the last 8 emitted tokens are identical).
    adapter = make_adapter([100] + [200] * 20, vocab)
    assert adapter._decode(None) == "hi"
