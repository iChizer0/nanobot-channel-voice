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


def make_adapter(script: list[int], vocab: dict[int, str]) -> w.WhisperOnDeviceStt:
    return w.WhisperOnDeviceStt(
        encoder=None,  # _decode never touches the encoder
        decoder=_ScriptedDecoder(script),
        vocab=vocab,
        mel_filters=np.zeros((1, 1), dtype=np.float32),
        lang_token=50259,
        chunk_length=20,
    )


def test_decode_window_follows_the_export():
    vocab = {50257: "<|endoftext|>"}
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
        100: "hi",
        50257: "<|endoftext|>",
        50259: "<|en|>",
        50362: "<|nospeech|>",
        50364: "<|0.00|>",
    }
    adapter = make_adapter([50362, 100, 50364, 50259, 50257], vocab)
    assert adapter._decode(None) == "hi"


def test_pure_nospeech_decode_is_empty():
    vocab = {50362: "<|nospeech|>", 50257: "<|endoftext|>"}
    adapter = make_adapter([50362, 50257], vocab)
    assert adapter._decode(None) == ""


def test_repetition_trap_bails_early_and_drops_the_loop():
    """Regression: a greedy loop stuck on one token burned all 448 decoder
    passes and returned ~440 junk tokens as user speech."""
    vocab = {100: "hi", 200: " la", 50257: "<|endoftext|>"}
    # Pure loop: bails within the scripted budget, keeping only the first two repeats
    # (the bail fires the moment the 30-token window is periodic).
    adapter = make_adapter([100] * 60, vocab)
    assert adapter._decode(None) == "hihi"
    # Real text followed by a loop: keeps the text, sheds the loop past two repeats.
    adapter = make_adapter([100] + [200] * 60, vocab)
    assert adapter._decode(None) == "hilala"


def test_phrase_loop_bails_like_a_single_token_loop():
    """Regression: the trap only saw IDENTICAL tokens, so whisper's characteristic
    2-token phrase loop ran the whole step cap and published 1792 junk characters."""
    vocab = {100: "hi", 200: "la", 50257: "<|endoftext|>"}
    adapter = make_adapter([100, 200] * 300, vocab)
    assert adapter._decode(None) == "hilahila"  # bails at the window, keeps two repeats

    # Real text first: it survives, only the cycle past two repeats is shed.
    vocab = {1: "one", 2: "two", 3: "three", 9: "real", 50257: "<|endoftext|>"}
    adapter = make_adapter([9] + [1, 2, 3] * 100, vocab)
    assert adapter._decode(None) == "realonetwothreeonetwothree"
    # Whisper's most-cited loop is FIVE tokens ("Thank you very much."): caught too.
    vocab = {i: f"w{i}" for i in range(1, 6)} | {50257: "<|endoftext|>"}
    adapter = make_adapter([1, 2, 3, 4, 5] * 20, vocab)
    assert adapter._decode(None) == "w1w2w3w4w5" * 2


def test_everyday_repetition_is_not_a_loop():
    """Regression: a 12-token window bailed on "milk, eggs, bread" said three times and
    dropped the rest of the utterance; the window must outlast real repetition."""
    # Byte-level tokens: "Ġ" is the encoded space.
    vocab = {1: "Ġmilk", 2: ",", 3: "Ġeggs", 4: "Ġbread", 9: "Ġto", 50257: "<|endoftext|>"}
    script = [1, 2, 3, 2, 4, 2] * 3 + [9, 50257]  # 18 periodic tokens, then real text
    adapter = make_adapter(script, vocab)
    assert adapter._decode(None) == " milk, eggs, bread," * 3 + " to"
    vocab = {1: "Ġno", 2: ",", 3: "Ġstop", 50257: "<|endoftext|>"}
    adapter = make_adapter([1, 2] * 10 + [3, 50257], vocab)  # ten "no,"s is still speech
    assert adapter._decode(None) == " no," * 10 + " stop"


def test_decode_steps_scale_with_the_encoder_window():
    """Regression: a flat 448-step cap meant a stuck decode burned 448 decoder passes
    whatever the audio length; 20 s of mel cannot hold 448 tokens."""
    adapter = make_adapter([50257], {50257: "<|endoftext|>"})
    assert adapter._max_steps == 2000 // w._STEPS_PER_MEL_FRAME  # 20 s window
    short = w.WhisperOnDeviceStt(
        encoder=None,
        decoder=_ScriptedDecoder([50257]),
        vocab={50257: "<|endoftext|>"},
        mel_filters=np.zeros((1, 1), dtype=np.float32),
        lang_token=50259,
        chunk_length=1,
        max_frames=100,
    )
    assert short._max_steps == w._MIN_DECODE_STEPS  # floor: a sentence still fits


def test_read_vocab_refuses_an_empty_table(tmp_path):
    """A flat file in the wrong shape (token-first, tabs) parsed to {} and every utterance
    decoded to "": a mute STT that looked healthy."""
    import pytest

    from nanobot_channel_voice.stt.whisper_tokenizer import read_vocab

    bad = tmp_path / "vocab.txt"
    bad.write_text("<blk> 0\na 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no <id> <token> entries"):
        read_vocab(str(bad))
