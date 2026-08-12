"""FbankCmvn streaming front-end: chunked == one-shot, and consumed frames are freed."""

from __future__ import annotations

import numpy as np
import pytest

knf = pytest.importorskip("kaldi_native_fbank")

from nanobot_channel_voice.vad import features as feat_mod  # noqa: E402 - after the skip gate


@pytest.fixture
def fbank(monkeypatch) -> feat_mod.FbankCmvn:
    # Identity CMVN: the test targets the fbank streaming bookkeeping, not the stats.
    monkeypatch.setattr(
        feat_mod, "_load_cmvn",
        lambda path: (
            np.zeros(feat_mod.NUM_MEL_BINS, np.float32),
            np.ones(feat_mod.NUM_MEL_BINS, np.float32),
        ),
    )
    return feat_mod.FbankCmvn("unused.ark")


def _tone(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(n) * 1000.0).astype(np.float32)


def test_chunked_accept_matches_one_shot(fbank):
    audio = _tone(16000, 7)  # 1 s
    chunks = [fbank.accept(audio[i : i + 320]) for i in range(0, len(audio), 320)]
    streamed = np.vstack([c for c in chunks if c.size])

    # The fixture's _load_cmvn patch is still installed, so both instances share stats.
    oneshot = feat_mod.FbankCmvn("unused.ark").accept(audio)
    assert streamed.shape == oneshot.shape
    np.testing.assert_allclose(streamed, oneshot, rtol=1e-5, atol=1e-5)


def test_consumed_frames_are_released(fbank):
    """knf retains every frame until popped: without the pop an idle-but-listening
    session grows without bound (~115 MB/h)."""
    for i in range(50):  # 5 simulated seconds, 100 ms chunks
        fbank.accept(_tone(1600, i))
    # Old frames must be gone from the retained window...
    with pytest.raises(IndexError):
        fbank._fbank.get_frame(0)
    # ...while the streaming cursor stays valid for what comes next.
    out = fbank.accept(_tone(1600, 99))
    assert out.shape[1] == feat_mod.NUM_MEL_BINS and out.shape[0] > 0
