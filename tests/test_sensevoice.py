"""SenseVoice adapter contracts (the original FunASR export): sidecar loading, the
dynamic .onnx path, the static .rknn window padding/masking/decode-slice/truncation,
and the registry's construction-failure fallbacks — all against a fake on-device model
(a tiny vocab keeps the fake's logits small; the adapter never hardcodes 25055)."""

from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("kaldi_native_fbank")

from nanobot_channel_voice.config import SenseVoiceSttConfig, SttConfig  # noqa: E402
from nanobot_channel_voice.stt import make_stt  # noqa: E402
from nanobot_channel_voice.stt import sensevoice as sv_mod  # noqa: E402
from nanobot_channel_voice.stt.sensevoice import SenseVoiceOnDeviceStt  # noqa: E402

_WINDOW = 40  # fake export window (frames); the real one is 500
_VOCAB = 100


class FakeModel:
    """Stands in for OnDeviceModel on the .rknn path: records inputs, returns
    scripted logits [1, window+4, vocab]."""

    def __init__(self):
        self.calls: list[list[tuple[str, np.ndarray]]] = []
        self.released = False
        self.script: list[int] | None = None  # per-frame argmax ids to force

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def run(self, inputs):
        self.calls.append([(n, np.asarray(a).copy()) for n, a in inputs])
        logits = np.full((1, _WINDOW + 4, _VOCAB), -10.0, dtype=np.float32)
        if self.script is not None:
            for t, tok in enumerate(self.script):
                logits[0, t, tok] = 10.0
        return [logits]

    def release(self):
        self.released = True


@pytest.fixture
def fake(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(sv_mod, "OnDeviceModel", lambda path, **kw: model)
    return model


@pytest.fixture
def sidecar(tmp_path):
    frontend = {
        "feats_len": _WINDOW,
        "neg_mean": ",".join(["0.1"] * 560),
        "inv_stddev": ",".join(["1.0"] * 560),
        "lfr_window_size": 7,
        "lfr_window_shift": 6,
        "languages": {"auto": 0, "zh": 3, "en": 4},
        "textnorm": {"withitn": 14, "woitn": 15},
    }
    (tmp_path / "frontend.json").write_text(json.dumps(frontend))
    (tmp_path / "tokens.txt").write_text("<blank> 0\nx 5\ny 7\n<|en|> 9\n")
    return tmp_path


def _cfg(sidecar, **kw):
    vals = {
        "modelPath": "model.rknn",
        "tokensPath": str(sidecar / "tokens.txt"),
        "frontendPath": str(sidecar / "frontend.json"),
    }
    vals.update(kw)
    return SenseVoiceSttConfig.model_validate(vals)


def _speech_seconds(s: float) -> bytes:
    return (np.full(int(16000 * s), 300, dtype="<i2")).tobytes()


def _lfr_frames(pcm: bytes) -> int:
    """The frame count the real frontend produces for this much 16 kHz audio."""
    n_fbank = 1 + (len(pcm) // 2 - 400) // 160  # snip_edges=True
    return int(np.ceil(n_fbank / 6))


# ---- construction / sidecar ----------------------------------------------------


def test_rknn_builds_from_sidecar_and_probes(fake, sidecar):
    adapter = SenseVoiceOnDeviceStt.from_config(_cfg(sidecar))
    assert adapter._feats_len == _WINDOW
    # transcribe_chunked must cut to the export window: frames x lfr shift 6 x 10 ms.
    assert adapter.max_decode_ms == _WINDOW * 60
    assert len(fake.calls) == 1  # the construction probe ran once
    inputs = dict(fake.calls[0])
    assert inputs["speech"].shape == (1, _WINDOW, 560)
    assert inputs["mask"].shape == (1, 1, 1, _WINDOW + 4)
    assert inputs["language"].dtype == np.int32
    adapter.release()
    assert fake.released


def test_language_and_itn_ids_come_from_the_sidecar(fake, sidecar):
    adapter = SenseVoiceOnDeviceStt.from_config(_cfg(sidecar, language="zh", use_itn=False))
    assert adapter._language_id == 3
    assert adapter._text_norm_id == 15


def test_unknown_language_raises_before_the_model_loads(fake, sidecar):
    with pytest.raises(RuntimeError, match="not in the model"):
        SenseVoiceOnDeviceStt.from_config(_cfg(sidecar, language="yue"))
    assert not fake.calls and not fake.released


def test_missing_frontend_path_degrades_to_none(fake):
    cfg = SttConfig.model_validate({
        "provider": "sensevoice",
        "sensevoice": {"modelPath": "model.rknn", "tokensPath": "t.txt"},
    })
    assert make_stt(cfg) is None
    assert not fake.calls and not fake.released  # rejected before the NPU load


def test_malformed_sidecar_fails_before_the_model_loads(fake, sidecar):
    side = json.loads((sidecar / "frontend.json").read_text())
    del side["languages"]
    (sidecar / "frontend.json").write_text(json.dumps(side))
    with pytest.raises(RuntimeError, match="frontend sidecar"):
        SenseVoiceOnDeviceStt.from_config(_cfg(sidecar))
    assert not fake.calls and not fake.released


def test_rknn_sidecar_without_feats_len_raises(fake, sidecar):
    side = json.loads((sidecar / "frontend.json").read_text())
    del side["feats_len"]
    (sidecar / "frontend.json").write_text(json.dumps(side))
    with pytest.raises(RuntimeError, match="feats_len"):
        SenseVoiceOnDeviceStt.from_config(_cfg(sidecar))
    assert not fake.calls and not fake.released


def test_bad_cmvn_dim_raises(fake, sidecar):
    side = json.loads((sidecar / "frontend.json").read_text())
    side["neg_mean"] = ",".join(["0.1"] * 100)
    (sidecar / "frontend.json").write_text(json.dumps(side))
    with pytest.raises(RuntimeError, match="CMVN stats"):
        SenseVoiceOnDeviceStt.from_config(_cfg(sidecar))


# ---- the static window contract -------------------------------------------------


def test_mask_covers_queries_plus_valid_frames_and_logits_tail_is_sliced(fake, sidecar):
    adapter = SenseVoiceOnDeviceStt.from_config(_cfg(sidecar))
    fake.calls.clear()
    pcm = _speech_seconds(1.0)  # ~17 LFR frames: inside the 40-frame fake window
    n_valid = min(_lfr_frames(pcm), _WINDOW)
    # Frames past 4+n_valid would decode as 'y' if the tail were NOT sliced away.
    fake.script = [0] * (4 + n_valid) + [7] * (_WINDOW - n_valid)
    fake.script[4] = 5  # one 'x' at the first speech frame
    assert adapter._transcribe_sync(pcm, 16000) == "x"
    inputs = dict(fake.calls[0])
    mask = inputs["mask"].reshape(-1)
    assert mask[: 4 + n_valid].all() and not mask[4 + n_valid:].any()
    speech = inputs["speech"]
    assert speech[0, n_valid:].sum() == 0.0  # zero-padded tail


def test_over_window_audio_truncates_not_crashes(fake, sidecar):
    adapter = SenseVoiceOnDeviceStt.from_config(_cfg(sidecar))
    fake.calls.clear()
    pcm = _speech_seconds(5.0)  # ~830 frames >> the 40-frame fake window
    fake.script = [0] * (_WINDOW + 4)
    fake.script[4] = 5
    assert adapter._transcribe_sync(pcm, 16000) == "x"
    assert dict(fake.calls[0])["speech"].shape == (1, _WINDOW, 560)


# ---- the dynamic .onnx contract -------------------------------------------------


def test_onnx_uses_the_dynamic_original_contract(fake, sidecar):
    """.onnx (official FunASR export): dynamic T, lengths by value, the sidecar's
    feats_len ignored (the graph is dynamic), same construction probe."""
    adapter = SenseVoiceOnDeviceStt.from_config(
        _cfg(sidecar, modelPath="model.onnx", language="en")
    )
    assert adapter._feats_len is None
    assert adapter.max_decode_ms == 30_000  # the dynamic policy bound, not a window
    assert adapter._language_id == 4
    assert len(fake.calls) == 1  # probed on this path too
    fake.calls.clear()
    pcm = _speech_seconds(0.5)
    n_valid = _lfr_frames(pcm)

    def dynamic_run(inputs):
        fake.calls.append([(n, np.asarray(a).copy()) for n, a in inputs])
        t = int(dict(fake.calls[-1])["speech_lengths"][0])
        logits = np.full((1, 4 + t, _VOCAB), -10.0, dtype=np.float32)
        logits[0, 4, 5] = 10.0  # one 'x' at the first speech frame
        return [logits]

    fake.run = dynamic_run  # type: ignore[method-assign]
    assert adapter._transcribe_sync(pcm, 16000) == "x"
    names = [n for n, _ in fake.calls[0]]
    assert names == ["speech", "speech_lengths", "language", "textnorm"]
    assert dict(fake.calls[0])["speech"].shape == (1, n_valid, 560)
