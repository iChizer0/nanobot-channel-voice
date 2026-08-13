"""Silero VAD adapter: windowing/context bookkeeping, hysteresis, the sr-input
autodetect, and the registry's construction-failure fallbacks — all against a fake
on-device model (the real-ONNX regression lives in ``test_ondevice_real.py``)."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from nanobot_channel_voice.config import VadConfig  # noqa: E402
from nanobot_channel_voice.vad import flag_lag_ms, make_vad, resolve_preroll_ms  # noqa: E402
from nanobot_channel_voice.vad import silero as silero_mod  # noqa: E402
from nanobot_channel_voice.vad.energy import EnergyVad  # noqa: E402
from nanobot_channel_voice.vad.silero import SileroVad  # noqa: E402


class FakeModel:
    """Stands in for OnDeviceModel: scripted probabilities, recorded inputs."""

    def __init__(self, *, with_sr: bool = True, state_shape=(2, 1, 128)):
        self.with_sr = with_sr
        self.state_shape = state_shape
        self.probs: list[float] = []
        self.calls: list[list[tuple[str, np.ndarray]]] = []
        self.released = False

    # the adapter loads inside an ExitStack
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def input_specs(self):
        specs = [
            ("input", [1, "N"], "tensor(float)"),
            ("state", list(self.state_shape), "tensor(float)"),
        ]
        if self.with_sr:
            specs.append(("sr", [], "tensor(int64)"))
        return specs

    def input_shape(self, name):
        return (2, 1, 128) if name == "state" else None

    def run(self, inputs):
        self.calls.append([(n, np.asarray(a).copy()) for n, a in inputs])
        prob = self.probs.pop(0) if self.probs else 0.0
        return [np.array([[prob]], dtype=np.float32),
                np.ones(self.state_shape, dtype=np.float32)]

    def release(self):
        self.released = True


@pytest.fixture
def fake(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr(silero_mod, "OnDeviceModel", lambda path, **kw: model)
    return model


def _adapter(**kw):
    kw.setdefault("model_path", "fake.onnx")
    kw.setdefault("sample_rate", 16000)
    return SileroVad(**kw)


def _frame(samples: int, value: int = 1000) -> bytes:
    return np.full(samples, value, dtype="<i2").tobytes()


# ---- windowing and context --------------------------------------------------

def test_decisions_land_on_512_sample_windows_and_hold_between(fake):
    vad = _adapter()
    fake.calls.clear()  # drop the construction probe
    fake.probs = [0.9, 0.9]

    assert vad.is_speech(_frame(320)) is False   # 320 < 512: nothing ran, initial state
    assert len(fake.calls) == 0
    assert vad.is_speech(_frame(320)) is True    # 640 buffered: one window ran
    assert len(fake.calls) == 1
    assert vad.is_speech(_frame(320)) is True    # 448 left: held decision, no new run
    assert len(fake.calls) == 1
    assert vad.is_speech(_frame(320)) is True    # 768: second window
    assert len(fake.calls) == 2


def test_model_input_is_context_plus_window_with_carried_context(fake):
    vad = _adapter()
    fake.calls.clear()
    fake.probs = [0.9, 0.9]

    stream = np.arange(1024, dtype="<i2")  # distinct values: positions are provable
    vad.is_speech(stream.tobytes())

    first = dict(fake.calls[0])["input"]
    assert first.shape == (1, 576)
    assert np.all(first[0, :64] == 0.0)  # fresh context is silence
    expected = stream[:512].astype(np.float32) / 32768.0
    assert np.allclose(first[0, 64:], expected)

    second = dict(fake.calls[1])["input"]
    # context = the previous window's tail, NOT zeros and NOT the stream tail
    assert np.allclose(second[0, :64], stream[448:512].astype(np.float32) / 32768.0)
    assert np.allclose(second[0, 64:], stream[512:1024].astype(np.float32) / 32768.0)


def test_8k_uses_256_sample_windows_and_32_context(fake):
    vad = _adapter(sample_rate=8000)
    fake.calls.clear()
    fake.probs = [0.9]
    vad.is_speech(_frame(256))
    assert dict(fake.calls[0])["input"].shape == (1, 288)


def test_unsupported_rate_raises_at_construction(fake):
    with pytest.raises(ValueError, match="8 or 16 kHz"):
        _adapter(sample_rate=48000)


# ---- sr input autodetect ----------------------------------------------------

def test_sr_is_passed_when_the_session_declares_it(fake):
    _adapter()
    names = [n for n, _ in fake.calls[0]]
    assert names == ["input", "state", "sr"]
    assert dict(fake.calls[0])["sr"] == 16000


def test_stripped_export_runs_without_sr(monkeypatch):
    model = FakeModel(with_sr=False)
    monkeypatch.setattr(silero_mod, "OnDeviceModel", lambda path, **kw: model)
    _adapter()
    assert [n for n, _ in model.calls[0]] == ["input", "state"]


# ---- hysteresis -------------------------------------------------------------

def test_hysteresis_holds_between_the_thresholds(fake):
    vad = _adapter(threshold=0.5, neg_threshold=0.35)
    fake.probs = [0.9, 0.45, 0.34, 0.45]
    assert vad.is_speech(_frame(512)) is True     # >= threshold: enters
    assert vad.is_speech(_frame(512)) is True     # in the gap: holds speech
    assert vad.is_speech(_frame(512)) is False    # < neg: leaves
    assert vad.is_speech(_frame(512)) is False    # in the gap: holds silence


def test_default_neg_threshold_is_threshold_minus_015(fake):
    vad = _adapter(threshold=0.5)
    fake.probs = [0.9, 0.4, 0.34]
    assert vad.is_speech(_frame(512)) is True
    assert vad.is_speech(_frame(512)) is True     # 0.4 >= 0.35: still speech
    assert vad.is_speech(_frame(512)) is False    # 0.34 < 0.35: released


# ---- gating, reset, state carry ---------------------------------------------

def test_min_volume_gates_the_held_decision_too(fake):
    vad = _adapter(min_volume=0.5)  # unreachable by the quiet test frames
    fake.probs = [0.99, 0.99]
    assert vad.is_speech(_frame(512)) is False           # model said speech; gate wins
    assert vad.is_speech(_frame(256)) is False           # held state is gated as well
    assert vad._last_speech is True                      # ...but the model state is intact


def test_reset_clears_stream_state(fake):
    vad = _adapter()
    fake.probs = [0.9]
    vad.is_speech(_frame(512 + 100))
    assert np.any(vad._lstm_state)                       # fake returns ones: state carried
    vad.reset()
    assert not np.any(vad._lstm_state)
    assert not np.any(vad._context)
    assert vad._pending.size == 0
    assert vad.is_speech(_frame(100)) is False           # decision cleared with it


def test_model_failure_returns_nonspeech_not_an_exception(fake):
    vad = _adapter()
    fake.run = None  # type: ignore[assignment] - every later call now raises
    assert vad.is_speech(_frame(512)) is False


# ---- registry ---------------------------------------------------------------

def test_make_vad_builds_silero(fake):
    cfg = VadConfig.model_validate({"engine": "silero", "silero": {"modelPath": "fake.onnx"}})
    assert isinstance(make_vad(cfg, 16000, 20), SileroVad)


def test_make_vad_missing_model_path_degrades_to_energy():
    cfg = VadConfig.model_validate({"engine": "silero"})
    assert isinstance(make_vad(cfg, 16000, 20), EnergyVad)


def test_incompatible_export_degrades_to_energy_and_releases(monkeypatch):
    model = FakeModel(state_shape=(2, 1, 64))  # wrong state out: _validate must raise
    monkeypatch.setattr(silero_mod, "OnDeviceModel", lambda path, **kw: model)
    cfg = VadConfig.model_validate({"engine": "silero", "silero": {"modelPath": "fake.onnx"}})
    assert isinstance(make_vad(cfg, 16000, 20), EnergyVad)
    assert model.released  # the ExitStack gave the session back


# ---- endpointing geometry ---------------------------------------------------

def test_preroll_floor_covers_the_window_lag():
    cfg = VadConfig.model_validate({"engine": "silero", "prerollMs": 0})
    assert flag_lag_ms(cfg, 20) == 144  # 2 windows + rise margin
    assert resolve_preroll_ms(cfg, 20) == 144
