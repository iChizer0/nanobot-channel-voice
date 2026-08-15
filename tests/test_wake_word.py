"""Wake-word unit surface: the transcript-tier phrase matcher/stripper, the
openWakeWord adapter's pipeline bookkeeping against fake on-device models (the
same pattern as test_silero_vad.py), the registry's degrade paths, and the
config validators. Conversation-level gating lives in test_wake_gating.py."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from pydantic import ValidationError  # noqa: E402

from nanobot_channel_voice.config import VoiceConfig, WakeConfig  # noqa: E402
from nanobot_channel_voice.wake import make_wake_detector  # noqa: E402
from nanobot_channel_voice.wake import openwakeword as oww_mod  # noqa: E402
from nanobot_channel_voice.wake.openwakeword import OpenWakeWord  # noqa: E402
from nanobot_channel_voice.wake.phrase import WakePhrase  # noqa: E402

# ---- transcript tier: WakePhrase --------------------------------------------


def test_strip_leading_phrase_and_punctuation():
    wp = WakePhrase(["hey nanobot"])
    assert wp.strip("Hey, Nanobot! what's the weather") == (True, "what's the weather")


def test_phrase_must_lead_not_merely_occur():
    wp = WakePhrase(["hey nanobot"])
    matched, text = wp.strip("I said hey nanobot yesterday")
    assert matched is False and text == "I said hey nanobot yesterday"


def test_hesitation_fillers_may_precede():
    wp = WakePhrase(["hey nanobot"])
    assert wp.strip("um, hey nanobot turn on the light")[0] is True


def test_fused_cjk_fillers_may_precede():
    # "嗯" and "那个" arrive as ONE fused token from unspaced STT output; the
    # prefix check must segment it instead of demanding exact membership.
    wp = WakePhrase(["小助手"])
    assert wp.strip("嗯那个小助手开灯")[0] is True


def test_cjk_fused_run_matches_and_strips():
    wp = WakePhrase(["小助手"])
    assert wp.strip("小助手今天天气怎么样") == (True, "今天天气怎么样")


def test_no_partial_word_match_in_spaced_scripts():
    wp = WakePhrase(["nanobot"])
    assert wp.strip("nanobots are cool") == (False, "nanobots are cool")
    wp2 = WakePhrase(["hey nanobot"])
    assert wp2.strip("hey nanobotics lab")[0] is False


def test_spaced_phrase_may_run_into_cjk():
    # A following ideograph starts a new word by definition (zh STT emits
    # "hey nanobot今天天气" with no separator).
    wp = WakePhrase(["hey nanobot"])
    assert wp.strip("hey nanobot今天天气") == (True, "今天天气")


def test_separator_strip_keeps_sign_characters():
    wp = WakePhrase(["hey nanobot"])
    assert wp.strip("hey nanobot -3 degrees is cold") == (True, "-3 degrees is cold")


def test_casefold_variants_match():
    wp = WakePhrase(["straße computer"])
    assert wp.leads("STRASSE COMPUTER an") is True


def test_bare_phrase_strips_to_empty():
    wp = WakePhrase(["hey nanobot"])
    assert wp.strip("hey nanobot.") == (True, "")


def test_earliest_widest_match_wins():
    wp = WakePhrase(["nanobot", "hey nanobot"])
    assert wp.strip("hey nanobot hello") == (True, "hello")


def test_leads_mirrors_strip():
    wp = WakePhrase(["ok computer"])
    assert wp.leads("OK computer, play something") is True
    assert wp.leads("that's ok computer stuff") is False


# ---- acoustic tier: OpenWakeWord against fake models ------------------------

_CHUNK = b"\x01\x00" * 1280  # one 80 ms step


class FakeMel:
    """Melspectrogram stand-in that PINS the upstream input contract: uniform
    [1, 1760] float32 windows of raw int16-VALUED samples (a /32768
    normalization would show up as sub-1.0 maxima), and the upstream frame
    count ceil(L/160) - 3."""

    def __init__(self):
        self.calls = 0
        self.released = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False

    def input_specs(self):
        return [("input", [1, "N"], "tensor(float)")]

    def input_shape(self, name):
        return None

    def run(self, inputs):
        self.calls += 1
        (_, arr) = inputs[0]
        assert arr.dtype == np.float32
        assert arr.size == 1760  # context 480 + chunk 1280, EVERY call (RKNN-portable)
        if np.any(arr):
            assert np.abs(arr).max() >= 1.0  # raw int16-valued, not normalized
        frames = max(0, -(-arr.size // 160) - 3)
        return [np.zeros((1, 1, frames, 32), dtype=np.float32)]

    def release(self):
        self.released = True


class FakeEmb(FakeMel):
    def input_specs(self):
        return [("input_1", [1, 76, 32, 1], "tensor(float)")]

    def run(self, inputs):
        self.calls += 1
        (_, arr) = inputs[0]
        assert arr.shape == (1, 76, 32, 1)
        # FakeMel emits zeros, so every frame must arrive as the x/10 + 2
        # transform of zero: dropping or misplacing the transform fails here.
        assert np.allclose(arr, 2.0)
        return [np.zeros((1, 1, 1, 96), dtype=np.float32)]


class FakeHead(FakeMel):
    def __init__(self, window: int = 16):
        super().__init__()
        self.window = window
        self.probs: list[float] = []

    def input_specs(self):
        return [("input_1", [1, self.window, 96], "tensor(float)")]

    def input_shape(self, name):
        return (1, self.window, 96)

    def run(self, inputs):
        self.calls += 1
        self.shapes = [np.asarray(a).shape for _, a in inputs]
        prob = self.probs.pop(0) if self.probs else 0.0
        return [np.array([[prob]], dtype=np.float32)]


@pytest.fixture
def fakes(monkeypatch):
    made = {"mel.onnx": FakeMel(), "emb.onnx": FakeEmb(), "head.onnx": FakeHead()}
    monkeypatch.setattr(oww_mod, "OnDeviceModel", lambda path, **kw: made[path])
    return made


def _detector(**kw):
    kw.setdefault("mel_path", "mel.onnx")
    kw.setdefault("embedding_path", "emb.onnx")
    kw.setdefault("model_path", "head.onnx")
    kw.setdefault("sample_rate", 16000)
    kw.setdefault("refractory_s", 0.0)
    return OpenWakeWord(**kw)


def _warm(det, chunks: int) -> bool:
    hit = False
    for _ in range(chunks):
        hit = det.push(_CHUNK) or hit
    return hit


def test_scores_start_once_the_mel_window_fills(fakes):
    det = _detector()
    head = fakes["head.onnx"]
    head.calls = 0
    # 8 mel frames per (uniform 1760-sample) mel call: the 76-frame embedding
    # window fills on chunk 10 after a reset.
    assert _warm(det, 9) is False
    assert head.calls == 0 and det.last_score is None
    det.push(_CHUNK)
    assert head.calls == 1 and det.last_score == 0.0
    assert head.shapes[0] == (1, 16, 96)


def test_sub_chunk_frames_buffer_without_inference(fakes):
    det = _detector()
    mel = fakes["mel.onnx"]
    mel.calls = 0
    det.push(b"\x01\x00" * 320)  # 20 ms: under one chunk
    assert mel.calls == 0
    det.push(b"\x01\x00" * 960)  # 1280 buffered: one step runs
    assert mel.calls == 1


def test_hit_on_threshold_with_rearm_hysteresis(fakes):
    det = _detector(threshold=0.5)
    head = fakes["head.onnx"]
    _warm(det, 9)
    head.probs = [0.9, 0.9, 0.2, 0.9]
    assert det.push(_CHUNK) is True    # crossing: hit
    assert det.push(_CHUNK) is False   # still above: no retrigger
    assert det.push(_CHUNK) is False   # dip: re-arms
    assert det.push(_CHUNK) is True    # second crossing
    assert det.last_score == pytest.approx(0.9, abs=1e-6)


def test_refractory_blocks_a_fast_second_hit(fakes):
    det = _detector(threshold=0.5, refractory_s=1000.0)
    head = fakes["head.onnx"]
    _warm(det, 9)
    head.probs = [0.9, 0.2, 0.9]
    assert det.push(_CHUNK) is True
    det.push(_CHUNK)
    assert det.push(_CHUNK) is False  # re-armed but inside the refractory window


def test_model_failure_returns_no_hit_not_an_exception(fakes):
    det = _detector()
    fakes["head.onnx"].run = None  # type: ignore[assignment]
    assert _warm(det, 12) is False


def test_reset_clears_stream_state(fakes):
    det = _detector()
    _warm(det, 10)
    assert det.last_score is not None
    det.reset()
    assert det.last_score is None
    assert _warm(det, 9) is False  # warmup starts over


def test_release_releases_all_three_models(fakes):
    det = _detector()
    det.release()
    assert all(m.released for m in fakes.values())


def test_non_16k_rate_raises_at_construction(fakes):
    with pytest.raises(ValueError, match="16 kHz"):
        _detector(sample_rate=8000)


def test_reset_restores_the_primed_embedding_window(fakes):
    det = _detector()
    prime = det._embs_prime
    assert prime.shape == (16, 96)
    _warm(det, 12)
    det.reset()
    assert np.array_equal(det._embs, prime)  # never the all-zero OOD window


def test_multi_output_head_is_rejected_at_construction(monkeypatch):
    class MultiHead(FakeHead):
        def run(self, inputs):
            self.calls += 1
            return [np.zeros((1, 3), dtype=np.float32)]

    made = {"mel.onnx": FakeMel(), "emb.onnx": FakeEmb(), "head.onnx": MultiHead()}
    monkeypatch.setattr(oww_mod, "OnDeviceModel", lambda path, **kw: made[path])
    with pytest.raises(RuntimeError, match="single-score"):
        _detector()
    assert all(m.released for m in made.values())  # ExitStack unwound


def test_non_sigmoid_head_degrades_registry_and_releases(monkeypatch):
    class LogitHead(FakeHead):
        def run(self, inputs):
            self.calls += 1
            return [np.array([[5.0]], dtype=np.float32)]

    made = {"mel.onnx": FakeMel(), "emb.onnx": FakeEmb(), "head.onnx": LogitHead()}
    monkeypatch.setattr(oww_mod, "OnDeviceModel", lambda path, **kw: made[path])
    assert make_wake_detector(_wake_cfg(), 16000, 20) is None
    assert all(m.released for m in made.values())


# ---- registry ---------------------------------------------------------------

def _wake_cfg(**over):
    base = {
        "mode": "gate",
        "phrases": ["hey nanobot"],
        "engine": "openwakeword",
        "openwakeword": {
            "melPath": "mel.onnx", "embeddingPath": "emb.onnx", "modelPath": "head.onnx",
        },
    }
    base.update(over)
    return WakeConfig.model_validate(base)


def test_make_wake_detector_builds(fakes):
    assert isinstance(make_wake_detector(_wake_cfg(), 16000, 20), OpenWakeWord)


def test_mode_off_and_text_engine_build_nothing(fakes):
    assert make_wake_detector(_wake_cfg(mode="off"), 16000, 20) is None
    assert make_wake_detector(_wake_cfg(engine="text"), 16000, 20) is None


def test_missing_model_paths_degrade_to_text_tier():
    cfg = _wake_cfg(openwakeword={})
    assert make_wake_detector(cfg, 16000, 20) is None


def test_incompatible_rate_degrades_to_text_tier(fakes):
    assert make_wake_detector(_wake_cfg(), 8000, 20) is None


# ---- config -----------------------------------------------------------------

def test_gating_requires_phrases_whatever_the_engine():
    # The transcript tier is the always-available fallback: an acoustic-only
    # config whose engine degrades would otherwise be PERMANENTLY deaf.
    with pytest.raises(ValidationError, match="wake.phrases"):
        WakeConfig.model_validate({"mode": "gate"})
    with pytest.raises(ValidationError, match="wake.phrases"):
        _wake_cfg(phrases=[])


def test_voice_config_carries_wake_block():
    cfg = VoiceConfig.model_validate(
        {"wake": {"mode": "strict", "phrases": ["hey nanobot"], "windowS": 10}}
    )
    assert cfg.wake.mode == "strict" and cfg.wake.window_s == 10.0
