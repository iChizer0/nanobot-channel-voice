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
from nanobot_channel_voice.wake.phrase import FuzzyWake, WakePhrase  # noqa: E402

# ---- transcript tier: WakePhrase --------------------------------------------


def test_strip_leading_phrase_and_punctuation():
    wp = WakePhrase(["hey nanobot"])
    assert wp.strip("Hey, Nanobot! what's the weather") == ("hey nanobot", "what's the weather")


def test_phrase_must_lead_not_merely_occur():
    wp = WakePhrase(["hey nanobot"])
    matched, text = wp.strip("I said hey nanobot yesterday")
    assert matched is None and text == "I said hey nanobot yesterday"


def test_hesitation_fillers_may_precede():
    wp = WakePhrase(["hey nanobot"])
    assert wp.strip("um, hey nanobot turn on the light")[0] == "hey nanobot"


def test_fused_cjk_fillers_may_precede():
    # "嗯" and "那个" arrive as ONE fused token from unspaced STT output; the
    # prefix check must segment it instead of demanding exact membership.
    wp = WakePhrase(["小助手"])
    assert wp.strip("嗯那个小助手开灯")[0] == "小助手"


def test_cjk_fused_run_matches_and_strips():
    wp = WakePhrase(["小助手"])
    assert wp.strip("小助手今天天气怎么样") == ("小助手", "今天天气怎么样")


def test_no_partial_word_match_in_spaced_scripts():
    wp = WakePhrase(["nanobot"])
    assert wp.strip("nanobots are cool") == (None, "nanobots are cool")
    wp2 = WakePhrase(["hey nanobot"])
    assert wp2.strip("hey nanobotics lab")[0] is None


def test_spaced_phrase_may_run_into_cjk():
    # A following ideograph starts a new word by definition (zh STT emits
    # "hey nanobot今天天气" with no separator).
    wp = WakePhrase(["hey nanobot"])
    assert wp.strip("hey nanobot今天天气") == ("hey nanobot", "今天天气")


def test_separator_strip_keeps_sign_characters():
    wp = WakePhrase(["hey nanobot"])
    assert wp.strip("hey nanobot -3 degrees is cold") == ("hey nanobot", "-3 degrees is cold")


def test_casefold_variants_match():
    wp = WakePhrase(["straße computer"])
    assert wp.leads("STRASSE COMPUTER an") is True


def test_bare_phrase_strips_to_empty():
    wp = WakePhrase(["hey nanobot"])
    assert wp.strip("hey nanobot.") == ("hey nanobot", "")


def test_earliest_widest_match_wins():
    wp = WakePhrase(["nanobot", "hey nanobot"])
    assert wp.strip("hey nanobot hello") == ("hey nanobot", "hello")


def test_leads_mirrors_strip():
    wp = WakePhrase(["ok computer"])
    assert wp.leads("OK computer, play something") is True
    assert wp.leads("that's ok computer stuff") is False


def test_present_finds_a_mention_anywhere():
    wp = WakePhrase(["hey nanobot"])
    assert wp.present("you can always say hey, nanobot! anytime") is True


def test_present_needs_the_ordered_phrase_not_its_units():
    # The wake echo veto's whole point: a reply saying the phrase's words APART
    # ("they" absorbs "hey" as a substring) is not a mention.
    wp = WakePhrase(["hey nanobot"])
    assert wp.present("they asked what nanobot can do") is False


def test_present_respects_word_boundaries_both_sides():
    wp = WakePhrase(["bot"])
    assert wp.present("the robot is here") is False
    assert wp.present("bots everywhere") is False
    assert wp.present("my bot, yes") is True


def test_present_matches_inside_fused_cjk_runs():
    wp = WakePhrase(["小助手"])
    assert wp.present("大家都叫我小助手呢") is True


def test_strip_extra_lead_admits_caller_approved_tokens():
    # The leak-tolerant lead the backend passes: known-echo tokens may precede
    # the phrase; anything else still demotes it to content.
    wp = WakePhrase(["hey nanobot"])
    leak = {"the", "weather", "is", "sunny"}
    assert wp.strip(
        "the weather is sunny hey nanobot stop", extra_lead=lambda t: t in leak
    ) == ("hey nanobot", "stop")
    assert wp.strip("the weather is sunny hey nanobot stop")[0] is None
    assert wp.strip(
        "the fresh guy said hey nanobot stop", extra_lead=lambda t: t in leak
    )[0] is None


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
        # FakeMel emits zeros, so every REAL frame must arrive as the x/10 + 2
        # transform of zero (2.0), newest last; rows still holding the reset
        # seed are exactly 1.0. Anything else = transform or seed lost.
        assert np.all((arr == 1.0) | np.isclose(arr, 2.0))
        assert np.allclose(arr[0, -8:, :, 0], 2.0)
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


def _push_chunks(det, chunks: int) -> bool:
    hit = False
    for _ in range(chunks):
        hit = det.push(_CHUNK) or hit
    return hit


def test_scores_start_on_the_first_chunk(fakes):
    det = _detector()
    head = fakes["head.onnx"]
    head.calls = 0
    # Pre-seeded mel window (upstream parity): no post-reset deaf window.
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
    head.probs = [0.9, 0.9, 0.2, 0.9]
    assert det.push(_CHUNK) is True    # crossing: hit
    assert det.push(_CHUNK) is False   # still above: no retrigger
    assert det.push(_CHUNK) is False   # dip: re-arms
    assert det.push(_CHUNK) is True    # second crossing
    assert det.last_score == pytest.approx(0.9, abs=1e-6)


def test_refractory_blocks_a_fast_second_hit(fakes):
    det = _detector(threshold=0.5, refractory_s=1000.0)
    head = fakes["head.onnx"]
    head.probs = [0.9, 0.2, 0.9]
    assert det.push(_CHUNK) is True
    det.push(_CHUNK)
    assert det.push(_CHUNK) is False  # re-armed but inside the refractory window


def test_model_failure_returns_no_hit_not_an_exception(fakes):
    det = _detector()
    fakes["head.onnx"].run = None  # type: ignore[assignment]
    assert _push_chunks(det, 12) is False


def test_reset_clears_stream_state(fakes):
    det = _detector()
    _push_chunks(det, 10)
    assert det.last_score is not None
    det.reset()
    assert det.last_score is None
    det.push(_CHUNK)
    assert det.last_score == 0.0  # re-seeded: scoring resumes immediately


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
    _push_chunks(det, 12)
    det.reset()
    assert np.array_equal(det._embs, prime)  # never the all-zero OOD window


# ---- python mel frontend (hybrid NPU packages) --------------------------------


def _filters_npy(tmp_path):
    path = tmp_path / "mel_filters.npy"
    rng = np.random.default_rng(3)
    np.save(path, (rng.random((257, 32)) * 0.01).astype(np.float32))
    return str(path)


def test_python_mel_frontend_contract(tmp_path):
    fe = oww_mod.PythonMelFrontend(_filters_npy(tmp_path))
    (out,) = fe.run([("input", np.zeros((1, 1760), dtype=np.float32))])
    assert out.shape == (1, 1, 8, 32) and out.dtype == np.float32
    assert np.allclose(out, -100.0)  # silence: 10*log10(1e-10) everywhere
    # A loud burst mid-window: the graph's dynamic floor pins the quiet
    # frames at exactly max - 80 dB.
    rng = np.random.default_rng(0)
    x = np.zeros(1760, dtype=np.float32)
    x[800:1000] = (rng.standard_normal(200) * 20000).astype(np.float32)
    (out,) = fe.run([("input", x.reshape(1, -1))])
    assert float(out.min()) == pytest.approx(float(out.max()) - 80.0)
    assert fe.input_specs() == []
    fe.release()  # no-op


def test_python_mel_frontend_rejects_a_wrong_filterbank(tmp_path):
    path = tmp_path / "mel_filters.npy"
    np.save(path, np.zeros((80, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="shape"):
        oww_mod.PythonMelFrontend(str(path))


def test_mel_filters_path_selects_the_python_frontend(monkeypatch):
    made = {"emb.onnx": FakeEmb(), "head.onnx": FakeHead()}
    monkeypatch.setattr(oww_mod, "OnDeviceModel", lambda path, **kw: made[path])
    monkeypatch.setattr(oww_mod, "PythonMelFrontend", lambda path: FakeMel())
    det = OpenWakeWord(
        mel_filters_path="mel_filters.npy", embedding_path="emb.onnx",
        model_path="head.onnx", sample_rate=16000,
    )
    assert isinstance(det._mel, FakeMel)


def test_exactly_one_mel_frontend_is_required(fakes):
    with pytest.raises(ValueError, match="exactly one mel frontend"):
        _detector(mel_path=None)
    with pytest.raises(ValueError, match="exactly one mel frontend"):
        _detector(mel_filters_path="mel_filters.npy")


def test_rknn_embedding_input_is_pretransposed(monkeypatch):
    made = {"mel.onnx": FakeMel(), "emb.onnx": FakeEmb(), "head.onnx": FakeHead()}
    kws: dict[str, dict] = {}

    def spy(path, **kw):
        kws[path] = kw
        return made[path]

    monkeypatch.setattr(oww_mod, "OnDeviceModel", spy)
    _detector()
    assert kws["emb.onnx"]["rknn_input_permutation"] == (0, 2, 3, 1)
    assert "rknn_input_permutation" not in kws["mel.onnx"]
    assert "rknn_input_permutation" not in kws["head.onnx"]


def test_rknn_input_permutation_validates_at_construction():
    from nanobot_channel_voice.ondevice.runtime import OnDeviceModel

    with pytest.raises(ValueError, match="permutation"):
        OnDeviceModel("x.rknn", rknn_input_permutation=(1, 1, 0))


def test_rknn_run_permutes_matching_rank_inputs_only():
    import threading

    from nanobot_channel_voice.ondevice import runtime as rt

    m = rt.OnDeviceModel.__new__(rt.OnDeviceModel)
    m._released = False
    m._rknn_lock = threading.Lock()
    m._sess = None
    m._np = np
    m._rknn_input_permutation = (0, 2, 3, 1)
    seen = {}

    class Stub:
        def inference(self, inputs):
            seen["arrs"] = inputs
            return [np.zeros(1, dtype=np.float32)]

    m._rknn = Stub()
    x4 = np.arange(60, dtype=np.float32).reshape(1, 3, 4, 5)
    x2 = np.zeros((1, 7), dtype=np.float32)
    m.run([("a", x4), ("b", x2)])
    sent = seen["arrs"][0]
    assert sent.shape == (1, 4, 5, 3) and sent.flags["C_CONTIGUOUS"]
    # Lite's standard NHWC->NCHW transpose must restore the original tensor.
    assert np.array_equal(np.transpose(sent, (0, 3, 1, 2)), x4)
    assert seen["arrs"][1] is x2  # rank-mismatched sibling untouched


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


def test_missing_mel_frontend_degrades_to_text_tier(fakes):
    cfg = _wake_cfg(openwakeword={"embeddingPath": "emb.onnx", "modelPath": "head.onnx"})
    assert make_wake_detector(cfg, 16000, 20) is None


def _warns_during(fn):
    from loguru import logger as _logger

    msgs: list[str] = []
    sink = _logger.add(lambda m: msgs.append(str(m)), level="WARNING")
    try:
        result = fn()
    finally:
        _logger.remove(sink)
    return result, msgs


def test_meta_advisories_warn_without_blocking(fakes, tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text('{"phrase": "hey mycroft", "target": "rv1126b"}')
    cfg = _wake_cfg(openwakeword={
        "melPath": "mel.onnx", "embeddingPath": "emb.onnx",
        "modelPath": "head.onnx", "metaPath": str(meta),
    })
    det, msgs = _warns_during(lambda: make_wake_detector(cfg, 16000, 20))
    assert isinstance(det, OpenWakeWord)          # advisory only, never fatal
    joined = "".join(msgs)
    assert "detects 'hey mycroft'" in joined      # phrases say "hey nanobot"
    assert "targets" not in joined                # .onnx embedding: target n/a


def test_meta_advisories_match_is_silent_and_garbage_tolerated(fakes, tmp_path):
    meta = tmp_path / "meta.json"
    meta.write_text('{"phrase": "Hey Nanobot", "target": "rv1126b"}')
    cfg = _wake_cfg(openwakeword={
        "melPath": "mel.onnx", "embeddingPath": "emb.onnx",
        "modelPath": "head.onnx", "metaPath": str(meta),
    })
    det, msgs = _warns_during(lambda: make_wake_detector(cfg, 16000, 20))
    assert isinstance(det, OpenWakeWord) and not msgs  # casefold match, .onnx target
    meta.write_text("{not json")
    det, msgs = _warns_during(lambda: make_wake_detector(cfg, 16000, 20))
    assert isinstance(det, OpenWakeWord)
    assert any("unreadable" in m for m in msgs)
    # Parseable but malformed: never fatal, and never a nonsense warning.
    meta.write_text("[1, 2]")
    det, msgs = _warns_during(lambda: make_wake_detector(cfg, 16000, 20))
    assert isinstance(det, OpenWakeWord)
    assert any("unreadable" in m for m in msgs)
    meta.write_text('{"phrase": 42, "target": ["rv1126b"]}')
    det, msgs = _warns_during(lambda: make_wake_detector(cfg, 16000, 20))
    assert isinstance(det, OpenWakeWord) and not msgs


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


def test_hit_position_marks_the_chunk_end(fakes):
    det = _detector(threshold=0.5)
    head = fakes["head.onnx"]
    head.probs = [0.9]
    # One full chunk plus a 400-sample tail: the hit chunk ends 800 bytes back.
    assert det.push(_CHUNK + b"\x01\x00" * 400) is True
    assert det.last_hit_back_bytes == 800
    head.probs = [0.2, 0.9]
    assert det.push(b"\x01\x00" * 880) is False  # completes the tail chunk; 0.2 re-arms
    assert det.push(_CHUNK) is True
    assert det.last_hit_back_bytes == 0          # hit landed on the frame boundary


# ---- ack language routing (script of the called name) ------------------------


def test_script_class_and_uniform_script():
    from nanobot_channel_voice.backend.local import _script_class, _uniform_script

    assert _script_class("hey nanobot") == "latin"
    assert _script_class("小助手") == "han"
    assert _script_class("ねえアシスタント") == "kana"
    assert _script_class("小娜ちゃん") == "kana"  # kana wins: ja mixes both scripts
    assert _script_class("비서야") == "hangul"
    assert _script_class("123 !") is None
    assert _uniform_script(["小助手", "小娜"]) == "han"
    assert _uniform_script(["hey nanobot", "小娜"]) is None
    assert _uniform_script([]) is None


def test_canned_language_infers_only_where_the_engine_declares_nothing():
    from nanobot_channel_voice.backend.local import (
        _PROLOGUE_BUILTINS,
        _WAKE_ACK_BUILTINS,
        _WAKE_ACK_FALLBACK,
        _canned_language,
        _prologue_phrases,
        _wake_ack_phrases,
    )

    class _Tts:
        def __init__(self, lang, langs=None):
            self.spoken_language = lang
            if langs is not None:
                self.spoken_languages = langs

    # MMS below en, `say` and the cloud adapters declare nothing, and an English ack a
    # zh-only engine cannot voice synthesizes to SILENCE: the wake phrases carry the language.
    assert (zh := _canned_language(_Tts(None), ["小娜"])) == "zh"
    assert _wake_ack_phrases(None, zh) == _WAKE_ACK_BUILTINS["zh"]
    assert _prologue_phrases(None, zh) == _PROLOGUE_BUILTINS["zh"]
    assert _canned_language(_Tts(None), ["ねえアシスタント"]) == "ja"
    # A DECLARED language always wins: the summon's script never overrides the engine.
    assert _canned_language(_Tts("zh"), ["hey nanobot"]) == "zh"
    assert _canned_language(_Tts("en"), ["小娜"]) == "en"
    # A bilingual that declares no single language still declares a PRIMARY: the router
    # (and any spoken_languages carrier) acks in it, never the English fallback.
    assert _canned_language(_Tts(None, ("zh", "en")), ["hey nanobot"]) == "zh"
    assert _canned_language(_Tts(None, ("en", "zh")), ["小娜"]) == "en"
    # Nothing to infer from (latin is en/de-ambiguous, or no phrases): English fallback.
    assert _canned_language(_Tts(None), ["hey nanobot"]) is None
    assert _canned_language(_Tts(None), []) is None
    assert _wake_ack_phrases(None, None) == _WAKE_ACK_FALLBACK


def test_ack_pool_follows_the_tts_not_the_called_name():
    import asyncio

    from eval_harness import EvalConversation

    from nanobot_channel_voice.backend.local import (
        _WAKE_ACK_BUILTINS,
        _WAKE_ACK_FALLBACK,
    )

    async def _pool(matched, *, resolved=None, ack=None):
        wake = {"mode": "gate", "phrases": ["hey nanobot", "小娜"], "ack": {"enabled": True}}
        if ack is not None:
            wake["ack"]["phrases"] = ack
        async with EvalConversation(wake=wake) as conv:
            b = conv.backend
            if resolved is not None:
                b._wake_ack_list = resolved  # as construction resolves spoken_language
            return b._ack_pool(matched)

    async def _t():
        zh, en = _WAKE_ACK_BUILTINS["zh"], _WAKE_ACK_FALLBACK
        # The engine's own language answers BOTH summons: the called name's script never
        # crosses languages, and the engine may not even voice the other one.
        assert await _pool("小娜", resolved=zh) == zh
        assert await _pool("hey nanobot", resolved=zh) == zh
        assert await _pool("小娜", resolved=en) == en
        assert await _pool("hey nanobot", resolved=en) == en
        # A CONFIGURED mixed list is the one place the summon's script picks -- listing
        # both languages is what asks for that; an unrouted summon keeps the whole list.
        mixed = ["在呢。", "I'm here."]
        assert await _pool("小娜", ack=mixed) == ["在呢。"]
        assert await _pool("hey nanobot", ack=mixed) == ["I'm here."]
        assert await _pool(None, ack=mixed) == mixed

    asyncio.run(_t())


def test_skeleton_and_fuzzy_wake():
    from nanobot_channel_voice.wake.phrase import FuzzyWake, _skeleton

    assert _skeleton("heynanobot") == "hnnbt"
    assert _skeleton("henineobt") == "hnnbt"   # measured render, in-command
    assert _skeleton("henineought") == "hnngt"  # measured render, punctuated
    fz = FuzzyWake(["hey nanobot"])
    assert fz.strip_head("he nine obt what time is it") == ("hey nanobot", "what time is it")
    assert fz.strip_head("he nine ought") == ("hey nanobot", "")
    assert fz.strip_head("hey nano bot you should go") == ("hey nanobot", "you should go")
    # real speech with a name-like head stays content
    assert fz.strip_head("hey no but seriously listen")[0] is None
    assert fz.strip_head("he never got the memo")[0] is None
    assert fz.strip_head("turn on the lights")[0] is None
    # hesitation fillers may precede, like the exact tier
    assert fz.strip_head("um he nine obt turn on the lights") == ("hey nanobot", "turn on the lights")
    assert fz.strip_head("so hey nano bot come here") == ("hey nanobot", "come here")
    assert fz.strip_head("no but he can go")[0] is None
    # cross-script renders are the alias layer's job, not fuzzy's
    assert fz.strip_head("嘿难道爸")[0] is None
    # zh and too-short names opt out entirely
    assert not FuzzyWake(["小娜"])
    assert not FuzzyWake(["nova"])


def test_wake_phrase_pair_entries_report_the_display():
    """An alias entry matches its spelling but reports the CANONICAL phrase,
    so an alias summon routes the ack by the name the user called."""
    wp = WakePhrase(["hey nanobot", ("hey nanobot", "嘿难道爸")])
    assert wp.strip("嘿难道爸") == ("hey nanobot", "")
    assert wp.strip("嘿难道爸今天天气") == ("hey nanobot", "今天天气")
    assert wp.present("嘿难道爸")
    assert wp.strip("hey nanobot hello") == ("hey nanobot", "hello")


def test_compatibility_forms_still_summon():
    """Regression: patterns were built from NFKC-folded tokens but matched against the RAW
    transcript, so a fullwidth rendering (zh/ja ASR with ITN) missed the summon AND left
    the phrase in the published text."""
    wake = WakePhrase(["hey nanobot"])
    assert wake.strip("ｈｅｙ　ｎａｎｏｂｏｔ 天气") == ("hey nanobot", "天气")
    bare = WakePhrase(["nanobot"])
    assert bare.present("I said Ｎａｎｏｂｏｔ once")
    assert not bare.present("nanobots")  # the word-boundary rule survives folding
    # The remainder is sliced from the ORIGINAL text: spelling and case are preserved.
    assert wake.strip("Hey Nanobot, Weather?") == ("hey nanobot", "Weather?")
    assert FuzzyWake(["hey nanobot"]).strip_head("ｈｅ ｎｉｎｅ ｏｂｔ 天气") == (
        "hey nanobot", "天气",
    )


def test_composed_forms_fold_as_one_alphabet():
    """Regression: folding per CHARACTER never composed a halfwidth kana with its voiced
    mark (or an NFD base with its accent), so the pattern's NFKC form could not match."""
    import unicodedata

    assert WakePhrase(["ナノボット"]).strip("ﾅﾉﾎﾞｯﾄ 天気は") == ("ナノボット", "天気は")
    nfd = unicodedata.normalize("NFD", "café")
    assert WakePhrase(["café"]).strip(nfd + " order one") == ("café", "order one")
    # A match ending inside one source char's expansion consumes the whole char.
    assert WakePhrase(["株式"]).strip("㍿の件です") == ("株式", "の件です")


def test_a_phrase_configured_in_a_compatibility_form_matches_itself():
    """Regression: tokens_of folded the CONFIGURED phrase too, so a fullwidth entry
    matched only the ASCII spelling and never the form it was written in."""
    wake = WakePhrase(["ｎａｎｏｂｏｔ"])
    assert wake.strip("ｎａｎｏｂｏｔ weather")[0] == "ｎａｎｏｂｏｔ"  # display stays as written
    assert wake.strip("nanobot weather")[0] == "ｎａｎｏｂｏｔ"
