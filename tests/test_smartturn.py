"""Smart Turn v3 end-of-turn scoring: registry fallbacks, the backend's
consult-verdict-close flow, mel parity with the vendored reference, and an
optional real-model regression against the checkout under ``reference/``."""

from __future__ import annotations

import asyncio
import importlib.util
import os
import wave
from pathlib import Path

import pytest

from nanobot_channel_voice.config import VadConfig, VoiceConfig
from nanobot_channel_voice.vad import make_turn_analyzer
from nanobot_channel_voice.vad.base import Vad

_REF = Path(os.environ.get(
    "NANOBOT_VOICE_REF_DIR", Path(__file__).resolve().parents[2] / "reference"
))
_MODEL = (
    _REF / "pipecat" / "src" / "pipecat" / "audio" / "turn" / "smart_turn"
    / "data" / "smart-turn-v3.2-cpu.onnx"
)
_REF_MEL = (
    _REF / "pipecat" / "src" / "pipecat" / "audio" / "turn" / "smart_turn"
    / "_whisper_features.py"
)
_WAV = _REF / "whisper" / "model" / "test_en.wav"
# The RKNN v3.2 port only runs where rknnlite does (the board); the second
# candidate is the workspace model store's copy.
_MODEL_RKNN_CANDIDATES = (
    _REF / "smartturn" / "rknn.rv1126b" / "model.rknn",
    Path(__file__).resolve().parents[2]
    / "nanobot-channel-voice-test" / "models" / "vad" / "smartturn" / "rknn.rv1126b" / "model.rknn",
)
_MODEL_RKNN = next(
    (p for p in _MODEL_RKNN_CANDIDATES if p.is_file()), _MODEL_RKNN_CANDIDATES[0]
)
_WAV_RKNN_CANDIDATES = (
    _REF / "silero-vad" / "tests" / "data" / "test.wav",  # 60 s, 16 kHz
    _WAV,
)
_WAV_RKNN = next((p for p in _WAV_RKNN_CANDIDATES if p.is_file()), _WAV_RKNN_CANDIDATES[0])

FRAME_MS = 20
FRAME = b"\x01\x00" * 320  # 20 ms @ 16 kHz


# ---- registry fallbacks -----------------------------------------------------

def test_engine_none_builds_nothing():
    assert make_turn_analyzer(VadConfig(), 16000, FRAME_MS) is None


def test_missing_model_path_degrades_to_none():
    cfg = VadConfig.model_validate({"turn": {"engine": "smartturn"}})
    assert make_turn_analyzer(cfg, 16000, FRAME_MS) is None


def test_consult_at_or_past_hangover_degrades_to_none():
    cfg = VadConfig.model_validate({
        "hangoverMs": 240,
        "turn": {"engine": "smartturn", "modelPath": "/nonexistent.onnx", "consultMs": 240},
    })
    assert make_turn_analyzer(cfg, 16000, FRAME_MS) is None


def test_bad_model_path_degrades_to_none():
    cfg = VadConfig.model_validate({
        "turn": {"engine": "smartturn", "modelPath": "/nonexistent.onnx"},
    })
    assert make_turn_analyzer(cfg, 16000, FRAME_MS) is None


# ---- backend consult flow ---------------------------------------------------

class ScriptedVad(Vad):
    def __init__(self, decisions: list[bool]):
        self._decisions = list(decisions)

    def is_speech(self, frame: bytes) -> bool:
        return self._decisions.pop(0) if self._decisions else False


class _FakeAnalyzer:
    last_probability = 0.9

    def __init__(self, complete: bool):
        self._complete = complete
        self.calls: list[int] = []

    def assess(self, pcm: bytes) -> bool:
        self.calls.append(len(pcm))
        return self._complete

    def release(self) -> None:
        pass


def _build_backend(decisions: list[bool], analyzer):
    from nanobot_channel_voice.audio.null import NullPlayback
    from nanobot_channel_voice.backend.audio_sink import AudioSink
    from nanobot_channel_voice.backend.local import LocalBackend

    cfg = VoiceConfig.model_validate({
        "vad": {
            "startFrames": 2, "hangoverMs": 200, "minUtteranceMs": 40, "prerollMs": 0,
            "turn": {"engine": "smartturn", "consultMs": 60},
        },
    })

    async def transcribe(pcm: bytes) -> str:
        return ""

    async def publish(text: str, token: str, notes: tuple[str, ...] = ()) -> None:
        pass

    async def interrupt() -> None:
        pass

    return LocalBackend(
        cfg, vad=ScriptedVad(decisions), tts=None,
        sink=AudioSink(NullPlayback(), mode="blob"),
        transcribe=transcribe, publish_text=publish, interrupt=interrupt,
        turn_analyzer=analyzer,
    )


def test_complete_verdict_closes_before_the_hangover():
    analyzer = _FakeAnalyzer(complete=True)
    backend = _build_backend([True] * 3 + [False] * 20, analyzer)

    async def scenario():
        for _ in range(3 + 3):  # speech, then silence up to the consult mark
            await backend.push_audio(FRAME)
        assert backend._consult_task is not None
        await backend._consult_task  # verdict lands: COMPLETE
        silence_after_consult = 0
        while backend._utt_queue.qsize() == 0:
            silence_after_consult += 1
            assert silence_after_consult < 7, "did not close before the hangover (10 frames)"
            await backend.push_audio(FRAME)
        pending = backend._utt_queue.get_nowait()
        assert pending.silence_ms < 200  # anchored at the EARLY close, not the hangover
        assert analyzer.calls  # the model really scored the pause
        assert backend._metrics.counters.get("eou_complete") == 1
        assert backend._metrics.counters.get("eou_close_early") == 1

    asyncio.run(scenario())


def test_incomplete_verdict_waits_out_the_full_hangover():
    analyzer = _FakeAnalyzer(complete=False)
    backend = _build_backend([True] * 3 + [False] * 20, analyzer)

    async def scenario():
        for _ in range(3 + 3):
            await backend.push_audio(FRAME)
        await backend._consult_task
        closed_at = None
        for extra in range(1, 12):
            await backend.push_audio(FRAME)
            if backend._utt_queue.qsize():
                closed_at = extra
                break
        pending = backend._utt_queue.get_nowait()
        assert pending.silence_ms == 200  # the full hangover ran
        assert closed_at == 7  # 3 + 7 = 10 silence frames = 200 ms
        assert backend._metrics.counters.get("eou_incomplete") == 1
        assert backend._metrics.counters.get("eou_close_early") is None

    asyncio.run(scenario())


def test_verdict_after_natural_close_counts_stale_not_close():
    analyzer = _FakeAnalyzer(complete=True)
    backend = _build_backend([True] * 3 + [False] * 30, analyzer)

    async def scenario():
        for _ in range(3 + 3):
            await backend.push_audio(FRAME)
        # Suppress the verdict until after the hangover closed the utterance.
        task = backend._consult_task
        for _ in range(8):
            await backend.push_audio(FRAME)
        assert backend._utt_queue.qsize() == 1  # closed naturally
        await task
        await backend.push_audio(FRAME)  # verdict consumed here: must be stale
        assert backend._metrics.counters.get("eou_close_stale") == 1
        assert backend._utt_queue.qsize() == 1  # no second phantom close

    asyncio.run(scenario())


# ---- feature extractor parity -----------------------------------------------

def test_mel_matches_the_vendored_reference():
    np = pytest.importorskip("numpy")
    if not _REF_MEL.is_file():
        pytest.skip(f"reference extractor not present at {_REF_MEL}")
    from nanobot_channel_voice.vad._whisper_mel import WINDOW_SAMPLES, log_mel_features

    spec = importlib.util.spec_from_file_location("ref_whisper_features", _REF_MEL)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)

    rng = np.random.default_rng(7)
    audio = (rng.standard_normal(WINDOW_SAMPLES) * 0.1).astype(np.float32)
    ours = log_mel_features(audio)
    theirs = ref.compute_whisper_log_mel_features(audio, do_normalize=True)
    assert ours.shape == theirs.shape == (80, 800)
    assert np.allclose(ours, theirs, atol=1e-6)


# ---- real model (skipped when the reference checkout is absent) -------------

def test_real_model_scores_speech_deterministically():
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    if not (_MODEL.is_file() and _WAV.is_file()):
        pytest.skip("smart-turn model / test wav not present under reference/")
    cfg = VadConfig.model_validate({
        "turn": {"engine": "smartturn", "modelPath": str(_MODEL)},
    })
    analyzer = make_turn_analyzer(cfg, 16000, FRAME_MS)
    assert analyzer is not None, "real model failed to build"
    try:
        with wave.open(str(_WAV), "rb") as w:
            assert w.getframerate() == 16000
            pcm = w.readframes(w.getnframes())

        analyzer.assess(pcm)
        p_full = analyzer.last_probability
        analyzer.assess(pcm)
        assert analyzer.last_probability == p_full
        assert 0.0 <= p_full <= 1.0

        # Cutting the clip mid-speech must not score MORE complete than the
        # finished sentence.
        analyzer.assess(pcm[: int(len(pcm) * 0.55) // 2 * 2])
        p_cut = analyzer.last_probability
        assert 0.0 <= p_cut <= 1.0
        assert p_cut <= p_full
    finally:
        analyzer.release()


def test_real_rknn_model_scores_speech_deterministically():
    """The fixed-shape RKNN v3.2 port on the NPU: same behavioral contract as the
    ONNX real-model test above. Board-only (rknnlite)."""
    pytest.importorskip("numpy")
    pytest.importorskip("rknnlite.api", reason="RKNN Lite runtime only exists on the board")
    if not (_MODEL_RKNN.is_file() and _WAV_RKNN.is_file()):
        pytest.skip("smart-turn .rknn / test wav not present")
    cfg = VadConfig.model_validate({
        "turn": {"engine": "smartturn", "modelPath": str(_MODEL_RKNN)},
    })
    analyzer = make_turn_analyzer(cfg, 16000, FRAME_MS)
    assert analyzer is not None, "RKNN model failed to build"
    try:
        with wave.open(str(_WAV_RKNN), "rb") as w:
            assert w.getframerate() == 16000
            pcm = w.readframes(w.getnframes())

        analyzer.assess(pcm)
        p_full = analyzer.last_probability
        analyzer.assess(pcm)
        assert analyzer.last_probability == p_full
        assert 0.0 <= p_full <= 1.0

        # Cutting the clip mid-speech must not score MORE complete than the
        # finished sentence.
        analyzer.assess(pcm[: int(len(pcm) * 0.55) // 2 * 2])
        p_cut = analyzer.last_probability
        assert 0.0 <= p_cut <= 1.0
        assert p_cut <= p_full
    finally:
        analyzer.release()
