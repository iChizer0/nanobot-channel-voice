"""OPTIONAL real-model regression tests.

These run the actual on-device adapters over the reference ONNX exports and
skip cleanly where the models are absent (they are not bundled: the dev
container mounts them under ``<workspace>/reference``; override with
``NANOBOT_VOICE_REF_DIR``). They exist to keep the adapter refactors honest:
the unit suite proves the shells, these prove the model math still runs.
"""

from __future__ import annotations

import asyncio
import io
import os
import wave
from pathlib import Path

import pytest

_REF = Path(os.environ.get(
    "NANOBOT_VOICE_REF_DIR", Path(__file__).resolve().parents[2] / "reference"
))
_MMS = _REF / "mms_tts" / "model"
_WHISPER = _REF / "whisper" / "model"
_FIRERED = _REF / "vad" / "FireRedVAD" / "pretrained_models" / "onnx_models"

pytestmark = pytest.mark.skipif(
    not _REF.is_dir(), reason=f"reference models not present at {_REF}"
)


def _need(*paths: Path):
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        pytest.skip(f"model files missing: {missing}")


def test_mms_real_synthesis_wav_and_pcm():
    _need(_MMS / "mms_tts_eng_encoder_200.onnx", _MMS / "mms_tts_eng_decoder_200.onnx")
    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.tts import make_tts

    tts = make_tts(TtsConfig.model_validate({
        "provider": "mms",
        "mms": {
            "encoderPath": str(_MMS / "mms_tts_eng_encoder_200.onnx"),
            "decoderPath": str(_MMS / "mms_tts_eng_decoder_200.onnx"),
        },
    }))
    assert type(tts).__name__ == "MmsTtsAdapter"  # no silent fallback to system

    blob = asyncio.run(tts.synthesize("Hello from the regression test."))
    with wave.open(io.BytesIO(blob), "rb") as w:
        duration_s = w.getnframes() / w.getframerate()
        assert w.getframerate() == 16000
    assert 0.5 < duration_s < 6.0, duration_s

    pcm = asyncio.run(tts.synthesize_pcm("Short check."))
    assert len(pcm) > 16000  # > 0.5 s of 16 kHz S16_LE


def test_whisper_real_transcription():
    _need(
        _WHISPER / "whisper_encoder_base_20s.onnx",
        _WHISPER / "whisper_decoder_base_20s.onnx",
        _WHISPER / "vocab_en.txt",
        _WHISPER / "mel_80_filters.txt",
        _WHISPER / "test_en.wav",
    )
    from nanobot_channel_voice.config import SttConfig
    from nanobot_channel_voice.stt import make_stt

    stt = make_stt(SttConfig.model_validate({
        "provider": "whisper",
        "whisper": {
            "encoderPath": str(_WHISPER / "whisper_encoder_base_20s.onnx"),
            "decoderPath": str(_WHISPER / "whisper_decoder_base_20s.onnx"),
            "vocabPath": str(_WHISPER / "vocab_en.txt"),
            "melFiltersPath": str(_WHISPER / "mel_80_filters.txt"),
            "language": "en",
            "chunkLength": 20,
        },
    }))
    assert type(stt).__name__ == "WhisperOnDeviceStt"  # no silent delegate fallback

    with wave.open(str(_WHISPER / "test_en.wav"), "rb") as w:
        rate = w.getframerate()
        pcm = w.readframes(w.getnframes())
    text = asyncio.run(stt.transcribe(pcm, rate))
    assert isinstance(text, str) and len(text.strip()) > 0
    assert any(c.isalpha() for c in text)


def test_firered_real_model_and_degrade_path():
    _need(_FIRERED / "fireredvad_stream_vad_with_cache.onnx", _FIRERED / "cmvn.ark")
    from nanobot_channel_voice.config import VadConfig
    from nanobot_channel_voice.vad import make_vad
    from nanobot_channel_voice.vad.energy import EnergyVad
    from nanobot_channel_voice.vad.firered import FireRedVad

    good = make_vad(VadConfig.model_validate({
        "engine": "firered",
        "firered": {
            "modelPath": str(_FIRERED / "fireredvad_stream_vad_with_cache.onnx"),
            "cmvnPath": str(_FIRERED / "cmvn.ark"),
        },
    }), 16000, 20)
    assert isinstance(good, FireRedVad)
    assert good.is_speech(b"\x00\x00" * 320) is False  # a 20 ms silence frame

    # The construction-failure path (this is where the model used to leak):
    # a bad side file must degrade to energy, releasing the claimed session.
    bad = make_vad(VadConfig.model_validate({
        "engine": "firered",
        "firered": {
            "modelPath": str(_FIRERED / "fireredvad_stream_vad_with_cache.onnx"),
            "cmvnPath": "/nonexistent/cmvn.ark",
        },
    }), 16000, 20)
    assert isinstance(bad, EnergyVad)


def test_firered_min_volume_gates_quiet_speech():
    _need(_FIRERED / "fireredvad_stream_vad_with_cache.onnx",
          _WHISPER / "test_en.wav")
    from nanobot_channel_voice.config import VadConfig
    from nanobot_channel_voice.vad import make_vad
    from nanobot_channel_voice.vad.firered import FireRedVad

    def _build_vad(min_volume: float):
        return make_vad(VadConfig.model_validate({
            "engine": "firered",
            "firered": {
                "modelPath": str(_FIRERED / "fireredvad_stream_vad_with_cache.onnx"),
                "cmvnPath": str(_FIRERED / "cmvn.ark"),
                "minVolume": min_volume,
            },
        }), 16000, 20)

    with wave.open(str(_WHISPER / "test_en.wav"), "rb") as w:
        pcm = w.readframes(w.getnframes())
    frames = [pcm[i:i + 640] for i in range(0, min(len(pcm), 640 * 150), 640)]

    plain = _build_vad(0.0)
    assert isinstance(plain, FireRedVad)
    flags_plain = [plain.is_speech(f) for f in frames]
    plain.release()
    assert any(flags_plain)  # the model does flag the reference speech

    gated = _build_vad(0.9)  # near-clipping RMS: nothing real reaches this
    flags_gated = [gated.is_speech(f) for f in frames]
    gated.release()
    assert not any(flags_gated)  # AND'd loudness gate wins
