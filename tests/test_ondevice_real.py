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
import itertools
import os
import wave
from pathlib import Path

import pytest

_REF = Path(os.environ.get(
    "NANOBOT_VOICE_REF_DIR", Path(__file__).resolve().parents[2] / "reference"
))
_MMS = _REF / "mms_tts" / "model"
_MATCHA = _REF / "matcha_tts" / "model"  # model dirs + shared vocoder + official exports
_WHISPER = _REF / "whisper" / "model"
_FIRERED = _REF / "vad" / "FireRedVAD" / "pretrained_models" / "onnx_models"
_SILERO_DIR = _REF / "vad" / "silero"
_SILERO_CANDIDATES = (
    _SILERO_DIR / "silero_vad_v6.onnx",  # v6.2.1 via download_model.sh
    _REF / "pipecat" / "src" / "pipecat" / "audio" / "vad" / "data" / "silero_vad.onnx",  # v6.0
)
_SILERO = next((p for p in _SILERO_CANDIDATES if p.is_file()), _SILERO_CANDIDATES[0])
# The RKNN v6 port only runs where rknnlite does (the board); the second
# candidate is the workspace model store's copy.
_SILERO_RKNN_CANDIDATES = (
    _SILERO_DIR / "rknn.rv1126b" / "model.rknn",
    Path(__file__).resolve().parents[2]
    / "nanobot-channel-voice-test" / "models" / "vad" / "silero" / "v6" / "rknn.rv1126b" / "model.rknn",
)
_SILERO_RKNN = next((p for p in _SILERO_RKNN_CANDIDATES if p.is_file()), _SILERO_RKNN_CANDIDATES[0])
# A 16 kHz speech wav: the upstream silero-vad repo's own test clip, else whisper's.
_SILERO_WAV_CANDIDATES = (
    _REF / "silero-vad" / "tests" / "data" / "test.wav",
    _WHISPER / "test_en.wav",
)
_SILERO_WAV = next((p for p in _SILERO_WAV_CANDIDATES if p.is_file()), _SILERO_WAV_CANDIDATES[0])

pytestmark = pytest.mark.skipif(
    not _REF.is_dir(), reason=f"reference models not present at {_REF}"
)


def _need(*paths: Path):
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        pytest.skip(f"model files missing: {missing}")


def _need_espeak():
    import importlib.util
    import shutil

    if not (shutil.which("espeak-ng") or shutil.which("espeak")
            or importlib.util.find_spec("espeakng_loader")):
        pytest.skip("no espeak-ng binary and no espeakng-loader ([espeak] extra)")


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


def test_mms_real_overrun_is_recut_where_it_fits():
    """The FIXED 2*max_length-frame window, not the encoder token count, is the binding
    ceiling for ~1 sentence in 8 at the budget; the re-cut must land the head inside the
    window on its first retry, and ordinary sentences must not be split at all."""
    _need(_MMS / "mms_tts_eng_encoder_200.onnx", _MMS / "mms_tts_eng_decoder_200.onnx")
    from nanobot_channel_voice.config import MmsTtsConfig
    from nanobot_channel_voice.tts.base import split_for_budget
    from nanobot_channel_voice.tts.mms import MmsTtsAdapter

    tts = MmsTtsAdapter.from_config(MmsTtsConfig(
        encoder_path=str(_MMS / "mms_tts_eng_encoder_200.onnx"),
        decoder_path=str(_MMS / "mms_tts_eng_decoder_200.onnx"),
    ))
    pieces: list[str] = []
    inner = tts._synthesize_piece
    tts._synthesize_piece = lambda text: (pieces.append(text), inner(text))[1]
    try:
        plain = "Well, the short answer is yes, although there are a few caveats."
        assert split_for_budget(tts._normalize(plain), tts._piece_budget()) == [plain]
        long = (
            "Your flight leaves at seven forty in the morning, so please leave the house "
            "by five thirty."
        )
        asyncio.run(tts.synthesize_pcm(long))
        # One overrun, one re-cut (2 pieces), nothing recursed: at most 3 passes.
        assert len(pieces) <= 3, pieces
    finally:
        tts.release()


def test_matcha_real_official_embedded_vocoder():
    """The preferred artifact: python -m matcha.onnx.export with the vocoder embedded
    (scales input, wav output, built-in symbol table - no side files at all)."""
    model = _MATCHA / "matcha_ljspeech_hifigan.onnx"
    _need(model)
    _need_espeak()
    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.tts import make_tts

    tts = make_tts(TtsConfig.model_validate({
        "provider": "matcha",
        "matcha": {"acousticModelPath": str(model)},
    }))
    assert type(tts).__name__ == "MatchaTtsAdapter"  # no silent fallback to system
    assert tts.output_rate == 22050
    assert tts.spoken_language == "en"

    pcm = asyncio.run(tts.synthesize_pcm("The official matcha export speaks for itself."))
    duration_s = len(pcm) / 2 / 22050
    assert 1.0 < duration_s < 8.0, duration_s
    tts.release()


def test_matcha_real_static_split_onnx():
    """The static split over its .onnx artifacts: same host glue the board runs."""
    en = _MATCHA / "matcha-icefall-en_US-ljspeech"
    _need(
        _MATCHA / "matcha_encoder_200.onnx",
        _MATCHA / "matcha_decoder_800.onnx",
        _MATCHA / "vocos_800.onnx",
        en / "tokens.txt",
    )
    _need_espeak()
    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.tts import make_tts

    tts = make_tts(TtsConfig.model_validate({
        "provider": "matcha",
        "matcha": {
            "encoderPath": str(_MATCHA / "matcha_encoder_200.onnx"),
            "decoderPath": str(_MATCHA / "matcha_decoder_800.onnx"),
            "vocoderPath": str(_MATCHA / "vocos_800.onnx"),
            "tokensPath": str(en / "tokens.txt"),
        },
    }))
    assert type(tts).__name__ == "SplitMatchaTtsAdapter"  # no silent fallback
    assert tts.output_rate == 22050
    assert tts.spoken_language == "en"

    pcm = asyncio.run(tts.synthesize_pcm("The static split speaks through fixed buckets."))
    duration_s = len(pcm) / 2 / 22050
    assert 1.0 < duration_s < 8.0, duration_s
    tts.release()


def test_matcha_split_encoder_tiling_is_identical_where_the_mask_is_honored():
    """The encoder bucket repeats the phonemes instead of padding. That is only safe if a
    graph honoring ``x_mask`` cannot see the tail — so prove it: byte-identical mu/logw."""
    en = _MATCHA / "matcha-icefall-en_US-ljspeech"
    _need(
        _MATCHA / "matcha_encoder_200.onnx",
        _MATCHA / "matcha_decoder_800.onnx",
        _MATCHA / "vocos_800.onnx",
        en / "tokens.txt",
    )
    _need_espeak()
    import numpy as np

    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.tts import make_tts

    tts = make_tts(TtsConfig.model_validate({
        "provider": "matcha",
        "matcha": {
            "encoderPath": str(_MATCHA / "matcha_encoder_200.onnx"),
            "decoderPath": str(_MATCHA / "matcha_decoder_800.onnx"),
            "vocoderPath": str(_MATCHA / "vocos_800.onnx"),
            "tokensPath": str(en / "tokens.txt"),
        },
    }))
    try:
        ids = tts._ids("Yes?")                      # the shortest thing we ever synthesize
        n = len(ids)
        assert n * 4 < tts._encoder_len             # pads would dominate the bucket
        mask = np.zeros((1, 1, tts._encoder_len), dtype=np.float32)
        mask[0, 0, :n] = 1.0
        padded = np.full((1, tts._encoder_len), tts._pad_id, dtype=np.int64)
        padded[0, :n] = ids
        tiled = np.resize(np.asarray(ids, dtype=np.int64), (1, tts._encoder_len))
        assert not np.array_equal(padded, tiled)    # the inputs really do differ
        for a, b in zip(
            tts._encoder.run([("x", padded), ("x_mask", mask)]),
            tts._encoder.run([("x", tiled), ("x_mask", mask)]),
        ):
            assert np.array_equal(np.asarray(a)[..., :n], np.asarray(b)[..., :n])
    finally:
        tts.release()


def test_matcha_real_zh_synthesis():
    zh = _MATCHA / "matcha-icefall-zh-baker"
    vocoder = _MATCHA / "vocos-22khz-univ.onnx"
    _need(zh / "model-steps-3.onnx", zh / "tokens.txt", zh / "lexicon.txt", vocoder)
    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.tts import make_tts

    tts = make_tts(TtsConfig.model_validate({
        "provider": "matcha",
        "matcha": {
            "acousticModelPath": str(zh / "model-steps-3.onnx"),
            "vocoderPath": str(vocoder),
            "tokensPath": str(zh / "tokens.txt"),
            "lexiconPath": str(zh / "lexicon.txt"),
        },
    }))
    assert type(tts).__name__ == "MatchaTtsAdapter"  # no silent fallback to system
    assert tts.output_rate == 22050
    assert tts.spoken_language == "zh"

    pcm = asyncio.run(tts.synthesize_pcm("你好，现在是7点45分。"))  # digits verbalize
    duration_s = len(pcm) / 2 / 22050
    assert 1.0 < duration_s < 10.0, duration_s

    # English fallback: acronym spells, word transliterates — both must add
    # real audio over the zh-only remainder instead of dropping to silence.
    base = asyncio.run(tts.synthesize_pcm("打开设置。"))
    mixed = asyncio.run(tts.synthesize_pcm("打开WiFi和USB设置。"))
    assert len(mixed) > len(base) + 22050 // 2  # >0.25 s of voiced English
    tts.release()


def test_matcha_real_en_synthesis():
    en = _MATCHA / "matcha-icefall-en_US-ljspeech"
    vocoder = _MATCHA / "vocos-22khz-univ.onnx"
    _need(en / "model-steps-3.onnx", en / "tokens.txt", vocoder)
    _need_espeak()
    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.tts import make_tts

    tts = make_tts(TtsConfig.model_validate({
        "provider": "matcha",
        "matcha": {
            "acousticModelPath": str(en / "model-steps-3.onnx"),
            "vocoderPath": str(vocoder),
            "tokensPath": str(en / "tokens.txt"),
        },
    }))
    assert type(tts).__name__ == "MatchaTtsAdapter"
    assert tts.spoken_language == "en"

    blob = asyncio.run(tts.synthesize("Hello from the matcha regression test."))
    with wave.open(io.BytesIO(blob), "rb") as w:
        duration_s = w.getnframes() / w.getframerate()
        assert w.getframerate() == 22050
    assert 0.5 < duration_s < 6.0, duration_s
    tts.release()


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


def test_silero_real_model_flags_speech_and_not_silence():
    _need(_SILERO, _WHISPER / "test_en.wav")
    from nanobot_channel_voice.config import VadConfig
    from nanobot_channel_voice.vad import make_vad
    from nanobot_channel_voice.vad.silero import SileroVad

    vad = make_vad(VadConfig.model_validate({
        "engine": "silero",
        "silero": {"modelPath": str(_SILERO)},
    }), 16000, 20)
    assert isinstance(vad, SileroVad)

    silence_flags = [vad.is_speech(b"\x00\x00" * 320) for _ in range(50)]
    assert not any(silence_flags)

    vad.reset()
    with wave.open(str(_WHISPER / "test_en.wav"), "rb") as w:
        pcm = w.readframes(w.getnframes())
    frames = [pcm[i:i + 640] for i in range(0, min(len(pcm), 640 * 150), 640)]
    flags = [vad.is_speech(f) for f in frames]
    vad.release()
    assert any(flags)
    # An utterance is a contiguous run, not isolated blips: the hysteresis pair
    # must hold through word-internal dips at this frame granularity.
    longest = max((len(list(g)) for k, g in itertools.groupby(flags) if k), default=0)
    assert longest >= 10  # >= 200 ms of continuous speech in a ~2 s utterance


def test_silero_real_model_construction_paths():
    _need(_SILERO)
    from nanobot_channel_voice.config import VadConfig
    from nanobot_channel_voice.vad import make_vad
    from nanobot_channel_voice.vad.energy import EnergyVad
    from nanobot_channel_voice.vad.silero import SileroVad

    # The combined export declares sr, so the SAME artifact runs at 8 kHz.
    at_8k = make_vad(VadConfig.model_validate({
        "engine": "silero", "silero": {"modelPath": str(_SILERO)},
    }), 8000, 20)
    assert isinstance(at_8k, SileroVad)
    assert at_8k.is_speech(b"\x00\x00" * 160) is False  # a 20 ms 8 kHz silence frame
    at_8k.release()

    bad = make_vad(VadConfig.model_validate({
        "engine": "silero", "silero": {"modelPath": "/nonexistent/silero_vad.onnx"},
    }), 16000, 20)
    assert isinstance(bad, EnergyVad)


def test_silero_single_rate_export_flags_speech():
    # The 16k-only v6 export: flattened graph, the shape a TensorRT/RKNN port starts from.
    model = _SILERO_DIR / "silero_vad_16k_op15.onnx"
    _need(model, _WHISPER / "test_en.wav")
    from nanobot_channel_voice.config import VadConfig
    from nanobot_channel_voice.vad import make_vad
    from nanobot_channel_voice.vad.silero import SileroVad

    vad = make_vad(VadConfig.model_validate({
        "engine": "silero", "silero": {"modelPath": str(model)},
    }), 16000, 20)
    assert isinstance(vad, SileroVad)
    with wave.open(str(_WHISPER / "test_en.wav"), "rb") as w:
        pcm = w.readframes(w.getnframes())
    flags = [vad.is_speech(pcm[i:i + 640]) for i in range(0, min(len(pcm), 640 * 150), 640)]
    vad.release()
    assert any(flags)


def test_silero_rknn_real_model_flags_speech_and_not_silence():
    """The fixed-shape RKNN v6 port on the NPU: same adapter contract as the ONNX
    exports, so the same speech/silence proof. Board-only (rknnlite)."""
    pytest.importorskip("rknnlite.api", reason="RKNN Lite runtime only exists on the board")
    _need(_SILERO_RKNN, _SILERO_WAV)
    from nanobot_channel_voice.config import VadConfig
    from nanobot_channel_voice.vad import make_vad
    from nanobot_channel_voice.vad.silero import SileroVad

    vad = make_vad(VadConfig.model_validate({
        "engine": "silero", "silero": {"modelPath": str(_SILERO_RKNN)},
    }), 16000, 20)
    assert isinstance(vad, SileroVad)

    silence_flags = [vad.is_speech(b"\x00\x00" * 320) for _ in range(50)]
    assert not any(silence_flags)

    vad.reset()
    with wave.open(str(_SILERO_WAV), "rb") as w:
        assert w.getframerate() == 16000, _SILERO_WAV  # the port is 16 kHz only
        pcm = w.readframes(w.getnframes())
    frames = [pcm[i:i + 640] for i in range(0, min(len(pcm), 640 * 1500), 640)]
    flags = [vad.is_speech(f) for f in frames]
    vad.release()
    assert any(flags)
    longest = max((len(list(g)) for k, g in itertools.groupby(flags) if k), default=0)
    assert longest >= 10  # >= 200 ms of continuous speech


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


# The RKNN SenseVoice-Small port + its frontend.json/tokens.txt sidecars; the
# second candidate is the workspace model store's copy. Board-only (rknnlite).
_SENSEVOICE_RKNN_CANDIDATES = (
    _REF / "sensevoice" / "rknn.rv1126b",
    Path(__file__).resolve().parents[2]
    / "nanobot-channel-voice-test" / "models" / "stt" / "sensevoice" / "small" / "rknn.rv1126b",
)
_SENSEVOICE_RKNN = next(
    (p for p in _SENSEVOICE_RKNN_CANDIDATES if p.is_dir()), _SENSEVOICE_RKNN_CANDIDATES[0]
)
_SENSEVOICE_WAV_CANDIDATES = (
    _REF / "sensevoice" / "en.wav",  # the upstream en.mp3 example, as 16 kHz wav
    _WHISPER / "test_en.wav",
)
_SENSEVOICE_WAV = next(
    (p for p in _SENSEVOICE_WAV_CANDIDATES if p.is_file()), _SENSEVOICE_WAV_CANDIDATES[0]
)


def test_sensevoice_rknn_real_transcription():
    pytest.importorskip("rknnlite.api", reason="RKNN Lite runtime only exists on the board")
    d = _SENSEVOICE_RKNN
    _need(d / "model.rknn", d / "frontend.json", d / "tokens.txt", _SENSEVOICE_WAV)
    from nanobot_channel_voice.config import SttConfig
    from nanobot_channel_voice.stt import make_stt
    from nanobot_channel_voice.stt.sensevoice import SenseVoiceOnDeviceStt

    stt = make_stt(SttConfig.model_validate({
        "provider": "sensevoice",
        "sensevoice": {
            "modelPath": str(d / "model.rknn"),
            "tokensPath": str(d / "tokens.txt"),
            "frontendPath": str(d / "frontend.json"),
            "language": "auto",
        },
    }))
    assert isinstance(stt, SenseVoiceOnDeviceStt)  # no silent delegate fallback
    with wave.open(str(_SENSEVOICE_WAV), "rb") as w:
        assert w.getframerate() == 16000
        pcm = w.readframes(w.getnframes())
    text = asyncio.run(stt.transcribe(pcm, 16000))
    stt.release()
    assert isinstance(text, str) and len(text.strip()) > 0
    assert any(c.isalpha() for c in text)
    assert stt.last_tags.startswith("<|")  # rich-transcription tags surfaced


# The RKNN streaming Zipformer trio + its meta.json/tokens.txt sidecars; the
# second candidate is the workspace model store's copy. Board-only (rknnlite).
_ZIPFORMER_RKNN_CANDIDATES = (
    _REF / "zipformer" / "rknn.rv1126b",
    Path(__file__).resolve().parents[2]
    / "nanobot-channel-voice-test" / "models" / "stt" / "zipformer" / "zh-en" / "rknn.rv1126b",
)
_ZIPFORMER_RKNN = next(
    (p for p in _ZIPFORMER_RKNN_CANDIDATES if p.is_dir()), _ZIPFORMER_RKNN_CANDIDATES[0]
)
_ZIPFORMER_ONNX = (
    _REF / "zipformer" / "sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"
)
_ZIPFORMER_WAV_CANDIDATES = (  # first: the sherpa export's bundled bilingual clip
    _REF / "zipformer" / "0.wav",
    _ZIPFORMER_ONNX / "test_wavs" / "0.wav",
    _WHISPER / "test_en.wav",
)
_ZIPFORMER_WAV = next(
    (p for p in _ZIPFORMER_WAV_CANDIDATES if p.is_file()), _ZIPFORMER_WAV_CANDIDATES[0]
)


def test_zipformer_onnx_real_streaming_matches_batch():
    d = _ZIPFORMER_ONNX
    enc, dec, join = (
        d / f"{n}-epoch-99-avg-1.int8.onnx" for n in ("encoder", "decoder", "joiner")
    )
    _need(enc, dec, join, d / "tokens.txt", _ZIPFORMER_WAV)
    from nanobot_channel_voice.config import SttConfig
    from nanobot_channel_voice.stt import make_stt
    from nanobot_channel_voice.stt.zipformer import ZipformerOnDeviceStt

    stt = make_stt(SttConfig.model_validate({
        "provider": "zipformer",
        "zipformer": {
            "encoderPath": str(enc), "decoderPath": str(dec),
            "joinerPath": str(join), "tokensPath": str(d / "tokens.txt"),
        },
    }))
    assert isinstance(stt, ZipformerOnDeviceStt)  # no silent delegate fallback
    asyncio.run(stt.warmup())  # the silent warmup decode must not disturb real decodes
    with wave.open(str(_ZIPFORMER_WAV), "rb") as w:
        assert w.getframerate() == 16000
        pcm = w.readframes(w.getnframes())
    batch = asyncio.run(stt.transcribe(pcm, 16000))
    stream = stt.stream_start()
    for i in range(0, len(pcm), 640):  # 20 ms frames, the live capture size
        stream.accept(pcm[i : i + 640])
    streamed = stream.finish()
    stt.release()
    assert batch.strip() and any(c.isalpha() for c in batch)
    assert streamed == batch  # the streaming path is transcript-identical to batch


def test_zipformer_onnx_real_start_context_matches_sherpa():
    """The decoder context opens as sherpa-onnx opens it ([-1, blank], wrapped by this
    clamp-free export's Gather to the last vocab row), not [blank, blank]: on the fp32
    export's own clips 2.wav read 这个是频繁的 instead of sherpa's 是不是平凡的."""
    d = _ZIPFORMER_ONNX
    enc, dec, join = (d / f"{n}-epoch-99-avg-1.onnx" for n in ("encoder", "decoder", "joiner"))
    wavs = {name: d / "test_wavs" / f"{name}.wav" for name in ("0", "1", "2", "3")}
    _need(enc, dec, join, d / "tokens.txt", *wavs.values())
    from nanobot_channel_voice.config import ZipformerSttConfig
    from nanobot_channel_voice.stt.zipformer import ZipformerOnDeviceStt

    stt = ZipformerOnDeviceStt.from_config(ZipformerSttConfig(
        encoder_path=str(enc), decoder_path=str(dec), joiner_path=str(join),
        tokens_path=str(d / "tokens.txt"),
    ))
    try:
        heard = {}
        for name, path in wavs.items():
            with wave.open(str(path), "rb") as w:
                assert w.getframerate() == 16000
                heard[name] = asyncio.run(stt.transcribe(w.readframes(w.getnframes()), 16000))
    finally:
        stt.release()
    assert heard == {
        "0": "昨天是 MONDAY TODAY IS LIBR THE DAY AFTER TOMORROW是星期三",
        "1": "这是第一种第二种叫呃与 ALWAYS ALWAYS什么意思啊",
        "2": "是不是平凡的啊不认识记下来 FREQUENTLY频繁的",
        "3": "第一句是个什么时态加了 ES是一般现在时对后面它时态写上",
    }


def test_zipformer_rknn_real_transcription():
    pytest.importorskip("rknnlite.api", reason="RKNN Lite runtime only exists on the board")
    d = _ZIPFORMER_RKNN
    _need(d / "encoder.rknn", d / "decoder.rknn", d / "joiner.rknn",
          d / "meta.json", d / "tokens.txt", _ZIPFORMER_WAV)
    from nanobot_channel_voice.config import SttConfig
    from nanobot_channel_voice.stt import make_stt
    from nanobot_channel_voice.stt.zipformer import ZipformerOnDeviceStt

    stt = make_stt(SttConfig.model_validate({
        "provider": "zipformer",
        "zipformer": {
            "encoderPath": str(d / "encoder.rknn"),
            "decoderPath": str(d / "decoder.rknn"),
            "joinerPath": str(d / "joiner.rknn"),
            "tokensPath": str(d / "tokens.txt"),
            "metaPath": str(d / "meta.json"),
        },
    }))
    assert isinstance(stt, ZipformerOnDeviceStt)  # no silent delegate fallback
    with wave.open(str(_ZIPFORMER_WAV), "rb") as w:
        assert w.getframerate() == 16000
        pcm = w.readframes(w.getnframes())
    text = asyncio.run(stt.transcribe(pcm, 16000))
    stt.release()
    assert isinstance(text, str) and len(text.strip()) > 0
    assert any(c.isalpha() for c in text)  # True for CJK too: bilingual-safe


def test_matcha_real_bilingual_router():
    zh = _MATCHA / "matcha-icefall-zh-baker"
    en = _MATCHA / "matcha-icefall-en_US-ljspeech"
    vocoder = _MATCHA / "vocos-22khz-univ.onnx"
    _need(zh / "model-steps-3.onnx", en / "model-steps-3.onnx", vocoder)
    _need_espeak()
    from nanobot_channel_voice.config import TtsConfig
    from nanobot_channel_voice.tts import make_tts
    from nanobot_channel_voice.tts.router import ScriptRoutedTts

    tts = make_tts(TtsConfig.model_validate({
        "provider": "matcha",
        "matcha": {
            "acousticModelPath": str(zh / "model-steps-3.onnx"),
            "vocoderPath": str(vocoder),
            "tokensPath": str(zh / "tokens.txt"),
            "lexiconPath": str(zh / "lexicon.txt"),
            "secondary": {
                "acousticModelPath": str(en / "model-steps-3.onnx"),
                "vocoderPath": str(vocoder),
                "tokensPath": str(en / "tokens.txt"),
            },
        },
    }))
    assert isinstance(tts, ScriptRoutedTts)
    assert tts.spoken_languages == ("zh", "en")
    assert tts._cjk._vocoder is tts._latin._vocoder  # one shared vocos session

    pcm = asyncio.run(tts.synthesize_pcm("你好，please turn on the WiFi，谢谢。"))
    duration_s = len(pcm) / 2 / 22050
    assert 2.0 < duration_s < 15.0, duration_s
    tts.release()


# ---- openWakeWord: python mel parity + the upstream fixture ------------------

_OWW = _REF / "oww" / "model"


def test_openwakeword_python_mel_matches_the_onnx_graph():
    _need(_OWW / "melspectrogram.onnx", _OWW / "mel_filters.npy")
    import numpy as np

    from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
    from nanobot_channel_voice.wake.openwakeword import PythonMelFrontend

    fe = PythonMelFrontend(str(_OWW / "mel_filters.npy"))
    rng = np.random.default_rng(7)
    with OnDeviceModel(str(_OWW / "melspectrogram.onnx"), intra_op_threads=1) as ref:
        worst = 0.0
        for scale in (0.0, 1.0, 30.0, 3000.0, 32000.0):
            x = (rng.standard_normal(1760) * scale).astype(np.float32).reshape(1, -1)
            (want,) = ref.run([("input", x)])
            (got,) = fe.run([("input", x)])
            assert got.shape == tuple(np.asarray(want).shape) == (1, 1, 8, 32)
            worst = max(worst, float(np.abs(np.asarray(want, dtype=np.float64) - got).max()))
    assert worst < 1e-3  # measured ~2e-5 dB; near a dB means real drift


def test_openwakeword_fixture_hits_with_both_frontends():
    _need(_OWW / "melspectrogram.onnx", _OWW / "mel_filters.npy",
          _OWW / "embedding_model.onnx", _OWW / "hey_mycroft_v0.1.onnx",
          _OWW / "hey_mycroft_test.wav")
    from nanobot_channel_voice.wake.openwakeword import OpenWakeWord

    with wave.open(str(_OWW / "hey_mycroft_test.wav")) as w:
        assert (w.getframerate(), w.getnchannels(), w.getsampwidth()) == (16000, 1, 2)
        pcm = w.readframes(w.getnframes())

    def run(**mel):
        det = OpenWakeWord(embedding_path=str(_OWW / "embedding_model.onnx"),
                           model_path=str(_OWW / "hey_mycroft_v0.1.onnx"),
                           sample_rate=16000, threshold=0.5, **mel)
        try:
            return det.push(pcm), det.last_score
        finally:
            det.release()

    hit_a, score_a = run(mel_path=str(_OWW / "melspectrogram.onnx"))
    hit_b, score_b = run(mel_filters_path=str(_OWW / "mel_filters.npy"))
    assert hit_a and hit_b  # an unseeded mel window misses this fixture entirely
    assert abs(score_a - score_b) < 1e-3


# ---- gated uplink on the real detectors: what leaves the device ---------------
# The offline half of the "SBC as smart speaker" validation: silero + openWakeWord
# drive GatedUplink over a scripted room, a fake cloud records what went up, and
# uplink_ms / capture_ms is the number the provider bills.


def _room(*parts, rate: int = 16000, seed: int = 3) -> bytes:
    """Clips (bytes) and seconds of noise floor (float, ~-66 dBFS) concatenated."""
    import numpy as np

    rng = np.random.default_rng(seed)
    out = []
    for part in parts:
        if isinstance(part, bytes):
            out.append(part)
        else:
            n = int(part * rate)
            out.append((rng.standard_normal(n) * 16).astype("<i2").tobytes())
    return b"".join(out)


class _FakeCloud:
    """A ManualTurnBackend that answers every committed turn at once."""

    pace_output_audio = False

    def __init__(self):
        self.calls: list = []
        self.on_event = None

    async def start(self, *, instructions, tools, on_event):
        self.on_event = on_event

    async def push_audio(self, pcm):
        self.calls.append(("push", pcm))

    async def begin_activity(self):
        from nanobot_channel_voice.backend.base import StateHint, VoiceState

        self.calls.append(("begin",))
        await self.on_event(StateHint(VoiceState.CAPTURING))

    async def end_activity(self, *, commit=True):
        from nanobot_channel_voice.backend.base import StateHint, VoiceState

        self.calls.append(("end", commit))
        if commit:
            await self.on_event(StateHint(VoiceState.THINKING))
        await self.on_event(StateHint(VoiceState.IDLE))

    async def park(self):
        self.calls.append(("park",))

    async def barge_in(self, played_ms):
        pass

    async def submit_tool_result(self, call_id, output):
        pass

    async def close(self):
        pass

    def commits(self) -> int:
        return sum(1 for c in self.calls if c == ("end", True))

    def uploaded(self) -> bytes:
        return b"".join(c[1] for c in self.calls if c[0] == "push")


class _AudioClock:
    """Stands in for the gate's ``time`` module: the attention window is wall-clock,
    and the room plays back far faster than real time here."""

    def __init__(self):
        self.now = 1000.0

    def monotonic(self) -> float:
        return self.now


async def _run_gate(mode: str, room: bytes, *, window_s: float, monkeypatch):
    from nanobot_channel_voice.audio.null import NullPlayback
    from nanobot_channel_voice.backend import gated
    from nanobot_channel_voice.backend.audio_sink import AudioSink
    from nanobot_channel_voice.backend.gated import GatedUplink
    from nanobot_channel_voice.config import VoiceConfig
    from nanobot_channel_voice.vad import make_vad
    from nanobot_channel_voice.vad.silero import SileroVad
    from nanobot_channel_voice.wake import make_wake_detector

    # The board config, minus the key: a cloud brain behind local ears.
    cfg = VoiceConfig.model_validate({
        "backend": "gemini",
        "realtime": {"uplink": mode, "idleParkS": 0},
        "vad": {"engine": "silero", "silero": {"modelPath": str(_SILERO)}},
        "wake": {
            "mode": "gate", "phrases": ["hey mycroft"], "engine": "openwakeword",
            "windowS": window_s,
            "openwakeword": {
                "melPath": str(_OWW / "melspectrogram.onnx"),
                "embeddingPath": str(_OWW / "embedding_model.onnx"),
                "modelPath": str(_OWW / "hey_mycroft_v0.1.onnx"),
            },
        },
    })
    frame_ms = cfg.audio.frame_ms
    vad = make_vad(cfg.vad, 16000, frame_ms)
    assert isinstance(vad, SileroVad)
    wake = make_wake_detector(cfg.wake, 16000, frame_ms)
    assert wake is not None
    cloud = _FakeCloud()
    gate = GatedUplink(
        cloud, config=cfg, sink=AudioSink(NullPlayback(), mode="stream"), vad=vad,
        wake_detector=wake, capture_rate=16000, uplink_rate=16000, open_mic=False,
    )

    async def on_event(e):
        pass

    clock = _AudioClock()
    monkeypatch.setattr(gated, "time", clock)
    await gate.start(instructions=None, tools=[], on_event=on_event)
    step = 16000 * 2 * frame_ms // 1000
    try:
        for i in range(0, len(room) - step + 1, step):
            clock.now += frame_ms / 1000.0
            await gate.push_audio(room[i:i + step])
    finally:
        await gate.close()  # releases both detectors
    return cloud, gate._metrics.snapshot()["counters"]


def test_gated_uplink_real_detectors_upload_only_the_summoned_command(monkeypatch):
    """uplink="wake": the phrase opens the window, the command after it goes up, the
    room's silence and the speech after the window lapses do not."""
    _need(_SILERO, _OWW / "melspectrogram.onnx", _OWW / "embedding_model.onnx",
          _OWW / "hey_mycroft_v0.1.onnx", _OWW / "hey_mycroft_test.wav",
          _WHISPER / "test_en.wav")
    with wave.open(str(_OWW / "hey_mycroft_test.wav")) as w:
        phrase = w.readframes(w.getnframes())  # 0.95 s
    with wave.open(str(_WHISPER / "test_en.wav")) as w:
        command = w.readframes(w.getnframes())  # 5.86 s of speech
    # summon, beat, command, a lapse longer than the window, unsummoned speech.
    room = _room(2.0, phrase, 0.6, command, 5.0, command, 1.0)
    cloud, m = asyncio.run(_run_gate("wake", room, window_s=3.0, monkeypatch=monkeypatch))

    assert m.get("wake_hit") == 1
    assert cloud.commits() == 1  # the summoned command; the bare summon commits nothing
    assert m.get("gate_bare_summon") == 1  # the hit adopted the phrase's own utterance
    assert m.get("uplink_utterances", 0) >= 1
    assert m.get("gate_dropped_onsets", 0) >= 1  # the unsummoned repeat
    # Engaged vs wall clock: one command (+ preroll and hangover) out of a ~21 s room.
    ratio = m["uplink_ms"] / m["capture_ms"]
    assert abs(m["capture_ms"] - len(room) * 1000 // 32000) <= 20
    assert 0.2 <= ratio <= 0.45, (m["uplink_ms"], m["capture_ms"])
    assert len(cloud.uploaded()) < len(phrase) + len(command) + 32000  # never both commands


def test_gated_uplink_real_detectors_vad_mode_uploads_every_utterance(monkeypatch):
    """uplink="vad": no summon needed; every endpointed utterance goes up, the silence
    (the bulk of the room) does not."""
    _need(_SILERO, _OWW / "melspectrogram.onnx", _OWW / "embedding_model.onnx",
          _OWW / "hey_mycroft_v0.1.onnx", _OWW / "hey_mycroft_test.wav",
          _WHISPER / "test_en.wav")
    with wave.open(str(_OWW / "hey_mycroft_test.wav")) as w:
        phrase = w.readframes(w.getnframes())
    with wave.open(str(_WHISPER / "test_en.wav")) as w:
        command = w.readframes(w.getnframes())
    room = _room(2.0, phrase, 0.6, command, 5.0, command, 1.0)
    cloud, m = asyncio.run(_run_gate("vad", room, window_s=3.0, monkeypatch=monkeypatch))

    assert cloud.commits() >= 3  # the phrase is speech too here
    assert "wake_hit" not in m  # the detector is not even run in vad mode
    speech_ms = (len(phrase) + 2 * len(command)) * 1000 // 32000
    assert speech_ms <= m["uplink_ms"] <= speech_ms + 3 * 1000  # + preroll/hangover per utterance
    assert m["uplink_ms"] / m["capture_ms"] < 0.8
