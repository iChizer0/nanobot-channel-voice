"""STT adapter selection.

Every fallback (unknown provider, failed weights resolve, missing config field, failed
import/construct) warns and delegates to nanobot's transcription layer, so registering
an engine is one :data:`ENGINES` spec plus one factory.
"""

from __future__ import annotations

from loguru import logger

from nanobot_channel_voice.config import SttConfig
from nanobot_channel_voice.engines import EngineSpec, describe_build_error, missing_fields
from nanobot_channel_voice.stt.base import SttAdapter, write_temp_wav
from nanobot_channel_voice.weights import WeightsError, apply_weights

__all__ = ["SttAdapter", "make_stt", "write_temp_wav"]


def _build_whisper(cfg: SttConfig) -> SttAdapter:
    from nanobot_channel_voice.stt.whisper import WhisperOnDeviceStt

    return WhisperOnDeviceStt.from_config(cfg.whisper)


def _build_sensevoice(cfg: SttConfig) -> SttAdapter:
    from nanobot_channel_voice.stt.sensevoice import SenseVoiceOnDeviceStt

    return SenseVoiceOnDeviceStt.from_config(cfg.sensevoice)


def _build_zipformer(cfg: SttConfig) -> SttAdapter:
    from nanobot_channel_voice.stt.zipformer import ZipformerOnDeviceStt

    return ZipformerOnDeviceStt.from_config(cfg.zipformer)


ENGINES: dict[str, EngineSpec] = {
    "whisper": EngineSpec(
        required=(
            ("whisper.encoder_path", "whisper.encoderPath"),
            ("whisper.decoder_path", "whisper.decoderPath"),
            ("whisper.vocab_path", "whisper.vocabPath"),
            ("whisper.mel_filters_path", "whisper.melFiltersPath"),
        ),
        build=_build_whisper,
        modules=("numpy",),
    ),
    "sensevoice": EngineSpec(
        required=(
            ("sensevoice.model_path", "sensevoice.modelPath"),
            ("sensevoice.tokens_path", "sensevoice.tokensPath"),
        ),
        build=_build_sensevoice,
        modules=("numpy", "kaldi_native_fbank"),
    ),
    "zipformer": EngineSpec(
        required=(
            ("zipformer.encoder_path", "zipformer.encoderPath"),
            ("zipformer.decoder_path", "zipformer.decoderPath"),
            ("zipformer.joiner_path", "zipformer.joinerPath"),
            ("zipformer.tokens_path", "zipformer.tokensPath"),
        ),
        build=_build_zipformer,
        modules=("numpy", "kaldi_native_fbank"),
    ),
}


def make_stt(cfg: SttConfig) -> SttAdapter | None:
    """Build the on-device STT adapter, or None to use nanobot's transcription."""
    if cfg.provider == "nanobot":
        return None

    spec = ENGINES.get(cfg.provider)
    if spec is None:
        logger.warning(
            "voice: unknown STT provider '{}'; delegating to nanobot transcription", cfg.provider
        )
        return None

    try:
        cfg = apply_weights(cfg, cfg.provider)  # stt.<provider>.weights -> store paths
    except WeightsError as exc:
        logger.warning(
            "voice: on-device {} STT unavailable ({}); delegating to nanobot transcription",
            cfg.provider, exc,
        )
        return None

    missing = missing_fields(cfg, spec)
    if missing:
        logger.warning(
            "voice: on-device {} STT needs stt.{}; delegating to nanobot transcription",
            cfg.provider,
            ", stt.".join(missing),
        )
        return None
    try:
        return spec.build(cfg)
    except Exception as exc:  # noqa: BLE001 - missing deps / models / assets
        logger.warning(
            "voice: on-device {} STT unavailable ({}); delegating to nanobot transcription",
            cfg.provider,
            describe_build_error(exc),
        )
        return None
