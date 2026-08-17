"""TTS adapter selection: a declarative :data:`ENGINES` table plus one fallback policy
(missing config or a failed import/construct warns and degrades to ``system``), so the
channel always speaks. ``openai``/``openai_compat`` (cloud or any OpenAI-compatible
local server) is the default; ``system`` (espeak-ng/``say``) is the always-available
floor.
"""

from __future__ import annotations

from loguru import logger

from nanobot_channel_voice.config import TtsConfig, resolve_openai_key
from nanobot_channel_voice.engines import EngineSpec, describe_build_error, missing_fields
from nanobot_channel_voice.tts.base import TtsAdapter
from nanobot_channel_voice.weights import WeightsError, apply_weights

__all__ = ["TtsAdapter", "make_tts"]


def _system_fallback(cfg: TtsConfig) -> TtsAdapter:
    from nanobot_channel_voice.tts.system import SystemTtsAdapter

    return SystemTtsAdapter(language=cfg.language)


def _build_openai(cfg: TtsConfig) -> TtsAdapter:
    from nanobot_channel_voice.tts.openai_compat import OpenAITtsAdapter

    return OpenAITtsAdapter(
        api_key=resolve_openai_key(cfg.api_key),
        api_base=cfg.api_base,
        model=cfg.model,
        voice=cfg.voice,
        audio_format=cfg.audio_format,
        timeout_s=cfg.timeout_s,
        pcm_sample_rate=cfg.pcm_sample_rate,
    )


def _build_mms(cfg: TtsConfig) -> TtsAdapter:
    from nanobot_channel_voice.tts.mms import MmsTtsAdapter

    return MmsTtsAdapter.from_config(cfg.mms)


def _build_supertonic(cfg: TtsConfig) -> TtsAdapter:
    from nanobot_channel_voice.tts.supertonic import SupertonicTtsAdapter

    return SupertonicTtsAdapter.from_config(cfg.supertonic)


def _build_matcha(cfg: TtsConfig) -> TtsAdapter:
    from nanobot_channel_voice.tts.matcha import MatchaTtsAdapter

    return MatchaTtsAdapter.from_config(cfg.matcha)


ENGINES: dict[str, EngineSpec] = {
    "openai": EngineSpec(build=_build_openai),
    "openai_compat": EngineSpec(build=_build_openai),
    "system": EngineSpec(build=_system_fallback),
    "mms": EngineSpec(
        required=(
            ("mms.encoder_path", "mms.encoderPath"),
            ("mms.decoder_path", "mms.decoderPath"),
        ),
        build=_build_mms,
        modules=("numpy",),
    ),
    "supertonic": EngineSpec(
        required=(
            ("supertonic.text_encoder_path", "supertonic.textEncoderPath"),
            ("supertonic.duration_predictor_path", "supertonic.durationPredictorPath"),
            ("supertonic.vector_estimator_path", "supertonic.vectorEstimatorPath"),
            ("supertonic.vocoder_path", "supertonic.vocoderPath"),
            ("supertonic.tts_json_path", "supertonic.ttsJsonPath"),
            ("supertonic.unicode_indexer_path", "supertonic.unicodeIndexerPath"),
            ("supertonic.voice_style_path", "supertonic.voiceStylePath"),
        ),
        build=_build_supertonic,
        modules=("numpy",),
    ),
    "matcha": EngineSpec(
        # Only the acoustic graph is universally required: an official embedded-vocoder
        # export needs nothing else; from_config names what a given export still wants.
        required=(("matcha.acoustic_model_path", "matcha.acousticModelPath"),),
        build=_build_matcha,
        modules=("numpy",),
    ),
}


def make_tts(cfg: TtsConfig) -> TtsAdapter | None:
    """Build the configured TTS adapter, or None when TTS is disabled.

    However the adapter was built (fallbacks included), ``tts.language`` settles its
    ``spoken_language``: an engine that knows its own language wins (a conflicting
    config claim is logged and ignored), while one that cannot know (an ``openai_compat``
    server, MMS with a supplied ``vocabPath``) takes the operator's declaration.
    Downstream (agent voice-context block, speakability warning) reads only the settled
    value."""
    if not cfg.enabled:
        return None
    adapter = _build(cfg)
    if adapter is not None:
        declared = getattr(adapter, "spoken_language", None)
        if declared is None:
            adapter.spoken_language = cfg.language
        elif cfg.language and cfg.language != declared:
            logger.warning(
                "voice: tts.language='{}' conflicts with the engine's own language "
                "'{}'; the engine wins: fix the engine block (or drop tts.language)",
                cfg.language, declared,
            )
    return adapter


def _build(cfg: TtsConfig) -> TtsAdapter | None:
    spec = ENGINES.get(cfg.provider)
    if spec is None:
        logger.warning("voice: unknown TTS provider '{}'; using system fallback", cfg.provider)
        return _system_fallback(cfg)

    try:
        cfg = apply_weights(cfg, cfg.provider)  # tts.<provider>.weights -> store paths
    except WeightsError as exc:
        logger.warning(
            "voice: TTS provider '{}' unavailable ({}); using system fallback",
            cfg.provider, exc,
        )
        return _system_fallback(cfg)

    missing = missing_fields(cfg, spec)
    if missing:
        logger.warning(
            "voice: TTS provider '{}' needs tts.{}; using system fallback",
            cfg.provider,
            ", tts.".join(missing),
        )
        return _system_fallback(cfg)
    try:
        return spec.build(cfg)
    except Exception as exc:  # noqa: BLE001 - missing extra / model / vocab
        logger.warning(
            "voice: TTS provider '{}' unavailable ({}); using system fallback",
            cfg.provider,
            describe_build_error(exc),
        )
        return _system_fallback(cfg)
