"""Wake-word gate backends (``wake.mode`` != "off").

Two tiers share the gate: the transcript tier (:class:`WakePhrase`, always
active when ``wake.phrases`` is set — zero models, every language the STT
covers) and an optional acoustic tier picked by ``wake.engine``
(``"openwakeword"`` = openWakeWord/livekit-wakeword-format ONNX/RKNN, one
model per phrase). Like the turn-analyzer registry, an unavailable acoustic
engine is never a startup failure: ``make_wake_detector`` warns and returns
None, and gating continues on transcripts alone.
"""

from __future__ import annotations

from loguru import logger

from nanobot_channel_voice.config import WakeConfig
from nanobot_channel_voice.engines import EngineSpec, describe_build_error, missing_fields
from nanobot_channel_voice.wake.base import WakeDetector
from nanobot_channel_voice.wake.phrase import WakePhrase
from nanobot_channel_voice.weights import WeightsError, apply_weights

__all__ = ["ENGINES", "WakeDetector", "WakePhrase", "make_wake_detector"]


def _build_openwakeword(cfg: WakeConfig, sample_rate: int, frame_ms: int) -> WakeDetector:
    from nanobot_channel_voice.wake.openwakeword import OpenWakeWord

    return OpenWakeWord.from_config(cfg.openwakeword, sample_rate)


ENGINES: dict[str, EngineSpec] = {
    # "text" is deliberately absent: the transcript tier lives in the backend
    # and needs no model.
    "openwakeword": EngineSpec(
        required=(
            ("openwakeword.mel_path", "openwakeword.melPath"),
            ("openwakeword.embedding_path", "openwakeword.embeddingPath"),
            ("openwakeword.model_path", "openwakeword.modelPath"),
        ),
        build=_build_openwakeword,
        modules=("numpy",),
    ),
}

_DEGRADE = "wake gating continues on transcripts only"


def make_wake_detector(
    cfg: WakeConfig, sample_rate: int, frame_ms: int
) -> WakeDetector | None:
    """Build the configured acoustic wake detector, or None (gate disabled, the
    text tier selected, or the engine unavailable — never a startup failure)."""
    if cfg.mode == "off":
        return None
    spec = ENGINES.get(cfg.engine)
    if spec is None:
        return None
    try:
        cfg = apply_weights(cfg, cfg.engine)  # wake.<engine>.weights -> store paths
    except WeightsError as exc:
        logger.warning(
            "voice: {} wake detector unavailable ({}); {}", cfg.engine, exc, _DEGRADE
        )
        return None
    missing = missing_fields(cfg, spec)
    if missing:
        logger.warning(
            "voice: {} wake detector needs wake.{}; {}",
            cfg.engine, ", wake.".join(missing), _DEGRADE,
        )
        return None
    try:
        return spec.build(cfg, sample_rate, frame_ms)
    except Exception as exc:  # noqa: BLE001 - missing deps / models / wrong rate
        logger.warning(
            "voice: {} wake detector unavailable ({}); {}",
            cfg.engine, describe_build_error(exc), _DEGRADE,
        )
        return None
