"""Wake-word gate backends (``wake.mode`` != "off").

Two tiers: the transcript tier (:class:`WakePhrase`, active whenever ``wake.phrases``
is set) and an optional acoustic tier picked by ``wake.engine``. An unavailable
acoustic engine is never a startup failure — gating continues on transcripts alone.
"""

from __future__ import annotations

import json
from pathlib import Path

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
    # "text" is deliberately absent: the transcript tier needs no model.
    "openwakeword": EngineSpec(
        required=(
            ("openwakeword.embedding_path", "openwakeword.embeddingPath"),
            ("openwakeword.model_path", "openwakeword.modelPath"),
        ),
        required_any=(  # the ONNX mel graph, or the filterbank for NPU packages
            (("openwakeword.mel_path", "openwakeword.melPath"),),
            (("openwakeword.mel_filters_path", "openwakeword.melFiltersPath"),),
        ),
        build=_build_openwakeword,
        modules=("numpy",),
    ),
}


def _meta_advisories(cfg: WakeConfig) -> None:
    """Package-sidecar checks, never fatal: a head disagreeing with the config still
    summons but misbehaves subtly, so say why at startup."""
    oww = cfg.openwakeword
    if not oww.meta_path:
        return
    try:
        meta = json.loads(Path(oww.meta_path).read_text(encoding="utf-8"))
        if not isinstance(meta, dict):
            raise ValueError("not a JSON object")
    except Exception as exc:  # noqa: BLE001 - provenance sidecar, never fatal
        logger.warning("voice: wake package meta '{}' unreadable ({})", oww.meta_path, exc)
        return
    phrase = meta.get("phrase")
    if not isinstance(phrase, str):
        phrase = None
    if phrase and cfg.phrases and phrase.casefold() not in (p.casefold() for p in cfg.phrases):
        logger.warning(
            "voice: the fetched wake head detects '{}' but wake.phrases is {}; "
            "hits still summon, but the transcript tier and the wake-phrase "
            "strip listen for different words (the phrase can leak into "
            "published text)",
            phrase, cfg.phrases,
        )
    target = meta.get("target")
    if not isinstance(target, str):
        target = None
    if target and (oww.embedding_path or "").endswith(".rknn") and target != oww.target:
        logger.warning(
            "voice: wake package targets '{}' but wake.openwakeword.target is '{}'",
            target, oww.target,
        )


_DEGRADE = "wake gating continues on transcripts only"


def make_wake_detector(
    cfg: WakeConfig, sample_rate: int, frame_ms: int
) -> WakeDetector | None:
    """The configured acoustic wake detector, or None (gate off, text tier selected,
    or engine unavailable — never a startup failure)."""
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
    if cfg.engine == "openwakeword":
        _meta_advisories(cfg)
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
