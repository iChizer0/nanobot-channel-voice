"""VAD backend selection.

``make_vad`` picks a per-frame detector by ``vad.engine``: ``energy`` (zero-dep default),
``webrtc`` (``[webrtc]`` extra), or ``firered`` (neural DFSMN over ONNX/RKNN,
``[ondevice]`` extra), via the declarative engine table the STT/TTS registries also use
(:mod:`..engines`)."""

from __future__ import annotations

from loguru import logger

from nanobot_channel_voice.config import VadConfig
from nanobot_channel_voice.engines import EngineSpec, describe_build_error, missing_fields
from nanobot_channel_voice.vad.base import Vad
from nanobot_channel_voice.vad.endpointer import Endpointer
from nanobot_channel_voice.vad.energy import EnergyVad
from nanobot_channel_voice.vad.webrtc import WebRtcVad
from nanobot_channel_voice.weights import WeightsError, apply_weights

__all__ = [
    "Vad", "Endpointer", "EnergyVad", "WebRtcVad",
    "flag_lag_ms", "make_vad", "make_turn_analyzer", "resolve_preroll_ms",
]

# Onset lag = time from true speech onset to the first speech-flagged frame, the only
# thing pre-roll must recover. Algorithmic (frame-counted), NOT compute time: slow
# inference delays decisions but never clips audio (the arecord pipe buffers in order).
_FBANK_WARMUP_MS = 25       # analysis window before the first fbank frame
_MODEL_RISE_MARGIN_MS = 80  # sigmoid rise to threshold + safety
_FIRERED_FRAME_MS = 10      # the model's own frame period (vad.firered.smoothFrames unit)


def flag_lag_ms(cfg: VadConfig, frame_ms: int) -> int:
    """Algorithmic decision lag between the acoustic edge and the flag following it
    (fbank window + smoothing settle + rise margin). Bounds both edges: pre-roll
    recovers the onset side, the pause-probe's leak-death window covers release."""
    if cfg.engine == "firered":
        return (
            _FBANK_WARMUP_MS
            + (cfg.firered.smooth_frames - 1) * _FIRERED_FRAME_MS
            + _MODEL_RISE_MARGIN_MS
        )
    return 2 * frame_ms + 40  # energy/webrtc: ~instant per-frame decision + margin


def resolve_preroll_ms(cfg: VadConfig, frame_ms: int) -> int:
    """The configured pre-roll, floored at the VAD's algorithmic onset lag, so a large
    ``vad.firered.smoothFrames`` (or an under-set ``prerollMs``) cannot clip the first
    word."""
    return max(cfg.preroll_ms, flag_lag_ms(cfg, frame_ms))


def _build_webrtc(cfg: VadConfig, sample_rate: int, frame_ms: int) -> Vad:
    return WebRtcVad(sample_rate, frame_ms, cfg.aggressiveness)


def _build_firered(cfg: VadConfig, sample_rate: int, frame_ms: int) -> Vad:
    from nanobot_channel_voice.vad.firered import FireRedVad

    return FireRedVad.from_config(cfg.firered, sample_rate)


ENGINES: dict[str, EngineSpec] = {
    "webrtc": EngineSpec(build=_build_webrtc, modules=("webrtcvad",)),
    "firered": EngineSpec(
        required=(
            ("firered.model_path", "firered.modelPath"),
            ("firered.cmvn_path", "firered.cmvnPath"),
        ),
        build=_build_firered,
        modules=("numpy",),
    ),
}


def _build_smartturn(cfg: VadConfig, sample_rate: int, frame_ms: int):
    from nanobot_channel_voice.vad.smartturn import SmartTurnAnalyzer

    return SmartTurnAnalyzer.from_config(cfg.turn, sample_rate)


TURN_ENGINES: dict[str, EngineSpec] = {
    "smartturn": EngineSpec(
        required=(("turn.model_path", "turn.modelPath"),),
        build=_build_smartturn,
        modules=("numpy", "onnxruntime"),
    ),
}


def make_turn_analyzer(cfg: VadConfig, sample_rate: int, frame_ms: int):
    """Build the configured end-of-turn analyzer (``vad.turn.engine``), or None:
    unavailable means endpointing stays silence-only, never a startup failure."""
    spec = TURN_ENGINES.get(cfg.turn.engine)
    if spec is None:
        return None
    if cfg.turn.consult_ms >= cfg.hangover_ms:
        logger.warning(
            "voice: vad.turn.consultMs ({}) >= vad.hangoverMs ({}); the turn model "
            "would never be consulted: endpointing by silence only",
            cfg.turn.consult_ms, cfg.hangover_ms,
        )
        return None
    try:
        cfg = apply_weights(cfg, "turn")  # vad.turn.weights -> store paths
    except WeightsError as exc:
        logger.warning(
            "voice: {} turn model unavailable ({}); endpointing by silence only",
            cfg.turn.engine, exc,
        )
        return None
    missing = missing_fields(cfg, spec)
    if missing:
        logger.warning(
            "voice: {} turn model needs vad.{}; endpointing by silence only",
            cfg.turn.engine, ", vad.".join(missing),
        )
        return None
    try:
        return spec.build(cfg, sample_rate, frame_ms)
    except Exception as exc:  # noqa: BLE001 - missing deps / models / wrong rate
        logger.warning(
            "voice: {} turn model unavailable ({}); endpointing by silence only",
            cfg.turn.engine, describe_build_error(exc),
        )
        return None


def make_vad(cfg: VadConfig, sample_rate: int, frame_ms: int) -> Vad:
    """Build the configured VAD, falling back to energy when unavailable."""
    spec = ENGINES.get(cfg.engine)
    if spec is not None:  # anything not in the table (i.e. "energy") skips to the floor
        try:
            cfg = apply_weights(cfg, cfg.engine)  # vad.<engine>.weights -> store paths
        except WeightsError as exc:
            logger.warning("voice: {} VAD unavailable ({}); using energy VAD", cfg.engine, exc)
            return EnergyVad(cfg.energy_threshold)
        missing = missing_fields(cfg, spec)
        if missing:
            logger.warning(
                "voice: {} VAD needs vad.{}; using energy VAD",
                cfg.engine, ", vad.".join(missing),
            )
        else:
            try:
                return spec.build(cfg, sample_rate, frame_ms)
            except Exception as exc:  # noqa: BLE001 - missing deps / models / wrong rate
                logger.warning(
                    "voice: {} VAD unavailable ({}); using energy VAD",
                    cfg.engine, describe_build_error(exc),
                )
    return EnergyVad(cfg.energy_threshold)
