"""Smart Turn v3: audio-native end-of-turn scoring (ONNX / RKNN).

A Whisper-Tiny-encoder classifier (BSD-2 weights, ~8 MB int8) answering "was that pause
terminal?". The endpointer consults it once per silence run at ``vad.turn.consultMs``;
COMPLETE closes the utterance immediately, INCOMPLETE lets the silence timer run to
``hangoverMs`` — the timer is the hard bound, the model only the decision. One
inference per PAUSE, not per frame.
"""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np
from loguru import logger

from nanobot_channel_voice.config import TurnConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.vad._whisper_mel import SAMPLE_RATE, WINDOW_SAMPLES, log_mel_features

_INPUT_NAME = "input_features"


class SmartTurnAnalyzer:
    """``assess(pcm) -> bool`` (turn complete?) over S16_LE mono 16 kHz bytes."""

    def __init__(
        self,
        *,
        model_path: str,
        threshold: float = 0.5,
        core_mask: str = "auto",
        target: str | None = None,
        device_id: str | None = None,
        providers: list | None = None,
        provider_options: list | None = None,
    ):
        with ExitStack() as models:
            self._model = models.enter_context(OnDeviceModel(
                model_path, core_mask=core_mask, target=target, device_id=device_id,
                providers=providers, provider_options=provider_options,
                intra_op_threads=1,  # one core per consult: never contend with the frame path
                profile="bulk",
            ))
            self._threshold = threshold
            self.window_bytes = 2 * WINDOW_SAMPLES  # consult-snapshot cap for the endpointer
            self.last_probability = 0.0
            self._log = logger.bind(component="vad-smartturn")
            # Contract probe + warmup: an incompatible export raises here and the
            # factory falls back to silence-only.
            self._score(np.zeros(WINDOW_SAMPLES, dtype=np.float32))
            models.pop_all()  # success: the adapter owns the model now

    @classmethod
    def from_config(cls, cfg: TurnConfig, sample_rate: int) -> SmartTurnAnalyzer:
        if sample_rate != SAMPLE_RATE:
            raise ValueError(f"smartturn requires 16 kHz audio, got {sample_rate}")
        return cls(
            model_path=cfg.model_path,  # type: ignore[arg-type]
            threshold=cfg.threshold,
            core_mask=cfg.core_mask,
            target=cfg.target,
            device_id=cfg.device_id,
            providers=cfg.execution_providers,
            provider_options=cfg.provider_options,
        )

    def _score(self, audio: np.ndarray) -> float:
        feats = log_mel_features(audio)[None, ...]
        out = self._model.run([(_INPUT_NAME, feats)])
        return float(np.asarray(out[0]).reshape(-1)[0])  # sigmoid probability

    def assess(self, pcm: bytes) -> bool:
        """Score the utterance-so-far (trailing pause included). Keeps the last 8 s;
        shorter audio is padded at the START so speech sits at the window end, as
        trained."""
        if len(pcm) % 2:
            pcm = pcm[:-1]  # tail-trim, like every other S16 helper: keep alignment
        if len(pcm) > 2 * WINDOW_SAMPLES:
            pcm = pcm[-2 * WINDOW_SAMPLES:]  # slice BEFORE converting: dropped samples never scale
        audio = np.multiply(
            np.frombuffer(pcm, dtype="<i2"), 1.0 / 32768.0, dtype=np.float32
        )
        if audio.size < WINDOW_SAMPLES:
            audio = np.pad(audio, (WINDOW_SAMPLES - audio.size, 0))
        self.last_probability = self._score(audio)
        return self.last_probability > self._threshold

    def release(self) -> None:
        self._model.release()
