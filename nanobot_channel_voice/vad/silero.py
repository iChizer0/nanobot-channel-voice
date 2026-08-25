"""Silero VAD: on-device neural VAD backend (ONNX / RKNN); v5 and v6 exports share the
I/O contract this adapter speaks.

Raw waveform in (the STFT lives inside the graph), one sigmoid per 32 ms window.
``is_speech(frame)`` buffers samples into 512-sample windows (256 @ 8 kHz), prepends
the model's required past context (64/32 samples, carried host-side — omitting it pins
every probability near 0.001), runs with the carried ``state[2,1,128]``, and applies an
enter/exit hysteresis; frames completing no window hold the last decision. The scalar
int64 ``sr`` is passed only when the session declares it (a stripped ``.rknn`` port
won't), and outputs are read as ``(output, stateN)``, in that order.
"""

from __future__ import annotations

from contextlib import ExitStack

import numpy as np
from loguru import logger

from nanobot_channel_voice.aio import Throttle
from nanobot_channel_voice.config import SileroVadConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.vad.base import Vad

# Upstream contract: 32 ms windows, 1/8 window of past context, LSTM state (2,1,128).
# Shape is read from the session (ONNX); the constant covers RKNN.
_WINDOW_BY_RATE = {16000: 512, 8000: 256}
_STATE_SHAPE = (2, 1, 128)
_INPUT_NAME = "input"
_STATE_NAME = "state"
_SR_NAME = "sr"
_DEFAULT_HYSTERESIS = 0.15  # upstream VADIterator: neg_threshold = threshold - 0.15
_FAIL_LOG_EVERY_S = 30.0    # throttle the per-frame failure warning


class SileroVad(Vad):
    heavy = True  # neural inference per window: run off the event loop

    def __init__(
        self,
        *,
        model_path: str,
        sample_rate: int,
        threshold: float = 0.5,
        neg_threshold: float | None = None,
        min_volume: float = 0.0,
        core_mask: str = "auto",
        target: str | None = None,
        device_id: str | None = None,
        providers: list | None = None,
        provider_options: list | None = None,
    ):
        window = _WINDOW_BY_RATE.get(sample_rate)
        if window is None:
            raise ValueError(f"Silero VAD requires 8 or 16 kHz audio, got {sample_rate}")
        # _validate() raising is the EXPECTED make_vad fallback path, so the ExitStack
        # must release the claimed NPU/ORT session on it.
        with ExitStack() as models:
            self._model = models.enter_context(OnDeviceModel(
                model_path, core_mask=core_mask, target=target, device_id=device_id,
                providers=providers, provider_options=provider_options,
                intra_op_threads=1,  # runs per window inside the hop: never fan out
            ))
            self._window = window
            self._context_n = window // 8
            self._sr = np.array(sample_rate, dtype=np.int64)
            # Stripped single-rate ports have no sr input (RKNN introspection: []).
            self._has_sr = any(n == _SR_NAME for n, _, _ in self._model.input_specs())
            self._threshold = threshold
            self._neg_threshold = (
                neg_threshold if neg_threshold is not None
                else max(threshold - _DEFAULT_HYSTERESIS, 0.01)
            )
            self._min_volume = min_volume
            self._state_shape = self._model.input_shape(_STATE_NAME) or _STATE_SHAPE
            self._fail_throttle = Throttle(_FAIL_LOG_EVERY_S)
            self._log = logger.bind(component="vad-silero")
            self.reset()
            self._validate()
            models.pop_all()  # success: the adapter owns the model now

    @classmethod
    def from_config(cls, cfg: SileroVadConfig, sample_rate: int) -> SileroVad:
        return cls(
            model_path=cfg.model_path,        # type: ignore[arg-type]
            sample_rate=sample_rate,
            threshold=cfg.threshold,
            neg_threshold=cfg.neg_threshold,
            min_volume=cfg.min_volume,
            core_mask=cfg.core_mask,
            target=cfg.target,
            device_id=cfg.device_id,
            providers=cfg.execution_providers,
            provider_options=cfg.provider_options,
        )

    def release(self) -> None:
        """Idempotent; refcount-GC does NOT free an RKNN context."""
        self._model.release()

    def reset(self) -> None:
        self._lstm_state = np.zeros(self._state_shape, dtype=np.float32)
        self._context = np.zeros(self._context_n, dtype=np.float32)
        self._pending = np.empty(0, dtype=np.float32)
        self._last_speech = False
        self.last_prob = None

    def _validate(self) -> None:
        """Prove the I/O contract HERE, so an incompatible export degrades to energy
        instead of failing per-frame in :meth:`is_speech` (a deaf channel)."""
        try:
            prob = self._infer_one(np.zeros(self._window, dtype=np.float32))
        except Exception as exc:  # noqa: BLE001 - re-raise as a catchable construction error
            raise RuntimeError(
                f"Silero VAD model rejected the expected inputs "
                f"('{_INPUT_NAME}'[1,{self._context_n + self._window}] + "
                f"'{_STATE_NAME}'{tuple(self._state_shape)}"
                f"{' + scalar sr' if self._has_sr else ''}): {exc}"
            ) from exc
        state_shape = self._lstm_state.shape
        expected = tuple(self._state_shape)
        if all(isinstance(d, int) and d > 0 for d in expected) and state_shape != expected:
            raise RuntimeError(
                f"Silero VAD state output shape {state_shape} != expected {expected} "
                "-- check the model export (output name/order)."
            )
        # Loose band: int8 dequantization jitters a real sigmoid slightly outside
        # [0, 1]; a swapped/logit output lands far outside. NaN fails the compare.
        if not (-0.05 <= prob <= 1.05):
            raise RuntimeError(
                f"Silero VAD probability {prob} is not sigmoid-like "
                "-- check the model export (output name/order)."
            )
        self.reset()  # the probe must not leave warm-up state behind


    def is_speech(self, frame: bytes) -> bool:
        try:
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float32)
            samples /= 32768.0
            self._pending = np.concatenate((self._pending, samples))
            while self._pending.size >= self._window:
                window, self._pending = (
                    self._pending[: self._window], self._pending[self._window:],
                )
                prob = self._infer_one(window)
                self.last_prob = prob
                # Hysteresis: between the thresholds the decision holds, so a mid-word
                # dip never flickers the flag.
                if prob >= self._threshold:
                    self._last_speech = True
                elif prob < self._neg_threshold:
                    self._last_speech = False
            return self._gated(self._last_speech, frame)
        except Exception as exc:  # noqa: BLE001 - never let VAD crash the capture loop
            if self._fail_throttle.ready():
                self._log.warning(
                    "Silero VAD failed: {}; treating frames as non-speech (throttled)", exc
                )
            return False

    def _infer_one(self, window: np.ndarray) -> float:
        x = np.concatenate((self._context, window)).reshape(1, -1)  # fresh, contiguous f32
        inputs = [(_INPUT_NAME, x), (_STATE_NAME, self._lstm_state)]
        if self._has_sr:
            inputs.append((_SR_NAME, self._sr))
        prob, state = self._model.run(inputs)
        self._lstm_state = np.ascontiguousarray(state, dtype=np.float32)
        self._context = window[-self._context_n:]
        return float(np.asarray(prob).reshape(-1)[-1])
