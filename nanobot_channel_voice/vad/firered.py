"""FireRedVAD: on-device neural VAD backend (ONNX / RKNN).

A DFSMN streaming VAD (beats webrtc/silero on noisy audio), run through the shared
:class:`OnDeviceModel` so ``.onnx`` (CPU) and ``.rknn`` (NPU) need no extra code. Each
``is_speech(frame)`` runs the new audio through fbank+CMVN, then the streaming model
**one 10 ms frame at a time** (``feat[1,1,80]`` + carried cache), and smooths the
sigmoid probability into a binary decision. Per-frame (``T=1``) inference is purely
causal (the model skips lookahead) and keeps every tensor a fixed shape, so the same
path converts cleanly to RKNN; the cache is carried in Python (``caches_in`` ->
``caches_out``) and zeroed on :meth:`reset`.
"""

from __future__ import annotations

from collections import deque
from contextlib import ExitStack

import numpy as np
from loguru import logger

from nanobot_channel_voice.aio import Throttle
from nanobot_channel_voice.audio.pcm import pcm_rms
from nanobot_channel_voice.config import FireRedVadConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.vad.base import Vad
from nanobot_channel_voice.vad.features import FbankCmvn

# Reference FireRedVAD stream export: 8 FSMN caches of [1, 128, 19]. Used when the
# shape can't be read (RKNN); ONNX is auto-detected from the session.
_CACHE_SHAPE = (8, 1, 128, 19)
_FEAT_NAME = "feat"
_CACHE_NAME = "caches_in"
_FEAT_DIM = 80              # fbank bins fed to the model (feat[1, 1, 80])
_FAIL_LOG_EVERY_S = 30.0    # throttle the per-frame failure warning


class FireRedVad(Vad):
    heavy = True  # numpy fbank + inference per frame: run off the event loop

    def __init__(
        self,
        *,
        model_path: str,
        cmvn_path: str,
        threshold: float = 0.5,
        smooth_frames: int = 5,
        min_volume: float = 0.0,
        core_mask: str = "auto",
        target: str | None = None,
        device_id: str | None = None,
        providers: list | None = None,
        provider_options: list | None = None,
    ):
        # _validate() raising is the EXPECTED path make_vad catches to fall back to
        # energy, so the ExitStack must release the claimed NPU/ORT session on it.
        with ExitStack() as models:
            self._model = models.enter_context(OnDeviceModel(
                model_path, core_mask=core_mask, target=target, device_id=device_id,
                providers=providers, provider_options=provider_options,
                intra_op_threads=1,  # runs per frame inside the hop: never fan out
            ))
            self._front = FbankCmvn(cmvn_path)
            self._threshold = threshold
            self._smooth_n = max(1, smooth_frames)
            self._min_volume = min_volume
            self._cache_shape = self._model.input_shape(_CACHE_NAME) or _CACHE_SHAPE
            self._fail_throttle = Throttle(_FAIL_LOG_EVERY_S)
            self._log = logger.bind(component="vad-firered")
            self.reset()
            self._validate()
            models.pop_all()  # success: the adapter owns the model now

    @classmethod
    def from_config(cls, cfg: FireRedVadConfig, sample_rate: int) -> FireRedVad:
        if sample_rate != 16000:
            raise ValueError(f"FireRedVAD requires 16 kHz audio, got {sample_rate}")
        return cls(
            model_path=cfg.model_path,        # type: ignore[arg-type]
            cmvn_path=cfg.cmvn_path,          # type: ignore[arg-type]
            threshold=cfg.threshold,
            smooth_frames=cfg.smooth_frames,
            min_volume=cfg.min_volume,
            core_mask=cfg.core_mask,
            target=cfg.target,
            device_id=cfg.device_id,
            providers=cfg.execution_providers,
            provider_options=cfg.provider_options,
        )

    def release(self) -> None:
        """Give the NPU context / ORT session back. Idempotent; refcount-GC does NOT
        free an RKNN context, so an in-process channel restart would otherwise exhaust
        the NPU cores."""
        self._model.release()

    def reset(self) -> None:
        self._caches = np.zeros(self._cache_shape, dtype=np.float32)
        self._front.reset()
        self._window: deque[float] = deque()
        self._window_sum = 0.0
        self._last_speech = False

    def _validate(self) -> None:
        """Prove the model's I/O contract once, at construction, on a zeroed frame, so
        that a loadable-but-incompatible export (wrong input/output name, shape or
        order) fails HERE and ``make_vad`` falls back to energy, rather than failing
        per-frame in :meth:`is_speech`, flooding the log and leaving the channel deaf.
        """
        feat0 = np.zeros((1, 1, _FEAT_DIM), dtype=np.float32)
        try:
            _, caches = self._model.run([(_FEAT_NAME, feat0), (_CACHE_NAME, self._caches)])
        except Exception as exc:  # noqa: BLE001 - re-raise as a catchable construction error
            raise RuntimeError(
                f"FireRedVAD model rejected the expected inputs "
                f"('{_FEAT_NAME}'[1,1,{_FEAT_DIM}] + '{_CACHE_NAME}'{tuple(self._cache_shape)}): {exc}"
            ) from exc
        out_shape = np.asarray(caches).shape
        expected = tuple(self._cache_shape)
        if all(isinstance(d, int) and d > 0 for d in expected) and out_shape != expected:
            raise RuntimeError(
                f"FireRedVAD cache output shape {out_shape} != expected {expected} "
                "-- check the model export (output name/order)."
            )

    def _gated(self, speech: bool, frame: bytes) -> bool:
        """Loudness AND'd with the model (Pipecat's gate): distant TV speech is real
        speech to the model but too quiet to be the user. Applied to EVERY returned
        decision (including held state on sub-window frames) never to the model
        run itself, so the FSMN cache stays coherent."""
        if speech and self._min_volume > 0.0:
            return pcm_rms(frame) >= self._min_volume
        return speech

    def is_speech(self, frame: bytes) -> bool:
        try:
            samples = np.frombuffer(frame, dtype="<i2")
            feats = self._front.accept(samples)  # (n, 80), n = new 10 ms frames
            if feats.shape[0] == 0:
                # Warmup / sub-frame boundary: hold last MODEL state, still gated.
                return self._gated(self._last_speech, frame)
            for i in range(feats.shape[0]):
                smoothed = self._smooth(self._infer_one(feats[i]))
                self._last_speech = smoothed >= self._threshold
            return self._gated(self._last_speech, frame)
        except Exception as exc:  # noqa: BLE001 - never let VAD crash the capture loop
            if self._fail_throttle.ready():
                self._log.warning(
                    "FireRedVAD failed: {}; treating frames as non-speech (throttled)", exc
                )
            return False

    def _infer_one(self, feat_row: np.ndarray) -> float:
        feat = np.ascontiguousarray(feat_row.reshape(1, 1, -1), dtype=np.float32)  # (1, 1, 80)
        probs, caches = self._model.run([(_FEAT_NAME, feat), (_CACHE_NAME, self._caches)])
        self._caches = np.ascontiguousarray(caches, dtype=np.float32)
        return float(np.asarray(probs).reshape(-1)[-1])

    def _smooth(self, prob: float) -> float:
        if self._smooth_n <= 1:
            return prob
        self._window.append(prob)
        self._window_sum += prob
        if len(self._window) > self._smooth_n:
            self._window_sum -= self._window.popleft()
        return self._window_sum / len(self._window)
