"""openWakeWord-format acoustic wake word over ONNX/RKNN (``wake.engine="openwakeword"``).

Three original upstream artifacts, chained host-side exactly as the upstream
runtime chains them: ``melspectrogram.onnx`` (raw int16-valued float samples in,
mel frames out, then the fixed ``x/10 + 2`` transform), the shared
``embedding_model.onnx`` (76 mel frames -> one 96-dim embedding, stride 8 = one
embedding per 80 ms chunk), and a per-phrase classifier head (a window of
embeddings -> one sigmoid score). livekit-wakeword heads follow the same
backbone contract and load through the same path. 16 kHz only; scoring cadence
is one decision per 1280-sample chunk (~12.5 Hz), cheap enough for the frame
hop even on A55-class CPUs.
"""

from __future__ import annotations

import time
from contextlib import ExitStack

import numpy as np
from loguru import logger

from nanobot_channel_voice.aio import Throttle
from nanobot_channel_voice.config import OpenWakeWordConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.wake.base import WakeDetector

# Upstream pipeline geometry (openWakeWord AudioFeatures): 80 ms steps at 16 kHz;
# the mel model sees each chunk with 3 hops of left context so edge frames get a
# full 25 ms analysis window; 8 new mel frames per chunk (160-sample hop).
_RATE = 16000
_CHUNK = 1280
_MEL_CONTEXT = 480
_MEL_BINS = 32
_MEL_FRAMES_PER_CHUNK = 8
_EMB_WINDOW = 76        # mel frames per embedding
_EMB_DIM = 96
_HEAD_WINDOW = 16       # embeddings per score; overridden by the head's declared shape
_PRIME_SECONDS = 4      # upstream prefill: the embedding window starts as REAL noise embeddings
_FAIL_LOG_EVERY_S = 30.0


class OpenWakeWord(WakeDetector):
    def __init__(
        self,
        *,
        mel_path: str,
        embedding_path: str,
        model_path: str,
        sample_rate: int,
        threshold: float = 0.5,
        refractory_s: float = 2.0,
        core_mask: str = "auto",
        target: str | None = None,
        device_id: str | None = None,
        providers: list | None = None,
        provider_options: list | None = None,
    ):
        if sample_rate != _RATE:
            raise ValueError(
                f"openWakeWord requires 16 kHz audio, got {sample_rate}"
            )
        # _validate() raising is the EXPECTED path make_wake_detector catches to
        # degrade to the transcript tier, so the ExitStack must release every
        # claimed session on it.
        with ExitStack() as models:
            kw = dict(
                core_mask=core_mask, target=target, device_id=device_id,
                providers=providers, provider_options=provider_options,
                intra_op_threads=1,  # runs per chunk inside the hop: never fan out
            )
            self._mel = models.enter_context(OnDeviceModel(mel_path, **kw))
            self._emb = models.enter_context(OnDeviceModel(embedding_path, **kw))
            self._head = models.enter_context(OnDeviceModel(model_path, **kw))
            self._mel_in = self._first_input(self._mel, "input")
            self._emb_in = self._first_input(self._emb, "input_1")
            self._head_in = self._first_input(self._head, "input_1")
            head_shape = self._head.input_shape(self._head_in)
            self._head_window = (
                int(head_shape[1])
                if head_shape is not None and len(head_shape) >= 2
                and isinstance(head_shape[1], int) and head_shape[1] > 0
                else _HEAD_WINDOW
            )
            self._threshold = threshold
            self._refractory_s = refractory_s
            self._fail_throttle = Throttle(_FAIL_LOG_EVERY_S)
            self._log = logger.bind(component="wake-oww")
            self._head_width = 0  # flattened head output size, proven by the prime run
            self._embs_prime = np.zeros((self._head_window, _EMB_DIM), dtype=np.float32)
            self.reset()
            self._prime_and_validate()
            models.pop_all()  # success: the adapter owns the models now

    @classmethod
    def from_config(cls, cfg: OpenWakeWordConfig, sample_rate: int) -> OpenWakeWord:
        return cls(
            mel_path=cfg.mel_path,              # type: ignore[arg-type]
            embedding_path=cfg.embedding_path,  # type: ignore[arg-type]
            model_path=cfg.model_path,          # type: ignore[arg-type]
            sample_rate=sample_rate,
            threshold=cfg.threshold,
            refractory_s=cfg.refractory_s,
            core_mask=cfg.core_mask,
            target=cfg.target,
            device_id=cfg.device_id,
            providers=cfg.execution_providers,
            provider_options=cfg.provider_options,
        )

    @staticmethod
    def _first_input(model: OnDeviceModel, fallback: str) -> str:
        specs = model.input_specs()
        return specs[0][0] if specs else fallback  # RKNN: no introspection, positional anyway

    def release(self) -> None:
        """Give the sessions/NPU contexts back. Idempotent; refcount-GC does NOT
        free an RKNN context."""
        for model in (self._mel, self._emb, self._head):
            model.release()

    def reset(self) -> None:
        self._pending = np.empty(0, dtype=np.float32)
        # A full-context zero tail makes EVERY mel call exactly
        # [1, context+chunk]: uniform shapes are what a fixed-graph RKNN port
        # needs, and the edge effect is one padded analysis window.
        self._raw_tail = np.zeros(_MEL_CONTEXT, dtype=np.float32)
        self._mels = np.empty((0, _MEL_BINS), dtype=np.float32)
        # Restore the primed embedding window: all-zero rows are outside the
        # embedding model's output distribution and can score arbitrarily on
        # some heads (upstream prefills with noise embeddings for the same
        # reason).
        self._embs = self._embs_prime.copy()
        self._armed = True          # re-armed by a sub-threshold score
        self._last_hit = float("-inf")
        self.last_score = None

    def _prime_and_validate(self) -> None:
        """One pass of noise audio through the whole pipeline at construction:
        proves all three models' I/O contracts (an incompatible export fails
        HERE and the registry degrades to the transcript tier instead of
        failing per-frame in the hop) and captures a REAL-embedding window for
        ``reset()`` to restore — upstream parity (its AudioFeatures prefills
        the feature buffer from 4 s of random audio)."""
        rng = np.random.default_rng(0)  # deterministic: the same prime every start
        noise = rng.integers(-1000, 1000, _PRIME_SECONDS * _RATE).astype(np.float32)
        score = None
        try:
            for i in range(0, noise.size - _CHUNK + 1, _CHUNK):
                s = self._score_chunk(noise[i:i + _CHUNK])
                if s is not None:
                    score = s
        except Exception as exc:  # noqa: BLE001 - re-raise as a catchable construction error
            raise RuntimeError(
                f"openWakeWord pipeline rejected the expected shapes (mel "
                f"[1,{_MEL_CONTEXT + _CHUNK}]->[...,{_MEL_BINS}], embedding "
                f"[1,{_EMB_WINDOW},{_MEL_BINS},1]->[...,{_EMB_DIM}], head "
                f"[1,{self._head_window},{_EMB_DIM}]): {exc}"
            ) from exc
        if self._head_width != 1:
            raise RuntimeError(
                f"openWakeWord head produced {self._head_width} outputs; only "
                "single-score per-phrase heads are supported (multi-class "
                "exports would silently score the wrong class)."
            )
        # Loose band: int8 dequantization may jitter a real sigmoid slightly
        # outside [0, 1]; a logit/swapped output lands far outside it.
        if score is None or not (-0.05 <= score <= 1.05):
            raise RuntimeError(
                f"openWakeWord head score {score} is not sigmoid-like "
                "-- check the model export (output order/shape)."
            )
        self._embs_prime = self._embs.copy()
        self.reset()  # the probe must not leave its own streaming state behind

    def push(self, frame: bytes) -> bool:
        try:
            samples = np.frombuffer(frame, dtype="<i2").astype(np.float32)
            # Upstream contract: raw int16-VALUED floats, not normalized.
            self._pending = np.concatenate((self._pending, samples))
            hit = False
            while self._pending.size >= _CHUNK:
                chunk, self._pending = self._pending[:_CHUNK], self._pending[_CHUNK:]
                score = self._score_chunk(chunk)
                if score is None:
                    continue
                self.last_score = score
                if self._debounce(score):
                    hit = True
                    # Everything still pending lies AFTER the hit chunk's end.
                    self.last_hit_back_bytes = 2 * self._pending.size
            return hit
        except Exception as exc:  # noqa: BLE001 - never let wake detection crash the hop
            if self._fail_throttle.ready():
                self._log.warning(
                    "wake detector failed: {}; treating audio as no-hit (throttled)", exc
                )
            return False

    def _debounce(self, score: float, now: float | None = None) -> bool:
        """Threshold + re-arm hysteresis + refractory: one hit per crossing, no
        retriggers while the score rides above the threshold, and a floor on
        hit spacing (the same phrase echoing in a room must not double-fire)."""
        t = time.monotonic() if now is None else now
        if score >= self._threshold:
            hit = self._armed and (t - self._last_hit) >= self._refractory_s
            self._armed = False
            if hit:
                self._last_hit = t
            return hit
        self._armed = True
        return False

    def _score_chunk(self, chunk: np.ndarray) -> float | None:
        """One 80 ms step through mel -> embedding -> head. None until the mel
        buffer covers an embedding window (start of stream only)."""
        mel_in = np.concatenate((self._raw_tail, chunk)).reshape(1, -1)
        (mel_out,) = self._mel.run([(self._mel_in, mel_in)])
        frames = np.asarray(mel_out, dtype=np.float32).reshape(-1, _MEL_BINS)
        frames = frames[-_MEL_FRAMES_PER_CHUNK:] / 10.0 + 2.0  # upstream transform
        self._raw_tail = chunk[-_MEL_CONTEXT:]
        self._mels = np.concatenate((self._mels, frames))[-_EMB_WINDOW:]
        if len(self._mels) < _EMB_WINDOW:
            return None
        emb_in = np.ascontiguousarray(
            self._mels.reshape(1, _EMB_WINDOW, _MEL_BINS, 1)
        )
        (emb_out,) = self._emb.run([(self._emb_in, emb_in)])
        emb = np.asarray(emb_out, dtype=np.float32).reshape(-1)[-_EMB_DIM:]
        self._embs = np.concatenate((self._embs[1:], emb.reshape(1, _EMB_DIM)))
        head_in = np.ascontiguousarray(
            self._embs.reshape(1, self._head_window, _EMB_DIM)
        )
        out = self._head.run([(self._head_in, head_in)])
        flat = np.asarray(out[0]).reshape(-1)
        self._head_width = flat.size  # proven == 1 by the construction prime
        return float(flat[0])  # upstream indexing: prediction[0][0][0]
