"""On-device SenseVoice-Small ASR (ONNX), the sherpa-onnx export.

Non-autoregressive CTC (one encoder pass, greedy per-frame argmax): ~5-15x faster than
Whisper on CPU and immune to hallucination loops (REPORT-asr-tts-model-survey.md 2.2);
one model covers zh / yue / en / ja / ko plus emotion/event tags.

I/O contract, verified against the pinned artifact
(``sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17``, 2026-07-29): inputs
``x [N,T,560]`` float32 (80-mel fbank x LFR window 7) plus per-row int32 ``x_length``,
``language``, ``text_norm``; output ``logits [N,T,25055]`` (CTC over the SentencePiece
vocab). The whole FRONT-END CONTRACT rides in the ONNX metadata (``neg_mean`` /
``inv_stddev`` 560-dim CMVN, ``lfr_window_size/shift``, the language / itn ids,
``normalize_samples=0`` = int16-scale waveform), so there is no side-file to mismatch;
``.rknn`` exports have no metadata surface and are rejected. The model prepends its 4
query embeddings INSIDE the graph, so this side supplies only real audio frames;
``tokens.txt`` is ``<token> <id>``, columns reversed vs the Whisper vocab files.
"""

from __future__ import annotations

import asyncio
import re
from contextlib import ExitStack

import numpy as np
from loguru import logger

from nanobot_channel_voice.config import SenseVoiceSttConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.stt.base import SttAdapter, pcm_to_float_mono, read_token_table

SAMPLE_RATE = 16000
_NUM_MEL_BINS = 80
_BLANK_ID = 0
# Rich-transcription tags (<|zh|><|NEUTRAL|><|Speech|><|woitn|>) arrive as ordinary CTC
# tokens: stripped from the transcript, surfaced on .last_tags.
_TAG_RE = re.compile(r"<\|[^|]*\|>")


def _apply_lfr(feats: np.ndarray, m: int, n: int) -> np.ndarray:
    """FunASR low-frame-rate stacking, (T, 80) -> (T', 80*m): window ``m`` frames,
    shift ``n``, first-frame left padding, last-frame right padding."""
    t = feats.shape[0]
    t_lfr = int(np.ceil(t / n))
    left = np.tile(feats[0], ((m - 1) // 2, 1))
    feats = np.vstack((left, feats))
    t = feats.shape[0]
    rows = []
    for i in range(t_lfr):
        if m <= t - i * n:
            rows.append(feats[i * n : i * n + m].reshape(-1))
        else:
            row = feats[i * n :].reshape(-1)
            pad = np.tile(feats[-1], m - (t - i * n))
            rows.append(np.concatenate((row, pad)))
    return np.vstack(rows).astype(np.float32)


class _Frontend:
    """FunASR's WavFrontend: fbank (hamming, 25/10 ms, int16-scale waveform) + LFR +
    CMVN, the CMVN stats taken from the model metadata."""

    def __init__(self, neg_mean: np.ndarray, inv_stddev: np.ndarray, lfr_m: int, lfr_n: int):
        import kaldi_native_fbank as knf  # lazy: [ondevice] extra

        self._knf = knf
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0.0
        opts.frame_opts.window_type = "hamming"
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = _NUM_MEL_BINS
        self._opts = opts
        self._neg_mean = neg_mean
        self._inv_stddev = inv_stddev
        self._lfr_m = lfr_m
        self._lfr_n = lfr_n

    def compute(self, samples: np.ndarray) -> np.ndarray:
        """int16-scale float samples -> CMVN'd LFR features (T', 560)."""
        fbank = self._knf.OnlineFbank(self._opts)
        fbank.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(samples, dtype=np.float32))
        fbank.input_finished()
        n = fbank.num_frames_ready
        if n == 0:
            return np.zeros((0, _NUM_MEL_BINS * self._lfr_m), dtype=np.float32)
        feats = np.vstack([fbank.get_frame(i) for i in range(n)]).astype(np.float32)
        feats = _apply_lfr(feats, self._lfr_m, self._lfr_n)
        return (feats + self._neg_mean) * self._inv_stddev


def ctc_greedy(ids) -> list[int]:
    """Collapse repeats, drop blanks: plain CTC greedy over per-frame argmax."""
    out: list[int] = []
    prev = -1
    for i in ids:
        i = int(i)
        if i != prev and i != _BLANK_ID:
            out.append(i)
        prev = i
    return out


class SenseVoiceOnDeviceStt(SttAdapter):
    def __init__(
        self,
        *,
        model: OnDeviceModel,
        tokens: dict[int, str],
        frontend: _Frontend,
        language_id: int,
        text_norm_id: int,
    ):
        self._model = model
        self._tokens = tokens
        self._frontend = frontend
        self._language_id = language_id
        self._text_norm_id = text_norm_id
        self.last_tags = ""  # stripped rich-transcription tags, for observers
        self._log = logger.bind(component="stt-sensevoice")

    @classmethod
    def from_config(cls, cfg: SenseVoiceSttConfig) -> SenseVoiceOnDeviceStt:
        sv = cfg
        with ExitStack() as models:  # any failure below releases the loaded model
            model = models.enter_context(OnDeviceModel(
                sv.model_path,  # type: ignore[arg-type]
                core_mask=sv.core_mask, target=sv.target, device_id=sv.device_id,
                providers=sv.execution_providers, provider_options=sv.provider_options,
            ))
            md = model.metadata()
            if "neg_mean" not in md or "inv_stddev" not in md:
                raise RuntimeError(
                    "SenseVoice model carries no front-end metadata (neg_mean/inv_stddev); "
                    "use the sherpa-onnx .onnx export (.rknn is not supported yet)"
                )
            lang_key = f"lang_{sv.language.lower()}"
            if lang_key not in md:
                known = sorted(k.removeprefix("lang_") for k in md if k.startswith("lang_"))
                raise RuntimeError(
                    f"SenseVoice language '{sv.language}' not in the model (has: {', '.join(known)})"
                )
            neg_mean = np.fromiter(
                (float(v) for v in md["neg_mean"].split(",")), dtype=np.float32
            )
            inv_stddev = np.fromiter(
                (float(v) for v in md["inv_stddev"].split(",")), dtype=np.float32
            )
            frontend = _Frontend(
                neg_mean, inv_stddev,
                lfr_m=int(md.get("lfr_window_size", 7)),
                lfr_n=int(md.get("lfr_window_shift", 6)),
            )
            adapter = cls(
                model=model,
                tokens=read_token_table(sv.tokens_path),  # type: ignore[arg-type]
                frontend=frontend,
                language_id=int(md[lang_key]),
                text_norm_id=int(md["with_itn" if sv.use_itn else "without_itn"]),
            )
            models.pop_all()  # success: the adapter owns the model now
            return adapter

    def release(self) -> None:
        self._model.release()

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if not pcm:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, pcm, sample_rate)

    async def warmup(self) -> None:
        await self.transcribe(b"\x00" * SAMPLE_RATE, SAMPLE_RATE)  # 0.5 s of silence

    def _transcribe_sync(self, pcm: bytes, sample_rate: int) -> str:
        try:
            # normalize_samples=0 in the export: the model wants int16-SCALE values.
            audio = pcm_to_float_mono(pcm, sample_rate, SAMPLE_RATE) * 32768.0
            feats = self._frontend.compute(audio)
            if feats.shape[0] == 0:
                return ""
            logits = self._model.run([
                ("x", feats[None, ...]),
                ("x_length", np.array([feats.shape[0]], dtype=np.int32)),
                ("language", np.array([self._language_id], dtype=np.int32)),
                ("text_norm", np.array([self._text_norm_id], dtype=np.int32)),
            ])[0]
            ids = ctc_greedy(np.asarray(logits)[0].argmax(axis=-1))
            text = "".join(self._tokens.get(i, "") for i in ids)
            self.last_tags = "".join(_TAG_RE.findall(text))
            return _TAG_RE.sub("", text).replace("▁", " ").strip()
        except Exception as exc:  # noqa: BLE001 - never let STT crash the capture loop
            self._log.warning("on-device SenseVoice STT failed: {}", exc)
            return ""
