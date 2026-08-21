"""On-device SenseVoice-Small ASR over the ORIGINAL FunASR export (ONNX / RKNN).

Non-autoregressive CTC (one encoder pass, greedy per-frame argmax): ~5-15x faster than
Whisper on CPU and immune to hallucination loops;
one model covers zh / yue / en / ja / ko plus emotion/event tags.

Both artifacts are converted from the upstream weights by the project's exporter, which
writes a ``frontend.json`` sidecar (560-dim CMVN, LFR window/shift, the language / itn
id tables — FunASR runtime constants living in NO model artifact — and the .rknn's
``feats_len``) plus a ``<token> <id>`` ``tokens.txt`` (columns reversed vs the Whisper
vocab files). Dispatch is on the file extension:

- ``.onnx`` — the official dynamic export (upstream ``export_meta.py``): inputs
  ``speech [N,T,560]`` float32 (80-mel fbank x LFR window 7, int16-scale waveform)
  plus int32 ``speech_lengths`` / ``language`` / ``textnorm``; outputs
  ``ctc_logits [N,4+T,25055]`` + ``encoder_out_lens``.
- ``.rknn`` — the static mask-input port: ``speech [1,L,560]`` zero-padded to the
  export window, a multiplicative ``mask [1,1,1,L+4]`` (1 = 4 query + valid frames),
  ``language``, ``textnorm``; ``ctc_logits [1,L+4,25055]`` decoded over frames
  ``0..4+n_valid`` (the padded tail's logits are meaningless).

The model prepends its 4 query embeddings INSIDE the graph (those frames yield the
rich-transcription tags), so this side supplies only real audio frames.
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import ExitStack
from pathlib import Path

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
    CMVN, the CMVN stats taken from the frontend.json sidecar."""

    def __init__(self, neg_mean: np.ndarray, inv_stddev: np.ndarray, lfr_m: int, lfr_n: int):
        dim = _NUM_MEL_BINS * lfr_m
        if neg_mean.size != dim or inv_stddev.size != dim:
            raise RuntimeError(
                f"SenseVoice CMVN stats must be {dim}-dim (80 mel x LFR {lfr_m}), "
                f"got {neg_mean.size}/{inv_stddev.size}"
            )
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
    decoder_family = "ctc"
    # Dynamic-.onnx policy bound (O(T^2) SAN-M activations, short-form training audio:
    # past ~30 s memory and accuracy slide); a static .rknn window tightens it per instance.
    max_decode_ms = 30_000

    def __init__(
        self,
        *,
        model: OnDeviceModel,
        tokens: dict[int, str],
        frontend: _Frontend,
        language_id: int,
        text_norm_id: int,
        feats_len: int | None = None,
    ):
        self._model = model
        self._tokens = tokens
        self._frontend = frontend
        self._language_id = language_id
        self._text_norm_id = text_norm_id
        # None = dynamic-length .onnx; an int = the static .rknn window (frames).
        self._feats_len = feats_len
        if feats_len is not None:
            # One frame = lfr_n fbank hops of 10 ms.
            self.max_decode_ms = min(self.max_decode_ms, feats_len * frontend._lfr_n * 10)
        self.last_tags = ""  # stripped rich-transcription tags, for observers
        self._log = logger.bind(component="stt-sensevoice")

    @staticmethod
    def _load_sidecar(path: str) -> tuple[_Frontend, dict[str, int], dict[str, int], int | None]:
        """``frontend.json`` -> (front end, language ids, textnorm ids, feats_len)."""
        try:
            side = json.loads(Path(path).read_text())
            frontend = _Frontend(
                np.fromiter((float(v) for v in side["neg_mean"].split(",")), np.float32),
                np.fromiter((float(v) for v in side["inv_stddev"].split(",")), np.float32),
                lfr_m=int(side.get("lfr_window_size", 7)),
                lfr_n=int(side.get("lfr_window_shift", 6)),
            )
            langs = {str(k).lower(): int(v) for k, v in side["languages"].items()}
            norms = {
                "withitn": int(side["textnorm"]["withitn"]),
                "woitn": int(side["textnorm"]["woitn"]),
            }
            feats_len = int(side["feats_len"]) if "feats_len" in side else None
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"bad SenseVoice frontend sidecar {path}: {exc!r}") from exc
        return frontend, langs, norms, feats_len

    @classmethod
    def from_config(cls, cfg: SenseVoiceSttConfig) -> SenseVoiceOnDeviceStt:
        sv = cfg
        # Sidecar first: a bad pairing must fail BEFORE the expensive model load.
        frontend, langs, norms, feats_len = cls._load_sidecar(sv.frontend_path)  # type: ignore[arg-type]
        if sv.language.lower() not in langs:
            raise RuntimeError(
                f"SenseVoice language '{sv.language}' not in the model (has: {', '.join(sorted(langs))})"
            )
        if not str(sv.model_path).endswith(".rknn"):
            feats_len = None  # the window is .rknn-only; the .onnx graph is dynamic
        elif feats_len is None:
            raise RuntimeError(f"SenseVoice .rknn sidecar {sv.frontend_path} lacks feats_len")
        with ExitStack() as models:  # any failure below releases the loaded model
            model = models.enter_context(OnDeviceModel(
                sv.model_path,  # type: ignore[arg-type]
                core_mask=sv.core_mask, target=sv.target, device_id=sv.device_id,
                providers=sv.execution_providers, provider_options=sv.provider_options,
            ))
            adapter = cls(
                model=model,
                tokens=read_token_table(sv.tokens_path),  # type: ignore[arg-type]
                frontend=frontend,
                language_id=langs[sv.language.lower()],
                text_norm_id=norms["withitn" if sv.use_itn else "woitn"],
                feats_len=feats_len,
            )
            # Contract probe doubling as warmup: a stale artifact (sherpa input names,
            # wrong feats_len) raises here, where the registry degrades loudly — via
            # _decode, not _transcribe_sync, whose blanket except would swallow it.
            adapter._decode(adapter._frontend.compute(np.zeros(SAMPLE_RATE, dtype=np.float32)))
            models.pop_all()  # success: the adapter owns the model now
            return adapter

    def release(self) -> None:
        self._model.release()

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if not pcm:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, pcm, sample_rate)

    def _transcribe_sync(self, pcm: bytes, sample_rate: int) -> str:
        try:
            # FunASR's WavFrontend feeds kaldi fbank int16-SCALE values.
            audio = pcm_to_float_mono(pcm, sample_rate, SAMPLE_RATE) * 32768.0
            feats = self._frontend.compute(audio)
            if feats.shape[0] == 0:
                return ""
            text = "".join(self._tokens.get(i, "") for i in self._decode(feats))
            self.last_tags = "".join(_TAG_RE.findall(text))
            return _TAG_RE.sub("", text).replace("▁", " ").strip()
        except Exception as exc:  # noqa: BLE001 - never let STT crash the capture loop
            self._log.warning("on-device SenseVoice STT failed: {}", exc)
            return ""

    def _decode(self, feats: np.ndarray) -> list[int]:
        return self._decode_dynamic(feats) if self._feats_len is None else self._decode_static(feats)

    def _decode_dynamic(self, feats: np.ndarray) -> list[int]:
        """The .onnx contract (official FunASR export): dynamic T, lengths by value."""
        logits = self._model.run([
            ("speech", feats[None, ...]),
            ("speech_lengths", np.array([feats.shape[0]], dtype=np.int32)),
            ("language", np.array([self._language_id], dtype=np.int32)),
            ("textnorm", np.array([self._text_norm_id], dtype=np.int32)),
        ])[0]
        return ctc_greedy(np.asarray(logits)[0].argmax(axis=-1))

    def _decode_static(self, feats: np.ndarray) -> list[int]:
        """The .rknn static contract: zero-pad to the export window, multiplicative
        mask over the 4 query + valid frames, decode only those frames (the padded
        tail's logits are meaningless)."""
        window = self._feats_len
        assert window is not None
        n_valid = feats.shape[0]
        if n_valid > window:  # direct callers only: transcribe_chunked cuts to the window
            self._log.warning(
                "utterance has {} feature frames, over the {}-frame export window; "
                "truncating the tail", n_valid, window,
            )
            n_valid = window
        speech = np.zeros((1, window, feats.shape[1]), dtype=np.float32)
        speech[0, :n_valid] = feats[:n_valid]
        mask = np.zeros((1, 1, 1, window + 4), dtype=np.float32)
        mask[0, 0, 0, : 4 + n_valid] = 1.0
        logits = self._model.run([
            ("speech", speech),
            ("mask", mask),
            ("language", np.array([self._language_id], dtype=np.int32)),
            ("textnorm", np.array([self._text_norm_id], dtype=np.int32)),
        ])[0]
        return ctc_greedy(np.asarray(logits)[0, : 4 + n_valid].argmax(axis=-1))
