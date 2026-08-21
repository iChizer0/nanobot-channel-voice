"""On-device streaming Zipformer transducer ASR (ONNX), the sherpa-onnx export.

A ``streaming = True`` engine: frames are decoded DURING speech (one encoder chunk per
320 ms of audio), so at the endpoint only the final tail remains and STT latency stops
existing as a pipeline stage. One model per language pair; the bilingual zh-en
artifact is the reference target.

Verified against the pinned artifact
(``sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20``, 2026-07-29):

* encoder ``x [N, T=39, 80]`` fbank frames + ~35 ``cached_*`` state tensors (int64
  ``cached_len_*``, float32 the rest) -> ``encoder_out [N, T', 512]`` +
  ``new_cached_*``. Metadata ``T = 39``, ``decode_chunk_len = 32``: each call consumes
  39 frames and ADVANCES by 32, re-reading 7 frames of right context.
* decoder (stateless) ``y [N, 2]`` int64 -> ``decoder_out [N, 512]`` (metadata
  ``context_size = 2``); joiner ``(encoder_out, decoder_out) -> logit [N, 6254]``.
* ``tokens.txt`` is ``<token> <id>`` with ``<blk> 0``; BPE ``▁`` marks word starts,
  CJK tokens are bare chars.

State plumbing is GENERIC: zero states come from the encoder's declared inputs and
each ``new_X`` output feeds the next call's ``X`` input, so a re-export with different
cache geometry still runs. For `.rknn` (no introspection surface) that contract rides
in the RKNN exporter's ``meta.json`` sidecar: state specs and output names IN DECLARED
ORDER (RKNN is fed positionally), plus per-state ``cached_len`` increments — the
rv1126b converter cannot emit the int64 ``new_cached_len_*`` assembly (toolkit 2.4.0
SIGABRT), so the port drops those outputs and the host advances the len states itself.
icefall front-end contract: fbank 80 (25/10 ms, dither 0) over waveform normalized to
[-1, 1], NOT the int16 scale SenseVoice uses; the streaming path requires 16 kHz
capture.
"""

from __future__ import annotations

import asyncio
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from loguru import logger

from nanobot_channel_voice.config import ZipformerSttConfig
from nanobot_channel_voice.ondevice.runtime import OnDeviceModel
from nanobot_channel_voice.stt.base import (
    SttAdapter,
    SttStream,
    pcm_to_float_mono,
    read_token_table,
)

SAMPLE_RATE = 16000
_NUM_MEL_BINS = 80
_BLANK_ID = 0
# Trailing zeros fed at finish so the last real audio clears the 7-frame lookahead and
# fills a final full chunk.
_FINISH_PAD_S = 0.66


class _ZipformerStream(SttStream):
    """One utterance's decode state (fbank + encoder caches + hypothesis); every
    consumer (live capture, a detached endpoint ``finish()`` thread, the batch path)
    builds its OWN handle per the :class:`SttStream` contract."""

    __slots__ = ("_engine", "fbank", "consumed", "states", "hyp", "decoder_out")

    def __init__(self, engine: ZipformerOnDeviceStt):
        self._engine = engine
        self.fbank = engine._knf.OnlineFbank(engine._opts)
        self.consumed = 0  # fbank frames already fed to the encoder (chunk starts)
        self.states = {
            name: np.zeros(
                [1 if not isinstance(d, int) else d for d in shape],
                dtype=np.int64 if "int64" in typ else np.float32,
            )
            for name, shape, typ in engine._state_specs
        }
        self.hyp: list[int] = []
        y = np.array([[_BLANK_ID] * engine._context], dtype=np.int64)
        self.decoder_out = engine._decoder.run([("y", y)])[0]

    def accept(self, pcm: bytes) -> None:
        """Frames must already be 16 kHz: the streaming path never resamples."""
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        self.fbank.accept_waveform(SAMPLE_RATE, samples)
        self._engine._decode_ready(self)

    def partial(self) -> str:
        """Live hypothesis for the min-words early-confirm gate: just a token join,
        since the decode already ran in accept()."""
        return self._engine._decode_text(self)

    def finish(self) -> str:
        pad = np.zeros(int(_FINISH_PAD_S * SAMPLE_RATE), dtype=np.float32)
        self.fbank.accept_waveform(SAMPLE_RATE, pad)
        self.fbank.input_finished()
        self._engine._decode_ready(self)
        return self._engine._decode_text(self)


class ZipformerOnDeviceStt(SttAdapter):
    decoder_family = "transducer"
    streaming = True

    def __init__(
        self,
        *,
        encoder: OnDeviceModel,
        decoder: OnDeviceModel,
        joiner: OnDeviceModel,
        tokens: dict[int, str],
        chunk_t: int,
        chunk_shift: int,
        context_size: int,
        state_specs: list | None = None,
        out_names: list | None = None,
        state_increments: dict | None = None,
        feedback_transpose: tuple | None = None,
    ):
        import kaldi_native_fbank as knf  # lazy: [ondevice] extra

        self._knf = knf
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0.0
        opts.mel_opts.num_bins = _NUM_MEL_BINS
        self._opts = opts

        self._encoder = encoder
        self._decoder = decoder
        self._joiner = joiner
        self._tokens = tokens
        self._chunk_t = chunk_t
        self._chunk_shift = chunk_shift
        self._context = context_size
        # Encoder state contract: from the model itself (ONNX) or the sidecar (RKNN).
        self._state_specs = state_specs or [
            (name, shape, typ) for name, shape, typ in encoder.input_specs() if name != "x"
        ]
        self._out_names = out_names or encoder.output_names()
        self._increments = state_increments or {}
        # .rknn: 4D caches return NCHW; the sidecar permutation restores the declared feed.
        self._feedback_transpose = feedback_transpose
        if not self._state_specs or not self._out_names:
            raise RuntimeError(
                "zipformer needs the encoder cache contract: ONNX introspection or "
                "the exporter's meta.json sidecar (stt.zipformer.metaPath)"
            )
        missing = [n for n, _, _ in self._state_specs
                   if f"new_{n}" not in self._out_names and n not in self._increments]
        if missing:
            raise RuntimeError(
                f"zipformer states with no feedback path (no new_* output, no "
                f"sidecar increment): {missing} -- meta.json/encoder mismatch"
            )
        self._log = logger.bind(component="stt-zipformer")
        # Contract probe doubling as warmup: one zero chunk (25 ms window + chunk_t
        # 10 ms hops) through encoder+joiner, so a stale export or meta.json mismatch
        # raises here — loud registry degrade, not _transcribe_sync's blanket except.
        self.stream_start().accept(bytes(2 * (25 + 10 * self._chunk_t) * (SAMPLE_RATE // 1000)))

    @classmethod
    def from_config(cls, cfg: ZipformerSttConfig) -> ZipformerOnDeviceStt:
        z = cfg
        # Sidecar first: a bad pairing must fail BEFORE the expensive model loads.
        side = cls._load_sidecar(z.meta_path) if str(z.encoder_path).endswith(".rknn") else {}
        kw = dict(
            core_mask=z.core_mask, target=z.target, device_id=z.device_id,
            providers=z.execution_providers, provider_options=z.provider_options,
            # Shared by the frame hop's chunk decode AND batch transcribe(); the frame
            # budget wins (sherpa-onnx ships this model single-threaded too).
            intra_op_threads=1,
        )
        with ExitStack() as models:  # any failure below releases every loaded model
            encoder, decoder, joiner = (
                models.enter_context(OnDeviceModel(path, **kw))  # type: ignore[arg-type]
                for path in (z.encoder_path, z.decoder_path, z.joiner_path)
            )
            if not side:
                md, dmd = encoder.metadata(), decoder.metadata()
                side = dict(
                    chunk_t=int(md.get("T", 39)),
                    chunk_shift=int(md.get("decode_chunk_len", 32)),
                    context_size=int(dmd.get("context_size", 2)),
                )
            adapter = cls(
                encoder=encoder, decoder=decoder, joiner=joiner,
                tokens=read_token_table(z.tokens_path),  # type: ignore[arg-type]
                **side,
            )
            models.pop_all()  # success: the adapter owns the models now
            return adapter

    @staticmethod
    def _load_sidecar(path: str | None) -> dict:
        """The exporter's ``meta.json`` -> constructor kwargs: the encoder contract
        an ``.rknn`` cannot introspect (see the module docstring)."""
        if not path:
            raise RuntimeError(
                "zipformer .rknn needs stt.zipformer.metaPath (the exporter's meta.json sidecar)"
            )
        try:
            side = json.loads(Path(path).read_text())
            if side["encoder_inputs"][0][0] != "x":
                raise ValueError("encoder_inputs must declare 'x' first (RKNN is fed positionally)")
            return dict(
                chunk_t=int(side.get("T", 39)),
                chunk_shift=int(side.get("decode_chunk_len", 32)),
                context_size=int(side.get("context_size", 2)),
                state_specs=[(n, s, t) for n, s, t in side["encoder_inputs"] if n != "x"],
                out_names=side["encoder_outputs"],
                state_increments={k: int(v) for k, v in side["state_increments"].items()},
                feedback_transpose=(
                    tuple(side["state_feedback_transpose"])
                    if "state_feedback_transpose" in side else None
                ),
            )
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(f"bad zipformer meta sidecar {path}: {exc!r}") from exc

    def stream_start(self) -> _ZipformerStream:
        return _ZipformerStream(self)

    async def warmup(self) -> None:
        """One dummy decode so the first real utterance pays no cold-start (ORT
        arena allocation, RKNN core spin-up). The batch path builds its own
        stream handle, so this warms the exact per-frame graphs the live capture
        uses — measured negligible on CPU, insurance on the NPU."""
        await self.transcribe(b"\x00" * SAMPLE_RATE, SAMPLE_RATE)  # 0.5 s of silence

    def release(self) -> None:
        for model in (self._encoder, self._decoder, self._joiner):
            model.release()

    # ---- batch path (same engine, whole utterance) --------------------------

    async def transcribe(self, pcm: bytes, sample_rate: int) -> str:
        if not pcm:
            return ""
        return await asyncio.to_thread(self._transcribe_sync, pcm, sample_rate)

    def _transcribe_sync(self, pcm: bytes, sample_rate: int) -> str:
        try:
            stream = self.stream_start()  # own handle; any live stream is untouched
            audio = pcm_to_float_mono(pcm, sample_rate, SAMPLE_RATE)
            stream.fbank.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(audio))
            self._decode_ready(stream)
            return stream.finish()
        except Exception as exc:  # noqa: BLE001 - never let STT crash the capture loop
            self._log.warning("on-device zipformer STT failed: {}", exc)
            return ""

    # ---- internals (all state via the handle argument) ----------------------

    def _decode_ready(self, s: _ZipformerStream) -> None:
        """Run the encoder over every complete chunk available, greedy-decoding its
        frames as they come: this is where streaming actually happens."""
        while s.fbank.num_frames_ready - s.consumed >= self._chunk_t:
            chunk = np.vstack([
                s.fbank.get_frame(s.consumed + i) for i in range(self._chunk_t)
            ]).astype(np.float32)
            outs = self._encoder.run(
                [("x", chunk[None, ...])] + [(n, v) for n, v in s.states.items()]
            )
            named = dict(zip(self._out_names, outs, strict=True))
            for name in s.states:
                new = named.get(f"new_{name}")
                if new is not None:
                    if (
                        self._feedback_transpose is not None
                        and np.asarray(new).ndim == len(self._feedback_transpose)
                    ):
                        new = np.ascontiguousarray(np.transpose(new, self._feedback_transpose))
                    s.states[name] = new
                else:
                    # .rknn drops the int64 new_cached_len_* outputs: advance host-side.
                    s.states[name] = s.states[name] + self._increments[name]
            # Commit the cursor exactly here: shift 32 of the 39 frames (7 of right
            # context re-read). Earlier, a raising encoder would skip the chunk with
            # stale caches; later, a raising joiner would re-feed already-advanced ones.
            s.consumed += self._chunk_shift
            self._greedy(s, np.asarray(named["encoder_out"])[0])

    def _greedy(self, s: _ZipformerStream, encoder_out: np.ndarray) -> None:
        """Stateless-transducer greedy search: at most one symbol per frame, decoder
        re-run only when a symbol is emitted."""
        for t in range(encoder_out.shape[0]):
            logit = self._joiner.run(
                [("encoder_out", encoder_out[t : t + 1]), ("decoder_out", s.decoder_out)]
            )[0]
            tok = int(np.asarray(logit)[0].argmax())
            if tok == _BLANK_ID:
                continue
            s.hyp.append(tok)
            y = np.array([s.hyp[-self._context :]], dtype=np.int64)
            if y.shape[1] < self._context:  # pad early context with blanks
                y = np.pad(y, ((0, 0), (self._context - y.shape[1], 0)))
            s.decoder_out = self._decoder.run([("y", y)])[0]

    def _decode_text(self, s: _ZipformerStream) -> str:
        pieces = [self._tokens.get(i, "") for i in s.hyp]
        text = "".join(p for p in pieces if not (p.startswith("<") and p.endswith(">")))
        return text.replace("▁", " ").strip()

