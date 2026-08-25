"""Kaldi fbank + global CMVN front-end for the FireRedVAD backend.

FireRedVAD's pipeline exactly: 80-bin Kaldi fbank (25/10 ms, ``snip_edges``,
``dither=0``) on the **int16-scale** waveform (sample values, NOT normalized), then
global CMVN ``(x - mean) * istd``. Stats come from ``cmvn.ark`` (a Kaldi binary
``DM ``/``FM `` matrix) via a tiny struct reader, so no ``kaldiio`` dependency. One
persistent ``OnlineFbank`` spans calls, so arbitrary PCM chunks yield the same 10 ms
frames as one continuous pass.
"""

from __future__ import annotations

import struct

import numpy as np

SAMPLE_RATE = 16000
NUM_MEL_BINS = 80
_VAR_FLOOR = 1e-20


def _read_kaldi_matrix(path: str) -> np.ndarray:
    """Read a single Kaldi binary float/double matrix into a numpy array."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:2] != b"\x00B":
        raise ValueError(f"{path}: not a Kaldi binary file (missing \\0B marker)")
    pos = 2
    token = data[pos:pos + 3]
    pos += 3
    if token == b"DM ":
        dtype = np.float64
    elif token == b"FM ":
        dtype = np.float32
    else:
        raise ValueError(f"{path}: unexpected matrix token {token!r} (want 'DM '/'FM ')")
    if data[pos] != 4:
        raise ValueError(f"{path}: unexpected int size marker {data[pos]}")
    rows = struct.unpack_from("<i", data, pos + 1)[0]
    pos += 5
    if data[pos] != 4:
        raise ValueError(f"{path}: unexpected int size marker {data[pos]}")
    cols = struct.unpack_from("<i", data, pos + 1)[0]
    pos += 5
    mat = np.frombuffer(data, dtype=dtype, count=rows * cols, offset=pos)
    return mat.reshape(rows, cols).astype(np.float64)


def _load_cmvn(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(means, inverse_std)`` (float32) from a Kaldi global-CMVN ark."""
    stats = _read_kaldi_matrix(path)
    if stats.shape[0] != 2:
        raise ValueError(f"{path}: expected 2-row CMVN stats, got {stats.shape}")
    dim = stats.shape[1] - 1
    count = stats[0, dim]
    if count < 1:
        raise ValueError(f"{path}: invalid CMVN count {count}")
    means = stats[0, :dim] / count
    variances = np.maximum(stats[1, :dim] / count - means * means, _VAR_FLOOR)
    return means.astype(np.float32), (1.0 / np.sqrt(variances)).astype(np.float32)


class FbankCmvn:
    """Persistent fbank + CMVN front-end; ``accept`` returns newly-ready frames."""

    def __init__(self, cmvn_path: str, num_mel_bins: int = NUM_MEL_BINS):
        import kaldi_native_fbank as knf  # lazy: only when FireRedVAD is used

        self._knf = knf
        self._num_bins = num_mel_bins
        opts = knf.FbankOptions()
        opts.frame_opts.samp_freq = SAMPLE_RATE
        opts.frame_opts.frame_length_ms = 25
        opts.frame_opts.frame_shift_ms = 10
        opts.frame_opts.dither = 0.0
        opts.frame_opts.snip_edges = True
        opts.mel_opts.num_bins = num_mel_bins
        opts.mel_opts.debug_mel = False
        self._opts = opts
        self._means, self._istd = _load_cmvn(cmvn_path)
        if self._means.shape[0] != num_mel_bins:
            raise ValueError(f"CMVN dim {self._means.shape[0]} != num_mel_bins {num_mel_bins}")
        self.reset()

    def reset(self) -> None:
        self._fbank = self._knf.OnlineFbank(self._opts)
        self._consumed = 0

    def accept(self, samples: np.ndarray) -> np.ndarray:
        """Feed int16-scale samples; return CMVN'd fbank for new frames ``[n, 80]``."""
        if samples.size:
            self._fbank.accept_waveform(SAMPLE_RATE, np.ascontiguousarray(samples, dtype=np.float32))
        ready = self._fbank.num_frames_ready
        if ready <= self._consumed:
            return np.zeros((0, self._num_bins), dtype=np.float32)
        frames = [self._fbank.get_frame(i) for i in range(self._consumed, ready)]
        feat = np.asarray(np.vstack(frames), dtype=np.float32)  # copies out of knf storage
        # Release consumed frames NOW, and only after the copy (get_frame returns views
        # into knf's deque): knf retains every frame until popped and reset() runs only
        # at utterance close, so idle listening grows ~115 MB/h at 80 bins. pop() keeps
        # num_frames_ready/get_frame indices ABSOLUTE.
        self._fbank.pop(ready - self._consumed)
        self._consumed = ready
        # copy=False: already float32, so this is a free guard that a widened CMVN
        # dtype cannot reach the ONNX/RKNN feed.
        return ((feat - self._means) * self._istd).astype(np.float32, copy=False)
