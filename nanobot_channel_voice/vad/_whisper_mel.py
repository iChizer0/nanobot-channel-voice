# Vendored from Pipecat's smart_turn/_whisper_features.py (BSD 2-Clause,
# Copyright (c) 2024-2026, Daily), itself mirroring transformers'
# WhisperFeatureExtractor (Apache-2.0) with chunk_length=8 defaults.
"""Numpy-only Whisper-style log-mel features for the Smart Turn v3 model.

The model was trained on ``WhisperFeatureExtractor(chunk_length=8)`` output:
8 s of 16 kHz audio -> float32 ``(80, 800)`` log-mel in roughly ``[-1, 1]``.
Callers pad/truncate to exactly 128 000 samples BEFORE calling (Smart Turn
pads at the START so speech sits at the window end); this module then matches
the reference bit-for-bit: normalize the waveform first, reflect-padded
centered power spectrogram, Slaney mel filterbank, log10, dynamic-range floor.
"""

from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

_N_FFT = 400
_HOP_LENGTH = 160
_N_MELS = 80
SAMPLE_RATE = 16000
WINDOW_SAMPLES = SAMPLE_RATE * 8  # the model's fixed 8 s input window
_MEL_FLOOR = 1e-10
_NORM_VARIANCE_EPS = 1e-7


def _hertz_to_mel_slaney(freq: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = 27.0 / np.log(6.4)
    freq = np.atleast_1d(np.asarray(freq, dtype=np.float64))
    mels = 3.0 * freq / 200.0
    log_region = freq >= min_log_hertz
    mels[log_region] = min_log_mel + np.log(freq[log_region] / min_log_hertz) * logstep
    return mels


def _mel_to_hertz_slaney(mels: np.ndarray) -> np.ndarray:
    min_log_hertz = 1000.0
    min_log_mel = 15.0
    logstep = np.log(6.4) / 27.0
    mels = np.atleast_1d(np.asarray(mels, dtype=np.float64))
    freq = 200.0 * mels / 3.0
    log_region = mels >= min_log_mel
    freq[log_region] = min_log_hertz * np.exp(logstep * (mels[log_region] - min_log_mel))
    return freq


def _build_mel_filterbank(
    num_frequency_bins: int,
    num_mel_filters: int,
    min_frequency: float,
    max_frequency: float,
    sampling_rate: int,
) -> np.ndarray:
    """Slaney-normalized triangular filterbank, ``(num_frequency_bins, num_mel_filters)``."""
    mel_min = float(_hertz_to_mel_slaney(np.array([min_frequency], dtype=np.float64))[0])
    mel_max = float(_hertz_to_mel_slaney(np.array([max_frequency], dtype=np.float64))[0])
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = _mel_to_hertz_slaney(mel_freqs)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    mel_filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
    mel_filters *= np.expand_dims(enorm, 0)
    return mel_filters


# Periodic Hann (np.hanning is symmetric; +1/[:-1] matches torch.hann_window).
_HANN_WINDOW = np.hanning(_N_FFT + 1)[:-1]
_MEL_FILTERS = _build_mel_filterbank(
    num_frequency_bins=_N_FFT // 2 + 1,
    num_mel_filters=_N_MELS,
    min_frequency=0.0,
    max_frequency=SAMPLE_RATE / 2.0,
    sampling_rate=SAMPLE_RATE,
)


def _power_spectrogram(
    waveform: np.ndarray, window: np.ndarray, frame_length: int, hop_length: int
) -> np.ndarray:
    """Centered power spectrogram (reflect pad, batched rFFT, |.|^2), float64 like
    the reference; ``(num_frequency_bins, num_frames)``."""
    pad = frame_length // 2
    padded = np.pad(waveform.astype(np.float64), (pad, pad), mode="reflect")
    win = window.astype(np.float64)
    windows = sliding_window_view(padded, frame_length)[::hop_length]
    spec = np.fft.rfft(windows * win, axis=-1)
    # re^2 + im^2, not |z|^2: same value without a sqrt per bin (and marginally
    # more accurate: the reference's abs() rounds through the sqrt first).
    power = np.square(spec.real)
    power += np.square(spec.imag)
    return power.T


def log_mel_features(audio: np.ndarray) -> np.ndarray:
    """Log-mel features for one exactly-8 s float window; float32 ``(80, 800)``."""
    if audio.ndim != 1 or audio.size != WINDOW_SAMPLES:
        raise ValueError(f"expected 1-D audio of {WINDOW_SAMPLES} samples, got {audio.shape}")

    # Zero-mean unit-variance BEFORE the spectrogram, in float32: mirrors
    # transformers' do_normalize=True precision exactly.
    x = np.asarray(audio, dtype=np.float32)
    x = (x - x.mean()) / np.sqrt(x.var() + _NORM_VARIANCE_EPS)

    magnitudes = _power_spectrogram(x, _HANN_WINDOW, _N_FFT, _HOP_LENGTH)
    mel_spec = np.maximum(_MEL_FLOOR, _MEL_FILTERS.T @ magnitudes)
    log_spec = np.log10(mel_spec)
    log_spec = log_spec[:, :-1]  # the reference drops the trailing frame
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    return ((log_spec + 4.0) / 4.0).astype(np.float32)
