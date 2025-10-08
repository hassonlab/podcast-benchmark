"""Utilities for volume-level encoding pipelines (audio + neural features)."""

from __future__ import annotations

import os
import warnings
from math import gcd
from typing import Any, Optional, Sequence

import numpy as np
from scipy.signal import butter, hilbert, resample_poly, sosfilt, sosfiltfilt

try:
    import soundfile as sf
except ImportError:  # pragma: no cover - optional dependency just like original notebook
    sf = None


def hilbert_envelope(waveform: np.ndarray) -> np.ndarray:
    """Compute the amplitude envelope of a 1D waveform via the Hilbert transform."""

    data = np.asarray(waveform, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError("Expected a 1D waveform for Hilbert envelope computation.")

    envelope = np.abs(hilbert(data))
    return envelope.astype(np.float32, copy=False)


def _ensure_float32(array: Any) -> np.ndarray:
    """Return a float32 NumPy array, staying zero-copy when supported."""

    try:
        return np.asarray(array, dtype=np.float32, copy=False)
    except TypeError:  # NumPy < 1.20 lacks the ``copy`` keyword for asarray
        return np.asarray(array, dtype=np.float32)


def load_audio_waveform(
    path: str,
    target_sr: int = 44100,
) -> tuple[np.ndarray, int]:
    """Load mono audio data with the assumptions used in the volume-level notebook."""

    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    if sf is None:
        raise ImportError(
            "soundfile is required to load audio waveforms for volume-level encoding."
        )

    waveform, sr = sf.read(path, dtype="float32", always_2d=False)

    if waveform.ndim > 1:
        waveform = np.mean(waveform, axis=1)

    if sr != target_sr:
        raise ValueError(f"Expected {target_sr}Hz audio, got {sr}Hz.")

    return waveform.astype(np.float32, copy=False), int(sr)


def butterworth_lowpass_envelope(
    envelope: np.ndarray,
    sr: int,
    cutoff_hz: float = 8.0,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Apply a Butterworth low-pass filter to a 1D envelope."""

    if cutoff_hz <= 0:
        raise ValueError("cutoff_hz must be positive.")
    if sr <= 0:
        raise ValueError("Sampling rate must be positive.")

    nyquist = 0.5 * sr
    if cutoff_hz >= nyquist:
        raise ValueError("cutoff_hz must be below the Nyquist frequency.")

    data = np.asarray(envelope, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError("Expected a 1D envelope for low-pass filtering.")

    sos = butter(order, cutoff_hz / nyquist, btype="low", output="sos")

    if zero_phase:
        try:
            filtered = sosfiltfilt(sos, data)
        except ValueError as exc:  # pragma: no cover - follows original behaviour
            warnings.warn(
                f"Zero-phase filtering failed ({exc}); falling back to causal filtering.",
                RuntimeWarning,
            )
            filtered = sosfilt(sos, data)
    else:
        filtered = sosfilt(sos, data)

    return np.asarray(filtered, dtype=np.float32)


def resample_envelope(
    envelope: np.ndarray,
    sr_in: int,
    sr_out: int,
) -> np.ndarray:
    """Resample an audio envelope to match the target sampling rate."""

    if sr_in <= 0 or sr_out <= 0:
        raise ValueError("Sampling rates must be positive integers.")

    data = np.asarray(envelope, dtype=np.float32)
    if data.ndim != 1:
        raise ValueError("Expected a 1D envelope for resampling.")

    factor = gcd(int(sr_out), int(sr_in))
    up = int(sr_out // factor)
    down = int(sr_in // factor)

    resampled = resample_poly(data, up, down)
    return np.asarray(resampled, dtype=np.float32)


def compress_envelope_db(
    envelope: np.ndarray,
    eps: Optional[float] = None,
) -> np.ndarray:
    """Compress an envelope to a decibel scale with a configurable epsilon."""

    data = np.asarray(envelope, dtype=float)
    if data.ndim != 1:
        raise ValueError("Expected a 1D envelope for log compression.")

    data = np.clip(data, 0.0, None)

    if eps is None:
        eps = max(1e-12, float(data.max()) * 1e-6)
    elif eps <= 0:
        raise ValueError("eps must be positive when provided.")

    compressed = 20.0 * np.log10(data + eps)
    return compressed.astype(np.float32, copy=False)


def _as_channel_matrix(
    array: np.ndarray, *, name: str
) -> tuple[np.ndarray, bool]:
    """Coerce input to (channels, samples) form while tracking squeezes."""

    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim == 1:
        return arr[np.newaxis, :], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError(
        f"Expected {name} with shape (channels, time); received shape {arr.shape}."
    )


def _restore_shape(arr: np.ndarray, squeeze: bool) -> np.ndarray:
    """Return the array with its original dimensionality."""

    return arr[0] if squeeze else arr


def _validate_channel_indices(
    n_channels: int, picks: Optional[Sequence[int]] = None
) -> np.ndarray:
    """Return valid channel indices limited to the available range."""

    if picks is None:
        return np.arange(n_channels, dtype=int)
    cleaned = sorted(
        {
            int(idx)
            for idx in picks
            if idx is not None and 0 <= int(idx) < n_channels
        }
    )
    return np.asarray(cleaned if cleaned else range(n_channels), dtype=int)


def _channelwise_mean_std(
    data: np.ndarray, picks: Optional[Sequence[int]] = None, eps: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-channel mean and std, optionally restricted to a subset."""

    if data.ndim != 2:
        raise ValueError(
            f"Expected (channels, time) input; received shape {data.shape}."
        )

    data64 = np.asarray(data, dtype=np.float64)
    means = np.nanmean(data64, axis=1, keepdims=True)
    stds = np.nanstd(data64, axis=1, keepdims=True)
    stds = np.where(~np.isfinite(stds) | (stds < eps), eps, stds)

    if picks is not None:
        idx = _validate_channel_indices(data.shape[0], picks)
        means[idx] = np.nanmean(data64[idx], axis=1, keepdims=True)
        stds[idx] = np.nanstd(data64[idx], axis=1, keepdims=True)
        stds[idx] = np.where(~np.isfinite(stds[idx]) | (stds[idx] < eps), eps, stds[idx])

    return means.astype(np.float64), stds.astype(np.float64)


def zscore_subjects(
    data_list: Sequence[np.ndarray],
    electrode_groups: Optional[Sequence[Optional[Sequence[int]]]] = None,
    eps: float = 1e-8,
) -> tuple[list[np.ndarray], dict[int, dict[str, float]]]:
    """Per-subject, per-channel z-score normalization for broadband ECoG."""

    normalized: list[np.ndarray] = []
    stats: dict[int, dict[str, float]] = {}

    for sid, data in enumerate(data_list):
        arr = np.asarray(data, dtype=np.float32)
        picks = None
        if electrode_groups is not None and sid < len(electrode_groups):
            picks = electrode_groups[sid]

        means, stds = _channelwise_mean_std(arr, picks=picks, eps=eps)
        arr_norm = ((arr.astype(np.float64, copy=False) - means) / stds).astype(
            np.float32,
            copy=False,
        )
        normalized.append(arr_norm)

        post_mean = float(np.nanmean(arr_norm))
        post_std = float(np.nanstd(arr_norm))
        stats[sid] = {
            "global_mean": float(np.nanmean(means)),
            "global_std": float(np.nanmean(stds)),
            "post_mean": post_mean,
            "post_std": post_std,
            "n_channels": int(arr.shape[0]),
            "n_timepoints": int(arr.shape[1]),
        }

    return normalized, stats


def bandpass_high_gamma(
    data: np.ndarray,
    sampling_rate: float,
    low: float = 70.0,
    high: float = 150.0,
    order: int = 4,
    zero_phase: bool = True,
) -> np.ndarray:
    """Band-pass ECoG data into the high-gamma range along the last axis."""

    arr, squeeze = _as_channel_matrix(data, name="data")

    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive.")
    nyquist = 0.5 * float(sampling_rate)
    if not (0 < low < high < nyquist):
        raise ValueError("Require 0 < low < high < Nyquist frequency.")

    sos = butter(order, [low / nyquist, high / nyquist], btype="band", output="sos")

    try:
        filtered = sosfiltfilt(sos, arr, axis=1) if zero_phase else sosfilt(
            sos, arr, axis=1
        )
    except ValueError as exc:
        warnings.warn(
            f"Zero-phase filtering failed ({exc}); falling back to causal filtering.",
            RuntimeWarning,
        )
        filtered = sosfilt(sos, arr, axis=1)

    filtered = _ensure_float32(filtered)
    return _restore_shape(filtered, squeeze)


def high_gamma_envelope(
    data: np.ndarray,
    sampling_rate: float,
    low: float = 70.0,
    high: float = 150.0,
    order: int = 4,
    zero_phase: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Band-pass data and compute the Hilbert amplitude envelope per channel."""

    filtered = bandpass_high_gamma(
        data,
        sampling_rate=sampling_rate,
        low=low,
        high=high,
        order=order,
        zero_phase=zero_phase,
    )

    filtered_2d, squeeze = _as_channel_matrix(filtered, name="filtered data")
    envelope = np.apply_along_axis(hilbert_envelope, 1, filtered_2d).astype(
        np.float32, copy=False
    )

    return _restore_shape(filtered_2d, squeeze), _restore_shape(envelope, squeeze)


def log_high_gamma_envelope(
    envelope: np.ndarray,
    eps: Optional[float] = None,
    eps_scale: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Apply a log10 transform to a high-gamma envelope with a shared epsilon."""

    arr, squeeze = _as_channel_matrix(envelope, name="envelope")

    if eps is None:
        eps_value = max(1e-12, float(arr.max()) * eps_scale)
    else:
        eps_value = float(eps)
        if eps_value <= 0:
            raise ValueError("eps must be positive when provided.")

    log_arr = np.log10(np.clip(arr, 0.0, None) + eps_value).astype(
        np.float32, copy=False
    )

    return _restore_shape(log_arr, squeeze), eps_value


def extract_high_gamma_features(
    data_list: Sequence[np.ndarray],
    sampling_rates: Sequence[float],
    electrode_groups: Optional[Sequence[Optional[Sequence[int]]]] = None,
    low: float = 70.0,
    high: float = 150.0,
    order: int = 4,
    eps_scale: float = 1e-6,
    zero_phase: bool = True,
) -> tuple[
    list[np.ndarray], list[np.ndarray], list[np.ndarray], list[dict[str, Any]]
]:
    """Compute band-passed, Hilbert, and log-transformed high-gamma features."""

    if len(data_list) != len(sampling_rates):
        raise ValueError("data_list and sampling_rates must have the same length.")

    filtered_list: list[np.ndarray] = []
    envelope_list: list[np.ndarray] = []
    log_list: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []

    for sid, (data, sr) in enumerate(zip(data_list, sampling_rates)):
        arr = np.asarray(data, dtype=np.float32, order="C")
        if arr.ndim != 2:
            raise ValueError(
                f"Subject {sid}: expected (channels, time); received shape {arr.shape}."
            )
        if sr <= 0:
            raise ValueError(f"Subject {sid}: sampling rate must be positive; got {sr}.")

        picks = _validate_channel_indices(
            arr.shape[0],
            electrode_groups[sid]
            if electrode_groups is not None and sid < len(electrode_groups)
            else None,
        )

        filtered, envelope = high_gamma_envelope(
            arr[picks, :],
            sampling_rate=sr,
            low=low,
            high=high,
            order=order,
            zero_phase=zero_phase,
        )
        log_envelope, eps_value = log_high_gamma_envelope(
            envelope, eps=None, eps_scale=eps_scale
        )

        filtered_arr = np.atleast_2d(_ensure_float32(filtered))
        envelope_arr = np.atleast_2d(_ensure_float32(envelope))
        log_arr = np.atleast_2d(_ensure_float32(log_envelope))

        filtered_list.append(filtered_arr)
        envelope_list.append(envelope_arr)
        log_list.append(log_arr)
        metadata.append(
            {
                "subject_index": sid,
                "sampling_rate": float(sr),
                "electrode_indices": picks.tolist(),
                "log_epsilon": float(eps_value),
                "band": (float(low), float(high)),
            }
        )

    return filtered_list, envelope_list, log_list, metadata
