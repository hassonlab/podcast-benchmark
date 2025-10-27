"""Helpers for the volume-level ridge CLI workflow.

This module mirrors only the pieces needed outside the notebook: loading the
preprocessed high-gamma derivatives and applying the same log compression the
notebook uses prior to ridge regression. Everything else intentionally lives in
notebook cells so the CLI footprint stays minimal.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from config import DataParams
from data_utils import load_raws, log_transform_preprocessor

DEFAULT_LOG_PARAMS = {
    "epsilon_scale": 1e-6,
    "epsilon_floor": 1e-12,
    "clip_min": 0.0,
    "log_base": 10.0,
}

__all__ = [
    "DEFAULT_LOG_PARAMS",
    "load_log_transformed_high_gamma",
    "trim_to_audio_length",
    "compute_window_hop",
    "sliding_window_rms",
]


def _apply_log_transform(neural: np.ndarray, *, log_params: Optional[dict]) -> Tuple[np.ndarray, dict]:
    """Log-compress a single subject's high-gamma matrix."""

    neural_arr = np.asarray(neural, dtype=np.float32)
    if neural_arr.ndim != 2:
        raise ValueError("Expected neural array shaped (channels, samples).")

    params = dict(DEFAULT_LOG_PARAMS)
    if log_params:
        params.update(log_params)

    transformed = log_transform_preprocessor(neural_arr[np.newaxis, ...], params)[0]
    epsilon = float(params["epsilon"])

    metadata = {
        "epsilon": epsilon,
        "log_params": params,
    }
    return transformed, metadata


def load_log_transformed_high_gamma(
    data_params: DataParams,
    *,
    log_params: Optional[dict] = None,
    apply_log: bool = True,
) -> Dict[int, dict]:
    """Fetch preprocessed raws and emit per-subject matrices ready for ridge."""

    raws = load_raws(data_params)
    subject_payloads: Dict[int, dict] = {}

    for subject_id, raw in zip(data_params.subject_ids, raws):
        neural = raw.get_data()
        params = dict(DEFAULT_LOG_PARAMS)
        if log_params:
            params.update(log_params)

        if apply_log:
            log_neural, meta = _apply_log_transform(neural, log_params=params)
            epsilon = meta["epsilon"]
            params_used = meta["log_params"]
        else:
            log_neural = np.asarray(neural, dtype=np.float32)
            epsilon = None
            params_used = params

        subject_payloads[subject_id] = {
            "log_highgamma": log_neural,
            "sampling_rate": float(raw.info.get("sfreq", 0.0)),
            "channel_names": list(raw.ch_names),
            "epsilon": epsilon,
            "log_params": params_used,
            "n_channels": int(log_neural.shape[0]),
            "n_samples": int(log_neural.shape[1]),
        }

    return subject_payloads


def trim_to_audio_length(neural: np.ndarray, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Crop neural/audio series to a shared minimum length."""

    neural_arr = np.asarray(neural, dtype=np.float32)
    if neural_arr.ndim != 2:
        raise ValueError("neural must be shaped (channels, samples).")

    audio_arr = np.asarray(audio, dtype=np.float32)
    min_len = min(neural_arr.shape[1], audio_arr.shape[0])
    if min_len <= 0:
        raise ValueError("Cannot trim arrays with zero overlapping length.")

    return neural_arr[:, :min_len], audio_arr[:min_len]


def compute_window_hop(
    sampling_rate: float,
    window_ms: float,
    hop_ms: float,
) -> Tuple[int, int, float]:
    """Convert millisecond window parameters into sample counts and effective rate."""

    sr = float(sampling_rate)
    if window_ms <= 0 or hop_ms <= 0:
        raise ValueError("window_ms and hop_ms must be positive.")

    window_samples = max(1, int(round(window_ms * sr / 1000.0)))
    hop_samples = max(1, int(round(hop_ms * sr / 1000.0)))
    effective_sr = sr / hop_samples
    return window_samples, hop_samples, effective_sr


def sliding_window_rms(
    series: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> np.ndarray:
    """Apply an RMS reducer across a sliding window along the last axis."""

    arr = np.asarray(series, dtype=np.float64, order="C")
    length = arr.shape[-1]
    if length < window_samples:
        raise ValueError(
            f"Window of {window_samples} samples exceeds series length {length}."
        )

    starts = np.arange(0, length - window_samples + 1, hop_samples, dtype=int)
    if starts.size == 0:
        raise ValueError(
            "hop_samples produced zero windows; choose a smaller hop or window."
        )

    if arr.ndim == 1:
        arr = arr[np.newaxis, :]
        squeeze = True
    else:
        squeeze = False

    sq = np.square(arr, dtype=np.float64)
    cumsum = np.concatenate(
        [np.zeros((sq.shape[0], 1), dtype=np.float64), np.cumsum(sq, axis=-1)],
        axis=-1,
    )
    start_vals = cumsum[:, starts]
    stop_vals = cumsum[:, starts + window_samples]
    window_means = (stop_vals - start_vals) / float(window_samples)
    rms = np.sqrt(np.maximum(window_means, 0.0))
    out = rms.astype(np.float32)
    if squeeze:
        out = out[0]
    return out
