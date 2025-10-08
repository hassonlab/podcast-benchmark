"""Utility functions for ridge regression latency analysis and sliding-window preprocessing.

These helpers are ported from the exploratory volume-level notebook so they can be
re-used in scripts and tests. They cover two main capabilities:

1. Sliding-window RMS down-sampling of neural/audio sequences to reduce temporal
   resolution while preserving amplitude envelopes.
2. Closed-form ridge regression solved via PyTorch with cross-validation across
   temporal lags.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class SlidingWindowMetadata:
    """Describe the parameters used for RMS-based sliding-window aggregation."""

    mode: str
    window_ms: float
    hop_ms: float
    window_samples: int
    hop_samples: int
    effective_sr: float


def rms_sliding_window(
    arr: np.ndarray,
    window_samples: int,
    hop_samples: int,
) -> np.ndarray:
    """Compute RMS values over sliding windows along the last axis.

    Args:
        arr: Input array where the last dimension is interpreted as time.
        window_samples: Number of samples per window (must be >= 1).
        hop_samples: Hop size between windows (must be >= 1).

    Returns:
        An array with the same leading dimensions as ``arr`` but with the last
        dimension replaced by the number of windows. Values are ``float32`` RMS.

    Raises:
        ValueError: If parameters are invalid or produce zero windows.
    """

    if window_samples <= 0:
        raise ValueError("window_samples must be a positive integer")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be a positive integer")

    arr64 = np.asarray(arr, dtype=np.float64, order="C")
    series_len = arr64.shape[-1]
    if series_len < window_samples:
        raise ValueError(
            f"Series length {series_len} is shorter than the requested window of {window_samples} samples."
        )

    starts = np.arange(0, series_len - window_samples + 1, hop_samples, dtype=int)
    if starts.size == 0:
        raise ValueError(
            f"Hop of {hop_samples} samples yields zero windows; adjust window/hop settings."
        )

    sq = np.square(arr64, dtype=np.float64)
    cumsum = np.cumsum(sq, axis=-1, dtype=np.float64)
    pad = np.zeros((*cumsum.shape[:-1], 1), dtype=np.float64)
    cumsum = np.concatenate([pad, cumsum], axis=-1)

    start_vals = np.take(cumsum, starts, axis=-1)
    end_vals = np.take(cumsum, starts + window_samples, axis=-1)
    window_means = (end_vals - start_vals) / float(window_samples)
    np.maximum(window_means, 0.0, out=window_means)
    rms = np.sqrt(window_means, dtype=np.float64)
    return rms.astype(np.float32, copy=False)


def apply_sliding_window_rms(
    audio: np.ndarray,
    neural: np.ndarray,
    sampling_rate: float,
    window_ms: float,
    hop_ms: float,
    *,
    trim_to_common: bool = True,
) -> Tuple[np.ndarray, np.ndarray, float, SlidingWindowMetadata]:
    """Down-sample audio and neural signals via sliding-window RMS.

    Args:
        audio: 1D float array of audio amplitude values.
        neural: 2D float array shaped ``(n_channels, n_samples)``.
        sampling_rate: Original sampling rate in Hz.
        window_ms: Window duration in milliseconds.
        hop_ms: Hop size in milliseconds.
        trim_to_common: If True, trim both sequences to the minimum number of
            windows so their lengths match exactly.

    Returns:
        Tuple ``(neural_windows, audio_windows, effective_sr, metadata)`` where
        both window arrays are ``float32`` and ``effective_sr`` is the new sample
        rate implied by the hop size.

    Raises:
        ValueError: If inputs are malformed or the window parameters are invalid.
    """

    if neural.ndim != 2:
        raise ValueError("neural array must have shape (n_channels, n_samples)")
    if audio.ndim != 1:
        raise ValueError("audio array must be 1D")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if window_ms <= 0 or hop_ms <= 0:
        raise ValueError("window_ms and hop_ms must be positive")

    sr = float(sampling_rate)
    window_samples = max(1, int(round(window_ms * sr / 1000.0)))
    hop_samples = max(1, int(round(hop_ms * sr / 1000.0)))

    if neural.shape[-1] < window_samples:
        raise ValueError(
            f"neural series length {neural.shape[-1]} is shorter than the window size {window_samples}."
        )
    if audio.shape[-1] < window_samples:
        raise ValueError(
            f"audio series length {audio.shape[-1]} is shorter than the window size {window_samples}."
        )

    neural_rms = rms_sliding_window(neural, window_samples, hop_samples)
    audio_rms = rms_sliding_window(audio[np.newaxis, :], window_samples, hop_samples)[0]

    if trim_to_common:
        n_windows = min(neural_rms.shape[-1], audio_rms.shape[0])
        neural_rms = neural_rms[..., :n_windows]
        audio_rms = audio_rms[:n_windows]

    effective_sr = sr / hop_samples
    metadata = SlidingWindowMetadata(
        mode="rms",
        window_ms=float(window_ms),
        hop_ms=float(hop_ms),
        window_samples=window_samples,
        hop_samples=hop_samples,
        effective_sr=effective_sr,
    )

    return (
        np.ascontiguousarray(neural_rms, dtype=np.float32),
        np.ascontiguousarray(audio_rms, dtype=np.float32),
        float(effective_sr),
        metadata,
    )


def align_for_lag(
    audio: np.ndarray,
    neural: np.ndarray,
    lag_ms: float,
    sampling_rate: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a design matrix and target vector for the requested lag.

    Args:
        audio: 1D array of audio values (length ``T``).
        neural: 2D array of neural data with shape ``(n_channels, T)``.
        lag_ms: Lag in milliseconds. Positive values shift neural data forward
            relative to audio (neural leads audio).
        sampling_rate: Sampling rate in Hz used to convert milliseconds to samples.

    Returns:
        ``(X, y)`` where ``X`` is shaped ``(n_samples, n_features)`` and ``y`` is
        1D with ``n_samples`` entries. The shorter series after shifting is chosen.

    Raises:
        ValueError: If arrays are malformed or the requested lag exceeds data length.
    """

    if neural.ndim != 2:
        raise ValueError("neural array must have shape (n_channels, n_samples)")
    if audio.ndim != 1:
        raise ValueError("audio array must be 1D")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")

    lag_samples = int(np.round(lag_ms / 1000.0 * float(sampling_rate)))

    if lag_samples > 0:
        if lag_samples >= neural.shape[1] or lag_samples >= audio.size:
            raise ValueError(
                f"Shift {lag_samples} exceeds series length (audio={audio.size}, neural={neural.shape[1]})."
            )
        X = neural[:, lag_samples:].T
        y = audio[:-lag_samples]
    elif lag_samples < 0:
        shift = abs(lag_samples)
        if shift >= neural.shape[1] or shift >= audio.size:
            raise ValueError(
                f"Shift {shift} exceeds series length (audio={audio.size}, neural={neural.shape[1]})."
            )
        X = neural[:, :-shift].T
        y = audio[shift:]
    else:
        X = neural.T
        y = audio

    length = min(X.shape[0], y.size)
    return X[:length], y[:length]


def _ridge_closed_form(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Solve ridge regression weights using the closed-form expression."""

    eye = torch.eye(X.shape[1], device=X.device, dtype=X.dtype)
    gram = X.T @ X
    return torch.linalg.solve(gram + alpha * eye, X.T @ y)


def ridge_r2_by_lag(
    audio: np.ndarray,
    neural: np.ndarray,
    sampling_rate: float,
    lags_ms: Sequence[float],
    *,
    alphas: Optional[Sequence[float]] = None,
    cv_splits: int = 10,
    device: Optional[str] = None,
    verbose: bool = True,
) -> dict[str, np.ndarray]:
    """Evaluate ridge regression R^2 across lags using torch closed-form solvers.

    Args:
        audio: 1D array of audio values.
        neural: 2D array (channels, samples) of neural data.
        sampling_rate: Sampling rate associated with the sequences.
        lags_ms: Iterable of lags (in milliseconds) to evaluate.
        alphas: Optional iterable of ridge regularisation strengths. Defaults to
            ``np.logspace(-4, 4, 17)`` if omitted.
        cv_splits: Number of K-fold splits used for cross validation.
        device: Torch device string (e.g. "cpu" or "cuda"). Uses GPU if
            available when omitted.
        verbose: If True, prints summary information.

    Returns:
        Dictionary mapping metric names to NumPy arrays (one entry per lag with
        usable data). Keys: ``lag_ms``, ``r2``, ``alpha``, ``coef_norm``,
        ``n_samples``, ``n_features``, ``train_r2``, ``device``, ``cv_splits``.
    """

    if neural.ndim != 2:
        raise ValueError("neural array must have shape (n_channels, n_samples)")
    if audio.ndim != 1:
        raise ValueError("audio array must be 1D")
    if sampling_rate <= 0:
        raise ValueError("sampling_rate must be positive")
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")

    lags = np.asarray(list(lags_ms), dtype=float)
    if alphas is None:
        alphas_arr = np.logspace(-4, 4, 17, dtype=float)
    else:
        alphas_arr = np.asarray(list(alphas), dtype=float)
        if alphas_arr.ndim != 1 or alphas_arr.size == 0:
            raise ValueError("alphas must be a 1D iterable with at least one value")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if verbose:
        print(f"[ridge_r2_by_lag] device={device}, cv_splits={cv_splits}")

    results: dict[str, list] = {
        "lag_ms": [],
        "r2": [],
        "alpha": [],
        "coef_norm": [],
        "n_samples": [],
        "n_features": [],
        "train_r2": [],
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=0)
    alpha_tensors = [torch.as_tensor(alpha, device=device, dtype=torch.float32) for alpha in alphas_arr]

    audio_np = np.asarray(audio, dtype=np.float32)
    neural_np = np.asarray(neural, dtype=np.float32)

    with torch.no_grad():
        for lag in lags:
            try:
                X_np, y_np = align_for_lag(audio_np, neural_np, float(lag), sampling_rate)
            except ValueError:
                continue

            if X_np.shape[0] <= cv_splits or y_np.size <= cv_splits:
                continue

            X = torch.as_tensor(X_np, device=device, dtype=torch.float32)
            y = torch.as_tensor(y_np, device=device, dtype=torch.float32)

            feature_std = torch.std(X, dim=0, unbiased=False)
            valid_mask = torch.isfinite(feature_std) & (feature_std > 0)
            if not torch.any(valid_mask):
                continue
            X = X[:, valid_mask]

            best_alpha = None
            best_cv_r2 = -np.inf

            for alpha_tensor, alpha_value in zip(alpha_tensors, alphas_arr):
                fold_scores: list[float] = []

                for train_idx, test_idx in cv.split(np.arange(X.shape[0])):
                    train_ids = torch.as_tensor(train_idx, device=device, dtype=torch.long)
                    test_ids = torch.as_tensor(test_idx, device=device, dtype=torch.long)

                    X_train = X.index_select(0, train_ids)
                    X_test = X.index_select(0, test_ids)
                    y_train = y.index_select(0, train_ids)
                    y_test = y.index_select(0, test_ids)

                    x_mean = torch.mean(X_train, dim=0)
                    x_std = torch.std(X_train, dim=0, unbiased=False)
                    x_std = torch.where(x_std < 1e-12, torch.ones_like(x_std), x_std)
                    X_train_norm = (X_train - x_mean) / x_std
                    X_test_norm = (X_test - x_mean) / x_std

                    y_mean = torch.mean(y_train)
                    y_train_centered = y_train - y_mean

                    weights = _ridge_closed_form(X_train_norm, y_train_centered, alpha_tensor)

                    y_pred = X_test_norm @ weights + y_mean
                    y_test_mean = torch.mean(y_test)

                    ss_res = torch.sum((y_test - y_pred) ** 2)
                    ss_tot = torch.sum((y_test - y_test_mean) ** 2)
                    if ss_tot <= 0:
                        continue
                    r2 = 1.0 - (ss_res / ss_tot)
                    fold_scores.append(float(r2.detach().cpu()))

                if len(fold_scores) != cv_splits:
                    continue

                mean_cv_r2 = float(np.mean(fold_scores))
                if mean_cv_r2 > best_cv_r2:
                    best_cv_r2 = mean_cv_r2
                    best_alpha = float(alpha_value)

            if best_alpha is None:
                continue

            best_alpha_tensor = torch.as_tensor(best_alpha, device=device, dtype=torch.float32)

            x_mean_full = torch.mean(X, dim=0)
            x_std_full = torch.std(X, dim=0, unbiased=False)
            x_std_full = torch.where(x_std_full < 1e-12, torch.ones_like(x_std_full), x_std_full)
            X_norm_full = (X - x_mean_full) / x_std_full
            y_mean_full = torch.mean(y)
            y_center_full = y - y_mean_full

            weights_full = _ridge_closed_form(X_norm_full, y_center_full, best_alpha_tensor)

            y_pred_full = X_norm_full @ weights_full + y_mean_full
            ss_res_full = torch.sum((y - y_pred_full) ** 2)
            ss_tot_full = torch.sum((y - torch.mean(y)) ** 2)
            train_r2 = float((1.0 - ss_res_full / ss_tot_full).detach().cpu()) if ss_tot_full > 0 else np.nan
            coef_norm = float(torch.norm(weights_full).detach().cpu())

            results["lag_ms"].append(float(lag))
            results["r2"].append(best_cv_r2)
            results["alpha"].append(best_alpha)
            results["coef_norm"].append(coef_norm)
            results["n_samples"].append(int(y.shape[0]))
            results["n_features"].append(int(X.shape[1]))
            results["train_r2"].append(train_r2)

    if not results["lag_ms"]:
        empty_output = {
            "lag_ms": np.asarray([], dtype=float),
            "r2": np.asarray([], dtype=float),
            "alpha": np.asarray([], dtype=float),
            "coef_norm": np.asarray([], dtype=float),
            "n_samples": np.asarray([], dtype=int),
            "n_features": np.asarray([], dtype=int),
            "train_r2": np.asarray([], dtype=float),
            "device": np.asarray([], dtype=object),
            "cv_splits": np.asarray([], dtype=int),
        }
        return empty_output

    n_rows = len(results["lag_ms"])
    output = {key: np.asarray(values) for key, values in results.items()}
    output["device"] = np.asarray([device] * n_rows, dtype=object)
    output["cv_splits"] = np.asarray([cv_splits] * n_rows, dtype=int)
    return output
