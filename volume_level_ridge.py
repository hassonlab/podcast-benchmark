"""Volume-level ridge regression utilities.

Helpers to align neural and audio time-series across candidate latencies
and evaluate ridge regressions using cross-validation. These functions power
lag sweeps for the volume-level encoding task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import torch
from sklearn.model_selection import KFold


@dataclass(frozen=True)
class RidgeLagResult:
    """Container for the results of a ridge latency sweep."""

    lag_ms: np.ndarray
    r2: np.ndarray
    alpha: np.ndarray
    coef_norm: np.ndarray
    n_samples: np.ndarray
    n_features: np.ndarray
    train_r2: np.ndarray
    device: np.ndarray
    cv_splits: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:  # pragma: no cover - convenience only
        """Return a dictionary representation compatible with plotting utilities."""

        return {
            "lag_ms": self.lag_ms,
            "r2": self.r2,
            "alpha": self.alpha,
            "coef_norm": self.coef_norm,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "train_r2": self.train_r2,
            "device": self.device,
            "cv_splits": self.cv_splits,
        }


def align_for_lag(
    audio: Sequence[float] | np.ndarray,
    neural: np.ndarray,
    lag_ms: float,
    sampling_rate_hz: float,
    *,
    allow_partial_overlap: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Align neural predictors and audio targets for a specific latency.

    Args:
        audio: One-dimensional audio envelope array (``shape == (n_samples,)``).
        neural: Neural data with ``shape == (n_electrodes, n_samples)``.
        lag_ms: Lag in milliseconds. Positive values shift neural activity forward
            relative to the audio target (i.e., audio leads neural).
        sampling_rate_hz: Sampling rate shared by ``audio`` and ``neural``.
        allow_partial_overlap: If ``True`` (default), truncate to the overlapping
            region when lagging would otherwise reduce the shared extent. If
            ``False`` and the lag prevents full overlap, a ``ValueError`` is raised.

    Returns:
        Tuple ``(X, y)`` where ``X`` is an ``(n_observations, n_features)`` design
        matrix and ``y`` is a length ``n_observations`` target vector.

    Raises:
        ValueError: If inputs have incompatible shapes or empty overlap.
    """

    audio_arr = np.asarray(audio, dtype=np.float64)
    neural_arr = np.asarray(neural, dtype=np.float64)

    if audio_arr.ndim != 1:
        raise ValueError("audio must be a 1D array")
    if neural_arr.ndim != 2:
        raise ValueError("neural must be a 2D array of shape (n_features, n_samples)")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")

    samples = audio_arr.size
    if neural_arr.shape[1] != samples:
        if not allow_partial_overlap:
            raise ValueError(
                "neural and audio must have the same length when allow_partial_overlap=False"
            )
        samples = min(samples, neural_arr.shape[1])
        audio_arr = audio_arr[:samples]
        neural_arr = neural_arr[:, :samples]

    shift_samples = int(np.round(lag_ms * sampling_rate_hz / 1000.0))

    if shift_samples > 0:
        if shift_samples >= samples:
            raise ValueError(
                f"Positive lag of {shift_samples} samples exceeds available length {samples}"
            )
        X = neural_arr[:, shift_samples:].T
        y = audio_arr[:-shift_samples]
    elif shift_samples < 0:
        shift = abs(shift_samples)
        if shift >= samples:
            raise ValueError(
                f"Negative lag of {shift} samples exceeds available length {samples}"
            )
        X = neural_arr[:, :-shift].T
        y = audio_arr[shift:]
    else:
        X = neural_arr.T
        y = audio_arr

    if not allow_partial_overlap and X.shape[0] != y.size:
        raise ValueError("Lag setting produced mismatched lengths without truncation allowed")

    n_observations = min(X.shape[0], y.size)
    if n_observations <= 0:
        raise ValueError("Lag alignment resulted in zero overlapping samples")

    return X[:n_observations], y[:n_observations]


def _ridge_closed_form(
    X: torch.Tensor,
    y: torch.Tensor,
    alpha: torch.Tensor,
) -> torch.Tensor:
    """Solve the ridge regression weights analytically."""

    n_features = X.shape[1]
    eye = torch.eye(n_features, dtype=X.dtype, device=X.device)
    gram = X.T @ X
    rhs = X.T @ y
    solution = torch.linalg.solve(gram + alpha * eye, rhs)
    return solution


def _as_numpy_array(values: Iterable[float] | np.ndarray, dtype=float) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=dtype)
    if array.ndim != 1:
        raise ValueError("lags_ms and alphas must be one-dimensional")
    return array


def ridge_r2_by_lag(
    audio: Sequence[float] | np.ndarray,
    neural: np.ndarray,
    sampling_rate_hz: float,
    lags_ms: Iterable[float] | np.ndarray,
    *,
    alphas: Optional[Iterable[float]] = None,
    cv_splits: int = 10,
    device: Optional[str] = None,
    random_state: Optional[int] = 0,
    verbose: bool = False,
) -> dict[str, np.ndarray]:
    """Evaluate ridge regression fits across multiple temporal lags.

    Args:
        audio: One-dimensional audio envelope array.
        neural: Neural data with shape ``(n_electrodes, n_samples)``.
        sampling_rate_hz: Sampling rate in Hertz shared by ``audio`` and ``neural``.
        lags_ms: Sequence of candidate lags in milliseconds.
        alphas: Optional iterable of ridge regularisation strengths. Defaults to
            ``np.logspace(-4, 4, 17)`` when ``None``.
        cv_splits: Number of K-fold splits to use for cross-validation.
        device: Torch device string. Defaults to ``"cuda"`` when available.
        random_state: Random seed for the K-fold splitter. ``None`` disables
            shuffling.
        verbose: If ``True``, print progress messages.

    Returns:
        Dictionary containing per-lag metrics suitable for plotting utilities.
    """

    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2")

    if alphas is None:
        alphas_arr = np.logspace(-4, 4, 17, dtype=float)
    else:
        alphas_arr = _as_numpy_array(alphas, dtype=float)
    if np.any(alphas_arr <= 0):
        raise ValueError("All alphas must be positive")

    lags_arr = _as_numpy_array(lags_ms, dtype=float)

    audio_arr = np.asarray(audio, dtype=np.float32)
    neural_arr = np.asarray(neural, dtype=np.float32)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    results: dict[str, list] = {
        "lag_ms": [],
        "r2": [],
        "alpha": [],
        "coef_norm": [],
        "n_samples": [],
        "n_features": [],
        "train_r2": [],
        "device": [],
        "cv_splits": [],
    }

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    alpha_tensors = [torch.as_tensor(alpha, device=torch_device, dtype=torch.float32) for alpha in alphas_arr]

    with torch.no_grad():
        for lag in lags_arr:
            try:
                X_np, y_np = align_for_lag(audio_arr, neural_arr, float(lag), sampling_rate_hz)
            except ValueError:
                if verbose:
                    print(f"Skipping lag {lag:.1f} ms: unable to align series")
                continue

            if X_np.shape[0] <= cv_splits or X_np.shape[1] == 0:
                if verbose:
                    print(f"Skipping lag {lag:.1f} ms: insufficient samples/features")
                continue

            X = torch.as_tensor(X_np, device=torch_device, dtype=torch.float32)
            y = torch.as_tensor(y_np, device=torch_device, dtype=torch.float32)

            # Drop constant features to avoid singular matrices
            feature_std = torch.std(X, dim=0, unbiased=False)
            valid_mask = torch.isfinite(feature_std) & (feature_std > 0)
            if not torch.any(valid_mask):
                if verbose:
                    print(f"Skipping lag {lag:.1f} ms: all features constant")
                continue
            X = X[:, valid_mask]

            best_alpha: Optional[float] = None
            best_cv_r2 = -np.inf

            for alpha_tensor, alpha_value in zip(alpha_tensors, alphas_arr):
                fold_scores: list[float] = []

                for train_idx, test_idx in cv.split(np.arange(X.shape[0])):
                    train_ids = torch.as_tensor(train_idx, device=torch_device, dtype=torch.long)
                    test_ids = torch.as_tensor(test_idx, device=torch_device, dtype=torch.long)

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
                if verbose:
                    print(f"Skipping lag {lag:.1f} ms: no valid folds produced")
                continue

            best_alpha_tensor = torch.as_tensor(best_alpha, device=torch_device, dtype=torch.float32)

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
            results["device"].append(device)
            results["cv_splits"].append(cv_splits)

    if not results["lag_ms"]:
        return results

    results["lag_ms"] = np.asarray(results["lag_ms"], dtype=float)
    results["r2"] = np.asarray(results["r2"], dtype=float)
    results["alpha"] = np.asarray(results["alpha"], dtype=float)
    results["coef_norm"] = np.asarray(results["coef_norm"], dtype=float)
    results["n_samples"] = np.asarray(results["n_samples"], dtype=int)
    results["n_features"] = np.asarray(results["n_features"], dtype=int)
    results["train_r2"] = np.asarray(results["train_r2"], dtype=float)
    results["device"] = np.asarray(results["device"], dtype=object)
    results["cv_splits"] = np.asarray(results["cv_splits"], dtype=int)

    return results
