"""Volume-level ridge regression utilities.

Helpers to align neural and audio time-series across candidate latencies
and evaluate ridge regressions using cross-validation. These functions power
lag sweeps for the volume-level encoding task.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold

import plot_utils
from config import ExperimentConfig, TrainingParams

logger = logging.getLogger(__name__)


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


def _materialize_sequence(value):
    if isinstance(value, range):
        return list(value)
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("range(") and text.endswith(")"):
            inner = text[6:-1]
            if inner:
                parts = [int(part.strip()) for part in inner.split(",")]
                return list(range(*parts))
            return []
        try:
            parsed = ast.literal_eval(text)
        except (SyntaxError, ValueError):
            return value
        if isinstance(parsed, range):
            return list(parsed)
        if isinstance(parsed, (list, tuple, np.ndarray)):
            return list(parsed)
        return parsed
    return value


def _ensure_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _infer_target_sr(df: pd.DataFrame, data_params) -> float:
    tp = getattr(data_params, "task_params", {}) or {}
    sr = tp.get("target_sr")
    if sr is not None:
        return float(sr)

    if "start" not in df.columns:
        raise ValueError("volume_level_encoding_task must return a 'start' column")

    starts = df["start"].to_numpy(dtype=float)
    if starts.size < 2:
        raise ValueError(
            "Unable to infer sampling rate from targets; provide task_params.target_sr."
        )

    deltas = np.diff(starts)
    positive = deltas[deltas > 0]
    if positive.size == 0:
        raise ValueError("Unable to infer sampling rate (no positive start deltas)")

    mean_step = float(np.mean(positive))
    if mean_step <= 0:
        raise ValueError("Mean start increment must be positive to infer sampling rate")

    return 1.0 / mean_step


def _load_neural_matrix_from_raws(
    raws: Sequence,
    target_sr: float,
    allow_resample: bool,
) -> tuple[np.ndarray, list[np.ndarray]]:
    neural_arrays: list[np.ndarray] = []
    for raw in raws:
        if raw is None:
            continue
        data = raw.get_data().astype(np.float32, copy=False)
        neural_arrays.append(data)

    if not neural_arrays:
        raise ValueError("No neural recordings provided; check subject_ids and data paths")

    min_len = min(arr.shape[1] for arr in neural_arrays)
    trimmed = [np.ascontiguousarray(arr[:, :min_len], dtype=np.float32) for arr in neural_arrays]
    neural_stacked = np.concatenate(trimmed, axis=0)
    return neural_stacked, trimmed


def _prepare_audio_neural_alignment(
    audio: np.ndarray, neural: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    min_len = min(audio.shape[0], neural.shape[1])
    if min_len <= 0:
        raise ValueError("Audio and neural data do not share any overlapping samples")

    if audio.shape[0] != min_len or neural.shape[1] != min_len:
        logger.info(
            "Trimming audio/neural length to %d samples for alignment (audio=%d, neural=%d)",
            min_len,
            audio.shape[0],
            neural.shape[1],
        )
    return audio[:min_len], neural[:, :min_len]


def _lag_grid_from_training(training_params: TrainingParams) -> np.ndarray:
    if training_params.lag is not None:
        return np.asarray([float(training_params.lag)], dtype=float)
    return np.arange(
        training_params.min_lag,
        training_params.max_lag,
        training_params.lag_step_size,
        dtype=float,
    )


def _resolve_output_path(base_dir: str, candidate: Optional[str], default_name: str) -> Path:
    if not candidate:
        return Path(base_dir) / default_name
    path = Path(candidate)
    if not path.is_absolute():
        path = Path(base_dir) / path
    return path


def run_volume_level_ridge_from_config(
    experiment_config: ExperimentConfig,
    raws: Sequence,
    df_targets: pd.DataFrame,
    output_dir: str,
    model_dir: Optional[str] = None,
) -> dict[str, np.ndarray]:
    model_params = experiment_config.model_params or {}

    requested_subject_ids = _materialize_sequence(experiment_config.data_params.subject_ids)
    if requested_subject_ids is None:
        subject_ids = list(range(len(raws)))
    else:
        subject_ids = [int(s) for s in requested_subject_ids]
        if len(subject_ids) < len(raws):
            subject_ids.extend(range(len(subject_ids), len(raws)))
        subject_ids = subject_ids[: len(raws)]
    experiment_config.data_params.subject_ids = subject_ids

    audio = df_targets["target"].to_numpy(dtype=np.float32)
    sampling_rate_hz = _infer_target_sr(df_targets, experiment_config.data_params)

    neural_stacked, per_subject_neural = _load_neural_matrix_from_raws(
        raws,
        sampling_rate_hz,
        bool(model_params.get("allow_neural_resample", False)),
    )

    requested_modes = model_params.get("analysis_modes")
    if requested_modes is None:
        analysis_modes = {"pooled_electrodes"}
    else:
        analysis_modes = {str(mode) for mode in requested_modes if mode}
        if not analysis_modes:
            analysis_modes = {"pooled_electrodes"}

    lags = _lag_grid_from_training(experiment_config.training_params)

    alphas = model_params.get("alphas")
    alphas = _materialize_sequence(alphas) if alphas is not None else None
    if alphas is not None and len(alphas) == 0:
        alphas = None
    if alphas is not None:
        alphas = [float(alpha) for alpha in alphas]

    ridge_kwargs = {
        "alphas": alphas,
        "cv_splits": int(model_params.get("cv_splits", 10)),
        "device": model_params.get("device"),
        "random_state": model_params.get("random_state", 0),
        "verbose": bool(model_params.get("verbose", False)),
    }

    results_catalog: dict[str, object] = {}

    output_path = _resolve_output_path(
        output_dir,
        model_params.get("output_csv"),
        "ridge_summary.csv",
    )

    if "pooled_electrodes" in analysis_modes:
        audio_pooled, neural_pooled = _prepare_audio_neural_alignment(audio, neural_stacked)
        pooled_results = ridge_r2_by_lag(
            audio_pooled,
            neural_pooled,
            sampling_rate_hz,
            lags,
            **ridge_kwargs,
        )

        if not pooled_results or len(pooled_results.get("lag_ms", [])) == 0:
            raise RuntimeError("Ridge sweep produced no results; check configuration")

        results_catalog["pooled_electrodes"] = pooled_results

        _ensure_directory(output_path)
        pd.DataFrame(pooled_results).to_csv(output_path, index=False)
        logger.info("Wrote ridge sweep summary to %s", output_path)

        plot_requested = bool(model_params.get("plot", False))
        save_plot_path = model_params.get("save_plot_path")
        if plot_requested or save_plot_path:
            fig, _axes = plot_utils.plot_ridge_results(pooled_results, show=plot_requested)
            if save_plot_path:
                fig_path = _resolve_output_path(output_dir, save_plot_path, "ridge_plot.png")
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(fig_path, bbox_inches="tight")
                logger.info("Saved ridge diagnostic plot to %s", fig_path)

        best_idx = int(np.argmax(pooled_results["r2"]))
        best_alpha = None
        if "alpha" in pooled_results and len(pooled_results["alpha"]) > best_idx:
            best_alpha = pooled_results["alpha"][best_idx]
        logger.info(
            "Best (pooled) lag %.1f ms with R^2=%.4f and alpha=%s",
            pooled_results["lag_ms"][best_idx],
            pooled_results["r2"][best_idx],
            best_alpha,
        )

    per_subject_results: dict[int, dict[str, np.ndarray]] = {}
    if "per_subject" in analysis_modes:
        per_subject_dir = output_path.parent / "per_subject"
        per_subject_dir.mkdir(parents=True, exist_ok=True)

        for subj_idx, neural_subject in enumerate(per_subject_neural):
            try:
                subject_id = subject_ids[subj_idx]
            except IndexError:
                subject_id = subj_idx

            audio_sub, neural_sub = _prepare_audio_neural_alignment(audio, neural_subject)
            subject_results = ridge_r2_by_lag(
                audio_sub,
                neural_sub,
                sampling_rate_hz,
                lags,
                **ridge_kwargs,
            )

            if not subject_results or len(subject_results.get("lag_ms", [])) == 0:
                logger.warning("Subject %s produced no usable lags; skipping CSV export.", subject_id)
                continue

            per_subject_results[subject_id] = subject_results
            results_catalog.setdefault("per_subject", {})[subject_id] = subject_results

            df_subject = pd.DataFrame(subject_results)
            df_subject.insert(0, "subject_id", subject_id)
            df_subject["n_input_channels"] = int(neural_subject.shape[0])
            subject_path = per_subject_dir / f"subject_{subject_id}_ridge.csv"
            df_subject.to_csv(subject_path, index=False)
            logger.info("Wrote per-subject ridge curve for %s to %s", subject_id, subject_path)

            subj_best_idx = int(np.argmax(subject_results["r2"]))
            subj_best_alpha = None
            if "alpha" in subject_results and len(subject_results["alpha"]) > subj_best_idx:
                subj_best_alpha = subject_results["alpha"][subj_best_idx]
            logger.info(
                "Subject %s best lag %.1f ms with R^2=%.4f and alpha=%s",
                subject_id,
                subject_results["lag_ms"][subj_best_idx],
                subject_results["r2"][subj_best_idx],
                subj_best_alpha,
            )

    if "average" in analysis_modes:
        if not per_subject_results:
            logger.warning(
                "Average analysis requested but no per-subject results were generated; skipping average curve."
            )
        else:
            combined_frames: list[pd.DataFrame] = []
            for subject_id, subject_results in per_subject_results.items():
                df = pd.DataFrame(subject_results)
                df["subject_id"] = subject_id
                combined_frames.append(df)

            combined_df = pd.concat(combined_frames, ignore_index=True)
            averaged_df = combined_df.groupby("lag_ms", as_index=False).agg(
                r2_mean=("r2", "mean"),
                r2_std=("r2", "std"),
                train_r2=("train_r2", "mean"),
                alpha=("alpha", "mean"),
                coef_norm=("coef_norm", "mean"),
                n_features=("n_features", "mean"),
                n_subjects=("subject_id", "nunique"),
            )
            averaged_df["r2"] = averaged_df["r2_mean"]
            results_catalog["average"] = {
                "curve": averaged_df,
                "per_subject": combined_df,
            }

            average_path = output_path.parent / "average_ridge.csv"
            averaged_df.to_csv(average_path, index=False)
            logger.info("Wrote average ridge curve to %s", average_path)

    if "pooled_electrodes" in results_catalog and len(results_catalog) == 1:
        return results_catalog["pooled_electrodes"]  # Backwards compatibility

    if results_catalog:
        results_catalog["analysis_modes"] = sorted(analysis_modes)

    return results_catalog
