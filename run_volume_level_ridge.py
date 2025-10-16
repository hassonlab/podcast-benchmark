from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
import torch

from config import ExperimentConfig
from config_utils import load_config_with_overrides, parse_override_args
from vol_lvl_ridge_utils import (
    compute_window_hop,
    load_log_transformed_high_gamma,
    sliding_window_rms,
)
from plot_utils import plot_ridge_results

# Import registries so decorators run.
import registry  # noqa: F401
import task_utils  # noqa: F401


def _parse_args():
    # Parse CLI config path and optional overrides.
    parser = argparse.ArgumentParser(description="Run the volume-level ridge pipeline.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args, unknown = parser.parse_known_args()
    overrides = parse_override_args(unknown)
    return args.config, overrides


def _format_trial_name(cfg: ExperimentConfig) -> str:
    # Build a unique trial identifier from config fields and timestamp.
    base = cfg.trial_name or cfg.model_constructor_name or "volume_level_ridge"
    if cfg.format_fields:
        values = []
        for path in cfg.format_fields:
            values.append(_traverse_config(cfg, path))
        base = base.format(*values)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    return f"{base}_{timestamp}"


def _traverse_config(cfg: ExperimentConfig, path: str):
    # Walk dot-separated attribute path on the config dataclasses.
    current = cfg
    for part in path.split("."):
        current = getattr(current, part)
    return current


def _prepare_output_dirs(cfg: ExperimentConfig, trial_name: str) -> Dict[str, Path]:
    # Create run-specific output folders and persist the resolved config.
    output_dir = Path(cfg.output_dir) / trial_name
    model_dir = Path(cfg.model_dir) / trial_name
    tensorboard_dir = Path(cfg.tensorboard_dir) / trial_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.yml", "w", encoding="utf-8") as fh:
        yaml.dump(asdict(cfg), fh, default_flow_style=False)
    return {
        "output_dir": output_dir,
        "model_dir": model_dir,
        "tensorboard_dir": tensorboard_dir,
    }


def _resolve_lags(training_params) -> np.ndarray:
    # Generate the array of lag values to scan during ridge CV.
    if training_params.lag is not None:
        return np.asarray([training_params.lag], dtype=float)
    return np.arange(
        training_params.min_lag,
        training_params.max_lag,
        training_params.lag_step_size,
        dtype=float,
    )


def _align_for_shift(neural: np.ndarray, audio: np.ndarray, shift: int) -> tuple[np.ndarray, np.ndarray]:
    # Offset neural and audio sequences to realize a specific lag.
    if shift > 0:
        X = neural[:, shift:].T
        y = audio[:-shift]
    elif shift < 0:
        step = abs(shift)
        X = neural[:, :-step].T
        y = audio[step:]
    else:
        X = neural.T
        y = audio
    cutoff = min(X.shape[0], y.shape[0])
    return X[:cutoff], y[:cutoff]


def _resolve_device(model_params: dict) -> torch.device:
    # Choose CPU or GPU device for the ridge solve based on config.
    device_str = model_params.get("device")
    if device_str is None:
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for ridge regression but no GPU is available.")
    return device


def _standardize_features(X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # Z-score features and return transformed data with stats for reuse.
    mean = X.mean(dim=0, keepdim=True)
    std = X.std(dim=0, unbiased=False, keepdim=True)
    std = torch.where(std == 0, torch.ones_like(std), std)
    X_standardized = (X - mean) / std
    return X_standardized, mean, std


def _solve_ridge_closed_form(X: torch.Tensor, y: torch.Tensor, alpha: float) -> torch.Tensor:
    # Solve ridge regression weights using the analytical solution.
    n_features = X.shape[1]
    eye = torch.eye(n_features, device=X.device, dtype=X.dtype)
    return torch.linalg.solve(X.T @ X + alpha * eye, X.T @ y)


def _r2_score_torch(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    # Compute coefficient of determination in PyTorch.
    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - y_true.mean()) ** 2)
    if torch.isclose(ss_tot, torch.tensor(0.0, device=y_true.device, dtype=y_true.dtype)):
        return float("nan")
    return float(1.0 - (ss_res / ss_tot).item())


def _ridge_lag_sweep(
    neural: np.ndarray,
    audio: np.ndarray,
    effective_sr: float,
    lags_ms: np.ndarray,
    cv_splits: int,
    alphas: Optional[Iterable[float]],
    device: torch.device,
) -> pd.DataFrame:
    # Run cross-validated ridge fits across lag offsets and alphas.
    if cv_splits < 2:
        raise ValueError("cv_splits must be at least 2 for cross-validation.")
    lag_values = np.asarray(lags_ms, dtype=float)
    if lag_values.size == 0:
        return pd.DataFrame()
    alpha_values = (
        np.logspace(-4, 4, 17, dtype=float)
        if alphas is None
        else np.asarray(list(alphas), dtype=float)
    )
    results: List[dict] = []
    kfold = KFold(n_splits=cv_splits, shuffle=True, random_state=0)
    for lag in lag_values:
        shift = int(round(lag / 1000.0 * effective_sr))
        X, y = _align_for_shift(neural, audio, shift)
        if X.shape[0] <= cv_splits or y.size <= cv_splits:
            continue
        best_alpha = None
        best_score = -np.inf
        X_tensor = torch.from_numpy(X).to(device=device, dtype=torch.float32)
        y_tensor = torch.from_numpy(y).to(device=device, dtype=torch.float32).unsqueeze(1)
        for alpha in alpha_values:
            fold_scores = []
            for train_idx, val_idx in kfold.split(X):
                train_idx_t = torch.as_tensor(train_idx, device=device)
                val_idx_t = torch.as_tensor(val_idx, device=device)
                X_train = torch.index_select(X_tensor, 0, train_idx_t)
                X_val = torch.index_select(X_tensor, 0, val_idx_t)
                y_train = torch.index_select(y_tensor, 0, train_idx_t)
                y_val = torch.index_select(y_tensor, 0, val_idx_t)
                y_train_mean = y_train.mean()
                y_train_centered = y_train - y_train_mean
                X_train_std, mean_t, std_t = _standardize_features(X_train)
                X_val_std = (X_val - mean_t) / std_t
                beta = _solve_ridge_closed_form(X_train_std, y_train_centered, float(alpha))
                y_pred = X_val_std @ beta + y_train_mean
                fold_scores.append(_r2_score_torch(y_val, y_pred))
            score = float(np.nanmean(fold_scores))
            if score > best_score:
                best_score = score
                best_alpha = float(alpha)
        if best_alpha is None:
            continue
        X_full_std, mean_full, std_full = _standardize_features(X_tensor)
        y_full_mean = y_tensor.mean()
        y_full_centered = y_tensor - y_full_mean
        beta = _solve_ridge_closed_form(X_full_std, y_full_centered, best_alpha)
        y_pred = X_full_std @ beta + y_full_mean
        records = {
            "lag_ms": float(lag),
            "r2": best_score,
            "alpha": best_alpha,
            "coef_norm": float(torch.linalg.norm(beta).cpu().item()),
            "n_samples": int(y.shape[0]),
            "n_features": int(X.shape[1]),
            "train_r2": _r2_score_torch(y_tensor, y_pred),
        }
        results.append(records)
    return pd.DataFrame(results)


def _merge_lengths(datasets: List[dict]) -> None:
    # Truncate each subject payload so all share the shortest duration.
    if not datasets:
        return
    min_len = min(entry["audio"].shape[0] for entry in datasets)
    for entry in datasets:
        entry["audio"] = entry["audio"][:min_len]
        entry["neural"] = entry["neural"][:, :min_len]


def _prepare_subject_datasets(
    cfg: ExperimentConfig,
    audio_targets: np.ndarray,
    window_ms: float,
    hop_ms: float,
    log_params: Optional[dict],
    apply_log: bool,
) -> List[dict]:
    # Load, preprocess, and window neural data per subject.
    model_params = cfg.model_params
    if model_params.get("allow_neural_resample"):
        raise NotImplementedError("Neural resampling is not implemented in this pipeline.")
    payloads = load_log_transformed_high_gamma(
        cfg.data_params,
        log_params=log_params,
        apply_log=apply_log,
    )
    drop_last = bool(model_params.get("drop_last_neural_channel"))
    datasets: List[dict] = []
    for subject_id in cfg.data_params.subject_ids:
        payload = payloads[subject_id]
        neural = np.asarray(payload["log_highgamma"], dtype=np.float32)
        if drop_last and neural.shape[0] > 1:
            neural = neural[:-1]
        window_samples, hop_samples, effective_sr = compute_window_hop(
            payload["sampling_rate"],
            window_ms,
            hop_ms,
        )
        neural_windows = sliding_window_rms(neural, window_samples, hop_samples)
        min_len = min(neural_windows.shape[1], audio_targets.shape[0])
        datasets.append(
            {
                "subject_id": int(subject_id),
                "neural": neural_windows[:, :min_len],
                "audio": audio_targets[:min_len],
                "effective_sr": effective_sr,
            }
        )
    _merge_lengths(datasets)
    return datasets


def _compute_modes(
    datasets: List[dict],
    lags_ms: np.ndarray,
    model_params: dict,
) -> Dict[str, object]:
    # Aggregate ridge results for per-subject, average, and pooled modes.
    cv_splits = int(model_params.get("cv_splits", 10))
    alphas = model_params.get("alphas")
    modes = model_params.get(
        "analysis_modes",
        ["per_subject", "average", "pooled_electrodes"],
    )
    device = _resolve_device(model_params)
    results: Dict[str, object] = {}
    per_subject: Dict[int, pd.DataFrame] = {}
    if "per_subject" in modes:
        for entry in datasets:
            df = _ridge_lag_sweep(
                entry["neural"],
                entry["audio"],
                entry["effective_sr"],
                lags_ms,
                cv_splits,
                alphas,
                device,
            )
            if not df.empty:
                df["subject_id"] = entry["subject_id"]
                per_subject[entry["subject_id"]] = df
        results["per_subject"] = per_subject
    if "average" in modes:
        if not per_subject:
            raise RuntimeError("Average analysis requested but no subject results were produced.")
        combined = pd.concat(per_subject.values(), ignore_index=True)
        grouped = combined.groupby("lag_ms", as_index=False).agg(
            r2=("r2", "mean"),
            r2_std=("r2", "std"),
            alpha_mean=("alpha", "mean"),
            train_r2=("train_r2", "mean"),
            n_subjects=("subject_id", "nunique"),
        )
        results["average"] = grouped
    if "pooled_electrodes" in modes:
        pooled_neural = np.concatenate([entry["neural"] for entry in datasets], axis=0)
        pooled_audio = datasets[0]["audio"]
        pooled_sr = datasets[0]["effective_sr"]
        pooled_df = _ridge_lag_sweep(
            pooled_neural,
            pooled_audio,
            pooled_sr,
            lags_ms,
            cv_splits,
            alphas,
            device,
        )
        pooled_df["subject_id"] = "pooled"
        results["pooled_electrodes"] = pooled_df
    return results


def _extract_audio(cfg: ExperimentConfig) -> tuple[np.ndarray, float, float]:
    # Retrieve windowed audio targets and RMS parameters from task getter.
    getter = registry.task_data_getter_registry[cfg.task_name]
    df = getter(cfg.data_params)
    targets = df["target"].to_numpy(dtype=np.float32)
    window_params = df.attrs.get("window_params")
    task_params = cfg.data_params.task_params or {}
    if window_params:
        window_ms = float(window_params["window_ms"])
        hop_ms = float(window_params["hop_ms"])
    else:
        window_ms = float(task_params.get("window_size", 0.0) or 0.0)
        hop_ms = float(task_params.get("hop_size", window_ms) or 0.0)
    if window_ms <= 0 or hop_ms <= 0:
        raise ValueError("Volume-level pipeline requires positive window and hop settings.")
    return targets, window_ms, hop_ms


def run_volume_level_ridge(cfg: ExperimentConfig, output_context: Dict[str, Path]) -> Dict[str, object]:
    # Orchestrate full ridge pipeline from loading to output writes.
    if cfg.task_name != "volume_level_encoding_task":
        raise ValueError(
            "Volume-level ridge pipeline expects task_name='volume_level_encoding_task'."
        )
    audio_targets, window_ms, hop_ms = _extract_audio(cfg)
    model_params = cfg.model_params
    apply_log = bool(model_params.get("neural_log_compress", True))
    log_params = None
    if apply_log:
        log_params = dict(model_params.get("neural_log_params", {}) or {})
        if model_params.get("neural_log_eps_scale") is not None:
            log_params["epsilon_scale"] = float(model_params["neural_log_eps_scale"])
    datasets = _prepare_subject_datasets(
        cfg,
        audio_targets,
        window_ms,
        hop_ms,
        log_params,
        apply_log,
    )
    lags_ms = _resolve_lags(cfg.training_params)
    results = _compute_modes(datasets, lags_ms, model_params)
    _write_outputs(results, output_context["output_dir"], model_params)
    return results


def _write_outputs(results: Dict[str, object], output_dir: Path, model_params: dict) -> None:
    # Save per-mode ridge summaries and combined CSV artifacts.
    records: List[pd.DataFrame] = []
    per_subject = results.get("per_subject", {})
    for subject_id, df in per_subject.items():
        df.to_csv(output_dir / f"subject_{int(subject_id):02d}_ridge.csv", index=False)
        temp = df.copy()
        temp["mode"] = "per_subject"
        records.append(temp)
    if "average" in results:
        avg_df = results["average"].copy()
        avg_df.insert(0, "subject_id", "mean")
        avg_df["mode"] = "average"
        avg_df.to_csv(output_dir / "average_ridge.csv", index=False)
        records.append(avg_df)
    if "pooled_electrodes" in results:
        pooled_df = results["pooled_electrodes"].copy()
        pooled_df["mode"] = "pooled_electrodes"
        pooled_df.to_csv(output_dir / "pooled_ridge.csv", index=False)
        records.append(pooled_df)
    if records:
        combined = pd.concat(records, ignore_index=True)
        summary_name = model_params.get("output_csv", "ridge_summary.csv")
        combined.to_csv(output_dir / summary_name, index=False)

    # Optionally create plots for each mode
    try:
        do_plot = bool(model_params.get("plot", False))
    except Exception:
        do_plot = False
    if do_plot:
        save_path = model_params.get("save_plot_path", "ridge_plot.png")
        # Per-subject plots
        for subject_id, df in per_subject.items():
            try:
                fig, axes = plot_ridge_results(df.to_dict(orient="list"), show=False)
                fig.savefig(output_dir / f"subject_{int(subject_id):02d}_ridge.png")
                fig.clf()
            except Exception as exc:  # keep pipeline robust
                print(f"Failed to plot subject {subject_id}: {exc}")
        # Average plot
        if "average" in results:
            try:
                fig, axes = plot_ridge_results(results["average"].to_dict(orient="list"), show=False)
                fig.savefig(output_dir / f"average_{save_path}")
                fig.clf()
            except Exception as exc:
                print(f"Failed to plot average results: {exc}")
        # Pooled plot
        if "pooled_electrodes" in results:
            try:
                fig, axes = plot_ridge_results(results["pooled_electrodes"].to_dict(orient="list"), show=False)
                fig.savefig(output_dir / f"pooled_{save_path}")
                fig.clf()
            except Exception as exc:
                print(f"Failed to plot pooled results: {exc}")


def main():
    # CLI entry point that parses config and executes the pipeline.
    config_path, overrides = _parse_args()
    cfg = load_config_with_overrides(config_path, overrides)
    trial_name = _format_trial_name(cfg)
    context = _prepare_output_dirs(cfg, trial_name)
    run_volume_level_ridge(cfg, context)


if __name__ == "__main__":
    main()
