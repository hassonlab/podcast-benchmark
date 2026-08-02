import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _resolve_lag_metric_column(results, metric):
    candidates = [metric]
    if not metric.endswith("_mean"):
        candidates.extend([f"{metric}_mean", f"test_{metric}_mean"])
    for candidate in candidates:
        if candidate in results.columns:
            return candidate
    raise ValueError(
        f"Metric '{metric}' was not found. Available mean metrics: "
        f"{sorted(column for column in results.columns if column.endswith('_mean'))}"
    )


def plot_lag_results(data, metric="test_roc_auc", ax=None):
    """Plot a lag-level metric with optional fold-standard-deviation bands.

    Args:
        data: A lag results DataFrame or path to ``lag_performance.csv``.
        metric: Metric column or friendly stem such as ``"test_roc_auc"`` or
            ``"roc_auc"``.
        ax: Optional matplotlib axes to draw on.

    Returns:
        Tuple of ``(figure, axes)``.
    """

    if isinstance(data, (str, Path)):
        results = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        results = data.copy()
    else:
        raise TypeError("data must be a pandas DataFrame or CSV path")

    if "lags" not in results.columns:
        raise ValueError("Lag results must contain a 'lags' column")
    if results.empty:
        raise ValueError("Lag results are empty")

    mean_column = _resolve_lag_metric_column(results, metric)
    std_column = (
        mean_column.removesuffix("_mean") + "_std"
        if mean_column.endswith("_mean")
        else None
    )

    if ax is None:
        figure, ax = plt.subplots(figsize=(7, 4))
    else:
        figure = ax.figure

    if "run_unit" not in results.columns:
        results["run_unit"] = "combined"

    for run_unit, unit_results in results.groupby("run_unit", sort=False):
        unit_results = unit_results.sort_values("lags")
        lags = pd.to_numeric(unit_results["lags"], errors="raise").to_numpy()
        means = pd.to_numeric(unit_results[mean_column], errors="raise").to_numpy()
        (line,) = ax.plot(lags, means, marker="o", label=str(run_unit))
        if std_column and std_column in unit_results.columns:
            stds = pd.to_numeric(
                unit_results[std_column], errors="coerce"
            ).to_numpy()
            if np.isfinite(stds).any():
                ax.fill_between(
                    lags,
                    means - stds,
                    means + stds,
                    color=line.get_color(),
                    alpha=0.2,
                )

    label_stem = mean_column.removeprefix("test_").removesuffix("_mean")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel(format_metric_name(label_stem))
    ax.set_title(f"{format_metric_name(label_stem)} across lags")
    ax.axvline(0, color="0.5", linewidth=1, linestyle="--")
    ax.grid(alpha=0.25)
    if results["run_unit"].nunique() > 1:
        ax.legend(title="Run unit")
    figure.tight_layout()
    return figure, ax


def extract_metric_names(history_dict):
    """
    Extract unique metric names from history dictionary keys.

    Args:
        history_dict: Dictionary with keys like 'train_metric', 'val_metric', etc.

    Returns:
        list: Unique metric names (without phase prefixes)
    """
    metric_names = set()

    for key in history_dict.keys():
        if key in ["num_epochs"]:  # Skip non-metric keys
            continue

        # Split on first underscore to separate phase from metric name
        parts = key.split("_", 1)
        if len(parts) == 2 and parts[0] in ["train", "val", "test"]:
            metric_names.add(parts[1])
        elif key in ["train_loss", "val_loss"]:  # Handle legacy loss keys
            metric_names.add("loss")

    return sorted(list(metric_names))


def format_metric_name(metric_name):
    """
    Format metric name for display in plots.

    Args:
        metric_name: Raw metric name (e.g., 'mse', 'cosine_sim')

    Returns:
        str: Formatted name for display (e.g., 'MSE', 'Cosine Similarity')
    """
    # Special cases for common metrics
    formatting_map = {
        "mse": "MSE",
        "loss": "Loss",
        "cosine": "Cosine Similarity",
        "cosine_sim": "Cosine Similarity",
        "cosine_dist": "Cosine Distance",
        "nll_embedding": "NLL Embedding",
        "auc_roc": "AUC-ROC",
        "roc_auc": "AUC-ROC",
        "perplexity": "Perplexity",
    }

    if metric_name in formatting_map:
        return formatting_map[metric_name]

    # General formatting: replace underscores and capitalize
    formatted = metric_name.replace("_", " ").title()
    return formatted


def get_subplot_layout(n_metrics):
    """
    Determine optimal subplot layout for given number of metrics.

    Args:
        n_metrics: Number of metrics to plot

    Returns:
        tuple: (rows, cols) for subplot layout
    """
    if n_metrics <= 0:
        return (1, 1)
    elif n_metrics == 1:
        return (1, 1)
    elif n_metrics == 2:
        return (1, 2)
    elif n_metrics <= 4:
        return (2, 2)
    elif n_metrics <= 6:
        return (2, 3)
    elif n_metrics <= 9:
        return (3, 3)
    else:
        # For larger numbers, prefer wider layouts
        cols = math.ceil(math.sqrt(n_metrics))
        rows = math.ceil(n_metrics / cols)
        return (rows, cols)


def plot_training_history(history, fold=None):
    """
    Plot the training and validation metrics from training history.

    Args:
        history: Dictionary containing training history with keys like 'train_metric', 'val_metric'
        fold: Fold number (optional)
    """
    # Extract unique metric names
    metric_names = extract_metric_names(history)

    if not metric_names:
        print("No metrics found in history dictionary")
        return

    # Determine subplot layout
    rows, cols = get_subplot_layout(len(metric_names))

    # Create figure with dynamic subplots
    fig_width = min(cols * 7, 20)  # Cap width to avoid overly wide plots
    fig_height = rows * 5
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Ensure axes is always a list for consistent indexing
    if len(metric_names) == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = axes.flatten()

    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]

        # Get metric data for train and val
        train_key = f"train_{metric_name}"
        val_key = f"val_{metric_name}"

        # Handle legacy loss keys
        if metric_name == "loss":
            train_key = "train_loss"
            val_key = "val_loss"

        # Plot data if available
        if train_key in history:
            ax.plot(
                history[train_key], label=f"Training {format_metric_name(metric_name)}"
            )
        if val_key in history:
            ax.plot(
                history[val_key], label=f"Validation {format_metric_name(metric_name)}"
            )

        # Set labels and title
        ax.set_xlabel("Epoch")
        ax.set_ylabel(format_metric_name(metric_name))

        title = f"Training and Validation {format_metric_name(metric_name)}"
        if fold is not None:
            title = f"Fold {fold}: {title}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True)

    # Hide unused subplots
    for i in range(len(metric_names), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_cv_results(cv_results):
    """
    Plot cross-validation results for all available metrics.

    Args:
        cv_results: Dictionary containing cross-validation results with keys like 'phase_metric'
    """
    # Extract unique metric names
    metric_names = extract_metric_names(cv_results)

    if not metric_names:
        print("No metrics found in cv_results dictionary")
        return

    # Determine subplot layout
    rows, cols = get_subplot_layout(len(metric_names))

    # Create figure with dynamic subplots
    fig_width = min(cols * 7, 20)  # Cap width to avoid overly wide plots
    fig_height = rows * 5
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height))

    # Ensure axes is always a list for consistent indexing
    if len(metric_names) == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    else:
        axes = axes.flatten()

    # Plot each metric
    for i, metric_name in enumerate(metric_names):
        ax = axes[i]

        # Get metric data for all phases
        phases_data = {}
        for phase in ["train", "val", "test"]:
            key = f"{phase}_{metric_name}"
            # Handle legacy loss keys
            if metric_name == "loss" and key not in cv_results:
                if f"{phase}_loss" in cv_results:
                    key = f"{phase}_loss"

            if key in cv_results and cv_results[key]:
                phases_data[phase] = cv_results[key]

        if not phases_data:
            # No data for this metric, skip
            ax.set_visible(False)
            continue

        # Determine number of folds from first available phase
        first_phase_data = list(phases_data.values())[0]
        folds = range(1, len(first_phase_data) + 1)

        # Plot each phase
        phase_labels = {"train": "Training", "val": "Validation", "test": "Test"}

        for phase, data in phases_data.items():
            if len(data) == len(folds):  # Ensure data length matches
                label = f"{phase_labels[phase]} {format_metric_name(metric_name)}"
                ax.plot(folds, data, "o-", label=label)

        # Set labels and title
        ax.set_xlabel("Fold")
        ax.set_ylabel(format_metric_name(metric_name))
        ax.set_title(f"Cross-Validation {format_metric_name(metric_name)}")
        ax.set_xticks(folds)
        ax.legend()
        ax.grid(True)

    # Hide unused subplots
    for i in range(len(metric_names), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.show()
