import math

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


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


def _prepare_ridge_plot_data(results):
    """Validate and prepare ridge results for plotting."""

    if results is None or not isinstance(results, dict):
        raise ValueError("results must be a dictionary produced by ridge_r2_by_lag")

    if "lag_ms" not in results or "r2" not in results:
        raise ValueError("results must contain 'lag_ms' and 'r2' entries")

    lags = np.asarray(results["lag_ms"], dtype=float)
    r2 = np.asarray(results["r2"], dtype=float)

    mask = np.isfinite(lags) & np.isfinite(r2)
    if not np.any(mask):
        raise ValueError("No finite lag/R^2 pairs available for plotting")

    lags = lags[mask]
    r2 = r2[mask]
    order = np.argsort(lags)
    lags = lags[order]
    r2 = r2[order]

    train_r2 = None
    if "train_r2" in results:
        train_vals = np.asarray(results["train_r2"], dtype=float)
        train_r2 = train_vals[mask][order]

    alphas = None
    if "alpha" in results:
        alphas = np.asarray(results["alpha"], dtype=float)[mask][order]

    coef_norm = None
    if "coef_norm" in results:
        coef_norm = np.asarray(results["coef_norm"], dtype=float)[mask][order]

    n_samples = None
    if "n_samples" in results:
        n_samples = np.asarray(results["n_samples"], dtype=float)[mask][order]

    n_features = None
    if "n_features" in results:
        n_features = np.asarray(results["n_features"], dtype=float)[mask][order]

    best_idx = int(np.argmax(r2))

    return {
        "lags": lags,
        "r2": r2,
        "train_r2": train_r2,
        "alphas": alphas,
        "coef_norm": coef_norm,
        "n_samples": n_samples,
        "n_features": n_features,
        "best_idx": best_idx,
    }


def plot_ridge_results(results, *, show: bool = True):
    """Visualise ridge lag search outputs from :func:`volume_level_ridge.ridge_r2_by_lag`.

    Args:
        results: Dictionary returned by ``ridge_r2_by_lag``.
        show: If True (default), call ``plt.show()`` after plotting.

    Returns:
        (fig, axes): Matplotlib figure and axes array for further tweaking.
    """

    data = _prepare_ridge_plot_data(results)
    lags = data["lags"]
    r2 = data["r2"]
    best_idx = data["best_idx"]
    best_lag = lags[best_idx]
    best_r2 = r2[best_idx]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # R^2 across lags
    ax = axes[0]
    ax.plot(lags, r2, marker="o", label="CV R$^2$", color="#1f77b4")
    ax.axvline(best_lag, linestyle="--", color="#ff7f0e", alpha=0.7)
    ax.scatter([best_lag], [best_r2], color="#ff7f0e", zorder=5, label="Best lag")
    if data["train_r2"] is not None:
        ax.plot(lags, data["train_r2"], marker="o", linestyle="--", color="#2ca02c", label="Train R$^2$")
    ax.set_xlabel("Lag (ms)")
    ax.set_ylabel("R$^2$")
    ax.set_title("Lag sweep performance")
    ax.grid(True)
    ax.legend()

    # Regularisation strength
    ax = axes[1]
    if data["alphas"] is not None:
        ax.plot(lags, data["alphas"], marker="o", color="#9467bd")
        ax.set_yscale("log")
        ax.set_ylabel("Alpha (log scale)")
    else:
        ax.text(0.5, 0.5, "No alpha data", ha="center", va="center")
        ax.set_ylabel("Alpha")
    ax.axvline(best_lag, linestyle="--", color="#ff7f0e", alpha=0.7)
    ax.set_xlabel("Lag (ms)")
    ax.set_title("Selected regularisation")
    ax.grid(True)

    # Coefficient norm
    ax = axes[2]
    if data["coef_norm"] is not None:
        ax.plot(lags, data["coef_norm"], marker="o", color="#8c564b")
        ax.set_ylabel("||w||$_2$")
    else:
        ax.text(0.5, 0.5, "No coefficient data", ha="center", va="center")
        ax.set_ylabel("Coefficient norm")
    ax.axvline(best_lag, linestyle="--", color="#ff7f0e", alpha=0.7)
    ax.set_xlabel("Lag (ms)")
    ax.set_title("Coefficient magnitude")
    ax.grid(True)

    # Sample/feature counts
    ax = axes[3]
    plotted_any = False
    if data["n_samples"] is not None:
        ax.plot(lags, data["n_samples"], marker="o", color="#17becf", label="Samples")
        plotted_any = True
    if data["n_features"] is not None:
        ax.plot(lags, data["n_features"], marker="s", color="#7f7f7f", label="Features")
        plotted_any = True
    if not plotted_any:
        ax.text(0.5, 0.5, "No size data", ha="center", va="center")
    ax.axvline(best_lag, linestyle="--", color="#ff7f0e", alpha=0.7)
    ax.set_xlabel("Lag (ms)")
    ax.set_title("Design matrix dimensions")
    ax.grid(True)
    if plotted_any:
        ax.legend()

    fig.suptitle("Ridge regression diagnostics", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))

    if show:
        plt.show()

    return fig, axes


def plot_combined_ridge(results_list, labels=None, *, save_path=None, show: bool = True):
    """Plot R^2 curves for multiple ridge-results dictionaries on a single axes.

    Args:
        results_list: Iterable of result dictionaries or pandas DataFrames (each must have
            keys 'lag_ms' and 'r2' or be convertible via to_dict(orient='list')).
        labels: Optional list of labels for the legend. If omitted, subjects will be numbered.
        save_path: Optional path to save the resulting figure (PNG).
        show: Whether to call plt.show(). Default True.

    Returns:
        fig, ax: Matplotlib figure and axis containing the overlay plot.
    """
    import matplotlib.pyplot as plt

    prepared = []
    for res in results_list:
        if hasattr(res, 'to_dict'):
            rd = res.to_dict(orient='list')
        elif isinstance(res, dict):
            rd = res
        else:
            raise ValueError('Each entry must be a dict or DataFrame')
        data = _prepare_ridge_plot_data(rd)
        prepared.append(data)

    n = len(prepared)
    if labels is None:
        labels = [f"entry_{i+1}" for i in range(n)]

    # Try to map subject-style labels (e.g. 'sub-01' or 'S01') to NYU IDs
    try:
        repo_root = Path(__file__).resolve().parents[0]
        participants_path = repo_root / "data" / "participants.tsv"
        participants_map = {}
        if participants_path.exists():
            try:
                import pandas as _pd

                p_df = _pd.read_csv(participants_path, sep="\t")
                participants_map = dict(zip(p_df["participant_id"], p_df["nyu_id"]))
            except Exception:
                participants_map = {}
        import re
        mapped_labels = []
        for lab in labels:
            m = re.match(r"^[sS]ub[-_]?0*(\d+)$", str(lab)) or re.match(r"^[sS](\d+)$", str(lab))
            if m:
                idx = int(m.group(1))
                subj_id = f"sub-{idx:02d}"
                mapped = participants_map.get(subj_id)
                mapped_labels.append(str(mapped) if mapped is not None else subj_id)
            else:
                mapped_labels.append(lab)
        labels = mapped_labels
    except Exception:
        # If mapping fails for any reason, fall back to original labels
        pass

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    cmap = plt.get_cmap('tab10')
    best_overall = {'r2': -float('inf'), 'lag': None, 'label': None}
    for i, data in enumerate(prepared):
        color = cmap(i % 10)
        # draw continuous line (no dots)
        ax.plot(data['lags'], data['r2'], linestyle='-', label=labels[i], color=color)
        # track best point globally
        bi = data['best_idx']
        r2_val = float(data['r2'][bi])
        lag_val = float(data['lags'][bi])
        if r2_val > best_overall['r2']:
            best_overall.update({'r2': r2_val, 'lag': lag_val, 'label': labels[i]})

    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('R$^2$')
    ax.set_title('Overlay: Ridge CV R$^2$ by lag (subjects / average / pooled)')
    ax.grid(True)
    ax.legend(loc='best', fontsize='small')
    # Annotate highest R^2 across plotted entries
    if best_overall['lag'] is not None:
        ann_text = f"Best R^2={best_overall['r2']:.3f} at {best_overall['lag']:.0f} ms ({best_overall['label']})"
        ax.annotate(ann_text, xy=(0.98, 0.02), xycoords='axes fraction', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7), fontsize='small')
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()
    return fig, ax
