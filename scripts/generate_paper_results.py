#!/usr/bin/env python3
"""Generate paper-ready result figures and summary tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize


CONDITIONS = ("super_subject", "per_subject")
PER_REGION_CONDITION = "per_region"
DEFAULT_NILEARN_DATA_DIR = (
    Path("processed_data") / "atlas_region_visualization" / "nilearn_data"
)
REGION_LEVEL_ORDER = ("EAC", "PC", "PRC", "IFG", "MTG", "ITG", "TPJ", "TP", "RIGHT")
DEFAULT_TASK_GROUP_ORDER = ("Semantic", "Syntactic", "Auditory", "Mixed")
BAR_SUMMARY_GRID_ROWS = 2
BAR_SUMMARY_GRID_COLS = 5
BAR_SUMMARY_GROUP_LAYOUT = {
    "Mixed": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "Semantic": ((0, 2), (1, 2)),
    "Syntactic": ((0, 3), (1, 3)),
    "Acoustic": ((0, 4), (1, 4)),
}
BAR_SUMMARY_GROUP_ALIASES = {
    "Mixed": ("Mixed",),
    "Semantic": ("Semantic",),
    "Syntactic": ("Syntactic",),
    "Acoustic": ("Acoustic", "Auditory"),
}
DEFAULT_COLORS = {
    "baseline": "#4C78A8",
    "diver": "#F58518",
    "brainbert": "#54A24B",
    "popt": "#E45756",
}


@dataclass(frozen=True)
class MetricConfig:
    column: str
    higher_is_better: bool
    label: str
    min_value: float | None = None
    max_value: float | None = None
    negate: bool = False


@dataclass(frozen=True)
class ResultSpec:
    model: str
    task: str
    condition: str
    path: Path


@dataclass(frozen=True)
class DestrieuxSurfaceAtlas:
    labels: Sequence[str]
    maps: Mapping[str, np.ndarray]
    mesh: object
    sulcal: object


def read_config(path: Path) -> Mapping:
    with path.open("r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, Mapping):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def load_current_style_run(run_dir: Path) -> pd.DataFrame:
    """Load a current-style result run.

    Super-subject runs contain ``lag_performance.csv`` directly. Per-subject
    runs contain one ``subject_*/lag_performance.csv`` per subject and are
    averaged by lag across subjects.
    """

    run_dir = Path(run_dir)
    root_csv = run_dir / "lag_performance.csv"
    if root_csv.exists():
        return pd.read_csv(root_csv)

    subject_frames = []
    for csv_path in sorted(run_dir.glob("subject_*/lag_performance.csv")):
        df = pd.read_csv(csv_path)
        subject_frames.append(df)

    if not subject_frames:
        raise FileNotFoundError(
            f"Expected {root_csv} or subject_*/lag_performance.csv files under {run_dir}"
        )

    combined = pd.concat(subject_frames, ignore_index=True)
    numeric_columns = [
        column
        for column in combined.select_dtypes(include="number").columns
        if column != "lags"
    ]
    averaged = (
        combined.groupby("lags", as_index=False)[numeric_columns]
        .mean()
        .sort_values("lags")
        .reset_index(drop=True)
    )
    return averaged


def iter_result_specs(config: Mapping) -> Iterable[ResultSpec]:
    results = config.get("results", {})
    if not isinstance(results, Mapping):
        raise ValueError("Config key 'results' must be a mapping")

    for model, tasks in results.items():
        if not isinstance(tasks, Mapping):
            continue
        for task, conditions in tasks.items():
            if not isinstance(conditions, Mapping):
                continue
            for condition in CONDITIONS:
                path = conditions.get(condition)
                if path:
                    yield ResultSpec(
                        model=model, task=task, condition=condition, path=Path(path)
                    )


def iter_per_region_result_specs(config: Mapping) -> Iterable[ResultSpec]:
    results = config.get("results", {})
    if not isinstance(results, Mapping):
        raise ValueError("Config key 'results' must be a mapping")

    for model, tasks in results.items():
        if not isinstance(tasks, Mapping):
            continue
        for task, conditions in tasks.items():
            if not isinstance(conditions, Mapping):
                continue
            path = conditions.get(PER_REGION_CONDITION)
            if path:
                yield ResultSpec(
                    model=model,
                    task=task,
                    condition=PER_REGION_CONDITION,
                    path=Path(path),
                )


def load_results(config: Mapping) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    loaded: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    for spec in iter_result_specs(config):
        loaded.setdefault(spec.condition, {}).setdefault(spec.task, {})[spec.model] = (
            load_current_style_run(spec.path)
        )
    return loaded


def normalize_region_name(region_dir_name: str) -> str:
    name = region_dir_name.removeprefix("region_")
    return name.upper()


def load_per_region_run(run_dir: Path) -> Dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    loaded = {}
    for csv_path in sorted(run_dir.glob("region_*/lag_performance.csv")):
        region = normalize_region_name(csv_path.parent.name)
        loaded[region] = pd.read_csv(csv_path)

    if not loaded:
        raise FileNotFoundError(
            f"Expected region_*/lag_performance.csv files under {run_dir}"
        )
    return loaded


def load_per_region_results(
    config: Mapping,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    loaded: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    for spec in iter_per_region_result_specs(config):
        loaded.setdefault(spec.task, {})[spec.model] = load_per_region_run(spec.path)
    return loaded


def get_metric_config(config: Mapping, task: str) -> MetricConfig:
    metrics = config.get("metrics", {})
    if not isinstance(metrics, Mapping) or task not in metrics:
        raise KeyError(f"No metric configuration found for task '{task}'")
    metric = metrics[task]
    min_value = _optional_float(metric.get("min", metric.get("vmin")))
    max_value = _optional_float(metric.get("max", metric.get("vmax")))
    if min_value is not None and max_value is not None and min_value >= max_value:
        raise ValueError(f"Metric bounds for task '{task}' must satisfy min < max")
    return MetricConfig(
        column=metric["column"],
        higher_is_better=bool(metric.get("higher_is_better", True)),
        label=metric.get("label", metric["column"]),
        min_value=min_value,
        max_value=max_value,
        negate=bool(metric.get("negate", metric.get("multiply_by_negative", False))),
    )


def _optional_float(value) -> float | None:
    if value is None:
        return None
    return float(value)


def apply_metric_ylim(ax: plt.Axes, metric: MetricConfig) -> None:
    if metric.min_value is None and metric.max_value is None:
        return
    current_lower, current_upper = ax.get_ylim()
    lower = metric.min_value if metric.min_value is not None else current_lower
    upper = metric.max_value if metric.max_value is not None else current_upper
    ax.set_ylim(lower, upper)


def metric_norm(values: Sequence[float], metric: MetricConfig) -> Normalize:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    lower = metric.min_value
    upper = metric.max_value
    if lower is None:
        lower = min(finite_values) if finite_values else 0.0
    if upper is None:
        upper = max(finite_values) if finite_values else 1.0
    if lower == upper:
        pad = abs(lower) * 0.05 or 0.05
        lower -= pad
        upper += pad
    return Normalize(vmin=lower, vmax=upper, clip=True)


def metric_values(df: pd.DataFrame, metric: MetricConfig) -> pd.Series:
    if metric.column not in df.columns:
        raise KeyError(f"Metric column '{metric.column}' is missing")
    values = pd.to_numeric(df[metric.column], errors="coerce")
    return -values if metric.negate else values


def select_best_lag(df: pd.DataFrame, metric: MetricConfig) -> pd.Series:
    values = metric_values(df, metric)
    if values.notna().sum() == 0:
        raise ValueError(f"Metric column '{metric.column}' has no numeric values")
    idx = values.idxmax() if metric.higher_is_better else values.idxmin()
    row = df.loc[idx].copy()
    row[metric.column] = values.loc[idx]
    return row


def metric_std_column(metric_column: str) -> str | None:
    if metric_column.endswith("_mean"):
        return f"{metric_column.removesuffix('_mean')}_std"
    return None


def best_lag_std_value(best: pd.Series, metric: MetricConfig) -> float:
    std_column = metric_std_column(metric.column)
    if std_column is None or std_column not in best.index:
        return float("nan")
    return float(pd.to_numeric(pd.Series([best[std_column]]), errors="coerce").iloc[0])


def best_lag_rows(
    condition_results: Mapping[str, Mapping[str, pd.DataFrame]],
    metrics: Mapping[str, MetricConfig],
) -> pd.DataFrame:
    rows = []
    for task, model_results in sorted(condition_results.items()):
        metric = metrics[task]
        for model, df in sorted(model_results.items()):
            best = select_best_lag(df, metric)
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "metric_min": metric.min_value,
                    "metric_max": metric.max_value,
                    "metric_negate": metric.negate,
                    "value": float(best[metric.column]),
                    "std": best_lag_std_value(best, metric),
                    "lag": best["lags"],
                    "higher_is_better": metric.higher_is_better,
                }
            )
    return pd.DataFrame(rows)


def best_region_lag_rows(
    region_results: Mapping[str, pd.DataFrame],
    metric: MetricConfig,
) -> pd.DataFrame:
    rows = []
    for region, df in sorted(region_results.items()):
        best = select_best_lag(df, metric)
        rows.append(
            {
                "region": region,
                "metric": metric.column,
                "metric_label": metric.label,
                "metric_min": metric.min_value,
                "metric_max": metric.max_value,
                "metric_negate": metric.negate,
                "value": float(best[metric.column]),
                "std": best_lag_std_value(best, metric),
                "lag": best["lags"],
                "higher_is_better": metric.higher_is_better,
            }
        )
    return pd.DataFrame(rows)


def best_model_by_task(summary: pd.DataFrame) -> Dict[tuple, str]:
    winners = {}
    group_columns = (
        ["condition", "task"] if "condition" in summary.columns else ["task"]
    )
    for keys, group in summary.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        higher = bool(group["higher_is_better"].iloc[0])
        idx = group["value"].idxmax() if higher else group["value"].idxmin()
        winners[keys] = str(summary.loc[idx, "model"])
    return winners


def format_value(value: float, lag) -> str:
    return f"{value:.3f} ({lag:g} ms)"


def summary_wide(
    summary: pd.DataFrame,
    config: Mapping | None = None,
    bold: bool = False,
    latex: bool = False,
) -> pd.DataFrame:
    config = config or {}
    winners = best_model_by_task(summary) if bold else {}
    models = sorted(summary["model"].unique())
    group_columns = (
        ["condition", "task"] if "condition" in summary.columns else ["task"]
    )
    rows = []
    for keys, group in summary.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        by_model = {item["model"]: item for item in group.to_dict("records")}
        for model in models:
            column = display_model_name(config, model)
            if model not in by_model:
                row[column] = ""
                continue
            item = by_model[model]
            text = format_value(item["value"], item["lag"])
            if bold and winners.get(keys) == model:
                text = f"\\textbf{{{text}}}" if latex else f"**{text}**"
            row[column] = text
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[*group_columns, *[display_model_name(config, model) for model in models]],
    )


def write_summary_tables(
    summary: pd.DataFrame, output_dir: Path, formats: Sequence[str], config: Mapping
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_table = summary_wide(summary, config, bold=False)
    display_table = summary_wide(summary, config, bold=True)
    latex_table = summary_wide(summary, config, bold=True, latex=True)

    if "csv" in formats:
        csv_table.to_csv(output_dir / "best_lag_summary.csv", index=False)
    if "markdown" in formats or "md" in formats:
        (output_dir / "best_lag_summary.md").write_text(
            to_markdown_table(display_table)
        )
    if "latex" in formats or "tex" in formats:
        (output_dir / "best_lag_summary.tex").write_text(
            latex_table.to_latex(index=False, escape=False)
        )


def to_markdown_table(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    rows = [[str(value) for value in row] for row in df.to_numpy()]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def model_colors(models: Iterable[str], config: Mapping) -> Dict[str, str]:
    configured = config.get("colors", {})
    colors = dict(DEFAULT_COLORS)
    if isinstance(configured, Mapping):
        colors.update(configured)

    cmap = plt.get_cmap("tab10")
    assigned = {}
    for idx, model in enumerate(sorted(models)):
        assigned[model] = colors.get(model, cmap(idx % 10))
    return assigned


def save_figure(fig: plt.Figure, output_base: Path, formats: Sequence[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plotting_config(config: Mapping) -> Mapping:
    configured = config.get("plotting", {})
    return configured if isinstance(configured, Mapping) else {}


def include_bar_error_bars(config: Mapping) -> bool:
    plot_config = plotting_config(config)
    if "include_error_bars" in plot_config:
        return bool(plot_config["include_error_bars"])
    if "bar_chart_error_bars" in config:
        return bool(config["bar_chart_error_bars"])
    return False


def check_best_lag_significance(config: Mapping) -> bool:
    plot_config = plotting_config(config)
    if "check_best_lag_significance" in plot_config:
        return bool(plot_config["check_best_lag_significance"])
    if "check_best_lag_significance" in config:
        return bool(config["check_best_lag_significance"])
    return False


def correct_best_lag_significance(config: Mapping) -> bool:
    plot_config = plotting_config(config)
    if "correct_best_lag_significance" in plot_config:
        return bool(plot_config["correct_best_lag_significance"])
    if "correct_best_lag_significance" in config:
        return bool(config["correct_best_lag_significance"])
    return True


def task_display_names(config: Mapping) -> Mapping[str, str]:
    plot_config = plotting_config(config)
    configured = plot_config.get(
        "task_display_names", config.get("task_display_names", {})
    )
    if not isinstance(configured, Mapping):
        return {}
    return {str(task): str(display_name) for task, display_name in configured.items()}


def display_task_name(config: Mapping, task: str) -> str:
    return task_display_names(config).get(task, task)


def model_display_names(config: Mapping) -> Mapping[str, str]:
    plot_config = plotting_config(config)
    configured = plot_config.get(
        "model_display_names", config.get("model_display_names", {})
    )
    if not isinstance(configured, Mapping):
        return {}
    return {str(model): str(display_name) for model, display_name in configured.items()}


def display_model_name(config: Mapping, model: str) -> str:
    return model_display_names(config).get(model, model)


def task_group_config(config: Mapping) -> Mapping:
    plot_config = plotting_config(config)
    configured = plot_config.get("task_groups", config.get("task_groups", {}))
    return configured if isinstance(configured, Mapping) else {}


def grouped_tasks_for_summary(
    config: Mapping, tasks: Sequence[str]
) -> list[tuple[str, list[str]]]:
    configured = task_group_config(config)
    task_set = set(tasks)
    grouped: dict[str, list[str]] = {}
    assigned: set[str] = set()

    if configured and all(isinstance(value, str) for value in configured.values()):
        for task in tasks:
            group = configured.get(task)
            if group is None:
                continue
            grouped.setdefault(str(group), []).append(task)
            assigned.add(task)
    else:
        for group, group_tasks in configured.items():
            if not isinstance(group_tasks, Sequence) or isinstance(group_tasks, str):
                continue
            for task in group_tasks:
                task = str(task)
                if task not in task_set:
                    continue
                grouped.setdefault(str(group), []).append(task)
                assigned.add(task)

    for task in sorted(task_set - assigned):
        grouped.setdefault("Other", []).append(task)

    ordered_groups = []
    for group in DEFAULT_TASK_GROUP_ORDER:
        if group in grouped:
            ordered_groups.append((group, grouped.pop(group)))
    ordered_groups.extend((group, grouped[group]) for group in grouped)
    return ordered_groups


def metric_config_from_summary(task_summary: pd.DataFrame) -> MetricConfig:
    return MetricConfig(
        column=str(task_summary["metric"].iloc[0]),
        higher_is_better=bool(task_summary["higher_is_better"].iloc[0]),
        label=str(task_summary["metric_label"].iloc[0]),
        min_value=(
            float(task_summary["metric_min"].iloc[0])
            if "metric_min" in task_summary
            and pd.notna(task_summary["metric_min"].iloc[0])
            else None
        ),
        max_value=(
            float(task_summary["metric_max"].iloc[0])
            if "metric_max" in task_summary
            and pd.notna(task_summary["metric_max"].iloc[0])
            else None
        ),
        negate=(
            bool(task_summary["metric_negate"].iloc[0])
            if "metric_negate" in task_summary
            and pd.notna(task_summary["metric_negate"].iloc[0])
            else False
        ),
    )


def metric_fold_columns(df: pd.DataFrame, metric: MetricConfig) -> Dict[int, str]:
    prefix = (
        metric.column.removesuffix("_mean")
        if metric.column.endswith("_mean")
        else metric.column
    )
    columns = {}
    for column in df.columns:
        stem, separator, fold = str(column).rpartition("_fold_")
        if stem != prefix or separator != "_fold_":
            continue
        try:
            columns[int(fold)] = str(column)
        except ValueError:
            continue
    return columns


def fold_values_at_lag(df: pd.DataFrame, lag, metric: MetricConfig) -> Dict[int, float]:
    fold_columns = metric_fold_columns(df, metric)
    if not fold_columns:
        return {}

    lag_values = pd.to_numeric(df["lags"], errors="coerce")
    lag_matches = df[lag_values == float(lag)]
    if lag_matches.empty:
        return {}
    row = lag_matches.iloc[0]
    values = {}
    for fold, column in fold_columns.items():
        value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
        if pd.notna(value):
            values[fold] = float(-value if metric.negate else value)
    return values


def fold_lag_performance_matrix(
    df: pd.DataFrame,
    metric: MetricConfig,
    folds: Sequence[int],
    lags: Sequence[float],
) -> np.ndarray:
    fold_columns = metric_fold_columns(df, metric)
    by_lag = df.assign(_numeric_lag=pd.to_numeric(df["lags"], errors="coerce"))
    by_lag = by_lag.set_index("_numeric_lag", drop=False)
    matrix = np.empty((len(folds), len(lags)), dtype=float)
    for lag_idx, lag in enumerate(lags):
        row = by_lag.loc[float(lag)]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        for fold_idx, fold in enumerate(folds):
            value = pd.to_numeric(
                pd.Series([row[fold_columns[fold]]]), errors="coerce"
            ).iloc[0]
            if pd.isna(value):
                matrix[fold_idx, lag_idx] = np.nan
                continue
            score = float(-value if metric.negate else value)
            matrix[fold_idx, lag_idx] = score if metric.higher_is_better else -score
    return matrix


def best_lag_permutation_p_value(
    winner_matrix: np.ndarray,
    other_matrix: np.ndarray,
) -> tuple[float, float]:
    valid_rows = np.isfinite(winner_matrix).all(axis=1) & np.isfinite(other_matrix).all(axis=1)
    winner_matrix = winner_matrix[valid_rows]
    other_matrix = other_matrix[valid_rows]
    if len(winner_matrix) < 2:
        return float("nan"), float("nan")

    observed = float(
        np.max(winner_matrix.mean(axis=0)) - np.max(other_matrix.mean(axis=0))
    )
    n_folds = winner_matrix.shape[0]
    count = 0
    total = 2**n_folds
    for mask_value in range(total):
        swap = np.array(
            [(mask_value >> fold_idx) & 1 for fold_idx in range(n_folds)],
            dtype=bool,
        )[:, None]
        permuted_winner = np.where(swap, other_matrix, winner_matrix)
        permuted_other = np.where(swap, winner_matrix, other_matrix)
        statistic = float(
            np.max(permuted_winner.mean(axis=0))
            - np.max(permuted_other.mean(axis=0))
        )
        if statistic >= observed - 1e-12:
            count += 1
    return observed, count / total


def holm_adjust_p_values(p_values: Sequence[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    finite = [(idx, float(p_value)) for idx, p_value in enumerate(p_values) if np.isfinite(p_value)]
    if not finite:
        return adjusted

    ordered = sorted(finite, key=lambda item: item[1])
    running_max = 0.0
    n_tests = len(ordered)
    for rank, (idx, p_value) in enumerate(ordered):
        corrected = min((n_tests - rank) * p_value, 1.0)
        running_max = max(running_max, corrected)
        adjusted[idx] = running_max
    return adjusted


def significance_label(p_value: float) -> str:
    if not np.isfinite(p_value):
        return "n.s."
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return "n.s."


def best_lag_significance_tests(
    task_summary: pd.DataFrame,
    task_results: Mapping[str, pd.DataFrame],
    metric: MetricConfig,
    correct_multiple_comparisons: bool = True,
) -> list[dict[str, object]]:
    if task_summary.empty or len(task_summary) < 2:
        return []

    idx = (
        task_summary["value"].idxmax()
        if metric.higher_is_better
        else task_summary["value"].idxmin()
    )
    winner = str(task_summary.loc[idx, "model"])
    if winner not in task_results:
        return []
    if not metric_fold_columns(task_results[winner], metric):
        return []

    comparisons = []
    for item in task_summary.to_dict("records"):
        model = str(item["model"])
        if model == winner or model not in task_results:
            continue
        winner_fold_columns = metric_fold_columns(task_results[winner], metric)
        other_fold_columns = metric_fold_columns(task_results[model], metric)
        folds = sorted(set(winner_fold_columns) & set(other_fold_columns))
        if len(folds) < 2:
            continue

        winner_lags = set(pd.to_numeric(task_results[winner]["lags"], errors="coerce").dropna())
        other_lags = set(pd.to_numeric(task_results[model]["lags"], errors="coerce").dropna())
        lags = sorted(winner_lags & other_lags)
        if not lags:
            continue

        winner_matrix = fold_lag_performance_matrix(
            task_results[winner],
            metric,
            folds,
            lags,
        )
        other_matrix = fold_lag_performance_matrix(
            task_results[model],
            metric,
            folds,
            lags,
        )
        statistic, raw_p_value = best_lag_permutation_p_value(
            winner_matrix,
            other_matrix,
        )
        comparisons.append(
            {
                "winner": winner,
                "other": model,
                "raw_p_value": raw_p_value,
                "p_value": raw_p_value,
                "statistic": statistic,
                "n": len(folds),
            }
        )
    display_p_values = (
        holm_adjust_p_values([float(comparison["raw_p_value"]) for comparison in comparisons])
        if correct_multiple_comparisons
        else [float(comparison["raw_p_value"]) for comparison in comparisons]
    )
    for comparison, p_value in zip(comparisons, display_p_values):
        comparison["p_value"] = p_value
        comparison["label"] = significance_label(p_value)
    return comparisons


def draw_significance_annotations(
    ax: plt.Axes,
    comparisons: Sequence[Mapping[str, object]],
    models: Sequence[str],
    values: Sequence[float],
    errors: Sequence[float],
) -> None:
    if not comparisons:
        return

    value_by_model = dict(zip(models, values))
    error_by_model = dict(zip(models, errors))
    x_by_model = {model: idx for idx, model in enumerate(models)}
    finite_heights = [
        float(value_by_model[model]) + abs(float(error_by_model.get(model, 0.0)))
        for model in models
        if np.isfinite(value_by_model.get(model, np.nan))
    ]
    if not finite_heights:
        return

    y_min, y_max = ax.get_ylim()
    data_span = y_max - y_min
    if data_span <= 0:
        data_span = max(abs(y_max), 1.0)
    bracket_height = data_span * 0.03
    step = data_span * 0.14
    text_pad = data_span * 0.01
    start = max(finite_heights) + data_span * 0.06

    ordered_comparisons = sorted(
        comparisons,
        key=lambda comparison: (
            abs(
                x_by_model.get(str(comparison["winner"]), 0)
                - x_by_model.get(str(comparison["other"]), 0)
            ),
            min(
                x_by_model.get(str(comparison["winner"]), 0),
                x_by_model.get(str(comparison["other"]), 0),
            ),
        ),
    )

    drawn_count = 0
    for comparison in ordered_comparisons:
        winner = str(comparison["winner"])
        other = str(comparison["other"])
        if winner not in x_by_model or other not in x_by_model:
            continue
        x1, x2 = sorted([x_by_model[winner], x_by_model[other]])
        y = start + drawn_count * step
        ax.plot(
            [x1, x1, x2, x2],
            [y, y + bracket_height, y + bracket_height, y],
            color="#333333",
            linewidth=0.9,
            clip_on=False,
        )
        ax.text(
            (x1 + x2) / 2,
            y + bracket_height + text_pad,
            str(comparison["label"]),
            ha="center",
            va="bottom",
            fontsize=9,
            clip_on=False,
        )
        drawn_count += 1

    if drawn_count:
        ax.set_ylim(y_min, start + drawn_count * step + bracket_height + text_pad * 2)


def group_matches_bar_layout(group: str, canonical_group: str) -> bool:
    aliases = BAR_SUMMARY_GROUP_ALIASES.get(canonical_group, (canonical_group,))
    normalized = group.casefold()
    return any(normalized == alias.casefold() for alias in aliases)


def best_lag_bar_group_slots(
    task_groups: Sequence[tuple[str, list[str]]],
) -> list[tuple[str, str, list[str], tuple[tuple[int, int], ...]]]:
    remaining_groups = [(group, list(tasks)) for group, tasks in task_groups]
    planned = []
    for canonical_group, slots in BAR_SUMMARY_GROUP_LAYOUT.items():
        match_idx = next(
            (
                idx
                for idx, (group, _tasks) in enumerate(remaining_groups)
                if group_matches_bar_layout(group, canonical_group)
            ),
            None,
        )
        if match_idx is None:
            group = canonical_group
            tasks = []
        else:
            group, tasks = remaining_groups.pop(match_idx)
        planned.append((canonical_group, group, tasks, slots))

    overflow_slots = tuple(
        (row, col)
        for row in range(BAR_SUMMARY_GRID_ROWS)
        for col in range(BAR_SUMMARY_GRID_COLS)
    )
    for group, tasks in remaining_groups:
        planned.append((group, group, tasks, overflow_slots))
    return planned


def draw_bar_group_box(
    fig: plt.Figure,
    group: str,
    axes: Sequence[plt.Axes],
) -> None:
    if not axes:
        return
    left = min(ax.get_position().x0 for ax in axes)
    right = max(ax.get_position().x1 for ax in axes)
    bottom = min(ax.get_position().y0 for ax in axes)
    top = max(ax.get_position().y1 for ax in axes)
    pad_x = 0.009
    pad_y = 0.018
    rect = plt.Rectangle(
        (left - pad_x, bottom - pad_y),
        (right - left) + pad_x * 2,
        (top - bottom) + pad_y * 2.7,
        transform=fig.transFigure,
        fill=False,
        linewidth=1.1,
        edgecolor="#666666",
        clip_on=False,
        zorder=2,
    )
    fig.add_artist(rect)
    fig.text(
        (left + right) / 2,
        top + pad_y * 1.2,
        group,
        ha="center",
        va="bottom",
        fontsize=plt.rcParams["axes.titlesize"],
        weight="bold",
    )


def plot_best_lag_summary(
    summary: pd.DataFrame,
    condition: str,
    output_dir: Path,
    formats: Sequence[str],
    colors: Mapping[str, str],
    config: Mapping | None = None,
    condition_results: Mapping[str, Mapping[str, pd.DataFrame]] | None = None,
) -> None:
    config = config or {}
    models = sorted(summary["model"].unique())
    tasks = sorted(summary["task"].unique())
    task_groups = grouped_tasks_for_summary(config, tasks)
    fig = plt.figure(figsize=(18, 8))
    outer_grid = fig.add_gridspec(
        BAR_SUMMARY_GRID_ROWS,
        BAR_SUMMARY_GRID_COLS,
        hspace=0.75 if check_best_lag_significance(config) else 0.5,
        wspace=0.36,
    )
    x_positions = list(range(len(models)))
    show_error_bars = include_bar_error_bars(config)
    show_significance = check_best_lag_significance(config)
    correct_significance = correct_best_lag_significance(config)
    group_title_axes = []
    used_slots: set[tuple[int, int]] = set()

    for canonical_group, _group, group_tasks, slots in best_lag_bar_group_slots(
        task_groups
    ):
        group_axes = []
        plotted_tasks = 0
        for task_idx, task in enumerate(group_tasks[: len(slots)]):
            row, col = slots[task_idx]
            if (row, col) in used_slots:
                continue
            used_slots.add((row, col))
            ax = fig.add_subplot(outer_grid[row, col])
            group_axes.append(ax)
            plotted_tasks += 1
            task_summary = summary[summary["task"] == task]
            metric = metric_config_from_summary(task_summary)
            values = []
            errors = []
            for model in models:
                match = task_summary[task_summary["model"] == model]
                values.append(
                    float(match["value"].iloc[0]) if not match.empty else float("nan")
                )
                errors.append(
                    float(match["std"].iloc[0])
                    if show_error_bars
                    and "std" in match
                    and not match.empty
                    and pd.notna(match["std"].iloc[0])
                    else 0.0
                )

            ax.bar(
                x_positions,
                values,
                yerr=errors if show_error_bars else None,
                error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
                color=[colors[model] for model in models],
                width=0.7,
            )
            label = (
                str(task_summary["metric_label"].iloc[0])
                if task_summary["metric_label"].nunique() == 1
                else "Metric"
            )
            ax.set_title(display_task_name(config, task))
            ax.set_ylabel(label)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(
                [display_model_name(config, model) for model in models],
                rotation=35,
                ha="right",
            )
            ax.grid(axis="y", alpha=0.25)
            apply_metric_ylim(ax, metric)
            if show_significance and condition_results is not None:
                comparisons = best_lag_significance_tests(
                    task_summary,
                    condition_results.get(task, {}),
                    metric,
                    correct_multiple_comparisons=correct_significance,
                )
                draw_significance_annotations(
                    ax,
                    comparisons,
                    models,
                    values,
                    errors,
                )

        if group_tasks and len(group_tasks) <= len(slots):
            empty_range = range(plotted_tasks, len(slots))
        else:
            empty_range = range(0)
        for empty_idx in empty_range:
            row, col = slots[empty_idx]
            if (row, col) in used_slots:
                continue
            used_slots.add((row, col))
            empty_ax = fig.add_subplot(outer_grid[row, col])
            empty_ax.set_axis_off()
            group_axes.append(empty_ax)

        if group_axes:
            group_title_axes.append((canonical_group, group_axes))

    handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            color=colors[model],
            label=display_model_name(config, model),
        )
        for model in models
    ]
    fig.text(
        0.01,
        0.985,
        condition.replace("_", " ").title(),
        ha="left",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=len(models),
        frameon=False,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.1, top=0.86)
    for group, group_axes in group_title_axes:
        draw_bar_group_box(fig, group, group_axes)
    save_figure(fig, output_dir / f"best_lag_summary_{condition}", formats)


def plot_lag_curves(
    loaded: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
    colors: Mapping[str, str],
) -> None:
    for condition, tasks in loaded.items():
        for task, model_results in tasks.items():
            metric = get_metric_config(config, task)
            fig, ax = plt.subplots(figsize=(6, 4))
            for model, df in sorted(model_results.items()):
                if metric.column not in df.columns:
                    continue
                curve = curve_for_metric(df, metric)
                ax.plot(
                    curve["lags"],
                    curve[metric.column],
                    marker="o",
                    linewidth=1.8,
                    label=display_model_name(config, model),
                    color=colors[model],
                )

            ax.set_title(
                f"{display_task_name(config, task)} - {condition.replace('_', ' ')}"
            )
            ax.set_xlabel("Lag relative to word onset (ms)")
            ax.set_ylabel(metric.label)
            ax.axvline(0, color="#333333", linewidth=0.8, alpha=0.5)
            ax.grid(alpha=0.25)
            apply_metric_ylim(ax, metric)
            ax.legend(frameon=False)
            save_figure(fig, output_dir / f"lag_curves_{task}_{condition}", formats)


def curve_for_metric(df: pd.DataFrame, metric: MetricConfig) -> pd.DataFrame:
    if metric.column not in df.columns:
        raise KeyError(f"Metric column '{metric.column}' is missing")
    curve = df[["lags", metric.column]].copy()
    curve[metric.column] = metric_values(curve, metric)
    return curve.dropna().sort_values("lags")


def region_sort_key(region: str) -> tuple[int, str]:
    if region in REGION_LEVEL_ORDER:
        return (REGION_LEVEL_ORDER.index(region), region)
    return (len(REGION_LEVEL_ORDER), region)


def region_gradient_colors(regions: Sequence[str]) -> Dict[str, object]:
    ordered = sorted(regions, key=region_sort_key)
    cmap = plt.get_cmap("viridis", max(len(ordered), 1))
    if len(ordered) == 1:
        return {ordered[0]: cmap(0.65)}
    return {region: cmap(idx / (len(ordered) - 1)) for idx, region in enumerate(ordered)}


def plot_per_region_lag_curves(
    per_region_results: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    models = sorted(
        {
            model
            for task_results in per_region_results.values()
            for model in task_results
        }
    )
    for model in models:
        task_items = [
            (task, model_results[model])
            for task, model_results in sorted(per_region_results.items())
            if model in model_results
        ]
        if not task_items:
            continue

        all_regions = sorted(
            {
                region
                for _, region_results in task_items
                for region in region_results
            },
            key=region_sort_key,
        )
        colors = region_gradient_colors(all_regions)
        n_tasks = len(task_items)
        ncols = min(3, n_tasks)
        nrows = int(np.ceil(n_tasks / ncols))
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.6 * ncols, 3.5 * nrows),
            squeeze=False,
            sharex=False,
            sharey=False,
        )
        axes_flat = axes.ravel()

        for ax, (task, region_results) in zip(axes_flat, task_items):
            metric = get_metric_config(config, task)
            for region in sorted(region_results, key=region_sort_key):
                curve = curve_for_metric(region_results[region], metric)
                ax.plot(
                    curve["lags"],
                    curve[metric.column],
                    marker="o",
                    linewidth=1.6,
                    markersize=3.5,
                    label=region,
                    color=colors[region],
                )
            ax.set_title(display_task_name(config, task))
            ax.set_xlabel("Lag relative to word onset (ms)")
            ax.set_ylabel(metric.label)
            ax.axvline(0, color="#777777", linewidth=0.8, alpha=0.6)
            ax.grid(alpha=0.25)
            apply_metric_ylim(ax, metric)

        for ax in axes_flat[n_tasks:]:
            ax.set_axis_off()

        handles = [
            plt.Line2D(
                [0],
                [0],
                color=colors[region],
                marker="o",
                linewidth=1.6,
                markersize=4,
                label=region,
            )
            for region in all_regions
        ]
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(len(handles), 6),
            frameon=False,
        )
        fig.suptitle(f"{display_model_name(config, model)} per-region lag curves", y=1.06)
        fig.tight_layout(rect=(0, 0, 1, 0.92))
        save_figure(fig, output_dir / f"per_region_lags_{model}", formats)


def _load_region_electrodes(
    data_root: Path,
    nilearn_data_dir: Path | None,
    include_bad: bool,
) -> pd.DataFrame:
    try:
        from scripts.plot_atlas_region_electrodes import (
            assign_region_groups,
            load_electrodes,
            load_region_groups,
        )
    except ModuleNotFoundError:
        from plot_atlas_region_electrodes import (
            assign_region_groups,
            load_electrodes,
            load_region_groups,
        )

    electrodes = load_electrodes(data_root, include_bad=include_bad)
    return assign_region_groups(
        electrodes,
        load_region_groups(None),
        nilearn_data_dir=nilearn_data_dir,
    )


def _load_destrieux_surface_atlas(
    nilearn_data_dir: Path | None,
) -> DestrieuxSurfaceAtlas:
    from nilearn import datasets

    fetch_kwargs = {}
    if nilearn_data_dir is not None:
        nilearn_data_dir.mkdir(parents=True, exist_ok=True)
        fetch_kwargs["data_dir"] = str(nilearn_data_dir)

    atlas = datasets.fetch_atlas_surf_destrieux(**fetch_kwargs)
    fsaverage = datasets.load_fsaverage("fsaverage5", **fetch_kwargs)
    sulcal = datasets.load_fsaverage_data(
        mesh="fsaverage5",
        mesh_type="inflated",
        data_type="sulcal",
        **fetch_kwargs,
    )
    return DestrieuxSurfaceAtlas(
        labels=list(atlas["labels"]),
        maps={"left": atlas["map_left"], "right": atlas["map_right"]},
        mesh=fsaverage["inflated"],
        sulcal=sulcal,
    )


def _hemisphere_label_name(label: str) -> tuple[str | None, str]:
    if label.startswith("L "):
        return "left", label[2:]
    if label.startswith("R "):
        return "right", label[2:]
    return None, label


def _surface_region_label_sets(
    region_groups: Mapping[str, Sequence[str]],
) -> dict[str, dict[str, set[str]]]:
    label_sets: dict[str, dict[str, set[str]]] = {"left": {}, "right": {}}
    for region, labels in region_groups.items():
        for label in labels:
            hemi, surface_label = _hemisphere_label_name(label)
            if hemi is None:
                label_sets["left"].setdefault(region, set()).add(surface_label)
                label_sets["right"].setdefault(region, set()).add(surface_label)
            else:
                label_sets[hemi].setdefault(region, set()).add(surface_label)
    return label_sets


def _build_surface_metric_maps(
    atlas_labels: Sequence[str],
    atlas_maps: Mapping[str, np.ndarray],
    region_groups: Mapping[str, Sequence[str]],
    metric_by_region: Mapping[str, float],
) -> tuple[dict[str, np.ndarray], dict[str, dict[str, np.ndarray]]]:
    label_sets = _surface_region_label_sets(region_groups)
    label_names = np.asarray(list(atlas_labels), dtype=object)
    metric_maps: dict[str, np.ndarray] = {}
    region_masks: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}

    for hemi in ("left", "right"):
        atlas_map = np.asarray(atlas_maps[hemi], dtype=int)
        valid = (atlas_map >= 0) & (atlas_map < len(label_names))
        surface_labels = np.full(atlas_map.shape, None, dtype=object)
        surface_labels[valid] = label_names[atlas_map[valid]]
        metric_map = np.full(atlas_map.shape, np.nan, dtype=float)

        for region, labels in label_sets[hemi].items():
            if region not in metric_by_region:
                continue
            mask = np.isin(surface_labels, list(labels))
            if not mask.any():
                continue
            metric_map[mask] = float(metric_by_region[region])
            region_masks[hemi][region] = mask

        metric_maps[hemi] = metric_map

    return metric_maps, region_masks


def _surface_part(surface_object: object, hemi: str):
    if isinstance(surface_object, Mapping):
        return surface_object[hemi]
    if hasattr(surface_object, "parts"):
        return surface_object.parts[hemi]
    if hasattr(surface_object, "data") and hasattr(surface_object.data, "parts"):
        return surface_object.data.parts[hemi]
    raise TypeError(f"Unsupported surface object: {type(surface_object)!r}")


def _mesh_coordinates(mesh_part: object) -> np.ndarray:
    if hasattr(mesh_part, "coordinates"):
        return np.asarray(mesh_part.coordinates, dtype=float)
    return np.asarray(mesh_part[0], dtype=float)


def _surface_contour_map(
    region_masks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray | None, list[int]]:
    if not region_masks:
        return None, []

    first_mask = next(iter(region_masks.values()))
    contour_map = np.zeros(first_mask.shape, dtype=int)
    levels = []
    for idx, region in enumerate(sorted(region_masks, key=region_sort_key), start=1):
        contour_map[region_masks[region]] = idx
        levels.append(idx)
    return contour_map, levels


def _draw_surface_region_boundaries(
    ax: plt.Axes,
    mesh_part: object,
    region_masks: Mapping[str, np.ndarray],
) -> None:
    from nilearn import plotting

    contour_map, levels = _surface_contour_map(region_masks)
    if contour_map is None or not levels:
        return

    plotting.plot_surf_contours(
        surf_mesh=mesh_part,
        roi_map=contour_map,
        levels=levels,
        colors=[(0.06, 0.06, 0.06, 0.95)] * len(levels),
        axes=ax,
        figure=ax.figure,
        legend=False,
    )


def _draw_surface_region_labels(
    ax: plt.Axes,
    mesh_part: object,
    region_masks: Mapping[str, np.ndarray],
    region_counts: Mapping[str, int],
) -> None:
    coords = _mesh_coordinates(mesh_part)
    for region, mask in sorted(region_masks.items(), key=lambda item: region_sort_key(item[0])):
        if not mask.any():
            continue
        center = coords[mask].mean(axis=0)
        text = ax.text(
            center[0],
            center[1],
            center[2],
            f"{region}\nn={region_counts.get(region, 0)}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            weight="bold",
            zorder=12,
        )
        text.set_path_effects(
            [patheffects.withStroke(linewidth=2.0, foreground="#1f1f1f")]
        )


def plot_per_region_brains(
    per_region_results: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
    data_root: Path,
    nilearn_data_dir: Path | None,
    include_bad: bool,
) -> None:
    if not per_region_results:
        return

    from nilearn import plotting
    from utils.atlas_utils import REGION_GROUPS

    electrodes = _load_region_electrodes(data_root, nilearn_data_dir, include_bad)
    electrodes = electrodes[electrodes["region_group"] != "unassigned"].copy()
    region_counts = electrodes["region_group"].value_counts().to_dict()
    surface_atlas = _load_destrieux_surface_atlas(nilearn_data_dir)

    for task, model_results in sorted(per_region_results.items()):
        metric = get_metric_config(config, task)
        cmap = plt.get_cmap("viridis" if metric.higher_is_better else "viridis_r")
        for model, region_results in sorted(model_results.items()):
            best_rows = best_region_lag_rows(region_results, metric)
            if best_rows.empty:
                continue
            metric_by_region = dict(zip(best_rows["region"], best_rows["value"]))
            norm = metric_norm(list(metric_by_region.values()), metric)
            metric_maps, region_masks = _build_surface_metric_maps(
                surface_atlas.labels,
                surface_atlas.maps,
                REGION_GROUPS,
                metric_by_region,
            )

            fig = plt.figure(figsize=(12, 5.5))
            fig.subplots_adjust(left=0.01, right=0.84, top=0.88, bottom=0.04, wspace=0.0)
            axes = [
                fig.add_subplot(1, 2, 1, projection="3d"),
                fig.add_subplot(1, 2, 2, projection="3d"),
            ]
            panels = [
                ("Left hemisphere", "left"),
                ("Right hemisphere", "right"),
            ]
            for ax, (title, hemi) in zip(axes, panels):
                mesh_part = _surface_part(surface_atlas.mesh, hemi)
                sulcal_part = _surface_part(surface_atlas.sulcal, hemi)
                plotting.plot_surf_stat_map(
                    surf_mesh=mesh_part,
                    stat_map=metric_maps[hemi],
                    bg_map=sulcal_part,
                    hemi=hemi,
                    view="lateral",
                    cmap=cmap,
                    colorbar=False,
                    bg_on_data=True,
                    alpha=0.9,
                    vmin=norm.vmin,
                    vmax=norm.vmax,
                    symmetric_cbar=False,
                    figure=fig,
                    axes=ax,
                    title=title,
                )
                _draw_surface_region_boundaries(
                    ax,
                    mesh_part,
                    region_masks[hemi],
                )
                _draw_surface_region_labels(
                    ax,
                    mesh_part,
                    region_masks[hemi],
                    region_counts,
                )

            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cbar_ax = fig.add_axes([0.88, 0.19, 0.018, 0.6])
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(metric.label)
            fig.suptitle(
                f"{display_model_name(config, model)} "
                f"{display_task_name(config, task)} per-region best lag",
                y=0.99,
            )
            save_figure(fig, output_dir / f"per_region_brain_{model}_{task}", formats)


def generate_paper_results(
    config_path: Path,
    output_dir: Path,
    formats: Sequence[str],
    table_formats: Sequence[str],
    data_root: Path = Path("data"),
    nilearn_data_dir: Path | None = None,
    include_bad: bool = False,
) -> None:
    config = read_config(config_path)
    loaded = load_results(config)
    per_region_results = load_per_region_results(config)
    all_models = {
        model
        for condition_results in loaded.values()
        for task_results in condition_results.values()
        for model in task_results
    }
    colors = model_colors(all_models, config)

    all_summaries = []
    metrics = {
        task: get_metric_config(config, task)
        for condition_results in loaded.values()
        for task in condition_results
    }
    for condition, condition_results in loaded.items():
        summary = best_lag_rows(condition_results, metrics)
        if summary.empty:
            continue
        summary.insert(0, "condition", condition)
        all_summaries.append(summary)
        plot_best_lag_summary(
            summary,
            condition,
            output_dir,
            formats,
            colors,
            config,
            condition_results,
        )

    if all_summaries:
        write_summary_tables(
            pd.concat(all_summaries, ignore_index=True),
            output_dir,
            table_formats,
            config,
        )
    plot_lag_curves(loaded, config, output_dir, formats, colors)
    plot_per_region_lag_curves(per_region_results, config, output_dir, formats)
    plot_per_region_brains(
        per_region_results,
        config,
        output_dir,
        formats,
        data_root,
        nilearn_data_dir,
        include_bad,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_results.yml", type=Path)
    parser.add_argument("--output-dir", default="paper-results", type=Path)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    parser.add_argument(
        "--table-formats", nargs="+", default=["csv", "markdown", "latex"]
    )
    parser.add_argument("--data-root", default="data", type=Path)
    parser.add_argument(
        "--nilearn-data-dir",
        default=None,
        type=Path,
        help="Directory for Nilearn atlas cache. Defaults to <output-dir>/nilearn_data.",
    )
    parser.add_argument(
        "--include-bad",
        action="store_true",
        help="Include channels marked bad in electrode localization sidecars.",
    )
    return parser.parse_args()


def resolve_nilearn_data_dir(output_dir: Path, explicit_data_dir: Path | None) -> Path:
    if explicit_data_dir is not None:
        return explicit_data_dir
    if DEFAULT_NILEARN_DATA_DIR.exists():
        return DEFAULT_NILEARN_DATA_DIR
    return output_dir / "nilearn_data"


def main() -> None:
    args = parse_args()
    generate_paper_results(
        args.config,
        args.output_dir,
        args.formats,
        args.table_formats,
        args.data_root,
        resolve_nilearn_data_dir(args.output_dir, args.nilearn_data_dir),
        args.include_bad,
    )


if __name__ == "__main__":
    main()
