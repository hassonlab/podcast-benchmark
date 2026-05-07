#!/usr/bin/env python3
"""Generate paper-ready result figures and summary tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from matplotlib import patheffects
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, to_hex, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

CONDITIONS = ("super_subject", "per_subject")
PER_REGION_CONDITION = "per_region"
DEFAULT_NILEARN_DATA_DIR = (
    Path("processed_data") / "atlas_region_visualization" / "nilearn_data"
)
DEFAULT_SELECTED_ELECTRODE_PATH = Path("processed_data") / "all_subject_sig.csv"
DEFAULT_NEURAL_CONV_SUMMARY_CONFIG = (
    Path("configs")
    / "baselines"
    / "content_noncontent_task"
    / "neural_conv_decoder"
    / "supersubject.yml"
)
REGION_LEVEL_ORDER = ("EAC", "PC", "PRC", "IFG", "MTG", "ITG", "TPJ", "TP", "RIGHT")
DEFAULT_TASK_GROUP_ORDER = ("Representations", "Semantic", "Syntactic", "Acoustic")
COMPACT_EXCLUDED_TASKS = ("llm_decoding", "gpt_surprise")
BAR_SUMMARY_GRID_ROWS = 2
BAR_SUMMARY_GRID_COLS = 5
BAR_SUMMARY_PRIMARY_GRID_COLS = 3
BAR_SUMMARY_GRID_TASK_COLS = (0, 1, 2, 3, 4)
BAR_SUMMARY_GROUP_LAYOUT = {
    "Representations": ((0, 0), (1, 0)),  # (0, 1)),
    "Semantic": ((0, 2), (1, 2)),  # (1, 1),),
    "Syntactic": ((0, 3), (1, 3)),
    "Acoustic": ((0, 4), (1, 4)),
}
BAR_SUMMARY_GROUP_ALIASES = {
    "Representations": ("Representations", "Mixed"),
    "Semantic": ("Semantic",),
    "Syntactic": ("Syntactic",),
    "Acoustic": ("Acoustic", "Auditory"),
}
DEFAULT_TASK_GROUP_BACKGROUNDS = {
    "Representations": "#C9DDF2",
    "Mixed": "#C9DDF2",
    "Semantic": "#F0D2A7",
    "Syntactic": "#CDE8BF",
    "Auditory": "#F1BED0",
    "Acoustic": "#F1BED0",
    "Other": "#E8E8E8",
}
DEFAULT_TASK_GROUP_BACKGROUND_ALPHA = 0.5
DEFAULT_COLORS = {
    "baseline": "#4C78A8",
    "diver": "#F58518",
    "brainbert": "#54A24B",
    "popt": "#E45756",
}
BEST_COLOR = "#F58518"
PLOT_TITLE_FONTSIZE = 18
GROUP_TITLE_FONTSIZE = 20
GROUP_TITLE_VERTICAL_PAD = 0.095
GROUP_TITLE_TOP_INSET = 0.005
BRAIN_MAP_COLORBAR_LABEL_FONTSIZE = 14
BRAIN_MAP_COLORBAR_TICK_FONTSIZE = 12
LAG_PLOT_AXIS_LABEL_FONTSIZE = 14
LAG_PLOT_TICK_LABEL_FONTSIZE = 12
BEST_LAG_AXIS_LABEL_FONTSIZE = 14
BEST_LAG_TICK_LABEL_FONTSIZE = 12

plt.rcParams["axes.titlesize"] = PLOT_TITLE_FONTSIZE
plt.rcParams["figure.titlesize"] = PLOT_TITLE_FONTSIZE
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


@dataclass(frozen=True)
class MetricConfig:
    column: str
    higher_is_better: bool
    label: str
    min_value: float | None = None
    max_value: float | None = None
    chance_level: float | None = None
    negate: bool = False


@dataclass(frozen=True)
class ResultSpec:
    model: str
    task: str
    condition: str
    paths: tuple[Path, ...]

    @property
    def path(self) -> Path:
        if len(self.paths) != 1:
            raise ValueError(
                f"Result spec for {self.model}/{self.task}/{self.condition} "
                f"contains {len(self.paths)} paths; use 'paths' instead"
            )
        return self.paths[0]


@dataclass(frozen=True)
class DestrieuxSurfaceAtlas:
    labels: Sequence[str]
    maps: Mapping[str, np.ndarray]
    mesh: object
    sulcal: object


@dataclass(frozen=True)
class GroupedTaskFigure:
    fig: plt.Figure
    task_axes: Mapping[str, plt.Axes]
    group_axes: Sequence[tuple[str, Sequence[plt.Axes]]]
    group_score_axes: Mapping[str, plt.Axes]


@dataclass(frozen=True)
class HalfPeakProfile:
    peak_value: float
    peak_lag: float
    reference_value: float
    half_peak_value: float
    ramp_half_peak_lag: float
    ramp_duration: float
    ramp_slope: float
    ramp_rate: float
    decay_half_peak_lag: float
    decay_duration: float
    decay_slope: float
    decay_rate: float
    half_peak_width: float


def read_config(path: Path) -> Mapping:
    with path.open("r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, Mapping):
        raise ValueError(f"Config must be a mapping: {path}")
    return config


def resolve_config_path(config_path: Path, value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    return config_path.parent / path


def destrieux_atlas_path(config: Mapping, config_path: Path) -> Path | None:
    """Return the configured Destrieux atlas path, if one is present.

    The plotting config historically used Nilearn's fetchers implicitly. Accept
    a few explicit key names so existing paper-result configs can add the path
    either at the plotting level or directly under per-region brain settings.
    """

    plot_config = plotting_config(config)
    brain_config = per_region_brain_plot_config(config)
    candidate_containers = (brain_config, plot_config, config)
    candidate_keys = (
        "destrieux_atlas_path",
        "destrieux_2009_atlas_path",
        "atlas_path",
    )
    for container in candidate_containers:
        if not isinstance(container, Mapping):
            continue
        for key in candidate_keys:
            value = container.get(key)
            if value:
                return resolve_config_path(config_path, value)
    return None


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


def result_paths(value) -> tuple[Path, ...]:
    if isinstance(value, (str, Path)):
        return (Path(value),)
    if isinstance(value, Sequence):
        paths = tuple(Path(path) for path in value)
        if not paths:
            raise ValueError("Result path lists must contain at least one path")
        return paths
    raise TypeError(
        f"Result path must be a path string or list of paths, got {value!r}"
    )


def combine_lag_dataframes(frames: Sequence[pd.DataFrame], label: str) -> pd.DataFrame:
    if not frames:
        raise ValueError(f"No lag performance dataframes to combine for {label}")
    if len(frames) == 1:
        return frames[0]

    combined = pd.concat(frames, ignore_index=True)
    if "lags" not in combined.columns:
        return combined

    duplicated_lags = combined.loc[combined["lags"].duplicated(), "lags"].unique()
    if len(duplicated_lags):
        duplicate_text = ", ".join(str(lag) for lag in sorted(duplicated_lags))
        raise ValueError(
            f"Multiple configured result directories for {label} contain duplicate "
            f"lags: {duplicate_text}"
        )
    return combined.sort_values("lags").reset_index(drop=True)


def average_subject_lag_dataframes(
    subject_frames: Mapping[str, pd.DataFrame],
    label: str,
) -> pd.DataFrame:
    if not subject_frames:
        raise ValueError(f"No per-subject lag performance dataframes for {label}")

    combined = pd.concat(subject_frames.values(), ignore_index=True)
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


def load_per_subject_run(run_dir: Path) -> Dict[str, pd.DataFrame]:
    run_dir = Path(run_dir)
    loaded = {}
    for csv_path in sorted(run_dir.glob("subject_*/lag_performance.csv")):
        loaded[csv_path.parent.name] = pd.read_csv(csv_path)

    if not loaded:
        raise FileNotFoundError(
            f"Expected subject_*/lag_performance.csv files under {run_dir}"
        )
    return loaded


def load_per_subject_runs(run_dirs: Sequence[Path], label: str) -> pd.DataFrame:
    loaded_runs = [load_per_subject_run(run_dir) for run_dir in run_dirs]
    subjects = sorted({subject for loaded in loaded_runs for subject in loaded})
    combined = {}
    for subject in subjects:
        subject_frames = [
            loaded[subject] for loaded in loaded_runs if subject in loaded
        ]
        combined[subject] = combine_lag_dataframes(
            subject_frames,
            f"{label}/{subject}",
        )
    return average_subject_lag_dataframes(combined, label)


def load_current_style_runs(run_dirs: Sequence[Path], label: str) -> pd.DataFrame:
    has_root_csv = [
        Path(run_dir, "lag_performance.csv").exists() for run_dir in run_dirs
    ]
    has_subject_csvs = [
        any(Path(run_dir).glob("subject_*/lag_performance.csv")) for run_dir in run_dirs
    ]

    if any(has_root_csv) and any(has_subject_csvs):
        raise ValueError(
            f"Configured result directories for {label} mix super-subject and "
            "per-subject result layouts"
        )
    if any(has_subject_csvs):
        return load_per_subject_runs(run_dirs, label)
    return combine_lag_dataframes(
        [load_current_style_run(run_dir) for run_dir in run_dirs],
        label,
    )


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
                        model=model,
                        task=task,
                        condition=condition,
                        paths=result_paths(path),
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
                    paths=result_paths(path),
                )


def load_results(config: Mapping) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    loaded: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    for spec in iter_result_specs(config):
        loaded.setdefault(spec.condition, {}).setdefault(spec.task, {})[spec.model] = (
            load_current_style_runs(
                spec.paths,
                f"{spec.model}/{spec.task}/{spec.condition}",
            )
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


def load_per_region_runs(
    run_dirs: Sequence[Path],
    label: str,
) -> Dict[str, pd.DataFrame]:
    loaded_runs = [load_per_region_run(run_dir) for run_dir in run_dirs]
    if len(loaded_runs) == 1:
        return loaded_runs[0]

    regions = sorted({region for loaded in loaded_runs for region in loaded})
    combined = {}
    for region in regions:
        region_frames = [loaded[region] for loaded in loaded_runs if region in loaded]
        combined[region] = combine_lag_dataframes(
            region_frames,
            f"{label}/{region}",
        )
    return combined


def load_per_region_results(
    config: Mapping,
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    loaded: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    for spec in iter_per_region_result_specs(config):
        loaded.setdefault(spec.task, {})[spec.model] = load_per_region_runs(
            spec.paths,
            f"{spec.model}/{spec.task}/{spec.condition}",
        )
    return loaded


def get_metric_config(config: Mapping, task: str) -> MetricConfig:
    metrics = config.get("metrics", {})
    if not isinstance(metrics, Mapping) or task not in metrics:
        raise KeyError(f"No metric configuration found for task '{task}'")
    metric = metrics[task]
    min_value = _optional_float(metric.get("min", metric.get("vmin")))
    max_value = _optional_float(metric.get("max", metric.get("vmax")))
    chance_level = _optional_float(metric.get("chance_level", metric.get("chance")))
    if min_value is not None and max_value is not None and min_value >= max_value:
        raise ValueError(f"Metric bounds for task '{task}' must satisfy min < max")
    return MetricConfig(
        column=metric["column"],
        higher_is_better=bool(metric.get("higher_is_better", True)),
        label=metric.get("label", metric["column"]),
        min_value=min_value,
        max_value=max_value,
        chance_level=chance_level,
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


def plot_chance_level(ax: plt.Axes, metric: MetricConfig) -> None:
    if metric.chance_level is None:
        return
    ax.axhline(
        metric.chance_level,
        color="#777777",
        linestyle="--",
        linewidth=1.0,
        alpha=0.85,
        zorder=1,
    )


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


def brain_map_metric_config(
    config: Mapping,
    task: str,
    metric: MetricConfig,
) -> MetricConfig:
    brain_config = per_region_brain_plot_config(config)
    configured_bounds = brain_config.get(
        "colorbar_bounds",
        brain_config.get("metric_bounds", brain_config.get("bounds", {})),
    )
    task_bounds = None
    if isinstance(configured_bounds, Mapping):
        if any(key in configured_bounds for key in ("min", "max", "vmin", "vmax")):
            task_bounds = configured_bounds
        else:
            task_bounds = configured_bounds.get(task, configured_bounds.get("default"))
    if not isinstance(task_bounds, Mapping):
        task_bounds = brain_config

    lower = _optional_float(task_bounds.get("min", task_bounds.get("vmin")))
    upper = _optional_float(task_bounds.get("max", task_bounds.get("vmax")))
    if lower is not None and upper is not None and lower >= upper:
        raise ValueError(
            f"Brain map colorbar bounds for task '{task}' must satisfy min < max"
        )

    return MetricConfig(
        column=metric.column,
        higher_is_better=metric.higher_is_better,
        label=metric.label,
        min_value=lower,
        max_value=upper,
        chance_level=metric.chance_level,
        negate=metric.negate,
    )


def brain_map_colormap(config: Mapping, task: str, metric: MetricConfig):
    brain_config = per_region_brain_plot_config(config)
    configured = brain_config.get(
        "colormaps",
        brain_config.get(
            "cmaps", brain_config.get("colormap", brain_config.get("cmap"))
        ),
    )
    reverse = not metric.higher_is_better

    if isinstance(configured, Mapping):
        task_config = configured.get(task, configured.get("default"))
        if isinstance(task_config, Mapping):
            name = task_config.get(
                "name", task_config.get("cmap", task_config.get("colormap"))
            )
            reverse = bool(
                task_config.get("reverse", task_config.get("reversed", reverse))
            )
        else:
            name = task_config
    else:
        name = configured

    if name is None:
        name = "viridis"
    name = str(name)
    if reverse and not name.endswith("_r"):
        name = f"{name}_r"
    return plt.get_cmap(name)


def metric_values(df: pd.DataFrame, metric: MetricConfig) -> pd.Series:
    if metric.column not in df.columns:
        raise KeyError(f"Metric column '{metric.column}' is missing")
    values = pd.to_numeric(df[metric.column], errors="coerce")
    return -values if metric.negate else values


def select_best_lag(
    df: pd.DataFrame,
    metric: MetricConfig,
    valid_lags: Sequence[float] | None = None,
    task: str | None = None,
    model: str | None = None,
) -> pd.Series:
    if valid_lags is not None:
        if "lags" not in df.columns:
            raise KeyError("Lag column 'lags' is missing")
        numeric_lags = pd.to_numeric(df["lags"], errors="coerce")
        valid_lag_values = set(float(lag) for lag in valid_lags)
        df = df.loc[numeric_lags.isin(valid_lag_values)]
        if df.empty:
            context = []
            if task is not None:
                context.append(f"task '{task}'")
            if model is not None:
                context.append(f"model '{model}'")
            context_text = f" for {', '.join(context)}" if context else ""
            raise ValueError(
                f"No rows remain{context_text} after filtering to valid best lags "
                f"{list(valid_lags)}"
            )

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
    valid_lags: Sequence[float] | None = None,
) -> pd.DataFrame:
    rows = []
    for task, model_results in sorted(condition_results.items()):
        metric = metrics[task]
        for model, df in sorted(model_results.items()):
            best = select_best_lag(df, metric, valid_lags, task=task, model=model)
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "metric_min": metric.min_value,
                    "metric_max": metric.max_value,
                    "metric_chance_level": metric.chance_level,
                    "metric_negate": metric.negate,
                    "value": float(best[metric.column]),
                    "std": best_lag_std_value(best, metric),
                    "lag": best["lags"],
                    "higher_is_better": metric.higher_is_better,
                }
            )
    return pd.DataFrame(rows)


def baseline_condition_comparison_rows(
    loaded: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    metrics: Mapping[str, MetricConfig],
    *,
    baseline_model: str = "baseline",
) -> pd.DataFrame:
    summaries = []
    for condition in CONDITIONS:
        condition_results = {
            task: {baseline_model: model_results[baseline_model]}
            for task, model_results in loaded.get(condition, {}).items()
            if baseline_model in model_results
        }
        if not condition_results:
            continue

        summary = best_lag_rows(condition_results, metrics)
        if summary.empty:
            continue
        summary.insert(0, "condition", condition)
        summaries.append(summary)

    if not summaries:
        return pd.DataFrame()
    return pd.concat(summaries, ignore_index=True)


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
                "metric_chance_level": metric.chance_level,
                "metric_negate": metric.negate,
                "value": float(best[metric.column]),
                "std": best_lag_std_value(best, metric),
                "lag": best["lags"],
                "higher_is_better": metric.higher_is_better,
            }
        )
    return pd.DataFrame(rows)


def baseline_region_peak_lag_rows(
    per_region_results: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    metrics: Mapping[str, MetricConfig],
    baseline_model: str = "baseline",
) -> pd.DataFrame:
    rows = []
    for task, model_results in sorted(per_region_results.items()):
        if baseline_model not in model_results:
            continue
        metric = metrics[task]
        task_rows = best_region_lag_rows(model_results[baseline_model], metric)
        if task_rows.empty:
            continue
        task_rows.insert(0, "task", task)
        task_rows.insert(1, "model", baseline_model)
        rows.append(task_rows)
    if not rows:
        return pd.DataFrame(
            columns=[
                "task",
                "model",
                "region",
                "metric",
                "metric_label",
                "metric_min",
                "metric_max",
                "metric_chance_level",
                "metric_negate",
                "value",
                "std",
                "lag",
                "higher_is_better",
            ]
        )
    return pd.concat(rows, ignore_index=True)


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


def percent_decrease_from_max(value: float, max_value: float) -> float:
    if not np.isfinite(value) or not np.isfinite(max_value) or max_value == 0:
        return float("nan")
    return 100.0 * (max_value - value) / max_value


def format_percent_decrease(percent_decrease: float) -> str:
    if not np.isfinite(percent_decrease) or np.isclose(percent_decrease, 0.0):
        return ""
    return f"-{percent_decrease:.0f}%"


def format_value(value: float, lag, percent_decrease: float | None = None) -> str:
    if percent_decrease is None:
        return f"{value:.3f} ({lag:g} ms)"
    decrease = format_percent_decrease(percent_decrease)
    if not decrease:
        return f"{value:.3f} ({lag:g} ms)"
    return f"{value:.3f} ({lag:g} ms; {decrease})"


def relative_decrease_by_model(group: pd.DataFrame) -> dict[str, float]:
    values = pd.to_numeric(group["value"], errors="coerce")
    max_value = float(values.max())
    return {
        str(item["model"]): percent_decrease_from_max(float(item["value"]), max_value)
        for item in group.to_dict("records")
    }


def latex_escape_text(text: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(text))


def best_lag_latex_condition_name(condition: str) -> str:
    names = {
        "super_subject": "Multi-Subject",
        "per_subject": "Single-Subject",
    }
    return names.get(condition, condition.replace("_", " ").title())


def best_lag_latex_rowcolor(group: str) -> str:
    colors = {
        "Representations": "mixed",
        "Mixed": "mixed",
        "Semantic": "semantic",
        "Syntactic": "syntactic",
        "Acoustic": "acoustic",
        "Auditory": "acoustic",
    }
    return colors.get(group, str(group).casefold().replace(" ", "_"))


def best_lag_latex_table(summary: pd.DataFrame, config: Mapping) -> str:
    models = configured_model_order(summary["model"].unique(), config)
    winners = best_model_by_task(summary)
    records = {
        (str(item["condition"]), str(item["task"]), str(item["model"])): item
        for item in summary.to_dict("records")
    }
    tasks = list(dict.fromkeys(str(task) for task in summary["task"]))
    task_groups = grouped_tasks_for_summary(config, tasks)
    conditions = [
        condition
        for condition in CONDITIONS
        if condition in set(str(item) for item in summary["condition"])
    ]
    extra_conditions = sorted(
        set(str(item) for item in summary["condition"]) - set(conditions)
    )
    conditions.extend(extra_conditions)

    header_columns = [
        "Task",
        "Condition",
        *[display_model_name(config, model) for model in models],
    ]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Best performance by each model on every task, grouped by task "
            r"and condition.}"
        ),
        r"\setlength{\aboverulesep}{0pt}",
        r"\setlength{\belowrulesep}{0pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{" + "l" * len(header_columns) + "}",
        r"\toprule",
        " & ".join(
            rf"\textbf{{{latex_escape_text(column)}}}" for column in header_columns
        )
        + r" \\",
        r"\midrule",
        "",
    ]

    first_task = True
    for group, group_tasks in task_groups:
        rowcolor = best_lag_latex_rowcolor(group)
        for task in group_tasks:
            task_conditions = [
                condition
                for condition in conditions
                if not summary[
                    (summary["task"].astype(str) == task)
                    & (summary["condition"].astype(str) == condition)
                ].empty
            ]
            if not task_conditions:
                continue

            if not first_task:
                lines.append(r"\midrule")
                lines.append("")
            first_task = False

            lines.append(
                f"% --- {latex_escape_text(display_task_name(config, task))} ---"
            )
            task_row_count = len(task_conditions)
            for condition_idx, condition in enumerate(task_conditions):
                task_cell = ""
                if task_row_count == 1:
                    task_cell = latex_escape_text(display_task_name(config, task))
                elif condition_idx == task_row_count - 1:
                    task_cell = (
                        rf"\multirow{{-{task_row_count}}}{{*}}"
                        rf"{{{latex_escape_text(display_task_name(config, task))}}}"
                    )

                cells = [
                    task_cell,
                    latex_escape_text(best_lag_latex_condition_name(condition)),
                ]
                task_condition_summary = summary[
                    (summary["task"].astype(str) == task)
                    & (summary["condition"].astype(str) == condition)
                ]
                percent_decreases = relative_decrease_by_model(task_condition_summary)
                for model in models:
                    item = records.get((condition, task, model))
                    if item is None:
                        cells.append("")
                        continue
                    percent_decrease = percent_decreases.get(model, float("nan"))
                    percent_text = format_percent_decrease(percent_decrease)
                    value_text = f"{float(item['value']):.3f}"
                    if percent_text:
                        value_text = f"{value_text} ({percent_text})"
                    value = latex_escape_text(value_text)
                    if winners.get((condition, task)) == model:
                        value = rf"\textbf{{{value}}}"
                    cells.append(value)
                lines.append(rf"\rowcolor{{{rowcolor}}}")
                lines.append(" & ".join(cells) + r" \\")
            lines.append("")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\label{time_analysis_FM}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def neural_conv_summary_options(config: Mapping) -> Mapping:
    model_summary_config = config.get("model_summary", {})
    if not isinstance(model_summary_config, Mapping):
        return {}
    options = model_summary_config.get("neural_conv_decoder", model_summary_config)
    return options if isinstance(options, Mapping) else {}


def neural_conv_summary_enabled(config: Mapping) -> bool:
    options = neural_conv_summary_options(config)
    return bool(options.get("enabled", True))


def neural_conv_summary_config_path(config: Mapping) -> Path:
    options = neural_conv_summary_options(config)
    return Path(options.get("config", DEFAULT_NEURAL_CONV_SUMMARY_CONFIG))


def neural_conv_summary_output_name(config: Mapping) -> str:
    options = neural_conv_summary_options(config)
    return str(options.get("output_name", "neural_conv_decoder_summary"))


def _iter_config_setter_names(config_setter_name) -> list[str]:
    if not config_setter_name:
        return []
    if isinstance(config_setter_name, str):
        return [config_setter_name]
    return list(config_setter_name)


def _load_configured_neural_conv_model(config_path: Path):
    from types import SimpleNamespace

    import torch

    from core import registry
    from core.config import MultiTaskConfig
    from utils.config_utils import load_config
    from utils.model_utils import build_model_from_spec
    from utils.module_loader_utils import import_all_from_package

    import_all_from_package("models", recursive=True)
    import_all_from_package("tasks", recursive=True)
    import_all_from_package("metrics", recursive=True)

    loaded_config = load_config(str(config_path), {})
    experiment_config = (
        loaded_config.tasks[0]
        if isinstance(loaded_config, MultiTaskConfig)
        else loaded_config
    )

    data_params = experiment_config.task_config.data_params
    per_subject_electrodes = data_params.per_subject_electrodes or {}
    if not per_subject_electrodes:
        raise ValueError(
            f"Config {config_path} did not resolve any per-subject electrodes"
        )

    raws = [
        SimpleNamespace(ch_names=list(per_subject_electrodes[subject_id]))
        for subject_id in data_params.subject_ids
    ]
    for config_setter_name in _iter_config_setter_names(
        experiment_config.config_setter_name
    ):
        setter = registry.config_setter_registry[config_setter_name]
        experiment_config = setter(experiment_config, raws, pd.DataFrame())

    model = build_model_from_spec(experiment_config.model_spec)
    model.eval()

    model_params = experiment_config.model_spec.params
    input_channels = int(model_params["input_channels"])
    input_timesteps = int(model_params.get("input_timesteps", 1))
    dummy_input = torch.zeros(2, input_channels, input_timesteps)
    return model, dummy_input, experiment_config


def _format_model_summary_shape(value: Any) -> str:
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and isinstance(value, torch.Tensor):
        return "(" + ", ".join(str(dim) for dim in value.shape) + ")"
    if isinstance(value, (list, tuple)):
        return (
            "[" + ", ".join(_format_model_summary_shape(item) for item in value) + "]"
        )
    if isinstance(value, Mapping):
        return (
            "{"
            + ", ".join(
                f"{key}: {_format_model_summary_shape(item)}"
                for key, item in value.items()
            )
            + "}"
        )
    return str(type(value).__name__)


def neural_conv_model_summary_rows(model, example_input) -> pd.DataFrame:
    import torch

    rows: list[dict[str, Any]] = []
    hooks = []

    def register_hook(name: str, module):
        if list(module.children()):
            return

        def hook(_module, _inputs, output):
            params = sum(param.numel() for param in _module.parameters(recurse=False))
            trainable = sum(
                param.numel()
                for param in _module.parameters(recurse=False)
                if param.requires_grad
            )
            rows.append(
                {
                    "layer": name,
                    "type": _module.__class__.__name__,
                    "output_shape": _format_model_summary_shape(output),
                    "parameters": params,
                    "trainable_parameters": trainable,
                }
            )

        hooks.append(module.register_forward_hook(hook))

    for name, module in model.named_modules():
        if name:
            register_hook(name, module)

    try:
        with torch.no_grad():
            model(example_input)
    finally:
        for hook in hooks:
            hook.remove()

    return pd.DataFrame(rows)


def neural_conv_model_summary_latex_table(
    summary: pd.DataFrame,
    model,
    config_path: Path,
) -> str:
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        (
            r"\caption{Neural convolution decoder model summary. "
            rf"Configuration: \texttt{{{latex_escape_text(config_path)}}}.}}"
        ),
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lllr}",
        r"\toprule",
        r"\textbf{Layer} & \textbf{Type} & \textbf{Output shape} & \textbf{Parameters} \\",
        r"\midrule",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            " & ".join(
                [
                    latex_escape_text(row["layer"]),
                    latex_escape_text(row["type"]),
                    latex_escape_text(row["output_shape"]),
                    f"{int(row['parameters']):,}",
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\midrule",
            r"\textbf{Total} &  &  & " + f"{total_params:,}" + r" \\",
            r"\textbf{Trainable} &  &  & " + f"{trainable_params:,}" + r" \\",
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\label{tab:neural_conv_decoder_summary}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_neural_conv_model_summary_table(
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    if not neural_conv_summary_enabled(config):
        return
    if "latex" not in formats and "tex" not in formats:
        return

    config_path = neural_conv_summary_config_path(config)
    model, example_input, _experiment_config = _load_configured_neural_conv_model(
        config_path
    )
    summary = neural_conv_model_summary_rows(model, example_input)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = neural_conv_summary_output_name(config)
    (output_dir / f"{output_name}.tex").write_text(
        neural_conv_model_summary_latex_table(summary, model, config_path)
    )


def summary_wide(
    summary: pd.DataFrame,
    config: Mapping | None = None,
    bold: bool = False,
    latex: bool = False,
) -> pd.DataFrame:
    config = config or {}
    winners = best_model_by_task(summary) if bold else {}
    models = configured_model_order(summary["model"].unique(), config)
    group_columns = (
        ["condition", "task"] if "condition" in summary.columns else ["task"]
    )
    rows = []
    for keys, group in summary.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        by_model = {item["model"]: item for item in group.to_dict("records")}
        percent_decreases = relative_decrease_by_model(group)
        for model in models:
            column = display_model_name(config, model)
            if model not in by_model:
                row[column] = ""
                continue
            item = by_model[model]
            text = format_value(
                item["value"],
                item["lag"],
                percent_decreases.get(model),
            )
            if latex:
                text = text.replace("%", r"\%")
            if bold and winners.get(keys) == model:
                text = f"\\textbf{{{text}}}" if latex else f"**{text}**"
            row[column] = text
        rows.append(row)
    return pd.DataFrame(
        rows,
        columns=[
            *group_columns,
            *[display_model_name(config, model) for model in models],
        ],
    )


def write_summary_tables(
    summary: pd.DataFrame, output_dir: Path, formats: Sequence[str], config: Mapping
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_table = summary_wide(summary, config, bold=False)
    display_table = summary_wide(summary, config, bold=True)

    if "csv" in formats:
        csv_table.to_csv(output_dir / "best_lag_summary.csv", index=False)
    if "markdown" in formats or "md" in formats:
        (output_dir / "best_lag_summary.md").write_text(
            to_markdown_table(display_table)
        )
    if "latex" in formats or "tex" in formats:
        (output_dir / "best_lag_summary.tex").write_text(
            best_lag_latex_table(summary, config)
        )


def baseline_region_peak_wide(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()

    regions = sorted(summary["region"].unique(), key=region_sort_key)
    rows = []
    for task, group in summary.groupby("task", sort=True):
        row = {"task": task}
        by_region = {item["region"]: item for item in group.to_dict("records")}
        for region in regions:
            if region not in by_region:
                row[region] = ""
                continue
            item = by_region[region]
            row[region] = format_value(item["value"], item["lag"])
        rows.append(row)
    return pd.DataFrame(rows, columns=["task", *regions])


def write_baseline_region_peak_tables(
    summary: pd.DataFrame, output_dir: Path, formats: Sequence[str]
) -> None:
    if summary.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    table = baseline_region_peak_wide(summary)
    if "csv" in formats:
        table.to_csv(output_dir / "baseline_region_peak_lags.csv", index=False)
    if "markdown" in formats or "md" in formats:
        (output_dir / "baseline_region_peak_lags.md").write_text(
            to_markdown_table(table)
        )
    if "latex" in formats or "tex" in formats:
        (output_dir / "baseline_region_peak_lags.tex").write_text(
            table.to_latex(index=False, escape=False)
        )


def write_half_peak_profile_tables(
    profile: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    output_name: str = "baseline_half_peak_profile",
) -> None:
    if profile.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if "csv" in formats:
        profile.to_csv(output_dir / f"{output_name}.csv", index=False)
    if "markdown" in formats or "md" in formats:
        (output_dir / f"{output_name}.md").write_text(to_markdown_table(profile))
    if "latex" in formats or "tex" in formats:
        (output_dir / f"{output_name}.tex").write_text(
            profile.to_latex(index=False, escape=False)
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


def configured_model_order(models: Iterable[str], config: Mapping) -> list[str]:
    model_set = {str(model) for model in models}
    results = config.get("results", {})
    configured_order = (
        [str(model) for model in results.keys()] if isinstance(results, Mapping) else []
    )
    ordered = [model for model in configured_order if model in model_set]
    ordered.extend(sorted(model_set - set(ordered)))
    return ordered


def rasterized_hatch_artists(fig: plt.Figure) -> dict[Patch, bool | None]:
    """Rasterize hatched patches so SVG textures survive Illustrator import."""
    original_states = {}
    for patch in fig.findobj(match=Patch):
        if patch.get_hatch():
            original_states[patch] = patch.get_rasterized()
            patch.set_rasterized(True)
    return original_states


def restore_rasterized_states(states: Mapping[Patch, bool | None]) -> None:
    for artist, rasterized in states.items():
        artist.set_rasterized(rasterized)


def trimmed_tick_decimal_count(value: float) -> int:
    if abs(value) < 0.005:
        value = 0.0
    text = f"{value:.2f}".rstrip("0").rstrip(".")
    return len(text.rsplit(".", 1)[1]) if "." in text else 0


def y_tick_decimal_count(ax: plt.Axes) -> int:
    lower, upper = sorted(ax.get_ylim())
    span = upper - lower
    tolerance = span * 1e-6 if span else 1e-6
    visible_ticks = [
        tick
        for tick in ax.get_yticks()
        if np.isfinite(tick) and lower - tolerance <= tick <= upper + tolerance
    ]
    if not visible_ticks:
        return 0
    return max(trimmed_tick_decimal_count(float(tick)) for tick in visible_ticks)


def format_tick_with_decimals(value: float, decimals: int) -> str:
    if abs(value) < 0.5 * 10 ** (-max(decimals, 0)):
        value = 0.0
    return f"{value:.{decimals}f}"


def apply_y_tick_formatting(fig: plt.Figure) -> None:
    fig.canvas.draw()
    for ax in fig.axes:
        decimals = y_tick_decimal_count(ax)
        ax.yaxis.set_major_formatter(
            FuncFormatter(
                lambda value, _position, decimals=decimals: format_tick_with_decimals(
                    value,
                    decimals,
                )
            )
        )


def save_figure(fig: plt.Figure, output_base: Path, formats: Sequence[str]) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    apply_y_tick_formatting(fig)
    for fmt in formats:
        normalized_fmt = fmt.lower().lstrip(".")
        rasterized_states = (
            rasterized_hatch_artists(fig) if normalized_fmt in {"svg", "svgz"} else {}
        )
        try:
            fig.savefig(
                output_base.with_suffix(f".{fmt}"),
                bbox_inches="tight",
                dpi=300,
            )
        finally:
            restore_rasterized_states(rasterized_states)
    plt.close(fig)


def plotting_config(config: Mapping) -> Mapping:
    configured = config.get("plotting", {})
    return configured if isinstance(configured, Mapping) else {}


def valid_best_lags(config: Mapping) -> tuple[float, ...] | None:
    configured = plotting_config(config).get("valid_best_lags")
    if configured is None:
        return None
    if isinstance(configured, str) or not isinstance(configured, Sequence):
        raise ValueError("plotting.valid_best_lags must be configured as a list")

    lags = pd.to_numeric(pd.Series(list(configured)), errors="coerce")
    if lags.isna().any():
        raise ValueError("plotting.valid_best_lags must contain only numeric values")
    return tuple(float(lag) for lag in lags)


def half_peak_profile_config(config: Mapping) -> Mapping:
    plot_config = plotting_config(config)
    configured = plot_config.get("half_peak_profile", {})
    return configured if isinstance(configured, Mapping) else {}


def half_peak_profile_enabled(config: Mapping) -> bool:
    profile_config = half_peak_profile_config(config)
    return bool(profile_config.get("enabled", False))


def half_peak_profile_model(config: Mapping) -> str:
    return str(half_peak_profile_config(config).get("model", "baseline"))


def half_peak_profile_output_name(config: Mapping) -> str:
    return str(
        half_peak_profile_config(config).get(
            "output_name", "baseline_half_peak_profile"
        )
    )


def half_peak_profile_bar_output_name(config: Mapping) -> str:
    return str(
        half_peak_profile_config(config).get(
            "bar_output_name", "baseline_half_peak_profile_bars"
        )
    )


def include_bar_error_bars(config: Mapping) -> bool:
    plot_config = plotting_config(config)
    if "include_error_bars" in plot_config:
        return bool(plot_config["include_error_bars"])
    if "bar_chart_error_bars" in config:
        return bool(config["bar_chart_error_bars"])
    return False


def include_overall_scores(config: Mapping) -> bool:
    plot_config = plotting_config(config)
    for key in (
        "include_overall_scores",
        "include_group_scores",
        "add_overall_scores",
    ):
        if key in plot_config:
            return bool(plot_config[key])
        if key in config:
            return bool(config[key])
    return False


def exclude_llm_decoding_and_gpt_surprise(config: Mapping) -> bool:
    plot_config = plotting_config(config)
    for key in (
        "exclude_llm_decoding_and_gpt_surprise",
        "drop_llm_decoding_and_gpt_surprise",
        "compact_task_summary",
    ):
        if key in plot_config:
            return bool(plot_config[key])
        if key in config:
            return bool(config[key])
    return False


def excluded_tasks(config: Mapping) -> set[str]:
    plot_config = plotting_config(config)
    tasks: set[str] = set()
    configured = plot_config.get("exclude_tasks", config.get("exclude_tasks", ()))
    if isinstance(configured, str):
        tasks.add(configured)
    elif isinstance(configured, Sequence):
        tasks.update(str(task) for task in configured)
    if exclude_llm_decoding_and_gpt_surprise(config):
        tasks.update(COMPACT_EXCLUDED_TASKS)
    return tasks


def filter_loaded_tasks(
    loaded: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    tasks_to_exclude: set[str],
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    if not tasks_to_exclude:
        return {
            condition: {
                task: dict(model_results)
                for task, model_results in condition_results.items()
            }
            for condition, condition_results in loaded.items()
        }
    return {
        condition: {
            task: dict(model_results)
            for task, model_results in condition_results.items()
            if task not in tasks_to_exclude
        }
        for condition, condition_results in loaded.items()
    }


def filter_per_region_tasks(
    per_region_results: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    tasks_to_exclude: set[str],
) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    if not tasks_to_exclude:
        return {
            task: {
                model: dict(region_results)
                for model, region_results in model_results.items()
            }
            for task, model_results in per_region_results.items()
        }
    return {
        task: {
            model: dict(region_results)
            for model, region_results in model_results.items()
        }
        for task, model_results in per_region_results.items()
        if task not in tasks_to_exclude
    }


def best_lag_summary_plot_style(config: Mapping) -> str:
    plot_config = plotting_config(config)
    style = plot_config.get(
        "best_lag_summary_plot_style",
        plot_config.get("best_lag_summary_style", config.get("best_lag_summary_style")),
    )
    if style is None and bool(plot_config.get("use_bar_charts", False)):
        return "bar"
    if style is None:
        return "point"

    style = str(style).strip().lower()
    if style in {"bar", "bars", "bar_chart", "bar_charts"}:
        return "bar"
    if style in {"point", "points", "dot", "dots", "summary", "mean"}:
        return "point"
    raise ValueError(
        "Best lag summary plot style must be 'point' or 'bar', " f"got {style!r}"
    )


def bar_start_for_task(config: Mapping, task: str) -> float:
    plot_config = plotting_config(config)
    task_starts = plot_config.get(
        "best_lag_bar_starts",
        plot_config.get("bar_starts", plot_config.get("bar_start_by_task", {})),
    )
    if isinstance(task_starts, Mapping) and task in task_starts:
        return float(task_starts[task])

    metrics = config.get("metrics", {})
    metric_config = metrics.get(task, {}) if isinstance(metrics, Mapping) else {}
    if isinstance(metric_config, Mapping) and "bar_start" in metric_config:
        return float(metric_config["bar_start"])

    if "bar_start" in plot_config:
        return float(plot_config["bar_start"])
    if "bar_chart_start" in config:
        return float(config["bar_chart_start"])
    return 0.0


def best_lag_model_bar_textures(config: Mapping) -> Mapping[str, str]:
    plot_config = plotting_config(config)
    configured = plot_config.get(
        "best_lag_model_bar_textures",
        plot_config.get(
            "best_lag_model_textures",
            plot_config.get(
                "model_bar_textures",
                config.get(
                    "best_lag_model_bar_textures",
                    config.get(
                        "best_lag_model_textures", config.get("model_bar_textures", {})
                    ),
                ),
            ),
        ),
    )
    if not isinstance(configured, Mapping):
        return {}

    textures = {}
    for model, texture in configured.items():
        if texture is None:
            textures[str(model)] = ""
            continue
        texture = str(texture)
        textures[str(model)] = (
            "" if texture.strip().lower() in {"", "none"} else texture
        )
    return textures


def lag_curve_models(config: Mapping) -> tuple[str, ...]:
    plot_config = plotting_config(config)
    configured = plot_config.get(
        "lag_curve_models",
        plot_config.get(
            "lag_models",
            plot_config.get(
                "lag_result_models",
                config.get(
                    "lag_curve_models",
                    config.get("lag_models", config.get("lag_result_models")),
                ),
            ),
        ),
    )
    if configured is None:
        return ("baseline",)
    if isinstance(configured, str):
        return (configured,)
    if not isinstance(configured, Sequence):
        raise ValueError("Lag curve models must be configured as a list of model names")

    models = tuple(str(model) for model in configured)
    if not models:
        raise ValueError("Lag curve models must contain at least one model name")
    return models


def lag_curve_conditions(config: Mapping) -> tuple[str, ...]:
    configured = plotting_config(config).get("lag_curve_conditions", CONDITIONS)
    if isinstance(configured, str):
        return (configured,)
    if not isinstance(configured, Sequence):
        raise ValueError("Lag curve conditions must be configured as a list")

    conditions = tuple(str(condition) for condition in configured)
    if not conditions:
        raise ValueError("Lag curve conditions must contain at least one condition")
    return conditions


def lag_plot_xlim(config: Mapping) -> tuple[float, float] | None:
    plot_config = plotting_config(config)
    configured = plot_config.get(
        "lag_plot_xlim",
        plot_config.get(
            "lag_curve_xlim",
            plot_config.get(
                "lag_xlim",
                config.get(
                    "lag_plot_xlim",
                    config.get("lag_curve_xlim", config.get("lag_xlim")),
                ),
            ),
        ),
    )
    if configured is None:
        return None
    if isinstance(configured, str) or not isinstance(configured, Sequence):
        raise ValueError(
            "Lag plot x-limits must be configured as a two-item list: [min_ms, max_ms]"
        )
    if len(configured) != 2:
        raise ValueError(
            "Lag plot x-limits must be configured as a two-item list: [min_ms, max_ms]"
        )

    limits = pd.to_numeric(pd.Series(list(configured)), errors="coerce")
    if limits.isna().any():
        raise ValueError("Lag plot x-limits must contain only numeric values")
    lower, upper = (float(limits.iloc[0]), float(limits.iloc[1]))
    if lower >= upper:
        raise ValueError("Lag plot x-limits must satisfy min_ms < max_ms")
    return lower, upper


def apply_lag_plot_xlim(ax: plt.Axes, config: Mapping) -> None:
    limits = lag_plot_xlim(config)
    if limits is not None:
        ax.set_xlim(*limits)


def apply_axis_text_sizes(
    ax: plt.Axes,
    *,
    label_fontsize: float,
    tick_fontsize: float,
) -> None:
    ax.xaxis.label.set_size(label_fontsize)
    ax.yaxis.label.set_size(label_fontsize)
    ax.tick_params(axis="both", which="both", labelsize=tick_fontsize)


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


def display_condition_name(condition: str) -> str:
    names = {
        "super_subject": "Multi-Subject",
        "per_subject": "Single Subject",
    }
    return names.get(condition, condition.replace("_", " ").title())


def task_group_config(config: Mapping) -> Mapping:
    plot_config = plotting_config(config)
    configured = plot_config.get("task_groups", config.get("task_groups", {}))
    return configured if isinstance(configured, Mapping) else {}


def normalize_task_group_name(group: str) -> str:
    for canonical_group, aliases in BAR_SUMMARY_GROUP_ALIASES.items():
        if any(group.casefold() == alias.casefold() for alias in aliases):
            return canonical_group
    return group


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
            grouped.setdefault(normalize_task_group_name(str(group)), []).append(task)
            assigned.add(task)
    else:
        for group, group_tasks in configured.items():
            if not isinstance(group_tasks, Sequence) or isinstance(group_tasks, str):
                continue
            group_name = normalize_task_group_name(str(group))
            for task in group_tasks:
                task = str(task)
                if task not in task_set:
                    continue
                grouped.setdefault(group_name, []).append(task)
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
        chance_level=(
            float(task_summary["metric_chance_level"].iloc[0])
            if "metric_chance_level" in task_summary
            and pd.notna(task_summary["metric_chance_level"].iloc[0])
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


def best_lag_fold_values(
    task_results: Mapping[str, pd.DataFrame] | None,
    model: str,
    lag,
    metric: MetricConfig,
) -> list[float]:
    if task_results is None or model not in task_results:
        return []
    values_by_fold = fold_values_at_lag(task_results[model], lag, metric)
    return [values_by_fold[fold] for fold in sorted(values_by_fold)]


def standard_error(values: Sequence[float]) -> float:
    finite_values = np.array(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if len(finite_values) < 2:
        return 0.0
    return float(np.std(finite_values, ddof=1) / np.sqrt(len(finite_values)))


def summary_std_error(match: pd.DataFrame) -> float:
    if "std" not in match or match.empty or pd.isna(match["std"].iloc[0]):
        return 0.0
    return float(match["std"].iloc[0])


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
    valid_rows = np.isfinite(winner_matrix).all(axis=1) & np.isfinite(other_matrix).all(
        axis=1
    )
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
            np.max(permuted_winner.mean(axis=0)) - np.max(permuted_other.mean(axis=0))
        )
        if statistic >= observed - 1e-12:
            count += 1
    return observed, count / total


def holm_adjust_p_values(p_values: Sequence[float]) -> list[float]:
    adjusted = [float("nan")] * len(p_values)
    finite = [
        (idx, float(p_value))
        for idx, p_value in enumerate(p_values)
        if np.isfinite(p_value)
    ]
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

        winner_lags = set(
            pd.to_numeric(task_results[winner]["lags"], errors="coerce").dropna()
        )
        other_lags = set(
            pd.to_numeric(task_results[model]["lags"], errors="coerce").dropna()
        )
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
        holm_adjust_p_values(
            [float(comparison["raw_p_value"]) for comparison in comparisons]
        )
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


def task_group_background_colors(config: Mapping) -> dict[str, str]:
    plot_config = plotting_config(config)
    configured = plot_config.get(
        "task_group_backgrounds",
        plot_config.get("task_group_background_colors", {}),
    )
    colors = dict(DEFAULT_TASK_GROUP_BACKGROUNDS)
    if isinstance(configured, Mapping):
        colors.update({str(group): str(color) for group, color in configured.items()})
    return colors


def task_group_background_alpha(config: Mapping) -> float:
    plot_config = plotting_config(config)
    alpha = plot_config.get(
        "task_group_background_alpha",
        plot_config.get("task_group_background_opacity"),
    )
    if alpha is None:
        return DEFAULT_TASK_GROUP_BACKGROUND_ALPHA
    return float(alpha)


def task_group_background_facecolors(
    config: Mapping,
    tasks: Sequence[str],
) -> dict[str, tuple[float, float, float, float]]:
    background_colors = task_group_background_colors(config)
    alpha = task_group_background_alpha(config)
    fallback_color = background_colors.get("Other", "#E8E8E8")
    facecolors = {}
    for group, group_tasks in grouped_tasks_for_summary(config, tasks):
        facecolor = to_rgba(background_colors.get(group, fallback_color), alpha)
        for task in group_tasks:
            facecolors[task] = facecolor
    return facecolors


def composite_rgba_over_background(
    foreground: tuple[float, float, float, float],
    background: object,
) -> tuple[float, float, float, float]:
    fg = np.asarray(to_rgba(foreground), dtype=float)
    bg = np.asarray(to_rgba(background), dtype=float)
    alpha = fg[3]
    rgb = fg[:3] * alpha + bg[:3] * (1.0 - alpha)
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def use_column_per_group_layout(config: Mapping) -> bool:
    plot_config = plotting_config(config)
    if "column_per_group_layout" in plot_config:
        return bool(plot_config["column_per_group_layout"])
    if "group_columns" in plot_config:
        return bool(plot_config["group_columns"])
    return exclude_llm_decoding_and_gpt_surprise(config)


def best_lag_bar_group_slots(
    config: Mapping,
    task_groups: Sequence[tuple[str, list[str]]],
    *,
    column_layout: bool | None = None,
) -> list[tuple[str, str, list[str], tuple[tuple[int, int], ...]]]:
    if column_layout is None:
        column_layout = use_column_per_group_layout(config)
    if column_layout:
        return [
            (
                group,
                group,
                tasks,
                tuple((row, col) for row in range(BAR_SUMMARY_GRID_ROWS)),
            )
            for col, (group, tasks) in enumerate(task_groups)
        ]

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
        for col in BAR_SUMMARY_GRID_TASK_COLS
    )
    for group, tasks in remaining_groups:
        planned.append((group, group, tasks, overflow_slots))
    return planned


def grouped_task_grid_cols(
    config: Mapping,
    task_groups: Sequence[tuple[str, list[str]]],
    *,
    column_layout: bool | None = None,
) -> int:
    if column_layout is None:
        column_layout = use_column_per_group_layout(config)
    if column_layout:
        return max(len(task_groups), 1)

    for group, tasks in task_groups:
        if not tasks:
            continue
        if group_matches_bar_layout(
            group, "Representations"
        ) or group_matches_bar_layout(group, "Semantic"):
            continue
        return BAR_SUMMARY_GRID_COLS
    return BAR_SUMMARY_PRIMARY_GRID_COLS


def grouped_axes_bounds(
    fig: plt.Figure,
    axes: Sequence[plt.Axes],
) -> tuple[float, float, float, float]:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    figure_bboxes = []
    for ax in axes:
        if ax.axison:
            bbox = ax.get_tightbbox(renderer)
            if bbox is not None:
                figure_bboxes.append(bbox.transformed(fig.transFigure.inverted()))
                continue
        figure_bboxes.append(ax.get_position())

    left = min(bbox.x0 for bbox in figure_bboxes)
    right = max(bbox.x1 for bbox in figure_bboxes)
    bottom = min(bbox.y0 for bbox in figure_bboxes)
    top = max(bbox.y1 for bbox in figure_bboxes)
    return left, right, bottom, top


def create_grouped_task_figure(
    config: Mapping,
    tasks: Sequence[str],
    *,
    figsize: tuple[float, float] = (18, 8),
    hspace: float = 0.5,
    wspace: float = 0.55,
    include_score_axes: bool = False,
) -> GroupedTaskFigure:
    task_groups = grouped_tasks_for_summary(config, sorted(tasks))
    include_scores = include_score_axes
    column_layout = use_column_per_group_layout(config) or include_scores
    fig = plt.figure(figsize=figsize)
    grid_rows = BAR_SUMMARY_GRID_ROWS + (1 if include_scores else 0)
    grid_cols = grouped_task_grid_cols(
        config,
        task_groups,
        column_layout=column_layout,
    )
    outer_grid = fig.add_gridspec(
        grid_rows,
        grid_cols,
        hspace=hspace,
        wspace=wspace,
    )
    task_axes: dict[str, plt.Axes] = {}
    group_score_axes: dict[str, plt.Axes] = {}
    group_title_axes = []
    used_slots: set[tuple[int, int]] = set()

    for group_idx, (canonical_group, _group, group_tasks, slots) in enumerate(
        best_lag_bar_group_slots(
            config,
            task_groups,
            column_layout=column_layout,
        )
    ):
        group_axes = []
        plotted_tasks = 0
        for task_idx, task in enumerate(group_tasks[: len(slots)]):
            row, col = slots[task_idx]
            if col >= grid_cols:
                continue
            if (row, col) in used_slots:
                continue
            used_slots.add((row, col))
            ax = fig.add_subplot(outer_grid[row, col])
            task_axes[task] = ax
            group_axes.append(ax)
            plotted_tasks += 1

        if group_tasks and len(group_tasks) <= len(slots):
            empty_range = range(plotted_tasks, len(slots))
        else:
            empty_range = range(0)
        for empty_idx in empty_range:
            row, col = slots[empty_idx]
            if col >= grid_cols:
                continue
            if (row, col) in used_slots:
                continue
            used_slots.add((row, col))
            empty_ax = fig.add_subplot(outer_grid[row, col])
            empty_ax.set_axis_off()
            group_axes.append(empty_ax)

        if include_scores and column_layout:
            score_slot = (BAR_SUMMARY_GRID_ROWS, group_idx)
            row, col = score_slot
            if col < grid_cols and score_slot not in used_slots:
                used_slots.add(score_slot)
                score_ax = fig.add_subplot(outer_grid[row, col])
                group_score_axes[canonical_group] = score_ax
                group_axes.append(score_ax)

        if group_axes:
            group_title_axes.append((canonical_group, group_axes))

    return GroupedTaskFigure(fig, task_axes, group_title_axes, group_score_axes)


def draw_grouped_task_backgrounds(
    layout: GroupedTaskFigure,
    config: Mapping,
    *,
    label_axis_off_groups: bool = False,
    draw_backgrounds: bool = True,
) -> None:
    background_colors = task_group_background_colors(config)
    alpha = task_group_background_alpha(config)
    fallback_color = background_colors.get("Other", "#E8E8E8")
    axes_by_slot: dict[tuple[int, int], plt.Axes] = {}
    for _group, group_axes in layout.group_axes:
        for ax in group_axes:
            spec = ax.get_subplotspec()
            axes_by_slot[(spec.rowspan.start, spec.colspan.start)] = ax
    if not axes_by_slot:
        return

    rows = sorted({row for row, _col in axes_by_slot})
    cols = sorted({col for _row, col in axes_by_slot})
    row_bounds = {}
    col_bounds = {}
    for row in rows:
        row_axes = [
            axes_by_slot[(slot_row, slot_col)]
            for slot_row, slot_col in axes_by_slot
            if slot_row == row
        ]
        row_bottom = min(ax.get_position().y0 for ax in row_axes)
        row_top = max(ax.get_position().y1 for ax in row_axes)
        row_bounds[row] = (row_bottom, row_top)
    for col in cols:
        col_axes = [
            axes_by_slot[(slot_row, slot_col)]
            for slot_row, slot_col in axes_by_slot
            if slot_col == col
        ]
        col_left = min(ax.get_position().x0 for ax in col_axes)
        col_right = max(ax.get_position().x1 for ax in col_axes)
        col_bounds[col] = (col_left, col_right)

    outer_pad_x = 0.04
    outer_pad_y = GROUP_TITLE_VERTICAL_PAD
    col_edges = {}
    for idx, col in enumerate(cols):
        left, right = col_bounds[col]
        if idx == 0:
            cell_left = max(0.0, left - outer_pad_x)
        else:
            prev_col = cols[idx - 1]
            cell_left = (col_bounds[prev_col][1] + left) / 2
        if idx == len(cols) - 1:
            cell_right = min(1.0, right + outer_pad_x)
        else:
            next_col = cols[idx + 1]
            cell_right = (right + col_bounds[next_col][0]) / 2
        col_edges[col] = (cell_left, cell_right)
    row_edges = {}
    rows_by_top = sorted(rows, key=lambda row: row_bounds[row][1], reverse=True)
    for idx, row in enumerate(rows_by_top):
        bottom, top = row_bounds[row]
        if idx == 0:
            cell_top = min(1.0, top + outer_pad_y)
        else:
            prev_row = rows_by_top[idx - 1]
            cell_top = (row_bounds[prev_row][0] + top) / 2
        if idx == len(rows_by_top) - 1:
            cell_bottom = max(0.0, bottom - outer_pad_y)
        else:
            next_row = rows_by_top[idx + 1]
            cell_bottom = (bottom + row_bounds[next_row][1]) / 2
        row_edges[row] = (cell_bottom, cell_top)

    if draw_backgrounds:
        for group, group_axes in layout.group_axes:
            color = background_colors.get(group, fallback_color)
            facecolor = to_rgba(color, alpha)
            for ax in group_axes:
                spec = ax.get_subplotspec()
                row = spec.rowspan.start
                col = spec.colspan.start
                rect_left, rect_right = col_edges[col]
                rect_bottom, rect_top = row_edges[row]
                rect = plt.Rectangle(
                    (rect_left, rect_bottom),
                    rect_right - rect_left,
                    rect_top - rect_bottom,
                    transform=layout.fig.transFigure,
                    facecolor=facecolor,
                    edgecolor="none",
                    clip_on=False,
                    zorder=-1,
                )
                layout.fig.add_artist(rect)

    for group, group_axes in layout.group_axes:
        visible_axes = (
            list(group_axes)
            if label_axis_off_groups
            else [ax for ax in group_axes if ax.axison]
        )
        if not visible_axes:
            continue
        slots = [
            (ax.get_subplotspec().rowspan.start, ax.get_subplotspec().colspan.start)
            for ax in group_axes
        ]
        left = min(col_edges[col][0] for _row, col in slots)
        right = max(col_edges[col][1] for _row, col in slots)
        top = max(row_edges[row][1] for row, _col in slots)
        top_slots = [
            (row, col) for row, col in slots if np.isclose(row_edges[row][1], top)
        ]
        if top_slots:
            left = min(col_edges[col][0] for _row, col in top_slots)
            right = max(col_edges[col][1] for _row, col in top_slots)
        layout.fig.text(
            (left + right) / 2,
            min(1.0, top - GROUP_TITLE_TOP_INSET),
            group,
            ha="center",
            va="top",
            fontsize=GROUP_TITLE_FONTSIZE,
            weight="bold",
        )


def task_score_value(value: float, metric: MetricConfig) -> float:
    metric_name = f"{metric.column} {metric.label}".casefold()
    if "roc_auc" in metric_name or "auc_roc" in metric_name or "auc-roc" in metric_name:
        return 2.0 * float(value) - 1.0
    if "correlation" in metric_name or "corr" in metric_name:
        return float(value)
    return float("nan")


def summary_with_scores(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return summary.copy()
    scored = summary.copy()
    scored["score"] = [
        task_score_value(
            float(row["value"]),
            metric_config_from_summary(pd.DataFrame([row])),
        )
        for row in scored.to_dict("records")
    ]
    return scored


def grouped_score_rows(
    summary: pd.DataFrame,
    config: Mapping,
    *,
    condition: str | None = None,
) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    scored = summary_with_scores(summary).dropna(subset=["score"])
    if scored.empty:
        return pd.DataFrame()

    task_groups = grouped_tasks_for_summary(config, sorted(scored["task"].unique()))
    task_to_group = {task: group for group, tasks in task_groups for task in tasks}
    scored = scored.assign(group=scored["task"].map(task_to_group))
    scored = scored.dropna(subset=["group"])

    group_columns = ["group", "model"]
    if condition is None and "condition" in scored.columns:
        group_columns.insert(0, "condition")
    grouped = (
        scored.groupby(group_columns, as_index=False)["score"]
        .mean()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )
    if condition is not None:
        grouped.insert(0, "condition", condition)
    return grouped


def overall_score_rows(summary: pd.DataFrame, config: Mapping) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    scored = summary_with_scores(summary).dropna(subset=["score"])
    if scored.empty:
        return pd.DataFrame()
    group_columns = ["model"]
    if "condition" in scored.columns:
        group_columns.insert(0, "condition")
    return (
        scored.groupby(group_columns, as_index=False)["score"]
        .mean()
        .sort_values(group_columns)
        .reset_index(drop=True)
    )


def plot_score_axis(
    ax: plt.Axes,
    score_summary: pd.DataFrame,
    models: Sequence[str],
    colors: Mapping[str, str],
    config: Mapping,
    *,
    title: str,
) -> None:
    x_positions = list(range(len(models)))
    values = []
    for model in models:
        match = score_summary[score_summary["model"] == model]
        values.append(
            float(match["score"].iloc[0]) if not match.empty else float("nan")
        )
    for x_position, model, value in zip(x_positions, models, values):
        if not np.isfinite(value):
            continue
        ax.bar(
            [x_position],
            [value],
            color=colors[model],
            edgecolor=colors[model],
            width=0.7,
            zorder=3,
        )
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.75)
    ax.set_title(title)
    ax.set_ylabel("Overall score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [display_model_name(config, model) for model in models],
        rotation=35,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.25)


def create_electrode_region_figure(electrodes: pd.DataFrame) -> plt.Figure:
    from nilearn import plotting

    electrodes = electrodes[electrodes["region_group"] != "unassigned"].copy()
    counts = (
        electrodes["region_group"]
        .value_counts()
        .sort_values(ascending=False, kind="mergesort")
    )
    groups = list(counts.index)
    cmap = plt.get_cmap("tab20", max(len(groups), 1))
    colors = {group: to_hex(cmap(i)) for i, group in enumerate(groups)}

    fig = plt.figure(figsize=(13, 5.5), constrained_layout=False)
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.15, 1.15, 0.95],
        left=0.04,
        right=0.985,
        bottom=0.18,
        top=0.86,
        wspace=0.18,
    )
    axes = [
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
    ]
    bar_ax = fig.add_subplot(grid[0, 2])

    views = [
        ("Left", "l"),
        ("Right", "r"),
    ]

    marker_coords = electrodes[["x", "y", "z"]].to_numpy(float)
    for ax, (title, display_mode) in zip(axes, views):
        display = plotting.plot_glass_brain(
            None,
            display_mode=display_mode,
            colorbar=False,
            figure=fig,
            axes=ax,
            title=title,
            black_bg=False,
            annotate=True,
        )
        for group in groups:
            mask = electrodes["region_group"] == group
            display.add_markers(
                marker_coords[mask],
                marker_color=colors[group],
                marker_size=34,
                alpha=0.9,
            )

    if counts.empty:
        bar_ax.text(
            0.5,
            0.5,
            "No electrodes matched REGION_GROUPS",
            ha="center",
            va="center",
            transform=bar_ax.transAxes,
            fontsize=13,
        )
        bar_ax.set_axis_off()
    else:
        bar_colors = [colors[group] for group in counts.index]
        x = np.arange(len(counts))
        bar_ax.bar(x, counts.values, color=bar_colors)
        bar_ax.set_xticks(x)
        bar_ax.set_xticklabels(
            [display_region_name(str(group)) for group in counts.index],
            rotation=45,
            ha="right",
        )
        bar_ax.set_ylabel("Electrodes")
        bar_ax.set_title("Electrodes per region")
        bar_ax.spines["top"].set_visible(False)
        bar_ax.spines["right"].set_visible(False)
        for idx, value in enumerate(counts.values):
            bar_ax.text(
                idx,
                value + max(counts.max() * 0.02, 0.2),
                str(int(value)),
                ha="center",
            )

    fig.suptitle("Destrieux Atlas Region Electrodes", fontsize=16, y=0.97)
    return fig


def plot_condition_score_axis(
    ax: plt.Axes,
    score_summary: pd.DataFrame,
    models: Sequence[str],
    conditions: Sequence[str],
    colors: Mapping[str, str],
    config: Mapping,
    *,
    title: str,
) -> None:
    x_positions = np.arange(len(models), dtype=float)
    bar_width = min(0.32, 0.72 / max(len(conditions), 1))
    offsets = (
        np.linspace(-0.18, 0.18, len(conditions))
        if len(conditions) > 1
        else np.array([0.0])
    )
    hatches = {
        "super_subject": "",
        "per_subject": "///",
    }
    for condition, offset in zip(conditions, offsets):
        condition_summary = score_summary[score_summary["condition"] == condition]
        for model_idx, model in enumerate(models):
            match = condition_summary[condition_summary["model"] == model]
            if match.empty:
                continue
            ax.bar(
                [x_positions[model_idx] + offset],
                [float(match["score"].iloc[0])],
                color=colors[model],
                edgecolor="#333333" if hatches.get(condition) else colors[model],
                hatch=hatches.get(condition, "\\\\"),
                linewidth=0.8 if hatches.get(condition) else 0.0,
                alpha=0.9,
                width=bar_width,
                zorder=3,
            )
    ax.axhline(0.0, color="#333333", linewidth=0.8, alpha=0.75)
    ax.set_title(title)
    ax.set_ylabel("Overall score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(
        [display_model_name(config, model) for model in models],
        rotation=35,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.25)


def plot_group_score_axes(
    layout: GroupedTaskFigure,
    summary: pd.DataFrame,
    models: Sequence[str],
    colors: Mapping[str, str],
    config: Mapping,
) -> None:
    if not layout.group_score_axes:
        return
    group_scores = grouped_score_rows(summary, config)
    if group_scores.empty:
        for ax in layout.group_score_axes.values():
            ax.set_axis_off()
        return
    for group, ax in layout.group_score_axes.items():
        score_summary = group_scores[group_scores["group"] == group]
        if score_summary.empty:
            ax.set_axis_off()
            continue
        plot_score_axis(ax, score_summary, models, colors, config, title="Overall")


def plot_combined_group_score_axes(
    layout: GroupedTaskFigure,
    summary: pd.DataFrame,
    models: Sequence[str],
    conditions: Sequence[str],
    colors: Mapping[str, str],
    config: Mapping,
) -> None:
    if not layout.group_score_axes:
        return
    group_scores = grouped_score_rows(summary, config)
    if group_scores.empty:
        for ax in layout.group_score_axes.values():
            ax.set_axis_off()
        return
    for group, ax in layout.group_score_axes.items():
        score_summary = group_scores[group_scores["group"] == group]
        if score_summary.empty:
            ax.set_axis_off()
            continue
        plot_condition_score_axis(
            ax,
            score_summary,
            models,
            conditions,
            colors,
            config,
            title="Overall",
        )


def plot_overall_score_summary(
    summary: pd.DataFrame,
    output_dir: Path,
    formats: Sequence[str],
    colors: Mapping[str, str],
    config: Mapping,
    *,
    suffix: str,
    title: str,
) -> None:
    score_summary = overall_score_rows(summary, config)
    if score_summary.empty:
        return
    models = configured_model_order(score_summary["model"].unique(), config)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if (
        "condition" in score_summary.columns
        and score_summary["condition"].nunique() > 1
    ):
        available_conditions = set(
            str(condition) for condition in score_summary["condition"]
        )
        conditions = [
            condition for condition in CONDITIONS if condition in available_conditions
        ]
        conditions.extend(sorted(available_conditions - set(conditions)))
        plot_condition_score_axis(
            ax,
            score_summary,
            models,
            conditions,
            colors,
            config,
            title=title,
        )
    else:
        plot_score_axis(ax, score_summary, models, colors, config, title=title)
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.24, top=0.88)
    save_figure(fig, output_dir / f"overall_score_{suffix}", formats)


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
    models = configured_model_order(summary["model"].unique(), config)
    tasks = sorted(summary["task"].unique())
    layout = create_grouped_task_figure(
        config,
        tasks,
        figsize=(18, 10.8) if include_overall_scores(config) else (18, 8.8),
        hspace=0.75 if check_best_lag_significance(config) else 0.5,
        include_score_axes=include_overall_scores(config),
    )
    fig = layout.fig
    x_positions = list(range(len(models)))
    plot_style = best_lag_summary_plot_style(config)
    show_bar_error_bars = include_bar_error_bars(config)
    show_significance = check_best_lag_significance(config)
    correct_significance = correct_best_lag_significance(config)
    model_bar_textures = best_lag_model_bar_textures(config)

    # Neutral color for non-winners
    neutral_color = "#95a5a6"

    for task, ax in layout.task_axes.items():
        task_summary = summary[summary["task"] == task]
        metric = metric_config_from_summary(task_summary)
        task_results = condition_results.get(task, {}) if condition_results else None
        bar_start = bar_start_for_task(config, task)

        # --- PER-SUBPLOT HIGHLIGHT LOGIC ---
        # Find the model with the highest value in this specific task
        if not task_summary.empty:
            best_model_in_task = task_summary.loc[
                task_summary["value"].idxmax(), "model"
            ]
        else:
            best_model_in_task = None
        # -----------------------------------

        values = []
        errors = []
        for model in models:
            match = task_summary[task_summary["model"] == model]
            if match.empty:
                values.append(float("nan"))
                errors.append(0.0)
                continue

            summary_value = float(match["value"].iloc[0])
            fold_values = best_lag_fold_values(
                task_results,
                model,
                match["lag"].iloc[0],
                metric,
            )
            mean_value = (
                float(np.nanmean(fold_values)) if fold_values else summary_value
            )
            sem = standard_error(fold_values)
            display_error = sem if fold_values else summary_std_error(match)
            values.append(mean_value)
            errors.append(display_error)

            # Assign color: highlight if it's the winner of THIS subplot
            is_best = model == best_model_in_task
            current_color = BEST_COLOR if is_best else neutral_color

            hatch = model_bar_textures.get(model, "")
            x_position = x_positions[models.index(model)]
            edge_color = "#333333" if (hatch or is_best) else current_color
            line_width = 0.8 if (hatch or is_best) else 0.0

            if plot_style == "bar":
                ax.bar(
                    [x_position],
                    [mean_value - bar_start],
                    bottom=bar_start,
                    yerr=[display_error] if show_bar_error_bars else None,
                    error_kw={"elinewidth": 1.0, "capsize": 3, "capthick": 1.0},
                    color=current_color,
                    edgecolor=edge_color,
                    linewidth=line_width,
                    hatch=hatch,
                    width=0.7,
                    zorder=3 if is_best else 2,
                )
            else:
                pt_alpha = 0.7 if is_best else 0.3
                if fold_values:
                    jitter = (
                        np.linspace(-0.18, 0.18, len(fold_values))
                        if len(fold_values) > 1
                        else np.array([0.0])
                    )
                    ax.scatter(
                        x_position + jitter,
                        fold_values,
                        color=current_color,
                        alpha=pt_alpha,
                        s=28,
                        edgecolors="none",
                        zorder=2,
                    )
                else:
                    ax.scatter(
                        [x_position],
                        [summary_value],
                        color=current_color,
                        alpha=pt_alpha,
                        s=28,
                        edgecolors="none",
                        zorder=2,
                    )

                ax.errorbar(
                    [x_position],
                    [mean_value],
                    yerr=[sem],
                    fmt="none",
                    ecolor=current_color,
                    elinewidth=1.6,
                    capsize=4,
                    capthick=1.6,
                    zorder=3,
                )
                ax.hlines(
                    mean_value,
                    x_position - 0.24,
                    x_position + 0.24,
                    color=current_color,
                    linewidth=4.0 if is_best else 2.0,
                    zorder=4,
                )

        # Standard axis formatting
        ax.set_title(display_task_name(config, task))
        ax.set_ylabel(
            str(task_summary["metric_label"].iloc[0])
            if not task_summary.empty
            else "Metric"
        )
        ax.set_xticks(x_positions)
        ax.set_xticklabels(
            [display_model_name(config, model) for model in models],
            rotation=35,
            ha="right",
        )
        apply_axis_text_sizes(
            ax,
            label_fontsize=BEST_LAG_AXIS_LABEL_FONTSIZE,
            tick_fontsize=BEST_LAG_TICK_LABEL_FONTSIZE,
        )
        ax.grid(axis="y", alpha=0.25)
        plot_chance_level(ax, metric)
        if plot_style == "bar":
            ax.axhline(bar_start, color="#333333", linewidth=0.8, alpha=0.75)
        apply_metric_ylim(ax, metric)

        if show_significance and condition_results is not None:
            comparisons = best_lag_significance_tests(
                task_summary,
                condition_results.get(task, {}),
                metric,
                correct_multiple_comparisons=correct_significance,
            )
            draw_significance_annotations(ax, comparisons, models, values, errors)

    # --- LEGEND ---
    # In this mode, the legend shows the "original" colors so the reader knows
    # what the colors mean when they do appear as highlights.
    if plot_style == "bar":
        handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=colors[model],
                edgecolor=(
                    "#333333" if model_bar_textures.get(model, "") else colors[model]
                ),
                linewidth=0.8 if model_bar_textures.get(model, "") else 0.0,
                hatch=model_bar_textures.get(model, ""),
                label=display_model_name(config, model),
            )
            for model in models
        ]
    else:
        handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="-",
                color=colors[model],
                markerfacecolor=colors[model],
                markeredgecolor="none",
                linewidth=3,
                markersize=6,
                label=display_model_name(config, model),
            )
            for model in models
        ]

    fig.text(
        0.01,
        0.985,
        display_condition_name(condition),
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
    draw_grouped_task_backgrounds(layout, config)
    save_figure(fig, output_dir / f"best_lag_summary_{condition}", formats)


def plot_combined_best_lag_summary(
    summary: pd.DataFrame,
    loaded: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    output_dir: Path,
    formats: Sequence[str],
    colors: Mapping[str, str],
    config: Mapping | None = None,
) -> None:
    if summary.empty or "condition" not in summary.columns:
        return

    config = config or {}
    models = configured_model_order(summary["model"].unique(), config)
    tasks = sorted(summary["task"].unique())
    available_conditions = set(str(condition) for condition in summary["condition"])
    conditions = [
        condition for condition in CONDITIONS if condition in available_conditions
    ]
    conditions.extend(sorted(available_conditions - set(conditions)))
    if not conditions:
        return

    layout = create_grouped_task_figure(
        config,
        tasks,
        figsize=(19, 10.8) if include_overall_scores(config) else (19, 8.8),
        hspace=0.5,
        include_score_axes=include_overall_scores(config),
    )
    fig = layout.fig
    condition_markers = {
        "super_subject": "o",
        "per_subject": "s",
    }
    condition_linestyles = {
        "super_subject": "-",
        "per_subject": "--",
    }
    condition_offsets = (
        np.linspace(-0.18, 0.18, len(conditions))
        if len(conditions) > 1
        else np.array([0.0])
    )
    condition_offset_by_name = dict(zip(conditions, condition_offsets))
    x_positions = list(range(len(models)))
    plot_style = best_lag_summary_plot_style(config)
    show_bar_error_bars = include_bar_error_bars(config)
    bar_width = min(0.32, 0.72 / max(len(conditions), 1))
    condition_hatches = {
        "super_subject": "",
        "per_subject": "///",
    }
    model_bar_textures = best_lag_model_bar_textures(config)

    for task, ax in layout.task_axes.items():
        task_summary = summary[summary["task"] == task]
        metric = metric_config_from_summary(task_summary)
        bar_start = bar_start_for_task(config, task)

        for condition in conditions:
            condition_summary = task_summary[task_summary["condition"] == condition]
            task_results = loaded.get(condition, {}).get(task, {})
            condition_offset = float(condition_offset_by_name[condition])
            marker = condition_markers.get(condition, "D")
            linestyle = condition_linestyles.get(condition, ":")
            condition_hatch = condition_hatches.get(condition, "\\\\")

            for model_idx, model in enumerate(models):
                match = condition_summary[condition_summary["model"] == model]
                if match.empty:
                    continue

                summary_value = float(match["value"].iloc[0])
                fold_values = best_lag_fold_values(
                    task_results,
                    model,
                    match["lag"].iloc[0],
                    metric,
                )
                mean_value = (
                    float(np.nanmean(fold_values)) if fold_values else summary_value
                )
                sem = standard_error(fold_values)
                display_error = sem if fold_values else summary_std_error(match)
                color = colors[model]
                x_position = model_idx + condition_offset
                hatch = f"{model_bar_textures.get(model, '')}{condition_hatch}"

                if plot_style == "bar":
                    ax.bar(
                        [x_position],
                        [mean_value - bar_start],
                        bottom=bar_start,
                        yerr=[display_error] if show_bar_error_bars else None,
                        error_kw={
                            "elinewidth": 1.0,
                            "capsize": 2.5,
                            "capthick": 1.0,
                        },
                        color=color,
                        edgecolor="#333333" if hatch else color,
                        linewidth=0.8 if hatch else 0.0,
                        hatch=hatch,
                        alpha=0.9,
                        width=bar_width,
                        zorder=3,
                    )
                else:
                    if fold_values:
                        jitter = (
                            np.linspace(-0.055, 0.055, len(fold_values))
                            if len(fold_values) > 1
                            else np.array([0.0])
                        )
                        ax.scatter(
                            x_position + jitter,
                            fold_values,
                            color=color,
                            alpha=0.26,
                            s=22,
                            marker=marker,
                            edgecolors="none",
                            zorder=2,
                        )
                    else:
                        ax.scatter(
                            [x_position],
                            [summary_value],
                            color=color,
                            alpha=0.26,
                            s=22,
                            marker=marker,
                            edgecolors="none",
                            zorder=2,
                        )

                    ax.errorbar(
                        [x_position],
                        [mean_value],
                        yerr=[sem],
                        fmt=marker,
                        color=color,
                        markerfacecolor=(
                            "white" if condition == "per_subject" else color
                        ),
                        markeredgecolor=color,
                        markersize=5,
                        ecolor=color,
                        elinewidth=1.3,
                        capsize=3,
                        capthick=1.3,
                        zorder=4,
                    )
                    ax.hlines(
                        mean_value,
                        x_position - 0.09,
                        x_position + 0.09,
                        color=color,
                        linewidth=3.0,
                        linestyle=linestyle,
                        zorder=3,
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
        apply_axis_text_sizes(
            ax,
            label_fontsize=BEST_LAG_AXIS_LABEL_FONTSIZE,
            tick_fontsize=BEST_LAG_TICK_LABEL_FONTSIZE,
        )
        ax.grid(axis="y", alpha=0.25)
        plot_chance_level(ax, metric)
        if plot_style == "bar":
            ax.axhline(bar_start, color="#333333", linewidth=0.8, alpha=0.75)
        apply_metric_ylim(ax, metric)

    plot_combined_group_score_axes(layout, summary, models, conditions, colors, config)

    if plot_style == "bar":
        model_handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=colors[model],
                edgecolor=(
                    "#333333" if model_bar_textures.get(model, "") else colors[model]
                ),
                linewidth=0.8 if model_bar_textures.get(model, "") else 0.0,
                hatch=model_bar_textures.get(model, ""),
                label=display_model_name(config, model),
            )
            for model in models
        ]
    else:
        model_handles = [
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="-",
                color=colors[model],
                markerfacecolor=colors[model],
                markeredgecolor="none",
                linewidth=3,
                markersize=6,
                label=display_model_name(config, model),
            )
            for model in models
        ]
    if plot_style == "bar":
        condition_handles = [
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor="white",
                edgecolor="#333333",
                hatch=condition_hatches.get(condition, "\\\\"),
                label=display_condition_name(condition),
            )
            for condition in conditions
        ]
    else:
        condition_handles = [
            Line2D(
                [0],
                [0],
                marker=condition_markers.get(condition, "D"),
                linestyle=condition_linestyles.get(condition, ":"),
                color="#333333",
                markerfacecolor="white" if condition == "per_subject" else "#333333",
                markeredgecolor="#333333",
                linewidth=2,
                markersize=6,
                label=display_condition_name(condition),
            )
            for condition in conditions
        ]
    fig.text(
        0.01,
        0.985,
        "Best Lag Summary",
        ha="left",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    fig.legend(
        handles=[*model_handles, *condition_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(model_handles) + len(condition_handles), 6),
        frameon=False,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.1, top=0.86)
    draw_grouped_task_backgrounds(layout, config)
    save_figure(fig, output_dir / "best_lag_summary_combined", formats)
    if include_overall_scores(config):
        plot_overall_score_summary(
            summary,
            output_dir,
            formats,
            colors,
            config,
            suffix="combined",
            title="Overall Score",
        )


def plot_baseline_condition_comparison(
    summary: pd.DataFrame,
    loaded: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    output_dir: Path,
    formats: Sequence[str],
    config: Mapping | None = None,
    *,
    baseline_model: str = "baseline",
) -> None:
    if summary.empty or "condition" not in summary.columns:
        return

    config = config or {}
    available_conditions = set(str(condition) for condition in summary["condition"])
    conditions = [
        condition for condition in CONDITIONS if condition in available_conditions
    ]
    if "super_subject" not in conditions or "per_subject" not in conditions:
        return

    baseline_summary = summary[
        (summary["model"] == baseline_model)
        & (summary["condition"].isin(("super_subject", "per_subject")))
    ].copy()
    if baseline_summary.empty:
        return

    condition_counts = baseline_summary.groupby("task")["condition"].nunique()
    tasks = sorted(condition_counts[condition_counts == 2].index)
    if not tasks:
        return

    layout = create_grouped_task_figure(
        config,
        tasks,
        figsize=(18, 8.8),
        hspace=0.5,
    )
    fig = layout.fig
    x_positions = np.arange(2, dtype=float)
    condition_labels = {
        "super_subject": "Multi-Subject",
        "per_subject": "Single",
    }
    loser_color = "#9A9A9A"

    for task, ax in layout.task_axes.items():
        task_summary = baseline_summary[baseline_summary["task"] == task]
        metric = metric_config_from_summary(task_summary)
        bar_start = bar_start_for_task(config, task)
        by_condition = {
            str(row["condition"]): row for row in task_summary.to_dict("records")
        }
        values = {
            condition: float(by_condition[condition]["value"])
            for condition in ("super_subject", "per_subject")
        }
        best_condition = (
            max(values, key=values.get)
            if metric.higher_is_better
            else min(values, key=values.get)
        )

        for condition_idx, condition in enumerate(("super_subject", "per_subject")):
            item = by_condition[condition]
            condition_results = loaded.get(condition, {}).get(task, {})
            fold_values = best_lag_fold_values(
                condition_results,
                baseline_model,
                item["lag"],
                metric,
            )
            mean_value = (
                float(np.nanmean(fold_values)) if fold_values else values[condition]
            )
            display_error = (
                standard_error(fold_values)
                if fold_values
                else summary_std_error(pd.DataFrame([item]))
            )
            color = BEST_COLOR if condition == best_condition else loser_color
            edgecolor = "#333333" if condition == best_condition else loser_color
            ax.bar(
                [x_positions[condition_idx]],
                [mean_value - bar_start],
                bottom=bar_start,
                yerr=[display_error] if display_error > 0 else None,
                error_kw={
                    "elinewidth": 1.0,
                    "capsize": 2.5,
                    "capthick": 1.0,
                },
                color=color,
                edgecolor=edgecolor,
                linewidth=0.8,
                width=0.62,
                zorder=3,
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
            [
                condition_labels[condition]
                for condition in ("super_subject", "per_subject")
            ]
        )
        apply_axis_text_sizes(
            ax,
            label_fontsize=BEST_LAG_AXIS_LABEL_FONTSIZE,
            tick_fontsize=BEST_LAG_TICK_LABEL_FONTSIZE,
        )
        ax.grid(axis="y", alpha=0.25)
        plot_chance_level(ax, metric)
        ax.axhline(bar_start, color="#333333", linewidth=0.8, alpha=0.75)
        apply_metric_ylim(ax, metric)

    model_name = display_model_name(config, baseline_model)
    fig.text(
        0.01,
        0.985,
        f"{model_name} Single- vs Multi-Subject",
        ha="left",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    fig.legend(
        handles=[
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=BEST_COLOR,
                edgecolor="#333333",
                linewidth=0.8,
                label="Better",
            ),
            plt.Rectangle(
                (0, 0),
                1,
                1,
                facecolor=loser_color,
                edgecolor=loser_color,
                linewidth=0.8,
                label="Lower bar",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.1, top=0.86)
    draw_grouped_task_backgrounds(layout, config)
    save_figure(fig, output_dir / "baseline_condition_comparison", formats)


def plot_lag_curves(
    loaded: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
    colors: Mapping[str, str],
) -> None:
    selected_models = lag_curve_models(config)
    selected_model_set = set(selected_models)
    selected_conditions = lag_curve_conditions(config)
    lag_tasks = sorted(
        {
            task
            for condition in selected_conditions
            for task, model_results in loaded.get(condition, {}).items()
            if selected_model_set.intersection(model_results)
        }
    )
    if not lag_tasks:
        return

    layout = create_grouped_task_figure(config, lag_tasks, figsize=(18, 8.8))
    fig = layout.fig
    condition_styles = {
        "super_subject": "-",
        "per_subject": "-",
    }
    multi_model_condition_styles = {
        "super_subject": "-",
        "per_subject": "--",
    }
    condition_colors = {
        "super_subject": "#1F4E79",
        "per_subject": "#2CA7A0",
    }

    for task, ax in layout.task_axes.items():
        metric = get_metric_config(config, task)
        for condition in selected_conditions:
            for model in selected_models:
                df = loaded.get(condition, {}).get(task, {}).get(model)
                if df is None or metric.column not in df.columns:
                    continue
                curve = curve_for_metric(df, metric)
                if len(selected_models) == 1:
                    line_color = condition_colors.get(
                        condition,
                        colors.get(model, DEFAULT_COLORS.get(model, "#333333")),
                    )
                    line_label = display_condition_name(condition)
                else:
                    line_color = colors.get(model, DEFAULT_COLORS.get(model, "#333333"))
                    line_label = (
                        f"{display_model_name(config, model)} "
                        f"{display_condition_name(condition)}"
                    )
                ax.plot(
                    curve["lags"],
                    curve[metric.column],
                    linestyle=(
                        multi_model_condition_styles
                        if len(selected_models) > 1
                        else condition_styles
                    ).get(condition, "-"),
                    linewidth=1.8,
                    label=line_label,
                    color=line_color,
                )
                errors = lag_curve_error_values(df.loc[curve.index], metric)
                if errors is not None:
                    y_values = curve[metric.column].to_numpy(dtype=float)
                    error_values = errors.to_numpy(dtype=float)
                    ax.fill_between(
                        curve["lags"],
                        y_values - error_values,
                        y_values + error_values,
                        color=line_color,
                        alpha=0.16,
                        linewidth=0,
                        zorder=1,
                    )

        ax.set_title(display_task_name(config, task))
        ax.set_xlabel("Lag relative to word onset (ms)")
        ax.set_ylabel(metric.label)
        apply_axis_text_sizes(
            ax,
            label_fontsize=LAG_PLOT_AXIS_LABEL_FONTSIZE,
            tick_fontsize=LAG_PLOT_TICK_LABEL_FONTSIZE,
        )
        ax.axvline(0, color="#333333", linewidth=0.8, alpha=0.5)
        plot_chance_level(ax, metric)
        ax.grid(alpha=0.25)
        apply_lag_plot_xlim(ax, config)
        apply_metric_ylim(ax, metric)

    handles = []
    for condition in selected_conditions:
        for model in selected_models:
            if not any(
                model in loaded.get(condition, {}).get(task, {}) for task in lag_tasks
            ):
                continue
            if len(selected_models) == 1:
                color = condition_colors.get(
                    condition,
                    colors.get(model, DEFAULT_COLORS.get(model, "#333333")),
                )
                label = display_condition_name(condition)
            else:
                color = colors.get(model, DEFAULT_COLORS.get(model, "#333333"))
                label = (
                    f"{display_model_name(config, model)} "
                    f"{display_condition_name(condition)}"
                )
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=(
                        multi_model_condition_styles
                        if len(selected_models) > 1
                        else condition_styles
                    ).get(condition, "-"),
                    linewidth=1.8,
                    label=label,
                )
            )
    fig.text(
        0.01,
        0.985,
        "Lag Curves",
        ha="left",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(handles)),
        frameon=False,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.1, top=0.86)
    draw_grouped_task_backgrounds(layout, config)
    remove_legacy_lag_curve_outputs(output_dir, loaded, formats)
    save_figure(fig, output_dir / "lag_curves", formats)


def plot_lag_curves_with_best_lags(
    loaded: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
    colors: Mapping[str, str],
) -> None:
    selected_models = lag_curve_models(config)
    selected_model_set = set(selected_models)
    selected_conditions = lag_curve_conditions(config)
    lag_tasks = sorted(
        {
            task
            for condition in selected_conditions
            for task, model_results in loaded.get(condition, {}).items()
            if selected_model_set.intersection(model_results)
        }
    )
    if not lag_tasks:
        return

    layout = create_grouped_task_figure(config, lag_tasks, figsize=(18, 8.8))
    fig = layout.fig
    condition_styles = {
        "super_subject": "-",
        "per_subject": "-",
    }
    multi_model_condition_styles = {
        "super_subject": "-",
        "per_subject": "--",
    }
    condition_colors = {
        "super_subject": "#1F4E79",
        "per_subject": "#2CA7A0",
    }

    for task, ax in layout.task_axes.items():
        metric = get_metric_config(config, task)
        for condition in selected_conditions:
            for model in selected_models:
                df = loaded.get(condition, {}).get(task, {}).get(model)
                if df is None or metric.column not in df.columns:
                    continue
                curve = curve_for_metric(df, metric)
                if curve.empty:
                    continue
                if len(selected_models) == 1:
                    line_color = condition_colors.get(
                        condition,
                        colors.get(model, DEFAULT_COLORS.get(model, "#333333")),
                    )
                    line_label = display_condition_name(condition)
                else:
                    line_color = colors.get(model, DEFAULT_COLORS.get(model, "#333333"))
                    line_label = (
                        f"{display_model_name(config, model)} "
                        f"{display_condition_name(condition)}"
                    )
                line_style = (
                    multi_model_condition_styles
                    if len(selected_models) > 1
                    else condition_styles
                ).get(condition, "-")
                ax.plot(
                    curve["lags"],
                    curve[metric.column],
                    linestyle=line_style,
                    linewidth=1.8,
                    label=line_label,
                    color=line_color,
                )
                errors = lag_curve_error_values(df.loc[curve.index], metric)
                if errors is not None:
                    y_values = curve[metric.column].to_numpy(dtype=float)
                    error_values = errors.to_numpy(dtype=float)
                    ax.fill_between(
                        curve["lags"],
                        y_values - error_values,
                        y_values + error_values,
                        color=line_color,
                        alpha=0.16,
                        linewidth=0,
                        zorder=1,
                    )

                best = select_best_lag(
                    df.loc[curve.index],
                    metric,
                    task=task,
                    model=model,
                )
                best_lag = float(best["lags"])
                best_value = float(best[metric.column])
                ax.axvline(
                    best_lag,
                    color=line_color,
                    linestyle="--",
                    linewidth=1.1,
                    alpha=0.85,
                    zorder=2,
                )
                ax.text(
                    best_lag,
                    best_value,
                    f"{best_lag:g} ms",
                    color=line_color,
                    fontsize=8.5,
                    ha="center",
                    va="bottom",
                    rotation=90,
                    rotation_mode="anchor",
                    path_effects=[
                        patheffects.withStroke(linewidth=2.5, foreground="white")
                    ],
                    zorder=4,
                )

        ax.set_title(display_task_name(config, task))
        ax.set_xlabel("Lag relative to word onset (ms)")
        ax.set_ylabel(metric.label)
        apply_axis_text_sizes(
            ax,
            label_fontsize=LAG_PLOT_AXIS_LABEL_FONTSIZE,
            tick_fontsize=LAG_PLOT_TICK_LABEL_FONTSIZE,
        )
        ax.axvline(0, color="#333333", linewidth=0.8, alpha=0.5)
        plot_chance_level(ax, metric)
        ax.grid(alpha=0.25)
        apply_lag_plot_xlim(ax, config)
        apply_metric_ylim(ax, metric)

    handles = []
    for condition in selected_conditions:
        for model in selected_models:
            if not any(
                model in loaded.get(condition, {}).get(task, {}) for task in lag_tasks
            ):
                continue
            if len(selected_models) == 1:
                color = condition_colors.get(
                    condition,
                    colors.get(model, DEFAULT_COLORS.get(model, "#333333")),
                )
                label = display_condition_name(condition)
            else:
                color = colors.get(model, DEFAULT_COLORS.get(model, "#333333"))
                label = (
                    f"{display_model_name(config, model)} "
                    f"{display_condition_name(condition)}"
                )
            handles.append(
                plt.Line2D(
                    [0],
                    [0],
                    color=color,
                    linestyle=(
                        multi_model_condition_styles
                        if len(selected_models) > 1
                        else condition_styles
                    ).get(condition, "-"),
                    linewidth=1.8,
                    label=label,
                )
            )
    fig.text(
        0.01,
        0.985,
        "Lag Curves with Best Lags",
        ha="left",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=max(1, len(handles)),
        frameon=False,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.1, top=0.86)
    draw_grouped_task_backgrounds(layout, config)
    save_figure(fig, output_dir / "lag_curves_best_lags", formats)


def remove_legacy_lag_curve_outputs(
    output_dir: Path,
    loaded: Mapping[str, Mapping[str, object]],
    formats: Sequence[str],
) -> None:
    output_bases = []
    for condition, tasks in loaded.items():
        output_bases.append(output_dir / f"lag_curves_{condition}")
        output_bases.extend(
            output_dir / f"lag_curves_{task}_{condition}" for task in tasks
        )
    for output_base in output_bases:
        for fmt in formats:
            output_base.with_suffix(f".{fmt}").unlink(missing_ok=True)


def curve_for_metric(df: pd.DataFrame, metric: MetricConfig) -> pd.DataFrame:
    if metric.column not in df.columns:
        raise KeyError(f"Metric column '{metric.column}' is missing")
    curve = df[["lags", metric.column]].copy()
    curve[metric.column] = metric_values(curve, metric)
    return curve.dropna().sort_values("lags")


def metric_reference_value(metric: MetricConfig) -> float:
    if metric.chance_level is not None:
        return float(metric.chance_level)
    if metric.min_value is not None:
        return float(metric.min_value)
    return 0.0


def _interpolate_threshold_crossing(
    lag_a: float,
    value_a: float,
    lag_b: float,
    value_b: float,
    threshold: float,
) -> float:
    if value_a == threshold:
        return float(lag_a)
    if value_b == threshold:
        return float(lag_b)
    if value_a == value_b:
        return float("nan")
    fraction = (threshold - value_a) / (value_b - value_a)
    if fraction < 0.0 or fraction > 1.0:
        return float("nan")
    return float(lag_a + fraction * (lag_b - lag_a))


def _crosses_threshold(value_a: float, value_b: float, threshold: float) -> bool:
    return (value_a - threshold) * (value_b - threshold) <= 0.0


def _closest_half_peak_crossing(
    lags: np.ndarray,
    values: np.ndarray,
    peak_index: int,
    half_peak: float,
    *,
    before_peak: bool,
) -> float:
    if before_peak:
        segment_indices = range(peak_index - 1, -1, -1)
    else:
        segment_indices = range(peak_index, len(values) - 1)

    for idx in segment_indices:
        lag_a = float(lags[idx])
        value_a = float(values[idx])
        lag_b = float(lags[idx + 1])
        value_b = float(values[idx + 1])
        if not (
            np.isfinite(lag_a)
            and np.isfinite(lag_b)
            and np.isfinite(value_a)
            and np.isfinite(value_b)
        ):
            continue
        if not _crosses_threshold(value_a, value_b, half_peak):
            continue
        return _interpolate_threshold_crossing(
            lag_a,
            value_a,
            lag_b,
            value_b,
            half_peak,
        )
    return float("nan")


def half_peak_profile_for_curve(
    curve: pd.DataFrame,
    metric: MetricConfig,
) -> HalfPeakProfile:
    if curve.empty:
        raise ValueError("Cannot compute half-peak profile for an empty curve")
    if "lags" not in curve.columns or metric.column not in curve.columns:
        raise KeyError("Curve must contain lags and the configured metric column")

    clean = curve[["lags", metric.column]].dropna().sort_values("lags")
    if clean.empty:
        raise ValueError("Cannot compute half-peak profile without numeric values")

    lags = clean["lags"].to_numpy(dtype=float)
    values = clean[metric.column].to_numpy(dtype=float)
    peak_index = int(
        np.nanargmax(values) if metric.higher_is_better else np.nanargmin(values)
    )
    peak_lag = float(lags[peak_index])
    peak_value = float(values[peak_index])
    reference_value = metric_reference_value(metric)
    half_peak_value = float(reference_value + 0.5 * (peak_value - reference_value))

    ramp_half_peak_lag = _closest_half_peak_crossing(
        lags,
        values,
        peak_index,
        half_peak_value,
        before_peak=True,
    )
    decay_half_peak_lag = _closest_half_peak_crossing(
        lags,
        values,
        peak_index,
        half_peak_value,
        before_peak=False,
    )

    ramp_duration = (
        float(peak_lag - ramp_half_peak_lag)
        if np.isfinite(ramp_half_peak_lag)
        else float("nan")
    )
    decay_duration = (
        float(decay_half_peak_lag - peak_lag)
        if np.isfinite(decay_half_peak_lag)
        else float("nan")
    )
    half_peak_width = (
        float(decay_half_peak_lag - ramp_half_peak_lag)
        if np.isfinite(ramp_half_peak_lag) and np.isfinite(decay_half_peak_lag)
        else float("nan")
    )
    ramp_slope = (
        float((peak_value - half_peak_value) / ramp_duration)
        if np.isfinite(ramp_duration) and ramp_duration != 0
        else float("nan")
    )
    decay_slope = (
        float((half_peak_value - peak_value) / decay_duration)
        if np.isfinite(decay_duration) and decay_duration != 0
        else float("nan")
    )

    return HalfPeakProfile(
        peak_value=peak_value,
        peak_lag=peak_lag,
        reference_value=reference_value,
        half_peak_value=half_peak_value,
        ramp_half_peak_lag=ramp_half_peak_lag,
        ramp_duration=ramp_duration,
        ramp_slope=ramp_slope,
        ramp_rate=abs(ramp_slope) if np.isfinite(ramp_slope) else float("nan"),
        decay_half_peak_lag=decay_half_peak_lag,
        decay_duration=decay_duration,
        decay_slope=decay_slope,
        decay_rate=abs(decay_slope) if np.isfinite(decay_slope) else float("nan"),
        half_peak_width=half_peak_width,
    )


def half_peak_profile_rows(
    loaded: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    config: Mapping,
) -> pd.DataFrame:
    model = half_peak_profile_model(config)
    rows = []
    tasks = sorted(
        {
            task
            for condition_results in loaded.values()
            for task, model_results in condition_results.items()
            if model in model_results
        }
    )
    for task in tasks:
        metric = get_metric_config(config, task)
        for condition in CONDITIONS:
            df = loaded.get(condition, {}).get(task, {}).get(model)
            if df is None or metric.column not in df.columns:
                continue
            curve = curve_for_metric(df, metric)
            profile = half_peak_profile_for_curve(curve, metric)
            rows.append(
                {
                    "task": task,
                    "condition": condition,
                    "model": model,
                    "metric": metric.column,
                    "metric_label": metric.label,
                    "reference_value": profile.reference_value,
                    "half_peak_value": profile.half_peak_value,
                    "peak_value": profile.peak_value,
                    "peak_lag": profile.peak_lag,
                    "ramp_half_peak_lag": profile.ramp_half_peak_lag,
                    "ramp_duration": profile.ramp_duration,
                    "decay_half_peak_lag": profile.decay_half_peak_lag,
                    "decay_duration": profile.decay_duration,
                    "half_peak_width": profile.half_peak_width,
                    "ramp_slope": profile.ramp_slope,
                    "ramp_rate": profile.ramp_rate,
                    "decay_slope": profile.decay_slope,
                    "decay_rate": profile.decay_rate,
                }
            )
    return pd.DataFrame(rows)


def task_group_lookup(config: Mapping, tasks: Sequence[str]) -> dict[str, str]:
    lookup = {}
    for group, group_tasks in grouped_tasks_for_summary(config, tasks):
        for task in group_tasks:
            lookup[task] = group
    return lookup


def ordered_tasks_by_group_average(
    profile: pd.DataFrame,
    config: Mapping,
    value_column: str,
) -> list[str]:
    if profile.empty or value_column not in profile.columns:
        return []

    tasks = sorted(str(task) for task in profile["task"].unique())
    group_by_task = task_group_lookup(config, tasks)
    sortable = profile.copy()
    sortable["group"] = sortable["task"].map(group_by_task).fillna("Other")
    group_means = (
        sortable.groupby("group", sort=False)[value_column]
        .mean()
        .replace([np.inf, -np.inf], np.nan)
    )
    group_order = sorted(
        group_means.index,
        key=lambda group: (
            (
                float(group_means.loc[group])
                if pd.notna(group_means.loc[group])
                else float("inf")
            ),
            str(group),
        ),
    )

    ordered_tasks = []
    for group in group_order:
        group_rows = sortable[sortable["group"] == group]
        task_means = (
            group_rows.groupby("task", sort=False)[value_column]
            .mean()
            .replace([np.inf, -np.inf], np.nan)
        )
        ordered_tasks.extend(
            sorted(
                task_means.index,
                key=lambda task: (
                    (
                        float(task_means.loc[task])
                        if pd.notna(task_means.loc[task])
                        else float("inf")
                    ),
                    str(task),
                ),
            )
        )
    return ordered_tasks


def plot_peak_profile_pyramid(
    profile: pd.DataFrame,
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    profile = profile[profile["condition"] == "super_subject"].copy()
    if profile.empty:
        return

    numeric_cols = ["peak_lag", "decay_duration"]
    for col in numeric_cols:
        profile[col] = pd.to_numeric(profile[col], errors="coerce")

    tasks = profile["task"].unique()
    group_by_task = task_group_lookup(config, tasks)
    background_colors = task_group_background_colors(config)
    fallback_color = "#E8E8E8"
    profile["group"] = profile["task"].map(lambda t: group_by_task.get(str(t), "Other"))

    metrics = [
        ("peak_lag", "Mean Peak Lag", "Time (ms)"),
        ("decay_duration", "Decay Half-Width", "Time (ms)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)
    for metric_idx, (value_column, title, ylabel) in enumerate(metrics):
        ax = axes[metric_idx]
        sort_order = (
            profile[profile[value_column].notna()]
            .groupby("group")[value_column]
            .mean()
            .sort_values()
            .index.tolist()
        )

        for i, group in enumerate(sort_order):
            group_data = profile[profile["group"] == group].dropna(
                subset=[value_column]
            )
            if group_data.empty:
                continue

            ax.bar(
                i,
                group_data[value_column].mean(),
                color=background_colors.get(group, fallback_color),
                alpha=0.7,
                edgecolor="#333333",
                linewidth=0.8,
                zorder=2,
            )

            rng = np.random.default_rng(metric_idx * 100 + i)
            ax.scatter(
                np.full(len(group_data), i) + rng.uniform(-0.15, 0.15, len(group_data)),
                group_data[value_column],
                color="white",
                edgecolor="#333333",
                linewidth=0.6,
                s=35,
                zorder=3,
            )

        ax.set_title(title, fontweight="bold", fontsize=PLOT_TITLE_FONTSIZE, pad=10)
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_xticks(range(len(sort_order)))
        ax.set_xticklabels(sort_order, rotation=25, ha="right", fontsize=9)
        ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)

    model = str(profile["model"].iloc[0])
    fig.suptitle(
        f"{display_model_name(config, model)}: Peak Timing Summary",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    save_figure(fig, output_dir / "peak_pyramid_summary", formats)


def plot_peak_profile_grid_2x2_column_sorted(
    profile: pd.DataFrame,
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    # 1. Filter for super_subject only
    profile = profile[profile["condition"] == "super_subject"].copy()
    if profile.empty:
        return

    numeric_cols = [
        "half_peak_width",
        "ramp_duration",
        "decay_duration",
        "ramp_slope",
        "decay_slope",
    ]
    for col in numeric_cols:
        profile[col] = pd.to_numeric(profile[col], errors="coerce")

    # [Row][Column]
    metric_grid = [
        [
            ("ramp_duration", "Ramp Duration (ms)"),
            ("decay_duration", "Decay Duration (ms)"),
        ],
        [("ramp_slope", "Ramp Slope"), ("decay_slope", "Decay Slope")],
    ]

    tasks = profile["task"].unique()
    group_by_task = task_group_lookup(config, tasks)
    background_colors = task_group_background_colors(config)
    fallback_color = "#E8E8E8"
    profile["group"] = profile["task"].map(lambda t: group_by_task.get(str(t), "Other"))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14, 9),  # Slightly narrower to bring plots closer
        sharex=False,  # We handle ticks manually to ensure sort consistency
    )

    sort_metric_col = ["ramp_duration", "decay_duration"]
    for col_idx in range(2):
        # Determine the metric to sort by for THIS column
        # If col 0 (Ramp), we check ramp data; if col 1 (Decay), we check decay data.
        # We filter for non-null values of the duration metric in this column first
        duration_col = metric_grid[0][col_idx][0]
        mask = profile[duration_col].notna()

        # Calculate sort order based on ramp/decay duration for groups present in this column
        column_sort_order = (
            profile[mask]
            .groupby("group")[sort_metric_col[col_idx]]
            .mean()
            .sort_values()
            .index.tolist()
        )

        for row_idx in range(2):
            ax = axes[row_idx, col_idx]
            value_column, ylabel = metric_grid[row_idx][col_idx]

            for i, group in enumerate(column_sort_order):
                group_data = profile[profile["group"] == group].dropna(
                    subset=[value_column]
                )
                if group_data.empty:
                    continue

                mean_val = group_data[value_column].mean()
                color = background_colors.get(group, fallback_color)

                ax.bar(
                    i,
                    mean_val,
                    color=color,
                    alpha=0.7,
                    edgecolor="#333333",
                    linewidth=0.8,
                    zorder=2,
                )

                rng = np.random.default_rng(i)
                x_jitter = rng.uniform(-0.15, 0.15, size=len(group_data))
                ax.scatter(
                    np.full(len(group_data), i) + x_jitter,
                    group_data[value_column],
                    color="white",
                    edgecolor="#333333",
                    linewidth=0.6,
                    s=35,
                    zorder=3,
                )

            # Formatting
            ax.set_ylabel(ylabel, fontweight="bold", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=1)
            ax.set_xticks(range(len(column_sort_order)))

            # X-labels only on bottom row
            if row_idx == 1:
                ax.set_xticklabels(
                    column_sort_order, rotation=25, ha="right", fontsize=9
                )
            else:
                ax.set_xticklabels([])
                side = "Ramp" if col_idx == 0 else "Decay"
                ax.set_title(
                    f"{side} Metrics",
                    fontweight="bold",
                    fontsize=PLOT_TITLE_FONTSIZE,
                    pad=10,
                )

    # Tighten the padding between subplots
    # hspace = vertical space, wspace = horizontal space
    plt.subplots_adjust(
        left=0.08, right=0.95, bottom=0.12, top=0.90, wspace=0.25, hspace=0.35
    )

    model = str(profile["model"].iloc[0])
    fig.suptitle(
        f"{display_model_name(config, model)}: Ramp/Decay Summary",
        fontsize=PLOT_TITLE_FONTSIZE,
        fontweight="bold",
    )

    save_figure(fig, output_dir / "ramp_decay_2x2_fixed", formats)


def plot_half_peak_profile_bars(
    profile: pd.DataFrame,
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    if profile.empty:
        return

    metrics = [
        ("peak_lag", "Peak lag (ms)"),
        ("ramp_duration", "Ramp duration (ms)"),
        ("decay_duration", "Decay duration (ms)"),
        ("half_peak_width", "Half-peak width (ms)"),
    ]
    available_metrics = [
        (column, label) for column, label in metrics if column in profile.columns
    ]
    if not available_metrics:
        return

    fig, axes = plt.subplots(
        len(available_metrics),
        1,
        figsize=(18, 4.2 * len(available_metrics)),
        squeeze=False,
    )
    axes = axes[:, 0]
    tasks = sorted(str(task) for task in profile["task"].unique())
    group_by_task = task_group_lookup(config, tasks)
    background_colors = task_group_background_colors(config)
    fallback_color = background_colors.get("Other", "#E8E8E8")
    conditions = [
        condition for condition in CONDITIONS if condition in set(profile["condition"])
    ]
    conditions.extend(
        sorted(
            set(str(condition) for condition in profile["condition"]) - set(conditions)
        )
    )
    condition_offsets = (
        np.linspace(-0.11, 0.11, len(conditions))
        if len(conditions) > 1
        else np.array([0.0])
    )
    bar_width = min(0.20, 0.46 / max(len(conditions), 1))
    hatches = {
        "super_subject": "",
        "per_subject": "///",
    }

    for ax, (value_column, ylabel) in zip(axes, available_metrics):
        ordered_tasks = ordered_tasks_by_group_average(profile, config, value_column)
        positions = {task: idx for idx, task in enumerate(ordered_tasks)}

        group_spans = []
        for group in dict.fromkeys(
            group_by_task.get(task, "Other") for task in ordered_tasks
        ):
            group_positions = [
                positions[task]
                for task in ordered_tasks
                if group_by_task.get(task, "Other") == group
            ]
            if group_positions:
                group_spans.append((group, min(group_positions), max(group_positions)))

        for group, start, end in group_spans:
            ax.axvspan(
                start - 0.5,
                end + 0.5,
                color=background_colors.get(group, fallback_color),
                alpha=0.22,
                linewidth=0,
                zorder=0,
            )
            ax.text(
                (start + end) / 2,
                1.015,
                group,
                transform=ax.get_xaxis_transform(),
                ha="center",
                va="bottom",
                fontsize=9,
            )
            group_tasks = [
                task
                for task in ordered_tasks
                if group_by_task.get(task, "Other") == group
            ]
            group_values = pd.to_numeric(
                profile.loc[
                    profile["task"].isin(group_tasks),
                    value_column,
                ],
                errors="coerce",
            ).dropna()
            if not group_values.empty:
                ax.hlines(
                    float(group_values.mean()),
                    start - 0.36,
                    end + 0.36,
                    color="#333333",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.85,
                    zorder=4,
                )

        for condition, offset in zip(conditions, condition_offsets):
            condition_profile = profile[profile["condition"] == condition]
            by_task = {
                str(row["task"]): row for row in condition_profile.to_dict("records")
            }
            x_values = []
            y_values = []
            colors = []
            for task in ordered_tasks:
                if task not in by_task:
                    continue
                value = by_task[task][value_column]
                if pd.isna(value):
                    continue
                x_values.append(positions[task] + float(offset))
                y_values.append(float(value))
                group = group_by_task.get(task, "Other")
                colors.append(background_colors.get(group, fallback_color))

            ax.bar(
                x_values,
                y_values,
                width=bar_width,
                color=colors,
                edgecolor="#333333",
                linewidth=0.7,
                hatch=hatches.get(condition, "\\\\"),
                label=display_condition_name(condition),
                zorder=3,
            )

        ax.set_ylabel(ylabel)
        ax.set_xticks(range(len(ordered_tasks)))
        ax.set_xticklabels(
            [display_task_name(config, task) for task in ordered_tasks],
            rotation=38,
            ha="right",
            fontsize=9,
        )
        ax.grid(axis="y", alpha=0.25, zorder=1)

    condition_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="white",
            edgecolor="#333333",
            hatch=hatches.get(condition, "\\\\"),
            label=display_condition_name(condition),
        )
        for condition in conditions
    ]
    group_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor=background_colors.get(group, fallback_color),
            edgecolor="none",
            label=group,
        )
        for group in dict.fromkeys(group_by_task.get(task, "Other") for task in tasks)
    ]
    model = str(profile["model"].iloc[0])
    fig.text(
        0.01,
        0.985,
        f"{display_model_name(config, model)} Half-Peak Profile Bars",
        ha="left",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    fig.legend(
        handles=[*condition_handles, *group_handles],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(condition_handles) + len(group_handles), 8),
        frameon=False,
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.08, top=0.92, hspace=1.08)
    save_figure(fig, output_dir / half_peak_profile_bar_output_name(config), formats)


def plot_half_peak_profile(
    profile: pd.DataFrame,
    loaded: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
    colors: Mapping[str, str],
) -> None:
    if profile.empty:
        return

    model = str(profile["model"].iloc[0])
    tasks = sorted(profile["task"].unique())
    layout = create_grouped_task_figure(config, tasks, figsize=(18, 8.8), hspace=0.6)
    fig = layout.fig
    condition_colors = {
        "super_subject": "#1F4E79",
        "per_subject": "#2CA7A0",
    }
    condition_markers = {
        "super_subject": "o",
        "per_subject": "s",
    }

    for task, ax in layout.task_axes.items():
        metric = get_metric_config(config, task)
        task_profile = profile[profile["task"] == task]
        for condition in config.get("plotting", {}).get(
            "half_peak_profile_conditions", CONDITIONS
        ):
            df = loaded.get(condition, {}).get(task, {}).get(model)
            if df is None or metric.column not in df.columns:
                continue
            curve = curve_for_metric(df, metric)
            row = task_profile[task_profile["condition"] == condition]
            if row.empty:
                continue
            item = row.iloc[0]
            color = condition_colors.get(
                condition, colors.get(model, DEFAULT_COLORS.get(model, "#333333"))
            )
            marker = condition_markers.get(condition, "D")
            ax.plot(
                curve["lags"],
                curve[metric.column],
                color=color,
                linewidth=1.7,
                label=display_condition_name(condition),
                zorder=2,
            )
            ax.scatter(
                [item["peak_lag"]],
                [item["peak_value"]],
                color=color,
                marker=marker,
                s=42,
                zorder=5,
            )
            ax.axhline(
                item["half_peak_value"],
                color=color,
                linestyle=":",
                linewidth=1.0,
                alpha=0.55,
                zorder=1,
            )
            if np.isfinite(item["ramp_half_peak_lag"]):
                ax.scatter(
                    [item["ramp_half_peak_lag"]],
                    [item["half_peak_value"]],
                    color="white",
                    edgecolor=color,
                    marker=marker,
                    s=34,
                    zorder=5,
                )
                ax.hlines(
                    item["half_peak_value"],
                    item["ramp_half_peak_lag"],
                    item["peak_lag"],
                    color=color,
                    linewidth=4.0,
                    alpha=0.28,
                    zorder=3,
                )
            if np.isfinite(item["decay_half_peak_lag"]):
                ax.scatter(
                    [item["decay_half_peak_lag"]],
                    [item["half_peak_value"]],
                    color="white",
                    edgecolor=color,
                    marker=marker,
                    s=34,
                    zorder=5,
                )
                ax.hlines(
                    item["half_peak_value"],
                    item["peak_lag"],
                    item["decay_half_peak_lag"],
                    color=color,
                    linewidth=4.0,
                    alpha=0.42,
                    zorder=3,
                )
            if np.isfinite(item["ramp_half_peak_lag"]) and np.isfinite(
                item["decay_half_peak_lag"]
            ):
                ax.axvspan(
                    item["ramp_half_peak_lag"],
                    item["decay_half_peak_lag"],
                    color=color,
                    alpha=0.07,
                    linewidth=0,
                    zorder=0,
                )

        ax.set_title(display_task_name(config, task))
        ax.set_xlabel("Lag relative to word onset (ms)")
        ax.set_ylabel(metric.label)
        apply_axis_text_sizes(
            ax,
            label_fontsize=LAG_PLOT_AXIS_LABEL_FONTSIZE,
            tick_fontsize=LAG_PLOT_TICK_LABEL_FONTSIZE,
        )
        ax.axvline(0, color="#333333", linewidth=0.8, alpha=0.5)
        plot_chance_level(ax, metric)
        ax.grid(alpha=0.25)
        apply_lag_plot_xlim(ax, config)
        apply_metric_ylim(ax, metric)

    handles = [
        Line2D(
            [0],
            [0],
            color=condition_colors.get(
                condition, colors.get(model, DEFAULT_COLORS.get(model, "#333333"))
            ),
            marker=condition_markers.get(condition, "D"),
            linewidth=1.8,
            markersize=6,
            label=display_condition_name(condition),
        )
        for condition in CONDITIONS
        if condition in set(profile["condition"])
    ]
    width_handle = plt.Rectangle(
        (0, 0),
        1,
        1,
        facecolor="#777777",
        edgecolor="none",
        alpha=0.16,
        label="Half-peak width",
    )
    fig.text(
        0.01,
        0.985,
        f"{display_model_name(config, model)} Half-Peak Temporal Profile",
        ha="left",
        va="top",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    fig.legend(
        handles=[*handles, width_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=min(len(handles) + 1, 4),
        frameon=False,
    )
    fig.subplots_adjust(left=0.055, right=0.985, bottom=0.1, top=0.86)
    draw_grouped_task_backgrounds(layout, config)
    save_figure(fig, output_dir / half_peak_profile_output_name(config), formats)


def lag_curve_error_values(df: pd.DataFrame, metric: MetricConfig) -> pd.Series | None:
    fold_columns = metric_fold_columns(df, metric)
    if fold_columns:
        rows = []
        for _, row in df.iterrows():
            values = pd.to_numeric(
                pd.Series([row[column] for column in fold_columns.values()]),
                errors="coerce",
            ).dropna()
            rows.append(standard_error(values.to_numpy(dtype=float)))
        errors = pd.Series(rows, index=df.index, dtype=float)
        return errors if errors.notna().any() else None

    std_column = metric_std_column(metric.column)
    if std_column is None or std_column not in df.columns:
        return None
    errors = pd.to_numeric(df[std_column], errors="coerce").abs()
    return errors if errors.notna().any() else None


def region_sort_key(region: str) -> tuple[int, str]:
    if region in REGION_LEVEL_ORDER:
        return (REGION_LEVEL_ORDER.index(region), region)
    return (len(REGION_LEVEL_ORDER), region)


def display_region_name(region: str) -> str:
    return "STG" if region == "EAC" else region


def region_gradient_colors(regions: Sequence[str]) -> Dict[str, object]:
    ordered = sorted(regions, key=region_sort_key)
    cmap = plt.get_cmap("viridis", max(len(ordered), 1))
    if len(ordered) == 1:
        return {ordered[0]: cmap(0.65)}
    return {
        region: cmap(idx / (len(ordered) - 1)) for idx, region in enumerate(ordered)
    }


def plot_per_region_lag_curves(
    per_region_results: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
) -> None:
    models = configured_model_order(
        {
            model
            for task_results in per_region_results.values()
            for model in task_results
        },
        config,
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
            {region for _, region_results in task_items for region in region_results},
            key=region_sort_key,
        )
        colors = region_gradient_colors(all_regions)
        layout = create_grouped_task_figure(
            config,
            [task for task, _ in task_items],
            figsize=(18, 8.8),
        )
        fig = layout.fig

        for task, region_results in task_items:
            ax = layout.task_axes[task]
            metric = get_metric_config(config, task)
            for region in sorted(region_results, key=region_sort_key):
                curve = curve_for_metric(region_results[region], metric)
                ax.plot(
                    curve["lags"],
                    curve[metric.column],
                    # marker="o",
                    linewidth=1.6,
                    # markersize=3.5,
                    label=display_region_name(region),
                    color=colors[region],
                )
            ax.set_title(display_task_name(config, task))
            ax.set_xlabel("Lag relative to word onset (ms)")
            ax.set_ylabel(metric.label)
            apply_axis_text_sizes(
                ax,
                label_fontsize=LAG_PLOT_AXIS_LABEL_FONTSIZE,
                tick_fontsize=LAG_PLOT_TICK_LABEL_FONTSIZE,
            )
            ax.axvline(0, color="#777777", linewidth=0.8, alpha=0.6)
            plot_chance_level(ax, metric)
            ax.grid(alpha=0.25)
            apply_lag_plot_xlim(ax, config)
            apply_metric_ylim(ax, metric)

        handles = [
            plt.Line2D(
                [0],
                [0],
                color=colors[region],
                # marker="o",
                linewidth=1.6,
                # markersize=4,
                label=display_region_name(region),
            )
            for region in all_regions
        ]
        fig.text(
            0.01,
            0.985,
            f"{display_model_name(config, model)} Per-Region Lag Curves",
            ha="left",
            va="top",
            fontsize=plt.rcParams["axes.titlesize"],
        )
        fig.legend(
            handles=handles,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.995),
            ncol=min(len(handles), 6),
            frameon=False,
        )
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.1, top=0.86)
        draw_grouped_task_backgrounds(layout, config)
        save_figure(fig, output_dir / f"per_region_lags_{model}", formats)


def assign_region_groups(
    electrodes: pd.DataFrame,
    region_groups: dict[str, list[str]],
    atlas_path: Path | None = None,
    nilearn_data_dir: Path | None = None,
) -> pd.DataFrame:
    from utils.atlas_utils import DESTRIEUX_2009_LABELS, _lookup_atlas_labels

    electrodes = electrodes.copy()
    if not region_groups:
        electrodes["atlas_label"] = "unassigned"
        electrodes["region_group"] = "unassigned"
        return electrodes

    atlas_image, affine = load_atlas(
        atlas_path=str(atlas_path) if atlas_path is not None else None,
        nilearn_data_dir=nilearn_data_dir,
    )
    coords = electrodes[["x", "y", "z"]].to_numpy(float)
    electrodes["atlas_label"] = _lookup_atlas_labels(
        coords, atlas_image, affine, DESTRIEUX_2009_LABELS
    )

    label_to_group = {
        label: region_name
        for region_name, labels in region_groups.items()
        for label in labels
    }
    electrodes["region_group"] = (
        electrodes["atlas_label"].map(label_to_group).fillna("unassigned")
    )
    return electrodes


def load_region_groups(path: Path | None) -> dict[str, list[str]]:
    from utils.atlas_utils import REGION_GROUPS
    import json

    if path is None:
        return REGION_GROUPS

    with path.open() as f:
        region_groups = json.load(f)

    if not isinstance(region_groups, dict):
        raise ValueError(
            "--region-groups-json must contain an object mapping names to labels"
        )

    return {str(name): list(labels) for name, labels in region_groups.items()}


def load_atlas(
    atlas_path: str | None = None,
    nilearn_data_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    from nilearn import datasets, image as nli_image

    if atlas_path is None:
        fetch_kwargs = {}
        if nilearn_data_dir is not None:
            nilearn_data_dir.mkdir(parents=True, exist_ok=True)
            fetch_kwargs["data_dir"] = str(nilearn_data_dir)
        atlas_path = datasets.fetch_atlas_destrieux_2009(**fetch_kwargs)["maps"]

    img = nli_image.load_img(atlas_path)
    return img.get_fdata().astype(int), img.affine


def _load_region_electrodes(
    data_root: Path,
    atlas_path: Path | None,
    nilearn_data_dir: Path | None,
    include_bad: bool,
) -> pd.DataFrame:
    electrodes = load_electrodes(data_root, include_bad=include_bad)
    return assign_region_groups(
        electrodes,
        load_region_groups(None),
        atlas_path=atlas_path,
        nilearn_data_dir=nilearn_data_dir,
    )


def read_good_channel_names(channels_path: Path) -> set[str] | None:
    if not channels_path.exists():
        return None

    channels = pd.read_csv(channels_path, sep="\t")
    if "status" not in channels or "name" not in channels:
        return None

    return set(
        channels.loc[channels["status"].fillna("").str.lower() == "good", "name"]
    )


def load_electrodes(data_root: Path, include_bad: bool) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for electrodes_path in sorted(
        data_root.glob("sub-*/ieeg/*_space-MNI152NLin2009aSym_electrodes.tsv")
    ):
        subject = electrodes_path.parts[-3]
        elecs = pd.read_csv(electrodes_path, sep="\t")
        required = {"name", "x", "y", "z"}
        missing = required - set(elecs.columns)
        if missing:
            raise ValueError(f"{electrodes_path} is missing columns: {sorted(missing)}")

        elecs = elecs.copy()
        elecs["subject"] = subject
        for axis in ("x", "y", "z"):
            elecs[axis] = pd.to_numeric(elecs[axis], errors="coerce")
        elecs = elecs.dropna(subset=["x", "y", "z"])

        if not include_bad:
            channels_path = electrodes_path.with_name(
                f"{subject}_task-podcast_channels.tsv"
            )
            good_names = read_good_channel_names(channels_path)
            if good_names is not None:
                elecs = elecs[elecs["name"].isin(good_names)]

        rows.append(elecs[["subject", "name", "x", "y", "z"]])

    if not rows:
        raise FileNotFoundError(f"No MNI electrode TSV files found under {data_root}")

    return pd.concat(rows, ignore_index=True)


def plot_atlas_region_electrodes(
    output_dir: Path,
    formats: Sequence[str],
    data_root: Path,
    atlas_path: Path,
    nilearn_data_dir: Path | None,
    include_bad: bool,
) -> None:

    electrodes = _load_region_electrodes(
        data_root, atlas_path, nilearn_data_dir, include_bad
    )
    fig = create_electrode_region_figure(electrodes)
    save_figure(fig, output_dir / "atlas_region_electrodes", formats)


def _bids_subject_label(subject_id: int) -> str:
    return f"sub-{int(subject_id):02d}"


def _bids_subject_id(subject_label: str) -> int:
    if not subject_label.startswith("sub-"):
        raise ValueError(
            f"Expected BIDS subject label like 'sub-01', got {subject_label!r}"
        )
    return int(subject_label.removeprefix("sub-"))


def _load_selected_electrode_coordinates(
    data_root: Path,
    include_bad: bool,
    selected_electrode_path: Path = DEFAULT_SELECTED_ELECTRODE_PATH,
) -> pd.DataFrame:
    from utils.data_utils import read_electrode_file, read_subject_mapping

    participant_mapping_path = data_root / "participants.tsv"
    subject_mapping = read_subject_mapping(
        str(participant_mapping_path),
        delimiter="\t",
    )
    selected_by_subject = read_electrode_file(
        str(selected_electrode_path),
        subject_mapping=subject_mapping,
    )

    selected_rows = [
        {
            "subject_id": int(subject_id),
            "subject": _bids_subject_label(subject_id),
            "name": electrode_name,
            "selection_order": selection_order,
        }
        for subject_id, electrode_names in selected_by_subject.items()
        for selection_order, electrode_name in enumerate(electrode_names)
    ]
    if not selected_rows:
        return pd.DataFrame(columns=["subject_id", "subject", "name", "x", "y", "z"])

    selected = pd.DataFrame(selected_rows)
    localized = load_electrodes(data_root, include_bad=include_bad).copy()
    localized["subject_id"] = localized["subject"].map(_bids_subject_id)

    duplicate_mask = localized.duplicated(["subject", "name"], keep=False)
    if duplicate_mask.any():
        duplicates = localized.loc[
            duplicate_mask, ["subject", "name"]
        ].drop_duplicates()
        duplicate_text = ", ".join(
            f"{row.subject}/{row.name}" for row in duplicates.itertuples(index=False)
        )
        raise ValueError(
            f"Localized electrode coordinates contain duplicate rows: {duplicate_text}"
        )

    merged = selected.merge(
        localized[["subject", "name", "x", "y", "z"]],
        on=["subject", "name"],
        how="left",
        validate="many_to_one",
    )
    missing = merged[["x", "y", "z"]].isna().any(axis=1)
    if missing.any():
        missing_rows = merged.loc[missing, ["subject", "name"]]
        missing_text = ", ".join(
            f"{row.subject}/{row.name}" for row in missing_rows.itertuples(index=False)
        )
        raise ValueError(
            "Selected electrodes are missing MNI coordinates in the localization sidecars: "
            f"{missing_text}"
        )

    return merged.sort_values(["subject_id", "selection_order"]).reset_index(drop=True)


def _load_all_electrode_coordinates(data_root: Path) -> pd.DataFrame:

    electrodes = load_electrodes(data_root, include_bad=True).copy()
    electrodes["subject_id"] = electrodes["subject"].map(_bids_subject_id)
    return electrodes.sort_values(["subject_id", "name"]).reset_index(drop=True)


def _participant_subject_ids(data_root: Path) -> list[int]:
    from utils.data_utils import read_subject_mapping

    subject_mapping = read_subject_mapping(
        str(data_root / "participants.tsv"),
        delimiter="\t",
    )
    return sorted(int(subject_id) for subject_id in subject_mapping.values())


def _subject_electrode_colors(subject_ids: Iterable[int]) -> dict[int, str]:
    ordered_subjects = sorted({int(subject_id) for subject_id in subject_ids})
    cmap = plt.get_cmap("tab10", max(len(ordered_subjects), 1))
    return {
        subject_id: to_hex(cmap(idx % cmap.N))
        for idx, subject_id in enumerate(ordered_subjects)
    }


def _add_electrode_markers_by_subject(
    display,
    electrodes: pd.DataFrame,
    subject_colors: Mapping[int, str],
    marker_size: float,
) -> None:
    marker_coords = electrodes[["x", "y", "z"]].to_numpy(float)
    for subject_id in sorted(electrodes["subject_id"].unique()):
        mask = electrodes["subject_id"] == subject_id
        display.add_markers(
            marker_coords[mask],
            marker_color=subject_colors[int(subject_id)],
            marker_size=marker_size,
            alpha=0.9,
        )


def _plot_electrode_glass_brain(
    electrodes: pd.DataFrame,
    subject_colors: Mapping[int, str],
    title: str,
    output_base: Path,
    formats: Sequence[str],
    marker_size: float,
    legend_subject_ids: Sequence[int] | None = None,
) -> None:
    from nilearn import plotting

    fig = plt.figure(figsize=(10, 4.8), constrained_layout=False)
    axes = [fig.add_subplot(1, 2, idx + 1) for idx in range(2)]
    panels = (("Left", "l"), ("Right", "r"))
    for ax, (panel_title, display_mode) in zip(axes, panels):
        display = plotting.plot_glass_brain(
            None,
            display_mode=display_mode,
            colorbar=False,
            figure=fig,
            axes=ax,
            title=panel_title,
            black_bg=False,
            annotate=True,
        )
        _add_electrode_markers_by_subject(
            display,
            electrodes,
            subject_colors,
            marker_size=marker_size,
        )

    if legend_subject_ids is None:
        legend_subject_ids = sorted(
            int(subject_id) for subject_id in electrodes["subject_id"].unique()
        )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=subject_colors[int(subject_id)],
            markeredgecolor="none",
            markersize=7,
            label=(
                f"Subject {int(subject_id):02d} "
                f"(n={int((electrodes['subject_id'] == subject_id).sum())})"
            ),
        )
        for subject_id in legend_subject_ids
    ]
    fig.suptitle(
        title,
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=plt.rcParams["axes.titlesize"],
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(len(handles), 5),
        frameon=False,
    )
    fig.subplots_adjust(left=0.03, right=0.985, bottom=0.18, top=0.86, wspace=0.08)
    save_figure(fig, output_base, formats)


def plot_selected_electrode_glass_brains(
    output_dir: Path,
    formats: Sequence[str],
    data_root: Path,
    include_bad: bool,
    selected_electrode_path: Path = DEFAULT_SELECTED_ELECTRODE_PATH,
) -> None:
    selected_electrodes = _load_selected_electrode_coordinates(
        data_root=data_root,
        include_bad=include_bad,
        selected_electrode_path=selected_electrode_path,
    )
    all_electrodes = _load_all_electrode_coordinates(data_root)

    participant_subject_ids = _participant_subject_ids(data_root)
    subject_colors = _subject_electrode_colors(participant_subject_ids)
    electrode_output_dir = output_dir / "electrode_glass_brains"
    for subject_id in participant_subject_ids:
        subject_electrodes = all_electrodes[all_electrodes["subject_id"] == subject_id]
        _plot_electrode_glass_brain(
            subject_electrodes,
            subject_colors,
            title=f"Subject {int(subject_id):02d} All Electrodes",
            output_base=electrode_output_dir
            / f"subject_{int(subject_id):02d}_electrodes",
            formats=formats,
            marker_size=42,
            legend_subject_ids=[subject_id],
        )

    if not selected_electrodes.empty:
        _plot_electrode_glass_brain(
            selected_electrodes,
            subject_colors,
            title="Multi-Subject Selected Electrodes",
            output_base=electrode_output_dir / "super_subject_electrodes",
            formats=formats,
            marker_size=26,
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
) -> None:
    coords = _mesh_coordinates(mesh_part)
    for region, mask in sorted(
        region_masks.items(), key=lambda item: region_sort_key(item[0])
    ):
        if not mask.any():
            continue
        center = coords[mask].mean(axis=0)
        text = ax.text(
            center[0],
            center[1],
            center[2],
            display_region_name(region),
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


def per_region_brain_plot_config(config: Mapping) -> Mapping:
    plot_config = plotting_config(config)
    brain_config = plot_config.get("per_region_brains", {})
    return brain_config if isinstance(brain_config, Mapping) else {}


def ignore_right_hemisphere_for_brain_maps(config: Mapping) -> bool:
    brain_config = per_region_brain_plot_config(config)
    if "ignore_right_hemisphere" in brain_config:
        return bool(brain_config["ignore_right_hemisphere"])
    if "left_hemisphere_only" in brain_config:
        return bool(brain_config["left_hemisphere_only"])
    return False


def brain_map_figsize(config: Mapping) -> tuple[float, float]:
    brain_config = per_region_brain_plot_config(config)
    configured = brain_config.get("figsize")
    if configured is None:
        return (30, 13)
    if not isinstance(configured, Sequence) or isinstance(configured, str):
        raise ValueError(
            "plotting.per_region_brains.figsize must be a two-item sequence"
        )
    if len(configured) != 2:
        raise ValueError(
            "plotting.per_region_brains.figsize must contain width and height"
        )
    return (float(configured[0]), float(configured[1]))


def add_brain_map_task_axes(
    fig: plt.Figure,
    container_ax: plt.Axes,
    panel_count: int = 2,
) -> tuple[list[plt.Axes], list[float]]:
    container_ax.set_axis_off()
    bbox = container_ax.get_position()
    if panel_count < 1:
        raise ValueError("Brain map panel_count must be at least 1")

    cbar_width = bbox.width * 0.065
    cbar_gap = bbox.width * 0.015
    right_pad = 0.0
    panel_area_width = bbox.width - cbar_width - cbar_gap - right_pad
    gap = bbox.width * 0.01 if panel_count > 1 else 0.0
    panel_width = (panel_area_width - gap * (panel_count - 1)) / panel_count
    axes = [
        fig.add_axes(
            [
                bbox.x0 + idx * (panel_width + gap),
                bbox.y0 - bbox.height * 0.02,
                panel_width,
                bbox.height * 1.08,
            ],
            projection="3d",
        )
        for idx in range(panel_count)
    ]
    colorbar_bounds = [
        bbox.x0 + panel_area_width + cbar_gap,
        bbox.y0 + bbox.height * 0.08,
        cbar_width,
        bbox.height * 0.84,
    ]
    return axes, colorbar_bounds


def set_brain_map_panel_title(ax: plt.Axes, title: str) -> None:
    ax.set_title(
        title,
        y=0.91,
        pad=0,
        fontsize=plt.rcParams["axes.titlesize"],
    )


def set_brain_map_panel_background(
    ax: plt.Axes,
    facecolor: tuple[float, float, float, float],
) -> None:
    ax.set_facecolor(facecolor)
    ax.patch.set_facecolor(facecolor)
    for axis_name in ("xaxis", "yaxis", "zaxis"):
        axis = getattr(ax, axis_name, None)
        pane = getattr(axis, "pane", None)
        if pane is not None:
            pane.set_facecolor(facecolor)
            pane.set_edgecolor(facecolor)


def rasterize_brain_map_surface_artists(ax: plt.Axes) -> None:
    for artist in ax.collections:
        artist.set_rasterized(True)


def region_count_legend_handles(
    regions: Iterable[str],
    region_counts: Mapping[str, int],
) -> list[Line2D]:
    return [
        Line2D(
            [],
            [],
            linestyle="none",
            label=f"{display_region_name(region)} (n={region_counts.get(region, 0)})",
        )
        for region in sorted(set(regions), key=region_sort_key)
    ]


def remove_legacy_per_region_brain_outputs(
    output_dir: Path,
    per_region_results: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    formats: Sequence[str],
) -> None:
    for task, model_results in per_region_results.items():
        for model in model_results:
            output_base = output_dir / f"per_region_brain_{model}_{task}"
            for fmt in formats:
                output_base.with_suffix(f".{fmt}").unlink(missing_ok=True)


def plot_per_region_brains(
    per_region_results: Mapping[str, Mapping[str, Mapping[str, pd.DataFrame]]],
    config: Mapping,
    output_dir: Path,
    formats: Sequence[str],
    data_root: Path,
    atlas_path: Path | None = None,
    nilearn_data_dir: Path | None = None,
    include_bad: bool = False,
) -> None:
    if not per_region_results:
        return

    from nilearn import plotting
    from utils.atlas_utils import REGION_GROUPS

    electrodes = _load_region_electrodes(
        data_root, atlas_path, nilearn_data_dir, include_bad
    )
    electrodes = electrodes[electrodes["region_group"] != "unassigned"].copy()
    region_counts = electrodes["region_group"].value_counts().to_dict()
    surface_atlas = _load_destrieux_surface_atlas(nilearn_data_dir)

    models = configured_model_order(
        {
            model
            for task_results in per_region_results.values()
            for model in task_results
        },
        config,
    )
    for model in models:
        task_items = [
            (task, model_results[model])
            for task, model_results in sorted(per_region_results.items())
            if model in model_results
        ]
        if not task_items:
            continue

        left_only = ignore_right_hemisphere_for_brain_maps(config)
        layout = create_grouped_task_figure(
            config,
            [task for task, _region_results in task_items],
            figsize=brain_map_figsize(config),
            hspace=0.32,
            wspace=0.18,
        )
        fig = layout.fig
        fig.subplots_adjust(left=0.02, right=0.99, bottom=0.055, top=0.9)
        colorbar_specs = []
        legend_regions = set()

        for task, region_results in task_items:
            metric = get_metric_config(config, task)
            brain_metric = brain_map_metric_config(config, task, metric)
            cmap = brain_map_colormap(config, task, metric)
            best_rows = best_region_lag_rows(region_results, metric)
            if left_only:
                best_rows = best_rows[best_rows["region"] != "RIGHT"]
            if best_rows.empty:
                continue
            metric_by_region = dict(zip(best_rows["region"], best_rows["value"]))
            legend_regions.update(metric_by_region)
            norm = metric_norm(list(metric_by_region.values()), brain_metric)
            metric_maps, region_masks = _build_surface_metric_maps(
                surface_atlas.labels,
                surface_atlas.maps,
                REGION_GROUPS,
                metric_by_region,
            )

            task_title = display_task_name(config, task)
            panels = [(task_title if left_only else "Left", "left")]
            if not left_only:
                panels.append(("Right", "right"))
            axes, colorbar_bounds = add_brain_map_task_axes(
                fig,
                layout.task_axes[task],
                panel_count=len(panels),
            )
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
                    title=None,
                )
                set_brain_map_panel_background(
                    ax,
                    fig.get_facecolor(),
                )
                set_brain_map_panel_title(ax, title)
                _draw_surface_region_boundaries(
                    ax,
                    mesh_part,
                    region_masks[hemi],
                )
                rasterize_brain_map_surface_artists(ax)
                _draw_surface_region_labels(
                    ax,
                    mesh_part,
                    region_masks[hemi],
                )

            sm = ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            colorbar_specs.append((task, brain_metric, sm, colorbar_bounds))

        fig.text(
            0.01,
            0.985,
            f"{display_model_name(config, model)} Per-Region Brain Maps",
            ha="left",
            va="top",
            fontsize=plt.rcParams["axes.titlesize"],
        )
        count_handles = region_count_legend_handles(legend_regions, region_counts)
        if count_handles:
            fig.legend(
                handles=count_handles,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.995),
                ncol=min(len(count_handles), 8),
                frameon=False,
                handlelength=0,
                handletextpad=0,
                columnspacing=1.2,
                title="Electrodes per group",
            )
        for _task, metric, sm, colorbar_bounds in colorbar_specs:
            cbar_ax = fig.add_axes(colorbar_bounds)
            cbar = fig.colorbar(sm, cax=cbar_ax)
            cbar.set_label(
                metric.label,
                fontsize=BRAIN_MAP_COLORBAR_LABEL_FONTSIZE,
                labelpad=8,
                weight="bold",
            )
            cbar.ax.yaxis.set_label_position("left")
            cbar.ax.tick_params(
                labelsize=BRAIN_MAP_COLORBAR_TICK_FONTSIZE,
                width=1.1,
                length=4,
            )
            cbar.outline.set_linewidth(1.0)

        draw_grouped_task_backgrounds(
            layout,
            config,
            label_axis_off_groups=True,
            draw_backgrounds=False,
        )
        save_figure(fig, output_dir / f"per_region_brains_{model}", formats)

    remove_legacy_per_region_brain_outputs(output_dir, per_region_results, formats)


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
    configured_destrieux_atlas_path = destrieux_atlas_path(config, config_path)
    tasks_to_exclude = excluded_tasks(config)
    loaded = filter_loaded_tasks(load_results(config), tasks_to_exclude)
    configured_valid_best_lags = valid_best_lags(config)
    per_region_results = filter_per_region_tasks(
        load_per_region_results(config),
        tasks_to_exclude,
    )
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
        for task in {
            *[
                task
                for condition_results in loaded.values()
                for task in condition_results
            ],
            *per_region_results.keys(),
        }
    }
    for condition, condition_results in loaded.items():
        summary = best_lag_rows(
            condition_results,
            metrics,
            valid_lags=configured_valid_best_lags,
        )
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
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        write_summary_tables(combined_summary, output_dir, table_formats, config)
        if include_overall_scores(config):
            score_dir = output_dir / "summary_tables"
            score_dir.mkdir(parents=True, exist_ok=True)
            summary_with_scores(combined_summary).to_csv(
                score_dir / "best_lag_scores.csv",
                index=False,
            )
            grouped_score_rows(combined_summary, config).to_csv(
                score_dir / "group_scores.csv",
                index=False,
            )
            overall_score_rows(combined_summary, config).to_csv(
                score_dir / "overall_scores.csv",
                index=False,
            )
        plot_combined_best_lag_summary(
            combined_summary,
            loaded,
            output_dir,
            formats,
            colors,
            config,
        )
        plot_baseline_condition_comparison(
            baseline_condition_comparison_rows(loaded, metrics),
            loaded,
            output_dir,
            formats,
            config,
        )
    write_baseline_region_peak_tables(
        baseline_region_peak_lag_rows(per_region_results, metrics),
        output_dir,
        table_formats,
    )
    if half_peak_profile_enabled(config):
        half_peak_profile = half_peak_profile_rows(loaded, config)
        write_half_peak_profile_tables(
            half_peak_profile,
            output_dir,
            table_formats,
            half_peak_profile_output_name(config),
        )
        plot_half_peak_profile(
            half_peak_profile,
            loaded,
            config,
            output_dir,
            formats,
            colors,
        )
        plot_half_peak_profile_bars(
            half_peak_profile,
            config,
            output_dir,
            formats,
        )
        plot_peak_profile_grid_2x2_column_sorted(
            half_peak_profile,
            config,
            output_dir,
            formats,
        )

        plot_peak_profile_pyramid(
            half_peak_profile,
            config,
            output_dir,
            formats,
        )
    write_neural_conv_model_summary_table(config, output_dir, table_formats)
    plot_lag_curves(loaded, config, output_dir, formats, colors)
    plot_lag_curves_with_best_lags(loaded, config, output_dir, formats, colors)
    plot_per_region_lag_curves(per_region_results, config, output_dir, formats)
    plot_selected_electrode_glass_brains(
        output_dir,
        formats,
        data_root,
        include_bad,
    )
    if configured_destrieux_atlas_path is None:
        print(
            "Skipping Destrieux atlas figures; configure "
            "plotting.destrieux_atlas_path in the paper-results config to generate them."
        )
    else:
        plot_atlas_region_electrodes(
            output_dir,
            formats,
            data_root,
            configured_destrieux_atlas_path,
            nilearn_data_dir,
            include_bad,
        )
        plot_per_region_brains(
            per_region_results,
            config,
            output_dir,
            formats,
            data_root,
            configured_destrieux_atlas_path,
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
