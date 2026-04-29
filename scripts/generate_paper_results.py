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
    summary: pd.DataFrame, bold: bool = False, latex: bool = False
) -> pd.DataFrame:
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
            if model not in by_model:
                row[model] = ""
                continue
            item = by_model[model]
            text = format_value(item["value"], item["lag"])
            if bold and winners.get(keys) == model:
                text = f"\\textbf{{{text}}}" if latex else f"**{text}**"
            row[model] = text
        rows.append(row)
    return pd.DataFrame(rows, columns=[*group_columns, *models])


def write_summary_tables(
    summary: pd.DataFrame, output_dir: Path, formats: Sequence[str]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_table = summary_wide(summary, bold=False)
    display_table = summary_wide(summary, bold=True)
    latex_table = summary_wide(summary, bold=True, latex=True)

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


def plot_best_lag_summary(
    summary: pd.DataFrame,
    condition: str,
    output_dir: Path,
    formats: Sequence[str],
    colors: Mapping[str, str],
) -> None:
    models = sorted(summary["model"].unique())
    tasks = sorted(summary["task"].unique())
    fig, axes = plt.subplots(
        1,
        len(tasks),
        figsize=(max(6, 2.4 * len(tasks)), 4),
        squeeze=False,
        sharey=False,
    )
    axes = axes[0]
    x_positions = list(range(len(models)))

    for ax, task in zip(axes, tasks):
        task_summary = summary[summary["task"] == task]
        metric = MetricConfig(
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
        values = []
        for model in models:
            match = task_summary[task_summary["model"] == model]
            values.append(
                float(match["value"].iloc[0]) if not match.empty else float("nan")
            )

        ax.bar(
            x_positions, values, color=[colors[model] for model in models], width=0.7
        )
        label = (
            str(task_summary["metric_label"].iloc[0])
            if task_summary["metric_label"].nunique() == 1
            else "Metric"
        )
        ax.set_title(task)
        ax.set_ylabel(label)
        ax.set_xticks(x_positions)
        ax.set_xticklabels(models, rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        apply_metric_ylim(ax, metric)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[model], label=model)
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
    fig.tight_layout(rect=(0, 0, 1, 0.92))
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
                    label=model,
                    color=colors[model],
                )

            ax.set_title(f"{task} - {condition.replace('_', ' ')}")
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
            ax.set_title(task)
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
        fig.suptitle(f"{model} per-region lag curves", y=1.06)
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
            fig.suptitle(f"{model} {task} per-region best lag", y=0.99)
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
        plot_best_lag_summary(summary, condition, output_dir, formats, colors)

    if all_summaries:
        write_summary_tables(
            pd.concat(all_summaries, ignore_index=True), output_dir, table_formats
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
