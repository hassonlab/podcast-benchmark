#!/usr/bin/env python3
"""Generate paper-ready result figures and summary tables."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import yaml


CONDITIONS = ("super_subject", "per_subject")
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


@dataclass(frozen=True)
class ResultSpec:
    model: str
    task: str
    condition: str
    path: Path


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
                    yield ResultSpec(model=model, task=task, condition=condition, path=Path(path))


def load_results(config: Mapping) -> Dict[str, Dict[str, Dict[str, pd.DataFrame]]]:
    loaded: Dict[str, Dict[str, Dict[str, pd.DataFrame]]] = {}
    for spec in iter_result_specs(config):
        loaded.setdefault(spec.condition, {}).setdefault(spec.task, {})[spec.model] = (
            load_current_style_run(spec.path)
        )
    return loaded


def get_metric_config(config: Mapping, task: str) -> MetricConfig:
    metrics = config.get("metrics", {})
    if not isinstance(metrics, Mapping) or task not in metrics:
        raise KeyError(f"No metric configuration found for task '{task}'")
    metric = metrics[task]
    return MetricConfig(
        column=metric["column"],
        higher_is_better=bool(metric.get("higher_is_better", True)),
        label=metric.get("label", metric["column"]),
    )


def select_best_lag(df: pd.DataFrame, metric: MetricConfig) -> pd.Series:
    if metric.column not in df.columns:
        raise KeyError(f"Metric column '{metric.column}' is missing")
    values = pd.to_numeric(df[metric.column], errors="coerce")
    if values.notna().sum() == 0:
        raise ValueError(f"Metric column '{metric.column}' has no numeric values")
    idx = values.idxmax() if metric.higher_is_better else values.idxmin()
    return df.loc[idx]


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
                    "value": float(best[metric.column]),
                    "lag": best["lags"],
                    "higher_is_better": metric.higher_is_better,
                }
            )
    return pd.DataFrame(rows)


def best_model_by_task(summary: pd.DataFrame) -> Dict[tuple, str]:
    winners = {}
    group_columns = ["condition", "task"] if "condition" in summary.columns else ["task"]
    for keys, group in summary.groupby(group_columns):
        if not isinstance(keys, tuple):
            keys = (keys,)
        higher = bool(group["higher_is_better"].iloc[0])
        idx = group["value"].idxmax() if higher else group["value"].idxmin()
        winners[keys] = str(summary.loc[idx, "model"])
    return winners


def format_value(value: float, lag) -> str:
    return f"{value:.3f} ({lag:g} ms)"


def summary_wide(summary: pd.DataFrame, bold: bool = False, latex: bool = False) -> pd.DataFrame:
    winners = best_model_by_task(summary) if bold else {}
    models = sorted(summary["model"].unique())
    group_columns = ["condition", "task"] if "condition" in summary.columns else ["task"]
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


def write_summary_tables(summary: pd.DataFrame, output_dir: Path, formats: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_table = summary_wide(summary, bold=False)
    display_table = summary_wide(summary, bold=True)
    latex_table = summary_wide(summary, bold=True, latex=True)

    if "csv" in formats:
        csv_table.to_csv(output_dir / "best_lag_summary.csv", index=False)
    if "markdown" in formats or "md" in formats:
        (output_dir / "best_lag_summary.md").write_text(to_markdown_table(display_table))
    if "latex" in formats or "tex" in formats:
        (output_dir / "best_lag_summary.tex").write_text(latex_table.to_latex(index=False, escape=False))


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
        values = []
        for model in models:
            match = task_summary[task_summary["model"] == model]
            values.append(float(match["value"].iloc[0]) if not match.empty else float("nan"))

        ax.bar(x_positions, values, color=[colors[model] for model in models], width=0.7)
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
                curve = df[["lags", metric.column]].copy()
                curve[metric.column] = pd.to_numeric(curve[metric.column], errors="coerce")
                curve = curve.dropna().sort_values("lags")
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
            ax.legend(frameon=False)
            save_figure(fig, output_dir / f"lag_curves_{task}_{condition}", formats)


def generate_paper_results(
    config_path: Path,
    output_dir: Path,
    formats: Sequence[str],
    table_formats: Sequence[str],
) -> None:
    config = read_config(config_path)
    loaded = load_results(config)
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
        write_summary_tables(pd.concat(all_summaries, ignore_index=True), output_dir, table_formats)
    plot_lag_curves(loaded, config, output_dir, formats, colors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_results.yml", type=Path)
    parser.add_argument("--output-dir", default="paper-results", type=Path)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf", "svg"])
    parser.add_argument("--table-formats", nargs="+", default=["csv", "markdown", "latex"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_paper_results(args.config, args.output_dir, args.formats, args.table_formats)


if __name__ == "__main__":
    main()
