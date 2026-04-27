from pathlib import Path

import pandas as pd

from scripts.generate_paper_results import (
    MetricConfig,
    best_lag_rows,
    load_current_style_run,
    plot_best_lag_summary,
    plot_lag_curves,
    select_best_lag,
    summary_wide,
)
from scripts.migrate_fm_results import migrate


def write_lag_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_loads_current_style_super_subject(tmp_path):
    run_dir = tmp_path / "run"
    write_lag_csv(run_dir / "lag_performance.csv", [{"lags": 0, "score": 0.5}])

    loaded = load_current_style_run(run_dir)

    assert loaded.to_dict("records") == [{"lags": 0, "score": 0.5}]


def test_loads_and_averages_current_style_per_subject(tmp_path):
    run_dir = tmp_path / "run"
    write_lag_csv(run_dir / "subject_1" / "lag_performance.csv", [{"lags": 0, "score": 0.5}])
    write_lag_csv(run_dir / "subject_2" / "lag_performance.csv", [{"lags": 0, "score": 0.7}])

    loaded = load_current_style_run(run_dir)

    assert loaded["lags"].tolist() == [0]
    assert loaded["score"].tolist() == [0.6]


def test_select_best_lag_maximize_and_minimize():
    df = pd.DataFrame({"lags": [-1, 0, 1], "score": [0.4, 0.8, 0.2], "loss": [3.0, 2.0, 1.0]})

    max_row = select_best_lag(df, MetricConfig("score", True, "Score"))
    min_row = select_best_lag(df, MetricConfig("loss", False, "Loss"))

    assert max_row["lags"] == 0
    assert min_row["lags"] == 1


def test_plot_lag_curves_preserves_unequal_lag_sets(tmp_path):
    loaded = {
        "super_subject": {
            "task": {
                "baseline": pd.DataFrame({"lags": [-1, 0], "score": [0.5, 0.6]}),
                "diver": pd.DataFrame({"lags": [-2, 1, 2], "score": [0.4, 0.7, 0.65]}),
            }
        }
    }
    config = {"metrics": {"task": {"column": "score", "higher_is_better": True, "label": "Score"}}}

    plot_lag_curves(
        loaded,
        config,
        tmp_path,
        formats=["png"],
        colors={"baseline": "#000000", "diver": "#ff0000"},
    )

    assert (tmp_path / "lag_curves_task_super_subject.png").exists()


def test_plot_best_lag_summary_uses_independent_task_y_axes(tmp_path, monkeypatch):
    captured = {}

    def capture_figure(fig, output_base, formats):
        captured["fig"] = fig
        captured["output_base"] = output_base
        captured["formats"] = formats

    monkeypatch.setattr("scripts.generate_paper_results.save_figure", capture_figure)
    summary = pd.DataFrame(
        [
            {
                "condition": "super_subject",
                "task": "small_scale",
                "model": "baseline",
                "metric": "score",
                "metric_label": "Score",
                "value": 0.1,
                "lag": 0,
                "higher_is_better": True,
            },
            {
                "condition": "super_subject",
                "task": "small_scale",
                "model": "diver",
                "metric": "score",
                "metric_label": "Score",
                "value": 0.2,
                "lag": 0,
                "higher_is_better": True,
            },
            {
                "condition": "super_subject",
                "task": "large_scale",
                "model": "baseline",
                "metric": "loss",
                "metric_label": "Loss",
                "value": 100.0,
                "lag": 0,
                "higher_is_better": False,
            },
            {
                "condition": "super_subject",
                "task": "large_scale",
                "model": "diver",
                "metric": "loss",
                "metric_label": "Loss",
                "value": 200.0,
                "lag": 0,
                "higher_is_better": False,
            },
        ]
    )

    plot_best_lag_summary(
        summary,
        "super_subject",
        tmp_path,
        formats=["png"],
        colors={"baseline": "#000000", "diver": "#ff0000"},
    )

    axes = captured["fig"].axes[:2]
    assert len(axes) == 2
    assert axes[0].get_ylim() != axes[1].get_ylim()


def test_summary_tables_bold_best_model_for_markdown_and_latex():
    summary = best_lag_rows(
        {
            "task": {
                "baseline": pd.DataFrame({"lags": [0], "score": [0.5]}),
                "diver": pd.DataFrame({"lags": [250], "score": [0.7]}),
            }
        },
        {"task": MetricConfig("score", True, "Score")},
    )

    markdown = summary_wide(summary, bold=True)
    latex = summary_wide(summary, bold=True, latex=True)

    assert markdown.loc[0, "diver"] == "**0.700 (250 ms)**"
    assert latex.loc[0, "diver"] == "\\textbf{0.700 (250 ms)}"


def test_fm_dry_run_migration_mapping_into_current_style_run_dirs(tmp_path):
    source = tmp_path / "results-fm" / "foundation_models"
    dest = tmp_path / "results-fm-normalized"

    old_super = source / "diver" / "content_noncontent" / "persubject_concat"
    write_lag_csv(
        old_super / "diver_persubject_concat_content_noncontent_2026-04-22-20-41-33" / "lag_performance.csv",
        [{"lags": 0, "score": 0.5}],
    )
    write_lag_csv(
        old_super / "diver_persubject_concat_content_noncontent_2026-04-22-21-56-26" / "lag_performance.csv",
        [{"lags": 0, "score": 0.6}],
    )

    old_subject = source / "diver" / "content_noncontent" / "subject_full"
    write_lag_csv(
        old_subject / "subject1_full" / "diver_subject1_full_content_noncontent_2026-04-23-07-41-18" / "lag_performance.csv",
        [{"lags": 0, "score": 0.6}],
    )
    write_lag_csv(
        old_subject / "subject1_full" / "diver_subject1_full_content_noncontent_2026-04-22-20-41-18" / "lag_performance.csv",
        [{"lags": 0, "score": 0.4}],
    )
    write_lag_csv(
        old_subject / "subject2_full" / "diver_subject2_full_content_noncontent_2026-04-23-07-41-20" / "lag_performance.csv",
        [{"lags": 0, "score": 0.7}],
    )

    report = migrate(source, dest, dry_run=True)

    selected = report[report["selected"]]
    assert (
        str(dest / "diver_content_noncontent_super_subject_2026-04-22-21-56-26")
        in selected["dest"].tolist()
    )
    assert (
        str(dest / "diver_content_noncontent_per_subject_2026-04-23-07-41-18" / "subject_1")
        in selected["dest"].tolist()
    )
    assert (
        str(dest / "diver_content_noncontent_per_subject_2026-04-23-07-41-18" / "subject_2")
        in selected["dest"].tolist()
    )
    assert (dest / "migration_report.csv").exists()
    assert not (dest / "diver_content_noncontent_super_subject_2026-04-22-21-56-26" / "lag_performance.csv").exists()
