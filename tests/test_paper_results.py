from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts.generate_paper_results import (
    MetricConfig,
    DestrieuxSurfaceAtlas,
    _build_surface_metric_maps,
    _draw_surface_region_boundaries,
    _draw_surface_region_labels,
    _surface_contour_map,
    best_lag_rows,
    best_region_lag_rows,
    get_metric_config,
    iter_per_region_result_specs,
    load_current_style_run,
    load_per_region_results,
    load_per_region_run,
    curve_for_metric,
    metric_norm,
    normalize_region_name,
    plot_best_lag_summary,
    plot_lag_curves,
    plot_per_region_brains,
    plot_per_region_lag_curves,
    region_gradient_colors,
    resolve_nilearn_data_dir,
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


def test_discovers_per_region_specs_from_existing_results_dictionary():
    config = {
        "results": {
            "baseline": {
                "content_noncontent": {
                    "super_subject": "results/super",
                    "per_subject": "results/subjects",
                    "per_region": "results/regions",
                }
            }
        }
    }

    specs = list(iter_per_region_result_specs(config))

    assert len(specs) == 1
    assert specs[0].model == "baseline"
    assert specs[0].task == "content_noncontent"
    assert specs[0].condition == "per_region"
    assert specs[0].path == Path("results/regions")


def test_loads_per_region_run_and_normalizes_region_names(tmp_path):
    run_dir = tmp_path / "run"
    write_lag_csv(run_dir / "region_eac" / "lag_performance.csv", [{"lags": 0, "score": 0.5}])
    write_lag_csv(run_dir / "region_right" / "lag_performance.csv", [{"lags": 0, "score": 0.6}])

    loaded = load_per_region_run(run_dir)

    assert normalize_region_name("region_eac") == "EAC"
    assert sorted(loaded) == ["EAC", "RIGHT"]
    assert loaded["RIGHT"].to_dict("records") == [{"lags": 0, "score": 0.6}]


def test_loads_per_region_results_by_task_and_model(tmp_path):
    run_dir = tmp_path / "regions"
    write_lag_csv(run_dir / "region_mtg" / "lag_performance.csv", [{"lags": 0, "score": 0.5}])
    config = {
        "results": {
            "baseline": {
                "task": {
                    "per_region": run_dir,
                }
            }
        }
    }

    loaded = load_per_region_results(config)

    assert list(loaded) == ["task"]
    assert list(loaded["task"]) == ["baseline"]
    assert list(loaded["task"]["baseline"]) == ["MTG"]


def test_resolve_nilearn_data_dir_prefers_explicit_path(tmp_path):
    explicit = tmp_path / "explicit"

    assert resolve_nilearn_data_dir(tmp_path / "out", explicit) == explicit


def test_metric_config_loads_optional_bounds():
    metric = get_metric_config(
        {
            "metrics": {
                "task": {
                    "column": "score",
                    "higher_is_better": True,
                    "label": "Score",
                    "min": 0.25,
                    "max": 0.75,
                }
            }
        },
        "task",
    )

    assert metric.min_value == 0.25
    assert metric.max_value == 0.75


def test_metric_config_loads_negate_option():
    metric = get_metric_config(
        {
            "metrics": {
                "task": {
                    "column": "loss",
                    "higher_is_better": True,
                    "label": "Negative loss",
                    "negate": True,
                }
            }
        },
        "task",
    )

    assert metric.negate is True


def test_metric_norm_uses_configured_bounds():
    norm = metric_norm([0.4, 0.6], MetricConfig("score", True, "Score", 0.0, 1.0))

    assert norm.vmin == 0.0
    assert norm.vmax == 1.0


def test_select_best_lag_maximize_and_minimize():
    df = pd.DataFrame({"lags": [-1, 0, 1], "score": [0.4, 0.8, 0.2], "loss": [3.0, 2.0, 1.0]})

    max_row = select_best_lag(df, MetricConfig("score", True, "Score"))
    min_row = select_best_lag(df, MetricConfig("loss", False, "Loss"))

    assert max_row["lags"] == 0
    assert min_row["lags"] == 1


def test_select_best_lag_uses_negated_metric_values():
    df = pd.DataFrame({"lags": [0, 1], "loss": [2.0, 1.0]})

    row = select_best_lag(df, MetricConfig("loss", True, "Negative loss", negate=True))

    assert row["lags"] == 1
    assert row["loss"] == -1.0


def test_curve_for_metric_uses_negated_metric_values():
    df = pd.DataFrame({"lags": [0, 1], "loss": [2.0, 1.0]})

    curve = curve_for_metric(df, MetricConfig("loss", True, "Negative loss", negate=True))

    assert curve["loss"].tolist() == [-2.0, -1.0]


def test_best_region_lag_rows_selects_best_lag_per_region():
    rows = best_region_lag_rows(
        {
            "EAC": pd.DataFrame({"lags": [0, 1], "score": [0.5, 0.8]}),
            "MTG": pd.DataFrame({"lags": [0, 1], "score": [0.7, 0.4]}),
        },
        MetricConfig("score", True, "Score"),
    )

    by_region = rows.set_index("region")
    assert by_region.loc["EAC", "value"] == 0.8
    assert by_region.loc["EAC", "lag"] == 1
    assert by_region.loc["MTG", "value"] == 0.7
    assert by_region.loc["MTG", "lag"] == 0


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


def test_plot_lag_curves_applies_metric_bounds(tmp_path, monkeypatch):
    captured = {}

    def capture_figure(fig, output_base, formats):
        captured["fig"] = fig
        captured["output_base"] = output_base
        captured["formats"] = formats

    monkeypatch.setattr("scripts.generate_paper_results.save_figure", capture_figure)
    loaded = {
        "super_subject": {
            "task": {
                "baseline": pd.DataFrame({"lags": [-1, 0], "score": [0.5, 0.6]}),
            }
        }
    }
    config = {
        "metrics": {
            "task": {
                "column": "score",
                "higher_is_better": True,
                "label": "Score",
                "min": 0.0,
                "max": 1.0,
            }
        }
    }

    plot_lag_curves(
        loaded,
        config,
        tmp_path,
        formats=["png"],
        colors={"baseline": "#000000"},
    )

    assert captured["fig"].axes[0].get_ylim() == (0.0, 1.0)


def test_region_gradient_colors_use_stable_low_to_high_order():
    colors = region_gradient_colors(["TP", "EAC", "MTG"])

    assert list(colors) == ["EAC", "MTG", "TP"]


def test_plot_per_region_lag_curves_writes_one_task_grid_per_model(tmp_path):
    per_region_results = {
        "content": {
            "baseline": {
                "EAC": pd.DataFrame({"lags": [-1, 0], "score": [0.5, 0.6]}),
                "MTG": pd.DataFrame({"lags": [-1, 0], "score": [0.4, 0.7]}),
            }
        },
        "syntax": {
            "baseline": {
                "EAC": pd.DataFrame({"lags": [-1, 0], "score": [0.3, 0.4]}),
                "TP": pd.DataFrame({"lags": [-1, 0], "score": [0.7, 0.8]}),
            }
        }
    }
    config = {
        "metrics": {
            "content": {"column": "score", "higher_is_better": True, "label": "Score"},
            "syntax": {"column": "score", "higher_is_better": True, "label": "Score"},
        }
    }

    plot_per_region_lag_curves(per_region_results, config, tmp_path, formats=["png"])

    assert (tmp_path / "per_region_lags_baseline.png").exists()


def test_plot_per_region_brains_writes_one_figure_per_model_task(tmp_path, monkeypatch):
    from nilearn import plotting

    electrodes = pd.DataFrame(
        [
            {"x": -42.0, "y": -20.0, "z": 18.0, "region_group": "EAC"},
            {"x": -38.0, "y": -18.0, "z": 22.0, "region_group": "EAC"},
            {"x": -36.0, "y": -24.0, "z": 20.0, "region_group": "EAC"},
            {"x": 42.0, "y": -18.0, "z": 24.0, "region_group": "RIGHT"},
            {"x": 44.0, "y": -14.0, "z": 26.0, "region_group": "RIGHT"},
            {"x": 39.0, "y": -20.0, "z": 21.0, "region_group": "RIGHT"},
        ]
    )
    monkeypatch.setattr("scripts.generate_paper_results._load_region_electrodes", lambda *args: electrodes)
    monkeypatch.setattr(
        "scripts.generate_paper_results._load_destrieux_surface_atlas",
        lambda *args: DestrieuxSurfaceAtlas(
            labels=["Unknown", "G_temp_sup-Lateral", "G_postcentral"],
            maps={
                "left": np.array([1, 1, 0, 0]),
                "right": np.array([2, 2, 0, 0]),
            },
            mesh={
                "left": SimpleNamespace(
                    coordinates=np.array(
                        [
                            [-2.0, 0.0, 0.0],
                            [-2.0, 1.0, 0.0],
                            [-1.0, 0.0, 0.0],
                            [-1.0, 1.0, 0.0],
                        ]
                    )
                ),
                "right": SimpleNamespace(
                    coordinates=np.array(
                        [
                            [2.0, 0.0, 0.0],
                            [2.0, 1.0, 0.0],
                            [1.0, 0.0, 0.0],
                            [1.0, 1.0, 0.0],
                        ]
                    )
                ),
            },
            sulcal={
                "left": np.zeros(4),
                "right": np.zeros(4),
            },
        ),
    )
    plotted_maps = []

    def fake_plot_surf_stat_map(**kwargs):
        plotted_maps.append((kwargs["hemi"], kwargs["stat_map"].copy()))

    monkeypatch.setattr(plotting, "plot_surf_stat_map", fake_plot_surf_stat_map)
    monkeypatch.setattr(plotting, "plot_surf_contours", lambda **kwargs: None)
    per_region_results = {
        "task": {
            "baseline": {
                "EAC": pd.DataFrame({"lags": [0, 1], "score": [0.5, 0.6]}),
                "RIGHT": pd.DataFrame({"lags": [0, 1], "score": [0.7, 0.4]}),
            }
        }
    }
    config = {"metrics": {"task": {"column": "score", "higher_is_better": True, "label": "Score"}}}

    plot_per_region_brains(
        per_region_results,
        config,
        tmp_path,
        formats=["png"],
        data_root=tmp_path,
        nilearn_data_dir=tmp_path / "nilearn",
        include_bad=False,
    )

    assert (tmp_path / "per_region_brain_baseline_task.png").exists()
    by_hemi = dict(plotted_maps)
    assert by_hemi["left"][:2].tolist() == [0.6, 0.6]
    assert np.isnan(by_hemi["left"][2:]).all()
    assert by_hemi["right"][:2].tolist() == [0.7, 0.7]
    assert np.isnan(by_hemi["right"][2:]).all()


def test_build_surface_metric_maps_respects_lateralized_region_labels():
    metric_maps, region_masks = _build_surface_metric_maps(
        atlas_labels=["Unknown", "G_temporal_middle", "G_temp_sup-Lateral"],
        atlas_maps={
            "left": np.array([1, 2, 0, 1]),
            "right": np.array([1, 2, 1, 0]),
        },
        region_groups={
            "MTG": ["L G_temporal_middle"],
            "RIGHT": ["R G_temporal_middle"],
        },
        metric_by_region={"MTG": 0.4, "RIGHT": 0.9},
    )

    assert metric_maps["left"][0] == 0.4
    assert metric_maps["left"][3] == 0.4
    assert np.isnan(metric_maps["left"][1])
    assert metric_maps["right"][0] == 0.9
    assert metric_maps["right"][2] == 0.9
    assert np.isnan(metric_maps["right"][1])
    assert set(region_masks["left"]) == {"MTG"}
    assert set(region_masks["right"]) == {"RIGHT"}


def test_surface_region_labels_include_electrode_counts():
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _draw_surface_region_labels(
        ax,
        SimpleNamespace(
            coordinates=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0],
                    [2.0, 2.0, 2.0],
                ]
            )
        ),
        {"EAC": np.array([True, True, False])},
        {"EAC": 3},
    )

    assert any(text.get_text() == "EAC\nn=3" for text in ax.texts)
    plt.close(fig)


def test_surface_region_boundaries_draw_between_regions(monkeypatch):
    from nilearn import plotting
    import matplotlib.pyplot as plt

    captured = {}

    def fake_plot_surf_contours(**kwargs):
        captured.update(kwargs)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    mesh = SimpleNamespace(coordinates=np.zeros((4, 3)))

    monkeypatch.setattr(plotting, "plot_surf_contours", fake_plot_surf_contours)
    _draw_surface_region_boundaries(
        ax,
        mesh,
        {
            "EAC": np.array([True, True, False, False]),
            "MTG": np.array([False, False, True, True]),
        },
    )

    assert captured["surf_mesh"] is mesh
    assert captured["levels"] == [1, 2]
    assert captured["roi_map"].tolist() == [1, 1, 2, 2]
    plt.close(fig)


def test_surface_contour_map_uses_stable_region_order():
    contour_map, levels = _surface_contour_map(
        {
            "MTG": np.array([True, False, False]),
            "EAC": np.array([False, True, False]),
        }
    )

    assert levels == [1, 2]
    assert contour_map.tolist() == [2, 1, 0]


def test_surface_region_boundaries_noop_without_regions(monkeypatch):
    from nilearn import plotting
    import matplotlib.pyplot as plt

    called = False

    def fake_plot_surf_contours(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(plotting, "plot_surf_contours", fake_plot_surf_contours)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    _draw_surface_region_boundaries(
        ax,
        SimpleNamespace(coordinates=np.zeros((0, 3))),
        {},
    )

    assert called is False
    plt.close(fig)


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
