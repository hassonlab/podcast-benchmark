from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from scripts.generate_paper_results import (
    MetricConfig,
    DestrieuxSurfaceAtlas,
    _build_surface_metric_maps,
    _draw_surface_region_boundaries,
    _draw_surface_region_labels,
    _surface_contour_map,
    bar_start_for_task,
    baseline_condition_comparison_rows,
    baseline_region_peak_lag_rows,
    baseline_region_peak_wide,
    best_lag_latex_table,
    best_lag_summary_plot_style,
    best_lag_significance_tests,
    best_lag_rows,
    best_region_lag_rows,
    brain_map_colormap,
    brain_map_metric_config,
    configured_model_order,
    create_grouped_task_figure,
    draw_grouped_task_backgrounds,
    get_metric_config,
    half_peak_profile_for_curve,
    half_peak_profile_rows,
    holm_adjust_p_values,
    iter_per_region_result_specs,
    load_current_style_run,
    load_results,
    load_per_region_results,
    load_per_region_run,
    curve_for_metric,
    metric_norm,
    neural_conv_model_summary_latex_table,
    neural_conv_model_summary_rows,
    neural_conv_summary_config_path,
    neural_conv_summary_enabled,
    neural_conv_summary_output_name,
    ordered_tasks_by_group_average,
    significance_label,
    normalize_region_name,
    plot_best_lag_summary,
    plot_lag_curves,
    plot_per_region_brains,
    plot_per_region_lag_curves,
    composite_rgba_over_background,
    rasterize_brain_map_surface_artists,
    region_count_legend_handles,
    region_gradient_colors,
    resolve_nilearn_data_dir,
    select_best_lag,
    summary_wide,
    valid_best_lags,
    write_baseline_region_peak_tables,
)


def write_lag_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_loads_current_style_super_subject(tmp_path):
    run_dir = tmp_path / "run"
    write_lag_csv(run_dir / "lag_performance.csv", [{"lags": 0, "score": 0.5}])

    loaded = load_current_style_run(run_dir)

    assert loaded.to_dict("records") == [{"lags": 0, "score": 0.5}]


def test_half_peak_profile_exact_crossing_on_symmetric_curve():
    curve = pd.DataFrame(
        {
            "lags": [-500, -250, 0, 250, 500],
            "score": [0.0, 0.5, 1.0, 0.5, 0.0],
        }
    )

    profile = half_peak_profile_for_curve(
        curve,
        MetricConfig("score", True, "Score"),
    )

    assert profile.peak_value == 1.0
    assert profile.peak_lag == 0.0
    assert profile.half_peak_value == 0.5
    assert profile.ramp_half_peak_lag == -250.0
    assert profile.decay_half_peak_lag == 250.0
    assert profile.ramp_duration == 250.0
    assert profile.decay_duration == 250.0
    assert profile.half_peak_width == 500.0
    assert profile.ramp_slope == pytest.approx(0.002)
    assert profile.decay_slope == pytest.approx(-0.002)
    assert profile.ramp_rate == pytest.approx(0.002)
    assert profile.decay_rate == pytest.approx(0.002)


def test_half_peak_profile_interpolates_between_lag_samples():
    curve = pd.DataFrame(
        {
            "lags": [-500, -250, 0, 250, 500],
            "score": [0.0, 0.25, 1.0, 0.25, 0.0],
        }
    )

    profile = half_peak_profile_for_curve(
        curve,
        MetricConfig("score", True, "Score"),
    )

    assert profile.ramp_half_peak_lag == pytest.approx(-166.6666667)
    assert profile.decay_half_peak_lag == pytest.approx(166.6666667)
    assert profile.half_peak_width == pytest.approx(333.3333333)


def test_half_peak_profile_missing_crossing_returns_nan():
    curve = pd.DataFrame(
        {
            "lags": [-500, -250, 0, 250, 500],
            "score": [0.75, 0.80, 1.0, 0.80, 0.75],
        }
    )

    profile = half_peak_profile_for_curve(
        curve,
        MetricConfig("score", True, "Score"),
    )

    assert profile.half_peak_value == 0.5
    assert np.isnan(profile.ramp_half_peak_lag)
    assert np.isnan(profile.decay_half_peak_lag)
    assert np.isnan(profile.ramp_duration)
    assert np.isnan(profile.decay_duration)
    assert np.isnan(profile.ramp_slope)
    assert np.isnan(profile.decay_slope)


def test_half_peak_profile_roc_auc_stays_auc_units_and_uses_chance():
    loaded = {
        "super_subject": {
            "task": {
                "baseline": pd.DataFrame(
                    {
                        "lags": [-250, 0, 250],
                        "test_roc_auc_mean": [0.5, 0.7, 0.5],
                    }
                )
            }
        }
    }
    config = {
        "plotting": {"half_peak_profile": {"model": "baseline"}},
        "metrics": {
            "task": {
                "column": "test_roc_auc_mean",
                "higher_is_better": True,
                "label": "ROC-AUC",
                "chance": 0.5,
            }
        },
    }

    rows = half_peak_profile_rows(loaded, config)

    assert rows.loc[0, "reference_value"] == 0.5
    assert rows.loc[0, "peak_value"] == 0.7
    assert rows.loc[0, "half_peak_value"] == 0.6


def test_half_peak_bar_task_order_sorts_groups_by_group_average():
    profile = pd.DataFrame(
        {
            "task": ["a1", "a2", "b1", "b2", "a1", "a2", "b1", "b2"],
            "condition": [
                "super_subject",
                "super_subject",
                "super_subject",
                "super_subject",
                "per_subject",
                "per_subject",
                "per_subject",
                "per_subject",
            ],
            "half_peak_width": [10.0, 30.0, 5.0, 7.0, 20.0, 40.0, 6.0, 8.0],
        }
    )
    config = {
        "plotting": {
            "task_groups": {
                "High": ["a1", "a2"],
                "Low": ["b1", "b2"],
            }
        }
    }

    ordered = ordered_tasks_by_group_average(profile, config, "half_peak_width")

    assert ordered == ["b1", "b2", "a1", "a2"]


def test_loads_and_averages_current_style_per_subject(tmp_path):
    run_dir = tmp_path / "run"
    write_lag_csv(run_dir / "subject_1" / "lag_performance.csv", [{"lags": 0, "score": 0.5}])
    write_lag_csv(run_dir / "subject_2" / "lag_performance.csv", [{"lags": 0, "score": 0.7}])

    loaded = load_current_style_run(run_dir)

    assert loaded["lags"].tolist() == [0]
    assert loaded["score"].tolist() == [0.6]


def test_load_results_combines_configured_path_lists_by_lag(tmp_path):
    early_run = tmp_path / "early"
    late_run = tmp_path / "late"
    write_lag_csv(early_run / "lag_performance.csv", [{"lags": -500, "score": 0.4}])
    write_lag_csv(late_run / "lag_performance.csv", [{"lags": 0, "score": 0.6}])
    config = {
        "results": {
            "baseline": {
                "task": {
                    "super_subject": [late_run, early_run],
                }
            }
        }
    }

    loaded = load_results(config)

    df = loaded["super_subject"]["task"]["baseline"]
    assert df.to_dict("records") == [
        {"lags": -500, "score": 0.4},
        {"lags": 0, "score": 0.6},
    ]


def test_load_results_combines_per_subject_path_lists_by_subject_and_lag(tmp_path):
    early_run = tmp_path / "subjects_early"
    late_run = tmp_path / "subjects_late"
    write_lag_csv(early_run / "subject_1" / "lag_performance.csv", [{"lags": -500, "score": 0.4}])
    write_lag_csv(early_run / "subject_2" / "lag_performance.csv", [{"lags": -500, "score": 0.8}])
    write_lag_csv(late_run / "subject_1" / "lag_performance.csv", [{"lags": 0, "score": 0.6}])
    write_lag_csv(late_run / "subject_2" / "lag_performance.csv", [{"lags": 0, "score": 1.0}])
    config = {
        "results": {
            "baseline": {
                "task": {
                    "per_subject": [late_run, early_run],
                }
            }
        }
    }

    loaded = load_results(config)

    df = loaded["per_subject"]["task"]["baseline"]
    assert df["lags"].tolist() == [-500, 0]
    assert np.allclose(df["score"].tolist(), [0.6, 0.8])


def test_load_results_combines_per_subject_path_lists_by_disjoint_subjects(tmp_path):
    first_run = tmp_path / "subjects_first"
    second_run = tmp_path / "subjects_second"
    write_lag_csv(first_run / "subject_1" / "lag_performance.csv", [{"lags": 0, "score": 0.4}])
    write_lag_csv(second_run / "subject_2" / "lag_performance.csv", [{"lags": 0, "score": 0.8}])
    config = {
        "results": {
            "baseline": {
                "task": {
                    "per_subject": [first_run, second_run],
                }
            }
        }
    }

    loaded = load_results(config)

    df = loaded["per_subject"]["task"]["baseline"]
    assert df["lags"].tolist() == [0]
    assert np.allclose(df["score"].tolist(), [0.6])


def test_load_results_rejects_duplicate_per_subject_lags(tmp_path):
    first_run = tmp_path / "subjects_first"
    second_run = tmp_path / "subjects_second"
    write_lag_csv(first_run / "subject_1" / "lag_performance.csv", [{"lags": 0, "score": 0.4}])
    write_lag_csv(second_run / "subject_1" / "lag_performance.csv", [{"lags": 0, "score": 0.8}])
    config = {
        "results": {
            "baseline": {
                "task": {
                    "per_subject": [first_run, second_run],
                }
            }
        }
    }

    with pytest.raises(ValueError, match="baseline/task/per_subject/subject_1"):
        load_results(config)


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


def test_discovers_path_list_specs_from_existing_results_dictionary():
    config = {
        "results": {
            "baseline": {
                "content_noncontent": {
                    "per_region": ["results/early", "results/late"],
                }
            }
        }
    }

    specs = list(iter_per_region_result_specs(config))

    assert specs[0].paths == (Path("results/early"), Path("results/late"))


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


def test_loads_per_region_results_combines_configured_path_lists_by_region_and_lag(tmp_path):
    early_run = tmp_path / "regions_early"
    late_run = tmp_path / "regions_late"
    write_lag_csv(
        early_run / "region_mtg" / "lag_performance.csv",
        [{"lags": -500, "score": 0.4}],
    )
    write_lag_csv(
        late_run / "region_mtg" / "lag_performance.csv",
        [{"lags": 0, "score": 0.6}],
    )
    write_lag_csv(
        late_run / "region_eac" / "lag_performance.csv",
        [{"lags": 0, "score": 0.7}],
    )
    config = {
        "results": {
            "baseline": {
                "task": {
                    "per_region": [late_run, early_run],
                }
            }
        }
    }

    loaded = load_per_region_results(config)

    model_results = loaded["task"]["baseline"]
    assert model_results["MTG"].to_dict("records") == [
        {"lags": -500, "score": 0.4},
        {"lags": 0, "score": 0.6},
    ]
    assert model_results["EAC"].to_dict("records") == [{"lags": 0, "score": 0.7}]


def test_loads_per_region_results_rejects_duplicate_region_lags(tmp_path):
    first_run = tmp_path / "regions_first"
    second_run = tmp_path / "regions_second"
    write_lag_csv(
        first_run / "region_mtg" / "lag_performance.csv",
        [{"lags": 0, "score": 0.4}],
    )
    write_lag_csv(
        second_run / "region_mtg" / "lag_performance.csv",
        [{"lags": 0, "score": 0.6}],
    )
    config = {
        "results": {
            "baseline": {
                "task": {
                    "per_region": [first_run, second_run],
                }
            }
        }
    }

    with pytest.raises(ValueError, match="baseline/task/per_region/MTG"):
        load_per_region_results(config)


def test_configured_model_order_follows_results_order_and_appends_unknown():
    config = {
        "results": {
            "diver": {},
            "baseline": {},
        }
    }

    assert configured_model_order(["baseline", "popt", "diver"], config) == [
        "diver",
        "baseline",
        "popt",
    ]


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


def test_brain_map_metric_config_ignores_standard_metric_bounds_by_default():
    metric = MetricConfig("score", True, "Score", 0.0, 1.0)

    brain_metric = brain_map_metric_config({}, "task", metric)

    assert brain_metric.min_value is None
    assert brain_metric.max_value is None
    assert brain_metric.column == metric.column
    assert brain_metric.label == metric.label


def test_brain_map_metric_config_uses_brain_specific_colorbar_bounds():
    metric = MetricConfig("score", True, "Score", 0.0, 1.0)

    brain_metric = brain_map_metric_config(
        {
            "plotting": {
                "per_region_brains": {
                    "colorbar_bounds": {
                        "task": {"min": 0.25, "max": 0.75},
                    }
                }
            }
        },
        "task",
        metric,
    )

    assert brain_metric.min_value == 0.25
    assert brain_metric.max_value == 0.75


def test_brain_map_colormap_defaults_to_metric_direction():
    assert brain_map_colormap(
        {},
        "task",
        MetricConfig("score", True, "Score"),
    ).name == "viridis"
    assert brain_map_colormap(
        {},
        "task",
        MetricConfig("score", False, "Score"),
    ).name == "viridis_r"


def test_brain_map_colormap_uses_task_specific_config():
    cmap = brain_map_colormap(
        {
            "plotting": {
                "per_region_brains": {
                    "colormaps": {
                        "default": "magma",
                        "task": {"name": "plasma", "reverse": False},
                    }
                }
            }
        },
        "task",
        MetricConfig("score", False, "Score"),
    )

    assert cmap.name == "plasma"


def test_best_lag_summary_plot_style_supports_bar_aliases():
    assert best_lag_summary_plot_style({}) == "point"
    assert best_lag_summary_plot_style(
        {"plotting": {"best_lag_summary_plot_style": "bars"}}
    ) == "bar"
    assert best_lag_summary_plot_style({"plotting": {"use_bar_charts": True}}) == "bar"


def test_bar_start_for_task_prefers_task_specific_plotting_config():
    config = {
        "plotting": {
            "bar_start": 0.0,
            "best_lag_bar_starts": {
                "task_a": 0.5,
            },
        },
        "metrics": {
            "task_b": {
                "column": "score",
                "label": "Score",
                "bar_start": 0.25,
            }
        },
    }

    assert bar_start_for_task(config, "task_a") == 0.5
    assert bar_start_for_task(config, "task_b") == 0.25
    assert bar_start_for_task(config, "task_c") == 0.0


def test_grouped_task_figure_interleaves_mixed_and_semantic_l_shapes():
    config = {
        "plotting": {
            "task_groups": {
                "Mixed": ["mixed_a", "mixed_b", "mixed_c"],
                "Semantic": ["semantic_a", "semantic_b", "semantic_c"],
            }
        }
    }

    layout = create_grouped_task_figure(
        config,
        ["semantic_c", "mixed_b", "semantic_a", "mixed_a", "semantic_b", "mixed_c"],
    )

    slots = {
        task: (
            ax.get_subplotspec().rowspan.start,
            ax.get_subplotspec().colspan.start,
        )
        for task, ax in layout.task_axes.items()
    }
    assert slots == {
        "mixed_a": (0, 0),
        "mixed_b": (1, 0),
        "semantic_a": (0, 2),
        "semantic_b": (1, 2),
    }


def test_grouped_task_backgrounds_color_whitespace_behind_axes():
    from matplotlib.colors import to_rgba

    config = {
        "plotting": {
            "task_group_background_alpha": 0.5,
            "task_group_backgrounds": {
                "Mixed": "#ddeeff",
                "Semantic": "#ffeecc",
            },
            "task_groups": {
                "Mixed": ["mixed"],
                "Semantic": ["semantic"],
            },
        }
    }
    layout = create_grouped_task_figure(config, ["mixed", "semantic"])

    draw_grouped_task_backgrounds(layout, config)

    assert np.allclose(
        layout.task_axes["mixed"].get_facecolor(),
        to_rgba("white"),
    )
    assert np.allclose(
        layout.task_axes["semantic"].get_facecolor(),
        to_rgba("white"),
    )
    background_artists = layout.fig.artists
    assert len(background_artists) == 4
    assert any(
        np.allclose(artist.get_facecolor(), to_rgba("#C9DDF2", 0.5))
        for artist in background_artists
    )
    assert any(
        np.allclose(artist.get_facecolor(), to_rgba("#ffeecc", 0.5))
        for artist in background_artists
    )
    assert all(artist.get_edgecolor()[3] == 0 for artist in background_artists)
    assert {text.get_text() for text in layout.fig.texts} == {
        "Representations",
        "Semantic",
    }
    assert all(text.get_fontweight() == "bold" for text in layout.fig.texts)


def test_l_shape_group_labels_stay_inside_top_group_segment():
    config = {
        "plotting": {
            "task_groups": {
                "Mixed": ["mixed_a", "mixed_b", "mixed_c"],
                "Semantic": ["semantic_a", "semantic_b", "semantic_c"],
            }
        }
    }
    layout = create_grouped_task_figure(
        config,
        ["mixed_a", "mixed_b", "mixed_c", "semantic_a", "semantic_b", "semantic_c"],
    )

    draw_grouped_task_backgrounds(layout, config)

    semantic_label = next(
        text for text in layout.fig.texts if text.get_text() == "Semantic"
    )
    semantic_top_axis = layout.task_axes["semantic_b"]
    left, _bottom, width, _height = semantic_top_axis.get_position().bounds
    assert left <= semantic_label.get_position()[0] <= left + width


def test_rasterize_brain_map_surface_artists_marks_axis_collections():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    collection = ax.scatter([0], [0])

    rasterize_brain_map_surface_artists(ax)

    assert collection.get_rasterized() is True
    plt.close(fig)


def test_select_best_lag_maximize_and_minimize():
    df = pd.DataFrame({"lags": [-1, 0, 1], "score": [0.4, 0.8, 0.2], "loss": [3.0, 2.0, 1.0]})

    max_row = select_best_lag(df, MetricConfig("score", True, "Score"))
    min_row = select_best_lag(df, MetricConfig("loss", False, "Loss"))

    assert max_row["lags"] == 0
    assert min_row["lags"] == 1


def test_valid_best_lags_selects_only_allowed_lags():
    condition_results = {
        "task": {
            "baseline": pd.DataFrame(
                {"lags": [0, 150, 250], "score_mean": [0.20, 0.50, 0.99]}
            ),
            "diver": pd.DataFrame(
                {"lags": [0.0, 150.0, 250.0], "score_mean": [0.60, 0.40, 0.95]}
            ),
        }
    }

    summary = best_lag_rows(
        condition_results,
        {"task": MetricConfig("score_mean", True, "Score")},
        valid_lags=valid_best_lags({"plotting": {"valid_best_lags": [0, 150]}}),
    )

    lags_by_model = dict(zip(summary["model"], summary["lag"]))
    assert lags_by_model == {"baseline": 150, "diver": 0.0}


def test_baseline_condition_comparison_rows_uses_all_baseline_lags():
    loaded = {
        "super_subject": {
            "task": {
                "baseline": pd.DataFrame(
                    {"lags": [0, 150, 250], "score_mean": [0.20, 0.50, 0.99]}
                ),
                "diver": pd.DataFrame(
                    {"lags": [0, 150, 250], "score_mean": [0.60, 0.70, 0.80]}
                ),
            }
        },
        "per_subject": {
            "task": {
                "baseline": pd.DataFrame(
                    {"lags": [0, 150, 250], "score_mean": [0.30, 0.40, 0.95]}
                ),
                "diver": pd.DataFrame(
                    {"lags": [0, 150, 250], "score_mean": [0.65, 0.75, 0.85]}
                ),
            }
        },
    }
    metrics = {"task": MetricConfig("score_mean", True, "Score")}

    filtered_summary = best_lag_rows(
        loaded["super_subject"],
        metrics,
        valid_lags=valid_best_lags({"plotting": {"valid_best_lags": [0, 150]}}),
    )
    comparison_summary = baseline_condition_comparison_rows(loaded, metrics)

    assert filtered_summary.loc[
        filtered_summary["model"] == "baseline", "lag"
    ].iloc[0] == 150
    assert comparison_summary["model"].unique().tolist() == ["baseline"]
    assert dict(zip(comparison_summary["condition"], comparison_summary["lag"])) == {
        "super_subject": 250,
        "per_subject": 250,
    }


def test_missing_valid_best_lags_preserves_existing_selection_behavior():
    assert valid_best_lags({}) is None
    df = pd.DataFrame({"lags": [0, 150, 250], "score": [0.20, 0.50, 0.99]})

    row = select_best_lag(df, MetricConfig("score", True, "Score"), valid_best_lags({}))

    assert row["lags"] == 250


def test_select_best_lag_raises_when_model_has_no_valid_best_lags():
    df = pd.DataFrame({"lags": [250, 500], "score": [0.90, 0.95]})

    with pytest.raises(ValueError, match="task 'task'.*model 'baseline'.*\\[0.0, 150.0\\]"):
        select_best_lag(
            df,
            MetricConfig("score", True, "Score"),
            valid_lags=(0.0, 150.0),
            task="task",
            model="baseline",
        )


def test_select_best_lag_uses_negated_metric_values():
    df = pd.DataFrame({"lags": [0, 1], "loss": [2.0, 1.0]})

    row = select_best_lag(df, MetricConfig("loss", True, "Negative loss", negate=True))

    assert row["lags"] == 1
    assert row["loss"] == -1.0


def test_select_best_lag_filters_before_negating_metric_values():
    df = pd.DataFrame({"lags": [0, 150, 250], "loss": [10.0, 2.0, 1.0]})

    row = select_best_lag(
        df,
        MetricConfig("loss", True, "Negative loss", negate=True),
        valid_lags=(0.0, 150.0),
    )

    assert row["lags"] == 150
    assert row["loss"] == -2.0


def test_curve_for_metric_uses_negated_metric_values():
    df = pd.DataFrame({"lags": [0, 1], "loss": [2.0, 1.0]})

    curve = curve_for_metric(df, MetricConfig("loss", True, "Negative loss", negate=True))

    assert curve["loss"].tolist() == [-2.0, -1.0]


def test_best_lag_significance_tests_compare_winner_to_others_at_best_lags():
    task_results = {
        "baseline": pd.DataFrame(
            {
                "lags": [0, 1],
                "score_mean": [0.60, 0.80],
                "score_fold_1": [0.58, 0.79],
                "score_fold_2": [0.59, 0.82],
                "score_fold_3": [0.60, 0.81],
                "score_fold_4": [0.61, 0.84],
                "score_fold_5": [0.62, 0.83],
            }
        ),
        "diver": pd.DataFrame(
            {
                "lags": [0, 1],
                "score_mean": [0.55, 0.62],
                "score_fold_1": [0.53, 0.61],
                "score_fold_2": [0.55, 0.64],
                "score_fold_3": [0.57, 0.60],
                "score_fold_4": [0.56, 0.65],
                "score_fold_5": [0.54, 0.59],
            }
        ),
    }
    summary = best_lag_rows(
        {"task": task_results},
        {"task": MetricConfig("score_mean", True, "Score")},
    )

    comparisons = best_lag_significance_tests(
        summary,
        task_results,
        MetricConfig("score_mean", True, "Score"),
    )

    assert len(comparisons) == 1
    assert comparisons[0]["winner"] == "baseline"
    assert comparisons[0]["other"] == "diver"
    assert comparisons[0]["label"] in {"*", "**", "***"}
    assert comparisons[0]["n"] == 5


def test_best_lag_significance_tests_can_skip_holm_correction():
    task_results = {
        "winner": pd.DataFrame(
            {
                "lags": [0],
                "score_mean": [0.80],
                "score_fold_1": [0.80],
                "score_fold_2": [0.82],
                "score_fold_3": [0.81],
                "score_fold_4": [0.83],
                "score_fold_5": [0.84],
            }
        ),
        "close": pd.DataFrame(
            {
                "lags": [0],
                "score_mean": [0.79],
                "score_fold_1": [0.79],
                "score_fold_2": [0.81],
                "score_fold_3": [0.80],
                "score_fold_4": [0.82],
                "score_fold_5": [0.83],
            }
        ),
        "low": pd.DataFrame(
            {
                "lags": [0],
                "score_mean": [0.60],
                "score_fold_1": [0.60],
                "score_fold_2": [0.61],
                "score_fold_3": [0.62],
                "score_fold_4": [0.63],
                "score_fold_5": [0.64],
            }
        ),
    }
    summary = best_lag_rows(
        {"task": task_results},
        {"task": MetricConfig("score_mean", True, "Score")},
    )

    uncorrected = best_lag_significance_tests(
        summary,
        task_results,
        MetricConfig("score_mean", True, "Score"),
        correct_multiple_comparisons=False,
    )
    corrected = best_lag_significance_tests(
        summary,
        task_results,
        MetricConfig("score_mean", True, "Score"),
        correct_multiple_comparisons=True,
    )

    uncorrected_by_model = {item["other"]: item for item in uncorrected}
    corrected_by_model = {item["other"]: item for item in corrected}
    assert uncorrected_by_model["low"]["label"] == "*"
    assert corrected_by_model["low"]["label"] == "n.s."
    assert corrected_by_model["low"]["p_value"] > uncorrected_by_model["low"]["p_value"]


def test_significance_label_formats_standard_thresholds():
    assert significance_label(0.0005) == "***"
    assert significance_label(0.005) == "**"
    assert significance_label(0.04) == "*"
    assert significance_label(0.20) == "n.s."


def test_holm_adjust_p_values_controls_familywise_error_in_order():
    adjusted = holm_adjust_p_values([0.03, 0.01, 0.04])

    assert adjusted == [0.06, 0.03, 0.06]


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


def test_baseline_region_peak_lag_rows_filters_to_baseline_model():
    per_region_results = {
        "task": {
            "baseline": {
                "EAC": pd.DataFrame({"lags": [0, 1], "score": [0.5, 0.8]}),
                "MTG": pd.DataFrame({"lags": [0, 1], "score": [0.7, 0.4]}),
            },
            "diver": {
                "EAC": pd.DataFrame({"lags": [0, 1], "score": [0.9, 0.1]}),
            },
        }
    }

    rows = baseline_region_peak_lag_rows(
        per_region_results,
        {"task": MetricConfig("score", True, "Score")},
    )

    assert set(rows["model"]) == {"baseline"}
    by_region = rows.set_index("region")
    assert by_region.loc["EAC", "value"] == 0.8
    assert by_region.loc["EAC", "lag"] == 1
    assert by_region.loc["MTG", "value"] == 0.7
    assert by_region.loc["MTG", "lag"] == 0


def test_baseline_region_peak_wide_orders_regions_and_formats_peak_cells():
    summary = pd.DataFrame(
        [
            {"task": "task", "region": "MTG", "value": 0.7, "lag": 0},
            {"task": "task", "region": "EAC", "value": 0.8, "lag": 1},
        ]
    )

    table = baseline_region_peak_wide(summary)

    assert table.columns.tolist() == ["task", "EAC", "MTG"]
    assert table.to_dict("records") == [
        {"task": "task", "EAC": "0.800 (1 ms)", "MTG": "0.700 (0 ms)"}
    ]


def test_write_baseline_region_peak_tables_writes_requested_formats(tmp_path):
    summary = pd.DataFrame(
        [{"task": "task", "region": "EAC", "value": 0.8, "lag": 1}]
    )

    write_baseline_region_peak_tables(
        summary,
        tmp_path,
        formats=["csv", "markdown", "latex"],
    )

    assert (tmp_path / "baseline_region_peak_lags.csv").exists()
    assert (tmp_path / "baseline_region_peak_lags.md").exists()
    assert (tmp_path / "baseline_region_peak_lags.tex").exists()


def test_plot_lag_curves_preserves_unequal_lag_sets(tmp_path):
    legacy_output = tmp_path / "lag_curves_task_super_subject.png"
    legacy_condition_output = tmp_path / "lag_curves_super_subject.png"
    legacy_output.write_text("stale")
    legacy_condition_output.write_text("stale")
    loaded = {
        "super_subject": {
            "task": {
                "baseline": pd.DataFrame({"lags": [-1, 0], "score": [0.5, 0.6]}),
                "diver": pd.DataFrame(
                    {"lags": [-2, 1, 2], "score": [0.4, 0.7, 0.65]}
                ),
            }
        },
        "per_subject": {
            "task": {
                "baseline": pd.DataFrame(
                    {"lags": [-2, 1, 2], "score": [0.4, 0.7, 0.65]}
                ),
                "diver": pd.DataFrame({"lags": [-1, 0], "score": [0.5, 0.6]}),
            }
        },
    }
    config = {
        "metrics": {
            "task": {"column": "score", "higher_is_better": True, "label": "Score"}
        }
    }

    plot_lag_curves(
        loaded,
        config,
        tmp_path,
        formats=["png"],
        colors={"baseline": "#000000", "diver": "#ff0000"},
    )

    assert (tmp_path / "lag_curves.png").exists()
    assert not legacy_output.exists()
    assert not legacy_condition_output.exists()


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
    assert captured["output_base"] == tmp_path / "lag_curves"


def test_plot_lag_curves_only_plots_baseline_conditions(tmp_path, monkeypatch):
    captured = {}

    def capture_figure(fig, output_base, formats):
        captured["fig"] = fig

    monkeypatch.setattr("scripts.generate_paper_results.save_figure", capture_figure)
    loaded = {
        "super_subject": {
            "task": {
                "baseline": pd.DataFrame({"lags": [-1, 0], "score": [0.5, 0.6]}),
                "diver": pd.DataFrame({"lags": [-1, 0], "score": [0.1, 0.2]}),
            }
        },
        "per_subject": {
            "task": {
                "baseline": pd.DataFrame({"lags": [-1, 0], "score": [0.4, 0.7]}),
                "brainbert": pd.DataFrame({"lags": [-1, 0], "score": [0.3, 0.4]}),
            }
        },
    }
    config = {
        "metrics": {
            "task": {"column": "score", "higher_is_better": True, "label": "Score"}
        }
    }

    plot_lag_curves(
        loaded,
        config,
        tmp_path,
        formats=["png"],
        colors={"baseline": "#000000", "diver": "#ff0000", "brainbert": "#00ff00"},
    )

    plotted_lines = [
        line for line in captured["fig"].axes[0].lines if line.get_label()[0] != "_"
    ]
    assert [line.get_label() for line in plotted_lines] == [
        "Multi-Subject",
        "Single Subject",
    ]
    assert [line.get_color() for line in plotted_lines] == ["#1F4E79", "#2CA7A0"]
    assert [line.get_linestyle() for line in plotted_lines] == ["-", "-"]


def test_plot_lag_curves_includes_configured_models(tmp_path, monkeypatch):
    captured = {}

    def capture_figure(fig, output_base, formats):
        captured["fig"] = fig

    monkeypatch.setattr("scripts.generate_paper_results.save_figure", capture_figure)
    loaded = {
        "super_subject": {
            "task": {
                "baseline": pd.DataFrame({"lags": [-1, 0], "score": [0.5, 0.6]}),
                "linear": pd.DataFrame({"lags": [-1, 0], "score": [0.4, 0.7]}),
                "diver": pd.DataFrame({"lags": [-1, 0], "score": [0.1, 0.2]}),
            }
        },
        "per_subject": {
            "task": {
                "baseline": pd.DataFrame({"lags": [-1, 0], "score": [0.45, 0.65]}),
                "linear": pd.DataFrame({"lags": [-1, 0], "score": [0.35, 0.75]}),
            }
        },
    }
    config = {
        "plotting": {
            "lag_curve_models": ["baseline", "linear"],
            "model_display_names": {"baseline": "CNN", "linear": "Linear"},
        },
        "metrics": {
            "task": {"column": "score", "higher_is_better": True, "label": "Score"}
        },
    }

    plot_lag_curves(
        loaded,
        config,
        tmp_path,
        formats=["png"],
        colors={"baseline": "#000000", "linear": "#ff00ff", "diver": "#ff0000"},
    )

    plotted_lines = [
        line for line in captured["fig"].axes[0].lines if line.get_label()[0] != "_"
    ]
    assert [line.get_label() for line in plotted_lines] == [
        "CNN Multi-Subject",
        "Linear Multi-Subject",
        "CNN Single Subject",
        "Linear Single Subject",
    ]
    assert [line.get_color() for line in plotted_lines] == [
        "#000000",
        "#ff00ff",
        "#000000",
        "#ff00ff",
    ]
    assert [line.get_linestyle() for line in plotted_lines] == ["-", "-", "--", "--"]


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
        },
    }
    config = {
        "metrics": {
            "content": {"column": "score", "higher_is_better": True, "label": "Score"},
            "syntax": {"column": "score", "higher_is_better": True, "label": "Score"},
        }
    }

    plot_per_region_lag_curves(per_region_results, config, tmp_path, formats=["png"])

    assert (tmp_path / "per_region_lags_baseline.png").exists()


def test_plot_per_region_brains_writes_one_task_grid_per_model(tmp_path, monkeypatch):
    from nilearn import plotting
    from matplotlib.colors import to_rgba

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
    monkeypatch.setattr(
        "scripts.generate_paper_results._load_region_electrodes",
        lambda *args: electrodes,
    )
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
    plotted_axes = []

    def fake_plot_surf_stat_map(**kwargs):
        plotted_maps.append(
            (
                kwargs["hemi"],
                kwargs["stat_map"].copy(),
                kwargs["vmin"],
                kwargs["vmax"],
                kwargs["cmap"],
            )
        )
        plotted_axes.append(kwargs["axes"])

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
    config = {
        "metrics": {
            "task": {
                "column": "score",
                "higher_is_better": True,
                "label": "Score",
                "min": 0.0,
                "max": 1.0,
            }
        },
        "plotting": {
            "task_group_background_alpha": 0.5,
            "task_group_backgrounds": {"Semantic": "#ffeecc"},
            "task_groups": {"Semantic": ["task"]},
            "per_region_brains": {
                "colorbar_bounds": {"task": {"min": 0.55, "max": 0.75}},
                "colormaps": {"task": "plasma"},
            },
        },
    }

    plot_per_region_brains(
        per_region_results,
        config,
        tmp_path,
        formats=["png"],
        data_root=tmp_path,
        nilearn_data_dir=tmp_path / "nilearn",
        include_bad=False,
    )

    assert (tmp_path / "per_region_brains_baseline.png").exists()
    by_hemi = {hemi: stat_map for hemi, stat_map, _vmin, _vmax, _cmap in plotted_maps}
    assert by_hemi["left"][:2].tolist() == [0.6, 0.6]
    assert np.isnan(by_hemi["left"][2:]).all()
    assert by_hemi["right"][:2].tolist() == [0.7, 0.7]
    assert np.isnan(by_hemi["right"][2:]).all()
    assert all(vmin == 0.55 for _hemi, _stat_map, vmin, _vmax, _cmap in plotted_maps)
    assert all(vmax == 0.75 for _hemi, _stat_map, _vmin, vmax, _cmap in plotted_maps)
    assert all(cmap.name == "plasma" for _hemi, _stat_map, _vmin, _vmax, cmap in plotted_maps)
    assert plotted_axes
    assert all(
        np.allclose(ax.get_facecolor(), plotted_axes[0].figure.get_facecolor())
        for ax in plotted_axes
    )
    assert any(
        text.get_text() == "Semantic" and text.get_fontweight() == "bold"
        for text in plotted_axes[0].figure.texts
    )


def test_plot_per_region_brains_can_plot_left_hemisphere_only(tmp_path, monkeypatch):
    from nilearn import plotting

    electrodes = pd.DataFrame(
        [
            {"x": -42.0, "y": -20.0, "z": 18.0, "region_group": "EAC"},
            {"x": 42.0, "y": -18.0, "z": 24.0, "region_group": "RIGHT"},
        ]
    )
    monkeypatch.setattr(
        "scripts.generate_paper_results._load_region_electrodes",
        lambda *args: electrodes,
    )
    monkeypatch.setattr(
        "scripts.generate_paper_results._load_destrieux_surface_atlas",
        lambda *args: DestrieuxSurfaceAtlas(
            labels=["Unknown", "G_temp_sup-Lateral", "G_postcentral"],
            maps={
                "left": np.array([1, 1, 0, 0]),
                "right": np.array([2, 2, 0, 0]),
            },
            mesh={
                "left": SimpleNamespace(coordinates=np.zeros((4, 3))),
                "right": SimpleNamespace(coordinates=np.zeros((4, 3))),
            },
            sulcal={
                "left": np.zeros(4),
                "right": np.zeros(4),
            },
        ),
    )
    plotted_hemis = []
    titled_axes = []

    def fake_plot_surf_stat_map(**kwargs):
        plotted_hemis.append(kwargs["hemi"])
        titled_axes.append(kwargs["axes"])

    monkeypatch.setattr(plotting, "plot_surf_stat_map", fake_plot_surf_stat_map)
    monkeypatch.setattr(plotting, "plot_surf_contours", lambda **kwargs: None)
    per_region_results = {
        "task": {
            "baseline": {
                "EAC": pd.DataFrame({"lags": [0], "score": [0.6]}),
                "RIGHT": pd.DataFrame({"lags": [0], "score": [0.9]}),
            }
        }
    }
    config = {
        "metrics": {
            "task": {"column": "score", "higher_is_better": True, "label": "Score"}
        },
        "plotting": {"per_region_brains": {"ignore_right_hemisphere": True}},
    }

    plot_per_region_brains(
        per_region_results,
        config,
        tmp_path,
        formats=["png"],
        data_root=tmp_path,
        nilearn_data_dir=tmp_path / "nilearn",
        include_bad=False,
    )

    assert plotted_hemis == ["left"]
    assert titled_axes[0].get_title() == "task"


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


def test_surface_region_labels_show_region_names_without_electrode_counts():
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
    )

    assert any(text.get_text() == "STG" for text in ax.texts)
    assert all("n=" not in text.get_text() for text in ax.texts)
    plt.close(fig)


def test_region_count_legend_handles_include_electrode_counts():
    handles = region_count_legend_handles(
        ["PC", "EAC"],
        {"EAC": 3, "PC": 5},
    )

    assert [handle.get_label() for handle in handles] == ["STG (n=3)", "PC (n=5)"]


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


def test_plot_best_lag_summary_orders_models_from_config(tmp_path, monkeypatch):
    captured = {}

    def capture_figure(fig, output_base, formats):
        captured["fig"] = fig

    monkeypatch.setattr("scripts.generate_paper_results.save_figure", capture_figure)
    summary = pd.DataFrame(
        [
            {
                "condition": "super_subject",
                "task": "task",
                "model": "baseline",
                "metric": "score",
                "metric_label": "Score",
                "value": 0.1,
                "lag": 0,
                "higher_is_better": True,
            },
            {
                "condition": "super_subject",
                "task": "task",
                "model": "diver",
                "metric": "score",
                "metric_label": "Score",
                "value": 0.2,
                "lag": 0,
                "higher_is_better": True,
            },
        ]
    )

    plot_best_lag_summary(
        summary,
        "super_subject",
        tmp_path,
        formats=["png"],
        colors={"baseline": "#000000", "diver": "#ff0000"},
        config={
            "results": {
                "diver": {"task": {"super_subject": "diver/run"}},
                "baseline": {"task": {"super_subject": "baseline/run"}},
            }
        },
    )

    labels = [label.get_text() for label in captured["fig"].axes[0].get_xticklabels()]
    assert labels == ["diver", "baseline"]


def test_plot_best_lag_summary_draws_significance_annotations(tmp_path, monkeypatch):
    captured = {}

    def capture_figure(fig, output_base, formats):
        captured["fig"] = fig

    monkeypatch.setattr("scripts.generate_paper_results.save_figure", capture_figure)
    condition_results = {
        "task": {
            "baseline": pd.DataFrame(
                {
                    "lags": [0],
                    "score_mean": [0.80],
                    "score_fold_1": [0.79],
                    "score_fold_2": [0.82],
                    "score_fold_3": [0.81],
                    "score_fold_4": [0.84],
                    "score_fold_5": [0.83],
                }
            ),
            "diver": pd.DataFrame(
                {
                    "lags": [0],
                    "score_mean": [0.62],
                    "score_fold_1": [0.61],
                    "score_fold_2": [0.64],
                    "score_fold_3": [0.60],
                    "score_fold_4": [0.65],
                    "score_fold_5": [0.59],
                }
            ),
        }
    }
    summary = best_lag_rows(
        condition_results,
        {"task": MetricConfig("score_mean", True, "Score")},
    )

    plot_best_lag_summary(
        summary,
        "super_subject",
        tmp_path,
        formats=["png"],
        colors={"baseline": "#000000", "diver": "#ff0000"},
        config={"plotting": {"check_best_lag_significance": True}},
        condition_results=condition_results,
    )

    labels = {text.get_text() for text in captured["fig"].axes[0].texts}
    assert labels & {"*", "**", "***", "n.s."}


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

    assert markdown.loc[0, "baseline"] == "0.500 (0 ms; -29%)"
    assert markdown.loc[0, "diver"] == "**0.700 (250 ms)**"
    assert latex.loc[0, "diver"] == "\\textbf{0.700 (250 ms)}"


def test_best_lag_latex_table_includes_relative_percent_for_non_max_values():
    summary = best_lag_rows(
        {
            "task": {
                "baseline": pd.DataFrame({"lags": [0], "score": [0.5]}),
                "diver": pd.DataFrame({"lags": [250], "score": [0.7]}),
            }
        },
        {"task": MetricConfig("score", True, "Score")},
    )
    summary.insert(0, "condition", "super_subject")

    table = best_lag_latex_table(summary, {})

    assert r"0.500 (-29\%)" in table
    assert r"\textbf{0.700}" in table


def test_neural_conv_summary_options_default_and_override_paths():
    assert neural_conv_summary_enabled({}) is True
    assert neural_conv_summary_output_name({}) == "neural_conv_decoder_summary"

    config = {
        "model_summary": {
            "neural_conv_decoder": {
                "enabled": False,
                "config": "custom.yml",
                "output_name": "custom_summary",
            }
        }
    }

    assert neural_conv_summary_enabled(config) is False
    assert neural_conv_summary_config_path(config) == Path("custom.yml")
    assert neural_conv_summary_output_name(config) == "custom_summary"


def test_neural_conv_model_summary_rows_and_latex_table():
    import torch

    model = torch.nn.Sequential(
        torch.nn.Conv1d(3, 4, kernel_size=2),
        torch.nn.ReLU(),
        torch.nn.AdaptiveMaxPool1d(1),
    )
    summary = neural_conv_model_summary_rows(model, torch.zeros(2, 3, 5))

    assert summary["type"].tolist() == ["Conv1d", "ReLU", "AdaptiveMaxPool1d"]
    assert summary.loc[0, "output_shape"] == "(2, 4, 4)"
    assert summary.loc[0, "parameters"] == 28

    latex = neural_conv_model_summary_latex_table(
        summary,
        model,
        Path(
            "configs/baselines/content_noncontent_task/"
            "neural_conv_decoder/supersubject.yml"
        ),
    )

    assert r"\caption{Neural convolution decoder model summary." in latex
    assert "content\\_noncontent\\_task" in latex
    assert r"\textbf{Total} &  &  & 28 \\" in latex
