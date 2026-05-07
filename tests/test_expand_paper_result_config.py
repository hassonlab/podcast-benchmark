from pathlib import Path

import pandas as pd
import yaml

from scripts.expand_paper_result_config import expand_config, expected_lags


def write_lag_csv(path: Path, lags):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"lags": lags, "score": [0.5] * len(lags)}).to_csv(path, index=False)


def write_run_config(
    run_dir: Path,
    *,
    task_name: str = "task_a",
    run_mode: str = "super_subject",
    constructor_name: str = "baseline_model",
    subject_ids=None,
):
    subject_ids = [1, 2] if subject_ids is None else subject_ids
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "config_setter_name": ["baseline"],
        "model_spec": {
            "constructor_name": constructor_name,
            "params": {"hidden": 4},
        },
        "run_mode": run_mode,
        "task_config": {
            "task_name": task_name,
            "data_params": {"subject_ids": subject_ids},
            "task_specific_config": {"source": "same"},
        },
        "training_params": {
            "min_lag": -800,
            "max_lag": -775,
            "lag_step_size": 25,
            "batch_size": 8,
        },
    }
    with (run_dir / "config.yml").open("w") as f:
        yaml.safe_dump(config, f)


def test_expand_config_adds_matching_super_subject_candidate(tmp_path):
    configured = tmp_path / "results" / "configured"
    candidate = tmp_path / "results" / "candidate"
    write_run_config(configured)
    write_run_config(candidate)
    write_lag_csv(configured / "lag_performance.csv", [-800])
    write_lag_csv(candidate / "lag_performance.csv", [-775])
    config = {
        "results": {
            "baseline": {
                "task": {
                    "super_subject": "results/configured",
                }
            }
        }
    }

    expanded = expand_config(
        config,
        tmp_path,
        tmp_path / "results",
        expected_lags(-800, -775, 25),
        "baseline",
    )

    assert expanded["results"]["baseline"]["task"]["super_subject"] == [
        "results/configured",
        "results/candidate",
    ]


def test_expand_config_rejects_mismatched_task_or_model_candidates(tmp_path):
    configured = tmp_path / "results" / "configured"
    wrong_task = tmp_path / "results" / "wrong_task"
    wrong_model = tmp_path / "results" / "wrong_model"
    write_run_config(configured)
    write_run_config(wrong_task, task_name="task_b")
    write_run_config(wrong_model, constructor_name="other_model")
    write_lag_csv(configured / "lag_performance.csv", [-800])
    write_lag_csv(wrong_task / "lag_performance.csv", [-775])
    write_lag_csv(wrong_model / "lag_performance.csv", [-775])
    config = {
        "results": {
            "baseline": {
                "task": {
                    "super_subject": "results/configured",
                }
            }
        }
    }

    expanded = expand_config(
        config,
        tmp_path,
        tmp_path / "results",
        expected_lags(-800, -775, 25),
        "baseline",
    )

    assert expanded["results"]["baseline"]["task"]["super_subject"] == "results/configured"


def test_expand_config_adds_per_subject_and_per_region_candidates(tmp_path):
    subject_configured = tmp_path / "results" / "subject_configured"
    subject_candidate = tmp_path / "results" / "subject_candidate"
    region_configured = tmp_path / "results" / "region_configured"
    region_candidate = tmp_path / "results" / "region_candidate"
    write_run_config(subject_configured, run_mode="per_subject", subject_ids=[1, 2])
    write_run_config(subject_candidate, run_mode="per_subject", subject_ids=[1, 2])
    write_run_config(region_configured, run_mode="per_region")
    write_run_config(region_candidate, run_mode="per_region")
    for subject in ("subject_1", "subject_2"):
        write_lag_csv(subject_configured / subject / "lag_performance.csv", [-800])
        write_lag_csv(subject_candidate / subject / "lag_performance.csv", [-775])
    for region in ("region_eac", "region_pc", "region_prc", "region_ifg", "region_mtg", "region_itg", "region_tpj", "region_tp", "region_right"):
        write_lag_csv(region_configured / region / "lag_performance.csv", [-800])
        write_lag_csv(region_candidate / region / "lag_performance.csv", [-775])
    config = {
        "results": {
            "baseline": {
                "task": {
                    "per_subject": "results/subject_configured",
                    "per_region": "results/region_configured",
                }
            }
        }
    }

    expanded = expand_config(
        config,
        tmp_path,
        tmp_path / "results",
        expected_lags(-800, -775, 25),
        "baseline",
    )

    assert expanded["results"]["baseline"]["task"]["per_subject"] == [
        "results/subject_configured",
        "results/subject_candidate",
    ]
    assert expanded["results"]["baseline"]["task"]["per_region"] == [
        "results/region_configured",
        "results/region_candidate",
    ]


def test_expand_config_skips_and_reports_partial_lag_overlap(tmp_path, capsys):
    configured = tmp_path / "results" / "configured"
    candidate = tmp_path / "results" / "candidate"
    write_run_config(configured)
    write_run_config(candidate)
    write_lag_csv(configured / "lag_performance.csv", [-800])
    write_lag_csv(candidate / "lag_performance.csv", [-800, -775])
    config = {
        "results": {
            "baseline": {
                "task": {
                    "super_subject": "results/configured",
                }
            }
        }
    }

    expanded = expand_config(
        config,
        tmp_path,
        tmp_path / "results",
        expected_lags(-800, -775, 25),
        "baseline",
    )

    assert expanded["results"]["baseline"]["task"]["super_subject"] == "results/configured"
    assert "Partial overlap blocks merge" in capsys.readouterr().out


def test_expand_config_ignores_non_baseline_models(tmp_path):
    configured = tmp_path / "results" / "configured"
    candidate = tmp_path / "results" / "candidate"
    write_run_config(configured)
    write_run_config(candidate)
    write_lag_csv(configured / "lag_performance.csv", [-800])
    write_lag_csv(candidate / "lag_performance.csv", [-775])
    config = {
        "results": {
            "diver": {
                "task": {
                    "super_subject": "results/configured",
                }
            }
        }
    }

    expanded = expand_config(
        config,
        tmp_path,
        tmp_path / "results",
        expected_lags(-800, -775, 25),
        "baseline",
    )

    assert expanded == config
