from pathlib import Path

import pandas as pd
import pytest
import yaml

from core.config import RunMode
from scripts.clean_paper_result_config import clean_config


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def write_config(
    run_dir: Path,
    *,
    lags=(-100, 100, 100),
    subject_ids=(1,),
    run_mode=RunMode.COMBINED,
    hidden=4,
) -> None:
    config = {
        "model_spec": {"constructor_name": "test_model", "params": {"hidden": hidden}},
        "task_config": {
            "task_name": "test_task",
            "data_params": {
                "subject_ids": list(subject_ids),
                "per_subject_electrodes": {
                    subject_id: [f"G{subject_id}"] for subject_id in subject_ids
                },
            },
            "task_specific_config": {},
        },
        "training_params": {
            "lag": None,
            "min_lag": lags[0],
            "max_lag": lags[1],
            "lag_step_size": lags[2],
            "epochs": 2,
        },
        "run_mode": run_mode,
        "regions": None,
        "trial_name": f"run_subject_{subject_ids[0]}",
        "format_fields": None,
        "output_dir": "results",
        "checkpoint_dir": "checkpoints",
        "tensorboard_dir": "event_logs",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "config.yml").open("w") as f:
        # Match the Python-tagged RunMode emitted by main.py's historical yaml.dump.
        yaml.dump(config, f, sort_keys=False)


def result_index(condition: str, paths: list[Path]) -> dict:
    return {
        "results": {
            "model": {
                "task": {
                    condition: [str(path) for path in paths],
                }
            }
        }
    }


def test_cleans_duplicate_lags_and_writes_regular_runnable_config(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_config(first, lags=(-100, 100, 100), subject_ids=(1, 2))
    write_config(second, lags=(0, 200, 100), subject_ids=(1, 2))
    write_csv(first / "lag_performance.csv", {"lags": [-100, 0], "score": [0.1, 0.2]})
    write_csv(second / "lag_performance.csv", {"lags": [0, 100], "score": [0.9, 1.0]})

    output_root = tmp_path / "cleaned"
    clean_config(
        result_index("super_subject", [first, second]), output_root, tmp_path, False
    )

    output_dir = output_root / "model" / "task" / "super_subject"
    result = pd.read_csv(output_dir / "lag_performance.csv")
    assert result.to_dict("records") == [
        {"lags": -100, "score": 0.1},
        {"lags": 0, "score": 0.9},
        {"lags": 100, "score": 1.0},
    ]
    with (output_dir / "config.yml").open() as f:
        config = yaml.safe_load(f)
    assert config["run_mode"] == "combined"
    assert config["training_params"]["lag"] is None
    assert config["training_params"]["min_lag"] == -100
    assert config["training_params"]["max_lag"] == 200
    assert config["training_params"]["lag_step_size"] == 100
    assert "!!python" not in (output_dir / "config.yml").read_text()


def test_combines_duplicate_direct_per_subject_shards_and_unions_subjects(tmp_path):
    subject_1_first = tmp_path / "model_subject_1_first"
    subject_1_second = tmp_path / "model_subject_1_second"
    subject_2 = tmp_path / "model_subject_2"
    for run_dir, subject_ids in (
        (subject_1_first, (1,)),
        (subject_1_second, (1,)),
        (subject_2, (2,)),
    ):
        write_config(run_dir, subject_ids=subject_ids)
    write_csv(
        subject_1_first / "lag_performance.csv",
        {"lags": [-100, 0], "score": [0.1, 0.2]},
    )
    write_csv(subject_1_second / "lag_performance.csv", {"lags": [0], "score": [0.8]})
    write_csv(
        subject_2 / "lag_performance.csv", {"lags": [-100, 0], "score": [0.3, 0.4]}
    )

    output_root = tmp_path / "cleaned"
    clean_config(
        result_index("per_subject", [subject_1_first, subject_1_second, subject_2]),
        output_root,
        tmp_path,
        False,
    )

    output_dir = output_root / "model" / "task" / "per_subject"
    subject_1 = pd.read_csv(output_dir / "subject_1" / "lag_performance.csv")
    assert subject_1["score"].tolist() == [0.1, 0.8]
    config = yaml.safe_load((output_dir / "config.yml").read_text())
    assert config["run_mode"] == "per_subject"
    assert config["task_config"]["data_params"]["subject_ids"] == [1, 2]
    assert config["task_config"]["data_params"]["per_subject_electrodes"] == {
        1: ["G1"],
        2: ["G2"],
    }


def test_writes_per_region_scope_and_single_lag_config(tmp_path):
    run_dir = tmp_path / "regions"
    write_config(run_dir, lags=(0, 1, 1), subject_ids=(1, 2))
    write_csv(
        run_dir / "region_eac" / "lag_performance.csv", {"lags": [0], "score": [0.5]}
    )
    write_csv(
        run_dir / "region_right" / "lag_performance.csv", {"lags": [0], "score": [0.6]}
    )

    output_root = tmp_path / "cleaned"
    clean_config(result_index("per_region", [run_dir]), output_root, tmp_path, False)

    config_path = output_root / "model" / "task" / "per_region" / "config.yml"
    config = yaml.safe_load(config_path.read_text())
    assert config["run_mode"] == "per_region"
    assert config["regions"] == ["EAC", "RIGHT"]
    assert config["training_params"]["lag"] == 0


def test_rejects_irregular_lags_before_replacing_existing_output(tmp_path):
    run_dir = tmp_path / "run"
    write_config(run_dir)
    write_csv(
        run_dir / "lag_performance.csv",
        {"lags": [-100, 0, 200], "score": [0.1, 0.2, 0.3]},
    )
    output_dir = tmp_path / "cleaned" / "model" / "task" / "super_subject"
    output_dir.mkdir(parents=True)
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("existing")

    with pytest.raises(ValueError, match="not one regular range"):
        clean_config(
            result_index("super_subject", [run_dir]),
            tmp_path / "cleaned",
            tmp_path,
            False,
        )

    assert sentinel.read_text() == "existing"


def test_rejects_conflicting_non_scope_config_settings(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_config(first, hidden=4)
    write_config(second, hidden=8)
    write_csv(first / "lag_performance.csv", {"lags": [-100], "score": [0.1]})
    write_csv(second / "lag_performance.csv", {"lags": [0], "score": [0.2]})

    with pytest.raises(ValueError, match="conflicts with other non-scope settings"):
        clean_config(
            result_index("super_subject", [first, second]),
            tmp_path / "cleaned",
            tmp_path,
            False,
        )


def test_allows_non_scientific_saved_config_differences(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_config(first, lags=(-100, 0, 100))
    write_config(second, lags=(0, 100, 100))

    first_config = yaml.unsafe_load((first / "config.yml").read_text())
    first_config.pop("atlas_path", None)
    first_config["training_params"]["tensorboard_logging"] = True
    first_config["task_config"]["data_params"]["chunked_preprocessing"] = {
        "enabled": True,
        "num_chunks": 2,
        "cache_dir": "/tmp/job-one",
    }
    with (first / "config.yml").open("w") as f:
        yaml.dump(first_config, f, sort_keys=False)

    second_config = yaml.unsafe_load((second / "config.yml").read_text())
    second_config["atlas_path"] = None
    second_config["training_params"]["tensorboard_logging"] = False
    second_config["task_config"]["data_params"]["chunked_preprocessing"] = {
        "enabled": True,
        "num_chunks": 2,
        "cache_dir": "/tmp/job-two",
    }
    with (second / "config.yml").open("w") as f:
        yaml.dump(second_config, f, sort_keys=False)

    write_csv(first / "lag_performance.csv", {"lags": [-100], "score": [0.1]})
    write_csv(second / "lag_performance.csv", {"lags": [0], "score": [0.2]})

    clean_config(
        result_index("super_subject", [first, second]),
        tmp_path / "cleaned",
        tmp_path,
        False,
    )


def test_applies_configured_lag_bounds_before_writing_and_validation(tmp_path):
    run_dir = tmp_path / "subjects"
    write_config(run_dir, lags=(-200, 300, 100), subject_ids=(1, 2))
    for subject in (1, 2):
        write_csv(
            run_dir / f"subject_{subject}" / "lag_performance.csv",
            {"lags": [-200, -100, 0, 100, 200], "score": range(5)},
        )
    config = result_index("per_subject", [run_dir])
    config["cleaning"] = {
        "lag_bounds": {"model": {"task": {"min": -100, "max": 100}}}
    }

    output_root = tmp_path / "cleaned"
    clean_config(config, output_root, tmp_path, False)

    output_dir = output_root / "model" / "task" / "per_subject"
    result = pd.read_csv(output_dir / "subject_1" / "lag_performance.csv")
    assert result["lags"].tolist() == [-100, 0, 100]
    cleaned_config = yaml.safe_load((output_dir / "config.yml").read_text())
    assert cleaned_config["training_params"]["min_lag"] == -100
    assert cleaned_config["training_params"]["max_lag"] == 200


def test_rejects_missing_source_config(tmp_path):
    run_dir = tmp_path / "run"
    write_csv(run_dir / "lag_performance.csv", {"lags": [0], "score": [0.2]})

    with pytest.raises(FileNotFoundError, match="config.yml"):
        clean_config(
            result_index("super_subject", [run_dir]),
            tmp_path / "cleaned",
            tmp_path,
            False,
        )


def test_rejects_entity_lag_coverage_that_one_config_cannot_reproduce(tmp_path):
    run_dir = tmp_path / "subjects"
    write_config(run_dir, run_mode=RunMode.PER_SUBJECT, subject_ids=(1, 2))
    write_csv(
        run_dir / "subject_1" / "lag_performance.csv",
        {"lags": [-100, 0], "score": [0.1, 0.2]},
    )
    write_csv(
        run_dir / "subject_2" / "lag_performance.csv",
        {"lags": [0], "score": [0.3]},
    )

    with pytest.raises(ValueError, match="do not have the same lag coverage"):
        clean_config(
            result_index("per_subject", [run_dir]),
            tmp_path / "cleaned",
            tmp_path,
            False,
        )
