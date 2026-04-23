from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import main
from core.config import ExperimentConfig, ModelSpec, TaskConfig, DataParams, TrainingParams


class FixedDatetime:
    @classmethod
    def now(cls):
        return cls()

    def strftime(self, _fmt):
        return "2026-04-23-12-34-56"


@pytest.fixture
def base_experiment_config(tmp_path):
    return ExperimentConfig(
        model_spec=ModelSpec(constructor_name="test_model"),
        task_config=TaskConfig(
            task_name="test_task",
            data_params=DataParams(subject_ids=[2, 5]),
        ),
        training_params=TrainingParams(lag=250, tensorboard_logging=True),
        trial_name="trial",
        output_dir=str(tmp_path / "results"),
        checkpoint_dir=str(tmp_path / "checkpoints"),
        tensorboard_dir=str(tmp_path / "event_logs"),
    )


@pytest.fixture
def task_df():
    return pd.DataFrame({"value": [1, 2, 3]})


def _configure_common_mocks(monkeypatch, raws, task_df):
    monkeypatch.setattr(main, "datetime", FixedDatetime)
    monkeypatch.setattr(
        main.registry,
        "task_registry",
        {"test_task": {"getter": lambda _task_config: task_df}},
    )
    monkeypatch.setattr(main.data_utils, "load_raws", lambda _data_params: raws)


def test_run_single_task_default_mode_uses_single_training_call(
    monkeypatch, base_experiment_config, task_df
):
    raws = [SimpleNamespace(name="raw-2"), SimpleNamespace(name="raw-5")]
    _configure_common_mocks(monkeypatch, raws, task_df)

    calls = []

    def fake_run_training_over_lags(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        main.decoding_utils, "run_training_over_lags", fake_run_training_over_lags
    )

    checkpoint_dir = main.run_single_task(base_experiment_config)

    assert checkpoint_dir.endswith("checkpoints/trial_2026-04-23-12-34-56")
    assert len(calls) == 1

    args, kwargs = calls[0]
    assert args[0] == [250]
    assert args[1] == raws
    assert args[2] is task_df
    assert kwargs["output_dir"].endswith("results/trial_2026-04-23-12-34-56")
    assert kwargs["checkpoint_dir"].endswith(
        "checkpoints/trial_2026-04-23-12-34-56"
    )
    assert kwargs["tensorboard_dir"].endswith(
        "event_logs/trial_2026-04-23-12-34-56"
    )

    config_path = (
        Path(base_experiment_config.output_dir)
        / "trial_2026-04-23-12-34-56"
        / "config.yml"
    )
    assert config_path.exists()


def test_run_single_task_per_subject_mode_splits_runs_by_subject(
    monkeypatch, base_experiment_config, task_df
):
    base_experiment_config.train_one_subject_at_a_time = True
    raws = [SimpleNamespace(name="raw-2"), SimpleNamespace(name="raw-5")]
    _configure_common_mocks(monkeypatch, raws, task_df)

    calls = []

    def fake_run_training_over_lags(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(
        main.decoding_utils, "run_training_over_lags", fake_run_training_over_lags
    )

    checkpoint_dir = main.run_single_task(base_experiment_config)

    assert checkpoint_dir.endswith("checkpoints/trial_2026-04-23-12-34-56")
    assert len(calls) == 2

    parent_output = "results/trial_2026-04-23-12-34-56"
    parent_checkpoint = "checkpoints/trial_2026-04-23-12-34-56"
    parent_tensorboard = "event_logs/trial_2026-04-23-12-34-56"

    expected = [
        (2, raws[0]),
        (5, raws[1]),
    ]
    for (subject_id, raw), (args, kwargs) in zip(expected, calls):
        assert args[0] == [250]
        assert args[1] == [raw]
        assert args[2] is task_df
        assert kwargs["output_dir"].endswith(f"{parent_output}/subject_{subject_id}")
        assert kwargs["checkpoint_dir"].endswith(
            f"{parent_checkpoint}/subject_{subject_id}"
        )
        assert kwargs["tensorboard_dir"].endswith(
            f"{parent_tensorboard}/subject_{subject_id}"
        )

    config_path = (
        Path(base_experiment_config.output_dir)
        / "trial_2026-04-23-12-34-56"
        / "config.yml"
    )
    assert config_path.exists()
    assert not (
        Path(base_experiment_config.output_dir)
        / "trial_2026-04-23-12-34-56"
        / "subject_2"
        / "config.yml"
    ).exists()


def test_run_single_task_rejects_per_subject_concat_mode(
    monkeypatch, base_experiment_config, task_df
):
    base_experiment_config.train_one_subject_at_a_time = True
    base_experiment_config.model_spec.per_subject_feature_concat = True
    raws = [SimpleNamespace(name="raw-2"), SimpleNamespace(name="raw-5")]
    _configure_common_mocks(monkeypatch, raws, task_df)

    monkeypatch.setattr(
        main.decoding_utils,
        "run_training_over_lags",
        lambda *args, **kwargs: pytest.fail("training should not run"),
    )

    with pytest.raises(
        ValueError, match="train_one_subject_at_a_time only supports non-concat runs"
    ):
        main.run_single_task(base_experiment_config)
