from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

import main
from core.config import (
    DataParams,
    ExperimentConfig,
    ModelSpec,
    RunMode,
    TaskConfig,
    TrainingParams,
)


class FixedDatetime:
    @classmethod
    def now(cls):
        return cls()

    def strftime(self, _fmt):
        return "2026-04-23-12-34-56"


@pytest.fixture
def base_experiment_config(tmp_path):
    return ExperimentConfig(
        model_spec=ModelSpec(constructor_name="test_model", model_data_getter="test_getter"),
        config_setter_name=["test_setter"],
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


def test_run_single_task_combined_mode_uses_single_training_call(
    monkeypatch, base_experiment_config, task_df
):
    raws = [
        SimpleNamespace(name="raw-2", ch_names=["A1", "A2"]),
        SimpleNamespace(name="raw-5", ch_names=["B1", "B2"]),
    ]
    _configure_common_mocks(monkeypatch, raws, task_df)

    setter_calls = []
    getter_calls = []
    training_calls = []

    def fake_setter(config, setter_raws, setter_task_df):
        setter_calls.append((list(config.task_config.data_params.subject_ids), setter_raws, setter_task_df))
        return config

    def fake_getter(df, getter_raws, _params):
        getter_calls.append((getter_raws, df.copy(deep=True)))
        return df.assign(extra=1), ["extra"]

    monkeypatch.setattr(
        main.registry,
        "config_setter_registry",
        {"test_setter": fake_setter},
    )
    monkeypatch.setattr(
        main.registry,
        "model_constructor_registry",
        {"test_model": {}},
    )
    monkeypatch.setattr(
        main.registry,
        "model_data_getter_registry",
        {"test_getter": fake_getter},
    )
    monkeypatch.setattr(
        main.decoding_utils,
        "run_training_over_lags",
        lambda *args, **kwargs: training_calls.append((args, kwargs)),
    )

    checkpoint_dir = main.run_single_task(base_experiment_config)

    assert checkpoint_dir.endswith("checkpoints/trial_2026-04-23-12-34-56")
    assert len(training_calls) == 1
    assert len(setter_calls) == 1
    assert len(getter_calls) == 1

    args, kwargs = training_calls[0]
    assert args[0] == [250]
    assert args[1] == raws
    assert list(kwargs["task_config"].data_params.subject_ids) == [2, 5]
    assert kwargs["task_config"].data_params.per_subject_electrodes is None
    assert kwargs["output_dir"].endswith("results/trial_2026-04-23-12-34-56")
    assert kwargs["checkpoint_dir"].endswith("checkpoints/trial_2026-04-23-12-34-56")
    assert kwargs["tensorboard_dir"].endswith("event_logs/trial_2026-04-23-12-34-56")

    config_path = (
        Path(base_experiment_config.output_dir)
        / "trial_2026-04-23-12-34-56"
        / "config.yml"
    )
    assert config_path.exists()


def test_run_single_task_per_subject_mode_splits_runs_by_subject(
    monkeypatch, base_experiment_config, task_df
):
    base_experiment_config.run_mode = RunMode.PER_SUBJECT
    raws = [
        SimpleNamespace(name="raw-2", ch_names=["A1", "A2"]),
        SimpleNamespace(name="raw-5", ch_names=["B1", "B2"]),
    ]
    _configure_common_mocks(monkeypatch, raws, task_df)

    setter_calls = []
    getter_calls = []
    training_calls = []

    def fake_setter(config, setter_raws, _task_df):
        setter_calls.append((list(config.task_config.data_params.subject_ids), setter_raws))
        return config

    def fake_getter(df, getter_raws, _params):
        getter_calls.append((getter_raws, list(df.columns)))
        return df.assign(extra=1), ["extra"]

    monkeypatch.setattr(main.registry, "config_setter_registry", {"test_setter": fake_setter})
    monkeypatch.setattr(main.registry, "model_constructor_registry", {"test_model": {}})
    monkeypatch.setattr(main.registry, "model_data_getter_registry", {"test_getter": fake_getter})
    monkeypatch.setattr(
        main.decoding_utils,
        "run_training_over_lags",
        lambda *args, **kwargs: training_calls.append((args, kwargs)),
    )

    checkpoint_dir = main.run_single_task(base_experiment_config)

    assert checkpoint_dir.endswith("checkpoints/trial_2026-04-23-12-34-56")
    assert len(training_calls) == 2
    assert [call[0] for call in setter_calls] == [[2], [5]]
    assert [len(call[0]) for call in getter_calls] == [1, 1]

    expected = [
        (2, raws[0]),
        (5, raws[1]),
    ]
    for (subject_id, raw), (args, kwargs) in zip(expected, training_calls):
        assert args[0] == [250]
        assert args[1] == [raw]
        assert list(kwargs["task_config"].data_params.subject_ids) == [subject_id]
        assert kwargs["output_dir"].endswith(f"results/trial_2026-04-23-12-34-56/subject_{subject_id}")
        assert kwargs["checkpoint_dir"].endswith(
            f"checkpoints/trial_2026-04-23-12-34-56/subject_{subject_id}"
        )
        assert kwargs["tensorboard_dir"].endswith(
            f"event_logs/trial_2026-04-23-12-34-56/subject_{subject_id}"
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


def test_run_single_task_per_region_mode_splits_runs_by_region(
    monkeypatch, base_experiment_config, task_df
):
    base_experiment_config.run_mode = RunMode.PER_REGION
    raws = [
        SimpleNamespace(name="raw-2", ch_names=["A1", "A2", "X9"]),
        SimpleNamespace(name="raw-5", ch_names=["B1"]),
    ]
    _configure_common_mocks(monkeypatch, raws, task_df)

    setter_calls = []
    getter_calls = []
    training_calls = []

    monkeypatch.setattr(
        main,
        "build_electrode_region_map",
        lambda **_kwargs: {
            "Temporal Pole": {2: ["A1"], 5: ["B1"]},
            "Frontal/Opercular": {2: ["A2"]},
        },
    )

    def fake_setter(config, setter_raws, _task_df):
        setter_calls.append(
            (
                list(config.task_config.data_params.subject_ids),
                config.task_config.data_params.per_subject_electrodes,
                setter_raws,
            )
        )
        return config

    def fake_getter(df, getter_raws, _params):
        getter_calls.append(getter_raws)
        return df.assign(extra=1), ["extra"]

    monkeypatch.setattr(main.registry, "config_setter_registry", {"test_setter": fake_setter})
    monkeypatch.setattr(main.registry, "model_constructor_registry", {"test_model": {}})
    monkeypatch.setattr(main.registry, "model_data_getter_registry", {"test_getter": fake_getter})
    monkeypatch.setattr(
        main.decoding_utils,
        "run_training_over_lags",
        lambda *args, **kwargs: training_calls.append((args, kwargs)),
    )

    checkpoint_dir = main.run_single_task(base_experiment_config)

    assert checkpoint_dir.endswith("checkpoints/trial_2026-04-23-12-34-56")
    assert len(training_calls) == 2
    assert len(setter_calls) == 2
    assert len(getter_calls) == 2

    first_args, first_kwargs = training_calls[0]
    assert first_kwargs["output_dir"].endswith("results/trial_2026-04-23-12-34-56/region_temporal_pole")
    assert list(first_kwargs["task_config"].data_params.subject_ids) == [2, 5]
    assert first_kwargs["task_config"].data_params.per_subject_electrodes == {
        2: ["A1"],
        5: ["B1"],
    }
    assert first_args[1] == raws

    second_args, second_kwargs = training_calls[1]
    assert second_kwargs["output_dir"].endswith(
        "results/trial_2026-04-23-12-34-56/region_frontal_opercular"
    )
    assert list(second_kwargs["task_config"].data_params.subject_ids) == [2]
    assert second_kwargs["task_config"].data_params.per_subject_electrodes == {
        2: ["A2"]
    }
    assert second_args[1] == [raws[0]]


def test_run_single_task_per_region_mode_filters_regions(
    monkeypatch, base_experiment_config, task_df
):
    base_experiment_config.run_mode = RunMode.PER_REGION
    base_experiment_config.regions = ["Frontal/Opercular"]
    raws = [
        SimpleNamespace(name="raw-2", ch_names=["A1", "A2", "X9"]),
        SimpleNamespace(name="raw-5", ch_names=["B1"]),
    ]
    _configure_common_mocks(monkeypatch, raws, task_df)

    training_calls = []

    monkeypatch.setattr(
        main,
        "build_electrode_region_map",
        lambda **_kwargs: {
            "Temporal Pole": {2: ["A1"], 5: ["B1"]},
            "Frontal/Opercular": {2: ["A2"]},
        },
    )

    monkeypatch.setattr(main.registry, "config_setter_registry", {"test_setter": lambda config, *_args: config})
    monkeypatch.setattr(main.registry, "model_constructor_registry", {"test_model": {}})
    monkeypatch.setattr(
        main.registry,
        "model_data_getter_registry",
        {"test_getter": lambda df, _raws, _params: (df.assign(extra=1), ["extra"])},
    )
    monkeypatch.setattr(
        main.decoding_utils,
        "run_training_over_lags",
        lambda *args, **kwargs: training_calls.append((args, kwargs)),
    )

    main.run_single_task(base_experiment_config)

    assert len(training_calls) == 1
    args, kwargs = training_calls[0]
    assert kwargs["output_dir"].endswith(
        "results/trial_2026-04-23-12-34-56/region_frontal_opercular"
    )
    assert list(kwargs["task_config"].data_params.subject_ids) == [2]
    assert kwargs["task_config"].data_params.per_subject_electrodes == {2: ["A2"]}
    assert args[1] == [raws[0]]


def test_run_single_task_per_region_mode_rejects_unknown_region(
    monkeypatch, base_experiment_config, task_df
):
    base_experiment_config.run_mode = RunMode.PER_REGION
    base_experiment_config.regions = ["Missing"]
    raws = [
        SimpleNamespace(name="raw-2", ch_names=["A1"]),
        SimpleNamespace(name="raw-5", ch_names=["B1"]),
    ]
    _configure_common_mocks(monkeypatch, raws, task_df)

    monkeypatch.setattr(
        main,
        "build_electrode_region_map",
        lambda **_kwargs: {
            "Temporal Pole": {2: ["A1"]},
        },
    )
    monkeypatch.setattr(
        main.decoding_utils,
        "run_training_over_lags",
        lambda *args, **kwargs: pytest.fail("training should not run"),
    )

    with pytest.raises(ValueError, match="Unknown regions requested"):
        main.run_single_task(base_experiment_config)


def test_run_single_task_rejects_split_concat_mode(
    monkeypatch, base_experiment_config, task_df
):
    base_experiment_config.run_mode = RunMode.PER_SUBJECT
    base_experiment_config.model_spec.per_subject_feature_concat = True
    raws = [
        SimpleNamespace(name="raw-2", ch_names=["A1"]),
        SimpleNamespace(name="raw-5", ch_names=["B1"]),
    ]
    _configure_common_mocks(monkeypatch, raws, task_df)

    monkeypatch.setattr(
        main.decoding_utils,
        "run_training_over_lags",
        lambda *args, **kwargs: pytest.fail("training should not run"),
    )

    with pytest.raises(
        ValueError, match="Split run modes only support non-concat runs"
    ):
        main.run_single_task(base_experiment_config)
