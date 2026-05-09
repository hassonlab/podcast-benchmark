import mne
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.config import (
    DataParams,
    ExperimentConfig,
    ModelSpec,
    RunMode,
    TaskConfig,
    TrainingParams,
)
from core import registry
from core.registry import register_data_preprocessor, register_model_constructor
import main
import metrics.regression_metrics  # noqa: F401 - registers mse
import models.foundation_feature_preprocessor  # noqa: F401 - registers preprocessor
import models.shared_preprocessors  # noqa: F401 - registers disk cache preprocessor
import models.shared_decoders  # noqa: F401 - registers shared probe
from utils import data_utils
from utils.decoding_utils import run_training_over_lags

SMOKE_STATS = {"preprocessor_calls": 0, "foundation_encode_calls": 0}


class SmokeFlattenRegressor(nn.Module):
    def __init__(self, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.linear = nn.LazyLinear(output_dim)

    def forward(self, x, **kwargs):
        out = self.linear(x.reshape(x.shape[0], -1))
        return out.squeeze(-1)


@register_model_constructor("smoke_flatten_regressor")
def build_smoke_flatten_regressor(model_params):
    return SmokeFlattenRegressor(output_dim=model_params.get("output_dim", 1))


class SmokeFoundationEncoder(nn.Module):
    def encode_features(self, x, **kwargs):
        SMOKE_STATS["foundation_encode_calls"] += 1
        return x.reshape(x.shape[0], x.shape[1], -1).mean(dim=-1)


@register_model_constructor("smoke_foundation_encoder")
def build_smoke_foundation_encoder(model_params):
    return SmokeFoundationEncoder()


@register_data_preprocessor("smoke_scale_preprocessor")
def smoke_scale_preprocessor(data, params, selected_rows=None, **kwargs):
    SMOKE_STATS["preprocessor_calls"] += 1
    assert selected_rows is not None
    return data * params.get("scale", 1.0)


def _fake_raws():
    sfreq = 1000
    n_samples = 1200
    raws = []
    for subject_idx, prefix in enumerate(("A", "B")):
        info = mne.create_info(
            ch_names=[f"{prefix}1", f"{prefix}2"],
            sfreq=sfreq,
            ch_types="seeg",
        )
        base = np.arange(2 * n_samples, dtype=np.float32).reshape(2, n_samples)
        data = base / 1000.0 + subject_idx
        raws.append(mne.io.RawArray(data, info, verbose=False))
    return raws


def _task_df():
    starts = np.linspace(0.10, 0.80, 8, dtype=np.float32)
    return pd.DataFrame(
        {
            "start": starts,
            "word": [f"word_{idx}" for idx in range(len(starts))],
            "target": np.linspace(-1.0, 1.0, len(starts), dtype=np.float32),
        }
    )


def _training_params():
    return TrainingParams(
        batch_size=2,
        epochs=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        early_stopping_patience=2,
        n_folds=2,
        losses=["mse"],
        metrics=[],
        early_stopping_metric="mse",
        smaller_is_better=True,
        tensorboard_logging=False,
        normalize_targets=False,
    )


def _single_lag_training_params():
    params = _training_params()
    params.lag = 0
    return params


def _task_config(data_params):
    return TaskConfig(
        task_name="smoke_task",
        data_params=data_params,
    )


def test_run_training_over_lags_smoke_with_preprocessor(tmp_path):
    SMOKE_STATS["preprocessor_calls"] = 0

    output_dir = tmp_path / "results"
    checkpoint_dir = tmp_path / "checkpoints"

    run_training_over_lags(
        [0],
        _fake_raws(),
        _task_df(),
        preprocessing_fns=[smoke_scale_preprocessor],
        model_spec=ModelSpec(
            constructor_name="smoke_flatten_regressor",
            params={"output_dim": 1},
        ),
        task_name="smoke_task",
        training_params=_training_params(),
        task_config=_task_config(
            DataParams(
                window_width=0.004,
                preprocessor_params=[{"scale": 0.5}],
            )
        ),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        write_to_tensorboard=False,
    )

    result_csv = output_dir / "lag_performance.csv"
    assert result_csv.exists()
    result = pd.read_csv(result_csv)
    assert result["lags"].tolist() == [0]
    assert np.isfinite(result.loc[0, "test_mse_mean"])
    assert (checkpoint_dir / "lag_0" / "best_model_fold1.pt").exists()
    assert (checkpoint_dir / "lag_0" / "best_model_fold2.pt").exists()
    assert SMOKE_STATS["preprocessor_calls"] == 1


def test_run_training_over_lags_smoke_with_foundation_feature_cache(tmp_path):
    SMOKE_STATS["foundation_encode_calls"] = 0

    output_dir = tmp_path / "cache_results"
    checkpoint_dir = tmp_path / "cache_checkpoints"

    run_training_over_lags(
        [0],
        _fake_raws(),
        _task_df(),
        preprocessing_fns=[
            models.foundation_feature_preprocessor.foundation_feature_cache
        ],
        model_spec=ModelSpec(
            constructor_name="mlp_probe_decoder",
            params={"layer_sizes": [1], "output_dim": 1},
        ),
        task_name="smoke_task",
        training_params=_training_params(),
        task_config=_task_config(
            DataParams(
                window_width=0.004,
                preprocessor_params=[
                    {
                        "foundation_model_spec": {
                            "constructor_name": "smoke_foundation_encoder",
                            "params": {},
                        },
                        "batch_size": 3,
                    }
                ],
            )
        ),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        write_to_tensorboard=False,
    )

    result = pd.read_csv(output_dir / "lag_performance.csv")
    assert result["lags"].tolist() == [0]
    assert np.isfinite(result.loc[0, "test_mse_mean"])
    assert (checkpoint_dir / "lag_0" / "best_model_fold1.pt").exists()
    assert (checkpoint_dir / "lag_0" / "best_model_fold2.pt").exists()
    assert SMOKE_STATS["foundation_encode_calls"] == 3


def test_run_training_over_lags_smoke_with_per_subject_feature_concat(tmp_path):
    SMOKE_STATS["foundation_encode_calls"] = 0

    output_dir = tmp_path / "concat_results"
    checkpoint_dir = tmp_path / "concat_checkpoints"

    run_training_over_lags(
        [0],
        _fake_raws(),
        _task_df(),
        preprocessing_fns=[
            models.foundation_feature_preprocessor.foundation_feature_cache
        ],
        model_spec=ModelSpec(
            constructor_name="mlp_probe_decoder",
            params={"layer_sizes": [1], "output_dim": 1},
        ),
        task_name="smoke_task",
        training_params=_training_params(),
        task_config=_task_config(
            DataParams(
                window_width=0.004,
                preprocessor_params=[
                    {
                        "foundation_model_spec": {
                            "constructor_name": "smoke_foundation_encoder",
                            "params": {},
                        },
                        "mode": "per_subject_feature_concat",
                        "batch_size": 3,
                    }
                ],
            )
        ),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        write_to_tensorboard=False,
    )

    result = pd.read_csv(output_dir / "lag_performance.csv")
    assert result["lags"].tolist() == [0]
    assert np.isfinite(result.loc[0, "test_mse_mean"])
    assert (checkpoint_dir / "lag_0" / "best_model_fold1.pt").exists()
    assert (checkpoint_dir / "lag_0" / "best_model_fold2.pt").exists()
    assert SMOKE_STATS["foundation_encode_calls"] == 6


def test_run_single_task_smoke_with_disk_cache_preprocessor(tmp_path, monkeypatch):
    SMOKE_STATS["preprocessor_calls"] = 0
    cache_dir = tmp_path / "preprocessor_cache"

    monkeypatch.setitem(
        registry.task_registry,
        "smoke_disk_cache_task",
        {
            "getter": lambda task_config: _task_df(),
            "config_type": TaskConfig,
        },
    )
    monkeypatch.setattr(data_utils, "load_raws", lambda data_params: _fake_raws())

    def build_config(trial_name):
        return ExperimentConfig(
            model_spec=ModelSpec(
                constructor_name="smoke_flatten_regressor",
                params={"output_dim": 1},
            ),
            task_config=TaskConfig(
                task_name="smoke_disk_cache_task",
                data_params=DataParams(
                    window_width=0.004,
                    preprocessing_fn_name="disk_cache_preprocessor",
                    preprocessor_params={
                        "base_preprocessing_fn_name": "smoke_scale_preprocessor",
                        "base_preprocessor_params": {"scale": 0.5},
                        "cache_dir": str(cache_dir),
                    },
                    subject_ids=[1, 2],
                ),
            ),
            training_params=_single_lag_training_params(),
            run_mode=RunMode.COMBINED,
            trial_name=trial_name,
            output_dir=str(tmp_path / "results"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            tensorboard_dir=str(tmp_path / "tensorboard"),
        )

    main.run_single_task(build_config("disk_cache_first"))
    main.run_single_task(build_config("disk_cache_second"))

    first_result_files = list(
        (tmp_path / "results").glob("disk_cache_first_*/lag_performance.csv")
    )
    second_result_files = list(
        (tmp_path / "results").glob("disk_cache_second_*/lag_performance.csv")
    )
    assert len(first_result_files) == 1
    assert len(second_result_files) == 1

    for result_file in first_result_files + second_result_files:
        result = pd.read_csv(result_file)
        assert result["lags"].tolist() == [0]
        assert np.isfinite(result.loc[0, "test_mse_mean"])

    assert (cache_dir).exists()
    assert len(list(cache_dir.glob("*.npz"))) == 1
    assert (checkpoint_dir := tmp_path / "checkpoints").exists()
    assert list(checkpoint_dir.glob("disk_cache_first_*/lag_0/best_model_fold1.pt"))
    assert list(checkpoint_dir.glob("disk_cache_second_*/lag_0/best_model_fold1.pt"))
    assert SMOKE_STATS["preprocessor_calls"] == 1
