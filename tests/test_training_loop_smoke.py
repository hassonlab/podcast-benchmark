import h5py
import mne
import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from core.config import (
    ChunkedPreprocessingParams,
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
from utils import decoding_utils
from utils.decoding_utils import run_training_over_lags
from utils.dataset import RawNeuralDataset
from utils.fold_utils import get_sequential_folds
from utils.model_utils import build_model_from_spec

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
    assert not (output_dir / "test_predictions.h5").exists()


def test_saves_original_scale_out_of_fold_predictions(tmp_path):
    output_dir = tmp_path / "prediction_results"
    checkpoint_dir = tmp_path / "prediction_checkpoints"
    params = _training_params()
    params.normalize_targets = True
    params.save_test_predictions = True

    run_training_over_lags(
        [0],
        _fake_raws(),
        _task_df(),
        preprocessing_fns=None,
        model_spec=ModelSpec(
            constructor_name="smoke_flatten_regressor",
            params={"output_dim": 1},
        ),
        task_name="smoke_task",
        training_params=params,
        task_config=_task_config(DataParams(window_width=0.004)),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        write_to_tensorboard=False,
    )

    with h5py.File(output_dir / "test_predictions.h5", "r") as artifact:
        assert artifact.attrs["schema_version"] == 1
        lag_group = artifact["lag_0"]
        saved = {}
        for fold_name in ("fold_1", "fold_2"):
            fold_group = lag_group[fold_name]
            assert bool(fold_group.attrs["normalized_during_training"])
            assert fold_group["prediction"].dtype == np.float32
            assert "target_mean" in fold_group
            assert "target_std" in fold_group
            ids = fold_group["sample_id"].asstr()[:]
            targets = fold_group["target"][:]
            saved.update(zip(ids, targets.tolist()))

    expected = {
        str(index): float(target)
        for index, target in enumerate(_task_df()["target"].to_numpy())
    }
    assert saved == expected


def test_rejects_llm_prediction_artifacts(tmp_path):
    params = _training_params()
    params.save_test_predictions = True

    with pytest.raises(ValueError, match="not supported for llm_decoding_task"):
        run_training_over_lags(
            [0],
            _fake_raws(),
            _task_df(),
            preprocessing_fns=None,
            model_spec=ModelSpec(constructor_name="smoke_flatten_regressor"),
            task_name="llm_decoding_task",
            training_params=params,
            task_config=_task_config(DataParams(window_width=0.004)),
            output_dir=str(tmp_path / "results"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
        )


def test_cnn_artifact_matches_reloaded_best_checkpoints(tmp_path):
    output_dir = tmp_path / "cnn_results"
    checkpoint_dir = tmp_path / "cnn_checkpoints"
    task_df = _task_df()
    raws = _fake_raws()
    model_spec = ModelSpec(
        constructor_name="pitom_model",
        params={
            "input_channels": 4,
            "output_dim": 1,
            "conv_filters": 4,
            "dropout": 0.0,
            "output_activation": "linear",
        },
    )
    params = _training_params()
    params.epochs = 3
    params.save_test_predictions = True
    data_params = DataParams(window_width=0.02)

    run_training_over_lags(
        [0],
        raws,
        task_df,
        preprocessing_fns=None,
        model_spec=model_spec,
        task_name="smoke_task",
        training_params=params,
        task_config=_task_config(data_params),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        write_to_tensorboard=False,
    )

    neural, _, selected_df, _ = RawNeuralDataset(
        raws,
        task_df,
        data_params.window_width,
        include_sample_ids=True,
    ).get_data_for_lag(0)
    folds = get_sequential_folds(neural, num_folds=params.n_folds)

    with h5py.File(output_dir / "test_predictions.h5", "r") as artifact:
        for fold, (_, _, test_indices) in enumerate(folds, start=1):
            model = build_model_from_spec(model_spec, lag=0, fold=fold)
            decoding_utils._load_checkpoint(
                model,
                checkpoint_dir / "lag_0" / f"best_model_fold{fold}.pt",
            )
            model.eval()
            with torch.no_grad():
                expected_predictions = model(neural[test_indices]).numpy()

            fold_group = artifact[f"lag_0/fold_{fold}"]
            np.testing.assert_array_equal(
                fold_group["sample_id"].asstr()[:],
                selected_df.iloc[test_indices]["sample_id"].to_numpy(),
            )
            np.testing.assert_allclose(
                fold_group["prediction"][:],
                expected_predictions,
                rtol=1e-6,
                atol=1e-7,
            )


def test_run_training_over_lags_releases_memory_after_each_lag(
    tmp_path, monkeypatch
):
    cleanup_calls = []
    monkeypatch.setattr(
        decoding_utils,
        "_release_accelerator_memory",
        lambda: cleanup_calls.append(True),
    )

    output_dir = tmp_path / "multi_lag_results"
    checkpoint_dir = tmp_path / "multi_lag_checkpoints"
    run_training_over_lags(
        [0, 1],
        _fake_raws(),
        _task_df(),
        preprocessing_fns=None,
        model_spec=ModelSpec(
            constructor_name="smoke_flatten_regressor",
            params={"output_dim": 1},
        ),
        task_name="smoke_task",
        training_params=_training_params(),
        task_config=_task_config(DataParams(window_width=0.004)),
        output_dir=str(output_dir),
        checkpoint_dir=str(checkpoint_dir),
        write_to_tensorboard=False,
    )

    result = pd.read_csv(output_dir / "lag_performance.csv")
    assert result["lags"].tolist() == [0, 1]
    assert cleanup_calls == [True, True]


def test_run_training_over_lags_smoke_with_chunked_preprocessing(tmp_path):
    SMOKE_STATS["preprocessor_calls"] = 0

    output_dir = tmp_path / "chunked_results"
    checkpoint_dir = tmp_path / "chunked_checkpoints"
    cache_dir = tmp_path / "chunks"
    params = _training_params()
    params.save_test_predictions = True

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
        training_params=params,
        task_config=_task_config(
            DataParams(
                window_width=0.004,
                preprocessor_params=[{"scale": 0.5}],
                chunked_preprocessing=ChunkedPreprocessingParams(
                    enabled=True,
                    num_chunks=3,
                    cache_dir=str(cache_dir),
                ),
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
    assert SMOKE_STATS["preprocessor_calls"] == 3
    assert not list(cache_dir.glob("*.npz"))
    with h5py.File(output_dir / "test_predictions.h5", "r") as artifact:
        saved_ids = np.concatenate(
            [
                artifact[f"lag_0/fold_{fold}/sample_id"].asstr()[:]
                for fold in (1, 2)
            ]
        )
    assert set(saved_ids) == {str(index) for index in range(len(_task_df()))}


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


def test_repeated_combined_null_reruns_random_foundation_features(tmp_path):
    SMOKE_STATS["foundation_encode_calls"] = 0
    output_dir = tmp_path / "null_results"
    checkpoint_dir = tmp_path / "null_checkpoints"
    params = _training_params()
    params.shuffle_targets = True
    params.num_null_repetitions = 2
    params.save_test_predictions = True

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
        training_params=params,
        task_config=_task_config(
            DataParams(
                window_width=0.004,
                preprocessor_params=[
                    {
                        "foundation_model_spec": {
                            "constructor_name": "smoke_foundation_encoder",
                            "params": {},
                            "random_init": True,
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

    results = pd.read_csv(output_dir / "lag_performance.csv")
    assert results["null_repetition"].tolist() == [1, 2]
    assert results["null_seed"].tolist() == [params.random_seed, params.random_seed + 1]
    summary = pd.read_csv(output_dir / "null_summary.csv")
    assert summary.loc[0, "num_null_repetitions"] == 2
    for repetition in (1, 2):
        repetition_dir = checkpoint_dir / "lag_0" / f"null_repetition_{repetition}"
        assert (repetition_dir / "best_model_fold1.pt").exists()
        assert (repetition_dir / "best_model_fold2.pt").exists()
    with h5py.File(output_dir / "test_predictions.h5", "r") as artifact:
        assert artifact.attrs["schema_version"] == 2
        assert "lag_0/null_repetition_1/fold_1" in artifact
        assert "lag_0/null_repetition_2/fold_2" in artifact
    assert SMOKE_STATS["foundation_encode_calls"] == 6


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
