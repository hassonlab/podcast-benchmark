import sys
import types

import numpy as np
import pandas as pd
import pytest
import torch

from core.config import DataParams, ModelSpec, TaskConfig, TrainingParams
import utils.decoding_utils as decoding_utils
from utils.decoding_utils import (
    train_decoding_model,
    train_ridge_logistic_regression,
    train_ridge_regression,
)


def _baseline_training_params(**overrides):
    params = TrainingParams(
        batch_size=4,
        epochs=1,
        n_folds=2,
        losses=["mse"],
        metrics=[],
        early_stopping_metric="mse",
        smaller_is_better=True,
        normalize_targets=False,
        tensorboard_logging=False,
    )
    for key, value in overrides.items():
        setattr(params, key, value)
    return params


def test_baseline_only_himalaya_ridge_populates_cv_results_without_model_build(
    monkeypatch, tmp_path
):
    def fail_model_build(*args, **kwargs):
        raise AssertionError("baseline-only mode should not build a neural model")

    monkeypatch.setattr("utils.decoding_utils.build_model_from_spec", fail_model_build)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(24, 3, 2)).astype(np.float32)
    y = X.reshape(24, -1) @ np.array([0.5, -0.2, 0.1, 0.4, -0.3, 0.2])

    models, histories, cv_results = train_decoding_model(
        torch.tensor(X),
        torch.tensor(y, dtype=torch.float32),
        pd.DataFrame({"row": range(24)}),
        ModelSpec(
            constructor_name="",
            params={
                "alphas": [0.1, 1.0],
                "cv": 2,
                "backend": "numpy",
                "force_cpu": True,
            },
        ),
        "toy_task",
        TaskConfig(),
        lag=0,
        training_params=_baseline_training_params(ridge_regression_baseline=True),
        checkpoint_dir=str(tmp_path),
    )

    assert models == []
    assert histories == []
    assert cv_results["fold_nums"] == [1, 2]
    assert cv_results["num_epochs"] == [0, 0]
    assert len(cv_results["train_mse"]) == 2
    assert len(cv_results["val_mse"]) == 2
    assert len(cv_results["test_mse"]) == 2
    assert not (tmp_path / "best_model_fold1.pt").exists()


def test_baseline_only_word_embedding_ridge_populates_embedding_metrics(
    monkeypatch, tmp_path
):
    def fail_model_build(*args, **kwargs):
        raise AssertionError("baseline-only mode should not build a neural model")

    monkeypatch.setattr("utils.decoding_utils.build_model_from_spec", fail_model_build)

    rng = np.random.default_rng(2)
    X = rng.normal(size=(24, 2, 2)).astype(np.float32)
    weights = np.array(
        [
            [0.5, -0.2],
            [0.1, 0.3],
            [-0.4, 0.2],
            [0.2, 0.1],
        ],
        dtype=np.float32,
    )
    y = X.reshape(24, -1) @ weights
    words = ["alpha", "beta", "gamma", "delta"] * 6

    _, _, cv_results = train_decoding_model(
        torch.tensor(X),
        torch.tensor(y, dtype=torch.float32),
        pd.DataFrame({"word": words}),
        ModelSpec(
            constructor_name="",
            params={
                "alphas": [0.1, 1.0],
                "cv": 2,
                "backend": "numpy",
                "force_cpu": True,
            },
        ),
        "word_embedding_decoding_task",
        TaskConfig(data_params=DataParams(word_column="word")),
        lag=0,
        training_params=_baseline_training_params(
            ridge_regression_baseline=True,
            metrics=["corr", "cosine_sim"],
            top_k_thresholds=[1, 2],
            min_train_freq_auc=1,
            min_test_freq_auc=1,
        ),
        checkpoint_dir=str(tmp_path),
    )

    assert len(cv_results["test_word_avg_auc_roc"]) == 2
    assert len(cv_results["test_occurence_top_1"]) == 2
    assert len(cv_results["test_word_top_2"]) == 2
    assert all(np.isfinite(v) for v in cv_results["test_word_avg_auc_roc"])


def test_baseline_only_ridge_logistic_uses_probabilities_and_classification_metrics(
    monkeypatch, tmp_path
):
    def fail_model_build(*args, **kwargs):
        raise AssertionError("baseline-only mode should not build a neural model")

    monkeypatch.setattr("utils.decoding_utils.build_model_from_spec", fail_model_build)

    rng = np.random.default_rng(1)
    X = rng.normal(size=(24, 2, 2)).astype(np.float32)
    score = X.reshape(24, -1)[:, 0] - 0.5 * X.reshape(24, -1)[:, 1]
    y = (score > np.median(score)).astype(np.int64)

    models, histories, cv_results = train_decoding_model(
        torch.tensor(X),
        torch.tensor(y),
        pd.DataFrame({"row": range(24)}),
        ModelSpec(
            constructor_name="",
            params={"Cs": [0.1, 1.0], "cv": 2, "solver": "lbfgs", "max_iter": 200},
        ),
        "toy_task",
        TaskConfig(),
        lag=0,
        training_params=_baseline_training_params(
            ridge_logistic_regression_baseline=True,
            losses=["cross_entropy"],
            metrics=["roc_auc"],
            early_stopping_metric="roc_auc",
            smaller_is_better=False,
        ),
        checkpoint_dir=str(tmp_path),
    )

    assert models == []
    assert histories == []
    assert len(cv_results["test_cross_entropy"]) == 2
    assert len(cv_results["test_roc_auc"]) == 2
    assert all(0.0 <= auc <= 1.0 for auc in cv_results["test_roc_auc"])


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({}, "Found 0"),
        (
            {
                "linear_regression_baseline": True,
                "ridge_regression_baseline": True,
            },
            "Found 2",
        ),
    ],
)
def test_baseline_only_requires_exactly_one_enabled_baseline(
    overrides, match, tmp_path
):
    with pytest.raises(ValueError, match=match):
        train_decoding_model(
            torch.randn(8, 2),
            torch.randn(8),
            pd.DataFrame({"row": range(8)}),
            ModelSpec(constructor_name=""),
            "toy_task",
            TaskConfig(),
            lag=0,
            training_params=_baseline_training_params(**overrides),
            checkpoint_dir=str(tmp_path),
        )


def test_ridge_regression_params_drive_himalaya_cv_and_backend(monkeypatch):
    captured = {}

    def fake_set_backend(backend, on_error="raise"):
        captured["backend"] = backend
        captured["on_error"] = on_error

    class FakeRidgeCV:
        def __init__(self, **kwargs):
            captured["ridge_kwargs"] = kwargs

        def fit(self, X, y):
            captured["X_shape"] = X.shape
            captured["X_dtype"] = X.dtype
            captured["y_dtype"] = y.dtype
            return self

    backend_module = types.ModuleType("himalaya.backend")
    backend_module.set_backend = fake_set_backend
    ridge_module = types.ModuleType("himalaya.ridge")
    ridge_module.RidgeCV = FakeRidgeCV
    monkeypatch.setitem(sys.modules, "himalaya.backend", backend_module)
    monkeypatch.setitem(sys.modules, "himalaya.ridge", ridge_module)

    train_ridge_regression(
        np.zeros((5, 2, 3)),
        np.ones(5),
        baseline_params={
            "alphas": [0.1, 1.0],
            "cv": 3,
            "backend": "numpy",
            "fit_intercept": False,
            "Y_in_cpu": True,
            "force_cpu": True,
        },
    )

    assert captured["backend"] == "numpy"
    assert captured["on_error"] == "warn"
    assert captured["ridge_kwargs"] == {
        "alphas": [0.1, 1.0],
        "cv": 3,
        "fit_intercept": False,
        "Y_in_cpu": True,
        "force_cpu": True,
    }
    assert captured["X_shape"] == (5, 6)
    assert captured["X_dtype"] == np.float32
    assert captured["y_dtype"] == np.float32


def test_ridge_logistic_params_drive_logistic_regression_cv(monkeypatch):
    captured = {}

    class FakeLogisticRegressionCV:
        def __init__(
            self,
            Cs=None,
            cv=None,
            penalty=None,
            solver=None,
            max_iter=None,
            class_weight=None,
            multi_class=None,
        ):
            captured["kwargs"] = {
                "Cs": Cs,
                "cv": cv,
                "penalty": penalty,
                "solver": solver,
                "max_iter": max_iter,
                "class_weight": class_weight,
                "multi_class": multi_class,
            }

        def fit(self, X, y):
            captured["X_shape"] = X.shape
            captured["y"] = y
            return self

    monkeypatch.setattr(
        decoding_utils, "LogisticRegressionCV", FakeLogisticRegressionCV
    )

    train_ridge_logistic_regression(
        np.zeros((6, 2, 2)),
        np.array([[0], [1], [0], [1], [0], [1]]),
        baseline_params={
            "Cs": [0.01, 0.1],
            "cv": 2,
            "solver": "liblinear",
            "max_iter": 123,
            "class_weight": "balanced",
            "multi_class": "ovr",
        },
    )

    assert captured["kwargs"] == {
        "Cs": [0.01, 0.1],
        "cv": 2,
        "penalty": "l2",
        "solver": "liblinear",
        "max_iter": 123,
        "class_weight": "balanced",
        "multi_class": "ovr",
    }
    assert captured["X_shape"] == (6, 4)
    assert captured["y"].shape == (6,)
