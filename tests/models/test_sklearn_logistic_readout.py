import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import metrics.classification_metrics  # noqa: F401 - registers the metrics used below
from core import registry
from core.config import DataParams, ModelSpec, TaskConfig, TrainingParams
from models.sklearn_logistic_readout import SklearnLogisticReadout
from utils import decoding_utils
from utils.model_utils import build_model_from_spec


def _data(n=120, d=6, n_classes=2, seed=0):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % n_classes
    centres = rng.normal(0, 1.5, (n_classes, d))
    X = centres[y] + rng.normal(0, 1, (n, d))
    # (n, C, T) like neural windows; the read-out flattens them.
    return torch.tensor(X.reshape(n, 2, d // 2), dtype=torch.float32), y


def _imindbench_reference(X, y, C=1.0):
    X = X.reshape(len(X), -1).numpy()
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(
        C=C, penalty="l2", solver="lbfgs", max_iter=10000, tol=1e-3, random_state=42
    ).fit(scaler.transform(X), y)
    return scaler, clf


def test_constructor_is_registered_and_takes_probe_params():
    model = build_model_from_spec(
        ModelSpec(
            constructor_name="sklearn_logistic_readout",
            params={"layer_sizes": [5], "dropout": 0.0, "embedding_dim": 5},
        )
    )
    assert isinstance(model, SklearnLogisticReadout)
    assert model.closed_form and model.C == 1.0 and model.output_dim == 5
    assert "sklearn_logistic_readout" in registry.model_constructor_registry
    linear = build_model_from_spec(
        ModelSpec(
            constructor_name="sklearn_logistic_readout",
            params={"output_dim": 1, "num_timepoints": 16, "activation": "sigmoid"},
        )
    )
    assert linear.output_activation == "sigmoid" and linear.output_dim == 1


def test_binary_matches_imindbench_estimator():
    X, y = _data()
    model = SklearnLogisticReadout().fit(X[:80], torch.tensor(y[:80], dtype=torch.float32))
    scaler, clf = _imindbench_reference(X[:80], y[:80])
    test = X[80:].reshape(40, -1).numpy()
    out = model(X[80:])
    assert out.shape == (40,)
    np.testing.assert_allclose(out.numpy(), clf.decision_function(scaler.transform(test)), rtol=1e-5, atol=1e-5)

    probs = SklearnLogisticReadout(output_activation="sigmoid").fit(X[:80], y[:80])(X[80:])
    np.testing.assert_allclose(probs.numpy(), clf.predict_proba(scaler.transform(test))[:, 1], rtol=1e-5, atol=1e-6)


def test_multiclass_softmax_of_output_recovers_probabilities():
    X, y = _data(n=150, n_classes=5)
    onehot = torch.nn.functional.one_hot(torch.tensor(y[:100]), 5).float()
    model = SklearnLogisticReadout(output_dim=5).fit(X[:100], onehot)
    scaler, clf = _imindbench_reference(X[:100], y[:100])
    out = model(X[100:])
    assert out.shape == (50, 5)
    expected = clf.predict_proba(scaler.transform(X[100:].reshape(50, -1).numpy()))
    np.testing.assert_allclose(torch.softmax(out, dim=1).numpy(), expected, rtol=1e-4, atol=1e-6)


def test_c_grid_picks_best_validation_score():
    X, y = _data(n=200, d=40, seed=3)
    score = registry.metric_registry["roc_auc"]
    model = SklearnLogisticReadout(C_grid=[1e-6, 1.0]).fit(
        X[:100], y[:100], X_val=X[100:150], y_val=torch.tensor(y[100:150]), score_fn=score
    )
    best = max(model.selection["val_scores"], key=model.selection["val_scores"].get)
    assert model.C == best
    with pytest.raises(ValueError):
        SklearnLogisticReadout(C_grid=[1.0]).fit(X[:100], y[:100])


@pytest.mark.parametrize("n_classes", [2, 4])
def test_train_decoding_model_fits_once_per_fold(tmp_path, n_classes):
    X, y = _data(n=200, n_classes=n_classes, seed=1)
    binary = n_classes == 2
    Y = torch.tensor(y, dtype=torch.float32) if binary else torch.tensor(y)
    params = TrainingParams(
        batch_size=32,
        epochs=50,
        n_folds=4,
        losses=["bce_with_logits" if binary else "cross_entropy"],
        metrics=["roc_auc" if binary else "roc_auc_multiclass"],
        early_stopping_metric="roc_auc" if binary else "roc_auc_multiclass",
        smaller_is_better=False,
        tensorboard_logging=False,
        normalize_targets=False,
    )
    _, histories, cv_results, _ = decoding_utils.train_decoding_model(
        X,
        Y,
        pd.DataFrame({"start": np.arange(len(y), dtype=float), "target": y}),
        ModelSpec(
            constructor_name="sklearn_logistic_readout",
            params={} if binary else {"output_dim": n_classes},
        ),
        "smoke_task",
        TaskConfig(task_name="smoke_task", data_params=DataParams()),
        0,
        params,
        str(tmp_path),
    )
    metric = params.early_stopping_metric
    assert len(cv_results[f"test_{metric}"]) == 4
    assert cv_results["num_epochs"] == [1, 1, 1, 1]
    assert all(len(h[f"val_{metric}"]) == 1 for h in histories)
    # Well-separated synthetic classes: a working read-out is far above chance.
    assert np.mean(cv_results[f"test_{metric}"]) > 0.8
