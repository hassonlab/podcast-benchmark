import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

import metrics.classification_metrics  # noqa: F401 - registers bce / roc_auc
from core.config import TrainingParams
from utils import decoding_utils


class _FixedScores(nn.Module):
    """Returns the first input feature as the score, so the scores are known in advance."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        return torch.sigmoid(x[:, 0])


def _loader(scores, y, batch_size):
    X = torch.tensor(scores, dtype=torch.float32)[:, None]
    Y = torch.tensor(y, dtype=torch.float32)
    return [(X[i : i + batch_size], {}, Y[i : i + batch_size]) for i in range(0, len(Y), batch_size)]


def _params(full_split):
    return TrainingParams(
        batch_size=4,
        losses=["bce"],
        metrics=["roc_auc"],
        early_stopping_metric="roc_auc",
        eval_metrics_over_full_split=full_split,
    )


def _evaluate(scores, y, full_split):
    params = _params(full_split)
    all_fns = decoding_utils.setup_metrics_and_loss(params)
    return decoding_utils._run_epoch(
        _FixedScores(), _loader(scores, y, 4), torch.device("cpu"), params, all_fns, all_fns.keys(), {}
    )


def test_full_split_roc_auc_equals_sklearn_on_the_whole_split():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    scores = rng.normal(y * 1.0, 1.0)
    full = _evaluate(scores, y, full_split=True)
    assert np.isclose(full["roc_auc"], roc_auc_score(y, scores))
    # The default per-batch average is pulled towards 0.5 at batch size 4.
    per_batch = _evaluate(scores, y, full_split=False)
    assert per_batch["roc_auc"] < full["roc_auc"] - 0.01


def test_single_class_batches_no_longer_score_half():
    # Sorted labels: every batch of 4 holds one class, so the per-batch average is exactly 0.5.
    y = np.r_[np.zeros(8), np.ones(8)]
    scores = np.r_[np.linspace(-2, -1, 8), np.linspace(1, 2, 8)]
    assert _evaluate(scores, y, full_split=False)["roc_auc"] == 0.5
    assert _evaluate(scores, y, full_split=True)["roc_auc"] == 1.0


def test_default_is_unchanged():
    assert TrainingParams().eval_metrics_over_full_split is False
