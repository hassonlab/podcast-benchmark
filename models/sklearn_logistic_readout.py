"""Closed-form logistic-regression read-out (`sklearn_logistic_readout`).

A drop-in replacement for a trained probe head (e.g. `mlp_probe_decoder`) on classification tasks: swap
`model_spec.constructor_name` and keep the rest of the config. Instead of the epoch loop with early stopping, the
training loop calls `fit` once per fold (see `decoding_utils._fit_closed_form_fold`), then scores the train, val
and test splits with the same `_run_epoch` and metric functions as a trained head. So folds, test rows and metric
code are identical, and only the read-out differs.

The estimator is iMindBench's (`neuroprobe_eval/models/logistic_model.py`): scikit-learn
`LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=10000, tol=1e-3, random_state=42)` on the
flattened input. The features are standardised with the TRAINING split's mean and SD (`StandardScaler`, ddof=0),
as iMindBench standardises before its logistic regression.

C is the inverse L2 regularisation strength (smaller C, stronger penalty). By default it is fixed, as in
iMindBench, and the validation split is not used for fitting. With `C_grid`, one model per C is fitted on the
training split, and the C with the best validation score on `training_params.early_stopping_metric` is kept.
There is no refit on train+val, so a model always sees the same training rows as a trained head.

Output, matching the probe head's `output_activation`: binary tasks give one score per row, the P(class 1) with
`sigmoid` and the log-odds otherwise; multiclass tasks give per-class P with `softmax` and log P otherwise (so a
softmax applied by the metric, e.g. `roc_auc_multiclass`, recovers P exactly).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from core import registry


def _labels(y) -> np.ndarray:
    """Class indices from (n,), (n, 1) or one-hot (n, K) targets."""
    y = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
    if y.ndim == 2 and y.shape[1] > 1:
        return y.argmax(axis=1)
    return np.rint(y.reshape(-1)).astype(int)


def _flatten(x) -> np.ndarray:
    x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    return x.reshape(x.shape[0], -1)


class SklearnLogisticReadout(nn.Module):
    """L2 logistic regression fitted in closed form; `closed_form = True` tells the training loop to call `fit`."""

    closed_form = True

    def __init__(
        self,
        C: float = 1.0,
        C_grid=None,
        penalty: str = "l2",
        solver: str = "lbfgs",
        class_weight=None,
        max_iter: int = 10000,
        tol: float = 1e-3,
        random_state: int = 42,
        standardize: bool = True,
        output_activation: str = "linear",
        output_dim: int | None = None,
    ):
        super().__init__()
        self.C = float(C)
        self.C_grid = [float(c) for c in C_grid] if C_grid else None
        self.lr_kwargs = dict(
            penalty=penalty,
            solver=solver,
            class_weight=class_weight,
            max_iter=int(max_iter),
            tol=float(tol),
            random_state=random_state,
        )
        self.standardize = bool(standardize)
        self.output_activation = output_activation
        self.output_dim = output_dim
        self.scaler = None
        self.clf = None
        self.n_classes = None
        self.selection = {}

    # -- fitting -------------------------------------------------------------------------------------------------

    def _new_clf(self, C):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=C, **self.lr_kwargs)

    def fit(self, X_train, y_train, X_val=None, y_val=None, score_fn=None, smaller_is_better=False):
        """Fit on the training split; with `C_grid`, choose C by `score_fn(forward(X_val), y_val)`."""
        from sklearn.preprocessing import StandardScaler

        Xtr = _flatten(X_train)
        ytr = _labels(y_train)
        self.n_classes = max(int(ytr.max()) + 1, 2)
        if self.output_dim and int(self.output_dim) > 1:
            self.n_classes = max(self.n_classes, int(self.output_dim))
        self.scaler = StandardScaler().fit(Xtr) if self.standardize else None
        Ztr = self.scaler.transform(Xtr) if self.scaler is not None else Xtr

        if not self.C_grid:
            self.clf = self._new_clf(self.C).fit(Ztr, ytr)
            self.selection = {"C": self.C, "n_iter": int(np.max(self.clf.n_iter_))}
            return self

        if X_val is None or y_val is None or score_fn is None:
            raise ValueError("C_grid needs the validation split and a score function")
        yval_t = y_val if torch.is_tensor(y_val) else torch.as_tensor(y_val)
        best, scores = None, {}
        for C in self.C_grid:
            self.clf = self._new_clf(C).fit(Ztr, ytr)
            val = score_fn(self.forward(X_val), yval_t)
            val = float(val.detach().mean().item()) if torch.is_tensor(val) else float(val)
            scores[C] = val
            if best is None or (val < best[1] if smaller_is_better else val > best[1]):
                best = (C, val, self.clf)
        self.C, _, self.clf = best
        self.selection = {"C": self.C, "val_scores": scores, "n_iter": int(np.max(self.clf.n_iter_))}
        return self

    # -- prediction ----------------------------------------------------------------------------------------------

    def _proba(self, X) -> np.ndarray:
        """(n, n_classes) probabilities, with zero columns for classes absent from the training split."""
        Z = _flatten(X)
        if self.scaler is not None:
            Z = self.scaler.transform(Z)
        p = self.clf.predict_proba(Z)
        full = np.zeros((len(Z), self.n_classes), dtype=np.float64)
        full[:, self.clf.classes_.astype(int)] = p
        return full

    def forward(self, x, **kwargs):
        if self.clf is None:
            raise RuntimeError("SklearnLogisticReadout.forward called before fit")
        device = x.device if torch.is_tensor(x) else torch.device("cpu")
        if self.n_classes == 2 and not (self.output_dim and int(self.output_dim) > 1):
            if self.output_activation == "sigmoid":
                out = self._proba(x)[:, 1]
            else:
                Z = _flatten(x)
                if self.scaler is not None:
                    Z = self.scaler.transform(Z)
                out = self.clf.decision_function(Z)
        else:
            p = self._proba(x)
            out = p if self.output_activation == "softmax" else np.log(np.clip(p, 1e-12, None))
        return torch.as_tensor(out, dtype=torch.float32, device=device)


@registry.register_model_constructor()
def sklearn_logistic_readout(model_params):
    keys = (
        "C",
        "C_grid",
        "penalty",
        "solver",
        "class_weight",
        "max_iter",
        "tol",
        "random_state",
        "standardize",
        "output_activation",
        "output_dim",
    )
    kwargs = {k: model_params[k] for k in keys if k in model_params}
    # `torch_linear_model` names its output activation `activation`; `mlp_probe_decoder` uses `output_activation`.
    if "output_activation" not in kwargs and "activation" in model_params:
        kwargs["output_activation"] = model_params["activation"]
    if "output_dim" not in kwargs and "embedding_dim" in model_params:
        kwargs["output_dim"] = model_params["embedding_dim"]
    return SklearnLogisticReadout(**kwargs)
