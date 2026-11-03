"""Closed-form logistic-regression read-out (`sklearn_logistic_readout`).

A drop-in replacement for a trained probe head (e.g. `mlp_probe_decoder`) on classification tasks: swap
`model_spec.constructor_name` and keep the rest of the config. Instead of the epoch loop with early stopping, the
training loop calls `fit` once per fold (see `decoding_utils._fit_closed_form_fold`), then scores the train, val
and test splits with the same `_run_epoch` and metric functions as a trained head. So folds, test rows and metric
code are identical, and only the read-out differs.

The estimator is iMindBench's (`neuroprobe_eval/models/logistic_model.py`): scikit-learn
`LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", max_iter=10000, tol=1e-3, random_state=42)` on the
flattened input. The features are standardised with the TRAINING split's mean and SD (`StandardScaler`
convention: ddof=0, zero-variance features get scale 1), as iMindBench standardises before its logistic regression.

C is the inverse L2 regularisation strength (smaller C, stronger penalty). By default it is fixed, as in
iMindBench, and the validation split is not used for fitting. With `C_grid`, one model per C is fitted on the
training split, and the C with the best validation score on `training_params.early_stopping_metric` is kept.
There is no refit on train+val, so a model always sees the same training rows as a trained head.

`row_space: true` solves the same problem in the span of the training rows, for inputs with many more features than
training rows (frozen foundation features reach ~10^6). The penalty is on the weights only and the standardised
training rows are centred, so the optimal weights lie in that span (representer theorem): with
G = Z_train Z_train^T = U S^2 U^T, fitting the same `LogisticRegression` on the coordinates U S (n_train columns
instead of n_features) reaches the same optimum, and the weights are mapped back as W = Z_train^T U S^-1 beta.
Only the lbfgs path differs, so the result agrees with `row_space: false` up to the solver tolerance. The Gram
matrix is accumulated in float64 over feature chunks on `gram_device` (default: CUDA when available).

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


def _flat_tensor(x) -> torch.Tensor:
    x = x if torch.is_tensor(x) else torch.as_tensor(np.asarray(x))
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
        row_space: bool = False,
        gram_device: str | None = None,
        gram_chunk: int = 1 << 16,
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
        self.row_space = bool(row_space)
        if self.row_space and penalty != "l2":
            raise ValueError("row_space needs penalty='l2'")
        self.gram_device = gram_device
        self.gram_chunk = int(gram_chunk)
        self.output_activation = output_activation
        self.output_dim = output_dim
        self.n_classes = None
        self.selection = {}
        # Fitted state on the flattened raw input: z = (x - mean) / scale; decision = z @ coef.T + intercept
        self.mean = self.scale = self.coef = self.intercept = None
        self.classes = None

    # -- fitting -------------------------------------------------------------------------------------------------

    def _new_clf(self, C):
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=C, **self.lr_kwargs)

    def _standardisation(self, Xtr: torch.Tensor):
        if not self.standardize:
            return torch.zeros(Xtr.shape[1]), torch.ones(Xtr.shape[1])
        Xd = Xtr.double()
        mean = Xd.mean(0)
        scale = Xd.std(0, unbiased=False)
        scale[scale == 0] = 1.0  # StandardScaler's convention for constant features
        return mean.float(), scale.float()

    def _gram_dev(self):
        if self.gram_device:
            return torch.device(self.gram_device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _z_chunk(self, X, a, b, dev):
        return ((X[:, a:b].to(dev) - self.mean[a:b].to(dev)) / self.scale[a:b].to(dev)).double()

    def _row_space_fit(self, Xtr, Xval, fit_one):
        """Fit in training-row coordinates; returns the classifier and the weights mapped back to features."""
        dev = self._gram_dev()
        n_tr, n_feat = Xtr.shape
        X_all = Xtr if Xval is None else torch.cat([Xtr, Xval])
        G = torch.zeros((len(X_all), n_tr), dtype=torch.float64, device=dev)
        for a in range(0, n_feat, self.gram_chunk):
            Zc = self._z_chunk(X_all, a, a + self.gram_chunk, dev)
            G += Zc @ Zc[:n_tr].T
            del Zc
        w, U = torch.linalg.eigh(G[:n_tr])
        keep = w > w.max() * 1e-12  # drops the null direction(s) of the centred training rows
        U, sv = U[:, keep], torch.sqrt(w[keep])
        proj = U / sv  # coordinates of any row = G_row @ proj
        T_all = (G @ proj).cpu().numpy()
        del G
        clf = fit_one(T_all[:n_tr], None if Xval is None else T_all[n_tr:])
        # W = Z_train^T @ proj @ beta^T, accumulated over feature chunks.
        A = proj @ torch.as_tensor(clf.coef_, dtype=torch.float64, device=dev).T  # (n_tr, n_out)
        coef = torch.empty((A.shape[1], n_feat), dtype=torch.float32)
        for a in range(0, n_feat, self.gram_chunk):
            Zc = self._z_chunk(Xtr, a, a + self.gram_chunk, dev)
            coef[:, a : a + self.gram_chunk] = (Zc.T @ A).T.float().cpu()
            del Zc
        self.selection["rank"] = int(keep.sum())
        return clf, coef

    def fit(self, X_train, y_train, X_val=None, y_val=None, score_fn=None, smaller_is_better=False):
        """Fit on the training split; with `C_grid`, choose C by `score_fn(output on X_val, y_val)`."""
        Xtr = _flat_tensor(X_train).detach().cpu().float()
        Xval = None if X_val is None else _flat_tensor(X_val).detach().cpu().float()
        ytr = _labels(y_train)
        self.n_classes = max(int(ytr.max()) + 1, 2)
        if self.output_dim and int(self.output_dim) > 1:
            self.n_classes = max(self.n_classes, int(self.output_dim))
        self.mean, self.scale = self._standardisation(Xtr)
        self.selection = {}
        grid = self.C_grid or [self.C]
        if self.C_grid and (Xval is None or y_val is None or score_fn is None):
            raise ValueError("C_grid needs the validation split and a score function")
        yval_t = None if y_val is None else (y_val if torch.is_tensor(y_val) else torch.as_tensor(y_val))

        def fit_one(T_tr, T_val):
            if not self.C_grid:
                return self._new_clf(self.C).fit(T_tr, ytr)
            best, scores = None, {}
            for C in grid:
                clf = self._new_clf(C).fit(T_tr, ytr)
                out = self._output(clf.decision_function(T_val), clf.classes_)
                val = score_fn(torch.as_tensor(out, dtype=torch.float32), yval_t)
                val = float(val.detach().mean().item()) if torch.is_tensor(val) else float(val)
                scores[C] = val
                if best is None or (val < best[1] if smaller_is_better else val > best[1]):
                    best = (C, val, clf)
            self.C = best[0]
            self.selection["val_scores"] = scores
            return best[2]

        if self.row_space:
            clf, self.coef = self._row_space_fit(Xtr, Xval, fit_one)
        else:
            Z = ((Xtr - self.mean) / self.scale).numpy()
            Zval = None if Xval is None else ((Xval - self.mean) / self.scale).numpy()
            clf = fit_one(Z, Zval)
            self.coef = torch.as_tensor(clf.coef_, dtype=torch.float32)
        self.intercept = torch.as_tensor(clf.intercept_, dtype=torch.float32)
        self.classes = clf.classes_.astype(int)
        self.selection.update({"C": self.C, "n_iter": int(np.max(clf.n_iter_)), "row_space": self.row_space})
        return self

    # -- prediction ----------------------------------------------------------------------------------------------

    def _binary_output(self):
        return self.n_classes == 2 and not (self.output_dim and int(self.output_dim) > 1)

    def _output(self, dec, classes):
        """The read-out's output from sklearn decision values ((n,) for two classes, (n, k) multinomial)."""
        dec = np.asarray(dec, dtype=np.float64)
        if self._binary_output():
            return 1.0 / (1.0 + np.exp(-dec)) if self.output_activation == "sigmoid" else dec
        if dec.ndim == 1:  # two classes seen but a multiclass output: P = sigmoid(d) = softmax([-d, d] / 2)
            dec = np.stack([-dec, dec], axis=1) / 2.0
        dec = dec - dec.max(axis=1, keepdims=True)
        p = np.exp(dec)
        p /= p.sum(axis=1, keepdims=True)
        full = np.zeros((len(p), self.n_classes))
        full[:, np.asarray(classes).astype(int)] = p
        return full if self.output_activation == "softmax" else np.log(np.clip(full, 1e-12, None))

    def decision_function(self, x) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("SklearnLogisticReadout used before fit")
        X = _flat_tensor(x).detach().float()
        dev = X.device
        Z = (X - self.mean.to(dev)) / self.scale.to(dev)
        dec = (Z @ self.coef.to(dev).T + self.intercept.to(dev)).cpu().double().numpy()
        return dec[:, 0] if dec.shape[1] == 1 else dec

    def forward(self, x, **kwargs):
        device = x.device if torch.is_tensor(x) else torch.device("cpu")
        out = self._output(self.decision_function(x), self.classes)
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
        "row_space",
        "gram_device",
        "gram_chunk",
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
