from collections import Counter
from typing import Optional
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset, Dataset
from mup import MuAdam, MuAdamW

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

from tqdm import tqdm

import mne

from utils import data_utils
from utils.dataset import NeuralDictDataset
from core.config import TrainingParams, TaskConfig, ModelSpec
from utils.fold_utils import get_sequential_folds, get_zero_shot_folds
from utils.model_utils import build_model_from_spec
import metrics
from utils.plot_utils import plot_cv_results, plot_training_history
from core.registry import metric_registry
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LinearRegression, Ridge
import time


def train_logistic_regression(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train: Training features
        y_train: Training labels

    Returns:
        Trained LogisticRegression model
    """
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model.

    Args:
        X_train: Training features
        y_train: Training targets

    Returns:
        Trained LinearRegression model
    """
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def train_ridge_regression(X_train, y_train, alpha=1.0):
    """
    Train a ridge regression model.

    Args:
        X_train: Training features
        y_train: Training targets
        alpha: Regularization strength (default=1.0)

    Returns:
        Trained Ridge model
    """
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    return model


def compute_baseline_metrics(model, X_splits, Y_splits, all_fns, model_params=None):
    """
    Compute all metrics for a baseline model (logistic or linear regression) on train/val/test splits.

    Args:
        model: Trained sklearn model (LogisticRegression or LinearRegression)
        X_splits: Dict with keys 'train', 'val', 'test' containing feature arrays
        Y_splits: Dict with keys 'train', 'val', 'test' containing target arrays
        all_fns: Dictionary mapping metric names to callable functions
        model_params: Optional model parameters dict (needed for confusion_matrix)

    Returns:
        dict: Dictionary with keys like 'train_metric_name', 'val_metric_name', 'test_metric_name'
    """
    results = {}

    for phase in ["train", "val", "test"]:
        X = X_splits[phase]
        Y = Y_splits[phase]

        # Flatten X if needed
        X_flat = np.reshape(X, (X.shape[0], -1))

        # Get predictions
        # For classification tasks, use predict_proba if available (needed for cross_entropy)
        if hasattr(model, "predict_proba"):
            predictions = model.predict_proba(X_flat)
            # Catch binary case.
            if predictions.shape[-1] == 2:
                predictions = predictions[:, 1]
        else:
            predictions = model.predict(X_flat)

        # Compute all metrics
        metrics = compute_all_metrics(predictions, Y, all_fns, model_params)

        # Store with phase prefix
        for metric_name, metric_value in metrics.items():
            results[f"{phase}_{metric_name}"] = metric_value

    return results


def log_metrics_to_tensorboard(writer, metrics, model_name, phase, step):
    """
    Log metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance
        metrics: Dict of metrics to log (e.g., {"mse": 0.5, "corr": 0.8} or {"train_mse": 0.5, "val_mse": 0.6})
                 Can be None, in which case nothing is logged.
        model_name: Name/namespace for the model (e.g., "model", "linear_regression", "logistic_regression")
        phase: Phase name (e.g., "train", "val", "test"). If None, will attempt to extract from metric names.
        step: Step number (epoch or fold number)
    """
    if not metrics:
        return

    for metric_name, metric_value in metrics.items():
        # If no phase provided, try to extract from metric name (e.g., "train_mse" -> "train", "mse")
        if phase is None:
            parts = metric_name.split("_", 1)
            if len(parts) == 2:
                metric_phase, metric_name = parts
            else:
                continue  # Skip if we can't extract phase
        else:
            metric_phase = phase

        if np.isscalar(metric_value) or (
            isinstance(metric_value, np.ndarray) and metric_value.size == 1
        ):
            writer.add_scalar(
                f"{model_name}/{metric_name}/{metric_phase}", metric_value, step
            )
        elif "confusion_matrix" in metric_name:
            writer.add_text(
                f"{model_name}/{metric_name}/{metric_phase}", str(metric_value), step
            )


def setup_metrics_and_loss(training_params: TrainingParams):
    """
    Set up metrics and loss functions from training parameters.

    Returns:
        dict: Dictionary mapping metric names to callable functions
    """
    # If user provided loss_name, set it as the loss.
    if training_params.loss_name:
        training_params.losses = [training_params.loss_name]
        training_params.loss_weights = [1]

    # Combine loss and metrics into single list
    metric_names = training_params.losses + training_params.metrics

    # Resolve all functions from registry
    all_fns = {name: metric_registry[name] for name in metric_names}

    return all_fns


def compute_loss(out, groundtruth, training_params, all_fns):
    loss = None
    for i, loss_name in enumerate(training_params.losses):
        loss_val = training_params.loss_weights[i] * all_fns[loss_name](
            out, groundtruth
        )
        if loss is None:
            loss = loss_val
        else:
            loss = loss + loss_val
    return loss


def compute_all_metrics(predictions, groundtruth, all_fns, model_params=None):
    """
    Compute all metrics given predictions and ground truth.

    Args:
        predictions: Model predictions (torch.Tensor or np.ndarray)
        groundtruth: Ground truth labels/values (torch.Tensor or np.ndarray)
        all_fns: Dictionary mapping metric names to callable functions
        model_params: Optional model parameters dict (needed for confusion_matrix)

    Returns:
        dict: Dictionary mapping metric names to computed values
    """
    metrics_dict = {}

    # Convert to tensors if needed
    if isinstance(predictions, np.ndarray):
        predictions = torch.tensor(predictions, dtype=torch.float32)
    if isinstance(groundtruth, np.ndarray):
        groundtruth = torch.tensor(groundtruth, dtype=torch.float32)

    for name, fn in all_fns.items():
        if name == "confusion_matrix":
            # Special handling for confusion matrix
            if model_params is None:
                continue
            if model_params.get("embedding_dim") == 1:
                num_classes = 2
            else:
                num_classes = model_params.get("embedding_dim")
            val = fn(predictions, groundtruth, num_classes)
            metrics_dict[name] = (
                val.detach().cpu().numpy() if torch.is_tensor(val) else np.array(val)
            )
        else:
            val = fn(predictions, groundtruth)
            # Convert to scalar
            if torch.is_tensor(val):
                val = val.detach().mean().item()
            metrics_dict[name] = val

    return metrics_dict


def validate_early_stopping_config(training_params: TrainingParams):
    """
    Validate that early stopping configuration is valid.

    Raises:
        ValueError: If early stopping metric is not in available metrics
    """
    available_metrics = [training_params.loss_name] + training_params.metrics

    if training_params.early_stopping_metric not in available_metrics:
        raise ValueError(
            f"Early stopping metric '{training_params.early_stopping_metric}' "
            f"must be either the loss function or in the metrics list. "
            f"Available: {available_metrics}"
        )


def get_fold_function_name(training_params: TrainingParams):
    """
    Get the name of the fold function to use based on training parameters.

    Returns:
        str: Name of the fold function

    Raises:
        ValueError: If fold_type is not recognized
    """
    if training_params.fold_type == "sequential_folds":
        return "get_sequential_folds"
    elif training_params.fold_type == "zero_shot_folds":
        return "get_zero_shot_folds"
    else:
        raise ValueError(f"Unknown fold_type: {training_params.fold_type}")


def setup_early_stopping_state(training_params: TrainingParams):
    """
    Set up initial state for early stopping.

    Returns:
        tuple: (best_val, patience) initial values
    """
    if training_params.smaller_is_better:
        best_val = float("inf")
    else:
        best_val = -float("inf")

    patience = 0

    return best_val, patience


def should_update_best(current_val, best_val, smaller_is_better):
    """
    Determine if current validation value is better than best.

    Returns:
        bool: True if current value is better
    """
    if smaller_is_better:
        return current_val < best_val
    else:
        return current_val > best_val


def create_lr_scheduler(optimizer, training_params: TrainingParams):
    """
    Create a ReduceLROnPlateau learning rate scheduler.

    Args:
        optimizer: PyTorch optimizer
        training_params: Training parameters containing scheduler config

    Returns:
        ReduceLROnPlateau scheduler or None if use_lr_scheduler is False
    """
    if not training_params.use_lr_scheduler:
        return None

    params = training_params.scheduler_params or {}

    # Auto-detect mode based on smaller_is_better unless explicitly provided
    mode = params.get("mode", "min" if training_params.smaller_is_better else "max")
    factor = params.get("factor", 0.5)
    patience = params.get("patience", 10)
    min_lr = params.get("min_lr", 1e-6)

    return lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr
    )


def should_update_gradient_accumulation(
    batch_idx, total_batches, grad_accumulation_steps
):
    """
    Determine if optimizer should step based on gradient accumulation.

    Returns:
        bool: True if optimizer should step
    """
    return (batch_idx + 1) % grad_accumulation_steps == 0 or (
        batch_idx + 1
    ) == total_batches


def _maybe_shuffle_targets(Y: torch.Tensor, training_params: TrainingParams):
    if not training_params.shuffle_targets:
        return Y

    print("WARNING: Shuffling targets for sanity check. Model should perform poorly.")
    rng = np.random.default_rng(training_params.random_seed)
    shuffle_indices = rng.permutation(len(Y))
    return Y[shuffle_indices]


def _get_fold_indices(
    neural_data: torch.Tensor,
    data_df: pd.DataFrame,
    task_config: TaskConfig,
    training_params: TrainingParams,
):
    if training_params.fold_type == "sequential_folds":
        return get_sequential_folds(neural_data, num_folds=training_params.n_folds)
    if training_params.fold_type == "zero_shot_folds":
        return get_zero_shot_folds(
            data_df[task_config.data_params.word_column].values,
            num_folds=training_params.n_folds,
        )
    raise ValueError(f"Unknown fold_type: {training_params.fold_type}")


def _select_requested_folds(fold_indices, training_params: TrainingParams):
    fold_nums = list(range(1, len(fold_indices) + 1))
    fold_ids = getattr(training_params, "fold_ids", None)
    if fold_ids is None:
        return fold_indices, fold_nums

    if len(fold_ids) == 0:
        raise ValueError(
            "training_params.fold_ids is empty. Provide at least one fold id or omit it."
        )

    bad = [k for k in fold_ids if (k < 1 or k > len(fold_indices))]
    if bad:
        raise ValueError(
            f"fold_ids must be 1-based integers in [1, {len(fold_indices)}]. "
            f"Got invalid: {bad}. If you intended the first fold, use [1] (not [0])."
        )

    seen = set()
    selected_fold_nums = [k for k in fold_ids if not (k in seen or seen.add(k))]
    selected_fold_indices = [fold_indices[k - 1] for k in selected_fold_nums]
    return selected_fold_indices, selected_fold_nums


def _maybe_visualize_fold_distribution(
    Y, fold_indices, task_name: str, lag: int, training_params: TrainingParams
):
    if not training_params.visualize_fold_distribution:
        return

    from utils.analysis_utils import visualize_fold_distribution

    Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
    visualize_fold_distribution(Y_np, fold_indices, task_name=task_name, lag=lag)


def _word_embedding_metric_names(training_params: TrainingParams):
    embedding_metrics = [
        "test_word_avg_auc_roc",
        "test_word_train_weighted_auc_roc",
        "test_word_test_weighted_auc_roc",
        "test_word_perplexity",
        "test_occurence_perplexity",
    ]
    for k_val in training_params.top_k_thresholds:
        for test_type in ["word", "occurence"]:
            embedding_metrics.append(f"test_{test_type}_top_{k_val}")
    return embedding_metrics


def _init_cv_results(metric_names, task_name: str, training_params: TrainingParams):
    phases = ("train", "val", "test")
    cv_results = {
        f"{phase}_{name}": []
        for phase in phases
        for name in metric_names
        if name != "confusion_matrix"
    }
    cv_results["num_epochs"] = []
    cv_results["fold_nums"] = []

    embedding_metrics = None
    if task_name == "word_embedding_decoding_task":
        embedding_metrics = _word_embedding_metric_names(training_params)
        for metric in embedding_metrics:
            cv_results[metric] = []

    return cv_results, embedding_metrics


def _print_fold_debug(fold, neural_data, Y, tr_idx, va_idx, te_idx):
    print(f"Fold {fold}")
    print(f"Train indices: {tr_idx}")
    print(f"Validation indices: {va_idx}")
    print(f"Test indices: {te_idx}")
    print(f"Train size: {len(tr_idx)}")
    print(f"Validation size: {len(va_idx)}")
    print(f"Test size: {len(te_idx)}")
    print(f"Train Input shape: {neural_data[tr_idx].shape}")
    print(f"Train targets: {Y[tr_idx]}, shape: {Y[tr_idx].shape}")
    print(f"Validation targets: {Y[va_idx]}, shape: {Y[va_idx].shape}")
    print(f"Test targets: {Y[te_idx]}, shape: {Y[te_idx].shape}")


def _create_tensorboard_writer(write_to_tensorboard, tensorboard_dir, lag, fold):
    if not write_to_tensorboard:
        return None
    if not TENSORBOARD_AVAILABLE:
        raise ImportError(
            "TensorBoard is not available. Please install it with: "
            "pip install tensorboard"
        )
    tb_path = os.path.join(tensorboard_dir, f"lag_{lag}", f"fold_{fold}")
    return SummaryWriter(log_dir=tb_path)


def _normalize_fold_targets(Y, tr_idx, va_idx, te_idx, training_params: TrainingParams):
    if not training_params.normalize_targets:
        return {"train": Y[tr_idx], "val": Y[va_idx], "test": Y[te_idx]}

    print("Normalizing targets...")
    Y_train = Y[tr_idx]
    y_mean = Y_train.mean(dim=0, keepdim=True)
    y_std = Y_train.std(dim=0, keepdim=True)
    y_std = torch.where(y_std < 1e-6, torch.ones_like(y_std), y_std)
    return {
        "train": (Y_train - y_mean) / y_std,
        "val": (Y[va_idx] - y_mean) / y_std,
        "test": (Y[te_idx] - y_mean) / y_std,
    }


def _build_fold_loaders(
    neural_data,
    data_df,
    task_config: TaskConfig,
    split_indices,
    target_splits,
    training_params: TrainingParams,
):
    input_fields = task_config.task_specific_config.input_fields
    extra_inputs = {
        phase: data_utils.df_columns_to_tensors(data_df, input_fields, indices)
        for phase, indices in split_indices.items()
    }
    datasets = {
        phase: NeuralDictDataset(
            neural_data[indices], extra_inputs[phase], target_splits[phase]
        )
        for phase, indices in split_indices.items()
    }
    return {
        phase: DataLoader(
            ds, batch_size=training_params.batch_size, shuffle=(phase == "train")
        )
        for phase, ds in datasets.items()
    }


def _build_full_lag_loader(
    neural_data,
    data_df,
    Y,
    task_config: TaskConfig,
    training_params: TrainingParams,
):
    input_fields = task_config.task_specific_config.input_fields
    indices = np.arange(len(neural_data))
    extra_inputs = data_utils.df_columns_to_tensors(data_df, input_fields, indices)
    dataset = NeuralDictDataset(neural_data, extra_inputs, Y)
    return DataLoader(dataset, batch_size=training_params.batch_size, shuffle=False)


def _as_index_tensor(indices, device=None):
    if isinstance(indices, np.ndarray):
        return torch.as_tensor(indices, dtype=torch.long, device=device)
    if torch.is_tensor(indices):
        return indices.to(device=device, dtype=torch.long)
    return torch.tensor(indices, dtype=torch.long, device=device)


def _slice_input_dict(inputs, indices):
    if torch.is_tensor(indices):
        base_indices = indices.to(dtype=torch.long)
    elif isinstance(indices, np.ndarray):
        base_indices = torch.as_tensor(indices, dtype=torch.long)
    else:
        base_indices = torch.tensor(indices, dtype=torch.long)

    sliced = {}
    for key, val in inputs.items():
        if torch.is_tensor(val):
            sliced[key] = val[base_indices.to(device=val.device)]
        else:
            sliced[key] = val
    return sliced


def _build_cached_fold_loaders(
    cached_features,
    cached_extra_inputs,
    split_indices,
    target_splits,
    training_params: TrainingParams,
):
    datasets = {
        phase: NeuralDictDataset(
            cached_features[_as_index_tensor(indices, device=cached_features.device)],
            _slice_input_dict(cached_extra_inputs, indices),
            target_splits[phase],
        )
        for phase, indices in split_indices.items()
    }
    return {
        phase: DataLoader(
            ds, batch_size=training_params.batch_size, shuffle=(phase == "train")
        )
        for phase, ds in datasets.items()
    }


def _model_spec_has_fold_checkpoint_template(model_spec: ModelSpec):
    checkpoint_path = getattr(model_spec, "checkpoint_path", None)
    if isinstance(checkpoint_path, str) and "{fold}" in checkpoint_path:
        return True

    for sub_spec in getattr(model_spec, "sub_models", {}).values():
        if _model_spec_has_fold_checkpoint_template(sub_spec):
            return True
    return False


def _validate_lag_level_feature_cache(model_spec: ModelSpec):
    if not (
        getattr(model_spec, "feature_cache", False)
        or getattr(model_spec, "per_subject_feature_concat", False)
    ):
        return

    if _model_spec_has_fold_checkpoint_template(model_spec):
        raise ValueError(
            "Lag-level feature caching cannot be used with checkpoint_path values "
            "containing '{fold}', because fold-specific encoders cannot safely share "
            "one activation cache."
        )


def _create_optimizer(model, training_params: TrainingParams):
    if training_params.optimizer == "MuAdamW":
        print("Using MuAdamW optimizer")
        return MuAdamW(
            model.parameters(),
            lr=float(training_params.learning_rate),
            weight_decay=float(training_params.weight_decay),
        )

    print("Using AdamW optimizer")
    return optim.AdamW(
        model.parameters(),
        lr=float(training_params.learning_rate),
        weight_decay=float(training_params.weight_decay),
    )


def _create_training_scheduler(optimizer, loaders, training_params: TrainingParams):
    scheduler = None
    if training_params.lr_scheduler:
        print(f"Using {training_params.lr_scheduler} LR scheduler")
        if training_params.lr_scheduler == "cosine_annealing":
            updates_per_epoch = math.ceil(
                len(loaders["train"])
                / max(1, int(training_params.grad_accumulation_steps))
            )
            t_max = max(1, int(training_params.epochs) * updates_per_epoch)
            eta_min = float(training_params.learning_rate) * float(
                getattr(training_params, "cosine_eta_min_factor", 1e-2)
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min
            )
        else:
            raise ValueError(
                f"Unknown lr_scheduler: {training_params.lr_scheduler}. "
                "Supported: None, 'cosine_annealing'"
            )

    scheduler = create_lr_scheduler(optimizer, training_params)
    return scheduler


def _build_model_optimizer_scheduler(
    model_spec,
    lag,
    fold,
    loaders,
    training_params,
    device,
):
    model = build_model_from_spec(model_spec, lag=lag, fold=fold).to(device)
    optimizer = _create_optimizer(model, training_params)
    scheduler = _create_training_scheduler(optimizer, loaders, training_params)
    return model, optimizer, scheduler


def _create_training_history(metric_names):
    history = {
        f"{phase}_{name}": [] for phase in ("train", "val") for name in metric_names
    }
    if "cross_entropy" in metric_names:
        for phase in ("train", "val"):
            history[f"{phase}_perplexity"] = []
    history["train_loss"] = []
    history["val_loss"] = []
    history["num_epochs"] = None
    return history


def _move_batch_to_device(batch_data, device):
    Xb, inputs_dict, yb = batch_data
    Xb = Xb.to(device)
    inputs_dict = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in inputs_dict.items()
    }
    yb = yb.to(device)
    return Xb, inputs_dict, yb


def _accumulate_batch_metrics(sums, batch_metrics):
    for name, val in batch_metrics.items():
        if sums[name] is None:
            sums[name] = val
        else:
            sums[name] += val


def _run_epoch(
    model,
    loader,
    device,
    training_params,
    all_fns,
    metric_names,
    model_params,
    optimizer=None,
    scheduler=None,
):
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    sums = {name: None if name == "confusion_matrix" else 0.0 for name in metric_names}
    sums["loss"] = 0.0
    grad_steps = training_params.grad_accumulation_steps

    if is_train:
        optimizer.zero_grad()

    for i, batch_data in enumerate(loader):
        Xb, inputs_dict, yb = _move_batch_to_device(batch_data, device)

        if is_train:
            out = model(Xb, **inputs_dict)
            loss = compute_loss(out, yb, training_params, all_fns)
            loss = loss / grad_steps
            loss.backward()

            if should_update_gradient_accumulation(i, len(loader), grad_steps):
                if (
                    getattr(training_params, "clip_grad_norm", 0.0)
                    and float(training_params.clip_grad_norm) > 0.0
                ):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=float(training_params.clip_grad_norm),
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                out = model(Xb, **inputs_dict)
                loss = compute_loss(out, yb, training_params, all_fns)

        batch_metrics = compute_all_metrics(out, yb, all_fns, model_params)
        _accumulate_batch_metrics(sums, batch_metrics)

        if torch.is_tensor(loss):
            loss = loss.detach().mean().item()
        sums["loss"] += loss

    result = {
        name: sums[name] if name == "confusion_matrix" else sums[name] / len(loader)
        for name in sums
    }

    if "cross_entropy" in result:
        result["perplexity"] = np.exp(result["cross_entropy"])

    return result


def _train_and_eval_baseline(
    training_fn,
    neural_data,
    split_indices,
    target_splits,
    all_fns,
    model_params,
    **kwargs,
):
    tr_idx = split_indices["train"]
    model = training_fn(
        neural_data[tr_idx].cpu().numpy(),
        target_splits["train"].cpu().numpy(),
        **kwargs,
    )
    X_splits = {
        phase: neural_data[indices].cpu().numpy()
        for phase, indices in split_indices.items()
    }
    Y_splits = {
        phase: targets.cpu().numpy() for phase, targets in target_splits.items()
    }
    return compute_baseline_metrics(model, X_splits, Y_splits, all_fns, model_params)


def _train_enabled_baselines(
    neural_data,
    split_indices,
    target_splits,
    training_params,
    all_fns,
    model_params,
):
    results = {}
    if training_params.logistic_regression_baseline:
        print("Training logistic regression baseline...")
        results["logistic_regression"] = _train_and_eval_baseline(
            train_logistic_regression,
            neural_data,
            split_indices,
            target_splits,
            all_fns,
            model_params,
        )
    if training_params.linear_regression_baseline:
        print("Training linear regression baseline...")
        results["linear_regression"] = _train_and_eval_baseline(
            train_linear_regression,
            neural_data,
            split_indices,
            target_splits,
            all_fns,
            model_params,
        )
    if training_params.ridge_regression_baseline:
        print("Training ridge regression baseline...")
        results["ridge_regression"] = _train_and_eval_baseline(
            train_ridge_regression,
            neural_data,
            split_indices,
            target_splits,
            all_fns,
            model_params,
            alpha=training_params.ridge_alpha,
        )
    return results


def _append_baseline_results(all_baseline_results, fold_baseline_results):
    for name, metrics_dict in fold_baseline_results.items():
        all_baseline_results[name].append(metrics_dict)


def _maybe_prepare_per_subject_concat_model(
    model,
    loaders,
    model_spec,
    training_params,
    device,
):
    if not getattr(model_spec, "per_subject_feature_concat", False):
        return model, loaders, None

    output_dim = getattr(model, "output_dim", None)
    if output_dim is None:
        raise NotImplementedError(
            "per_subject_feature_concat requires the model to expose "
            f"output_dim. Got model: {model.__class__.__name__}"
        )

    sample_batch = next(iter(loaders["train"]))
    concat_dim = sample_batch[0].shape[-1]
    print(f"Linear probe: {concat_dim} -> {output_dim}")

    probe = nn.Linear(concat_dim, output_dim)
    model = SqueezeWrapper(
        feature_head=MakeIgnoreKwargsDuringForward(probe), output_dim=output_dim
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_params.learning_rate),
        weight_decay=float(training_params.weight_decay),
    )
    return model, loaders, optimizer


def _maybe_prepare_feature_cache_model(
    model_spec,
    lag,
    full_lag_loader,
    training_params,
    device,
    subject_channel_counts=None,
):
    if not (
        getattr(model_spec, "feature_cache", False)
        or getattr(model_spec, "per_subject_feature_concat", False)
    ):
        return None

    cache_loader_generation_start_time = time.time()
    cache_model = build_model_from_spec(model_spec, lag=lag, fold=1).to(device)
    features, input_dicts, _ = extract_features_for_caching(
        cache_model,
        full_lag_loader,
        device,
        subject_channel_counts=subject_channel_counts,
    )
    print(
        "Time taken for feature extraction and loader generation: "
        f"{time.time() - cache_loader_generation_start_time}"
    )

    if subject_channel_counts is not None:
        n_subjects = len(subject_channel_counts)
        embed_dim = features.shape[-1] // n_subjects
        print(
            "Per-subject-concat features: "
            f"{n_subjects} subjects x {embed_dim}d = {features.shape[-1]}d total"
        )
    elif not hasattr(cache_model, "forward_from_features"):
        raise NotImplementedError(
            "Feature caching requires the model to implement "
            f"forward_from_features(...). Got model: {cache_model.__class__.__name__}"
        )
    return features, _merge_input_dicts(input_dicts)


def _save_checkpoint(model, model_path):
    if hasattr(model, "save_checkpoint") and callable(getattr(model, "save_checkpoint")):
        model.save_checkpoint(model_path)
    else:
        torch.save(model.state_dict(), model_path)


def _load_checkpoint(model, model_path):
    if hasattr(model, "load_checkpoint") and callable(getattr(model, "load_checkpoint")):
        model.load_checkpoint(model_path)
    else:
        model.load_state_dict(torch.load(model_path))


def _append_epoch_metrics(history, train_mets, val_mets):
    for name, val in train_mets.items():
        history[f"train_{name}"].append(val)
    for name, val in val_mets.items():
        history[f"val_{name}"].append(val)


def _train_fold(
    model,
    loaders,
    optimizer,
    scheduler,
    model_path,
    lag,
    fold,
    training_params,
    all_fns,
    metric_names,
    model_params,
    device,
    writer=None,
):
    best_val, patience = setup_early_stopping_state(training_params)
    best_epoch = 0
    history = _create_training_history(metric_names)
    loop = tqdm(range(training_params.epochs), desc=f"Lag {lag}, Fold {fold}")

    loop_start_time = time.time()
    for epoch in loop:
        train_mets = _run_epoch(
            model,
            loaders["train"],
            device,
            training_params,
            all_fns,
            metric_names,
            model_params,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        val_mets = _run_epoch(
            model,
            loaders["val"],
            device,
            training_params,
            all_fns,
            metric_names,
            model_params,
        )
        _append_epoch_metrics(history, train_mets, val_mets)

        if writer is not None:
            log_metrics_to_tensorboard(writer, train_mets, "model", "train", epoch)
            log_metrics_to_tensorboard(writer, val_mets, "model", "val", epoch)

        cur = val_mets[training_params.early_stopping_metric]
        if should_update_best(cur, best_val, training_params.smaller_is_better):
            best_val = cur
            best_epoch = epoch
            _save_checkpoint(model, model_path)
            patience = 0
        else:
            patience += 1
            if patience >= training_params.early_stopping_patience:
                break

        if scheduler is not None:
            scheduler.step(cur)

        if writer is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("learning_rate", current_lr, epoch)

        loop.set_postfix(
            {
                training_params.early_stopping_metric: f"{best_val:.4f}",
                **{f"train_{name}": val for name, val in train_mets.items()},
                **{f"val_{name}": val for name, val in val_mets.items()},
            }
        )
    print(f"Time taken for training loop: {time.time() - loop_start_time}")

    history["num_epochs"] = best_epoch + 1
    _load_checkpoint(model, model_path)
    test_mets = _run_epoch(
        model,
        loaders["test"],
        device,
        training_params,
        all_fns,
        metric_names,
        model_params,
    )
    return history, test_mets, best_epoch


def _record_fold_results(cv_results, history, test_mets, metric_names, best_epoch):
    conf_matrices = {}
    for name in metric_names:
        if name != "confusion_matrix":
            cv_results[f"train_{name}"].append(history[f"train_{name}"][best_epoch])
            cv_results[f"val_{name}"].append(history[f"val_{name}"][best_epoch])
            cv_results[f"test_{name}"].append(test_mets[name])
        else:
            conf_matrices = {
                "train": history[f"train_{name}"][best_epoch],
                "val": history[f"val_{name}"][best_epoch],
                "test": test_mets[name],
            }
    cv_results["num_epochs"].append(history["num_epochs"])
    return conf_matrices


def _log_fold_tensorboard_results(writer, test_mets, fold_baseline_results, fold):
    if writer is None:
        return

    log_metrics_to_tensorboard(writer, test_mets, "model", "test", fold)
    for model_name, metrics_dict in fold_baseline_results.items():
        log_metrics_to_tensorboard(writer, metrics_dict, model_name, None, fold)
    writer.close()


def _collect_loader_features(loader):
    test_features = []
    test_targets = []
    with torch.no_grad():
        for batch_data in loader:
            features, _, y_b = batch_data
            test_features.append(features)
            test_targets.append(y_b)
    return torch.cat(test_features, dim=0), torch.cat(test_targets, dim=0)


def _maybe_compute_word_embedding_metrics(
    cv_results,
    embedding_metrics,
    loaders,
    model,
    device,
    data_df,
    task_config,
    tr_idx,
    te_idx,
    training_params,
):
    if embedding_metrics is None:
        return

    test_extra_inputs = data_utils.df_columns_to_tensors(
        data_df, task_config.task_specific_config.input_fields, te_idx
    )
    test_features, test_targets = _collect_loader_features(loaders["test"])
    results = metrics.embedding_metrics.compute_word_embedding_task_metrics(
        test_features,
        test_targets,
        model,
        device,
        data_df[task_config.data_params.word_column],
        te_idx,
        tr_idx,
        training_params.top_k_thresholds,
        training_params.min_train_freq_auc,
        training_params.min_test_freq_auc,
        extra_inputs=test_extra_inputs,
        preserve_ensemble=True,
    )
    for key, val in results.items():
        cv_results[key].append(val)


def _print_single_baseline_summary(title, baseline_results):
    if not baseline_results:
        return

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    for metric_name in baseline_results[0].keys():
        values = [result[metric_name] for result in baseline_results]
        if np.isscalar(values[0]) or (
            isinstance(values[0], np.ndarray) and values[0].size == 1
        ):
            print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def _print_baseline_summaries(training_params, baseline_results):
    if training_params.logistic_regression_baseline:
        _print_single_baseline_summary(
            "LOGISTIC REGRESSION BASELINE RESULTS",
            baseline_results["logistic_regression"],
        )
    if training_params.linear_regression_baseline:
        _print_single_baseline_summary(
            "LINEAR REGRESSION BASELINE RESULTS",
            baseline_results["linear_regression"],
        )
    if training_params.ridge_regression_baseline:
        _print_single_baseline_summary(
            f"RIDGE REGRESSION BASELINE RESULTS (alpha={training_params.ridge_alpha})",
            baseline_results["ridge_regression"],
        )


def _print_main_cv_summary(cv_results, metric_names, conf_matrices, embedding_metrics):
    print("\n" + "=" * 60)
    print("MAIN MODEL CROSS-VALIDATION RESULTS")
    print("=" * 60)

    for phase in ("train", "val", "test"):
        for name in metric_names:
            if name != "confusion_matrix":
                vals = cv_results[f"{phase}_{name}"]
                print(f"--- Individual Folds ({phase}_{name}) ---")
                fold_nums = cv_results.get("fold_nums", list(range(1, len(vals) + 1)))
                for i, val in enumerate(vals):
                    fold_num = fold_nums[i]
                    print(f"Fold {fold_num}: {val:.4f}")
                print(
                    f"Mean {phase} {name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n"
                )
            elif name == "confusion_matrix":
                print(f"{phase} confusion matrix:\n{conf_matrices[phase]}")

    if "cross_entropy" in metric_names:
        for phase in ("train", "val", "test"):
            ce_vals = cv_results[f"{phase}_cross_entropy"]
            ppl_vals = np.exp(ce_vals)
            print(
                f"Mean {phase} perplexity: {np.mean(ppl_vals):.4f} ± {np.std(ppl_vals):.4f}"
            )

    if embedding_metrics is not None:
        for metric_name in embedding_metrics:
            vals = cv_results[metric_name]
            print(f"Mean {metric_name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


def train_decoding_model(
    neural_data: torch.Tensor,
    Y: torch.Tensor,
    data_df: pd.DataFrame,
    model_spec: ModelSpec,
    task_name: str,
    task_config: TaskConfig,
    lag: int,
    training_params: TrainingParams,
    checkpoint_dir: str,
    plot_results: bool = False,
    write_to_tensorboard: bool = False,
    tensorboard_dir: str = "event_logs",
    subject_channel_counts: list[int] = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    Y = _maybe_shuffle_targets(Y, training_params)
    fold_indices = _get_fold_indices(neural_data, data_df, task_config, training_params)
    fold_indices, fold_nums = _select_requested_folds(fold_indices, training_params)
    _maybe_visualize_fold_distribution(Y, fold_indices, task_name, lag, training_params)

    all_fns = setup_metrics_and_loss(training_params)
    metric_names = all_fns.keys()
    cv_results, embedding_metrics = _init_cv_results(
        metric_names, task_name, training_params
    )

    models, histories = [], []
    baseline_results = {
        "logistic_regression": [],
        "linear_regression": [],
        "ridge_regression": [],
    }
    conf_matrices = {}
    cached_lag_features = None
    cached_lag_extra_inputs = None
    use_lag_feature_cache = getattr(model_spec, "feature_cache", False) or getattr(
        model_spec, "per_subject_feature_concat", False
    )
    if use_lag_feature_cache:
        _validate_lag_level_feature_cache(model_spec)
        full_lag_loader = _build_full_lag_loader(
            neural_data, data_df, Y, task_config, training_params
        )
        cached_lag_features, cached_lag_extra_inputs = _maybe_prepare_feature_cache_model(
            model_spec,
            lag,
            full_lag_loader,
            training_params,
            device,
            subject_channel_counts=(
                subject_channel_counts
                if getattr(model_spec, "per_subject_feature_concat", False)
                else None
            ),
        )

    for fold, (tr_idx, va_idx, te_idx) in zip(fold_nums, fold_indices):
        _print_fold_debug(fold, neural_data, Y, tr_idx, va_idx, te_idx)
        cv_results["fold_nums"].append(fold)
        model_path = os.path.join(checkpoint_dir, f"best_model_fold{fold}.pt")
        writer = _create_tensorboard_writer(
            write_to_tensorboard, tensorboard_dir, lag, fold
        )

        split_indices = {"train": tr_idx, "val": va_idx, "test": te_idx}
        target_splits = _normalize_fold_targets(
            Y, tr_idx, va_idx, te_idx, training_params
        )
        if use_lag_feature_cache:
            loaders = _build_cached_fold_loaders(
                cached_lag_features,
                cached_lag_extra_inputs,
                split_indices,
                target_splits,
                training_params,
            )
        else:
            loaders = _build_fold_loaders(
                neural_data,
                data_df,
                task_config,
                split_indices,
                target_splits,
                training_params,
            )

        fold_baseline_results = _train_enabled_baselines(
            neural_data,
            split_indices,
            target_splits,
            training_params,
            all_fns,
            model_spec.params,
        )
        _append_baseline_results(baseline_results, fold_baseline_results)

        model, optimizer, scheduler = _build_model_optimizer_scheduler(
            model_spec,
            lag,
            fold,
            loaders,
            training_params,
            device,
        )

        model, loaders, probe_optimizer = _maybe_prepare_per_subject_concat_model(
            model,
            loaders,
            model_spec,
            training_params,
            device,
        )
        if probe_optimizer is not None:
            optimizer = probe_optimizer
            scheduler = _create_training_scheduler(optimizer, loaders, training_params)
        elif getattr(model_spec, "feature_cache", False):
            model = CachedFeatureModel(model).to(device)

        history, test_mets, best_epoch = _train_fold(
            model,
            loaders,
            optimizer,
            scheduler,
            model_path,
            lag,
            fold,
            training_params,
            all_fns,
            metric_names,
            model_spec.params,
            device,
            writer,
        )
        fold_conf_matrices = _record_fold_results(
            cv_results, history, test_mets, metric_names, best_epoch
        )
        if fold_conf_matrices:
            conf_matrices = fold_conf_matrices

        _log_fold_tensorboard_results(writer, test_mets, fold_baseline_results, fold)
        _maybe_compute_word_embedding_metrics(
            cv_results,
            embedding_metrics,
            loaders,
            model,
            device,
            data_df,
            task_config,
            tr_idx,
            te_idx,
            training_params,
        )

        models.append(model)
        histories.append(history)

        if plot_results:
            plot_training_history(history, fold=fold)

    _print_baseline_summaries(training_params, baseline_results)
    _print_main_cv_summary(cv_results, metric_names, conf_matrices, embedding_metrics)

    if plot_results:
        plot_cv_results(cv_results)

    return models, histories, cv_results


def run_training_over_lags(
    lags,
    raws: list[mne.io.Raw],
    task_df: pd.DataFrame,
    preprocessing_fns: list[callable] | None,
    model_spec: ModelSpec,
    task_name: str,
    training_params: TrainingParams,
    task_config: TaskConfig,
    output_dir="results/",
    checkpoint_dir="checkpoints/",
    write_to_tensorboard=False,
    tensorboard_dir="event_log",
):
    data_params = task_config.data_params
    os.makedirs(output_dir, exist_ok=True)

    filename = os.path.join(output_dir, f"lag_performance.csv")

    # Load existing results if they exist
    if os.path.exists(filename):
        roc_df = pd.read_csv(filename)
        already_read_lags = roc_df["lags"].tolist()
        existing_df = roc_df
    else:
        already_read_lags = []
        existing_df = pd.DataFrame()

    all_new_results = []

    from utils.dataset import RawNeuralDataset

    raw_dataset = RawNeuralDataset(
        raws,
        task_df,
        data_params.window_width,
        preprocessing_fns,
        data_params.preprocessor_params,
    )

    for lag in lags:
        if lag in already_read_lags:
            print(f"Lag {lag} already done, skipping...")
            continue

        print("=" * 60)
        print("running lag:", lag)
        print("=" * 60)

        (
            neural_tensor,
            targets_tensor,
            data_df,
            subject_channel_counts,
        ) = raw_dataset.get_data_for_lag(lag)

        print(f"neural_tensor shape: {neural_tensor.shape}")
        models, histories, cv_results = train_decoding_model(
            neural_tensor,
            targets_tensor,
            data_df,
            model_spec,
            task_name,
            task_config,
            lag,
            training_params=training_params,
            checkpoint_dir=os.path.join(checkpoint_dir, f"lag_{lag}"),
            write_to_tensorboard=write_to_tensorboard,
            tensorboard_dir=tensorboard_dir,
            subject_channel_counts=subject_channel_counts,
        )

        # Aggregate metrics
        lag_metrics = {}
        lag_metrics["lags"] = lag  # lag information first

        fold_nums = cv_results.get("fold_nums", None)
        for metric, values in cv_results.items():
            if metric == "fold_nums":
                continue
            if len(values) > 0:
                # 1. 기존: 평균과 표준편차 저장
                lag_metrics[f"{metric}_mean"] = np.mean(values)
                lag_metrics[f"{metric}_std"] = np.std(values)

                # 2. Add: Individual values for each Fold (e.g., test_acc_fold_0, test_acc_fold_1 ...)
                for i, val in enumerate(values):
                    fold_num = (
                        fold_nums[i]
                        if (fold_nums is not None and i < len(fold_nums))
                        else (i + 1)
                    )
                    lag_metrics[f"{metric}_fold_{fold_num}"] = val
            else:
                lag_metrics[f"{metric}_mean"] = np.nan
                lag_metrics[f"{metric}_std"] = np.nan
        # ---------------------------------------------------------

        # Append new row to existing DataFrame and write to file
        existing_df = pd.concat(
            [existing_df, pd.DataFrame([lag_metrics])], ignore_index=True
        )
        existing_df.to_csv(filename, index=False)


### below : are stuff for feature caching!
def extract_features_for_caching(model, loader, device, subject_channel_counts=None):
    model.eval()
    if not hasattr(model, "encode_features"):
        raise NotImplementedError(
            "Feature caching requires the model to implement "
            f"encode_features(...). Got model: {model.__class__.__name__}"
        )
    if subject_channel_counts is not None and len(subject_channel_counts) <= 1:
        raise ValueError(
            "per_subject_feature_concat requires multiple subjects. "
            f"Got subject_channel_counts={subject_channel_counts}"
        )

    # feature aggregation
    all_features, input_dicts, y_bs = [], [], []
    with torch.no_grad():
        for batch_data in loader:
            Xb, inputs_dict, y_b = batch_data
            Xb = Xb.to(device)
            inputs_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in inputs_dict.items()
            }
            if subject_channel_counts is None:
                features = model.encode_features(Xb, **inputs_dict)
            else:
                subject_chunks = torch.split(Xb, subject_channel_counts, dim=1)
                coord_chunks = {}
                for coord_key in ("xyz_id", "lip_coords"):
                    if coord_key in inputs_dict and torch.is_tensor(
                        inputs_dict[coord_key]
                    ):
                        coord_chunks[coord_key] = torch.split(
                            inputs_dict[coord_key], subject_channel_counts, dim=1
                        )

                subject_embeddings = []
                for s_idx, chunk in enumerate(subject_chunks):
                    sub_kwargs = {
                        coord_key: chunks[s_idx]
                        for coord_key, chunks in coord_chunks.items()
                    }
                    subject_embeddings.append(model.encode_features(chunk, **sub_kwargs))
                features = torch.cat(subject_embeddings, dim=-1)
            all_features.append(features)
            input_dicts.append({} if subject_channel_counts is not None else inputs_dict)
            y_bs.append(y_b)
    return torch.cat(all_features, dim=0), input_dicts, torch.cat(y_bs, dim=0)


def extract_per_subject_concat_features(model, loader, subject_channel_counts, device):
    """Extract features per-subject, then concatenate across subjects.

    For each batch, splits the channel dimension by subject, runs each subject's
    data through the model independently, and concatenates the resulting embeddings.

    Returns:
        concat_features: [n_samples, n_subjects * embed_dim]
        input_dicts: list of input dicts (empty dicts, since features replace raw input)
        targets: [n_samples]
    """
    model.eval()
    if not hasattr(model, "encode_features"):
        raise NotImplementedError(
            "per_subject_feature_concat requires the model to implement "
            f"encode_features(...). Got model: {model.__class__.__name__}"
        )
    all_features, y_bs = [], []
    with torch.no_grad():
        for batch_data in loader:
            Xb, inputs_dict, y_b = batch_data
            Xb = Xb.to(device)

            # Split channel dimension by subject
            # Xb shape: [batch, total_channels, ...] (3D for raw, 4D for STFT)
            subject_chunks = torch.split(Xb, subject_channel_counts, dim=1)

            # Split coordinate tensors by subject if present
            # DIVER uses xyz_id, POPT uses lip_coords — both are [batch, total_channels, 3]
            coord_chunks = {}
            for coord_key in ("xyz_id", "lip_coords"):
                if coord_key in inputs_dict and torch.is_tensor(inputs_dict[coord_key]):
                    coord_tensor = inputs_dict[coord_key].to(device)
                    coord_chunks[coord_key] = torch.split(
                        coord_tensor, subject_channel_counts, dim=1
                    )

            subject_embeddings = []
            for s_idx, chunk in enumerate(subject_chunks):
                sub_kwargs = {}
                for coord_key, chunks in coord_chunks.items():
                    sub_kwargs[coord_key] = chunks[s_idx]
                emb = model.encode_features(chunk, **sub_kwargs)
                subject_embeddings.append(emb)

            # Concatenate per-subject embeddings: [batch, n_subjects * embed_dim]
            concat_emb = torch.cat(subject_embeddings, dim=-1)
            all_features.append(concat_emb)
            y_bs.append(y_b)
    concat_features = torch.cat(all_features, dim=0)
    n_subjects = len(subject_channel_counts)
    embed_dim = concat_features.shape[-1] // n_subjects
    print(
        f"Per-subject-concat features: {n_subjects} subjects x {embed_dim}d = {concat_features.shape[-1]}d total"
    )
    # Return empty input_dicts (features replace raw input, no extra kwargs needed)
    empty_dicts = [{} for _ in range(len(all_features))]
    return concat_features, empty_dicts, torch.cat(y_bs, dim=0)


def _merge_input_dicts(batch_dicts):
    if not batch_dicts:
        return {}
    merged = {}
    for key in batch_dicts[0].keys():
        vals = [d[key] for d in batch_dicts if torch.is_tensor(d[key])]
        merged[key] = torch.cat(vals, dim=0) if vals else [d[key] for d in batch_dicts]
    return merged


def generate_loaders_from_features(
    features, input_dicts, y_bs, batch_size, shuffle=False
):
    def _make_loader(feat, inp, y, shuffle):
        if isinstance(inp, list):
            inp = _merge_input_dicts(inp)
        ds = NeuralDictDataset(feat, inp, y)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

    if isinstance(features, dict):
        return {
            phase: _make_loader(
                features[phase],
                input_dicts[phase],
                y_bs[phase],
                shuffle=shuffle,
            )
            for phase in features
        }

    return _make_loader(features, input_dicts, y_bs, shuffle=shuffle)


class SqueezeWrapper(nn.Module):
    def __init__(self, feature_head: nn.Module, output_dim=None):
        super().__init__()
        self.feature_head = feature_head
        self.output_dim = output_dim

    def forward(self, x, **kwargs):
        out = self.feature_head(x, **kwargs)
        if self.output_dim == 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out


class MakeIgnoreKwargsDuringForward(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x, **kwargs):
        return self.module(x)


class CachedFeatureModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.output_dim = getattr(model, "output_dim", None)

    def forward(self, x, **kwargs):
        return self.model.forward_from_features(x, **kwargs)

    def save_checkpoint(self, path):
        if hasattr(self.model, "save_checkpoint") and callable(
            getattr(self.model, "save_checkpoint")
        ):
            return self.model.save_checkpoint(path)
        return torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, path):
        if hasattr(self.model, "load_checkpoint") and callable(
            getattr(self.model, "load_checkpoint")
        ):
            return self.model.load_checkpoint(path)
        return self.model.load_state_dict(torch.load(path, map_location="cpu"))


def check_model_train_eval_and_requires_grads(
    model: nn.Module, print_requires_grad_params=False
):
    print(f"Model is in training mode: {model.training}")
    num_params_requires_grad = sum(1 for p in model.parameters() if p.requires_grad)

    print(
        "Parameter tensors requiring grad: "
        f"{num_params_requires_grad} out of "
        f"{sum(1 for p in model.parameters())} total parameters"
    )
    if print_requires_grad_params:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter '{name}' requires grad")
