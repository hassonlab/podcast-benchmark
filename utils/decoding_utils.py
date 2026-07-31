from typing import Optional
import gc
import os
import math
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader
from mup import MuAdamW

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
import time


def log_metrics_to_tensorboard(writer, metrics, model_name, phase, step):
    """
    Log metrics to TensorBoard.

    Args:
        writer: TensorBoard SummaryWriter instance
        metrics: Dict of metrics to log (e.g., {"mse": 0.5, "corr": 0.8} or {"train_mse": 0.5, "val_mse": 0.6})
                 Can be None, in which case nothing is logged.
        model_name: Name/namespace for the model (e.g., "model")
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
            output_dim = model_params.get(
                "output_dim", model_params.get("embedding_dim")
            )
            if output_dim == 1:
                num_classes = 2
            else:
                num_classes = output_dim
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
    factor = float(params.get("factor", 0.5))
    patience = int(params.get("patience", 10))
    min_lr = float(params.get("min_lr", 1e-6))

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


def _maybe_shuffle_training_targets(
    target_splits, training_params: TrainingParams, fold: int
):
    """Shuffle only a fold's training labels for the sanity-check control."""
    if not training_params.shuffle_targets:
        return target_splits

    print(
        "WARNING: Shuffling training targets for sanity check. "
        "Model should perform poorly."
    )
    rng = np.random.default_rng(training_params.random_seed + fold * 9173)
    shuffle_indices = torch.as_tensor(
        rng.permutation(len(target_splits["train"])), dtype=torch.long
    )
    shuffled = dict(target_splits)
    shuffled["train"] = target_splits["train"][shuffle_indices]
    return shuffled


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


def _init_cv_results(
    metric_names,
    task_name: str,
    training_params: TrainingParams,
    include_embedding_metrics: bool = True,
):
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
    if include_embedding_metrics and task_name == "word_embedding_decoding_task":
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


def _fold_target_normalization_stats(Y, tr_idx, training_params: TrainingParams):
    if not training_params.normalize_targets:
        return None
    Y_train = Y[tr_idx]
    y_mean = Y_train.mean(dim=0, keepdim=True)
    y_std = Y_train.std(dim=0, keepdim=True)
    y_std = torch.where(y_std < 1e-6, torch.ones_like(y_std), y_std)
    return y_mean, y_std


def _normalize_full_targets(Y, tr_idx, va_idx, te_idx, training_params: TrainingParams):
    target_splits = _normalize_fold_targets(Y, tr_idx, va_idx, te_idx, training_params)
    if not training_params.normalize_targets:
        return Y

    normalized = torch.empty_like(Y)
    for phase, indices in {
        "train": tr_idx,
        "val": va_idx,
        "test": te_idx,
    }.items():
        normalized[_as_index_tensor(indices)] = target_splits[phase]
    return normalized


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


def _as_index_tensor(indices, device=None):
    if isinstance(indices, np.ndarray):
        return torch.as_tensor(indices, dtype=torch.long, device=device)
    if torch.is_tensor(indices):
        return indices.to(device=device, dtype=torch.long)
    return torch.tensor(indices, dtype=torch.long, device=device)


def _get_fold_indices_for_length(
    n_examples: int,
    data_df: pd.DataFrame,
    task_config: TaskConfig,
    training_params: TrainingParams,
):
    dummy = torch.empty((n_examples, 1), dtype=torch.float32)
    return _get_fold_indices(dummy, data_df, task_config, training_params)


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
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=t_max, eta_min=eta_min
            )
        raise ValueError(
            f"Unknown lr_scheduler: {training_params.lr_scheduler}. "
            "Supported: None, 'cosine_annealing'"
        )

    return create_lr_scheduler(optimizer, training_params)


def _step_scheduler_after_optimizer_update(scheduler):
    if scheduler is None or isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        return
    scheduler.step()


def _step_scheduler_after_validation(scheduler, metric_value):
    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric_value)


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
        k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs_dict.items()
    }
    yb = yb.to(device)
    return Xb, inputs_dict, yb


def _accumulate_batch_metrics(sums, batch_metrics):
    for name, val in batch_metrics.items():
        if sums[name] is None:
            sums[name] = val
        else:
            sums[name] += val


def _init_streaming_corr_state(device):
    return {
        "n": 0,
        "sum_pred": torch.tensor(0.0, device=device),
        "sum_true": torch.tensor(0.0, device=device),
        "sum_pred_sq": torch.tensor(0.0, device=device),
        "sum_true_sq": torch.tensor(0.0, device=device),
        "sum_prod": torch.tensor(0.0, device=device),
    }


def _update_streaming_corr_state(state, pred, true):
    pred = pred.detach().to(dtype=torch.float32).reshape(-1)
    true = true.detach().to(dtype=torch.float32).reshape(-1)

    state["n"] += pred.numel()
    state["sum_pred"] += pred.sum()
    state["sum_true"] += true.sum()
    state["sum_pred_sq"] += torch.sum(pred * pred)
    state["sum_true_sq"] += torch.sum(true * true)
    state["sum_prod"] += torch.sum(pred * true)


def _compute_streaming_corr(state):
    n = state["n"]
    if n < 2:
        return 0.0

    n_tensor = torch.tensor(float(n), device=state["sum_pred"].device)
    cov = state["sum_prod"] / n_tensor
    cov = cov - (state["sum_pred"] / n_tensor) * (state["sum_true"] / n_tensor)
    pred_var = state["sum_pred_sq"] / n_tensor
    pred_var = pred_var - (state["sum_pred"] / n_tensor) ** 2
    true_var = state["sum_true_sq"] / n_tensor
    true_var = true_var - (state["sum_true"] / n_tensor) ** 2

    if pred_var <= 0 or true_var <= 0:
        return 0.0

    corr = cov / torch.sqrt(pred_var * true_var)
    return corr.item() if torch.isfinite(corr) else 0.0


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

    accumulate_corr = "corr" in metric_names
    batch_metric_fns = {
        name: fn
        for name, fn in all_fns.items()
        if not (accumulate_corr and name == "corr")
    }
    batch_metric_names = batch_metric_fns.keys()
    corr_state = _init_streaming_corr_state(device) if accumulate_corr else None

    sums = {
        name: None if name == "confusion_matrix" else 0.0 for name in batch_metric_names
    }
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
                _step_scheduler_after_optimizer_update(scheduler)
                optimizer.zero_grad()
        else:
            with torch.no_grad():
                out = model(Xb, **inputs_dict)
                loss = compute_loss(out, yb, training_params, all_fns)

        if accumulate_corr:
            _update_streaming_corr_state(corr_state, out, yb)

        batch_metrics = compute_all_metrics(out, yb, batch_metric_fns, model_params)
        _accumulate_batch_metrics(sums, batch_metrics)

        if torch.is_tensor(loss):
            loss = loss.detach().mean().item()
        sums["loss"] += loss

    result = {
        name: sums[name] if name == "confusion_matrix" else sums[name] / len(loader)
        for name in sums
    }

    if accumulate_corr:
        result["corr"] = _compute_streaming_corr(corr_state)

    if "cross_entropy" in result:
        result["perplexity"] = np.exp(result["cross_entropy"])

    return result


def _save_checkpoint(model, model_path):
    if hasattr(model, "save_checkpoint") and callable(
        getattr(model, "save_checkpoint")
    ):
        model.save_checkpoint(model_path)
    else:
        torch.save(model.state_dict(), model_path)


def _load_checkpoint(model, model_path):
    if hasattr(model, "load_checkpoint") and callable(
        getattr(model, "load_checkpoint")
    ):
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

        _step_scheduler_after_validation(scheduler, cur)

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


def _log_fold_tensorboard_results(writer, test_mets, fold):
    if writer is None:
        return

    log_metrics_to_tensorboard(writer, test_mets, "model", "test", fold)
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


def _collect_model_outputs(model, loader, device):
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            Xb, inputs_dict, _ = _move_batch_to_device(batch_data, device)
            predictions.append(model(Xb, **inputs_dict).detach().cpu())
    return torch.cat(predictions, dim=0)


def _loader_output_indices(loader, split_indices):
    if hasattr(loader, "ordered_indices"):
        return np.asarray(loader.ordered_indices(), dtype=np.int64)
    return np.asarray(split_indices, dtype=np.int64)


def _prediction_record(
    model,
    loader,
    device,
    data_df,
    original_targets,
    te_idx,
    fold,
    normalization_stats,
):
    predictions = _collect_model_outputs(model, loader, device)
    output_indices = _loader_output_indices(loader, te_idx)
    if len(output_indices) != len(predictions):
        raise RuntimeError("Prediction rows do not align with test sample indices")

    if normalization_stats is not None:
        y_mean, y_std = normalization_stats
        predictions = predictions * y_std.cpu() + y_mean.cpu()

    rows = data_df.iloc[output_indices]
    raw_targets = original_targets[
        torch.as_tensor(output_indices, dtype=torch.long)
    ].detach().cpu()
    record = {
        "fold": int(fold),
        "sample_id": rows["sample_id"].astype(str).to_numpy(),
        "start": rows["start"].to_numpy(dtype=np.float64),
        "prediction": predictions.numpy().astype(np.float32, copy=False),
        "target": raw_targets.numpy().astype(np.float32, copy=False),
    }
    if normalization_stats is not None:
        record["target_mean"] = normalization_stats[0].detach().cpu().numpy()
        record["target_std"] = normalization_stats[1].detach().cpu().numpy()
    return record


def _create_compressed_dataset(group, name, values):
    values = np.asarray(values)
    kwargs = {}
    if values.size:
        kwargs = {"compression": "gzip", "shuffle": True}
    group.create_dataset(name, data=values, **kwargs)


def _write_prediction_artifact(
    filename, lag, task_name, records, null_repetition=None
):
    """Commit one completed lag of out-of-fold predictions to HDF5."""
    lag_group_name = f"lag_{int(lag)}"
    final_group_name = (
        lag_group_name
        if null_repetition is None
        else f"{lag_group_name}/null_repetition_{int(null_repetition)}"
    )
    pending_group_name = (
        f"_pending_lag_{int(lag)}"
        if null_repetition is None
        else f"_pending_lag_{int(lag)}_null_repetition_{int(null_repetition)}"
    )
    string_dtype = h5py.string_dtype(encoding="utf-8")

    with h5py.File(filename, "a") as artifact:
        artifact.attrs["schema_version"] = 1 if null_repetition is None else 2
        artifact.attrs["task_name"] = task_name
        if null_repetition is not None:
            artifact.require_group(lag_group_name).attrs["lag_ms"] = int(lag)
        for stale_name in (pending_group_name, final_group_name):
            if stale_name in artifact:
                del artifact[stale_name]
        lag_group = artifact.create_group(pending_group_name)
        lag_group.attrs["lag_ms"] = int(lag)
        if null_repetition is not None:
            lag_group.attrs["null_repetition"] = int(null_repetition)

        for record in records:
            fold_group = lag_group.create_group(f"fold_{record['fold']}")
            fold_group.attrs["fold"] = record["fold"]
            fold_group.attrs["normalized_during_training"] = (
                "target_mean" in record
            )
            fold_group.create_dataset(
                "sample_id",
                data=np.asarray(record["sample_id"], dtype=object),
                dtype=string_dtype,
            )
            _create_compressed_dataset(fold_group, "start", record["start"])
            _create_compressed_dataset(
                fold_group, "prediction", record["prediction"]
            )
            _create_compressed_dataset(fold_group, "target", record["target"])
            if "target_mean" in record:
                _create_compressed_dataset(
                    fold_group, "target_mean", record["target_mean"]
                )
                _create_compressed_dataset(
                    fold_group, "target_std", record["target_std"]
                )

        artifact.move(pending_group_name, final_group_name)
        artifact.flush()


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
            if not vals:
                continue
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
    if not model_spec.constructor_name:
        raise ValueError("model_spec.constructor_name is required for neural training")
    os.makedirs(checkpoint_dir, exist_ok=True)

    fold_indices = _get_fold_indices(neural_data, data_df, task_config, training_params)
    fold_indices, fold_nums = _select_requested_folds(fold_indices, training_params)
    _maybe_visualize_fold_distribution(Y, fold_indices, task_name, lag, training_params)

    all_fns = setup_metrics_and_loss(training_params)
    metric_names = all_fns.keys()
    cv_results, embedding_metrics = _init_cv_results(
        metric_names,
        task_name,
        training_params,
        include_embedding_metrics=True,
    )

    models, histories = [], []
    prediction_records = []
    conf_matrices = {}

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
        target_splits = _maybe_shuffle_training_targets(
            target_splits, training_params, fold
        )
        loaders = _build_fold_loaders(
            neural_data,
            data_df,
            task_config,
            split_indices,
            target_splits,
            training_params,
        )

        model, optimizer, scheduler = _build_model_optimizer_scheduler(
            model_spec,
            lag,
            fold,
            loaders,
            training_params,
            device,
        )

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

        _log_fold_tensorboard_results(writer, test_mets, fold)
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
        if training_params.save_test_predictions:
            prediction_records.append(
                _prediction_record(
                    model,
                    loaders["test"],
                    device,
                    data_df,
                    Y,
                    te_idx,
                    fold,
                    _fold_target_normalization_stats(
                        Y, tr_idx, training_params
                    ),
                )
            )

        models.append(model)
        histories.append(history)

        if plot_results:
            plot_training_history(history, fold=fold)

    _print_main_cv_summary(cv_results, metric_names, conf_matrices, embedding_metrics)

    if plot_results:
        plot_cv_results(cv_results)

    return models, histories, cv_results, prediction_records


def train_decoding_model_chunked(
    chunk_store,
    model_spec: ModelSpec,
    task_name: str,
    task_config: TaskConfig,
    lag: int,
    training_params: TrainingParams,
    checkpoint_dir: str,
    plot_results: bool = False,
    write_to_tensorboard: bool = False,
    tensorboard_dir: str = "event_logs",
):
    if not model_spec.constructor_name:
        raise ValueError("model_spec.constructor_name is required for neural training")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    Y = chunk_store.targets
    fold_indices = _get_fold_indices_for_length(
        len(Y), chunk_store.data_df, task_config, training_params
    )
    fold_indices, fold_nums = _select_requested_folds(fold_indices, training_params)
    _maybe_visualize_fold_distribution(Y, fold_indices, task_name, lag, training_params)

    all_fns = setup_metrics_and_loss(training_params)
    metric_names = all_fns.keys()
    cv_results, embedding_metrics = _init_cv_results(
        metric_names,
        task_name,
        training_params,
        include_embedding_metrics=True,
    )

    models, histories = [], []
    prediction_records = []
    conf_matrices = {}

    for fold, (tr_idx, va_idx, te_idx) in zip(fold_nums, fold_indices):
        print(f"Fold {fold}")
        print(f"Train indices: {tr_idx}")
        print(f"Validation indices: {va_idx}")
        print(f"Test indices: {te_idx}")
        print(f"Train size: {len(tr_idx)}")
        print(f"Validation size: {len(va_idx)}")
        print(f"Test size: {len(te_idx)}")
        cv_results["fold_nums"].append(fold)

        model_path = os.path.join(checkpoint_dir, f"best_model_fold{fold}.pt")
        writer = _create_tensorboard_writer(
            write_to_tensorboard, tensorboard_dir, lag, fold
        )

        split_indices = {"train": tr_idx, "val": va_idx, "test": te_idx}
        target_splits = _normalize_fold_targets(
            Y, tr_idx, va_idx, te_idx, training_params
        )
        target_splits = _maybe_shuffle_training_targets(
            target_splits, training_params, fold
        )
        full_targets = torch.empty_like(Y)
        for phase, indices in split_indices.items():
            full_targets[_as_index_tensor(indices)] = target_splits[phase]
        loaders = {
            phase: chunk_store.get_loader(
                indices,
                task_config,
                full_targets,
                training_params.batch_size,
                shuffle=(phase == "train"),
                seed=training_params.random_seed + fold * 9173,
            )
            for phase, indices in split_indices.items()
        }

        model, optimizer, scheduler = _build_model_optimizer_scheduler(
            model_spec,
            lag,
            fold,
            loaders,
            training_params,
            device,
        )

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

        _log_fold_tensorboard_results(writer, test_mets, fold)
        _maybe_compute_word_embedding_metrics(
            cv_results,
            embedding_metrics,
            loaders,
            model,
            device,
            chunk_store.data_df,
            task_config,
            tr_idx,
            te_idx,
            training_params,
        )
        if training_params.save_test_predictions:
            prediction_records.append(
                _prediction_record(
                    model,
                    loaders["test"],
                    device,
                    chunk_store.data_df,
                    Y,
                    te_idx,
                    fold,
                    _fold_target_normalization_stats(
                        Y, tr_idx, training_params
                    ),
                )
            )

        models.append(model)
        histories.append(history)

        if plot_results:
            plot_training_history(history, fold=fold)

    _print_main_cv_summary(cv_results, metric_names, conf_matrices, embedding_metrics)

    if plot_results:
        plot_cv_results(cv_results)

    return models, histories, cv_results, prediction_records


def _chunked_preprocessing_value(chunked_params, name, default=None):
    if isinstance(chunked_params, dict):
        return chunked_params.get(name, default)
    return getattr(chunked_params, name, default)


def _release_accelerator_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _contains_enabled_random_init(value) -> bool:
    """Find an enabled random-init flag in nested model/preprocessor config."""
    if isinstance(value, ModelSpec):
        return value.random_init or any(
            _contains_enabled_random_init(spec) for spec in value.sub_models.values()
        )
    if isinstance(value, dict):
        if value.get("random_init") is True:
            return True
        return any(_contains_enabled_random_init(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_enabled_random_init(item) for item in value)
    return False


def _preprocessor_params_for_null_seed(value, seed):
    """Copy params and salt caches that wrap a random foundation model."""
    copied = deepcopy(value)

    def add_seed_marker(item):
        if isinstance(item, dict):
            foundation_spec = item.get("foundation_model_spec")
            if foundation_spec is not None and _contains_enabled_random_init(
                foundation_spec
            ):
                item["_null_repetition_seed"] = seed
            for nested in item.values():
                add_seed_marker(nested)
        elif isinstance(item, list):
            for nested in item:
                add_seed_marker(nested)

    add_seed_marker(copied)
    return copied


def _set_null_repetition_seed(seed: int, cudnn_deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if cudnn_deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _validate_null_repetitions(lags, model_spec, training_params, data_params):
    repetitions = training_params.num_null_repetitions
    if isinstance(repetitions, bool) or not isinstance(repetitions, int):
        raise ValueError("num_null_repetitions must be a positive integer")
    if repetitions < 1:
        raise ValueError("num_null_repetitions must be a positive integer")

    model_random_init = _contains_enabled_random_init(model_spec)
    preprocessor_random_init = _contains_enabled_random_init(
        data_params.preprocessor_params
    )
    if repetitions > 1:
        if len(lags) != 1:
            raise ValueError(
                "num_null_repetitions > 1 requires exactly one resolved lag"
            )
        if not (
            training_params.shuffle_targets
            or model_random_init
            or preprocessor_random_init
        ):
            raise ValueError(
                "num_null_repetitions > 1 requires shuffle_targets and/or "
                "an enabled random_init ModelSpec"
            )
    return repetitions, preprocessor_random_init


def _aggregate_lag_metrics(lag, cv_results, repetition=None, seed=None):
    lag_metrics = {"lags": lag}
    if repetition is not None:
        lag_metrics["null_repetition"] = repetition
        lag_metrics["null_seed"] = seed

    fold_nums = cv_results.get("fold_nums")
    for metric, values in cv_results.items():
        if metric == "fold_nums":
            continue
        if len(values) > 0:
            lag_metrics[f"{metric}_mean"] = np.mean(values)
            lag_metrics[f"{metric}_std"] = np.std(values)
            for i, val in enumerate(values):
                fold_num = (
                    fold_nums[i]
                    if fold_nums is not None and i < len(fold_nums)
                    else i + 1
                )
                lag_metrics[f"{metric}_fold_{fold_num}"] = val
        else:
            lag_metrics[f"{metric}_mean"] = np.nan
            lag_metrics[f"{metric}_std"] = np.nan
    return lag_metrics


def _write_null_summary(existing_df, output_dir):
    repeated_df = existing_df.dropna(subset=["null_repetition"])
    summary = {"lags": repeated_df["lags"].iloc[0], "num_null_repetitions": len(repeated_df)}
    for column in repeated_df.columns:
        if not column.endswith("_mean"):
            continue
        values = pd.to_numeric(repeated_df[column], errors="coerce")
        summary[f"{column}_null_mean"] = values.mean()
        summary[f"{column}_null_std"] = values.std(ddof=0)
    pd.DataFrame([summary]).to_csv(
        os.path.join(output_dir, "null_summary.csv"), index=False
    )


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
    per_raw_event_times=None,
):
    data_params = task_config.data_params
    os.makedirs(output_dir, exist_ok=True)
    lags = list(lags)
    repetitions, preprocessor_random_init = _validate_null_repetitions(
        lags, model_spec, training_params, data_params
    )
    repeated_run = repetitions > 1

    if training_params.save_test_predictions and task_name == "llm_decoding_task":
        raise ValueError(
            "save_test_predictions is not supported for llm_decoding_task because "
            "full token-vocabulary logits are prohibitively large"
        )

    filename = os.path.join(output_dir, f"lag_performance.csv")
    prediction_filename = os.path.join(output_dir, "test_predictions.h5")

    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
    else:
        existing_df = pd.DataFrame()

    from utils.dataset import RawNeuralDataset

    raw_dataset = RawNeuralDataset(
        raws,
        task_df,
        data_params.window_width,
        preprocessing_fns,
        data_params.preprocessor_params,
        per_raw_event_times=per_raw_event_times,
        include_sample_ids=training_params.save_test_predictions,
    )

    chunked_params = getattr(data_params, "chunked_preprocessing", None)
    use_chunks = chunked_params is not None and _chunked_preprocessing_value(
        chunked_params, "enabled", False
    )

    def prepare_lag(lag):
        if use_chunks:
            store = raw_dataset.build_preprocessed_chunks(
                lag,
                num_chunks=_chunked_preprocessing_value(chunked_params, "num_chunks", 1),
                cache_dir=_chunked_preprocessing_value(
                    chunked_params, "cache_dir", ".cache/preprocessed_chunks"
                ),
            )
            print(
                "chunked preprocessing rows: "
                f"{len(store.data_df)}, chunks: {len(store.chunk_paths)}"
            )
            return store
        tensors = raw_dataset.get_data_for_lag(lag)
        print(f"neural_tensor shape: {tensors[0].shape}")
        return tensors

    for lag in lags:
        if not repeated_run and "lags" in existing_df and lag in existing_df["lags"].tolist():
            print(f"Lag {lag} already done, skipping...")
            continue

        print("=" * 60)
        print("running lag:", lag)
        print("=" * 60)
        reusable_prepared = None
        prepared = None
        models = histories = cv_results = prediction_records = lag_metrics = None
        try:
            if not preprocessor_random_init:
                _set_null_repetition_seed(
                    training_params.random_seed,
                    training_params.cudnn_deterministic,
                )
                reusable_prepared = prepare_lag(lag)

            for repetition_index in range(repetitions):
                repetition = repetition_index + 1
                if repeated_run and not existing_df.empty and {
                    "lags", "null_repetition"
                }.issubset(existing_df.columns):
                    completed = (
                        (existing_df["lags"] == lag)
                        & (existing_df["null_repetition"] == repetition)
                    ).any()
                    if completed:
                        print(
                            f"Lag {lag}, null repetition {repetition} already done, skipping..."
                        )
                        continue

                repetition_seed = training_params.random_seed + repetition_index
                repetition_training_params = deepcopy(training_params)
                repetition_training_params.random_seed = repetition_seed
                _set_null_repetition_seed(
                    repetition_seed, training_params.cudnn_deterministic
                )
                if preprocessor_random_init:
                    raw_dataset.preprocessor_params = _preprocessor_params_for_null_seed(
                        data_params.preprocessor_params, repetition_seed
                    )
                prepared = (
                    prepare_lag(lag) if preprocessor_random_init else reusable_prepared
                )
                repetition_component = (
                    f"null_repetition_{repetition}" if repeated_run else None
                )
                repetition_checkpoint_dir = os.path.join(checkpoint_dir, f"lag_{lag}")
                repetition_tensorboard_dir = tensorboard_dir
                if repetition_component:
                    repetition_checkpoint_dir = os.path.join(
                        repetition_checkpoint_dir, repetition_component
                    )
                    repetition_tensorboard_dir = os.path.join(
                        tensorboard_dir, repetition_component
                    )

                try:
                    if use_chunks:
                        models, histories, cv_results, prediction_records = (
                            train_decoding_model_chunked(
                                prepared,
                                model_spec,
                                task_name,
                                task_config,
                                lag,
                                training_params=repetition_training_params,
                                checkpoint_dir=repetition_checkpoint_dir,
                                write_to_tensorboard=write_to_tensorboard,
                                tensorboard_dir=repetition_tensorboard_dir,
                            )
                        )
                    else:
                        neural_tensor, targets_tensor, data_df, channel_counts = prepared
                        models, histories, cv_results, prediction_records = train_decoding_model(
                            neural_tensor,
                            targets_tensor,
                            data_df,
                            model_spec,
                            task_name,
                            task_config,
                            lag,
                            training_params=repetition_training_params,
                            checkpoint_dir=repetition_checkpoint_dir,
                            write_to_tensorboard=write_to_tensorboard,
                            tensorboard_dir=repetition_tensorboard_dir,
                            subject_channel_counts=channel_counts,
                        )

                    if training_params.save_test_predictions:
                        _write_prediction_artifact(
                            prediction_filename,
                            lag,
                            task_name,
                            prediction_records,
                            null_repetition=repetition if repeated_run else None,
                        )

                    lag_metrics = _aggregate_lag_metrics(
                        lag,
                        cv_results,
                        repetition=repetition if repeated_run else None,
                        seed=repetition_seed if repeated_run else None,
                    )
                    existing_df = pd.concat(
                        [existing_df, pd.DataFrame([lag_metrics])], ignore_index=True
                    )
                    existing_df.to_csv(filename, index=False)
                    if repeated_run:
                        _write_null_summary(existing_df, output_dir)
                finally:
                    if preprocessor_random_init and use_chunks:
                        prepared.cleanup()
        finally:
            if reusable_prepared is not None and use_chunks:
                reusable_prepared.cleanup()

        # Do not retain fold models and lag-sized tensors while preprocessing the
        # next lag. Foundation feature extraction temporarily loads another large
        # model, so overlapping these objects can exceed host or accelerator RAM.
        models = histories = cv_results = prediction_records = lag_metrics = None
        prepared = reusable_prepared = None
        _release_accelerator_memory()


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
