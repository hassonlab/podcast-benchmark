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
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    if training_params.shuffle_targets:
        print(
            "WARNING: Shuffling targets for sanity check. Model should perform poorly."
        )
        # Set seed for reproducibility
        rng = np.random.default_rng(training_params.random_seed)
        shuffle_indices = rng.permutation(len(Y))
        Y = Y[shuffle_indices]

    # 3. Get fold indices
    if training_params.fold_type == "sequential_folds":
        fold_indices = get_sequential_folds(
            neural_data, num_folds=training_params.n_folds
        )
    elif training_params.fold_type == "zero_shot_folds":
        fold_indices = get_zero_shot_folds(
            data_df[task_config.data_params.word_column].values,
            num_folds=training_params.n_folds,
        )
    else:
        raise ValueError(f"Unknown fold_type: {training_params.fold_type}")

    # 3.25. Optionally restrict to specific folds
    # Internal fold numbering in this file is 1..n_folds (see enumerate(start=1) below).
    fold_nums_all = list(range(1, len(fold_indices) + 1))  # [1..n]
    fold_ids = getattr(training_params, "fold_ids", None)
    if fold_ids is not None:
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
        selected_fold_nums = list(fold_ids)

        # Keep order as provided by user; de-duplicate while preserving order
        seen = set()
        selected_fold_nums = [
            k for k in selected_fold_nums if not (k in seen or seen.add(k))
        ]

        fold_indices = [
            fold_indices[k - 1] for k in selected_fold_nums
        ]  # map to 0-based list index
        fold_nums_all = (
            selected_fold_nums  # now fold_nums_all matches fold_indices order
        )

    # 3.5. Visualize fold distribution if requested
    if training_params.visualize_fold_distribution:
        from utils.analysis_utils import visualize_fold_distribution

        # Convert Y to numpy if it's a tensor
        Y_np = Y.cpu().numpy() if isinstance(Y, torch.Tensor) else Y
        visualize_fold_distribution(Y_np, fold_indices, task_name=task_name, lag=lag)

    # 4. Build a single dict of all metric functions (including loss)
    all_fns = setup_metrics_and_loss(training_params)
    metric_names = all_fns.keys()

    # 5. Initialize CV containers
    phases = ("train", "val", "test")
    # cv_results = {f"{phase}_{name}": [] for phase in phases for name in metric_names}

    cv_results = {
        f"{phase}_{name}": []
        for phase in phases
        for name in metric_names
        if name != "confusion_matrix"
    }

    cv_results["num_epochs"] = []
    cv_results["fold_nums"] = []

    # Hardcode embedding task metrics for now since they need to be handled a bit differently.
    # Clean this up later. Hardcoding for now since generalizing this like other metrics would
    # get complicated.
    is_word_embedding_decoding_task = task_name == "word_embedding_decoding_task"
    if is_word_embedding_decoding_task:
        # Test type is split between "word" and "occ" where word is averaged over
        # each time a word occurs and occ is per-each occurence of the word so is
        # more difficult and depends on contextual embeddings.
        embedding_metrics = [
            "test_word_avg_auc_roc",
            "test_word_train_weighted_auc_roc",
            "test_word_test_weighted_auc_roc",
            "test_word_perplexity",
            "test_occurence_perplexity",
        ]

        # Top-K metrics.
        for k_val in training_params.top_k_thresholds:
            for test_type in ["word", "occurence"]:
                embedding_metrics.append(f"test_{test_type}_top_{k_val}")

        for metric in embedding_metrics:
            cv_results[metric] = []

    models, histories = [], []

    # Store baseline results across folds
    logistic_regression_results = []
    linear_regression_results = []
    ridge_regression_results = []

    def run_epoch(model, loader, optimizer=None):
        """
        If optimizer is provided: does a training pass.
        Otherwise: does an eval pass.
        Returns a dict { metric_name: average_value }.
        """
        is_train = optimizer is not None
        if is_train:
            model.train()
        else:
            model.eval()

        # Initialize sums with None for confusion matrix, 0.0 for others
        sums = {}
        for name in metric_names:
            sums[name] = None if name == "confusion_matrix" else 0.0
        sums["loss"] = 0.0

        grad_steps = training_params.grad_accumulation_steps

        if is_train:
            optimizer.zero_grad()

        for i, batch_data in enumerate(loader):
            # NeuralDictDataset returns (X, inputs_dict, Y)
            Xb, inputs_dict, yb = batch_data
            Xb = Xb.to(device)
            # Move all input tensors to device (handles data_info_list and other inputs)
            inputs_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in inputs_dict.items()
            }
            yb = yb.to(device)

            if is_train:
                # Forward pass
                out = model(Xb, **inputs_dict)
                # Loss calculation
                loss = compute_loss(out, yb, training_params, all_fns)
                # Normalize loss to account for gradient accumulation
                loss = loss / grad_steps
                # Backward pass
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
                    # Loss calculation
                    loss = compute_loss(out, yb, training_params, all_fns)

            # Compute all metrics for this batch using the helper function
            batch_metrics = compute_all_metrics(out, yb, all_fns, model_spec.params)

            # Accumulate metrics
            for name, val in batch_metrics.items():
                if sums[name] is None:
                    # First batch - initialize with the value
                    sums[name] = val
                else:
                    # Accumulate (works for both scalars and arrays)
                    sums[name] += val

            # Add loss to sums
            if torch.is_tensor(loss):
                loss = loss.detach().mean().item()
            sums["loss"] += loss

        result = {
            name: (
                sums[name] if name == "confusion_matrix" else sums[name] / len(loader)
            )
            for name in sums
        }

        # Calculate perplexity as derived metric from averaged cross_entropy
        if "cross_entropy" in result:
            result["perplexity"] = np.exp(result["cross_entropy"])

        return result

    # 6. Cross‐val loop
    for fold, (tr_idx, va_idx, te_idx) in zip(fold_nums_all, fold_indices):
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
        cv_results["fold_nums"].append(fold)
        model_path = os.path.join(checkpoint_dir, f"best_model_fold{fold}.pt")

        # TensorBoard writer
        if write_to_tensorboard:
            if not TENSORBOARD_AVAILABLE:
                raise ImportError(
                    "TensorBoard is not available. Please install it with: "
                    "pip install tensorboard"
                )
            tb_path = os.path.join(tensorboard_dir, f"lag_{lag}", f"fold_{fold}")
            writer = SummaryWriter(log_dir=tb_path)

        # Normalize targets if requested (compute stats on training set only)
        if training_params.normalize_targets:
            print("Normalizing targets...")
            Y_train = Y[tr_idx]
            Y_val = Y[va_idx]
            Y_test = Y[te_idx]

            # Compute mean and std on training set only
            y_mean = Y_train.mean(dim=0, keepdim=True)
            y_std = Y_train.std(dim=0, keepdim=True)

            # Prevent division by zero
            y_std = torch.where(y_std < 1e-6, torch.ones_like(y_std), y_std)

            # Apply normalization using training statistics
            Y_train_norm = (Y_train - y_mean) / y_std
            Y_val_norm = (Y_val - y_mean) / y_std
            Y_test_norm = (Y_test - y_mean) / y_std
        else:
            Y_train_norm = Y[tr_idx]
            Y_val_norm = Y[va_idx]
            Y_test_norm = Y[te_idx]

        # DataLoaders - unified path using NeuralDictDataset for all models
        # Model-specific columns (like data_info_list) are added via model_data_getter
        # and included in input_fields, so they flow through automatically
        extra_train_inputs = data_utils.df_columns_to_tensors(
            data_df, task_config.task_specific_config.input_fields, tr_idx
        )
        extra_val_inputs = data_utils.df_columns_to_tensors(
            data_df, task_config.task_specific_config.input_fields, va_idx
        )
        extra_test_inputs = data_utils.df_columns_to_tensors(
            data_df, task_config.task_specific_config.input_fields, te_idx
        )
        datasets = {
            "train": NeuralDictDataset(
                neural_data[tr_idx], extra_train_inputs, Y_train_norm
            ),
            "val": NeuralDictDataset(neural_data[va_idx], extra_val_inputs, Y_val_norm),
            "test": NeuralDictDataset(
                neural_data[te_idx], extra_test_inputs, Y_test_norm
            ),
        }
        loaders = {
            phase: DataLoader(
                ds, batch_size=training_params.batch_size, shuffle=(phase == "train")
            )
            for phase, ds in datasets.items()
        }

        # Train baseline models and compute all metrics
        def train_and_eval_baseline(training_fn, **kwargs):
            model = training_fn(
                neural_data[tr_idx].cpu().numpy(),
                Y_train_norm.cpu().numpy(),
                **kwargs,
            )
            # Prepare data splits for metric computation (use normalized targets)
            X_splits = {
                "train": neural_data[tr_idx].cpu().numpy(),
                "val": neural_data[va_idx].cpu().numpy(),
                "test": neural_data[te_idx].cpu().numpy(),
            }
            Y_splits = {
                "train": Y_train_norm.cpu().numpy(),
                "val": Y_val_norm.cpu().numpy(),
                "test": Y_test_norm.cpu().numpy(),
            }
            # Compute all metrics
            return compute_baseline_metrics(
                model, X_splits, Y_splits, all_fns, model_spec.params
            )

        if training_params.logistic_regression_baseline:
            print("Training logistic regression baseline...")
            logistic_baseline_metrics = train_and_eval_baseline(
                train_logistic_regression
            )
            logistic_regression_results.append(logistic_baseline_metrics)
        if training_params.linear_regression_baseline:
            print("Training linear regression baseline...")
            linear_baseline_metrics = train_and_eval_baseline(train_linear_regression)
            linear_regression_results.append(linear_baseline_metrics)
        if training_params.ridge_regression_baseline:
            print("Training ridge regression baseline...")
            ridge_baseline_metrics = train_and_eval_baseline(
                train_ridge_regression, alpha=training_params.ridge_alpha
            )
            ridge_regression_results.append(ridge_baseline_metrics)

        # Model, optimizer, early‐stop setup
        model = build_model_from_spec(model_spec, lag=lag, fold=fold).to(device)

        if training_params.optimizer == "MuAdamW":
            print("Using MuAdamW optimizer")
            optimizer = MuAdamW(
                model.parameters(),
                lr=float(training_params.learning_rate),
                weight_decay=float(training_params.weight_decay),
            )

        else:
            print("Using AdamW optimizer")
            optimizer = optim.AdamW(
                model.parameters(),
                lr=float(training_params.learning_rate),
                weight_decay=float(training_params.weight_decay),
            )

        # Optional LR scheduler (per optimizer update, not per epoch).
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

        # Create learning rate scheduler if specified
        scheduler = create_lr_scheduler(optimizer, training_params)

        best_val, patience = setup_early_stopping_state(training_params)
        best_epoch = 0

        # per‐fold history (only train & val, for plotting)
        history = {
            f"{phase}_{name}": [] for phase in ("train", "val") for name in metric_names
        }

        if "cross_entropy" in metric_names:
            for phase in ("train", "val"):
                history[f"{phase}_perplexity"] = []
        history["train_loss"] = []
        history["val_loss"] = []
        history["num_epochs"] = None

        loop = tqdm(range(training_params.epochs), desc=f"Lag {lag}, Fold {fold}")
        
        #if feature cache, overwrite loaders and model => not ideal but we wanna do it quickly for now. Clean up later.
        if getattr(model_spec, "feature_cache", False): 
            cache_loader_generation_start_time = time.time()
            loaders = {
                "train" : generate_loaders_from_features(*extract_features_for_caching(model, loaders["train"], device), training_params.batch_size, shuffle=False), #*training already shuffled we assume
                "val" : generate_loaders_from_features(*extract_features_for_caching(model, loaders["val"], device), training_params.batch_size, shuffle=False),
                "test" : generate_loaders_from_features(*extract_features_for_caching(model, loaders["test"], device), training_params.batch_size, shuffle=False),
            }
            print(f"Time taken for feature extraction and loader generation: {time.time() - cache_loader_generation_start_time}")
            model = SqueezeWrapper(feature_head = model.projector, output_dim=model.output_dim).to(device)
        
        loop_start_time = time.time() #! remove later 
        for epoch in loop:
            train_mets = run_epoch(model, loaders["train"], optimizer)
            val_mets = run_epoch(model, loaders["val"])

            # record + TensorBoard
            for name, val in train_mets.items():
                history[f"train_{name}"].append(val)
            if write_to_tensorboard:
                log_metrics_to_tensorboard(writer, train_mets, "model", "train", epoch)

            for name, val in val_mets.items():
                history[f"val_{name}"].append(val)
            if write_to_tensorboard:
                log_metrics_to_tensorboard(writer, val_mets, "model", "val", epoch)

            # early stopping on requested metric
            cur = val_mets[training_params.early_stopping_metric]
            if should_update_best(cur, best_val, training_params.smaller_is_better):
                best_val = cur
                best_epoch = epoch
                # Use model's save_checkpoint method if available, otherwise save state_dict
                if hasattr(model, "save_checkpoint") and callable(
                    getattr(model, "save_checkpoint")
                ):
                    model.save_checkpoint(model_path)
                else:
                    torch.save(model.state_dict(), model_path)
                patience = 0
            else:
                patience += 1
                if patience >= training_params.early_stopping_patience:
                    break

            # learning rate scheduler
            if scheduler is not None:
                scheduler.step(cur)

            # Log learning rate to TensorBoard
            if write_to_tensorboard:
                current_lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("learning_rate", current_lr, epoch)

            loop.set_postfix(
                {
                    training_params.early_stopping_metric: f"{best_val:.4f}",
                    **{f"train_{name}": val for name, val in train_mets.items()},
                    **{f"val_{name}": val for name, val in val_mets.items()},
                }
            )
        print(f"Time taken for training loop: {time.time() - loop_start_time}") #! remove later

        history["num_epochs"] = best_epoch + 1

        # load best and eval on test set
        # Use model's load_checkpoint method if available, otherwise save state_dict
        if hasattr(model, "load_checkpoint") and callable(
            getattr(model, "load_checkpoint")
        ):
            model.load_checkpoint(model_path)
        else:
            model.load_state_dict(torch.load(model_path))
        test_mets = run_epoch(model, loaders["test"])

        # record into cv_results
        for name in metric_names:

            if name != "confusion_matrix":
                cv_results[f"train_{name}"].append(history[f"train_{name}"][best_epoch])
                cv_results[f"val_{name}"].append(history[f"val_{name}"][best_epoch])
                cv_results[f"test_{name}"].append(test_mets[name])
            elif name == "confusion_matrix":
                conf_matrix_train = history[f"train_{name}"][best_epoch]
                conf_matrix_val = history[f"val_{name}"][best_epoch]
                conf_matrix_test = test_mets[name]
        cv_results["num_epochs"].append(history["num_epochs"])

        if write_to_tensorboard:
            # Log main model test metrics
            log_metrics_to_tensorboard(writer, test_mets, "model", "test", fold)

            # Log baseline metrics
            if training_params.logistic_regression_baseline:
                log_metrics_to_tensorboard(
                    writer, logistic_baseline_metrics, "logistic_regression", None, fold
                )
            if training_params.linear_regression_baseline:
                log_metrics_to_tensorboard(
                    writer, linear_baseline_metrics, "linear_regression", None, fold
                )
            if training_params.ridge_regression_baseline:
                log_metrics_to_tensorboard(
                    writer, ridge_baseline_metrics, "ridge_regression", None, fold
                )

            writer.close()

        # word‐level ROC and top-k. Only useful for word embedding task.
        # Hardcoded for now since this would be a bit complicated
        # to generalize at the moment.
        if is_word_embedding_decoding_task:
            # Get extra inputs for test set (includes model-specific data like data_info_list)
            test_extra_inputs = data_utils.df_columns_to_tensors(
                data_df, task_config.task_specific_config.input_fields, te_idx
            )

            results = metrics.embedding_metrics.compute_word_embedding_task_metrics(
                neural_data[te_idx],
                Y[te_idx],
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

        models.append(model)
        histories.append(history)

        if plot_results:
            plot_training_history(history, fold=fold)

    # 7. Print CV summary
    if training_params.logistic_regression_baseline and logistic_regression_results:
        print("\n" + "=" * 60)
        print("LOGISTIC REGRESSION BASELINE RESULTS")
        print("=" * 60)

        # Aggregate metrics across folds
        for metric_name in logistic_regression_results[0].keys():
            values = [result[metric_name] for result in logistic_regression_results]
            if np.isscalar(values[0]) or (
                isinstance(values[0], np.ndarray) and values[0].size == 1
            ):
                print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    if training_params.linear_regression_baseline and linear_regression_results:
        print("\n" + "=" * 60)
        print("LINEAR REGRESSION BASELINE RESULTS")
        print("=" * 60)

        # Aggregate metrics across folds
        for metric_name in linear_regression_results[0].keys():
            values = [result[metric_name] for result in linear_regression_results]
            if np.isscalar(values[0]) or (
                isinstance(values[0], np.ndarray) and values[0].size == 1
            ):
                print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    if training_params.ridge_regression_baseline and ridge_regression_results:
        print("\n" + "=" * 60)
        print(
            f"RIDGE REGRESSION BASELINE RESULTS (alpha={training_params.ridge_alpha})"
        )
        print("=" * 60)

        # Aggregate metrics across folds
        for metric_name in ridge_regression_results[0].keys():
            values = [result[metric_name] for result in ridge_regression_results]
            if np.isscalar(values[0]) or (
                isinstance(values[0], np.ndarray) and values[0].size == 1
            ):
                print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("\n" + "=" * 60)
    print("MAIN MODEL CROSS-VALIDATION RESULTS")
    print("=" * 60)

    if "confusion_matrix" in metric_names:
        conf_matrices = {
            "train": conf_matrix_train,
            "val": conf_matrix_val,
            "test": conf_matrix_test,
        }

    for phase in ("train", "val", "test"):
        for name in metric_names:
            if name != "confusion_matrix":
                vals = cv_results[f"{phase}_{name}"]

                # Individual Folds
                print(f"--- Individual Folds ({phase}_{name}) ---")
                fold_nums = cv_results.get("fold_nums", list(range(1, len(vals) + 1)))
                for i, val in enumerate(vals):
                    fold_num = fold_nums[i]
                    print(f"Fold {fold_num}: {val:.4f}")

                # Mean
                print(
                    f"Mean {phase} {name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}\n"
                )

            elif name == "confusion_matrix":
                print(f"{phase} confusion matrix:\n{conf_matrices[phase]}")

    if "cross_entropy" in metric_names:
        for phase in phases:
            ce_vals = cv_results[f"{phase}_cross_entropy"]
            ppl_vals = np.exp(ce_vals)
            print(
                f"Mean {phase} perplexity: {np.mean(ppl_vals):.4f} ± {np.std(ppl_vals):.4f}"
            )

    if is_word_embedding_decoding_task:
        for metric_name in embedding_metrics:
            vals = cv_results[metric_name]
            print(f"Mean {metric_name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

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

    for lag in lags:
        if lag in already_read_lags:
            print(f"Lag {lag} already done, skipping...")
            continue

        print("=" * 60)
        print("running lag:", lag)
        print("=" * 60)

        # TODO: Support lazy-loading for larger datasets on future tasks.
        neural_data, targets, data_df = data_utils.get_data(
            lag,
            raws,
            task_df,
            data_params.window_width,
            preprocessing_fns,
            data_params.preprocessor_params,
            per_subject_preprocessing=data_params.per_subject_preprocessing,
        )

        neural_tensor = torch.FloatTensor(neural_data)
        # Handle case where Y contains arrays (e.g., word embeddings)
        if targets.dtype == object:
            targets = np.stack(targets)
        targets_tensor = torch.FloatTensor(targets)

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
def extract_features_for_caching(model, loader, device):
    model.eval()
    
    #feature aggregation
    all_features, input_dicts, y_bs = [], [], []
    with torch.no_grad():
        for batch_data in loader:
            Xb, inputs_dict, y_b = batch_data
            Xb = Xb.to(device)
            inputs_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in inputs_dict.items()
            }
            features = model(Xb, **inputs_dict, return_feature_emb_instead_of_projection=True) 
            #TODO the return_feature_emb_instead_of_projection flag is important!! must be implemented for each integration.py of the FM model 
            #* BrainBERt => done, PopT/DIVER => NO! 
            all_features.append(features)
            input_dicts.append(inputs_dict)
            y_bs.append(y_b)
    return torch.cat(all_features, dim=0), input_dicts, torch.cat(y_bs, dim=0)

def generate_loaders_from_features(features, input_dicts, y_bs, batch_size, shuffle = False):
    def _merge_input_dicts(batch_dicts):
        if not batch_dicts:
            return {}
        merged = {}
        for key in batch_dicts[0].keys():
            vals = [d[key] for d in batch_dicts if torch.is_tensor(d[key])]
            merged[key] = torch.cat(vals, dim=0) if vals else [d[key] for d in batch_dicts]
        return merged

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
    def __init__(self, feature_head : nn.Module, output_dim = None):
        super().__init__()
        self.feature_head = feature_head
        self.output_dim = output_dim

    def forward(self, x, **kwargs):
        out = self.feature_head(x, **kwargs)
        if self.output_dim == 1 and out.shape[-1] == 1:
            out = out.squeeze(-1)
        return out