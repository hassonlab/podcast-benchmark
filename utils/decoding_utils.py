from collections import Counter
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

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
from core.config import TrainingParams, DataParams
from utils.fold_utils import get_sequential_folds, get_zero_shot_folds
import metrics
from utils.plot_utils import plot_cv_results, plot_training_history
from core.registry import metric_registry
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from sklearn.linear_model import LinearRegression, Ridge
from scipy.stats import pearsonr


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
    loss = 0.0
    for i, loss_name in enumerate(training_params.losses):
        loss += training_params.loss_weights[i] * all_fns[loss_name](out, groundtruth)
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
    X: np.ndarray,
    Y: np.ndarray,
    selected_words: list[str],
    model_constructor_fn,
    task_name: str,
    lag: int,
    model_params: dict,
    training_params: TrainingParams,
    checkpoint_dir: str,
    plot_results: bool = False,
    write_to_tensorboard: bool = False,
    tensorboard_dir: str = "event_logs",
):
    # 1. Prepare device & output dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 2. Convert to tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float32)

    # 2.5. Shuffle targets if requested (sanity check to verify model is working)
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
        fold_indices = get_sequential_folds(X, num_folds=training_params.n_folds)
    elif training_params.fold_type == "zero_shot_folds":
        fold_indices = get_zero_shot_folds(
            selected_words, num_folds=training_params.n_folds
        )
    else:
        raise ValueError(f"Unknown fold_type: {training_params.fold_type}")

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

        for i, (Xb, yb) in enumerate(loader):
            Xb, yb = Xb.to(device), yb.to(device)

            if is_train:
                out = model(Xb)
                loss = compute_loss(out, yb, training_params, all_fns)
                # Normalize loss to account for gradient accumulation
                loss = loss / grad_steps
                loss.backward()

                if should_update_gradient_accumulation(i, len(loader), grad_steps):
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                with torch.no_grad():
                    out = model(Xb)
                    loss = compute_loss(out, yb, training_params, all_fns)

            # Compute all metrics for this batch using the helper function
            batch_metrics = compute_all_metrics(out, yb, all_fns, model_params)

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

        return {
            name: (
                sums[name] if name == "confusion_matrix" else sums[name] / len(loader)
            )
            for name in sums
        }

    # 6. Cross‐val loop
    for fold, (tr_idx, va_idx, te_idx) in enumerate(fold_indices, start=1):
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

        # DataLoaders
        datasets = {
            "train": TensorDataset(X[tr_idx], Y_train_norm),
            "val": TensorDataset(X[va_idx], Y_val_norm),
            "test": TensorDataset(X[te_idx], Y_test_norm),
        }
        loaders = {
            phase: DataLoader(
                ds, batch_size=training_params.batch_size, shuffle=(phase == "train")
            )
            for phase, ds in datasets.items()
        }

        # Train baseline models and compute all metrics
        logistic_baseline_metrics = None
        linear_baseline_metrics = None
        ridge_baseline_metrics = None

        if training_params.logistic_regression_baseline:
            print("Training logistic regression baseline...")
            # Train logistic regression model (use normalized targets if applicable)
            logistic_model = train_logistic_regression(
                X[tr_idx].cpu().numpy(),
                Y_train_norm.cpu().numpy(),
            )

            # Prepare data splits for metric computation (use normalized targets)
            X_splits = {
                "train": X[tr_idx].cpu().numpy(),
                "val": X[va_idx].cpu().numpy(),
                "test": X[te_idx].cpu().numpy(),
            }
            Y_splits = {
                "train": Y_train_norm.cpu().numpy(),
                "val": Y_val_norm.cpu().numpy(),
                "test": Y_test_norm.cpu().numpy(),
            }

            # Compute all metrics
            logistic_baseline_metrics = compute_baseline_metrics(
                logistic_model, X_splits, Y_splits, all_fns, model_params
            )
            logistic_regression_results.append(logistic_baseline_metrics)

        if training_params.linear_regression_baseline:
            print("Training linear regression baseline...")
            # Train linear regression model (use normalized targets if applicable)
            linear_model = train_linear_regression(
                X[tr_idx].cpu().numpy(),
                Y_train_norm.cpu().numpy(),
            )

            # Prepare data splits for metric computation (use normalized targets)
            X_splits = {
                "train": X[tr_idx].cpu().numpy(),
                "val": X[va_idx].cpu().numpy(),
                "test": X[te_idx].cpu().numpy(),
            }
            Y_splits = {
                "train": Y_train_norm.cpu().numpy(),
                "val": Y_val_norm.cpu().numpy(),
                "test": Y_test_norm.cpu().numpy(),
            }

            # Compute all metrics
            linear_baseline_metrics = compute_baseline_metrics(
                linear_model, X_splits, Y_splits, all_fns, model_params
            )
            linear_regression_results.append(linear_baseline_metrics)

        if training_params.ridge_regression_baseline:
            print("Training ridge regression baseline...")
            # Train ridge regression model (use normalized targets if applicable)
            ridge_model = train_ridge_regression(
                X[tr_idx].cpu().numpy(),
                Y_train_norm.cpu().numpy(),
                alpha=training_params.ridge_alpha,
            )

            # Prepare data splits for metric computation (use normalized targets)
            X_splits = {
                "train": X[tr_idx].cpu().numpy(),
                "val": X[va_idx].cpu().numpy(),
                "test": X[te_idx].cpu().numpy(),
            }
            Y_splits = {
                "train": Y_train_norm.cpu().numpy(),
                "val": Y_val_norm.cpu().numpy(),
                "test": Y_test_norm.cpu().numpy(),
            }

            # Compute all metrics
            ridge_baseline_metrics = compute_baseline_metrics(
                ridge_model, X_splits, Y_splits, all_fns, model_params
            )
            ridge_regression_results.append(ridge_baseline_metrics)

        # Model, optimizer, early‐stop setup
        model = model_constructor_fn(model_params).to(device)
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_params.learning_rate,
            weight_decay=training_params.weight_decay,
        )

        best_val, patience = setup_early_stopping_state(training_params)
        best_epoch = 0

        # per‐fold history (only train & val, for plotting)
        history = {
            f"{phase}_{name}": [] for phase in ("train", "val") for name in metric_names
        }
        history["train_loss"] = []
        history["val_loss"] = []
        history["num_epochs"] = None

        loop = tqdm(range(training_params.epochs), desc=f"Lag {lag}, Fold {fold}")
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
                torch.save(model.state_dict(), model_path)
                patience = 0
            else:
                patience += 1
                if patience >= training_params.early_stopping_patience:
                    break

            loop.set_postfix(
                {
                    training_params.early_stopping_metric: f"{best_val:.4f}",
                    **{f"train_{name}": val for name, val in train_mets.items()},
                    **{f"val_{name}": val for name, val in val_mets.items()},
                }
            )

        history["num_epochs"] = best_epoch + 1

        # load best and eval on test set
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
            log_metrics_to_tensorboard(
                writer, logistic_baseline_metrics, "logistic_regression", None, fold
            )
            log_metrics_to_tensorboard(
                writer, linear_baseline_metrics, "linear_regression", None, fold
            )
            log_metrics_to_tensorboard(
                writer, ridge_baseline_metrics, "ridge_regression", None, fold
            )

            writer.close()

        # word‐level ROC and top-k. Only useful for word embedding task.
        # Hardcoded for now since this would be a bit complicated
        # to generalize at the moment.
        if is_word_embedding_decoding_task:
            # TODO: figure out how we want to generalize evaluation inference vs training inference better.
            # Key focus on preserve_ensemble argument.
            results = metrics.embedding_metrics.compute_word_embedding_task_metrics(
                X[te_idx],
                Y[te_idx],
                model,
                device,
                selected_words,
                te_idx,
                tr_idx,
                training_params.top_k_thresholds,
                training_params.min_train_freq_auc,
                training_params.min_test_freq_auc,
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

    for phase in phases:
        for name in metric_names:
            if name != "confusion_matrix":
                vals = cv_results[f"{phase}_{name}"]
                print(f"Mean {phase} {name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
            elif name == "confusion_matrix":
                print(f"{phase} confusion matrix:\n{conf_matrices[phase]}")

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
    model_constructor_fn,
    task_name: str,
    model_params: dict,
    training_params: TrainingParams,
    data_params: DataParams,
    output_dir="results/",
    checkpoint_dir="checkpoints/",
    write_to_tensorboard=False,
    tensorboard_dir="event_log",
):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

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
        X, Y, selected_words = data_utils.get_data(
            lag,
            raws,
            task_df,
            data_params.window_width,
            preprocessing_fns,
            data_params.preprocessor_params,
            word_column=data_params.word_column,
        )

        X_tensor = torch.FloatTensor(X)
        # Handle case where Y contains arrays (e.g., word embeddings)
        if Y.dtype == object:
            Y = np.stack(Y)
        Y_tensor = torch.FloatTensor(Y)

        print(f"X_tensor shape: {X_tensor.shape}, Y_tensor shape: {Y_tensor.shape}")

        models, histories, cv_results = train_decoding_model(
            X_tensor,
            Y_tensor,
            selected_words,
            model_constructor_fn,
            task_name,
            lag,
            model_params=model_params,
            training_params=training_params,
            checkpoint_dir=os.path.join(checkpoint_dir, f"lag_{lag}"),
            write_to_tensorboard=write_to_tensorboard,
            tensorboard_dir=tensorboard_dir,
        )

        # Aggregate metrics
        lag_metrics = {
            f"{metric}_mean": np.mean(values) if len(values) > 0 else np.nan
            for metric, values in cv_results.items()
        }
        lag_metrics.update(
            {
                f"{metric}_std": np.std(values) if len(values) > 0 else np.nan
                for metric, values in cv_results.items()
            }
        )
        lag_metrics["lags"] = lag

        # Append new row to existing DataFrame and write to file
        existing_df = pd.concat(
            [existing_df, pd.DataFrame([lag_metrics])], ignore_index=True
        )
        existing_df.to_csv(filename, index=False)
