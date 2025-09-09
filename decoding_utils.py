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
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

import mne
from sklearn.metrics import roc_auc_score

from scipy.spatial.distance import cosine

import data_utils
from config import TrainingParams, DataParams
from fold_utils import get_sequential_folds, get_zero_shot_folds
import metrics
from registry import metric_registry


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
    model_dir: str,
    plot_results: bool = False,
    write_to_tensorboard: bool = False,
    tensorboard_dir: str = "event_logs",
):
    # 1. Prepare device & output dir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(model_dir, exist_ok=True)

    # 2. Convert to tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float32)

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
    cv_results = {f"{phase}_{name}": [] for phase in phases for name in metric_names}
    cv_results["num_epochs"] = []

    models, histories = [], []
    # Clean this up later to get rid of this weird optional return value. Hardcoding for now since generalizing this
    # would get complicated.
    is_word_embedding_decoding_task = task_name == "word_embedding_decoding_task"
    if is_word_embedding_decoding_task:
        roc_results = []
    else:
        roc_results = None

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

        sums = {name: 0.0 for name in metric_names}
        sums["loss"] = 0.0

        grad_steps = training_params.grad_accumulation_steps
        if is_train:
            optimizer.zero_grad()

        for i, (Xb, yb) in enumerate(loader):
            Xb, yb = Xb.to(device), yb.to(device)
            bsz = Xb.size(0)

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

            # accumulate each metric
            for name, fn in all_fns.items():
                val = fn(out, yb)
                # get a scalar float
                if torch.is_tensor(val):
                    val = val.detach().mean().item()
                sums[name] += val

            # add loss to sums
            if torch.is_tensor(loss):
                loss = loss.detach().mean().item()
            sums["loss"] += loss
        return {name: sums[name] / len(loader) for name in sums}

    # 6. Cross‐val loop
    for fold, (tr_idx, va_idx, te_idx) in enumerate(fold_indices, start=1):
        model_path = os.path.join(model_dir, f"best_model_fold{fold}.pt")

        # TensorBoard writer
        if write_to_tensorboard:
            tb_path = os.path.join(tensorboard_dir, f"lag_{lag}", f"fold_{fold}")
            writer = SummaryWriter(log_dir=tb_path)

        # DataLoaders
        datasets = {
            "train": TensorDataset(X[tr_idx], Y[tr_idx]),
            "val": TensorDataset(X[va_idx], Y[va_idx]),
            "test": TensorDataset(X[te_idx], Y[te_idx]),
        }
        loaders = {
            phase: DataLoader(
                ds, batch_size=training_params.batch_size, shuffle=(phase == "train")
            )
            for phase, ds in datasets.items()
        }

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
                    writer.add_scalar(f"{name}/train", val, epoch)
            for name, val in val_mets.items():
                history[f"val_{name}"].append(val)
                if write_to_tensorboard:
                    writer.add_scalar(f"{name}/val", val, epoch)

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
            cv_results[f"train_{name}"].append(history[f"train_{name}"][-1])
            cv_results[f"val_{name}"].append(max(history[f"val_{name}"]))
            cv_results[f"test_{name}"].append(test_mets[name])
        cv_results["num_epochs"].append(history["num_epochs"])

        if write_to_tensorboard:
            for name, val in test_mets.items():
                writer.add_scalar(f"{name}/test", val, fold)
            writer.close()

        # word‐level ROC. Only useful for word embedding task. Hardcoded for now since this would be a bit complicated
        # to generalize at the moment.
        if is_word_embedding_decoding_task:
            roc = calculate_word_embeddings_roc_auc_logits(
                model, X[te_idx], Y[te_idx], selected_words[te_idx], device
            )
            roc_results.append(roc)

        models.append(model)
        histories.append(history)

        if plot_results:
            plot_training_history(history, fold=fold)

    # 7. Print CV summary
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    for phase in phases:
        for name in metric_names:
            vals = cv_results[f"{phase}_{name}"]
            print(f"Mean {phase} {name}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    if plot_results:
        plot_cv_results(cv_results)

    # 8. Aggregate final word‐AUC if task is for word embedding decoding.
    # Clean this up later to get rid of this weird optional return value.
    weighted_roc = None
    if is_word_embedding_decoding_task:
        final_word_auc = {}
        for r in roc_results:
            final_word_auc.update(r["word_aucs"])
        weighted_roc = summarize_roc_results(final_word_auc, selected_words)

    return models, histories, cv_results, roc_results, weighted_roc


def plot_training_history(history, fold=None):
    """
    Plot the training and validation loss and cosine similarity.

    Args:
        history: Dictionary containing training history
        fold: Fold number (optional)
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot loss
    ax1.plot(history["train_loss"], label="Training Loss")
    ax1.plot(history["val_loss"], label="Validation Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    title = "Training and Validation Loss"
    if fold is not None:
        title = f"Fold {fold}: {title}"
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    # Plot cosine similarity
    ax2.plot(history["train_cosine"], label="Training Cosine Similarity")
    ax2.plot(history["val_cosine"], label="Validation Cosine Similarity")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cosine Similarity")
    title = "Training and Validation Cosine Similarity"
    if fold is not None:
        title = f"Fold {fold}: {title}"
    ax2.set_title(title)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def plot_cv_results(cv_results):
    """
    Plot cross-validation results.

    Args:
        cv_results: Dictionary containing cross-validation results
    """
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Prepare data
    folds = range(1, len(cv_results["train_loss"]) + 1)

    # Plot loss
    ax1.plot(folds, cv_results["train_loss"], "o-", label="Training Loss")
    ax1.plot(folds, cv_results["val_loss"], "o-", label="Validation Loss")
    ax1.plot(folds, cv_results["test_loss"], "o-", label="Test Loss")
    ax1.set_xlabel("Fold")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Cross-Validation Loss")
    ax1.set_xticks(folds)
    ax1.legend()
    ax1.grid(True)

    # Plot cosine similarity
    ax2.plot(
        folds, cv_results["train_cosine"], "o-", label="Training Cosine Similarity"
    )
    ax2.plot(
        folds, cv_results["val_cosine"], "o-", label="Validation Cosine Similarity"
    )
    ax2.plot(folds, cv_results["test_cosine"], "o-", label="Test Cosine Similarity")
    ax2.set_xlabel("Fold")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_title("Cross-Validation Cosine Similarity")
    ax2.set_xticks(folds)
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def calculate_word_embeddings_roc_auc_logits(
    model, X, Y, selected_words, device, min_repetitions=5
):
    """
    Calculate ROC-AUC for word embedding predictions using logits approach.

    This follows the described method more closely, converting distances to logits
    via softmax transformation.
    """
    # Step 1: Get word frequency counts and filter for words with minimum repetitions
    word_counts = Counter(selected_words)
    frequent_words = [
        word for word, count in word_counts.items() if count >= min_repetitions
    ]
    print(
        f"Found {len(frequent_words)} words with at least {min_repetitions} repetitions ({len(frequent_words)/len(set(selected_words))*100:.1f}% of unique words)"
    )

    X = X.to(device)

    # Step 2: Get predicted embeddings for all neural data
    model_predictions = []
    for i in range(len(X)):
        with torch.no_grad():
            input_data = X[i : i + 1]
            pred = model(input_data).cpu().numpy()

        model_predictions.append(pred.squeeze())

    predicted_embeddings = np.array(model_predictions)

    # Step 3: Group all embeddings for each unique word
    # Y should be on cpu for comparisons.
    Y = Y.cpu()
    unique_words = list(set(selected_words))
    word_to_embeddings = {
        word: np.array(Y[np.array(selected_words) == word]) for word in unique_words
    }

    # Step 4: Calculate average embeddings for each unique word
    avg_word_embeddings = {
        word: np.mean(embs, axis=0) for word, embs in word_to_embeddings.items()
    }

    # Step 5: Calculate cosine distances and convert to logits
    word_aucs = {}
    word_to_idx = {}

    for idx, pred_embedding in enumerate(predicted_embeddings):
        # Calculate distances to all unique words
        distances = []
        for word in unique_words:
            avg_embedding = avg_word_embeddings[word]
            distance = cosine(pred_embedding, avg_embedding)
            # Convert distance to similarity
            similarity = 1 - distance
            distances.append(similarity)

        # Convert similarities to logits using softmax
        logits = torch.tensor(distances)
        logits = F.softmax(logits, dim=0).numpy()

        # For each instance, collect logits for the correct label and all other labels
        true_word = selected_words[idx]
        true_word_idx = unique_words.index(true_word)

        # Update the logits for each word
        if true_word not in word_to_idx:
            word_to_idx[true_word] = {"logits": [], "is_true": []}

        for word_idx, word in enumerate(unique_words):
            if word not in word_to_idx:
                word_to_idx[word] = {"logits": [], "is_true": []}

            word_to_idx[word]["logits"].append(logits[word_idx])
            word_to_idx[word]["is_true"].append(1 if word == true_word else 0)

    # Step 6: Calculate ROC-AUC for each frequent word
    for word in frequent_words:
        try:
            roc_auc = roc_auc_score(
                np.array(word_to_idx[word]["is_true"]),
                np.array(word_to_idx[word]["logits"]),
            )
            word_aucs[word] = roc_auc
        except ValueError:
            print(
                f"Skipping ROC-AUC calculation for '{word}' - insufficient class variety"
            )

    # Step 7: Calculate weighted ROC-AUC based on word frequency
    total_count = sum(word_counts[word] for word in frequent_words if word in word_aucs)
    weighted_auc = (
        sum(word_aucs[word] * word_counts[word] for word in word_aucs) / total_count
    )

    return {
        "word_aucs": word_aucs,
        "weighted_auc": weighted_auc,
        "frequent_words": frequent_words,
    }


def summarize_roc_results(word_aucs, selected_words, min_repetitions=5):
    word_counts = Counter(selected_words)
    frequent_words = [
        word for word, count in word_counts.items() if count >= min_repetitions
    ]

    total_count = sum(word_counts[word] for word in frequent_words if word in word_aucs)
    weighted_auc = (
        sum(word_aucs[word] * word_counts[word] for word in word_aucs) / total_count
    )

    return weighted_auc


def run_training_over_lags(
    lags,
    raws: list[mne.io.Raw],
    df_word: pd.DataFrame,
    preprocessing_fn,
    model_constructor_fn,
    task_name: str,
    model_params: dict,
    training_params: TrainingParams,
    data_params: DataParams,
    output_dir="results/",
    model_dir="models/",
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

        X, Y, selected_words = data_utils.get_data(
            lag,
            raws,
            df_word,
            data_params.window_width,
            preprocessing_fn,
            data_params.preprocessor_params,
        )

        X_tensor = torch.FloatTensor(X)
        Y_tensor = torch.FloatTensor(Y)

        print(f"X_tensor shape: {X_tensor.shape}, Y_tensor shape: {Y_tensor.shape}")

        models, histories, cv_results, roc_results, weighted_roc_mean = (
            train_decoding_model(
                X_tensor,
                Y_tensor,
                selected_words,
                model_constructor_fn,
                task_name,
                lag,
                model_params=model_params,
                training_params=training_params,
                model_dir=os.path.join(model_dir, f"lag_{lag}"),
                write_to_tensorboard=write_to_tensorboard,
                tensorboard_dir=tensorboard_dir,
            )
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
        lag_metrics["rocs"] = weighted_roc_mean

        # Append new row to existing DataFrame and write to file
        existing_df = pd.concat(
            [existing_df, pd.DataFrame([lag_metrics])], ignore_index=True
        )
        existing_df.to_csv(filename, index=False)

    return [row["rocs"] for row in all_new_results]
