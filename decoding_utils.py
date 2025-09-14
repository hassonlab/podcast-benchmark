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

    # Hardcode embedding task metrics for now since they need to be handled a bit differently.
    # Clean this up later to get rid of this weird optional return value. Hardcoding for now since generalizing this
    # would get complicated.
    is_word_embedding_decoding_task = task_name == "word_embedding_decoding_task"
    if is_word_embedding_decoding_task:
        # Metrics for specific occurence. Useful for checking contextual embeddings.
        # Can't use AUC-ROC since each example is treated as its own class.
        # cv_results["test_occ_top_1"] = []
        # cv_results["test_occ_top_5"] = []

        # Metrics averaged over words and all their occurences.
        cv_results["test_word_auc_roc"] = []
        # cv_results["test_word_top_1"] = []
        # cv_results["test_word_top_5"] = []

    models, histories = [], []

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
            cv_results[f"train_{name}"].append(history[f"train_{name}"][best_epoch])
            cv_results[f"val_{name}"].append(history[f"val_{name}"][best_epoch])
            cv_results[f"test_{name}"].append(test_mets[name])
        cv_results["num_epochs"].append(history["num_epochs"])

        if write_to_tensorboard:
            for name, val in test_mets.items():
                writer.add_scalar(f"{name}/test", val, fold)
            writer.close()

        # word‐level ROC and top-k. Only useful for word embedding task.
        # Hardcoded for now since this would be a bit complicated
        # to generalize at the moment.
        if is_word_embedding_decoding_task:
            _, _, position_to_id = build_vocabulary(selected_words)
            predictions = get_predictions(X[te_idx], model, device)
            distances, _ = compute_cosine_distances(predictions, Y)

            # Measure performance based on each individual word occurence. Can the model
            # predict which occurence of "dog" in context we care about?
            occurence_scores = compute_class_scores(distances)

            # Group by words for a slightly easier task.
            word_scores = compute_class_scores(distances, position_to_id[te_idx])
            train_frequencies = np.bincount(position_to_id[tr_idx])
            word_auc = metrics.calculate_auc_roc(
                word_scores,
                position_to_id[te_idx],
                train_frequencies,
                training_params.min_train_freq_auc,
            )
            cv_results["test_word_auc_roc"].append(word_auc)

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

    return models, histories, cv_results


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


def get_predictions(X, model, device):
    X = X.to(device)

    # Step 2: Get predicted embeddings for all neural data
    model_predictions = []
    for i in range(len(X)):
        with torch.no_grad():
            input_data = X[i : i + 1]
            pred = model(input_data)

        model_predictions.append(pred.squeeze())

    return torch.cat(model_predictions)


def build_vocabulary(words):
    """
    Build vocabulary mappings from a list of words.

    Args:
        words: List of words (may contain repetitions)

    Returns:
        word_to_id: Dictionary mapping word -> unique_id
        id_to_word: Dictionary mapping unique_id -> word
        postion_to_id: List specifying what the ith word maps to which unique id.
    """
    word_to_id = {}
    position_to_id = []
    next_id = 0

    for word in words:
        if word not in word_to_id:
            # First time seeing this word - assign new ID
            word_to_id[word] = next_id
            next_id += 1

        # Add current position to the word's position list
        word_id = word_to_id[word]
        position_to_id.append(word_id)

    # Build reverse mapping
    id_to_word = {word_id: word for word, word_id in word_to_id.items()}

    return word_to_id, id_to_word, position_to_id


def compute_cosine_distances(predictions, word_embeddings):
    """
    Compute cosine distances between predicted embeddings and word embeddings.
    Supports ensemble predictions.

    Args:
        predictions: torch tensor of shape [num_samples, embedding_dim] or
                    [num_samples, n_ensemble, embedding_dim] for ensemble predictions
        word_embeddings: torch tensor of shape [num_words, embedding_dim]

    Returns:
        scores: torch tensor of shape [num_samples, num_words] containing
                cosine distances for each word given each prediction
    """
    # Normalize word embeddings once
    word_embeddings_norm = F.normalize(word_embeddings, p=2, dim=1)

    # Handle both 2D and 3D cases by reshaping 2D to 3D with singleton dimension
    if predictions.dim() == 2:
        # Single prediction case: [num_samples, embedding_dim] -> [num_samples, 1, embedding_dim]
        predictions = predictions.unsqueeze(1)
    elif predictions.dim() != 3:
        raise ValueError(
            f"Predictions must be 2D or 3D tensor, got {predictions.dim()}D"
        )

    # Now we have: [num_samples, n_ensemble, embedding_dim]
    num_samples, n_ensemble, embedding_dim = predictions.shape

    # Reshape to treat each ensemble prediction separately
    predictions_reshaped = predictions.view(num_samples * n_ensemble, embedding_dim)

    # Normalize ensemble predictions
    predictions_norm = F.normalize(predictions_reshaped, p=2, dim=1)

    # Compute cosine similarity for all ensemble predictions
    cosine_similarities = torch.mm(predictions_norm, word_embeddings_norm.t())
    # Shape: [num_samples * n_ensemble, num_words]

    # Convert to cosine distance
    cosine_distances = 1 - cosine_similarities

    # Reshape back to separate ensemble dimension
    cosine_distances = cosine_distances.view(
        num_samples, n_ensemble, word_embeddings.shape[0]
    )

    # Average across ensemble dimension (for single predictions, n_ensemble=1, so this is a no-op)
    return cosine_distances.mean(dim=1)  # [num_samples, num_words]


def compute_class_scores(cosine_distances, word_labels=None):
    """
    Compute class scores from cosine distances by averaging over word embeddings
    belonging to the same class and applying softmax transformation.

    This implements the logic: "we computed the cosine distance between each of
    the predicted embeddings and the embeddings of all instances of each unique
    word label. The distances were averaged across unique word labels, yielding
    one score for each word label (that is, logit). We used a Softmax
    transformation on these scores (logits)."

    Args:
        cosine_distances: torch tensor of shape [num_samples, num_words] containing
                         cosine distances between predictions and word embeddings
        word_labels: Optional torch tensor of shape [num_words] containing integer class IDs
                    for each word embedding

    Returns:
        class_probabilities: torch tensor of shape [num_samples, num_classes] containing
                           softmax probabilities for each class
        class_logits: torch tensor of shape [num_samples, num_classes] containing
                     the logits (negative averaged distances) before softmax
    """
    if word_labels is not None:
        device = cosine_distances.device
        num_samples = cosine_distances.shape[0]

        # Get unique class labels and sort them for consistent ordering
        unique_classes = torch.unique(word_labels).sort()[0]
        num_classes = len(unique_classes)

        # Initialize tensors for class-averaged distances
        class_distances = torch.zeros(num_samples, num_classes, device=device)

        # For each unique class, average the distances across all word embeddings
        # belonging to that class
        for i, class_id in enumerate(unique_classes):
            # Find indices of word embeddings belonging to this class
            class_mask = word_labels == class_id
            class_indices = torch.where(class_mask)[0]

            if len(class_indices) > 0:
                # Average distances for this class across all its word embeddings
                class_distances[:, i] = cosine_distances[:, class_indices].mean(dim=1)
    else:
        class_distances = cosine_distances

    # Convert distances to similarities (logits)
    # Since cosine distance = 1 - cosine_similarity, we convert back:
    # logits = 1 - distance = cosine_similarity
    class_logits = 1 - class_distances

    # Apply softmax transformation to get probabilities
    class_probabilities = F.softmax(class_logits, dim=1)

    return class_probabilities, class_logits


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

        models, histories, cv_results = train_decoding_model(
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

    return [row["rocs"] for row in all_new_results]
