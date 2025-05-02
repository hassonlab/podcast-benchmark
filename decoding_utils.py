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
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import roc_auc_score

from scipy.spatial.distance import cosine

import data_utils
from config import TrainingParams, DataParams


def train_decoding_model(
    X: np.array,
    Y: np.array,
    selected_words: list[str],
    model_constructor_fn,
    lag: int,
    model_params: dict,
    training_params: TrainingParams,
    model_dir: str,
    plot_results=False,
    write_to_tensorboard=False,
    tensorboard_dir="event_logs",
):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Convert numpy arrays to torch tensors if needed
    if isinstance(X, np.ndarray):
        X = torch.tensor(X, dtype=torch.float32)
    if isinstance(Y, np.ndarray):
        Y = torch.tensor(Y, dtype=torch.float32)

    # Initialize cross-validation results
    models = []
    histories = []
    cv_results = {
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "train_cosine": [],
        "val_cosine": [],
        "test_cosine": [],
        "train_nll": [],
        "val_nll": [],
        "test_nll": [],
        "num_epochs": [],
    }
    roc_results = []

    kf = KFold(n_splits=5, shuffle=False)
    fold_indices = list(kf.split(range(X.shape[0])))
    best_epoch = 0

    for fold, (train_val_idx, test_idx) in enumerate(fold_indices):
        model_path = os.path.join(model_dir, f"best_pitom_model_fold{fold+1}.pt")

        # TensorBoard writer per fold/lag/run
        if write_to_tensorboard:
            writer = SummaryWriter(
                log_dir=os.path.join(
                    tensorboard_dir,
                    f"lag_{lag}",
                    f"fold_{fold+1}",
                )
            )

        train_idx, val_idx = train_test_split(
            np.array(train_val_idx),
            test_size=0.25,
            shuffle=False,
        )

        X_train, Y_train = X[train_idx], Y[train_idx]
        X_val, Y_val = X[val_idx], Y[val_idx]
        X_test, Y_test = X[test_idx], Y[test_idx]

        train_dataset = TensorDataset(X_train, Y_train)
        val_dataset = TensorDataset(X_val, Y_val)
        test_dataset = TensorDataset(X_test, Y_test)

        train_loader = DataLoader(
            train_dataset, batch_size=training_params.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=training_params.batch_size, shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, batch_size=training_params.batch_size, shuffle=False
        )

        model = model_constructor_fn(model_params).to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=training_params.learning_rate,
            weight_decay=training_params.weight_decay,
        )

        best_val_cosine = -float("inf")
        patience_counter = 0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_cosine": [],
            "val_cosine": [],
            "train_nll": [],
            "val_nll": [],
            "num_epochs": np.nan,
        }

        progress_bar = tqdm(
            range(training_params.epochs), desc=f"Lag {lag}, Fold {fold + 1}"
        )
        for epoch in progress_bar:
            model.train()
            train_loss = 0.0
            train_cosine = 0.0
            train_nll = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * inputs.size(0)
                batch_cosines = calculate_cosine_similarity(
                    outputs.detach(), targets.detach()
                )
                train_cosine += batch_cosines.sum().item()
                nll = compute_nll_contextual(outputs.detach(), targets.detach())
                train_nll += nll.sum().item()

            train_loss = train_loss / len(train_loader.dataset)
            train_cosine = train_cosine / len(train_loader.dataset)
            train_nll = train_nll / len(train_loader.dataset)

            model.eval()
            val_loss = 0.0
            val_cosine = 0.0
            val_nll = 0.0

            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    batch_cosines = calculate_cosine_similarity(
                        outputs.detach(), targets.detach()
                    )
                    val_cosine += batch_cosines.sum().item()
                    nll = compute_nll_contextual(outputs.detach(), targets.detach())
                    val_nll += nll.sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            val_cosine = val_cosine / len(val_loader.dataset)
            val_nll = val_nll / len(val_loader.dataset)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_cosine"].append(train_cosine)
            history["val_cosine"].append(val_cosine)
            history["train_nll"].append(train_nll)
            history["val_nll"].append(val_nll)

            if write_to_tensorboard:
                writer.add_scalar("Loss/Train", train_loss, epoch)
                writer.add_scalar("Loss/Validation", val_loss, epoch)
                writer.add_scalar("Cosine/Train", train_cosine, epoch)
                writer.add_scalar("Cosine/Validation", val_cosine, epoch)
                writer.add_scalar("NLL/Train", train_nll, epoch)
                writer.add_scalar("NLL/Validation", val_cosine, epoch)

            if val_cosine > best_val_cosine:
                best_val_cosine = val_cosine
                patience_counter = 0
                best_epoch = epoch
                torch.save(model.state_dict(), model_path)
            else:
                patience_counter += 1
                if patience_counter >= training_params.early_stopping_patience:
                    history["num_epochs"] = best_epoch + 1
                    break

            progress_bar.set_postfix(
                {
                    "train_loss": train_loss,
                    "train_cosine": train_cosine,
                    "val_loss": val_loss,
                    "val_cosine": val_cosine,
                    "best_epoch": best_epoch + 1,
                    "epoch": epoch,
                }
            )

        if np.isnan(history["num_epochs"]):
            history["num_epochs"] = training_params.epochs

        model.load_state_dict(torch.load(model_path))
        model.eval()
        test_loss = 0.0
        test_cosine = 0.0
        test_nll = 0.0

        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                batch_cosines = calculate_cosine_similarity(
                    outputs.detach(), targets.detach()
                )
                test_cosine += batch_cosines.sum().item()
                nll = compute_nll_contextual(outputs.detach(), targets.detach())
                test_nll += nll.sum().item()

        roc_result = calculate_word_embeddings_roc_auc_logits(
            model, X_test, Y_test, selected_words[test_idx], device
        )
        roc_results.append(roc_result)

        test_loss = test_loss / len(test_loader.dataset)
        test_cosine = test_cosine / len(test_loader.dataset)
        test_nll = test_nll / len(test_loader.dataset)

        if write_to_tensorboard:
            writer.add_scalar("Loss/Test", test_loss, fold)
            writer.add_scalar("Cosine/Test", test_cosine, fold)
            writer.add_scalar("NLL/Test", test_nll, fold)
            writer.close()

        print(
            f"\nFold {fold+1} Test Results: Loss = {test_loss:.4f}, Cosine Similarity = {test_cosine:.4f}"
        )

        cv_results["train_loss"].append(history["train_loss"][-1])
        cv_results["val_loss"].append(
            history["val_loss"][history["val_cosine"].index(max(history["val_cosine"]))]
        )
        cv_results["test_loss"].append(test_loss)
        cv_results["train_cosine"].append(history["train_cosine"][-1])
        cv_results["val_cosine"].append(best_val_cosine)
        cv_results["test_cosine"].append(test_cosine)
        cv_results["train_nll"].append(history["train_nll"][-1])
        cv_results["val_nll"].append(
            history["val_nll"][history["val_cosine"].index(max(history["val_cosine"]))]
        )
        cv_results["test_nll"].append(test_nll)
        cv_results["num_epochs"].append(history["num_epochs"])

        models.append(model)
        histories.append(history)

        if plot_results:
            plot_training_history(history, fold=fold + 1)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(
        f"Mean Train Loss: {np.mean(cv_results['train_loss']):.4f} ± {np.std(cv_results['train_loss']):.4f}"
    )
    print(
        f"Mean Val Loss: {np.mean(cv_results['val_loss']):.4f} ± {np.std(cv_results['val_loss']):.4f}"
    )
    print(
        f"Mean Test Loss: {np.mean(cv_results['test_loss']):.4f} ± {np.std(cv_results['test_loss']):.4f}"
    )
    print(
        f"Mean Train Cosine: {np.mean(cv_results['train_cosine']):.4f} ± {np.std(cv_results['train_cosine']):.4f}"
    )
    print(
        f"Mean Val Cosine: {np.mean(cv_results['val_cosine']):.4f} ± {np.std(cv_results['val_cosine']):.4f}"
    )
    print(
        f"Mean Test Cosine: {np.mean(cv_results['test_cosine']):.4f} ± {np.std(cv_results['test_cosine']):.4f}"
    )
    print(
        f"Mean Train Negative Log-Likelihood: {np.mean(cv_results['train_nll']):.4f} ± {np.std(cv_results['train_nll']):.4f}"
    )
    print(
        f"Mean Val Negative Log-Likelihood: {np.mean(cv_results['val_nll']):.4f} ± {np.std(cv_results['val_nll']):.4f}"
    )
    print(
        f"Mean Test Negative Log-Likelihood: {np.mean(cv_results['test_nll']):.4f} ± {np.std(cv_results['test_nll']):.4f}"
    )
    print(
        f"Mean Num Epochs: {np.mean(cv_results['num_epochs']):.4f} ± {np.std(cv_results['num_epochs']):.4f}"
    )

    if plot_results:
        plot_cv_results(cv_results)

    final_word_auc = {}
    for roc_result in roc_results:
        final_word_auc.update(roc_result["word_aucs"])
    weighted_roc_mean = summarize_roc_results(final_word_auc, selected_words)

    return models, histories, cv_results, roc_results, weighted_roc_mean


def calculate_cosine_similarity(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Calculate cosine similarity between each pair of rows in predictions and targets.

    Args:
        predictions: Tensor of shape [batch_size, embedding_dim]
        targets: Tensor of shape [batch_size, embedding_dim]

    Returns:
        cosine_similarities: Tensor of shape [batch_size], with cosine similarity per sample
    """
    # Normalize along the embedding dimension
    predictions = F.normalize(predictions, p=2, dim=1)
    targets = F.normalize(targets, p=2, dim=1)

    # Element-wise dot product across batch
    cosine_similarities = torch.sum(predictions * targets, dim=1)

    return cosine_similarities


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


def compute_nll_contextual(predicted_embeddings, actual_embeddings):
    """
    Computes a contrastive NLL where each predicted embedding is scored against all actual embeddings.
    """
    # Normalize embeddings
    pred_norm = F.normalize(predicted_embeddings, dim=1)
    actual_norm = F.normalize(actual_embeddings, dim=1)

    # Similarity matrix: [n_samples, n_samples]
    logits = torch.matmul(pred_norm, actual_norm.T)

    # Labels: diagonal = correct match
    targets = torch.arange(
        len(predicted_embeddings), device=predicted_embeddings.device
    )

    # Cross-entropy over rows
    return F.cross_entropy(logits, targets)


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
