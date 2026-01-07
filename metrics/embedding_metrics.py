"""
Metrics for embedding and similarity-based tasks.
"""

import numpy as np

import torch
import torch.nn.functional as F

from core.registry import register_metric
from metrics.classification_metrics import perplexity
from metrics.utils import (
    entropy,
    compute_cosine_distances,
    top_k_accuracy,
    compute_class_scores,
    calculate_auc_roc,
)


@register_metric("cosine_sim")
def cosine_similarity(pred: torch.Tensor, true: torch.Tensor) -> float:
    return F.cosine_similarity(pred, true, dim=-1).mean()


@register_metric("cosine_dist")
def cosine_distance(pred: torch.Tensor, true: torch.Tensor) -> float:
    sim = F.cosine_similarity(pred, true, dim=-1)
    return (1 - sim).mean()


@register_metric("nll_embedding")
def compute_nll_contextual(predicted_embeddings, actual_embeddings):
    """
    Computes a contrastive NLL where each predicted embedding is scored against all actual embeddings.
    """
    logits = 1 - compute_cosine_distances(predicted_embeddings, actual_embeddings)

    # Labels: diagonal = correct match
    targets = torch.arange(
        len(predicted_embeddings), device=predicted_embeddings.device
    )

    # Cross-entropy over rows
    return F.cross_entropy(logits, targets)


@register_metric("similarity_entropy")
def similarity_entropy(predicted_embeddings, actual_embeddings):
    logits = 1 - compute_cosine_distances(predicted_embeddings, actual_embeddings)
    probs = F.softmax(logits, dim=1)
    return entropy(probs).mean()


def compute_word_embedding_task_metrics(
    X_test,
    Y_test,
    model,
    device,
    selected_words,
    test_index,
    train_index,
    top_k_thresholds,
    min_train_freq_auc,
    min_test_freq_auc,
    batch_size=16,
    **kwargs,
):
    """
    Calculate top-k metrics and AUC-ROC for decoding from brain data.

    Args:
        X_test: Test brain data
        Y: All word embeddings across all folds.
        model: Trained model
        device: PyTorch device
        selected_words: List of selected words for vocabulary.
        test_index: Test indices for indexing into position_to_id
        train_index: Train indices for indexing into position_to_id
        top_k_thresholds: List of k values for top-k accuracy
        min_train_freq_auc: Minimum training frequency for AUC calculation
        min_test_freq_auc: Minimum test frequency for AUC calculation

    Returns:
        dict: Dictionary containing computed metrics
    """
    results = {}

    # Put model in evaluation mode
    model.eval()

    X_test, Y_test = X_test.to(device), Y_test.to(device)

    # Get predictions
    predictions = get_predictions(X_test, model, device, batch_size, **kwargs)

    # Compute cosine distances
    distances = compute_cosine_distances(predictions, Y_test)

    # Measure performance based on each individual word occurrence
    occurence_scores, _, _ = compute_class_scores(distances)
    for k_val in top_k_thresholds:
        # Labels are in order of test set since we are hoping the ith example is predicted as the ith class.
        results[f"test_occurence_top_{k_val}"] = top_k_accuracy(
            occurence_scores, torch.arange(occurence_scores.shape[0], device=device), k_val
        )
    results["test_occurence_perplexity"] = perplexity(
        occurence_scores, torch.arange(occurence_scores.shape[0], device=device)
    )

    # Group by words for an easier task.
    # Build vocabulary. While in most cases we would not want to include
    # the test set in the vocabulary building process, we remove any
    _, _, position_to_id = build_vocabulary(selected_words)
    position_to_id = np.array(position_to_id)

    word_scores, _, test_class_idxs = compute_class_scores(
        distances, torch.from_numpy(position_to_id[test_index]).to(device)
    )
    # Get a mapping from over-all class index -> test class index.
    test_class_idxs_np = test_class_idxs.cpu().numpy()
    class_to_test_idxs = np.empty(np.max(position_to_id) + 1, dtype=int)
    class_to_test_idxs[test_class_idxs_np] = np.arange(len(test_class_idxs_np))

    train_frequencies = np.bincount(
        position_to_id[train_index], minlength=np.max(position_to_id) + 1
    )
    # Limit train frequencies to only those in the test set.
    train_frequencies = train_frequencies[test_class_idxs_np]

    test_frequencies = np.bincount(
        position_to_id[test_index], minlength=np.max(position_to_id) + 1
    )
    test_frequencies = test_frequencies[test_class_idxs_np]

    # Translate to vocab of word ID's.
    test_word_ids = class_to_test_idxs[position_to_id[test_index]]

    # For calculate_auc_roc, pass numpy arrays (it converts internally anyway)
    word_scores_np = word_scores.cpu().numpy()
    avg_auc, train_weighted_auc, test_weighted_auc = calculate_auc_roc(
        word_scores_np,
        test_word_ids,
        train_frequencies,
        test_frequencies,
        min_train_freq_auc,
        min_test_freq_auc,
    )
    results["test_word_avg_auc_roc"] = avg_auc
    results["test_word_train_weighted_auc_roc"] = train_weighted_auc
    results["test_word_test_weighted_auc_roc"] = test_weighted_auc

    # For top_k_accuracy and perplexity, use torch tensors
    test_word_ids_tensor = torch.from_numpy(test_word_ids).to(device)
    for k_val in top_k_thresholds:
        results[f"test_word_top_{k_val}"] = top_k_accuracy(
            word_scores, test_word_ids_tensor, k_val
        )
    results["test_word_perplexity"] = perplexity(word_scores, test_word_ids_tensor)

    return results


def build_vocabulary(words):
    """
    Build vocabulary mappings from a list of words.

    Args:
        words: List of words (may contain repetitions)

    Returns:
        word_to_id: Dictionary mapping word -> unique_id
        id_to_word: Dictionary mapping unique_id -> word
        postion_to_id: Numpy array specifying what the ith word maps to which unique id.
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


def get_predictions(X, model, device, batch_size, **kwargs):
    X = X.to(device)

    # Step 2: Get predicted embeddings for all neural data
    model_predictions = []
    for i in range(0, len(X), batch_size):
        with torch.no_grad():
            input_data = X[i : i + batch_size]
            pred = model(input_data, **kwargs)

        model_predictions.append(pred)

    # Stack to ensure we get a 2D tensor [num_samples, embedding_dim]
    return torch.cat(model_predictions)
