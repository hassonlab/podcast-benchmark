import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score


from registry import register_metric


@register_metric("mse")
def mse_metric(predicted: torch.Tensor, groundtruth: torch.Tensor) -> float:
    return F.mse_loss(predicted, groundtruth)


@register_metric("cosine_sim")
def cosine_similarity(pred: torch.Tensor, true: torch.Tensor) -> float:
    return F.cosine_similarity(pred, true, dim=-1).mean()


@register_metric("cosine_dist")
def cosine_distance(pred: torch.Tensor, true: torch.Tensor) -> float:
    sim = F.cosine_similarity(pred, true, dim=-1)
    return (1 - sim).mean()


def get_logits(predicted_embeddings, actual_embeddings):
    """Shared function to compute similarity logits over embeddings."""
    # Normalize embeddings
    pred_norm = F.normalize(predicted_embeddings, dim=1)
    actual_norm = F.normalize(actual_embeddings, dim=1)

    # Similarity matrix: [n_samples, n_samples]
    return torch.matmul(pred_norm, actual_norm.T)


@register_metric("nll_embedding")
def compute_nll_contextual(predicted_embeddings, actual_embeddings):
    """
    Computes a contrastive NLL where each predicted embedding is scored against all actual embeddings.
    """
    logits = get_logits(predicted_embeddings, actual_embeddings)

    # Labels: diagonal = correct match
    targets = torch.arange(
        len(predicted_embeddings), device=predicted_embeddings.device
    )

    # Cross-entropy over rows
    return F.cross_entropy(logits, targets)


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Compute the entropy for each row in a batch of categorical distributions.

    Args:
        p: Tensor of shape [B, C], where each row is a probability distribution.
        eps: Small constant to prevent log(0).

    Returns:
        Tensor of shape [B], the entropy for each distribution in the batch.
    """
    p = p.clamp(min=eps)  # Avoid log(0)
    return -(p * p.log()).sum(dim=1)


@register_metric("similarity_entropy")
def similarity_entropy(predicted_embeddings, actual_embeddings):
    logits = get_logits(predicted_embeddings, actual_embeddings)
    probs = F.softmax(logits, dim=1)
    return entropy(probs).mean()


def calculate_auc_roc(
    predictions,
    groundtruth,
    frequencies,
    min_frequencies,
    average="weighted",
):
    """
    Calculate AUC-ROC score with frequency-based filtering.

    Args:
        predictions: Array of shape [num_samples, num_vocab] where each row is a
                    probability distribution over the vocabulary
        groundtruth: Array of shape [num_samples] containing class predictions
        frequencies: Lost of Arrays of shape [num_vocab] containing the number of appearances
                    of each vocab item for filtering
        min_frequencies: List of inimum number of appearances required for a vocab item
                       to be included in the calculation
        average: How the AUC ROC score should be averaged across classes. Options
                include 'weighted', 'macro', 'micro', etc.

    Returns:
        float: AUC-ROC score calculated only for vocabulary items that meet
               the minimum frequency threshold
    """
    # Only include labels that meet the minimum frequency level.
    include_class = None
    for freq, thresh in zip(frequencies, min_frequencies):
        if include_class is None:
            include_class = np.ones_like(freq, dtype=bool)
        print(freq)
        include_class = include_class & (freq >= thresh)
    print(
        "Fraction of examples included in AUC-ROC calculation:",
        include_class.sum() / include_class.shape[0],
    )
    include_example = include_class[groundtruth]

    # Filter examples
    filtered_groundtruth = groundtruth[include_example]
    filtered_predictions = predictions[include_example]

    # Get the original class indices that are included
    included_class_indices = np.where(include_class)[0]

    # Filter prediction columns to only include frequent classes
    filtered_predictions = filtered_predictions[:, included_class_indices]

    # Remap groundtruth classes to consecutive indices
    # Create mapping from original class index to new consecutive index
    class_mapping = {
        original_idx: new_idx
        for new_idx, original_idx in enumerate(included_class_indices)
    }
    remapped_groundtruth = np.array(
        [class_mapping[cls] for cls in filtered_groundtruth]
    )

    # Handle binary vs multiclass cases
    n_classes = len(included_class_indices)
    if n_classes == 2:
        # Binary classification: use probabilities of positive class (class 1)
        return roc_auc_score(remapped_groundtruth, filtered_predictions[:, 1])
    else:
        # Multiclass classification
        return roc_auc_score(
            remapped_groundtruth,
            filtered_predictions,
            average=average,
            multi_class="ovr",
        )


def top_k_accuracy(predictions: np.ndarray, ground_truth: np.ndarray, k: int) -> float:
    """
    Calculate top-k accuracy for multiclass classification.

    Args:
        predictions: Array of shape [num_samples, num_classes] where each row
                    contains prediction scores/probabilities for each class
        ground_truth: Array of shape [num_samples] containing the true class
                     indices for each sample
        k: Number of top predictions to consider

    Returns:
        float: Top-k accuracy as a fraction between 0 and 1
    """
    if len(predictions) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # Get the indices of the top-k predictions for each sample
    # argsort returns indices in ascending order, so we take the last k and reverse
    top_k_indices = np.argsort(predictions, axis=1)[:, -k:]

    # Check if ground truth is in the top-k predictions for each sample
    correct = np.array(
        [ground_truth[i] in top_k_indices[i] for i in range(len(ground_truth))]
    )

    return np.mean(correct)
