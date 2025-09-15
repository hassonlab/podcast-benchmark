import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc


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
    train_frequencies,
    test_frequencies,
    min_train_freq,
    min_test_freq,
):
    """
    Calculate AUC-ROC score with frequency-based filtering.

    Args:
        predictions: Array of shape [num_samples, num_vocab] where each row is a
                    probability distribution over the vocabulary
        groundtruth: Array of shape [num_samples] containing class predictions
        train_frequencies: Array of shape [num_vocab] containing the number of appearances
                    of each vocab item in train set.
        test_frequencies: Array of shape [num_vocab] containing the number of appearances
                    of each vocab item in test set.
        min_train_freq: Minimum number of occurences in train set to include class.
        min_test_freq: Minimum number of occurences in test set to include class.

    Returns:
        tuple[float, float, flaot]: AUC-ROC score calculated only for vocabulary items that meet
               the minimum frequency threshold. 0th is unwieghted, 1st is weighted by train frequency,
               2nd is weighted by test frequency.
    """
    # Ensure frequencies are always arrays for consistent handling
    train_frequencies = np.atleast_1d(train_frequencies)
    test_frequencies = np.atleast_1d(test_frequencies)

    # Only include labels that meet the minimum frequency level.
    include_trains = train_frequencies >= min_train_freq
    include_tests = test_frequencies >= min_test_freq
    include_class = include_trains & include_tests

    print(
        f"Fraction of examples included in AUC-ROC calculation:",
        f"{include_class.sum() / include_class.shape[0]:.4f},",
        f"({include_class.sum()} / {include_class.shape[0]})",
    )
    # Get the original class indices that are included
    included_class_indices = np.where(include_class)[0]
    scores = []

    one_hots = np.eye(groundtruth.max() + 1)[groundtruth]

    # Due to limitations in sklearn roc_auc_score we calculate this ourselves here.
    for class_index in included_class_indices:
        probs = predictions[:, class_index]
        c_labels = one_hots[:, class_index]
        fpr, tpr, _ = roc_curve(c_labels, probs)
        score = auc(fpr, tpr)
        scores.append(score)

    scores = np.array(scores)
    avg_auc = np.mean(scores)

    # Only use frequencies for included classes
    included_train_freqs = train_frequencies[included_class_indices]
    normed_freqs = included_train_freqs / included_train_freqs.sum()
    train_weighted_auc = (scores * normed_freqs).sum()

    included_test_freqs = test_frequencies[included_class_indices]
    normed_freqs = included_test_freqs / included_test_freqs.sum()
    test_weighted_auc = (scores * normed_freqs).sum()

    return avg_auc, train_weighted_auc, test_weighted_auc


@register_metric("perplexity")
def perplexity(predictions: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Calculate perplexity of predictions as used for LLM evaluation.

    Perplexity = 2^(cross_entropy) where cross_entropy is the average negative
    log-likelihood of the true labels.

    Args:
        predictions: Array of shape [num_samples, num_classes] where each row
                    contains prediction probabilities for each class (should sum to 1)
        ground_truth: Array of shape [num_samples] containing the true class
                     indices for each sample

    Returns:
        float: Perplexity score (lower is better, minimum is 1.0)
    """
    if len(predictions) == 0:
        return float("inf")

    # Ensure predictions are valid probabilities
    predictions = np.clip(predictions, 1e-12, 1.0)

    # Calculate cross-entropy: -1/N * sum(log(p_i)) where p_i is probability of true class
    true_class_probs = predictions[np.arange(len(ground_truth)), ground_truth]
    cross_entropy = -np.mean(np.log2(true_class_probs))

    # Perplexity = 2^(cross_entropy)
    return 2**cross_entropy


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
