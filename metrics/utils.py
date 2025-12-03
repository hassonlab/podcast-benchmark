"""
Utility functions shared across different metric implementations.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc


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
        unique_classes = torch.arange(cosine_distances.shape[0])

    # Convert distances to similarities (logits)
    # Since cosine distance = 1 - cosine_similarity, we convert back:
    # logits = 1 - distance = cosine_similarity
    class_logits = 1 - class_distances

    # Apply softmax transformation to get probabilities
    class_probabilities = F.softmax(class_logits, dim=1)

    return class_probabilities, class_logits, unique_classes


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
        predictions: Array or Tensor of shape [num_samples, num_vocab] where each row is a
                    probability distribution over the vocabulary
        groundtruth: Array or Tensor of shape [num_samples] containing class predictions
        train_frequencies: Array or Tensor of shape [num_vocab] containing the number of appearances
                    of each vocab item in train set.
        test_frequencies: Array or Tensor of shape [num_vocab] containing the number of appearances
                    of each vocab item in test set.
        min_train_freq: Minimum number of occurences in train set to include class.
        min_test_freq: Minimum number of occurences in test set to include class.

    Returns:
        tuple[float, float, flaot]: AUC-ROC score calculated only for vocabulary items that meet
               the minimum frequency threshold. 0th is unwieghted, 1st is weighted by train frequency,
               2nd is weighted by test frequency.
    """
    # Convert torch tensors to numpy arrays for sklearn compatibility
    if isinstance(predictions, torch.Tensor):
        predictions_np = predictions.detach().cpu().numpy()
    else:
        predictions_np = np.asarray(predictions)

    if isinstance(groundtruth, torch.Tensor):
        groundtruth_np = groundtruth.detach().cpu().numpy()
    else:
        groundtruth_np = np.asarray(groundtruth)

    if isinstance(train_frequencies, torch.Tensor):
        train_frequencies_np = train_frequencies.detach().cpu().numpy()
    else:
        train_frequencies_np = np.asarray(train_frequencies)

    if isinstance(test_frequencies, torch.Tensor):
        test_frequencies_np = test_frequencies.detach().cpu().numpy()
    else:
        test_frequencies_np = np.asarray(test_frequencies)

    # Ensure frequencies are always arrays for consistent handling
    train_frequencies_np = np.atleast_1d(train_frequencies_np)
    test_frequencies_np = np.atleast_1d(test_frequencies_np)

    # Only include labels that meet the minimum frequency level.
    include_trains = train_frequencies_np >= min_train_freq
    include_tests = test_frequencies_np >= min_test_freq
    include_class = include_trains & include_tests

    print(
        f"Fraction of examples included in AUC-ROC calculation:",
        f"{include_class.sum() / include_class.shape[0]:.4f},",
        f"({include_class.sum()} / {include_class.shape[0]})",
    )
    # Get the original class indices that are included
    included_class_indices = np.where(include_class)[0]
    scores = []

    one_hots = np.eye(groundtruth_np.max() + 1)[groundtruth_np]

    # Due to limitations in sklearn roc_auc_score we calculate this ourselves here.
    for class_index in included_class_indices:
        probs = predictions_np[:, class_index]
        c_labels = one_hots[:, class_index]
        fpr, tpr, _ = roc_curve(c_labels, probs)
        score = auc(fpr, tpr)
        scores.append(score)

    scores = np.array(scores)
    avg_auc = np.mean(scores)

    # Only use frequencies for included classes
    included_train_freqs = train_frequencies_np[included_class_indices]
    normed_freqs = included_train_freqs / included_train_freqs.sum()
    train_weighted_auc = (scores * normed_freqs).sum()

    included_test_freqs = test_frequencies_np[included_class_indices]
    normed_freqs = included_test_freqs / included_test_freqs.sum()
    test_weighted_auc = (scores * normed_freqs).sum()

    return avg_auc, train_weighted_auc, test_weighted_auc


def top_k_accuracy(predictions: torch.Tensor, ground_truth: torch.Tensor, k: int) -> float:
    """
    Calculate top-k accuracy for multiclass classification.

    Args:
        predictions: Tensor of shape [num_samples, num_classes] where each row
                    contains prediction scores/probabilities for each class
        ground_truth: Tensor of shape [num_samples] containing the true class
                     indices for each sample
        k: Number of top predictions to consider

    Returns:
        float: Top-k accuracy as a fraction between 0 and 1
    """
    if len(predictions) == 0:
        return 0.0

    if k <= 0:
        return 0.0

    # Clamp k to the number of classes
    num_classes = predictions.shape[1]
    k = min(k, num_classes)

    # Get the indices of the top-k predictions for each sample
    # topk returns (values, indices) in descending order
    _, top_k_indices = torch.topk(predictions, k, dim=1)

    # Expand ground truth to match top_k_indices shape for comparison
    ground_truth_expanded = ground_truth.unsqueeze(1).expand_as(top_k_indices)

    # Check if ground truth is in the top-k predictions for each sample
    correct = (top_k_indices == ground_truth_expanded).any(dim=1)

    return correct.float().mean().item()
