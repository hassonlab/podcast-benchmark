import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from sklearn.metrics import roc_curve, auc

from sklearn.metrics import roc_auc_score, f1_score

from registry import register_metric


@register_metric("mse")
def mse_metric(predicted: torch.Tensor, groundtruth: torch.Tensor) -> float:
    return F.mse_loss(predicted, groundtruth)

@register_metric("bce")
def bce_metric(predicted: torch.Tensor, groundtruth: torch.Tensor) -> float:
    """BCE loss for binary classification, expects probabilities in [0, 1].

    If inputs do not look like probabilities, applies a sigmoid to convert logits to probs.
    """
    probs = predicted
    # Heuristic: if values are outside [0,1], treat as logits and apply sigmoid
    if probs.detach().min() < 0 or probs.detach().max() > 1:
        probs = torch.sigmoid(probs)
    return F.binary_cross_entropy(probs, groundtruth)

@register_metric("cosine_sim")
def cosine_similarity(pred: torch.Tensor, true: torch.Tensor) -> float:
    return F.cosine_similarity(pred, true, dim=-1).mean()


@register_metric("cosine_dist")
def cosine_distance(pred: torch.Tensor, true: torch.Tensor) -> float:
    sim = F.cosine_similarity(pred, true, dim=-1)
    return (1 - sim).mean()

@register_metric("cross_entropy")
def cross_entropy_metric(predicted: torch.Tensor, groundtruth: torch.Tensor) -> float:
    """
    Cross-entropy loss for multi-class classification, expects raw logits.
    Groundtruth should contain class indices (not one-hot).
    """
    return F.cross_entropy(predicted, groundtruth.long())


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
    logits = 1 - compute_cosine_distances(predicted_embeddings, actual_embeddings)
    probs = F.softmax(logits, dim=1)
    return entropy(probs).mean()


@register_metric("roc_auc")
def roc_auc_binary(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    ROC-AUC for binary classification. Accepts raw scores; any monotonic
    transform (e.g., tanh, sigmoid) is fine for AUC.
    """
    # Ensure 1D
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if true.ndim > 1:
        true = true.squeeze(-1)

    y_true = true.detach().cpu().numpy()
    y_score = pred.detach().cpu().numpy()

    # Handle batches with a single class gracefully
    if len(set(y_true.tolist())) < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5
    
@register_metric("roc_auc_multiclass")
def roc_auc_multiclass(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    ROC-AUC for multiclass classification.
    Accepts raw scores (logits or probabilities).
    """
    # Convert to CPU numpy arrays
    y_true = true.detach().cpu().numpy()
    y_score = pred.detach().cpu().numpy()

    # Handle edge case: fewer than 2 classes in batch
    if len(set(y_true.tolist())) < 2:
        return 0.5

    try:
        return float(roc_auc_score(
            y_true,
            y_score,
            multi_class='ovr',     # or 'ovo' (one-vs-one)
            average='macro'        # or 'weighted', depending on your use case
        ))
    except Exception:
        return 0.5
    
@register_metric("corr")
def pearson_correlation(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Pearson correlation coefficient between predictions and ground truth.
    """
    pred_mean = pred.mean()
    true_mean = true.mean()

    cov = ((pred - pred_mean) * (true - true_mean)).mean()
    pred_std = pred.std()
    true_std = true.std()

    if pred_std.item() == 0 or true_std.item() == 0:
        return 0.0

    corr = cov / (pred_std * true_std)

    # corr=torch.corrcoef(true.unsqueeze(0), pred.unsqueeze(0))[0,1]

    return corr.item()



@register_metric("f1")
def f1_binary(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    F1 score at a 0.5 threshold after sigmoid.
    """
    # Ensure 1D
    if pred.ndim > 1 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if true.ndim > 1:
        true = true.squeeze(-1)

    y_true = true.detach().cpu().numpy().astype(int)
    
    if pred.ndim > 1 and pred.shape[1] > 1:
        # If logits, apply softmax
        if pred.detach().min() < 0 or pred.detach().max() > 1:
            probs = torch.softmax(pred, dim=1)
        else:
            probs = pred
        y_pred = probs.detach().cpu().numpy().argmax(axis=1)
        average = "weighted"
    else:
        # Binary case
        if (pred.detach().min() < 0 or pred.detach().max() > 1):
            probs = torch.sigmoid(pred)
        else:
            probs = pred
        y_pred = (probs.detach().cpu().numpy() >= 0.5).astype(int)
        # average = "binary" if y_true.max() == 1 else "weighted"
        average = "weighted"

    try:
        return float(f1_score(y_true, y_pred, zero_division=0, average=average))
    except Exception:
        return 0.0


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

@register_metric("confusion_matrix")
def conf_matrix(predictions: np.ndarray, ground_truth: np.ndarray, num_classes=None) -> np.ndarray:
    """
    Compute the confusion matrix for predictions and ground truth.

    Args:
        predictions: Array of predicted class indices or probabilities (shape [N] or [N, num_classes])
        ground_truth: Array of true class indices (shape [N])

    Returns:
        Confusion matrix as a 2D numpy array.
    """
    from sklearn.metrics import confusion_matrix

    # Move to CPU and convert to numpy if tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # If predictions are probabilities, convert to class indices
    if predictions.ndim > 1:
        pred_labels = np.argmax(predictions, axis=1)
    else:
        pred_labels =(predictions>= 0.5).astype(int)

    # if num_classes is None:
    #     num_classes = max(ground_truth.max(), pred_labels.max()) + 1

    return confusion_matrix(ground_truth, pred_labels,labels=np.arange(num_classes))




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
