"""
Metrics for binary and multiclass classification tasks.
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import warnings

from core.registry import register_metric


@register_metric("bce")
def bce_metric(predicted: torch.Tensor, groundtruth: torch.Tensor) -> float:
    """Weighted BCE loss for binary classification using PyTorch's built-in functionality.

    Expects probabilities in [0,1] range. Uses sklearn's compute_class_weight for automatic class balancing.
    """

    # Check if input looks like logits and warn user
    if predicted.detach().min() < 0 or predicted.detach().max() > 1:
        warnings.warn(
            f"BCE metric received values outside [0,1] range (min={predicted.detach().min():.3f}, "
            f"max={predicted.detach().max():.3f}). Function expects probabilities in [0,1] range.",
            UserWarning,
        )
    else:
        probs = predicted

    # Convert to numpy for sklearn
    y_true = groundtruth.detach().cpu().numpy().astype(int)

    # Check if we have both classes in the batch
    unique_classes = np.unique(y_true)

    # If only one class present, use regular BCE (no weighting needed)
    if len(unique_classes) == 1:
        return F.binary_cross_entropy(probs, groundtruth)

    try:
        # Compute balanced class weights using sklearn
        class_weights = compute_class_weight(
            "balanced", classes=unique_classes, y=y_true
        )

        # Map class weights to [class_0_weight, class_1_weight]
        weight_dict = dict(zip(unique_classes, class_weights))
        weight_0 = weight_dict.get(0, 1.0)
        weight_1 = weight_dict.get(1, 1.0)

        # Create per-sample weights based on class
        sample_weights = torch.where(groundtruth == 1, weight_1, weight_0)
        sample_weights = sample_weights.to(dtype=probs.dtype, device=probs.device)

        return F.binary_cross_entropy(probs, groundtruth, weight=sample_weights)

    except Exception as e:
        print(f"Using: regular BCE instead. Error in weighted BCE: {e}")
        # Fallback to regular BCE if class weight computation fails
        return F.binary_cross_entropy(probs, groundtruth)


@register_metric("cross_entropy")
def cross_entropy_metric(predicted: torch.Tensor, groundtruth: torch.Tensor) -> float:
    """
    Cross-entropy loss for multi-class classification, expects raw logits.
    Groundtruth should contain class indices (not one-hot).

    Handles both standard classification (batch, num_classes) and sequence prediction
    (batch, seq_len, num_classes) by automatically reshaping.

    Supports -100 as ignore_index for padding tokens.
    """
    # Handle sequence prediction case: reshape (batch, seq_len, num_classes) -> (batch*seq_len, num_classes)
    if predicted.ndim == 3:
        batch_size, seq_len, num_classes = predicted.shape
        predicted = predicted.reshape(batch_size * seq_len, num_classes)
        groundtruth = groundtruth.reshape(batch_size * seq_len)

    return F.cross_entropy(predicted, groundtruth.long(), ignore_index=-100)

@register_metric("weighted_cross_entropy")
def weighted_cross_entropy_metric(predicted: torch.Tensor, groundtruth: torch.Tensor) -> torch.Tensor:
    """
    Weighted cross-entropy loss for classification (binary and multiclass).
    
    Returns:
        Scalar tensor containing weighted cross-entropy loss.
    
    Raises:
        ValueError: If input shapes are invalid or don't match PyTorch requirements.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    # Input must be at least 2D: (N, C) or (N, C, d1, d2, ..., dK)
    if predicted.ndim < 2:
        raise ValueError(
            f"predicted must have at least 2 dimensions (N, C), got shape {predicted.shape}. "
            f"Per PyTorch docs: input should be (N, C) or (N, C, d1, d2, ..., dK)"
        )
    
    # Get number of classes from second dimension (C)
    num_classes = predicted.shape[1]
    
    # Validate batch dimension matches between input and target
    # Per PyTorch docs: target should be (N,) or (N, d1, d2, ..., dK)
    if groundtruth.ndim == 0:
        raise ValueError(
            f"groundtruth cannot be scalar when predicted has batch dimension. "
            f"Expected shape (N,) or (N, d1, ...), got shape {groundtruth.shape}"
        )
    
    if predicted.shape[0] != groundtruth.shape[0]:
        raise ValueError(
            f"Batch size mismatch: predicted has {predicted.shape[0]} samples, "
            f"groundtruth has {groundtruth.shape[0]} samples"
        )
    
    groundtruth_long = groundtruth.long()
    
    # Flatten target to 1D array for scikit-learn compute_class_weight
    # Per scikit-learn docs: y should be array-like of shape (n_samples,)
    y_true_flat = groundtruth_long.detach().cpu().numpy().flatten()
    
    # Get unique classes present in the batch
    unique_classes = np.unique(y_true_flat)
    
    # If only one class present, no weighting needed (all samples are same class)
    if len(unique_classes) == 1:
        return F.cross_entropy(predicted, groundtruth_long)
    
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_classes,
        y=y_true_flat
    )
    
    weight_tensor = torch.ones(
        num_classes,
        dtype=predicted.dtype,
        device=predicted.device
    )
    
    # Assign computed weights to corresponding class indices
    for class_idx, weight in zip(unique_classes, class_weights):
        if 0 <= class_idx < num_classes:
            weight_tensor[class_idx] = float(weight)
    
    return F.cross_entropy(predicted, groundtruth_long, weight=weight_tensor)


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

    y_true = true.detach().cpu().numpy().astype(int)
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
        return float(
            roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",  # or 'ovo' (one-vs-one)
                average="macro",  # or 'weighted', depending on your use case
            )
        )
    except Exception:
        return 0.5


@register_metric("f1")
def f1_binary(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    F1 score for binary classification at 0.5 threshold.
    Expects probabilities in [0,1] range. (for binary classification)
    Also works for multiclass by taking argmax after softmax.
    """
    # Ensure 1D
    if pred.ndim > 1 and pred.shape[-1] == 1:
        pred = pred.squeeze(-1)
    if true.ndim > 1:
        true = true.squeeze(-1)

    y_true = true.detach().cpu().numpy().astype(int)

    # Check if input looks like logits and warn user
    if pred.detach().min() < 0 or pred.detach().max() > 1:
        warnings.warn(
            f"F1 metric received values outside [0,1] range (min={pred.detach().min():.3f}, "
            f"max={pred.detach().max():.3f}). Function expects probabilities in [0,1] range.",
            UserWarning,
        )

    y_pred = (pred.detach().cpu().numpy() >= 0.5).astype(int)
    try:
        return float(f1_score(y_true, y_pred, zero_division=0, average="weighted"))
    except Exception:
        return 0.0


@register_metric("sensitivity")
def sensitivity_binary(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Sensitivity (True Positive Rate) for binary classification.
    Sensitivity = TP / (TP + FN) = Recall

    Measures the proportion of actual positives that are correctly identified.
    Expects probabilities in [0,1] range.
    """

    # Ensure 1D
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if true.ndim > 1:
        true = true.squeeze(-1)

    y_true = true.detach().cpu().numpy().astype(int)

    # Check if input looks like logits and warn user
    if pred.detach().min() < 0 or pred.detach().max() > 1:
        warnings.warn(
            f"Sensitivity metric received values outside [0,1] range (min={pred.detach().min():.3f}, "
            f"max={pred.detach().max():.3f}). Function expects probabilities in [0,1] range.",
            UserWarning,
        )

    y_pred = (pred.detach().cpu().numpy() >= 0.5).astype(int)

    try:
        # Calculate True Positives and False Negatives
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        # Avoid division by zero
        if tp + fn == 0:
            return 0.0

        sensitivity = tp / (tp + fn)
        return float(sensitivity)
    except Exception:
        return 0.0


@register_metric("precision")
def precision_binary(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Precision for binary classification.
    Precision = TP / (TP + FP)

    Measures the proportion of predicted positives that are actually positive.
    Expects probabilities in [0,1] range.
    """

    # Ensure 1D
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if true.ndim > 1:
        true = true.squeeze(-1)

    y_true = true.detach().cpu().numpy().astype(int)

    # Check if input looks like logits and warn user
    if pred.detach().min() < 0 or pred.detach().max() > 1:
        warnings.warn(
            f"Precision metric received values outside [0,1] range (min={pred.detach().min():.3f}, "
            f"max={pred.detach().max():.3f}). Function expects probabilities in [0,1] range.",
            UserWarning,
        )

    y_pred = (pred.detach().cpu().numpy() >= 0.5).astype(int)

    try:
        # Calculate True Positives and False Positives
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        # Avoid division by zero
        if tp + fp == 0:
            return 0.0

        precision = tp / (tp + fp)
        return float(precision)
    except Exception:
        return 0.0


@register_metric("specificity")
def specificity_binary(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Specificity (True Negative Rate) for binary classification.
    Specificity = TN / (TN + FP)

    Measures the proportion of actual negatives that are correctly identified.
    Expects probabilities in [0,1] range.
    """

    # Ensure 1D
    if pred.ndim > 1:
        pred = pred.squeeze(-1)
    if true.ndim > 1:
        true = true.squeeze(-1)

    y_true = true.detach().cpu().numpy().astype(int)

    # Check if input looks like logits and warn user
    if pred.detach().min() < 0 or pred.detach().max() > 1:
        warnings.warn(
            f"Specificity metric received values outside [0,1] range (min={pred.detach().min():.3f}, "
            f"max={pred.detach().max():.3f}). Function expects probabilities in [0,1] range.",
            UserWarning,
        )

    y_pred = (pred.detach().cpu().numpy() >= 0.5).astype(int)

    try:
        # Calculate True Negatives and False Positives
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()

        # Avoid division by zero
        if tn + fp == 0:
            return 0.0

        specificity = tn / (tn + fp)
        return float(specificity)
    except Exception:
        return 0.0


@register_metric("confusion_matrix")
def conf_matrix(
    predictions: np.ndarray, ground_truth: np.ndarray, num_classes=None
) -> np.ndarray:
    """
    Compute the confusion matrix for predictions and ground truth.

    Args:
        predictions: Array of predicted class indices or probabilities (shape [N] or [N, num_classes])
        ground_truth: Array of true class indices (shape [N])

    Returns:
        Confusion matrix as a 2D numpy array.
    """

    # Move to CPU and convert to numpy if tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(ground_truth, torch.Tensor):
        ground_truth = ground_truth.detach().cpu().numpy()

    # If predictions are probabilities, convert to class indices
    if predictions.ndim > 1:
        pred_labels = np.argmax(predictions, axis=1)
    else:
        pred_labels = (predictions >= 0.5).astype(int)

    # if num_classes is None:
    #     num_classes = max(ground_truth.max(), pred_labels.max()) + 1

    return confusion_matrix(ground_truth, pred_labels, labels=np.arange(num_classes))


def perplexity(predictions: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """
    Calculate perplexity of predictions as used for LLM evaluation.

    Perplexity = exp(cross_entropy) where cross_entropy is the average negative
    log-likelihood of the true labels.

    Handles both standard classification (batch, num_classes) and sequence prediction
    (batch, seq_len, num_classes) by automatically reshaping.

    Supports -100 as ignore_index for padding tokens.

    Args:
        predictions: Tensor of shape [num_samples, num_classes] or [batch, seq_len, num_classes] containing raw logits
        ground_truth: Tensor of shape [num_samples] or [batch, seq_len] containing the true class
                     indices for each sample

    Returns:
        float: Perplexity score (lower is better, minimum is 1.0)
    """
    if len(predictions) == 0:
        return float("inf")

    # Handle sequence prediction case: reshape (batch, seq_len, num_classes) -> (batch*seq_len, num_classes)
    if predictions.ndim == 3:
        batch_size, seq_len, num_classes = predictions.shape
        predictions = predictions.reshape(batch_size * seq_len, num_classes)
        ground_truth = ground_truth.reshape(batch_size * seq_len)

    # Calculate cross-entropy using PyTorch's implementation (expects raw logits)
    cross_entropy = F.cross_entropy(predictions, ground_truth.long(), ignore_index=-100)

    # Perplexity = exp(cross_entropy)
    return torch.exp(cross_entropy).item()
