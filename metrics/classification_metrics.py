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


@register_metric("bce_with_logits")
def bce_with_logits_metric(predicted: torch.Tensor, groundtruth: torch.Tensor) -> float:
    """
    Binary Cross Entropy with Logits.
    Combines a Sigmoid layer and the BCELoss in one single class.
    
    Args:
        predicted: Raw logits from the model (before Sigmoid). Shape: (N, ) or (N, 1)
        groundtruth: True labels (0 or 1). Shape: (N, ) or (N, 1)
    """
    # 타겟 데이터(groundtruth)는 float 타입이어야 계산 가능
    target = groundtruth.float()
    
    # 차원이 맞지 않는 경우를 대비해 squeeze (예: [N, 1] -> [N])
    if predicted.shape != target.shape:
         # predicted와 target의 차원을 맞춤 (보통 target을 predicted에 맞추거나 그 반대)
         # 여기서는 안전하게 둘 다 1차원으로 펴서 계산하거나, 상황에 맞춰 조정
         pass 

    # PyTorch의 안정적인 구현체 사용 (내부적으로 Sigmoid -> BCE 수행)
    return F.binary_cross_entropy_with_logits(predicted, target)

@register_metric("cross_entropy")
def cross_entropy_metric(predicted: torch.Tensor, groundtruth: torch.Tensor) -> float:
    """
    Cross-entropy loss for multi-class classification, expects raw logits.
    Groundtruth should contain class indices (not one-hot).
    """
    return F.cross_entropy(predicted, groundtruth.long())

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
    y_true = true.detach().cpu().numpy()
    if y_true.ndim == 2 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1) # One-Hot -> Index
    else:
        y_true = y_true.astype(int).ravel()

    y_score = torch.softmax(pred, dim=1).detach().cpu().numpy()

    present_classes = np.unique(y_true)
    
    if len(present_classes) < 2:
        return 0.5

    auc_scores = []
    for cls in present_classes:
        binary_true = (y_true == cls).astype(int)
        binary_score = y_score[:, cls]
        
        try:
            cls_auc = roc_auc_score(binary_true, binary_score)
            auc_scores.append(cls_auc)
        except ValueError:
            continue

    if len(auc_scores) == 0:
        return 0.5
        
    return float(np.mean(auc_scores))


@register_metric("f1")
def f1_binary(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Binary F1 Metric
    """
    # 1. Tensor
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    
    # 2. Multiclass
    if pred.ndim > 1 and pred.shape[-1] > 1:
        # [Case A: Multiclass] Argmax
        y_pred = torch.argmax(pred, dim=-1).numpy()
    else:
        # [Case B: Binary] Threshold
        pred = pred.squeeze()
        true = true.squeeze()
        
        # Logit(negative values included) then apply Sigmoid to convert to probabilities
        if pred.min() < 0 or pred.max() > 1.0:
            pred = torch.sigmoid(pred)
            
        # Apply Threshold (0.5)
        # Tip: If AUROC is high but F1 is low, try lowering this value to 0.3 or 0.4.
        y_pred = (pred.numpy() >= 0.5).astype(int)

    y_true = true.astype(int)

    try:
        # Multiclass Shape Mismatch resolved
        return float(f1_score(y_true, y_pred, zero_division=0, average="weighted"))
    except Exception as e:
        # print(f"F1 Error: {e}")
        return 0.0

@register_metric("acc")
def accuracy_metric(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Accuracy metric for both binary and multiclass classification.
    
    Args:
        pred: Predictions from the model. 
              - For binary: Probabilities (0~1) or Logits.
              - For multiclass: Logits or Probabilities (N, C).
        true: Ground truth labels.
              - For binary: (N, ) or (N, 1) with values 0 or 1.
              - For multiclass: (N, ) with class indices.
    """
    # 1. Tensor 변환 및 Detach
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu()

    # 2. Numpy 변환
    pred = pred.numpy() if isinstance(pred, torch.Tensor) else np.array(pred)
    true = true.numpy() if isinstance(true, torch.Tensor) else np.array(true)

    # 3. 차원 정리 (Squeeze)
    # Binary의 경우 (N, 1) -> (N, )
    if pred.ndim > 1 and pred.shape[-1] == 1:
        pred = pred.squeeze()
    if true.ndim > 1:
        true = true.squeeze()

    # 4. 예측값(Label) 결정
    if pred.ndim > 1:
        # [Case A: Multiclass] (N, C) -> 가장 높은 확률의 인덱스 선택
        y_pred = np.argmax(pred, axis=1)
    else:
        # [Case B: Binary] (N, ) -> 0.5 Threshold 적용 (Logit인 경우 Sigmoid 처리 필요할 수 있으나, 일반적으로 부호나 0.5 기준으로 판단 가능)
        # 만약 Logit(음수 포함)이 들어온다면 0.0을 기준으로, 확률(0~1)이 들어온다면 0.5를 기준으로 해야 함.
        # 기존 코드들이 확률을 기대하고 warn을 띄우므로, 여기선 0.5 기준을 기본으로 함.
        
        # 입력이 확률 범위(0~1)를 벗어난 경우 (Logit으로 추정) Sigmoid 적용
        if pred.min() < 0 or pred.max() > 1.0:
            pred = 1 / (1 + np.exp(-pred))  # Sigmoid

        y_pred = (pred >= 0.5).astype(int)

    # 5. 정답값(Label) 정수 변환
    y_true = true.astype(int)

    # 6. 정확도 계산
    try:
        correct = (y_pred == y_true).sum()
        total = len(y_true)
        if total == 0:
            return 0.0
        return float(correct / total)
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
