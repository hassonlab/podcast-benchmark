"""
Metrics for regression and continuous prediction tasks.
"""

import torch
import torch.nn.functional as F

from core.registry import register_metric


@register_metric("mse")
def mse_metric(predicted: torch.Tensor, groundtruth: torch.Tensor) -> float:
    return F.mse_loss(predicted, groundtruth)


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


@register_metric("r2")
def r2_score_metric(pred: torch.Tensor, true: torch.Tensor) -> float:
    """
    Coefficient of determination (R^2) for regression tasks.

    Supports inputs of shape [B, 1] or [B]. Returns 1 - SSE / SST, or 0.0 if variance is zero.
    """
    # Pure-PyTorch implementation so this metric works on CUDA tensors.
    # Squeeze to 1D for computation
    if pred.ndim > 1:
        pred_s = pred.squeeze(-1)
    else:
        pred_s = pred
    if true.ndim > 1:
        true_s = true.squeeze(-1)
    else:
        true_s = true

    # Convert to float32 and flatten
    y_pred = pred_s.to(dtype=torch.float32).reshape(-1)
    y_true = true_s.to(dtype=torch.float32).reshape(-1)

    if y_true.numel() == 0:
        return 0.0

    ss_res = torch.sum((y_true - y_pred) ** 2)
    ss_tot = torch.sum((y_true - torch.mean(y_true)) ** 2)

    # If there's zero variance in the true targets, return 0.0
    if torch.isclose(ss_tot, torch.tensor(0.0, device=ss_tot.device)):
        return 0.0

    r2 = 1.0 - ss_res / ss_tot
    return float(float(r2.cpu().item()))
