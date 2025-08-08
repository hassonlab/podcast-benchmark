import torch
import torch.nn as nn
from torch.nn import functional as F


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
