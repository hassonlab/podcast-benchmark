import torch
from torch.nn import functional as F

def cosine_similarity(pred, true) -> float:
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred)
    if not isinstance(true, torch.Tensor):
        true = torch.as_tensor(true)
    return F.cosine_similarity(pred, true, dim=-1).mean()


def cosine_distance(pred, true) -> float:
    if not isinstance(pred, torch.Tensor):
        pred = torch.as_tensor(pred)
    if not isinstance(true, torch.Tensor):
        true = torch.as_tensor(true)
    sim = F.cosine_similarity(pred, true, dim=-1)
    return (1 - sim).mean()


def compute_nll_contextual(predicted_embeddings, actual_embeddings):
    if not isinstance(predicted_embeddings, torch.Tensor):
        predicted_embeddings = torch.as_tensor(predicted_embeddings)
    if not isinstance(actual_embeddings, torch.Tensor):
        actual_embeddings = torch.as_tensor(actual_embeddings)

    logits = 1 - compute_cosine_distances(predicted_embeddings, actual_embeddings)
    targets = torch.arange(
        len(predicted_embeddings), device=predicted_embeddings.device
    )
    return F.cross_entropy(logits, targets)


def entropy(p: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    p = p.clamp(min=eps) 
    return -(p * p.log()).sum(dim=1)


def similarity_entropy(predicted_embeddings, actual_embeddings):
    logits = 1 - compute_cosine_distances(predicted_embeddings, actual_embeddings)
    probs = F.softmax(logits, dim=1)
    return entropy(probs).mean()

def compute_cosine_distances(predictions, word_embeddings):
    word_embeddings_norm = F.normalize(word_embeddings, p=2, dim=1)

    if predictions.dim() == 2:
        predictions = predictions.unsqueeze(1)
    elif predictions.dim() != 3:
        raise ValueError(
            f"Predictions must be 2D or 3D tensor, got {predictions.dim()}D"
        )

    num_samples, n_ensemble, embedding_dim = predictions.shape

    predictions_reshaped = predictions.view(num_samples * n_ensemble, embedding_dim)

    predictions_norm = F.normalize(predictions_reshaped, p=2, dim=1)

    cosine_similarities = torch.mm(predictions_norm, word_embeddings_norm.t())

    cosine_distances = 1 - cosine_similarities

    cosine_distances = cosine_distances.view(
        num_samples, n_ensemble, word_embeddings.shape[0]
    )

    return cosine_distances.mean(dim=1)