import torch
import torch.nn as nn
from torch.nn import functional as F

from models.brainbert.integration import (
    ReferenceBrainBERTDecoder,
    _adaptive_avg_pool_temporal_patches,
)


def test_adaptive_avg_pool_temporal_patches_exact_length_noop():
    features = torch.arange(2 * 10 * 3, dtype=torch.float32).view(2, 10, 3)

    pooled = _adaptive_avg_pool_temporal_patches(features, 10)

    torch.testing.assert_close(pooled, features)


def test_adaptive_avg_pool_temporal_patches_downsamples_with_means():
    features = torch.arange(1 * 8 * 1, dtype=torch.float32).view(1, 8, 1)

    pooled = _adaptive_avg_pool_temporal_patches(features, 4)

    expected = torch.tensor([[[0.5], [2.5], [4.5], [6.5]]])
    torch.testing.assert_close(pooled, expected)


def test_adaptive_avg_pool_temporal_patches_uses_all_tokens():
    features = torch.arange(1 * 13 * 2, dtype=torch.float32).view(1, 13, 2)

    pooled = _adaptive_avg_pool_temporal_patches(features, 10)

    expected = F.adaptive_avg_pool1d(features.transpose(1, 2), 10).transpose(1, 2)
    torch.testing.assert_close(pooled, expected)


def test_adaptive_avg_pool_temporal_patches_can_expand_to_requested_length():
    features = torch.arange(2 * 3 * 4, dtype=torch.float32).view(2, 3, 4)

    pooled = _adaptive_avg_pool_temporal_patches(features, 5)

    assert pooled.shape == (2, 5, 4)


class StubBrainBERTUpstream(nn.Module):
    def __init__(self, seq_len, hidden_dim):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim

    def forward(self, inputs, pad_mask, intermediate_rep=False):
        assert intermediate_rep is True
        assert pad_mask is None
        return torch.arange(
            inputs.shape[0] * self.seq_len * self.hidden_dim,
            dtype=inputs.dtype,
            device=inputs.device,
        ).view(inputs.shape[0], self.seq_len, self.hidden_dim)


class StubFinetuneModel(nn.Module):
    def __init__(self, upstream):
        super().__init__()
        self.upstream = upstream
        self.frozen_upstream = False


def test_brainbert_decoder_pools_temporal_tokens_flattened():
    batch_size = 2
    num_electrodes = 3
    temporal_patches_to_keep = 4
    hidden_dim = 5
    decoder = ReferenceBrainBERTDecoder(
        StubFinetuneModel(StubBrainBERTUpstream(seq_len=8, hidden_dim=hidden_dim)),
        num_electrodes=num_electrodes,
        hidden_dim=hidden_dim,
        temporal_patches_to_keep=temporal_patches_to_keep,
    )
    x = torch.zeros(batch_size, num_electrodes, 6, 7)

    features = decoder(x, return_feature_emb_instead_of_projection=True)

    assert features.shape == (
        batch_size,
        num_electrodes * temporal_patches_to_keep * hidden_dim,
    )
