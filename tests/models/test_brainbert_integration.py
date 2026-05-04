import pytest
import torch
import torch.nn as nn

from models.brainbert.integration import (
    ReferenceBrainBERTDecoder,
    _center_crop_temporal_patches,
)


def test_center_crop_temporal_patches_exact_length_noop():
    features = torch.arange(2 * 10 * 3).view(2, 10, 3)

    cropped = _center_crop_temporal_patches(features, 10)

    torch.testing.assert_close(cropped, features)


def test_center_crop_temporal_patches_even_surplus():
    features = torch.arange(1 * 14 * 2).view(1, 14, 2)

    cropped = _center_crop_temporal_patches(features, 10)

    torch.testing.assert_close(cropped, features[:, 2:12, :])


def test_center_crop_temporal_patches_odd_surplus():
    features = torch.arange(1 * 13 * 2).view(1, 13, 2)

    cropped = _center_crop_temporal_patches(features, 10)

    torch.testing.assert_close(cropped, features[:, 1:11, :])


def test_center_crop_temporal_patches_too_short_raises():
    features = torch.zeros(2, 9, 3)

    with pytest.raises(ValueError, match="fewer temporal tokens"):
        _center_crop_temporal_patches(features, 10)


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


def test_brainbert_decoder_keeps_central_temporal_tokens_flattened():
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
